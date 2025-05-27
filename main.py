#!/usr/bin/env python3
"""
Enhanced Hyperliquid Trading Bot
--------------------------------
This module implements the main trading bot logic for the Hyperliquid exchange.
"""

import os
import sys
import json
import asyncio
import logging
import datetime
import time
import traceback
import signal
import threading
from typing import Dict, List, Optional, Any, Callable

# Import core components
from core.hyperliquid_adapter import HyperliquidAdapter as HyperliquidExchangeAdapter
from core.error_handler import ErrorHandler
from sentiment.llm_analyzer import LLMSentimentAnalyzer
from config_compatibility import ConfigManager

class WatchdogTimer:
    """
    Watchdog timer to detect and recover from frozen states.
    """
    def __init__(self, timeout: float, callback: Callable[[], None]):
        """
        Initialize the watchdog timer.
        
        Args:
            timeout: Timeout in seconds
            callback: Function to call when timeout is reached
        """
        self.timeout = timeout
        self.callback = callback
        self.timer = None
        self.is_running = False
        self.last_reset = 0
        self.lock = threading.Lock()
    
    def start(self):
        """Start the watchdog timer."""
        with self.lock:
            if self.is_running:
                return
            
            self.is_running = True
            self.last_reset = time.time()
            self._schedule()
    
    def stop(self):
        """Stop the watchdog timer."""
        with self.lock:
            if not self.is_running:
                return
            
            self.is_running = False
            if self.timer:
                self.timer.cancel()
                self.timer = None
    
    def reset(self):
        """Reset the watchdog timer."""
        with self.lock:
            if not self.is_running:
                return
            
            self.last_reset = time.time()
            if self.timer:
                self.timer.cancel()
            
            self._schedule()
    
    def _schedule(self):
        """Schedule the watchdog timer."""
        if not self.is_running:
            return
        
        self.timer = threading.Timer(self.timeout, self._timeout)
        self.timer.daemon = True
        self.timer.start()
    
    def _timeout(self):
        """Handle watchdog timeout."""
        with self.lock:
            if not self.is_running:
                return
            
            # Call the callback
            try:
                self.callback()
            except Exception as e:
                print(f"Error in watchdog callback: {e}")
            
            # Reschedule
            self._schedule()

class EnhancedTradingBot:
    """
    Main trading bot class that integrates all components and manages the trading lifecycle.
    """
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the Enhanced Trading Bot.
        
        Args:
            config_path: Path to the configuration file
        """
        # Store config path
        self.config_path = config_path
        
        # Setup logging
        self.logger = self._setup_logger()
        self.logger.info("Initializing Enhanced Hyperliquid Trading Bot...")    
        
        # Load configuration
        self.config_manager = ConfigManager(config_path, self.logger)
        self.config = self.config_manager.get_config()
        
        # Initialize error handler
        self.error_handler = ErrorHandler(self.logger)
        
        # Initialize exchange adapter
        self.exchange = HyperliquidExchangeAdapter(
            self.config_path
        )
        
        # Initialize sentiment analyzer if enabled
        self.sentiment_analyzer = None
        if self.config.get("use_sentiment_analysis", True):
            self.sentiment_analyzer = LLMSentimentAnalyzer(
                self.config,
                self.logger
            )
            
        # Runtime variables
        self.running = False
        self.last_trade_time = 0
        self.market_data = {}
        self.positions = {}
        self.last_successful_cycle = 0
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        self.error_backoff_base = 2  # seconds
        
        # State recovery
        self.state_file = "bot_state.json"
        self.load_state()
        
        # Watchdog timer
        self.watchdog_timeout = self.config.get("watchdog_timeout", 60)  # seconds
        self.watchdog = WatchdogTimer(self.watchdog_timeout, self._watchdog_callback)
        
        # Signal handlers
        self._setup_signal_handlers()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up the logger."""
        logger = logging.getLogger("EnhancedTradingBot")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("hyperliquid_bot.log", mode="a")
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            self.logger.info("Signal handlers registered")
        except Exception as e:
            self.logger.warning(f"Failed to register signal handlers: {e}")
    
    def _signal_handler(self, sig, frame):
        """Handle signals for graceful shutdown."""
        self.logger.info(f"Received signal {sig}, shutting down...")
        self.running = False
    
    def _watchdog_callback(self):
        """Watchdog callback to detect and recover from frozen states."""
        current_time = time.time()
        time_since_last_cycle = current_time - self.last_successful_cycle
        
        if time_since_last_cycle > self.watchdog_timeout:
            self.logger.warning(f"Watchdog detected potential freeze: {time_since_last_cycle:.2f}s since last successful cycle")
            
            # Try to recover
            self.logger.info("Attempting recovery...")
            
            # Save state
            self.save_state()
            
            # Force reconnection to exchange
            try:
                self.exchange.reload_config()
                self.logger.info("Forced exchange reconnection")
            except Exception as e:
                self.logger.error(f"Failed to reconnect to exchange: {e}")
    
    def save_state(self):
        """Save bot state to file."""
        try:
            state = {
                "last_trade_time": self.last_trade_time,
                "market_data": self.market_data,
                "positions": self.positions,
                "timestamp": time.time()
            }
            
            with open(self.state_file, "w") as f:
                json.dump(state, f)
                
            self.logger.info("Bot state saved")
        except Exception as e:
            self.logger.error(f"Failed to save bot state: {e}")
    
    def load_state(self):
        """Load bot state from file."""
        try:
            if not os.path.exists(self.state_file):
                self.logger.info("No state file found, starting fresh")
                return
                
            with open(self.state_file, "r") as f:
                state = json.load(f)
                
            # Check if state is recent enough (within 1 hour)
            if time.time() - state.get("timestamp", 0) > 3600:
                self.logger.info("State file too old, starting fresh")
                return
                
            self.last_trade_time = state.get("last_trade_time", 0)
            self.market_data = state.get("market_data", {})
            self.positions = state.get("positions", {})
            
            self.logger.info("Bot state loaded")
        except Exception as e:
            self.logger.error(f"Failed to load bot state: {e}")
    
    async def start(self):
        """Start the trading bot."""
        if self.running:
            self.logger.warning("Bot is already running")
            return
            
        self.running = True
        self.logger.info("Starting trading bot...")
        
        # Start watchdog
        self.watchdog.start()
        
        try:
            # Initialize exchange connection
            await self.initialize_exchange()
            
            # Main trading loop with resilience
            await self.run_trading_loop()
                    
        except Exception as e:
            self.logger.error(f"Critical error in start method: {str(e)}")
            self.logger.error(traceback.format_exc())
        finally:
            # Stop watchdog
            self.watchdog.stop()
            
            # Save final state
            self.save_state()
            
            self.running = False
            self.logger.info("Trading bot stopped")
    
    async def initialize_exchange(self):
        """Initialize exchange connection with retry logic."""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Initialize exchange connection
                self.logger.info(f"Initializing exchange connection (attempt {retry_count + 1}/{max_retries})...")
                await self.exchange.initialize()
                self.logger.info("Exchange connection initialized successfully")
                return
            except Exception as e:
                retry_count += 1
                wait_time = self.error_backoff_base ** retry_count
                
                self.logger.error(f"Failed to initialize exchange connection: {e}")
                if retry_count < max_retries:
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error("Max retries reached, giving up")
                    raise
    
    async def run_trading_loop(self):
        """Run the main trading loop with resilience."""
        self.logger.info("Starting main trading loop...")
        
        while self.running:
            try:
                # Reset watchdog
                self.watchdog.reset()
                
                # Fetch market data
                await self.fetch_market_data()
                
                # Check positions
                await self.check_positions()
                
                # Generate trading signals
                await self.generate_signals()
                
                # Execute trades if needed
                await self.execute_trades()
                
                # Update successful cycle time
                self.last_successful_cycle = time.time()
                
                # Reset consecutive errors
                self.consecutive_errors = 0
                
                # Save state periodically
                if time.time() - self.last_trade_time > 300:  # 5 minutes
                    self.save_state()
                
                # Wait for next cycle
                poll_interval = self.config.get("poll_interval_seconds", 5)
                await asyncio.sleep(poll_interval)
                
            except asyncio.CancelledError:
                self.logger.info("Trading loop cancelled")
                break
            except Exception as e:
                # Handle errors with exponential backoff
                self.consecutive_errors += 1
                wait_time = min(60, self.error_backoff_base ** self.consecutive_errors)
                
                self.logger.error(f"Error in trading loop (attempt {self.consecutive_errors}): {str(e)}")
                self.logger.error(traceback.format_exc())
                
                # Check if we should continue
                if self.consecutive_errors >= self.max_consecutive_errors:
                    self.logger.error(f"Too many consecutive errors ({self.consecutive_errors}), stopping bot")
                    self.running = False
                    break
                
                self.logger.info(f"Waiting {wait_time} seconds before retrying...")
                await asyncio.sleep(wait_time)
    
    async def stop(self):
        """Stop the trading bot."""
        if not self.running:
            self.logger.warning("Bot is not running")
            return
            
        self.logger.info("Stopping trading bot...")
        self.running = False
        
        # Save state
        self.save_state()
    
    async def fetch_market_data(self):
        """Fetch market data from the exchange."""
        self.logger.info("Fetching market data...")
        
        # Get symbols to monitor
        symbols = self.config.get("symbols", ["BTC", "ETH", "SOL"])
        
        for symbol in symbols:
            try:
                # Fetch market data for symbol
                market_data = await self.exchange.fetch_market_data(symbol)
                
                # Store market data
                self.market_data[symbol] = market_data
                
                self.logger.info(f"Market data for {symbol}: price={market_data['price']}, funding_rate={market_data.get('funding_rate', 0)}")
                
            except Exception as e:
                error_info = self.error_handler.handle_error(e, {
                    "function": "fetch_market_data",
                    "symbol": symbol
                })
                
                self.logger.error(f"Error fetching market data for {symbol}: {str(e)}")
                
                # Use cached data if available
                if symbol in self.market_data:
                    self.logger.info(f"Using cached market data for {symbol}")
                else:
                    self.logger.warning(f"No market data available for {symbol}")
    
    async def check_positions(self):
        """Check current positions."""
        self.logger.info("Checking positions...")
        
        try:
            # Fetch positions from exchange
            positions = await self.exchange.get_user_positions()
            
            # Store positions
            self.positions = positions
            
            # Log positions
            for symbol, position in positions.items():
                self.logger.info(f"Position for {symbol}: size={position['size']}, entry_price={position['entry_price']}, pnl={position['pnl']}")
                
        except Exception as e:
            error_info = self.error_handler.handle_error(e, {
                "function": "check_positions"
            })
            
            self.logger.error(f"Error checking positions: {str(e)}")
            
            # Use cached positions if available
            if self.positions:
                self.logger.info("Using cached positions")
            else:
                self.logger.warning("No position data available")
    
    async def generate_signals(self):
        """Generate trading signals."""
        self.logger.info("Generating trading signals...")
        
        # Get symbols to monitor
        symbols = self.config.get("symbols", ["BTC", "ETH", "SOL"])
        
        for symbol in symbols:
            try:
                if symbol not in self.market_data:
                    self.logger.warning(f"No market data for {symbol}, skipping signal generation")
                    continue
                    
                # Get market data for symbol
                market_data = self.market_data[symbol]
                
                # Generate signal using sentiment analysis if available
                sentiment_signal = None
                if self.sentiment_analyzer:
                    try:
                        sentiment_signal = await self.sentiment_analyzer.analyze_market_sentiment(symbol)
                    except Exception as e:
                        self.logger.error(f"Error in sentiment analysis for {symbol}: {str(e)}")
                        sentiment_signal = None
                    
                # TODO: Implement signal generation logic
                # For now, just log a placeholder signal
                signal = "none"
                quantity = 0.0
                
                self.logger.info(f"Generated signal for {symbol}: {signal} with quantity {quantity}")
                
            except Exception as e:
                error_info = self.error_handler.handle_error(e, {
                    "function": "generate_signals",
                    "symbol": symbol
                })
                
                self.logger.error(f"Error generating signal for {symbol}: {str(e)}")
    
    async def execute_trades(self):
        """Execute trades based on signals."""
        self.logger.info("Executing trades...")
        
        # Check if enough time has passed since last trade
        current_time = time.time()
        min_trade_interval = self.config.get("min_trade_interval", 60)
        
        if current_time - self.last_trade_time < min_trade_interval:
            self.logger.info(f"Skipping trade execution, min interval not reached ({min_trade_interval}s)")
            return
            
        # TODO: Implement trade execution logic
        # For now, just log a placeholder message
        self.logger.info("No trades executed")
        
        # Update last trade time
        self.last_trade_time = current_time

async def main():
    """Main entry point with resilience."""
    # Get config path from command line or use default
    config_path = "config.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # Setup root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("main_loop.log", mode="a")
        ]
    )
    
    logger = logging.getLogger("MainLoop")
    logger.info("Starting main loop...")
    
    # Create bot instance
    bot = None
    
    try:
        # Create bot
        bot = EnhancedTradingBot(config_path)
        
        # Start bot
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        if bot and bot.running:
            await bot.stop()
    except Exception as e:
        logger.error(f"Critical error in main loop: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Try to stop bot gracefully
        if bot and bot.running:
            try:
                await bot.stop()
            except Exception as stop_error:
                logger.error(f"Error stopping bot: {stop_error}")
        
        # Exit with error code
        sys.exit(1)
    
    logger.info("Main loop exited")

if __name__ == "__main__":
    # Use asyncio.run with error handling
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
