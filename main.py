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
from core.trading_mode import TradingModeManager, TradingMode
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
        
        # Initialize trading mode manager
        self.mode_manager = TradingModeManager(config_path, self.logger)
        self.logger.info(f"Current trading mode: {self.mode_manager.get_current_mode().value}")
        
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
                "timestamp": time.time(),
                "trading_mode": self.mode_manager.get_current_mode().value
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
            
            # Restore trading mode if present
            if "trading_mode" in state:
                try:
                    self.mode_manager.set_mode(state["trading_mode"])
                    self.logger.info(f"Restored trading mode: {state['trading_mode']}")
                except Exception as e:
                    self.logger.error(f"Failed to restore trading mode: {e}")
            
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
        self.logger.info(f"Trading mode: {self.mode_manager.get_current_mode().value}")
        
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
        """Generate trading signals using advanced strategy logic."""
        self.logger.info("Generating trading signals...")
        
        # Get symbols to monitor
        symbols = self.config.get("symbols", ["BTC", "ETH", "SOL"])
        
        # Initialize signals dictionary if not exists
        if not hasattr(self, 'signals') or self.signals is None:
            self.signals = {}
        
        # Initialize strategy if not exists
        if not hasattr(self, 'strategy'):
            from strategies.master_omni_overlord import MasterOmniOverlordStrategy
            self.strategy = MasterOmniOverlordStrategy(self.config, self.logger)
        
        for symbol in symbols:
            try:
                if symbol not in self.market_data:
                    self.logger.warning(f"No market data available for {symbol}, skipping signal generation")
                    continue
                
                # Get market data for the symbol
                market_data = self.market_data[symbol]
                
                # Check if we have historical data
                if 'historical_data' not in market_data or market_data['historical_data'] is None:
                    # Fetch historical data if not available
                    try:
                        historical_data = await self.exchange.fetch_historical_data(symbol)
                        if historical_data is not None and not isinstance(historical_data, dict):
                            # Convert to DataFrame if it's not already
                            if not isinstance(historical_data, pd.DataFrame):
                                # Assume it's a list of dictionaries
                                import pandas as pd
                                historical_data = pd.DataFrame(historical_data)
                            
                            # Add to market data
                            self.market_data[symbol]['historical_data'] = historical_data
                        else:
                            self.logger.warning(f"Could not fetch historical data for {symbol}")
                    except Exception as e:
                        self.logger.error(f"Error fetching historical data for {symbol}: {e}")
                
                # Prepare market data with indicators
                from strategies.strategy_integration import StrategyIntegration
                enhanced_market_data = StrategyIntegration.prepare_market_data(self.market_data[symbol])
                self.market_data[symbol] = enhanced_market_data
                
                # Detect market regime if we have historical data
                if 'historical_data' in enhanced_market_data and enhanced_market_data['historical_data'] is not None:
                    try:
                        market_regime = self.strategy.detect_market_regime(enhanced_market_data['historical_data'])
                        self.logger.info(f"Market regime for {symbol}: {market_regime}")
                    except Exception as e:
                        self.logger.error(f"Error detecting market regime for {symbol}: {e}")
                        market_regime = "unknown"
                else:
                    market_regime = "unknown"
                
                # Adjust strategy parameters based on market conditions
                self.strategy.adjust_adaptive_parameters()
                self.strategy.adjust_strategy_weights()
                
                # Generate signal
                signal = {
                    "symbol": symbol,
                    "timestamp": time.time(),
                    "price": enhanced_market_data.get("price", 0),
                    "signal": "NEUTRAL",  # Default signal
                    "strength": 0,
                    "confidence": 0,
                    "regime": market_regime,
                    "indicators": enhanced_market_data.get("indicators", {})
                }
                
                # Get position size if we have a signal
                current_position = self.positions.get(symbol, {"size": 0})
                current_position_size = current_position.get("size", 0)
                
                # Generate signals from sub-strategies
                try:
                    # Get signals from each sub-strategy
                    tc_signal = self.strategy.triple_confluence.generate_signal(enhanced_market_data)
                    ou_signal = self.strategy.oracle_update.generate_signal(enhanced_market_data)
                    
                    # Combine signals using strategy weights
                    tc_weight = self.strategy.strategy_weights.get("triple_confluence", 0.5)
                    ou_weight = self.strategy.strategy_weights.get("oracle_update", 0.5)
                    
                    # Calculate weighted signal strength
                    signal_strength = (
                        tc_signal.get("strength", 0) * tc_weight +
                        ou_signal.get("strength", 0) * ou_weight
                    )
                    
                    # Calculate weighted confidence
                    confidence = (
                        tc_signal.get("confidence", 0) * tc_weight +
                        ou_signal.get("confidence", 0) * ou_weight
                    )
                    
                    # Determine signal direction
                    if signal_strength > self.strategy.adaptive_params.get("signal_threshold", 0.7):
                        signal["signal"] = "LONG"
                        signal["strength"] = signal_strength
                        signal["confidence"] = confidence
                    elif signal_strength < -self.strategy.adaptive_params.get("signal_threshold", 0.7):
                        signal["signal"] = "SHORT"
                        signal["strength"] = abs(signal_strength)
                        signal["confidence"] = confidence
                    else:
                        signal["signal"] = "NEUTRAL"
                        signal["strength"] = abs(signal_strength)
                        signal["confidence"] = confidence
                    
                    # Calculate target position size
                    if signal["signal"] != "NEUTRAL":
                        # Get current price and calculate stop loss
                        current_price = enhanced_market_data.get("price", 0)
                        
                        # Calculate stop loss based on ATR or support/resistance
                        if "indicators" in enhanced_market_data and enhanced_market_data["indicators"] is not None:
                            indicators = enhanced_market_data["indicators"]
                            if signal["signal"] == "LONG":
                                # For long positions, use support level or percentage-based stop
                                if "support_level" in indicators and indicators["support_level"] is not None:
                                    stop_loss = indicators["support_level"]
                                else:
                                    stop_loss = current_price * 0.98  # 2% default stop loss
                            else:
                                # For short positions, use resistance level or percentage-based stop
                                if "resistance_level" in indicators and indicators["resistance_level"] is not None:
                                    stop_loss = indicators["resistance_level"]
                                else:
                                    stop_loss = current_price * 1.02  # 2% default stop loss
                        else:
                            # Default stop loss if no indicators
                            stop_loss = current_price * (0.98 if signal["signal"] == "LONG" else 1.02)
                        
                        # Calculate position size
                        target_position_size = self.strategy.calculate_position_size(
                            symbol=symbol,
                            entry_price=current_price,
                            stop_loss=stop_loss
                        )
                        
                        # Adjust for direction
                        if signal["signal"] == "SHORT":
                            target_position_size = -target_position_size
                        
                        signal["target_position_size"] = target_position_size
                        signal["stop_loss"] = stop_loss
                        
                        # Calculate take profit based on risk-reward ratio
                        risk = abs(current_price - stop_loss)
                        reward_ratio = 2.0  # Default 1:2 risk-reward ratio
                        
                        if signal["signal"] == "LONG":
                            take_profit = current_price + (risk * reward_ratio)
                        else:
                            take_profit = current_price - (risk * reward_ratio)
                        
                        signal["take_profit"] = take_profit
                    
                    self.logger.info(f"Generated signal for {symbol}: {signal['signal']} (strength: {signal['strength']:.2f}, confidence: {signal['confidence']:.2f})")
                    
                    # Store signal
                    self.signals[symbol] = signal
                    
                except Exception as e:
                    self.logger.error(f"Error generating signal for {symbol}: {e}")
                    # Use neutral signal as fallback
                    self.signals[symbol] = signal
                
            except Exception as e:
                error_info = self.error_handler.handle_error(e, {
                    "function": "generate_signals",
                    "symbol": symbol
                })
                
                self.logger.error(f"Error generating signals for {symbol}: {str(e)}")
    
    async def execute_trades(self):
        """Execute trades based on generated signals."""
        # Skip trade execution if in monitor-only mode
        if self.mode_manager.get_current_mode() == TradingMode.MONITOR_ONLY:
            self.logger.info("Monitor-only mode active, skipping trade execution")
            return
            
        self.logger.info("Executing trades...")
        
        # Get symbols to monitor
        symbols = self.config.get("symbols", ["BTC", "ETH", "SOL"])
        
        # Check if we have signals
        if not hasattr(self, 'signals') or not self.signals:
            self.logger.warning("No signals available, skipping trade execution")
            return
        
        for symbol in symbols:
            try:
                # Skip if no signal for this symbol
                if symbol not in self.signals:
                    self.logger.info(f"No signal for {symbol}, skipping trade execution")
                    continue
                
                # Get signal
                signal = self.signals[symbol]
                
                # Skip if neutral signal
                if signal["signal"] == "NEUTRAL":
                    self.logger.info(f"Neutral signal for {symbol}, no trade execution needed")
                    continue
                
                # Get current position
                current_position = self.positions.get(symbol, {"size": 0})
                current_position_size = current_position.get("size", 0)
                
                # Get target position size
                target_position_size = signal.get("target_position_size", 0)
                
                # Calculate position delta
                position_delta = target_position_size - current_position_size
                
                # Skip if position delta is too small
                min_trade_size = self.config.get("min_trade_size", 0.001)
                if abs(position_delta) < min_trade_size:
                    self.logger.info(f"Position delta for {symbol} too small ({position_delta}), skipping trade")
                    continue
                
                # Determine if this is a new position, position increase, or position decrease
                is_new_position = current_position_size == 0
                is_position_increase = (current_position_size > 0 and position_delta > 0) or (current_position_size < 0 and position_delta < 0)
                is_position_decrease = (current_position_size > 0 and position_delta < 0) or (current_position_size < 0 and position_delta > 0)
                is_position_flip = (current_position_size > 0 and target_position_size < 0) or (current_position_size < 0 and target_position_size > 0)
                
                # Get current price
                current_price = self.market_data.get(symbol, {}).get("price", 0)
                if current_price == 0:
                    self.logger.warning(f"Invalid price for {symbol}, skipping trade")
                    continue
                
                # Execute trade based on position change type
                if is_new_position or is_position_increase:
                    # New position or increase existing position
                    is_buy = position_delta > 0
                    size = abs(position_delta)
                    
                    self.logger.info(f"Opening {'long' if is_buy else 'short'} position for {symbol}: {size} units at {current_price}")
                    
                    # Only execute real orders if in live trading mode
                    if self.mode_manager.is_real_trading():
                        # Place market order
                        order_result = await self.exchange.place_order(
                            symbol=symbol,
                            size=size,
                            price=None,  # Market order
                            is_buy=is_buy,
                            reduce_only=False
                        )
                        
                        if "error" in order_result:
                            self.logger.error(f"Error placing order for {symbol}: {order_result['error']}")
                        else:
                            self.logger.info(f"Order placed successfully for {symbol}")
                            
                            # Update last trade time
                            self.last_trade_time = time.time()
                            
                            # Set stop loss and take profit if available
                            if "stop_loss" in signal and signal["stop_loss"] > 0:
                                await self.exchange.place_stop_loss(
                                    symbol=symbol,
                                    size=size,
                                    stop_price=signal["stop_loss"],
                                    is_buy=not is_buy  # Opposite direction for stop loss
                                )
                                
                            if "take_profit" in signal and signal["take_profit"] > 0:
                                await self.exchange.place_take_profit(
                                    symbol=symbol,
                                    size=size,
                                    take_profit_price=signal["take_profit"],
                                    is_buy=not is_buy  # Opposite direction for take profit
                                )
                    else:
                        self.logger.info(f"Simulating {'buy' if is_buy else 'sell'} order for {symbol}: {size} units at {current_price}")
                        
                        # Simulate order execution
                        # In a real implementation, this would update a simulated portfolio
                        
                        # Update last trade time
                        self.last_trade_time = time.time()
                
                elif is_position_decrease and not is_position_flip:
                    # Decrease existing position
                    is_buy = position_delta > 0
                    size = abs(position_delta)
                    
                    self.logger.info(f"Reducing {'long' if current_position_size > 0 else 'short'} position for {symbol}: {size} units at {current_price}")
                    
                    # Only execute real orders if in live trading mode
                    if self.mode_manager.is_real_trading():
                        # Place market order with reduce_only=True
                        order_result = await self.exchange.place_order(
                            symbol=symbol,
                            size=size,
                            price=None,  # Market order
                            is_buy=is_buy,
                            reduce_only=True
                        )
                        
                        if "error" in order_result:
                            self.logger.error(f"Error placing order for {symbol}: {order_result['error']}")
                        else:
                            self.logger.info(f"Order placed successfully for {symbol}")
                            
                            # Update last trade time
                            self.last_trade_time = time.time()
                    else:
                        self.logger.info(f"Simulating {'buy' if is_buy else 'sell'} order for {symbol}: {size} units at {current_price}")
                        
                        # Simulate order execution
                        # In a real implementation, this would update a simulated portfolio
                        
                        # Update last trade time
                        self.last_trade_time = time.time()
                
                elif is_position_flip:
                    # Close existing position and open new one in opposite direction
                    
                    # First, close existing position
                    is_buy = current_position_size < 0  # Buy to close short, sell to close long
                    size = abs(current_position_size)
                    
                    self.logger.info(f"Closing {'long' if current_position_size > 0 else 'short'} position for {symbol}: {size} units at {current_price}")
                    
                    # Only execute real orders if in live trading mode
                    if self.mode_manager.is_real_trading():
                        # Place market order to close position
                        close_result = await self.exchange.place_order(
                            symbol=symbol,
                            size=size,
                            price=None,  # Market order
                            is_buy=is_buy,
                            reduce_only=True
                        )
                        
                        if "error" in close_result:
                            self.logger.error(f"Error closing position for {symbol}: {close_result['error']}")
                            continue  # Skip opening new position if closing failed
                        
                        # Then, open new position
                        is_buy = target_position_size > 0
                        size = abs(target_position_size)
                        
                        self.logger.info(f"Opening {'long' if is_buy else 'short'} position for {symbol}: {size} units at {current_price}")
                        
                        # Place market order for new position
                        open_result = await self.exchange.place_order(
                            symbol=symbol,
                            size=size,
                            price=None,  # Market order
                            is_buy=is_buy,
                            reduce_only=False
                        )
                        
                        if "error" in open_result:
                            self.logger.error(f"Error opening new position for {symbol}: {open_result['error']}")
                        else:
                            self.logger.info(f"New position opened successfully for {symbol}")
                            
                            # Update last trade time
                            self.last_trade_time = time.time()
                            
                            # Set stop loss and take profit if available
                            if "stop_loss" in signal and signal["stop_loss"] > 0:
                                await self.exchange.place_stop_loss(
                                    symbol=symbol,
                                    size=size,
                                    stop_price=signal["stop_loss"],
                                    is_buy=not is_buy  # Opposite direction for stop loss
                                )
                                
                            if "take_profit" in signal and signal["take_profit"] > 0:
                                await self.exchange.place_take_profit(
                                    symbol=symbol,
                                    size=size,
                                    take_profit_price=signal["take_profit"],
                                    is_buy=not is_buy  # Opposite direction for take profit
                                )
                    else:
                        self.logger.info(f"Simulating position flip for {symbol} from {current_position_size} to {target_position_size}")
                        
                        # Simulate order execution
                        # In a real implementation, this would update a simulated portfolio
                        
                        # Update last trade time
                        self.last_trade_time = time.time()
                
            except Exception as e:
                error_info = self.error_handler.handle_error(e, {
                    "function": "execute_trades",
                    "symbol": symbol
                })
                
                self.logger.error(f"Error executing trades for {symbol}: {str(e)}")
    
    # Mode management methods
    def get_current_mode(self) -> TradingMode:
        """
        Get the current trading mode.
        
        Returns:
            Current trading mode
        """
        return self.mode_manager.get_current_mode()
    
    def set_mode(self, mode: str) -> bool:
        """
        Set the current trading mode.
        
        Args:
            mode: Trading mode to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.mode_manager.set_mode(mode)
            if result:
                self.logger.info(f"Trading mode set to: {mode}")
                # Save state to persist mode change
                self.save_state()
            else:
                self.logger.error(f"Failed to set trading mode to: {mode}")
            return result
        except Exception as e:
            self.logger.error(f"Error setting trading mode: {e}")
            return False
    
    def get_available_modes(self) -> List[str]:
        """
        Get list of available trading modes.
        
        Returns:
            List of available trading modes
        """
        return self.mode_manager.get_available_modes()
    
    def get_mode_settings(self) -> Dict[str, Any]:
        """
        Get settings for current mode.
        
        Returns:
            Dict containing mode settings
        """
        return self.mode_manager.get_mode_settings()
    
    def update_mode_settings(self, settings: Dict[str, Any]) -> bool:
        """
        Update settings for current mode.
        
        Args:
            settings: Dict containing settings to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.mode_manager.update_mode_settings(settings)
            if result:
                self.logger.info(f"Mode settings updated: {settings}")
                # Save state to persist settings change
                self.save_state()
            else:
                self.logger.error(f"Failed to update mode settings: {settings}")
            return result
        except Exception as e:
            self.logger.error(f"Error updating mode settings: {e}")
            return False

# Main entry point
if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Hyperliquid Trading Bot")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--mode", type=str, help="Trading mode to use")
    args = parser.parse_args()
    
    # Create bot instance
    bot = EnhancedTradingBot(args.config)
    
    # Set trading mode if specified
    if args.mode:
        if bot.set_mode(args.mode):
            print(f"Trading mode set to: {args.mode}")
        else:
            print(f"Failed to set trading mode to: {args.mode}")
            print(f"Available modes: {', '.join(bot.get_available_modes())}")
            sys.exit(1)
    
    # Print current mode
    print(f"Current trading mode: {bot.get_current_mode().value}")
    print(f"Mode settings: {json.dumps(bot.get_mode_settings(), indent=2)}")
    
    # Run bot
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("Bot stopped by user")
    except Exception as e:
        print(f"Error running bot: {e}")
        traceback.print_exc()
