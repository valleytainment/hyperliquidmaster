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
from typing import Dict, List, Optional, Any

# Import core components
from core.hyperliquid_adapter import HyperliquidAdapter as HyperliquidExchangeAdapter
from core.error_handler import ErrorHandler
from sentiment.llm_analyzer import LLMSentimentAnalyzer
from config_compatibility import ConfigManager

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
    
    async def start(self):
        """Start the trading bot."""
        if self.running:
            self.logger.warning("Bot is already running")
            return
            
        self.running = True
        self.logger.info("Starting trading bot...")
        
        try:
            # Initialize exchange connection
            await self.exchange.initialize()
            
            # Main trading loop
            while self.running:
                try:
                    # Fetch market data
                    await self.fetch_market_data()
                    
                    # Check positions
                    await self.check_positions()
                    
                    # Generate trading signals
                    await self.generate_signals()
                    
                    # Execute trades if needed
                    await self.execute_trades()
                    
                    # Wait for next cycle
                    await asyncio.sleep(self.config.get("poll_interval_seconds", 5))
                    
                except Exception as e:
                    self.logger.error(f"Error in trading loop: {str(e)}")
                    await asyncio.sleep(5)  # Wait before retrying
                    
        except Exception as e:
            self.logger.error(f"Critical error: {str(e)}")
            self.running = False
            
        self.logger.info("Trading bot stopped")
    
    async def stop(self):
        """Stop the trading bot."""
        if not self.running:
            self.logger.warning("Bot is not running")
            return
            
        self.logger.info("Stopping trading bot...")
        self.running = False
    
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
                self.logger.error(f"Error fetching market data for {symbol}: {str(e)}")
    
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
            self.logger.error(f"Error checking positions: {str(e)}")
    
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
                    sentiment_signal = await self.sentiment_analyzer.analyze_market_sentiment(symbol)
                    
                # TODO: Implement signal generation logic
                # For now, just log a placeholder signal
                signal = "none"
                quantity = 0.0
                
                self.logger.info(f"Generated signal for {symbol}: {signal} with quantity {quantity}")
                
            except Exception as e:
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
    """Main entry point."""
    # Get config path from command line or use default
    config_path = "config.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        
    # Create and start bot
    bot = EnhancedTradingBot(config_path)
    
    # Register signal handlers for graceful shutdown
    try:
        # Start bot
        await bot.start()
    except KeyboardInterrupt:
        # Stop bot on keyboard interrupt
        await bot.stop()

if __name__ == "__main__":
    asyncio.run(main())
