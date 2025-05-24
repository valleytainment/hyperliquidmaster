"""
Enhanced Hyperliquid Trading Bot - Main Application

This is the main entry point for the enhanced trading bot, integrating all components
into a complete, production-ready cryptocurrency trading system.
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from typing import Dict, List, Optional, Any

# Import core components
from core.hyperliquid_adapter import HyperliquidExchangeAdapter
from core.error_handler import ErrorHandler
from sentiment.llm_analyzer import LLMSentimentAnalyzer
from config_compatibility import ConfigManager

class EnhancedTradingBot:
    """
    Main trading bot class that integrates all components and manages the trading lifecycle.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the trading bot.
        
        Args:
            config_path: Path to the configuration file
        """
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
            self.config,
            self.logger,
            self.error_handler
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
        self.last_data_update = 0
        self.last_sentiment_update = 0
        self.market_data = {}
        self.sentiment_data = {}
        self.positions = {}
        
        # Signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._signal_handler)
            
    def _setup_logger(self) -> logging.Logger:
        """
        Set up the logger.
        
        Returns:
            Configured logger
        """
        logger = logging.getLogger("EnhancedTradingBot")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("enhanced_bot.log", mode="a")
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
        
    def _signal_handler(self, sig, frame):
        """
        Handle termination signals.
        
        Args:
            sig: Signal number
            frame: Current stack frame
        """
        self.logger.info(f"Received signal {sig}, initiating shutdown...")
        self.running = False
        
    async def start(self):
        """Start the trading bot."""
        self.logger.info("Starting Enhanced Hyperliquid Trading Bot...")
        self.running = True
        
        try:
            # Initialize exchange connection
            await self.exchange.initialize()
            
            # Start main loop
            await self._main_loop()
        except Exception as e:
            self.logger.exception(f"Error in main bot execution: {str(e)}")
        finally:
            await self.shutdown()
            
    async def shutdown(self):
        """Gracefully shut down the trading bot."""
        self.logger.info("Shutting down Enhanced Hyperliquid Trading Bot...")
        self.running = False
        
        # Close exchange connection
        await self.exchange.close()
        
        self.logger.info("Trading bot shutdown complete.")
        
    async def _main_loop(self):
        """Main trading loop."""
        self.logger.info("Entering main trading loop...")
        
        # Warm-up period
        await self._warm_up()
        
        while self.running:
            try:
                # Update market data
                await self._update_market_data()
                
                # Update sentiment data (less frequently)
                await self._update_sentiment_data()
                
                # Update positions
                await self._update_positions()
                
                # Display status
                self._display_status()
                
                # Sleep to avoid excessive API calls
                await asyncio.sleep(self.config.get("main_loop_interval", 5))
                
            except Exception as e:
                self.logger.error(f"Error in main loop iteration: {str(e)}")
                await asyncio.sleep(10)  # Longer sleep on error
                
    async def _warm_up(self):
        """Warm-up period to gather initial data."""
        warm_up_duration = self.config.get("warm_up_duration", 20)
        self.logger.info(f"Starting warm-up period ({warm_up_duration} seconds)...")
        
        # Get initial market data
        symbols = self.config.get("symbols", [])
        if not symbols:
            self.logger.warning("No symbols configured for trading")
            symbols = [self.config.get("trade_symbol", "BTC-USD-PERP")]
            
        # Fetch initial market data
        self.market_data = await self.exchange.fetch_market_data(symbols)
        
        # Fetch initial positions
        self.positions = await self.exchange.get_user_positions()
        
        # Wait for warm-up period to complete
        await asyncio.sleep(warm_up_duration)
        
        self.logger.info("Warm-up period complete.")
        
    async def _update_market_data(self):
        """Update market data."""
        current_time = time.time()
        update_interval = self.config.get("data_update_interval", 5)
        
        if current_time - self.last_data_update < update_interval:
            return
            
        symbols = self.config.get("symbols", [])
        if not symbols:
            return
            
        try:
            # Fetch market data
            self.market_data = await self.exchange.fetch_market_data(symbols)
            self.last_data_update = current_time
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {str(e)}")
            
    async def _update_sentiment_data(self):
        """Update sentiment data."""
        if not self.sentiment_analyzer:
            return
            
        current_time = time.time()
        update_interval = self.config.get("sentiment_update_interval", 300)  # 5 minutes default
        
        if current_time - self.last_sentiment_update < update_interval:
            return
            
        try:
            # For demonstration purposes, we'll use empty lists
            # In a real implementation, you would fetch actual news and social media data
            news_items = []
            social_posts = []
            
            # Analyze sentiment
            self.sentiment_data = await self.sentiment_analyzer.detect_market_narratives(
                news_items, social_posts
            )
            
            self.last_sentiment_update = current_time
            
        except Exception as e:
            self.logger.error(f"Error updating sentiment data: {str(e)}")
            
    async def _update_positions(self):
        """Update positions."""
        try:
            self.positions = await self.exchange.get_user_positions()
        except Exception as e:
            self.logger.error(f"Error updating positions: {str(e)}")
            
    def _display_status(self):
        """Display current status."""
        # Display market data
        for symbol, data in self.market_data.items():
            self.logger.info(f"{symbol}: Price=${data.get('price', 0):.2f}, Funding={data.get('funding_rate', 0):.6f}")
            
        # Display positions
        if self.positions:
            self.logger.info("Current positions:")
            for symbol, position in self.positions.items():
                self.logger.info(f"  {symbol}: {position['side']} {position['size']:.4f} @ {position['entry_price']:.2f}")
        else:
            self.logger.info("No open positions")
            
        # Display sentiment data
        if self.sentiment_data:
            market_regime = self.sentiment_data.get("market_regime", "unknown")
            confidence = self.sentiment_data.get("confidence", 0)
            self.logger.info(f"Market regime: {market_regime} (confidence: {confidence:.2f})")
            
            narratives = self.sentiment_data.get("narratives", [])
            if narratives:
                self.logger.info("Top market narratives:")
                for narrative in narratives[:2]:
                    self.logger.info(f"  {narrative.get('theme')}: {narrative.get('sentiment')} (impact: {narrative.get('impact', 0):.2f})")

async def main():
    """Main entry point."""
    # Get configuration path from command line arguments
    config_path = "config.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        
    # Create and start the bot
    bot = EnhancedTradingBot(config_path)
    await bot.start()

if __name__ == "__main__":
    asyncio.run(main())
