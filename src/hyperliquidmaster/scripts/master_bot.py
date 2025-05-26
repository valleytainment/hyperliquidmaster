"""
Main entry point for the hyperliquidmaster trading bot.

This module provides the main entry point for running the trading bot,
with support for both GUI and headless modes.
"""

import os
import sys
import argparse
import logging
import asyncio
import time
import signal
from pathlib import Path
from typing import Optional, Dict, Any

from hyperliquidmaster.config import load_config, BotSettings
from hyperliquidmaster.core.hyperliquid_adapter import HyperliquidAdapter
from hyperliquidmaster.core.error_handler import ErrorHandler
from hyperliquidmaster.risk import RiskManager
from hyperliquidmaster.events import event_bus, EVENT_MARKET_DATA, EVENT_STATUS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hyperliquid_bot.log")
    ]
)
logger = logging.getLogger(__name__)

class EnhancedTradingBot:
    """
    Main trading bot class that integrates all components and manages the trading lifecycle.
    """
    
    def __init__(self, settings: BotSettings):
        """
        Initialize the trading bot.
        
        Args:
            settings: Bot configuration settings
        """
        # Store settings
        self.settings = settings
        
        # Initialize error handler
        self.error_handler = ErrorHandler(logger)
        
        # Initialize exchange adapter
        self.exchange = HyperliquidAdapter(self.settings)
        
        # Initialize risk manager
        self.risk_manager = RiskManager(self.settings)
        
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
            
    def _signal_handler(self, sig, frame):
        """
        Handle termination signals.
        
        Args:
            sig: Signal number
            frame: Current stack frame
        """
        logger.info(f"Received signal {sig}, initiating shutdown...")
        self.running = False
        
    async def start(self):
        """Start the trading bot."""
        logger.info("Starting Enhanced Hyperliquid Trading Bot...")
        self.running = True
        
        try:
            # Start main loop
            await self._main_loop()
        except Exception as e:
            logger.exception(f"Error in main bot execution: {str(e)}")
        finally:
            await self.shutdown()
            
    async def shutdown(self):
        """Gracefully shut down the trading bot."""
        logger.info("Shutting down Enhanced Hyperliquid Trading Bot...")
        self.running = False
        
        # Publish shutdown event
        await event_bus.publish(EVENT_STATUS, {
            "type": "shutdown"
        })
        
        logger.info("Trading bot shutdown complete.")
        
    async def _main_loop(self):
        """Main trading loop."""
        logger.info("Entering main trading loop...")
        
        # Warm-up period
        await self._warm_up()
        
        while self.running:
            try:
                # Update market data
                await self._update_market_data()
                
                # Update positions
                await self._update_positions()
                
                # Display status
                self._display_status()
                
                # Sleep to avoid excessive API calls
                await asyncio.sleep(self.settings.data_update_interval)
                
            except Exception as e:
                logger.error(f"Error in main loop iteration: {str(e)}")
                await asyncio.sleep(10)  # Longer sleep on error
                
    async def _warm_up(self):
        """Warm-up period to gather initial data."""
        warm_up_duration = 20  # seconds
        logger.info(f"Starting warm-up period ({warm_up_duration} seconds)...")
        
        # Get initial market data
        symbol = self.settings.trade_symbol
        
        try:
            # Fetch initial market data
            market_data = await self.exchange.get_market_data(symbol)
            self.market_data[symbol] = market_data
            
            # Fetch initial positions
            account_info = await self.exchange.get_account_info()
            self.positions = account_info.get("positions", {})
            
            # Wait for warm-up period to complete
            await asyncio.sleep(warm_up_duration)
            
            logger.info("Warm-up period complete.")
        except Exception as e:
            logger.error(f"Error during warm-up: {str(e)}")
        
    async def _update_market_data(self):
        """Update market data."""
        current_time = time.time()
        update_interval = self.settings.data_update_interval
        
        if current_time - self.last_data_update < update_interval:
            return
            
        symbol = self.settings.trade_symbol
            
        try:
            # Fetch market data
            market_data = await self.exchange.get_market_data(symbol)
            self.market_data[symbol] = market_data
            self.last_data_update = current_time
            
            # Publish market data event
            await event_bus.publish(EVENT_MARKET_DATA, {
                "symbol": symbol,
                "data": market_data
            })
            
        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")
            
    async def _update_positions(self):
        """Update positions."""
        try:
            account_info = await self.exchange.get_account_info()
            self.positions = account_info.get("positions", {})
        except Exception as e:
            logger.error(f"Error updating positions: {str(e)}")
            
    def _display_status(self):
        """Display current status."""
        # Display market data
        for symbol, data in self.market_data.items():
            logger.info(f"{symbol}: Price=${data.get('last_price', 0):.2f}")
            
        # Display positions
        if self.positions:
            logger.info("Current positions:")
            for symbol, position in self.positions.items():
                logger.info(f"  {symbol}: {position['side']} {position['size']:.4f} @ {position['entry_price']:.2f}")
        else:
            logger.info("No open positions")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hyperliquid Trading Bot")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/example.json",
        help="Path to configuration file (JSON or YAML)"
    )
    parser.add_argument(
        "--headless", 
        action="store_true",
        help="Run in headless mode (no GUI)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    return parser.parse_args()

async def run_headless(settings: BotSettings) -> None:
    """
    Run the bot in headless mode.
    
    Args:
        settings: Bot configuration settings
    """
    logger.info("Starting bot in headless mode")
    
    # Initialize event bus
    event_bus.start()
    
    # Initialize and run bot
    bot = EnhancedTradingBot(settings)
    await bot.start()
    
    # Stop event bus
    event_bus.stop()

def run_gui(settings: BotSettings) -> None:
    """
    Run the bot with GUI.
    
    Args:
        settings: Bot configuration settings
    """
    logger.info("Starting bot with GUI")
    
    try:
        # Import GUI components here to avoid dependencies in headless mode
        from hyperliquidmaster.ui.bot_ui import launch_gui
        launch_gui(settings)
    except ImportError as e:
        logger.error(f"Failed to import GUI components: {e}")
        logger.info("Falling back to headless mode")
        asyncio.run(run_headless(settings))

def main() -> None:
    """Main entry point for the trading bot."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    try:
        # Load and validate configuration
        logger.info(f"Loading configuration from {args.config}")
        settings = load_config(args.config)
        logger.info("Configuration loaded successfully")
        
        # Run in appropriate mode
        if args.headless:
            asyncio.run(run_headless(settings))
        else:
            run_gui(settings)
            
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
