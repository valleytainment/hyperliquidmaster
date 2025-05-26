"""
Main entry point for the hyperliquidmaster trading bot.

This module provides the main entry point for running the trading bot,
with support for both GUI and headless modes.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from hyperliquidmaster.config import load_config, BotSettings
from hyperliquidmaster.core.hyperliquid_adapter import HyperliquidAdapter
from hyperliquidmaster.risk import RiskManager

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

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hyperliquid Trading Bot")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.json",
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

def run_headless(settings: BotSettings) -> None:
    """
    Run the bot in headless mode.
    
    Args:
        settings: Bot configuration settings
    """
    logger.info("Starting bot in headless mode")
    
    # Initialize components
    adapter = HyperliquidAdapter(settings)
    risk_manager = RiskManager(settings)
    
    # TODO: Initialize and run trading logic
    logger.info("Bot initialized successfully")
    
    # Placeholder for actual trading logic
    logger.info("Trading bot is running...")

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
        run_headless(settings)

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
            run_headless(settings)
        else:
            run_gui(settings)
            
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
