#!/usr/bin/env python3
"""
Main entry point for the Hyperliquid Trading Bot.

This script initializes and runs the integrated trading bot with all features
from all branches, ensuring no loss of functionality or settings.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Set up base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Add directories to path
sys.path.append(str(BASE_DIR))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(BASE_DIR, "logs", "hyperliquid_bot.log"))
    ]
)
logger = logging.getLogger(__name__)

def ensure_directories():
    """Ensure all required directories exist."""
    dirs = ["logs", "data/cache", "config/backup"]
    for dir_path in dirs:
        os.makedirs(os.path.join(BASE_DIR, dir_path), exist_ok=True)
    logger.info("Directory structure verified")

def load_config():
    """Load configuration from config files."""
    config = {}
    
    # Load trading mode
    try:
        with open(os.path.join(BASE_DIR, "config", "trading_mode.json"), "r") as f:
            config["trading_mode"] = json.load(f)
        logger.info(f"Trading mode loaded: {config['trading_mode']['trading_mode']}")
    except Exception as e:
        logger.error(f"Failed to load trading mode: {e}")
        config["trading_mode"] = {"trading_mode": "PAPER_TRADING"}
    
    # Load mode settings
    try:
        with open(os.path.join(BASE_DIR, "config", "mode_settings.json"), "r") as f:
            config["mode_settings"] = json.load(f)
        logger.info(f"Mode settings loaded for {len(config['mode_settings'])} modes")
    except Exception as e:
        logger.error(f"Failed to load mode settings: {e}")
        config["mode_settings"] = {}
    
    # Load risk metrics
    try:
        with open(os.path.join(BASE_DIR, "config", "risk_metrics.json"), "r") as f:
            config["risk_metrics"] = json.load(f)
        logger.info("Risk metrics loaded")
    except Exception as e:
        logger.error(f"Failed to load risk metrics: {e}")
        config["risk_metrics"] = {}
    
    return config

def main():
    """Main entry point for the Hyperliquid Trading Bot."""
    logger.info("Starting Hyperliquid Trading Bot - Integrated Version")
    
    # Ensure directories exist
    ensure_directories()
    
    # Load configuration
    config = load_config()
    
    # Determine which interface to launch based on arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--headless":
        logger.info("Starting in headless mode")
        from tests.headless_gui_test import run_headless_test
        run_headless_test()
    else:
        logger.info("Starting GUI interface")
        try:
            # Try to import the optimized GUI first
            from gui.gui_main_optimized_fixed import run_gui
            logger.info("Using optimized GUI")
        except ImportError:
            # Fall back to standard GUI if optimized is not available
            from gui.gui_main import run_gui
            logger.info("Using standard GUI")
        
        # Run the GUI with the loaded configuration
        run_gui(config)

if __name__ == "__main__":
    main()
