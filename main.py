#!/usr/bin/env python3
"""
Main entry point for the Hyperliquid Trading Bot.

This script launches the trading bot in either GUI or headless mode.
It handles command-line arguments, environment setup, and application initialization.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f"hyperliquid_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Hyperliquid Trading Bot")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI)")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

def run_gui_mode(args):
    """Run in GUI mode"""
    try:
        logger.info("Starting Hyperliquid Trading Bot in GUI mode")
        
        # Import GUI module
        from gui.enhanced_gui import main as gui_main
        
        # Run GUI
        gui_main()
    except ImportError:
        logger.error("Failed to import GUI module. Make sure all dependencies are installed.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running GUI mode: {str(e)}")
        sys.exit(1)

def run_headless_mode(args):
    """Run in headless mode"""
    try:
        logger.info("Starting Hyperliquid Trading Bot in headless mode")
        
        # Import headless module
        from tests.headless_test import main as headless_main
        
        # Run headless
        headless_main()
    except ImportError:
        logger.error("Failed to import headless module. Make sure all dependencies are installed.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running headless mode: {str(e)}")
        sys.exit(1)

def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Run in appropriate mode
    if args.headless:
        run_headless_mode(args)
    else:
        run_gui_mode(args)

if __name__ == "__main__":
    main()
