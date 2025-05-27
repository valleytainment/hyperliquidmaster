#!/usr/bin/env python3
"""
XRP Trading Test with Mode Management
------------------------------------
This script tests the HyperliquidMaster trading bot with different trading modes,
focusing on opening and closing both long and short positions for XRP.
"""

import os
import sys
import json
import time
import logging
import asyncio
import traceback
from typing import Dict, List, Any, Optional

# Import core components
from core.hyperliquid_adapter import HyperliquidAdapter
from core.trading_mode import TradingModeManager, TradingMode
from core.position_manager_wrapper import PositionManager
from core.error_handler import ErrorHandler

class XRPTradingTest:
    """
    Test class for XRP trading with different modes.
    Tests opening and closing both long and short positions.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the XRP trading test.
        
        Args:
            config_path: Path to the configuration file
        """
        # Setup logging
        self.logger = self._setup_logger()
        self.logger.info("Initializing XRP Trading Test with Mode Management...")
        
        # Store config path
        self.config_path = config_path
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize error handler
        self.error_handler = ErrorHandler(self.logger)
        
        # Initialize trading mode manager
        self.mode_manager = TradingModeManager(config_path, self.logger)
        
        # Initialize exchange adapter
        self.exchange = HyperliquidAdapter(config_path)
        
        # Initialize position manager
        self.position_manager = PositionManager(self.exchange, self.logger)
        
        # Test parameters
        self.symbol = "XRP"
        self.test_size = 0.1  # Small position size for testing
        self.test_modes = [
            TradingMode.PAPER_TRADING,
            TradingMode.LIVE_TRADING,
            TradingMode.AGGRESSIVE,
            TradingMode.CONSERVATIVE
        ]
        
        # Test results
        self.test_results = {
            "long_entry": {},
            "long_exit": {},
            "short_entry": {},
            "short_exit": {},
            "mode_switch": {}
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Set up the logger."""
        logger = logging.getLogger("XRPTradingTest")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("xrp_trading_test_with_modes.log", mode="w")
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Dict containing the configuration
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"Config file not found at {self.config_path}, using empty config")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}
    
    def _save_test_results(self):
        """Save test results to file."""
        try:
            with open("xrp_trading_test_results.json", "w") as f:
                json.dump(self.test_results, f, indent=2)
            self.logger.info("Test results saved to xrp_trading_test_results.json")
        except Exception as e:
            self.logger.error(f"Error saving test results: {e}")
    
    async def initialize(self):
        """Initialize the test environment."""
        self.logger.info("Initializing test environment...")
        
        try:
            # Initialize exchange connection
            self.logger.info("Initializing exchange connection...")
            await self.exchange.initialize()
            self.logger.info("Exchange connection initialized successfully")
            
            # Test connection
            test_result = self.exchange.test_connection()
            if "error" in test_result:
                self.logger.error(f"Connection test failed: {test_result['error']}")
                return False
            
            self.logger.info("Connection test successful")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing test environment: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    async def get_market_data(self):
        """Get market data for XRP."""
        self.logger.info(f"Getting market data for {self.symbol}...")
        
        try:
            market_data = self.exchange.get_market_data(self.symbol)
            
            if "error" in market_data:
                self.logger.error(f"Error getting market data: {market_data['error']}")
                return None
            
            self.logger.info(f"Market data for {self.symbol}: price={market_data['price']}, funding_rate={market_data.get('funding_rate', 0)}")
            return market_data
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    async def test_mode_switching(self):
        """Test switching between different trading modes."""
        self.logger.info("Testing mode switching...")
        
        for mode in self.test_modes:
            try:
                self.logger.info(f"Switching to mode: {mode.value}")
                result = self.mode_manager.set_mode(mode)
                
                if result:
                    self.logger.info(f"Successfully switched to mode: {mode.value}")
                    current_mode = self.mode_manager.get_current_mode()
                    mode_settings = self.mode_manager.get_mode_settings()
                    
                    self.logger.info(f"Current mode: {current_mode.value}")
                    self.logger.info(f"Mode settings: {json.dumps(mode_settings, indent=2)}")
                    
                    self.test_results["mode_switch"][mode.value] = {
                        "success": True,
                        "current_mode": current_mode.value,
                        "settings": mode_settings
                    }
                else:
                    self.logger.error(f"Failed to switch to mode: {mode.value}")
                    self.test_results["mode_switch"][mode.value] = {
                        "success": False,
                        "error": "Failed to switch mode"
                    }
            except Exception as e:
                self.logger.error(f"Error switching to mode {mode.value}: {e}")
                self.test_results["mode_switch"][mode.value] = {
                    "success": False,
                    "error": str(e)
                }
            
            # Wait a bit between mode switches
            await asyncio.sleep(1)
    
    async def test_long_position(self):
        """Test opening and closing a long position."""
        self.logger.info("Testing long position...")
        
        # Switch to paper trading mode for safety
        self.mode_manager.set_mode(TradingMode.PAPER_TRADING)
        self.logger.info(f"Switched to mode: {self.mode_manager.get_current_mode().value}")
        
        try:
            # Get current market data
            market_data = await self.get_market_data()
            if not market_data:
                self.logger.error("Failed to get market data, aborting long position test")
                self.test_results["long_entry"]["paper_trading"] = {
                    "success": False,
                    "error": "Failed to get market data"
                }
                return
            
            current_price = market_data["price"]
            self.logger.info(f"Current {self.symbol} price: {current_price}")
            
            # Open long position
            self.logger.info(f"Opening long position for {self.symbol} with size {self.test_size}...")
            
            try:
                # Use position manager to open position
                position_result = await self.position_manager.open_position(
                    symbol=self.symbol,
                    size=self.test_size,
                    is_long=True,
                    price=current_price * 1.001,  # Slightly above market for limit order
                    reduce_only=False
                )
                
                if position_result.get("success", False):
                    self.logger.info(f"Successfully opened long position: {position_result}")
                    self.test_results["long_entry"]["paper_trading"] = {
                        "success": True,
                        "price": current_price,
                        "size": self.test_size,
                        "order_id": position_result.get("order_id", "unknown"),
                        "timestamp": time.time()
                    }
                    
                    # Wait a bit before closing
                    self.logger.info("Waiting 5 seconds before closing position...")
                    await asyncio.sleep(5)
                    
                    # Close long position
                    self.logger.info(f"Closing long position for {self.symbol}...")
                    close_result = await self.position_manager.close_position(
                        symbol=self.symbol,
                        is_long=True,
                        price=current_price * 0.999  # Slightly below market for limit order
                    )
                    
                    if close_result.get("success", False):
                        self.logger.info(f"Successfully closed long position: {close_result}")
                        self.test_results["long_exit"]["paper_trading"] = {
                            "success": True,
                            "price": current_price,
                            "order_id": close_result.get("order_id", "unknown"),
                            "timestamp": time.time()
                        }
                    else:
                        self.logger.error(f"Failed to close long position: {close_result}")
                        self.test_results["long_exit"]["paper_trading"] = {
                            "success": False,
                            "error": close_result.get("message", "Unknown error"),
                            "timestamp": time.time()
                        }
                else:
                    self.logger.error(f"Failed to open long position: {position_result}")
                    self.test_results["long_entry"]["paper_trading"] = {
                        "success": False,
                        "error": position_result.get("message", "Unknown error"),
                        "timestamp": time.time()
                    }
            except Exception as e:
                self.logger.error(f"Error in long position test: {e}")
                self.logger.error(traceback.format_exc())
                self.test_results["long_entry"]["paper_trading"] = {
                    "success": False,
                    "error": str(e),
                    "timestamp": time.time()
                }
        except Exception as e:
            self.logger.error(f"Error in long position test: {e}")
            self.logger.error(traceback.format_exc())
            self.test_results["long_entry"]["paper_trading"] = {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def test_short_position(self):
        """Test opening and closing a short position."""
        self.logger.info("Testing short position...")
        
        # Switch to paper trading mode for safety
        self.mode_manager.set_mode(TradingMode.PAPER_TRADING)
        self.logger.info(f"Switched to mode: {self.mode_manager.get_current_mode().value}")
        
        try:
            # Get current market data
            market_data = await self.get_market_data()
            if not market_data:
                self.logger.error("Failed to get market data, aborting short position test")
                self.test_results["short_entry"]["paper_trading"] = {
                    "success": False,
                    "error": "Failed to get market data"
                }
                return
            
            current_price = market_data["price"]
            self.logger.info(f"Current {self.symbol} price: {current_price}")
            
            # Open short position
            self.logger.info(f"Opening short position for {self.symbol} with size {self.test_size}...")
            
            try:
                # Use position manager to open position
                position_result = await self.position_manager.open_position(
                    symbol=self.symbol,
                    size=-self.test_size,  # Negative size for short
                    is_long=False,
                    price=current_price * 0.999,  # Slightly below market for limit order
                    reduce_only=False
                )
                
                if position_result.get("success", False):
                    self.logger.info(f"Successfully opened short position: {position_result}")
                    self.test_results["short_entry"]["paper_trading"] = {
                        "success": True,
                        "price": current_price,
                        "size": -self.test_size,
                        "order_id": position_result.get("order_id", "unknown"),
                        "timestamp": time.time()
                    }
                    
                    # Wait a bit before closing
                    self.logger.info("Waiting 5 seconds before closing position...")
                    await asyncio.sleep(5)
                    
                    # Close short position
                    self.logger.info(f"Closing short position for {self.symbol}...")
                    close_result = await self.position_manager.close_position(
                        symbol=self.symbol,
                        is_long=False,
                        price=current_price * 1.001  # Slightly above market for limit order
                    )
                    
                    if close_result.get("success", False):
                        self.logger.info(f"Successfully closed short position: {close_result}")
                        self.test_results["short_exit"]["paper_trading"] = {
                            "success": True,
                            "price": current_price,
                            "order_id": close_result.get("order_id", "unknown"),
                            "timestamp": time.time()
                        }
                    else:
                        self.logger.error(f"Failed to close short position: {close_result}")
                        self.test_results["short_exit"]["paper_trading"] = {
                            "success": False,
                            "error": close_result.get("message", "Unknown error"),
                            "timestamp": time.time()
                        }
                else:
                    self.logger.error(f"Failed to open short position: {position_result}")
                    self.test_results["short_entry"]["paper_trading"] = {
                        "success": False,
                        "error": position_result.get("message", "Unknown error"),
                        "timestamp": time.time()
                    }
            except Exception as e:
                self.logger.error(f"Error in short position test: {e}")
                self.logger.error(traceback.format_exc())
                self.test_results["short_entry"]["paper_trading"] = {
                    "success": False,
                    "error": str(e),
                    "timestamp": time.time()
                }
        except Exception as e:
            self.logger.error(f"Error in short position test: {e}")
            self.logger.error(traceback.format_exc())
            self.test_results["short_entry"]["paper_trading"] = {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def test_live_mode_simulation(self):
        """Test live mode simulation with small positions."""
        self.logger.info("Testing live mode simulation...")
        
        # Switch to live trading mode
        self.mode_manager.set_mode(TradingMode.LIVE_TRADING)
        self.logger.info(f"Switched to mode: {self.mode_manager.get_current_mode().value}")
        
        # Update mode settings to use very small position size
        self.mode_manager.update_mode_settings({
            "max_position_size": 0.05,
            "risk_percent": 0.001
        })
        
        try:
            # Get current market data
            market_data = await self.get_market_data()
            if not market_data:
                self.logger.error("Failed to get market data, aborting live mode simulation")
                self.test_results["long_entry"]["live_trading"] = {
                    "success": False,
                    "error": "Failed to get market data"
                }
                return
            
            current_price = market_data["price"]
            self.logger.info(f"Current {self.symbol} price: {current_price}")
            
            # Open tiny long position
            tiny_size = 0.01  # Very small position size
            self.logger.info(f"Opening tiny long position for {self.symbol} with size {tiny_size}...")
            
            try:
                # Use position manager to open position
                position_result = await self.position_manager.open_position(
                    symbol=self.symbol,
                    size=tiny_size,
                    is_long=True,
                    price=current_price * 1.001,  # Slightly above market for limit order
                    reduce_only=False
                )
                
                if position_result.get("success", False):
                    self.logger.info(f"Successfully opened tiny long position: {position_result}")
                    self.test_results["long_entry"]["live_trading"] = {
                        "success": True,
                        "price": current_price,
                        "size": tiny_size,
                        "order_id": position_result.get("order_id", "unknown"),
                        "timestamp": time.time()
                    }
                    
                    # Wait a bit before closing
                    self.logger.info("Waiting 5 seconds before closing position...")
                    await asyncio.sleep(5)
                    
                    # Close long position
                    self.logger.info(f"Closing tiny long position for {self.symbol}...")
                    close_result = await self.position_manager.close_position(
                        symbol=self.symbol,
                        is_long=True,
                        price=current_price * 0.999  # Slightly below market for limit order
                    )
                    
                    if close_result.get("success", False):
                        self.logger.info(f"Successfully closed tiny long position: {close_result}")
                        self.test_results["long_exit"]["live_trading"] = {
                            "success": True,
                            "price": current_price,
                            "order_id": close_result.get("order_id", "unknown"),
                            "timestamp": time.time()
                        }
                    else:
                        self.logger.error(f"Failed to close tiny long position: {close_result}")
                        self.test_results["long_exit"]["live_trading"] = {
                            "success": False,
                            "error": close_result.get("message", "Unknown error"),
                            "timestamp": time.time()
                        }
                else:
                    self.logger.error(f"Failed to open tiny long position: {position_result}")
                    self.test_results["long_entry"]["live_trading"] = {
                        "success": False,
                        "error": position_result.get("message", "Unknown error"),
                        "timestamp": time.time()
                    }
            except Exception as e:
                self.logger.error(f"Error in live mode simulation: {e}")
                self.logger.error(traceback.format_exc())
                self.test_results["long_entry"]["live_trading"] = {
                    "success": False,
                    "error": str(e),
                    "timestamp": time.time()
                }
        except Exception as e:
            self.logger.error(f"Error in live mode simulation: {e}")
            self.logger.error(traceback.format_exc())
            self.test_results["long_entry"]["live_trading"] = {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
        
        # Switch back to paper trading mode for safety
        self.mode_manager.set_mode(TradingMode.PAPER_TRADING)
        self.logger.info(f"Switched back to mode: {self.mode_manager.get_current_mode().value}")
    
    async def run_tests(self):
        """Run all tests."""
        self.logger.info("Starting XRP trading tests with mode management...")
        
        try:
            # Initialize test environment
            if not await self.initialize():
                self.logger.error("Failed to initialize test environment, aborting tests")
                return
            
            # Test mode switching
            await self.test_mode_switching()
            
            # Test long position
            await self.test_long_position()
            
            # Test short position
            await self.test_short_position()
            
            # Test live mode simulation
            await self.test_live_mode_simulation()
            
            # Save test results
            self._save_test_results()
            
            self.logger.info("XRP trading tests completed")
            
            # Print summary
            self._print_summary()
        except Exception as e:
            self.logger.error(f"Error running tests: {e}")
            self.logger.error(traceback.format_exc())
    
    def _print_summary(self):
        """Print test summary."""
        self.logger.info("=== XRP Trading Test Summary ===")
        
        # Mode switching summary
        self.logger.info("Mode Switching:")
        for mode, result in self.test_results["mode_switch"].items():
            status = "SUCCESS" if result.get("success", False) else "FAILED"
            self.logger.info(f"  {mode}: {status}")
        
        # Long position summary
        self.logger.info("Long Position:")
        for mode, result in self.test_results["long_entry"].items():
            entry_status = "SUCCESS" if result.get("success", False) else "FAILED"
            exit_result = self.test_results["long_exit"].get(mode, {})
            exit_status = "SUCCESS" if exit_result.get("success", False) else "FAILED"
            self.logger.info(f"  {mode}: Entry: {entry_status}, Exit: {exit_status}")
        
        # Short position summary
        self.logger.info("Short Position:")
        for mode, result in self.test_results["short_entry"].items():
            entry_status = "SUCCESS" if result.get("success", False) else "FAILED"
            exit_result = self.test_results["short_exit"].get(mode, {})
            exit_status = "SUCCESS" if exit_result.get("success", False) else "FAILED"
            self.logger.info(f"  {mode}: Entry: {entry_status}, Exit: {exit_status}")
        
        # Overall success rate
        total_tests = 0
        successful_tests = 0
        
        for category in ["mode_switch", "long_entry", "long_exit", "short_entry", "short_exit"]:
            for mode, result in self.test_results[category].items():
                total_tests += 1
                if result.get("success", False):
                    successful_tests += 1
        
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        self.logger.info(f"Overall Success Rate: {success_rate:.2f}% ({successful_tests}/{total_tests})")
        self.logger.info("===============================")

# Main entry point
if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="XRP Trading Test with Mode Management")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    args = parser.parse_args()
    
    # Create test instance
    test = XRPTradingTest(args.config)
    
    # Run tests
    asyncio.run(test.run_tests())
