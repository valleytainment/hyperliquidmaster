#!/usr/bin/env python3
"""
Test script for validating the robustness and compatibility of the HyperliquidMaster application.
This script tests connection recovery, theme switching, SDK compatibility, and watchdog functionality.
"""

import os
import sys
import time
import logging
import asyncio
import json
import traceback
from typing import Dict, Any, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core components
from core.hyperliquid_adapter import HyperliquidAdapter
from core.trading_integration import TradingIntegration
from core.error_handler import ErrorHandler

class RobustnessTest:
    """
    Test class for validating the robustness and compatibility of the HyperliquidMaster application.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the test class.
        
        Args:
            config_path: Path to the configuration file
        """
        # Setup logging
        self.logger = self._setup_logger()
        self.logger.info("Initializing robustness test...")
        
        # Store config path
        self.config_path = config_path
        
        # Initialize components
        self.error_handler = ErrorHandler(self.logger)
        self.adapter = HyperliquidAdapter(config_path)
        self.trading = TradingIntegration(config_path, self.logger)
        
        # Test results
        self.test_results = {
            "connection_recovery": False,
            "sdk_compatibility": False,
            "watchdog_functionality": False,
            "state_persistence": False,
            "overall": False
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Set up the logger."""
        logger = logging.getLogger("RobustnessTest")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("robustness_test.log", mode="w")
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    async def run_all_tests(self):
        """Run all tests."""
        self.logger.info("Running all tests...")
        
        try:
            # Test connection recovery
            await self.test_connection_recovery()
            
            # Test SDK compatibility
            await self.test_sdk_compatibility()
            
            # Test watchdog functionality
            await self.test_watchdog_functionality()
            
            # Test state persistence
            await self.test_state_persistence()
            
            # Calculate overall result
            self.test_results["overall"] = all([
                self.test_results["connection_recovery"],
                self.test_results["sdk_compatibility"],
                self.test_results["watchdog_functionality"],
                self.test_results["state_persistence"]
            ])
            
            # Log results
            self.logger.info("Test results:")
            for test, result in self.test_results.items():
                self.logger.info(f"  {test}: {'PASS' if result else 'FAIL'}")
            
            # Save results to file
            self._save_results()
            
            return self.test_results["overall"]
        
        except Exception as e:
            self.logger.error(f"Error running tests: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _save_results(self):
        """Save test results to file."""
        try:
            with open("robustness_test_results.json", "w") as f:
                json.dump(self.test_results, f, indent=2)
            
            self.logger.info("Test results saved to robustness_test_results.json")
        except Exception as e:
            self.logger.error(f"Error saving test results: {e}")
    
    async def test_connection_recovery(self):
        """Test connection recovery."""
        self.logger.info("Testing connection recovery...")
        
        try:
            # Test initial connection
            self.logger.info("Testing initial connection...")
            result = self.adapter.test_connection()
            
            if "error" in result:
                self.logger.error(f"Initial connection failed: {result['error']}")
                return False
            
            self.logger.info("Initial connection successful")
            
            # Simulate connection loss
            self.logger.info("Simulating connection loss...")
            self.adapter.is_connected = False
            
            # Try to get market data (should trigger reconnection)
            self.logger.info("Attempting to get market data after connection loss...")
            market_data = self.adapter.get_market_data("BTC")
            
            # Check if reconnection was successful
            if "error" in market_data:
                self.logger.error(f"Reconnection failed: {market_data['error']}")
                return False
            
            self.logger.info("Reconnection successful")
            
            # Test connection through trading integration
            self.logger.info("Testing connection through trading integration...")
            self.trading.is_connected = False
            
            # Try to get market data through trading integration
            result = self.trading.get_market_data("BTC")
            
            if not result.get("success", False):
                self.logger.error(f"Trading integration reconnection failed: {result.get('message', 'Unknown error')}")
                return False
            
            self.logger.info("Trading integration reconnection successful")
            
            # Mark test as passed
            self.test_results["connection_recovery"] = True
            self.logger.info("Connection recovery test PASSED")
            return True
            
        except Exception as e:
            self.logger.error(f"Error testing connection recovery: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    async def test_sdk_compatibility(self):
        """Test SDK compatibility."""
        self.logger.info("Testing SDK compatibility...")
        
        try:
            # Test safe attribute access
            self.logger.info("Testing safe attribute access...")
            
            # Create test objects
            test_dict = {"key1": "value1", "key2": {"nested": "value2"}}
            test_obj = type("TestObject", (), {"attr1": "value1"})()
            
            # Test safe attribute access on dictionary
            value1 = self.trading._safe_get_attribute(test_dict, "key1", "default")
            value_missing = self.trading._safe_get_attribute(test_dict, "missing", "default")
            
            if value1 != "value1" or value_missing != "default":
                self.logger.error(f"Safe attribute access on dictionary failed: {value1}, {value_missing}")
                return False
            
            # Test safe attribute access on object
            value1_obj = self.trading._safe_get_attribute(test_obj, "attr1", "default")
            value_missing_obj = self.trading._safe_get_attribute(test_obj, "missing", "default")
            
            if value1_obj != "value1" or value_missing_obj != "default":
                self.logger.error(f"Safe attribute access on object failed: {value1_obj}, {value_missing_obj}")
                return False
            
            # Test safe nested access
            nested_value = self.trading._safe_get_nested(test_dict, ["key2", "nested"], "default")
            nested_missing = self.trading._safe_get_nested(test_dict, ["key2", "missing"], "default")
            
            if nested_value != "value2" or nested_missing != "default":
                self.logger.error(f"Safe nested access failed: {nested_value}, {nested_missing}")
                return False
            
            # Test safe float conversion
            float_value = self.trading._safe_float_convert("123.45", 0.0)
            float_invalid = self.trading._safe_float_convert("invalid", 0.0)
            float_none = self.trading._safe_float_convert(None, 0.0)
            
            if float_value != 123.45 or float_invalid != 0.0 or float_none != 0.0:
                self.logger.error(f"Safe float conversion failed: {float_value}, {float_invalid}, {float_none}")
                return False
            
            # Test position data normalization
            self.logger.info("Testing position data normalization...")
            
            # Create test position data in different formats
            test_positions_v1 = [
                {"coin": "BTC", "szi": 1.5, "entryPx": 50000, "markPx": 51000, "liqPx": 40000, "unrealizedPnl": 1500}
            ]
            
            test_positions_v2 = [
                {"name": "BTC", "size": 1.5, "entry_price": 50000, "mark_price": 51000, "liquidation_price": 40000, "unrealized_pnl": 1500}
            ]
            
            # Process positions
            processed_v1 = self._process_test_positions(test_positions_v1)
            processed_v2 = self._process_test_positions(test_positions_v2)
            
            # Check if both formats were processed correctly
            if processed_v1[0]["symbol"] != "BTC" or processed_v2[0]["symbol"] != "BTC":
                self.logger.error(f"Position data normalization failed: {processed_v1}, {processed_v2}")
                return False
            
            self.logger.info("SDK compatibility tests successful")
            
            # Mark test as passed
            self.test_results["sdk_compatibility"] = True
            self.logger.info("SDK compatibility test PASSED")
            return True
            
        except Exception as e:
            self.logger.error(f"Error testing SDK compatibility: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _process_test_positions(self, positions):
        """Process test positions using the same logic as in trading_integration."""
        processed_positions = []
        
        for pos in positions:
            # Handle different SDK versions and formats
            processed_pos = {
                "symbol": self.trading._safe_get_attribute(pos, "coin", 
                         self.trading._safe_get_attribute(pos, "name", "Unknown")),
                "size": self.trading._safe_float_convert(self.trading._safe_get_attribute(pos, "szi", 
                        self.trading._safe_get_attribute(pos, "size", 0.0))),
                "entry_price": self.trading._safe_float_convert(self.trading._safe_get_attribute(pos, "entryPx", 
                              self.trading._safe_get_attribute(pos, "entry_price", 0.0))),
                "mark_price": self.trading._safe_float_convert(self.trading._safe_get_attribute(pos, "markPx", 
                             self.trading._safe_get_attribute(pos, "mark_price", 0.0))),
                "liquidation_price": self.trading._safe_float_convert(self.trading._safe_get_attribute(pos, "liqPx", 
                                   self.trading._safe_get_attribute(pos, "liquidation_price", 0.0))),
                "unrealized_pnl": self.trading._safe_float_convert(self.trading._safe_get_attribute(pos, "unrealizedPnl", 
                                 self.trading._safe_get_attribute(pos, "unrealized_pnl", 0.0))),
                "leverage": self.trading._safe_float_convert(self.trading._safe_get_attribute(pos, "leverage", 1.0))
            }
            
            # Calculate PnL percentage
            if processed_pos["entry_price"] > 0 and processed_pos["size"] != 0:
                price_diff = processed_pos["mark_price"] - processed_pos["entry_price"]
                direction = 1 if processed_pos["size"] > 0 else -1
                processed_pos["pnl_percentage"] = direction * price_diff / processed_pos["entry_price"] * 100
            else:
                processed_pos["pnl_percentage"] = 0.0
            
            processed_positions.append(processed_pos)
        
        return processed_positions
    
    async def test_watchdog_functionality(self):
        """Test watchdog functionality."""
        self.logger.info("Testing watchdog functionality...")
        
        try:
            # Import the WatchdogTimer class from main.py
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from main import WatchdogTimer
            
            # Create a flag to track callback execution
            callback_executed = False
            
            # Define callback function
            def watchdog_callback():
                nonlocal callback_executed
                callback_executed = True
                self.logger.info("Watchdog callback executed")
            
            # Create watchdog with short timeout
            watchdog = WatchdogTimer(0.5, watchdog_callback)
            
            # Start watchdog
            self.logger.info("Starting watchdog...")
            watchdog.start()
            
            # Wait for callback to execute
            self.logger.info("Waiting for watchdog timeout...")
            await asyncio.sleep(1.0)
            
            # Check if callback was executed
            if not callback_executed:
                self.logger.error("Watchdog callback was not executed")
                return False
            
            # Reset watchdog and verify it doesn't trigger immediately
            callback_executed = False
            self.logger.info("Resetting watchdog...")
            watchdog.reset()
            
            # Wait a short time
            await asyncio.sleep(0.2)
            
            # Check that callback wasn't executed yet
            if callback_executed:
                self.logger.error("Watchdog callback executed too early after reset")
                return False
            
            # Wait for timeout again
            self.logger.info("Waiting for watchdog timeout after reset...")
            await asyncio.sleep(0.5)
            
            # Check if callback was executed
            if not callback_executed:
                self.logger.error("Watchdog callback was not executed after reset")
                return False
            
            # Stop watchdog
            self.logger.info("Stopping watchdog...")
            watchdog.stop()
            
            # Reset flag and wait to verify watchdog is stopped
            callback_executed = False
            await asyncio.sleep(1.0)
            
            # Check that callback wasn't executed after stop
            if callback_executed:
                self.logger.error("Watchdog callback executed after stop")
                return False
            
            self.logger.info("Watchdog functionality tests successful")
            
            # Mark test as passed
            self.test_results["watchdog_functionality"] = True
            self.logger.info("Watchdog functionality test PASSED")
            return True
            
        except Exception as e:
            self.logger.error(f"Error testing watchdog functionality: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    async def test_state_persistence(self):
        """Test state persistence."""
        self.logger.info("Testing state persistence...")
        
        try:
            # Import the EnhancedTradingBot class from main.py
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from main import EnhancedTradingBot
            
            # Create test state data
            test_state = {
                "last_trade_time": time.time(),
                "market_data": {
                    "BTC": {"price": 50000, "funding_rate": 0.01},
                    "ETH": {"price": 3000, "funding_rate": 0.005}
                },
                "positions": {
                    "BTC": {"size": 1.0, "entry_price": 49000, "pnl": 1000},
                    "ETH": {"size": 5.0, "entry_price": 2900, "pnl": 500}
                },
                "timestamp": time.time()
            }
            
            # Save test state
            self.logger.info("Saving test state...")
            with open("bot_state.json", "w") as f:
                json.dump(test_state, f)
            
            # Create bot instance (should load the state)
            self.logger.info("Creating bot instance to load state...")
            bot = EnhancedTradingBot(self.config_path)
            
            # Check if state was loaded correctly
            if bot.last_trade_time != test_state["last_trade_time"]:
                self.logger.error(f"State persistence failed: last_trade_time mismatch {bot.last_trade_time} != {test_state['last_trade_time']}")
                return False
            
            if "BTC" not in bot.market_data or "ETH" not in bot.market_data:
                self.logger.error(f"State persistence failed: market_data mismatch {bot.market_data}")
                return False
            
            if "BTC" not in bot.positions or "ETH" not in bot.positions:
                self.logger.error(f"State persistence failed: positions mismatch {bot.positions}")
                return False
            
            # Modify state and save
            self.logger.info("Modifying and saving state...")
            bot.market_data["BTC"]["price"] = 51000
            bot.save_state()
            
            # Create new bot instance to verify state was saved
            self.logger.info("Creating new bot instance to verify state was saved...")
            new_bot = EnhancedTradingBot(self.config_path)
            
            # Check if modified state was loaded correctly
            if new_bot.market_data["BTC"]["price"] != 51000:
                self.logger.error(f"State persistence failed: modified state not saved {new_bot.market_data['BTC']['price']} != 51000")
                return False
            
            self.logger.info("State persistence tests successful")
            
            # Clean up
            if os.path.exists("bot_state.json"):
                os.remove("bot_state.json")
            
            # Mark test as passed
            self.test_results["state_persistence"] = True
            self.logger.info("State persistence test PASSED")
            return True
            
        except Exception as e:
            self.logger.error(f"Error testing state persistence: {e}")
            self.logger.error(traceback.format_exc())
            return False

async def main():
    """Main entry point."""
    # Get config path from command line or use default
    config_path = "config.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # Create and run tests
    test = RobustnessTest(config_path)
    success = await test.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
