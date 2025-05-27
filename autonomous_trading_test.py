#!/usr/bin/env python3
"""
Autonomous Trading Test for HyperliquidMaster

This script tests the autonomous trading capabilities of the HyperliquidMaster bot,
focusing on both long and short positions with various market conditions.
"""

import os
import sys
import time
import json
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("autonomous_trading_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import core modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.enhanced_connection_manager import ConnectionManager
from core.trading_mode import TradingModeManager, TradingMode
from core.settings_manager import SettingsManager
from core.enhanced_risk_manager import RiskManager
from core.position_manager_wrapper import PositionManagerWrapper
from core.advanced_order_manager import AdvancedOrderManager

class AutonomousTradingTest:
    """
    Test harness for validating autonomous trading capabilities.
    """
    
    def __init__(self):
        """Initialize the test harness."""
        self.logger = logger
        self.logger.info("Initializing Autonomous Trading Test")
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize core components
        self.connection_manager = ConnectionManager(self.logger)
        self.mode_manager = TradingModeManager(self.config, self.logger)
        self.settings_manager = SettingsManager(self.config, self.logger)
        self.risk_manager = RiskManager(self.config, self.logger)
        self.position_manager = PositionManagerWrapper(self.config, self.logger)
        self.order_manager = AdvancedOrderManager(self.config, self.logger)
        
        # Test results
        self.test_results = {
            "long_entry_tests": [],
            "long_exit_tests": [],
            "short_entry_tests": [],
            "short_exit_tests": [],
            "mode_switching_tests": [],
            "error_handling_tests": [],
            "overall_success": False,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.logger.info("Autonomous Trading Test initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        try:
            if os.path.exists("config.json"):
                with open("config.json", "r") as f:
                    config = json.load(f)
                self.logger.info("Configuration loaded from config.json")
                return config
            else:
                self.logger.warning("config.json not found, using default configuration")
                return self._create_default_config()
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """
        Create default configuration.
        
        Returns:
            Default configuration dictionary
        """
        config = {
            "api_key": "test_api_key",
            "api_secret": "test_api_secret",
            "base_url": "https://api.hyperliquid.xyz",
            "ws_url": "wss://api.hyperliquid.xyz/ws",
            "default_mode": "paper_trading",
            "risk_level": 0.01,  # 1% risk per trade
            "max_drawdown": 0.1,  # 10% max drawdown
            "daily_loss_limit": 0.05,  # 5% daily loss limit
            "max_position_size": 0.1,  # 10% max position size
            "max_open_positions": 5,  # Max 5 open positions
            "default_leverage": 5,  # 5x leverage
            "symbols": ["BTC", "ETH", "XRP", "SOL", "DOGE", "AVAX", "LINK", "MATIC"]
        }
        
        self.logger.info("Created default configuration")
        return config
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all autonomous trading tests.
        
        Returns:
            Test results dictionary
        """
        try:
            self.logger.info("Starting all autonomous trading tests")
            
            # Connect to exchange
            if not self.connection_manager.connect():
                self.logger.error("Failed to connect to exchange, aborting tests")
                return {
                    "success": False,
                    "error": "Failed to connect to exchange"
                }
            
            # Run tests
            self._test_mode_switching()
            self._test_long_entry()
            self._test_long_exit()
            self._test_short_entry()
            self._test_short_exit()
            self._test_error_handling()
            
            # Calculate overall success
            total_tests = (
                len(self.test_results["long_entry_tests"]) +
                len(self.test_results["long_exit_tests"]) +
                len(self.test_results["short_entry_tests"]) +
                len(self.test_results["short_exit_tests"]) +
                len(self.test_results["mode_switching_tests"]) +
                len(self.test_results["error_handling_tests"])
            )
            
            successful_tests = 0
            for category in ["long_entry_tests", "long_exit_tests", "short_entry_tests", "short_exit_tests", "mode_switching_tests", "error_handling_tests"]:
                for test in self.test_results[category]:
                    if test.get("success", False):
                        successful_tests += 1
            
            if total_tests > 0:
                success_rate = successful_tests / total_tests
                self.test_results["success_rate"] = success_rate
                self.test_results["overall_success"] = success_rate >= 0.9  # 90% success rate required
            else:
                self.test_results["success_rate"] = 0.0
                self.test_results["overall_success"] = False
            
            self.logger.info(f"All tests completed. Success rate: {self.test_results.get('success_rate', 0.0):.2%}")
            
            # Save test results
            self._save_test_results()
            
            return self.test_results
        except Exception as e:
            self.logger.error(f"Error running tests: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _test_mode_switching(self) -> None:
        """Test trading mode switching functionality."""
        self.logger.info("Testing trading mode switching")
        
        # Test all modes
        modes = [
            TradingMode.PAPER_TRADING,
            TradingMode.LIVE_TRADING,
            TradingMode.MONITOR_ONLY,
            TradingMode.AGGRESSIVE,
            TradingMode.CONSERVATIVE
        ]
        
        for mode in modes:
            try:
                # Set mode
                self.mode_manager.set_mode(mode)
                
                # Verify mode
                current_mode = self.mode_manager.get_current_mode()
                
                # Record result
                result = {
                    "test_name": f"Switch to {mode.name}",
                    "success": current_mode == mode,
                    "expected": mode.name,
                    "actual": current_mode.name if current_mode else "None",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                self.test_results["mode_switching_tests"].append(result)
                
                self.logger.info(f"Mode switching test to {mode.name}: {'SUCCESS' if result['success'] else 'FAILED'}")
            except Exception as e:
                self.logger.error(f"Error testing mode switching to {mode.name}: {e}")
                
                # Record error
                result = {
                    "test_name": f"Switch to {mode.name}",
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                self.test_results["mode_switching_tests"].append(result)
        
        # Reset to paper trading mode
        self.mode_manager.set_mode(TradingMode.PAPER_TRADING)
    
    def _test_long_entry(self) -> None:
        """Test long position entry functionality."""
        self.logger.info("Testing long position entry")
        
        # Set paper trading mode
        self.mode_manager.set_mode(TradingMode.PAPER_TRADING)
        
        # Test symbols
        symbols = ["XRP", "ETH"]
        
        for symbol in symbols:
            try:
                # Calculate position size
                entry_price = self._get_current_price(symbol)
                stop_loss_price = entry_price * 0.95  # 5% stop loss
                
                position_size = self.risk_manager.calculate_position_size(
                    symbol=symbol,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss_price
                )
                
                # Open long position
                result = self.position_manager.open_position(
                    symbol=symbol,
                    is_long=True,
                    size=position_size,
                    stop_loss_percent=5.0,
                    take_profit_percent=15.0
                )
                
                # Record result
                test_result = {
                    "test_name": f"Long entry for {symbol}",
                    "success": result.get("success", False),
                    "position_size": position_size,
                    "entry_price": entry_price,
                    "stop_loss_price": stop_loss_price,
                    "result": result,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                self.test_results["long_entry_tests"].append(test_result)
                
                self.logger.info(f"Long entry test for {symbol}: {'SUCCESS' if test_result['success'] else 'FAILED'}")
            except Exception as e:
                self.logger.error(f"Error testing long entry for {symbol}: {e}")
                
                # Record error
                test_result = {
                    "test_name": f"Long entry for {symbol}",
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                self.test_results["long_entry_tests"].append(test_result)
    
    def _test_long_exit(self) -> None:
        """Test long position exit functionality."""
        self.logger.info("Testing long position exit")
        
        # Set paper trading mode
        self.mode_manager.set_mode(TradingMode.PAPER_TRADING)
        
        # Get open positions
        positions = self.position_manager.get_positions()
        
        # Filter long positions
        long_positions = [p for p in positions if p["side"] == "LONG"]
        
        if not long_positions:
            self.logger.warning("No long positions found for exit testing")
            
            # Record result
            test_result = {
                "test_name": "Long exit (no positions)",
                "success": True,
                "message": "No long positions found for exit testing",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.test_results["long_exit_tests"].append(test_result)
            return
        
        # Test exit for each long position
        for position in long_positions:
            symbol = position["symbol"]
            
            try:
                # Close long position
                result = self.position_manager.close_position(
                    symbol=symbol,
                    is_long=True
                )
                
                # Record result
                test_result = {
                    "test_name": f"Long exit for {symbol}",
                    "success": result.get("success", False),
                    "position": position,
                    "result": result,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                self.test_results["long_exit_tests"].append(test_result)
                
                self.logger.info(f"Long exit test for {symbol}: {'SUCCESS' if test_result['success'] else 'FAILED'}")
            except Exception as e:
                self.logger.error(f"Error testing long exit for {symbol}: {e}")
                
                # Record error
                test_result = {
                    "test_name": f"Long exit for {symbol}",
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                self.test_results["long_exit_tests"].append(test_result)
    
    def _test_short_entry(self) -> None:
        """Test short position entry functionality."""
        self.logger.info("Testing short position entry")
        
        # Set paper trading mode
        self.mode_manager.set_mode(TradingMode.PAPER_TRADING)
        
        # Test symbols
        symbols = ["XRP", "ETH"]
        
        for symbol in symbols:
            try:
                # Calculate position size
                entry_price = self._get_current_price(symbol)
                stop_loss_price = entry_price * 1.05  # 5% stop loss for short
                
                position_size = self.risk_manager.calculate_position_size(
                    symbol=symbol,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss_price
                )
                
                # Open short position
                result = self.position_manager.open_position(
                    symbol=symbol,
                    is_long=False,
                    size=position_size,
                    stop_loss_percent=5.0,
                    take_profit_percent=15.0
                )
                
                # Record result
                test_result = {
                    "test_name": f"Short entry for {symbol}",
                    "success": result.get("success", False),
                    "position_size": position_size,
                    "entry_price": entry_price,
                    "stop_loss_price": stop_loss_price,
                    "result": result,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                self.test_results["short_entry_tests"].append(test_result)
                
                self.logger.info(f"Short entry test for {symbol}: {'SUCCESS' if test_result['success'] else 'FAILED'}")
            except Exception as e:
                self.logger.error(f"Error testing short entry for {symbol}: {e}")
                
                # Record error
                test_result = {
                    "test_name": f"Short entry for {symbol}",
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                self.test_results["short_entry_tests"].append(test_result)
    
    def _test_short_exit(self) -> None:
        """Test short position exit functionality."""
        self.logger.info("Testing short position exit")
        
        # Set paper trading mode
        self.mode_manager.set_mode(TradingMode.PAPER_TRADING)
        
        # Get open positions
        positions = self.position_manager.get_positions()
        
        # Filter short positions
        short_positions = [p for p in positions if p["side"] == "SHORT"]
        
        if not short_positions:
            self.logger.warning("No short positions found for exit testing")
            
            # Record result
            test_result = {
                "test_name": "Short exit (no positions)",
                "success": True,
                "message": "No short positions found for exit testing",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.test_results["short_exit_tests"].append(test_result)
            return
        
        # Test exit for each short position
        for position in short_positions:
            symbol = position["symbol"]
            
            try:
                # Close short position
                result = self.position_manager.close_position(
                    symbol=symbol,
                    is_long=False
                )
                
                # Record result
                test_result = {
                    "test_name": f"Short exit for {symbol}",
                    "success": result.get("success", False),
                    "position": position,
                    "result": result,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                self.test_results["short_exit_tests"].append(test_result)
                
                self.logger.info(f"Short exit test for {symbol}: {'SUCCESS' if test_result['success'] else 'FAILED'}")
            except Exception as e:
                self.logger.error(f"Error testing short exit for {symbol}: {e}")
                
                # Record error
                test_result = {
                    "test_name": f"Short exit for {symbol}",
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                self.test_results["short_exit_tests"].append(test_result)
    
    def _test_error_handling(self) -> None:
        """Test error handling and recovery functionality."""
        self.logger.info("Testing error handling and recovery")
        
        # Test connection loss and recovery
        try:
            # Simulate connection loss
            self.connection_manager.disconnect()
            
            # Verify disconnected state
            is_disconnected = not self.connection_manager.is_connected()
            
            # Attempt reconnection
            reconnect_success = self.connection_manager.reconnect()
            
            # Verify reconnected state
            is_reconnected = self.connection_manager.is_connected()
            
            # Record result
            test_result = {
                "test_name": "Connection loss and recovery",
                "success": is_disconnected and reconnect_success and is_reconnected,
                "is_disconnected": is_disconnected,
                "reconnect_success": reconnect_success,
                "is_reconnected": is_reconnected,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.test_results["error_handling_tests"].append(test_result)
            
            self.logger.info(f"Connection loss and recovery test: {'SUCCESS' if test_result['success'] else 'FAILED'}")
        except Exception as e:
            self.logger.error(f"Error testing connection loss and recovery: {e}")
            
            # Record error
            test_result = {
                "test_name": "Connection loss and recovery",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.test_results["error_handling_tests"].append(test_result)
        
        # Test invalid order handling
        try:
            # Attempt to open position with invalid parameters
            result = self.position_manager.open_position(
                symbol="INVALID_SYMBOL",
                is_long=True,
                size=0.0  # Invalid size
            )
            
            # Verify error handling
            error_handled = not result.get("success", False) and "error" in result
            
            # Record result
            test_result = {
                "test_name": "Invalid order handling",
                "success": error_handled,
                "result": result,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.test_results["error_handling_tests"].append(test_result)
            
            self.logger.info(f"Invalid order handling test: {'SUCCESS' if test_result['success'] else 'FAILED'}")
        except Exception as e:
            self.logger.error(f"Error testing invalid order handling: {e}")
            
            # Record error
            test_result = {
                "test_name": "Invalid order handling",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.test_results["error_handling_tests"].append(test_result)
    
    def _save_test_results(self) -> None:
        """Save test results to file."""
        try:
            with open("autonomous_trading_test_results.json", "w") as f:
                json.dump(self.test_results, f, indent=4)
            
            self.logger.info("Test results saved to autonomous_trading_test_results.json")
        except Exception as e:
            self.logger.error(f"Error saving test results: {e}")
    
    def _get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price
        """
        # This is a placeholder for actual price fetching
        # In a real implementation, this would call the exchange API
        
        # Return dummy prices for testing
        prices = {
            "BTC": 50000.0,
            "ETH": 3000.0,
            "XRP": 0.5,
            "SOL": 100.0,
            "DOGE": 0.1,
            "AVAX": 30.0,
            "LINK": 15.0,
            "MATIC": 1.0
        }
        
        return prices.get(symbol, 1.0)

if __name__ == "__main__":
    # Run autonomous trading tests
    test_harness = AutonomousTradingTest()
    results = test_harness.run_all_tests()
    
    # Print summary
    print("\n=== AUTONOMOUS TRADING TEST SUMMARY ===")
    print(f"Overall Success: {'YES' if results.get('overall_success', False) else 'NO'}")
    print(f"Success Rate: {results.get('success_rate', 0.0):.2%}")
    print(f"Long Entry Tests: {len(results.get('long_entry_tests', []))}")
    print(f"Long Exit Tests: {len(results.get('long_exit_tests', []))}")
    print(f"Short Entry Tests: {len(results.get('short_entry_tests', []))}")
    print(f"Short Exit Tests: {len(results.get('short_exit_tests', []))}")
    print(f"Mode Switching Tests: {len(results.get('mode_switching_tests', []))}")
    print(f"Error Handling Tests: {len(results.get('error_handling_tests', []))}")
    print("==========================================")
