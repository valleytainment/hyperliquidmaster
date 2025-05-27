#!/usr/bin/env python3
"""
Position Management Test Script for HyperliquidMaster

This script tests the advanced position management capabilities of the HyperliquidMaster
trading bot, including opening and closing both long and short positions, handling
partial exits, trailing stops, and position reversals.
"""

import os
import sys
import time
import json
import logging
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("position_management_test.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("PositionManagementTest")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from core.hyperliquid_adapter import HyperliquidAdapter
from core.position_manager import AdvancedPositionManager
from strategies.master_omni_overlord_robust import MasterOmniOverlordRobustStrategy
from strategies.robust_signal_generator import RobustSignalGenerator

class PositionManagementTester:
    """
    Test harness for position management functionality
    """
    
    def __init__(self, config_path="config.json"):
        """Initialize the tester with configuration"""
        self.logger = logger
        self.config_path = config_path
        self.test_results = {
            "long_entry": {"status": "Not Tested", "details": {}},
            "long_exit": {"status": "Not Tested", "details": {}},
            "short_entry": {"status": "Not Tested", "details": {}},
            "short_exit": {"status": "Not Tested", "details": {}},
            "position_reversal": {"status": "Not Tested", "details": {}},
            "partial_exit": {"status": "Not Tested", "details": {}},
            "trailing_stop": {"status": "Not Tested", "details": {}},
            "emergency_close": {"status": "Not Tested", "details": {}}
        }
        self.success_score = 0.0
        self.total_tests = len(self.test_results)
        self.bot_mode = "test"  # Default mode: test, live, paper
        
        # Load configuration
        self._load_config()
        
        # Initialize components
        self._initialize_components()
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            
            # Set bot mode from config if available
            if "bot_mode" in self.config:
                self.bot_mode = self.config["bot_mode"]
            
            self.logger.info(f"Configuration loaded from {self.config_path}")
            self.logger.info(f"Bot operating mode: {self.bot_mode}")
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise
    
    def _initialize_components(self):
        """Initialize trading components"""
        try:
            # Initialize adapter with config_path
            self.adapter = HyperliquidAdapter(config_path=self.config_path)
            
            # Initialize position manager
            self.position_manager = AdvancedPositionManager(
                adapter=self.adapter,
                logger=self.logger
            )
            
            # Initialize strategy
            self.strategy = MasterOmniOverlordRobustStrategy(
                logger=self.logger
            )
            
            # Initialize signal generator
            self.signal_generator = RobustSignalGenerator()
            
            self.logger.info("All components initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    def set_bot_mode(self, mode):
        """
        Set the bot's operating mode
        
        Args:
            mode: Operating mode ('test', 'paper', 'live')
        
        Returns:
            True if successful, False otherwise
        """
        valid_modes = ['test', 'paper', 'live']
        if mode not in valid_modes:
            self.logger.error(f"Invalid mode: {mode}. Must be one of {valid_modes}")
            return False
        
        # Update mode in memory
        self.bot_mode = mode
        
        # Update mode in config
        self.config["bot_mode"] = mode
        
        # Save config
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            self.logger.info(f"Bot mode changed to: {mode}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving bot mode to config: {e}")
            return False
    
    def get_bot_mode(self):
        """
        Get the current bot operating mode
        
        Returns:
            Current operating mode
        """
        return self.bot_mode
    
    def _get_test_symbol(self):
        """Get symbol for testing"""
        return "XRP"  # Using XRP as requested by the user
    
    def _generate_test_signal(self, signal_type="BUY"):
        """Generate a test signal for the specified direction"""
        symbol = self._get_test_symbol()
        
        # Get market data
        market_data = self.adapter.get_market_data(symbol)
        
        if "error" in market_data:
            self.logger.error(f"Error getting market data: {market_data['error']}")
            return None
        
        # Generate signal
        signal = self.signal_generator.generate_signal(market_data)
        
        # Override signal direction for testing
        signal["signal"] = signal_type
        signal["confidence"] = 0.85  # High confidence for testing
        
        # Ensure we have price and stop loss
        if "price" not in signal or signal["price"] <= 0:
            current_price = self.adapter.get_current_price(symbol)
            signal["price"] = current_price
        
        if "stop_loss" not in signal or signal["stop_loss"] <= 0:
            if signal_type == "BUY":
                signal["stop_loss"] = signal["price"] * 0.95  # 5% below for long
            else:
                signal["stop_loss"] = signal["price"] * 1.05  # 5% above for short
        
        if "take_profit" not in signal or signal["take_profit"] <= 0:
            if signal_type == "BUY":
                signal["take_profit"] = signal["price"] * 1.1  # 10% above for long
            else:
                signal["take_profit"] = signal["price"] * 0.9  # 10% below for short
        
        signal["symbol"] = symbol
        
        return signal
    
    def test_long_entry(self):
        """Test opening a long position"""
        self.logger.info("Testing long position entry...")
        
        try:
            # Check if we're in test mode
            if self.bot_mode == "live":
                self.logger.warning("WARNING: Running in LIVE mode. Real trades will be executed!")
            
            # Generate buy signal
            signal = self._generate_test_signal("BUY")
            
            if not signal:
                self.test_results["long_entry"] = {
                    "status": "Failed",
                    "details": {"error": "Could not generate signal"}
                }
                return False
            
            # Open position
            result = self.position_manager.open_position(signal["symbol"], signal)
            
            if "error" in result:
                self.test_results["long_entry"] = {
                    "status": "Failed",
                    "details": {"error": result["error"]}
                }
                return False
            
            # Verify position was opened
            positions = self.position_manager.get_current_positions()
            symbol = signal["symbol"]
            
            if symbol not in positions:
                self.test_results["long_entry"] = {
                    "status": "Failed",
                    "details": {"error": "Position not found after opening"}
                }
                return False
            
            position = positions[symbol]
            is_long = position.get("size", 0) > 0
            
            if not is_long:
                self.test_results["long_entry"] = {
                    "status": "Failed",
                    "details": {"error": "Position is not long"}
                }
                return False
            
            self.test_results["long_entry"] = {
                "status": "Passed",
                "details": {
                    "position": position,
                    "order_result": result
                }
            }
            return True
            
        except Exception as e:
            self.logger.error(f"Error testing long entry: {e}")
            self.test_results["long_entry"] = {
                "status": "Failed",
                "details": {"error": str(e), "traceback": traceback.format_exc()}
            }
            return False
    
    def test_long_exit(self):
        """Test closing a long position"""
        self.logger.info("Testing long position exit...")
        
        try:
            # Check if we're in test mode
            if self.bot_mode == "live":
                self.logger.warning("WARNING: Running in LIVE mode. Real trades will be executed!")
            
            # Check if we have a long position
            positions = self.position_manager.get_current_positions()
            symbol = self._get_test_symbol()
            
            if symbol not in positions:
                # Open a long position first
                if not self.test_long_entry():
                    self.test_results["long_exit"] = {
                        "status": "Failed",
                        "details": {"error": "Could not open long position for exit test"}
                    }
                    return False
            
            # Close position
            result = self.position_manager.close_position(symbol)
            
            if "error" in result:
                self.test_results["long_exit"] = {
                    "status": "Failed",
                    "details": {"error": result["error"]}
                }
                return False
            
            # Verify position was closed
            positions = self.position_manager.get_current_positions()
            
            if symbol in positions:
                self.test_results["long_exit"] = {
                    "status": "Failed",
                    "details": {"error": "Position still exists after closing"}
                }
                return False
            
            self.test_results["long_exit"] = {
                "status": "Passed",
                "details": {
                    "close_result": result
                }
            }
            return True
            
        except Exception as e:
            self.logger.error(f"Error testing long exit: {e}")
            self.test_results["long_exit"] = {
                "status": "Failed",
                "details": {"error": str(e), "traceback": traceback.format_exc()}
            }
            return False
    
    def test_short_entry(self):
        """Test opening a short position"""
        self.logger.info("Testing short position entry...")
        
        try:
            # Check if we're in test mode
            if self.bot_mode == "live":
                self.logger.warning("WARNING: Running in LIVE mode. Real trades will be executed!")
            
            # Generate sell signal
            signal = self._generate_test_signal("SELL")
            
            if not signal:
                self.test_results["short_entry"] = {
                    "status": "Failed",
                    "details": {"error": "Could not generate signal"}
                }
                return False
            
            # Open position
            result = self.position_manager.open_position(signal["symbol"], signal)
            
            if "error" in result:
                self.test_results["short_entry"] = {
                    "status": "Failed",
                    "details": {"error": result["error"]}
                }
                return False
            
            # Verify position was opened
            positions = self.position_manager.get_current_positions()
            symbol = signal["symbol"]
            
            if symbol not in positions:
                self.test_results["short_entry"] = {
                    "status": "Failed",
                    "details": {"error": "Position not found after opening"}
                }
                return False
            
            position = positions[symbol]
            is_short = position.get("size", 0) < 0
            
            if not is_short:
                self.test_results["short_entry"] = {
                    "status": "Failed",
                    "details": {"error": "Position is not short"}
                }
                return False
            
            self.test_results["short_entry"] = {
                "status": "Passed",
                "details": {
                    "position": position,
                    "order_result": result
                }
            }
            return True
            
        except Exception as e:
            self.logger.error(f"Error testing short entry: {e}")
            self.test_results["short_entry"] = {
                "status": "Failed",
                "details": {"error": str(e), "traceback": traceback.format_exc()}
            }
            return False
    
    def test_short_exit(self):
        """Test closing a short position"""
        self.logger.info("Testing short position exit...")
        
        try:
            # Check if we're in test mode
            if self.bot_mode == "live":
                self.logger.warning("WARNING: Running in LIVE mode. Real trades will be executed!")
            
            # Check if we have a short position
            positions = self.position_manager.get_current_positions()
            symbol = self._get_test_symbol()
            
            if symbol not in positions:
                # Open a short position first
                if not self.test_short_entry():
                    self.test_results["short_exit"] = {
                        "status": "Failed",
                        "details": {"error": "Could not open short position for exit test"}
                    }
                    return False
            
            # Close position
            result = self.position_manager.close_position(symbol)
            
            if "error" in result:
                self.test_results["short_exit"] = {
                    "status": "Failed",
                    "details": {"error": result["error"]}
                }
                return False
            
            # Verify position was closed
            positions = self.position_manager.get_current_positions()
            
            if symbol in positions:
                self.test_results["short_exit"] = {
                    "status": "Failed",
                    "details": {"error": "Position still exists after closing"}
                }
                return False
            
            self.test_results["short_exit"] = {
                "status": "Passed",
                "details": {
                    "close_result": result
                }
            }
            return True
            
        except Exception as e:
            self.logger.error(f"Error testing short exit: {e}")
            self.test_results["short_exit"] = {
                "status": "Failed",
                "details": {"error": str(e), "traceback": traceback.format_exc()}
            }
            return False
    
    def test_position_reversal(self):
        """Test reversing a position from long to short or vice versa"""
        self.logger.info("Testing position reversal...")
        
        try:
            # Check if we're in test mode
            if self.bot_mode == "live":
                self.logger.warning("WARNING: Running in LIVE mode. Real trades will be executed!")
            
            symbol = self._get_test_symbol()
            positions = self.position_manager.get_current_positions()
            
            # Determine current position state
            has_position = symbol in positions
            is_long = False
            
            if has_position:
                is_long = positions[symbol].get("size", 0) > 0
                self.logger.info(f"Current position is {'long' if is_long else 'short'}")
            else:
                # Open a long position first
                if not self.test_long_entry():
                    self.test_results["position_reversal"] = {
                        "status": "Failed",
                        "details": {"error": "Could not open initial position for reversal test"}
                    }
                    return False
                is_long = True
            
            # Generate opposite signal
            signal_type = "SELL" if is_long else "BUY"
            signal = self._generate_test_signal(signal_type)
            
            if not signal:
                self.test_results["position_reversal"] = {
                    "status": "Failed",
                    "details": {"error": "Could not generate signal for reversal"}
                }
                return False
            
            # Ensure high confidence for reversal
            signal["confidence"] = 0.9
            
            # Open position (should trigger reversal)
            result = self.position_manager.open_position(symbol, signal)
            
            if "error" in result:
                self.test_results["position_reversal"] = {
                    "status": "Failed",
                    "details": {"error": result["error"]}
                }
                return False
            
            # Verify position was reversed
            positions = self.position_manager.get_current_positions()
            
            if symbol not in positions:
                self.test_results["position_reversal"] = {
                    "status": "Failed",
                    "details": {"error": "Position not found after reversal"}
                }
                return False
            
            new_position = positions[symbol]
            new_is_long = new_position.get("size", 0) > 0
            
            if new_is_long == is_long:
                self.test_results["position_reversal"] = {
                    "status": "Failed",
                    "details": {"error": "Position direction did not change after reversal"}
                }
                return False
            
            self.test_results["position_reversal"] = {
                "status": "Passed",
                "details": {
                    "original_direction": "long" if is_long else "short",
                    "new_direction": "long" if new_is_long else "short",
                    "reversal_result": result
                }
            }
            return True
            
        except Exception as e:
            self.logger.error(f"Error testing position reversal: {e}")
            self.test_results["position_reversal"] = {
                "status": "Failed",
                "details": {"error": str(e), "traceback": traceback.format_exc()}
            }
            return False
    
    def test_partial_exit(self):
        """Test partial position exit"""
        self.logger.info("Testing partial position exit...")
        
        try:
            # Check if we're in test mode
            if self.bot_mode == "live":
                self.logger.warning("WARNING: Running in LIVE mode. Real trades will be executed!")
            
            symbol = self._get_test_symbol()
            positions = self.position_manager.get_current_positions()
            
            # Ensure we have a position
            if symbol not in positions:
                # Open a position first
                if not self.test_long_entry():
                    self.test_results["partial_exit"] = {
                        "status": "Failed",
                        "details": {"error": "Could not open position for partial exit test"}
                    }
                    return False
            
            # Get initial position size
            initial_position = self.position_manager.get_current_positions()[symbol]
            initial_size = abs(initial_position.get("size", 0))
            
            if initial_size <= 0:
                self.test_results["partial_exit"] = {
                    "status": "Failed",
                    "details": {"error": "Invalid initial position size"}
                }
                return False
            
            # Close 50% of position
            result = self.position_manager.close_position(symbol, 50.0)
            
            if "error" in result:
                self.test_results["partial_exit"] = {
                    "status": "Failed",
                    "details": {"error": result["error"]}
                }
                return False
            
            # Verify position was partially closed
            positions = self.position_manager.get_current_positions()
            
            if symbol not in positions:
                self.test_results["partial_exit"] = {
                    "status": "Failed",
                    "details": {"error": "Position completely closed instead of partially"}
                }
                return False
            
            new_position = positions[symbol]
            new_size = abs(new_position.get("size", 0))
            
            # Check if size reduced by approximately 50%
            # Allow for some rounding/precision differences
            size_ratio = new_size / initial_size
            if size_ratio < 0.4 or size_ratio > 0.6:
                self.test_results["partial_exit"] = {
                    "status": "Failed",
                    "details": {
                        "error": f"Position not reduced by expected amount. Initial: {initial_size}, New: {new_size}, Ratio: {size_ratio}"
                    }
                }
                return False
            
            self.test_results["partial_exit"] = {
                "status": "Passed",
                "details": {
                    "initial_size": initial_size,
                    "new_size": new_size,
                    "reduction_percentage": (1 - size_ratio) * 100,
                    "close_result": result
                }
            }
            return True
            
        except Exception as e:
            self.logger.error(f"Error testing partial exit: {e}")
            self.test_results["partial_exit"] = {
                "status": "Failed",
                "details": {"error": str(e), "traceback": traceback.format_exc()}
            }
            return False
    
    def test_trailing_stop(self):
        """Test trailing stop functionality"""
        self.logger.info("Testing trailing stop...")
        
        try:
            # Check if we're in test mode
            if self.bot_mode == "live":
                self.logger.warning("WARNING: Running in LIVE mode. Real trades will be executed!")
            
            symbol = self._get_test_symbol()
            
            # Ensure we have a position
            positions = self.position_manager.get_current_positions()
            if symbol not in positions:
                # Open a position first
                if not self.test_long_entry():
                    self.test_results["trailing_stop"] = {
                        "status": "Failed",
                        "details": {"error": "Could not open position for trailing stop test"}
                    }
                    return False
            
            # Get initial stop loss
            position_history = self.position_manager.state["position_history"].get(symbol, {})
            initial_stop = position_history.get("stop_loss", 0.0)
            
            if initial_stop <= 0:
                # No stop loss set, use signal data
                signal = position_history.get("signal", {})
                initial_stop = signal.get("stop_loss", 0.0)
            
            if initial_stop <= 0:
                self.test_results["trailing_stop"] = {
                    "status": "Failed",
                    "details": {"error": "No initial stop loss found"}
                }
                return False
            
            # Apply trailing stop
            result = self.position_manager.apply_trailing_stop(symbol)
            
            # Check if trailing stop was applied
            # Note: In a real test, we would simulate price movement to trigger the trailing stop
            # Here we're just checking if the method executes without error
            
            if "error" in result:
                self.test_results["trailing_stop"] = {
                    "status": "Failed",
                    "details": {"error": result["error"]}
                }
                return False
            
            self.test_results["trailing_stop"] = {
                "status": "Passed",
                "details": {
                    "initial_stop": initial_stop,
                    "trailing_stop_result": result
                }
            }
            return True
            
        except Exception as e:
            self.logger.error(f"Error testing trailing stop: {e}")
            self.test_results["trailing_stop"] = {
                "status": "Failed",
                "details": {"error": str(e), "traceback": traceback.format_exc()}
            }
            return False
    
    def test_emergency_close(self):
        """Test emergency close functionality"""
        self.logger.info("Testing emergency close...")
        
        try:
            # Check if we're in test mode
            if self.bot_mode == "live":
                self.logger.warning("WARNING: Running in LIVE mode. Real trades will be executed!")
            
            # Ensure we have at least one position
            positions = self.position_manager.get_current_positions()
            
            if not positions:
                # Open a position first
                if not self.test_long_entry():
                    self.test_results["emergency_close"] = {
                        "status": "Failed",
                        "details": {"error": "Could not open position for emergency close test"}
                    }
                    return False
            
            # Execute emergency close
            result = self.position_manager.emergency_close_all()
            
            if "error" in result:
                self.test_results["emergency_close"] = {
                    "status": "Failed",
                    "details": {"error": result["error"]}
                }
                return False
            
            # Verify all positions were closed
            positions = self.position_manager.get_current_positions()
            
            if positions:
                self.test_results["emergency_close"] = {
                    "status": "Failed",
                    "details": {"error": "Positions still exist after emergency close"}
                }
                return False
            
            # Verify emergency mode is active
            if not self.position_manager.state["emergency_mode"]:
                self.test_results["emergency_close"] = {
                    "status": "Failed",
                    "details": {"error": "Emergency mode not activated after emergency close"}
                }
                return False
            
            self.test_results["emergency_close"] = {
                "status": "Passed",
                "details": {
                    "emergency_close_result": result
                }
            }
            return True
            
        except Exception as e:
            self.logger.error(f"Error testing emergency close: {e}")
            self.test_results["emergency_close"] = {
                "status": "Failed",
                "details": {"error": str(e), "traceback": traceback.format_exc()}
            }
            return False
    
    def run_all_tests(self):
        """Run all position management tests"""
        self.logger.info(f"Starting position management tests in {self.bot_mode.upper()} mode...")
        
        # Reset position manager state
        self.position_manager.state["emergency_mode"] = False
        self.position_manager.state["trading_paused"] = False
        
        # Run tests
        tests = [
            self.test_long_entry,
            self.test_long_exit,
            self.test_short_entry,
            self.test_short_exit,
            self.test_position_reversal,
            self.test_partial_exit,
            self.test_trailing_stop,
            self.test_emergency_close
        ]
        
        passed_tests = 0
        for test in tests:
            try:
                if test():
                    passed_tests += 1
            except Exception as e:
                self.logger.error(f"Unhandled exception in test {test.__name__}: {e}")
                self.logger.error(traceback.format_exc())
        
        # Calculate success score
        self.success_score = (passed_tests / len(tests)) * 100
        
        self.logger.info(f"Tests completed. Success score: {self.success_score:.2f}%")
        
        return self.get_results()
    
    def get_results(self):
        """Get test results and success score"""
        return {
            "success_score": self.success_score,
            "tests": self.test_results,
            "bot_mode": self.bot_mode,
            "timestamp": datetime.now().isoformat()
        }
    
    def save_results(self, file_path="position_management_test_results.json"):
        """Save test results to file"""
        try:
            results = self.get_results()
            
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Test results saved to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving test results: {e}")
            return False

if __name__ == "__main__":
    try:
        # Run tests
        tester = PositionManagementTester()
        
        # Set bot mode to test by default for safety
        if tester.get_bot_mode() == "live":
            print("\n" + "="*50)
            print("WARNING: Bot is in LIVE mode. Real trades will be executed!")
            print("="*50)
            response = input("Do you want to continue in LIVE mode? (yes/no): ")
            if response.lower() != "yes":
                print("Switching to TEST mode for safety...")
                tester.set_bot_mode("test")
        
        print(f"\nRunning tests in {tester.get_bot_mode().upper()} mode...\n")
        
        # Run tests
        results = tester.run_all_tests()
        
        # Save results
        tester.save_results()
        
        # Print summary
        print("\n" + "="*50)
        print(f"POSITION MANAGEMENT TEST RESULTS: {results['success_score']:.2f}% SUCCESS")
        print(f"Bot Mode: {results['bot_mode'].upper()}")
        print("="*50)
        
        for test_name, test_result in results['tests'].items():
            status = test_result['status']
            status_str = f"[{'✓' if status == 'Passed' else '✗'}] {status}"
            print(f"{test_name.ljust(20)}: {status_str}")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        logger.error(traceback.format_exc())
        print(f"Error running tests: {e}")
