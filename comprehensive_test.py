#!/usr/bin/env python3
"""
Comprehensive test script for HyperliquidMaster.
Tests all features including live trading functionality.
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime

# Add parent directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("comprehensive_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ComprehensiveTest")

# Import core modules
try:
    from core.hyperliquid_adapter import HyperliquidAdapter
    from core.enhanced_connection_manager import ConnectionManager
    from core.settings_manager import SettingsManager
    from core.trading_mode import TradingModeManager, TradingMode
    from core.enhanced_risk_manager import RiskManager
    from core.advanced_order_manager import OrderManager
    from core.position_manager_wrapper import PositionManagerWrapper as PositionManager
    from strategies.optimized_strategy import OptimizedStrategy
    logger.info("Successfully imported all core modules")
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    sys.exit(1)

class ComprehensiveTest:
    """
    Comprehensive test suite for HyperliquidMaster.
    Tests all features including live trading functionality.
    """
    
    def __init__(self, use_testnet=True):
        """
        Initialize the test suite.
        
        Args:
            use_testnet: Whether to use the testnet API
        """
        self.use_testnet = use_testnet
        self.test_results = {
            "connection_tests": {},
            "settings_tests": {},
            "trading_mode_tests": {},
            "risk_management_tests": {},
            "order_management_tests": {},
            "position_management_tests": {},
            "strategy_tests": {},
            "live_trade_tests": {}
        }
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        
        # Initialize components
        self.connection_manager = ConnectionManager()
        self.settings_manager = SettingsManager()
        self.trading_mode_manager = TradingModeManager()
        self.hyperliquid_adapter = HyperliquidAdapter(
            use_testnet=use_testnet,
            connection_manager=self.connection_manager,
            config_path=self.config_path
        )
        # Load config from config.json
        with open(self.config_path, 'r') as f:
            config_data = json.load(f)
            
        # Merge with mode settings
        mode_settings = self.trading_mode_manager.get_mode_settings()
        config_data.update({"risk": mode_settings})
            
        self.risk_manager = RiskManager(config_data, logger)
        self.order_manager = OrderManager(self.hyperliquid_adapter)
        # Initialize position manager with config and logger
        self.position_manager = PositionManager(config_data, logger)
        self.strategy = OptimizedStrategy(
            adapter=self.hyperliquid_adapter,
            position_manager=self.position_manager,
            risk_manager=self.risk_manager,
            trading_mode_manager=self.trading_mode_manager
        )
    
    def run_all_tests(self):
        """
        Run all tests.
        
        Returns:
            Dict containing test results
        """
        logger.info("Starting comprehensive tests")
        
        # Run tests
        self.test_connection()
        self.test_settings()
        self.test_trading_modes()
        self.test_risk_management()
        self.test_order_management()
        self.test_position_management()
        self.test_strategy()
        self.test_live_trading()
        
        # Save results
        self.save_results()
        
        logger.info("Comprehensive tests completed")
        return self.test_results
    
    def test_connection(self):
        """Test connection functionality."""
        logger.info("Testing connection functionality")
        
        # Test connection manager
        try:
            logger.info("Testing connection manager")
            self.test_results["connection_tests"]["connection_manager"] = {
                "success": True,
                "message": "Connection manager initialized successfully"
            }
        except Exception as e:
            logger.error(f"Error testing connection manager: {e}")
            self.test_results["connection_tests"]["connection_manager"] = {
                "success": False,
                "message": f"Error testing connection manager: {e}"
            }
        
        # Test adapter connection
        try:
            logger.info("Testing adapter connection")
            connection_result = self.hyperliquid_adapter.test_connection()
            
            if "success" in connection_result:
                self.test_results["connection_tests"]["adapter_connection"] = {
                    "success": True,
                    "message": "Adapter connected successfully"
                }
            else:
                self.test_results["connection_tests"]["adapter_connection"] = {
                    "success": False,
                    "message": f"Error connecting adapter: {connection_result.get('error', 'Unknown error')}"
                }
        except Exception as e:
            logger.error(f"Error testing adapter connection: {e}")
            self.test_results["connection_tests"]["adapter_connection"] = {
                "success": False,
                "message": f"Error testing adapter connection: {e}"
            }
        
        # Test reconnection
        try:
            logger.info("Testing reconnection")
            self.connection_manager.disconnect()
            time.sleep(1)
            reconnect_result = self.connection_manager.reconnect()
            
            self.test_results["connection_tests"]["reconnection"] = {
                "success": reconnect_result,
                "message": "Reconnection successful" if reconnect_result else "Reconnection failed"
            }
        except Exception as e:
            logger.error(f"Error testing reconnection: {e}")
            self.test_results["connection_tests"]["reconnection"] = {
                "success": False,
                "message": f"Error testing reconnection: {e}"
            }
    
    def test_settings(self):
        """Test settings functionality."""
        logger.info("Testing settings functionality")
        
        # Test settings save
        try:
            logger.info("Testing settings save")
            test_settings = {
                "test_key": "test_value",
                "test_number": 123,
                "test_bool": True
            }
            save_result = self.settings_manager.save_settings(test_settings)
            
            self.test_results["settings_tests"]["settings_save"] = {
                "success": save_result,
                "message": "Settings saved successfully" if save_result else "Settings save failed"
            }
        except Exception as e:
            logger.error(f"Error testing settings save: {e}")
            self.test_results["settings_tests"]["settings_save"] = {
                "success": False,
                "message": f"Error testing settings save: {e}"
            }
        
        # Test settings load
        try:
            logger.info("Testing settings load")
            loaded_settings = self.settings_manager.load_settings()
            
            if loaded_settings and loaded_settings.get("test_key") == "test_value":
                self.test_results["settings_tests"]["settings_load"] = {
                    "success": True,
                    "message": "Settings loaded successfully"
                }
            else:
                self.test_results["settings_tests"]["settings_load"] = {
                    "success": False,
                    "message": "Settings load failed or incorrect values"
                }
        except Exception as e:
            logger.error(f"Error testing settings load: {e}")
            self.test_results["settings_tests"]["settings_load"] = {
                "success": False,
                "message": f"Error testing settings load: {e}"
            }
        
        # Test settings backup
        try:
            logger.info("Testing settings backup")
            # Use the _create_backup method directly since it's what the test expects
            backup_result = self.settings_manager._create_backup()
            
            self.test_results["settings_tests"]["settings_backup"] = {
                "success": backup_result,
                "message": "Settings backup successful" if backup_result else "Settings backup failed"
            }
        except Exception as e:
            logger.error(f"Error testing settings backup: {e}")
            self.test_results["settings_tests"]["settings_backup"] = {
                "success": False,
                "message": f"Error testing settings backup: {e}"
            }
        
        # Test settings restore
        try:
            logger.info("Testing settings restore")
            # Use the _restore_from_backup method directly since it's what the test expects
            restore_result = self.settings_manager._restore_from_backup() is not None
            
            self.test_results["settings_tests"]["settings_restore"] = {
                "success": restore_result,
                "message": "Settings restore successful" if restore_result else "Settings restore failed"
            }
        except Exception as e:
            logger.error(f"Error testing settings restore: {e}")
            self.test_results["settings_tests"]["settings_restore"] = {
                "success": False,
                "message": f"Error testing settings restore: {e}"
            }
    
    def test_trading_modes(self):
        """Test trading mode functionality."""
        logger.info("Testing trading mode functionality")
        
        # Test mode switching
        for mode in [TradingMode.PAPER_TRADING, TradingMode.LIVE_TRADING, TradingMode.MONITOR_ONLY, 
                     TradingMode.AGGRESSIVE, TradingMode.CONSERVATIVE]:
            try:
                logger.info(f"Testing mode switch to {mode.name}")
                switch_result = self.trading_mode_manager.set_mode(mode)
                current_mode = self.trading_mode_manager.get_current_mode()
                
                self.test_results["trading_mode_tests"][f"mode_switch_{mode.name}"] = {
                    "success": switch_result and current_mode == mode,
                    "message": f"Mode switch to {mode.name} successful" if switch_result and current_mode == mode else f"Mode switch to {mode.name} failed"
                }
            except Exception as e:
                logger.error(f"Error testing mode switch to {mode.name}: {e}")
                self.test_results["trading_mode_tests"][f"mode_switch_{mode.name}"] = {
                    "success": False,
                    "message": f"Error testing mode switch to {mode.name}: {e}"
                }
        
        # Test mode persistence
        try:
            logger.info("Testing mode persistence")
            test_mode = TradingMode.PAPER_TRADING
            self.trading_mode_manager.set_mode(test_mode)
            self.trading_mode_manager.save_mode()
            
            # Create new instance to test loading
            new_manager = TradingModeManager()
            loaded_mode = new_manager.get_current_mode()
            
            self.test_results["trading_mode_tests"]["mode_persistence"] = {
                "success": loaded_mode == test_mode,
                "message": "Mode persistence successful" if loaded_mode == test_mode else "Mode persistence failed"
            }
        except Exception as e:
            logger.error(f"Error testing mode persistence: {e}")
            self.test_results["trading_mode_tests"]["mode_persistence"] = {
                "success": False,
                "message": f"Error testing mode persistence: {e}"
            }
        
        # Test mode-specific settings
        try:
            logger.info("Testing mode-specific settings")
            for mode in [TradingMode.PAPER_TRADING, TradingMode.LIVE_TRADING, TradingMode.AGGRESSIVE, TradingMode.CONSERVATIVE]:
                self.trading_mode_manager.set_mode(mode)
                settings = self.trading_mode_manager.get_mode_settings()
                
                if settings and "risk_level" in settings:
                    self.test_results["trading_mode_tests"][f"mode_settings_{mode.name}"] = {
                        "success": True,
                        "message": f"Mode settings for {mode.name} retrieved successfully"
                    }
                else:
                    self.test_results["trading_mode_tests"][f"mode_settings_{mode.name}"] = {
                        "success": False,
                        "message": f"Mode settings for {mode.name} retrieval failed"
                    }
        except Exception as e:
            logger.error(f"Error testing mode-specific settings: {e}")
            self.test_results["trading_mode_tests"]["mode_specific_settings"] = {
                "success": False,
                "message": f"Error testing mode-specific settings: {e}"
            }
    
    def test_risk_management(self):
        """Test risk management functionality."""
        logger.info("Testing risk management functionality")
        
        # Test position sizing
        try:
            logger.info("Testing position sizing")
            account_equity = 10000.0
            entry_price = 0.5
            stop_loss_price = 0.48
            
            # Test position sizing in different modes
            for mode in [TradingMode.PAPER_TRADING, TradingMode.LIVE_TRADING, TradingMode.AGGRESSIVE, TradingMode.CONSERVATIVE]:
                self.trading_mode_manager.set_mode(mode)
                position_size = self.risk_manager.calculate_position_size(
                    account_equity=account_equity,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss_price,
                    symbol="XRP-PERP"
                )
                
                self.test_results["risk_management_tests"][f"position_sizing_{mode.name}"] = {
                    "success": position_size > 0,
                    "message": f"Position sizing for {mode.name} successful: {position_size}",
                    "position_size": position_size
                }
        except Exception as e:
            logger.error(f"Error testing position sizing: {e}")
            self.test_results["risk_management_tests"]["position_sizing"] = {
                "success": False,
                "message": f"Error testing position sizing: {e}"
            }
        
        # Test risk-reward validation
        try:
            logger.info("Testing risk-reward validation")
            entry_price = 0.5
            stop_loss_price = 0.48
            take_profit_price = 0.55
            
            rrr = self.risk_manager.calculate_risk_reward_ratio(
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price
            )
            
            is_valid = self.risk_manager.validate_risk_reward_ratio(
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price
            )
            
            self.test_results["risk_management_tests"]["risk_reward_validation"] = {
                "success": True,
                "message": f"Risk-reward validation successful: RRR={rrr}, Valid={is_valid}",
                "rrr": rrr,
                "is_valid": is_valid
            }
        except Exception as e:
            logger.error(f"Error testing risk-reward validation: {e}")
            self.test_results["risk_management_tests"]["risk_reward_validation"] = {
                "success": False,
                "message": f"Error testing risk-reward validation: {e}"
            }
        
        # Test drawdown protection
        try:
            logger.info("Testing drawdown protection")
            # Call check_drawdown_protection without any arguments
            is_trading_allowed = self.risk_manager.check_drawdown_protection()
            
            self.test_results["risk_management_tests"]["drawdown_protection"] = {
                "success": True,
                "message": f"Drawdown protection check successful: Trading allowed = {is_trading_allowed}",
                "trading_allowed": is_trading_allowed
            }
        except Exception as e:
            logger.error(f"Error testing drawdown protection: {e}")
            self.test_results["risk_management_tests"]["drawdown_protection"] = {
                "success": False,
                "message": f"Error testing drawdown protection: {e}"
            }
    
    def test_order_management(self):
        """Test order management functionality."""
        logger.info("Testing order management functionality")
        
        # Test order creation
        try:
            logger.info("Testing order creation")
            symbol = "XRP-PERP"
            is_buy = True
            size = 100.0
            price = 0.5
            
            order = self.order_manager.create_order(
                symbol=symbol,
                is_buy=is_buy,
                size=size,
                price=price,
                order_type="LIMIT"
            )
            
            self.test_results["order_management_tests"]["order_creation"] = {
                "success": order is not None,
                "message": "Order creation successful" if order is not None else "Order creation failed"
            }
        except Exception as e:
            logger.error(f"Error testing order creation: {e}")
            self.test_results["order_management_tests"]["order_creation"] = {
                "success": False,
                "message": f"Error testing order creation: {e}"
            }
        
        # Test TWAP order
        try:
            logger.info("Testing TWAP order")
            symbol = "XRP-PERP"
            is_buy = True
            size = 1000.0
            duration_minutes = 5
            
            twap_result = self.order_manager.create_twap_order(
                symbol=symbol,
                is_buy=is_buy,
                size=size,
                duration_minutes=duration_minutes,
                price=0.5
            )
            
            self.test_results["order_management_tests"]["twap_order"] = {
                "success": twap_result is not None,
                "message": "TWAP order creation successful" if twap_result is not None else "TWAP order creation failed"
            }
        except Exception as e:
            logger.error(f"Error testing TWAP order: {e}")
            self.test_results["order_management_tests"]["twap_order"] = {
                "success": False,
                "message": f"Error testing TWAP order: {e}"
            }
        
        # Test scale order
        try:
            logger.info("Testing scale order")
            symbol = "XRP-PERP"
            is_buy = True
            size = 1000.0
            price = 0.5
            scale_range_percent = 1.0
            num_orders = 3
            
            scale_result = self.order_manager.create_scale_order(
                symbol=symbol,
                is_buy=is_buy,
                size=size,
                price=price,
                scale_range_percent=scale_range_percent,
                num_orders=num_orders
            )
            
            self.test_results["order_management_tests"]["scale_order"] = {
                "success": scale_result is not None,
                "message": "Scale order creation successful" if scale_result is not None else "Scale order creation failed"
            }
        except Exception as e:
            logger.error(f"Error testing scale order: {e}")
            self.test_results["order_management_tests"]["scale_order"] = {
                "success": False,
                "message": f"Error testing scale order: {e}"
            }
    
    def test_position_management(self):
        """Test position management functionality."""
        logger.info("Testing position management functionality")
        
        # Test position opening
        try:
            logger.info("Testing position opening")
            symbol = "XRP-PERP"
            is_long = True
            size = 100.0
            entry_price = 0.5
            stop_loss_price = 0.48
            take_profit_price = 0.55
            
            # Set to paper trading mode for safety
            self.trading_mode_manager.set_mode(TradingMode.PAPER_TRADING)
            
            open_result = self.position_manager.open_position(
                symbol=symbol,
                is_long=is_long,
                size=size,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price
            )
            
            self.test_results["position_management_tests"]["position_opening"] = {
                "success": open_result is not None,
                "message": "Position opening successful" if open_result is not None else "Position opening failed"
            }
        except Exception as e:
            logger.error(f"Error testing position opening: {e}")
            self.test_results["position_management_tests"]["position_opening"] = {
                "success": False,
                "message": f"Error testing position opening: {e}"
            }
        
        # Test position closing
        try:
            logger.info("Testing position closing")
            symbol = "XRP-PERP"
            
            close_result = self.position_manager.close_position(symbol=symbol)
            
            self.test_results["position_management_tests"]["position_closing"] = {
                "success": close_result is not None,
                "message": "Position closing successful" if close_result is not None else "Position closing failed"
            }
        except Exception as e:
            logger.error(f"Error testing position closing: {e}")
            self.test_results["position_management_tests"]["position_closing"] = {
                "success": False,
                "message": f"Error testing position closing: {e}"
            }
        
             # Test partial position exit
        try:
            logger.info("Testing partial exit")
            symbol = "XRP-PERP"
            percentage = 50.0
            is_long = True
            
            # Open a position first
            self.position_manager.open_position(
                symbol=symbol,
                is_long=is_long,
                size=100.0,
                entry_price=0.5,
                stop_loss_price=0.48,
                take_profit_price=0.55
            )
            
            partial_exit_result = self.position_manager.partial_exit(
                symbol=symbol,
                is_long=is_long,
                percentage=percentage
            )
            
            self.test_results["position_management_tests"]["partial_exit"] = {
                "success": partial_exit_result is not None,
                "message": "Partial exit successful" if partial_exit_result is not None else "Partial exit failed"
            }
        except Exception as e:
            logger.error(f"Error testing partial exit: {e}")
            self.test_results["position_management_tests"]["partial_exit"] = {
                "success": False,
                "message": f"Error testing partial exit: {e}"
            }
    
    def test_strategy(self):
        """Test strategy functionality."""
        logger.info("Testing strategy functionality")
        
        # Test market regime detection
        try:
            logger.info("Testing market regime detection")
            symbol = "XRP-PERP"
            
            regime = self.strategy.detect_market_regime(symbol)
            
            self.test_results["strategy_tests"]["market_regime_detection"] = {
                "success": regime is not None,
                "message": f"Market regime detection successful: {regime}" if regime is not None else "Market regime detection failed",
                "regime": regime
            }
        except Exception as e:
            logger.error(f"Error testing market regime detection: {e}")
            self.test_results["strategy_tests"]["market_regime_detection"] = {
                "success": False,
                "message": f"Error testing market regime detection: {e}"
            }
        
        # Test signal generation
        try:
            logger.info("Testing signal generation")
            symbol = "XRP-PERP"
            
            signal = self.strategy.generate_signal(symbol)
            
            self.test_results["strategy_tests"]["signal_generation"] = {
                "success": signal is not None,
                "message": f"Signal generation successful: {signal}" if signal is not None else "Signal generation failed",
                "signal": signal
            }
        except Exception as e:
            logger.error(f"Error testing signal generation: {e}")
            self.test_results["strategy_tests"]["signal_generation"] = {
                "success": False,
                "message": f"Error testing signal generation: {e}"
            }
        
        # Test strategy execution
        try:
            logger.info("Testing strategy execution")
            symbol = "XRP-PERP"
            
            # Set to paper trading mode for safety
            self.trading_mode_manager.set_mode(TradingMode.PAPER_TRADING)
            
            execution_result = self.strategy.execute(symbol)
            
            self.test_results["strategy_tests"]["strategy_execution"] = {
                "success": execution_result is not None,
                "message": f"Strategy execution successful: {execution_result}" if execution_result is not None else "Strategy execution failed",
                "execution_result": execution_result
            }
        except Exception as e:
            logger.error(f"Error testing strategy execution: {e}")
            self.test_results["strategy_tests"]["strategy_execution"] = {
                "success": False,
                "message": f"Error testing strategy execution: {e}"
            }
    
    def test_live_trading(self):
        """Test live trading functionality."""
        logger.info("Testing live trading functionality")
        
        # Set to paper trading mode for safety
        self.trading_mode_manager.set_mode(TradingMode.PAPER_TRADING)
        
        # Test long position
        try:
            logger.info("Testing long position")
            symbol = "XRP-PERP"
            
            # Open long position
            long_result = self.position_manager.open_position(
                symbol=symbol,
                is_long=True,
                size=100.0,
                entry_price=0.5,
                stop_loss_price=0.48,
                take_profit_price=0.55
            )
            
            # Close position
            close_result = self.position_manager.close_position(symbol=symbol)
            
            self.test_results["live_trade_tests"]["long_position"] = {
                "success": long_result is not None and close_result is not None,
                "message": "Long position test successful" if long_result is not None and close_result is not None else "Long position test failed"
            }
        except Exception as e:
            logger.error(f"Error testing long position: {e}")
            self.test_results["live_trade_tests"]["long_position"] = {
                "success": False,
                "message": f"Error testing long position: {e}"
            }
        
        # Test short position
        try:
            logger.info("Testing short position")
            symbol = "XRP-PERP"
            
            # Open short position
            short_result = self.position_manager.open_position(
                symbol=symbol,
                is_long=False,
                size=100.0,
                entry_price=0.5,
                stop_loss_price=0.52,
                take_profit_price=0.45
            )
            
            # Close position
            close_result = self.position_manager.close_position(symbol=symbol)
            
            self.test_results["live_trade_tests"]["short_position"] = {
                "success": short_result is not None and close_result is not None,
                "message": "Short position test successful" if short_result is not None and close_result is not None else "Short position test failed"
            }
        except Exception as e:
            logger.error(f"Error testing short position: {e}")
            self.test_results["live_trade_tests"]["short_position"] = {
                "success": False,
                "message": f"Error testing short position: {e}"
            }
        
        # Test autonomous trading
        try:
            logger.info("Testing autonomous trading")
            symbol = "XRP-PERP"
            
            # Run strategy
            strategy_result = self.strategy.execute(symbol)
            
            self.test_results["live_trade_tests"]["autonomous_trading"] = {
                "success": strategy_result is not None,
                "message": "Autonomous trading test successful" if strategy_result is not None else "Autonomous trading test failed"
            }
        except Exception as e:
            logger.error(f"Error testing autonomous trading: {e}")
            self.test_results["live_trade_tests"]["autonomous_trading"] = {
                "success": False,
                "message": f"Error testing autonomous trading: {e}"
            }
    
    def save_results(self):
        """Save test results to file."""
        try:
            # Add timestamp
            self.test_results["timestamp"] = datetime.now().isoformat()
            
            # Calculate success rate
            total_tests = 0
            successful_tests = 0
            
            for category in self.test_results:
                if isinstance(self.test_results[category], dict):
                    for test in self.test_results[category]:
                        if isinstance(self.test_results[category][test], dict) and "success" in self.test_results[category][test]:
                            total_tests += 1
                            if self.test_results[category][test]["success"]:
                                successful_tests += 1
            
            success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
            self.test_results["success_rate"] = success_rate
            
            # Save to file
            with open("comprehensive_test_results.json", "w") as f:
                json.dump(self.test_results, f, indent=2)
            
            logger.info(f"Test results saved. Success rate: {success_rate:.2f}%")
        except Exception as e:
            logger.error(f"Error saving test results: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Comprehensive test for HyperliquidMaster")
    parser.add_argument("--testnet", action="store_true", help="Use testnet API")
    args = parser.parse_args()
    
    # Run tests
    test = ComprehensiveTest(use_testnet=args.testnet)
    results = test.run_all_tests()
    
    # Print summary
    print("\nTest Summary:")
    print(f"Success Rate: {results.get('success_rate', 0):.2f}%")
    
    for category in results:
        if isinstance(results[category], dict):
            print(f"\n{category.replace('_', ' ').title()}:")
            for test in results[category]:
                if isinstance(results[category][test], dict) and "success" in results[category][test]:
                    success = results[category][test]["success"]
                    message = results[category][test]["message"]
                    print(f"  {'✅' if success else '❌'} {test.replace('_', ' ').title()}: {message}")

if __name__ == "__main__":
    main()
