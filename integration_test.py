#!/usr/bin/env python3
"""
Integration test script for HyperliquidMaster

This script tests the core functionality and integration between components
to ensure everything works correctly before committing changes.
"""

import os
import sys
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("integration_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("IntegrationTest")

# Import core modules
try:
    from core.trading_integration import TradingIntegration
    from core.error_handler import ErrorHandler
    from core.trading_mode import TradingModeManager, TradingMode
    from core.enhanced_connection_manager import ConnectionManager
    from core.settings_manager import SettingsManager
    from core.enhanced_risk_manager import RiskManager
    from core.advanced_order_manager import AdvancedOrderManager
    from core.position_manager_wrapper import PositionManagerWrapper
    from core.hyperliquid_adapter import HyperliquidAdapter
    
    logger.info("Successfully imported all core modules")
except ImportError as e:
    logger.error(f"Error importing core modules: {e}")
    sys.exit(1)

def test_config_loading():
    """Test configuration loading"""
    logger.info("Testing configuration loading...")
    
    config_path = "config.json"
    
    # Ensure config exists
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} does not exist, creating test config")
        
        test_config = {
            "api_key": "test_api_key",
            "api_secret": "test_api_secret",
            "account_address": "test_account_address",
            "secret_key": "test_secret_key",
            "use_testnet": True,
            "symbol": "XRP",
            "position_size": 0.01,
            "stop_loss": 1.0,
            "take_profit": 2.0,
            "risk_level": 0.02,
            "use_volatility_filters": True,
            "use_trend_filters": True,
            "use_volume_filters": True,
            "tp_multiplier": 2.0,
            "sl_multiplier": 1.0
        }
        
        with open(config_path, 'w') as f:
            json.dump(test_config, f, indent=2)
        
        logger.info(f"Created test config at {config_path}")
    
    # Test loading config
    try:
        settings_manager = SettingsManager(config_path, logger)
        config = settings_manager.load_settings()
        
        logger.info(f"Successfully loaded config: {config.keys()}")
        return True
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return False

def test_error_handler():
    """Test error handler functionality"""
    logger.info("Testing error handler...")
    
    try:
        error_handler = ErrorHandler(logger)
        
        # Test handling an error
        try:
            # Deliberately cause an error
            result = 1 / 0
        except Exception as e:
            error_record = error_handler.handle_error(e, "test_division", {"test": True})
            
            logger.info(f"Successfully handled error: {error_record['type']}")
        
        # Test getting error history
        history = error_handler.get_error_history()
        
        logger.info(f"Successfully retrieved error history: {len(history)} errors")
        return True
    except Exception as e:
        logger.error(f"Error testing error handler: {e}")
        return False

def test_connection_manager():
    """Test connection manager functionality"""
    logger.info("Testing connection manager...")
    
    try:
        connection_manager = ConnectionManager(logger)
        
        # Test connection
        connected = connection_manager.connect()
        
        logger.info(f"Connection test result: {connected}")
        
        # Test getting connection status
        status = connection_manager.get_connection_status()
        
        logger.info(f"Connection status: {status}")
        
        # Test is_connected method
        is_connected = connection_manager.is_connected()
        
        logger.info(f"Is connected: {is_connected}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing connection manager: {e}")
        return False

def test_trading_mode_manager():
    """Test trading mode manager functionality"""
    logger.info("Testing trading mode manager...")
    
    try:
        config_path = "config.json"
        settings_manager = SettingsManager(config_path, logger)
        config = settings_manager.load_settings()
        
        mode_manager = TradingModeManager(config, logger)
        
        # Test getting current mode
        current_mode = mode_manager.get_current_mode()
        
        logger.info(f"Current mode: {current_mode}")
        
        # Test setting mode
        for mode in [TradingMode.PAPER_TRADING, TradingMode.MONITOR_ONLY, TradingMode.CONSERVATIVE]:
            result = mode_manager.set_mode(mode)
            logger.info(f"Set mode to {mode}: {result}")
            
            # Verify mode was set
            new_mode = mode_manager.get_current_mode()
            logger.info(f"New mode: {new_mode}")
            
            if new_mode != mode:
                logger.error(f"Mode was not set correctly: expected {mode}, got {new_mode}")
                return False
        
        # Reset to PAPER_TRADING
        mode_manager.set_mode(TradingMode.PAPER_TRADING)
        
        return True
    except Exception as e:
        logger.error(f"Error testing trading mode manager: {e}")
        return False

def test_risk_manager():
    """Test risk manager functionality"""
    logger.info("Testing risk manager...")
    
    try:
        config_path = "config.json"
        settings_manager = SettingsManager(config_path, logger)
        config = settings_manager.load_settings()
        
        risk_manager = RiskManager(config, logger)
        
        # Test calculating position size
        position_size = risk_manager.calculate_position_size(
            symbol="XRP",
            entry_price=1.0,
            stop_loss_price=0.9,
            account_equity=1000.0
        )
        
        logger.info(f"Calculated position size: {position_size}")
        
        # Test calculating risk-reward ratio
        ratio = risk_manager.calculate_risk_reward_ratio(
            entry_price=1.0,
            stop_loss_price=0.9,
            take_profit_price=1.2
        )
        
        logger.info(f"Calculated risk-reward ratio: {ratio}")
        
        # Test validating risk-reward ratio
        is_valid = risk_manager.validate_risk_reward_ratio(
            entry_price=1.0,
            stop_loss_price=0.9,
            take_profit_price=1.2
        )
        
        logger.info(f"Risk-reward validation: {is_valid}")
        
        # Test checking drawdown protection
        trading_allowed = risk_manager.check_drawdown_protection()
        
        logger.info(f"Drawdown protection check: {trading_allowed}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing risk manager: {e}")
        return False

def test_hyperliquid_adapter():
    """Test hyperliquid adapter functionality"""
    logger.info("Testing hyperliquid adapter...")
    
    try:
        # Test with testnet
        adapter = HyperliquidAdapter(use_testnet=True)
        
        # Test is_connected attribute
        logger.info(f"Adapter is_connected attribute: {adapter.is_connected}")
        
        # Test connection
        connection_result = adapter.connect()
        
        logger.info(f"Connection result: {connection_result}")
        
        # Test is_connected attribute after connection attempt
        logger.info(f"Adapter is_connected attribute after connection: {adapter.is_connected}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing hyperliquid adapter: {e}")
        return False

def test_trading_integration():
    """Test trading integration functionality"""
    logger.info("Testing trading integration...")
    
    try:
        config_path = "config.json"
        
        # Initialize trading integration
        trading = TradingIntegration(config_path, logger)
        
        # Test is_connected attribute
        logger.info(f"Trading integration is_connected attribute: {trading.is_connected}")
        
        # Test getting connection status
        status = trading.get_connection_status()
        
        logger.info(f"Connection status: {status}")
        
        # Test getting trading mode
        mode = trading.get_trading_mode()
        
        logger.info(f"Trading mode: {mode}")
        
        # Test setting trading mode
        result = trading.set_trading_mode("PAPER_TRADING")
        
        logger.info(f"Set trading mode result: {result}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing trading integration: {e}")
        return False

def run_all_tests():
    """Run all integration tests"""
    logger.info("Starting integration tests...")
    
    tests = [
        ("Config Loading", test_config_loading),
        ("Error Handler", test_error_handler),
        ("Connection Manager", test_connection_manager),
        ("Trading Mode Manager", test_trading_mode_manager),
        ("Risk Manager", test_risk_manager),
        ("Hyperliquid Adapter", test_hyperliquid_adapter),
        ("Trading Integration", test_trading_integration)
    ]
    
    results = {}
    all_passed = True
    
    for name, test_func in tests:
        logger.info(f"Running test: {name}")
        
        try:
            result = test_func()
            results[name] = result
            
            if not result:
                all_passed = False
                
            logger.info(f"Test {name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            logger.error(f"Error running test {name}: {e}")
            results[name] = False
            all_passed = False
    
    # Print summary
    logger.info("Integration Test Summary:")
    for name, result in results.items():
        logger.info(f"  {name}: {'PASSED' if result else 'FAILED'}")
    
    logger.info(f"Overall result: {'PASSED' if all_passed else 'FAILED'}")
    
    return all_passed, results

if __name__ == "__main__":
    success, results = run_all_tests()
    
    # Save results to file
    with open("integration_test_results.json", 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "results": {k: "PASSED" if v else "FAILED" for k, v in results.items()}
        }, f, indent=2)
    
    sys.exit(0 if success else 1)
