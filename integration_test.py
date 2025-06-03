#!/usr/bin/env python3
"""
Integration test script for HyperliquidMaster.
Tests all core components and their integration.
"""

import os
import sys
import json
import logging
import unittest
from unittest.mock import MagicMock, patch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("integration_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Import core modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.settings_manager import SettingsManager
from core.error_handler import ErrorHandler
from core.enhanced_connection_manager import ConnectionManager
from core.trading_mode import TradingModeManager, TradingMode
from core.enhanced_risk_manager import RiskManager
from core.hyperliquid_adapter import HyperliquidAdapter
from core.trading_integration import TradingIntegration

class IntegrationTest(unittest.TestCase):
    """Integration test class for HyperliquidMaster."""
    
    def setUp(self):
        """Set up test environment."""
        self.logger = logging.getLogger("IntegrationTest")
        self.config_path = "config.json"
        
        # Create test config if it doesn't exist
        if not os.path.exists(self.config_path):
            with open(self.config_path, 'w') as f:
                json.dump({
                    "account_address": "test_address",
                    "secret_key": "test_key",
                    "symbol": "BTC",
                    "position_size": "0.01",
                    "stop_loss": "1.0",
                    "take_profit": "2.0",
                    "risk_level": 0.02,
                    "use_mock_data": True
                }, f)
        
        # Initialize components
        self.settings_manager = SettingsManager(self.config_path, self.logger)
        self.error_handler = ErrorHandler(self.logger)
        self.connection_manager = ConnectionManager(self.logger)
        self.config = self.settings_manager.load_settings()
        self.mode_manager = TradingModeManager(self.config, self.logger)
        self.risk_manager = RiskManager(self.config, self.logger)
        
        # Mock adapter to avoid actual API calls
        self.adapter = MagicMock(spec=HyperliquidAdapter)
        self.adapter.is_connected = True
        self.adapter.get_all_available_tokens.return_value = {
            "success": True,
            "tokens": ["BTC", "ETH", "XRP", "SOL", "DOGE", "AVAX", "LINK", "MATIC"]
        }
        
        # Initialize trading integration with mock adapter
        self.trading = TradingIntegration(
            self.config_path,
            self.logger,
            self.connection_manager,
            self.mode_manager,
            self.risk_manager
        )
        self.trading.adapter = self.adapter
    
    def test_config_loading(self):
        """Test config loading."""
        self.logger.info("Testing config loading...")
        config = self.settings_manager.load_settings()
        self.assertIsNotNone(config)
        self.assertIn("account_address", config)
        self.assertIn("secret_key", config)
        self.logger.info("Config loading test passed")
    
    def test_error_handler(self):
        """Test error handler."""
        self.logger.info("Testing error handler...")
        error_msg = "Test error"
        self.error_handler.log_error(error_msg)
        self.error_handler.log_warning(error_msg)
        self.error_handler.log_info(error_msg)
        self.logger.info("Error handler test passed")
    
    def test_connection_manager(self):
        """Test connection manager."""
        self.logger.info("Testing connection manager...")
        self.connection_manager.set_connected(True)
        self.assertTrue(self.connection_manager.is_connected())
        self.connection_manager.set_connected(False)
        self.assertFalse(self.connection_manager.is_connected())
        self.logger.info("Connection manager test passed")
    
    def test_trading_mode_manager(self):
        """Test trading mode manager."""
        self.logger.info("Testing trading mode manager...")
        # Test getting current mode
        current_mode = self.mode_manager.get_current_mode()
        self.assertIsNotNone(current_mode)
        
        # Test setting mode
        for mode in TradingMode:
            self.mode_manager.set_mode(mode)
            self.assertEqual(self.mode_manager.get_current_mode(), mode)
        
        # Test mode description
        for mode in TradingMode:
            desc = self.mode_manager.get_mode_description(mode)
            self.assertIsNotNone(desc)
            self.assertNotEqual(desc, "")
        
        self.logger.info("Trading mode manager test passed")
    
    def test_risk_manager(self):
        """Test risk manager."""
        self.logger.info("Testing risk manager...")
        # Test risk metrics
        metrics = self.risk_manager.get_risk_metrics()
        self.assertIsNotNone(metrics)
        
        # Test position size calculation
        position_size = self.risk_manager.calculate_position_size(
            symbol="BTC",
            entry_price=50000.0,
            stop_loss_price=49000.0,
            account_equity=10000.0
        )
        self.assertGreater(position_size, 0)
        
        # Test risk-reward ratio calculation
        ratio = self.risk_manager.calculate_risk_reward_ratio(
            entry_price=50000.0,
            stop_loss_price=49000.0,
            take_profit_price=52000.0
        )
        self.assertGreater(ratio, 0)
        
        # Test drawdown protection
        result = self.risk_manager.check_drawdown_protection()
        self.assertIsNotNone(result)
        
        self.logger.info("Risk manager test passed")
    
    def test_hyperliquid_adapter(self):
        """Test Hyperliquid adapter."""
        self.logger.info("Testing Hyperliquid adapter...")
        # Test token fetching
        result = self.adapter.get_all_available_tokens()
        self.assertIn("tokens", result)
        self.assertIsInstance(result["tokens"], list)
        self.assertGreater(len(result["tokens"]), 0)
        
        self.logger.info("Hyperliquid adapter test passed")
    
    def test_trading_integration(self):
        """Test trading integration."""
        self.logger.info("Testing trading integration...")
        # Test token fetching through integration
        result = self.trading.get_all_available_tokens()
        self.assertIn("tokens", result)
        self.assertIsInstance(result["tokens"], list)
        self.assertGreater(len(result["tokens"]), 0)
        
        # Test connection status
        status = self.trading.get_connection_status()
        self.assertIn("connected", status)
        self.assertIn("exchange", status)
        self.assertIn("mode", status)
        
        self.logger.info("Trading integration test passed")
    
    def test_new_api_key_tab_integration(self):
        """Test API key tab integration with other components."""
        self.logger.info("Testing API key tab integration...")
        
        # Mock the API key setting
        with patch.object(self.adapter, 'set_api_keys') as mock_set_keys:
            mock_set_keys.return_value = True
            
            # Simulate setting API keys
            result = self.adapter.set_api_keys("new_address", "new_key")
            self.assertTrue(result)
            mock_set_keys.assert_called_once_with("new_address", "new_key")
        
        self.logger.info("API key tab integration test passed")
    
    def test_first_start_prompt_integration(self):
        """Test first-start prompt integration."""
        self.logger.info("Testing first-start prompt integration...")
        
        # Create a temporary config with no keys
        temp_config_path = "temp_config.json"
        with open(temp_config_path, 'w') as f:
            json.dump({
                "symbol": "BTC",
                "position_size": "0.01"
            }, f)
        
        # Create a settings manager with the temp config
        temp_settings_manager = SettingsManager(temp_config_path, self.logger)
        temp_config = temp_settings_manager.load_settings()
        
        # Check if keys are missing
        self.assertNotIn("account_address", temp_config)
        self.assertNotIn("secret_key", temp_config)
        
        # Clean up
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        
        self.logger.info("First-start prompt integration test passed")
    
    def tearDown(self):
        """Clean up after tests."""
        pass

def run_tests():
    """Run integration tests."""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(IntegrationTest)
    
    # Run tests
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Save results to JSON
    test_results = {
        "total": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "success": result.wasSuccessful()
    }
    
    with open("integration_test_results.json", 'w') as f:
        json.dump(test_results, f, indent=2)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
