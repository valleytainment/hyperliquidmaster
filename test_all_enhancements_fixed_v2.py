"""
Test script to verify all enhancements
"""

import os
import sys
import unittest
import logging
from unittest.mock import MagicMock, patch

# Configure keyring to use file backend for testing
import keyring
from keyrings.alt.file import PlaintextKeyring
keyring.set_keyring(PlaintextKeyring())

# Import modules to test
from utils.config_manager_fixed import ConfigManager
from utils.security_fixed_v2 import SecurityManager
from core.connection_manager_enhanced import EnhancedConnectionManager
from core.api_fixed_v2 import EnhancedHyperliquidAPI
from strategies.strategy_manager import StrategyManager
from strategies.base_strategy import BaseStrategy

# Disable logging during tests
logging.disable(logging.CRITICAL)


class TestEnhancements(unittest.TestCase):
    """Test all enhancements"""
    
    def setUp(self):
        """Set up test environment"""
        # Create mock API
        self.mock_api = MagicMock()
        self.mock_api.get_account_state.return_value = {
            "marginSummary": {
                "accountValue": "10000.0",
                "totalMarginUsed": "1000.0",
                "totalNtlPos": "5000.0"
            },
            "assetPositions": [
                {
                    "coin": "BTC",
                    "position": "0.5"
                },
                {
                    "coin": "ETH",
                    "position": "5.0"
                }
            ]
        }
        self.mock_api.get_open_orders.return_value = [
            {
                "coin": "BTC",
                "side": "buy",
                "size": "0.1",
                "price": "50000.0"
            },
            {
                "coin": "ETH",
                "side": "sell",
                "size": "1.0",
                "price": "3000.0"
            }
        ]
        self.mock_api.get_markets.return_value = [
            {"name": "BTC"},
            {"name": "ETH"},
            {"name": "SOL"}
        ]
        self.mock_api.test_connection.return_value = True
        
        # Create test objects
        self.config_manager = ConfigManager()
        self.security_manager = SecurityManager()
        
        # Patch the API creation in connection manager
        patcher = patch('core.api_fixed_v2.EnhancedHyperliquidAPI')
        self.mock_api_class = patcher.start()
        self.mock_api_class.return_value = self.mock_api
        self.addCleanup(patcher.stop)
        
        # Create connection manager with default credentials
        self.connection_manager = EnhancedConnectionManager()
        self.connection_manager.default_address = "0x306D29F56EA1345c7E6F1ff27657ba05cEE15D4F"
        self.connection_manager.default_private_key = "43ba46de58067dd1ef3794c653bf3b11fa78866623cc515a5aff5f4be31fd3b8"
        self.connection_manager.api = self.mock_api
        
        # Add methods to mock API for connection manager
        self.connection_manager.set_auto_reconnect = MagicMock(return_value=True)
        
        # Create strategy manager
        self.strategy_manager = StrategyManager(self.connection_manager)
    
    def test_auto_connection_with_default_credentials(self):
        """Test auto-connection with default credentials"""
        # Test connection with default credentials
        self.connection_manager._connect_with_default_credentials()
        
        # Verify connection status
        status = self.connection_manager.get_connection_status()
        self.assertTrue(status['connected'])
        self.assertEqual(status['address'], "0x306D29F56EA1345c7E6F1ff27657ba05cEE15D4F")
        self.assertTrue(status['using_default'])
    
    def test_connection_manager_methods(self):
        """Test connection manager methods"""
        # Test ensure_connection
        self.assertTrue(self.connection_manager.ensure_connection())
        
        # Test get_account_state
        account_state = self.connection_manager.get_account_state()
        self.assertIsNotNone(account_state)
        self.assertIn("marginSummary", account_state)
        
        # Test check_connection_health
        self.assertTrue(self.connection_manager.check_connection_health())
        
        # Test update_network
        self.assertTrue(self.connection_manager.update_network(True))
        self.assertTrue(self.connection_manager.testnet)
        
        # Test set_auto_reconnect
        self.assertTrue(self.connection_manager.set_auto_reconnect(True))
        
        # Test disconnect
        self.connection_manager.disconnect()
        self.assertFalse(self.connection_manager.is_connected)
        
        # Test reconnect
        self.assertTrue(self.connection_manager.ensure_connection())
        self.assertTrue(self.connection_manager.is_connected)
    
    def test_wallet_generation(self):
        """Test wallet generation"""
        # Test connect_with_new_wallet
        success, address, private_key = self.connection_manager.connect_with_new_wallet()
        
        # Verify results
        self.assertTrue(success)
        self.assertIsNotNone(address)
        self.assertIsNotNone(private_key)
        self.assertTrue(address.startswith("0x"))
        self.assertEqual(len(private_key), 64)  # 32 bytes = 64 hex chars
    
    def test_strategy_manager_methods(self):
        """Test strategy manager methods"""
        # Test get_available_strategies
        available_strategies = self.strategy_manager.get_available_strategies()
        self.assertIn("BB_RSI_ADX", available_strategies)
        self.assertIn("Hull_Suite", available_strategies)
        
        # Test add_strategy
        self.assertTrue(self.strategy_manager.add_strategy("BB_RSI_ADX"))
        
        # Test get_active_strategies
        active_strategies = self.strategy_manager.get_active_strategies()
        self.assertIn("BB_RSI_ADX", active_strategies)
        
        # Test start
        self.assertTrue(self.strategy_manager.start())
        self.assertTrue(self.strategy_manager.running)
        
        # Test stop
        self.assertTrue(self.strategy_manager.stop())
        self.assertFalse(self.strategy_manager.running)
        
        # Test remove_strategy
        self.assertTrue(self.strategy_manager.remove_strategy("BB_RSI_ADX"))
        active_strategies = self.strategy_manager.get_active_strategies()
        self.assertNotIn("BB_RSI_ADX", active_strategies)
    
    def test_base_strategy_methods(self):
        """Test base strategy methods"""
        # Create base strategy
        strategy = BaseStrategy(api=self.mock_api)
        
        # Add methods to mock API for base strategy
        self.mock_api.place_limit_order = MagicMock(return_value=True)
        self.mock_api.cancel_all_orders = MagicMock(return_value=True)
        self.mock_api.place_market_order = MagicMock(return_value=True)
        
        # Test update_positions
        strategy.update_positions()
        self.assertIn("BTC", strategy.positions)
        self.assertIn("ETH", strategy.positions)
        
        # Test update_orders
        strategy.update_orders()
        self.assertIn("BTC", strategy.orders)
        self.assertIn("ETH", strategy.orders)
        
        # Test get_account_value
        account_value = strategy.get_account_value()
        self.assertEqual(account_value, 10000.0)
        
        # Test place_order
        strategy.place_order("BTC", "buy", 0.1, 50000.0)
        self.mock_api.place_limit_order.assert_called_with("BTC", "buy", 0.1, 50000.0)
        
        # Test cancel_all_orders
        strategy.cancel_all_orders()
        self.mock_api.cancel_all_orders.assert_called()
        
        # Test close_position
        strategy.close_position("BTC")
        self.mock_api.place_market_order.assert_called()
    
    def test_config_manager_methods(self):
        """Test config manager methods"""
        # Test set and get
        self.config_manager.set("test.key", "test_value")
        self.assertEqual(self.config_manager.get("test.key"), "test_value")
        
        # Test save_config and load_config
        self.config_manager.save_config()
        
        # Create new config manager to test loading
        new_config_manager = ConfigManager()
        self.assertEqual(new_config_manager.get("test.key"), "test_value")
    
    def test_security_manager_methods(self):
        """Test security manager methods"""
        # Test store and retrieve private key
        self.security_manager.store_private_key("test_private_key")
        retrieved_key = self.security_manager.get_private_key()
        self.assertEqual(retrieved_key, "test_private_key")
        
        # Test clear private key
        self.security_manager.clear_private_key()
        retrieved_key = self.security_manager.get_private_key()
        self.assertIsNone(retrieved_key)


if __name__ == "__main__":
    unittest.main()

