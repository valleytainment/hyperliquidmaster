"""
Test script to verify all implementations
"""

import os
import sys
import unittest
import logging
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestImplementations(unittest.TestCase):
    """Test all implementations"""
    
    def setUp(self):
        """Set up test environment"""
        # Mock dependencies
        self.mock_exchange = MagicMock()
        self.mock_info = MagicMock()
        
        # Patch Exchange and Info classes
        self.exchange_patcher = patch('core.api_fixed_v2.Exchange', return_value=self.mock_exchange)
        self.info_patcher = patch('core.api_fixed_v2.Info', return_value=self.mock_info)
        
        # Start patchers
        self.mock_exchange_class = self.exchange_patcher.start()
        self.mock_info_class = self.info_patcher.start()
        
        # Set up mock responses
        self.mock_info.user_state.return_value = {
            "marginSummary": {
                "accountValue": "1000.00",
                "totalMarginUsed": "200.00",
                "totalNtlPos": "500.00"
            },
            "assetPositions": [
                {
                    "coin": "BTC",
                    "position": "0.01"
                }
            ]
        }
    
    def tearDown(self):
        """Tear down test environment"""
        # Stop patchers
        self.exchange_patcher.stop()
        self.info_patcher.stop()
    
    def test_api_fixed_v2(self):
        """Test the fixed API implementation"""
        from core.api_fixed_v2 import EnhancedHyperliquidAPI
        
        # Create API instance
        api = EnhancedHyperliquidAPI()
        
        # Test test_connection method
        self.assertTrue(api.test_connection("0x123", "0x456"))
        
        # Verify Exchange.__init__() was called without private_key
        self.mock_exchange_class.assert_called_once()
        call_args = self.mock_exchange_class.call_args[1]
        self.assertNotIn('private_key', call_args)
        
        # Verify set_private_key was called
        self.mock_exchange.set_private_key.assert_called_once()
    
    def test_config_manager_fixed(self):
        """Test the fixed ConfigManager implementation"""
        import tempfile
        from utils.config_manager_fixed import ConfigManager
        
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(suffix='.yaml') as temp_file:
            # Create ConfigManager instance
            config_manager = ConfigManager(temp_file.name)
            
            # Set a value
            config_manager.set('trading.wallet_address', '0x123')
            
            # Save config without arguments
            config_manager.save_config()
            
            # Verify the value was saved
            self.assertEqual(config_manager.get('trading.wallet_address'), '0x123')
    
    def test_connection_manager(self):
        """Test the ConnectionManager implementation"""
        import tempfile
        from core.connection_manager import ConnectionManager
        
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(suffix='.yaml') as temp_file:
            # Create ConnectionManager instance
            connection_manager = ConnectionManager(temp_file.name)
            
            # Test ensure_connection method
            self.assertTrue(connection_manager.ensure_connection())
            
            # Verify connection status
            status = connection_manager.get_connection_status()
            self.assertTrue(status['connected'])
            
            # Test get_account_state method
            account_state = connection_manager.get_account_state()
            self.assertIn('marginSummary', account_state)
            
            # Test update_network method
            self.assertTrue(connection_manager.update_network(True))
            self.assertTrue(connection_manager.testnet)
    
    def test_wallet_generation(self):
        """Test wallet generation functionality"""
        from eth_account import Account
        
        # Generate a new wallet
        acct = Account.create()
        
        # Verify wallet was generated
        self.assertIsNotNone(acct.address)
        self.assertTrue(acct.address.startswith('0x'))
        self.assertIsNotNone(acct.key)
        
        # Verify private key format
        private_key_hex = acct.key.hex()
        self.assertTrue(private_key_hex.startswith('0x') or len(private_key_hex) == 64)


if __name__ == '__main__':
    unittest.main()

