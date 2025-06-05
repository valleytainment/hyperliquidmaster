"""
Test script to verify fixes for private key input and connection testing
Uses a file-based keyring backend that doesn't require a password
"""

import os
import sys
import unittest
import logging
from unittest.mock import patch, MagicMock

# Set keyring backend to plaintext file
import keyring
from keyrings.alt.file import PlaintextKeyring
keyring.set_keyring(PlaintextKeyring())

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logging, get_logger
from utils.config_manager import ConfigManager
from utils.security_fixed_v2 import SecurityManager
from core.api_robust import RobustHyperliquidAPI
from eth_account import Account

# Setup logging
setup_logging()
logger = get_logger(__name__)


class TestPrivateKeyHandling(unittest.TestCase):
    """Test private key handling fixes"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a test config file
        self.config_path = "test_config.yaml"
        self.config_manager = ConfigManager(self.config_path)
        self.security_manager = SecurityManager()
        
        # Generate a test wallet
        self.test_account = Account.create()
        self.test_address = self.test_account.address
        self.test_private_key = self.test_account.key.hex()
        
        logger.info(f"Test wallet generated: {self.test_address}")
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove test config file
        try:
            if os.path.exists(self.config_path):
                os.remove(self.config_path)
        except Exception as e:
            logger.error(f"Error cleaning up: {e}")
        
        # Clear test private key
        try:
            self.security_manager.clear_private_key()
        except Exception as e:
            logger.error(f"Error clearing private key: {e}")
    
    def test_config_manager(self):
        """Test ConfigManager fixes"""
        logger.info("Testing ConfigManager...")
        
        # Test saving wallet address
        self.config_manager.set('trading.wallet_address', self.test_address)
        self.config_manager.save_config()
        
        # Test loading wallet address
        loaded_address = self.config_manager.get('trading.wallet_address')
        self.assertEqual(loaded_address, self.test_address)
        
        logger.info("ConfigManager test passed")
    
    def test_security_manager(self):
        """Test SecurityManager fixes"""
        logger.info("Testing SecurityManager...")
        
        # Test storing private key
        success = self.security_manager.store_private_key(self.test_private_key)
        self.assertTrue(success)
        
        # Test retrieving private key
        retrieved_key = self.security_manager.get_private_key()
        self.assertEqual(retrieved_key, self.test_private_key)
        
        # Test clearing private key
        success = self.security_manager.clear_private_key()
        self.assertTrue(success)
        
        # Verify key is cleared
        retrieved_key = self.security_manager.get_private_key()
        self.assertIsNone(retrieved_key)
        
        logger.info("SecurityManager test passed")
    
    @patch('core.api_robust.Exchange')
    @patch('core.api_robust.Info')
    def test_api_connection(self, mock_info, mock_exchange):
        """Test API connection with mocked SDK"""
        logger.info("Testing API connection...")
        
        # Mock Info.user_state to return a valid response
        mock_info_instance = MagicMock()
        mock_info_instance.user_state.return_value = {
            "marginSummary": {
                "accountValue": "1000.0",
                "totalMarginUsed": "100.0"
            }
        }
        mock_info.return_value = mock_info_instance
        
        # Mock Exchange
        mock_exchange_instance = MagicMock()
        mock_exchange_instance.set_private_key.return_value = None
        mock_exchange.return_value = mock_exchange_instance
        
        # Create API client
        api = RobustHyperliquidAPI(testnet=True)
        
        # Test connection
        success = api.test_connection(self.test_private_key, self.test_address)
        self.assertTrue(success)
        
        # Verify Exchange.set_private_key was called
        mock_exchange_instance.set_private_key.assert_called_once_with(self.test_private_key)
        
        # Verify Info.user_state was called
        mock_info_instance.user_state.assert_called_once_with(self.test_address)
        
        logger.info("API connection test passed")
    
    def test_private_key_format(self):
        """Test private key format handling"""
        logger.info("Testing private key format handling...")
        
        # Test with 0x prefix
        key_with_prefix = self.test_private_key
        
        # Test without 0x prefix
        key_without_prefix = self.test_private_key[2:] if self.test_private_key.startswith('0x') else self.test_private_key
        
        # Store key without prefix
        success = self.security_manager.store_private_key(key_without_prefix)
        self.assertTrue(success)
        
        # Retrieve key
        retrieved_key = self.security_manager.get_private_key()
        
        # Should have 0x prefix
        self.assertTrue(retrieved_key.startswith('0x'))
        
        logger.info("Private key format test passed")
    
    @patch('core.api_robust.Exchange')
    @patch('core.api_robust.Info')
    def test_wallet_generation(self, mock_info, mock_exchange):
        """Test wallet generation functionality"""
        logger.info("Testing wallet generation...")
        
        # Generate a wallet
        acct = Account.create()
        address = acct.address
        private_key = acct.key.hex()
        
        # Verify wallet format
        self.assertTrue(address.startswith('0x'))
        self.assertEqual(len(address), 42)  # 0x + 40 hex chars
        
        # Mock Info.user_state to return a valid response
        mock_info_instance = MagicMock()
        mock_info_instance.user_state.return_value = {
            "marginSummary": {
                "accountValue": "1000.0",
                "totalMarginUsed": "100.0"
            }
        }
        mock_info.return_value = mock_info_instance
        
        # Mock Exchange
        mock_exchange_instance = MagicMock()
        mock_exchange_instance.set_private_key.return_value = None
        mock_exchange.return_value = mock_exchange_instance
        
        # Create API client
        api = RobustHyperliquidAPI(testnet=True)
        
        # Test connection with generated wallet
        success = api.test_connection(private_key, address)
        self.assertTrue(success)
        
        logger.info("Wallet generation test passed")


def main():
    """Run the tests"""
    logger.info("Starting tests...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    logger.info("All tests completed")


if __name__ == "__main__":
    main()

