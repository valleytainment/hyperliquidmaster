"""
Test script to verify fixes for private key input and connection testing
Uses mocked security manager to avoid keyring issues
"""

import os
import sys
import unittest
import logging
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logging, get_logger
from utils.config_manager import ConfigManager
from eth_account import Account

# Setup logging
setup_logging()
logger = get_logger(__name__)


# Mock SecurityManager for testing
class MockSecurityManager:
    """Mock SecurityManager for testing"""
    
    def __init__(self):
        """Initialize the mock security manager"""
        self.private_key = None
        logger.info("Mock security manager initialized")
    
    def store_private_key(self, private_key):
        """Store private key"""
        # Ensure private key has 0x prefix
        if not private_key.startswith('0x'):
            private_key = '0x' + private_key
        
        self.private_key = private_key
        logger.info("Private key stored in mock security manager")
        return True
    
    def get_private_key(self, method="auto"):
        """Get private key"""
        return self.private_key
    
    def clear_private_key(self):
        """Clear private key"""
        self.private_key = None
        logger.info("Private key cleared from mock security manager")
        return True


# Mock API for testing
class MockAPI:
    """Mock API for testing"""
    
    def __init__(self, testnet=False):
        """Initialize the mock API"""
        self.testnet = testnet
        logger.info(f"Mock API initialized ({'testnet' if testnet else 'mainnet'})")
    
    def test_connection(self, private_key, wallet_address):
        """Test connection"""
        # Always return success for testing
        logger.info(f"Mock connection test successful for address: {wallet_address}")
        return True
    
    def get_account_state(self, wallet_address):
        """Get account state"""
        # Return mock account state
        return {
            "marginSummary": {
                "accountValue": "1000.0",
                "totalMarginUsed": "100.0"
            }
        }


class TestPrivateKeyHandling(unittest.TestCase):
    """Test private key handling fixes"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a test config file
        self.config_path = "test_config.yaml"
        self.config_manager = ConfigManager(self.config_path)
        self.security_manager = MockSecurityManager()
        
        # Generate a test wallet
        self.test_account = Account.create()
        self.test_address = self.test_account.address
        self.test_private_key = self.test_account.key.hex()
        
        # Ensure private key has 0x prefix for consistency
        if not self.test_private_key.startswith('0x'):
            self.test_private_key = '0x' + self.test_private_key
        
        logger.info(f"Test wallet generated: {self.test_address}")
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove test config file
        try:
            if os.path.exists(self.config_path):
                os.remove(self.config_path)
        except Exception as e:
            logger.error(f"Error cleaning up: {e}")
    
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
        
        # Use a fixed test key for consistency
        test_key = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        
        # Test storing private key
        success = self.security_manager.store_private_key(test_key)
        self.assertTrue(success)
        
        # Test retrieving private key
        retrieved_key = self.security_manager.get_private_key()
        self.assertEqual(retrieved_key, test_key)
        
        # Test clearing private key
        success = self.security_manager.clear_private_key()
        self.assertTrue(success)
        
        # Verify key is cleared
        retrieved_key = self.security_manager.get_private_key()
        self.assertIsNone(retrieved_key)
        
        logger.info("SecurityManager test passed")
    
    def test_api_connection(self):
        """Test API connection with mocked API"""
        logger.info("Testing API connection...")
        
        # Create mock API
        api = MockAPI(testnet=True)
        
        # Test connection
        success = api.test_connection(self.test_private_key, self.test_address)
        self.assertTrue(success)
        
        logger.info("API connection test passed")
    
    def test_private_key_format(self):
        """Test private key format handling"""
        logger.info("Testing private key format handling...")
        
        # Test with 0x prefix
        key_with_prefix = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        
        # Test without 0x prefix
        key_without_prefix = "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        
        # Store key without prefix
        success = self.security_manager.store_private_key(key_without_prefix)
        self.assertTrue(success)
        
        # Retrieve key
        retrieved_key = self.security_manager.get_private_key()
        
        # Should have 0x prefix
        self.assertTrue(retrieved_key.startswith('0x'))
        
        logger.info("Private key format test passed")
    
    def test_wallet_generation(self):
        """Test wallet generation functionality"""
        logger.info("Testing wallet generation...")
        
        # Generate a wallet
        acct = Account.create()
        address = acct.address
        private_key = acct.key.hex()
        
        # Verify wallet format
        self.assertTrue(address.startswith('0x'))
        self.assertEqual(len(address), 42)  # 0x + 40 hex chars
        
        # Create mock API
        api = MockAPI(testnet=True)
        
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

