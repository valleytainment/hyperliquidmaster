"""
Test script for all fixes and new features
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import tempfile
import json
import keyring
from keyrings.alt.file import PlaintextKeyring

# Set keyring backend to file-based for testing
keyring.set_keyring(PlaintextKeyring())

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from utils.config_manager import ConfigManager
from utils.security_fixed_v2 import SecurityManager
from core.api_fixed import EnhancedHyperliquidAPI
from eth_account import Account


class TestConfigManager(unittest.TestCase):
    """Test the ConfigManager class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary config file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "config.yaml")
        
        # Create a basic config
        with open(self.config_path, "w") as f:
            f.write("""
trading:
  wallet_address: "0x1234567890abcdef1234567890abcdef12345678"
  testnet: false
  active_strategies:
    - bb_rsi_adx
    - hull_suite
api_url: "https://api.hyperliquid.xyz"
""")
        
        # Initialize ConfigManager
        self.config_manager = ConfigManager(self.config_path)
    
    def tearDown(self):
        """Clean up test environment"""
        self.temp_dir.cleanup()
    
    def test_get_config(self):
        """Test getting config"""
        config = self.config_manager.get_config()
        self.assertIsNotNone(config)
        self.assertIn("trading", config)
        self.assertIn("wallet_address", config["trading"])
        self.assertEqual(config["trading"]["wallet_address"], "0x1234567890abcdef1234567890abcdef12345678")
    
    def test_set_config(self):
        """Test setting config"""
        self.config_manager.set("trading.wallet_address", "0xabcdef1234567890abcdef1234567890abcdef12")
        config = self.config_manager.get_config()
        self.assertEqual(config["trading"]["wallet_address"], "0xabcdef1234567890abcdef1234567890abcdef12")
    
    def test_save_config(self):
        """Test saving config"""
        self.config_manager.set("trading.wallet_address", "0xabcdef1234567890abcdef1234567890abcdef12")
        self.config_manager.save_config()
        
        # Reload config
        new_config_manager = ConfigManager(self.config_path)
        config = new_config_manager.get_config()
        self.assertEqual(config["trading"]["wallet_address"], "0xabcdef1234567890abcdef1234567890abcdef12")


class TestSecurityManager(unittest.TestCase):
    """Test the SecurityManager class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for security files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Initialize SecurityManager with custom path
        self.security_manager = SecurityManager(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up test environment"""
        self.temp_dir.cleanup()
    
    def test_store_private_key(self):
        """Test storing private key"""
        private_key = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        success = self.security_manager.store_private_key(private_key)
        self.assertTrue(success)
    
    def test_get_private_key(self):
        """Test getting private key"""
        private_key = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        self.security_manager.store_private_key(private_key)
        retrieved_key = self.security_manager.get_private_key()
        self.assertEqual(retrieved_key, private_key)
    
    def test_clear_private_key(self):
        """Test clearing private key"""
        private_key = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        self.security_manager.store_private_key(private_key)
        success = self.security_manager.clear_private_key()
        self.assertTrue(success)


class TestEnhancedHyperliquidAPI(unittest.TestCase):
    """Test the EnhancedHyperliquidAPI class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary config file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "config.yaml")
        
        # Create a basic config
        with open(self.config_path, "w") as f:
            f.write("""
trading:
  wallet_address: "0x1234567890abcdef1234567890abcdef12345678"
  testnet: false
api_url: "https://api.hyperliquid.xyz"
""")
        
        # Initialize API with mock Exchange
        with patch("core.api_fixed.Exchange") as mock_exchange:
            self.mock_exchange_instance = MagicMock()
            mock_exchange.return_value = self.mock_exchange_instance
            self.api = EnhancedHyperliquidAPI(self.config_path)
    
    def tearDown(self):
        """Clean up test environment"""
        self.temp_dir.cleanup()
    
    def test_authenticate(self):
        """Test authentication"""
        private_key = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        wallet_address = "0x1234567890abcdef1234567890abcdef12345678"
        
        # Mock successful authentication
        self.mock_exchange_instance.set_private_key.return_value = True
        
        success = self.api.authenticate(private_key, wallet_address)
        self.assertTrue(success)
        
        # Verify set_private_key was called
        self.mock_exchange_instance.set_private_key.assert_called_once_with(private_key)


class TestWalletGeneration(unittest.TestCase):
    """Test wallet generation functionality"""
    
    def test_generate_wallet(self):
        """Test generating a new wallet"""
        # Generate a new wallet
        acct = Account.create()
        address = acct.address
        private_key = acct.key.hex()
        
        # Verify address format
        self.assertTrue(address.startswith("0x"))
        self.assertEqual(len(address), 42)
        
        # Verify private key format
        self.assertTrue(private_key.startswith("0x"))
        self.assertEqual(len(private_key), 66)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestConfigManager))
    suite.addTests(loader.loadTestsFromTestCase(TestSecurityManager))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedHyperliquidAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestWalletGeneration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

