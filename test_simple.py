"""
Simple test script to verify the most important functionality
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
from core.api_fixed_v2 import EnhancedHyperliquidAPI

# Disable logging during tests
logging.disable(logging.CRITICAL)


class TestSimple(unittest.TestCase):
    """Simple test case for the most important functionality"""
    
    def test_config_manager(self):
        """Test ConfigManager functionality"""
        # Create config manager
        config_manager = ConfigManager()
        
        # Test set and get
        config_manager.set("test.key", "test_value")
        self.assertEqual(config_manager.get("test.key"), "test_value")
        
        # Test save_config
        config_manager.save_config()
        
        # Create new config manager to test loading
        new_config_manager = ConfigManager()
        self.assertEqual(new_config_manager.get("test.key"), "test_value")
    
    def test_security_manager(self):
        """Test SecurityManager functionality"""
        # Create security manager
        security_manager = SecurityManager()
        
        # Test store_private_key and get_private_key
        security_manager.store_private_key("test_private_key")
        retrieved_key = security_manager.get_private_key()
        self.assertEqual(retrieved_key, "test_private_key")
    
    def test_api(self):
        """Test EnhancedHyperliquidAPI functionality"""
        # Mock the Exchange and Info classes
        with patch('hyperliquid.exchange.Exchange') as mock_exchange, \
             patch('hyperliquid.info.Info') as mock_info:
            
            # Create API
            api = EnhancedHyperliquidAPI()
            
            # Test default credentials
            self.assertEqual(api.default_address, "0x306D29F56EA1345c7E6F1ff27657ba05cEE15D4F")
            self.assertEqual(api.default_private_key, "0x43ba46de58067dd1ef3794c653bf3b11fa78866623cc515a5aff5f4be31fd3b8")
            
            # Test save_credentials
            mock_exchange.return_value.set_private_key = MagicMock()
            mock_info.return_value.user_state = MagicMock(return_value={"account": "test"})
            
            result = api.save_credentials("test_private_key", "test_address")
            self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()

