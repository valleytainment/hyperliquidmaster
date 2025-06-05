"""
Simplified API for Hyperliquid Trading Bot
Focused on fixing connection testing issues
"""

import logging
import time
from typing import Dict, Any, Optional

# Import Hyperliquid SDK
try:
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange
    from hyperliquid.utils import constants
except ImportError as e:
    print(f"Error importing Hyperliquid SDK: {e}")
    print("Please install the latest version: pip install hyperliquid-python-sdk")
    raise

from utils.logger import get_logger
from utils.config_manager import ConfigManager
from utils.security_fixed_v2 import SecurityManager

logger = get_logger(__name__)


class SimpleHyperliquidAPI:
    """Simplified Hyperliquid API focused on fixing connection testing issues"""
    
    def __init__(self, config_path: str = None, testnet: bool = False):
        """
        Initialize the API
        
        Args:
            config_path: Path to configuration file
            testnet: Whether to use testnet
        """
        self.config_manager = ConfigManager(config_path)
        self.security_manager = SecurityManager()
        self.testnet = testnet
        
        # API URLs
        self.api_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        
        # Exchange client
        self.exchange = None
        self.info = None
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.2  # 200ms
        
        logger.info(f"Simple Hyperliquid API initialized ({'testnet' if testnet else 'mainnet'})")
    
    def test_connection(self, private_key: str = None, wallet_address: str = None) -> bool:
        """
        Test connection to Hyperliquid API
        
        Args:
            private_key: Private key for authentication
            wallet_address: Wallet address
            
        Returns:
            bool: True if connection successful
        """
        try:
            # Get credentials if not provided
            if not private_key:
                private_key = self.security_manager.get_private_key()
                if not private_key:
                    logger.error("No private key provided or found")
                    return False
            
            if not wallet_address:
                wallet_address = self.config_manager.get('trading.wallet_address')
                if not wallet_address:
                    logger.error("No wallet address provided or found")
                    return False
            
            # Initialize temporary exchange for testing
            # Important: Do NOT pass private_key to the constructor
            temp_exchange = Exchange(
                base_url=self.api_url,
                skip_ws=True  # Skip WebSocket for testing
            )
            
            # Set the private key separately
            temp_exchange.set_private_key(private_key)
            
            # Initialize info client
            info_client = Info(self.api_url, skip_ws=True)
            
            # Try to get user state
            user_state = info_client.user_state(wallet_address)
            
            # Check if we got a valid response
            if user_state and isinstance(user_state, dict):
                logger.info(f"Connection test successful for address: {wallet_address}")
                return True
            else:
                logger.warning(f"Connection test failed: Invalid response for address: {wallet_address}")
                return False
                
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def authenticate(self, private_key: str, wallet_address: str) -> bool:
        """
        Authenticate with Hyperliquid API
        
        Args:
            private_key: Private key for authentication
            wallet_address: Wallet address
            
        Returns:
            bool: True if authentication successful
        """
        try:
            # Initialize exchange client
            # Important: Do NOT pass private_key to the constructor
            self.exchange = Exchange(
                base_url=self.api_url,
                skip_ws=True  # Skip WebSocket for now
            )
            
            # Set the private key separately
            self.exchange.set_private_key(private_key)
            
            # Initialize info client
            self.info = Info(self.api_url, skip_ws=True)
            
            # Try to get user state to verify authentication
            user_state = self.info.user_state(wallet_address)
            
            # Check if we got a valid response
            if user_state and isinstance(user_state, dict):
                logger.info(f"Authentication successful for address: {wallet_address}")
                return True
            else:
                logger.warning(f"Authentication failed: Invalid response for address: {wallet_address}")
                return False
                
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def get_account_state(self, wallet_address: str = None) -> Dict[str, Any]:
        """
        Get account state
        
        Args:
            wallet_address: Wallet address
            
        Returns:
            Dict: Account state
        """
        try:
            if not wallet_address:
                wallet_address = self.config_manager.get('trading.wallet_address')
                if not wallet_address:
                    logger.error("No wallet address provided or found")
                    return {}
            
            if not self.info:
                self.info = Info(self.api_url, skip_ws=True)
            
            # Get user state
            user_state = self.info.user_state(wallet_address)
            
            return user_state or {}
            
        except Exception as e:
            logger.error(f"Failed to get account state: {e}")
            return {}
    
    def save_private_key(self, private_key: str) -> bool:
        """
        Save private key securely
        
        Args:
            private_key: Private key to save
            
        Returns:
            bool: True if successful
        """
        try:
            return self.security_manager.store_private_key(private_key)
        except Exception as e:
            logger.error(f"Failed to save private key: {e}")
            return False

