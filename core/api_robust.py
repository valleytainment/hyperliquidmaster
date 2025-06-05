"""
Robust API for Hyperliquid Trading Bot
Focused on fixing connection testing issues with comprehensive error handling
"""

import logging
import time
import json
from typing import Dict, Any, Optional, Union, List

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


class RobustHyperliquidAPI:
    """Robust Hyperliquid API with comprehensive error handling"""
    
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
        
        logger.info(f"Robust Hyperliquid API initialized ({'testnet' if testnet else 'mainnet'})")
    
    def _rate_limit(self):
        """Implement rate limiting to avoid API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def test_connection(self, private_key: str = None, wallet_address: str = None) -> bool:
        """
        Test connection to Hyperliquid API with comprehensive error handling
        
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
            
            # Validate private key format
            if not private_key.startswith('0x'):
                logger.warning("Private key doesn't start with 0x, adding prefix")
                private_key = '0x' + private_key
            
            # Validate wallet address format
            if not wallet_address.startswith('0x'):
                logger.warning("Wallet address doesn't start with 0x, adding prefix")
                wallet_address = '0x' + wallet_address
            
            # Apply rate limiting
            self._rate_limit()
            
            # Initialize temporary exchange for testing
            # Important: Do NOT pass private_key to the constructor
            try:
                temp_exchange = Exchange(
                    base_url=self.api_url,
                    skip_ws=True  # Skip WebSocket for testing
                )
            except Exception as e:
                logger.error(f"Failed to initialize Exchange: {e}")
                return False
            
            # Set the private key separately
            try:
                temp_exchange.set_private_key(private_key)
            except Exception as e:
                logger.error(f"Failed to set private key: {e}")
                return False
            
            # Initialize info client
            try:
                info_client = Info(self.api_url, skip_ws=True)
            except Exception as e:
                logger.error(f"Failed to initialize Info client: {e}")
                return False
            
            # Try to get user state
            try:
                user_state = info_client.user_state(wallet_address)
            except Exception as e:
                logger.error(f"Failed to get user state: {e}")
                return False
            
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
        Authenticate with Hyperliquid API with comprehensive error handling
        
        Args:
            private_key: Private key for authentication
            wallet_address: Wallet address
            
        Returns:
            bool: True if authentication successful
        """
        try:
            # Validate private key format
            if not private_key.startswith('0x'):
                logger.warning("Private key doesn't start with 0x, adding prefix")
                private_key = '0x' + private_key
            
            # Validate wallet address format
            if not wallet_address.startswith('0x'):
                logger.warning("Wallet address doesn't start with 0x, adding prefix")
                wallet_address = '0x' + wallet_address
            
            # Apply rate limiting
            self._rate_limit()
            
            # Initialize exchange client
            # Important: Do NOT pass private_key to the constructor
            try:
                self.exchange = Exchange(
                    base_url=self.api_url,
                    skip_ws=True  # Skip WebSocket for now
                )
            except Exception as e:
                logger.error(f"Failed to initialize Exchange: {e}")
                return False
            
            # Set the private key separately
            try:
                self.exchange.set_private_key(private_key)
            except Exception as e:
                logger.error(f"Failed to set private key: {e}")
                return False
            
            # Initialize info client
            try:
                self.info = Info(self.api_url, skip_ws=True)
            except Exception as e:
                logger.error(f"Failed to initialize Info client: {e}")
                return False
            
            # Try to get user state to verify authentication
            try:
                user_state = self.info.user_state(wallet_address)
            except Exception as e:
                logger.error(f"Failed to get user state: {e}")
                return False
            
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
        Get account state with comprehensive error handling
        
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
            
            # Validate wallet address format
            if not wallet_address.startswith('0x'):
                logger.warning("Wallet address doesn't start with 0x, adding prefix")
                wallet_address = '0x' + wallet_address
            
            # Apply rate limiting
            self._rate_limit()
            
            if not self.info:
                try:
                    self.info = Info(self.api_url, skip_ws=True)
                except Exception as e:
                    logger.error(f"Failed to initialize Info client: {e}")
                    return {}
            
            # Get user state
            try:
                user_state = self.info.user_state(wallet_address)
                return user_state or {}
            except Exception as e:
                logger.error(f"Failed to get user state: {e}")
                return {}
            
        except Exception as e:
            logger.error(f"Failed to get account state: {e}")
            return {}
    
    def save_private_key(self, private_key: str) -> bool:
        """
        Save private key securely with comprehensive error handling
        
        Args:
            private_key: Private key to save
            
        Returns:
            bool: True if successful
        """
        try:
            # Validate private key format
            if not private_key.startswith('0x'):
                logger.warning("Private key doesn't start with 0x, adding prefix")
                private_key = '0x' + private_key
            
            return self.security_manager.store_private_key(private_key)
        except Exception as e:
            logger.error(f"Failed to save private key: {e}")
            return False
    
    def get_market_data(self, coin: str) -> Dict[str, Any]:
        """
        Get market data for a specific coin with comprehensive error handling
        
        Args:
            coin: Coin symbol (e.g., BTC)
            
        Returns:
            Dict: Market data
        """
        try:
            # Apply rate limiting
            self._rate_limit()
            
            if not self.info:
                try:
                    self.info = Info(self.api_url, skip_ws=True)
                except Exception as e:
                    logger.error(f"Failed to initialize Info client: {e}")
                    return {}
            
            # Get market data
            try:
                meta = self.info.meta()
                if not meta or not isinstance(meta, dict) or "universe" not in meta:
                    logger.error("Failed to get market metadata")
                    return {}
                
                # Find the coin in the universe
                coin_upper = coin.upper()
                for asset in meta["universe"]:
                    if asset["name"].upper() == coin_upper:
                        # Get market data
                        market_data = {}
                        
                        # Get ticker
                        try:
                            ticker = self.info.ticker()
                            for item in ticker:
                                if item["coin"].upper() == coin_upper:
                                    market_data["price"] = float(item["markPrice"])
                                    market_data["funding_rate"] = float(item["fundingRate"])
                                    break
                        except Exception as e:
                            logger.error(f"Failed to get ticker: {e}")
                        
                        # Get stats
                        try:
                            stats = self.info.stats()
                            for item in stats:
                                if item["coin"].upper() == coin_upper:
                                    market_data["volume_24h"] = float(item["volume24h"])
                                    break
                        except Exception as e:
                            logger.error(f"Failed to get stats: {e}")
                        
                        return market_data
                
                logger.warning(f"Coin {coin} not found in universe")
                return {}
                
            except Exception as e:
                logger.error(f"Failed to get market data: {e}")
                return {}
            
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return {}

