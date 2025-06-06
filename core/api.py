"""
Enhanced Hyperliquid API Wrapper
Fixed version that properly handles Exchange.__init__() error
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
from utils.security import SecurityManager

logger = get_logger(__name__)


class EnhancedHyperliquidAPI:
    """Enhanced Hyperliquid API with proper error handling"""
    
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
        
        # Default credentials
        self.default_address = "0x306D29F56EA1345c7E6F1ff27657ba05cEE15D4F"
        self.default_private_key = "0x43ba46de58067dd1ef3794c653bf3b11fa78866623cc515a5aff5f4be31fd3b8"
        
        logger.info(f"Enhanced Hyperliquid API initialized ({'testnet' if testnet else 'mainnet'})")
    
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
            
            # Validate private key format
            if not private_key.startswith('0x'):
                logger.warning("Private key doesn't start with 0x, adding prefix")
                private_key = '0x' + private_key
            
            # Validate wallet address format
            if not wallet_address.startswith('0x'):
                logger.warning("Wallet address doesn't start with 0x, adding prefix")
                wallet_address = '0x' + wallet_address
            
            # Initialize temporary exchange for testing
            # IMPORTANT: Do NOT pass private_key to the constructor
            try:
                temp_exchange = Exchange(
                    base_url=self.api_url
                )
                
                # Set the private key separately
                temp_exchange.set_private_key(private_key)
            except Exception as e:
                logger.error(f"Failed to initialize Exchange: {e}")
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
                if user_state:
                    logger.info(f"Connection test successful for address: {wallet_address}")
                    return True
                else:
                    logger.warning(f"Connection test failed: No user state for address: {wallet_address}")
                    return False
            except Exception as e:
                logger.error(f"Connection test failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def initialize_exchange(self, private_key: str = None, wallet_address: str = None) -> bool:
        """
        Initialize exchange client
        
        Args:
            private_key: Private key for authentication
            wallet_address: Wallet address
            
        Returns:
            bool: True if initialization successful
        """
        try:
            # Get credentials if not provided
            if not private_key:
                private_key = self.security_manager.get_private_key()
                if not private_key:
                    # Use default private key if none provided
                    logger.warning("No private key provided or found, using default")
                    private_key = self.default_private_key
            
            if not wallet_address:
                wallet_address = self.config_manager.get('trading.wallet_address')
                if not wallet_address:
                    # Use default wallet address if none provided
                    logger.warning("No wallet address provided or found, using default")
                    wallet_address = self.default_address
            
            # Validate private key format
            if not private_key.startswith('0x'):
                logger.warning("Private key doesn't start with 0x, adding prefix")
                private_key = '0x' + private_key
            
            # Validate wallet address format
            if not wallet_address.startswith('0x'):
                logger.warning("Wallet address doesn't start with 0x, adding prefix")
                wallet_address = '0x' + wallet_address
            
            # Initialize exchange client
            # IMPORTANT: Do NOT pass private_key to the constructor
            try:
                self.exchange = Exchange(
                    base_url=self.api_url
                )
                
                # Set the private key separately
                self.exchange.set_private_key(private_key)
            except Exception as e:
                logger.error(f"Failed to initialize Exchange: {e}")
                return False
            
            # Initialize info client
            try:
                self.info = Info(self.api_url, skip_ws=False)
            except Exception as e:
                logger.error(f"Failed to initialize Info client: {e}")
                return False
            
            # Store credentials
            self.config_manager.set('trading.wallet_address', wallet_address)
            self.security_manager.store_private_key(private_key)
            
            logger.info(f"Exchange initialized for address: {wallet_address}")
            return True
                
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
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
                    wallet_address = self.default_address
            
            # Validate wallet address format
            if not wallet_address.startswith('0x'):
                logger.warning("Wallet address doesn't start with 0x, adding prefix")
                wallet_address = '0x' + wallet_address
            
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
    
    def save_credentials(self, private_key: str, wallet_address: str) -> bool:
        """
        Save credentials
        
        Args:
            private_key: Private key
            wallet_address: Wallet address
            
        Returns:
            bool: True if successful
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
            
            # Save wallet address to config
            self.config_manager.set('trading.wallet_address', wallet_address)
            self.config_manager.save_config()
            
            # Save private key securely
            success = self.security_manager.store_private_key(private_key)
            
            if success:
                logger.info("Credentials saved successfully")
                return True
            else:
                logger.error("Failed to save private key")
                return False
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
            return False
    
    def get_market_data(self, coin: str) -> Dict[str, Any]:
        """
        Get market data for a specific coin
        
        Args:
            coin: Coin symbol (e.g., BTC)
            
        Returns:
            Dict: Market data
        """
        try:
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

