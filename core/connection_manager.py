"""
Connection Manager for Hyperliquid Trading Bot
Ensures the application is always connected to Hyperliquid
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple, Union

from utils.logger import get_logger
from utils.config_manager_fixed import ConfigManager
from utils.security_fixed_v2 import SecurityManager
from core.api_fixed_v2 import EnhancedHyperliquidAPI

logger = get_logger(__name__)


class ConnectionManager:
    """
    Connection Manager for Hyperliquid Trading Bot
    Ensures the application is always connected to Hyperliquid
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the connection manager
        
        Args:
            config_path: Path to configuration file
        """
        self.config_manager = ConfigManager(config_path)
        self.security_manager = SecurityManager()
        self.api = None
        
        # Default credentials
        self.default_address = "0x306D29F56EA1345c7E6F1ff27657ba05cEE15D4F"
        self.default_private_key = "0x43ba46de58067dd1ef3794c653bf3b11fa78866623cc515a5aff5f4be31fd3b8"
        
        # Connection state
        self.is_connected = False
        self.current_address = None
        self.testnet = False
        
        # Initialize API
        self._initialize_api()
        
        logger.info("Connection Manager initialized")
    
    def _initialize_api(self):
        """Initialize the API"""
        try:
            # Get testnet setting
            self.testnet = self.config_manager.get('trading.testnet', False)
            
            # Initialize API
            self.api = EnhancedHyperliquidAPI(testnet=self.testnet)
            
            logger.info(f"API initialized ({'testnet' if self.testnet else 'mainnet'})")
        except Exception as e:
            logger.error(f"Failed to initialize API: {e}")
    
    def ensure_connection(self) -> bool:
        """
        Ensure the application is connected to Hyperliquid
        
        Returns:
            bool: True if connected
        """
        # If already connected, return True
        if self.is_connected and self.api:
            return True
        
        # Try to connect with saved credentials
        if self._connect_with_saved_credentials():
            return True
        
        # Try to connect with default credentials
        if self._connect_with_default_credentials():
            return True
        
        logger.error("Failed to connect to Hyperliquid")
        return False
    
    def _connect_with_saved_credentials(self) -> bool:
        """
        Connect with saved credentials
        
        Returns:
            bool: True if connected
        """
        try:
            # Get saved credentials
            private_key = self.security_manager.get_private_key()
            wallet_address = self.config_manager.get('trading.wallet_address')
            
            if not private_key or not wallet_address:
                logger.warning("No saved credentials found")
                return False
            
            # Connect with saved credentials
            if self._connect_with_credentials(private_key, wallet_address):
                logger.info(f"Connected with saved credentials: {wallet_address}")
                return True
            else:
                logger.warning("Failed to connect with saved credentials")
                return False
        except Exception as e:
            logger.error(f"Error connecting with saved credentials: {e}")
            return False
    
    def _connect_with_default_credentials(self) -> bool:
        """
        Connect with default credentials
        
        Returns:
            bool: True if connected
        """
        try:
            # Connect with default credentials
            if self._connect_with_credentials(self.default_private_key, self.default_address):
                logger.info(f"Connected with default credentials: {self.default_address}")
                
                # Save default credentials
                self.config_manager.set('trading.wallet_address', self.default_address)
                self.security_manager.store_private_key(self.default_private_key)
                self.config_manager.save_config()
                
                return True
            else:
                logger.error("Failed to connect with default credentials")
                return False
        except Exception as e:
            logger.error(f"Error connecting with default credentials: {e}")
            return False
    
    def _connect_with_credentials(self, private_key: str, wallet_address: str) -> bool:
        """
        Connect with the provided credentials
        
        Args:
            private_key: Private key
            wallet_address: Wallet address
            
        Returns:
            bool: True if connected
        """
        try:
            # Initialize API if needed
            if not self.api:
                self._initialize_api()
            
            # Initialize exchange
            success = self.api.initialize_exchange(private_key, wallet_address)
            
            if success:
                self.is_connected = True
                self.current_address = wallet_address
                return True
            else:
                self.is_connected = False
                self.current_address = None
                return False
        except Exception as e:
            logger.error(f"Error connecting with credentials: {e}")
            self.is_connected = False
            self.current_address = None
            return False
    
    def connect_with_new_wallet(self) -> Tuple[bool, str, str]:
        """
        Generate a new wallet and connect with it
        
        Returns:
            Tuple[bool, str, str]: (success, wallet_address, private_key)
        """
        try:
            from eth_account import Account
            
            # Create a new random account
            acct = Account.create()
            new_address = acct.address
            new_privkey = acct.key.hex()
            
            # Connect with new wallet
            if self._connect_with_credentials(new_privkey, new_address):
                logger.info(f"Connected with new wallet: {new_address}")
                
                # Save new credentials
                self.config_manager.set('trading.wallet_address', new_address)
                self.security_manager.store_private_key(new_privkey)
                self.config_manager.save_config()
                
                return True, new_address, new_privkey
            else:
                logger.error("Failed to connect with new wallet")
                return False, new_address, new_privkey
        except Exception as e:
            logger.error(f"Error generating and connecting with new wallet: {e}")
            return False, "", ""
    
    def disconnect(self):
        """Disconnect from Hyperliquid"""
        self.is_connected = False
        self.current_address = None
        logger.info("Disconnected from Hyperliquid")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get connection status
        
        Returns:
            Dict: Connection status
        """
        return {
            "connected": self.is_connected,
            "address": self.current_address,
            "network": "testnet" if self.testnet else "mainnet"
        }
    
    def get_account_state(self) -> Dict[str, Any]:
        """
        Get account state
        
        Returns:
            Dict: Account state
        """
        if not self.is_connected or not self.api:
            return {}
        
        return self.api.get_account_state(self.current_address)
    
    def update_network(self, testnet: bool) -> bool:
        """
        Update network setting
        
        Args:
            testnet: Whether to use testnet
            
        Returns:
            bool: True if successful
        """
        try:
            # Update testnet setting
            self.testnet = testnet
            self.config_manager.set('trading.testnet', testnet)
            self.config_manager.save_config()
            
            # Reinitialize API
            self._initialize_api()
            
            # Reconnect
            self.is_connected = False
            self.ensure_connection()
            
            logger.info(f"Network updated: {'testnet' if testnet else 'mainnet'}")
            return True
        except Exception as e:
            logger.error(f"Error updating network: {e}")
            return False

