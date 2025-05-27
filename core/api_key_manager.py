#!/usr/bin/env python3
"""
API key management module for HyperliquidMaster.
Handles secure storage and retrieval of API keys.
"""

import os
import json
import logging
from typing import Tuple, Dict, Any, Optional

class ApiKeyManager:
    """
    Manages API keys for the Hyperliquid exchange.
    Handles secure storage and retrieval of API keys.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the API key manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.logger = logging.getLogger("ApiKeyManager")
    
    def get_api_keys(self) -> Tuple[str, str]:
        """
        Get the API keys from the configuration file.
        
        Returns:
            Tuple containing (account_address, secret_key)
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                account_address = config.get("account_address", "")
                secret_key = config.get("secret_key", "")
                
                return account_address, secret_key
            else:
                return "", ""
        except Exception as e:
            self.logger.error(f"Error getting API keys: {e}")
            return "", ""
    
    def set_api_keys(self, account_address: str, secret_key: str) -> bool:
        """
        Set the API keys in the configuration file.
        
        Args:
            account_address: The account address
            secret_key: The secret key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load existing config if it exists
            config = {}
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
            
            # Update API keys
            config["account_address"] = account_address
            config["secret_key"] = secret_key
            
            # Save config
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            return True
        except Exception as e:
            self.logger.error(f"Error setting API keys: {e}")
            return False
    
    def has_valid_keys(self) -> bool:
        """
        Check if valid API keys are available.
        
        Returns:
            True if valid keys are available, False otherwise
        """
        account_address, secret_key = self.get_api_keys()
        return bool(account_address and secret_key)
    
    def clear_keys(self) -> bool:
        """
        Clear the API keys from the configuration file.
        
        Returns:
            True if successful, False otherwise
        """
        return self.set_api_keys("", "")
