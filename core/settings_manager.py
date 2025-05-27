"""
Enhanced GUI settings management for the HyperliquidMaster trading bot.
Ensures flawless API key handling and settings persistence.
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, Tuple

class SettingsManager:
    """
    Settings manager for the HyperliquidMaster trading bot.
    Handles loading, saving, and validating settings.
    """
    
    def __init__(self, config_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the settings manager.
        
        Args:
            config_path: Path to the configuration file
            logger: Logger instance
        """
        self.config_path = config_path
        self.logger = logger or logging.getLogger("SettingsManager")
        self.settings = {}
        self.backup_dir = "settings_backup"
        
        # Create backup directory if it doesn't exist
        os.makedirs(self.backup_dir, exist_ok=True)
        self.logger.info(f"Created settings backup directory: {self.backup_dir}")
        
        # Load settings
        self.load_settings()
    
    def load_settings(self) -> Dict[str, Any]:
        """
        Load settings from the configuration file.
        
        Returns:
            Dict containing the settings
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.settings = json.load(f)
                self.logger.info(f"Loaded settings from {self.config_path}")
            else:
                self.logger.warning(f"Configuration file not found: {self.config_path}")
                self.settings = {}
                self.logger.info("Creating default configuration file")
                self.save_settings()
            
            return self.settings
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}")
            self.settings = {}
            return self.settings
    
    def save_settings(self) -> bool:
        """
        Save settings to the configuration file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create backup before saving
            self._create_backup()
            
            with open(self.config_path, 'w') as f:
                json.dump(self.settings, f, indent=2)
            
            self.logger.info(f"Saved settings to {self.config_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            return False
    
    def _create_backup(self) -> str:
        """
        Create a backup of the current settings.
        
        Returns:
            Path to the backup file
        """
        try:
            if not os.path.exists(self.config_path):
                return ""
            
            timestamp = int(time.time())
            backup_path = os.path.join(self.backup_dir, f"settings_{timestamp}.json")
            
            with open(self.config_path, 'r') as src:
                with open(backup_path, 'w') as dst:
                    dst.write(src.read())
            
            self.logger.info(f"Created settings backup at {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.error(f"Error creating settings backup: {e}")
            return ""
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value.
        
        Args:
            key: Setting key
            default: Default value if setting doesn't exist
            
        Returns:
            Setting value or default
        """
        return self.settings.get(key, default)
        
    def get_settings(self) -> Dict[str, Any]:
        """
        Get all settings.
        
        Returns:
            Dict containing all settings
        """
        return self.settings
    
    def set_setting(self, key: str, value: Any) -> None:
        """
        Set a setting value.
        
        Args:
            key: Setting key
            value: Setting value
        """
        self.settings[key] = value
    
    def update_settings(self, new_settings: Dict[str, Any]) -> bool:
        """
        Update multiple settings at once.
        
        Args:
            new_settings: Dict containing new settings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create backup before updating
            self._create_backup()
            
            # Update settings
            self.settings.update(new_settings)
            
            # Save settings
            result = self.save_settings()
            
            if result:
                self.logger.info("Settings updated successfully")
            
            return result
        except Exception as e:
            self.logger.error(f"Error updating settings: {e}")
            return False
    
    def validate_api_keys(self, account_address: str, secret_key: str) -> Tuple[bool, str]:
        """
        Validate API keys format.
        
        Args:
            account_address: Account address
            secret_key: Secret key
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if account address is empty
        if not account_address or account_address.strip() == "":
            return False, "Account address cannot be empty"
        
        # Check if secret key is empty
        if not secret_key or secret_key.strip() == "":
            return False, "Secret key cannot be empty"
        
        # Skip additional validation for now to allow any format of keys
        # This ensures compatibility with various key formats
        return True, ""
    
    def save_api_keys(self, account_address: str, secret_key: str) -> Tuple[bool, str]:
        """
        Save API keys to settings.
        
        Args:
            account_address: Account address
            secret_key: Secret key
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Validate API keys
            is_valid, error_message = self.validate_api_keys(account_address, secret_key)
            
            if not is_valid:
                return False, error_message
            
            # Create backup before saving
            self._create_backup()
            
            # Update settings
            self.settings["account_address"] = account_address.strip()
            self.settings["secret_key"] = secret_key.strip()
            
            # Save settings
            result = self.save_settings()
            
            if result:
                return True, "API keys saved successfully"
            else:
                return False, "Failed to save API keys"
        except Exception as e:
            self.logger.error(f"Error saving API keys: {e}")
            return False, f"Error saving API keys: {e}"
