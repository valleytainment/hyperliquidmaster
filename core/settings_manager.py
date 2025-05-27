"""
Settings Manager for HyperLiquid Trading Bot

This module provides a robust settings management system with
validation, backup, and secure storage of sensitive information.
"""

import os
import json
import time
import logging
import shutil
from typing import Dict, Any, Optional, List

class SettingsManager:
    """
    Settings manager for HyperLiquid Trading Bot.
    Handles loading, saving, validating, and backing up configuration.
    """
    
    def __init__(self, config_path: str, logger=None):
        """
        Initialize the settings manager.
        
        Args:
            config_path: Path to the configuration file
            logger: Optional logger instance
        """
        # Setup logging
        self.logger = logger or logging.getLogger("SettingsManager")
        
        # Store configuration path
        self.config_path = config_path
        
        # Default configuration
        self.default_config = {
            "account_address": "",
            "secret_key": "",
            "api_url": "https://api.hyperliquid.xyz",
            "symbols": ["BTC", "ETH", "SOL"],
            "poll_interval_seconds": 5,
            "risk_percent": 0.01,
            "max_open_positions": 3,
            "trading_mode": "monitor_only",  # monitor_only, paper_trading, live_trading
            "strategy": "master_omni_overlord",
            "strategy_params": {
                "signal_threshold": 0.7,
                "trend_filter_strength": 0.5,
                "mean_reversion_factor": 0.3,
                "volatility_adjustment": 1.0
            },
            "logging": {
                "level": "INFO",
                "file_path": "logs/trading_bot.log",
                "max_file_size_mb": 10,
                "backup_count": 5
            },
            "notifications": {
                "enabled": False,
                "email": "",
                "webhook_url": "",
                "notify_on_trade": True,
                "notify_on_error": True
            }
        }
        
        # Create backup directory if it doesn't exist
        backup_dir = os.path.join(os.path.dirname(config_path), "settings_backup")
        if not os.path.exists(backup_dir):
            try:
                os.makedirs(backup_dir)
                self.logger.info(f"Created settings backup directory: {backup_dir}")
            except Exception as e:
                self.logger.error(f"Failed to create settings backup directory: {e}")
        
        # Load configuration
        self.config = self.load_settings()
        
        # Create backup of initial configuration
        self.create_backup()
        
        self.logger.info("Settings manager initialized")
    
    def load_settings(self) -> Dict[str, Any]:
        """
        Load settings from configuration file.
        
        Returns:
            Configuration dictionary
        """
        try:
            # Check if configuration file exists
            if not os.path.exists(self.config_path):
                self.logger.warning(f"Configuration file not found: {self.config_path}")
                self.logger.info("Creating default configuration file")
                
                # Create default configuration
                self.save_settings(self.default_config)
                return self.default_config.copy()
            
            # Load configuration
            with open(self.config_path, "r") as f:
                config = json.load(f)
                
            # Validate configuration
            validated_config = self.validate_config(config)
            
            self.logger.info(f"Loaded settings from {self.config_path}")
            return validated_config
            
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}")
            self.logger.info("Using default configuration")
            return self.default_config.copy()
    
    def save_settings(self, config: Dict[str, Any]) -> bool:
        """
        Save settings to configuration file.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            config_dir = os.path.dirname(self.config_path)
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
            
            # Save configuration
            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=4)
                
            self.logger.info(f"Saved settings to {self.config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            return False
    
    def update_settings(self, new_settings: Dict[str, Any]) -> bool:
        """
        Update settings with new values.
        
        Args:
            new_settings: New settings dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create backup before updating
            self.create_backup()
            
            # Merge new settings with existing settings
            updated_config = self.merge_configs(self.config, new_settings)
            
            # Validate updated configuration
            validated_config = self.validate_config(updated_config)
            
            # Save updated configuration
            if self.save_settings(validated_config):
                # Update current configuration
                self.config = validated_config
                self.logger.info("Settings updated successfully")
                return True
            else:
                self.logger.error("Failed to save updated settings")
                return False
                
        except Exception as e:
            self.logger.error(f"Error updating settings: {e}")
            return False
    
    def get_settings(self) -> Dict[str, Any]:
        """
        Get current settings.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def create_backup(self) -> bool:
        """
        Create a backup of the current configuration.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Skip if configuration file doesn't exist
            if not os.path.exists(self.config_path):
                return False
                
            # Create backup directory if it doesn't exist
            backup_dir = os.path.join(os.path.dirname(self.config_path), "settings_backup")
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
                
            # Create backup filename with timestamp
            timestamp = int(time.time())
            backup_path = os.path.join(backup_dir, f"settings_{timestamp}.json")
            
            # Copy configuration file to backup
            shutil.copy2(self.config_path, backup_path)
            
            self.logger.info(f"Created settings backup at {backup_path}")
            
            # Clean up old backups (keep last 10)
            self.cleanup_old_backups(10)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating settings backup: {e}")
            return False
    
    def cleanup_old_backups(self, keep_count: int = 10):
        """
        Clean up old backups, keeping only the specified number of most recent backups.
        
        Args:
            keep_count: Number of backups to keep
        """
        try:
            # Get backup directory
            backup_dir = os.path.join(os.path.dirname(self.config_path), "settings_backup")
            if not os.path.exists(backup_dir):
                return
                
            # Get list of backup files
            backup_files = [os.path.join(backup_dir, f) for f in os.listdir(backup_dir) if f.startswith("settings_") and f.endswith(".json")]
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Remove old backups
            if len(backup_files) > keep_count:
                for old_backup in backup_files[keep_count:]:
                    os.remove(old_backup)
                    self.logger.debug(f"Removed old settings backup: {old_backup}")
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up old backups: {e}")
    
    def restore_backup(self, backup_path: str) -> bool:
        """
        Restore configuration from backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if backup file exists
            if not os.path.exists(backup_path):
                self.logger.error(f"Backup file not found: {backup_path}")
                return False
                
            # Create backup of current configuration before restoring
            self.create_backup()
            
            # Copy backup file to configuration file
            shutil.copy2(backup_path, self.config_path)
            
            # Reload configuration
            self.config = self.load_settings()
            
            self.logger.info(f"Restored settings from backup: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error restoring settings from backup: {e}")
            return False
    
    def get_latest_backup(self) -> Optional[str]:
        """
        Get the path to the latest backup file.
        
        Returns:
            Path to latest backup file, or None if no backups exist
        """
        try:
            # Get backup directory
            backup_dir = os.path.join(os.path.dirname(self.config_path), "settings_backup")
            if not os.path.exists(backup_dir):
                return None
                
            # Get list of backup files
            backup_files = [os.path.join(backup_dir, f) for f in os.listdir(backup_dir) if f.startswith("settings_") and f.endswith(".json")]
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Return latest backup
            if backup_files:
                return backup_files[0]
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting latest backup: {e}")
            return None
    
    def get_backup_list(self) -> List[str]:
        """
        Get list of available backup files.
        
        Returns:
            List of backup file paths
        """
        try:
            # Get backup directory
            backup_dir = os.path.join(os.path.dirname(self.config_path), "settings_backup")
            if not os.path.exists(backup_dir):
                return []
                
            # Get list of backup files
            backup_files = [os.path.join(backup_dir, f) for f in os.listdir(backup_dir) if f.startswith("settings_") and f.endswith(".json")]
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            return backup_files
            
        except Exception as e:
            self.logger.error(f"Error getting backup list: {e}")
            return []
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration and fill in missing values with defaults.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Validated configuration dictionary
        """
        # Start with a copy of the default configuration
        validated = self.default_config.copy()
        
        # Update with provided configuration
        validated = self.merge_configs(validated, config)
        
        # Validate specific fields
        # Ensure poll interval is reasonable
        if validated.get("poll_interval_seconds", 0) < 1:
            validated["poll_interval_seconds"] = self.default_config["poll_interval_seconds"]
            self.logger.warning(f"Invalid poll interval, using default: {validated['poll_interval_seconds']}s")
            
        # Ensure risk percent is reasonable
        risk_percent = validated.get("risk_percent", 0)
        if risk_percent <= 0 or risk_percent > 0.1:  # Max 10% risk per trade
            validated["risk_percent"] = self.default_config["risk_percent"]
            self.logger.warning(f"Invalid risk percent, using default: {validated['risk_percent'] * 100}%")
            
        # Ensure trading mode is valid
        valid_modes = ["monitor_only", "paper_trading", "live_trading"]
        if validated.get("trading_mode", "") not in valid_modes:
            validated["trading_mode"] = self.default_config["trading_mode"]
            self.logger.warning(f"Invalid trading mode, using default: {validated['trading_mode']}")
            
        # Ensure strategy is valid
        valid_strategies = ["master_omni_overlord", "triple_confluence", "oracle_update", "enhanced_vwma"]
        if validated.get("strategy", "") not in valid_strategies:
            validated["strategy"] = self.default_config["strategy"]
            self.logger.warning(f"Invalid strategy, using default: {validated['strategy']}")
            
        return validated
    
    def merge_configs(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries, with overlay taking precedence.
        
        Args:
            base: Base configuration dictionary
            overlay: Overlay configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        result = base.copy()
        
        for key, value in overlay.items():
            # If both values are dictionaries, merge them recursively
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_configs(result[key], value)
            else:
                # Otherwise, overlay value takes precedence
                result[key] = value
                
        return result
    
    def get_sensitive_keys(self) -> List[str]:
        """
        Get list of sensitive configuration keys that should be masked in logs.
        
        Returns:
            List of sensitive key paths (dot notation)
        """
        return [
            "secret_key",
            "api_key",
            "api_secret",
            "private_key",
            "password",
            "notifications.email_password",
            "notifications.webhook_secret"
        ]
    
    def mask_sensitive_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a copy of the configuration with sensitive data masked.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration dictionary with sensitive data masked
        """
        # Create a deep copy of the configuration
        masked = json.loads(json.dumps(config))
        
        # Get sensitive keys
        sensitive_keys = self.get_sensitive_keys()
        
        # Mask sensitive data
        for key_path in sensitive_keys:
            parts = key_path.split(".")
            current = masked
            
            # Navigate to the parent object
            for i, part in enumerate(parts[:-1]):
                if part in current and isinstance(current[part], dict):
                    current = current[part]
                else:
                    break
                    
            # Mask the value if it exists and is not empty
            last_part = parts[-1]
            if last_part in current and current[last_part]:
                if isinstance(current[last_part], str):
                    # Mask string values
                    if len(current[last_part]) > 8:
                        # Show first 4 and last 4 characters
                        current[last_part] = current[last_part][:4] + "****" + current[last_part][-4:]
                    else:
                        # Mask completely
                        current[last_part] = "********"
                else:
                    # Mask non-string values
                    current[last_part] = "********"
                    
        return masked
    
    def print_config_summary(self):
        """Print a summary of the current configuration."""
        try:
            # Get masked configuration
            masked_config = self.mask_sensitive_data(self.config)
            
            # Print summary
            self.logger.info("Configuration summary:")
            self.logger.info(f"  Trading mode: {masked_config.get('trading_mode', 'unknown')}")
            self.logger.info(f"  Strategy: {masked_config.get('strategy', 'unknown')}")
            self.logger.info(f"  Symbols: {', '.join(masked_config.get('symbols', []))}")
            self.logger.info(f"  Poll interval: {masked_config.get('poll_interval_seconds', 0)}s")
            self.logger.info(f"  Risk per trade: {masked_config.get('risk_percent', 0) * 100}%")
            self.logger.info(f"  Max open positions: {masked_config.get('max_open_positions', 0)}")
            
            # Print API connection info
            self.logger.info(f"  API URL: {masked_config.get('api_url', 'unknown')}")
            self.logger.info(f"  Account address: {masked_config.get('account_address', 'not set')}")
            self.logger.info(f"  Secret key: {masked_config.get('secret_key', 'not set')}")
            
            # Print notification settings
            notifications = masked_config.get('notifications', {})
            if notifications.get('enabled', False):
                self.logger.info("  Notifications: Enabled")
                if notifications.get('email'):
                    self.logger.info(f"    Email: {notifications.get('email')}")
                if notifications.get('webhook_url'):
                    self.logger.info(f"    Webhook: {notifications.get('webhook_url')}")
            else:
                self.logger.info("  Notifications: Disabled")
                
        except Exception as e:
            self.logger.error(f"Error printing configuration summary: {e}")
