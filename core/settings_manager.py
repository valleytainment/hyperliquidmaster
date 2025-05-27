"""
Settings Manager for HyperliquidMaster

This module provides functionality to manage settings with backup and restore capabilities.
"""

import os
import json
import logging
import shutil
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

class SettingsManager:
    """
    Manages settings with backup and restore capabilities.
    
    This class provides functionality to load, save, backup, and restore settings
    for the HyperliquidMaster application.
    """
    
    def __init__(self, config_path: str = None, logger: logging.Logger = None):
        """
        Initialize the settings manager.
        
        Args:
            config_path: Path to the configuration file (optional, defaults to config.json)
            logger: Logger instance (optional, creates one if not provided)
        """
        # Set default config path if not provided
        if config_path is None:
            self.config_path = "config.json"
        else:
            self.config_path = config_path
            
        # Set default logger if not provided
        if logger is None:
            self.logger = logging.getLogger("SettingsManager")
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger
            
        self.backup_dir = "settings_backup"
        self.max_backups = 10
        
        # Create backup directory if it doesn't exist
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def load_settings(self) -> Dict[str, Any]:
        """
        Load settings from file with automatic restore if needed.
        
        Returns:
            Dictionary containing the settings
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    settings = json.load(f)
                
                self.logger.info(f"Settings loaded from {self.config_path}")
                return settings
            else:
                self.logger.warning(f"Settings file {self.config_path} not found")
                
                # Try to restore from backup
                restored_settings = self._restore_from_backup()
                if restored_settings:
                    self.logger.info("Settings restored from backup")
                    return restored_settings
                
                # Return empty settings if no backup available
                self.logger.warning("No backup available, using default settings")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}")
            
            # Try to restore from backup
            restored_settings = self._restore_from_backup()
            if restored_settings:
                self.logger.info("Settings restored from backup after error")
                return restored_settings
            
            # Return empty settings if no backup available
            self.logger.warning("No backup available, using default settings")
            return {}
    
    def save_settings(self, settings: Dict[str, Any]) -> bool:
        """
        Save settings to file with automatic backup.
        
        Args:
            settings: Dictionary containing the settings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create backup before saving
            if os.path.exists(self.config_path):
                self._create_backup()
            
            # Save settings
            with open(self.config_path, 'w') as f:
                json.dump(settings, f, indent=4)
            
            self.logger.info(f"Settings saved to {self.config_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            return False
    
    def _create_backup(self) -> bool:
        """
        Create a backup of the current settings file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.backup_dir, f"config_{timestamp}.json")
            
            # Copy current settings to backup
            shutil.copy2(self.config_path, backup_path)
            
            self.logger.info(f"Settings backup created at {backup_path}")
            
            # Clean up old backups
            self._cleanup_old_backups()
            
            return True
        except Exception as e:
            self.logger.error(f"Error creating settings backup: {e}")
            return False
    
    def _cleanup_old_backups(self) -> None:
        """
        Clean up old backups, keeping only the most recent ones.
        """
        try:
            # Get list of backup files
            backup_files = [os.path.join(self.backup_dir, f) for f in os.listdir(self.backup_dir) 
                           if f.startswith("config_") and f.endswith(".json")]
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Remove old backups
            if len(backup_files) > self.max_backups:
                for old_backup in backup_files[self.max_backups:]:
                    os.remove(old_backup)
                    self.logger.info(f"Removed old backup: {old_backup}")
        except Exception as e:
            self.logger.error(f"Error cleaning up old backups: {e}")
    
    def _restore_from_backup(self) -> Optional[Dict[str, Any]]:
        """
        Restore settings from the most recent backup.
        
        Returns:
            Dictionary containing the restored settings, or None if no backup available
        """
        try:
            # Get list of backup files
            backup_files = [os.path.join(self.backup_dir, f) for f in os.listdir(self.backup_dir) 
                           if f.startswith("config_") and f.endswith(".json")]
            
            if not backup_files:
                self.logger.warning("No backup files found")
                return None
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Load settings from most recent backup
            with open(backup_files[0], 'r') as f:
                settings = json.load(f)
            
            # Save restored settings to main config file
            with open(self.config_path, 'w') as f:
                json.dump(settings, f, indent=4)
            
            self.logger.info(f"Settings restored from {backup_files[0]}")
            return settings
        except Exception as e:
            self.logger.error(f"Error restoring settings from backup: {e}")
            return None
    
    def get_backup_list(self) -> List[Dict[str, Any]]:
        """
        Get list of available backups.
        
        Returns:
            List of dictionaries containing backup information
        """
        try:
            # Get list of backup files
            backup_files = [os.path.join(self.backup_dir, f) for f in os.listdir(self.backup_dir) 
                           if f.startswith("config_") and f.endswith(".json")]
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Create list of backup information
            backups = []
            for backup_file in backup_files:
                backup_time = datetime.fromtimestamp(os.path.getmtime(backup_file))
                backup_size = os.path.getsize(backup_file)
                
                backups.append({
                    "file": backup_file,
                    "time": backup_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "size": backup_size
                })
            
            return backups
        except Exception as e:
            self.logger.error(f"Error getting backup list: {e}")
            return []
    
    def restore_specific_backup(self, backup_file: str) -> bool:
        """
        Restore settings from a specific backup file.
        
        Args:
            backup_file: Path to the backup file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if backup file exists
            if not os.path.exists(backup_file):
                self.logger.error(f"Backup file {backup_file} not found")
                return False
            
            # Create backup of current settings before restoring
            if os.path.exists(self.config_path):
                self._create_backup()
            
            # Load settings from backup
            with open(backup_file, 'r') as f:
                settings = json.load(f)
            
            # Save restored settings to main config file
            with open(self.config_path, 'w') as f:
                json.dump(settings, f, indent=4)
            
            self.logger.info(f"Settings restored from {backup_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error restoring settings from {backup_file}: {e}")
            return False
