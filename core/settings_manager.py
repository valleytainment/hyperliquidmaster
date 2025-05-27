"""
Settings Manager for Hyperliquid Trading Bot
-------------------------------------------
Provides robust settings management with automatic backup and restore functionality.
"""

import os
import json
import time
import logging
import shutil
import threading
from typing import Dict, Any, Optional, List

class SettingsManager:
    """
    Settings manager with automatic backup and restore functionality.
    Ensures settings persistence across sessions and prevents data loss.
    """
    
    def __init__(self, config_path: str, logger=None):
        """
        Initialize the settings manager.
        
        Args:
            config_path: Path to the main configuration file
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger("SettingsManager")
        self.config_path = config_path
        
        # Backup settings
        self.backup_dir = os.path.join(os.path.dirname(config_path), "settings_backup")
        self.max_backups = 5
        self.backup_interval = 3600  # 1 hour in seconds
        self.last_backup = 0
        
        # Settings lock for thread safety
        self.settings_lock = threading.Lock()
        
        # Current settings
        self.settings = {}
        
        # Create backup directory if it doesn't exist
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Load settings
        self._load_settings()
        
        # Create initial backup
        self._create_backup()
    
    def _load_settings(self) -> None:
        """Load settings from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.settings = json.load(f)
                self.logger.info(f"Loaded settings from {self.config_path}")
            else:
                self.logger.warning(f"Settings file not found at {self.config_path}, using empty settings")
                self.settings = {}
                
                # Try to restore from backup if available
                if self._restore_from_backup():
                    self.logger.info("Settings restored from backup")
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}")
            self.settings = {}
            
            # Try to restore from backup if available
            if self._restore_from_backup():
                self.logger.info("Settings restored from backup after load error")
    
    def _save_settings(self) -> bool:
        """
        Save settings to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a temporary file first
            temp_path = f"{self.config_path}.tmp"
            with open(temp_path, 'w') as f:
                json.dump(self.settings, f, indent=2)
            
            # Rename the temporary file to the actual file
            # This ensures atomic write to prevent corruption
            shutil.move(temp_path, self.config_path)
            
            # Check if backup is due
            current_time = time.time()
            if current_time - self.last_backup > self.backup_interval:
                self._create_backup()
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            return False
    
    def _create_backup(self) -> bool:
        """
        Create a backup of the current settings.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Skip if settings file doesn't exist
            if not os.path.exists(self.config_path):
                return False
            
            # Create backup filename with timestamp
            timestamp = int(time.time())
            backup_path = os.path.join(self.backup_dir, f"settings_{timestamp}.json")
            
            # Copy settings file to backup
            shutil.copy2(self.config_path, backup_path)
            
            # Update last backup time
            self.last_backup = time.time()
            
            # Clean up old backups
            self._cleanup_old_backups()
            
            self.logger.info(f"Created settings backup at {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error creating settings backup: {e}")
            return False
    
    def _cleanup_old_backups(self) -> None:
        """Clean up old backups, keeping only the most recent ones."""
        try:
            # Get all backup files
            backup_files = []
            for filename in os.listdir(self.backup_dir):
                if filename.startswith("settings_") and filename.endswith(".json"):
                    backup_path = os.path.join(self.backup_dir, filename)
                    backup_files.append((backup_path, os.path.getmtime(backup_path)))
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old backups
            if len(backup_files) > self.max_backups:
                for backup_path, _ in backup_files[self.max_backups:]:
                    os.remove(backup_path)
                    self.logger.debug(f"Removed old backup: {backup_path}")
        except Exception as e:
            self.logger.error(f"Error cleaning up old backups: {e}")
    
    def _restore_from_backup(self) -> bool:
        """
        Restore settings from the most recent backup.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all backup files
            backup_files = []
            for filename in os.listdir(self.backup_dir):
                if filename.startswith("settings_") and filename.endswith(".json"):
                    backup_path = os.path.join(self.backup_dir, filename)
                    backup_files.append((backup_path, os.path.getmtime(backup_path)))
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: x[1], reverse=True)
            
            # Restore from the most recent backup
            if backup_files:
                most_recent_backup = backup_files[0][0]
                with open(most_recent_backup, 'r') as f:
                    self.settings = json.load(f)
                
                # Save restored settings
                self._save_settings()
                
                self.logger.info(f"Restored settings from backup: {most_recent_backup}")
                return True
            else:
                self.logger.warning("No backup files found")
                return False
        except Exception as e:
            self.logger.error(f"Error restoring from backup: {e}")
            return False
    
    def get_settings(self) -> Dict[str, Any]:
        """
        Get current settings.
        
        Returns:
            Dict containing current settings
        """
        with self.settings_lock:
            return self.settings.copy()
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a specific setting.
        
        Args:
            key: Setting key
            default: Default value if setting doesn't exist
            
        Returns:
            Setting value or default
        """
        with self.settings_lock:
            return self.settings.get(key, default)
    
    def set_setting(self, key: str, value: Any) -> bool:
        """
        Set a specific setting.
        
        Args:
            key: Setting key
            value: Setting value
            
        Returns:
            True if successful, False otherwise
        """
        with self.settings_lock:
            self.settings[key] = value
            return self._save_settings()
    
    def update_settings(self, new_settings: Dict[str, Any]) -> bool:
        """
        Update multiple settings.
        
        Args:
            new_settings: Dict containing new settings
            
        Returns:
            True if successful, False otherwise
        """
        with self.settings_lock:
            self.settings.update(new_settings)
            return self._save_settings()
    
    def delete_setting(self, key: str) -> bool:
        """
        Delete a specific setting.
        
        Args:
            key: Setting key
            
        Returns:
            True if successful, False otherwise
        """
        with self.settings_lock:
            if key in self.settings:
                del self.settings[key]
                return self._save_settings()
            return True
    
    def reset_settings(self) -> bool:
        """
        Reset all settings to default (empty).
        
        Returns:
            True if successful, False otherwise
        """
        with self.settings_lock:
            # Create backup before reset
            self._create_backup()
            
            # Reset settings
            self.settings = {}
            return self._save_settings()
    
    def force_backup(self) -> bool:
        """
        Force creation of a backup.
        
        Returns:
            True if successful, False otherwise
        """
        with self.settings_lock:
            return self._create_backup()
    
    def restore_backup(self, backup_index: int = 0) -> bool:
        """
        Restore settings from a specific backup.
        
        Args:
            backup_index: Index of the backup to restore (0 = most recent)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all backup files
            backup_files = []
            for filename in os.listdir(self.backup_dir):
                if filename.startswith("settings_") and filename.endswith(".json"):
                    backup_path = os.path.join(self.backup_dir, filename)
                    backup_files.append((backup_path, os.path.getmtime(backup_path)))
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: x[1], reverse=True)
            
            # Check if backup index is valid
            if backup_index < 0 or backup_index >= len(backup_files):
                self.logger.error(f"Invalid backup index: {backup_index}")
                return False
            
            # Restore from the specified backup
            backup_path = backup_files[backup_index][0]
            with self.settings_lock:
                with open(backup_path, 'r') as f:
                    self.settings = json.load(f)
                
                # Save restored settings
                self._save_settings()
                
                self.logger.info(f"Restored settings from backup: {backup_path}")
                return True
        except Exception as e:
            self.logger.error(f"Error restoring from backup: {e}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all available backups.
        
        Returns:
            List of dicts containing backup information
        """
        try:
            # Get all backup files
            backup_files = []
            for filename in os.listdir(self.backup_dir):
                if filename.startswith("settings_") and filename.endswith(".json"):
                    backup_path = os.path.join(self.backup_dir, filename)
                    mtime = os.path.getmtime(backup_path)
                    backup_files.append({
                        "path": backup_path,
                        "timestamp": int(mtime),
                        "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
                        "size": os.path.getsize(backup_path)
                    })
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return backup_files
        except Exception as e:
            self.logger.error(f"Error listing backups: {e}")
            return []
