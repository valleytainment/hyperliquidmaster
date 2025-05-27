"""
Advanced API Management for HyperLiquid Trading Bot
Supports multiple exchange accounts, key rotation, and usage monitoring
"""

import os
import json
import time
import logging
import uuid
import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

@dataclass
class APIKeyConfig:
    """Data class for API key configuration"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    account_address: str = ""
    secret_key: str = ""
    exchange: str = "hyperliquid"
    permissions: List[str] = field(default_factory=lambda: ["read", "trade"])
    created_at: float = field(default_factory=time.time)
    last_used: float = 0
    last_rotation: float = 0
    usage_count: int = 0
    is_active: bool = True
    notes: str = ""

class AdvancedAPIManager:
    """
    Advanced API Manager for the HyperLiquid trading bot.
    Handles multiple API keys, rotation, permissions, and usage monitoring.
    """
    
    def __init__(self, config_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the advanced API manager.
        
        Args:
            config_path: Path to the configuration directory
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger("AdvancedAPIManager")
        self.config_dir = config_path
        self.api_keys_file = os.path.join(self.config_dir, "api_keys.json")
        self.backup_dir = os.path.join(self.config_dir, "api_keys_backup")
        self.api_keys: Dict[str, APIKeyConfig] = {}
        self.active_key_id: Optional[str] = None
        
        # Create directories if they don't exist
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Load API keys
        self.load_api_keys()
    
    def load_api_keys(self) -> Dict[str, APIKeyConfig]:
        """
        Load API keys from the configuration file.
        
        Returns:
            Dict containing the API keys
        """
        try:
            if os.path.exists(self.api_keys_file):
                with open(self.api_keys_file, 'r') as f:
                    data = json.load(f)
                
                # Convert dict to APIKeyConfig objects
                self.api_keys = {}
                for key_id, key_data in data.get("keys", {}).items():
                    self.api_keys[key_id] = APIKeyConfig(**key_data)
                
                # Set active key
                self.active_key_id = data.get("active_key_id")
                
                self.logger.info(f"Loaded {len(self.api_keys)} API keys from {self.api_keys_file}")
            else:
                self.logger.warning(f"API keys file not found: {self.api_keys_file}")
                self.api_keys = {}
                self.active_key_id = None
            
            return self.api_keys
        except Exception as e:
            self.logger.error(f"Error loading API keys: {e}")
            self.api_keys = {}
            self.active_key_id = None
            return self.api_keys
    
    def save_api_keys(self) -> bool:
        """
        Save API keys to the configuration file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create backup before saving
            self._create_backup()
            
            # Convert APIKeyConfig objects to dict
            keys_dict = {}
            for key_id, key_config in self.api_keys.items():
                keys_dict[key_id] = asdict(key_config)
            
            data = {
                "keys": keys_dict,
                "active_key_id": self.active_key_id,
                "last_updated": time.time()
            }
            
            with open(self.api_keys_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Saved {len(self.api_keys)} API keys to {self.api_keys_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving API keys: {e}")
            return False
    
    def _create_backup(self) -> str:
        """
        Create a backup of the current API keys.
        
        Returns:
            Path to the backup file
        """
        try:
            if not os.path.exists(self.api_keys_file):
                return ""
            
            timestamp = int(time.time())
            backup_path = os.path.join(self.backup_dir, f"api_keys_{timestamp}.json")
            
            with open(self.api_keys_file, 'r') as src:
                with open(backup_path, 'w') as dst:
                    dst.write(src.read())
            
            self.logger.info(f"Created API keys backup at {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.error(f"Error creating API keys backup: {e}")
            return ""
    
    def add_api_key(self, name: str, account_address: str, secret_key: str, 
                   exchange: str = "hyperliquid", permissions: List[str] = None,
                   notes: str = "") -> Tuple[bool, str, Optional[str]]:
        """
        Add a new API key.
        
        Args:
            name: Name for the API key
            account_address: Account address
            secret_key: Secret key
            exchange: Exchange name
            permissions: List of permissions
            notes: Additional notes
            
        Returns:
            Tuple of (success, message, key_id)
        """
        try:
            # Validate inputs
            if not name or name.strip() == "":
                return False, "Name cannot be empty", None
            
            if not account_address or account_address.strip() == "":
                return False, "Account address cannot be empty", None
            
            if not secret_key or secret_key.strip() == "":
                return False, "Secret key cannot be empty", None
            
            # Create new API key config
            key_id = str(uuid.uuid4())
            new_key = APIKeyConfig(
                id=key_id,
                name=name.strip(),
                account_address=account_address.strip(),
                secret_key=secret_key.strip(),
                exchange=exchange.strip(),
                permissions=permissions or ["read", "trade"],
                created_at=time.time(),
                last_used=0,
                last_rotation=time.time(),
                usage_count=0,
                is_active=True,
                notes=notes.strip()
            )
            
            # Add to API keys
            self.api_keys[key_id] = new_key
            
            # If this is the first key, set it as active
            if self.active_key_id is None:
                self.active_key_id = key_id
            
            # Save API keys
            if self.save_api_keys():
                self.logger.info(f"Added new API key: {name}")
                return True, f"Added new API key: {name}", key_id
            else:
                # Remove key if save failed
                if key_id in self.api_keys:
                    del self.api_keys[key_id]
                return False, "Failed to save API keys", None
        except Exception as e:
            self.logger.error(f"Error adding API key: {e}")
            return False, f"Error adding API key: {e}", None
    
    def update_api_key(self, key_id: str, **kwargs) -> Tuple[bool, str]:
        """
        Update an existing API key.
        
        Args:
            key_id: ID of the API key to update
            **kwargs: Fields to update
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Check if key exists
            if key_id not in self.api_keys:
                return False, f"API key not found: {key_id}"
            
            # Update fields
            key_config = self.api_keys[key_id]
            for field, value in kwargs.items():
                if hasattr(key_config, field):
                    setattr(key_config, field, value)
            
            # Save API keys
            if self.save_api_keys():
                self.logger.info(f"Updated API key: {key_config.name}")
                return True, f"Updated API key: {key_config.name}"
            else:
                return False, "Failed to save API keys"
        except Exception as e:
            self.logger.error(f"Error updating API key: {e}")
            return False, f"Error updating API key: {e}"
    
    def delete_api_key(self, key_id: str) -> Tuple[bool, str]:
        """
        Delete an API key.
        
        Args:
            key_id: ID of the API key to delete
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Check if key exists
            if key_id not in self.api_keys:
                return False, f"API key not found: {key_id}"
            
            # Get key name for logging
            key_name = self.api_keys[key_id].name
            
            # Delete key
            del self.api_keys[key_id]
            
            # If this was the active key, set active key to None
            if self.active_key_id == key_id:
                self.active_key_id = next(iter(self.api_keys.keys())) if self.api_keys else None
            
            # Save API keys
            if self.save_api_keys():
                self.logger.info(f"Deleted API key: {key_name}")
                return True, f"Deleted API key: {key_name}"
            else:
                return False, "Failed to save API keys"
        except Exception as e:
            self.logger.error(f"Error deleting API key: {e}")
            return False, f"Error deleting API key: {e}"
    
    def set_active_key(self, key_id: str) -> Tuple[bool, str]:
        """
        Set the active API key.
        
        Args:
            key_id: ID of the API key to set as active
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Check if key exists
            if key_id not in self.api_keys:
                return False, f"API key not found: {key_id}"
            
            # Set active key
            self.active_key_id = key_id
            
            # Save API keys
            if self.save_api_keys():
                self.logger.info(f"Set active API key: {self.api_keys[key_id].name}")
                return True, f"Set active API key: {self.api_keys[key_id].name}"
            else:
                return False, "Failed to save API keys"
        except Exception as e:
            self.logger.error(f"Error setting active API key: {e}")
            return False, f"Error setting active API key: {e}"
    
    def get_active_key(self) -> Optional[APIKeyConfig]:
        """
        Get the active API key.
        
        Returns:
            Active API key config or None
        """
        if self.active_key_id and self.active_key_id in self.api_keys:
            return self.api_keys[self.active_key_id]
        return None
    
    def get_active_key_credentials(self) -> Tuple[str, str]:
        """
        Get the active API key credentials.
        
        Returns:
            Tuple of (account_address, secret_key)
        """
        active_key = self.get_active_key()
        if active_key:
            # Update usage statistics
            active_key.last_used = time.time()
            active_key.usage_count += 1
            self.save_api_keys()
            
            return active_key.account_address, active_key.secret_key
        return "", ""
    
    def rotate_key(self, key_id: str, new_secret_key: str) -> Tuple[bool, str]:
        """
        Rotate an API key's secret key.
        
        Args:
            key_id: ID of the API key to rotate
            new_secret_key: New secret key
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Check if key exists
            if key_id not in self.api_keys:
                return False, f"API key not found: {key_id}"
            
            # Update secret key
            self.api_keys[key_id].secret_key = new_secret_key
            self.api_keys[key_id].last_rotation = time.time()
            
            # Save API keys
            if self.save_api_keys():
                self.logger.info(f"Rotated API key: {self.api_keys[key_id].name}")
                return True, f"Rotated API key: {self.api_keys[key_id].name}"
            else:
                return False, "Failed to save API keys"
        except Exception as e:
            self.logger.error(f"Error rotating API key: {e}")
            return False, f"Error rotating API key: {e}"
    
    def import_api_keys(self, file_path: str) -> Tuple[bool, str, int]:
        """
        Import API keys from a file.
        
        Args:
            file_path: Path to the file to import from
            
        Returns:
            Tuple of (success, message, number of keys imported)
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return False, f"File not found: {file_path}", 0
            
            # Load keys from file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Import keys
            imported_count = 0
            
            # Handle different import formats
            if "keys" in data:
                # Our format
                for key_id, key_data in data["keys"].items():
                    if key_id not in self.api_keys:
                        self.api_keys[key_id] = APIKeyConfig(**key_data)
                        imported_count += 1
            elif isinstance(data, list):
                # List of keys
                for key_data in data:
                    key_id = key_data.get("id", str(uuid.uuid4()))
                    if key_id not in self.api_keys:
                        self.api_keys[key_id] = APIKeyConfig(**key_data)
                        imported_count += 1
            elif "account_address" in data and "secret_key" in data:
                # Single key
                key_id = data.get("id", str(uuid.uuid4()))
                if key_id not in self.api_keys:
                    self.api_keys[key_id] = APIKeyConfig(**data)
                    imported_count += 1
            
            # Set active key if none is set
            if self.active_key_id is None and self.api_keys:
                self.active_key_id = next(iter(self.api_keys.keys()))
            
            # Save API keys
            if self.save_api_keys():
                self.logger.info(f"Imported {imported_count} API keys from {file_path}")
                return True, f"Imported {imported_count} API keys", imported_count
            else:
                return False, "Failed to save imported API keys", 0
        except Exception as e:
            self.logger.error(f"Error importing API keys: {e}")
            return False, f"Error importing API keys: {e}", 0
    
    def export_api_keys(self, file_path: str, include_secrets: bool = False) -> Tuple[bool, str]:
        """
        Export API keys to a file.
        
        Args:
            file_path: Path to the file to export to
            include_secrets: Whether to include secret keys
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Convert APIKeyConfig objects to dict
            keys_dict = {}
            for key_id, key_config in self.api_keys.items():
                key_data = asdict(key_config)
                
                # Remove secret key if not including secrets
                if not include_secrets:
                    key_data["secret_key"] = "********"
                
                keys_dict[key_id] = key_data
            
            data = {
                "keys": keys_dict,
                "active_key_id": self.active_key_id,
                "exported_at": time.time(),
                "exported_by": "HyperliquidMaster"
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Exported {len(self.api_keys)} API keys to {file_path}")
            return True, f"Exported {len(self.api_keys)} API keys to {file_path}"
        except Exception as e:
            self.logger.error(f"Error exporting API keys: {e}")
            return False, f"Error exporting API keys: {e}"
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """
        Get usage statistics for all API keys.
        
        Returns:
            Dict containing usage statistics
        """
        stats = {
            "total_keys": len(self.api_keys),
            "active_keys": sum(1 for k in self.api_keys.values() if k.is_active),
            "total_usage": sum(k.usage_count for k in self.api_keys.values()),
            "last_used": max((k.last_used for k in self.api_keys.values()), default=0),
            "oldest_key": min((k.created_at for k in self.api_keys.values()), default=0),
            "newest_key": max((k.created_at for k in self.api_keys.values()), default=0),
            "keys": {}
        }
        
        for key_id, key_config in self.api_keys.items():
            stats["keys"][key_id] = {
                "name": key_config.name,
                "usage_count": key_config.usage_count,
                "last_used": key_config.last_used,
                "last_used_formatted": datetime.datetime.fromtimestamp(key_config.last_used).strftime("%Y-%m-%d %H:%M:%S") if key_config.last_used > 0 else "Never",
                "created_at": key_config.created_at,
                "created_at_formatted": datetime.datetime.fromtimestamp(key_config.created_at).strftime("%Y-%m-%d %H:%M:%S"),
                "age_days": (time.time() - key_config.created_at) / (60 * 60 * 24),
                "is_active": key_config.is_active,
                "is_current": key_id == self.active_key_id
            }
        
        return stats
    
    def validate_all_keys(self) -> Dict[str, bool]:
        """
        Validate all API keys.
        
        Returns:
            Dict mapping key IDs to validation results
        """
        results = {}
        for key_id, key_config in self.api_keys.items():
            # Basic validation
            is_valid = (
                key_config.account_address and 
                key_config.secret_key and 
                key_config.is_active
            )
            results[key_id] = is_valid
        
        return results
