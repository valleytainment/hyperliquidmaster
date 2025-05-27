"""
Config compatibility module for the Hyperliquid trading bot.
This is a mock implementation for testing purposes.
"""

import json
import os

class ConfigManager:
    def __init__(self, config_path, logger):
        self.config_path = config_path
        self.logger = logger
        self.config = self._load_config()
        
    def _load_config(self):
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                # Create default config
                default_config = {
                    "account_address": "",
                    "secret_key": "",
                    "symbols": ["BTC", "ETH", "SOL"],
                    "theme": "dark"
                }
                
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                
                return default_config
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}
            
    def get_config(self):
        """Get the current configuration."""
        return self.config
        
    def save_config(self, config):
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.config = config
            return True
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            return False
