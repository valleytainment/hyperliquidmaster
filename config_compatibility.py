"""
Configuration Compatibility Module

This module ensures compatibility between the original config.json format
and the enhanced trading bot architecture, providing seamless migration
and backward compatibility.
"""

import json
import os
import logging
from typing import Dict, Any, Optional

class ConfigManager:
    """
    Configuration manager that handles loading, validation, and migration
    of configuration settings between original and enhanced bot versions.
    """
    
    def __init__(self, config_path: str = "config.json", logger: Optional[logging.Logger] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
            logger: Logger instance
        """
        self.config_path = config_path
        self.logger = logger or logging.getLogger(__name__)
        self.config = {}
        self.default_config = self._get_default_config()
        
        # Load configuration
        self._load_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration values.
        
        Returns:
            Default configuration dictionary
        """
        return {
            # Core settings
            "account_address": "",
            "secret_key": "",
            "api_url": "https://api.hyperliquid.xyz",
            "use_testnet": False,
            "poll_interval_seconds": 2,
            "micro_poll_interval": 2,
            
            # Trading settings
            "trade_symbol": "BTC-USD-PERP",
            "trade_mode": "perp",
            "symbols": ["BTC-USD-PERP", "ETH-USD-PERP", "SOL-USD-PERP"],
            
            # Technical indicators
            "fast_ma": 5,
            "slow_ma": 15,
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "boll_period": 20,
            "boll_stddev": 2.0,
            
            # Risk management
            "stop_loss_pct": 0.005,
            "take_profit_pct": 0.01,
            "use_trailing_stop": True,
            "trail_start_profit": 0.005,
            "trail_offset": 0.0025,
            "use_partial_tp": True,
            "partial_tp_levels": [0.005, 0.01],
            "partial_tp_ratios": [0.2, 0.2],
            "min_trade_interval": 60,
            "risk_percent": 0.01,
            "min_scrap_value": 0.03,
            "circuit_breaker_threshold": 0.05,
            "max_drawdown_threshold": 0.07,
            "max_consecutive_losses": 3,
            
            # Neural network settings
            "nn_lookback_bars": 30,
            "nn_hidden_size": 64,
            "nn_lr": 0.0003,
            "synergy_conf_threshold": 0.8,
            
            # Order settings
            "order_size": 0.25,
            "use_manual_entry_size": True,
            "manual_entry_size": 55.0,
            "use_manual_close_size": True,
            "position_close_size": 10.0,
            "taker_fee": 0.00042,
            
            # Enhanced bot settings
            "log_level": "INFO",
            "data_update_interval": 5,
            "state_update_interval": 5,
            "sentiment_update_interval": 300,
            "model_update_interval": 3600,
            "main_loop_interval": 5,
            "signal_confidence_threshold": 0.7,
            "min_training_samples": 100,
            
            # Strategy settings
            "use_triple_confluence_strategy": True,
            "use_oracle_update_strategy": True,
            "use_funding_arbitrage_strategy": True,
            "use_multi_timeframe_strategy": True,
            
            # Sentiment analysis settings
            "use_sentiment_analysis": True,
            "openai_api_key": "",
            "use_local_llm": False,
            "local_llm_url": "http://localhost:8000/v1",
            "sentiment_model": "gpt-3.5-turbo",
            "sentiment_cache_ttl": 3600,
            "sentiment_impact_weight": 0.2,
            
            # Advanced settings
            "rate_limit_max_calls": 10,
            "rate_limit_period": 1.0,
            "max_retries": 3,
            "retry_delay": 1.0,
            "retry_backoff_factor": 2.0
        }
        
    def _load_config(self):
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    user_config = json.load(f)
                    
                # Merge with default config
                self.config = self.default_config.copy()
                self.config.update(user_config)
                
                # Ensure backward compatibility
                self._ensure_compatibility()
                
                self.logger.info(f"Configuration loaded from {self.config_path}")
            else:
                self.logger.warning(f"Configuration file {self.config_path} not found, using defaults")
                self.config = self.default_config.copy()
                
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            self.config = self.default_config.copy()
            
    def _ensure_compatibility(self):
        """Ensure compatibility with original config format."""
        # Handle symbol list conversion
        if "trade_symbol" in self.config and "symbols" not in self.config:
            self.config["symbols"] = [self.config["trade_symbol"]]
            
        # Handle API URL format
        if "api_url" in self.config and not self.config["api_url"].startswith("http"):
            self.config["api_url"] = f"https://{self.config['api_url']}"
            
        # Convert legacy parameter names
        param_mapping = {
            "poll_interval": "poll_interval_seconds",
            "order_size_usd": "manual_entry_size",
            "circuit_breaker": "circuit_breaker_threshold"
        }
        
        for old_param, new_param in param_mapping.items():
            if old_param in self.config and new_param not in self.config:
                self.config[new_param] = self.config[old_param]
                
    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=2)
                
            self.logger.info(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            return False
            
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Configuration dictionary
        """
        return self.config
        
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.config.update(updates)
            return self.save_config()
        except Exception as e:
            self.logger.error(f"Error updating configuration: {str(e)}")
            return False
            
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate the current configuration.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check required fields
        required_fields = ["account_address"]
        for field in required_fields:
            if not self.config.get(field):
                validation_results["warnings"].append(f"Missing required field: {field}")
                
        # Check for secret key if not in read-only mode
        if not self.config.get("secret_key") and not self.config.get("read_only_mode", False):
            validation_results["warnings"].append("Secret key not provided, bot will run in read-only mode")
            
        # Check for valid symbols
        if not self.config.get("symbols"):
            validation_results["errors"].append("No trading symbols specified")
            validation_results["valid"] = False
            
        # Check risk parameters
        if self.config.get("risk_percent", 0) > 0.05:
            validation_results["warnings"].append("Risk percentage is set above 5%, which is relatively high")
            
        # Check for sentiment analysis configuration
        if self.config.get("use_sentiment_analysis", False) and not self.config.get("openai_api_key") and not self.config.get("use_local_llm", False):
            validation_results["warnings"].append("Sentiment analysis enabled but no API key or local LLM configured")
            
        return validation_results
        
    def get_original_config_format(self) -> Dict[str, Any]:
        """
        Get configuration in the original format for backward compatibility.
        
        Returns:
            Configuration in original format
        """
        original_format = {
            "account_address": self.config.get("account_address", ""),
            "secret_key": self.config.get("secret_key", ""),
            "api_url": self.config.get("api_url", "https://api.hyperliquid.xyz"),
            "poll_interval_seconds": self.config.get("poll_interval_seconds", 2),
            "micro_poll_interval": self.config.get("micro_poll_interval", 2),
            "trade_symbol": self.config.get("symbols", ["BTC-USD-PERP"])[0],
            "trade_mode": self.config.get("trade_mode", "perp"),
            
            # Technical indicators
            "fast_ma": self.config.get("fast_ma", 5),
            "slow_ma": self.config.get("slow_ma", 15),
            "rsi_period": self.config.get("rsi_period", 14),
            "macd_fast": self.config.get("macd_fast", 12),
            "macd_slow": self.config.get("macd_slow", 26),
            "macd_signal": self.config.get("macd_signal", 9),
            "boll_period": self.config.get("boll_period", 20),
            "boll_stddev": self.config.get("boll_stddev", 2.0),
            
            # Risk management
            "stop_loss_pct": self.config.get("stop_loss_pct", 0.005),
            "take_profit_pct": self.config.get("take_profit_pct", 0.01),
            "use_trailing_stop": self.config.get("use_trailing_stop", True),
            "trail_start_profit": self.config.get("trail_start_profit", 0.005),
            "trail_offset": self.config.get("trail_offset", 0.0025),
            "use_partial_tp": self.config.get("use_partial_tp", True),
            "partial_tp_levels": self.config.get("partial_tp_levels", [0.005, 0.01]),
            "partial_tp_ratios": self.config.get("partial_tp_ratios", [0.2, 0.2]),
            "min_trade_interval": self.config.get("min_trade_interval", 60),
            "risk_percent": self.config.get("risk_percent", 0.01),
            "min_scrap_value": self.config.get("min_scrap_value", 0.03),
            
            # Neural network settings
            "nn_lookback_bars": self.config.get("nn_lookback_bars", 30),
            "nn_hidden_size": self.config.get("nn_hidden_size", 64),
            "nn_lr": self.config.get("nn_lr", 0.0003),
            "synergy_conf_threshold": self.config.get("synergy_conf_threshold", 0.8),
            
            # Order settings
            "order_size": self.config.get("order_size", 0.25),
            "use_manual_entry_size": self.config.get("use_manual_entry_size", True),
            "manual_entry_size": self.config.get("manual_entry_size", 55.0),
            "use_manual_close_size": self.config.get("use_manual_close_size", True),
            "position_close_size": self.config.get("position_close_size", 10.0),
            "taker_fee": self.config.get("taker_fee", 0.00042),
            "circuit_breaker_threshold": self.config.get("circuit_breaker_threshold", 0.05)
        }
        
        return original_format


def create_or_migrate_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Create a new configuration or migrate an existing one.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(config_path):
        logger.info(f"Creating new configuration file: {config_path}")
        
        # Get user input for essential parameters
        account_address = input("Main wallet address (0x...): ").strip()
        secret_key = input("Enter your private key (0x...): ").strip()
        symbol = input("Default trading pair (e.g. BTC-USD-PERP): ").strip() or "BTC-USD-PERP"
        
        # Create basic configuration
        config = {
            "account_address": account_address,
            "secret_key": secret_key,
            "api_url": "https://api.hyperliquid.xyz",
            "symbols": [symbol],
            "trade_mode": "perp"
        }
        
        # Save configuration
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Configuration file created: {config_path}")
        
        # Initialize config manager with the new config
        config_manager = ConfigManager(config_path, logger)
        return config_manager.get_config()
    else:
        logger.info(f"Migrating existing configuration: {config_path}")
        
        # Load existing configuration
        config_manager = ConfigManager(config_path, logger)
        config = config_manager.get_config()
        
        # Validate configuration
        validation_results = config_manager.validate_config()
        
        # Display warnings and errors
        for warning in validation_results["warnings"]:
            logger.warning(f"Configuration warning: {warning}")
            
        for error in validation_results["errors"]:
            logger.error(f"Configuration error: {error}")
            
        # Save migrated configuration
        config_manager.save_config()
        
        return config
