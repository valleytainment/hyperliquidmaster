"""
Trading mode management module for the HyperliquidMaster trading bot.
Provides functionality to check and change the bot's operating mode.
"""

import os
import json
import logging
import time
from enum import Enum
from typing import Dict, Any, Optional, List, Union

class TradingMode(Enum):
    """Enum representing the different trading modes available."""
    PAPER_TRADING = "paper_trading"  # Simulated trading with no real orders
    LIVE_TRADING = "live_trading"    # Real trading with actual orders
    MONITOR_ONLY = "monitor_only"    # Only monitor markets, no trading
    BACKTEST = "backtest"            # Backtesting mode using historical data
    AGGRESSIVE = "aggressive"        # Aggressive trading strategy
    CONSERVATIVE = "conservative"    # Conservative trading strategy
    CUSTOM = "custom"                # Custom trading strategy

class TradingModeManager:
    """
    Trading mode manager for the HyperliquidMaster trading bot.
    Provides functionality to check and change the bot's operating mode.
    """
    
    def __init__(self, config_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the trading mode manager.
        
        Args:
            config_path: Path to the configuration file
            logger: Logger instance
        """
        self.config_path = config_path
        self.logger = logger or logging.getLogger("TradingModeManager")
        self.config = self._load_config()
        
        # Default mode settings
        self.default_mode_settings = {
            TradingMode.PAPER_TRADING: {
                "use_real_orders": False,
                "risk_percent": 0.01,
                "max_position_size": 1.0,
                "use_market_orders": False
            },
            TradingMode.LIVE_TRADING: {
                "use_real_orders": True,
                "risk_percent": 0.01,
                "max_position_size": 1.0,
                "use_market_orders": False
            },
            TradingMode.MONITOR_ONLY: {
                "use_real_orders": False,
                "risk_percent": 0.0,
                "max_position_size": 0.0,
                "use_market_orders": False
            },
            TradingMode.AGGRESSIVE: {
                "use_real_orders": True,
                "risk_percent": 0.02,
                "max_position_size": 2.0,
                "use_market_orders": True
            },
            TradingMode.CONSERVATIVE: {
                "use_real_orders": True,
                "risk_percent": 0.005,
                "max_position_size": 0.5,
                "use_market_orders": False
            }
        }
        
        # Initialize current mode
        self._initialize_mode()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Dict containing the configuration
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"Config file not found at {self.config_path}, using empty config")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}
    
    def _save_config(self) -> bool:
        """
        Save configuration to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            return False
    
    def _initialize_mode(self) -> None:
        """Initialize the trading mode from config."""
        # Get mode from config or default to PAPER_TRADING
        mode_str = self.config.get("trading_mode", TradingMode.PAPER_TRADING.value)
        
        # Convert string to enum
        try:
            self.current_mode = TradingMode(mode_str)
        except ValueError:
            self.logger.warning(f"Invalid trading mode '{mode_str}', defaulting to PAPER_TRADING")
            self.current_mode = TradingMode.PAPER_TRADING
            
            # Update config with default mode
            self.config["trading_mode"] = self.current_mode.value
            self._save_config()
        
        # Initialize mode settings if not present
        if "mode_settings" not in self.config:
            self.config["mode_settings"] = {}
        
        # Initialize settings for current mode if not present
        if self.current_mode.value not in self.config["mode_settings"]:
            default_settings = self.default_mode_settings.get(
                self.current_mode, 
                self.default_mode_settings[TradingMode.PAPER_TRADING]
            )
            self.config["mode_settings"][self.current_mode.value] = default_settings
            self._save_config()
        
        self.logger.info(f"Trading mode initialized to: {self.current_mode.value}")
    
    def get_current_mode(self) -> TradingMode:
        """
        Get the current trading mode.
        
        Returns:
            Current trading mode
        """
        return self.current_mode
    
    def get_mode_settings(self, mode: Optional[TradingMode] = None) -> Dict[str, Any]:
        """
        Get settings for a specific mode or current mode if not specified.
        
        Args:
            mode: Trading mode to get settings for
            
        Returns:
            Dict containing mode settings
        """
        mode = mode or self.current_mode
        
        # Get settings from config or default
        if "mode_settings" in self.config and mode.value in self.config["mode_settings"]:
            return self.config["mode_settings"][mode.value]
        else:
            return self.default_mode_settings.get(
                mode, 
                self.default_mode_settings[TradingMode.PAPER_TRADING]
            )
    
    def set_mode(self, mode: Union[TradingMode, str]) -> bool:
        """
        Set the current trading mode.
        
        Args:
            mode: Trading mode to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert string to enum if necessary
            if isinstance(mode, str):
                try:
                    mode = TradingMode(mode)
                except ValueError:
                    self.logger.error(f"Invalid trading mode: {mode}")
                    return False
            
            # Update current mode
            self.current_mode = mode
            
            # Update config
            self.config["trading_mode"] = mode.value
            
            # Initialize settings for mode if not present
            if "mode_settings" not in self.config:
                self.config["mode_settings"] = {}
            
            if mode.value not in self.config["mode_settings"]:
                default_settings = self.default_mode_settings.get(
                    mode, 
                    self.default_mode_settings[TradingMode.PAPER_TRADING]
                )
                self.config["mode_settings"][mode.value] = default_settings
            
            # Save config
            success = self._save_config()
            
            if success:
                self.logger.info(f"Trading mode set to: {mode.value}")
            
            return success
        except Exception as e:
            self.logger.error(f"Error setting trading mode: {e}")
            return False
    
    def update_mode_settings(self, settings: Dict[str, Any], mode: Optional[TradingMode] = None) -> bool:
        """
        Update settings for a specific mode or current mode if not specified.
        
        Args:
            settings: Dict containing settings to update
            mode: Trading mode to update settings for
            
        Returns:
            True if successful, False otherwise
        """
        try:
            mode = mode or self.current_mode
            
            # Initialize mode settings if not present
            if "mode_settings" not in self.config:
                self.config["mode_settings"] = {}
            
            # Initialize settings for mode if not present
            if mode.value not in self.config["mode_settings"]:
                default_settings = self.default_mode_settings.get(
                    mode, 
                    self.default_mode_settings[TradingMode.PAPER_TRADING]
                )
                self.config["mode_settings"][mode.value] = default_settings
            
            # Update settings
            for key, value in settings.items():
                self.config["mode_settings"][mode.value][key] = value
            
            # Save config
            success = self._save_config()
            
            if success:
                self.logger.info(f"Settings updated for mode: {mode.value}")
            
            return success
        except Exception as e:
            self.logger.error(f"Error updating mode settings: {e}")
            return False
    
    def get_available_modes(self) -> List[str]:
        """
        Get list of available trading modes.
        
        Returns:
            List of available trading modes
        """
        return [mode.value for mode in TradingMode]
    
    def is_real_trading(self) -> bool:
        """
        Check if current mode is using real orders.
        
        Returns:
            True if using real orders, False otherwise
        """
        settings = self.get_mode_settings()
        return settings.get("use_real_orders", False)
    
    def get_risk_percent(self) -> float:
        """
        Get risk percent for current mode.
        
        Returns:
            Risk percent
        """
        settings = self.get_mode_settings()
        return settings.get("risk_percent", 0.01)
    
    def get_max_position_size(self) -> float:
        """
        Get maximum position size for current mode.
        
        Returns:
            Maximum position size
        """
        settings = self.get_mode_settings()
        return settings.get("max_position_size", 1.0)
    
    def should_use_market_orders(self) -> bool:
        """
        Check if current mode should use market orders.
        
        Returns:
            True if should use market orders, False otherwise
        """
        settings = self.get_mode_settings()
        return settings.get("use_market_orders", False)
