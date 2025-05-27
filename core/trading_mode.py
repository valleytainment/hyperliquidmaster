"""
Trading Mode Manager for HyperliquidMaster

This module provides functionality to manage different trading modes
with specific risk parameters and behavior for each mode.
"""

import logging
import json
import os
from enum import Enum
from typing import Dict, Any, Optional

class TradingMode(Enum):
    """Trading modes for the HyperliquidMaster bot."""
    PAPER_TRADING = 1
    LIVE_TRADING = 2
    MONITOR_ONLY = 3
    AGGRESSIVE = 4
    CONSERVATIVE = 5

class TradingModeManager:
    """
    Manages trading modes and their specific settings.
    
    This class provides functionality to switch between different trading modes,
    each with its own risk parameters and behavior.
    """
    
    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Initialize the trading mode manager.
        
        Args:
            config: Configuration dictionary (optional, creates empty config if not provided)
            logger: Logger instance (optional, creates one if not provided)
        """
        # Initialize config if not provided
        if config is None:
            self.config = {}
        else:
            self.config = config
            
        # Initialize logger if not provided
        if logger is None:
            self.logger = logging.getLogger("TradingModeManager")
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger
            
        self.mode_settings_file = "mode_settings.json"
        
        # Initialize mode settings
        self.mode_settings = self._load_mode_settings()
        
        # Set default mode
        mode_name = self.config.get("trading_mode", "PAPER_TRADING")
        try:
            self.current_mode = TradingMode[mode_name]
        except (KeyError, ValueError):
            self.logger.warning(f"Invalid trading mode: {mode_name}, defaulting to PAPER_TRADING")
            self.current_mode = TradingMode.PAPER_TRADING
        
        self.logger.info(f"Trading mode initialized to {self.current_mode.name}")
    
    def _load_mode_settings(self) -> Dict[str, Dict[str, Any]]:
        """
        Load mode settings from file or create default settings.
        
        Returns:
            Dictionary of mode settings
        """
        if os.path.exists(self.mode_settings_file):
            try:
                with open(self.mode_settings_file, 'r') as f:
                    settings = json.load(f)
                
                # Convert string keys to enum keys
                return {TradingMode[k].name: v for k, v in settings.items()}
            except Exception as e:
                self.logger.error(f"Error loading mode settings: {e}")
                return self._create_default_settings()
        else:
            return self._create_default_settings()
    
    def _create_default_settings(self) -> Dict[str, Dict[str, Any]]:
        """
        Create default settings for all trading modes.
        
        Returns:
            Dictionary of default mode settings
        """
        settings = {
            TradingMode.PAPER_TRADING.name: {
                "risk_level": 0.02,
                "max_position_size": 0.1,
                "max_open_positions": 5,
                "use_stop_loss": True,
                "use_take_profit": True,
                "tp_multiplier": 2.0,
                "sl_multiplier": 1.0,
                "use_trailing_stop": True,
                "description": "Paper trading mode with simulated trades. No real money at risk."
            },
            TradingMode.LIVE_TRADING.name: {
                "risk_level": 0.01,
                "max_position_size": 0.05,
                "max_open_positions": 3,
                "use_stop_loss": True,
                "use_take_profit": True,
                "tp_multiplier": 2.0,
                "sl_multiplier": 1.0,
                "use_trailing_stop": True,
                "description": "Live trading mode with real money. Standard risk parameters."
            },
            TradingMode.MONITOR_ONLY.name: {
                "risk_level": 0.0,
                "max_position_size": 0.0,
                "max_open_positions": 0,
                "use_stop_loss": False,
                "use_take_profit": False,
                "tp_multiplier": 0.0,
                "sl_multiplier": 0.0,
                "use_trailing_stop": False,
                "description": "Monitor only mode. Generates signals but does not execute trades."
            },
            TradingMode.AGGRESSIVE.name: {
                "risk_level": 0.03,
                "max_position_size": 0.15,
                "max_open_positions": 7,
                "use_stop_loss": True,
                "use_take_profit": True,
                "tp_multiplier": 3.0,
                "sl_multiplier": 1.5,
                "use_trailing_stop": True,
                "description": "Aggressive trading mode with higher risk and reward parameters."
            },
            TradingMode.CONSERVATIVE.name: {
                "risk_level": 0.005,
                "max_position_size": 0.03,
                "max_open_positions": 2,
                "use_stop_loss": True,
                "use_take_profit": True,
                "tp_multiplier": 1.5,
                "sl_multiplier": 0.7,
                "use_trailing_stop": True,
                "description": "Conservative trading mode with lower risk and reward parameters."
            }
        }
        
        # Save default settings
        self._save_mode_settings(settings)
        
        return settings
    
    def _save_mode_settings(self, settings: Dict[str, Dict[str, Any]]) -> None:
        """
        Save mode settings to file.
        
        Args:
            settings: Dictionary of mode settings
        """
        try:
            with open(self.mode_settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving mode settings: {e}")
    
    def get_current_mode(self) -> TradingMode:
        """
        Get the current trading mode.
        
        Returns:
            Current trading mode
        """
        return self.current_mode
    
    def set_mode(self, mode: TradingMode) -> bool:
        """
        Set the trading mode.
        
        Args:
            mode: Trading mode to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.current_mode = mode
            
            # Update config
            self.config["trading_mode"] = mode.name
            
            self.logger.info(f"Trading mode set to {mode.name}")
            return True
        except Exception as e:
            self.logger.error(f"Error setting trading mode: {e}")
            return False
            
    def save_mode(self) -> bool:
        """
        Save the current trading mode to persistent storage.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a settings dictionary with the current mode
            settings = {
                "trading_mode": self.current_mode.name
            }
            
            # Save to a mode file
            with open("trading_mode.json", 'w') as f:
                json.dump(settings, f, indent=4)
                
            self.logger.info(f"Trading mode {self.current_mode.name} saved to persistent storage")
            return True
        except Exception as e:
            self.logger.error(f"Error saving trading mode: {e}")
            return False
    
    def get_mode_settings(self, mode: Optional[TradingMode] = None) -> Dict[str, Any]:
        """
        Get settings for the specified mode or current mode.
        
        Args:
            mode: Trading mode to get settings for, or None for current mode
            
        Returns:
            Dictionary of mode settings
        """
        mode = mode or self.current_mode
        return self.mode_settings.get(mode.name, {})
    
    def update_mode_settings(self, mode: TradingMode, settings: Dict[str, Any]) -> bool:
        """
        Update settings for the specified mode.
        
        Args:
            mode: Trading mode to update settings for
            settings: New settings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update settings
            self.mode_settings[mode.name].update(settings)
            
            # Save settings
            self._save_mode_settings(self.mode_settings)
            
            self.logger.info(f"Settings updated for {mode.name}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating mode settings: {e}")
            return False
    
    def get_mode_description(self, mode: Optional[TradingMode] = None) -> str:
        """
        Get description for the specified mode or current mode.
        
        Args:
            mode: Trading mode to get description for, or None for current mode
            
        Returns:
            Mode description
        """
        mode = mode or self.current_mode
        settings = self.get_mode_settings(mode)
        return settings.get("description", "No description available")
    
    def is_trading_allowed(self) -> bool:
        """
        Check if trading is allowed in the current mode.
        
        Returns:
            True if trading is allowed, False otherwise
        """
        return self.current_mode not in [TradingMode.MONITOR_ONLY]
    
    def get_risk_level(self) -> float:
        """
        Get risk level for the current mode.
        
        Returns:
            Risk level as a decimal (0.01 = 1%)
        """
        settings = self.get_mode_settings()
        return settings.get("risk_level", 0.01)
    
    def get_max_position_size(self) -> float:
        """
        Get maximum position size for the current mode.
        
        Returns:
            Maximum position size as a decimal (0.1 = 10%)
        """
        settings = self.get_mode_settings()
        return settings.get("max_position_size", 0.05)
    
    def get_max_open_positions(self) -> int:
        """
        Get maximum number of open positions for the current mode.
        
        Returns:
            Maximum number of open positions
        """
        settings = self.get_mode_settings()
        return settings.get("max_open_positions", 3)
    
    def should_use_stop_loss(self) -> bool:
        """
        Check if stop loss should be used in the current mode.
        
        Returns:
            True if stop loss should be used, False otherwise
        """
        settings = self.get_mode_settings()
        return settings.get("use_stop_loss", True)
    
    def should_use_take_profit(self) -> bool:
        """
        Check if take profit should be used in the current mode.
        
        Returns:
            True if take profit should be used, False otherwise
        """
        settings = self.get_mode_settings()
        return settings.get("use_take_profit", True)
    
    def get_tp_multiplier(self) -> float:
        """
        Get take profit multiplier for the current mode.
        
        Returns:
            Take profit multiplier
        """
        settings = self.get_mode_settings()
        return settings.get("tp_multiplier", 2.0)
    
    def get_sl_multiplier(self) -> float:
        """
        Get stop loss multiplier for the current mode.
        
        Returns:
            Stop loss multiplier
        """
        settings = self.get_mode_settings()
        return settings.get("sl_multiplier", 1.0)
    
    def should_use_trailing_stop(self) -> bool:
        """
        Check if trailing stop should be used in the current mode.
        
        Returns:
            True if trailing stop should be used, False otherwise
        """
        settings = self.get_mode_settings()
        return settings.get("use_trailing_stop", True)
