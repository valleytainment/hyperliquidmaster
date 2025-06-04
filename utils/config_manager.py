"""
Configuration management for the trading bot
"""

import json
import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, asdict
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradingConfig:
    """Trading configuration parameters"""
    # Account settings
    wallet_address: str = ""
    testnet: bool = False
    
    # Risk management
    max_position_size: float = 1000.0  # USD
    max_leverage: int = 10
    stop_loss_percentage: float = 2.0  # %
    take_profit_percentage: float = 4.0  # %
    max_daily_loss: float = 500.0  # USD
    max_drawdown: float = 10.0  # %
    
    # Trading parameters
    default_order_size: float = 100.0  # USD
    slippage_tolerance: float = 0.1  # %
    min_profit_threshold: float = 0.5  # %
    
    # Strategy settings
    active_strategies: list = None
    strategy_allocation: dict = None
    
    # API settings
    rate_limit_requests_per_second: float = 10.0
    websocket_reconnect_attempts: int = 5
    
    # Logging
    log_level: str = "INFO"
    log_trades: bool = True
    log_performance: bool = True
    
    def __post_init__(self):
        if self.active_strategies is None:
            self.active_strategies = ["bb_rsi_adx", "hull_suite"]
        if self.strategy_allocation is None:
            self.strategy_allocation = {
                "bb_rsi_adx": 0.6,
                "hull_suite": 0.4
            }


@dataclass
class StrategyConfig:
    """Base strategy configuration"""
    enabled: bool = True
    allocation: float = 1.0  # Percentage of capital to allocate
    max_positions: int = 3
    timeframe: str = "15m"
    
    # Risk parameters
    position_size: float = 100.0  # USD
    stop_loss: float = 2.0  # %
    take_profit: float = 4.0  # %
    
    # Technical indicators
    indicators: dict = None
    
    def __post_init__(self):
        if self.indicators is None:
            self.indicators = {}


@dataclass
class BBRSIADXConfig(StrategyConfig):
    """Bollinger Bands + RSI + ADX strategy configuration"""
    
    def __post_init__(self):
        super().__post_init__()
        if not self.indicators:
            self.indicators = {
                "bollinger_bands": {
                    "period": 20,
                    "std_dev": 2.0
                },
                "rsi": {
                    "period": 14,
                    "overbought": 75,
                    "oversold": 25
                },
                "adx": {
                    "period": 14,
                    "threshold": 25
                }
            }


@dataclass
class HullSuiteConfig(StrategyConfig):
    """Hull Suite strategy configuration"""
    
    def __post_init__(self):
        super().__post_init__()
        if not self.indicators:
            self.indicators = {
                "hull_ma": {
                    "period": 34,
                    "source": "close"
                },
                "atr": {
                    "period": 14,
                    "multiplier": 2.0
                }
            }


class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else Path("config/config.yaml")
        self.config_dir = self.config_path.parent
        self.config_dir.mkdir(exist_ok=True)
        
        # Configuration data
        self._config_data = {}
        self._trading_config = None
        self._strategy_configs = {}
        
        # Load configuration
        self.load_config()
        
        logger.info(f"Configuration manager initialized with {self.config_path}")
    
    def load_config(self) -> None:
        """Load configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    if self.config_path.suffix.lower() == '.json':
                        self._config_data = json.load(f)
                    else:  # YAML
                        self._config_data = yaml.safe_load(f) or {}
                
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.warning(f"Configuration file not found: {self.config_path}")
                self._config_data = {}
                self.create_default_config()
            
            # Parse configurations
            self._parse_configurations()
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self._config_data = {}
            self.create_default_config()
    
    def save_config(self) -> None:
        """Save current configuration to file"""
        try:
            # Update config data with current objects
            self._update_config_data()
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                if self.config_path.suffix.lower() == '.json':
                    json.dump(self._config_data, f, indent=2)
                else:  # YAML
                    yaml.dump(self._config_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def create_default_config(self) -> None:
        """Create default configuration"""
        self._trading_config = TradingConfig()
        self._strategy_configs = {
            "bb_rsi_adx": BBRSIADXConfig(),
            "hull_suite": HullSuiteConfig()
        }
        
        self._update_config_data()
        self.save_config()
        
        logger.info("Default configuration created")
    
    def _parse_configurations(self) -> None:
        """Parse configuration data into objects"""
        # Parse trading config
        trading_data = self._config_data.get('trading', {})
        self._trading_config = TradingConfig(**trading_data)
        
        # Parse strategy configs
        strategies_data = self._config_data.get('strategies', {})
        self._strategy_configs = {}
        
        for strategy_name, strategy_data in strategies_data.items():
            if strategy_name == "bb_rsi_adx":
                self._strategy_configs[strategy_name] = BBRSIADXConfig(**strategy_data)
            elif strategy_name == "hull_suite":
                self._strategy_configs[strategy_name] = HullSuiteConfig(**strategy_data)
            else:
                self._strategy_configs[strategy_name] = StrategyConfig(**strategy_data)
    
    def _update_config_data(self) -> None:
        """Update config data from objects"""
        self._config_data['trading'] = asdict(self._trading_config) if self._trading_config else {}
        self._config_data['strategies'] = {
            name: asdict(config) for name, config in self._strategy_configs.items()
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = self._config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key"""
        keys = key.split('.')
        config = self._config_data
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        
        # Re-parse configurations
        self._parse_configurations()
        
        logger.info(f"Configuration updated: {key} = {value}")
    
    @property
    def trading(self) -> TradingConfig:
        """Get trading configuration"""
        return self._trading_config
    
    @trading.setter
    def trading(self, config: TradingConfig) -> None:
        """Set trading configuration"""
        self._trading_config = config
        self._update_config_data()
    
    def get_strategy_config(self, strategy_name: str) -> Optional[StrategyConfig]:
        """Get strategy configuration by name"""
        return self._strategy_configs.get(strategy_name)
    
    def set_strategy_config(self, strategy_name: str, config: StrategyConfig) -> None:
        """Set strategy configuration"""
        self._strategy_configs[strategy_name] = config
        self._update_config_data()
    
    def get_all_strategy_configs(self) -> Dict[str, StrategyConfig]:
        """Get all strategy configurations"""
        return self._strategy_configs.copy()
    
    def update_strategy_parameter(self, strategy_name: str, parameter: str, value: Any) -> None:
        """Update a specific strategy parameter"""
        if strategy_name in self._strategy_configs:
            config = self._strategy_configs[strategy_name]
            
            # Handle nested parameters (e.g., "indicators.rsi.period")
            if '.' in parameter:
                keys = parameter.split('.')
                obj = config
                
                for key in keys[:-1]:
                    if hasattr(obj, key):
                        obj = getattr(obj, key)
                    else:
                        return
                
                if isinstance(obj, dict):
                    obj[keys[-1]] = value
                else:
                    setattr(obj, keys[-1], value)
            else:
                setattr(config, parameter, value)
            
            self._update_config_data()
            logger.info(f"Strategy parameter updated: {strategy_name}.{parameter} = {value}")
    
    def validate_config(self) -> bool:
        """Validate configuration parameters"""
        try:
            # Validate trading config
            if not self._trading_config:
                logger.error("Trading configuration is missing")
                return False
            
            # Check required fields
            if not self._trading_config.wallet_address:
                logger.warning("Wallet address not configured")
            
            # Validate risk parameters
            if self._trading_config.max_leverage > 50:
                logger.warning("Very high leverage configured")
            
            if self._trading_config.stop_loss_percentage > 10:
                logger.warning("Very high stop loss percentage")
            
            # Validate strategy allocations
            total_allocation = sum(self._trading_config.strategy_allocation.values())
            if abs(total_allocation - 1.0) > 0.01:
                logger.warning(f"Strategy allocations don't sum to 1.0: {total_allocation}")
            
            # Validate strategy configs
            for name, config in self._strategy_configs.items():
                if config.allocation < 0 or config.allocation > 1:
                    logger.warning(f"Invalid allocation for strategy {name}: {config.allocation}")
            
            logger.info("Configuration validation completed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def export_config(self, export_path: str) -> None:
        """Export configuration to a different file"""
        try:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                if export_path.suffix.lower() == '.json':
                    json.dump(self._config_data, f, indent=2)
                else:  # YAML
                    yaml.dump(self._config_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration exported to {export_path}")
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
    
    def import_config(self, import_path: str) -> None:
        """Import configuration from a different file"""
        try:
            import_path = Path(import_path)
            
            if not import_path.exists():
                logger.error(f"Import file not found: {import_path}")
                return
            
            with open(import_path, 'r', encoding='utf-8') as f:
                if import_path.suffix.lower() == '.json':
                    imported_data = json.load(f)
                else:  # YAML
                    imported_data = yaml.safe_load(f)
            
            # Merge with existing config
            self._config_data.update(imported_data)
            self._parse_configurations()
            self.save_config()
            
            logger.info(f"Configuration imported from {import_path}")
            
        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults"""
        self.create_default_config()
        logger.info("Configuration reset to defaults")
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"ConfigManager(path={self.config_path}, strategies={list(self._strategy_configs.keys())})"


    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values
        
        Args:
            updates: Dictionary of configuration updates
        """
        try:
            self._config_data.update(updates)
            logger.info(f"Configuration updated with {len(updates)} changes")
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            raise

