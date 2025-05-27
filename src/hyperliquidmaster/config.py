"""
Configuration module for the hyperliquidmaster package.

This module provides Pydantic models for configuration validation and loading.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class BotSettings(BaseModel):
    """
    Pydantic model for validating and storing bot configuration.
    
    This model ensures all required configuration parameters are present
    and provides sensible defaults where appropriate.
    """
    # Account settings
    account_address: str = Field(
        default_factory=lambda: os.environ.get("HYPERLIQUID_ACCOUNT_ADDRESS", ""),
        description="Hyperliquid account address"
    )
    secret_key: str = Field(
        default_factory=lambda: os.environ.get("HYPERLIQUID_SECRET_KEY", ""),
        description="Secret key for API authentication"
    )
    
    # Trading symbols
    symbols: List[str] = Field(
        default_factory=lambda: ["BTC", "ETH", "SOL"],
        description="List of trading symbols"
    )
    trade_symbol: str = Field(
        default="BTC",
        description="Primary trading symbol"
    )
    
    # Strategy settings
    use_sentiment_analysis: bool = Field(
        default=True,
        description="Whether to use sentiment analysis"
    )
    use_triple_confluence_strategy: bool = Field(
        default=True,
        description="Whether to use triple confluence strategy"
    )
    use_oracle_update_strategy: bool = Field(
        default=True,
        description="Whether to use oracle update strategy"
    )
    
    # Risk management
    risk_percent: float = Field(
        default=0.01,
        description="Risk percentage per trade"
    )
    max_drawdown_threshold: float = Field(
        default=0.07,
        description="Maximum drawdown threshold"
    )
    circuit_breaker_threshold: float = Field(
        default=0.05,
        description="Circuit breaker threshold"
    )
    max_consecutive_losses: int = Field(
        default=3,
        description="Maximum consecutive losses before pausing"
    )
    
    # Trading parameters
    min_order_imbalance: float = Field(
        default=1.3,
        description="Minimum order imbalance for signal"
    )
    funding_threshold: float = Field(
        default=0.00001,
        description="Funding rate threshold"
    )
    min_price_deviation: float = Field(
        default=0.0015,
        description="Minimum price deviation for signal"
    )
    
    # Technical indicators
    vwma_fast_period: int = Field(
        default=20,
        description="VWMA fast period"
    )
    vwma_slow_period: int = Field(
        default=50,
        description="VWMA slow period"
    )
    oracle_update_interval: int = Field(
        default=3,
        description="Oracle update interval in seconds"
    )
    max_trade_duration: int = Field(
        default=30,
        description="Maximum trade duration in minutes"
    )
    
    # Update intervals
    data_update_interval: int = Field(
        default=5,
        description="Data update interval in seconds"
    )
    sentiment_update_interval: int = Field(
        default=300,
        description="Sentiment update interval in seconds"
    )
    signal_confidence_threshold: float = Field(
        default=0.7,
        description="Signal confidence threshold"
    )
    
    # Fee structure
    taker_fee: float = Field(
        default=0.00042,
        description="Taker fee"
    )
    maker_fee: float = Field(
        default=0.00021,
        description="Maker fee"
    )
    
    # API settings
    base_url: str = Field(
        default="https://api.hyperliquid.xyz",
        description="Base URL for Hyperliquid API"
    )
    
    @validator("account_address")
    def validate_account_address(cls, v):
        """Validate account address format."""
        if not v:
            raise ValueError("Account address is required")
        if not v.startswith("0x"):
            raise ValueError("Account address must start with '0x'")
        return v
    
    @validator("secret_key")
    def validate_secret_key(cls, v):
        """Validate secret key is provided."""
        if not v:
            raise ValueError("Secret key is required")
        return v

def load_config(config_path: Union[str, Path] = None) -> BotSettings:
    """
    Load configuration from file and/or environment variables.
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        Validated BotSettings object
    """
    import json
    import yaml
    
    config_data = {}
    
    # Load from file if provided
    if config_path:
        path = Path(config_path)
        if path.exists():
            if path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'r') as f:
                    config_data = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                with open(path, 'r') as f:
                    config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
        else:
            raise FileNotFoundError(f"Config file not found: {path}")
    
    # Create and validate settings
    try:
        settings = BotSettings(**config_data)
        return settings
    except Exception as e:
        raise ValueError(f"Invalid configuration: {str(e)}")
