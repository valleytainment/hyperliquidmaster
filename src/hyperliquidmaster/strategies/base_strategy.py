"""
Base strategy interface for the hyperliquidmaster package.

This module provides the base strategy interface that all trading strategies should implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

from hyperliquidmaster.config import BotSettings

class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    
    This abstract class defines the interface that all trading strategies must implement.
    """
    
    def __init__(self, settings: BotSettings, logger: Optional[logging.Logger] = None):
        """
        Initialize the strategy.
        
        Args:
            settings: Bot configuration settings
            logger: Logger instance (optional)
        """
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)
        self.name = self.__class__.__name__
        
    @abstractmethod
    def generate_signal(self, 
                       symbol: str, 
                       market_data: Dict[str, Any], 
                       order_book: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate trading signal based on market data.
        
        Args:
            symbol: Trading symbol
            market_data: Market data dictionary
            order_book: Order book data (optional)
            
        Returns:
            Signal dictionary with action, strength, entry price, etc.
        """
        pass
    
    @abstractmethod
    def update_state(self, symbol: str, market_data: Dict[str, Any]) -> None:
        """
        Update strategy state with new market data.
        
        Args:
            symbol: Trading symbol
            market_data: Market data dictionary
        """
        pass
    
    def get_name(self) -> str:
        """
        Get strategy name.
        
        Returns:
            Strategy name
        """
        return self.name
    
    def get_description(self) -> str:
        """
        Get strategy description.
        
        Returns:
            Strategy description
        """
        return "Base strategy interface"
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get strategy parameters.
        
        Returns:
            Dictionary of strategy parameters
        """
        return {}
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set strategy parameters.
        
        Args:
            parameters: Dictionary of strategy parameters
        """
        pass
