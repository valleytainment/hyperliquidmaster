"""
Risk Management for Hyperliquid Trading Bot
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List

from utils.logger import get_logger
from strategies.trading_types import TradingSignal, SignalType, MarketData

logger = get_logger(__name__)


class RiskManager:
    """
    Risk Management for Hyperliquid Trading Bot
    """
    
    def __init__(self, max_position_size=1.0, max_leverage=5.0, max_drawdown=0.1):
        """
        Initialize the risk manager
        
        Args:
            max_position_size: Maximum position size as a fraction of account value
            max_leverage: Maximum leverage
            max_drawdown: Maximum drawdown as a fraction of account value
        """
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.max_drawdown = max_drawdown
        
        # Initialize state
        self.account_value = 0.0
        self.positions = {}
        self.drawdown = 0.0
        self.peak_account_value = 0.0
        
        logger.info("Risk management system initialized")
    
    def update_account_state(self, account_value, positions):
        """
        Update account state
        
        Args:
            account_value: Current account value
            positions: Current positions
        """
        self.account_value = account_value
        self.positions = positions
        
        # Update peak account value
        if account_value > self.peak_account_value:
            self.peak_account_value = account_value
        
        # Update drawdown
        if self.peak_account_value > 0:
            self.drawdown = 1.0 - (account_value / self.peak_account_value)
        
        logger.debug(f"Account state updated: value={account_value}, drawdown={self.drawdown:.2f}")
    
    def check_signal(self, signal):
        """
        Check if a trading signal is within risk limits
        
        Args:
            signal: Trading signal
        
        Returns:
            bool: True if signal is within risk limits, False otherwise
        """
        try:
            # Check if we have account value
            if self.account_value <= 0:
                logger.warning("No account value available")
                return False
            
            # Check drawdown
            if self.drawdown >= self.max_drawdown:
                logger.warning(f"Maximum drawdown reached: {self.drawdown:.2f} >= {self.max_drawdown:.2f}")
                return False
            
            # Check position size
            if signal.size is not None:
                position_value = signal.size * signal.price if signal.price is not None else 0.0
                position_fraction = position_value / self.account_value
                
                if position_fraction > self.max_position_size:
                    logger.warning(f"Position size too large: {position_fraction:.2f} > {self.max_position_size:.2f}")
                    return False
            
            # Check leverage (placeholder)
            # This would require more sophisticated calculation
            
            return True
        except Exception as e:
            logger.error(f"Failed to check signal: {e}")
            return False
    
    def check_order(self, coin, side, size, price=None):
        """
        Check if an order is within risk limits
        
        Args:
            coin: Coin to trade
            side: Order side (buy or sell)
            size: Order size
            price: Order price (optional)
        
        Returns:
            bool: True if order is within risk limits, False otherwise
        """
        try:
            # Create signal from order
            signal_type = SignalType.BUY if side.lower() == "buy" else SignalType.SELL
            signal = TradingSignal(coin, signal_type, 1.0, price, size)
            
            # Check signal
            return self.check_signal(signal)
        except Exception as e:
            logger.error(f"Failed to check order: {e}")
            return False
    
    def adjust_position_size(self, coin, side, size, price=None):
        """
        Adjust position size to be within risk limits
        
        Args:
            coin: Coin to trade
            side: Order side (buy or sell)
            size: Requested position size
            price: Order price (optional)
        
        Returns:
            float: Adjusted position size
        """
        try:
            # Check if we have account value
            if self.account_value <= 0:
                logger.warning("No account value available")
                return 0.0
            
            # Calculate position value
            position_value = size * price if price is not None else 0.0
            
            # Calculate maximum position value
            max_position_value = self.account_value * self.max_position_size
            
            # Adjust position size if necessary
            if position_value > max_position_value:
                adjusted_size = max_position_value / price if price is not None and price > 0 else 0.0
                logger.info(f"Adjusted position size from {size} to {adjusted_size}")
                return adjusted_size
            
            return size
        except Exception as e:
            logger.error(f"Failed to adjust position size: {e}")
            return 0.0
    
    def get_risk_metrics(self):
        """
        Get risk metrics
        
        Returns:
            dict: Risk metrics
        """
        return {
            "account_value": self.account_value,
            "peak_account_value": self.peak_account_value,
            "drawdown": self.drawdown,
            "max_drawdown": self.max_drawdown,
            "max_position_size": self.max_position_size,
            "max_leverage": self.max_leverage
        }

