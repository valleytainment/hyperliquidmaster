"""
Risk Management for Hyperliquid Trading Bot
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from utils.logger import get_logger
from strategies.trading_types_fixed import TradingSignal, SignalType, MarketData

logger = get_logger(__name__)


@dataclass
class RiskLimits:
    """
    Risk limits configuration
    """
    max_portfolio_risk: float = 0.02  # 2% max portfolio risk
    max_daily_loss: float = 0.05      # 5% max daily loss
    max_drawdown: float = 0.10        # 10% max drawdown
    max_leverage: float = 3.0         # 3x max leverage
    max_position_size: float = 0.10   # 10% max position size


class RiskManager:
    """
    Risk Management for Hyperliquid Trading Bot
    """
    
    def __init__(self, risk_limits=None, api=None):
        """
        Initialize the risk manager
        
        Args:
            risk_limits: Risk limits configuration
            api: API instance
        """
        self.risk_limits = risk_limits or RiskLimits()
        self.api = api
        
        # Initialize state
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.max_drawdown_reached = 0.0
        self.positions = {}
        self.orders = {}
        
        logger.info("Risk management system initialized")
    
    def check_signal(self, signal):
        """
        Check if a trading signal is within risk limits
        
        Args:
            signal: Trading signal to check
        
        Returns:
            bool: True if signal is acceptable, False otherwise
        """
        try:
            if not signal:
                return False
            
            # Check daily loss limit
            if self.daily_pnl <= -self.risk_limits.max_daily_loss:
                logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2%}")
                return False
            
            # Check max drawdown
            if self.max_drawdown_reached >= self.risk_limits.max_drawdown:
                logger.warning(f"Max drawdown reached: {self.max_drawdown_reached:.2%}")
                return False
            
            # Check position size
            if hasattr(signal, 'size') and signal.size:
                if signal.size > self.risk_limits.max_position_size:
                    logger.warning(f"Position size too large: {signal.size:.2%} > {self.risk_limits.max_position_size:.2%}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Failed to check signal: {e}")
            return False
    
    def check_order(self, coin, side, size, price):
        """
        Check if an order is within risk limits
        
        Args:
            coin: Coin to trade
            side: Order side (buy or sell)
            size: Order size
            price: Order price
        
        Returns:
            bool: True if order is acceptable, False otherwise
        """
        try:
            # Check daily loss limit
            if self.daily_pnl <= -self.risk_limits.max_daily_loss:
                logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2%}")
                return False
            
            # Check position size
            position_value = size * price if price else size
            if position_value > self.risk_limits.max_position_size:
                logger.warning(f"Position size too large: {position_value:.2%}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Failed to check order: {e}")
            return False
    
    def adjust_position_size(self, coin, side, size, price):
        """
        Adjust position size based on risk limits
        
        Args:
            coin: Coin to trade
            side: Order side (buy or sell)
            size: Requested position size
            price: Order price
        
        Returns:
            float: Adjusted position size
        """
        try:
            # Calculate maximum allowed position size
            max_size = self.risk_limits.max_position_size
            
            # Adjust size if necessary
            if size > max_size:
                logger.info(f"Adjusting position size from {size:.4f} to {max_size:.4f}")
                return max_size
            
            return size
        except Exception as e:
            logger.error(f"Failed to adjust position size: {e}")
            return size
    
    def update_pnl(self, pnl):
        """
        Update PnL tracking
        
        Args:
            pnl: Profit/loss amount
        """
        try:
            self.daily_pnl += pnl
            self.total_pnl += pnl
            
            # Update max drawdown
            if pnl < 0:
                drawdown = abs(pnl)
                if drawdown > self.max_drawdown_reached:
                    self.max_drawdown_reached = drawdown
            
            logger.debug(f"Updated PnL: daily={self.daily_pnl:.2f}, total={self.total_pnl:.2f}")
        except Exception as e:
            logger.error(f"Failed to update PnL: {e}")
    
    def reset_daily_pnl(self):
        """
        Reset daily PnL (call at start of each trading day)
        """
        try:
            self.daily_pnl = 0.0
            logger.info("Daily PnL reset")
        except Exception as e:
            logger.error(f"Failed to reset daily PnL: {e}")
    
    def get_risk_status(self):
        """
        Get current risk status
        
        Returns:
            dict: Risk status information
        """
        try:
            return {
                "daily_pnl": self.daily_pnl,
                "total_pnl": self.total_pnl,
                "max_drawdown_reached": self.max_drawdown_reached,
                "daily_loss_limit": self.risk_limits.max_daily_loss,
                "max_drawdown_limit": self.risk_limits.max_drawdown,
                "max_position_size": self.risk_limits.max_position_size,
                "max_leverage": self.risk_limits.max_leverage
            }
        except Exception as e:
            logger.error(f"Failed to get risk status: {e}")
            return {}
    
    def is_trading_allowed(self):
        """
        Check if trading is allowed based on current risk status
        
        Returns:
            bool: True if trading is allowed, False otherwise
        """
        try:
            # Check daily loss limit
            if self.daily_pnl <= -self.risk_limits.max_daily_loss:
                return False
            
            # Check max drawdown
            if self.max_drawdown_reached >= self.risk_limits.max_drawdown:
                return False
            
            return True
        except Exception as e:
            logger.error(f"Failed to check if trading is allowed: {e}")
            return False

