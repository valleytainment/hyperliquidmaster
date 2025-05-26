"""
Risk management module for the hyperliquidmaster package.

This module provides risk management functionality including position sizing,
drawdown protection, and circuit breakers.
"""

from typing import Dict, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta

from hyperliquidmaster.config import BotSettings

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Risk management system for trading operations.
    
    This class handles position sizing, drawdown protection, circuit breakers,
    and other risk management functionality.
    """
    
    def __init__(self, settings: BotSettings):
        """
        Initialize the risk manager.
        
        Args:
            settings: Bot configuration settings
        """
        self.settings = settings
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.circuit_breaker_triggered = False
        self.last_reset = datetime.now()
        self.trades_history = []
    
    def calculate_position_size(self, 
                               account_size: float, 
                               entry_price: float, 
                               stop_loss_price: Optional[float] = None,
                               risk_override: Optional[float] = None) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            account_size: Current account size in USD
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price (optional)
            risk_override: Override default risk percentage (optional)
            
        Returns:
            Position size in contract units
        """
        # Use provided risk or default from settings
        risk_percent = risk_override if risk_override is not None else self.settings.risk_percent
        
        # Calculate risk amount
        risk_amount = account_size * risk_percent
        
        # If stop loss is provided, calculate position size based on stop distance
        if stop_loss_price and stop_loss_price > 0:
            # Calculate stop distance as percentage
            if entry_price > stop_loss_price:  # Long position
                stop_distance_pct = (entry_price - stop_loss_price) / entry_price
            else:  # Short position
                stop_distance_pct = (stop_loss_price - entry_price) / entry_price
            
            # Avoid division by zero
            if stop_distance_pct <= 0:
                logger.warning("Invalid stop distance. Using default risk calculation.")
                position_size = risk_amount / (entry_price * 0.01)  # Default 1% risk
            else:
                position_size = risk_amount / (entry_price * stop_distance_pct)
        else:
            # Default calculation using a standard 1% risk per trade
            position_size = risk_amount / (entry_price * 0.01)
        
        return position_size
    
    def check_drawdown_limit(self, current_equity: float) -> bool:
        """
        Check if current drawdown exceeds the maximum allowed threshold.
        
        Args:
            current_equity: Current account equity
            
        Returns:
            True if trading should continue, False if drawdown limit exceeded
        """
        self.current_equity = current_equity
        
        # Update peak equity if current equity is higher
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Calculate current drawdown
        if self.peak_equity > 0:
            current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # Check if drawdown exceeds threshold
            if current_drawdown > self.settings.max_drawdown_threshold:
                logger.warning(f"Maximum drawdown threshold exceeded: {current_drawdown:.2%}")
                return False
        
        return True
    
    def update_trade_result(self, pnl: float) -> bool:
        """
        Update trade history and check circuit breaker conditions.
        
        Args:
            pnl: Profit/loss from the trade
            
        Returns:
            True if trading should continue, False if circuit breaker triggered
        """
        # Update daily PnL
        self.daily_pnl += pnl
        
        # Update consecutive losses counter
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Add trade to history
        self.trades_history.append({
            'timestamp': datetime.now(),
            'pnl': pnl
        })
        
        # Check circuit breaker conditions
        if self.consecutive_losses >= self.settings.max_consecutive_losses:
            logger.warning(f"Circuit breaker triggered: {self.consecutive_losses} consecutive losses")
            self.circuit_breaker_triggered = True
            return False
        
        # Check daily loss limit
        if self.daily_pnl < -self.settings.circuit_breaker_threshold * self.peak_equity:
            logger.warning(f"Daily loss limit exceeded: {self.daily_pnl:.2f}")
            self.circuit_breaker_triggered = True
            return False
        
        return True
    
    def reset_daily_metrics(self) -> None:
        """Reset daily metrics if a new day has started."""
        now = datetime.now()
        if now.date() > self.last_reset.date():
            logger.info("Resetting daily risk metrics")
            self.daily_pnl = 0.0
            self.last_reset = now
            self.circuit_breaker_triggered = False
    
    def can_trade(self) -> bool:
        """
        Check if trading is allowed based on all risk parameters.
        
        Returns:
            True if trading is allowed, False otherwise
        """
        self.reset_daily_metrics()
        
        if self.circuit_breaker_triggered:
            logger.warning("Trading halted: Circuit breaker is active")
            return False
        
        return True
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get current risk metrics.
        
        Returns:
            Dictionary of risk metrics
        """
        return {
            'consecutive_losses': self.consecutive_losses,
            'daily_pnl': self.daily_pnl,
            'max_drawdown': self.max_drawdown,
            'peak_equity': self.peak_equity,
            'current_equity': self.current_equity,
            'circuit_breaker_triggered': self.circuit_breaker_triggered
        }
