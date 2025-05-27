"""
Enhanced Risk Management Module for HyperliquidMaster
----------------------------------------------------
Implements advanced risk management techniques based on expert recommendations
including volatility-based position sizing, strict risk percentage rules,
risk-reward ratio validation, and liquidation proximity warnings.
"""

import logging
import math
import time
from typing import Dict, Any, Optional, Tuple, List, Union

class EnhancedRiskManager:
    """
    Advanced risk management system for crypto derivatives trading.
    Implements the 1% rule, volatility-based position sizing, risk-reward ratio validation,
    and liquidation proximity warnings.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the risk manager.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger("EnhancedRiskManager")
        
        # Default risk parameters
        self.max_risk_percentage = 1.0  # Default 1% rule
        self.min_risk_reward_ratio = 2.0  # Default 1:2 risk-reward ratio
        self.liquidation_warning_threshold = 15.0  # Percentage distance for warning
        self.critical_warning_threshold = 7.5  # Percentage distance for critical warning
        self.max_leverage = 10.0  # Default maximum leverage
        
        # Volatility settings
        self.use_volatility_adjustment = True
        self.volatility_lookback_periods = 14  # For ATR calculation
        self.volatility_multiplier = 1.5  # Adjustment factor for volatility
        
        # Drawdown protection
        self.max_daily_drawdown = 5.0  # Maximum allowed daily drawdown percentage
        self.max_total_drawdown = 15.0  # Maximum allowed total drawdown percentage
        self.pause_trading_on_max_drawdown = True
        
        # Trading streak protection
        self.winning_streak_threshold = 5  # Number of consecutive wins that triggers warning
        self.losing_streak_threshold = 3  # Number of consecutive losses that triggers warning
        
        # Trading history
        self.trade_history = []
        self.current_streak = {"type": None, "count": 0}
        
        # Performance tracking
        self.initial_equity = 0.0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.daily_starting_equity = 0.0
        self.daily_low_equity = 0.0
        self.trading_paused = False
        
        self.logger.info("Enhanced Risk Manager initialized with 1% risk rule and volatility-based sizing")
    
    def set_account_equity(self, equity: float) -> None:
        """
        Set the current account equity and update tracking metrics.
        
        Args:
            equity: Current account equity
        """
        self.current_equity = equity
        
        # Initialize values if this is the first update
        if self.initial_equity == 0.0:
            self.initial_equity = equity
            self.peak_equity = equity
            self.daily_starting_equity = equity
            self.daily_low_equity = equity
        
        # Update peak equity if current equity is higher
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # Update daily low equity if current equity is lower
        if equity < self.daily_low_equity:
            self.daily_low_equity = equity
        
        # Check for drawdown limits
        self._check_drawdown_limits()
    
    def reset_daily_metrics(self) -> None:
        """Reset daily tracking metrics."""
        self.daily_starting_equity = self.current_equity
        self.daily_low_equity = self.current_equity
    
    def _check_drawdown_limits(self) -> bool:
        """
        Check if drawdown limits have been exceeded.
        
        Returns:
            True if trading should continue, False if it should be paused
        """
        # Calculate drawdowns
        total_drawdown = 0.0
        if self.peak_equity > 0:
            total_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity * 100
        
        daily_drawdown = 0.0
        if self.daily_starting_equity > 0:
            daily_drawdown = (self.daily_starting_equity - self.daily_low_equity) / self.daily_starting_equity * 100
        
        # Log drawdown status
        self.logger.info(f"Current drawdown - Daily: {daily_drawdown:.2f}%, Total: {total_drawdown:.2f}%")
        
        # Check if limits are exceeded
        if daily_drawdown > self.max_daily_drawdown:
            self.logger.warning(f"Daily drawdown limit exceeded: {daily_drawdown:.2f}% > {self.max_daily_drawdown:.2f}%")
            if self.pause_trading_on_max_drawdown:
                self.trading_paused = True
                self.logger.warning("Trading paused due to excessive daily drawdown")
                return False
        
        if total_drawdown > self.max_total_drawdown:
            self.logger.warning(f"Total drawdown limit exceeded: {total_drawdown:.2f}% > {self.max_total_drawdown:.2f}%")
            if self.pause_trading_on_max_drawdown:
                self.trading_paused = True
                self.logger.warning("Trading paused due to excessive total drawdown")
                return False
        
        return True
    
    def resume_trading(self) -> None:
        """Resume trading after a pause."""
        self.trading_paused = False
        self.logger.info("Trading resumed")
    
    def calculate_position_size(self, 
                               account_equity: float, 
                               entry_price: float, 
                               stop_loss_price: float, 
                               volatility_factor: Optional[float] = None,
                               risk_percentage: Optional[float] = None) -> float:
        """
        Calculate position size based on account equity, risk percentage, and stop loss distance.
        Optionally adjust for volatility using ATR.
        
        Args:
            account_equity: Total account equity
            entry_price: Planned entry price
            stop_loss_price: Planned stop loss price
            volatility_factor: Optional ATR or other volatility measure
            risk_percentage: Optional override for risk percentage
            
        Returns:
            Appropriate position size in contracts/coins
        """
        # Use provided risk percentage or default
        risk_pct = risk_percentage if risk_percentage is not None else self.max_risk_percentage
        
        # Calculate risk amount
        risk_amount = account_equity * (risk_pct / 100)
        
        # Calculate price distance
        price_distance = abs(entry_price - stop_loss_price)
        
        # Ensure price distance is not zero
        if price_distance <= 0:
            self.logger.warning("Invalid price distance (zero or negative). Using minimum distance.")
            price_distance = entry_price * 0.001  # Use 0.1% as minimum distance
        
        # Adjust for volatility if provided and enabled
        if self.use_volatility_adjustment and volatility_factor:
            # Reduce position size for higher volatility
            volatility_ratio = volatility_factor / entry_price
            adjusted_distance = price_distance * (1 + (volatility_ratio * self.volatility_multiplier))
            position_size = risk_amount / adjusted_distance
            self.logger.info(f"Volatility adjustment applied: {volatility_ratio:.4f}, adjusted distance: {adjusted_distance:.4f}")
        else:
            position_size = risk_amount / price_distance
        
        self.logger.info(f"Calculated position size: {position_size:.6f} based on {risk_pct:.1f}% risk")
        return position_size
    
    def validate_risk_reward_ratio(self, 
                                  entry_price: float, 
                                  stop_loss_price: float, 
                                  take_profit_price: float,
                                  min_ratio: Optional[float] = None) -> Tuple[bool, float]:
        """
        Validate if a trade setup meets the minimum risk-reward ratio requirement.
        
        Args:
            entry_price: Planned entry price
            stop_loss_price: Planned stop loss price
            take_profit_price: Planned take profit price
            min_ratio: Optional override for minimum risk-reward ratio
            
        Returns:
            Tuple of (is_valid, actual_ratio)
        """
        # Use provided minimum ratio or default
        min_rrr = min_ratio if min_ratio is not None else self.min_risk_reward_ratio
        
        # Calculate risk and reward
        risk = abs(entry_price - stop_loss_price)
        reward = abs(entry_price - take_profit_price)
        
        # Ensure risk is not zero
        if risk <= 0:
            self.logger.warning("Invalid risk distance (zero or negative). Using minimum distance.")
            risk = entry_price * 0.001  # Use 0.1% as minimum distance
        
        # Calculate ratio
        ratio = reward / risk
        
        # Validate against minimum requirement
        is_valid = ratio >= min_rrr
        
        if is_valid:
            self.logger.info(f"Trade setup meets risk-reward requirement: {ratio:.2f} >= {min_rrr:.2f}")
        else:
            self.logger.warning(f"Trade setup does NOT meet risk-reward requirement: {ratio:.2f} < {min_rrr:.2f}")
        
        return is_valid, ratio
    
    def calculate_liquidation_proximity(self, 
                                       current_price: float, 
                                       liquidation_price: float) -> Tuple[str, str, float]:
        """
        Calculate proximity to liquidation and provide appropriate warnings.
        
        Args:
            current_price: Current market price
            liquidation_price: Calculated liquidation price
            
        Returns:
            Tuple of (warning_level, message, distance_percentage)
        """
        # Calculate distance percentage
        distance_percentage = abs(current_price - liquidation_price) / current_price * 100
        
        # Determine warning level
        if distance_percentage <= self.critical_warning_threshold:
            level = "CRITICAL"
            message = f"EXTREME RISK: Only {distance_percentage:.2f}% from liquidation!"
        elif distance_percentage <= self.liquidation_warning_threshold:
            level = "WARNING"
            message = f"HIGH RISK: {distance_percentage:.2f}% from liquidation"
        else:
            level = "SAFE"
            message = f"Position {distance_percentage:.2f}% from liquidation"
        
        self.logger.info(f"Liquidation proximity: {level} - {distance_percentage:.2f}%")
        return level, message, distance_percentage
    
    def recommend_max_leverage(self, 
                              volatility_factor: Optional[float] = None, 
                              experience_level: str = "intermediate") -> float:
        """
        Recommend maximum leverage based on volatility and experience level.
        
        Args:
            volatility_factor: Optional ATR or other volatility measure
            experience_level: Trader experience level (beginner, intermediate, advanced)
            
        Returns:
            Recommended maximum leverage
        """
        # Base leverage by experience level
        base_leverage = {
            "beginner": 3.0,
            "intermediate": 5.0,
            "advanced": 10.0
        }.get(experience_level.lower(), 5.0)
        
        # Adjust for volatility if provided
        if volatility_factor:
            # Normalize volatility (higher volatility = lower leverage)
            volatility_adjustment = 1.0 / (1.0 + volatility_factor)
            recommended_leverage = base_leverage * volatility_adjustment
            
            # Ensure leverage is within reasonable bounds
            recommended_leverage = max(1.0, min(recommended_leverage, self.max_leverage))
        else:
            recommended_leverage = base_leverage
        
        self.logger.info(f"Recommended max leverage: {recommended_leverage:.1f}x for {experience_level} trader")
        return recommended_leverage
    
    def record_trade_result(self, trade_result: Dict[str, Any]) -> None:
        """
        Record a trade result and update streak tracking.
        
        Args:
            trade_result: Dictionary containing trade details
        """
        # Add timestamp if not present
        if "timestamp" not in trade_result:
            trade_result["timestamp"] = int(time.time())
        
        # Add to history
        self.trade_history.append(trade_result)
        
        # Update streak tracking
        is_win = trade_result.get("is_win", False)
        
        if self.current_streak["type"] is None:
            # First trade
            self.current_streak["type"] = "win" if is_win else "loss"
            self.current_streak["count"] = 1
        elif (self.current_streak["type"] == "win" and is_win) or (self.current_streak["type"] == "loss" and not is_win):
            # Continuing streak
            self.current_streak["count"] += 1
        else:
            # Streak broken
            self.current_streak["type"] = "win" if is_win else "loss"
            self.current_streak["count"] = 1
        
        # Check for streak warnings
        self._check_streak_warnings()
        
        self.logger.info(f"Trade recorded: {'Win' if is_win else 'Loss'}, current streak: {self.current_streak['count']} {self.current_streak['type']}(s)")
    
    def _check_streak_warnings(self) -> None:
        """Check for streak warnings and log appropriate messages."""
        if self.current_streak["type"] == "win" and self.current_streak["count"] >= self.winning_streak_threshold:
            self.logger.warning(f"WINNING STREAK ALERT: {self.current_streak['count']} consecutive wins. "
                               f"Be cautious of overconfidence and maintain strict risk management.")
        
        if self.current_streak["type"] == "loss" and self.current_streak["count"] >= self.losing_streak_threshold:
            self.logger.warning(f"LOSING STREAK ALERT: {self.current_streak['count']} consecutive losses. "
                               f"Consider reducing position size or taking a break.")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate and return performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        # Count wins and losses
        wins = sum(1 for trade in self.trade_history if trade.get("is_win", False))
        losses = len(self.trade_history) - wins
        
        # Calculate win rate
        win_rate = 0.0
        if len(self.trade_history) > 0:
            win_rate = wins / len(self.trade_history) * 100
        
        # Calculate profit factor
        total_profit = sum(trade.get("profit", 0) for trade in self.trade_history if trade.get("profit", 0) > 0)
        total_loss = abs(sum(trade.get("profit", 0) for trade in self.trade_history if trade.get("profit", 0) < 0))
        profit_factor = 0.0
        if total_loss > 0:
            profit_factor = total_profit / total_loss
        
        # Calculate drawdown
        current_drawdown = 0.0
        if self.peak_equity > 0:
            current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity * 100
        
        # Calculate daily drawdown
        daily_drawdown = 0.0
        if self.daily_starting_equity > 0:
            daily_drawdown = (self.daily_starting_equity - self.daily_low_equity) / self.daily_starting_equity * 100
        
        # Calculate equity growth
        equity_growth = 0.0
        if self.initial_equity > 0:
            equity_growth = (self.current_equity - self.initial_equity) / self.initial_equity * 100
        
        metrics = {
            "total_trades": len(self.trade_history),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "current_drawdown": current_drawdown,
            "daily_drawdown": daily_drawdown,
            "equity_growth": equity_growth,
            "current_streak": self.current_streak,
            "trading_paused": self.trading_paused
        }
        
        return metrics
    
    def create_scale_orders(self, 
                           symbol: str, 
                           is_buy: bool, 
                           total_size: float, 
                           price_range: Tuple[float, float], 
                           num_orders: int) -> List[Dict[str, Any]]:
        """
        Create multiple limit orders distributed across a price range.
        
        Args:
            symbol: Trading pair symbol
            is_buy: True for buy orders, False for sell orders
            total_size: Total position size to be distributed
            price_range: Tuple of (start_price, end_price)
            num_orders: Number of orders to create
            
        Returns:
            List of order details
        """
        start_price, end_price = price_range
        
        # Ensure start_price < end_price for buy orders and start_price > end_price for sell orders
        if is_buy and start_price > end_price:
            start_price, end_price = end_price, start_price
        elif not is_buy and start_price < end_price:
            start_price, end_price = end_price, start_price
        
        # Calculate price step
        price_step = (end_price - start_price) / (num_orders - 1) if num_orders > 1 else 0
        
        # Calculate size per order
        size_per_order = total_size / num_orders
        
        orders = []
        for i in range(num_orders):
            price = start_price + (i * price_step)
            orders.append({
                "symbol": symbol,
                "is_buy": is_buy,
                "size": size_per_order,
                "price": price,
                "order_type": "LIMIT"
            })
        
        self.logger.info(f"Created {num_orders} scale orders for {symbol}, total size: {total_size}")
        return orders
    
    def should_allow_trade(self) -> Tuple[bool, str]:
        """
        Determine if trading should be allowed based on risk parameters.
        
        Returns:
            Tuple of (is_allowed, reason)
        """
        if self.trading_paused:
            return False, "Trading paused due to excessive drawdown"
        
        # Check drawdown limits
        total_drawdown = 0.0
        if self.peak_equity > 0:
            total_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity * 100
        
        daily_drawdown = 0.0
        if self.daily_starting_equity > 0:
            daily_drawdown = (self.daily_starting_equity - self.daily_low_equity) / self.daily_starting_equity * 100
        
        if daily_drawdown > self.max_daily_drawdown:
            return False, f"Daily drawdown limit exceeded: {daily_drawdown:.2f}% > {self.max_daily_drawdown:.2f}%"
        
        if total_drawdown > self.max_total_drawdown:
            return False, f"Total drawdown limit exceeded: {total_drawdown:.2f}% > {self.max_total_drawdown:.2f}%"
        
        # Check losing streak
        if self.current_streak["type"] == "loss" and self.current_streak["count"] >= self.losing_streak_threshold * 2:
            return False, f"Excessive losing streak: {self.current_streak['count']} consecutive losses"
        
        return True, "Trading allowed"
