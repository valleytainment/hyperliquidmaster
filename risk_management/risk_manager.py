"""
Advanced Risk Management System for Hyperliquid Trading Bot
Provides comprehensive risk controls and portfolio protection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from ..strategies.base_strategy import TradingSignal, SignalType
from ..utils.logger import get_logger, TradingLogger

logger = get_logger(__name__)
trading_logger = TradingLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio monitoring"""
    portfolio_value: float
    total_exposure: float
    leverage_ratio: float
    var_1d: float  # 1-day Value at Risk
    var_7d: float  # 7-day Value at Risk
    max_drawdown: float
    sharpe_ratio: float
    correlation_risk: float
    concentration_risk: float
    risk_level: RiskLevel
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'portfolio_value': self.portfolio_value,
            'total_exposure': self.total_exposure,
            'leverage_ratio': self.leverage_ratio,
            'var_1d': self.var_1d,
            'var_7d': self.var_7d,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'correlation_risk': self.correlation_risk,
            'concentration_risk': self.concentration_risk,
            'risk_level': self.risk_level.value
        }


@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_portfolio_risk: float = 0.02  # 2% max portfolio risk per trade
    max_daily_loss: float = 0.05  # 5% max daily loss
    max_drawdown: float = 0.10  # 10% max drawdown
    max_leverage: float = 10.0  # 10x max leverage
    max_position_size: float = 0.20  # 20% max position size
    max_correlation: float = 0.7  # Max correlation between positions
    max_concentration: float = 0.30  # 30% max concentration in single asset
    var_limit: float = 0.03  # 3% VaR limit
    
    # Position limits
    max_positions: int = 10
    max_positions_per_coin: int = 1
    
    # Time-based limits
    max_trades_per_hour: int = 10
    max_trades_per_day: int = 50
    
    # Volatility limits
    max_volatility: float = 0.05  # 5% max daily volatility
    min_liquidity: float = 100000  # $100k min daily volume


class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, risk_limits: RiskLimits = None, api_client=None):
        """
        Initialize risk manager
        
        Args:
            risk_limits: Risk limits configuration
            api_client: API client for portfolio data
        """
        self.risk_limits = risk_limits or RiskLimits()
        self.api_client = api_client
        
        # Risk tracking
        self.portfolio_history = []
        self.trade_history = []
        self.daily_pnl = {}
        self.position_correlations = {}
        
        # Current state
        self.current_positions = {}
        self.current_orders = {}
        self.daily_trades = 0
        self.hourly_trades = 0
        self.last_trade_time = None
        
        # Risk alerts
        self.active_alerts = []
        
        logger.info("Risk management system initialized")
    
    def validate_signal(self, signal: TradingSignal, current_portfolio: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate a trading signal against risk limits
        
        Args:
            signal: Trading signal to validate
            current_portfolio: Current portfolio state
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Check if signal should be processed
            if signal.signal_type == SignalType.NONE:
                return True, "No signal to validate"
            
            # Get current portfolio metrics
            portfolio_value = current_portfolio.get('account_value', 0)
            if portfolio_value <= 0:
                return False, "Invalid portfolio value"
            
            # Entry signal validations
            if signal.signal_type in [SignalType.LONG, SignalType.SHORT]:
                
                # Check maximum positions limit
                active_positions = len([p for p in current_portfolio.get('positions', []) if p.get('size', 0) != 0])
                if active_positions >= self.risk_limits.max_positions:
                    return False, f"Maximum positions limit reached ({self.risk_limits.max_positions})"
                
                # Check position size limit
                position_size = signal.size or 0
                position_percentage = position_size / portfolio_value
                if position_percentage > self.risk_limits.max_position_size:
                    return False, f"Position size exceeds limit ({position_percentage:.2%} > {self.risk_limits.max_position_size:.2%})"
                
                # Check portfolio risk per trade
                if signal.stop_loss and signal.entry_price:
                    risk_per_trade = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
                    portfolio_risk = (position_size / portfolio_value) * risk_per_trade
                    if portfolio_risk > self.risk_limits.max_portfolio_risk:
                        return False, f"Portfolio risk per trade exceeds limit ({portfolio_risk:.2%} > {self.risk_limits.max_portfolio_risk:.2%})"
                
                # Check concentration risk
                coin_exposure = self._calculate_coin_exposure(signal.coin, current_portfolio)
                new_exposure = coin_exposure + position_percentage
                if new_exposure > self.risk_limits.max_concentration:
                    return False, f"Concentration risk in {signal.coin} exceeds limit ({new_exposure:.2%} > {self.risk_limits.max_concentration:.2%})"
                
                # Check correlation risk
                if not self._check_correlation_risk(signal, current_portfolio):
                    return False, "Correlation risk too high with existing positions"
                
                # Check daily trade limits
                if not self._check_trade_frequency_limits():
                    return False, "Daily/hourly trade limits exceeded"
                
                # Check leverage limits
                total_exposure = self._calculate_total_exposure(current_portfolio) + position_size
                leverage = total_exposure / portfolio_value
                if leverage > self.risk_limits.max_leverage:
                    return False, f"Leverage exceeds limit ({leverage:.1f}x > {self.risk_limits.max_leverage:.1f}x)"
                
                # Check volatility limits
                if not self._check_volatility_limits(signal.coin):
                    return False, f"Volatility too high for {signal.coin}"
                
                # Check liquidity requirements
                if not self._check_liquidity_requirements(signal.coin):
                    return False, f"Insufficient liquidity for {signal.coin}"
            
            # Exit signal validations
            elif signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
                # Check if position exists
                position_exists = any(p.get('coin') == signal.coin and p.get('size', 0) != 0 
                                    for p in current_portfolio.get('positions', []))
                if not position_exists:
                    return False, f"No position to close for {signal.coin}"
            
            # Check daily loss limits
            if not self._check_daily_loss_limits(current_portfolio):
                return False, "Daily loss limit exceeded"
            
            # Check drawdown limits
            if not self._check_drawdown_limits(current_portfolio):
                return False, "Maximum drawdown limit exceeded"
            
            return True, "Signal validated successfully"
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False, f"Validation error: {e}"
    
    def _calculate_coin_exposure(self, coin: str, portfolio: Dict[str, Any]) -> float:
        """Calculate current exposure to a specific coin"""
        portfolio_value = portfolio.get('account_value', 1)
        total_exposure = 0
        
        for position in portfolio.get('positions', []):
            if position.get('coin') == coin:
                position_value = abs(position.get('position_value', 0))
                total_exposure += position_value / portfolio_value
        
        return total_exposure
    
    def _calculate_total_exposure(self, portfolio: Dict[str, Any]) -> float:
        """Calculate total portfolio exposure"""
        total_exposure = 0
        
        for position in portfolio.get('positions', []):
            position_value = abs(position.get('position_value', 0))
            total_exposure += position_value
        
        return total_exposure
    
    def _check_correlation_risk(self, signal: TradingSignal, portfolio: Dict[str, Any]) -> bool:
        """Check if new position would create excessive correlation risk"""
        # Simplified correlation check - in practice would use historical price correlations
        coin = signal.coin
        
        # Get list of current positions
        current_coins = [p.get('coin') for p in portfolio.get('positions', []) if p.get('size', 0) != 0]
        
        # Check against known high-correlation pairs
        high_correlation_pairs = [
            ['BTC', 'ETH'],
            ['ETH', 'AVAX'],
            ['SOL', 'AVAX'],
            # Add more correlation pairs as needed
        ]
        
        for pair in high_correlation_pairs:
            if coin in pair:
                other_coin = pair[1] if coin == pair[0] else pair[0]
                if other_coin in current_coins:
                    # Check if combined exposure would be too high
                    combined_exposure = (self._calculate_coin_exposure(coin, portfolio) + 
                                       self._calculate_coin_exposure(other_coin, portfolio))
                    if combined_exposure > self.risk_limits.max_correlation:
                        return False
        
        return True
    
    def _check_trade_frequency_limits(self) -> bool:
        """Check daily and hourly trade frequency limits"""
        current_time = datetime.now()
        
        # Check hourly limit
        if self.last_trade_time:
            if (current_time - self.last_trade_time).total_seconds() < 3600:  # Same hour
                if self.hourly_trades >= self.risk_limits.max_trades_per_hour:
                    return False
            else:
                self.hourly_trades = 0  # Reset hourly counter
        
        # Check daily limit
        if self.daily_trades >= self.risk_limits.max_trades_per_day:
            return False
        
        return True
    
    def _check_volatility_limits(self, coin: str) -> bool:
        """Check if coin volatility is within acceptable limits"""
        # This would typically fetch recent volatility data
        # For now, return True (implement with real volatility calculation)
        return True
    
    def _check_liquidity_requirements(self, coin: str) -> bool:
        """Check if coin has sufficient liquidity"""
        # This would typically check 24h volume
        # For now, return True (implement with real liquidity check)
        return True
    
    def _check_daily_loss_limits(self, portfolio: Dict[str, Any]) -> bool:
        """Check if daily loss limit has been exceeded"""
        today = datetime.now().date()
        daily_pnl = self.daily_pnl.get(today, 0)
        
        portfolio_value = portfolio.get('account_value', 0)
        daily_loss_limit = portfolio_value * self.risk_limits.max_daily_loss
        
        return daily_pnl > -daily_loss_limit
    
    def _check_drawdown_limits(self, portfolio: Dict[str, Any]) -> bool:
        """Check if maximum drawdown limit has been exceeded"""
        if not self.portfolio_history:
            return True
        
        current_value = portfolio.get('account_value', 0)
        peak_value = max(p.get('account_value', 0) for p in self.portfolio_history)
        
        if peak_value > 0:
            drawdown = (peak_value - current_value) / peak_value
            return drawdown <= self.risk_limits.max_drawdown
        
        return True
    
    def calculate_position_size(self, signal: TradingSignal, portfolio: Dict[str, Any]) -> float:
        """
        Calculate optimal position size based on risk management
        
        Args:
            signal: Trading signal
            portfolio: Current portfolio state
            
        Returns:
            Recommended position size in USD
        """
        portfolio_value = portfolio.get('account_value', 0)
        if portfolio_value <= 0:
            return 0
        
        # Base position size from signal
        base_size = signal.size or (portfolio_value * 0.1)  # Default 10%
        
        # Apply position size limit
        max_position_value = portfolio_value * self.risk_limits.max_position_size
        size_limited = min(base_size, max_position_value)
        
        # Apply portfolio risk limit
        if signal.stop_loss and signal.entry_price:
            risk_per_trade = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
            max_risk_size = (portfolio_value * self.risk_limits.max_portfolio_risk) / risk_per_trade
            size_limited = min(size_limited, max_risk_size)
        
        # Apply concentration limit
        current_exposure = self._calculate_coin_exposure(signal.coin, portfolio)
        remaining_concentration = self.risk_limits.max_concentration - current_exposure
        max_concentration_size = portfolio_value * remaining_concentration
        size_limited = min(size_limited, max_concentration_size)
        
        # Apply leverage limit
        current_exposure_total = self._calculate_total_exposure(portfolio)
        remaining_leverage = (portfolio_value * self.risk_limits.max_leverage) - current_exposure_total
        size_limited = min(size_limited, remaining_leverage)
        
        return max(size_limited, 0)
    
    def calculate_stop_loss(self, signal: TradingSignal, portfolio: Dict[str, Any]) -> float:
        """
        Calculate stop loss based on risk management rules
        
        Args:
            signal: Trading signal
            portfolio: Current portfolio state
            
        Returns:
            Stop loss price
        """
        if not signal.entry_price:
            return 0
        
        portfolio_value = portfolio.get('account_value', 0)
        position_size = self.calculate_position_size(signal, portfolio)
        
        if portfolio_value <= 0 or position_size <= 0:
            return signal.stop_loss or 0
        
        # Calculate maximum acceptable loss
        max_loss = portfolio_value * self.risk_limits.max_portfolio_risk
        
        # Calculate stop loss distance
        max_loss_per_unit = max_loss / (position_size / signal.entry_price)
        
        if signal.signal_type == SignalType.LONG:
            calculated_stop = signal.entry_price - max_loss_per_unit
            # Use the more conservative stop loss
            if signal.stop_loss:
                return max(calculated_stop, signal.stop_loss)
            return calculated_stop
        
        elif signal.signal_type == SignalType.SHORT:
            calculated_stop = signal.entry_price + max_loss_per_unit
            # Use the more conservative stop loss
            if signal.stop_loss:
                return min(calculated_stop, signal.stop_loss)
            return calculated_stop
        
        return signal.stop_loss or 0
    
    def update_portfolio_state(self, portfolio: Dict[str, Any]) -> None:
        """Update portfolio state for risk monitoring"""
        # Add to portfolio history
        portfolio_entry = {
            'timestamp': datetime.now(),
            'account_value': portfolio.get('account_value', 0),
            'positions': portfolio.get('positions', []),
            'total_pnl': sum(p.get('unrealized_pnl', 0) for p in portfolio.get('positions', []))
        }
        self.portfolio_history.append(portfolio_entry)
        
        # Keep only last 1000 entries
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-1000:]
        
        # Update daily PnL tracking
        today = datetime.now().date()
        if len(self.portfolio_history) >= 2:
            yesterday_value = self.portfolio_history[-2]['account_value']
            today_value = portfolio_entry['account_value']
            self.daily_pnl[today] = today_value - yesterday_value
    
    def calculate_risk_metrics(self, portfolio: Dict[str, Any]) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics
        
        Args:
            portfolio: Current portfolio state
            
        Returns:
            RiskMetrics object
        """
        portfolio_value = portfolio.get('account_value', 0)
        
        # Calculate total exposure
        total_exposure = self._calculate_total_exposure(portfolio)
        
        # Calculate leverage
        leverage_ratio = total_exposure / portfolio_value if portfolio_value > 0 else 0
        
        # Calculate VaR (simplified)
        var_1d = self._calculate_var(portfolio, days=1)
        var_7d = self._calculate_var(portfolio, days=7)
        
        # Calculate max drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        # Calculate Sharpe ratio (simplified)
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Calculate correlation risk
        correlation_risk = self._calculate_correlation_risk(portfolio)
        
        # Calculate concentration risk
        concentration_risk = self._calculate_concentration_risk(portfolio)
        
        # Determine overall risk level
        risk_level = self._determine_risk_level(leverage_ratio, max_drawdown, var_1d, concentration_risk)
        
        return RiskMetrics(
            portfolio_value=portfolio_value,
            total_exposure=total_exposure,
            leverage_ratio=leverage_ratio,
            var_1d=var_1d,
            var_7d=var_7d,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            correlation_risk=correlation_risk,
            concentration_risk=concentration_risk,
            risk_level=risk_level
        )
    
    def _calculate_var(self, portfolio: Dict[str, Any], days: int = 1, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(self.portfolio_history) < days + 1:
            return 0
        
        # Get recent portfolio values
        recent_values = [p['account_value'] for p in self.portfolio_history[-days-1:]]
        
        # Calculate returns
        returns = [(recent_values[i] - recent_values[i-1]) / recent_values[i-1] 
                  for i in range(1, len(recent_values))]
        
        if not returns:
            return 0
        
        # Calculate VaR
        var_percentile = (1 - confidence) * 100
        var_return = np.percentile(returns, var_percentile)
        var_amount = abs(var_return * portfolio.get('account_value', 0))
        
        return var_amount
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from portfolio history"""
        if len(self.portfolio_history) < 2:
            return 0
        
        values = [p['account_value'] for p in self.portfolio_history]
        peak = values[0]
        max_dd = 0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from portfolio history"""
        if len(self.portfolio_history) < 30:  # Need at least 30 data points
            return 0
        
        values = [p['account_value'] for p in self.portfolio_history[-30:]]
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
        
        if not returns or np.std(returns) == 0:
            return 0
        
        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
    
    def _calculate_correlation_risk(self, portfolio: Dict[str, Any]) -> float:
        """Calculate correlation risk score"""
        # Simplified correlation risk calculation
        positions = portfolio.get('positions', [])
        if len(positions) <= 1:
            return 0
        
        # This would typically calculate actual correlations between assets
        # For now, return a simplified score based on position concentration
        return min(len(positions) * 0.1, 1.0)
    
    def _calculate_concentration_risk(self, portfolio: Dict[str, Any]) -> float:
        """Calculate concentration risk score"""
        portfolio_value = portfolio.get('account_value', 1)
        positions = portfolio.get('positions', [])
        
        if not positions:
            return 0
        
        # Calculate largest position percentage
        max_position_pct = 0
        for position in positions:
            position_value = abs(position.get('position_value', 0))
            position_pct = position_value / portfolio_value
            max_position_pct = max(max_position_pct, position_pct)
        
        return max_position_pct
    
    def _determine_risk_level(self, leverage: float, drawdown: float, 
                            var: float, concentration: float) -> RiskLevel:
        """Determine overall risk level"""
        risk_score = 0
        
        # Leverage risk
        if leverage > self.risk_limits.max_leverage * 0.8:
            risk_score += 3
        elif leverage > self.risk_limits.max_leverage * 0.6:
            risk_score += 2
        elif leverage > self.risk_limits.max_leverage * 0.4:
            risk_score += 1
        
        # Drawdown risk
        if drawdown > self.risk_limits.max_drawdown * 0.8:
            risk_score += 3
        elif drawdown > self.risk_limits.max_drawdown * 0.6:
            risk_score += 2
        elif drawdown > self.risk_limits.max_drawdown * 0.4:
            risk_score += 1
        
        # Concentration risk
        if concentration > self.risk_limits.max_concentration * 0.8:
            risk_score += 2
        elif concentration > self.risk_limits.max_concentration * 0.6:
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 6:
            return RiskLevel.CRITICAL
        elif risk_score >= 4:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def generate_risk_report(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        risk_metrics = self.calculate_risk_metrics(portfolio)
        
        # Check for risk violations
        violations = []
        
        if risk_metrics.leverage_ratio > self.risk_limits.max_leverage:
            violations.append(f"Leverage exceeds limit: {risk_metrics.leverage_ratio:.1f}x > {self.risk_limits.max_leverage:.1f}x")
        
        if risk_metrics.max_drawdown > self.risk_limits.max_drawdown:
            violations.append(f"Drawdown exceeds limit: {risk_metrics.max_drawdown:.2%} > {self.risk_limits.max_drawdown:.2%}")
        
        if risk_metrics.concentration_risk > self.risk_limits.max_concentration:
            violations.append(f"Concentration risk too high: {risk_metrics.concentration_risk:.2%} > {self.risk_limits.max_concentration:.2%}")
        
        if risk_metrics.var_1d > portfolio.get('account_value', 0) * self.risk_limits.var_limit:
            violations.append(f"VaR exceeds limit: ${risk_metrics.var_1d:.2f}")
        
        # Generate recommendations
        recommendations = []
        
        if risk_metrics.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append("Consider reducing position sizes")
            recommendations.append("Review stop loss levels")
        
        if risk_metrics.leverage_ratio > self.risk_limits.max_leverage * 0.8:
            recommendations.append("Reduce leverage by closing some positions")
        
        if risk_metrics.concentration_risk > self.risk_limits.max_concentration * 0.8:
            recommendations.append("Diversify portfolio to reduce concentration risk")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'risk_metrics': risk_metrics.to_dict(),
            'risk_limits': asdict(self.risk_limits),
            'violations': violations,
            'recommendations': recommendations,
            'overall_assessment': f"Portfolio risk level: {risk_metrics.risk_level.value}"
        }
    
    def emergency_stop(self, reason: str) -> Dict[str, Any]:
        """
        Trigger emergency stop of all trading activities
        
        Args:
            reason: Reason for emergency stop
            
        Returns:
            Emergency stop report
        """
        logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
        trading_logger.log_error(f"Emergency stop: {reason}")
        
        # Add to active alerts
        alert = {
            'timestamp': datetime.now(),
            'type': 'EMERGENCY_STOP',
            'reason': reason,
            'severity': 'CRITICAL'
        }
        self.active_alerts.append(alert)
        
        return {
            'status': 'EMERGENCY_STOP_ACTIVATED',
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'action_required': 'Manual intervention required to resume trading'
        }
    
    def reset_daily_limits(self) -> None:
        """Reset daily trading limits (call at start of new day)"""
        self.daily_trades = 0
        self.hourly_trades = 0
        logger.info("Daily trading limits reset")
    
    def record_trade(self, trade_info: Dict[str, Any]) -> None:
        """Record a trade for risk tracking"""
        self.trade_history.append({
            'timestamp': datetime.now(),
            **trade_info
        })
        
        self.daily_trades += 1
        self.hourly_trades += 1
        self.last_trade_time = datetime.now()
        
        # Update daily PnL if trade has PnL info
        if 'pnl' in trade_info:
            today = datetime.now().date()
            self.daily_pnl[today] = self.daily_pnl.get(today, 0) + trade_info['pnl']

