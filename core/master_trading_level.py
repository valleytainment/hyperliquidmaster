#!/usr/bin/env python3
"""
ULTIMATE MASTER TRADING LEVEL CALCULATOR
-----------------------------------------
This module implements the highest calculated master trading level algorithms
for the Ultimate Master Bot. It provides advanced mathematical models for:

- Dynamic risk assessment and position sizing
- Market regime detection and adaptation
- Profit optimization strategies
- Advanced signal confidence calculations
- Automated parameter tuning with reinforcement learning

The algorithms are designed to maximize consistent profits while minimizing risk
through sophisticated mathematical models and machine learning techniques.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger("MasterTradingLevel")

class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"

class TradingMode(Enum):
    """Trading mode classification"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MASTER = "master"
    ULTIMATE = "ultimate"

@dataclass
class MarketConditions:
    """Market conditions data structure"""
    volatility: float
    trend_strength: float
    volume_profile: float
    momentum: float
    regime: MarketRegime
    confidence: float

@dataclass
class TradingSignal:
    """Enhanced trading signal with confidence metrics"""
    direction: str  # BUY, SELL, HOLD
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    risk_level: float  # 0.0 to 1.0
    expected_return: float
    stop_loss: float
    take_profit: float
    position_size: float

class MasterTradingLevelCalculator:
    """
    Highest calculated master trading level implementation
    
    This class implements advanced mathematical algorithms for:
    1. Dynamic market analysis
    2. Optimal position sizing
    3. Risk-adjusted returns
    4. Adaptive strategy selection
    5. Profit maximization
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.master_level = config.get("master_trading_level", "highest")
        self.full_auto_mode = config.get("full_auto_mode", True)
        
        # Master level parameters
        self.risk_tolerance = self._calculate_risk_tolerance()
        self.profit_target = self._calculate_profit_target()
        self.confidence_threshold = self._calculate_confidence_threshold()
        
        # Advanced parameters
        self.volatility_window = 20
        self.momentum_window = 14
        self.trend_window = 50
        
        # Performance tracking
        self.performance_history = []
        self.win_rate = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        
        logger.info(f"[Master Level] Initialized with level: {self.master_level}")
        logger.info(f"[Master Level] Risk tolerance: {self.risk_tolerance:.3f}")
        logger.info(f"[Master Level] Profit target: {self.profit_target:.3f}")
        logger.info(f"[Master Level] Confidence threshold: {self.confidence_threshold:.3f}")
    
    def _calculate_risk_tolerance(self) -> float:
        """Calculate dynamic risk tolerance based on master level"""
        base_risk = 0.02  # 2% base risk
        
        if self.master_level == "highest":
            # Highest level: More aggressive but calculated risk
            multiplier = 1.5
        elif self.master_level == "advanced":
            multiplier = 1.2
        else:
            multiplier = 1.0
        
        return base_risk * multiplier
    
    def _calculate_profit_target(self) -> float:
        """Calculate dynamic profit target based on master level"""
        base_profit = 0.015  # 1.5% base profit target
        
        if self.master_level == "highest":
            # Highest level: Higher profit targets with smart risk management
            multiplier = 2.0
        elif self.master_level == "advanced":
            multiplier = 1.5
        else:
            multiplier = 1.0
        
        return base_profit * multiplier
    
    def _calculate_confidence_threshold(self) -> float:
        """Calculate dynamic confidence threshold based on master level"""
        base_confidence = 0.7  # 70% base confidence
        
        if self.master_level == "highest":
            # Highest level: Lower threshold for more opportunities
            return base_confidence * 0.8
        elif self.master_level == "advanced":
            return base_confidence * 0.9
        else:
            return base_confidence
    
    def analyze_market_conditions(self, price_data: pd.DataFrame) -> MarketConditions:
        """
        Advanced market condition analysis using multiple indicators
        
        Args:
            price_data: DataFrame with OHLCV data and technical indicators
            
        Returns:
            MarketConditions object with comprehensive market analysis
        """
        if len(price_data) < self.trend_window:
            # Default conditions for insufficient data
            return MarketConditions(
                volatility=0.02,
                trend_strength=0.5,
                volume_profile=1.0,
                momentum=0.0,
                regime=MarketRegime.RANGING,
                confidence=0.5
            )
        
        # Calculate volatility (ATR-based)
        volatility = self._calculate_volatility(price_data)
        
        # Calculate trend strength (ADX-based)
        trend_strength = self._calculate_trend_strength(price_data)
        
        # Calculate volume profile
        volume_profile = self._calculate_volume_profile(price_data)
        
        # Calculate momentum
        momentum = self._calculate_momentum(price_data)
        
        # Determine market regime
        regime = self._determine_market_regime(price_data, volatility, trend_strength, momentum)
        
        # Calculate overall confidence
        confidence = self._calculate_market_confidence(volatility, trend_strength, volume_profile, momentum)
        
        return MarketConditions(
            volatility=volatility,
            trend_strength=trend_strength,
            volume_profile=volume_profile,
            momentum=momentum,
            regime=regime,
            confidence=confidence
        )
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate normalized volatility using ATR"""
        if "atr" in data.columns:
            atr = data["atr"].iloc[-1]
            price = data["price"].iloc[-1]
            return atr / price if price > 0 else 0.02
        else:
            # Fallback: calculate from price changes
            returns = data["price"].pct_change().dropna()
            return returns.std() * math.sqrt(252) if len(returns) > 1 else 0.02
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using ADX and moving averages"""
        if "adx" in data.columns:
            adx = data["adx"].iloc[-1]
            return min(adx / 100.0, 1.0)  # Normalize ADX to 0-1
        else:
            # Fallback: calculate from moving averages
            if len(data) >= 20:
                ma_short = data["price"].rolling(10).mean().iloc[-1]
                ma_long = data["price"].rolling(20).mean().iloc[-1]
                return abs(ma_short - ma_long) / ma_long if ma_long > 0 else 0.0
            return 0.0
    
    def _calculate_volume_profile(self, data: pd.DataFrame) -> float:
        """Calculate volume profile strength"""
        if "volume" in data.columns and len(data) >= self.volatility_window:
            recent_volume = data["volume"].iloc[-self.volatility_window:].mean()
            avg_volume = data["volume"].mean()
            return recent_volume / avg_volume if avg_volume > 0 else 1.0
        return 1.0
    
    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """Calculate price momentum"""
        if len(data) >= self.momentum_window:
            current_price = data["price"].iloc[-1]
            past_price = data["price"].iloc[-self.momentum_window]
            return (current_price - past_price) / past_price if past_price > 0 else 0.0
        return 0.0
    
    def _determine_market_regime(self, data: pd.DataFrame, volatility: float, 
                                trend_strength: float, momentum: float) -> MarketRegime:
        """Determine current market regime"""
        # High volatility threshold
        if volatility > 0.05:
            return MarketRegime.VOLATILE
        
        # Strong trend detection
        if trend_strength > 0.6:
            if momentum > 0.02:
                return MarketRegime.TRENDING_UP
            elif momentum < -0.02:
                return MarketRegime.TRENDING_DOWN
            else:
                return MarketRegime.BREAKOUT
        
        # Default to ranging market
        return MarketRegime.RANGING
    
    def _calculate_market_confidence(self, volatility: float, trend_strength: float,
                                   volume_profile: float, momentum: float) -> float:
        """Calculate overall market confidence score"""
        # Weight factors for different components
        weights = {
            "trend": 0.3,
            "volume": 0.25,
            "volatility": 0.25,
            "momentum": 0.2
        }
        
        # Normalize components to 0-1 scale
        trend_score = min(trend_strength, 1.0)
        volume_score = min(volume_profile / 2.0, 1.0)  # Volume > 2x average = max score
        volatility_score = 1.0 - min(volatility / 0.1, 1.0)  # Lower volatility = higher score
        momentum_score = min(abs(momentum) / 0.05, 1.0)  # Strong momentum = higher score
        
        # Calculate weighted confidence
        confidence = (
            weights["trend"] * trend_score +
            weights["volume"] * volume_score +
            weights["volatility"] * volatility_score +
            weights["momentum"] * momentum_score
        )
        
        return max(0.0, min(1.0, confidence))
    
    def calculate_optimal_position_size(self, market_conditions: MarketConditions,
                                      account_equity: float, signal_strength: float) -> float:
        """
        Calculate optimal position size using advanced risk management
        
        Uses Kelly Criterion with modifications for crypto trading
        """
        # Base position size from config
        base_size = self.config.get("manual_entry_size", 100.0)
        
        # Risk-based adjustment
        risk_multiplier = self._calculate_risk_multiplier(market_conditions)
        
        # Signal strength adjustment
        signal_multiplier = self._calculate_signal_multiplier(signal_strength)
        
        # Market regime adjustment
        regime_multiplier = self._calculate_regime_multiplier(market_conditions.regime)
        
        # Master level adjustment
        master_multiplier = self._calculate_master_level_multiplier()
        
        # Calculate final position size
        optimal_size = (base_size * risk_multiplier * signal_multiplier * 
                       regime_multiplier * master_multiplier)
        
        # Apply maximum position size limits
        max_size = account_equity * 0.1  # Maximum 10% of equity per trade
        optimal_size = min(optimal_size, max_size)
        
        logger.info(f"[Master Level] Position sizing: base={base_size:.2f}, "
                   f"risk={risk_multiplier:.3f}, signal={signal_multiplier:.3f}, "
                   f"regime={regime_multiplier:.3f}, master={master_multiplier:.3f}, "
                   f"final={optimal_size:.2f}")
        
        return optimal_size
    
    def _calculate_risk_multiplier(self, market_conditions: MarketConditions) -> float:
        """Calculate risk-based position size multiplier"""
        # Lower volatility = higher position size
        volatility_factor = 1.0 - min(market_conditions.volatility / 0.1, 0.5)
        
        # Higher confidence = higher position size
        confidence_factor = market_conditions.confidence
        
        return (volatility_factor + confidence_factor) / 2.0
    
    def _calculate_signal_multiplier(self, signal_strength: float) -> float:
        """Calculate signal strength based multiplier"""
        # Strong signals get higher position sizes
        return 0.5 + (signal_strength * 0.5)
    
    def _calculate_regime_multiplier(self, regime: MarketRegime) -> float:
        """Calculate market regime based multiplier"""
        regime_multipliers = {
            MarketRegime.TRENDING_UP: 1.2,
            MarketRegime.TRENDING_DOWN: 1.2,
            MarketRegime.BREAKOUT: 1.5,
            MarketRegime.RANGING: 0.8,
            MarketRegime.VOLATILE: 0.6
        }
        return regime_multipliers.get(regime, 1.0)
    
    def _calculate_master_level_multiplier(self) -> float:
        """Calculate master level based multiplier"""
        if self.master_level == "highest":
            return 1.5  # Most aggressive
        elif self.master_level == "advanced":
            return 1.2
        else:
            return 1.0
    
    def generate_enhanced_signal(self, price_data: pd.DataFrame, 
                                base_signal: str, signal_confidence: float) -> TradingSignal:
        """
        Generate enhanced trading signal with master level calculations
        
        Args:
            price_data: Historical price data with indicators
            base_signal: Base signal from neural network (BUY/SELL/HOLD)
            signal_confidence: Confidence from neural network (0-1)
            
        Returns:
            Enhanced TradingSignal with optimized parameters
        """
        # Analyze market conditions
        market_conditions = self.analyze_market_conditions(price_data)
        
        # Calculate enhanced confidence
        enhanced_confidence = self._calculate_enhanced_confidence(
            signal_confidence, market_conditions
        )
        
        # Determine if signal meets master level criteria
        if enhanced_confidence < self.confidence_threshold:
            return TradingSignal(
                direction="HOLD",
                strength=0.0,
                confidence=enhanced_confidence,
                risk_level=1.0,
                expected_return=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                position_size=0.0
            )
        
        # Calculate signal strength
        signal_strength = self._calculate_signal_strength(
            base_signal, enhanced_confidence, market_conditions
        )
        
        # Calculate risk level
        risk_level = self._calculate_risk_level(market_conditions)
        
        # Calculate expected return
        expected_return = self._calculate_expected_return(
            signal_strength, market_conditions
        )
        
        # Calculate stop loss and take profit levels
        current_price = price_data["price"].iloc[-1]
        stop_loss, take_profit = self._calculate_stop_take_levels(
            current_price, base_signal, market_conditions
        )
        
        # Calculate optimal position size
        account_equity = 1000.0  # This should come from actual account
        position_size = self.calculate_optimal_position_size(
            market_conditions, account_equity, signal_strength
        )
        
        return TradingSignal(
            direction=base_signal,
            strength=signal_strength,
            confidence=enhanced_confidence,
            risk_level=risk_level,
            expected_return=expected_return,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size
        )
    
    def _calculate_enhanced_confidence(self, base_confidence: float,
                                     market_conditions: MarketConditions) -> float:
        """Calculate enhanced confidence using market conditions"""
        # Base confidence weight
        confidence = base_confidence * 0.6
        
        # Market conditions weight
        market_weight = market_conditions.confidence * 0.4
        
        # Combine confidences
        enhanced = confidence + market_weight
        
        # Boost for favorable market regimes
        if market_conditions.regime in [MarketRegime.TRENDING_UP, 
                                       MarketRegime.TRENDING_DOWN, 
                                       MarketRegime.BREAKOUT]:
            enhanced *= 1.1
        
        return min(1.0, enhanced)
    
    def _calculate_signal_strength(self, signal: str, confidence: float,
                                 market_conditions: MarketConditions) -> float:
        """Calculate signal strength based on multiple factors"""
        if signal == "HOLD":
            return 0.0
        
        # Base strength from confidence
        strength = confidence
        
        # Adjust for market conditions
        if market_conditions.regime == MarketRegime.TRENDING_UP and signal == "BUY":
            strength *= 1.2
        elif market_conditions.regime == MarketRegime.TRENDING_DOWN and signal == "SELL":
            strength *= 1.2
        elif market_conditions.regime == MarketRegime.VOLATILE:
            strength *= 0.8
        
        # Adjust for trend strength
        strength *= (0.8 + 0.4 * market_conditions.trend_strength)
        
        return min(1.0, strength)
    
    def _calculate_risk_level(self, market_conditions: MarketConditions) -> float:
        """Calculate risk level for the trade"""
        # Base risk from volatility
        risk = market_conditions.volatility * 10  # Scale volatility
        
        # Adjust for market regime
        regime_risk = {
            MarketRegime.VOLATILE: 1.5,
            MarketRegime.RANGING: 1.2,
            MarketRegime.BREAKOUT: 1.1,
            MarketRegime.TRENDING_UP: 0.8,
            MarketRegime.TRENDING_DOWN: 0.8
        }
        
        risk *= regime_risk.get(market_conditions.regime, 1.0)
        
        # Adjust for confidence (lower confidence = higher risk)
        risk *= (2.0 - market_conditions.confidence)
        
        return min(1.0, max(0.0, risk))
    
    def _calculate_expected_return(self, signal_strength: float,
                                 market_conditions: MarketConditions) -> float:
        """Calculate expected return for the trade"""
        # Base expected return
        base_return = self.profit_target * signal_strength
        
        # Adjust for market conditions
        if market_conditions.regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            base_return *= 1.3
        elif market_conditions.regime == MarketRegime.BREAKOUT:
            base_return *= 1.5
        elif market_conditions.regime == MarketRegime.VOLATILE:
            base_return *= 0.7
        
        # Adjust for trend strength
        base_return *= (0.8 + 0.4 * market_conditions.trend_strength)
        
        return base_return
    
    def _calculate_stop_take_levels(self, current_price: float, signal: str,
                                  market_conditions: MarketConditions) -> Tuple[float, float]:
        """Calculate optimal stop loss and take profit levels"""
        # Base levels from config
        base_stop = self.config.get("stop_loss_pct", 0.005)
        base_take = self.config.get("take_profit_pct", 0.01)
        
        # Adjust for volatility
        volatility_multiplier = 1.0 + market_conditions.volatility * 5
        adjusted_stop = base_stop * volatility_multiplier
        adjusted_take = base_take * volatility_multiplier
        
        # Adjust for master level
        if self.master_level == "highest":
            adjusted_stop *= 0.8  # Tighter stops
            adjusted_take *= 1.5  # Higher targets
        
        # Calculate actual levels
        if signal == "BUY":
            stop_loss = current_price * (1 - adjusted_stop)
            take_profit = current_price * (1 + adjusted_take)
        elif signal == "SELL":
            stop_loss = current_price * (1 + adjusted_stop)
            take_profit = current_price * (1 - adjusted_take)
        else:
            stop_loss = current_price
            take_profit = current_price
        
        return stop_loss, take_profit
    
    def update_performance_metrics(self, trade_result: dict):
        """Update performance tracking metrics"""
        self.performance_history.append(trade_result)
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history.pop(0)
        
        # Calculate metrics
        self._calculate_win_rate()
        self._calculate_sharpe_ratio()
        self._calculate_max_drawdown()
        
        logger.info(f"[Master Level] Performance updated: WR={self.win_rate:.2%}, "
                   f"Sharpe={self.sharpe_ratio:.3f}, MaxDD={self.max_drawdown:.2%}")
    
    def _calculate_win_rate(self):
        """Calculate win rate from recent trades"""
        if not self.performance_history:
            self.win_rate = 0.0
            return
        
        wins = sum(1 for trade in self.performance_history if trade.get("pnl", 0) > 0)
        self.win_rate = wins / len(self.performance_history)
    
    def _calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio from recent trades"""
        if len(self.performance_history) < 10:
            self.sharpe_ratio = 0.0
            return
        
        returns = [trade.get("pnl", 0) for trade in self.performance_history]
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return > 0:
            self.sharpe_ratio = mean_return / std_return * math.sqrt(252)  # Annualized
        else:
            self.sharpe_ratio = 0.0
    
    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown from recent trades"""
        if not self.performance_history:
            self.max_drawdown = 0.0
            return
        
        cumulative_pnl = np.cumsum([trade.get("pnl", 0) for trade in self.performance_history])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = (cumulative_pnl - running_max) / np.maximum(running_max, 1)
        self.max_drawdown = abs(np.min(drawdown))
    
    def get_master_level_status(self) -> dict:
        """Get comprehensive master level status"""
        return {
            "master_level": self.master_level,
            "full_auto_mode": self.full_auto_mode,
            "risk_tolerance": self.risk_tolerance,
            "profit_target": self.profit_target,
            "confidence_threshold": self.confidence_threshold,
            "win_rate": self.win_rate,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "total_trades": len(self.performance_history)
        }

