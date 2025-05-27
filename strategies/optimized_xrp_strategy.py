"""
Optimized Strategy for XRP Trading with Mode Management
------------------------------------------------------
This module implements an optimized trading strategy for XRP with mode-specific parameters.
"""

import os
import json
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from core.trading_mode import TradingMode

class OptimizedXRPStrategy:
    """
    Optimized trading strategy for XRP with mode-specific parameters.
    Implements sophisticated signal generation and risk management.
    """
    
    def __init__(self, mode_manager, position_manager, logger=None):
        """
        Initialize the optimized XRP strategy.
        
        Args:
            mode_manager: Trading mode manager instance
            position_manager: Position manager instance
            logger: Optional logger instance
        """
        self.mode_manager = mode_manager
        self.position_manager = position_manager
        self.logger = logger or logging.getLogger("OptimizedXRPStrategy")
        
        # Strategy parameters - base values
        self.base_params = {
            "ema_short": 12,
            "ema_medium": 26,
            "ema_long": 50,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "volume_threshold": 1.5,
            "funding_threshold": 0.0001,
            "min_price_move": 0.005,
            "profit_target": 0.02,
            "stop_loss": 0.01,
            "trailing_stop": 0.005,
            "max_trade_duration": 24 * 60 * 60,  # 24 hours in seconds
            "entry_confirmation_threshold": 0.7,
            "exit_confirmation_threshold": 0.6
        }
        
        # Mode-specific parameter adjustments
        self.mode_params = {
            TradingMode.PAPER_TRADING: {
                # Standard parameters for paper trading
            },
            TradingMode.LIVE_TRADING: {
                "profit_target": 0.015,
                "stop_loss": 0.008,
                "entry_confirmation_threshold": 0.75
            },
            TradingMode.AGGRESSIVE: {
                "ema_short": 8,
                "rsi_overbought": 75,
                "rsi_oversold": 25,
                "profit_target": 0.03,
                "stop_loss": 0.015,
                "trailing_stop": 0.008,
                "entry_confirmation_threshold": 0.65
            },
            TradingMode.CONSERVATIVE: {
                "ema_short": 16,
                "ema_medium": 32,
                "rsi_overbought": 65,
                "rsi_oversold": 35,
                "profit_target": 0.01,
                "stop_loss": 0.005,
                "entry_confirmation_threshold": 0.8
            },
            TradingMode.MONITOR_ONLY: {
                # Same as paper trading but no actual trades
            }
        }
        
        # Market state tracking
        self.market_state = {
            "trend": "neutral",  # "bullish", "bearish", or "neutral"
            "volatility": "medium",  # "low", "medium", or "high"
            "liquidity": "normal",  # "low", "normal", or "high"
            "regime": "normal",  # "trending", "ranging", "volatile", or "normal"
            "last_update": 0
        }
        
        # Historical data
        self.price_history = []
        self.volume_history = []
        self.funding_history = []
        
        # Performance metrics
        self.performance = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_profit": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_profit_per_trade": 0.0,
            "avg_holding_time": 0.0
        }
    
    def get_current_params(self) -> Dict[str, Any]:
        """
        Get strategy parameters for the current trading mode.
        
        Returns:
            Dict containing strategy parameters
        """
        # Get current mode
        current_mode = self.mode_manager.get_current_mode()
        
        # Start with base parameters
        params = self.base_params.copy()
        
        # Apply mode-specific adjustments
        if current_mode in self.mode_params:
            mode_adjustments = self.mode_params[current_mode]
            for key, value in mode_adjustments.items():
                params[key] = value
        
        return params
    
    def update_market_state(self, market_data: Dict[str, Any]) -> None:
        """
        Update market state based on new market data.
        
        Args:
            market_data: Dict containing market data
        """
        # Extract data
        current_price = market_data.get("price", 0.0)
        current_volume = market_data.get("volume_24h", 0.0)
        current_funding = market_data.get("funding_rate", 0.0)
        
        # Update historical data
        self.price_history.append(current_price)
        self.volume_history.append(current_volume)
        self.funding_history.append(current_funding)
        
        # Keep only the last 100 data points
        max_history = 100
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
            self.volume_history = self.volume_history[-max_history:]
            self.funding_history = self.funding_history[-max_history:]
        
        # Calculate trend
        if len(self.price_history) >= 3:
            short_trend = self.price_history[-1] > self.price_history[-3]
            if short_trend:
                self.market_state["trend"] = "bullish"
            else:
                self.market_state["trend"] = "bearish"
        
        # Calculate volatility
        if len(self.price_history) >= 20:
            recent_prices = self.price_history[-20:]
            volatility = np.std(recent_prices) / np.mean(recent_prices)
            
            if volatility < 0.01:
                self.market_state["volatility"] = "low"
            elif volatility > 0.03:
                self.market_state["volatility"] = "high"
            else:
                self.market_state["volatility"] = "medium"
        
        # Calculate liquidity
        if len(self.volume_history) >= 5:
            recent_volume = self.volume_history[-1]
            avg_volume = np.mean(self.volume_history[-5:])
            
            if recent_volume < 0.7 * avg_volume:
                self.market_state["liquidity"] = "low"
            elif recent_volume > 1.5 * avg_volume:
                self.market_state["liquidity"] = "high"
            else:
                self.market_state["liquidity"] = "normal"
        
        # Determine market regime
        if self.market_state["volatility"] == "high":
            self.market_state["regime"] = "volatile"
        elif self.market_state["trend"] != "neutral" and self.market_state["volatility"] != "low":
            self.market_state["regime"] = "trending"
        elif self.market_state["volatility"] == "low":
            self.market_state["regime"] = "ranging"
        else:
            self.market_state["regime"] = "normal"
        
        # Update timestamp
        self.market_state["last_update"] = time.time()
    
    def calculate_ema(self, prices: List[float], period: int) -> float:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: List of prices
            period: EMA period
            
        Returns:
            EMA value
        """
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
            
        return ema
    
    def calculate_rsi(self, prices: List[float], period: int) -> float:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: List of prices
            period: RSI period
            
        Returns:
            RSI value
        """
        if len(prices) < period + 1:
            return 50.0
        
        # Calculate price changes
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Calculate gains and losses
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        # Calculate average gains and losses
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signal(self, market_data: Dict[str, Any], positions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signal based on market data and current positions.
        
        Args:
            market_data: Dict containing market data
            positions: Dict containing current positions
            
        Returns:
            Dict containing trading signal
        """
        # Update market state
        self.update_market_state(market_data)
        
        # Get current parameters
        params = self.get_current_params()
        
        # Extract data
        symbol = market_data.get("symbol", "XRP")
        current_price = market_data.get("price", 0.0)
        funding_rate = market_data.get("funding_rate", 0.0)
        
        # Check if we have enough data
        if len(self.price_history) < params["ema_long"]:
            return {
                "symbol": symbol,
                "signal": "WAIT",
                "confidence": 0.0,
                "reason": "Insufficient data"
            }
        
        # Calculate indicators
        ema_short = self.calculate_ema(self.price_history, params["ema_short"])
        ema_medium = self.calculate_ema(self.price_history, params["ema_medium"])
        ema_long = self.calculate_ema(self.price_history, params["ema_long"])
        rsi = self.calculate_rsi(self.price_history, params["rsi_period"])
        
        # Check current position
        has_position = symbol in positions
        position_size = positions.get(symbol, {}).get("size", 0.0) if has_position else 0.0
        is_long = position_size > 0 if has_position else False
        
        # Initialize signal variables
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        # Generate signal based on market regime
        if self.market_state["regime"] == "trending":
            # Trending market strategy
            if self.market_state["trend"] == "bullish":
                # Bullish trend
                if not has_position or not is_long:
                    # No position or short position
                    if ema_short > ema_medium and ema_medium > ema_long and rsi < 65:
                        signal = "BUY"
                        confidence = min(0.9, 0.6 + (ema_short / ema_long - 1) * 10)
                        reason = "Strong uptrend with EMA alignment"
                elif is_long:
                    # Long position
                    if ema_short < ema_medium or rsi > params["rsi_overbought"]:
                        signal = "SELL"
                        confidence = 0.7 if ema_short < ema_medium else 0.8
                        reason = "Trend weakening or overbought"
            else:
                # Bearish trend
                if not has_position or is_long:
                    # No position or long position
                    if ema_short < ema_medium and ema_medium < ema_long and rsi > 35:
                        signal = "SELL"
                        confidence = min(0.9, 0.6 + (1 - ema_short / ema_long) * 10)
                        reason = "Strong downtrend with EMA alignment"
                elif not is_long:
                    # Short position
                    if ema_short > ema_medium or rsi < params["rsi_oversold"]:
                        signal = "BUY"
                        confidence = 0.7 if ema_short > ema_medium else 0.8
                        reason = "Trend weakening or oversold"
        
        elif self.market_state["regime"] == "ranging":
            # Ranging market strategy
            if not has_position:
                # No position
                if rsi < params["rsi_oversold"]:
                    signal = "BUY"
                    confidence = 0.6 + (params["rsi_oversold"] - rsi) / 30
                    reason = "Oversold in ranging market"
                elif rsi > params["rsi_overbought"]:
                    signal = "SELL"
                    confidence = 0.6 + (rsi - params["rsi_overbought"]) / 30
                    reason = "Overbought in ranging market"
            elif is_long:
                # Long position
                if rsi > 60:
                    signal = "SELL"
                    confidence = 0.6 + (rsi - 60) / 40
                    reason = "Take profit in ranging market"
            else:
                # Short position
                if rsi < 40:
                    signal = "BUY"
                    confidence = 0.6 + (40 - rsi) / 40
                    reason = "Take profit in ranging market"
        
        elif self.market_state["regime"] == "volatile":
            # Volatile market strategy - more conservative
            if has_position:
                # Already have position - consider taking profit or cutting loss
                if is_long and (rsi > 65 or ema_short < ema_medium):
                    signal = "SELL"
                    confidence = 0.8
                    reason = "Reducing risk in volatile market"
                elif not is_long and (rsi < 35 or ema_short > ema_medium):
                    signal = "BUY"
                    confidence = 0.8
                    reason = "Reducing risk in volatile market"
            else:
                # No position - wait for clear signals
                if rsi < 25 and ema_short > ema_medium:
                    signal = "BUY"
                    confidence = 0.7
                    reason = "Strong oversold in volatile market"
                elif rsi > 75 and ema_short < ema_medium:
                    signal = "SELL"
                    confidence = 0.7
                    reason = "Strong overbought in volatile market"
        
        else:
            # Normal market strategy
            if not has_position:
                # No position
                if ema_short > ema_medium and rsi < 60:
                    signal = "BUY"
                    confidence = 0.6 + (ema_short / ema_medium - 1) * 5
                    reason = "Bullish trend in normal market"
                elif ema_short < ema_medium and rsi > 40:
                    signal = "SELL"
                    confidence = 0.6 + (1 - ema_short / ema_medium) * 5
                    reason = "Bearish trend in normal market"
            elif is_long:
                # Long position
                if ema_short < ema_medium or rsi > params["rsi_overbought"]:
                    signal = "SELL"
                    confidence = 0.7
                    reason = "Exit long in normal market"
            else:
                # Short position
                if ema_short > ema_medium or rsi < params["rsi_oversold"]:
                    signal = "BUY"
                    confidence = 0.7
                    reason = "Exit short in normal market"
        
        # Funding rate adjustment
        if abs(funding_rate) > params["funding_threshold"]:
            if funding_rate > 0 and signal == "BUY":
                # Positive funding rate reduces long confidence
                confidence *= 0.9
            elif funding_rate < 0 and signal == "SELL":
                # Negative funding rate reduces short confidence
                confidence *= 0.9
        
        # Mode-specific adjustments
        current_mode = self.mode_manager.get_current_mode()
        
        if current_mode == TradingMode.CONSERVATIVE:
            # More conservative signals
            if signal != "WAIT" and confidence < params["entry_confirmation_threshold"]:
                signal = "WAIT"
                reason = "Confidence below threshold for conservative mode"
        elif current_mode == TradingMode.AGGRESSIVE:
            # More aggressive signals
            if signal != "WAIT":
                confidence = min(0.95, confidence * 1.1)
        elif current_mode == TradingMode.MONITOR_ONLY:
            # No actual trades in monitor mode
            if signal != "WAIT":
                original_signal = signal
                signal = "MONITOR"
                reason = f"Would {original_signal} in normal mode"
        
        # Construct final signal
        return {
            "symbol": symbol,
            "signal": signal,
            "price": current_price,
            "confidence": confidence,
            "reason": reason,
            "market_state": self.market_state.copy(),
            "indicators": {
                "ema_short": ema_short,
                "ema_medium": ema_medium,
                "ema_long": ema_long,
                "rsi": rsi,
                "funding_rate": funding_rate
            },
            "params": params,
            "timestamp": time.time()
        }
    
    def calculate_position_size(self, signal: Dict[str, Any], account_info: Dict[str, Any]) -> float:
        """
        Calculate optimal position size based on signal and account information.
        
        Args:
            signal: Signal dictionary
            account_info: Account information dictionary
            
        Returns:
            Position size
        """
        # Get current mode settings
        mode_settings = self.mode_manager.get_mode_settings()
        risk_percent = mode_settings.get("risk_percent", 0.01)
        max_position_size = mode_settings.get("max_position_size", 1.0)
        
        # Extract data
        equity = account_info.get("equity", 0.0)
        confidence = signal.get("confidence", 0.5)
        current_price = signal.get("price", 0.0)
        
        if equity <= 0 or current_price <= 0:
            return 0.0
        
        # Adjust risk based on confidence and market regime
        confidence_factor = confidence / 0.7  # Normalize around 0.7
        regime_factor = 1.0
        
        if signal.get("market_state", {}).get("regime") == "volatile":
            regime_factor = 0.7
        elif signal.get("market_state", {}).get("regime") == "ranging":
            regime_factor = 0.9
        
        adjusted_risk = risk_percent * confidence_factor * regime_factor
        
        # Calculate position size
        risk_amount = equity * adjusted_risk
        position_value = risk_amount * 10  # 10x leverage implied
        
        # Convert to position size
        position_size = position_value / current_price
        
        # Apply maximum position size limit
        max_size_value = equity * max_position_size
        max_size = max_size_value / current_price
        
        return min(position_size, max_size)
    
    def update_performance(self, trade_result: Dict[str, Any]) -> None:
        """
        Update performance metrics based on trade result.
        
        Args:
            trade_result: Dict containing trade result
        """
        # Extract data
        profit = trade_result.get("profit", 0.0)
        is_win = profit > 0
        
        # Update metrics
        self.performance["total_trades"] += 1
        
        if is_win:
            self.performance["winning_trades"] += 1
        else:
            self.performance["losing_trades"] += 1
        
        self.performance["total_profit"] += profit
        
        # Calculate win rate
        if self.performance["total_trades"] > 0:
            self.performance["win_rate"] = self.performance["winning_trades"] / self.performance["total_trades"] * 100
        
        # Calculate average profit per trade
        if self.performance["total_trades"] > 0:
            self.performance["avg_profit_per_trade"] = self.performance["total_profit"] / self.performance["total_trades"]
        
        # Calculate profit factor
        if self.performance["losing_trades"] > 0:
            total_wins = trade_result.get("total_wins", 0.0)
            total_losses = trade_result.get("total_losses", 0.0)
            
            if total_losses != 0:
                self.performance["profit_factor"] = abs(total_wins / total_losses)
    
    def get_performance(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dict containing performance metrics
        """
        return self.performance.copy()
    
    def get_market_state(self) -> Dict[str, Any]:
        """
        Get current market state.
        
        Returns:
            Dict containing market state
        """
        return self.market_state.copy()
    
    def get_stop_loss_price(self, entry_price: float, is_long: bool) -> float:
        """
        Calculate stop loss price based on entry price and position direction.
        
        Args:
            entry_price: Entry price
            is_long: Whether position is long
            
        Returns:
            Stop loss price
        """
        params = self.get_current_params()
        stop_loss_percent = params["stop_loss"]
        
        if is_long:
            return entry_price * (1 - stop_loss_percent)
        else:
            return entry_price * (1 + stop_loss_percent)
    
    def get_take_profit_price(self, entry_price: float, is_long: bool) -> float:
        """
        Calculate take profit price based on entry price and position direction.
        
        Args:
            entry_price: Entry price
            is_long: Whether position is long
            
        Returns:
            Take profit price
        """
        params = self.get_current_params()
        profit_target = params["profit_target"]
        
        if is_long:
            return entry_price * (1 + profit_target)
        else:
            return entry_price * (1 - profit_target)
    
    def should_use_trailing_stop(self, unrealized_pnl: float, entry_price: float) -> bool:
        """
        Determine if trailing stop should be activated.
        
        Args:
            unrealized_pnl: Unrealized profit/loss
            entry_price: Entry price
            
        Returns:
            True if trailing stop should be activated, False otherwise
        """
        params = self.get_current_params()
        
        # Activate trailing stop if profit exceeds threshold
        pnl_percent = abs(unrealized_pnl / entry_price)
        return pnl_percent >= params["profit_target"] / 2
