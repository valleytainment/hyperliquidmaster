"""
Hull Suite Strategy
Advanced trend-following strategy using Hull Moving Average
Ported and enhanced from the analyzed repositories
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from strategies.base_strategy import BaseStrategy, TradingSignal, SignalType, MarketData, OrderType
from utils.logger import get_logger

logger = get_logger(__name__)


class HullSuiteStrategy(BaseStrategy):
    """
    Hull Suite Strategy
    
    This strategy uses the Hull Moving Average (HMA) for trend identification
    and Average True Range (ATR) for volatility-based position sizing and stops.
    
    The Hull Moving Average is designed to reduce lag while maintaining smoothness.
    It's calculated using weighted moving averages and provides faster signals
    than traditional moving averages.
    
    Entry Conditions:
    LONG: Price crosses above Hull MA and Hull MA is trending up (green)
    SHORT: Price crosses below Hull MA and Hull MA is trending down (red)
    
    Exit Conditions:
    - Hull MA changes color (trend reversal)
    - ATR-based stop loss hit
    - Take profit target reached
    """
    
    def __init__(self, name: str = "Hull_Suite", config=None, api_client=None):
        """Initialize Hull Suite strategy"""
        super().__init__(name, config, api_client)
        
        # Default configuration if none provided
        if config is None:
            from types import SimpleNamespace
            config = SimpleNamespace()
            config.indicators = {
                'hull_ma': {'period': 21},
                'atr': {'period': 14, 'multiplier': 1.5}
            }
            config.max_positions = 3
        
        # Strategy parameters from config
        indicators_config = config.indicators if hasattr(config, 'indicators') else {}
        
        # Hull MA parameters
        hull_config = indicators_config.get('hull_ma', {})
        self.hull_period = hull_config.get('period', 34)
        self.hull_source = hull_config.get('source', 'close')
        
        # ATR parameters
        atr_config = indicators_config.get('atr', {})
        self.atr_period = atr_config.get('period', 14)
        self.atr_multiplier = atr_config.get('multiplier', 2.0)
        
        # Additional parameters
        self.min_candles = max(self.hull_period, self.atr_period) + 10
        self.trend_confirmation_periods = 3  # Number of periods to confirm trend
        
        logger.info(f"Hull Suite Strategy initialized with Hull MA({self.hull_period}), "
                   f"ATR({self.atr_period}, {self.atr_multiplier})")
    
    def calculate_hull_ma(self, data: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate Hull Moving Average
        
        Hull MA formula:
        1. WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        
        Where:
        - WMA = Weighted Moving Average
        - n = period
        """
        def weighted_moving_average(series: pd.Series, period: int) -> pd.Series:
            """Calculate Weighted Moving Average"""
            weights = np.arange(1, period + 1)
            
            def wma_calc(x):
                if len(x) < period:
                    return np.nan
                return np.dot(x[-period:], weights) / weights.sum()
            
            return series.rolling(window=period).apply(wma_calc, raw=True)
        
        # Calculate WMA components
        half_period = int(self.hull_period / 2)
        sqrt_period = int(np.sqrt(self.hull_period))
        
        wma_half = weighted_moving_average(data, half_period)
        wma_full = weighted_moving_average(data, self.hull_period)
        
        # Calculate Hull MA
        hull_raw = 2 * wma_half - wma_full
        hull_ma = weighted_moving_average(hull_raw, sqrt_period)
        
        # Determine Hull MA color (trend direction)
        hull_color = pd.Series(index=hull_ma.index, dtype=str)
        hull_color[hull_ma > hull_ma.shift(1)] = 'green'  # Uptrend
        hull_color[hull_ma < hull_ma.shift(1)] = 'red'    # Downtrend
        hull_color[hull_ma == hull_ma.shift(1)] = 'gray'  # Sideways
        
        # Calculate Hull MA slope for trend strength
        hull_slope = hull_ma.diff()
        hull_slope_normalized = hull_slope / data * 100  # Normalize by price
        
        return {
            'hull_ma': hull_ma,
            'hull_color': hull_color,
            'hull_slope': hull_slope,
            'hull_slope_normalized': hull_slope_normalized,
            'hull_raw': hull_raw
        }
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """Calculate Average True Range and related indicators"""
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Average True Range
        atr = true_range.rolling(window=self.atr_period).mean()
        
        # ATR-based levels
        atr_upper = close + (atr * self.atr_multiplier)
        atr_lower = close - (atr * self.atr_multiplier)
        
        # ATR percentage (volatility measure)
        atr_percentage = (atr / close) * 100
        
        return {
            'atr': atr,
            'atr_upper': atr_upper,
            'atr_lower': atr_lower,
            'atr_percentage': atr_percentage,
            'true_range': true_range
        }
    
    def calculate_indicators(self, coin: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all technical indicators"""
        if len(data) < self.min_candles:
            return {}
        
        # Get price data
        close = data['close']
        high = data['high']
        low = data['low']
        
        # Use specified source for Hull MA
        if self.hull_source == 'high':
            hull_source_data = high
        elif self.hull_source == 'low':
            hull_source_data = low
        elif self.hull_source == 'hl2':
            hull_source_data = (high + low) / 2
        elif self.hull_source == 'hlc3':
            hull_source_data = (high + low + close) / 3
        elif self.hull_source == 'ohlc4':
            hull_source_data = (data['open'] + high + low + close) / 4
        else:  # default to close
            hull_source_data = close
        
        # Calculate Hull MA
        hull_indicators = self.calculate_hull_ma(hull_source_data)
        
        # Calculate ATR
        atr_indicators = self.calculate_atr(high, low, close)
        
        # Combine all indicators
        indicators = {
            **hull_indicators,
            **atr_indicators
        }
        
        # Store indicators for this coin
        self.indicators[coin] = indicators
        
        return indicators
    
    def detect_hull_cross(self, price: pd.Series, hull_ma: pd.Series, 
                         periods: int = 2) -> str:
        """
        Detect price crossing Hull MA
        
        Args:
            price: Price series
            hull_ma: Hull MA series
            periods: Number of periods to look back for confirmation
            
        Returns:
            'bullish_cross', 'bearish_cross', or 'none'
        """
        if len(price) < periods + 1 or len(hull_ma) < periods + 1:
            return 'none'
        
        # Current and previous values
        current_price = price.iloc[-1]
        current_hull = hull_ma.iloc[-1]
        prev_price = price.iloc[-2]
        prev_hull = hull_ma.iloc[-2]
        
        # Check for bullish cross (price crosses above Hull MA)
        if prev_price <= prev_hull and current_price > current_hull:
            return 'bullish_cross'
        
        # Check for bearish cross (price crosses below Hull MA)
        elif prev_price >= prev_hull and current_price < current_hull:
            return 'bearish_cross'
        
        return 'none'
    
    def confirm_trend_direction(self, hull_color: pd.Series, periods: int = 3) -> str:
        """
        Confirm trend direction based on Hull MA color consistency
        
        Args:
            hull_color: Hull MA color series
            periods: Number of periods to check for consistency
            
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        if len(hull_color) < periods:
            return 'neutral'
        
        recent_colors = hull_color.iloc[-periods:]
        
        # Check for consistent bullish trend
        if all(color == 'green' for color in recent_colors):
            return 'bullish'
        
        # Check for consistent bearish trend
        elif all(color == 'red' for color in recent_colors):
            return 'bearish'
        
        return 'neutral'
    
    def calculate_signal_confidence(self, indicators: Dict[str, Any], 
                                  signal_type: SignalType, price: float) -> float:
        """
        Calculate signal confidence based on Hull Suite indicators
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_type: Type of signal being generated
            price: Current price
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.0
        
        # Get latest values
        hull_ma = indicators['hull_ma'].iloc[-1]
        hull_color = indicators['hull_color'].iloc[-1]
        hull_slope = indicators['hull_slope_normalized'].iloc[-1]
        atr_percentage = indicators['atr_percentage'].iloc[-1]
        
        if signal_type == SignalType.LONG:
            # Hull MA color confirmation
            if hull_color == 'green':
                confidence += 0.3
            
            # Price above Hull MA
            if price > hull_ma:
                confidence += 0.2
            
            # Hull MA slope (positive slope for uptrend)
            if hull_slope > 0:
                confidence += 0.2
                if hull_slope > 0.1:  # Strong uptrend
                    confidence += 0.1
            
            # Trend consistency
            trend_direction = self.confirm_trend_direction(indicators['hull_color'])
            if trend_direction == 'bullish':
                confidence += 0.2
            
            # Volatility consideration (moderate volatility preferred)
            if 1.0 <= atr_percentage <= 5.0:
                confidence += 0.1
        
        elif signal_type == SignalType.SHORT:
            # Hull MA color confirmation
            if hull_color == 'red':
                confidence += 0.3
            
            # Price below Hull MA
            if price < hull_ma:
                confidence += 0.2
            
            # Hull MA slope (negative slope for downtrend)
            if hull_slope < 0:
                confidence += 0.2
                if hull_slope < -0.1:  # Strong downtrend
                    confidence += 0.1
            
            # Trend consistency
            trend_direction = self.confirm_trend_direction(indicators['hull_color'])
            if trend_direction == 'bearish':
                confidence += 0.2
            
            # Volatility consideration (moderate volatility preferred)
            if 1.0 <= atr_percentage <= 5.0:
                confidence += 0.1
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    async def generate_signal(self, coin: str, market_data: List[MarketData]) -> TradingSignal:
        """Generate trading signal based on Hull Suite analysis"""
        
        # Convert market data to DataFrame
        if len(market_data) < self.min_candles:
            return TradingSignal(
                signal_type=SignalType.NONE,
                coin=coin,
                confidence=0.0,
                metadata={'reason': 'insufficient_data'}
            )
        
        df = pd.DataFrame([d.to_dict() for d in market_data])
        df.set_index('timestamp', inplace=True)
        
        # Calculate indicators
        indicators = self.calculate_indicators(coin, df)
        if not indicators:
            return TradingSignal(
                signal_type=SignalType.NONE,
                coin=coin,
                confidence=0.0,
                metadata={'reason': 'indicator_calculation_failed'}
            )
        
        # Get current values
        current_price = df['close'].iloc[-1]
        current_hull_ma = indicators['hull_ma'].iloc[-1]
        current_hull_color = indicators['hull_color'].iloc[-1]
        current_atr = indicators['atr'].iloc[-1]
        
        # Check for NaN values
        if pd.isna(current_hull_ma) or pd.isna(current_atr):
            return TradingSignal(
                signal_type=SignalType.NONE,
                coin=coin,
                confidence=0.0,
                metadata={'reason': 'nan_indicators'}
            )
        
        # Detect Hull MA crosses
        hull_cross = self.detect_hull_cross(df['close'], indicators['hull_ma'])
        
        # Confirm trend direction
        trend_direction = self.confirm_trend_direction(indicators['hull_color'], 
                                                     self.trend_confirmation_periods)
        
        signal_type = SignalType.NONE
        metadata = {
            'hull_ma': current_hull_ma,
            'hull_color': current_hull_color,
            'hull_cross': hull_cross,
            'trend_direction': trend_direction,
            'atr': current_atr,
            'atr_percentage': indicators['atr_percentage'].iloc[-1],
            'price': current_price
        }
        
        # Check for existing position to determine exit signals
        existing_position = self.positions.get(coin, {})
        position_size = existing_position.get('size', 0)
        
        # Exit signal logic
        if position_size != 0:
            if position_size > 0:  # Long position
                # Exit long if Hull MA turns red or bearish cross
                if current_hull_color == 'red' or hull_cross == 'bearish_cross':
                    signal_type = SignalType.CLOSE_LONG
                    metadata['exit_reason'] = 'hull_bearish' if current_hull_color == 'red' else 'bearish_cross'
            
            elif position_size < 0:  # Short position
                # Exit short if Hull MA turns green or bullish cross
                if current_hull_color == 'green' or hull_cross == 'bullish_cross':
                    signal_type = SignalType.CLOSE_SHORT
                    metadata['exit_reason'] = 'hull_bullish' if current_hull_color == 'green' else 'bullish_cross'
        
        # Entry signal logic (only if no existing position)
        else:
            # Long signal conditions
            if (hull_cross == 'bullish_cross' and 
                current_hull_color == 'green' and 
                trend_direction in ['bullish', 'neutral']):
                signal_type = SignalType.LONG
                metadata['entry_reason'] = 'bullish_cross_green_hull'
            
            # Short signal conditions
            elif (hull_cross == 'bearish_cross' and 
                  current_hull_color == 'red' and 
                  trend_direction in ['bearish', 'neutral']):
                signal_type = SignalType.SHORT
                metadata['entry_reason'] = 'bearish_cross_red_hull'
        
        # Calculate confidence
        confidence = 0.0
        if signal_type != SignalType.NONE:
            if signal_type in [SignalType.LONG, SignalType.SHORT]:
                confidence = self.calculate_signal_confidence(indicators, signal_type, current_price)
            else:  # Exit signals
                confidence = 0.9  # High confidence for exit signals based on Hull MA
        
        # Create signal
        signal = TradingSignal(
            signal_type=signal_type,
            coin=coin,
            confidence=confidence,
            metadata=metadata
        )
        
        # Add stop loss and take profit for entry signals
        if signal_type in [SignalType.LONG, SignalType.SHORT]:
            signal.entry_price = current_price
            
            # ATR-based stop loss
            if signal_type == SignalType.LONG:
                signal.stop_loss = current_price - (current_atr * self.atr_multiplier)
                signal.take_profit = current_price + (current_atr * self.atr_multiplier * 1.5)
            else:  # SHORT
                signal.stop_loss = current_price + (current_atr * self.atr_multiplier)
                signal.take_profit = current_price - (current_atr * self.atr_multiplier * 1.5)
            
            signal.size = self.calculate_position_size(coin, signal)
        
        # Add to signal history
        self.add_signal_to_history(signal)
        
        # Log signal if not NONE
        if signal_type != SignalType.NONE:
            logger.info(f"Hull Suite Signal: {coin} {signal_type.value} "
                       f"(confidence: {confidence:.2f}, Hull: {current_hull_color}, "
                       f"Cross: {hull_cross}, Trend: {trend_direction})")
        
        return signal
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and current state"""
        return {
            'name': self.name,
            'type': 'Hull_Suite',
            'parameters': {
                'hull_period': self.hull_period,
                'hull_source': self.hull_source,
                'atr_period': self.atr_period,
                'atr_multiplier': self.atr_multiplier,
                'trend_confirmation_periods': self.trend_confirmation_periods
            },
            'status': self.get_status(),
            'description': 'Hull Moving Average trend-following strategy with ATR-based stops'
        }

