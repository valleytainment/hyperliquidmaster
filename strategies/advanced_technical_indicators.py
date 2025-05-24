"""
Advanced Technical Indicators with Robust Data Type Handling

This module provides a comprehensive set of technical indicators with robust data type handling
to ensure reliable calculations regardless of input data format.
"""

import numpy as np
import pandas as pd
import logging
from typing import Union, Dict, List, Tuple, Optional, Any
from .data_type_handling import (
    ensure_pandas_series, ensure_pandas_dataframe, safe_shift, safe_rolling,
    safe_ewm, safe_diff, safe_pct_change, extract_column, get_ohlc, handle_missing_values
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class AdvancedTechnicalIndicators:
    """
    Advanced technical indicators with robust data type handling.
    """
    
    @staticmethod
    def sma(data: Any, period: int = 14) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).
        
        Args:
            data: Price data
            period: SMA period
            
        Returns:
            SMA values as pandas Series
        """
        try:
            series = ensure_pandas_series(data)
            return safe_rolling(series, period).mean()
        except Exception as e:
            logger.error(f"Error calculating SMA: {str(e)}")
            return pd.Series(np.nan, index=ensure_pandas_series(data).index)
    
    @staticmethod
    def ema(data: Any, period: int = 14) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).
        
        Args:
            data: Price data
            period: EMA period
            
        Returns:
            EMA values as pandas Series
        """
        try:
            series = ensure_pandas_series(data)
            return safe_ewm(series, span=period).mean()
        except Exception as e:
            logger.error(f"Error calculating EMA: {str(e)}")
            return pd.Series(np.nan, index=ensure_pandas_series(data).index)
    
    @staticmethod
    def wma(data: Any, period: int = 14) -> pd.Series:
        """
        Calculate Weighted Moving Average (WMA).
        
        Args:
            data: Price data
            period: WMA period
            
        Returns:
            WMA values as pandas Series
        """
        try:
            series = ensure_pandas_series(data)
            weights = np.arange(1, period + 1)
            return safe_rolling(series, period).apply(
                lambda x: np.sum(weights * x) / weights.sum(), raw=True
            )
        except Exception as e:
            logger.error(f"Error calculating WMA: {str(e)}")
            return pd.Series(np.nan, index=ensure_pandas_series(data).index)
    
    @staticmethod
    def vwma(data: Any, volume: Any, period: int = 14) -> pd.Series:
        """
        Calculate Volume Weighted Moving Average (VWMA).
        
        Args:
            data: Price data
            volume: Volume data
            period: VWMA period
            
        Returns:
            VWMA values as pandas Series
        """
        try:
            price_series = ensure_pandas_series(data)
            volume_series = ensure_pandas_series(volume)
            
            # Ensure both series have the same index
            if not price_series.index.equals(volume_series.index):
                volume_series = volume_series.reindex(price_series.index)
                
            vp = price_series * volume_series
            return safe_rolling(vp, period).sum() / safe_rolling(volume_series, period).sum()
        except Exception as e:
            logger.error(f"Error calculating VWMA: {str(e)}")
            return pd.Series(np.nan, index=ensure_pandas_series(data).index)
    
    @staticmethod
    def rsi(data: Any, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: Price data
            period: RSI period
            
        Returns:
            RSI values as pandas Series
        """
        try:
            series = ensure_pandas_series(data)
            delta = safe_diff(series)
            
            # Make two series: one for gains, one for losses
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            # Calculate average gain and average loss
            avg_gain = safe_ewm(gain, span=period).mean()
            avg_loss = safe_ewm(loss, span=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series(np.nan, index=ensure_pandas_series(data).index)
    
    @staticmethod
    def macd(data: Any, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            data: Price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal EMA period
            
        Returns:
            Dictionary with MACD, signal, and histogram values
        """
        try:
            series = ensure_pandas_series(data)
            
            # Calculate fast and slow EMAs
            fast_ema = AdvancedTechnicalIndicators.ema(series, fast_period)
            slow_ema = AdvancedTechnicalIndicators.ema(series, slow_period)
            
            # Calculate MACD line
            macd_line = fast_ema - slow_ema
            
            # Calculate signal line
            signal_line = AdvancedTechnicalIndicators.ema(macd_line, signal_period)
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            empty_series = pd.Series(np.nan, index=ensure_pandas_series(data).index)
            return {
                'macd': empty_series,
                'signal': empty_series,
                'histogram': empty_series
            }
    
    @staticmethod
    def bollinger_bands(data: Any, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: Price data
            period: SMA period
            std_dev: Standard deviation multiplier
            
        Returns:
            Dictionary with upper, middle, and lower band values
        """
        try:
            series = ensure_pandas_series(data)
            
            # Calculate middle band (SMA)
            middle_band = AdvancedTechnicalIndicators.sma(series, period)
            
            # Calculate standard deviation
            rolling_std = safe_rolling(series, period).std()
            
            # Calculate upper and lower bands
            upper_band = middle_band + (rolling_std * std_dev)
            lower_band = middle_band - (rolling_std * std_dev)
            
            return {
                'upper': upper_band,
                'middle': middle_band,
                'lower': lower_band
            }
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            empty_series = pd.Series(np.nan, index=ensure_pandas_series(data).index)
            return {
                'upper': empty_series,
                'middle': empty_series,
                'lower': empty_series
            }
    
    @staticmethod
    def stochastic(high: Any, low: Any, close: Any, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high: High price data
            low: Low price data
            close: Close price data
            k_period: %K period
            d_period: %D period
            
        Returns:
            Dictionary with %K and %D values
        """
        try:
            high_series = ensure_pandas_series(high)
            low_series = ensure_pandas_series(low)
            close_series = ensure_pandas_series(close)
            
            # Ensure all series have the same index
            if not high_series.index.equals(low_series.index) or not high_series.index.equals(close_series.index):
                index = high_series.index
                low_series = low_series.reindex(index)
                close_series = close_series.reindex(index)
            
            # Calculate %K
            lowest_low = safe_rolling(low_series, k_period).min()
            highest_high = safe_rolling(high_series, k_period).max()
            k = 100 * ((close_series - lowest_low) / (highest_high - lowest_low))
            
            # Calculate %D
            d = AdvancedTechnicalIndicators.sma(k, d_period)
            
            return {
                'k': k,
                'd': d
            }
        except Exception as e:
            logger.error(f"Error calculating Stochastic Oscillator: {str(e)}")
            empty_series = pd.Series(np.nan, index=ensure_pandas_series(close).index)
            return {
                'k': empty_series,
                'd': empty_series
            }
    
    @staticmethod
    def atr(high: Any, low: Any, close: Any, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            high: High price data
            low: Low price data
            close: Close price data
            period: ATR period
            
        Returns:
            ATR values as pandas Series
        """
        try:
            high_series = ensure_pandas_series(high)
            low_series = ensure_pandas_series(low)
            close_series = ensure_pandas_series(close)
            
            # Ensure all series have the same index
            if not high_series.index.equals(low_series.index) or not high_series.index.equals(close_series.index):
                index = high_series.index
                low_series = low_series.reindex(index)
                close_series = close_series.reindex(index)
            
            # Calculate true range
            prev_close = safe_shift(close_series, 1)
            tr1 = high_series - low_series
            tr2 = (high_series - prev_close).abs()
            tr3 = (low_series - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR
            atr = safe_ewm(tr, span=period).mean()
            
            return atr
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series(np.nan, index=ensure_pandas_series(close).index)
    
    @staticmethod
    def adx(high: Any, low: Any, close: Any, period: int = 14) -> Dict[str, pd.Series]:
        """
        Calculate Average Directional Index (ADX).
        
        Args:
            high: High price data
            low: Low price data
            close: Close price data
            period: ADX period
            
        Returns:
            Dictionary with ADX, +DI, and -DI values
        """
        try:
            high_series = ensure_pandas_series(high)
            low_series = ensure_pandas_series(low)
            close_series = ensure_pandas_series(close)
            
            # Ensure all series have the same index
            if not high_series.index.equals(low_series.index) or not high_series.index.equals(close_series.index):
                index = high_series.index
                low_series = low_series.reindex(index)
                close_series = close_series.reindex(index)
            
            # Calculate +DM and -DM
            high_diff = high_series - safe_shift(high_series, 1)
            low_diff = safe_shift(low_series, 1) - low_series
            
            plus_dm = ((high_diff > low_diff) & (high_diff > 0)) * high_diff
            minus_dm = ((low_diff > high_diff) & (low_diff > 0)) * low_diff
            
            # Calculate ATR
            atr = AdvancedTechnicalIndicators.atr(high_series, low_series, close_series, period)
            
            # Calculate +DI and -DI
            plus_di = 100 * safe_ewm(plus_dm, span=period).mean() / atr
            minus_di = 100 * safe_ewm(minus_dm, span=period).mean() / atr
            
            # Calculate DX and ADX
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
            adx = safe_ewm(dx, span=period).mean()
            
            return {
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di
            }
        except Exception as e:
            logger.error(f"Error calculating ADX: {str(e)}")
            empty_series = pd.Series(np.nan, index=ensure_pandas_series(close).index)
            return {
                'adx': empty_series,
                'plus_di': empty_series,
                'minus_di': empty_series
            }
    
    @staticmethod
    def ichimoku(high: Any, low: Any, close: Any, 
                conversion_period: int = 9, 
                base_period: int = 26, 
                span_b_period: int = 52, 
                displacement: int = 26) -> Dict[str, pd.Series]:
        """
        Calculate Ichimoku Cloud.
        
        Args:
            high: High price data
            low: Low price data
            close: Close price data
            conversion_period: Conversion line period (Tenkan-sen)
            base_period: Base line period (Kijun-sen)
            span_b_period: Span B period (Senkou Span B)
            displacement: Displacement period
            
        Returns:
            Dictionary with Ichimoku components
        """
        try:
            high_series = ensure_pandas_series(high)
            low_series = ensure_pandas_series(low)
            close_series = ensure_pandas_series(close)
            
            # Ensure all series have the same index
            if not high_series.index.equals(low_series.index) or not high_series.index.equals(close_series.index):
                index = high_series.index
                low_series = low_series.reindex(index)
                close_series = close_series.reindex(index)
            
            # Calculate Conversion Line (Tenkan-sen)
            conversion_high = safe_rolling(high_series, conversion_period).max()
            conversion_low = safe_rolling(low_series, conversion_period).min()
            conversion_line = (conversion_high + conversion_low) / 2
            
            # Calculate Base Line (Kijun-sen)
            base_high = safe_rolling(high_series, base_period).max()
            base_low = safe_rolling(low_series, base_period).min()
            base_line = (base_high + base_low) / 2
            
            # Calculate Leading Span A (Senkou Span A)
            leading_span_a = ((conversion_line + base_line) / 2).shift(displacement)
            
            # Calculate Leading Span B (Senkou Span B)
            span_b_high = safe_rolling(high_series, span_b_period).max()
            span_b_low = safe_rolling(low_series, span_b_period).min()
            leading_span_b = ((span_b_high + span_b_low) / 2).shift(displacement)
            
            # Calculate Lagging Span (Chikou Span)
            lagging_span = close_series.shift(-displacement)
            
            return {
                'conversion_line': conversion_line,
                'base_line': base_line,
                'leading_span_a': leading_span_a,
                'leading_span_b': leading_span_b,
                'lagging_span': lagging_span
            }
        except Exception as e:
            logger.error(f"Error calculating Ichimoku Cloud: {str(e)}")
            empty_series = pd.Series(np.nan, index=ensure_pandas_series(close).index)
            return {
                'conversion_line': empty_series,
                'base_line': empty_series,
                'leading_span_a': empty_series,
                'leading_span_b': empty_series,
                'lagging_span': empty_series
            }
    
    @staticmethod
    def supertrend(high: Any, low: Any, close: Any, period: int = 10, multiplier: float = 3.0) -> Dict[str, pd.Series]:
        """
        Calculate SuperTrend indicator.
        
        Args:
            high: High price data
            low: Low price data
            close: Close price data
            period: ATR period
            multiplier: ATR multiplier
            
        Returns:
            Dictionary with SuperTrend values and trend direction
        """
        try:
            high_series = ensure_pandas_series(high)
            low_series = ensure_pandas_series(low)
            close_series = ensure_pandas_series(close)
            
            # Ensure all series have the same index
            if not high_series.index.equals(low_series.index) or not high_series.index.equals(close_series.index):
                index = high_series.index
                low_series = low_series.reindex(index)
                close_series = close_series.reindex(index)
            
            # Calculate ATR
            atr = AdvancedTechnicalIndicators.atr(high_series, low_series, close_series, period)
            
            # Calculate basic upper and lower bands
            hl2 = (high_series + low_series) / 2
            basic_upper = hl2 + (multiplier * atr)
            basic_lower = hl2 - (multiplier * atr)
            
            # Initialize SuperTrend
            supertrend = pd.Series(0.0, index=close_series.index)
            direction = pd.Series(1, index=close_series.index)  # 1 for uptrend, -1 for downtrend
            
            # Calculate SuperTrend
            for i in range(1, len(close_series)):
                if close_series.iloc[i] > basic_upper.iloc[i-1]:
                    direction.iloc[i] = 1
                elif close_series.iloc[i] < basic_lower.iloc[i-1]:
                    direction.iloc[i] = -1
                else:
                    direction.iloc[i] = direction.iloc[i-1]
                    
                    if direction.iloc[i] == 1 and basic_lower.iloc[i] < basic_lower.iloc[i-1]:
                        basic_lower.iloc[i] = basic_lower.iloc[i-1]
                    if direction.iloc[i] == -1 and basic_upper.iloc[i] > basic_upper.iloc[i-1]:
                        basic_upper.iloc[i] = basic_upper.iloc[i-1]
                
                if direction.iloc[i] == 1:
                    supertrend.iloc[i] = basic_lower.iloc[i]
                else:
                    supertrend.iloc[i] = basic_upper.iloc[i]
            
            return {
                'supertrend': supertrend,
                'direction': direction
            }
        except Exception as e:
            logger.error(f"Error calculating SuperTrend: {str(e)}")
            empty_series = pd.Series(np.nan, index=ensure_pandas_series(close).index)
            return {
                'supertrend': empty_series,
                'direction': empty_series
            }
    
    @staticmethod
    def detect_market_regime(close: Any, high: Any = None, low: Any = None, volume: Any = None, 
                           period: int = 20) -> Dict[str, Union[str, float]]:
        """
        Detect market regime (trending, ranging, volatile).
        
        Args:
            close: Close price data
            high: High price data (optional)
            low: Low price data (optional)
            volume: Volume data (optional)
            period: Analysis period
            
        Returns:
            Dictionary with market regime information
        """
        try:
            close_series = ensure_pandas_series(close)
            
            # Calculate ADX if high and low are provided
            adx_value = None
            if high is not None and low is not None:
                high_series = ensure_pandas_series(high)
                low_series = ensure_pandas_series(low)
                adx_result = AdvancedTechnicalIndicators.adx(high_series, low_series, close_series, period)
                adx_value = adx_result['adx'].iloc[-1] if not adx_result['adx'].empty else None
            
            # Calculate Bollinger Bands
            bb_result = AdvancedTechnicalIndicators.bollinger_bands(close_series, period)
            
            # Calculate BB width
            bb_width = (bb_result['upper'] - bb_result['lower']) / bb_result['middle']
            current_bb_width = bb_width.iloc[-1] if not bb_width.empty else None
            
            # Calculate price volatility
            returns = safe_pct_change(close_series)
            volatility = safe_rolling(returns, period).std().iloc[-1] * np.sqrt(252)
            
            # Calculate RSI
            rsi = AdvancedTechnicalIndicators.rsi(close_series, period).iloc[-1]
            
            # Determine market regime
            regime = "unknown"
            confidence = 0.0
            
            if adx_value is not None and not np.isnan(adx_value):
                if adx_value > 25:
                    regime = "trending"
                    confidence = min(1.0, (adx_value - 25) / 25)
                elif adx_value < 20:
                    regime = "ranging"
                    confidence = min(1.0, (20 - adx_value) / 10)
            
            # Adjust based on BB width
            if current_bb_width is not None and not np.isnan(current_bb_width):
                bb_width_mean = safe_rolling(bb_width, period * 2).mean().iloc[-1]
                
                if current_bb_width > bb_width_mean * 1.5:
                    if regime != "trending":
                        regime = "volatile"
                        confidence = min(1.0, (current_bb_width / bb_width_mean - 1.5) / 0.5)
                elif current_bb_width < bb_width_mean * 0.75:
                    if regime != "trending":
                        regime = "ranging"
                        confidence = min(1.0, (1 - current_bb_width / bb_width_mean) / 0.25)
            
            # Adjust based on RSI
            if not np.isnan(rsi):
                if rsi > 70 or rsi < 30:
                    # Extreme RSI values suggest potential trend reversal
                    reversal_probability = min(1.0, (max(abs(rsi - 50) - 20, 0)) / 20)
                    
                    if reversal_probability > confidence:
                        regime = "reversal_potential"
                        confidence = reversal_probability
            
            return {
                'regime': regime,
                'confidence': confidence,
                'adx': adx_value,
                'bb_width': current_bb_width,
                'volatility': volatility,
                'rsi': rsi
            }
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return {
                'regime': "unknown",
                'confidence': 0.0,
                'adx': None,
                'bb_width': None,
                'volatility': None,
                'rsi': None
            }
    
    @staticmethod
    def generate_master_signal(close: Any, high: Any = None, low: Any = None, volume: Any = None,
                             order_book: Dict = None, funding_rate: float = None) -> Dict[str, Any]:
        """
        Generate a master trading signal combining multiple indicators.
        
        Args:
            close: Close price data
            high: High price data (optional)
            low: Low price data (optional)
            volume: Volume data (optional)
            order_book: Order book data (optional)
            funding_rate: Funding rate (optional)
            
        Returns:
            Dictionary with signal information
        """
        try:
            close_series = ensure_pandas_series(close)
            
            # Prepare high and low series if provided
            high_series = ensure_pandas_series(high) if high is not None else None
            low_series = ensure_pandas_series(low) if low is not None else None
            volume_series = ensure_pandas_series(volume) if volume is not None else None
            
            # Calculate technical indicators
            signals = {}
            
            # Trend indicators
            macd_result = AdvancedTechnicalIndicators.macd(close_series)
            macd_signal = 1 if macd_result['histogram'].iloc[-1] > 0 else -1
            signals['macd'] = macd_signal
            
            # Momentum indicators
            rsi = AdvancedTechnicalIndicators.rsi(close_series)
            rsi_value = rsi.iloc[-1]
            if rsi_value > 70:
                rsi_signal = -1  # Overbought
            elif rsi_value < 30:
                rsi_signal = 1   # Oversold
            else:
                rsi_signal = 0   # Neutral
            signals['rsi'] = rsi_signal
            
            # Volatility indicators
            if high_series is not None and low_series is not None:
                bb_result = AdvancedTechnicalIndicators.bollinger_bands(close_series)
                if close_series.iloc[-1] < bb_result['lower'].iloc[-1]:
                    bb_signal = 1  # Price below lower band
                elif close_series.iloc[-1] > bb_result['upper'].iloc[-1]:
                    bb_signal = -1  # Price above upper band
                else:
                    bb_signal = 0  # Price within bands
                signals['bollinger'] = bb_signal
                
                # SuperTrend
                supertrend_result = AdvancedTechnicalIndicators.supertrend(high_series, low_series, close_series)
                supertrend_signal = supertrend_result['direction'].iloc[-1]
                signals['supertrend'] = supertrend_signal
            
            # Order book analysis
            if order_book is not None:
                try:
                    bids = order_book.get('bids', [])
                    asks = order_book.get('asks', [])
                    
                    if bids and asks:
                        # Calculate bid-ask imbalance
                        bid_volume = sum(bid['quantity'] for bid in bids[:5])
                        ask_volume = sum(ask['quantity'] for ask in asks[:5])
                        
                        if bid_volume > ask_volume * 1.5:
                            ob_signal = 1  # Strong buying pressure
                        elif ask_volume > bid_volume * 1.5:
                            ob_signal = -1  # Strong selling pressure
                        else:
                            ob_signal = 0  # Balanced
                        
                        signals['order_book'] = ob_signal
                except Exception as e:
                    logger.warning(f"Error analyzing order book: {str(e)}")
            
            # Funding rate analysis
            if funding_rate is not None:
                try:
                    if funding_rate < -0.01:
                        funding_signal = 1  # Negative funding rate favors longs
                    elif funding_rate > 0.01:
                        funding_signal = -1  # Positive funding rate favors shorts
                    else:
                        funding_signal = 0  # Neutral
                    
                    signals['funding'] = funding_signal
                except Exception as e:
                    logger.warning(f"Error analyzing funding rate: {str(e)}")
            
            # Detect market regime
            regime_result = AdvancedTechnicalIndicators.detect_market_regime(
                close_series, high_series, low_series, volume_series
            )
            
            # Calculate weighted signal based on market regime
            weights = {
                'trending': {
                    'macd': 0.3,
                    'rsi': 0.1,
                    'bollinger': 0.1,
                    'supertrend': 0.3,
                    'order_book': 0.1,
                    'funding': 0.1
                },
                'ranging': {
                    'macd': 0.1,
                    'rsi': 0.3,
                    'bollinger': 0.3,
                    'supertrend': 0.1,
                    'order_book': 0.1,
                    'funding': 0.1
                },
                'volatile': {
                    'macd': 0.1,
                    'rsi': 0.2,
                    'bollinger': 0.2,
                    'supertrend': 0.1,
                    'order_book': 0.3,
                    'funding': 0.1
                },
                'reversal_potential': {
                    'macd': 0.2,
                    'rsi': 0.3,
                    'bollinger': 0.2,
                    'supertrend': 0.1,
                    'order_book': 0.1,
                    'funding': 0.1
                },
                'unknown': {
                    'macd': 0.2,
                    'rsi': 0.2,
                    'bollinger': 0.2,
                    'supertrend': 0.2,
                    'order_book': 0.1,
                    'funding': 0.1
                }
            }
            
            regime = regime_result['regime']
            regime_weights = weights.get(regime, weights['unknown'])
            
            # Calculate weighted signal
            weighted_signal = 0
            total_weight = 0
            
            for indicator, signal in signals.items():
                if indicator in regime_weights:
                    weight = regime_weights[indicator]
                    weighted_signal += signal * weight
                    total_weight += weight
            
            if total_weight > 0:
                final_signal = weighted_signal / total_weight
            else:
                final_signal = 0
            
            # Determine signal strength and direction
            if final_signal > 0.5:
                signal_type = "strong_buy"
            elif final_signal > 0.2:
                signal_type = "buy"
            elif final_signal < -0.5:
                signal_type = "strong_sell"
            elif final_signal < -0.2:
                signal_type = "sell"
            else:
                signal_type = "neutral"
            
            return {
                'signal': final_signal,
                'signal_type': signal_type,
                'regime': regime,
                'confidence': regime_result['confidence'],
                'indicators': signals,
                'timestamp': pd.Timestamp.now().timestamp()
            }
        except Exception as e:
            logger.error(f"Error generating master signal: {str(e)}")
            return {
                'signal': 0,
                'signal_type': "error",
                'regime': "unknown",
                'confidence': 0,
                'indicators': {},
                'timestamp': pd.Timestamp.now().timestamp(),
                'error': str(e)
            }
