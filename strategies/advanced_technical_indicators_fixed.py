"""
Advanced Technical Indicators with Robust Data Type Handling

This module provides a comprehensive set of technical indicators with robust data type handling
to ensure reliable calculations regardless of input data format.
"""

import numpy as np
import pandas as pd
import logging
from typing import Union, Dict, List, Tuple, Optional, Any

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
    def ensure_pandas_series(data: Any) -> pd.Series:
        """
        Ensure data is a pandas Series.
        
        Args:
            data: Input data
            
        Returns:
            Pandas Series
        """
        if isinstance(data, pd.Series):
            return data
        elif isinstance(data, pd.DataFrame):
            if len(data.columns) == 1:
                return data.iloc[:, 0]
            else:
                return data['close'] if 'close' in data.columns else data.iloc[:, 0]
        elif isinstance(data, (list, tuple, np.ndarray)):
            return pd.Series(data)
        else:
            return pd.Series([data])
    
    @staticmethod
    def safe_rolling(series: pd.Series, window: int, min_periods: int = 1):
        """
        Safely apply rolling window.
        
        Args:
            series: Input series
            window: Window size
            min_periods: Minimum periods
            
        Returns:
            Rolling window object
        """
        if len(series) < window:
            min_periods = min(min_periods, len(series))
        return series.rolling(window=window, min_periods=min_periods)
    
    @staticmethod
    def safe_ewm(series: pd.Series, span: int, min_periods: int = 1):
        """
        Safely apply exponential weighted window.
        
        Args:
            series: Input series
            span: Span
            min_periods: Minimum periods
            
        Returns:
            EWM object
        """
        if len(series) < span:
            min_periods = min(min_periods, len(series))
        return series.ewm(span=span, min_periods=min_periods)
    
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
            series = AdvancedTechnicalIndicators.ensure_pandas_series(data)
            return AdvancedTechnicalIndicators.safe_rolling(series, period).mean()
        except Exception as e:
            logger.error(f"Error calculating SMA: {str(e)}")
            return pd.Series(np.nan, index=AdvancedTechnicalIndicators.ensure_pandas_series(data).index)
    
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
            series = AdvancedTechnicalIndicators.ensure_pandas_series(data)
            return AdvancedTechnicalIndicators.safe_ewm(series, span=period).mean()
        except Exception as e:
            logger.error(f"Error calculating EMA: {str(e)}")
            return pd.Series(np.nan, index=AdvancedTechnicalIndicators.ensure_pandas_series(data).index)
    
    @staticmethod
    def calculate_rsi(data: Any, period: int = 14) -> List[float]:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: Price data
            period: RSI period
            
        Returns:
            RSI values as list
        """
        try:
            if data is None:
                raise ValueError("Input data cannot be None")
                
            # Convert to numpy array if needed
            if isinstance(data, (list, tuple)):
                prices = np.array(data)
            elif isinstance(data, pd.Series):
                prices = data.values
            elif isinstance(data, pd.DataFrame):
                if 'close' in data.columns:
                    prices = data['close'].values
                else:
                    prices = data.iloc[:, 0].values
            elif isinstance(data, np.ndarray):
                prices = data
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            
            if len(prices) < period + 1:
                return []
                
            # Calculate price changes
            deltas = np.diff(prices)
            
            # Calculate gains and losses
            gains = deltas.copy()
            losses = deltas.copy()
            
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = abs(losses)
            
            # Calculate average gains and losses
            avg_gain = np.zeros_like(prices)
            avg_loss = np.zeros_like(prices)
            
            # First average is simple average
            avg_gain[period] = np.mean(gains[:period])
            avg_loss[period] = np.mean(losses[:period])
            
            # Calculate subsequent values using smoothing
            for i in range(period + 1, len(prices)):
                avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
                
            # Calculate RS and RSI
            rs = np.zeros_like(prices)
            rsi = np.zeros_like(prices)
            
            for i in range(period, len(prices)):
                if avg_loss[i] == 0:
                    rs[i] = 100.0
                else:
                    rs[i] = avg_gain[i] / avg_loss[i]
                    
                rsi[i] = 100 - (100 / (1 + rs[i]))
                
            # Return only valid values
            return rsi[period:].tolist()
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return []
    
    @staticmethod
    def rsi(data: Any, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI) as pandas Series.
        
        Args:
            data: Price data
            period: RSI period
            
        Returns:
            RSI values as pandas Series
        """
        try:
            series = AdvancedTechnicalIndicators.ensure_pandas_series(data)
            delta = series.diff()
            
            # Make two series: one for gains, one for losses
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            # Calculate average gain and average loss
            avg_gain = AdvancedTechnicalIndicators.safe_ewm(gain, span=period).mean()
            avg_loss = AdvancedTechnicalIndicators.safe_ewm(loss, span=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series(np.nan, index=AdvancedTechnicalIndicators.ensure_pandas_series(data).index)
    
    @staticmethod
    def calculate_macd(data: Any, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, List[float]]:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            data: Price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal EMA period
            
        Returns:
            Dictionary with MACD, signal, and histogram values as lists
        """
        try:
            if data is None:
                raise ValueError("Input data cannot be None")
                
            # Convert to numpy array if needed
            if isinstance(data, (list, tuple)):
                prices = np.array(data)
            elif isinstance(data, pd.Series):
                prices = data.values
            elif isinstance(data, pd.DataFrame):
                if 'close' in data.columns:
                    prices = data['close'].values
                else:
                    prices = data.iloc[:, 0].values
            elif isinstance(data, np.ndarray):
                prices = data
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            
            if len(prices) < slow_period + signal_period:
                return {'macd': [], 'signal': [], 'histogram': []}
            
            # Calculate EMAs
            ema_fast = np.zeros_like(prices)
            ema_slow = np.zeros_like(prices)
            
            # Initialize with SMA
            ema_fast[fast_period-1] = np.mean(prices[:fast_period])
            ema_slow[slow_period-1] = np.mean(prices[:slow_period])
            
            # Calculate subsequent EMAs
            alpha_fast = 2 / (fast_period + 1)
            alpha_slow = 2 / (slow_period + 1)
            
            for i in range(fast_period, len(prices)):
                ema_fast[i] = prices[i] * alpha_fast + ema_fast[i-1] * (1 - alpha_fast)
                
            for i in range(slow_period, len(prices)):
                ema_slow[i] = prices[i] * alpha_slow + ema_slow[i-1] * (1 - alpha_slow)
            
            # Calculate MACD line
            macd_line = ema_fast - ema_slow
            
            # Calculate signal line
            signal_line = np.zeros_like(prices)
            alpha_signal = 2 / (signal_period + 1)
            
            # Initialize with SMA
            signal_line[slow_period+signal_period-2] = np.mean(macd_line[slow_period-1:slow_period+signal_period-1])
            
            # Calculate subsequent signal values
            for i in range(slow_period+signal_period-1, len(prices)):
                signal_line[i] = macd_line[i] * alpha_signal + signal_line[i-1] * (1 - alpha_signal)
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            # Return only valid values
            valid_idx = slow_period + signal_period - 1
            return {
                'macd': macd_line[valid_idx:].tolist(),
                'signal': signal_line[valid_idx:].tolist(),
                'histogram': histogram[valid_idx:].tolist()
            }
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return {'macd': [], 'signal': [], 'histogram': []}
    
    @staticmethod
    def macd(data: Any, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate Moving Average Convergence Divergence (MACD) as pandas Series.
        
        Args:
            data: Price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal EMA period
            
        Returns:
            Dictionary with MACD, signal, and histogram values as pandas Series
        """
        try:
            series = AdvancedTechnicalIndicators.ensure_pandas_series(data)
            
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
            empty_series = pd.Series(np.nan, index=AdvancedTechnicalIndicators.ensure_pandas_series(data).index)
            return {
                'macd': empty_series,
                'signal': empty_series,
                'histogram': empty_series
            }
    
    @staticmethod
    def calculate_bollinger_bands(data: Any, period: int = 20, std_dev: float = 2.0) -> Dict[str, List[float]]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: Price data
            period: SMA period
            std_dev: Standard deviation multiplier
            
        Returns:
            Dictionary with upper, middle, and lower band values as lists
        """
        try:
            if data is None:
                raise ValueError("Input data cannot be None")
                
            # Convert to numpy array if needed
            if isinstance(data, (list, tuple)):
                prices = np.array(data)
            elif isinstance(data, pd.Series):
                prices = data.values
            elif isinstance(data, pd.DataFrame):
                if 'close' in data.columns:
                    prices = data['close'].values
                else:
                    prices = data.iloc[:, 0].values
            elif isinstance(data, np.ndarray):
                prices = data
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            
            if len(prices) < period:
                return {'upper': [], 'middle': [], 'lower': []}
            
            # Calculate middle band (SMA)
            middle_band = np.zeros_like(prices)
            for i in range(period-1, len(prices)):
                middle_band[i] = np.mean(prices[i-period+1:i+1])
            
            # Calculate standard deviation
            std = np.zeros_like(prices)
            for i in range(period-1, len(prices)):
                std[i] = np.std(prices[i-period+1:i+1])
            
            # Calculate upper and lower bands
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)
            
            # Return only valid values
            valid_idx = period - 1
            return {
                'upper': upper_band[valid_idx:].tolist(),
                'middle': middle_band[valid_idx:].tolist(),
                'lower': lower_band[valid_idx:].tolist()
            }
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return {'upper': [], 'middle': [], 'lower': []}
    
    @staticmethod
    def bollinger_bands(data: Any, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands as pandas Series.
        
        Args:
            data: Price data
            period: SMA period
            std_dev: Standard deviation multiplier
            
        Returns:
            Dictionary with upper, middle, and lower band values as pandas Series
        """
        try:
            series = AdvancedTechnicalIndicators.ensure_pandas_series(data)
            
            # Calculate middle band (SMA)
            middle_band = AdvancedTechnicalIndicators.sma(series, period)
            
            # Calculate standard deviation
            rolling_std = AdvancedTechnicalIndicators.safe_rolling(series, period).std()
            
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
            empty_series = pd.Series(np.nan, index=AdvancedTechnicalIndicators.ensure_pandas_series(data).index)
            return {
                'upper': empty_series,
                'middle': empty_series,
                'lower': empty_series
            }
    
    @staticmethod
    def calculate_supertrend(ohlc: Dict[str, List[float]], period: int = 10, multiplier: float = 3.0) -> List[float]:
        """
        Calculate SuperTrend indicator.
        
        Args:
            ohlc: Dictionary with open, high, low, close price data as lists
            period: ATR period
            multiplier: ATR multiplier
            
        Returns:
            SuperTrend values as list
        """
        try:
            if not all(key in ohlc for key in ['high', 'low', 'close']):
                raise ValueError("OHLC data must contain 'high', 'low', and 'close' keys")
                
            high = np.array(ohlc['high'])
            low = np.array(ohlc['low'])
            close = np.array(ohlc['close'])
            
            if len(high) < period or len(low) < period or len(close) < period:
                return []
            
            # Calculate ATR
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            
            # Replace NaN values in the first element
            tr2[0] = tr1[0]
            tr3[0] = tr1[0]
            
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr = np.zeros_like(close)
            
            # Calculate first ATR value
            atr[period-1] = np.mean(tr[:period])
            
            # Calculate subsequent ATR values
            for i in range(period, len(close)):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            
            # Calculate basic upper and lower bands
            basic_upper = (high + low) / 2 + (multiplier * atr)
            basic_lower = (high + low) / 2 - (multiplier * atr)
            
            # Calculate SuperTrend
            supertrend = np.zeros_like(close)
            direction = np.zeros_like(close)  # 1 for uptrend, -1 for downtrend
            
            # Initialize
            supertrend[period-1] = basic_upper[period-1] if close[period-1] <= basic_upper[period-1] else basic_lower[period-1]
            direction[period-1] = -1 if close[period-1] <= basic_upper[period-1] else 1
            
            # Calculate subsequent values
            for i in range(period, len(close)):
                if direction[i-1] == 1:  # Previous direction was uptrend
                    if close[i] <= basic_upper[i]:
                        supertrend[i] = basic_upper[i]
                        direction[i] = -1
                    else:
                        supertrend[i] = max(basic_lower[i], supertrend[i-1])
                        direction[i] = 1
                else:  # Previous direction was downtrend
                    if close[i] >= basic_lower[i]:
                        supertrend[i] = basic_lower[i]
                        direction[i] = 1
                    else:
                        supertrend[i] = min(basic_upper[i], supertrend[i-1])
                        direction[i] = -1
            
            # Return only valid values
            return supertrend[period:].tolist()
        except Exception as e:
            logger.error(f"Error calculating SuperTrend: {str(e)}")
            return []
    
    @staticmethod
    def calculate_ichimoku_cloud(ohlc: Dict[str, List[float]], 
                               conversion_period: int = 9, 
                               base_period: int = 26, 
                               span_b_period: int = 52, 
                               displacement: int = 26) -> Dict[str, List[float]]:
        """
        Calculate Ichimoku Cloud.
        
        Args:
            ohlc: Dictionary with open, high, low, close price data as lists
            conversion_period: Conversion line period (Tenkan-sen)
            base_period: Base line period (Kijun-sen)
            span_b_period: Span B period (Senkou Span B)
            displacement: Displacement period
            
        Returns:
            Dictionary with Ichimoku components as lists
        """
        try:
            if not all(key in ohlc for key in ['high', 'low']):
                raise ValueError("OHLC data must contain 'high' and 'low' keys")
                
            high = np.array(ohlc['high'])
            low = np.array(ohlc['low'])
            
            if len(high) < span_b_period or len(low) < span_b_period:
                return {'tenkan': [], 'kijun': [], 'senkou_a': [], 'senkou_b': [], 'chikou': []}
            
            # Calculate Conversion Line (Tenkan-sen)
            tenkan = np.zeros_like(high)
            for i in range(conversion_period-1, len(high)):
                tenkan[i] = (np.max(high[i-conversion_period+1:i+1]) + np.min(low[i-conversion_period+1:i+1])) / 2
            
            # Calculate Base Line (Kijun-sen)
            kijun = np.zeros_like(high)
            for i in range(base_period-1, len(high)):
                kijun[i] = (np.max(high[i-base_period+1:i+1]) + np.min(low[i-base_period+1:i+1])) / 2
            
            # Calculate Leading Span A (Senkou Span A)
            senkou_a = np.zeros_like(high)
            for i in range(base_period-1, len(high)):
                if i + displacement < len(high):
                    senkou_a[i+displacement] = (tenkan[i] + kijun[i]) / 2
            
            # Calculate Leading Span B (Senkou Span B)
            senkou_b = np.zeros_like(high)
            for i in range(span_b_period-1, len(high)):
                if i + displacement < len(high):
                    senkou_b[i+displacement] = (np.max(high[i-span_b_period+1:i+1]) + np.min(low[i-span_b_period+1:i+1])) / 2
            
            # Calculate Lagging Span (Chikou Span)
            chikou = np.zeros_like(high)
            if 'close' in ohlc:
                close = np.array(ohlc['close'])
                for i in range(displacement, len(high)):
                    chikou[i-displacement] = close[i]
            
            # Return only valid values
            valid_idx = max(conversion_period, base_period, span_b_period) - 1
            return {
                'tenkan': tenkan[valid_idx:].tolist(),
                'kijun': kijun[valid_idx:].tolist(),
                'senkou_a': senkou_a[valid_idx:].tolist(),
                'senkou_b': senkou_b[valid_idx:].tolist(),
                'chikou': chikou[valid_idx:].tolist()
            }
        except Exception as e:
            logger.error(f"Error calculating Ichimoku Cloud: {str(e)}")
            return {'tenkan': [], 'kijun': [], 'senkou_a': [], 'senkou_b': [], 'chikou': []}
    
    @staticmethod
    def detect_candlestick_patterns(ohlc: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """
        Detect candlestick patterns.
        
        Args:
            ohlc: Dictionary with open, high, low, close price data as lists
            
        Returns:
            List of detected patterns with details
        """
        try:
            if not all(key in ohlc for key in ['open', 'high', 'low', 'close']):
                raise ValueError("OHLC data must contain 'open', 'high', 'low', and 'close' keys")
                
            open_prices = np.array(ohlc['open'])
            high_prices = np.array(ohlc['high'])
            low_prices = np.array(ohlc['low'])
            close_prices = np.array(ohlc['close'])
            
            if len(open_prices) < 3:
                return []
            
            patterns = []
            
            # Calculate body size and shadow size
            body_size = np.abs(close_prices - open_prices)
            upper_shadow = high_prices - np.maximum(close_prices, open_prices)
            lower_shadow = np.minimum(close_prices, open_prices) - low_prices
            
            # Calculate average body size for reference
            avg_body = np.mean(body_size[-20:]) if len(body_size) >= 20 else np.mean(body_size)
            
            # Detect patterns for each candle
            for i in range(2, len(close_prices)):
                current_pattern = {
                    'index': i,
                    'patterns': []
                }
                
                # Doji
                if body_size[i] <= 0.1 * avg_body:
                    current_pattern['patterns'].append({
                        'name': 'Doji',
                        'type': 'reversal',
                        'strength': 0.6
                    })
                
                # Hammer
                if (body_size[i] > 0 and  # Bullish
                    lower_shadow[i] >= 2 * body_size[i] and
                    upper_shadow[i] <= 0.1 * body_size[i]):
                    current_pattern['patterns'].append({
                        'name': 'Hammer',
                        'type': 'bullish',
                        'strength': 0.7
                    })
                
                # Shooting Star
                if (body_size[i] > 0 and  # Bearish
                    upper_shadow[i] >= 2 * body_size[i] and
                    lower_shadow[i] <= 0.1 * body_size[i]):
                    current_pattern['patterns'].append({
                        'name': 'Shooting Star',
                        'type': 'bearish',
                        'strength': 0.7
                    })
                
                # Engulfing patterns
                if i >= 1:
                    # Bullish Engulfing
                    if (close_prices[i-1] < open_prices[i-1] and  # Previous bearish
                        close_prices[i] > open_prices[i] and  # Current bullish
                        open_prices[i] <= close_prices[i-1] and
                        close_prices[i] >= open_prices[i-1]):
                        current_pattern['patterns'].append({
                            'name': 'Bullish Engulfing',
                            'type': 'bullish',
                            'strength': 0.8
                        })
                    
                    # Bearish Engulfing
                    if (close_prices[i-1] > open_prices[i-1] and  # Previous bullish
                        close_prices[i] < open_prices[i] and  # Current bearish
                        open_prices[i] >= close_prices[i-1] and
                        close_prices[i] <= open_prices[i-1]):
                        current_pattern['patterns'].append({
                            'name': 'Bearish Engulfing',
                            'type': 'bearish',
                            'strength': 0.8
                        })
                
                # Add pattern if any detected
                if current_pattern['patterns']:
                    patterns.append(current_pattern)
            
            return patterns
        except Exception as e:
            logger.error(f"Error detecting candlestick patterns: {str(e)}")
            return []
    
    @staticmethod
    def detect_market_regime(ohlc: Dict[str, List[float]]) -> str:
        """
        Detect market regime (trending, ranging, volatile).
        
        Args:
            ohlc: Dictionary with open, high, low, close price data as lists
            
        Returns:
            Market regime as string
        """
        try:
            if 'close' not in ohlc:
                raise ValueError("OHLC data must contain 'close' key")
                
            close = np.array(ohlc['close'])
            
            if len(close) < 20:
                return "unknown"
            
            # Calculate returns
            returns = np.diff(close) / close[:-1]
            
            # Calculate volatility
            volatility = np.std(returns[-20:]) * np.sqrt(252)
            
            # Calculate directional movement
            direction = (close[-1] - close[-20]) / close[-20]
            
            # Calculate price range
            if 'high' in ohlc and 'low' in ohlc:
                high = np.array(ohlc['high'])
                low = np.array(ohlc['low'])
                price_range = (np.max(high[-20:]) - np.min(low[-20:])) / np.mean(close[-20:])
            else:
                price_range = (np.max(close[-20:]) - np.min(close[-20:])) / np.mean(close[-20:])
            
            # Determine regime
            if volatility > 0.5:  # High volatility threshold
                return "volatile"
            elif abs(direction) > 0.1:  # Strong trend threshold
                return "trending_up" if direction > 0 else "trending_down"
            elif price_range < 0.05:  # Tight range threshold
                return "ranging"
            else:
                return "ranging"
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return "unknown"
