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
    def calculate_macd(data: Any, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            data: Price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal EMA period
            
        Returns:
            Tuple of (MACD line, signal line, histogram)
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
            
            return macd_line, signal_line, histogram
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            empty_series = pd.Series(np.nan, index=AdvancedTechnicalIndicators.ensure_pandas_series(data).index)
            return empty_series, empty_series, empty_series
    
    @staticmethod
    def calculate_bollinger_bands(data: Any, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: Price data
            period: SMA period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (upper band, middle band, lower band)
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
            
            return upper_band, middle_band, lower_band
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            empty_series = pd.Series(np.nan, index=AdvancedTechnicalIndicators.ensure_pandas_series(data).index)
            return empty_series, empty_series, empty_series
    
    @staticmethod
    def calculate_atr(high: Any, low: Any, close: Any, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period
            
        Returns:
            ATR values as pandas Series
        """
        try:
            high_series = AdvancedTechnicalIndicators.ensure_pandas_series(high)
            low_series = AdvancedTechnicalIndicators.ensure_pandas_series(low)
            close_series = AdvancedTechnicalIndicators.ensure_pandas_series(close)
            
            # Ensure all series have the same index
            if not high_series.index.equals(low_series.index) or not high_series.index.equals(close_series.index):
                raise ValueError("High, low, and close series must have the same index")
            
            # Calculate true range
            prev_close = close_series.shift(1)
            tr1 = high_series - low_series
            tr2 = (high_series - prev_close).abs()
            tr3 = (low_series - prev_close).abs()
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR
            atr = AdvancedTechnicalIndicators.safe_ewm(true_range, span=period).mean()
            
            return atr
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series(np.nan, index=AdvancedTechnicalIndicators.ensure_pandas_series(high).index)
    
    @staticmethod
    def calculate_stochastic(high: Any, low: Any, close: Any, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_period: %K period
            d_period: %D period
            
        Returns:
            Tuple of (%K, %D)
        """
        try:
            high_series = AdvancedTechnicalIndicators.ensure_pandas_series(high)
            low_series = AdvancedTechnicalIndicators.ensure_pandas_series(low)
            close_series = AdvancedTechnicalIndicators.ensure_pandas_series(close)
            
            # Ensure all series have the same index
            if not high_series.index.equals(low_series.index) or not high_series.index.equals(close_series.index):
                raise ValueError("High, low, and close series must have the same index")
            
            # Calculate %K
            lowest_low = AdvancedTechnicalIndicators.safe_rolling(low_series, k_period).min()
            highest_high = AdvancedTechnicalIndicators.safe_rolling(high_series, k_period).max()
            
            k = 100 * ((close_series - lowest_low) / (highest_high - lowest_low).replace(0, 1e-10))
            
            # Calculate %D
            d = AdvancedTechnicalIndicators.sma(k, d_period)
            
            return k, d
        except Exception as e:
            logger.error(f"Error calculating Stochastic Oscillator: {str(e)}")
            empty_series = pd.Series(np.nan, index=AdvancedTechnicalIndicators.ensure_pandas_series(high).index)
            return empty_series, empty_series
    
    @staticmethod
    def calculate_adx(high: Any, low: Any, close: Any, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ADX period
            
        Returns:
            ADX values as pandas Series
        """
        try:
            high_series = AdvancedTechnicalIndicators.ensure_pandas_series(high)
            low_series = AdvancedTechnicalIndicators.ensure_pandas_series(low)
            close_series = AdvancedTechnicalIndicators.ensure_pandas_series(close)
            
            # Ensure all series have the same index
            if not high_series.index.equals(low_series.index) or not high_series.index.equals(close_series.index):
                raise ValueError("High, low, and close series must have the same index")
            
            # Calculate +DM and -DM
            high_diff = high_series.diff()
            low_diff = low_series.diff()
            
            plus_dm = high_diff.copy()
            plus_dm[plus_dm < 0] = 0
            plus_dm[(high_diff <= 0) | (high_diff < low_diff.abs())] = 0
            
            minus_dm = low_diff.abs().copy()
            minus_dm[minus_dm < 0] = 0
            minus_dm[(low_diff >= 0) | (low_diff.abs() < high_diff)] = 0
            
            # Calculate ATR
            atr = AdvancedTechnicalIndicators.calculate_atr(high_series, low_series, close_series, period)
            
            # Calculate +DI and -DI
            plus_di = 100 * AdvancedTechnicalIndicators.safe_ewm(plus_dm, span=period).mean() / atr
            minus_di = 100 * AdvancedTechnicalIndicators.safe_ewm(minus_dm, span=period).mean() / atr
            
            # Calculate DX
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10))
            
            # Calculate ADX
            adx = AdvancedTechnicalIndicators.safe_ewm(dx, span=period).mean()
            
            return adx
        except Exception as e:
            logger.error(f"Error calculating ADX: {str(e)}")
            return pd.Series(np.nan, index=AdvancedTechnicalIndicators.ensure_pandas_series(high).index)
    
    @staticmethod
    def calculate_vwap(high: Any, low: Any, close: Any, volume: Any) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume
            
        Returns:
            VWAP values as pandas Series
        """
        try:
            high_series = AdvancedTechnicalIndicators.ensure_pandas_series(high)
            low_series = AdvancedTechnicalIndicators.ensure_pandas_series(low)
            close_series = AdvancedTechnicalIndicators.ensure_pandas_series(close)
            volume_series = AdvancedTechnicalIndicators.ensure_pandas_series(volume)
            
            # Ensure all series have the same index
            if (not high_series.index.equals(low_series.index) or 
                not high_series.index.equals(close_series.index) or 
                not high_series.index.equals(volume_series.index)):
                raise ValueError("High, low, close, and volume series must have the same index")
            
            # Calculate typical price
            typical_price = (high_series + low_series + close_series) / 3
            
            # Calculate VWAP
            vwap = (typical_price * volume_series).cumsum() / volume_series.cumsum()
            
            return vwap
        except Exception as e:
            logger.error(f"Error calculating VWAP: {str(e)}")
            return pd.Series(np.nan, index=AdvancedTechnicalIndicators.ensure_pandas_series(high).index)
