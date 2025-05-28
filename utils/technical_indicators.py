"""
Technical Indicators Calculator for Hyperliquid Trading Bot

This module provides a comprehensive set of technical indicators for trading strategy development.
It includes standard indicators like RSI, MACD, Bollinger Bands as well as advanced indicators
and custom implementations optimized for cryptocurrency markets.

Features:
- Standard technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Advanced indicators (Ichimoku Cloud, VWAP, etc.)
- Custom indicators optimized for cryptocurrency markets
- Efficient calculation with NumPy and Pandas
- Configurable parameters for all indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

# Configure logging
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    Technical indicators calculator for trading strategy development.
    
    Provides a comprehensive set of standard and advanced technical indicators
    optimized for cryptocurrency markets.
    """
    
    def __init__(self):
        """Initialize the technical indicators calculator."""
        logger.info("Technical indicators calculator initialized")
    
    def calculate_rsi(self, data: Union[pd.DataFrame, pd.Series, List[float], np.ndarray], 
                     period: int = 14, column: Optional[str] = None) -> np.ndarray:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: Price data (DataFrame, Series, List, or ndarray)
            period: RSI period
            column: Column name if data is DataFrame
            
        Returns:
            ndarray with RSI values
        """
        # Convert data to numpy array
        if isinstance(data, pd.DataFrame):
            if column is None:
                column = 'close'
            prices = data[column].values
        elif isinstance(data, pd.Series):
            prices = data.values
        else:
            prices = np.array(data)
        
        # Calculate price changes
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            # Avoid division by zero
            rs = float('inf')
        else:
            rs = up / down
        
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - (100. / (1. + rs))
        
        # Calculate RSI
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            
            if down == 0:
                # Avoid division by zero
                rs = float('inf')
            else:
                rs = up / down
            
            rsi[i] = 100. - (100. / (1. + rs))
        
        return rsi
    
    def calculate_macd(self, data: Union[pd.DataFrame, pd.Series, List[float], np.ndarray],
                      fast_period: int = 12, slow_period: int = 26, signal_period: int = 9,
                      column: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            data: Price data (DataFrame, Series, List, or ndarray)
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal EMA period
            column: Column name if data is DataFrame
            
        Returns:
            Tuple of (macd, signal, histogram)
        """
        # Convert data to numpy array
        if isinstance(data, pd.DataFrame):
            if column is None:
                column = 'close'
            prices = data[column].values
        elif isinstance(data, pd.Series):
            prices = data.values
        else:
            prices = np.array(data)
        
        # Calculate EMAs
        ema_fast = self.calculate_ema(prices, fast_period)
        ema_slow = self.calculate_ema(prices, slow_period)
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = self.calculate_ema(macd_line, signal_period)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, data: Union[pd.DataFrame, pd.Series, List[float], np.ndarray],
                                 period: int = 20, std_dev: float = 2.0,
                                 column: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: Price data (DataFrame, Series, List, or ndarray)
            period: SMA period
            std_dev: Standard deviation multiplier
            column: Column name if data is DataFrame
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        # Convert data to numpy array
        if isinstance(data, pd.DataFrame):
            if column is None:
                column = 'close'
            prices = data[column].values
        elif isinstance(data, pd.Series):
            prices = data.values
        else:
            prices = np.array(data)
        
        # Calculate middle band (SMA)
        middle_band = self.calculate_sma(prices, period)
        
        # Calculate standard deviation
        rolling_std = np.zeros_like(prices)
        
        for i in range(period - 1, len(prices)):
            rolling_std[i] = np.std(prices[i - period + 1:i + 1])
        
        # Calculate upper and lower bands
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    def calculate_sma(self, data: Union[pd.DataFrame, pd.Series, List[float], np.ndarray],
                     period: int = 20, column: Optional[str] = None) -> np.ndarray:
        """
        Calculate Simple Moving Average (SMA).
        
        Args:
            data: Price data (DataFrame, Series, List, or ndarray)
            period: SMA period
            column: Column name if data is DataFrame
            
        Returns:
            ndarray with SMA values
        """
        # Convert data to numpy array
        if isinstance(data, pd.DataFrame):
            if column is None:
                column = 'close'
            prices = data[column].values
        elif isinstance(data, pd.Series):
            prices = data.values
        else:
            prices = np.array(data)
        
        # Calculate SMA
        sma = np.zeros_like(prices)
        
        for i in range(period - 1, len(prices)):
            sma[i] = np.mean(prices[i - period + 1:i + 1])
        
        return sma
    
    def calculate_ema(self, data: Union[pd.DataFrame, pd.Series, List[float], np.ndarray],
                     period: int = 20, column: Optional[str] = None) -> np.ndarray:
        """
        Calculate Exponential Moving Average (EMA).
        
        Args:
            data: Price data (DataFrame, Series, List, or ndarray)
            period: EMA period
            column: Column name if data is DataFrame
            
        Returns:
            ndarray with EMA values
        """
        # Convert data to numpy array
        if isinstance(data, pd.DataFrame):
            if column is None:
                column = 'close'
            prices = data[column].values
        elif isinstance(data, pd.Series):
            prices = data.values
        else:
            prices = np.array(data)
        
        # Calculate EMA
        ema = np.zeros_like(prices)
        ema[period - 1] = np.mean(prices[:period])
        
        # Calculate multiplier
        multiplier = 2.0 / (period + 1)
        
        # Calculate EMA
        for i in range(period, len(prices)):
            ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1]
        
        return ema
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """
        Calculate Average True Range (ATR).
        
        Args:
            data: OHLC DataFrame with 'high', 'low', and 'close' columns
            period: ATR period
            
        Returns:
            ndarray with ATR values
        """
        # Check if data has required columns
        required_columns = ['high', 'low', 'close']
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Data must have '{column}' column")
        
        # Calculate true range
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i - 1])
            tr3 = abs(low[i] - close[i - 1])
            tr[i] = max(tr1, tr2, tr3)
        
        # Calculate ATR
        atr = np.zeros_like(tr)
        atr[period - 1] = np.mean(tr[1:period])
        
        for i in range(period, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        
        return atr
    
    def calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            data: OHLC DataFrame with 'high', 'low', and 'close' columns
            k_period: %K period
            d_period: %D period
            
        Returns:
            Tuple of (%K, %D)
        """
        # Check if data has required columns
        required_columns = ['high', 'low', 'close']
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Data must have '{column}' column")
        
        # Get data
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # Calculate %K
        k = np.zeros(len(close))
        
        for i in range(k_period - 1, len(close)):
            highest_high = np.max(high[i - k_period + 1:i + 1])
            lowest_low = np.min(low[i - k_period + 1:i + 1])
            
            if highest_high == lowest_low:
                k[i] = 50.0
            else:
                k[i] = 100.0 * (close[i] - lowest_low) / (highest_high - lowest_low)
        
        # Calculate %D (SMA of %K)
        d = self.calculate_sma(k, d_period)
        
        return k, d
    
    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Average Directional Index (ADX).
        
        Args:
            data: OHLC DataFrame with 'high', 'low', and 'close' columns
            period: ADX period
            
        Returns:
            Tuple of (ADX, +DI, -DI)
        """
        # Check if data has required columns
        required_columns = ['high', 'low', 'close']
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Data must have '{column}' column")
        
        # Get data
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # Calculate True Range
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i - 1])
            tr3 = abs(low[i] - close[i - 1])
            tr[i] = max(tr1, tr2, tr3)
        
        # Calculate +DM and -DM
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        
        for i in range(1, len(high)):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            else:
                plus_dm[i] = 0
            
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            else:
                minus_dm[i] = 0
        
        # Calculate smoothed TR, +DM, and -DM
        smoothed_tr = np.zeros(len(tr))
        smoothed_plus_dm = np.zeros(len(plus_dm))
        smoothed_minus_dm = np.zeros(len(minus_dm))
        
        # Initialize
        smoothed_tr[period] = np.sum(tr[1:period+1])
        smoothed_plus_dm[period] = np.sum(plus_dm[1:period+1])
        smoothed_minus_dm[period] = np.sum(minus_dm[1:period+1])
        
        # Calculate smoothed values
        for i in range(period + 1, len(tr)):
            smoothed_tr[i] = smoothed_tr[i - 1] - (smoothed_tr[i - 1] / period) + tr[i]
            smoothed_plus_dm[i] = smoothed_plus_dm[i - 1] - (smoothed_plus_dm[i - 1] / period) + plus_dm[i]
            smoothed_minus_dm[i] = smoothed_minus_dm[i - 1] - (smoothed_minus_dm[i - 1] / period) + minus_dm[i]
        
        # Calculate +DI and -DI
        plus_di = np.zeros(len(smoothed_tr))
        minus_di = np.zeros(len(smoothed_tr))
        
        for i in range(period, len(smoothed_tr)):
            if smoothed_tr[i] == 0:
                plus_di[i] = 0
                minus_di[i] = 0
            else:
                plus_di[i] = 100.0 * smoothed_plus_dm[i] / smoothed_tr[i]
                minus_di[i] = 100.0 * smoothed_minus_dm[i] / smoothed_tr[i]
        
        # Calculate DX
        dx = np.zeros(len(plus_di))
        
        for i in range(period, len(plus_di)):
            if plus_di[i] + minus_di[i] == 0:
                dx[i] = 0
            else:
                dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # Calculate ADX
        adx = np.zeros(len(dx))
        adx[2 * period - 1] = np.mean(dx[period:2 * period])
        
        for i in range(2 * period, len(dx)):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
        
        return adx, plus_di, minus_di
    
    def calculate_ichimoku(self, data: pd.DataFrame, 
                          tenkan_period: int = 9, 
                          kijun_period: int = 26, 
                          senkou_span_b_period: int = 52,
                          displacement: int = 26) -> Dict[str, np.ndarray]:
        """
        Calculate Ichimoku Cloud.
        
        Args:
            data: OHLC DataFrame with 'high' and 'low' columns
            tenkan_period: Tenkan-sen (Conversion Line) period
            kijun_period: Kijun-sen (Base Line) period
            senkou_span_b_period: Senkou Span B (Leading Span B) period
            displacement: Displacement period for Senkou Span
            
        Returns:
            Dictionary with Ichimoku components
        """
        # Check if data has required columns
        required_columns = ['high', 'low']
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Data must have '{column}' column")
        
        # Get data
        high = data['high'].values
        low = data['low'].values
        
        # Calculate Tenkan-sen (Conversion Line)
        tenkan_sen = np.zeros(len(high))
        
        for i in range(tenkan_period - 1, len(high)):
            highest_high = np.max(high[i - tenkan_period + 1:i + 1])
            lowest_low = np.min(low[i - tenkan_period + 1:i + 1])
            tenkan_sen[i] = (highest_high + lowest_low) / 2.0
        
        # Calculate Kijun-sen (Base Line)
        kijun_sen = np.zeros(len(high))
        
        for i in range(kijun_period - 1, len(high)):
            highest_high = np.max(high[i - kijun_period + 1:i + 1])
            lowest_low = np.min(low[i - kijun_period + 1:i + 1])
            kijun_sen[i] = (highest_high + lowest_low) / 2.0
        
        # Calculate Senkou Span A (Leading Span A)
        senkou_span_a = np.zeros(len(high) + displacement)
        
        for i in range(kijun_period - 1, len(high)):
            senkou_span_a[i + displacement] = (tenkan_sen[i] + kijun_sen[i]) / 2.0
        
        # Calculate Senkou Span B (Leading Span B)
        senkou_span_b = np.zeros(len(high) + displacement)
        
        for i in range(senkou_span_b_period - 1, len(high)):
            highest_high = np.max(high[i - senkou_span_b_period + 1:i + 1])
            lowest_low = np.min(low[i - senkou_span_b_period + 1:i + 1])
            senkou_span_b[i + displacement] = (highest_high + lowest_low) / 2.0
        
        # Calculate Chikou Span (Lagging Span)
        chikou_span = np.zeros(len(high))
        
        for i in range(displacement, len(high)):
            chikou_span[i - displacement] = high[i]
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a[:len(high)],
            'senkou_span_b': senkou_span_b[:len(high)],
            'chikou_span': chikou_span
        }
    
    def calculate_vwap(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            data: OHLCV DataFrame with 'high', 'low', 'close', and 'volume' columns
            
        Returns:
            ndarray with VWAP values
        """
        # Check if data has required columns
        required_columns = ['high', 'low', 'close', 'volume']
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Data must have '{column}' column")
        
        # Get data
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values
        
        # Calculate typical price
        typical_price = (high + low + close) / 3.0
        
        # Calculate VWAP
        vwap = np.zeros(len(typical_price))
        cumulative_tp_vol = 0
        cumulative_vol = 0
        
        for i in range(len(typical_price)):
            cumulative_tp_vol += typical_price[i] * volume[i]
            cumulative_vol += volume[i]
            
            if cumulative_vol == 0:
                vwap[i] = 0
            else:
                vwap[i] = cumulative_tp_vol / cumulative_vol
        
        return vwap
    
    def calculate_obv(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate On-Balance Volume (OBV).
        
        Args:
            data: OHLCV DataFrame with 'close' and 'volume' columns
            
        Returns:
            ndarray with OBV values
        """
        # Check if data has required columns
        required_columns = ['close', 'volume']
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Data must have '{column}' column")
        
        # Get data
        close = data['close'].values
        volume = data['volume'].values
        
        # Calculate OBV
        obv = np.zeros(len(close))
        
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]
        
        return obv
    
    def calculate_mfi(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """
        Calculate Money Flow Index (MFI).
        
        Args:
            data: OHLCV DataFrame with 'high', 'low', 'close', and 'volume' columns
            period: MFI period
            
        Returns:
            ndarray with MFI values
        """
        # Check if data has required columns
        required_columns = ['high', 'low', 'close', 'volume']
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Data must have '{column}' column")
        
        # Get data
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values
        
        # Calculate typical price
        typical_price = (high + low + close) / 3.0
        
        # Calculate money flow
        money_flow = typical_price * volume
        
        # Calculate positive and negative money flow
        positive_flow = np.zeros(len(typical_price))
        negative_flow = np.zeros(len(typical_price))
        
        for i in range(1, len(typical_price)):
            if typical_price[i] > typical_price[i - 1]:
                positive_flow[i] = money_flow[i]
                negative_flow[i] = 0
            elif typical_price[i] < typical_price[i - 1]:
                positive_flow[i] = 0
                negative_flow[i] = money_flow[i]
            else:
                positive_flow[i] = 0
                negative_flow[i] = 0
        
        # Calculate MFI
        mfi = np.zeros(len(typical_price))
        
        for i in range(period, len(typical_price)):
            positive_sum = np.sum(positive_flow[i - period + 1:i + 1])
            negative_sum = np.sum(negative_flow[i - period + 1:i + 1])
            
            if negative_sum == 0:
                mfi[i] = 100.0
            else:
                money_ratio = positive_sum / negative_sum
                mfi[i] = 100.0 - (100.0 / (1.0 + money_ratio))
        
        return mfi
    
    def calculate_roc(self, data: Union[pd.DataFrame, pd.Series, List[float], np.ndarray],
                     period: int = 12, column: Optional[str] = None) -> np.ndarray:
        """
        Calculate Rate of Change (ROC).
        
        Args:
            data: Price data (DataFrame, Series, List, or ndarray)
            period: ROC period
            column: Column name if data is DataFrame
            
        Returns:
            ndarray with ROC values
        """
        # Convert data to numpy array
        if isinstance(data, pd.DataFrame):
            if column is None:
                column = 'close'
            prices = data[column].values
        elif isinstance(data, pd.Series):
            prices = data.values
        else:
            prices = np.array(data)
        
        # Calculate ROC
        roc = np.zeros(len(prices))
        
        for i in range(period, len(prices)):
            if prices[i - period] == 0:
                roc[i] = 0
            else:
                roc[i] = ((prices[i] - prices[i - period]) / prices[i - period]) * 100.0
        
        return roc
    
    def calculate_cci(self, data: pd.DataFrame, period: int = 20) -> np.ndarray:
        """
        Calculate Commodity Channel Index (CCI).
        
        Args:
            data: OHLC DataFrame with 'high', 'low', and 'close' columns
            period: CCI period
            
        Returns:
            ndarray with CCI values
        """
        # Check if data has required columns
        required_columns = ['high', 'low', 'close']
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Data must have '{column}' column")
        
        # Get data
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # Calculate typical price
        typical_price = (high + low + close) / 3.0
        
        # Calculate SMA of typical price
        tp_sma = self.calculate_sma(typical_price, period)
        
        # Calculate mean deviation
        mean_deviation = np.zeros(len(typical_price))
        
        for i in range(period - 1, len(typical_price)):
            mean_deviation[i] = np.mean(np.abs(typical_price[i - period + 1:i + 1] - tp_sma[i]))
        
        # Calculate CCI
        cci = np.zeros(len(typical_price))
        
        for i in range(period - 1, len(typical_price)):
            if mean_deviation[i] == 0:
                cci[i] = 0
            else:
                cci[i] = (typical_price[i] - tp_sma[i]) / (0.015 * mean_deviation[i])
        
        return cci
    
    def calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """
        Calculate Williams %R.
        
        Args:
            data: OHLC DataFrame with 'high', 'low', and 'close' columns
            period: Williams %R period
            
        Returns:
            ndarray with Williams %R values
        """
        # Check if data has required columns
        required_columns = ['high', 'low', 'close']
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Data must have '{column}' column")
        
        # Get data
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # Calculate Williams %R
        williams_r = np.zeros(len(close))
        
        for i in range(period - 1, len(close)):
            highest_high = np.max(high[i - period + 1:i + 1])
            lowest_low = np.min(low[i - period + 1:i + 1])
            
            if highest_high == lowest_low:
                williams_r[i] = -50.0
            else:
                williams_r[i] = -100.0 * (highest_high - close[i]) / (highest_high - lowest_low)
        
        return williams_r
    
    def calculate_trix(self, data: Union[pd.DataFrame, pd.Series, List[float], np.ndarray],
                      period: int = 15, column: Optional[str] = None) -> np.ndarray:
        """
        Calculate Triple Exponential Average (TRIX).
        
        Args:
            data: Price data (DataFrame, Series, List, or ndarray)
            period: TRIX period
            column: Column name if data is DataFrame
            
        Returns:
            ndarray with TRIX values
        """
        # Convert data to numpy array
        if isinstance(data, pd.DataFrame):
            if column is None:
                column = 'close'
            prices = data[column].values
        elif isinstance(data, pd.Series):
            prices = data.values
        else:
            prices = np.array(data)
        
        # Calculate triple EMA
        ema1 = self.calculate_ema(prices, period)
        ema2 = self.calculate_ema(ema1, period)
        ema3 = self.calculate_ema(ema2, period)
        
        # Calculate TRIX
        trix = np.zeros(len(prices))
        
        for i in range(1, len(prices)):
            if ema3[i - 1] == 0:
                trix[i] = 0
            else:
                trix[i] = (ema3[i] - ema3[i - 1]) / ema3[i - 1] * 100.0
        
        return trix
    
    def calculate_keltner_channels(self, data: pd.DataFrame, ema_period: int = 20, 
                                  atr_period: int = 10, multiplier: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Keltner Channels.
        
        Args:
            data: OHLC DataFrame with 'high', 'low', and 'close' columns
            ema_period: EMA period
            atr_period: ATR period
            multiplier: ATR multiplier
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        # Check if data has required columns
        required_columns = ['high', 'low', 'close']
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Data must have '{column}' column")
        
        # Get data
        close = data['close'].values
        
        # Calculate middle band (EMA)
        middle_band = self.calculate_ema(close, ema_period)
        
        # Calculate ATR
        atr = self.calculate_atr(data, atr_period)
        
        # Calculate upper and lower bands
        upper_band = middle_band + (multiplier * atr)
        lower_band = middle_band - (multiplier * atr)
        
        return upper_band, middle_band, lower_band
    
    def calculate_supertrend(self, data: pd.DataFrame, period: int = 10, 
                            multiplier: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate SuperTrend.
        
        Args:
            data: OHLC DataFrame with 'high', 'low', and 'close' columns
            period: ATR period
            multiplier: ATR multiplier
            
        Returns:
            Tuple of (supertrend, direction)
        """
        # Check if data has required columns
        required_columns = ['high', 'low', 'close']
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Data must have '{column}' column")
        
        # Get data
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # Calculate ATR
        atr = self.calculate_atr(data, period)
        
        # Calculate basic upper and lower bands
        basic_upper_band = (high + low) / 2.0 + (multiplier * atr)
        basic_lower_band = (high + low) / 2.0 - (multiplier * atr)
        
        # Calculate SuperTrend
        supertrend = np.zeros(len(close))
        direction = np.zeros(len(close))  # 1 for uptrend, -1 for downtrend
        
        # Initialize
        supertrend[period - 1] = close[period - 1]
        direction[period - 1] = 1
        
        # Calculate SuperTrend
        for i in range(period, len(close)):
            # Adjust upper and lower bands
            if basic_upper_band[i] < basic_upper_band[i - 1] or close[i - 1] > basic_upper_band[i - 1]:
                basic_upper_band[i] = basic_upper_band[i]
            else:
                basic_upper_band[i] = basic_upper_band[i - 1]
            
            if basic_lower_band[i] > basic_lower_band[i - 1] or close[i - 1] < basic_lower_band[i - 1]:
                basic_lower_band[i] = basic_lower_band[i]
            else:
                basic_lower_band[i] = basic_lower_band[i - 1]
            
            # Calculate SuperTrend
            if supertrend[i - 1] == basic_upper_band[i - 1]:
                if close[i] <= basic_upper_band[i]:
                    supertrend[i] = basic_upper_band[i]
                    direction[i] = -1
                else:
                    supertrend[i] = basic_lower_band[i]
                    direction[i] = 1
            elif supertrend[i - 1] == basic_lower_band[i - 1]:
                if close[i] >= basic_lower_band[i]:
                    supertrend[i] = basic_lower_band[i]
                    direction[i] = 1
                else:
                    supertrend[i] = basic_upper_band[i]
                    direction[i] = -1
            else:
                # Should not happen, but just in case
                supertrend[i] = supertrend[i - 1]
                direction[i] = direction[i - 1]
        
        return supertrend, direction
    
    def calculate_pivot_points(self, data: pd.DataFrame, method: str = 'standard') -> Dict[str, float]:
        """
        Calculate Pivot Points.
        
        Args:
            data: OHLC DataFrame with 'high', 'low', and 'close' columns
            method: Pivot point method ('standard', 'fibonacci', 'woodie', 'camarilla', 'demark')
            
        Returns:
            Dictionary with pivot points
        """
        # Check if data has required columns
        required_columns = ['high', 'low', 'close']
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Data must have '{column}' column")
        
        # Get data from the last row (most recent)
        high = data['high'].values[-1]
        low = data['low'].values[-1]
        close = data['close'].values[-1]
        
        # Add open if available
        if 'open' in data.columns:
            open_price = data['open'].values[-1]
        else:
            open_price = None
        
        # Calculate pivot points based on method
        if method == 'standard':
            pivot = (high + low + close) / 3.0
            s1 = (2.0 * pivot) - high
            s2 = pivot - (high - low)
            s3 = low - 2.0 * (high - pivot)
            r1 = (2.0 * pivot) - low
            r2 = pivot + (high - low)
            r3 = high + 2.0 * (pivot - low)
            
            return {
                'pivot': pivot,
                's1': s1,
                's2': s2,
                's3': s3,
                'r1': r1,
                'r2': r2,
                'r3': r3
            }
        
        elif method == 'fibonacci':
            pivot = (high + low + close) / 3.0
            s1 = pivot - 0.382 * (high - low)
            s2 = pivot - 0.618 * (high - low)
            s3 = pivot - 1.0 * (high - low)
            r1 = pivot + 0.382 * (high - low)
            r2 = pivot + 0.618 * (high - low)
            r3 = pivot + 1.0 * (high - low)
            
            return {
                'pivot': pivot,
                's1': s1,
                's2': s2,
                's3': s3,
                'r1': r1,
                'r2': r2,
                'r3': r3
            }
        
        elif method == 'woodie':
            if open_price is None:
                raise ValueError("Woodie pivot points require 'open' column in data")
            
            pivot = (high + low + 2.0 * open_price) / 4.0
            s1 = (2.0 * pivot) - high
            s2 = pivot - (high - low)
            s3 = low - 2.0 * (high - pivot)
            r1 = (2.0 * pivot) - low
            r2 = pivot + (high - low)
            r3 = high + 2.0 * (pivot - low)
            
            return {
                'pivot': pivot,
                's1': s1,
                's2': s2,
                's3': s3,
                'r1': r1,
                'r2': r2,
                'r3': r3
            }
        
        elif method == 'camarilla':
            pivot = (high + low + close) / 3.0
            s1 = close - 1.1 * (high - low) / 12.0
            s2 = close - 1.1 * (high - low) / 6.0
            s3 = close - 1.1 * (high - low) / 4.0
            s4 = close - 1.1 * (high - low) / 2.0
            r1 = close + 1.1 * (high - low) / 12.0
            r2 = close + 1.1 * (high - low) / 6.0
            r3 = close + 1.1 * (high - low) / 4.0
            r4 = close + 1.1 * (high - low) / 2.0
            
            return {
                'pivot': pivot,
                's1': s1,
                's2': s2,
                's3': s3,
                's4': s4,
                'r1': r1,
                'r2': r2,
                'r3': r3,
                'r4': r4
            }
        
        elif method == 'demark':
            if open_price is None:
                raise ValueError("DeMark pivot points require 'open' column in data")
            
            if close < open_price:
                x = high + (2.0 * low) + close
            elif close > open_price:
                x = (2.0 * high) + low + close
            else:
                x = high + low + (2.0 * close)
            
            pivot = x / 4.0
            s1 = x / 2.0 - high
            r1 = x / 2.0 - low
            
            return {
                'pivot': pivot,
                's1': s1,
                'r1': r1
            }
        
        else:
            raise ValueError(f"Invalid pivot point method: {method}")
    
    def calculate_zigzag(self, data: pd.DataFrame, deviation: float = 5.0, 
                        column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate ZigZag indicator.
        
        Args:
            data: DataFrame with price data
            deviation: Minimum percentage deviation for a pivot
            column: Column name to use for calculation
            
        Returns:
            Tuple of (zigzag, pivots)
        """
        # Check if data has required column
        if column not in data.columns:
            raise ValueError(f"Data must have '{column}' column")
        
        # Get data
        prices = data[column].values
        
        # Initialize variables
        zigzag = np.zeros(len(prices))
        pivots = np.zeros(len(prices))
        
        # Set initial pivot
        last_pivot_idx = 0
        last_pivot_price = prices[0]
        last_pivot_type = 0  # 0 for none, 1 for high, -1 for low
        
        # Calculate ZigZag
        for i in range(1, len(prices)):
            # Calculate percentage change from last pivot
            if last_pivot_price == 0:
                pct_change = 0
            else:
                pct_change = (prices[i] - last_pivot_price) / last_pivot_price * 100.0
            
            # Check if we have a new pivot
            if abs(pct_change) >= deviation:
                # Determine pivot type
                if pct_change > 0:
                    # New high pivot
                    if last_pivot_type != 1:
                        # Set pivot
                        pivots[i] = 1
                        zigzag[i] = prices[i]
                        
                        # Update last pivot
                        last_pivot_idx = i
                        last_pivot_price = prices[i]
                        last_pivot_type = 1
                    elif prices[i] > last_pivot_price:
                        # Update existing high pivot
                        pivots[last_pivot_idx] = 0
                        zigzag[last_pivot_idx] = 0
                        
                        pivots[i] = 1
                        zigzag[i] = prices[i]
                        
                        # Update last pivot
                        last_pivot_idx = i
                        last_pivot_price = prices[i]
                        last_pivot_type = 1
                else:
                    # New low pivot
                    if last_pivot_type != -1:
                        # Set pivot
                        pivots[i] = -1
                        zigzag[i] = prices[i]
                        
                        # Update last pivot
                        last_pivot_idx = i
                        last_pivot_price = prices[i]
                        last_pivot_type = -1
                    elif prices[i] < last_pivot_price:
                        # Update existing low pivot
                        pivots[last_pivot_idx] = 0
                        zigzag[last_pivot_idx] = 0
                        
                        pivots[i] = -1
                        zigzag[i] = prices[i]
                        
                        # Update last pivot
                        last_pivot_idx = i
                        last_pivot_price = prices[i]
                        last_pivot_type = -1
        
        return zigzag, pivots
    
    def calculate_heikin_ashi(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Heikin-Ashi candles.
        
        Args:
            data: OHLC DataFrame with 'open', 'high', 'low', and 'close' columns
            
        Returns:
            DataFrame with Heikin-Ashi candles
        """
        # Check if data has required columns
        required_columns = ['open', 'high', 'low', 'close']
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Data must have '{column}' column")
        
        # Create result DataFrame
        ha = pd.DataFrame(index=data.index)
        
        # Calculate Heikin-Ashi candles
        ha['close'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4.0
        
        # Calculate open
        ha['open'] = data['open'].copy()
        for i in range(1, len(data)):
            ha.iloc[i, ha.columns.get_loc('open')] = (ha.iloc[i-1, ha.columns.get_loc('open')] + 
                                                     ha.iloc[i-1, ha.columns.get_loc('close')]) / 2.0
        
        # Calculate high and low
        ha['high'] = ha[['open', 'close']].max(axis=1).combine(data['high'], max)
        ha['low'] = ha[['open', 'close']].min(axis=1).combine(data['low'], min)
        
        return ha
    
    def calculate_custom_xrp_oscillator(self, data: pd.DataFrame, 
                                       rsi_period: int = 14, 
                                       stoch_period: int = 14,
                                       stoch_smooth: int = 3,
                                       ema_period: int = 9) -> np.ndarray:
        """
        Calculate Custom XRP Oscillator.
        
        This is a custom indicator specifically designed for XRP trading.
        It combines RSI, Stochastic, and price momentum.
        
        Args:
            data: OHLC DataFrame with 'high', 'low', and 'close' columns
            rsi_period: RSI period
            stoch_period: Stochastic period
            stoch_smooth: Stochastic smoothing period
            ema_period: EMA period for final smoothing
            
        Returns:
            ndarray with XRP Oscillator values
        """
        # Check if data has required columns
        required_columns = ['high', 'low', 'close']
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Data must have '{column}' column")
        
        # Get data
        close = data['close'].values
        
        # Calculate RSI
        rsi = self.calculate_rsi(close, rsi_period)
        
        # Calculate Stochastic
        k, d = self.calculate_stochastic(data, stoch_period, stoch_smooth)
        
        # Calculate price momentum (ROC)
        momentum = self.calculate_roc(close, 10)
        
        # Normalize momentum to 0-100 scale
        momentum_min = np.min(momentum[momentum != 0])
        momentum_max = np.max(momentum[momentum != 0])
        
        if momentum_max == momentum_min:
            normalized_momentum = np.zeros_like(momentum)
        else:
            normalized_momentum = 100.0 * (momentum - momentum_min) / (momentum_max - momentum_min)
            normalized_momentum[momentum == 0] = 0
        
        # Combine indicators
        xrp_oscillator = (rsi * 0.4) + (k * 0.3) + (normalized_momentum * 0.3)
        
        # Smooth with EMA
        xrp_oscillator_smooth = self.calculate_ema(xrp_oscillator, ema_period)
        
        return xrp_oscillator_smooth
    
    def calculate_custom_volatility_index(self, data: pd.DataFrame, 
                                         atr_period: int = 14,
                                         bb_period: int = 20,
                                         bb_std: float = 2.0) -> np.ndarray:
        """
        Calculate Custom Volatility Index.
        
        This is a custom indicator that combines ATR and Bollinger Bands width
        to measure market volatility.
        
        Args:
            data: OHLC DataFrame with 'high', 'low', and 'close' columns
            atr_period: ATR period
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviation
            
        Returns:
            ndarray with Volatility Index values
        """
        # Check if data has required columns
        required_columns = ['high', 'low', 'close']
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Data must have '{column}' column")
        
        # Get data
        close = data['close'].values
        
        # Calculate ATR
        atr = self.calculate_atr(data, atr_period)
        
        # Calculate Bollinger Bands
        upper, middle, lower = self.calculate_bollinger_bands(close, bb_period, bb_std)
        
        # Calculate Bollinger Bands width
        bb_width = (upper - lower) / middle * 100.0
        
        # Normalize ATR to 0-100 scale
        atr_min = np.min(atr[atr != 0])
        atr_max = np.max(atr[atr != 0])
        
        if atr_max == atr_min:
            normalized_atr = np.zeros_like(atr)
        else:
            normalized_atr = 100.0 * (atr - atr_min) / (atr_max - atr_min)
            normalized_atr[atr == 0] = 0
        
        # Normalize BB width to 0-100 scale
        bb_width_min = np.min(bb_width[bb_width != 0])
        bb_width_max = np.max(bb_width[bb_width != 0])
        
        if bb_width_max == bb_width_min:
            normalized_bb_width = np.zeros_like(bb_width)
        else:
            normalized_bb_width = 100.0 * (bb_width - bb_width_min) / (bb_width_max - bb_width_min)
            normalized_bb_width[bb_width == 0] = 0
        
        # Combine indicators
        volatility_index = (normalized_atr * 0.6) + (normalized_bb_width * 0.4)
        
        return volatility_index
    
    def calculate_custom_trend_strength(self, data: pd.DataFrame,
                                       adx_period: int = 14,
                                       ema_short: int = 9,
                                       ema_long: int = 21) -> np.ndarray:
        """
        Calculate Custom Trend Strength Index.
        
        This is a custom indicator that combines ADX and EMA crossovers
        to measure trend strength.
        
        Args:
            data: OHLC DataFrame with 'high', 'low', and 'close' columns
            adx_period: ADX period
            ema_short: Short EMA period
            ema_long: Long EMA period
            
        Returns:
            ndarray with Trend Strength values
        """
        # Check if data has required columns
        required_columns = ['high', 'low', 'close']
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Data must have '{column}' column")
        
        # Get data
        close = data['close'].values
        
        # Calculate ADX
        adx, plus_di, minus_di = self.calculate_adx(data, adx_period)
        
        # Calculate EMAs
        ema_short_values = self.calculate_ema(close, ema_short)
        ema_long_values = self.calculate_ema(close, ema_long)
        
        # Calculate EMA crossover signal
        ema_signal = np.zeros_like(close)
        
        for i in range(1, len(close)):
            if ema_short_values[i] > ema_long_values[i] and ema_short_values[i-1] <= ema_long_values[i-1]:
                # Bullish crossover
                ema_signal[i] = 100.0
            elif ema_short_values[i] < ema_long_values[i] and ema_short_values[i-1] >= ema_long_values[i-1]:
                # Bearish crossover
                ema_signal[i] = -100.0
            else:
                # No crossover
                ema_signal[i] = ema_signal[i-1] * 0.9  # Decay signal
        
        # Calculate DI crossover signal
        di_signal = np.zeros_like(close)
        
        for i in range(1, len(close)):
            if plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1]:
                # Bullish crossover
                di_signal[i] = 100.0
            elif plus_di[i] < minus_di[i] and plus_di[i-1] >= minus_di[i-1]:
                # Bearish crossover
                di_signal[i] = -100.0
            else:
                # No crossover
                di_signal[i] = di_signal[i-1] * 0.9  # Decay signal
        
        # Combine indicators
        trend_strength = np.zeros_like(close)
        
        for i in range(len(close)):
            # ADX component (0-100)
            adx_component = adx[i]
            
            # EMA signal component (-100 to 100)
            ema_component = ema_signal[i]
            
            # DI signal component (-100 to 100)
            di_component = di_signal[i]
            
            # Combine components
            if ema_component > 0 and di_component > 0:
                # Strong bullish trend
                trend_strength[i] = adx_component
            elif ema_component < 0 and di_component < 0:
                # Strong bearish trend
                trend_strength[i] = -adx_component
            else:
                # Mixed signals
                trend_strength[i] = (ema_component + di_component) / 2.0
        
        return trend_strength
