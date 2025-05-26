"""
Technical Indicators Calculator

This module provides a comprehensive set of technical indicators for market analysis.

Classes:
    TechnicalIndicators: Calculates various technical indicators
"""

import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/technical_indicators.log")
    ]
)
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    Technical indicators calculator for market analysis.
    
    This class provides methods for calculating various technical indicators
    used in market analysis and trading strategies.
    """
    
    def __init__(self):
        """
        Initialize the technical indicators calculator.
        """
        logger.info("Technical indicators calculator initialized")
    
    def calculate_rsi(self, prices, period=14):
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices (array-like): Price data
            period (int): RSI period
        
        Returns:
            array-like: RSI values
        """
        try:
            # Convert to numpy array if needed
            prices = np.array(prices)
            
            # Calculate price changes
            deltas = np.diff(prices)
            
            # Initialize seed values
            seed = deltas[:period+1]
            up = seed[seed >= 0].sum() / period
            down = -seed[seed < 0].sum() / period
            
            # Initialize RSI values
            rs_values = np.zeros_like(prices)
            rs_values[period] = 100 - (100 / (1 + up / down if down != 0 else 9999))
            
            # Calculate RSI values
            for i in range(period + 1, len(prices)):
                delta = deltas[i - 1]
                
                if delta > 0:
                    upval = delta
                    downval = 0
                else:
                    upval = 0
                    downval = -delta
                
                up = (up * (period - 1) + upval) / period
                down = (down * (period - 1) + downval) / period
                
                rs = up / down if down != 0 else 9999
                rs_values[i] = 100 - (100 / (1 + rs))
            
            # Fill initial values
            rs_values[:period] = rs_values[period]
            
            return rs_values
        
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return np.zeros_like(prices)
    
    def calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            prices (array-like): Price data
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal EMA period
        
        Returns:
            dict: MACD values including 'macd', 'signal', and 'histogram'
        """
        try:
            # Convert to numpy array if needed
            prices = np.array(prices)
            
            # Calculate EMAs
            fast_ema = self.calculate_ema(prices, fast_period)
            slow_ema = self.calculate_ema(prices, slow_period)
            
            # Calculate MACD line
            macd_line = fast_ema - slow_ema
            
            # Calculate signal line
            signal_line = self.calculate_ema(macd_line, signal_period)
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
        
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return {
                'macd': np.zeros_like(prices),
                'signal': np.zeros_like(prices),
                'histogram': np.zeros_like(prices)
            }
    
    def calculate_ema(self, prices, period):
        """
        Calculate Exponential Moving Average (EMA).
        
        Args:
            prices (array-like): Price data
            period (int): EMA period
        
        Returns:
            array-like: EMA values
        """
        try:
            # Convert to numpy array if needed
            prices = np.array(prices)
            
            # Calculate EMA
            ema = np.zeros_like(prices)
            ema[:period] = np.mean(prices[:period])
            
            # Calculate multiplier
            multiplier = 2 / (period + 1)
            
            # Calculate EMA values
            for i in range(period, len(prices)):
                ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
            
            return ema
        
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return np.zeros_like(prices)
    
    def calculate_sma(self, prices, period):
        """
        Calculate Simple Moving Average (SMA).
        
        Args:
            prices (array-like): Price data
            period (int): SMA period
        
        Returns:
            array-like: SMA values
        """
        try:
            # Convert to numpy array if needed
            prices = np.array(prices)
            
            # Calculate SMA
            sma = np.zeros_like(prices)
            
            for i in range(len(prices)):
                if i < period - 1:
                    sma[i] = np.mean(prices[:i+1])
                else:
                    sma[i] = np.mean(prices[i-period+1:i+1])
            
            return sma
        
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return np.zeros_like(prices)
    
    def calculate_bollinger_bands(self, prices, period=20, num_std=2):
        """
        Calculate Bollinger Bands.
        
        Args:
            prices (array-like): Price data
            period (int): Period for moving average
            num_std (float): Number of standard deviations
        
        Returns:
            dict: Bollinger Bands values including 'upper', 'middle', and 'lower'
        """
        try:
            # Convert to numpy array if needed
            prices = np.array(prices)
            
            # Calculate middle band (SMA)
            middle_band = self.calculate_sma(prices, period)
            
            # Calculate standard deviation
            std = np.zeros_like(prices)
            
            for i in range(len(prices)):
                if i < period - 1:
                    std[i] = np.std(prices[:i+1])
                else:
                    std[i] = np.std(prices[i-period+1:i+1])
            
            # Calculate upper and lower bands
            upper_band = middle_band + num_std * std
            lower_band = middle_band - num_std * std
            
            return {
                'upper': upper_band,
                'middle': middle_band,
                'lower': lower_band
            }
        
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return {
                'upper': np.zeros_like(prices),
                'middle': np.zeros_like(prices),
                'lower': np.zeros_like(prices)
            }
    
    def calculate_vwma(self, prices, volumes, period=20):
        """
        Calculate Volume Weighted Moving Average (VWMA).
        
        Args:
            prices (array-like): Price data
            volumes (array-like): Volume data
            period (int): VWMA period
        
        Returns:
            array-like: VWMA values
        """
        try:
            # Convert to numpy arrays if needed
            prices = np.array(prices)
            volumes = np.array(volumes)
            
            # Calculate VWMA
            vwma = np.zeros_like(prices)
            
            for i in range(len(prices)):
                if i < period - 1:
                    if np.sum(volumes[:i+1]) != 0:
                        vwma[i] = np.sum(prices[:i+1] * volumes[:i+1]) / np.sum(volumes[:i+1])
                    else:
                        vwma[i] = np.mean(prices[:i+1])
                else:
                    if np.sum(volumes[i-period+1:i+1]) != 0:
                        vwma[i] = np.sum(prices[i-period+1:i+1] * volumes[i-period+1:i+1]) / np.sum(volumes[i-period+1:i+1])
                    else:
                        vwma[i] = np.mean(prices[i-period+1:i+1])
            
            return vwma
        
        except Exception as e:
            logger.error(f"Error calculating VWMA: {e}")
            return np.zeros_like(prices)
    
    def calculate_atr(self, high_prices, low_prices, close_prices, period=14):
        """
        Calculate Average True Range (ATR).
        
        Args:
            high_prices (array-like): High price data
            low_prices (array-like): Low price data
            close_prices (array-like): Close price data
            period (int): ATR period
        
        Returns:
            array-like: ATR values
        """
        try:
            # Convert to numpy arrays if needed
            high_prices = np.array(high_prices)
            low_prices = np.array(low_prices)
            close_prices = np.array(close_prices)
            
            # Calculate true range
            tr = np.zeros_like(close_prices)
            
            for i in range(len(close_prices)):
                if i == 0:
                    tr[i] = high_prices[i] - low_prices[i]
                else:
                    tr[i] = max(
                        high_prices[i] - low_prices[i],
                        abs(high_prices[i] - close_prices[i-1]),
                        abs(low_prices[i] - close_prices[i-1])
                    )
            
            # Calculate ATR
            atr = np.zeros_like(close_prices)
            atr[0] = tr[0]
            
            for i in range(1, len(close_prices)):
                if i < period:
                    atr[i] = np.mean(tr[:i+1])
                else:
                    atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            
            return atr
        
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return np.zeros_like(close_prices)
    
    def calculate_stochastic(self, high_prices, low_prices, close_prices, k_period=14, d_period=3):
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high_prices (array-like): High price data
            low_prices (array-like): Low price data
            close_prices (array-like): Close price data
            k_period (int): %K period
            d_period (int): %D period
        
        Returns:
            dict: Stochastic values including 'k' and 'd'
        """
        try:
            # Convert to numpy arrays if needed
            high_prices = np.array(high_prices)
            low_prices = np.array(low_prices)
            close_prices = np.array(close_prices)
            
            # Calculate %K
            k = np.zeros_like(close_prices)
            
            for i in range(len(close_prices)):
                if i < k_period - 1:
                    highest_high = np.max(high_prices[:i+1])
                    lowest_low = np.min(low_prices[:i+1])
                else:
                    highest_high = np.max(high_prices[i-k_period+1:i+1])
                    lowest_low = np.min(low_prices[i-k_period+1:i+1])
                
                if highest_high != lowest_low:
                    k[i] = 100 * (close_prices[i] - lowest_low) / (highest_high - lowest_low)
                else:
                    k[i] = 50
            
            # Calculate %D (SMA of %K)
            d = self.calculate_sma(k, d_period)
            
            return {
                'k': k,
                'd': d
            }
        
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            return {
                'k': np.zeros_like(close_prices),
                'd': np.zeros_like(close_prices)
            }
    
    def calculate_obv(self, close_prices, volumes):
        """
        Calculate On-Balance Volume (OBV).
        
        Args:
            close_prices (array-like): Close price data
            volumes (array-like): Volume data
        
        Returns:
            array-like: OBV values
        """
        try:
            # Convert to numpy arrays if needed
            close_prices = np.array(close_prices)
            volumes = np.array(volumes)
            
            # Calculate OBV
            obv = np.zeros_like(close_prices)
            
            for i in range(len(close_prices)):
                if i == 0:
                    obv[i] = volumes[i]
                else:
                    if close_prices[i] > close_prices[i-1]:
                        obv[i] = obv[i-1] + volumes[i]
                    elif close_prices[i] < close_prices[i-1]:
                        obv[i] = obv[i-1] - volumes[i]
                    else:
                        obv[i] = obv[i-1]
            
            return obv
        
        except Exception as e:
            logger.error(f"Error calculating OBV: {e}")
            return np.zeros_like(close_prices)
    
    def calculate_ichimoku(self, high_prices, low_prices, close_prices, tenkan_period=9, kijun_period=26, senkou_b_period=52, displacement=26):
        """
        Calculate Ichimoku Cloud.
        
        Args:
            high_prices (array-like): High price data
            low_prices (array-like): Low price data
            close_prices (array-like): Close price data
            tenkan_period (int): Tenkan-sen (Conversion Line) period
            kijun_period (int): Kijun-sen (Base Line) period
            senkou_b_period (int): Senkou Span B period
            displacement (int): Displacement period
        
        Returns:
            dict: Ichimoku values including 'tenkan', 'kijun', 'senkou_a', 'senkou_b', and 'chikou'
        """
        try:
            # Convert to numpy arrays if needed
            high_prices = np.array(high_prices)
            low_prices = np.array(low_prices)
            close_prices = np.array(close_prices)
            
            # Calculate Tenkan-sen (Conversion Line)
            tenkan = np.zeros_like(close_prices)
            
            for i in range(len(close_prices)):
                if i < tenkan_period - 1:
                    highest_high = np.max(high_prices[:i+1])
                    lowest_low = np.min(low_prices[:i+1])
                else:
                    highest_high = np.max(high_prices[i-tenkan_period+1:i+1])
                    lowest_low = np.min(low_prices[i-tenkan_period+1:i+1])
                
                tenkan[i] = (highest_high + lowest_low) / 2
            
            # Calculate Kijun-sen (Base Line)
            kijun = np.zeros_like(close_prices)
            
            for i in range(len(close_prices)):
                if i < kijun_period - 1:
                    highest_high = np.max(high_prices[:i+1])
                    lowest_low = np.min(low_prices[:i+1])
                else:
                    highest_high = np.max(high_prices[i-kijun_period+1:i+1])
                    lowest_low = np.min(low_prices[i-kijun_period+1:i+1])
                
                kijun[i] = (highest_high + lowest_low) / 2
            
            # Calculate Senkou Span A (Leading Span A)
            senkou_a = np.zeros_like(close_prices)
            
            for i in range(len(close_prices)):
                senkou_a[i] = (tenkan[i] + kijun[i]) / 2
            
            # Calculate Senkou Span B (Leading Span B)
            senkou_b = np.zeros_like(close_prices)
            
            for i in range(len(close_prices)):
                if i < senkou_b_period - 1:
                    highest_high = np.max(high_prices[:i+1])
                    lowest_low = np.min(low_prices[:i+1])
                else:
                    highest_high = np.max(high_prices[i-senkou_b_period+1:i+1])
                    lowest_low = np.min(low_prices[i-senkou_b_period+1:i+1])
                
                senkou_b[i] = (highest_high + lowest_low) / 2
            
            # Calculate Chikou Span (Lagging Span)
            chikou = np.zeros_like(close_prices)
            
            for i in range(len(close_prices)):
                if i < displacement:
                    chikou[i] = close_prices[0]
                else:
                    chikou[i] = close_prices[i-displacement]
            
            return {
                'tenkan': tenkan,
                'kijun': kijun,
                'senkou_a': senkou_a,
                'senkou_b': senkou_b,
                'chikou': chikou
            }
        
        except Exception as e:
            logger.error(f"Error calculating Ichimoku: {e}")
            return {
                'tenkan': np.zeros_like(close_prices),
                'kijun': np.zeros_like(close_prices),
                'senkou_a': np.zeros_like(close_prices),
                'senkou_b': np.zeros_like(close_prices),
                'chikou': np.zeros_like(close_prices)
            }
    
    def calculate_all_indicators(self, data):
        """
        Calculate all technical indicators.
        
        Args:
            data (DataFrame): Price data with 'open', 'high', 'low', 'close', and 'volume' columns
        
        Returns:
            DataFrame: Data with all technical indicators
        """
        try:
            # Create a copy of the data
            result = data.copy()
            
            # Calculate RSI
            result['rsi'] = self.calculate_rsi(result['close'].values)
            
            # Calculate MACD
            macd_result = self.calculate_macd(result['close'].values)
            result['macd'] = macd_result['macd']
            result['macd_signal'] = macd_result['signal']
            result['macd_histogram'] = macd_result['histogram']
            
            # Calculate Bollinger Bands
            bb_result = self.calculate_bollinger_bands(result['close'].values)
            result['bb_upper'] = bb_result['upper']
            result['bb_middle'] = bb_result['middle']
            result['bb_lower'] = bb_result['lower']
            
            # Calculate VWMA
            result['vwma_20'] = self.calculate_vwma(result['close'].values, result['volume'].values, 20)
            result['vwma_50'] = self.calculate_vwma(result['close'].values, result['volume'].values, 50)
            
            # Calculate ATR
            result['atr'] = self.calculate_atr(result['high'].values, result['low'].values, result['close'].values)
            
            # Calculate Stochastic
            stoch_result = self.calculate_stochastic(result['high'].values, result['low'].values, result['close'].values)
            result['stoch_k'] = stoch_result['k']
            result['stoch_d'] = stoch_result['d']
            
            # Calculate OBV
            result['obv'] = self.calculate_obv(result['close'].values, result['volume'].values)
            
            return result
        
        except Exception as e:
            logger.error(f"Error calculating all indicators: {e}")
            return data
