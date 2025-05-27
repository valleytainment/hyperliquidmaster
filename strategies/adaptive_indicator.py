"""
Adaptive Technical Indicators

This module provides technical indicators that can adapt to limited data scenarios
by using fallback logic and synthetic data enhancement.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class AdaptiveIndicator:
    """
    Technical indicators that adapt to limited data scenarios.
    """
    
    @staticmethod
    def vwma(df: pd.DataFrame, period: int = 20, price_col: str = 'close', volume_col: str = 'volume') -> pd.Series:
        """
        Calculate Volume Weighted Moving Average with fallback for limited data.
        
        Args:
            df: DataFrame with price and volume data
            period: VWMA period
            price_col: Column name for price data
            volume_col: Column name for volume data
            
        Returns:
            Series with VWMA values
        """
        try:
            # Check if we have enough data
            if len(df) < period:
                logger.warning(f"Not enough data for VWMA({period}): {len(df)} < {period}")
                
                # Fallback: Use simple moving average with available data
                return df[price_col].rolling(window=min(period, max(2, len(df)))).mean()
                
            # Calculate VWMA normally
            vwma = df[price_col].multiply(df[volume_col]).rolling(window=period).sum() / df[volume_col].rolling(window=period).sum()
            return vwma
        except Exception as e:
            logger.error(f"Error calculating VWMA: {str(e)}")
            
            # Emergency fallback: Return price series
            return df[price_col]
            
    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14, price_col: str = 'close') -> pd.Series:
        """
        Calculate Relative Strength Index with fallback for limited data.
        
        Args:
            df: DataFrame with price data
            period: RSI period
            price_col: Column name for price data
            
        Returns:
            Series with RSI values
        """
        try:
            # Check if we have enough data
            if len(df) < period + 1:
                logger.warning(f"Not enough data for RSI({period}): {len(df)} < {period + 1}")
                
                # Fallback: Generate synthetic price changes to calculate RSI
                if len(df) >= 2:
                    # Use available data plus synthetic extension
                    actual_changes = df[price_col].diff().dropna()
                    
                    # Generate synthetic changes with similar characteristics
                    mean_change = actual_changes.mean()
                    std_change = max(actual_changes.std(), df[price_col].iloc[-1] * 0.001)  # Ensure some volatility
                    
                    # Generate synthetic changes
                    np.random.seed(42)  # For reproducibility
                    synthetic_count = period + 1 - len(df)
                    synthetic_changes = np.random.normal(mean_change, std_change, synthetic_count)
                    
                    # Combine actual and synthetic changes
                    all_changes = np.concatenate([synthetic_changes, actual_changes.values])
                else:
                    # No actual changes available, use default values
                    all_changes = np.random.normal(0, df[price_col].iloc[-1] * 0.01, period + 1)
                    
                # Calculate RSI from combined changes
                gains = np.maximum(all_changes, 0)
                losses = np.maximum(-all_changes, 0)
                
                avg_gain = np.mean(gains[-period:])
                avg_loss = np.mean(losses[-period:])
                
                if avg_loss == 0:
                    rsi_value = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi_value = 100 - (100 / (1 + rs))
                    
                # Return a Series with the same index as the input DataFrame
                result = pd.Series(index=df.index, dtype=float)
                result.iloc[-1] = rsi_value
                return result
                
            # Calculate RSI normally
            delta = df[price_col].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            
            # Emergency fallback: Return middle value
            return pd.Series(50, index=df.index)
            
    @staticmethod
    def bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, price_col: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands with fallback for limited data.
        
        Args:
            df: DataFrame with price data
            period: Bollinger Bands period
            std_dev: Standard deviation multiplier
            price_col: Column name for price data
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        try:
            # Check if we have enough data
            if len(df) < period:
                logger.warning(f"Not enough data for Bollinger Bands({period}): {len(df)} < {period}")
                
                # Fallback: Use available data with adjusted period
                adjusted_period = min(period, max(2, len(df)))
                
                # Calculate with adjusted period
                middle_band = df[price_col].rolling(window=adjusted_period).mean()
                
                # For very limited data, use a fixed percentage for bands
                if len(df) < 5:
                    price = df[price_col].iloc[-1]
                    band_width = price * 0.02 * std_dev  # 2% per std_dev
                    
                    upper_band = pd.Series(index=df.index, dtype=float)
                    lower_band = pd.Series(index=df.index, dtype=float)
                    
                    upper_band.iloc[-1] = price + band_width
                    lower_band.iloc[-1] = price - band_width
                else:
                    # Use rolling standard deviation with available data
                    rolling_std = df[price_col].rolling(window=adjusted_period).std()
                    upper_band = middle_band + (rolling_std * std_dev)
                    lower_band = middle_band - (rolling_std * std_dev)
                    
                return upper_band, middle_band, lower_band
                
            # Calculate Bollinger Bands normally
            middle_band = df[price_col].rolling(window=period).mean()
            rolling_std = df[price_col].rolling(window=period).std()
            
            upper_band = middle_band + (rolling_std * std_dev)
            lower_band = middle_band - (rolling_std * std_dev)
            
            return upper_band, middle_band, lower_band
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            
            # Emergency fallback: Return price +/- 2%
            price = df[price_col].iloc[-1] if not df.empty else 100
            band_width = price * 0.02 * std_dev
            
            middle_band = pd.Series(price, index=df.index)
            upper_band = pd.Series(price + band_width, index=df.index)
            lower_band = pd.Series(price - band_width, index=df.index)
            
            return upper_band, middle_band, lower_band
            
    @staticmethod
    def macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, price_col: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD with fallback for limited data.
        
        Args:
            df: DataFrame with price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal EMA period
            price_col: Column name for price data
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        try:
            # Check if we have enough data
            if len(df) < slow_period:
                logger.warning(f"Not enough data for MACD: {len(df)} < {slow_period}")
                
                # Fallback: Use simple moving averages with available data
                adjusted_fast = min(fast_period, max(2, len(df) - 1))
                adjusted_slow = min(slow_period, max(3, len(df)))
                adjusted_signal = min(signal_period, max(2, len(df) - adjusted_slow))
                
                # Calculate with adjusted periods
                fast_ma = df[price_col].rolling(window=adjusted_fast).mean()
                slow_ma = df[price_col].rolling(window=adjusted_slow).mean()
                
                macd_line = fast_ma - slow_ma
                signal_line = macd_line.rolling(window=adjusted_signal).mean()
                histogram = macd_line - signal_line
                
                return macd_line, signal_line, histogram
                
            # Calculate MACD normally
            exp1 = df[price_col].ewm(span=fast_period, adjust=False).mean()
            exp2 = df[price_col].ewm(span=slow_period, adjust=False).mean()
            
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            histogram = macd_line - signal_line
            
            return macd_line, signal_line, histogram
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            
            # Emergency fallback: Return zeros
            zeros = pd.Series(0, index=df.index)
            return zeros, zeros, zeros
            
    @staticmethod
    def detect_market_regime(df: pd.DataFrame, lookback: int = 20, price_col: str = 'close') -> str:
        """
        Detect market regime (trending, ranging, volatile) with fallback for limited data.
        
        Args:
            df: DataFrame with price data
            lookback: Lookback period
            price_col: Column name for price data
            
        Returns:
            Market regime as string: 'trending', 'ranging', or 'volatile'
        """
        try:
            # Check if we have enough data
            if len(df) < lookback:
                logger.warning(f"Not enough data for market regime detection: {len(df)} < {lookback}")
                
                # Fallback: Use available data or return default
                if len(df) < 5:
                    # Not enough data for any meaningful detection
                    return 'unknown'
                    
                # Use available data with adjusted lookback
                adjusted_lookback = min(lookback, max(5, len(df)))
                
                # Get recent prices
                recent_prices = df[price_col].iloc[-adjusted_lookback:]
            else:
                # Use normal lookback
                recent_prices = df[price_col].iloc[-lookback:]
                
            # Calculate metrics for regime detection
            returns = recent_prices.pct_change().dropna()
            
            # Volatility (annualized)
            volatility = returns.std() * np.sqrt(252) * 100  # As percentage
            
            # Trend strength (absolute value of cumulative return)
            trend_strength = abs(returns.sum()) * 100  # As percentage
            
            # Efficiency ratio (directional movement / total movement)
            price_change = abs(recent_prices.iloc[-1] - recent_prices.iloc[0])
            total_movement = abs(recent_prices.diff()).sum()
            efficiency_ratio = price_change / total_movement if total_movement > 0 else 0
            
            # Determine regime
            if volatility > 5:  # High volatility threshold
                regime = 'volatile'
            elif efficiency_ratio > 0.5 and trend_strength > 2:  # Strong directional movement
                regime = 'trending'
            else:
                regime = 'ranging'
                
            logger.info(f"Market regime detected: {regime} (volatility: {volatility:.2f}%, trend: {trend_strength:.2f}%, efficiency: {efficiency_ratio:.2f})")
            return regime
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            
            # Emergency fallback: Return unknown
            return 'unknown'
            
    @staticmethod
    def detect_support_resistance(df: pd.DataFrame, lookback: int = 50, price_col: str = 'close', threshold: float = 0.02) -> Tuple[float, float]:
        """
        Detect support and resistance levels with fallback for limited data.
        
        Args:
            df: DataFrame with price data
            lookback: Lookback period
            price_col: Column name for price data
            threshold: Threshold for level detection (as percentage)
            
        Returns:
            Tuple of (support_level, resistance_level)
        """
        try:
            # Check if we have enough data
            if len(df) < lookback:
                logger.warning(f"Not enough data for support/resistance detection: {len(df)} < {lookback}")
                
                # Fallback: Use Bollinger Bands
                upper, middle, lower = AdaptiveIndicator.bollinger_bands(df, period=min(20, max(2, len(df))), price_col=price_col)
                
                # Use the last values
                support = lower.iloc[-1]
                resistance = upper.iloc[-1]
                
                return support, resistance
                
            # Use normal detection
            recent_prices = df[price_col].iloc[-lookback:]
            current_price = recent_prices.iloc[-1]
            
            # Find potential support/resistance levels using price clusters
            price_range = recent_prices.max() - recent_prices.min()
            bin_width = price_range * threshold
            
            # Create histogram bins
            bins = int(price_range / bin_width) + 1
            hist, bin_edges = np.histogram(recent_prices, bins=bins)
            
            # Find bins with high frequency
            threshold_count = np.percentile(hist, 80)  # Top 20% of bins
            high_freq_bins = [i for i, count in enumerate(hist) if count >= threshold_count]
            
            # Convert bins to price levels
            levels = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in high_freq_bins]
            
            # Find nearest support (below current price) and resistance (above current price)
            support_levels = [level for level in levels if level < current_price]
            resistance_levels = [level for level in levels if level > current_price]
            
            support = max(support_levels) if support_levels else current_price * 0.95
            resistance = min(resistance_levels) if resistance_levels else current_price * 1.05
            
            return support, resistance
        except Exception as e:
            logger.error(f"Error detecting support/resistance: {str(e)}")
            
            # Emergency fallback: Return current price +/- 5%
            current_price = df[price_col].iloc[-1] if not df.empty else 100
            return current_price * 0.95, current_price * 1.05
