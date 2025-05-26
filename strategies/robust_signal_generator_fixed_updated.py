"""
Robust Signal Generator

This module provides robust signal generation functions that can handle
limited data, missing values, and edge cases in DataFrames.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from strategies.enhanced_vwma import EnhancedVWMAIndicator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class RobustSignalGenerator:
    """
    Generates robust trading signals that can handle limited data and edge cases.
    """
    
    def __init__(self, technical_indicators=None, error_handler=None):
        """
        Initialize the RobustSignalGenerator with technical indicators and error handler.
        
        Args:
            technical_indicators: Technical indicators calculator instance
            error_handler: Error handler instance for managing exceptions
        """
        self.technical_indicators = technical_indicators
        self.error_handler = error_handler
        logger.info("RobustSignalGenerator initialized with technical indicators and error handler")
    
    @staticmethod
    def safe_get(df: pd.DataFrame, col: str, idx: int, default: float = None) -> float:
        """
        Safely get a value from a DataFrame with fallback.
        
        Args:
            df: DataFrame to get value from
            col: Column name
            idx: Index position (-1 for last element)
            default: Default value if not available
            
        Returns:
            Value or default
        """
        try:
            if df is None or df.empty or col not in df.columns:
                return default
                
            if idx >= len(df) or idx < -len(df):
                return default
                
            value = df.iloc[idx][col]
            
            if pd.isna(value):
                return default
                
            return value
        except Exception as e:
            logger.error(f"Error in safe_get({col}, {idx}): {str(e)}")
            return default
            
    @staticmethod
    def safe_rolling(series: pd.Series, window: int, func: str = 'mean', min_periods: int = 1) -> pd.Series:
        """
        Safely apply a rolling function with fallback for short series.
        
        Args:
            series: Series to apply rolling function to
            window: Rolling window size
            func: Function to apply ('mean', 'std', 'min', 'max')
            min_periods: Minimum number of periods required
            
        Returns:
            Series with rolling function applied
        """
        try:
            if series is None or len(series) == 0:
                return pd.Series()
                
            # Adjust window if series is too short
            adjusted_window = min(window, len(series))
            
            if adjusted_window < min_periods:
                # Not enough data, return series of NaN
                return pd.Series(index=series.index)
                
            # Apply rolling function
            if func == 'mean':
                return series.rolling(window=adjusted_window, min_periods=min_periods).mean()
            elif func == 'std':
                return series.rolling(window=adjusted_window, min_periods=min_periods).std()
            elif func == 'min':
                return series.rolling(window=adjusted_window, min_periods=min_periods).min()
            elif func == 'max':
                return series.rolling(window=adjusted_window, min_periods=min_periods).max()
            else:
                logger.warning(f"Unknown rolling function: {func}")
                return series.rolling(window=adjusted_window, min_periods=min_periods).mean()
        except Exception as e:
            logger.error(f"Error in safe_rolling: {str(e)}")
            return pd.Series(index=series.index if series is not None else [])
    
    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """
        Detect market regime (trending, ranging, volatile) based on price action.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Market regime as string: 'trending_up', 'trending_down', 'ranging', 'volatile', or 'unknown'
        """
        try:
            if df is None or df.empty or len(df) < 10:
                logger.warning("Insufficient data for market regime detection")
                return "unknown"
                
            # Calculate metrics for regime detection
            close = df['close']
            
            # Use adjusted periods based on available data
            atr_period = min(14, len(df) - 1)
            volatility_period = min(20, len(df) - 1)
            trend_period = min(50, len(df) - 1)
            
            # Calculate ATR (Average True Range) for volatility
            high = df['high'] if 'high' in df.columns else close
            low = df['low'] if 'low' in df.columns else close
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=atr_period, min_periods=1).mean()
            
            # Calculate volatility as percentage of price
            volatility = atr.iloc[-1] / close.iloc[-1] * 100
            
            # Calculate directional movement
            direction = (close.iloc[-1] - close.iloc[-trend_period]) / close.iloc[-trend_period] * 100
            
            # Calculate price range as percentage of average price
            price_range = (high.iloc[-volatility_period:].max() - low.iloc[-volatility_period:].min()) / close.mean() * 100
            
            # Determine regime based on metrics
            if volatility > 5:  # High volatility threshold
                return "volatile"
            elif abs(direction) > 10:  # Strong trend threshold
                return "trending_up" if direction > 0 else "trending_down"
            elif price_range < 10:  # Tight range threshold
                return "ranging"
            else:
                # Default to ranging if no clear pattern
                return "ranging"
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error("detect_market_regime", str(e))
            else:
                logger.error(f"Error detecting market regime: {str(e)}")
            return "unknown"
            
    def check_vwma_crossover(self, df: pd.DataFrame, fast_period: int = 20, slow_period: int = 50) -> Tuple[bool, bool, float]:
        """
        Check for VWMA crossover with robust handling of limited data.
        
        Args:
            df: DataFrame with price and volume data
            fast_period: Fast VWMA period
            slow_period: Slow VWMA period
            
        Returns:
            Tuple of (bullish_crossover, bearish_crossover, strength)
        """
        try:
            # Use the enhanced VWMA indicator for more robust crossover detection
            return EnhancedVWMAIndicator.check_vwma_crossover(df, fast_period, slow_period)
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error("check_vwma_crossover", str(e))
            else:
                logger.error(f"Error checking VWMA crossover: {str(e)}")
            return False, False, 0.0
            
    def check_rsi_signal(self, df: pd.DataFrame, period: int = 14, overbought: float = 70, oversold: float = 30) -> Tuple[bool, bool, float]:
        """
        Check for RSI signals with robust handling of limited data.
        
        Args:
            df: DataFrame with price data
            period: RSI period
            overbought: Overbought threshold
            oversold: Oversold threshold
            
        Returns:
            Tuple of (buy_signal, sell_signal, strength)
        """
        try:
            if df is None or df.empty:
                logger.warning("No data for RSI signal check")
                return False, False, 0.0
                
            # Use technical indicators if available
            if self.technical_indicators and hasattr(self.technical_indicators, 'calculate_rsi'):
                rsi_values = self.technical_indicators.calculate_rsi(df, period)
                if isinstance(rsi_values, pd.Series) or isinstance(rsi_values, pd.DataFrame):
                    if len(rsi_values) >= 2:
                        current_rsi = rsi_values.iloc[-1]
                        prev_rsi = rsi_values.iloc[-2]
                    else:
                        return False, False, 0.0
                elif isinstance(rsi_values, list) and len(rsi_values) >= 2:
                    # Handle list type
                    current_rsi = rsi_values[-1]
                    prev_rsi = rsi_values[-2]
                else:
                    return False, False, 0.0
            else:
                # Check if RSI column exists
                if f'rsi_{period}' in df.columns:
                    # Use pre-calculated RSI
                    rsi = df[f'rsi_{period}']
                    current_rsi = RobustSignalGenerator.safe_get(rsi.to_frame('value'), 'value', -1, 50)
                    prev_rsi = RobustSignalGenerator.safe_get(rsi.to_frame('value'), 'value', -2, 50)
                else:
                    # Not enough data for reliable RSI
                    if len(df) < 3:
                        logger.warning("Not enough data for RSI calculation")
                        return False, False, 0.0
                        
                    # Calculate RSI using available data
                    delta = df['close'].diff().dropna()
                    
                    if len(delta) == 0:
                        logger.warning("No price changes for RSI calculation")
                        return False, False, 0.0
                        
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    
                    # Use adjusted period if not enough data
                    adjusted_period = min(period, len(gain))
                    
                    if adjusted_period < 2:
                        logger.warning("Not enough data for RSI calculation")
                        return False, False, 0.0
                        
                    avg_gain = gain.rolling(window=adjusted_period, min_periods=1).mean()
                    avg_loss = loss.rolling(window=adjusted_period, min_periods=1).mean()
                    
                    rs = avg_gain / avg_loss.replace(0, 0.001)  # Avoid division by zero
                    rsi = 100 - (100 / (1 + rs))
                    
                    current_rsi = RobustSignalGenerator.safe_get(rsi.to_frame('value'), 'value', -1, 50)
                    prev_rsi = RobustSignalGenerator.safe_get(rsi.to_frame('value'), 'value', -2, 50)
                
            # Check for signals
            buy_signal = prev_rsi < oversold and current_rsi >= oversold
            sell_signal = prev_rsi > overbought and current_rsi <= overbought
            
            # Calculate strength
            if buy_signal:
                strength = (oversold - current_rsi) / oversold
            elif sell_signal:
                strength = (current_rsi - overbought) / (100 - overbought)
            else:
                strength = 0.0
                
            return buy_signal, sell_signal, abs(strength)
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error("check_rsi_signal", str(e))
            else:
                logger.error(f"Error checking RSI signal: {str(e)}")
            return False, False, 0.0
            
    def check_bollinger_signal(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Tuple[bool, bool, float]:
        """
        Check for Bollinger Band signals with robust handling of limited data.
        
        Args:
            df: DataFrame with price data
            period: Bollinger Band period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (buy_signal, sell_signal, strength)
        """
        try:
            if df is None or df.empty or len(df) < 3:
                logger.warning("Not enough data for Bollinger Band signal check")
                return False, False, 0.0
                
            # Check if Bollinger Band columns exist
            if 'bb_upper' in df.columns and 'bb_middle' in df.columns and 'bb_lower' in df.columns:
                # Use pre-calculated Bollinger Bands
                upper = df['bb_upper']
                middle = df['bb_middle']
                lower = df['bb_lower']
            else:
                # Calculate Bollinger Bands using available data
                adjusted_period = min(period, len(df))
                
                if adjusted_period < 2:
                    logger.warning("Not enough data for Bollinger Band calculation")
                    return False, False, 0.0
                    
                middle = df['close'].rolling(window=adjusted_period, min_periods=1).mean()
                std = df['close'].rolling(window=adjusted_period, min_periods=1).std()
                
                upper = middle + (std * std_dev)
                lower = middle - (std * std_dev)
                
            # Get current values
            current_price = RobustSignalGenerator.safe_get(df, 'close', -1, None)
            current_upper = RobustSignalGenerator.safe_get(upper.to_frame('value'), 'value', -1, None)
            current_middle = RobustSignalGenerator.safe_get(middle.to_frame('value'), 'value', -1, None)
            current_lower = RobustSignalGenerator.safe_get(lower.to_frame('value'), 'value', -1, None)
            
            if current_price is None or current_upper is None or current_middle is None or current_lower is None:
                logger.warning("Missing values for Bollinger Band signal check")
                return False, False, 0.0
                
            # Check for signals
            buy_signal = current_price <= current_lower
            sell_signal = current_price >= current_upper
            
            # Calculate strength
            band_width = current_upper - current_lower
            
            if band_width == 0:
                strength = 0.0
            elif buy_signal:
                strength = (current_lower - current_price) / band_width
            elif sell_signal:
                strength = (current_price - current_upper) / band_width
            else:
                strength = 0.0
                
            return buy_signal, sell_signal, abs(strength)
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error("check_bollinger_signal", str(e))
            else:
                logger.error(f"Error checking Bollinger Band signal: {str(e)}")
            return False, False, 0.0
            
    def check_macd_signal(self, df: pd.DataFrame) -> Tuple[bool, bool, float]:
        """
        Check for MACD signals with robust handling of limited data.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Tuple of (buy_signal, sell_signal, strength)
        """
        try:
            if df is None or df.empty or len(df) < 3:
                logger.warning("Not enough data for MACD signal check")
                return False, False, 0.0
                
            # Check if MACD columns exist
            if 'macd_line' in df.columns and 'macd_signal' in df.columns and 'macd_histogram' in df.columns:
                # Use pre-calculated MACD
                macd_line = df['macd_line']
                signal_line = df['macd_signal']
                histogram = df['macd_histogram']
            else:
                # Not enough data for reliable MACD
                if len(df) < 10:
                    logger.warning("Not enough data for MACD calculation")
                    return False, False, 0.0
                    
                # Calculate MACD using available data
                close = df['close']
                
                # Use adjusted periods if not enough data
                fast_period = min(12, len(close) - 1)
                slow_period = min(26, len(close) - 1)
                signal_period = min(9, len(close) - 1)
                
                if fast_period < 2 or slow_period < 2 or signal_period < 2:
                    logger.warning("Not enough data for MACD calculation")
                    return False, False, 0.0
                    
                # Calculate EMAs
                fast_ema = close.ewm(span=fast_period, adjust=False).mean()
                slow_ema = close.ewm(span=slow_period, adjust=False).mean()
                
                # Calculate MACD line and signal line
                macd_line = fast_ema - slow_ema
                signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
                
                # Calculate histogram
                histogram = macd_line - signal_line
                
            # Get current and previous values
            current_macd = RobustSignalGenerator.safe_get(macd_line.to_frame('value'), 'value', -1, 0)
            current_signal = RobustSignalGenerator.safe_get(signal_line.to_frame('value'), 'value', -1, 0)
            current_hist = RobustSignalGenerator.safe_get(histogram.to_frame('value'), 'value', -1, 0)
            
            prev_macd = RobustSignalGenerator.safe_get(macd_line.to_frame('value'), 'value', -2, 0)
            prev_signal = RobustSignalGenerator.safe_get(signal_line.to_frame('value'), 'value', -2, 0)
            prev_hist = RobustSignalGenerator.safe_get(histogram.to_frame('value'), 'value', -2, 0)
            
            # Check for signals
            buy_signal = prev_macd < prev_signal and current_macd >= current_signal
            sell_signal = prev_macd > prev_signal and current_macd <= current_signal
            
            # Calculate strength
            if current_macd == 0 and current_signal == 0:
                strength = 0.0
            else:
                strength = abs(current_hist) / max(abs(current_macd), abs(current_signal), 0.001)
                
            return buy_signal, sell_signal, min(strength, 1.0)
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error("check_macd_signal", str(e))
            else:
                logger.error(f"Error checking MACD signal: {str(e)}")
            return False, False, 0.0
            
    def check_volume_signal(self, df: pd.DataFrame, period: int = 20) -> Tuple[bool, bool, float]:
        """
        Check for volume signals with robust handling of limited data.
        
        Args:
            df: DataFrame with price and volume data
            period: Volume average period
            
        Returns:
            Tuple of (buy_signal, sell_signal, strength)
        """
        try:
            if df is None or df.empty or 'volume' not in df.columns:
                logger.warning("No volume data for volume signal check")
                return False, False, 0.0
                
            # Not enough data for reliable volume analysis
            if len(df) < 3:
                logger.warning("Not enough data for volume signal check")
                return False, False, 0.0
                
            # Get current and previous values
            current_close = RobustSignalGenerator.safe_get(df, 'close', -1, None)
            prev_close = RobustSignalGenerator.safe_get(df, 'close', -2, None)
            
            current_volume = RobustSignalGenerator.safe_get(df, 'volume', -1, 0)
            
            # Use adjusted period if not enough data
            adjusted_period = min(period, len(df) - 1)
            
            # Calculate average volume
            avg_volume = df['volume'].iloc[-adjusted_period:].mean()
            
            if current_close is None or prev_close is None or avg_volume == 0:
                logger.warning("Missing values for volume signal check")
                return False, False, 0.0
                
            # Calculate price change
            price_change = (current_close - prev_close) / prev_close
            
            # Check for signals
            volume_ratio = current_volume / avg_volume
            
            buy_signal = price_change > 0 and volume_ratio > 1.5
            sell_signal = price_change < 0 and volume_ratio > 1.5
            
            # Calculate strength
            strength = min(volume_ratio / 3.0, 1.0)
                
            return buy_signal, sell_signal, strength
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error("check_volume_signal", str(e))
            else:
                logger.error(f"Error checking volume signal: {str(e)}")
            return False, False, 0.0
            
    def check_support_resistance(self, df: pd.DataFrame, lookback: int = 50, threshold: float = 0.02) -> Tuple[bool, bool, float]:
        """
        Check for support and resistance levels with robust handling of limited data.
        
        Args:
            df: DataFrame with price data
            lookback: Lookback period for support/resistance levels
            threshold: Price threshold for support/resistance (percentage)
            
        Returns:
            Tuple of (buy_signal, sell_signal, strength)
        """
        try:
            if df is None or df.empty:
                logger.warning("No data for support/resistance check")
                return False, False, 0.0
                
            # Not enough data for reliable support/resistance
            if len(df) < 10:
                logger.warning("Not enough data for support/resistance check")
                return False, False, 0.0
                
            # Use adjusted lookback if not enough data
            adjusted_lookback = min(lookback, len(df) - 1)
            
            # Get current price
            current_price = RobustSignalGenerator.safe_get(df, 'close', -1, None)
            
            if current_price is None:
                logger.warning("Missing current price for support/resistance check")
                return False, False, 0.0
                
            # Get high and low prices
            highs = df['high'].iloc[-adjusted_lookback:] if 'high' in df.columns else df['close'].iloc[-adjusted_lookback:]
            lows = df['low'].iloc[-adjusted_lookback:] if 'low' in df.columns else df['close'].iloc[-adjusted_lookback:]
            
            # Find potential support and resistance levels
            support_levels = []
            resistance_levels = []
            
            # Simple method: find local minima and maxima
            for i in range(1, len(highs) - 1):
                # Check for local maximum (resistance)
                if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                    resistance_levels.append(highs.iloc[i])
                    
                # Check for local minimum (support)
                if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                    support_levels.append(lows.iloc[i])
                    
            # Check if current price is near support or resistance
            buy_signal = False
            sell_signal = False
            strength = 0.0
            
            # Check support levels
            for level in support_levels:
                distance = abs(current_price - level) / current_price
                if distance < threshold and current_price >= level:
                    buy_signal = True
                    strength = max(strength, 1.0 - distance / threshold)
                    
            # Check resistance levels
            for level in resistance_levels:
                distance = abs(current_price - level) / current_price
                if distance < threshold and current_price <= level:
                    sell_signal = True
                    strength = max(strength, 1.0 - distance / threshold)
                    
            return buy_signal, sell_signal, strength
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error("check_support_resistance", str(e))
            else:
                logger.error(f"Error checking support/resistance: {str(e)}")
            return False, False, 0.0
            
    def generate_master_signal(self, df: pd.DataFrame, additional_data: Dict = None) -> int:
        """
        Generate master trading signal by combining multiple indicators.
        
        Args:
            df: DataFrame with price and indicator data
            additional_data: Additional data dictionary
            
        Returns:
            Signal: 1 for buy, -1 for sell, 0 for neutral
        """
        try:
            if df is None or df.empty:
                logger.warning("No data for signal generation")
                return 0
                
            # Initialize signal components
            signal_components = {}
            
            # Check RSI signal
            rsi_buy, rsi_sell, rsi_strength = self.check_rsi_signal(df)
            signal_components["rsi"] = {
                "buy": rsi_buy,
                "sell": rsi_sell,
                "strength": rsi_strength
            }
            
            # Check Bollinger Band signal
            bb_buy, bb_sell, bb_strength = self.check_bollinger_signal(df)
            signal_components["bollinger"] = {
                "buy": bb_buy,
                "sell": bb_sell,
                "strength": bb_strength
            }
            
            # Check MACD signal
            macd_buy, macd_sell, macd_strength = self.check_macd_signal(df)
            signal_components["macd"] = {
                "buy": macd_buy,
                "sell": macd_sell,
                "strength": macd_strength
            }
            
            # Check volume signal
            vol_buy, vol_sell, vol_strength = self.check_volume_signal(df)
            signal_components["volume"] = {
                "buy": vol_buy,
                "sell": vol_sell,
                "strength": vol_strength
            }
            
            # Check support/resistance signal
            sr_buy, sr_sell, sr_strength = self.check_support_resistance(df)
            signal_components["support_resistance"] = {
                "buy": sr_buy,
                "sell": sr_sell,
                "strength": sr_strength
            }
            
            # Calculate buy and sell scores
            buy_score = 0.0
            sell_score = 0.0
            
            # RSI (weight: 0.2)
            if rsi_buy:
                buy_score += 0.2 * rsi_strength
            if rsi_sell:
                sell_score += 0.2 * rsi_strength
                
            # Bollinger Bands (weight: 0.2)
            if bb_buy:
                buy_score += 0.2 * bb_strength
            if bb_sell:
                sell_score += 0.2 * bb_strength
                
            # MACD (weight: 0.3)
            if macd_buy:
                buy_score += 0.3 * macd_strength
            if macd_sell:
                sell_score += 0.3 * macd_strength
                
            # Volume (weight: 0.15)
            if vol_buy:
                buy_score += 0.15 * vol_strength
            if vol_sell:
                sell_score += 0.15 * vol_strength
                
            # Support/Resistance (weight: 0.15)
            if sr_buy:
                buy_score += 0.15 * sr_strength
            if sr_sell:
                sell_score += 0.15 * sr_strength
                
            # Determine final signal
            if buy_score > 0.4 and buy_score > sell_score:
                signal = 1  # Buy
            elif sell_score > 0.4 and sell_score > buy_score:
                signal = -1  # Sell
            else:
                signal = 0  # Neutral
                
            logger.info(f"Generated master signal: {signal} (buy_score: {buy_score:.2f}, sell_score: {sell_score:.2f})")
            return signal
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error("generate_master_signal", str(e))
            else:
                logger.error(f"Error generating master signal: {str(e)}")
            return 0
