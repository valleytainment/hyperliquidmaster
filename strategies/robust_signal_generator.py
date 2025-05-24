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
    
    @staticmethod
    def detect_market_regime(df: pd.DataFrame) -> str:
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
            logger.error(f"Error detecting market regime: {str(e)}")
            return "unknown"
            
    @staticmethod
    def check_vwma_crossover(df: pd.DataFrame, fast_period: int = 20, slow_period: int = 50) -> Tuple[bool, bool, float]:
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
            logger.error(f"Error checking VWMA crossover: {str(e)}")
            return False, False, 0.0
            
    @staticmethod
    def check_rsi_signal(df: pd.DataFrame, period: int = 14, overbought: float = 70, oversold: float = 30) -> Tuple[bool, bool, float]:
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
            logger.error(f"Error checking RSI signal: {str(e)}")
            return False, False, 0.0
            
    @staticmethod
    def check_bollinger_signal(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Tuple[bool, bool, float]:
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
            logger.error(f"Error checking Bollinger Band signal: {str(e)}")
            return False, False, 0.0
            
    @staticmethod
    def check_macd_signal(df: pd.DataFrame) -> Tuple[bool, bool, float]:
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
                fast_period = min(12, len(df) - 1)
                slow_period = min(26, len(df))
                signal_period = min(9, len(df) - slow_period)
                
                if fast_period < 2 or slow_period < 3 or signal_period < 2:
                    logger.warning("Not enough data for MACD calculation")
                    return False, False, 0.0
                    
                # Use simple moving averages for limited data
                fast_ma = df['close'].rolling(window=fast_period, min_periods=1).mean()
                slow_ma = df['close'].rolling(window=slow_period, min_periods=1).mean()
                
                macd_line = fast_ma - slow_ma
                signal_line = macd_line.rolling(window=signal_period, min_periods=1).mean()
                histogram = macd_line - signal_line
                
            # Get current and previous values
            current_macd = RobustSignalGenerator.safe_get(macd_line.to_frame('value'), 'value', -1, 0)
            current_signal = RobustSignalGenerator.safe_get(signal_line.to_frame('value'), 'value', -1, 0)
            current_hist = RobustSignalGenerator.safe_get(histogram.to_frame('value'), 'value', -1, 0)
            prev_hist = RobustSignalGenerator.safe_get(histogram.to_frame('value'), 'value', -2, 0)
            
            # Check for signals
            buy_signal = prev_hist < 0 and current_hist > 0
            sell_signal = prev_hist > 0 and current_hist < 0
            
            # Calculate strength
            if current_signal != 0:
                strength = abs(current_macd - current_signal) / abs(current_signal)
            else:
                strength = 0.0
                
            return buy_signal, sell_signal, strength
        except Exception as e:
            logger.error(f"Error checking MACD signal: {str(e)}")
            return False, False, 0.0
            
    @staticmethod
    def _generate_technical_signal(market_data: Dict) -> Dict:
        """
        Generate signal based on technical indicators.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Signal dictionary
        """
        try:
            # Get historical data
            df = market_data.get("historical_data")
            
            if df is None or len(df) < 2:
                logger.warning("Insufficient historical data for technical signal")
                return {"action": "none", "signal": 0.0, "strength": 0.0}
                
            # Use enhanced VWMA indicator for robust signal generation
            vwma_signal = EnhancedVWMAIndicator.generate_vwma_signal(df)
            
            # Get RSI if available
            rsi_signal = 0.0
            if 'rsi' in df.columns:
                last_rsi = df['rsi'].iloc[-1]
                if not pd.isna(last_rsi):
                    if last_rsi < 30:
                        rsi_signal = 0.5  # Bullish
                    elif last_rsi > 70:
                        rsi_signal = -0.5  # Bearish
            
            # Combine signals
            action = vwma_signal["action"]
            signal = vwma_signal["signal"] + rsi_signal
            
            # Adjust action based on combined signal
            if signal > 0.2:
                action = "long"
            elif signal < -0.2:
                action = "short"
            else:
                action = "none"
                
            return {
                "action": action,
                "signal": signal,
                "strength": abs(signal),
                "components": {
                    "vwma": vwma_signal,
                    "rsi": rsi_signal
                }
            }
        except Exception as e:
            logger.error(f"Error generating technical signal: {str(e)}")
            return {"action": "none", "signal": 0.0, "strength": 0.0}
            
    @staticmethod
    def _generate_order_book_signal(market_data: Dict) -> Dict:
        """
        Generate signal based on order book imbalance.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Signal dictionary
        """
        try:
            # Get order book metrics
            order_book_metrics = market_data.get("order_book_metrics", {})
            
            if not order_book_metrics:
                logger.warning("No order book metrics available")
                return {"action": "none", "signal": 0.0, "strength": 0.0}
                
            # Get bid-ask imbalance
            bid_volume = order_book_metrics.get("bid_volume", 0)
            ask_volume = order_book_metrics.get("ask_volume", 0)
            
            if bid_volume == 0 or ask_volume == 0:
                logger.warning("Invalid order book volumes")
                return {"action": "none", "signal": 0.0, "strength": 0.0}
                
            # Calculate imbalance ratio
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            
            # Generate signal based on imbalance
            action = "none"
            if imbalance > 0.2:
                action = "long"
            elif imbalance < -0.2:
                action = "short"
                
            return {
                "action": action,
                "signal": imbalance,
                "strength": abs(imbalance)
            }
        except Exception as e:
            logger.error(f"Error generating order book signal: {str(e)}")
            return {"action": "none", "signal": 0.0, "strength": 0.0}
            
    @staticmethod
    def _generate_funding_rate_signal(market_data: Dict) -> Dict:
        """
        Generate signal based on funding rate.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Signal dictionary
        """
        try:
            # Get funding rate
            funding_rate = market_data.get("funding_rate", 0)
            
            if funding_rate == 0:
                logger.warning("No funding rate available")
                return {"action": "none", "signal": 0.0, "strength": 0.0}
                
            # Generate signal based on funding rate
            action = "none"
            signal = 0.0
            
            # Negative funding rate means longs pay shorts, indicating bearish sentiment
            # Positive funding rate means shorts pay longs, indicating bullish sentiment
            if funding_rate < -0.01:
                action = "long"  # Contrarian approach
                signal = -funding_rate * 10  # Scale for signal strength
            elif funding_rate > 0.01:
                action = "short"  # Contrarian approach
                signal = -funding_rate * 10  # Scale for signal strength
                
            return {
                "action": action,
                "signal": signal,
                "strength": abs(signal)
            }
        except Exception as e:
            logger.error(f"Error generating funding rate signal: {str(e)}")
            return {"action": "none", "signal": 0.0, "strength": 0.0}
            
    @staticmethod
    def generate_combined_signal(symbol: str, market_data: Dict, order_book: Dict = None) -> Dict:
        """
        Generate combined signal from multiple sources.
        
        Args:
            symbol: Trading symbol
            market_data: Market data dictionary
            order_book: Order book dictionary
            
        Returns:
            Combined signal dictionary
        """
        try:
            # Generate signals from different sources
            technical_signal = RobustSignalGenerator._generate_technical_signal(market_data)
            order_book_signal = RobustSignalGenerator._generate_order_book_signal(market_data)
            funding_rate_signal = RobustSignalGenerator._generate_funding_rate_signal(market_data)
            
            # Get historical data for market regime detection
            df = market_data.get("historical_data")
            market_regime = "unknown"
            
            if df is not None and len(df) > 10:
                market_regime = RobustSignalGenerator.detect_market_regime(df)
                
            # Adjust weights based on market regime
            technical_weight = 0.6
            order_book_weight = 0.3
            funding_rate_weight = 0.1
            
            if market_regime == "trending_up" or market_regime == "trending_down":
                technical_weight = 0.7
                order_book_weight = 0.2
                funding_rate_weight = 0.1
            elif market_regime == "ranging":
                technical_weight = 0.5
                order_book_weight = 0.3
                funding_rate_weight = 0.2
            elif market_regime == "volatile":
                technical_weight = 0.4
                order_book_weight = 0.4
                funding_rate_weight = 0.2
                
            # Calculate combined signal
            combined_signal = (
                technical_signal["signal"] * technical_weight +
                order_book_signal["signal"] * order_book_weight +
                funding_rate_signal["signal"] * funding_rate_weight
            )
            
            # Determine action based on combined signal
            action = "none"
            if combined_signal > 0.3:
                action = "long"
            elif combined_signal < -0.3:
                action = "short"
                
            # Calculate position size based on signal strength
            position_size = 0.0
            if action != "none":
                position_size = min(1.0, abs(combined_signal))
                
            # Get current price
            current_price = market_data.get("last_price", 0)
            
            # Calculate stop loss and take profit levels
            stop_loss = 0.0
            take_profit = 0.0
            
            if action == "long" and current_price > 0:
                stop_loss = current_price * 0.95  # 5% stop loss
                take_profit = current_price * 1.15  # 15% take profit
            elif action == "short" and current_price > 0:
                stop_loss = current_price * 1.05  # 5% stop loss
                take_profit = current_price * 0.85  # 15% take profit
                
            # Create combined signal dictionary
            return {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "action": action,
                "quantity": position_size,
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "signal_strength": abs(combined_signal),
                "market_regime": market_regime,
                "components": {
                    "technical": technical_signal,
                    "order_book": order_book_signal,
                    "funding_rate": funding_rate_signal
                }
            }
        except Exception as e:
            logger.error(f"Error generating combined signal: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "action": "none",
                "quantity": 0.0,
                "entry_price": market_data.get("last_price", 0),
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "signal_strength": 0.0,
                "market_regime": "unknown",
                "components": {
                    "technical": {"action": "none", "signal": 0.0, "strength": 0.0},
                    "order_book": {"action": "none", "signal": 0.0, "strength": 0.0},
                    "funding_rate": {"action": "none", "signal": 0.0, "strength": 0.0}
                }
            }
