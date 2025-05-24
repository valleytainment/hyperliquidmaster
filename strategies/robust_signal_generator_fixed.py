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
    def calculate_rsi(prices, period=14):
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: List or array of prices
            period: RSI period
            
        Returns:
            List of RSI values
        """
        try:
            if prices is None or len(prices) < period + 1:
                return []
                
            # Convert to numpy array if needed
            if not isinstance(prices, np.ndarray):
                prices = np.array(prices)
                
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
            if df is None or df.empty or len(df) < slow_period:
                logger.warning("Not enough data for VWMA crossover check")
                return False, False, 0.0
                
            # Check if required columns exist
            if 'close' not in df.columns or 'volume' not in df.columns:
                logger.warning("Missing required columns for VWMA calculation")
                return False, False, 0.0
                
            # Calculate VWMA
            close = df['close']
            volume = df['volume']
            
            # Adjust periods if not enough data
            adjusted_fast_period = min(fast_period, len(df) - 1)
            adjusted_slow_period = min(slow_period, len(df) - 1)
            
            if adjusted_fast_period < 2 or adjusted_slow_period < 2:
                logger.warning("Not enough data for VWMA calculation")
                return False, False, 0.0
                
            # Calculate VWMA
            close_volume = close * volume
            
            fast_vwma = close_volume.rolling(window=adjusted_fast_period).sum() / volume.rolling(window=adjusted_fast_period).sum()
            slow_vwma = close_volume.rolling(window=adjusted_slow_period).sum() / volume.rolling(window=adjusted_slow_period).sum()
            
            # Check for crossover
            current_fast = fast_vwma.iloc[-1]
            current_slow = slow_vwma.iloc[-1]
            
            prev_fast = fast_vwma.iloc[-2] if len(fast_vwma) > 1 else None
            prev_slow = slow_vwma.iloc[-2] if len(slow_vwma) > 1 else None
            
            if prev_fast is None or prev_slow is None:
                return False, False, 0.0
                
            bullish_crossover = prev_fast <= prev_slow and current_fast > current_slow
            bearish_crossover = prev_fast >= prev_slow and current_fast < current_slow
            
            # Calculate strength
            if bullish_crossover or bearish_crossover:
                strength = abs(current_fast - current_slow) / current_slow
                strength = min(strength * 10, 1.0)  # Scale and cap strength
            else:
                strength = 0.0
                
            return bullish_crossover, bearish_crossover, strength
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
                strength = min(strength * 5, 1.0)  # Scale and cap strength
            else:
                strength = 0.0
                
            return buy_signal, sell_signal, strength
        except Exception as e:
            logger.error(f"Error checking MACD signal: {str(e)}")
            return False, False, 0.0
    
    @staticmethod
    def generate_master_signal(market_data: Dict[str, Any], order_book: Dict[str, Any] = None, 
                              strategy_weights: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Generate master trading signal by combining multiple indicators and data sources.
        
        Args:
            market_data: Market data dictionary
            order_book: Order book data dictionary
            strategy_weights: Strategy weights dictionary
            
        Returns:
            Signal dictionary with action, strength, and components
        """
        try:
            # Initialize signal components
            technical_signal = {"action": "none", "signal": 0.0, "strength": 0.0, "components": {}}
            order_book_signal = {"action": "none", "signal": 0.0, "strength": 0.0}
            funding_rate_signal = {"action": "none", "signal": 0.0, "strength": 0.0}
            
            # Default strategy weights if not provided
            if strategy_weights is None:
                strategy_weights = {
                    "triple_confluence": 0.6,
                    "oracle_update": 0.4
                }
            
            # Get current price
            current_price = market_data.get("price", market_data.get("last_price", 0.0))
            
            # Check if we have historical data
            if "historical_data" in market_data and market_data["historical_data"] is not None:
                df = market_data["historical_data"]
                
                # Generate technical signals
                
                # VWMA Crossover
                bullish_crossover, bearish_crossover, vwma_strength = RobustSignalGenerator.check_vwma_crossover(df)
                
                if bullish_crossover:
                    vwma_signal = {"action": "buy", "signal": vwma_strength, "strength": vwma_strength, "indicator": "vwma", "bullish_crossover": bullish_crossover, "bearish_crossover": bearish_crossover}
                elif bearish_crossover:
                    vwma_signal = {"action": "sell", "signal": -vwma_strength, "strength": vwma_strength, "indicator": "vwma", "bullish_crossover": bullish_crossover, "bearish_crossover": bearish_crossover}
                else:
                    vwma_signal = {"action": "none", "signal": 0.0, "strength": 0.0, "indicator": "vwma", "bullish_crossover": bullish_crossover, "bearish_crossover": bearish_crossover}
                
                # RSI
                rsi_buy, rsi_sell, rsi_strength = RobustSignalGenerator.check_rsi_signal(df)
                rsi_signal = 0.0
                
                if rsi_buy:
                    rsi_signal = rsi_strength
                elif rsi_sell:
                    rsi_signal = -rsi_strength
                
                # Combine technical signals
                technical_components = {
                    "vwma": vwma_signal,
                    "rsi": rsi_signal
                }
                
                # Triple Confluence strategy weight
                triple_confluence_weight = strategy_weights.get("triple_confluence", 0.6)
                
                # Determine technical signal
                if vwma_signal["action"] == "buy" and rsi_buy:
                    # Strong buy signal (confluence)
                    technical_signal = {
                        "action": "buy",
                        "signal": (vwma_strength + rsi_strength) / 2 * triple_confluence_weight,
                        "strength": (vwma_strength + rsi_strength) / 2,
                        "components": technical_components
                    }
                elif vwma_signal["action"] == "sell" and rsi_sell:
                    # Strong sell signal (confluence)
                    technical_signal = {
                        "action": "sell",
                        "signal": -(vwma_strength + rsi_strength) / 2 * triple_confluence_weight,
                        "strength": (vwma_strength + rsi_strength) / 2,
                        "components": technical_components
                    }
                elif vwma_signal["action"] == "buy":
                    # Moderate buy signal
                    technical_signal = {
                        "action": "buy",
                        "signal": vwma_strength * 0.7 * triple_confluence_weight,
                        "strength": vwma_strength * 0.7,
                        "components": technical_components
                    }
                elif vwma_signal["action"] == "sell":
                    # Moderate sell signal
                    technical_signal = {
                        "action": "sell",
                        "signal": -vwma_strength * 0.7 * triple_confluence_weight,
                        "strength": vwma_strength * 0.7,
                        "components": technical_components
                    }
                elif rsi_buy:
                    # Weak buy signal
                    technical_signal = {
                        "action": "buy",
                        "signal": rsi_strength * 0.5 * triple_confluence_weight,
                        "strength": rsi_strength * 0.5,
                        "components": technical_components
                    }
                elif rsi_sell:
                    # Weak sell signal
                    technical_signal = {
                        "action": "sell",
                        "signal": -rsi_strength * 0.5 * triple_confluence_weight,
                        "strength": rsi_strength * 0.5,
                        "components": technical_components
                    }
            else:
                logger.warning("No historical data available for technical signal generation")
            
            # Generate order book signal
            if order_book and "metrics" in order_book:
                metrics = order_book["metrics"]
                
                # Extract order book metrics
                bid_ask_imbalance = metrics.get("bid_ask_imbalance", 0.0)
                depth_imbalance = metrics.get("depth_imbalance", 0.0)
                
                # Oracle Update strategy weight
                oracle_update_weight = strategy_weights.get("oracle_update", 0.4)
                
                # Calculate order book signal
                if bid_ask_imbalance > 0.2 and depth_imbalance > 0.2:
                    # Strong buy signal
                    order_book_signal = {
                        "action": "buy",
                        "signal": (bid_ask_imbalance + depth_imbalance) / 2 * oracle_update_weight,
                        "strength": (bid_ask_imbalance + depth_imbalance) / 2
                    }
                elif bid_ask_imbalance < -0.2 and depth_imbalance < -0.2:
                    # Strong sell signal
                    order_book_signal = {
                        "action": "sell",
                        "signal": (bid_ask_imbalance + depth_imbalance) / 2 * oracle_update_weight,
                        "strength": abs((bid_ask_imbalance + depth_imbalance) / 2)
                    }
                elif bid_ask_imbalance > 0.1 or depth_imbalance > 0.1:
                    # Weak buy signal
                    order_book_signal = {
                        "action": "buy",
                        "signal": max(bid_ask_imbalance, depth_imbalance) * 0.5 * oracle_update_weight,
                        "strength": max(bid_ask_imbalance, depth_imbalance) * 0.5
                    }
                elif bid_ask_imbalance < -0.1 or depth_imbalance < -0.1:
                    # Weak sell signal
                    order_book_signal = {
                        "action": "sell",
                        "signal": min(bid_ask_imbalance, depth_imbalance) * 0.5 * oracle_update_weight,
                        "strength": abs(min(bid_ask_imbalance, depth_imbalance)) * 0.5
                    }
            else:
                logger.warning("No order book metrics available")
            
            # Generate funding rate signal
            funding_rate = market_data.get("funding_rate", 0.0)
            
            if abs(funding_rate) > 0.001:  # 0.1% threshold
                # Oracle Update strategy weight
                oracle_update_weight = strategy_weights.get("oracle_update", 0.4)
                
                if funding_rate > 0:
                    # Positive funding rate - short signal
                    funding_rate_signal = {
                        "action": "sell",
                        "signal": -min(funding_rate * 100, 1.0) * oracle_update_weight,
                        "strength": min(funding_rate * 100, 1.0)
                    }
                else:
                    # Negative funding rate - long signal
                    funding_rate_signal = {
                        "action": "buy",
                        "signal": min(abs(funding_rate) * 100, 1.0) * oracle_update_weight,
                        "strength": min(abs(funding_rate) * 100, 1.0)
                    }
            else:
                logger.warning("No funding rate available or funding rate too small")
            
            # Combine all signals
            combined_signal = technical_signal["signal"] + order_book_signal["signal"] + funding_rate_signal["signal"]
            
            # Determine final action
            if combined_signal > 0.1:
                action = "buy"
            elif combined_signal < -0.1:
                action = "sell"
            else:
                action = "none"
            
            # Calculate stop loss and take profit
            stop_loss = 0.0
            take_profit = 0.0
            
            if action == "buy":
                # Stop loss 2% below entry
                stop_loss = current_price * 0.98
                # Take profit 3% above entry
                take_profit = current_price * 1.03
            elif action == "sell":
                # Stop loss 2% above entry
                stop_loss = current_price * 1.02
                # Take profit 3% below entry
                take_profit = current_price * 0.97
            
            # Prepare final signal
            signal = {
                "action": action,
                "signal": combined_signal,
                "strength": abs(combined_signal),
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "components": {
                    "technical": technical_signal,
                    "order_book": order_book_signal,
                    "funding_rate": funding_rate_signal
                }
            }
            
            return signal
        except Exception as e:
            logger.error(f"Error generating master signal: {str(e)}")
            return {
                "action": "none",
                "signal": 0.0,
                "strength": 0.0,
                "entry_price": 0.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "components": {},
                "error": str(e)
            }
