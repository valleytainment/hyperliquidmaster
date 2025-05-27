"""
Enhanced Robust Signal Generator for the Hyperliquid trading bot.
Implements advanced technical analysis and market regime detection for optimal trading signals.
"""

import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

class RobustSignalGenerator:
    """
    Advanced signal generator with optimized detection for both long and short opportunities.
    Implements multiple technical indicators, pattern recognition, and market regime adaptation.
    """
    
    def __init__(self):
        """Initialize the signal generator with configuration parameters"""
        self.logger = logging.getLogger("RobustSignalGenerator")
        
        # Configuration parameters
        self.config = {
            # Technical indicators
            "ema_short": 8,
            "ema_medium": 21,
            "ema_long": 55,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "stoch_k_period": 14,
            "stoch_d_period": 3,
            "stoch_overbought": 80,
            "stoch_oversold": 20,
            "bollinger_period": 20,
            "bollinger_std": 2.0,
            "atr_period": 14,
            
            # Pattern recognition
            "pattern_lookback": 5,
            "pattern_threshold": 0.8,
            
            # Market regime detection
            "regime_volatility_period": 20,
            "regime_trend_period": 50,
            
            # Signal thresholds
            "min_confidence": 0.65,
            "strong_confidence": 0.8,
            
            # Adaptive parameters
            "adapt_to_volatility": True,
            "adapt_to_volume": True,
            "adapt_to_trend": True
        }
        
        # State tracking
        self.state = {
            "last_signal": {},
            "market_regime": "neutral",
            "volatility_level": "medium",
            "trend_strength": "neutral",
            "historical_signals": []
        }
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for signal generation.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        try:
            # Moving averages
            df["ema_short"] = df["close"].ewm(span=self.config["ema_short"], adjust=False).mean()
            df["ema_medium"] = df["close"].ewm(span=self.config["ema_medium"], adjust=False).mean()
            df["ema_long"] = df["close"].ewm(span=self.config["ema_long"], adjust=False).mean()
            
            # RSI
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=self.config["rsi_period"]).mean()
            avg_loss = loss.rolling(window=self.config["rsi_period"]).mean()
            rs = avg_gain / avg_loss
            df["rsi"] = 100 - (100 / (1 + rs))
            
            # Stochastic oscillator
            df["stoch_k"] = ((df["close"] - df["low"].rolling(window=self.config["stoch_k_period"]).min()) / 
                            (df["high"].rolling(window=self.config["stoch_k_period"]).max() - 
                             df["low"].rolling(window=self.config["stoch_k_period"]).min())) * 100
            df["stoch_d"] = df["stoch_k"].rolling(window=self.config["stoch_d_period"]).mean()
            
            # Bollinger Bands
            df["bb_middle"] = df["close"].rolling(window=self.config["bollinger_period"]).mean()
            df["bb_std"] = df["close"].rolling(window=self.config["bollinger_period"]).std()
            df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * self.config["bollinger_std"])
            df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * self.config["bollinger_std"])
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
            
            # ATR (Average True Range)
            tr1 = df["high"] - df["low"]
            tr2 = abs(df["high"] - df["close"].shift())
            tr3 = abs(df["low"] - df["close"].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df["atr"] = tr.rolling(window=self.config["atr_period"]).mean()
            
            # Volatility
            df["volatility"] = df["close"].pct_change().rolling(window=self.config["regime_volatility_period"]).std() * 100
            
            # Volume indicators
            df["volume_sma"] = df["volume"].rolling(window=10).mean()
            df["volume_ratio"] = df["volume"] / df["volume_sma"]
            
            # Trend indicators
            df["trend_direction"] = np.where(df["ema_medium"] > df["ema_long"], 1, -1)
            df["price_distance"] = (df["close"] - df["ema_medium"]) / df["ema_medium"] * 100
            
            # Momentum
            df["momentum"] = df["close"] / df["close"].shift(10) - 1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return df
    
    def _detect_market_regime(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect current market regime characteristics.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Dictionary with market regime characteristics
        """
        try:
            if df.empty or len(df) < self.config["regime_trend_period"]:
                return {
                    "regime": "neutral",
                    "volatility": "medium",
                    "trend": "neutral"
                }
            
            # Get recent data
            recent = df.iloc[-30:]
            
            # Detect volatility level
            volatility = recent["volatility"].mean()
            volatility_percentile = np.percentile(df["volatility"].dropna(), [25, 50, 75])
            
            if volatility < volatility_percentile[0]:
                volatility_level = "low"
            elif volatility > volatility_percentile[2]:
                volatility_level = "high"
            else:
                volatility_level = "medium"
            
            # Detect trend strength
            trend_consistency = abs(recent["trend_direction"].sum()) / len(recent)
            
            if trend_consistency > 0.8:
                trend_strength = "strong"
            elif trend_consistency > 0.5:
                trend_strength = "moderate"
            else:
                trend_strength = "weak"
            
            # Determine overall regime
            if trend_strength == "strong" and volatility_level == "low":
                regime = "trending"
            elif trend_strength == "strong" and volatility_level == "high":
                regime = "volatile_trending"
            elif trend_strength == "weak" and volatility_level == "low":
                regime = "ranging"
            elif trend_strength == "weak" and volatility_level == "high":
                regime = "choppy"
            else:
                regime = "mixed"
            
            return {
                "regime": regime,
                "volatility": volatility_level,
                "trend": trend_strength
            }
                
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            return {
                "regime": "neutral",
                "volatility": "medium",
                "trend": "neutral"
            }
    
    def _detect_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Detect chart patterns in price data.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Dictionary with pattern confidence scores
        """
        try:
            if df.empty or len(df) < 20:
                return {"patterns": {}}
            
            patterns = {}
            recent = df.iloc[-self.config["pattern_lookback"]:]
            
            # Double bottom pattern
            if (recent["low"].iloc[0] < recent["low"].iloc[1:-1].min() and 
                recent["low"].iloc[-1] < recent["low"].iloc[1:-1].min() and
                abs(recent["low"].iloc[0] - recent["low"].iloc[-1]) / recent["low"].iloc[0] < 0.03):
                patterns["double_bottom"] = 0.7
            
            # Double top pattern
            if (recent["high"].iloc[0] > recent["high"].iloc[1:-1].max() and 
                recent["high"].iloc[-1] > recent["high"].iloc[1:-1].max() and
                abs(recent["high"].iloc[0] - recent["high"].iloc[-1]) / recent["high"].iloc[0] < 0.03):
                patterns["double_top"] = 0.7
            
            # Bullish engulfing
            if (recent["open"].iloc[-2] > recent["close"].iloc[-2] and  # Previous candle is bearish
                recent["open"].iloc[-1] < recent["close"].iloc[-1] and  # Current candle is bullish
                recent["open"].iloc[-1] <= recent["close"].iloc[-2] and  # Current open below previous close
                recent["close"].iloc[-1] >= recent["open"].iloc[-2]):   # Current close above previous open
                patterns["bullish_engulfing"] = 0.75
            
            # Bearish engulfing
            if (recent["open"].iloc[-2] < recent["close"].iloc[-2] and  # Previous candle is bullish
                recent["open"].iloc[-1] > recent["close"].iloc[-1] and  # Current candle is bearish
                recent["open"].iloc[-1] >= recent["close"].iloc[-2] and  # Current open above previous close
                recent["close"].iloc[-1] <= recent["open"].iloc[-2]):   # Current close below previous open
                patterns["bearish_engulfing"] = 0.75
            
            # Hammer (potential bullish reversal)
            if (recent["close"].iloc[-1] > recent["open"].iloc[-1] and  # Bullish candle
                (recent["high"].iloc[-1] - recent["close"].iloc[-1]) < 0.3 * (recent["close"].iloc[-1] - recent["low"].iloc[-1]) and  # Small upper shadow
                (recent["open"].iloc[-1] - recent["low"].iloc[-1]) > 2 * (recent["high"].iloc[-1] - recent["close"].iloc[-1])):  # Long lower shadow
                patterns["hammer"] = 0.65
            
            # Shooting star (potential bearish reversal)
            if (recent["close"].iloc[-1] < recent["open"].iloc[-1] and  # Bearish candle
                (recent["high"].iloc[-1] - recent["open"].iloc[-1]) > 2 * (recent["open"].iloc[-1] - recent["close"].iloc[-1]) and  # Long upper shadow
                (recent["close"].iloc[-1] - recent["low"].iloc[-1]) < 0.3 * (recent["high"].iloc[-1] - recent["open"].iloc[-1])):  # Small lower shadow
                patterns["shooting_star"] = 0.65
            
            return {"patterns": patterns}
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {e}")
            return {"patterns": {}}
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Calculate support and resistance levels.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with support and resistance levels
        """
        try:
            if df.empty or len(df) < 30:
                return {"support": [], "resistance": []}
            
            # Use recent data for calculation
            recent = df.iloc[-100:].copy() if len(df) >= 100 else df.copy()
            
            # Find local minima and maxima
            recent["min"] = recent["low"].rolling(window=5, center=True).min()
            recent["max"] = recent["high"].rolling(window=5, center=True).max()
            
            # Identify potential support levels (local minima)
            support_points = recent[(recent["low"] == recent["min"]) & 
                                   (recent["min"].shift(1) > recent["min"]) & 
                                   (recent["min"].shift(-1) > recent["min"])]["low"].tolist()
            
            # Identify potential resistance levels (local maxima)
            resistance_points = recent[(recent["high"] == recent["max"]) & 
                                      (recent["max"].shift(1) < recent["max"]) & 
                                      (recent["max"].shift(-1) < recent["max"])]["high"].tolist()
            
            # Cluster nearby levels
            support_levels = self._cluster_price_levels(support_points, df["close"].iloc[-1] * 0.005)
            resistance_levels = self._cluster_price_levels(resistance_points, df["close"].iloc[-1] * 0.005)
            
            # Sort levels
            support_levels.sort()
            resistance_levels.sort()
            
            # Filter levels that are too close to current price
            current_price = df["close"].iloc[-1]
            support_levels = [level for level in support_levels if level < current_price * 0.98]
            resistance_levels = [level for level in resistance_levels if level > current_price * 1.02]
            
            return {
                "support": support_levels[-3:] if support_levels else [],  # Return 3 closest support levels
                "resistance": resistance_levels[:3] if resistance_levels else []  # Return 3 closest resistance levels
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance: {e}")
            return {"support": [], "resistance": []}
    
    def _cluster_price_levels(self, levels: List[float], threshold: float) -> List[float]:
        """
        Cluster nearby price levels.
        
        Args:
            levels: List of price levels
            threshold: Threshold for clustering
            
        Returns:
            List of clustered price levels
        """
        if not levels:
            return []
        
        # Sort levels
        sorted_levels = sorted(levels)
        
        # Cluster nearby levels
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        for i in range(1, len(sorted_levels)):
            if sorted_levels[i] - sorted_levels[i-1] <= threshold:
                current_cluster.append(sorted_levels[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sorted_levels[i]]
        
        clusters.append(current_cluster)
        
        # Calculate average of each cluster
        return [sum(cluster) / len(cluster) for cluster in clusters]
    
    def _generate_signal_score(self, df: pd.DataFrame, regime: Dict[str, str], 
                              patterns: Dict[str, Dict[str, float]], 
                              levels: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Generate signal score based on technical indicators and market regime.
        
        Args:
            df: DataFrame with technical indicators
            regime: Market regime information
            patterns: Detected patterns
            levels: Support and resistance levels
            
        Returns:
            Dictionary with signal information
        """
        try:
            if df.empty or len(df) < 20:
                return {
                    "signal": "NEUTRAL",
                    "confidence": 0.5,
                    "reason": "Insufficient data"
                }
            
            # Get latest data
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Initialize scores
            buy_score = 0.0
            sell_score = 0.0
            reasons = []
            
            # Trend analysis (30% weight)
            if latest["ema_short"] > latest["ema_medium"] > latest["ema_long"]:
                buy_score += 0.3
                reasons.append("Strong uptrend")
            elif latest["ema_short"] < latest["ema_medium"] < latest["ema_long"]:
                sell_score += 0.3
                reasons.append("Strong downtrend")
            elif latest["ema_short"] > latest["ema_medium"]:
                buy_score += 0.15
                reasons.append("Short-term uptrend")
            elif latest["ema_short"] < latest["ema_medium"]:
                sell_score += 0.15
                reasons.append("Short-term downtrend")
            
            # Momentum indicators (25% weight)
            # RSI
            if latest["rsi"] < self.config["rsi_oversold"]:
                buy_score += 0.125
                reasons.append("Oversold RSI")
            elif latest["rsi"] > self.config["rsi_overbought"]:
                sell_score += 0.125
                reasons.append("Overbought RSI")
            
            # Stochastic
            if latest["stoch_k"] < self.config["stoch_oversold"] and latest["stoch_k"] > latest["stoch_d"]:
                buy_score += 0.125
                reasons.append("Bullish stochastic crossover in oversold region")
            elif latest["stoch_k"] > self.config["stoch_overbought"] and latest["stoch_k"] < latest["stoch_d"]:
                sell_score += 0.125
                reasons.append("Bearish stochastic crossover in overbought region")
            
            # Bollinger Bands (15% weight)
            if latest["close"] < latest["bb_lower"]:
                buy_score += 0.15
                reasons.append("Price below lower Bollinger Band")
            elif latest["close"] > latest["bb_upper"]:
                sell_score += 0.15
                reasons.append("Price above upper Bollinger Band")
            
            # Pattern recognition (20% weight)
            pattern_dict = patterns.get("patterns", {})
            for pattern, confidence in pattern_dict.items():
                if "double_bottom" in pattern or "bullish_engulfing" in pattern or "hammer" in pattern:
                    buy_score += 0.2 * confidence
                    reasons.append(f"Bullish pattern: {pattern}")
                elif "double_top" in pattern or "bearish_engulfing" in pattern or "shooting_star" in pattern:
                    sell_score += 0.2 * confidence
                    reasons.append(f"Bearish pattern: {pattern}")
            
            # Support/Resistance (10% weight)
            current_price = latest["close"]
            
            # Check if price is near support
            for support in levels.get("support", []):
                if 0.99 * support <= current_price <= 1.01 * support:
                    buy_score += 0.1
                    reasons.append(f"Price at support level ({support:.2f})")
                    break
            
            # Check if price is near resistance
            for resistance in levels.get("resistance", []):
                if 0.99 * resistance <= current_price <= 1.01 * resistance:
                    sell_score += 0.1
                    reasons.append(f"Price at resistance level ({resistance:.2f})")
                    break
            
            # Adjust based on market regime
            regime_type = regime.get("regime", "neutral")
            volatility = regime.get("volatility", "medium")
            trend_strength = regime.get("trend", "neutral")
            
            # In trending markets, increase trend signal
            if regime_type == "trending":
                if buy_score > sell_score:
                    buy_score *= 1.2
                    reasons.append("Trending market favors continuation")
                elif sell_score > buy_score:
                    sell_score *= 1.2
                    reasons.append("Trending market favors continuation")
            
            # In ranging markets, increase mean reversion signal
            elif regime_type == "ranging":
                if buy_score > sell_score and latest["price_distance"] < -1.0:
                    buy_score *= 1.2
                    reasons.append("Ranging market favors mean reversion from oversold")
                elif sell_score > buy_score and latest["price_distance"] > 1.0:
                    sell_score *= 1.2
                    reasons.append("Ranging market favors mean reversion from overbought")
            
            # In volatile markets, reduce confidence
            if volatility == "high":
                buy_score *= 0.9
                sell_score *= 0.9
                reasons.append("High volatility reduces confidence")
            
            # Determine final signal
            if buy_score > sell_score and buy_score >= self.config["min_confidence"]:
                signal = "BUY"
                confidence = buy_score
                primary_reason = "bullish"
            elif sell_score > buy_score and sell_score >= self.config["min_confidence"]:
                signal = "SELL"
                confidence = sell_score
                primary_reason = "bearish"
            else:
                signal = "NEUTRAL"
                confidence = max(0.5, max(buy_score, sell_score))
                primary_reason = "neutral"
            
            # Cap confidence at 0.95
            confidence = min(0.95, confidence)
            
            # Construct reason string
            reason = f"Signal is {primary_reason} based on: " + ", ".join(reasons[:3])
            
            # Calculate stop loss and take profit levels
            stop_loss = 0.0
            take_profit = 0.0
            
            if signal == "BUY":
                # For buy signals, use recent low or support level for stop loss
                recent_low = df["low"].iloc[-10:].min()
                atr_distance = latest["atr"] * 1.5
                
                # Use support levels if available, otherwise use ATR
                if levels.get("support"):
                    closest_support = max([s for s in levels["support"] if s < current_price], default=recent_low)
                    stop_loss = min(closest_support, current_price - atr_distance)
                else:
                    stop_loss = current_price - atr_distance
                
                # Take profit based on nearest resistance or risk:reward ratio
                if levels.get("resistance"):
                    closest_resistance = min([r for r in levels["resistance"] if r > current_price], default=current_price * 1.05)
                    take_profit = closest_resistance
                else:
                    risk = current_price - stop_loss
                    take_profit = current_price + (risk * 2)  # 1:2 risk:reward ratio
                
            elif signal == "SELL":
                # For sell signals, use recent high or resistance level for stop loss
                recent_high = df["high"].iloc[-10:].max()
                atr_distance = latest["atr"] * 1.5
                
                # Use resistance levels if available, otherwise use ATR
                if levels.get("resistance"):
                    closest_resistance = min([r for r in levels["resistance"] if r > current_price], default=recent_high)
                    stop_loss = max(closest_resistance, current_price + atr_distance)
                else:
                    stop_loss = current_price + atr_distance
                
                # Take profit based on nearest support or risk:reward ratio
                if levels.get("support"):
                    closest_support = max([s for s in levels["support"] if s < current_price], default=current_price * 0.95)
                    take_profit = closest_support
                else:
                    risk = stop_loss - current_price
                    take_profit = current_price - (risk * 2)  # 1:2 risk:reward ratio
            
            # Create signal dictionary
            signal_dict = {
                "signal": signal,
                "confidence": confidence,
                "reason": reason,
                "price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "atr": latest["atr"],
                "regime": regime_type,
                "volatility": volatility,
                "timestamp": time.time()
            }
            
            return signal_dict
            
        except Exception as e:
            self.logger.error(f"Error generating signal score: {e}")
            return {
                "signal": "NEUTRAL",
                "confidence": 0.5,
                "reason": f"Error: {str(e)}"
            }
    
    def generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a trading signal based on the provided data.
        
        Args:
            data: Dictionary containing market data
            
        Returns:
            Signal dictionary with action recommendation
        """
        try:
            # Extract market data
            candles = data.get("candles", [])
            symbol = data.get("symbol", "Unknown")
            
            if not candles or len(candles) < 30:
                return {
                    "signal": "NEUTRAL",
                    "confidence": 0.0,
                    "reason": "Insufficient historical data",
                    "symbol": symbol
                }
            
            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            
            # Ensure numeric types
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Fill missing values
            df.fillna(method="ffill", inplace=True)
            
            # Calculate technical indicators
            df = self._calculate_technical_indicators(df)
            
            # Detect market regime
            regime = self._detect_market_regime(df)
            self.state["market_regime"] = regime.get("regime", "neutral")
            self.state["volatility_level"] = regime.get("volatility", "medium")
            self.state["trend_strength"] = regime.get("trend", "neutral")
            
            # Detect patterns
            patterns = self._detect_patterns(df)
            
            # Calculate support/resistance levels
            levels = self._calculate_support_resistance(df)
            
            # Generate signal
            signal = self._generate_signal_score(df, regime, patterns, levels)
            signal["symbol"] = symbol
            
            # Record signal
            self.state["last_signal"] = signal
            self.state["historical_signals"].append(signal)
            
            # Limit historical signals to last 100
            if len(self.state["historical_signals"]) > 100:
                self.state["historical_signals"] = self.state["historical_signals"][-100:]
            
            # Log signal
            self.logger.info(f"Generated signal for {symbol}: {signal['signal']} "
                            f"(confidence: {signal['confidence']:.2f}) - {signal['reason']}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return {
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "reason": f"Error: {str(e)}",
                "symbol": data.get("symbol", "Unknown")
            }
