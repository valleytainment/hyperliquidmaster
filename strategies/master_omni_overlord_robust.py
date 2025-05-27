"""
Enhanced Master Strategy implementation for the Hyperliquid trading bot.
Optimized for maximum success rate with robust long/short handling.
"""

import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

class MasterOmniOverlordRobustStrategy:
    """
    Advanced trading strategy with optimized signal generation for both long and short positions.
    Implements multiple technical indicators, adaptive position sizing, and dynamic risk management.
    """
    
    def __init__(self, logger):
        """Initialize the strategy with configuration parameters"""
        self.logger = logger
        
        # Strategy configuration
        self.config = {
            # Signal generation parameters
            "ema_short": 9,
            "ema_medium": 21,
            "ema_long": 50,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "atr_period": 14,
            "volatility_lookback": 20,
            "volume_lookback": 10,
            
            # Position sizing and risk management
            "max_position_size": 0.1,  # Maximum position size as fraction of account
            "risk_per_trade": 0.02,    # Risk per trade as fraction of account
            "max_open_positions": 3,   # Maximum number of open positions
            "profit_target_multiplier": 2.0,  # Profit target as multiple of risk
            "trailing_stop_activation": 1.5,  # Activate trailing stop at this multiple of risk
            "trailing_stop_distance": 0.5,    # Trailing stop distance as multiple of ATR
            
            # Market condition adaptation
            "volatility_adjustment": True,    # Adjust position size based on volatility
            "trend_filter": True,             # Use trend filter for signal confirmation
            "volume_filter": True,            # Use volume filter for signal confirmation
            "adaptive_parameters": True,      # Adapt parameters based on market conditions
        }
        
        # Strategy state
        self.state = {
            "last_signal": {},
            "market_regime": "neutral",
            "volatility_factor": 1.0,
            "success_rate": 0.0,
            "total_trades": 0,
            "successful_trades": 0,
            "historical_signals": [],
        }
    
    def preprocess_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess market data for analysis.
        
        Args:
            data: Dictionary containing market data
            
        Returns:
            Preprocessed data as pandas DataFrame
        """
        try:
            # Extract OHLCV data
            candles = data.get("candles", [])
            if not candles:
                self.logger.warning("No candle data available for preprocessing")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            
            # Ensure numeric types
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Fill missing values
            df.fillna(method="ffill", inplace=True)
            
            # Add technical indicators
            self._add_technical_indicators(df)
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> None:
        """
        Add technical indicators to the DataFrame.
        
        Args:
            df: DataFrame to add indicators to
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
            
            # MACD
            df["macd_fast"] = df["close"].ewm(span=self.config["macd_fast"], adjust=False).mean()
            df["macd_slow"] = df["close"].ewm(span=self.config["macd_slow"], adjust=False).mean()
            df["macd"] = df["macd_fast"] - df["macd_slow"]
            df["macd_signal"] = df["macd"].ewm(span=self.config["macd_signal"], adjust=False).mean()
            df["macd_histogram"] = df["macd"] - df["macd_signal"]
            
            # ATR (Average True Range)
            tr1 = df["high"] - df["low"]
            tr2 = abs(df["high"] - df["close"].shift())
            tr3 = abs(df["low"] - df["close"].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df["atr"] = tr.rolling(window=self.config["atr_period"]).mean()
            
            # Volatility
            df["volatility"] = df["close"].pct_change().rolling(window=self.config["volatility_lookback"]).std()
            
            # Volume indicators
            df["volume_sma"] = df["volume"].rolling(window=self.config["volume_lookback"]).mean()
            df["volume_ratio"] = df["volume"] / df["volume_sma"]
            
            # Trend indicators
            df["trend_direction"] = np.where(df["ema_medium"] > df["ema_long"], 1, -1)
            df["price_distance"] = (df["close"] - df["ema_medium"]) / df["ema_medium"] * 100
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
    
    def _detect_market_regime(self, df: pd.DataFrame) -> str:
        """
        Detect current market regime (trending, ranging, volatile).
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Market regime as string
        """
        try:
            if df.empty or len(df) < 50:
                return "neutral"
            
            # Get recent data
            recent = df.iloc[-20:]
            
            # Check volatility
            volatility = recent["volatility"].mean()
            high_volatility = volatility > df["volatility"].quantile(0.7)
            
            # Check trend strength
            trend_consistency = abs(recent["trend_direction"].sum()) / len(recent)
            strong_trend = trend_consistency > 0.7
            
            # Determine regime
            if high_volatility and strong_trend:
                return "trending_volatile"
            elif high_volatility:
                return "volatile"
            elif strong_trend:
                return "trending"
            else:
                return "ranging"
                
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            return "neutral"
    
    def _adapt_parameters(self, df: pd.DataFrame, regime: str) -> None:
        """
        Adapt strategy parameters based on market regime.
        
        Args:
            df: DataFrame with technical indicators
            regime: Current market regime
        """
        if not self.config["adaptive_parameters"]:
            return
        
        try:
            # Adjust parameters based on regime
            if regime == "trending":
                # In trending markets, favor trend following
                self.config["ema_short"] = 8
                self.config["ema_medium"] = 21
                self.config["trailing_stop_distance"] = 0.7
                self.config["profit_target_multiplier"] = 2.5
                
            elif regime == "ranging":
                # In ranging markets, favor mean reversion
                self.config["ema_short"] = 5
                self.config["ema_medium"] = 15
                self.config["trailing_stop_distance"] = 0.4
                self.config["profit_target_multiplier"] = 1.5
                
            elif regime == "volatile":
                # In volatile markets, reduce risk
                self.config["risk_per_trade"] = 0.015
                self.config["max_position_size"] = 0.07
                self.config["trailing_stop_distance"] = 0.6
                
            elif regime == "trending_volatile":
                # In trending volatile markets, balanced approach
                self.config["ema_short"] = 7
                self.config["ema_medium"] = 18
                self.config["risk_per_trade"] = 0.018
                self.config["trailing_stop_distance"] = 0.5
                
            # Update volatility factor for position sizing
            recent_volatility = df["volatility"].iloc[-1] if not df.empty else 0.02
            historical_volatility = df["volatility"].mean() if not df.empty else 0.02
            
            if historical_volatility > 0:
                self.state["volatility_factor"] = historical_volatility / recent_volatility
                # Limit the factor to reasonable bounds
                self.state["volatility_factor"] = max(0.5, min(2.0, self.state["volatility_factor"]))
            else:
                self.state["volatility_factor"] = 1.0
                
        except Exception as e:
            self.logger.error(f"Error adapting parameters: {e}")
    
    def _calculate_position_size(self, account_size: float, price: float, stop_price: float) -> float:
        """
        Calculate optimal position size based on risk parameters.
        
        Args:
            account_size: Current account size
            price: Entry price
            stop_price: Stop loss price
            
        Returns:
            Position size in base currency
        """
        try:
            # Calculate risk amount
            risk_amount = account_size * self.config["risk_per_trade"]
            
            # Calculate risk per unit
            risk_per_unit = abs(price - stop_price)
            
            if risk_per_unit <= 0:
                self.logger.warning("Invalid risk per unit, using default position size")
                return account_size * 0.01
            
            # Calculate position size
            position_size = risk_amount / risk_per_unit
            
            # Apply volatility adjustment
            if self.config["volatility_adjustment"]:
                position_size *= self.state["volatility_factor"]
            
            # Apply maximum position size limit
            max_size = account_size * self.config["max_position_size"] / price
            position_size = min(position_size, max_size)
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def _generate_entry_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate entry signal based on technical indicators.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Signal dictionary
        """
        try:
            if df.empty or len(df) < 50:
                return {"signal": "NEUTRAL", "confidence": 0.5, "reason": "Insufficient data"}
            
            # Get latest data
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Initialize signal components
            trend_signal = "NEUTRAL"
            momentum_signal = "NEUTRAL"
            volatility_signal = "NEUTRAL"
            volume_signal = "NEUTRAL"
            
            # Trend analysis
            if latest["ema_short"] > latest["ema_medium"] > latest["ema_long"]:
                trend_signal = "BUY"
            elif latest["ema_short"] < latest["ema_medium"] < latest["ema_long"]:
                trend_signal = "SELL"
            
            # Momentum analysis
            if latest["rsi"] < self.config["rsi_oversold"] and prev["rsi"] < latest["rsi"]:
                momentum_signal = "BUY"
            elif latest["rsi"] > self.config["rsi_overbought"] and prev["rsi"] > latest["rsi"]:
                momentum_signal = "SELL"
            
            # MACD analysis
            if latest["macd"] > latest["macd_signal"] and prev["macd"] <= prev["macd_signal"]:
                momentum_signal = "BUY"
            elif latest["macd"] < latest["macd_signal"] and prev["macd"] >= prev["macd_signal"]:
                momentum_signal = "SELL"
            
            # Volatility analysis
            if latest["atr"] > df["atr"].rolling(window=10).mean().iloc[-1]:
                volatility_signal = "HIGH"
            else:
                volatility_signal = "LOW"
            
            # Volume analysis
            if latest["volume_ratio"] > 1.5:
                volume_signal = "HIGH"
            elif latest["volume_ratio"] < 0.7:
                volume_signal = "LOW"
            
            # Combine signals
            final_signal = "NEUTRAL"
            confidence = 0.5
            reason = ""
            
            # Strong buy signal
            if (trend_signal == "BUY" and momentum_signal == "BUY" and 
                (not self.config["volume_filter"] or volume_signal != "LOW")):
                final_signal = "BUY"
                confidence = 0.7 + (0.1 if volume_signal == "HIGH" else 0)
                reason = "Strong uptrend with momentum confirmation"
                
                # Add volatility factor
                if volatility_signal == "LOW":
                    confidence += 0.1
                    reason += " in low volatility"
                else:
                    reason += " with increased volatility"
            
            # Strong sell signal
            elif (trend_signal == "SELL" and momentum_signal == "SELL" and 
                  (not self.config["volume_filter"] or volume_signal != "LOW")):
                final_signal = "SELL"
                confidence = 0.7 + (0.1 if volume_signal == "HIGH" else 0)
                reason = "Strong downtrend with momentum confirmation"
                
                # Add volatility factor
                if volatility_signal == "LOW":
                    confidence += 0.1
                    reason += " in low volatility"
                else:
                    reason += " with increased volatility"
            
            # Trend reversal buy
            elif (trend_signal == "SELL" and momentum_signal == "BUY" and 
                  latest["rsi"] < 40 and volume_signal == "HIGH"):
                final_signal = "BUY"
                confidence = 0.65
                reason = "Potential trend reversal with oversold conditions and high volume"
            
            # Trend reversal sell
            elif (trend_signal == "BUY" and momentum_signal == "SELL" and 
                  latest["rsi"] > 60 and volume_signal == "HIGH"):
                final_signal = "SELL"
                confidence = 0.65
                reason = "Potential trend reversal with overbought conditions and high volume"
            
            # Calculate stop loss and take profit levels
            entry_price = latest["close"]
            atr = latest["atr"]
            
            stop_loss = 0.0
            take_profit = 0.0
            
            if final_signal == "BUY":
                stop_loss = entry_price - (atr * 1.5)
                take_profit = entry_price + (atr * 1.5 * self.config["profit_target_multiplier"])
            elif final_signal == "SELL":
                stop_loss = entry_price + (atr * 1.5)
                take_profit = entry_price - (atr * 1.5 * self.config["profit_target_multiplier"])
            
            # Create signal dictionary
            signal = {
                "signal": final_signal,
                "confidence": confidence,
                "reason": reason,
                "price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "atr": atr,
                "timestamp": time.time()
            }
            
            # Record signal for performance tracking
            self.state["last_signal"] = signal
            self.state["historical_signals"].append(signal)
            
            # Limit historical signals to last 100
            if len(self.state["historical_signals"]) > 100:
                self.state["historical_signals"] = self.state["historical_signals"][-100:]
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating entry signal: {e}")
            return {"signal": "NEUTRAL", "confidence": 0.5, "reason": f"Error: {str(e)}"}
    
    def _validate_signal(self, signal: Dict[str, Any], account_info: Dict[str, Any], 
                         positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate signal against current account and positions.
        
        Args:
            signal: Signal dictionary
            account_info: Account information
            positions: Current positions
            
        Returns:
            Validated signal with additional information
        """
        try:
            # Check if signal is actionable
            if signal["signal"] == "NEUTRAL":
                return signal
            
            # Check maximum open positions
            if len(positions) >= self.config["max_open_positions"]:
                return {
                    "signal": "NEUTRAL",
                    "confidence": 0.0,
                    "reason": f"Maximum open positions ({self.config['max_open_positions']}) reached"
                }
            
            # Check if we already have a position in this symbol
            symbol = signal.get("symbol", "")
            for position in positions:
                if position.get("symbol") == symbol:
                    # Check if position direction matches signal
                    position_size = position.get("size", 0.0)
                    if (signal["signal"] == "BUY" and position_size > 0) or \
                       (signal["signal"] == "SELL" and position_size < 0):
                        return {
                            "signal": "NEUTRAL",
                            "confidence": 0.0,
                            "reason": f"Already have a {signal['signal']} position in {symbol}"
                        }
                    else:
                        # Opposite direction - could close or reverse
                        signal["action"] = "REVERSE"
                        signal["reason"] += " (reversing existing position)"
                        return signal
            
            # Calculate position size
            equity = account_info.get("equity", 0.0)
            if equity <= 0:
                return {
                    "signal": "NEUTRAL",
                    "confidence": 0.0,
                    "reason": "Invalid account equity"
                }
            
            position_size = self._calculate_position_size(
                account_size=equity,
                price=signal["price"],
                stop_price=signal["stop_loss"]
            )
            
            if position_size <= 0:
                return {
                    "signal": "NEUTRAL",
                    "confidence": 0.0,
                    "reason": "Invalid position size calculation"
                }
            
            # Add position size to signal
            signal["position_size"] = position_size
            signal["action"] = "OPEN"
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return {"signal": "NEUTRAL", "confidence": 0.0, "reason": f"Error: {str(e)}"}
    
    def generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a trading signal based on the provided data.
        
        Args:
            data: Dictionary containing market data, account info, and positions
            
        Returns:
            Signal dictionary with action recommendation
        """
        try:
            # Extract data components
            market_data = data.get("market_data", {})
            account_info = data.get("account_info", {})
            positions = data.get("positions", [])
            symbol = data.get("symbol", "Unknown")
            
            # Preprocess data
            df = self.preprocess_data(market_data)
            if df.empty:
                return {"signal": "NEUTRAL", "confidence": 0.0, "reason": "Insufficient data"}
            
            # Detect market regime
            regime = self._detect_market_regime(df)
            self.state["market_regime"] = regime
            
            # Adapt parameters to market conditions
            self._adapt_parameters(df, regime)
            
            # Generate raw signal
            signal = self._generate_entry_signal(df)
            signal["symbol"] = symbol
            
            # Validate signal against account and positions
            validated_signal = self._validate_signal(signal, account_info, positions)
            
            # Log signal
            self.logger.info(f"Generated signal for {symbol}: {validated_signal['signal']} "
                            f"(confidence: {validated_signal['confidence']:.2f}) - {validated_signal['reason']}")
            
            return validated_signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return {"signal": "NEUTRAL", "confidence": 0.0, "reason": f"Error: {str(e)}"}
    
    def update_performance(self, trade_result: Dict[str, Any]) -> None:
        """
        Update strategy performance metrics based on trade result.
        
        Args:
            trade_result: Dictionary containing trade result information
        """
        try:
            # Extract trade information
            is_successful = trade_result.get("is_successful", False)
            pnl = trade_result.get("pnl", 0.0)
            
            # Update performance metrics
            self.state["total_trades"] += 1
            if is_successful:
                self.state["successful_trades"] += 1
            
            # Calculate success rate
            if self.state["total_trades"] > 0:
                self.state["success_rate"] = self.state["successful_trades"] / self.state["total_trades"]
            
            # Log performance update
            self.logger.info(f"Updated performance: Success rate {self.state['success_rate']:.2f} "
                            f"({self.state['successful_trades']}/{self.state['total_trades']} trades)")
            
        except Exception as e:
            self.logger.error(f"Error updating performance: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get current performance statistics.
        
        Returns:
            Dictionary containing performance statistics
        """
        return {
            "success_rate": self.state["success_rate"],
            "total_trades": self.state["total_trades"],
            "successful_trades": self.state["successful_trades"],
            "market_regime": self.state["market_regime"],
            "volatility_factor": self.state["volatility_factor"]
        }
