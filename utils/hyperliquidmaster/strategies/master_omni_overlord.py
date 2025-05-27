"""
Master Omni Overlord Trading Strategy

This module implements the ultimate trading strategy that combines multiple advanced techniques
and adapts to market conditions for maximum performance on Hyperliquid.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

# Import strategy components
from strategies.triple_confluence import TripleConfluenceStrategy
from strategies.oracle_update import OracleUpdateStrategy

class MasterOmniOverlordStrategy:
    """
    Master Omni Overlord Strategy - The ultimate trading strategy that combines
    multiple advanced techniques and adapts to market conditions.
    """
    
    def __init__(self, config: Dict, logger=None):
        """
        Initialize the Master Omni Overlord Strategy.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        # Setup logging
        self.logger = logger or self._setup_logger()
        self.logger.info("Initializing Master Omni Overlord Strategy...")
        
        # Store configuration
        self.config = config
        
        # Initialize sub-strategies
        self.triple_confluence = TripleConfluenceStrategy(config, self.logger)
        self.oracle_update = OracleUpdateStrategy(config, self.logger)
        
        # Strategy weights - will be dynamically adjusted
        self.strategy_weights = {
            "triple_confluence": 0.5,
            "oracle_update": 0.5
        }
        
        # Market regime detection parameters
        self.volatility_lookback = 20
        self.trend_lookback = 50
        self.regime_threshold = 0.5
        
        # Multi-timeframe parameters
        self.timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        self.timeframe_weights = {
            "1m": 0.05,
            "5m": 0.10,
            "15m": 0.15,
            "1h": 0.30,
            "4h": 0.25,
            "1d": 0.15
        }
        
        # Advanced risk management
        self.max_risk_per_trade = config.get("risk_percent", 0.01)  # 1% default
        self.max_open_positions = 3
        self.max_correlated_exposure = 0.15  # 15% max exposure to correlated assets
        self.dynamic_position_sizing = True
        
        # Performance tracking
        self.strategy_performance = {
            "triple_confluence": {
                "win_rate": 0.5,
                "profit_factor": 1.5,
                "recent_trades": []
            },
            "oracle_update": {
                "win_rate": 0.5,
                "profit_factor": 1.5,
                "recent_trades": []
            }
        }
        
        # Adaptive parameters
        self.adaptive_params = {
            "signal_threshold": 0.7,
            "trend_filter_strength": 0.5,
            "mean_reversion_factor": 0.3,
            "volatility_adjustment": 1.0
        }
        
        # Initialize market state
        self.market_state = {
            "regime": "unknown",
            "volatility": "medium",
            "trend_strength": 0.5,
            "sentiment": "neutral",
            "liquidity": "normal"
        }
        
        self.logger.info("Master Omni Overlord Strategy initialized")
        
    def _setup_logger(self) -> logging.Logger:
        """
        Set up the logger.
        
        Returns:
            Configured logger
        """
        logger = logging.getLogger("MasterOmniOverlordStrategy")
        logger.setLevel(logging.INFO)
        
        # Check if handlers already exist
        if not logger.handlers:
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            console_handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(console_handler)
            
        return logger
        
    def detect_market_regime(self, data: pd.DataFrame) -> str:
        """
        Detect the current market regime (trending, ranging, volatile).
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Market regime string
        """
        try:
            # Calculate volatility
            returns = data["close"].pct_change().dropna()
            volatility = returns.rolling(self.volatility_lookback).std().iloc[-1]
            
            # Calculate trend strength using ADX-like measure
            high_low_range = data["high"] - data["low"]
            true_range = pd.concat([
                high_low_range,
                (data["high"] - data["close"].shift(1)).abs(),
                (data["low"] - data["close"].shift(1)).abs()
            ], axis=1).max(axis=1)
            
            directional_movement_plus = (data["high"] - data["high"].shift(1)).clip(lower=0)
            directional_movement_minus = (data["low"].shift(1) - data["low"]).clip(lower=0)
            
            condition = directional_movement_plus > directional_movement_minus
            directional_indicator_plus = (directional_movement_plus / true_range).rolling(self.trend_lookback).mean().iloc[-1]
            directional_indicator_minus = (directional_movement_minus / true_range).rolling(self.trend_lookback).mean().iloc[-1]
            
            trend_strength = abs(directional_indicator_plus - directional_indicator_minus) / (directional_indicator_plus + directional_indicator_minus)
            
            # Determine regime
            if volatility > 0.03:  # High volatility threshold
                regime = "volatile"
            elif trend_strength > self.regime_threshold:
                regime = "trending"
            else:
                regime = "ranging"
                
            self.market_state["regime"] = regime
            self.market_state["volatility"] = "high" if volatility > 0.03 else ("medium" if volatility > 0.01 else "low")
            self.market_state["trend_strength"] = trend_strength
            
            self.logger.info(f"Market regime detected: {regime} (volatility: {volatility:.4f}, trend strength: {trend_strength:.4f})")
            return regime
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            return "unknown"
            
    def adjust_strategy_weights(self):
        """
        Dynamically adjust strategy weights based on market regime and performance.
        """
        try:
            # Adjust based on market regime
            if self.market_state["regime"] == "trending":
                # Triple Confluence performs better in trending markets
                base_weight_tc = 0.7
                base_weight_ou = 0.3
            elif self.market_state["regime"] == "ranging":
                # Oracle Update performs better in ranging markets
                base_weight_tc = 0.3
                base_weight_ou = 0.7
            elif self.market_state["regime"] == "volatile":
                # More balanced in volatile markets with slight edge to Oracle Update
                base_weight_tc = 0.4
                base_weight_ou = 0.6
            else:
                # Default balanced weights
                base_weight_tc = 0.5
                base_weight_ou = 0.5
                
            # Adjust based on recent performance
            tc_performance = self.strategy_performance["triple_confluence"]["win_rate"] * self.strategy_performance["triple_confluence"]["profit_factor"]
            ou_performance = self.strategy_performance["oracle_update"]["win_rate"] * self.strategy_performance["oracle_update"]["profit_factor"]
            
            # Normalize performance scores
            total_performance = tc_performance + ou_performance
            if total_performance > 0:
                performance_weight_tc = tc_performance / total_performance
                performance_weight_ou = ou_performance / total_performance
                
                # Blend base weights with performance weights (70% base, 30% performance)
                final_weight_tc = 0.7 * base_weight_tc + 0.3 * performance_weight_tc
                final_weight_ou = 0.7 * base_weight_ou + 0.3 * performance_weight_ou
                
                # Normalize to ensure weights sum to 1
                sum_weights = final_weight_tc + final_weight_ou
                self.strategy_weights["triple_confluence"] = final_weight_tc / sum_weights
                self.strategy_weights["oracle_update"] = final_weight_ou / sum_weights
            else:
                # Fallback to base weights if performance data is insufficient
                self.strategy_weights["triple_confluence"] = base_weight_tc
                self.strategy_weights["oracle_update"] = base_weight_ou
                
            self.logger.info(f"Adjusted strategy weights: Triple Confluence={self.strategy_weights['triple_confluence']:.2f}, Oracle Update={self.strategy_weights['oracle_update']:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error adjusting strategy weights: {str(e)}")
            # Fallback to default weights
            self.strategy_weights["triple_confluence"] = 0.5
            self.strategy_weights["oracle_update"] = 0.5
            
    def adjust_adaptive_parameters(self):
        """
        Adjust adaptive parameters based on market conditions.
        """
        try:
            # Adjust signal threshold based on volatility
            if self.market_state["volatility"] == "high":
                self.adaptive_params["signal_threshold"] = 0.8  # Higher threshold in volatile markets
            elif self.market_state["volatility"] == "low":
                self.adaptive_params["signal_threshold"] = 0.6  # Lower threshold in calm markets
            else:
                self.adaptive_params["signal_threshold"] = 0.7  # Default
                
            # Adjust trend filter strength based on trend strength
            self.adaptive_params["trend_filter_strength"] = min(0.8, max(0.2, self.market_state["trend_strength"]))
            
            # Adjust mean reversion factor based on regime
            if self.market_state["regime"] == "ranging":
                self.adaptive_params["mean_reversion_factor"] = 0.6  # Stronger mean reversion in ranging markets
            elif self.market_state["regime"] == "trending":
                self.adaptive_params["mean_reversion_factor"] = 0.2  # Weaker mean reversion in trending markets
            else:
                self.adaptive_params["mean_reversion_factor"] = 0.3  # Default
                
            # Adjust volatility adjustment based on volatility
            if self.market_state["volatility"] == "high":
                self.adaptive_params["volatility_adjustment"] = 0.7  # Reduce position size in volatile markets
            elif self.market_state["volatility"] == "low":
                self.adaptive_params["volatility_adjustment"] = 1.2  # Increase position size in calm markets
            else:
                self.adaptive_params["volatility_adjustment"] = 1.0  # Default
                
            self.logger.info(f"Adjusted adaptive parameters: {self.adaptive_params}")
            
        except Exception as e:
            self.logger.error(f"Error adjusting adaptive parameters: {str(e)}")
            # Reset to defaults
            self.adaptive_params = {
                "signal_threshold": 0.7,
                "trend_filter_strength": 0.5,
                "mean_reversion_factor": 0.3,
                "volatility_adjustment": 1.0
            }
            
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk parameters and market conditions.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            
        Returns:
            Position size in USD
        """
        try:
            # Base risk percentage
            risk_pct = self.max_risk_per_trade
            
            # Adjust risk based on market conditions
            if self.dynamic_position_sizing:
                # Reduce risk in volatile markets
                if self.market_state["volatility"] == "high":
                    risk_pct *= 0.7
                elif self.market_state["volatility"] == "low":
                    risk_pct *= 1.2
                    
                # Adjust based on trend strength
                trend_confidence = min(1.5, max(0.5, self.market_state["trend_strength"] * 2))
                risk_pct *= trend_confidence
                
                # Apply volatility adjustment from adaptive parameters
                risk_pct *= self.adaptive_params["volatility_adjustment"]
                
                # Cap maximum risk
                risk_pct = min(risk_pct, self.max_risk_per_trade * 1.5)
            
            # Calculate dollar risk
            account_size = 10000  # Default account size, should be replaced with actual balance
            dollar_risk = account_size * risk_pct
            
            # Calculate position size based on stop loss distance
            stop_distance_pct = abs(entry_price - stop_loss) / entry_price
            if stop_distance_pct == 0:
                self.logger.warning(f"Stop distance is zero for {symbol}, using default 1% stop")
                stop_distance_pct = 0.01
                
            position_size = dollar_risk / (entry_price * stop_distance_pct)
            
            self.logger.info(f"Calculated position size for {symbol}: {position_size:.2f} units (risk: {risk_pct:.2%}, stop distance: {stop_distance_pct:.2%})")
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            # Return conservative default
            return 0.1  # Small default position
            
    def analyze_multi_timeframe(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Perform multi-timeframe analysis to generate weighted signals.
        
        Args:
            data_dict: Dictionary of DataFrames for different timeframes
            
        Returns:
            Dictionary with signal strengths
        """
        try:
            signals = {}
            
            # Get signals from each timeframe
            for timeframe, data in data_dict.items():
                if timeframe not in self.timeframe_weights:
                    continue
                    
                # Detect trend direction
                sma_short = data["close"].rolling(20).mean()
                sma_long = data["close"].rolling(50).mean()
                trend_direction = 1 if sma_short.iloc[-1] > sma_long.iloc[-1] else -1
                
                # Calculate momentum
                roc = (data["close"].iloc[-1] / data["close"].iloc[-10] - 1) * 100
                momentum = np.tanh(roc / 10)  # Normalize between -1 and 1
                
                # Calculate volatility
                returns = data["close"].pct_change().dropna()
                volatility = returns.rolling(20).std().iloc[-1]
                
                # Calculate signal strength
                signal_strength = trend_direction * (0.7 + 0.3 * momentum)
                
                # Adjust signal based on volatility
                volatility_factor = 1 - min(0.5, volatility * 10)  # Reduce signal in high volatility
                signal_strength *= volatility_factor
                
                signals[timeframe] = signal_strength
                
            # Calculate weighted signal
            weighted_signal = 0
            total_weight = 0
            
            for timeframe, signal in signals.items():
                weight = self.timeframe_weights.get(timeframe, 0)
                weighted_signal += signal * weight
                total_weight += weight
                
            if total_weight > 0:
                final_signal = weighted_signal / total_weight
            else:
                final_signal = 0
                
            self.logger.info(f"Multi-timeframe analysis result: {final_signal:.4f}")
            
            return {
                "signal": final_signal,
                "signals": signals
            }
            
        except Exception as e:
            self.logger.error(f"Error in multi-timeframe analysis: {str(e)}")
            return {"signal": 0, "signals": {}}
            
    async def generate_signal(self, symbol: str, market_data: Dict, order_book: Dict = None, positions: Dict = None) -> Dict:
        """
        Generate a trading signal for a specific symbol using the Master Omni Overlord Strategy.
        
        Args:
            symbol: Symbol to generate signal for
            market_data: Market data dictionary
            order_book: Optional order book dictionary
            positions: Optional positions dictionary
            
        Returns:
            Dictionary with signal information
        """
        try:
            self.logger.info(f"Generating signal for {symbol} using Master Omni Overlord Strategy")
            
            # Convert market data to DataFrame format expected by generate_signals
            # This is a simple adapter to maintain compatibility with existing code
            data = pd.DataFrame({
                "timestamp": [market_data.get("timestamp", datetime.now().timestamp())],
                "open": [market_data.get("last_price", 0) * 0.99],  # Approximate
                "high": [market_data.get("last_price", 0) * 1.01],  # Approximate
                "low": [market_data.get("last_price", 0) * 0.99],   # Approximate
                "close": [market_data.get("last_price", 0)],
                "volume": [0]  # Not available in market_data
            })
            
            # Call the synchronous generate_signals method
            signal_info = self.generate_signals(data)
            
            # Adapt the result to the expected format for live_simulation
            action = "none"
            if signal_info.get("decision") == "buy":
                action = "long"
            elif signal_info.get("decision") == "sell":
                action = "short"
                
            quantity = signal_info.get("position_size", 0.0)
            
            # Create the signal dictionary expected by live_simulation
            signal = {
                "action": action,
                "symbol": symbol,
                "quantity": quantity,
                "entry_price": signal_info.get("entry_price", market_data.get("last_price", 0)),
                "stop_loss": signal_info.get("stop_loss", 0),
                "take_profit": signal_info.get("take_profit", 0),
                "confidence": abs(signal_info.get("signal_strength", 0)),
                "timestamp": datetime.now().timestamp(),
                "market_regime": signal_info.get("market_regime", "unknown"),
                "strategy_weights": signal_info.get("strategy_weights", {}),
                "adaptive_params": signal_info.get("adaptive_params", {})
            }
            
            self.logger.info(f"Generated signal for {symbol}: {action} with quantity {quantity}")
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {str(e)}")
            return {"action": "none", "reason": str(e)}
    
    def generate_signals(self, data: pd.DataFrame, multi_tf_data: Dict[str, pd.DataFrame] = None) -> Dict:
        """
        Generate trading signals using the Master Omni Overlord Strategy.
        
        Args:
            data: Market data DataFrame
            multi_tf_data: Optional dictionary of DataFrames for different timeframes
            
        Returns:
            Dictionary with signal information
        """
        try:
            # Detect market regime
            regime = self.detect_market_regime(data)
            
            # Adjust strategy weights and parameters
            self.adjust_strategy_weights()
            self.adjust_adaptive_parameters()
            
            # Get signals from sub-strategies
            tc_signal = self.triple_confluence.generate_signals(data)
            ou_signal = self.oracle_update.generate_signals(data)
            
            # Multi-timeframe analysis if data is available
            mtf_signal = 0
            if multi_tf_data:
                mtf_result = self.analyze_multi_timeframe(multi_tf_data)
                mtf_signal = mtf_result["signal"]
            
            # Combine signals with weights
            tc_weight = self.strategy_weights["triple_confluence"]
            ou_weight = self.strategy_weights["oracle_update"]
            
            # Base signal from weighted sub-strategies
            base_signal = (
                tc_weight * tc_signal.get("signal", 0) +
                ou_weight * ou_signal.get("signal", 0)
            )
            
            # Incorporate multi-timeframe signal (30% weight)
            if multi_tf_data:
                final_signal = 0.7 * base_signal + 0.3 * mtf_signal
            else:
                final_signal = base_signal
                
            # Apply adaptive threshold
            signal_threshold = self.adaptive_params["signal_threshold"]
            
            # Determine final trading decision
            if final_signal > signal_threshold:
                decision = "buy"
            elif final_signal < -signal_threshold:
                decision = "sell"
            else:
                decision = "neutral"
                
            # Calculate entry, stop loss, and take profit levels
            current_price = data["close"].iloc[-1]
            
            if decision == "buy":
                # Entry slightly above current price to ensure execution
                entry_price = current_price * 1.001
                
                # Stop loss based on recent volatility and support levels
                volatility = data["close"].pct_change().rolling(20).std().iloc[-1]
                stop_distance = max(0.01, volatility * 2)  # At least 1% stop distance
                
                # Find recent support level
                recent_lows = data["low"].rolling(10).min().iloc[-5:]
                support_level = recent_lows.min()
                
                # Use the higher of volatility-based stop or support-based stop
                support_stop = support_level * 0.995  # Slightly below support
                volatility_stop = entry_price * (1 - stop_distance)
                stop_loss = max(support_stop, volatility_stop)
                
                # Take profit based on risk-reward ratio
                risk = entry_price - stop_loss
                take_profit = entry_price + (risk * 2)  # 1:2 risk-reward
                
            elif decision == "sell":
                # Entry slightly below current price to ensure execution
                entry_price = current_price * 0.999
                
                # Stop loss based on recent volatility and resistance levels
                volatility = data["close"].pct_change().rolling(20).std().iloc[-1]
                stop_distance = max(0.01, volatility * 2)  # At least 1% stop distance
                
                # Find recent resistance level
                recent_highs = data["high"].rolling(10).max().iloc[-5:]
                resistance_level = recent_highs.max()
                
                # Use the lower of volatility-based stop or resistance-based stop
                resistance_stop = resistance_level * 1.005  # Slightly above resistance
                volatility_stop = entry_price * (1 + stop_distance)
                stop_loss = min(resistance_stop, volatility_stop)
                
                # Take profit based on risk-reward ratio
                risk = stop_loss - entry_price
                take_profit = entry_price - (risk * 2)  # 1:2 risk-reward
                
            else:
                entry_price = current_price
                stop_loss = current_price * 0.95  # Default 5% stop loss
                take_profit = current_price * 1.1  # Default 10% take profit
                
            # Calculate position size
            position_size = self.calculate_position_size("BTC-USD-PERP", entry_price, stop_loss)
            
            # Prepare signal result
            signal_result = {
                "timestamp": datetime.now().isoformat(),
                "symbol": "BTC-USD-PERP",  # Default symbol, should be replaced with actual symbol
                "decision": decision,
                "signal_strength": final_signal,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "position_size": position_size,
                "market_regime": regime,
                "strategy_weights": self.strategy_weights.copy(),
                "adaptive_params": self.adaptive_params.copy(),
                "sub_signals": {
                    "triple_confluence": tc_signal.get("signal", 0),
                    "oracle_update": ou_signal.get("signal", 0),
                    "multi_timeframe": mtf_signal
                }
            }
            
            self.logger.info(f"Generated signal: {decision} with strength {final_signal:.4f}")
            return signal_result
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "decision": "error",
                "signal_strength": 0,
                "error": str(e)
            }
            
    def update_performance(self, strategy: str, trade_result: Dict):
        """
        Update performance metrics for a strategy.
        
        Args:
            strategy: Strategy name
            trade_result: Trade result dictionary
        """
        try:
            if strategy not in self.strategy_performance:
                self.logger.warning(f"Unknown strategy: {strategy}")
                return
                
            # Add to recent trades
            self.strategy_performance[strategy]["recent_trades"].append(trade_result)
            
            # Keep only the last 20 trades
            if len(self.strategy_performance[strategy]["recent_trades"]) > 20:
                self.strategy_performance[strategy]["recent_trades"] = self.strategy_performance[strategy]["recent_trades"][-20:]
                
            # Calculate win rate
            wins = sum(1 for trade in self.strategy_performance[strategy]["recent_trades"] if trade.get("profit", 0) > 0)
            total = len(self.strategy_performance[strategy]["recent_trades"])
            
            if total > 0:
                win_rate = wins / total
                self.strategy_performance[strategy]["win_rate"] = win_rate
                
            # Calculate profit factor
            profits = sum(trade.get("profit", 0) for trade in self.strategy_performance[strategy]["recent_trades"] if trade.get("profit", 0) > 0)
            losses = sum(abs(trade.get("profit", 0)) for trade in self.strategy_performance[strategy]["recent_trades"] if trade.get("profit", 0) < 0)
            
            if losses > 0:
                profit_factor = profits / losses
                self.strategy_performance[strategy]["profit_factor"] = profit_factor
            elif profits > 0:
                self.strategy_performance[strategy]["profit_factor"] = 10.0  # High value for no losses
                
            self.logger.info(f"Updated performance for {strategy}: win rate={self.strategy_performance[strategy]['win_rate']:.2f}, profit factor={self.strategy_performance[strategy]['profit_factor']:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error updating performance: {str(e)}")
            
    def backtest(self, data: pd.DataFrame, multi_tf_data: Dict[str, pd.DataFrame] = None) -> Dict:
        """
        Backtest the Master Omni Overlord Strategy.
        
        Args:
            data: Market data DataFrame
            multi_tf_data: Optional dictionary of DataFrames for different timeframes
            
        Returns:
            Dictionary with backtest results
        """
        try:
            self.logger.info("Starting backtest of Master Omni Overlord Strategy...")
            
            # Initialize results
            results = {
                "trades": [],
                "equity_curve": [],
                "metrics": {}
            }
            
            # Initial equity
            equity = 10000
            results["equity_curve"].append({"timestamp": data.iloc[0]["timestamp"], "equity": equity})
            
            # Minimum lookback period
            min_lookback = 50
            
            # Iterate through data
            for i in range(min_lookback, len(data)):
                # Get historical data up to current point
                historical_data = data.iloc[:i+1]
                
                # Prepare multi-timeframe data if available
                current_mtf_data = None
                if multi_tf_data:
                    current_mtf_data = {
                        tf: df.iloc[:i+1] for tf, df in multi_tf_data.items()
                    }
                
                # Generate signal
                signal = self.generate_signals(historical_data, current_mtf_data)
                
                # Skip if no clear decision
                if signal["decision"] not in ["buy", "sell"]:
                    continue
                    
                # Simulate trade execution
                entry_price = signal["entry_price"]
                stop_loss = signal["stop_loss"]
                take_profit = signal["take_profit"]
                position_size = signal["position_size"]
                
                # Determine trade direction
                direction = 1 if signal["decision"] == "buy" else -1
                
                # Look ahead to see if stop loss or take profit was hit
                exit_price = None
                exit_timestamp = None
                trade_result = "open"
                
                for j in range(i+1, min(i+100, len(data))):
                    future_data = data.iloc[j]
                    
                    if direction == 1:  # Long trade
                        if future_data["low"] <= stop_loss:
                            exit_price = stop_loss
                            exit_timestamp = future_data["timestamp"]
                            trade_result = "stop_loss"
                            break
                        elif future_data["high"] >= take_profit:
                            exit_price = take_profit
                            exit_timestamp = future_data["timestamp"]
                            trade_result = "take_profit"
                            break
                    else:  # Short trade
                        if future_data["high"] >= stop_loss:
                            exit_price = stop_loss
                            exit_timestamp = future_data["timestamp"]
                            trade_result = "stop_loss"
                            break
                        elif future_data["low"] <= take_profit:
                            exit_price = take_profit
                            exit_timestamp = future_data["timestamp"]
                            trade_result = "take_profit"
                            break
                
                # If trade wasn't closed, use the last available price
                if exit_price is None:
                    exit_price = data.iloc[-1]["close"]
                    exit_timestamp = data.iloc[-1]["timestamp"]
                    trade_result = "open"
                
                # Calculate profit/loss
                profit_pct = direction * (exit_price - entry_price) / entry_price
                profit_amount = equity * position_size * profit_pct
                
                # Update equity
                equity += profit_amount
                
                # Record trade
                trade = {
                    "entry_timestamp": historical_data.iloc[-1]["timestamp"],
                    "entry_price": entry_price,
                    "exit_timestamp": exit_timestamp,
                    "exit_price": exit_price,
                    "direction": "long" if direction == 1 else "short",
                    "position_size": position_size,
                    "profit_pct": profit_pct,
                    "profit_amount": profit_amount,
                    "result": trade_result,
                    "strategy_weights": signal["strategy_weights"],
                    "market_regime": signal["market_regime"]
                }
                
                results["trades"].append(trade)
                results["equity_curve"].append({"timestamp": exit_timestamp, "equity": equity})
                
                # Update performance metrics for sub-strategies
                # Attribute trade to the strategy with the highest weight
                if signal["strategy_weights"]["triple_confluence"] > signal["strategy_weights"]["oracle_update"]:
                    self.update_performance("triple_confluence", {"profit": profit_amount, "result": trade_result})
                else:
                    self.update_performance("oracle_update", {"profit": profit_amount, "result": trade_result})
                
                # Skip ahead to avoid overlapping trades
                i = j
            
            # Calculate performance metrics
            if results["trades"]:
                # Total return
                initial_equity = 10000
                final_equity = results["equity_curve"][-1]["equity"]
                total_return = (final_equity - initial_equity) / initial_equity
                
                # Win rate
                wins = sum(1 for trade in results["trades"] if trade["profit_amount"] > 0)
                total_trades = len(results["trades"])
                win_rate = wins / total_trades if total_trades > 0 else 0
                
                # Profit factor
                gross_profit = sum(trade["profit_amount"] for trade in results["trades"] if trade["profit_amount"] > 0)
                gross_loss = sum(abs(trade["profit_amount"]) for trade in results["trades"] if trade["profit_amount"] < 0)
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
                # Maximum drawdown
                max_equity = initial_equity
                max_drawdown = 0
                
                for point in results["equity_curve"]:
                    equity = point["equity"]
                    max_equity = max(max_equity, equity)
                    drawdown = (max_equity - equity) / max_equity
                    max_drawdown = max(max_drawdown, drawdown)
                
                # Sharpe ratio (assuming risk-free rate of 0)
                if len(results["equity_curve"]) > 1:
                    equity_values = [point["equity"] for point in results["equity_curve"]]
                    returns = [(equity_values[i] - equity_values[i-1]) / equity_values[i-1] for i in range(1, len(equity_values))]
                    avg_return = sum(returns) / len(returns)
                    std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
                    sharpe_ratio = avg_return / std_return if std_return > 0 else 0
                else:
                    sharpe_ratio = 0
                
                # Store metrics
                results["metrics"] = {
                    "total_return": total_return,
                    "win_rate": win_rate,
                    "profit_factor": profit_factor,
                    "max_drawdown": max_drawdown,
                    "sharpe_ratio": sharpe_ratio,
                    "total_trades": total_trades
                }
                
                self.logger.info(f"Backtest completed with {total_trades} trades, {total_return:.2%} return, {win_rate:.2%} win rate")
            else:
                self.logger.warning("No trades executed in backtest")
                results["metrics"] = {
                    "total_return": 0,
                    "win_rate": 0,
                    "profit_factor": 0,
                    "max_drawdown": 0,
                    "sharpe_ratio": 0,
                    "total_trades": 0
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in backtest: {str(e)}")
            return {
                "trades": [],
                "equity_curve": [],
                "metrics": {
                    "error": str(e)
                }
            }
