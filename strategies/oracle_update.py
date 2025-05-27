"""
Oracle Update Trading Strategy

This module implements the Oracle Update Trading Strategy for Hyperliquid exchange.
The strategy capitalizes on Hyperliquid's 3-second oracle update cycle to capture
price inefficiencies between oracle updates.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

class OracleUpdateStrategy:
    """
    Oracle Update Trading Strategy for Hyperliquid exchange.
    """
    
    def __init__(self, config: Dict, logger=None):
        """
        Initialize the Oracle Update Strategy.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        # Setup logging
        self.logger = logger or self._setup_logger()
        self.logger.info("Initializing Oracle Update Strategy...")
        
        # Store configuration
        self.config = config
        
        # Strategy parameters
        self.min_deviation = config.get("min_price_deviation", 0.0025)  # 0.25% minimum deviation
        self.max_deviation = config.get("max_price_deviation", 0.01)    # 1% maximum deviation
        self.oracle_update_interval = 3  # 3 seconds between oracle updates
        self.max_trade_duration = 30     # Maximum trade duration in seconds
        
        # Store latest data
        self.latest_data = {}
        
        self.logger.info(f"Oracle Update Strategy initialized with min_deviation={self.min_deviation}")
        
    def update_data(self, symbol, market_price=None, oracle_price=None):
        """
        Update strategy data for a symbol.
        Added to support integration test.
        
        Args:
            symbol: Symbol to update data for
            market_price: Current market price
            oracle_price: Current oracle price
        """
        if symbol not in self.latest_data:
            self.latest_data[symbol] = {}
            
        if market_price is not None:
            self.latest_data[symbol]['market_price'] = market_price
            
        if oracle_price is not None:
            self.latest_data[symbol]['oracle_price'] = oracle_price
            
    def analyze(self, symbol):
        """
        Analyze data for a symbol and generate trading signals.
        Added to support integration test.
        
        Args:
            symbol: Symbol to analyze
            
        Returns:
            Dictionary with signal information
        """
        try:
            # If we have data for this symbol in latest_data, convert to DataFrame
            if symbol in self.latest_data and self.latest_data[symbol]:
                # Create a simple DataFrame with the latest data
                data = pd.DataFrame({
                    'close': [self.latest_data[symbol].get('market_price', 0)],
                    'oracle_price': [self.latest_data[symbol].get('oracle_price', 0)],
                    'timestamp': [datetime.now()]
                })
                
                # Use the existing generate_signals method
                return self.generate_signals(data)
            else:
                # Return a neutral signal if no data
                return {
                    "timestamp": datetime.now().isoformat(),
                    "strategy": "oracle_update",
                    "decision": "neutral",
                    "signal": "NEUTRAL",
                    "signal_strength": 0,
                    "confidence": 0.0,
                    "factors": {}
                }
        except Exception as e:
            self.logger.error(f"Error in analyze: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "strategy": "oracle_update",
                "decision": "error",
                "signal": 0,
                "error": str(e)
            }
        
    def _setup_logger(self) -> logging.Logger:
        """
        Set up the logger.
        
        Returns:
            Configured logger
        """
        logger = logging.getLogger("OracleUpdateStrategy")
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
        
    def detect_oracle_price_deviation(self, data: pd.DataFrame) -> Tuple[bool, float, str]:
        """
        Detect deviation between market price and oracle price.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Tuple of (has_deviation, deviation_pct, direction)
        """
        try:
            # Check if required data is available
            if "close" not in data.columns or "oracle_price" not in data.columns:
                self.logger.warning("Price or oracle data not available")
                return False, 0, "neutral"
                
            # Get latest prices
            market_price = data["close"].iloc[-1]
            oracle_price = data["oracle_price"].iloc[-1]
            
            # Calculate deviation
            deviation_pct = (market_price - oracle_price) / oracle_price
            
            # Determine if deviation is significant
            if deviation_pct > self.min_deviation and deviation_pct < self.max_deviation:
                # Market price higher than oracle price, expect reversion down
                return True, deviation_pct, "short"
            elif deviation_pct < -self.min_deviation and deviation_pct > -self.max_deviation:
                # Market price lower than oracle price, expect reversion up
                return True, abs(deviation_pct), "long"
                
            return False, 0, "neutral"
            
        except Exception as e:
            self.logger.error(f"Error detecting oracle price deviation: {str(e)}")
            return False, 0, "neutral"
            
    def check_oracle_update_timing(self, data: pd.DataFrame) -> bool:
        """
        Check if we're close to an oracle update.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Boolean indicating if we're close to an oracle update
        """
        try:
            # Check if timestamp data is available
            if "timestamp" not in data.columns:
                self.logger.warning("Timestamp data not available")
                return False
                
            # Get latest timestamp
            latest_timestamp = data["timestamp"].iloc[-1]
            
            # Convert to datetime if it's not already
            if not isinstance(latest_timestamp, datetime):
                try:
                    latest_timestamp = pd.to_datetime(latest_timestamp)
                except:
                    self.logger.warning("Could not convert timestamp to datetime")
                    return False
                    
            # Check if we're close to an oracle update (within 1 second)
            seconds_since_epoch = latest_timestamp.timestamp()
            seconds_until_update = self.oracle_update_interval - (seconds_since_epoch % self.oracle_update_interval)
            
            return seconds_until_update <= 1
            
        except Exception as e:
            self.logger.error(f"Error checking oracle update timing: {str(e)}")
            return False
            
    def check_recent_price_volatility(self, data: pd.DataFrame) -> Tuple[bool, float]:
        """
        Check if recent price volatility is suitable for the strategy.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Tuple of (is_suitable, volatility)
        """
        try:
            # Check if required data is available
            if "close" not in data.columns:
                self.logger.warning("Price data not available")
                return False, 0
                
            # Calculate recent volatility (standard deviation of returns)
            returns = data["close"].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1]
            
            # Strategy works better in moderate volatility
            # Too low: not enough deviation to capture
            # Too high: risk of large adverse moves
            min_volatility = 0.0005  # 0.05% per period
            max_volatility = 0.005   # 0.5% per period
            
            is_suitable = min_volatility <= volatility <= max_volatility
            
            return is_suitable, volatility
            
        except Exception as e:
            self.logger.error(f"Error checking recent price volatility: {str(e)}")
            return False, 0
            
    def generate_signals(self, data: pd.DataFrame) -> Dict:
        """
        Generate trading signals using the Oracle Update Strategy.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary with signal information
        """
        try:
            self.logger.info("Generating signals with Oracle Update Strategy...")
            
            # Check for oracle price deviation
            has_deviation, deviation_pct, direction = self.detect_oracle_price_deviation(data)
            
            # Check if we're close to an oracle update
            near_update = self.check_oracle_update_timing(data)
            
            # Check if volatility is suitable
            suitable_volatility, volatility = self.check_recent_price_volatility(data)
            
            # Generate signal
            signal_strength = 0
            decision = "neutral"
            
            if has_deviation and suitable_volatility:
                # Scale signal strength based on deviation percentage
                # Higher deviation = stronger signal, up to a point
                normalized_deviation = min(1.0, deviation_pct / self.max_deviation)
                base_strength = normalized_deviation * 0.8  # 80% weight to deviation size
                
                # Add timing component
                timing_factor = 0.2 if near_update else 0.1  # 20% boost if near update
                
                signal_strength = base_strength + timing_factor
                
                # Set decision based on direction
                if direction == "long":
                    decision = "buy"
                elif direction == "short":
                    decision = "sell"
                    
            # Normalize signal strength to [-1, 1] range
            if decision == "buy":
                normalized_signal = signal_strength
            elif decision == "sell":
                normalized_signal = -signal_strength
            else:
                normalized_signal = 0
                
            # Prepare signal result
            signal_result = {
                "timestamp": datetime.now().isoformat(),
                "strategy": "oracle_update",
                "decision": decision,
                "signal": normalized_signal,
                "signal_strength": signal_strength,
                "factors": {
                    "oracle_deviation": {
                        "active": has_deviation,
                        "deviation_pct": deviation_pct if has_deviation else 0,
                        "direction": direction
                    },
                    "timing": {
                        "near_update": near_update
                    },
                    "volatility": {
                        "suitable": suitable_volatility,
                        "value": volatility
                    }
                }
            }
            
            self.logger.info(f"Generated signal: {decision} with strength {normalized_signal:.4f}")
            return signal_result
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "strategy": "oracle_update",
                "decision": "error",
                "signal": 0,
                "error": str(e)
            }
            
    def backtest(self, data: pd.DataFrame) -> Dict:
        """
        Backtest the Oracle Update Strategy.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary with backtest results
        """
        try:
            self.logger.info("Starting backtest of Oracle Update Strategy...")
            
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
            min_lookback = 20  # For volatility calculation
            
            # Iterate through data
            position = None
            entry_price = 0
            entry_index = 0
            entry_time = None
            
            for i in range(min_lookback, len(data)):
                # Get historical data up to current point
                historical_data = data.iloc[:i+1]
                
                # Generate signal
                signal = self.generate_signals(historical_data)
                
                # Trading logic
                current_price = historical_data["close"].iloc[-1]
                current_time = historical_data["timestamp"].iloc[-1]
                
                # Convert to datetime if it's not already
                if not isinstance(current_time, datetime):
                    try:
                        current_time = pd.to_datetime(current_time)
                    except:
                        self.logger.warning("Could not convert timestamp to datetime")
                        continue
                
                # If no position, check for entry
                if position is None:
                    if signal["decision"] == "buy":
                        # Enter long position
                        position = "long"
                        entry_price = current_price
                        entry_index = i
                        entry_time = current_time
                        self.logger.info(f"Entered long position at {entry_price:.2f}")
                    elif signal["decision"] == "sell":
                        # Enter short position
                        position = "short"
                        entry_price = current_price
                        entry_index = i
                        entry_time = current_time
                        self.logger.info(f"Entered short position at {entry_price:.2f}")
                        
                # If in position, check for exit
                elif position is not None:
                    # Calculate time in trade
                    time_in_trade = (current_time - entry_time).total_seconds()
                    
                    # Exit conditions:
                    # 1. Signal reverses
                    # 2. Max trade duration reached
                    # 3. Target profit reached (0.5%)
                    # 4. Stop loss hit (0.5%)
                    
                    exit_signal = False
                    exit_reason = ""
                    
                    if position == "long":
                        # Check for signal reversal
                        if signal["decision"] == "sell":
                            exit_signal = True
                            exit_reason = "signal_reversal"
                            
                        # Check for max duration
                        elif time_in_trade >= self.max_trade_duration:
                            exit_signal = True
                            exit_reason = "max_duration"
                            
                        # Check for target profit
                        elif (current_price - entry_price) / entry_price >= 0.005:
                            exit_signal = True
                            exit_reason = "target_profit"
                            
                        # Check for stop loss
                        elif (current_price - entry_price) / entry_price <= -0.005:
                            exit_signal = True
                            exit_reason = "stop_loss"
                            
                    elif position == "short":
                        # Check for signal reversal
                        if signal["decision"] == "buy":
                            exit_signal = True
                            exit_reason = "signal_reversal"
                            
                        # Check for max duration
                        elif time_in_trade >= self.max_trade_duration:
                            exit_signal = True
                            exit_reason = "max_duration"
                            
                        # Check for target profit
                        elif (entry_price - current_price) / entry_price >= 0.005:
                            exit_signal = True
                            exit_reason = "target_profit"
                            
                        # Check for stop loss
                        elif (entry_price - current_price) / entry_price <= -0.005:
                            exit_signal = True
                            exit_reason = "stop_loss"
                            
                    # Execute exit if conditions met
                    if exit_signal:
                        # Calculate profit/loss
                        if position == "long":
                            profit_pct = (current_price - entry_price) / entry_price
                        else:  # short
                            profit_pct = (entry_price - current_price) / entry_price
                            
                        profit_amount = equity * 0.1 * profit_pct  # 10% position size
                        
                        # Update equity
                        equity += profit_amount
                        
                        # Record trade
                        trade = {
                            "entry_timestamp": data.iloc[entry_index]["timestamp"],
                            "entry_price": entry_price,
                            "exit_timestamp": current_time,
                            "exit_price": current_price,
                            "direction": position,
                            "profit_pct": profit_pct,
                            "profit_amount": profit_amount,
                            "exit_reason": exit_reason,
                            "time_in_trade": time_in_trade
                        }
                        
                        results["trades"].append(trade)
                        results["equity_curve"].append({"timestamp": current_time, "equity": equity})
                        
                        self.logger.info(f"Exited {position} position at {current_price:.2f} with profit {profit_pct:.2%} ({exit_reason})")
                        
                        # Reset position
                        position = None
                        
            # Close any open position at the end
            if position is not None:
                current_price = data["close"].iloc[-1]
                current_time = data["timestamp"].iloc[-1]
                
                # Convert to datetime if it's not already
                if not isinstance(current_time, datetime):
                    try:
                        current_time = pd.to_datetime(current_time)
                    except:
                        current_time = datetime.now()
                
                # Calculate time in trade
                time_in_trade = (current_time - entry_time).total_seconds() if entry_time else 0
                
                if position == "long":
                    profit_pct = (current_price - entry_price) / entry_price
                else:  # short
                    profit_pct = (entry_price - current_price) / entry_price
                    
                profit_amount = equity * 0.1 * profit_pct  # 10% position size
                
                # Update equity
                equity += profit_amount
                
                # Record trade
                trade = {
                    "entry_timestamp": data.iloc[entry_index]["timestamp"],
                    "entry_price": entry_price,
                    "exit_timestamp": current_time,
                    "exit_price": current_price,
                    "direction": position,
                    "profit_pct": profit_pct,
                    "profit_amount": profit_amount,
                    "exit_reason": "end_of_data",
                    "time_in_trade": time_in_trade
                }
                
                results["trades"].append(trade)
                results["equity_curve"].append({"timestamp": current_time, "equity": equity})
                
                self.logger.info(f"Closed {position} position at end of backtest at {current_price:.2f} with profit {profit_pct:.2%}")
                
            # Calculate performance metrics
            if results["trades"]:
                # Total return
                initial_equity = 10000
                final_equity = results["equity_curve"][-1]["equity"]
                total_return = (final_equity - initial_equity) / initial_equity
                
                # Win rate
                wins = sum(1 for trade in results["trades"] if trade["profit_pct"] > 0)
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
