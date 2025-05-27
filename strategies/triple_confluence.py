"""
Strategy Module - Triple Confluence Strategy

This module implements the Triple Confluence Strategy for Hyperliquid trading.
The strategy looks for confluence of three factors:
1. Funding Rate Edge
2. Order Book Imbalance
3. Technical Triggers (VWMA crossover, hidden divergence, liquidity sweep)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

class TripleConfluenceStrategy:
    """
    Triple Confluence Strategy for Hyperliquid trading.
    """
    
    def __init__(self, config: Dict, logger=None):
        """
        Initialize the Triple Confluence Strategy.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        # Setup logging
        self.logger = logger or self._setup_logger()
        self.logger.info("Initializing Triple Confluence Strategy...")
        
        # Store configuration
        self.config = config
        
        # Strategy parameters
        self.funding_threshold = 0.0001  # 0.01% per hour
        self.order_imbalance_threshold = 1.3  # 30% more on one side
        self.vwma_short_period = 20
        self.vwma_long_period = 50
        self.divergence_lookback = 14
        self.liquidity_threshold = 0.005  # 0.5% from recent high/low
        
        # Store latest data
        self.latest_data = {}
        
        self.logger.info("Triple Confluence Strategy initialized")
        
    def update_data(self, symbol, price=None, volume=None, funding_rate=None, order_book=None):
        """
        Update strategy data for a symbol.
        Added to support integration test.
        
        Args:
            symbol: Symbol to update data for
            price: Current price
            volume: Current volume
            funding_rate: Current funding rate
            order_book: Current order book
        """
        if symbol not in self.latest_data:
            self.latest_data[symbol] = {}
            
        if price is not None:
            self.latest_data[symbol]['price'] = price
            
        if volume is not None:
            self.latest_data[symbol]['volume'] = volume
            
        if funding_rate is not None:
            self.latest_data[symbol]['funding_rate'] = funding_rate
            
        if order_book is not None:
            self.latest_data[symbol]['order_book'] = order_book
            
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
                    'close': [self.latest_data[symbol].get('price', 0)],
                    'volume': [self.latest_data[symbol].get('volume', 0)],
                    'funding_rate': [self.latest_data[symbol].get('funding_rate', 0)]
                })
                
                # Use the existing generate_signals method
                return self.generate_signals(data)
            else:
                # Return a neutral signal if no data
                return {
                    "timestamp": datetime.now().isoformat(),
                    "strategy": "triple_confluence",
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
                "strategy": "triple_confluence",
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
        logger = logging.getLogger("TripleConfluenceStrategy")
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
        
    def check_funding_edge(self, data: pd.DataFrame) -> Tuple[bool, float, str]:
        """
        Check for funding rate edge.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Tuple of (has_edge, edge_strength, direction)
        """
        try:
            # Get latest funding rate
            if "funding_rate" not in data.columns:
                self.logger.warning("Funding rate data not available")
                return False, 0, "neutral"
                
            funding_rate = data["funding_rate"].iloc[-1]
            
            # Determine edge
            if funding_rate > self.funding_threshold:
                # Positive funding rate, short has edge (collecting funding)
                return True, abs(funding_rate) / self.funding_threshold, "short"
            elif funding_rate < -self.funding_threshold:
                # Negative funding rate, long has edge (collecting funding)
                return True, abs(funding_rate) / self.funding_threshold, "long"
            else:
                # No significant edge
                return False, 0, "neutral"
                
        except Exception as e:
            self.logger.error(f"Error checking funding edge: {str(e)}")
            return False, 0, "neutral"
            
    def check_order_imbalance(self, data: pd.DataFrame) -> Tuple[bool, float, str]:
        """
        Check for order book imbalance.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Tuple of (has_imbalance, imbalance_ratio, direction)
        """
        try:
            # Check if order book data is available
            if "bid_volume" not in data.columns or "ask_volume" not in data.columns:
                self.logger.warning("Order book data not available")
                return False, 1.0, "neutral"
                
            # Get latest bid and ask volumes
            bid_volume = data["bid_volume"].iloc[-1]
            ask_volume = data["ask_volume"].iloc[-1]
            
            # Calculate imbalance ratio
            if ask_volume > 0 and bid_volume > 0:
                ratio = bid_volume / ask_volume
                
                if ratio > self.order_imbalance_threshold:
                    # More bids than asks, bullish
                    return True, ratio, "long"
                elif 1 / ratio > self.order_imbalance_threshold:
                    # More asks than bids, bearish
                    return True, 1 / ratio, "short"
                    
            return False, 1.0, "neutral"
            
        except Exception as e:
            self.logger.error(f"Error checking order imbalance: {str(e)}")
            return False, 1.0, "neutral"
            
    def check_vwma_crossover(self, data: pd.DataFrame) -> Tuple[bool, float, str]:
        """
        Check for VWMA (Volume Weighted Moving Average) crossover.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Tuple of (has_crossover, strength, direction)
        """
        try:
            # Check if required data is available
            if "close" not in data.columns or "volume" not in data.columns:
                self.logger.warning("Price or volume data not available")
                return False, 0, "neutral"
                
            # Calculate VWMA
            close = data["close"]
            volume = data["volume"]
            
            # Short-term VWMA
            vwma_short = (close * volume).rolling(self.vwma_short_period).sum() / volume.rolling(self.vwma_short_period).sum()
            
            # Long-term VWMA
            vwma_long = (close * volume).rolling(self.vwma_long_period).sum() / volume.rolling(self.vwma_long_period).sum()
            
            # Check for recent crossover (within last 3 periods)
            crossover_window = 3
            
            # Check for bullish crossover (short crosses above long)
            bullish_crossover = (vwma_short.iloc[-crossover_window-1] <= vwma_long.iloc[-crossover_window-1]) and (vwma_short.iloc[-1] > vwma_long.iloc[-1])
            
            # Check for bearish crossover (short crosses below long)
            bearish_crossover = (vwma_short.iloc[-crossover_window-1] >= vwma_long.iloc[-crossover_window-1]) and (vwma_short.iloc[-1] < vwma_long.iloc[-1])
            
            # Calculate strength as percentage difference between VWMAs
            strength = abs(vwma_short.iloc[-1] - vwma_long.iloc[-1]) / vwma_long.iloc[-1]
            
            if bullish_crossover:
                return True, strength * 100, "long"
            elif bearish_crossover:
                return True, strength * 100, "short"
                
            return False, 0, "neutral"
            
        except Exception as e:
            self.logger.error(f"Error checking VWMA crossover: {str(e)}")
            return False, 0, "neutral"
            
    def check_hidden_divergence(self, data: pd.DataFrame) -> Tuple[bool, float, str]:
        """
        Check for hidden divergence in RSI.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Tuple of (has_divergence, strength, direction)
        """
        try:
            # Check if required data is available
            if "close" not in data.columns:
                self.logger.warning("Price data not available")
                return False, 0, "neutral"
                
            # Calculate RSI
            close = data["close"]
            delta = close.diff()
            
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Look for hidden bullish divergence
            # Price making lower lows but RSI making higher lows
            lookback = self.divergence_lookback
            
            # Find recent price lows
            price_window = close.iloc[-lookback:]
            rsi_window = rsi.iloc[-lookback:]
            
            # Simple approach: check if price trend is down but RSI trend is up
            price_trend = price_window.iloc[-1] < price_window.iloc[0]
            rsi_trend = rsi_window.iloc[-1] > rsi_window.iloc[0]
            
            bullish_divergence = price_trend and not rsi_trend
            bearish_divergence = not price_trend and rsi_trend
            
            # Calculate strength based on the divergence magnitude
            if bullish_divergence or bearish_divergence:
                price_change = abs(price_window.iloc[-1] - price_window.iloc[0]) / price_window.iloc[0]
                rsi_change = abs(rsi_window.iloc[-1] - rsi_window.iloc[0]) / 100
                
                strength = (price_change + rsi_change) / 2 * 100
                
                if bullish_divergence:
                    return True, strength, "long"
                else:
                    return True, strength, "short"
                    
            return False, 0, "neutral"
            
        except Exception as e:
            self.logger.error(f"Error checking hidden divergence: {str(e)}")
            return False, 0, "neutral"
            
    def check_liquidity_sweep(self, data: pd.DataFrame) -> Tuple[bool, float, str]:
        """
        Check for liquidity sweep (price briefly breaking support/resistance then reversing).
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Tuple of (has_sweep, strength, direction)
        """
        try:
            # Check if required data is available
            if "high" not in data.columns or "low" not in data.columns or "close" not in data.columns:
                self.logger.warning("Price data not available")
                return False, 0, "neutral"
                
            # Get recent price data
            high = data["high"]
            low = data["low"]
            close = data["close"]
            
            # Find recent high and low (excluding last 3 periods)
            lookback = 20
            recent_high = high.iloc[-lookback:-3].max()
            recent_low = low.iloc[-lookback:-3].min()
            
            # Check if price briefly broke above recent high then closed below
            broke_high = high.iloc[-3:].max() > recent_high
            closed_below_high = close.iloc[-1] < recent_high
            
            # Check if price briefly broke below recent low then closed above
            broke_low = low.iloc[-3:].min() < recent_low
            closed_above_low = close.iloc[-1] > recent_low
            
            # Calculate distance from recent high/low as percentage
            high_distance = abs(high.iloc[-3:].max() - recent_high) / recent_high
            low_distance = abs(low.iloc[-3:].min() - recent_low) / recent_low
            
            # Check for bearish liquidity sweep (broke above high then reversed)
            if broke_high and closed_below_high and high_distance > self.liquidity_threshold:
                return True, high_distance * 100, "short"
                
            # Check for bullish liquidity sweep (broke below low then reversed)
            if broke_low and closed_above_low and low_distance > self.liquidity_threshold:
                return True, low_distance * 100, "long"
                
            return False, 0, "neutral"
            
        except Exception as e:
            self.logger.error(f"Error checking liquidity sweep: {str(e)}")
            return False, 0, "neutral"
            
    def generate_signals(self, data: pd.DataFrame) -> Dict:
        """
        Generate trading signals using the Triple Confluence Strategy.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary with signal information
        """
        try:
            self.logger.info("Generating signals with Triple Confluence Strategy...")
            
            # Check all three factors
            funding_edge = self.check_funding_edge(data)
            order_imbalance = self.check_order_imbalance(data)
            
            # Check technical triggers
            vwma_crossover = self.check_vwma_crossover(data)
            hidden_divergence = self.check_hidden_divergence(data)
            liquidity_sweep = self.check_liquidity_sweep(data)
            
            # Combine technical triggers (use the strongest one)
            technical_triggers = [vwma_crossover, hidden_divergence, liquidity_sweep]
            technical_triggers = [t for t in technical_triggers if t[0]]  # Filter for valid triggers
            
            if technical_triggers:
                # Sort by strength and take the strongest
                technical_triggers.sort(key=lambda x: x[1], reverse=True)
                technical_trigger = technical_triggers[0]
            else:
                technical_trigger = (False, 0, "neutral")
                
            # Check for confluence
            has_funding_edge, funding_strength, funding_direction = funding_edge
            has_imbalance, imbalance_ratio, imbalance_direction = order_imbalance
            has_trigger, trigger_strength, trigger_direction = technical_trigger
            
            # Calculate overall signal
            # For a valid signal, we need at least 2 out of 3 factors in the same direction
            directions = [d for _, _, d in [funding_edge, order_imbalance, technical_trigger] if _ and d != "neutral"]
            
            if len(directions) >= 2:
                # Count occurrences of each direction
                long_count = directions.count("long")
                short_count = directions.count("short")
                
                if long_count > short_count:
                    decision = "buy"
                    signal_strength = (long_count / len(directions)) * (
                        (funding_strength if funding_direction == "long" else 0) +
                        (imbalance_ratio if imbalance_direction == "long" else 0) +
                        (trigger_strength if trigger_direction == "long" else 0)
                    ) / 3
                elif short_count > long_count:
                    decision = "sell"
                    signal_strength = (short_count / len(directions)) * (
                        (funding_strength if funding_direction == "short" else 0) +
                        (imbalance_ratio if imbalance_direction == "short" else 0) +
                        (trigger_strength if trigger_direction == "short" else 0)
                    ) / 3
                else:
                    decision = "neutral"
                    signal_strength = 0
            else:
                decision = "neutral"
                signal_strength = 0
                
            # Normalize signal strength to [-1, 1] range
            if decision == "buy":
                normalized_signal = min(1.0, signal_strength / 10)
            elif decision == "sell":
                normalized_signal = -min(1.0, signal_strength / 10)
            else:
                normalized_signal = 0
                
            # Prepare signal result
            signal_result = {
                "timestamp": datetime.now().isoformat(),
                "strategy": "triple_confluence",
                "decision": decision,
                "signal": normalized_signal,
                "signal_strength": signal_strength,
                "confidence": min(1.0, signal_strength / 5.0),  # Add confidence field for integration test
                "factors": {
                    "funding_edge": {
                        "active": has_funding_edge,
                        "strength": funding_strength,
                        "direction": funding_direction
                    },
                    "order_imbalance": {
                        "active": has_imbalance,
                        "strength": imbalance_ratio,
                        "direction": imbalance_direction
                    },
                    "technical_trigger": {
                        "active": has_trigger,
                        "strength": trigger_strength,
                        "direction": trigger_direction,
                        "type": next((t for t in ["vwma_crossover", "hidden_divergence", "liquidity_sweep"] 
                                     if [vwma_crossover, hidden_divergence, liquidity_sweep][["vwma_crossover", "hidden_divergence", "liquidity_sweep"].index(t)][0]), None)
                    }
                }
            }
            
            self.logger.info(f"Generated signal: {decision} with strength {normalized_signal:.4f}")
            return signal_result
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "strategy": "triple_confluence",
                "decision": "error",
                "signal": 0,
                "error": str(e)
            }
            
    def backtest(self, data: pd.DataFrame) -> Dict:
        """
        Backtest the Triple Confluence Strategy.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary with backtest results
        """
        try:
            self.logger.info("Starting backtest of Triple Confluence Strategy...")
            
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
            min_lookback = max(self.vwma_long_period, self.divergence_lookback) + 5
            
            # Iterate through data
            position = None
            entry_price = 0
            entry_index = 0
            
            for i in range(min_lookback, len(data)):
                # Get historical data up to current point
                historical_data = data.iloc[:i+1]
                
                # Generate signal
                signal = self.generate_signals(historical_data)
                
                # Trading logic
                current_price = historical_data["close"].iloc[-1]
                
                # If no position, check for entry
                if position is None:
                    if signal["decision"] == "buy":
                        # Enter long position
                        position = "long"
                        entry_price = current_price
                        entry_index = i
                        self.logger.info(f"Entered long position at {entry_price:.2f}")
                    elif signal["decision"] == "sell":
                        # Enter short position
                        position = "short"
                        entry_price = current_price
                        entry_index = i
                        self.logger.info(f"Entered short position at {entry_price:.2f}")
                        
                # If in position, check for exit
                elif position == "long":
                    # Exit if signal turns bearish or we've been in the trade for too long
                    if signal["decision"] == "sell" or (i - entry_index) > 20:
                        # Calculate profit/loss
                        profit_pct = (current_price - entry_price) / entry_price
                        profit_amount = equity * 0.1 * profit_pct  # 10% position size
                        
                        # Update equity
                        equity += profit_amount
                        
                        # Record trade
                        trade = {
                            "entry_timestamp": data.iloc[entry_index]["timestamp"],
                            "entry_price": entry_price,
                            "exit_timestamp": data.iloc[i]["timestamp"],
                            "exit_price": current_price,
                            "direction": position,
                            "profit_pct": profit_pct,
                            "profit_amount": profit_amount
                        }
                        
                        results["trades"].append(trade)
                        results["equity_curve"].append({"timestamp": data.iloc[i]["timestamp"], "equity": equity})
                        
                        self.logger.info(f"Exited long position at {current_price:.2f} with profit {profit_pct:.2%}")
                        
                        # Reset position
                        position = None
                        
                elif position == "short":
                    # Exit if signal turns bullish or we've been in the trade for too long
                    if signal["decision"] == "buy" or (i - entry_index) > 20:
                        # Calculate profit/loss
                        profit_pct = (entry_price - current_price) / entry_price
                        profit_amount = equity * 0.1 * profit_pct  # 10% position size
                        
                        # Update equity
                        equity += profit_amount
                        
                        # Record trade
                        trade = {
                            "entry_timestamp": data.iloc[entry_index]["timestamp"],
                            "entry_price": entry_price,
                            "exit_timestamp": data.iloc[i]["timestamp"],
                            "exit_price": current_price,
                            "direction": position,
                            "profit_pct": profit_pct,
                            "profit_amount": profit_amount
                        }
                        
                        results["trades"].append(trade)
                        results["equity_curve"].append({"timestamp": data.iloc[i]["timestamp"], "equity": equity})
                        
                        self.logger.info(f"Exited short position at {current_price:.2f} with profit {profit_pct:.2%}")
                        
                        # Reset position
                        position = None
                        
            # Close any open position at the end
            if position is not None:
                current_price = data["close"].iloc[-1]
                
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
                    "exit_timestamp": data.iloc[-1]["timestamp"],
                    "exit_price": current_price,
                    "direction": position,
                    "profit_pct": profit_pct,
                    "profit_amount": profit_amount
                }
                
                results["trades"].append(trade)
                results["equity_curve"].append({"timestamp": data.iloc[-1]["timestamp"], "equity": equity})
                
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
