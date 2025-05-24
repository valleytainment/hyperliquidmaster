"""
Live Simulation with Historical Data Accumulation and Rate Limiting (Fixed)

This module provides a live simulation environment for testing trading strategies
with real market data, historical data accumulation, and rate limiting.
All async/sync interfaces have been properly fixed.
"""

import os
import sys
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np

from strategies.master_omni_overlord_robust import MasterOmniOverlordRobustStrategy
from historical_data_accumulator import HistoricalDataAccumulator
from hyperliquid_api_integration import get_market_data, get_meta
from order_book_handler import get_order_book, analyze_order_book
from strategies.strategy_integration import StrategyIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class LiveSimulationFixed:
    """
    Live simulation environment with historical data accumulation and rate limiting.
    All async/sync interfaces have been properly fixed.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the live simulation.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.strategy = MasterOmniOverlordRobustStrategy(config=self.config)
        self.data_accumulator = HistoricalDataAccumulator()
        
        self.capital = 10000.0  # Initial capital
        self.positions = {}  # Current positions
        self.trades = []  # Trade history
        
        self.running = False
        
    def _load_config(self) -> Dict:
        """
        Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
            
    def _get_market_data(self, symbol: str) -> Dict:
        """
        Get market data for a symbol using rate-limited API client.
        
        Args:
            symbol: Symbol to get market data for
            
        Returns:
            Market data dictionary
        """
        try:
            # Use the rate-limited API client
            market_data = get_market_data(symbol)
            
            if market_data:
                # Add data point to accumulator
                self.data_accumulator.add_data_point(symbol, market_data)
                
            return market_data
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return {}
            
    def _generate_signals(self) -> Dict:
        """
        Generate trading signals for all symbols.
        
        Returns:
            Dictionary of signals by symbol
        """
        try:
            signals = {}
            
            for symbol in self.config.get("symbols", []):
                # Get market data
                market_data = self._get_market_data(symbol)
                
                if not market_data:
                    logger.warning(f"Market data not available for {symbol}")
                    continue
                    
                # Get order book data with robust handler
                order_book = get_order_book(symbol)
                
                # Analyze order book
                order_book_metrics = analyze_order_book(order_book, market_data.get("last_price"))
                
                # Add order book to market data
                market_data["order_book"] = order_book
                market_data["order_book_metrics"] = order_book_metrics
                
                # Get historical data (real + synthetic if needed)
                historical_df = self.data_accumulator.get_synthetic_dataframe(symbol, periods=100)
                
                # Attach historical data to market data for strategy use
                market_data["historical_data"] = historical_df
                
                # Apply adaptive indicators to market data
                market_data = StrategyIntegration.prepare_market_data(market_data)
                
                # Generate signal (synchronous call)
                signal = self.strategy.generate_signal(
                    symbol=symbol,
                    market_data=market_data,
                    order_book=market_data.get("order_book"),
                    positions=self.positions
                )
                
                signals[symbol] = signal
                
            return signals
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return {}
            
    def _execute_signals(self, signals: Dict) -> None:
        """
        Execute trading signals.
        
        Args:
            signals: Dictionary of signals by symbol
        """
        try:
            for symbol, signal in signals.items():
                action = signal.get("action", "none")
                quantity = signal.get("quantity", 0.0)
                entry_price = signal.get("entry_price", 0.0)
                stop_loss = signal.get("stop_loss", 0.0)
                take_profit = signal.get("take_profit", 0.0)
                
                # Skip if no action or zero quantity
                if action == "none" or quantity <= 0:
                    continue
                    
                # Check if we already have a position for this symbol
                current_position = self.positions.get(symbol)
                
                if current_position:
                    # We already have a position, check if we need to close or reverse it
                    current_side = current_position.get("side")
                    
                    if (current_side == "long" and action == "short") or (current_side == "short" and action == "long"):
                        # Close existing position
                        self._close_position(symbol, entry_price)
                        
                        # Open new position
                        self._open_position(symbol, action, quantity, entry_price, stop_loss, take_profit)
                    else:
                        # Same direction, update stop loss and take profit
                        self.positions[symbol]["stop_loss"] = stop_loss
                        self.positions[symbol]["take_profit"] = take_profit
                else:
                    # No existing position, open a new one
                    self._open_position(symbol, action, quantity, entry_price, stop_loss, take_profit)
        except Exception as e:
            logger.error(f"Error executing signals: {str(e)}")
            
    def _open_position(self, symbol: str, side: str, quantity: float, entry_price: float, stop_loss: float, take_profit: float) -> None:
        """
        Open a new position.
        
        Args:
            symbol: Symbol to open position for
            side: Position side ('long' or 'short')
            quantity: Position quantity
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
        """
        try:
            # Calculate position value
            position_value = quantity * entry_price
            
            # Check if we have enough capital
            if position_value > self.capital:
                logger.warning(f"Not enough capital to open position for {symbol}")
                return
                
            # Create position
            position = {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "entry_price": entry_price,
                "entry_time": datetime.now().isoformat(),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "value": position_value
            }
            
            # Add to positions
            self.positions[symbol] = position
            
            # Update capital
            self.capital -= position_value
            
            # Add to trades
            trade = {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": entry_price,
                "time": datetime.now().isoformat(),
                "type": "open"
            }
            
            self.trades.append(trade)
            
            logger.info(f"Opened {side} position for {symbol}: {quantity} @ {entry_price}")
        except Exception as e:
            logger.error(f"Error opening position for {symbol}: {str(e)}")
            
    def _close_position(self, symbol: str, exit_price: float) -> None:
        """
        Close an existing position.
        
        Args:
            symbol: Symbol to close position for
            exit_price: Exit price
        """
        try:
            # Check if we have a position for this symbol
            if symbol not in self.positions:
                logger.warning(f"No position to close for {symbol}")
                return
                
            # Get position
            position = self.positions[symbol]
            
            # Calculate profit/loss
            if position["side"] == "long":
                pnl = (exit_price - position["entry_price"]) * position["quantity"]
            else:
                pnl = (position["entry_price"] - exit_price) * position["quantity"]
                
            # Update capital
            self.capital += position["value"] + pnl
            
            # Add to trades
            trade = {
                "symbol": symbol,
                "side": "sell" if position["side"] == "long" else "buy",
                "quantity": position["quantity"],
                "price": exit_price,
                "time": datetime.now().isoformat(),
                "type": "close",
                "pnl": pnl
            }
            
            self.trades.append(trade)
            
            logger.info(f"Closed {position['side']} position for {symbol}: {position['quantity']} @ {exit_price} (PnL: {pnl:.2f})")
            
            # Remove from positions
            del self.positions[symbol]
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {str(e)}")
            
    def _check_stop_loss_take_profit(self) -> None:
        """
        Check if any positions have hit their stop loss or take profit levels.
        """
        try:
            for symbol, position in list(self.positions.items()):
                # Get current price
                market_data = self._get_market_data(symbol)
                
                if not market_data:
                    continue
                    
                current_price = market_data.get("last_price", 0.0)
                
                if current_price <= 0:
                    continue
                    
                # Check stop loss
                if position["side"] == "long" and current_price <= position["stop_loss"]:
                    logger.info(f"Stop loss triggered for {symbol} long position: {current_price} <= {position['stop_loss']}")
                    self._close_position(symbol, current_price)
                elif position["side"] == "short" and current_price >= position["stop_loss"]:
                    logger.info(f"Stop loss triggered for {symbol} short position: {current_price} >= {position['stop_loss']}")
                    self._close_position(symbol, current_price)
                    
                # Check take profit
                elif position["side"] == "long" and current_price >= position["take_profit"]:
                    logger.info(f"Take profit triggered for {symbol} long position: {current_price} >= {position['take_profit']}")
                    self._close_position(symbol, current_price)
                elif position["side"] == "short" and current_price <= position["take_profit"]:
                    logger.info(f"Take profit triggered for {symbol} short position: {current_price} <= {position['take_profit']}")
                    self._close_position(symbol, current_price)
        except Exception as e:
            logger.error(f"Error checking stop loss/take profit: {str(e)}")
            
    def _print_status(self) -> None:
        """
        Print current status.
        """
        try:
            # Calculate total position value
            total_position_value = sum(position["value"] for position in self.positions.values())
            
            # Calculate total PnL
            total_pnl = sum(trade["pnl"] for trade in self.trades if "pnl" in trade)
            
            # Calculate PnL percentage
            initial_capital = 10000.0
            pnl_pct = (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0.0
            
            logger.info(f"Capital: {self.capital:.2f} ({pnl_pct:.2f}%)")
            logger.info(f"Positions: {len(self.positions)}")
            logger.info(f"Trades: {len(self.trades)}")
        except Exception as e:
            logger.error(f"Error printing status: {str(e)}")
            
    def run(self, duration_seconds: int = 3600) -> None:
        """
        Run the simulation for a specified duration.
        
        Args:
            duration_seconds: Duration in seconds
        """
        try:
            logger.info(f"Starting live simulation for {duration_seconds} seconds")
            
            self.running = True
            start_time = time.time()
            
            while self.running and time.time() - start_time < duration_seconds:
                # Generate signals
                signals = self._generate_signals()
                
                # Execute signals
                self._execute_signals(signals)
                
                # Check stop loss/take profit
                self._check_stop_loss_take_profit()
                
                # Print status
                self._print_status()
                
                # Wait for next update
                logger.info("Waiting 5 seconds for next update...")
                time.sleep(5)
                
            logger.info("Simulation completed")
            
            # Print final status
            self._print_status()
            
            # Print trade history
            logger.info("Trade history:")
            for trade in self.trades:
                logger.info(f"{trade['time']}: {trade['type']} {trade['side']} {trade['symbol']} {trade['quantity']} @ {trade['price']}")
                
            # Calculate performance metrics
            self._calculate_performance_metrics()
        except Exception as e:
            logger.error(f"Error running simulation: {str(e)}")
            
    def _calculate_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        try:
            # Calculate total PnL
            total_pnl = sum(trade["pnl"] for trade in self.trades if "pnl" in trade)
            
            # Calculate PnL percentage
            initial_capital = 10000.0
            pnl_pct = (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0.0
            
            # Calculate win rate
            winning_trades = [trade for trade in self.trades if "pnl" in trade and trade["pnl"] > 0]
            losing_trades = [trade for trade in self.trades if "pnl" in trade and trade["pnl"] <= 0]
            
            total_closed_trades = len(winning_trades) + len(losing_trades)
            win_rate = (len(winning_trades) / total_closed_trades) * 100 if total_closed_trades > 0 else 0.0
            
            # Calculate average win and loss
            avg_win = sum(trade["pnl"] for trade in winning_trades) / len(winning_trades) if winning_trades else 0.0
            avg_loss = sum(trade["pnl"] for trade in losing_trades) / len(losing_trades) if losing_trades else 0.0
            
            # Calculate profit factor
            gross_profit = sum(trade["pnl"] for trade in winning_trades)
            gross_loss = abs(sum(trade["pnl"] for trade in losing_trades))
            
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Calculate max drawdown
            equity_curve = []
            current_equity = initial_capital
            
            for trade in self.trades:
                if "pnl" in trade:
                    current_equity += trade["pnl"]
                    equity_curve.append(current_equity)
                    
            if not equity_curve:
                max_drawdown = 0.0
            else:
                max_equity = initial_capital
                max_drawdown = 0.0
                
                for equity in equity_curve:
                    max_equity = max(max_equity, equity)
                    drawdown = (max_equity - equity) / max_equity * 100
                    max_drawdown = max(max_drawdown, drawdown)
                    
            # Print metrics
            logger.info(f"Performance Metrics:")
            logger.info(f"Total PnL: ${total_pnl:.2f} ({pnl_pct:.2f}%)")
            logger.info(f"Win Rate: {win_rate:.2f}% ({len(winning_trades)}/{total_closed_trades})")
            logger.info(f"Average Win: ${avg_win:.2f}")
            logger.info(f"Average Loss: ${avg_loss:.2f}")
            logger.info(f"Profit Factor: {profit_factor:.2f}")
            logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
            
            return {
                "total_pnl": total_pnl,
                "pnl_pct": pnl_pct,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor,
                "max_drawdown": max_drawdown
            }
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}

if __name__ == "__main__":
    # Run simulation
    simulation = LiveSimulationFixed()
    simulation.run()
