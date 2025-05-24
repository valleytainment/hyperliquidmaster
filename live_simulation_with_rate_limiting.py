"""
Live Simulation with Historical Data Accumulation and Rate Limiting

This module provides a live simulation environment for testing trading strategies
with real market data, historical data accumulation, and API rate limiting.
"""

import os
import sys
import json
import logging
import asyncio
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

class LiveSimulationWithRateLimiting:
    """
    Live simulation environment with historical data accumulation and rate limiting.
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
            
    async def _get_market_data(self, symbol: str) -> Dict:
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
            
    async def _generate_signals(self) -> Dict:
        """
        Generate trading signals for all symbols.
        
        Returns:
            Dictionary of signals by symbol
        """
        try:
            signals = {}
            
            for symbol in self.config.get("symbols", []):
                      # Get market data
                market_data = await get_market_data(symbol)
                
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
                
                # Get historical data (real + synthetic if needed)ynthetic if needed)
                historical_df = self.data_accumulator.get_synthetic_dataframe(symbol, periods=100)
                
                # Attach historical data to market data for strategy use
                market_data["historical_data"] = historical_df
                
                # Apply adaptive indicators to market data
                market_data = StrategyIntegration.prepare_market_data(market_data)
                
                # Generate signal
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
                
                if action == "none":
                    continue
                    
                quantity = signal.get("quantity", 0.0)
                entry_price = signal.get("entry_price", 0.0)
                stop_loss = signal.get("stop_loss", 0.0)
                take_profit = signal.get("take_profit", 0.0)
                
                if quantity <= 0 or entry_price <= 0:
                    logger.warning(f"Invalid quantity or entry price for {symbol}")
                    continue
                    
                # Check if we already have a position
                if symbol in self.positions:
                    existing_position = self.positions[symbol]
                    existing_side = existing_position.get("side", "")
                    
                    # Close existing position if signal is in opposite direction
                    if (existing_side == "long" and action == "short") or (existing_side == "short" and action == "long"):
                        # Calculate profit/loss
                        entry_value = existing_position.get("entry_price", 0.0) * existing_position.get("quantity", 0.0)
                        exit_value = entry_price * existing_position.get("quantity", 0.0)
                        
                        if existing_side == "long":
                            profit = exit_value - entry_value
                        else:
                            profit = entry_value - exit_value
                            
                        # Update capital
                        self.capital += profit
                        
                        # Record trade
                        trade = {
                            "symbol": symbol,
                            "side": existing_side,
                            "quantity": existing_position.get("quantity", 0.0),
                            "entry_price": existing_position.get("entry_price", 0.0),
                            "exit_price": entry_price,
                            "entry_time": existing_position.get("entry_time", ""),
                            "exit_time": datetime.now().isoformat(),
                            "profit": profit,
                            "profit_percent": (profit / entry_value) * 100 if entry_value > 0 else 0.0
                        }
                        
                        self.trades.append(trade)
                        
                        logger.info(f"Closed {existing_side} position for {symbol} with profit: {profit:.2f} ({trade['profit_percent']:.2f}%)")
                        
                        # Remove position
                        del self.positions[symbol]
                
                # Open new position
                position_value = entry_price * quantity
                
                if position_value > self.capital:
                    logger.warning(f"Insufficient capital for {symbol} position: {position_value:.2f} > {self.capital:.2f}")
                    continue
                    
                # Create position
                position = {
                    "symbol": symbol,
                    "side": "long" if action == "long" else "short",
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "entry_time": datetime.now().isoformat()
                }
                
                self.positions[symbol] = position
                
                logger.info(f"Opened {position['side']} position for {symbol}: {quantity} @ {entry_price:.2f}")
        except Exception as e:
            logger.error(f"Error executing signals: {str(e)}")
            
    def _update_positions(self, market_data: Dict) -> None:
        """
        Update positions with current market data.
        
        Args:
            market_data: Dictionary of market data by symbol
        """
        try:
            for symbol, position in list(self.positions.items()):
                if symbol not in market_data:
                    continue
                    
                current_price = market_data[symbol].get("last_price", 0.0)
                
                if current_price <= 0:
                    continue
                    
                side = position.get("side", "")
                entry_price = position.get("entry_price", 0.0)
                quantity = position.get("quantity", 0.0)
                stop_loss = position.get("stop_loss", 0.0)
                take_profit = position.get("take_profit", 0.0)
                
                # Calculate current value
                entry_value = entry_price * quantity
                current_value = current_price * quantity
                
                # Calculate profit/loss
                if side == "long":
                    profit = current_value - entry_value
                    
                    # Check stop loss
                    if stop_loss > 0 and current_price <= stop_loss:
                        logger.info(f"Stop loss triggered for {symbol} long position: {current_price:.2f} <= {stop_loss:.2f}")
                        self._close_position(symbol, current_price, "stop_loss")
                        continue
                        
                    # Check take profit
                    if take_profit > 0 and current_price >= take_profit:
                        logger.info(f"Take profit triggered for {symbol} long position: {current_price:.2f} >= {take_profit:.2f}")
                        self._close_position(symbol, current_price, "take_profit")
                        continue
                else:
                    profit = entry_value - current_value
                    
                    # Check stop loss
                    if stop_loss > 0 and current_price >= stop_loss:
                        logger.info(f"Stop loss triggered for {symbol} short position: {current_price:.2f} >= {stop_loss:.2f}")
                        self._close_position(symbol, current_price, "stop_loss")
                        continue
                        
                    # Check take profit
                    if take_profit > 0 and current_price <= take_profit:
                        logger.info(f"Take profit triggered for {symbol} short position: {current_price:.2f} <= {take_profit:.2f}")
                        self._close_position(symbol, current_price, "take_profit")
                        continue
                        
                # Update position with current profit
                position["current_price"] = current_price
                position["current_value"] = current_value
                position["profit"] = profit
                position["profit_percent"] = (profit / entry_value) * 100 if entry_value > 0 else 0.0
        except Exception as e:
            logger.error(f"Error updating positions: {str(e)}")
            
    def _close_position(self, symbol: str, price: float, reason: str) -> None:
        """
        Close a position.
        
        Args:
            symbol: Symbol to close position for
            price: Price to close at
            reason: Reason for closing
        """
        try:
            if symbol not in self.positions:
                return
                
            position = self.positions[symbol]
            side = position.get("side", "")
            entry_price = position.get("entry_price", 0.0)
            quantity = position.get("quantity", 0.0)
            
            # Calculate profit/loss
            entry_value = entry_price * quantity
            exit_value = price * quantity
            
            if side == "long":
                profit = exit_value - entry_value
            else:
                profit = entry_value - exit_value
                
            # Update capital
            self.capital += profit
            
            # Record trade
            trade = {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "entry_price": entry_price,
                "exit_price": price,
                "entry_time": position.get("entry_time", ""),
                "exit_time": datetime.now().isoformat(),
                "profit": profit,
                "profit_percent": (profit / entry_value) * 100 if entry_value > 0 else 0.0,
                "reason": reason
            }
            
            self.trades.append(trade)
            
            logger.info(f"Closed {side} position for {symbol} with profit: {profit:.2f} ({trade['profit_percent']:.2f}%) - Reason: {reason}")
            
            # Remove position
            del self.positions[symbol]
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {str(e)}")
            
    def _print_status(self) -> None:
        """
        Print current status.
        """
        try:
            # Calculate total profit
            total_profit = sum(trade.get("profit", 0.0) for trade in self.trades)
            profit_percent = (total_profit / 10000.0) * 100  # Relative to initial capital
            
            logger.info(f"Capital: {self.capital:.2f} ({profit_percent:.2f}%)")
            logger.info(f"Positions: {len(self.positions)}")
            logger.info(f"Trades: {len(self.trades)}")
            
            # Print positions
            for symbol, position in self.positions.items():
                logger.info(f"Position: {symbol} {position.get('side', '')} {position.get('quantity', 0.0)} @ {position.get('entry_price', 0.0):.2f} - Profit: {position.get('profit', 0.0):.2f} ({position.get('profit_percent', 0.0):.2f}%)")
        except Exception as e:
            logger.error(f"Error printing status: {str(e)}")
            
    async def run(self, duration_seconds: int = 3600) -> None:
        """
        Run the simulation.
        
        Args:
            duration_seconds: Duration in seconds
        """
        try:
            logger.info(f"Starting live simulation for {duration_seconds} seconds")
            
            self.running = True
            start_time = time.time()
            
            while self.running and time.time() - start_time < duration_seconds:
                # Get market data for all symbols
                market_data = {}
                
                for symbol in self.config.get("symbols", []):
                    data = await self._get_market_data(symbol)
                    
                    if data:
                        market_data[symbol] = data
                        
                # Update positions
                self._update_positions(market_data)
                
                # Generate signals
                signals = await self._generate_signals()
                
                # Execute signals
                self._execute_signals(signals)
                
                # Print status
                self._print_status()
                
                # Wait for next update - use a longer interval to avoid rate limits
                update_interval = self.config.get("data_update_interval", 5)
                logger.info(f"Waiting {update_interval} seconds for next update...")
                await asyncio.sleep(update_interval)
                
            logger.info("Simulation completed")
            
            # Print final results
            logger.info("Final results:")
            self._print_status()
            
            # Print trade statistics
            if self.trades:
                win_count = sum(1 for trade in self.trades if trade.get("profit", 0.0) > 0)
                loss_count = sum(1 for trade in self.trades if trade.get("profit", 0.0) <= 0)
                win_rate = win_count / len(self.trades) if self.trades else 0.0
                
                total_profit = sum(trade.get("profit", 0.0) for trade in self.trades if trade.get("profit", 0.0) > 0)
                total_loss = sum(abs(trade.get("profit", 0.0)) for trade in self.trades if trade.get("profit", 0.0) < 0)
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
                
                logger.info(f"Win rate: {win_rate:.2f} ({win_count}/{len(self.trades)})")
                logger.info(f"Profit factor: {profit_factor:.2f}")
                logger.info(f"Average win: {total_profit / win_count:.2f}" if win_count > 0 else "Average win: N/A")
                logger.info(f"Average loss: {total_loss / loss_count:.2f}" if loss_count > 0 else "Average loss: N/A")
                
                # Save trade history to file
                self._save_trade_history()
        except Exception as e:
            logger.error(f"Error running simulation: {str(e)}")
        finally:
            self.running = False
            
    def _save_trade_history(self) -> None:
        """
        Save trade history to file.
        """
        try:
            if not self.trades:
                return
                
            # Create trades directory if it doesn't exist
            os.makedirs("trades", exist_ok=True)
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"trades/trade_history_{timestamp}.json"
            
            with open(file_path, "w") as f:
                json.dump(self.trades, f, indent=2)
                
            logger.info(f"Saved trade history to {file_path}")
        except Exception as e:
            logger.error(f"Error saving trade history: {str(e)}")
            
    def stop(self) -> None:
        """
        Stop the simulation.
        """
        self.running = False
        logger.info("Stopping simulation")

async def main():
    """
    Main function.
    """
    try:
        # Create simulation
        simulation = LiveSimulationWithRateLimiting()
        
        # Run simulation for 1 hour
        await simulation.run(duration_seconds=3600)
    except KeyboardInterrupt:
        logger.info("Simulation interrupted")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
