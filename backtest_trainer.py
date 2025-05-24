"""
Backtesting and Training Module

This module provides comprehensive backtesting and training capabilities
for the enhanced trading bot, allowing for strategy optimization and validation.
"""

import asyncio
import json
import logging
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import requests
from tqdm import tqdm

# Import core components
from strategies.triple_confluence import TripleConfluenceStrategy
from strategies.oracle_update import OracleUpdateStrategy
from config_compatibility import ConfigManager

class BacktestTrainer:
    """
    Backtesting and training system for the enhanced trading bot.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the backtest trainer.
        
        Args:
            config_path: Path to the configuration file
        """
        # Setup logging
        self.logger = self._setup_logger()
        self.logger.info("Initializing Backtest Trainer...")
        
        # Load configuration
        self.config_manager = ConfigManager(config_path, self.logger)
        self.config = self.config_manager.get_config()
        
        # Initialize strategies
        self.triple_confluence_strategy = TripleConfluenceStrategy(
            self.config,
            self.logger
        )
        
        self.oracle_update_strategy = OracleUpdateStrategy(
            self.config,
            self.logger
        )
        
        # Data storage
        self.historical_data = {}
        self.backtest_results = {}
        self.optimization_results = {}
        
        # Create output directories
        os.makedirs("backtest_results", exist_ok=True)
        os.makedirs("backtest_results/charts", exist_ok=True)
        os.makedirs("backtest_results/data", exist_ok=True)
        
    def _setup_logger(self) -> logging.Logger:
        """
        Set up the logger.
        
        Returns:
            Configured logger
        """
        logger = logging.getLogger("BacktestTrainer")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("backtest_training.log", mode="a")
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
        
    async def fetch_historical_data(self, symbol: str, days: int = 30, interval: str = "1h"):
        """
        Fetch historical data for backtesting.
        
        Args:
            symbol: Trading symbol
            days: Number of days of historical data
            interval: Data interval (1m, 5m, 15m, 1h, 4h, 1d)
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Fetching {days} days of {interval} historical data for {symbol}...")
        
        try:
            # Calculate start and end timestamps
            end_time = int(time.time())
            start_time = end_time - (days * 24 * 60 * 60)
            
            # Convert symbol format if needed (e.g., BTC-USD-PERP to BTC)
            base_symbol = symbol.split("-")[0] if "-" in symbol else symbol
            
            # Use CoinGecko API for historical data
            url = f"https://api.coingecko.com/api/v3/coins/{base_symbol.lower()}/market_chart/range"
            params = {
                "vs_currency": "usd",
                "from": start_time,
                "to": end_time
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                self.logger.error(f"Error fetching historical data: {response.status_code} {response.text}")
                return False
                
            data = response.json()
            
            # Process price data
            prices = data.get("prices", [])
            volumes = data.get("total_volumes", [])
            
            if not prices or not volumes:
                self.logger.error("No price or volume data returned")
                return False
                
            # Convert to DataFrame
            price_df = pd.DataFrame(prices, columns=["timestamp", "price"])
            volume_df = pd.DataFrame(volumes, columns=["timestamp", "volume"])
            
            # Merge dataframes
            price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], unit="ms")
            volume_df["timestamp"] = pd.to_datetime(volume_df["timestamp"], unit="ms")
            
            df = pd.merge(price_df, volume_df, on="timestamp")
            
            # Resample to desired interval
            df = df.set_index("timestamp")
            
            if interval == "1m":
                df = df.resample("1min").last().ffill()
            elif interval == "5m":
                df = df.resample("5min").last().ffill()
            elif interval == "15m":
                df = df.resample("15min").last().ffill()
            elif interval == "1h":
                df = df.resample("1H").last().ffill()
            elif interval == "4h":
                df = df.resample("4H").last().ffill()
            elif interval == "1d":
                df = df.resample("1D").last().ffill()
                
            # Reset index
            df = df.reset_index()
            
            # Add additional columns for backtesting
            df["open"] = df["price"].shift(1)
            df["high"] = df["price"] * (1 + np.random.uniform(0, 0.005, len(df)))  # Simulate high prices
            df["low"] = df["price"] * (1 - np.random.uniform(0, 0.005, len(df)))   # Simulate low prices
            df["close"] = df["price"]
            
            # Add funding rate (simulated)
            df["funding_rate"] = np.sin(np.linspace(0, 10, len(df))) * 0.0001
            
            # Add oracle price (simulated)
            df["oracle_price"] = df["price"] * (1 + np.sin(np.linspace(0, 20, len(df))) * 0.001)
            
            # Fill NaN values
            df = df.fillna(method="ffill")
            
            # Store historical data
            self.historical_data[symbol] = df
            
            # Save to CSV
            df.to_csv(f"backtest_results/data/{symbol}_{interval}_{days}d.csv", index=False)
            
            self.logger.info(f"Successfully fetched {len(df)} data points for {symbol}")
            return True
            
        except Exception as e:
            self.logger.exception(f"Error fetching historical data: {str(e)}")
            return False
            
    def generate_simulated_data(self, symbol: str, days: int = 30, interval: str = "1h"):
        """
        Generate simulated data for backtesting when API data is unavailable.
        
        Args:
            symbol: Trading symbol
            days: Number of days of simulated data
            interval: Data interval (1m, 5m, 15m, 1h, 4h, 1d)
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Generating {days} days of simulated {interval} data for {symbol}...")
        
        try:
            # Determine number of data points
            points_per_day = {
                "1m": 1440,
                "5m": 288,
                "15m": 96,
                "1h": 24,
                "4h": 6,
                "1d": 1
            }
            
            points = days * points_per_day.get(interval, 24)
            
            # Generate timestamps
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            if interval == "1m":
                timestamps = pd.date_range(start=start_time, end=end_time, freq="1min")
            elif interval == "5m":
                timestamps = pd.date_range(start=start_time, end=end_time, freq="5min")
            elif interval == "15m":
                timestamps = pd.date_range(start=start_time, end=end_time, freq="15min")
            elif interval == "1h":
                timestamps = pd.date_range(start=start_time, end=end_time, freq="1H")
            elif interval == "4h":
                timestamps = pd.date_range(start=start_time, end=end_time, freq="4H")
            elif interval == "1d":
                timestamps = pd.date_range(start=start_time, end=end_time, freq="1D")
                
            # Generate price data with realistic patterns
            base_price = 50000 if symbol.startswith("BTC") else (
                3000 if symbol.startswith("ETH") else (
                100 if symbol.startswith("SOL") else 1.0
            ))
            
            # Generate price with trend, cycles, and noise
            trend = np.linspace(0, 0.2, len(timestamps))  # Upward trend
            cycles = 0.1 * np.sin(np.linspace(0, 15, len(timestamps)))  # Cycles
            noise = np.random.normal(0, 0.02, len(timestamps))  # Random noise
            
            price_changes = trend + cycles + noise
            prices = base_price * np.cumprod(1 + price_changes)
            
            # Generate volume with correlation to price changes
            volume_base = base_price * 10
            volumes = volume_base * (1 + np.abs(price_changes) * 5 + np.random.normal(0, 0.5, len(timestamps)))
            
            # Create DataFrame
            df = pd.DataFrame({
                "timestamp": timestamps,
                "price": prices,
                "volume": volumes
            })
            
            # Add OHLC data
            df["open"] = df["price"].shift(1)
            df["high"] = df["price"] * (1 + np.random.uniform(0, 0.005, len(df)))
            df["low"] = df["price"] * (1 - np.random.uniform(0, 0.005, len(df)))
            df["close"] = df["price"]
            
            # Add funding rate (simulated)
            df["funding_rate"] = np.sin(np.linspace(0, 10, len(df))) * 0.0001
            
            # Add oracle price (simulated)
            df["oracle_price"] = df["price"] * (1 + np.sin(np.linspace(0, 20, len(df))) * 0.001)
            
            # Fill NaN values
            df = df.fillna(method="ffill")
            
            # Store historical data
            self.historical_data[symbol] = df
            
            # Save to CSV
            df.to_csv(f"backtest_results/data/{symbol}_{interval}_{days}d_simulated.csv", index=False)
            
            self.logger.info(f"Successfully generated {len(df)} simulated data points for {symbol}")
            return True
            
        except Exception as e:
            self.logger.exception(f"Error generating simulated data: {str(e)}")
            return False
            
    def backtest_triple_confluence(self, symbol: str, optimize: bool = False):
        """
        Backtest the Triple Confluence strategy.
        
        Args:
            symbol: Trading symbol
            optimize: Whether to optimize strategy parameters
            
        Returns:
            Backtest results dictionary
        """
        self.logger.info(f"Backtesting Triple Confluence strategy for {symbol}...")
        
        if symbol not in self.historical_data:
            self.logger.error(f"No historical data available for {symbol}")
            return None
            
        try:
            # Get historical data
            df = self.historical_data[symbol].copy()
            
            # Initialize strategy
            strategy = TripleConfluenceStrategy(
                self.config,
                self.logger
            )
            
            # Initialize results
            trades = []
            positions = []
            current_position = None
            equity_curve = [1.0]  # Start with 1.0 (100%)
            
            # Backtest parameters
            initial_equity = 10000.0
            risk_per_trade = self.config.get("risk_percent", 0.01)
            commission_rate = self.config.get("taker_fee", 0.00042)
            
            # Optimization parameters
            if optimize:
                best_params = self._optimize_triple_confluence(symbol)
                
                if best_params:
                    self.logger.info(f"Using optimized parameters: {best_params}")
                    
                    # Update strategy with optimized parameters
                    strategy = TripleConfluenceStrategy(
                        {**self.config, **best_params},
                        self.logger
                    )
                    
            # Run backtest
            for i in tqdm(range(100, len(df)), desc="Backtesting"):
                # Get current data
                current_data = df.iloc[i]
                price = current_data["price"]
                volume = current_data["volume"]
                funding_rate = current_data["funding_rate"]
                
                # Create mock order book
                mock_order_book = {
                    "bids": [[price * (1 - j * 0.001), 10 + j] for j in range(10)],
                    "asks": [[price * (1 + j * 0.001), 5 + j] for j in range(10)]
                }
                
                # Update strategy data
                strategy.update_data(
                    symbol,
                    price,
                    volume,
                    funding_rate,
                    mock_order_book
                )
                
                # Get signal
                signal = strategy.analyze(symbol)
                
                # Process signal
                if current_position is None:
                    # No position, check for entry
                    if signal["signal"] == "LONG" and signal["confidence"] > 0.7:
                        # Enter long position
                        position_size = initial_equity * risk_per_trade / (price * 0.01)  # 1% stop loss
                        entry_price = price
                        stop_loss = price * 0.99  # 1% stop loss
                        take_profit = price * 1.03  # 3% take profit
                        
                        current_position = {
                            "type": "LONG",
                            "entry_price": entry_price,
                            "entry_time": current_data["timestamp"],
                            "size": position_size,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "current_price": price
                        }
                        
                        trades.append({
                            "type": "ENTRY",
                            "position": "LONG",
                            "time": current_data["timestamp"],
                            "price": entry_price,
                            "size": position_size,
                            "reason": signal["reason"]
                        })
                        
                    elif signal["signal"] == "SHORT" and signal["confidence"] > 0.7:
                        # Enter short position
                        position_size = initial_equity * risk_per_trade / (price * 0.01)  # 1% stop loss
                        entry_price = price
                        stop_loss = price * 1.01  # 1% stop loss
                        take_profit = price * 0.97  # 3% take profit
                        
                        current_position = {
                            "type": "SHORT",
                            "entry_price": entry_price,
                            "entry_time": current_data["timestamp"],
                            "size": position_size,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "current_price": price
                        }
                        
                        trades.append({
                            "type": "ENTRY",
                            "position": "SHORT",
                            "time": current_data["timestamp"],
                            "price": entry_price,
                            "size": position_size,
                            "reason": signal["reason"]
                        })
                        
                else:
                    # Update current position
                    current_position["current_price"] = price
                    
                    # Check for exit conditions
                    exit_reason = None
                    
                    if current_position["type"] == "LONG":
                        # Check stop loss
                        if price <= current_position["stop_loss"]:
                            exit_reason = "STOP_LOSS"
                        # Check take profit
                        elif price >= current_position["take_profit"]:
                            exit_reason = "TAKE_PROFIT"
                        # Check for exit signal
                        elif signal["signal"] == "SHORT" and signal["confidence"] > 0.8:
                            exit_reason = "SIGNAL_REVERSAL"
                            
                    elif current_position["type"] == "SHORT":
                        # Check stop loss
                        if price >= current_position["stop_loss"]:
                            exit_reason = "STOP_LOSS"
                        # Check take profit
                        elif price <= current_position["take_profit"]:
                            exit_reason = "TAKE_PROFIT"
                        # Check for exit signal
                        elif signal["signal"] == "LONG" and signal["confidence"] > 0.8:
                            exit_reason = "SIGNAL_REVERSAL"
                            
                    # Exit position if needed
                    if exit_reason:
                        # Calculate profit/loss
                        if current_position["type"] == "LONG":
                            pnl_pct = (price / current_position["entry_price"]) - 1.0
                        else:  # SHORT
                            pnl_pct = 1.0 - (price / current_position["entry_price"])
                            
                        # Apply commission
                        pnl_pct -= commission_rate * 2  # Entry and exit
                        
                        # Update equity
                        trade_equity_impact = pnl_pct * risk_per_trade
                        new_equity = equity_curve[-1] * (1 + trade_equity_impact)
                        equity_curve.append(new_equity)
                        
                        # Record trade
                        trades.append({
                            "type": "EXIT",
                            "position": current_position["type"],
                            "time": current_data["timestamp"],
                            "price": price,
                            "size": current_position["size"],
                            "entry_price": current_position["entry_price"],
                            "pnl_pct": pnl_pct,
                            "reason": exit_reason
                        })
                        
                        # Record position
                        positions.append({
                            "type": current_position["type"],
                            "entry_time": current_position["entry_time"],
                            "exit_time": current_data["timestamp"],
                            "entry_price": current_position["entry_price"],
                            "exit_price": price,
                            "size": current_position["size"],
                            "pnl_pct": pnl_pct,
                            "exit_reason": exit_reason
                        })
                        
                        # Clear position
                        current_position = None
                        
                # If no equity update from trade, copy last value
                if len(equity_curve) <= i - 99:
                    equity_curve.append(equity_curve[-1])
                    
            # Calculate performance metrics
            if positions:
                win_trades = [p for p in positions if p["pnl_pct"] > 0]
                loss_trades = [p for p in positions if p["pnl_pct"] <= 0]
                
                win_rate = len(win_trades) / len(positions) if positions else 0
                avg_win = np.mean([p["pnl_pct"] for p in win_trades]) if win_trades else 0
                avg_loss = np.mean([p["pnl_pct"] for p in loss_trades]) if loss_trades else 0
                profit_factor = abs(sum(p["pnl_pct"] for p in win_trades) / sum(p["pnl_pct"] for p in loss_trades)) if loss_trades and sum(p["pnl_pct"] for p in loss_trades) != 0 else 0
                
                # Calculate drawdown
                equity_array = np.array(equity_curve)
                max_equity = np.maximum.accumulate(equity_array)
                drawdown = (equity_array / max_equity) - 1.0
                max_drawdown = abs(min(drawdown))
                
                # Calculate Sharpe ratio (assuming risk-free rate of 0)
                returns = np.diff(equity_array) / equity_array[:-1]
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
                
                # Final equity
                final_equity = equity_curve[-1] * initial_equity
                total_return = (final_equity / initial_equity) - 1.0
                
                # Store results
                results = {
                    "symbol": symbol,
                    "strategy": "triple_confluence",
                    "initial_equity": initial_equity,
                    "final_equity": final_equity,
                    "total_return": total_return,
                    "total_trades": len(positions),
                    "win_rate": win_rate,
                    "avg_win": avg_win,
                    "avg_loss": avg_loss,
                    "profit_factor": profit_factor,
                    "max_drawdown": max_drawdown,
                    "sharpe_ratio": sharpe_ratio,
                    "equity_curve": equity_curve,
                    "trades": trades,
                    "positions": positions
                }
                
                # Save results
                self.backtest_results[f"{symbol}_triple_confluence"] = results
                
                # Generate charts
                self._generate_backtest_charts(symbol, "triple_confluence", results)
                
                # Save detailed results to file
                with open(f"backtest_results/{symbol}_triple_confluence_results.json", "w") as f:
                    # Convert non-serializable objects
                    serializable_results = results.copy()
                    serializable_results["equity_curve"] = [float(e) for e in equity_curve]
                    
                    # Convert timestamps to strings
                    for trade in serializable_results["trades"]:
                        trade["time"] = str(trade["time"])
                        
                    for position in serializable_results["positions"]:
                        position["entry_time"] = str(position["entry_time"])
                        position["exit_time"] = str(position["exit_time"])
                        
                    json.dump(serializable_results, f, indent=2)
                    
                self.logger.info(f"Triple Confluence backtest completed for {symbol}: {len(positions)} trades, {win_rate:.2%} win rate, {total_return:.2%} return")
                
                return results
            else:
                self.logger.warning(f"No trades executed in backtest for {symbol}")
                return None
                
        except Exception as e:
            self.logger.exception(f"Error in Triple Confluence backtest: {str(e)}")
            return None
            
    def _optimize_triple_confluence(self, symbol: str) -> Dict[str, Any]:
        """
        Optimize Triple Confluence strategy parameters.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of optimized parameters
        """
        self.logger.info(f"Optimizing Triple Confluence parameters for {symbol}...")
        
        try:
            # Define parameter ranges
            param_ranges = {
                "min_order_imbalance": [1.1, 1.2, 1.3, 1.4, 1.5],
                "funding_threshold": [0.000005, 0.00001, 0.00002],
                "vwma_fast_period": [10, 20, 30],
                "vwma_slow_period": [40, 50, 60]
            }
            
            # Generate parameter combinations
            param_combinations = []
            
            for min_order_imbalance in param_ranges["min_order_imbalance"]:
                for funding_threshold in param_ranges["funding_threshold"]:
                    for vwma_fast_period in param_ranges["vwma_fast_period"]:
                        for vwma_slow_period in param_ranges["vwma_slow_period"]:
                            if vwma_slow_period > vwma_fast_period:  # Ensure slow > fast
                                param_combinations.append({
                                    "min_order_imbalance": min_order_imbalance,
                                    "funding_threshold": funding_threshold,
                                    "vwma_fast_period": vwma_fast_period,
                                    "vwma_slow_period": vwma_slow_period
                                })
                                
            # Run optimization
            best_params = None
            best_score = -float("inf")
            
            for params in tqdm(param_combinations, desc="Optimizing"):
                # Create strategy with these parameters
                strategy_config = {**self.config, **params}
                strategy = TripleConfluenceStrategy(strategy_config, self.logger)
                
                # Run simplified backtest
                score = self._evaluate_parameters(symbol, strategy)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            # Save optimization results
            self.optimization_results[f"{symbol}_triple_confluence"] = {
                "best_params": best_params,
                "best_score": best_score,
                "param_combinations": len(param_combinations)
            }
            
            self.logger.info(f"Optimization completed for {symbol}: best score {best_score:.4f} with params {best_params}")
            
            return best_params
            
        except Exception as e:
            self.logger.exception(f"Error in Triple Confluence optimization: {str(e)}")
            return None
            
    def _evaluate_parameters(self, symbol: str, strategy) -> float:
        """
        Evaluate a set of strategy parameters.
        
        Args:
            symbol: Trading symbol
            strategy: Strategy instance with parameters to evaluate
            
        Returns:
            Evaluation score (higher is better)
        """
        try:
            # Get historical data
            df = self.historical_data[symbol].copy()
            
            # Initialize results
            trades = []
            current_position = None
            equity_curve = [1.0]  # Start with 1.0 (100%)
            
            # Backtest parameters
            risk_per_trade = self.config.get("risk_percent", 0.01)
            commission_rate = self.config.get("taker_fee", 0.00042)
            
            # Run simplified backtest (fewer iterations for speed)
            step = max(1, len(df) // 500)  # Sample points for faster optimization
            
            for i in range(100, len(df), step):
                # Get current data
                current_data = df.iloc[i]
                price = current_data["price"]
                volume = current_data["volume"]
                funding_rate = current_data["funding_rate"]
                
                # Create mock order book
                mock_order_book = {
                    "bids": [[price * (1 - j * 0.001), 10 + j] for j in range(10)],
                    "asks": [[price * (1 + j * 0.001), 5 + j] for j in range(10)]
                }
                
                # Update strategy data
                strategy.update_data(
                    symbol,
                    price,
                    volume,
                    funding_rate,
                    mock_order_book
                )
                
                # Get signal
                signal = strategy.analyze(symbol)
                
                # Process signal
                if current_position is None:
                    # No position, check for entry
                    if signal["signal"] == "LONG" and signal["confidence"] > 0.7:
                        # Enter long position
                        entry_price = price
                        stop_loss = price * 0.99  # 1% stop loss
                        take_profit = price * 1.03  # 3% take profit
                        
                        current_position = {
                            "type": "LONG",
                            "entry_price": entry_price,
                            "entry_index": i,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit
                        }
                        
                    elif signal["signal"] == "SHORT" and signal["confidence"] > 0.7:
                        # Enter short position
                        entry_price = price
                        stop_loss = price * 1.01  # 1% stop loss
                        take_profit = price * 0.97  # 3% take profit
                        
                        current_position = {
                            "type": "SHORT",
                            "entry_price": entry_price,
                            "entry_index": i,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit
                        }
                        
                else:
                    # Check for exit conditions
                    exit_reason = None
                    
                    if current_position["type"] == "LONG":
                        # Check stop loss
                        if price <= current_position["stop_loss"]:
                            exit_reason = "STOP_LOSS"
                        # Check take profit
                        elif price >= current_position["take_profit"]:
                            exit_reason = "TAKE_PROFIT"
                        # Check for exit signal
                        elif signal["signal"] == "SHORT" and signal["confidence"] > 0.8:
                            exit_reason = "SIGNAL_REVERSAL"
                            
                    elif current_position["type"] == "SHORT":
                        # Check stop loss
                        if price >= current_position["stop_loss"]:
                            exit_reason = "STOP_LOSS"
                        # Check take profit
                        elif price <= current_position["take_profit"]:
                            exit_reason = "TAKE_PROFIT"
                        # Check for exit signal
                        elif signal["signal"] == "LONG" and signal["confidence"] > 0.8:
                            exit_reason = "SIGNAL_REVERSAL"
                            
                    # Exit position if needed
                    if exit_reason:
                        # Calculate profit/loss
                        if current_position["type"] == "LONG":
                            pnl_pct = (price / current_position["entry_price"]) - 1.0
                        else:  # SHORT
                            pnl_pct = 1.0 - (price / current_position["entry_price"])
                            
                        # Apply commission
                        pnl_pct -= commission_rate * 2  # Entry and exit
                        
                        # Update equity
                        trade_equity_impact = pnl_pct * risk_per_trade
                        new_equity = equity_curve[-1] * (1 + trade_equity_impact)
                        equity_curve.append(new_equity)
                        
                        # Record trade
                        trades.append({
                            "type": current_position["type"],
                            "entry_index": current_position["entry_index"],
                            "exit_index": i,
                            "entry_price": current_position["entry_price"],
                            "exit_price": price,
                            "pnl_pct": pnl_pct,
                            "exit_reason": exit_reason
                        })
                        
                        # Clear position
                        current_position = None
                        
            # Calculate evaluation score
            if trades:
                win_trades = [t for t in trades if t["pnl_pct"] > 0]
                loss_trades = [t for t in trades if t["pnl_pct"] <= 0]
                
                win_rate = len(win_trades) / len(trades) if trades else 0
                avg_win = np.mean([t["pnl_pct"] for t in win_trades]) if win_trades else 0
                avg_loss = np.mean([t["pnl_pct"] for t in loss_trades]) if loss_trades else 0
                
                # Calculate drawdown
                equity_array = np.array(equity_curve)
                max_equity = np.maximum.accumulate(equity_array)
                drawdown = (equity_array / max_equity) - 1.0
                max_drawdown = abs(min(drawdown)) if len(drawdown) > 0 else 0
                
                # Calculate Sharpe ratio
                returns = np.diff(equity_array) / equity_array[:-1]
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 and np.std(returns) != 0 else 0
                
                # Final equity
                total_return = equity_curve[-1] - 1.0
                
                # Combine metrics into score
                # Weight factors based on importance
                score = (
                    total_return * 0.4 +
                    win_rate * 0.2 +
                    sharpe_ratio * 0.2 -
                    max_drawdown * 0.2
                )
                
                return score
            else:
                return -1.0  # Penalize strategies that don't generate trades
                
        except Exception as e:
            self.logger.error(f"Error evaluating parameters: {str(e)}")
            return -float("inf")  # Worst possible score
            
    def backtest_oracle_update(self, symbol: str, optimize: bool = False):
        """
        Backtest the Oracle Update strategy.
        
        Args:
            symbol: Trading symbol
            optimize: Whether to optimize strategy parameters
            
        Returns:
            Backtest results dictionary
        """
        self.logger.info(f"Backtesting Oracle Update strategy for {symbol}...")
        
        if symbol not in self.historical_data:
            self.logger.error(f"No historical data available for {symbol}")
            return None
            
        try:
            # Get historical data
            df = self.historical_data[symbol].copy()
            
            # Initialize strategy
            strategy = OracleUpdateStrategy(
                self.config,
                self.logger
            )
            
            # Initialize results
            trades = []
            positions = []
            current_position = None
            equity_curve = [1.0]  # Start with 1.0 (100%)
            
            # Backtest parameters
            initial_equity = 10000.0
            risk_per_trade = self.config.get("risk_percent", 0.01)
            commission_rate = self.config.get("taker_fee", 0.00042)
            
            # Optimization parameters
            if optimize:
                best_params = self._optimize_oracle_update(symbol)
                
                if best_params:
                    self.logger.info(f"Using optimized parameters: {best_params}")
                    
                    # Update strategy with optimized parameters
                    strategy = OracleUpdateStrategy(
                        {**self.config, **best_params},
                        self.logger
                    )
                    
            # Run backtest
            for i in tqdm(range(10, len(df)), desc="Backtesting"):
                # Get current data
                current_data = df.iloc[i]
                market_price = current_data["price"]
                oracle_price = current_data["oracle_price"]
                
                # Update strategy data
                strategy.update_data(
                    symbol,
                    market_price,
                    oracle_price
                )
                
                # Get signal
                signal = strategy.analyze(symbol)
                
                # Process signal
                if current_position is None:
                    # No position, check for entry
                    if signal["signal"] == "LONG" and signal["confidence"] > 0.7:
                        # Enter long position
                        position_size = initial_equity * risk_per_trade / (market_price * 0.01)  # 1% stop loss
                        entry_price = market_price
                        stop_loss = market_price * 0.99  # 1% stop loss
                        
                        current_position = {
                            "type": "LONG",
                            "entry_price": entry_price,
                            "entry_time": current_data["timestamp"],
                            "size": position_size,
                            "stop_loss": stop_loss,
                            "current_price": market_price,
                            "max_duration": 10  # Max 10 bars for Oracle Update strategy
                        }
                        
                        trades.append({
                            "type": "ENTRY",
                            "position": "LONG",
                            "time": current_data["timestamp"],
                            "price": entry_price,
                            "size": position_size,
                            "reason": signal["reason"]
                        })
                        
                    elif signal["signal"] == "SHORT" and signal["confidence"] > 0.7:
                        # Enter short position
                        position_size = initial_equity * risk_per_trade / (market_price * 0.01)  # 1% stop loss
                        entry_price = market_price
                        stop_loss = market_price * 1.01  # 1% stop loss
                        
                        current_position = {
                            "type": "SHORT",
                            "entry_price": entry_price,
                            "entry_time": current_data["timestamp"],
                            "size": position_size,
                            "stop_loss": stop_loss,
                            "current_price": market_price,
                            "max_duration": 10,  # Max 10 bars for Oracle Update strategy
                            "duration": 0
                        }
                        
                        trades.append({
                            "type": "ENTRY",
                            "position": "SHORT",
                            "time": current_data["timestamp"],
                            "price": entry_price,
                            "size": position_size,
                            "reason": signal["reason"]
                        })
                        
                else:
                    # Update current position
                    current_position["current_price"] = market_price
                    current_position["duration"] = current_position.get("duration", 0) + 1
                    
                    # Check for exit conditions
                    exit_reason = None
                    
                    # Check for explicit exit signals
                    if signal["signal"].startswith("CLOSE_"):
                        exit_reason = "SIGNAL_EXIT"
                        
                    # Check duration limit
                    elif current_position["duration"] >= current_position["max_duration"]:
                        exit_reason = "MAX_DURATION"
                        
                    # Check stop loss
                    elif (current_position["type"] == "LONG" and market_price <= current_position["stop_loss"]) or \
                         (current_position["type"] == "SHORT" and market_price >= current_position["stop_loss"]):
                        exit_reason = "STOP_LOSS"
                        
                    # Check for reversal signal
                    elif (current_position["type"] == "LONG" and signal["signal"] == "SHORT" and signal["confidence"] > 0.8) or \
                         (current_position["type"] == "SHORT" and signal["signal"] == "LONG" and signal["confidence"] > 0.8):
                        exit_reason = "SIGNAL_REVERSAL"
                        
                    # Exit position if needed
                    if exit_reason:
                        # Calculate profit/loss
                        if current_position["type"] == "LONG":
                            pnl_pct = (market_price / current_position["entry_price"]) - 1.0
                        else:  # SHORT
                            pnl_pct = 1.0 - (market_price / current_position["entry_price"])
                            
                        # Apply commission
                        pnl_pct -= commission_rate * 2  # Entry and exit
                        
                        # Update equity
                        trade_equity_impact = pnl_pct * risk_per_trade
                        new_equity = equity_curve[-1] * (1 + trade_equity_impact)
                        equity_curve.append(new_equity)
                        
                        # Record trade
                        trades.append({
                            "type": "EXIT",
                            "position": current_position["type"],
                            "time": current_data["timestamp"],
                            "price": market_price,
                            "size": current_position["size"],
                            "entry_price": current_position["entry_price"],
                            "pnl_pct": pnl_pct,
                            "reason": exit_reason
                        })
                        
                        # Record position
                        positions.append({
                            "type": current_position["type"],
                            "entry_time": current_position["entry_time"],
                            "exit_time": current_data["timestamp"],
                            "entry_price": current_position["entry_price"],
                            "exit_price": market_price,
                            "size": current_position["size"],
                            "pnl_pct": pnl_pct,
                            "exit_reason": exit_reason,
                            "duration": current_position["duration"]
                        })
                        
                        # Clear position
                        current_position = None
                        
                # If no equity update from trade, copy last value
                if len(equity_curve) <= i - 9:
                    equity_curve.append(equity_curve[-1])
                    
            # Calculate performance metrics
            if positions:
                win_trades = [p for p in positions if p["pnl_pct"] > 0]
                loss_trades = [p for p in positions if p["pnl_pct"] <= 0]
                
                win_rate = len(win_trades) / len(positions) if positions else 0
                avg_win = np.mean([p["pnl_pct"] for p in win_trades]) if win_trades else 0
                avg_loss = np.mean([p["pnl_pct"] for p in loss_trades]) if loss_trades else 0
                profit_factor = abs(sum(p["pnl_pct"] for p in win_trades) / sum(p["pnl_pct"] for p in loss_trades)) if loss_trades and sum(p["pnl_pct"] for p in loss_trades) != 0 else 0
                
                # Calculate drawdown
                equity_array = np.array(equity_curve)
                max_equity = np.maximum.accumulate(equity_array)
                drawdown = (equity_array / max_equity) - 1.0
                max_drawdown = abs(min(drawdown))
                
                # Calculate Sharpe ratio (assuming risk-free rate of 0)
                returns = np.diff(equity_array) / equity_array[:-1]
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
                
                # Final equity
                final_equity = equity_curve[-1] * initial_equity
                total_return = (final_equity / initial_equity) - 1.0
                
                # Store results
                results = {
                    "symbol": symbol,
                    "strategy": "oracle_update",
                    "initial_equity": initial_equity,
                    "final_equity": final_equity,
                    "total_return": total_return,
                    "total_trades": len(positions),
                    "win_rate": win_rate,
                    "avg_win": avg_win,
                    "avg_loss": avg_loss,
                    "profit_factor": profit_factor,
                    "max_drawdown": max_drawdown,
                    "sharpe_ratio": sharpe_ratio,
                    "equity_curve": equity_curve,
                    "trades": trades,
                    "positions": positions
                }
                
                # Save results
                self.backtest_results[f"{symbol}_oracle_update"] = results
                
                # Generate charts
                self._generate_backtest_charts(symbol, "oracle_update", results)
                
                # Save detailed results to file
                with open(f"backtest_results/{symbol}_oracle_update_results.json", "w") as f:
                    # Convert non-serializable objects
                    serializable_results = results.copy()
                    serializable_results["equity_curve"] = [float(e) for e in equity_curve]
                    
                    # Convert timestamps to strings
                    for trade in serializable_results["trades"]:
                        trade["time"] = str(trade["time"])
                        
                    for position in serializable_results["positions"]:
                        position["entry_time"] = str(position["entry_time"])
                        position["exit_time"] = str(position["exit_time"])
                        
                    json.dump(serializable_results, f, indent=2)
                    
                self.logger.info(f"Oracle Update backtest completed for {symbol}: {len(positions)} trades, {win_rate:.2%} win rate, {total_return:.2%} return")
                
                return results
            else:
                self.logger.warning(f"No trades executed in backtest for {symbol}")
                return None
                
        except Exception as e:
            self.logger.exception(f"Error in Oracle Update backtest: {str(e)}")
            return None
            
    def _optimize_oracle_update(self, symbol: str) -> Dict[str, Any]:
        """
        Optimize Oracle Update strategy parameters.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of optimized parameters
        """
        self.logger.info(f"Optimizing Oracle Update parameters for {symbol}...")
        
        try:
            # Define parameter ranges
            param_ranges = {
                "min_price_deviation": [0.0005, 0.001, 0.0015, 0.002, 0.0025],
                "max_price_deviation": [0.005, 0.0075, 0.01, 0.0125, 0.015],
                "oracle_update_interval": [2, 3, 4],
                "max_trade_duration": [10, 20, 30, 40, 50]
            }
            
            # Generate parameter combinations
            param_combinations = []
            
            for min_dev in param_ranges["min_price_deviation"]:
                for max_dev in param_ranges["max_price_deviation"]:
                    for update_interval in param_ranges["oracle_update_interval"]:
                        for max_duration in param_ranges["max_trade_duration"]:
                            if max_dev > min_dev:  # Ensure max > min
                                param_combinations.append({
                                    "min_price_deviation": min_dev,
                                    "max_price_deviation": max_dev,
                                    "oracle_update_interval": update_interval,
                                    "max_trade_duration": max_duration
                                })
                                
            # Run optimization
            best_params = None
            best_score = -float("inf")
            
            for params in tqdm(param_combinations, desc="Optimizing"):
                # Create strategy with these parameters
                strategy_config = {**self.config, **params}
                strategy = OracleUpdateStrategy(strategy_config, self.logger)
                
                # Run simplified backtest
                score = self._evaluate_oracle_parameters(symbol, strategy)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            # Save optimization results
            self.optimization_results[f"{symbol}_oracle_update"] = {
                "best_params": best_params,
                "best_score": best_score,
                "param_combinations": len(param_combinations)
            }
            
            self.logger.info(f"Optimization completed for {symbol}: best score {best_score:.4f} with params {best_params}")
            
            return best_params
            
        except Exception as e:
            self.logger.exception(f"Error in Oracle Update optimization: {str(e)}")
            return None
            
    def _evaluate_oracle_parameters(self, symbol: str, strategy) -> float:
        """
        Evaluate a set of Oracle Update strategy parameters.
        
        Args:
            symbol: Trading symbol
            strategy: Strategy instance with parameters to evaluate
            
        Returns:
            Evaluation score (higher is better)
        """
        try:
            # Get historical data
            df = self.historical_data[symbol].copy()
            
            # Initialize results
            trades = []
            current_position = None
            equity_curve = [1.0]  # Start with 1.0 (100%)
            
            # Backtest parameters
            risk_per_trade = self.config.get("risk_percent", 0.01)
            commission_rate = self.config.get("taker_fee", 0.00042)
            
            # Run simplified backtest (fewer iterations for speed)
            step = max(1, len(df) // 500)  # Sample points for faster optimization
            
            for i in range(10, len(df), step):
                # Get current data
                current_data = df.iloc[i]
                market_price = current_data["price"]
                oracle_price = current_data["oracle_price"]
                
                # Update strategy data
                strategy.update_data(
                    symbol,
                    market_price,
                    oracle_price
                )
                
                # Get signal
                signal = strategy.analyze(symbol)
                
                # Process signal
                if current_position is None:
                    # No position, check for entry
                    if signal["signal"] == "LONG" and signal["confidence"] > 0.7:
                        # Enter long position
                        entry_price = market_price
                        stop_loss = market_price * 0.99  # 1% stop loss
                        
                        current_position = {
                            "type": "LONG",
                            "entry_price": entry_price,
                            "entry_index": i,
                            "stop_loss": stop_loss,
                            "max_duration": strategy.max_trade_duration // step,
                            "duration": 0
                        }
                        
                    elif signal["signal"] == "SHORT" and signal["confidence"] > 0.7:
                        # Enter short position
                        entry_price = market_price
                        stop_loss = market_price * 1.01  # 1% stop loss
                        
                        current_position = {
                            "type": "SHORT",
                            "entry_price": entry_price,
                            "entry_index": i,
                            "stop_loss": stop_loss,
                            "max_duration": strategy.max_trade_duration // step,
                            "duration": 0
                        }
                        
                else:
                    # Update position
                    current_position["duration"] += 1
                    
                    # Check for exit conditions
                    exit_reason = None
                    
                    # Check for explicit exit signals
                    if signal["signal"].startswith("CLOSE_"):
                        exit_reason = "SIGNAL_EXIT"
                        
                    # Check duration limit
                    elif current_position["duration"] >= current_position["max_duration"]:
                        exit_reason = "MAX_DURATION"
                        
                    # Check stop loss
                    elif (current_position["type"] == "LONG" and market_price <= current_position["stop_loss"]) or \
                         (current_position["type"] == "SHORT" and market_price >= current_position["stop_loss"]):
                        exit_reason = "STOP_LOSS"
                        
                    # Exit position if needed
                    if exit_reason:
                        # Calculate profit/loss
                        if current_position["type"] == "LONG":
                            pnl_pct = (market_price / current_position["entry_price"]) - 1.0
                        else:  # SHORT
                            pnl_pct = 1.0 - (market_price / current_position["entry_price"])
                            
                        # Apply commission
                        pnl_pct -= commission_rate * 2  # Entry and exit
                        
                        # Update equity
                        trade_equity_impact = pnl_pct * risk_per_trade
                        new_equity = equity_curve[-1] * (1 + trade_equity_impact)
                        equity_curve.append(new_equity)
                        
                        # Record trade
                        trades.append({
                            "type": current_position["type"],
                            "entry_index": current_position["entry_index"],
                            "exit_index": i,
                            "entry_price": current_position["entry_price"],
                            "exit_price": market_price,
                            "pnl_pct": pnl_pct,
                            "exit_reason": exit_reason,
                            "duration": current_position["duration"]
                        })
                        
                        # Clear position
                        current_position = None
                        
            # Calculate evaluation score
            if trades:
                win_trades = [t for t in trades if t["pnl_pct"] > 0]
                loss_trades = [t for t in trades if t["pnl_pct"] <= 0]
                
                win_rate = len(win_trades) / len(trades) if trades else 0
                avg_win = np.mean([t["pnl_pct"] for t in win_trades]) if win_trades else 0
                avg_loss = np.mean([t["pnl_pct"] for t in loss_trades]) if loss_trades else 0
                
                # Calculate drawdown
                equity_array = np.array(equity_curve)
                max_equity = np.maximum.accumulate(equity_array)
                drawdown = (equity_array / max_equity) - 1.0
                max_drawdown = abs(min(drawdown)) if len(drawdown) > 0 else 0
                
                # Calculate Sharpe ratio
                returns = np.diff(equity_array) / equity_array[:-1]
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 and np.std(returns) != 0 else 0
                
                # Final equity
                total_return = equity_curve[-1] - 1.0
                
                # Combine metrics into score
                # Weight factors based on importance
                score = (
                    total_return * 0.4 +
                    win_rate * 0.2 +
                    sharpe_ratio * 0.2 -
                    max_drawdown * 0.2
                )
                
                return score
            else:
                return -1.0  # Penalize strategies that don't generate trades
                
        except Exception as e:
            self.logger.error(f"Error evaluating Oracle Update parameters: {str(e)}")
            return -float("inf")  # Worst possible score
            
    def _generate_backtest_charts(self, symbol: str, strategy_name: str, results: Dict[str, Any]):
        """
        Generate charts for backtest results.
        
        Args:
            symbol: Trading symbol
            strategy_name: Strategy name
            results: Backtest results
        """
        try:
            # Create figure with subplots
            fig, axs = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [2, 1, 1]})
            
            # Plot equity curve
            equity_curve = results["equity_curve"]
            axs[0].plot(equity_curve)
            axs[0].set_title(f"{symbol} - {strategy_name} Equity Curve")
            axs[0].set_ylabel("Equity (normalized)")
            axs[0].grid(True)
            
            # Add key metrics as text
            metrics_text = (
                f"Total Return: {results['total_return']:.2%}\n"
                f"Win Rate: {results['win_rate']:.2%}\n"
                f"Profit Factor: {results['profit_factor']:.2f}\n"
                f"Max Drawdown: {results['max_drawdown']:.2%}\n"
                f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n"
                f"Total Trades: {results['total_trades']}"
            )
            axs[0].text(0.02, 0.95, metrics_text, transform=axs[0].transAxes,
                      verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5})
                      
            # Plot drawdown
            equity_array = np.array(equity_curve)
            max_equity = np.maximum.accumulate(equity_array)
            drawdown = (equity_array / max_equity) - 1.0
            
            axs[1].fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
            axs[1].set_title(f"{symbol} - {strategy_name} Drawdown")
            axs[1].set_ylabel("Drawdown")
            axs[1].grid(True)
            
            # Plot trade outcomes
            if results["positions"]:
                trade_returns = [p["pnl_pct"] for p in results["positions"]]
                trade_types = [p["type"] for p in results["positions"]]
                
                colors = ['green' if r > 0 else 'red' for r in trade_returns]
                
                axs[2].bar(range(len(trade_returns)), trade_returns, color=colors)
                axs[2].set_title(f"{symbol} - {strategy_name} Trade Returns")
                axs[2].set_xlabel("Trade Number")
                axs[2].set_ylabel("Return (%)")
                axs[2].grid(True)
                
                # Add trade type annotations
                for i, (r, t) in enumerate(zip(trade_returns, trade_types)):
                    axs[2].annotate(t[0], (i, 0), ha='center', va='center', 
                                  xytext=(0, 10 if r > 0 else -10),
                                  textcoords='offset points')
                                  
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(f"backtest_results/charts/{symbol}_{strategy_name}_backtest.png")
            plt.close()
            
            self.logger.info(f"Generated backtest charts for {symbol} {strategy_name}")
            
        except Exception as e:
            self.logger.error(f"Error generating backtest charts: {str(e)}")
            
    def run_combined_backtest(self, symbol: str, optimize: bool = False):
        """
        Run a combined backtest using multiple strategies.
        
        Args:
            symbol: Trading symbol
            optimize: Whether to optimize strategy parameters
            
        Returns:
            Combined backtest results
        """
        self.logger.info(f"Running combined backtest for {symbol}...")
        
        # Run individual strategy backtests first
        tc_results = self.backtest_triple_confluence(symbol, optimize)
        ou_results = self.backtest_oracle_update(symbol, optimize)
        
        if not tc_results or not ou_results:
            self.logger.error("Individual strategy backtests failed, cannot run combined backtest")
            return None
            
        try:
            # Get historical data
            df = self.historical_data[symbol].copy()
            
            # Initialize strategies with optimized parameters if available
            tc_strategy = TripleConfluenceStrategy(
                self.config if not optimize else {**self.config, **self.optimization_results.get(f"{symbol}_triple_confluence", {}).get("best_params", {})},
                self.logger
            )
            
            ou_strategy = OracleUpdateStrategy(
                self.config if not optimize else {**self.config, **self.optimization_results.get(f"{symbol}_oracle_update", {}).get("best_params", {})},
                self.logger
            )
            
            # Initialize results
            trades = []
            positions = []
            current_position = None
            equity_curve = [1.0]  # Start with 1.0 (100%)
            
            # Backtest parameters
            initial_equity = 10000.0
            risk_per_trade = self.config.get("risk_percent", 0.01)
            commission_rate = self.config.get("taker_fee", 0.00042)
            
            # Run backtest
            for i in tqdm(range(100, len(df)), desc="Combined Backtest"):
                # Get current data
                current_data = df.iloc[i]
                price = current_data["price"]
                volume = current_data["volume"]
                funding_rate = current_data["funding_rate"]
                oracle_price = current_data["oracle_price"]
                
                # Create mock order book
                mock_order_book = {
                    "bids": [[price * (1 - j * 0.001), 10 + j] for j in range(10)],
                    "asks": [[price * (1 + j * 0.001), 5 + j] for j in range(10)]
                }
                
                # Update strategy data
                tc_strategy.update_data(
                    symbol,
                    price,
                    volume,
                    funding_rate,
                    mock_order_book
                )
                
                ou_strategy.update_data(
                    symbol,
                    price,
                    oracle_price
                )
                
                # Get signals
                tc_signal = tc_strategy.analyze(symbol)
                ou_signal = ou_strategy.analyze(symbol)
                
                # Combine signals
                combined_signal = self._combine_signals(tc_signal, ou_signal)
                
                # Process signal
                if current_position is None:
                    # No position, check for entry
                    if combined_signal["signal"] == "LONG" and combined_signal["confidence"] > 0.7:
                        # Enter long position
                        position_size = initial_equity * risk_per_trade / (price * 0.01)  # 1% stop loss
                        entry_price = price
                        stop_loss = price * 0.99  # 1% stop loss
                        take_profit = price * 1.03  # 3% take profit
                        
                        current_position = {
                            "type": "LONG",
                            "entry_price": entry_price,
                            "entry_time": current_data["timestamp"],
                            "size": position_size,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "current_price": price,
                            "strategy": combined_signal["strategy"]
                        }
                        
                        trades.append({
                            "type": "ENTRY",
                            "position": "LONG",
                            "time": current_data["timestamp"],
                            "price": entry_price,
                            "size": position_size,
                            "reason": combined_signal["reason"],
                            "strategy": combined_signal["strategy"]
                        })
                        
                    elif combined_signal["signal"] == "SHORT" and combined_signal["confidence"] > 0.7:
                        # Enter short position
                        position_size = initial_equity * risk_per_trade / (price * 0.01)  # 1% stop loss
                        entry_price = price
                        stop_loss = price * 1.01  # 1% stop loss
                        take_profit = price * 0.97  # 3% take profit
                        
                        current_position = {
                            "type": "SHORT",
                            "entry_price": entry_price,
                            "entry_time": current_data["timestamp"],
                            "size": position_size,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "current_price": price,
                            "strategy": combined_signal["strategy"]
                        }
                        
                        trades.append({
                            "type": "ENTRY",
                            "position": "SHORT",
                            "time": current_data["timestamp"],
                            "price": entry_price,
                            "size": position_size,
                            "reason": combined_signal["reason"],
                            "strategy": combined_signal["strategy"]
                        })
                        
                else:
                    # Update current position
                    current_position["current_price"] = price
                    
                    # Check for exit conditions
                    exit_reason = None
                    
                    # Check for explicit exit signals from Oracle Update strategy
                    if ou_signal["signal"].startswith("CLOSE_"):
                        exit_reason = "SIGNAL_EXIT"
                        
                    # Check for position-specific conditions
                    if current_position["type"] == "LONG":
                        # Check stop loss
                        if price <= current_position["stop_loss"]:
                            exit_reason = "STOP_LOSS"
                        # Check take profit
                        elif price >= current_position["take_profit"]:
                            exit_reason = "TAKE_PROFIT"
                        # Check for exit signal
                        elif combined_signal["signal"] == "SHORT" and combined_signal["confidence"] > 0.8:
                            exit_reason = "SIGNAL_REVERSAL"
                            
                    elif current_position["type"] == "SHORT":
                        # Check stop loss
                        if price >= current_position["stop_loss"]:
                            exit_reason = "STOP_LOSS"
                        # Check take profit
                        elif price <= current_position["take_profit"]:
                            exit_reason = "TAKE_PROFIT"
                        # Check for exit signal
                        elif combined_signal["signal"] == "LONG" and combined_signal["confidence"] > 0.8:
                            exit_reason = "SIGNAL_REVERSAL"
                            
                    # Exit position if needed
                    if exit_reason:
                        # Calculate profit/loss
                        if current_position["type"] == "LONG":
                            pnl_pct = (price / current_position["entry_price"]) - 1.0
                        else:  # SHORT
                            pnl_pct = 1.0 - (price / current_position["entry_price"])
                            
                        # Apply commission
                        pnl_pct -= commission_rate * 2  # Entry and exit
                        
                        # Update equity
                        trade_equity_impact = pnl_pct * risk_per_trade
                        new_equity = equity_curve[-1] * (1 + trade_equity_impact)
                        equity_curve.append(new_equity)
                        
                        # Record trade
                        trades.append({
                            "type": "EXIT",
                            "position": current_position["type"],
                            "time": current_data["timestamp"],
                            "price": price,
                            "size": current_position["size"],
                            "entry_price": current_position["entry_price"],
                            "pnl_pct": pnl_pct,
                            "reason": exit_reason,
                            "strategy": current_position["strategy"]
                        })
                        
                        # Record position
                        positions.append({
                            "type": current_position["type"],
                            "entry_time": current_position["entry_time"],
                            "exit_time": current_data["timestamp"],
                            "entry_price": current_position["entry_price"],
                            "exit_price": price,
                            "size": current_position["size"],
                            "pnl_pct": pnl_pct,
                            "exit_reason": exit_reason,
                            "strategy": current_position["strategy"]
                        })
                        
                        # Clear position
                        current_position = None
                        
                # If no equity update from trade, copy last value
                if len(equity_curve) <= i - 99:
                    equity_curve.append(equity_curve[-1])
                    
            # Calculate performance metrics
            if positions:
                win_trades = [p for p in positions if p["pnl_pct"] > 0]
                loss_trades = [p for p in positions if p["pnl_pct"] <= 0]
                
                win_rate = len(win_trades) / len(positions) if positions else 0
                avg_win = np.mean([p["pnl_pct"] for p in win_trades]) if win_trades else 0
                avg_loss = np.mean([p["pnl_pct"] for p in loss_trades]) if loss_trades else 0
                profit_factor = abs(sum(p["pnl_pct"] for p in win_trades) / sum(p["pnl_pct"] for p in loss_trades)) if loss_trades and sum(p["pnl_pct"] for p in loss_trades) != 0 else 0
                
                # Calculate drawdown
                equity_array = np.array(equity_curve)
                max_equity = np.maximum.accumulate(equity_array)
                drawdown = (equity_array / max_equity) - 1.0
                max_drawdown = abs(min(drawdown))
                
                # Calculate Sharpe ratio (assuming risk-free rate of 0)
                returns = np.diff(equity_array) / equity_array[:-1]
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
                
                # Final equity
                final_equity = equity_curve[-1] * initial_equity
                total_return = (final_equity / initial_equity) - 1.0
                
                # Strategy breakdown
                strategy_breakdown = {}
                for p in positions:
                    strategy = p["strategy"]
                    if strategy not in strategy_breakdown:
                        strategy_breakdown[strategy] = {
                            "count": 0,
                            "wins": 0,
                            "losses": 0,
                            "total_pnl": 0.0
                        }
                        
                    strategy_breakdown[strategy]["count"] += 1
                    strategy_breakdown[strategy]["total_pnl"] += p["pnl_pct"]
                    
                    if p["pnl_pct"] > 0:
                        strategy_breakdown[strategy]["wins"] += 1
                    else:
                        strategy_breakdown[strategy]["losses"] += 1
                        
                # Calculate strategy win rates
                for strategy, stats in strategy_breakdown.items():
                    stats["win_rate"] = stats["wins"] / stats["count"] if stats["count"] > 0 else 0
                    stats["avg_pnl"] = stats["total_pnl"] / stats["count"] if stats["count"] > 0 else 0
                    
                # Store results
                results = {
                    "symbol": symbol,
                    "strategy": "combined",
                    "initial_equity": initial_equity,
                    "final_equity": final_equity,
                    "total_return": total_return,
                    "total_trades": len(positions),
                    "win_rate": win_rate,
                    "avg_win": avg_win,
                    "avg_loss": avg_loss,
                    "profit_factor": profit_factor,
                    "max_drawdown": max_drawdown,
                    "sharpe_ratio": sharpe_ratio,
                    "equity_curve": equity_curve,
                    "trades": trades,
                    "positions": positions,
                    "strategy_breakdown": strategy_breakdown
                }
                
                # Save results
                self.backtest_results[f"{symbol}_combined"] = results
                
                # Generate charts
                self._generate_combined_backtest_charts(symbol, results)
                
                # Save detailed results to file
                with open(f"backtest_results/{symbol}_combined_results.json", "w") as f:
                    # Convert non-serializable objects
                    serializable_results = results.copy()
                    serializable_results["equity_curve"] = [float(e) for e in equity_curve]
                    
                    # Convert timestamps to strings
                    for trade in serializable_results["trades"]:
                        trade["time"] = str(trade["time"])
                        
                    for position in serializable_results["positions"]:
                        position["entry_time"] = str(position["entry_time"])
                        position["exit_time"] = str(position["exit_time"])
                        
                    json.dump(serializable_results, f, indent=2)
                    
                self.logger.info(f"Combined backtest completed for {symbol}: {len(positions)} trades, {win_rate:.2%} win rate, {total_return:.2%} return")
                
                return results
            else:
                self.logger.warning(f"No trades executed in combined backtest for {symbol}")
                return None
                
        except Exception as e:
            self.logger.exception(f"Error in combined backtest: {str(e)}")
            return None
            
    def _combine_signals(self, tc_signal: Dict[str, Any], ou_signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine signals from multiple strategies.
        
        Args:
            tc_signal: Triple Confluence signal
            ou_signal: Oracle Update signal
            
        Returns:
            Combined signal
        """
        # Default to neutral
        combined_signal = {
            "signal": "NEUTRAL",
            "confidence": 0.0,
            "reason": "No strong signals",
            "strategy": "none"
        }
        
        # Check for Oracle Update exit signals
        if ou_signal["signal"].startswith("CLOSE_"):
            return {
                "signal": ou_signal["signal"],
                "confidence": ou_signal["confidence"],
                "reason": ou_signal["reason"],
                "strategy": "oracle_update"
            }
            
        # Check for strong Oracle Update signals (higher priority for short-term trades)
        if ou_signal["signal"] != "NEUTRAL" and ou_signal["confidence"] > 0.8:
            return {
                "signal": ou_signal["signal"],
                "confidence": ou_signal["confidence"],
                "reason": ou_signal["reason"],
                "strategy": "oracle_update"
            }
            
        # Check for Triple Confluence signals
        if tc_signal["signal"] != "NEUTRAL" and tc_signal["confidence"] > 0.7:
            return {
                "signal": tc_signal["signal"],
                "confidence": tc_signal["confidence"],
                "reason": tc_signal["reason"],
                "strategy": "triple_confluence"
            }
            
        # Check for agreement between strategies
        if tc_signal["signal"] == ou_signal["signal"] and tc_signal["signal"] != "NEUTRAL":
            # Combine confidence (weighted average)
            confidence = (tc_signal["confidence"] * 0.6) + (ou_signal["confidence"] * 0.4)
            
            return {
                "signal": tc_signal["signal"],
                "confidence": confidence,
                "reason": f"Strategy agreement: {tc_signal['reason']} & {ou_signal['reason']}",
                "strategy": "combined"
            }
            
        # Check for weaker Oracle Update signals
        if ou_signal["signal"] != "NEUTRAL" and ou_signal["confidence"] > 0.7:
            return {
                "signal": ou_signal["signal"],
                "confidence": ou_signal["confidence"],
                "reason": ou_signal["reason"],
                "strategy": "oracle_update"
            }
            
        # Default to neutral
        return combined_signal
        
    def _generate_combined_backtest_charts(self, symbol: str, results: Dict[str, Any]):
        """
        Generate charts for combined backtest results.
        
        Args:
            symbol: Trading symbol
            results: Backtest results
        """
        try:
            # Create figure with subplots
            fig, axs = plt.subplots(4, 1, figsize=(12, 24), gridspec_kw={'height_ratios': [2, 1, 1, 1]})
            
            # Plot equity curve
            equity_curve = results["equity_curve"]
            axs[0].plot(equity_curve)
            axs[0].set_title(f"{symbol} - Combined Strategy Equity Curve")
            axs[0].set_ylabel("Equity (normalized)")
            axs[0].grid(True)
            
            # Add key metrics as text
            metrics_text = (
                f"Total Return: {results['total_return']:.2%}\n"
                f"Win Rate: {results['win_rate']:.2%}\n"
                f"Profit Factor: {results['profit_factor']:.2f}\n"
                f"Max Drawdown: {results['max_drawdown']:.2%}\n"
                f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n"
                f"Total Trades: {results['total_trades']}"
            )
            axs[0].text(0.02, 0.95, metrics_text, transform=axs[0].transAxes,
                      verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5})
                      
            # Plot drawdown
            equity_array = np.array(equity_curve)
            max_equity = np.maximum.accumulate(equity_array)
            drawdown = (equity_array / max_equity) - 1.0
            
            axs[1].fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
            axs[1].set_title(f"{symbol} - Combined Strategy Drawdown")
            axs[1].set_ylabel("Drawdown")
            axs[1].grid(True)
            
            # Plot trade outcomes
            if results["positions"]:
                trade_returns = [p["pnl_pct"] for p in results["positions"]]
                trade_types = [p["type"] for p in results["positions"]]
                trade_strategies = [p["strategy"] for p in results["positions"]]
                
                colors = ['green' if r > 0 else 'red' for r in trade_returns]
                
                axs[2].bar(range(len(trade_returns)), trade_returns, color=colors)
                axs[2].set_title(f"{symbol} - Combined Strategy Trade Returns")
                axs[2].set_xlabel("Trade Number")
                axs[2].set_ylabel("Return (%)")
                axs[2].grid(True)
                
                # Add trade type annotations
                for i, (r, t) in enumerate(zip(trade_returns, trade_types)):
                    axs[2].annotate(t[0], (i, 0), ha='center', va='center', 
                                  xytext=(0, 10 if r > 0 else -10),
                                  textcoords='offset points')
                                  
                # Plot strategy breakdown
                strategy_breakdown = results["strategy_breakdown"]
                strategies = list(strategy_breakdown.keys())
                counts = [strategy_breakdown[s]["count"] for s in strategies]
                win_rates = [strategy_breakdown[s]["win_rate"] * 100 for s in strategies]
                avg_pnls = [strategy_breakdown[s]["avg_pnl"] * 100 for s in strategies]
                
                x = np.arange(len(strategies))
                width = 0.25
                
                axs[3].bar(x - width, counts, width, label='Trade Count')
                axs[3].bar(x, win_rates, width, label='Win Rate (%)')
                axs[3].bar(x + width, avg_pnls, width, label='Avg PnL (%)')
                
                axs[3].set_title(f"{symbol} - Strategy Performance Breakdown")
                axs[3].set_xticks(x)
                axs[3].set_xticklabels(strategies)
                axs[3].legend()
                axs[3].grid(True)
                
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(f"backtest_results/charts/{symbol}_combined_backtest.png")
            plt.close()
            
            self.logger.info(f"Generated combined backtest charts for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error generating combined backtest charts: {str(e)}")
            
    def generate_summary_report(self):
        """Generate a summary report of all backtest results."""
        self.logger.info("Generating summary report...")
        
        try:
            # Collect results
            summary = {
                "strategies": [],
                "symbols": [],
                "results": []
            }
            
            for key, result in self.backtest_results.items():
                if isinstance(result, dict) and "symbol" in result and "strategy" in result:
                    summary["results"].append({
                        "symbol": result["symbol"],
                        "strategy": result["strategy"],
                        "total_return": result.get("total_return", 0),
                        "win_rate": result.get("win_rate", 0),
                        "total_trades": result.get("total_trades", 0),
                        "profit_factor": result.get("profit_factor", 0),
                        "max_drawdown": result.get("max_drawdown", 0),
                        "sharpe_ratio": result.get("sharpe_ratio", 0)
                    })
                    
                    if result["symbol"] not in summary["symbols"]:
                        summary["symbols"].append(result["symbol"])
                        
                    if result["strategy"] not in summary["strategies"]:
                        summary["strategies"].append(result["strategy"])
                        
            # Generate summary table
            if summary["results"]:
                # Create DataFrame
                df = pd.DataFrame(summary["results"])
                
                # Format percentages
                df["total_return"] = df["total_return"].apply(lambda x: f"{x:.2%}")
                df["win_rate"] = df["win_rate"].apply(lambda x: f"{x:.2%}")
                df["max_drawdown"] = df["max_drawdown"].apply(lambda x: f"{x:.2%}")
                
                # Format other columns
                df["profit_factor"] = df["profit_factor"].apply(lambda x: f"{x:.2f}")
                df["sharpe_ratio"] = df["sharpe_ratio"].apply(lambda x: f"{x:.2f}")
                
                # Save to CSV
                df.to_csv("backtest_results/summary_report.csv", index=False)
                
                # Generate HTML report
                html = """
                <html>
                <head>
                    <title>Backtest Summary Report</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        h1 { color: #333366; }
                        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                        th { background-color: #f2f2f2; }
                        tr:hover { background-color: #f5f5f5; }
                        .positive { color: green; }
                        .negative { color: red; }
                    </style>
                </head>
                <body>
                    <h1>Backtest Summary Report</h1>
                    <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
                    <table>
                        <tr>
                            <th>Symbol</th>
                            <th>Strategy</th>
                            <th>Total Return</th>
                            <th>Win Rate</th>
                            <th>Total Trades</th>
                            <th>Profit Factor</th>
                            <th>Max Drawdown</th>
                            <th>Sharpe Ratio</th>
                        </tr>
                """
                
                for _, row in df.iterrows():
                    html += f"""
                        <tr>
                            <td>{row['symbol']}</td>
                            <td>{row['strategy']}</td>
                            <td class="{'positive' if float(row['total_return'].strip('%')) > 0 else 'negative'}">{row['total_return']}</td>
                            <td>{row['win_rate']}</td>
                            <td>{row['total_trades']}</td>
                            <td>{row['profit_factor']}</td>
                            <td>{row['max_drawdown']}</td>
                            <td>{row['sharpe_ratio']}</td>
                        </tr>
                    """
                    
                html += """
                    </table>
                </body>
                </html>
                """
                
                with open("backtest_results/summary_report.html", "w") as f:
                    f.write(html)
                    
                self.logger.info("Summary report generated successfully")
                
                return {
                    "csv_path": "backtest_results/summary_report.csv",
                    "html_path": "backtest_results/summary_report.html"
                }
            else:
                self.logger.warning("No backtest results available for summary report")
                return None
                
        except Exception as e:
            self.logger.exception(f"Error generating summary report: {str(e)}")
            return None
            
    async def run_full_training(self, symbols: List[str] = None, days: int = 30, interval: str = "1h", optimize: bool = True):
        """
        Run full training process for all strategies.
        
        Args:
            symbols: List of symbols to train on (defaults to config symbols)
            days: Number of days of historical data
            interval: Data interval
            optimize: Whether to optimize strategy parameters
            
        Returns:
            Training results
        """
        self.logger.info("Starting full training process...")
        
        # Use symbols from config if not provided
        if not symbols:
            symbols = self.config.get("symbols", ["BTC-USD-PERP"])
            
        # Ensure symbols is a list
        if isinstance(symbols, str):
            symbols = [symbols]
            
        # Create results structure
        training_results = {
            "symbols": symbols,
            "days": days,
            "interval": interval,
            "optimize": optimize,
            "results": {}
        }
        
        # Process each symbol
        for symbol in symbols:
            self.logger.info(f"Training on {symbol}...")
            symbol_results = {}
            
            # Fetch or generate historical data
            data_success = await self.fetch_historical_data(symbol, days, interval)
            
            if not data_success:
                self.logger.warning(f"Could not fetch historical data for {symbol}, generating simulated data")
                data_success = self.generate_simulated_data(symbol, days, interval)
                
            if not data_success:
                self.logger.error(f"Failed to obtain data for {symbol}, skipping")
                continue
                
            # Run individual strategy backtests
            tc_results = self.backtest_triple_confluence(symbol, optimize)
            ou_results = self.backtest_oracle_update(symbol, optimize)
            
            # Run combined backtest
            combined_results = self.run_combined_backtest(symbol, optimize)
            
            # Store results
            if tc_results:
                symbol_results["triple_confluence"] = {
                    "total_return": tc_results["total_return"],
                    "win_rate": tc_results["win_rate"],
                    "total_trades": tc_results["total_trades"],
                    "sharpe_ratio": tc_results["sharpe_ratio"]
                }
                
            if ou_results:
                symbol_results["oracle_update"] = {
                    "total_return": ou_results["total_return"],
                    "win_rate": ou_results["win_rate"],
                    "total_trades": ou_results["total_trades"],
                    "sharpe_ratio": ou_results["sharpe_ratio"]
                }
                
            if combined_results:
                symbol_results["combined"] = {
                    "total_return": combined_results["total_return"],
                    "win_rate": combined_results["win_rate"],
                    "total_trades": combined_results["total_trades"],
                    "sharpe_ratio": combined_results["sharpe_ratio"]
                }
                
            # Store optimization results if available
            if optimize:
                if f"{symbol}_triple_confluence" in self.optimization_results:
                    symbol_results["triple_confluence_optimization"] = self.optimization_results[f"{symbol}_triple_confluence"]
                    
                if f"{symbol}_oracle_update" in self.optimization_results:
                    symbol_results["oracle_update_optimization"] = self.optimization_results[f"{symbol}_oracle_update"]
                    
            # Store in overall results
            training_results["results"][symbol] = symbol_results
            
        # Generate summary report
        summary_report = self.generate_summary_report()
        if summary_report:
            training_results["summary_report"] = summary_report
            
        # Save training results
        with open("backtest_results/training_results.json", "w") as f:
            # Convert non-serializable objects
            json.dump(training_results, f, indent=2, default=str)
            
        self.logger.info("Full training process completed successfully")
        
        return training_results

async def main():
    """Main entry point."""
    # Create trainer
    trainer = BacktestTrainer("config.json")
    
    # Run full training
    symbols = ["BTC-USD-PERP", "ETH-USD-PERP", "SOL-USD-PERP"]
    await trainer.run_full_training(symbols, days=30, interval="1h", optimize=True)
    
    print("Training completed successfully. Results saved in backtest_results/ directory.")

if __name__ == "__main__":
    asyncio.run(main())
