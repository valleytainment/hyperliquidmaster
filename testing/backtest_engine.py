"""
Backtesting Framework for HyperLiquid Trading Bot

This module provides a comprehensive backtesting framework to evaluate
trading strategies using historical data.
"""

import os
import time
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta

class BacktestEngine:
    """
    Backtesting engine for evaluating trading strategies.
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """
        Initialize backtesting engine.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        # Setup logging
        self.logger = logger or logging.getLogger("BacktestEngine")
        
        # Store configuration
        self.config = config
        
        # Initialize results storage
        self.results = {}
        
        # Default parameters
        self.initial_capital = config.get("initial_capital", 10000.0)
        self.fee_rate = config.get("fee_rate", 0.0005)  # 0.05% default
        self.slippage = config.get("slippage", 0.0002)  # 0.02% default
        
        # Output directory
        self.output_dir = config.get("output_dir", "backtest_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger.info("Backtest engine initialized")
    
    def load_historical_data(self, symbol: str, start_date: str, end_date: str, timeframe: str = "1h") -> pd.DataFrame:
        """
        Load historical data for backtesting.
        
        Args:
            symbol: Symbol to load data for
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Timeframe for data (e.g., "1m", "5m", "1h", "1d")
            
        Returns:
            DataFrame with historical data
        """
        self.logger.info(f"Loading historical data for {symbol} from {start_date} to {end_date} ({timeframe})")
        
        try:
            # Check if data file exists
            data_file = os.path.join(self.output_dir, f"{symbol}_{timeframe}_{start_date}_{end_date}.csv")
            
            if os.path.exists(data_file):
                # Load from file
                self.logger.info(f"Loading data from file: {data_file}")
                df = pd.read_csv(data_file, parse_dates=["timestamp"])
                return df
            
            # If file doesn't exist, generate mock data for testing
            # In a real implementation, this would fetch data from an API or database
            self.logger.info(f"Generating mock data for {symbol}")
            
            # Parse dates
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Determine time delta based on timeframe
            if timeframe == "1m":
                delta = timedelta(minutes=1)
            elif timeframe == "5m":
                delta = timedelta(minutes=5)
            elif timeframe == "15m":
                delta = timedelta(minutes=15)
            elif timeframe == "1h":
                delta = timedelta(hours=1)
            elif timeframe == "4h":
                delta = timedelta(hours=4)
            elif timeframe == "1d":
                delta = timedelta(days=1)
            else:
                delta = timedelta(hours=1)
            
            # Generate timestamps
            timestamps = []
            current = start
            while current <= end:
                timestamps.append(current)
                current += delta
            
            # Generate price data with random walk
            np.random.seed(42)  # For reproducibility
            
            # Initial price depends on symbol
            if symbol == "BTC":
                initial_price = 50000.0
            elif symbol == "ETH":
                initial_price = 3000.0
            elif symbol == "SOL":
                initial_price = 100.0
            else:
                initial_price = 100.0
            
            # Generate prices with random walk and some trend
            price_changes = np.random.normal(0.0001, 0.01, len(timestamps))
            prices = [initial_price]
            
            for i in range(1, len(timestamps)):
                # Add some trend and cyclical patterns
                trend = 0.0002 * np.sin(i / 100)
                cycle = 0.001 * np.sin(i / 1000)
                
                # Calculate new price
                new_price = prices[-1] * (1 + price_changes[i] + trend + cycle)
                prices.append(new_price)
            
            # Create DataFrame
            df = pd.DataFrame({
                "timestamp": timestamps,
                "open": prices,
                "high": [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
                "low": [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
                "close": prices,
                "volume": [np.random.uniform(100, 1000) * p for p in prices]
            })
            
            # Save to file for future use
            df.to_csv(data_file, index=False)
            
            self.logger.info(f"Generated and saved mock data for {symbol} with {len(df)} data points")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading historical data: {e}")
            # Return empty DataFrame
            return pd.DataFrame()
    
    def run_backtest(self, strategy_func: Callable, data: Dict[str, pd.DataFrame], params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run backtest with specified strategy and data.
        
        Args:
            strategy_func: Strategy function that takes data and params and returns signals
            data: Dictionary of DataFrames with historical data for each symbol
            params: Optional parameters for the strategy
            
        Returns:
            Dictionary with backtest results
        """
        start_time = time.time()
        self.logger.info("Starting backtest")
        
        # Initialize parameters
        params = params or {}
        initial_capital = params.get("initial_capital", self.initial_capital)
        fee_rate = params.get("fee_rate", self.fee_rate)
        slippage = params.get("slippage", self.slippage)
        
        # Initialize portfolio
        portfolio = {
            "cash": initial_capital,
            "positions": {},
            "equity": initial_capital,
            "trades": [],
            "equity_curve": []
        }
        
        # Get common timestamps across all symbols
        common_timestamps = self._get_common_timestamps(data)
        
        if not common_timestamps:
            self.logger.error("No common timestamps found across symbols")
            return {"error": "No common timestamps found"}
        
        self.logger.info(f"Running backtest with {len(common_timestamps)} data points")
        
        # Run backtest for each timestamp
        for i, timestamp in enumerate(common_timestamps):
            # Skip first few timestamps to allow for indicator calculation
            if i < params.get("warmup_period", 20):
                continue
            
            # Get current data for each symbol
            current_data = {}
            for symbol, df in data.items():
                # Get data up to current timestamp
                symbol_data = df[df["timestamp"] <= timestamp].copy()
                current_data[symbol] = symbol_data
            
            # Generate signals
            try:
                signals = strategy_func(current_data, params)
            except Exception as e:
                self.logger.error(f"Error generating signals at {timestamp}: {e}")
                signals = {}
            
            # Execute signals
            for symbol, signal in signals.items():
                if symbol not in data:
                    continue
                
                # Get current price
                current_price = self._get_price_at_timestamp(data[symbol], timestamp)
                if current_price is None:
                    continue
                
                # Apply slippage
                if signal.get("action") == "buy":
                    execution_price = current_price * (1 + slippage)
                elif signal.get("action") == "sell":
                    execution_price = current_price * (1 - slippage)
                else:
                    execution_price = current_price
                
                # Execute trade
                self._execute_trade(portfolio, symbol, signal, execution_price, fee_rate, timestamp)
            
            # Update portfolio value
            portfolio_value = portfolio["cash"]
            for symbol, position in portfolio["positions"].items():
                if position["size"] == 0:
                    continue
                
                # Get current price
                current_price = self._get_price_at_timestamp(data[symbol], timestamp)
                if current_price is None:
                    continue
                
                # Update position value
                position["current_price"] = current_price
                position["value"] = position["size"] * current_price
                position["pnl"] = position["value"] - position["cost"]
                
                # Add to portfolio value
                portfolio_value += position["value"]
            
            # Update equity curve
            portfolio["equity"] = portfolio_value
            portfolio["equity_curve"].append({
                "timestamp": timestamp,
                "equity": portfolio_value
            })
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(portfolio, initial_capital)
        
        # Combine results
        results = {
            "portfolio": portfolio,
            "metrics": metrics,
            "params": params,
            "duration": time.time() - start_time
        }
        
        self.logger.info(f"Backtest completed in {results['duration']:.2f} seconds")
        self.logger.info(f"Final equity: ${portfolio['equity']:.2f}")
        self.logger.info(f"Return: {metrics['total_return']:.2f}%")
        self.logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        self.logger.info(f"Max drawdown: {metrics['max_drawdown']:.2f}%")
        
        # Store results
        self.results = results
        
        return results
    
    def _get_common_timestamps(self, data: Dict[str, pd.DataFrame]) -> List[datetime]:
        """
        Get common timestamps across all symbols.
        
        Args:
            data: Dictionary of DataFrames with historical data for each symbol
            
        Returns:
            List of common timestamps
        """
        if not data:
            return []
            
        # Get timestamps for each symbol
        timestamps_by_symbol = {}
        for symbol, df in data.items():
            timestamps_by_symbol[symbol] = set(df["timestamp"].dt.to_pydatetime())
        
        # Find common timestamps
        common_timestamps = set.intersection(*timestamps_by_symbol.values())
        
        # Sort timestamps
        return sorted(common_timestamps)
    
    def _get_price_at_timestamp(self, df: pd.DataFrame, timestamp: datetime) -> Optional[float]:
        """
        Get price at specific timestamp.
        
        Args:
            df: DataFrame with historical data
            timestamp: Timestamp to get price for
            
        Returns:
            Price at timestamp, or None if not found
        """
        # Find row with matching timestamp
        row = df[df["timestamp"] == timestamp]
        if row.empty:
            return None
            
        # Return close price
        return row.iloc[0]["close"]
    
    def _execute_trade(self, portfolio: Dict[str, Any], symbol: str, signal: Dict[str, Any], price: float, fee_rate: float, timestamp: datetime):
        """
        Execute trade in portfolio.
        
        Args:
            portfolio: Portfolio dictionary
            symbol: Symbol to trade
            signal: Signal dictionary with action and size
            price: Execution price
            fee_rate: Fee rate
            timestamp: Timestamp of trade
        """
        action = signal.get("action")
        size = signal.get("size", 0)
        
        if action not in ["buy", "sell"]:
            return
            
        # Initialize position if not exists
        if symbol not in portfolio["positions"]:
            portfolio["positions"][symbol] = {
                "size": 0,
                "cost": 0,
                "value": 0,
                "current_price": price,
                "pnl": 0
            }
        
        position = portfolio["positions"][symbol]
        
        # Calculate trade value and fees
        trade_value = size * price
        fees = trade_value * fee_rate
        
        if action == "buy":
            # Check if enough cash
            if portfolio["cash"] < trade_value + fees:
                # Adjust size based on available cash
                available_cash = portfolio["cash"] - fees
                size = available_cash / price
                trade_value = size * price
                fees = trade_value * fee_rate
            
            # Update position
            new_size = position["size"] + size
            new_cost = position["cost"] + trade_value
            
            # Update portfolio
            portfolio["cash"] -= (trade_value + fees)
            position["size"] = new_size
            position["cost"] = new_cost
            position["current_price"] = price
            position["value"] = new_size * price
            position["pnl"] = position["value"] - position["cost"]
            
        elif action == "sell":
            # Check if enough position size
            if position["size"] < size:
                size = position["size"]
                trade_value = size * price
                fees = trade_value * fee_rate
            
            # Calculate realized P&L
            if position["size"] > 0:
                cost_per_unit = position["cost"] / position["size"]
                realized_pnl = (price - cost_per_unit) * size
            else:
                realized_pnl = 0
            
            # Update position
            new_size = position["size"] - size
            if new_size > 0:
                # Reduce cost proportionally
                new_cost = position["cost"] * (new_size / position["size"])
            else:
                new_cost = 0
            
            # Update portfolio
            portfolio["cash"] += (trade_value - fees)
            position["size"] = new_size
            position["cost"] = new_cost
            position["current_price"] = price
            position["value"] = new_size * price
            position["pnl"] = position["value"] - position["cost"]
        
        # Record trade
        portfolio["trades"].append({
            "timestamp": timestamp,
            "symbol": symbol,
            "action": action,
            "size": size,
            "price": price,
            "value": trade_value,
            "fees": fees,
            "cash_after": portfolio["cash"]
        })
    
    def _calculate_performance_metrics(self, portfolio: Dict[str, Any], initial_capital: float) -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Args:
            portfolio: Portfolio dictionary
            initial_capital: Initial capital
            
        Returns:
            Dictionary with performance metrics
        """
        # Extract equity curve
        equity_curve = pd.DataFrame(portfolio["equity_curve"])
        if equity_curve.empty:
            return {
                "total_return": 0,
                "annualized_return": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "profit_factor": 0
            }
        
        # Calculate returns
        equity_curve["return"] = equity_curve["equity"].pct_change()
        
        # Calculate metrics
        final_equity = portfolio["equity"]
        total_return = (final_equity / initial_capital - 1) * 100
        
        # Calculate annualized return
        days = (equity_curve["timestamp"].max() - equity_curve["timestamp"].min()).days
        if days > 0:
            annualized_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100
        else:
            annualized_return = 0
        
        # Calculate Sharpe ratio
        if len(equity_curve) > 1:
            returns = equity_curve["return"].dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5)  # Annualized
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Calculate maximum drawdown
        equity_curve["cummax"] = equity_curve["equity"].cummax()
        equity_curve["drawdown"] = (equity_curve["equity"] / equity_curve["cummax"] - 1) * 100
        max_drawdown = equity_curve["drawdown"].min()
        
        # Calculate win rate and profit factor
        trades = portfolio["trades"]
        if trades:
            # Calculate P&L for each trade
            for i, trade in enumerate(trades):
                if i == 0:
                    trade["pnl"] = 0
                    continue
                
                prev_trade = trades[i-1]
                if trade["action"] == "buy":
                    # Buy increases cost
                    trade["pnl"] = 0
                elif trade["action"] == "sell":
                    # Sell realizes P&L
                    # Simple approximation, not accounting for multiple buys
                    trade["pnl"] = trade["value"] - prev_trade["value"]
            
            # Filter out trades with P&L
            pnl_trades = [t for t in trades if "pnl" in t and t["action"] == "sell"]
            
            if pnl_trades:
                winning_trades = [t for t in pnl_trades if t["pnl"] > 0]
                losing_trades = [t for t in pnl_trades if t["pnl"] <= 0]
                
                win_rate = len(winning_trades) / len(pnl_trades) * 100
                
                total_profit = sum(t["pnl"] for t in winning_trades)
                total_loss = abs(sum(t["pnl"] for t in losing_trades))
                
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            else:
                win_rate = 0
                profit_factor = 0
        else:
            win_rate = 0
            profit_factor = 0
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor
        }
    
    def plot_results(self, filename: str = None):
        """
        Plot backtest results.
        
        Args:
            filename: Optional filename to save plot
        """
        if not self.results or "portfolio" not in self.results:
            self.logger.warning("No backtest results to plot")
            return
            
        # Extract equity curve
        equity_curve = pd.DataFrame(self.results["portfolio"]["equity_curve"])
        if equity_curve.empty:
            self.logger.warning("Empty equity curve, nothing to plot")
            return
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(equity_curve["timestamp"], equity_curve["equity"])
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        
        # Plot drawdown
        plt.subplot(2, 1, 2)
        equity_curve["cummax"] = equity_curve["equity"].cummax()
        equity_curve["drawdown"] = (equity_curve["equity"] / equity_curve["cummax"] - 1) * 100
        plt.fill_between(equity_curve["timestamp"], equity_curve["drawdown"], 0, color="red", alpha=0.3)
        plt.title("Drawdown")
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%)")
        plt.grid(True)
        
        # Add metrics as text
        metrics = self.results["metrics"]
        plt.figtext(0.01, 0.01, f"Total Return: {metrics['total_return']:.2f}%\n"
                             f"Annualized Return: {metrics['annualized_return']:.2f}%\n"
                             f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                             f"Max Drawdown: {metrics['max_drawdown']:.2f}%\n"
                             f"Win Rate: {metrics['win_rate']:.2f}%\n"
                             f"Profit Factor: {metrics['profit_factor']:.2f}",
                  fontsize=10, bbox=dict(facecolor="white", alpha=0.5))
        
        plt.tight_layout()
        
        # Save or show plot
        if filename:
            plt.savefig(filename)
            self.logger.info(f"Plot saved to {filename}")
        else:
            plt.show()
    
    def save_results(self, filename: str = None):
        """
        Save backtest results to file.
        
        Args:
            filename: Optional filename to save results
        """
        if not self.results:
            self.logger.warning("No backtest results to save")
            return
            
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"backtest_results_{timestamp}.json")
        
        # Create a copy of results for serialization
        results_copy = self.results.copy()
        
        # Convert timestamps to strings
        if "portfolio" in results_copy and "equity_curve" in results_copy["portfolio"]:
            for point in results_copy["portfolio"]["equity_curve"]:
                if isinstance(point["timestamp"], datetime):
                    point["timestamp"] = point["timestamp"].isoformat()
        
        if "portfolio" in results_copy and "trades" in results_copy["portfolio"]:
            for trade in results_copy["portfolio"]["trades"]:
                if isinstance(trade["timestamp"], datetime):
                    trade["timestamp"] = trade["timestamp"].isoformat()
        
        # Save to file
        with open(filename, "w") as f:
            json.dump(results_copy, f, indent=4)
            
        self.logger.info(f"Results saved to {filename}")
        return filename
    
    def optimize_strategy(self, strategy_func: Callable, data: Dict[str, pd.DataFrame], param_grid: Dict[str, List[Any]], base_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            strategy_func: Strategy function that takes data and params and returns signals
            data: Dictionary of DataFrames with historical data for each symbol
            param_grid: Dictionary of parameter names and lists of values to try
            base_params: Base parameters to use for all tests
            
        Returns:
            Dictionary with optimization results
        """
        self.logger.info("Starting strategy optimization")
        start_time = time.time()
        
        # Initialize base parameters
        base_params = base_params or {}
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        self.logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        # Run backtest for each parameter combination
        results = []
        for i, params in enumerate(param_combinations):
            # Combine with base parameters
            test_params = {**base_params, **params}
            
            # Run backtest
            self.logger.info(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
            backtest_result = self.run_backtest(strategy_func, data, test_params)
            
            # Store result
            results.append({
                "params": params,
                "metrics": backtest_result["metrics"]
            })
        
        # Find best parameters
        if results:
            # Sort by total return (could use other metrics)
            results.sort(key=lambda x: x["metrics"]["total_return"], reverse=True)
            best_result = results[0]
            
            self.logger.info(f"Optimization completed in {time.time() - start_time:.2f} seconds")
            self.logger.info(f"Best parameters: {best_result['params']}")
            self.logger.info(f"Best return: {best_result['metrics']['total_return']:.2f}%")
            
            # Return optimization results
            return {
                "best_params": best_result["params"],
                "best_metrics": best_result["metrics"],
                "all_results": results,
                "duration": time.time() - start_time
            }
        else:
            self.logger.warning("No optimization results")
            return {
                "error": "No optimization results",
                "duration": time.time() - start_time
            }
    
    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generate all combinations of parameters.
        
        Args:
            param_grid: Dictionary of parameter names and lists of values
            
        Returns:
            List of parameter dictionaries
        """
        # Get parameter names and values
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Generate combinations
        combinations = []
        
        def generate_combinations(index, current_params):
            if index == len(param_names):
                combinations.append(current_params.copy())
                return
                
            param_name = param_names[index]
            for value in param_values[index]:
                current_params[param_name] = value
                generate_combinations(index + 1, current_params)
        
        generate_combinations(0, {})
        return combinations
