"""
Enhanced Backtesting Framework with Real Market Data Support

This module provides a robust backtesting framework that properly handles
mixed real and synthetic market data, with appropriate risk adjustments
and performance metrics for real-money trading optimization.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterGrid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("real_data_backtesting.log")
    ]
)
logger = logging.getLogger(__name__)

class EnhancedBacktester:
    """
    Enhanced backtesting framework with real market data support.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the enhanced backtester.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.base_dir = "real_market_data"
        self.results_dir = "backtest_results"
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/charts", exist_ok=True)
        
        # Risk adjustment factors for synthetic data
        self.risk_adjustment_factors = {
            "position_size": 0.7,  # Reduce position size by 30% for synthetic data
            "stop_loss": 1.2,      # Increase stop loss distance by 20% for synthetic data
            "take_profit": 1.2,    # Increase take profit distance by 20% for synthetic data
            "max_drawdown": 0.8    # Reduce max drawdown threshold by 20% for synthetic data
        }
        
        # Performance metrics to track
        self.performance_metrics = [
            "total_return",
            "annualized_return",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "avg_profit_per_trade",
            "avg_loss_per_trade",
            "risk_adjusted_return"
        ]
        
        logger.info("Enhanced Backtester initialized")
        
    def run_backtest(self, 
                    strategy: Callable, 
                    symbols: List[str], 
                    interval: str, 
                    start_date: str = None, 
                    end_date: str = None,
                    params: Dict = None,
                    initial_capital: float = 10000.0,
                    position_sizing: str = "fixed",
                    position_size: float = 0.1,
                    max_positions: int = 3,
                    use_stop_loss: bool = True,
                    stop_loss_pct: float = 0.05,
                    use_take_profit: bool = True,
                    take_profit_pct: float = 0.15,
                    max_drawdown_pct: float = 0.25,
                    trading_fee_pct: float = 0.0006,
                    slippage_pct: float = 0.0005) -> Dict:
        """
        Run a backtest with the specified strategy and parameters.
        
        Args:
            strategy: Strategy function to backtest
            symbols: List of symbols to backtest on
            interval: Interval to backtest on
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            params: Strategy parameters
            initial_capital: Initial capital for backtest
            position_sizing: Position sizing method ("fixed", "percent", "risk", "kelly")
            position_size: Position size (interpretation depends on position_sizing)
            max_positions: Maximum number of concurrent positions
            use_stop_loss: Whether to use stop loss
            stop_loss_pct: Stop loss percentage
            use_take_profit: Whether to use take profit
            take_profit_pct: Take profit percentage
            max_drawdown_pct: Maximum drawdown percentage
            trading_fee_pct: Trading fee percentage
            slippage_pct: Slippage percentage
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running backtest for symbols: {symbols}, interval: {interval}")
        
        # Initialize results
        results = {
            "strategy": strategy.__name__,
            "symbols": symbols,
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date,
            "params": params,
            "initial_capital": initial_capital,
            "position_sizing": position_sizing,
            "position_size": position_size,
            "max_positions": max_positions,
            "use_stop_loss": use_stop_loss,
            "stop_loss_pct": stop_loss_pct,
            "use_take_profit": use_take_profit,
            "take_profit_pct": take_profit_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "trading_fee_pct": trading_fee_pct,
            "slippage_pct": slippage_pct,
            "timestamp": datetime.now().isoformat(),
            "performance": {},
            "trades": [],
            "equity_curve": [],
            "drawdowns": [],
            "positions": [],
            "data_quality": {}
        }
        
        # Load and prepare data
        data = {}
        data_quality = {}
        
        for symbol in symbols:
            symbol_data, quality = self.load_data(symbol, interval, start_date, end_date)
            data[symbol] = symbol_data
            data_quality[symbol] = quality
            
            # Apply risk adjustments based on data quality
            if quality["synthetic_ratio"] > 0.5:
                logger.warning(f"High synthetic data ratio ({quality['synthetic_ratio']:.2f}) for {symbol}, applying risk adjustments")
                
                # Adjust risk parameters
                if position_sizing == "fixed" or position_sizing == "percent":
                    position_size *= self.risk_adjustment_factors["position_size"]
                
                if use_stop_loss:
                    stop_loss_pct *= self.risk_adjustment_factors["stop_loss"]
                    
                if use_take_profit:
                    take_profit_pct *= self.risk_adjustment_factors["take_profit"]
                    
                max_drawdown_pct *= self.risk_adjustment_factors["max_drawdown"]
                
        # Store data quality in results
        results["data_quality"] = data_quality
        
        # Initialize portfolio
        portfolio = {
            "cash": initial_capital,
            "positions": {},
            "equity": initial_capital,
            "high_watermark": initial_capital,
            "drawdown": 0.0,
            "max_drawdown": 0.0
        }
        
        # Initialize trade tracking
        trades = []
        equity_curve = []
        drawdowns = []
        
        # Get common date range across all symbols
        common_dates = self.get_common_dates(data)
        
        # Main backtest loop
        for date_idx, date in enumerate(common_dates):
            # Update portfolio value
            portfolio_value = portfolio["cash"]
            for symbol, position in portfolio["positions"].items():
                current_price = self.get_price_at_date(data[symbol], date, "close")
                position["current_value"] = position["quantity"] * current_price
                portfolio_value += position["current_value"]
                
            # Update equity curve and drawdown
            portfolio["equity"] = portfolio_value
            equity_curve.append({
                "date": date,
                "equity": portfolio_value
            })
            
            # Calculate drawdown
            if portfolio_value > portfolio["high_watermark"]:
                portfolio["high_watermark"] = portfolio_value
                portfolio["drawdown"] = 0.0
            else:
                portfolio["drawdown"] = (portfolio["high_watermark"] - portfolio_value) / portfolio["high_watermark"]
                
            if portfolio["drawdown"] > portfolio["max_drawdown"]:
                portfolio["max_drawdown"] = portfolio["drawdown"]
                
            drawdowns.append({
                "date": date,
                "drawdown": portfolio["drawdown"]
            })
            
            # Check for stop loss and take profit
            for symbol, position in list(portfolio["positions"].items()):
                current_price = self.get_price_at_date(data[symbol], date, "close")
                
                # Check stop loss
                if use_stop_loss and current_price <= position["stop_loss"]:
                    # Close position at stop loss
                    trade_result = self.close_position(portfolio, symbol, current_price, date, "stop_loss")
                    trades.append(trade_result)
                    continue
                    
                # Check take profit
                if use_take_profit and current_price >= position["take_profit"]:
                    # Close position at take profit
                    trade_result = self.close_position(portfolio, symbol, current_price, date, "take_profit")
                    trades.append(trade_result)
                    continue
                    
            # Check for max drawdown circuit breaker
            if portfolio["drawdown"] >= max_drawdown_pct:
                logger.warning(f"Max drawdown ({portfolio['drawdown']:.2f}) exceeded threshold ({max_drawdown_pct:.2f}), closing all positions")
                
                # Close all positions
                for symbol, position in list(portfolio["positions"].items()):
                    current_price = self.get_price_at_date(data[symbol], date, "close")
                    trade_result = self.close_position(portfolio, symbol, current_price, date, "max_drawdown")
                    trades.append(trade_result)
                    
                # Skip to next date
                continue
                
            # Generate trading signals
            for symbol in symbols:
                # Skip if already at max positions
                if len(portfolio["positions"]) >= max_positions:
                    break
                    
                # Skip if already have position in this symbol
                if symbol in portfolio["positions"]:
                    continue
                    
                # Prepare data for strategy
                symbol_data_slice = self.prepare_data_slice(data[symbol], date_idx)
                
                # Generate signal
                signal = strategy(symbol_data_slice, params)
                
                # Process signal
                if signal["action"] == "buy":
                    # Calculate position size
                    current_price = self.get_price_at_date(data[symbol], date, "close")
                    
                    if position_sizing == "fixed":
                        # Fixed dollar amount
                        position_value = position_size
                    elif position_sizing == "percent":
                        # Percentage of portfolio
                        position_value = portfolio_value * position_size
                    elif position_sizing == "risk":
                        # Risk-based position sizing
                        risk_amount = portfolio_value * position_size
                        position_value = risk_amount / stop_loss_pct
                    elif position_sizing == "kelly":
                        # Kelly criterion
                        win_rate = 0.5  # Default
                        if len(trades) > 0:
                            win_trades = [t for t in trades if t["profit_pct"] > 0]
                            win_rate = len(win_trades) / len(trades)
                            
                        avg_win = 0.1  # Default
                        avg_loss = 0.05  # Default
                        if len(win_trades) > 0:
                            avg_win = sum([t["profit_pct"] for t in win_trades]) / len(win_trades)
                            
                        lose_trades = [t for t in trades if t["profit_pct"] <= 0]
                        if len(lose_trades) > 0:
                            avg_loss = sum([abs(t["profit_pct"]) for t in lose_trades]) / len(lose_trades)
                            
                        kelly_pct = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
                        kelly_pct = max(0.01, min(0.2, kelly_pct))  # Limit between 1% and 20%
                        position_value = portfolio_value * kelly_pct
                    else:
                        # Default to fixed
                        position_value = position_size
                        
                    # Check if enough cash
                    if position_value > portfolio["cash"]:
                        position_value = portfolio["cash"]
                        
                    # Calculate quantity
                    quantity = position_value / current_price
                    
                    # Calculate fees
                    fees = position_value * trading_fee_pct
                    
                    # Calculate slippage
                    slippage = position_value * slippage_pct
                    
                    # Calculate entry price with slippage
                    entry_price = current_price * (1 + slippage_pct)
                    
                    # Calculate stop loss and take profit
                    stop_loss_price = entry_price * (1 - stop_loss_pct)
                    take_profit_price = entry_price * (1 + take_profit_pct)
                    
                    # Open position
                    portfolio["positions"][symbol] = {
                        "quantity": quantity,
                        "entry_price": entry_price,
                        "entry_date": date,
                        "current_value": quantity * current_price,
                        "stop_loss": stop_loss_price,
                        "take_profit": take_profit_price,
                        "fees": fees,
                        "slippage": slippage
                    }
                    
                    # Deduct cash
                    portfolio["cash"] -= (position_value + fees + slippage)
                    
                    logger.info(f"Opened position in {symbol} at {entry_price:.2f}, quantity: {quantity:.6f}, value: {position_value:.2f}")
                    
        # Close any remaining positions at the end
        final_date = common_dates[-1]
        for symbol, position in list(portfolio["positions"].items()):
            current_price = self.get_price_at_date(data[symbol], final_date, "close")
            trade_result = self.close_position(portfolio, symbol, current_price, final_date, "end_of_backtest")
            trades.append(trade_result)
            
        # Calculate performance metrics
        performance = self.calculate_performance_metrics(initial_capital, portfolio["equity"], equity_curve, trades)
        
        # Store results
        results["performance"] = performance
        results["trades"] = trades
        results["equity_curve"] = equity_curve
        results["drawdowns"] = drawdowns
        results["final_equity"] = portfolio["equity"]
        
        # Generate charts
        self.generate_charts(results)
        
        # Save results
        self.save_results(results)
        
        logger.info(f"Backtest completed with final equity: {portfolio['equity']:.2f}, return: {performance['total_return']:.2f}%")
        return results
        
    def run_parameter_optimization(self, 
                                  strategy: Callable, 
                                  symbols: List[str], 
                                  interval: str, 
                                  param_grid: Dict,
                                  start_date: str = None, 
                                  end_date: str = None,
                                  initial_capital: float = 10000.0,
                                  position_sizing: str = "fixed",
                                  position_size: float = 0.1,
                                  max_positions: int = 3,
                                  use_stop_loss: bool = True,
                                  stop_loss_pct: float = 0.05,
                                  use_take_profit: bool = True,
                                  take_profit_pct: float = 0.15,
                                  max_drawdown_pct: float = 0.25,
                                  trading_fee_pct: float = 0.0006,
                                  slippage_pct: float = 0.0005,
                                  optimization_metric: str = "sharpe_ratio") -> Dict:
        """
        Run parameter optimization for a strategy.
        
        Args:
            strategy: Strategy function to optimize
            symbols: List of symbols to backtest on
            interval: Interval to backtest on
            param_grid: Dictionary of parameter grids to search
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            initial_capital: Initial capital for backtest
            position_sizing: Position sizing method ("fixed", "percent", "risk", "kelly")
            position_size: Position size (interpretation depends on position_sizing)
            max_positions: Maximum number of concurrent positions
            use_stop_loss: Whether to use stop loss
            stop_loss_pct: Stop loss percentage
            use_take_profit: Whether to use take profit
            take_profit_pct: Take profit percentage
            max_drawdown_pct: Maximum drawdown percentage
            trading_fee_pct: Trading fee percentage
            slippage_pct: Slippage percentage
            optimization_metric: Metric to optimize for
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Running parameter optimization for strategy: {strategy.__name__}")
        
        # Generate parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        # Run backtest for each parameter combination
        results = []
        for params in param_combinations:
            logger.info(f"Testing parameters: {params}")
            
            # Run backtest
            backtest_result = self.run_backtest(
                strategy=strategy,
                symbols=symbols,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                params=params,
                initial_capital=initial_capital,
                position_sizing=position_sizing,
                position_size=position_size,
                max_positions=max_positions,
                use_stop_loss=use_stop_loss,
                stop_loss_pct=stop_loss_pct,
                use_take_profit=use_take_profit,
                take_profit_pct=take_profit_pct,
                max_drawdown_pct=max_drawdown_pct,
                trading_fee_pct=trading_fee_pct,
                slippage_pct=slippage_pct
            )
            
            # Store result
            results.append({
                "params": params,
                "performance": backtest_result["performance"],
                "final_equity": backtest_result["final_equity"]
            })
            
        # Sort results by optimization metric
        results.sort(key=lambda x: x["performance"][optimization_metric], reverse=True)
        
        # Get best parameters
        best_params = results[0]["params"]
        best_performance = results[0]["performance"]
        
        # Generate optimization report
        optimization_results = {
            "strategy": strategy.__name__,
            "symbols": symbols,
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date,
            "optimization_metric": optimization_metric,
            "best_params": best_params,
            "best_performance": best_performance,
            "all_results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save optimization results
        self.save_optimization_results(optimization_results)
        
        logger.info(f"Parameter optimization completed. Best parameters: {best_params}")
        logger.info(f"Best {optimization_metric}: {best_performance[optimization_metric]}")
        
        return optimization_results
        
    def run_walk_forward_optimization(self, 
                                     strategy: Callable, 
                                     symbols: List[str], 
                                     interval: str, 
                                     param_grid: Dict,
                                     start_date: str = None, 
                                     end_date: str = None,
                                     window_size: int = 60,
                                     step_size: int = 30,
                                     initial_capital: float = 10000.0,
                                     position_sizing: str = "fixed",
                                     position_size: float = 0.1,
                                     max_positions: int = 3,
                                     use_stop_loss: bool = True,
                                     stop_loss_pct: float = 0.05,
                                     use_take_profit: bool = True,
                                     take_profit_pct: float = 0.15,
                                     max_drawdown_pct: float = 0.25,
                                     trading_fee_pct: float = 0.0006,
                                     slippage_pct: float = 0.0005,
                                     optimization_metric: str = "sharpe_ratio") -> Dict:
        """
        Run walk-forward optimization for a strategy.
        
        Args:
            strategy: Strategy function to optimize
            symbols: List of symbols to backtest on
            interval: Interval to backtest on
            param_grid: Dictionary of parameter grids to search
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            window_size: Size of in-sample window in days
            step_size: Size of out-of-sample window in days
            initial_capital: Initial capital for backtest
            position_sizing: Position sizing method ("fixed", "percent", "risk", "kelly")
            position_size: Position size (interpretation depends on position_sizing)
            max_positions: Maximum number of concurrent positions
            use_stop_loss: Whether to use stop loss
            stop_loss_pct: Stop loss percentage
            use_take_profit: Whether to use take profit
            take_profit_pct: Take profit percentage
            max_drawdown_pct: Maximum drawdown percentage
            trading_fee_pct: Trading fee percentage
            slippage_pct: Slippage percentage
            optimization_metric: Metric to optimize for
            
        Returns:
            Dictionary with walk-forward optimization results
        """
        logger.info(f"Running walk-forward optimization for strategy: {strategy.__name__}")
        
        # Load data
        data = {}
        for symbol in symbols:
            symbol_data, _ = self.load_data(symbol, interval, start_date, end_date)
            data[symbol] = symbol_data
            
        # Get common date range
        common_dates = self.get_common_dates(data)
        
        # Convert start_date and end_date to datetime if provided
        if start_date:
            start_dt = pd.to_datetime(start_date)
            common_dates = [d for d in common_dates if d >= start_dt]
            
        if end_date:
            end_dt = pd.to_datetime(end_date)
            common_dates = [d for d in common_dates if d <= end_dt]
            
        # Initialize results
        walk_forward_results = {
            "strategy": strategy.__name__,
            "symbols": symbols,
            "interval": interval,
            "window_size": window_size,
            "step_size": step_size,
            "optimization_metric": optimization_metric,
            "windows": [],
            "trades": [],
            "equity_curve": [],
            "drawdowns": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Initialize portfolio
        portfolio = {
            "cash": initial_capital,
            "positions": {},
            "equity": initial_capital,
            "high_watermark": initial_capital,
            "drawdown": 0.0,
            "max_drawdown": 0.0
        }
        
        # Initialize tracking
        trades = []
        equity_curve = []
        drawdowns = []
        
        # Walk-forward loop
        for i in range(0, len(common_dates) - window_size - step_size, step_size):
            # Define in-sample and out-of-sample periods
            in_sample_start = common_dates[i]
            in_sample_end = common_dates[i + window_size - 1]
            out_sample_start = common_dates[i + window_size]
            out_sample_end = common_dates[min(i + window_size + step_size - 1, len(common_dates) - 1)]
            
            logger.info(f"Window {i//step_size + 1}: In-sample {in_sample_start} to {in_sample_end}, Out-of-sample {out_sample_start} to {out_sample_end}")
            
            # Optimize parameters on in-sample data
            optimization_results = self.run_parameter_optimization(
                strategy=strategy,
                symbols=symbols,
                interval=interval,
                param_grid=param_grid,
                start_date=in_sample_start.strftime("%Y-%m-%d"),
                end_date=in_sample_end.strftime("%Y-%m-%d"),
                initial_capital=initial_capital,
                position_sizing=position_sizing,
                position_size=position_size,
                max_positions=max_positions,
                use_stop_loss=use_stop_loss,
                stop_loss_pct=stop_loss_pct,
                use_take_profit=use_take_profit,
                take_profit_pct=take_profit_pct,
                max_drawdown_pct=max_drawdown_pct,
                trading_fee_pct=trading_fee_pct,
                slippage_pct=slippage_pct,
                optimization_metric=optimization_metric
            )
            
            # Get best parameters
            best_params = optimization_results["best_params"]
            
            # Run backtest on out-of-sample data with best parameters
            out_sample_results = self.run_backtest(
                strategy=strategy,
                symbols=symbols,
                interval=interval,
                start_date=out_sample_start.strftime("%Y-%m-%d"),
                end_date=out_sample_end.strftime("%Y-%m-%d"),
                params=best_params,
                initial_capital=portfolio["equity"],  # Use current equity
                position_sizing=position_sizing,
                position_size=position_size,
                max_positions=max_positions,
                use_stop_loss=use_stop_loss,
                stop_loss_pct=stop_loss_pct,
                use_take_profit=use_take_profit,
                take_profit_pct=take_profit_pct,
                max_drawdown_pct=max_drawdown_pct,
                trading_fee_pct=trading_fee_pct,
                slippage_pct=slippage_pct
            )
            
            # Update portfolio
            portfolio["equity"] = out_sample_results["final_equity"]
            portfolio["cash"] = out_sample_results["final_equity"]  # Reset positions for next window
            portfolio["positions"] = {}
            
            # Update high watermark and drawdown
            if portfolio["equity"] > portfolio["high_watermark"]:
                portfolio["high_watermark"] = portfolio["equity"]
                portfolio["drawdown"] = 0.0
            else:
                portfolio["drawdown"] = (portfolio["high_watermark"] - portfolio["equity"]) / portfolio["high_watermark"]
                
            if portfolio["drawdown"] > portfolio["max_drawdown"]:
                portfolio["max_drawdown"] = portfolio["drawdown"]
                
            # Store window results
            walk_forward_results["windows"].append({
                "window": i//step_size + 1,
                "in_sample_start": in_sample_start.strftime("%Y-%m-%d"),
                "in_sample_end": in_sample_end.strftime("%Y-%m-%d"),
                "out_sample_start": out_sample_start.strftime("%Y-%m-%d"),
                "out_sample_end": out_sample_end.strftime("%Y-%m-%d"),
                "best_params": best_params,
                "in_sample_performance": optimization_results["best_performance"],
                "out_sample_performance": out_sample_results["performance"]
            })
            
            # Append trades and equity curve
            trades.extend(out_sample_results["trades"])
            equity_curve.extend(out_sample_results["equity_curve"])
            drawdowns.extend(out_sample_results["drawdowns"])
            
        # Store results
        walk_forward_results["trades"] = trades
        walk_forward_results["equity_curve"] = equity_curve
        walk_forward_results["drawdowns"] = drawdowns
        walk_forward_results["final_equity"] = portfolio["equity"]
        
        # Calculate overall performance
        overall_performance = self.calculate_performance_metrics(initial_capital, portfolio["equity"], equity_curve, trades)
        walk_forward_results["performance"] = overall_performance
        
        # Generate charts
        self.generate_walk_forward_charts(walk_forward_results)
        
        # Save results
        self.save_walk_forward_results(walk_forward_results)
        
        logger.info(f"Walk-forward optimization completed with final equity: {portfolio['equity']:.2f}, return: {overall_performance['total_return']:.2f}%")
        return walk_forward_results
        
    def load_data(self, symbol: str, interval: str, start_date: str = None, end_date: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Load data for a symbol and interval.
        
        Args:
            symbol: Symbol to load data for
            interval: Interval to load data for
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)
            
        Returns:
            Tuple of (DataFrame with data, Dictionary with data quality metrics)
        """
        logger.info(f"Loading data for {symbol}, interval {interval}")
        
        # Check if processed data exists
        processed_path = f"{self.base_dir}/processed/{symbol}_{interval}_processed.csv"
        if os.path.exists(processed_path):
            df = pd.read_csv(processed_path)
            
            # Ensure timestamp is datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                
            # Filter by date range if provided
            if start_date:
                start_dt = pd.to_datetime(start_date)
                df = df[df["timestamp"] >= start_dt]
                
            if end_date:
                end_dt = pd.to_datetime(end_date)
                df = df[df["timestamp"] <= end_dt]
                
            # Check if data is synthetic
            is_synthetic = False
            synthetic_path = f"{self.base_dir}/raw/{symbol}_{interval}_synthetic_klines.csv"
            if os.path.exists(synthetic_path):
                is_synthetic = True
                
            # Check if alternative data was used
            alternative_sources = []
            for source in ["binance", "coingecko"]:
                alt_path = f"{self.base_dir}/raw/{symbol}_{interval}_{source}_klines.csv"
                if os.path.exists(alt_path):
                    alternative_sources.append(source)
                    
            # Calculate data quality metrics
            data_quality = {
                "is_synthetic": is_synthetic,
                "alternative_sources": alternative_sources,
                "synthetic_ratio": 1.0 if is_synthetic else 0.0,
                "missing_values": df.isnull().sum().sum() / (df.shape[0] * df.shape[1]),
                "data_points": df.shape[0],
                "start_date": df["timestamp"].min().strftime("%Y-%m-%d"),
                "end_date": df["timestamp"].max().strftime("%Y-%m-%d")
            }
            
            logger.info(f"Loaded {df.shape[0]} rows of data for {symbol}, interval {interval}")
            return df, data_quality
            
        # If processed data doesn't exist, check raw data
        raw_path = f"{self.base_dir}/raw/{symbol}_{interval}_klines.csv"
        if os.path.exists(raw_path):
            df = pd.read_csv(raw_path)
            
            # Ensure timestamp is datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                
            # Filter by date range if provided
            if start_date:
                start_dt = pd.to_datetime(start_date)
                df = df[df["timestamp"] >= start_dt]
                
            if end_date:
                end_dt = pd.to_datetime(end_date)
                df = df[df["timestamp"] <= end_dt]
                
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Check if data is synthetic
            is_synthetic = False
            synthetic_path = f"{self.base_dir}/raw/{symbol}_{interval}_synthetic_klines.csv"
            if os.path.exists(synthetic_path):
                is_synthetic = True
                
            # Check if alternative data was used
            alternative_sources = []
            for source in ["binance", "coingecko"]:
                alt_path = f"{self.base_dir}/raw/{symbol}_{interval}_{source}_klines.csv"
                if os.path.exists(alt_path):
                    alternative_sources.append(source)
                    
            # Calculate data quality metrics
            data_quality = {
                "is_synthetic": is_synthetic,
                "alternative_sources": alternative_sources,
                "synthetic_ratio": 1.0 if is_synthetic else 0.0,
                "missing_values": df.isnull().sum().sum() / (df.shape[0] * df.shape[1]),
                "data_points": df.shape[0],
                "start_date": df["timestamp"].min().strftime("%Y-%m-%d"),
                "end_date": df["timestamp"].max().strftime("%Y-%m-%d")
            }
            
            logger.info(f"Loaded {df.shape[0]} rows of data for {symbol}, interval {interval}")
            return df, data_quality
            
        # If no data exists, generate synthetic data
        logger.warning(f"No data found for {symbol}, interval {interval}, generating synthetic data")
        
        # Generate synthetic data
        df = self.generate_backtest_data(symbol, interval, start_date, end_date)
        
        # Calculate data quality metrics
        data_quality = {
            "is_synthetic": True,
            "alternative_sources": [],
            "synthetic_ratio": 1.0,
            "missing_values": 0.0,
            "data_points": df.shape[0],
            "start_date": df["timestamp"].min().strftime("%Y-%m-%d"),
            "end_date": df["timestamp"].max().strftime("%Y-%m-%d")
        }
        
        logger.info(f"Generated {df.shape[0]} rows of synthetic data for {symbol}, interval {interval}")
        return df, data_quality
        
    def generate_backtest_data(self, symbol: str, interval: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Generate synthetic data for a symbol and interval.
        
        Args:
            symbol: Symbol to generate data for
            interval: Interval to generate data for
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)
            
        Returns:
            DataFrame with synthetic data
        """
        logger.info(f"Generating synthetic data for {symbol}, interval {interval}")
        
        # Determine start and end dates
        if end_date:
            end_dt = pd.to_datetime(end_date)
        else:
            end_dt = datetime.now()
            
        if start_date:
            start_dt = pd.to_datetime(start_date)
        else:
            # Default to 30 days
            start_dt = end_dt - timedelta(days=30)
            
        # Determine number of periods
        hours_per_period = self.interval_to_hours(interval)
        periods = int((end_dt - start_dt).total_seconds() / (3600 * hours_per_period))
        
        # Generate timestamps
        timestamps = [end_dt - timedelta(hours=i * hours_per_period) for i in range(periods)]
        timestamps.reverse()
        
        # Base price depends on symbol
        if "BTC" in symbol:
            base_price = 60000
            volatility = 0.02
        elif "ETH" in symbol:
            base_price = 3500
            volatility = 0.025
        elif "SOL" in symbol:
            base_price = 150
            volatility = 0.035
        else:
            base_price = 100
            volatility = 0.03
            
        # Generate price data with realistic patterns
        np.random.seed(42)  # For reproducibility
        
        # Generate returns with slight upward bias and autocorrelation
        returns = np.random.normal(0.0001, volatility, periods)
        
        # Add autocorrelation
        for i in range(1, periods):
            returns[i] = 0.7 * returns[i] + 0.3 * returns[i-1]
            
        # Generate prices
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
            
        # Generate OHLCV data
        data = []
        for i, timestamp in enumerate(timestamps):
            price = prices[i]
            
            # Generate realistic OHLC
            high_low_range = price * volatility * np.random.uniform(0.5, 1.5)
            high = price + high_low_range / 2
            low = price - high_low_range / 2
            
            # Randomly determine if open > close or close > open
            if np.random.random() > 0.5:
                open_price = price - high_low_range * np.random.uniform(0, 0.4)
                close = price + high_low_range * np.random.uniform(0, 0.4)
            else:
                open_price = price + high_low_range * np.random.uniform(0, 0.4)
                close = price - high_low_range * np.random.uniform(0, 0.4)
                
            # Ensure high >= max(open, close) and low <= min(open, close)
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Generate volume
            volume = base_price * 10 * np.random.uniform(0.5, 1.5)
            
            data.append([timestamp, open_price, high, low, close, volume])
            
        # Create DataFrame
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        return df
        
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Convert columns to numeric if they aren't already
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                
        # Calculate SMA
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()
        df["sma_200"] = df["close"].rolling(window=200).mean()
        
        # Calculate EMA
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
        
        # Calculate MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # Calculate RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        df["bb_std"] = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
        df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]
        
        # Calculate ATR
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df["atr"] = true_range.rolling(window=14).mean()
        
        # Calculate VWAP
        df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()
        
        # Calculate Stochastic Oscillator
        low_14 = df["low"].rolling(window=14).min()
        high_14 = df["high"].rolling(window=14).max()
        
        df["stoch_k"] = 100 * ((df["close"] - low_14) / (high_14 - low_14))
        df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()
        
        # Calculate OBV (On-Balance Volume)
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        
        return df
        
    def get_common_dates(self, data: Dict[str, pd.DataFrame]) -> List[datetime]:
        """
        Get common dates across all symbols.
        
        Args:
            data: Dictionary of DataFrames with data for each symbol
            
        Returns:
            List of common dates
        """
        # Get all dates for each symbol
        all_dates = {}
        for symbol, df in data.items():
            all_dates[symbol] = set(df["timestamp"].dt.date)
            
        # Get intersection of all dates
        common_dates = set.intersection(*all_dates.values())
        
        # Convert to list and sort
        common_dates = sorted(list(common_dates))
        
        # Convert to datetime
        common_dates = [pd.to_datetime(d) for d in common_dates]
        
        return common_dates
        
    def get_price_at_date(self, df: pd.DataFrame, date: datetime, price_type: str = "close") -> float:
        """
        Get price at a specific date.
        
        Args:
            df: DataFrame with price data
            date: Date to get price for
            price_type: Type of price to get (open, high, low, close)
            
        Returns:
            Price at date
        """
        # Filter by date
        date_df = df[df["timestamp"].dt.date == date.date()]
        
        # If no data for date, return last available price
        if date_df.empty:
            return df[price_type].iloc[-1]
            
        # Return price
        return date_df[price_type].iloc[-1]
        
    def prepare_data_slice(self, df: pd.DataFrame, end_idx: int, lookback: int = 100) -> pd.DataFrame:
        """
        Prepare data slice for strategy.
        
        Args:
            df: DataFrame with price data
            end_idx: End index for slice
            lookback: Number of periods to look back
            
        Returns:
            DataFrame with data slice
        """
        # Calculate start index
        start_idx = max(0, end_idx - lookback + 1)
        
        # Get slice
        return df.iloc[start_idx:end_idx+1]
        
    def close_position(self, portfolio: Dict, symbol: str, price: float, date: datetime, reason: str) -> Dict:
        """
        Close a position.
        
        Args:
            portfolio: Portfolio dictionary
            symbol: Symbol to close position for
            price: Price to close at
            date: Date to close at
            reason: Reason for closing
            
        Returns:
            Dictionary with trade result
        """
        position = portfolio["positions"][symbol]
        
        # Calculate exit value
        exit_value = position["quantity"] * price
        
        # Calculate fees
        fees = exit_value * 0.0006  # Trading fee
        
        # Calculate slippage
        slippage = exit_value * 0.0005  # Slippage
        
        # Calculate profit/loss
        entry_value = position["quantity"] * position["entry_price"]
        profit_loss = exit_value - entry_value - fees - slippage - position["fees"] - position["slippage"]
        profit_pct = profit_loss / entry_value
        
        # Add cash
        portfolio["cash"] += (exit_value - fees - slippage)
        
        # Remove position
        del portfolio["positions"][symbol]
        
        # Create trade result
        trade_result = {
            "symbol": symbol,
            "entry_date": position["entry_date"],
            "entry_price": position["entry_price"],
            "exit_date": date,
            "exit_price": price,
            "quantity": position["quantity"],
            "entry_value": entry_value,
            "exit_value": exit_value,
            "fees": position["fees"] + fees,
            "slippage": position["slippage"] + slippage,
            "profit_loss": profit_loss,
            "profit_pct": profit_pct,
            "reason": reason
        }
        
        logger.info(f"Closed position in {symbol} at {price:.2f}, profit/loss: {profit_loss:.2f} ({profit_pct:.2%}), reason: {reason}")
        
        return trade_result
        
    def calculate_performance_metrics(self, initial_capital: float, final_equity: float, equity_curve: List[Dict], trades: List[Dict]) -> Dict:
        """
        Calculate performance metrics.
        
        Args:
            initial_capital: Initial capital
            final_equity: Final equity
            equity_curve: List of equity curve points
            trades: List of trades
            
        Returns:
            Dictionary with performance metrics
        """
        # Calculate total return
        total_return = (final_equity - initial_capital) / initial_capital * 100
        
        # Calculate annualized return
        if len(equity_curve) > 1:
            start_date = equity_curve[0]["date"]
            end_date = equity_curve[-1]["date"]
            days = (end_date - start_date).days
            if days > 0:
                annualized_return = ((final_equity / initial_capital) ** (365 / days) - 1) * 100
            else:
                annualized_return = 0.0
        else:
            annualized_return = 0.0
            
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(equity_curve)):
            daily_return = (equity_curve[i]["equity"] - equity_curve[i-1]["equity"]) / equity_curve[i-1]["equity"]
            daily_returns.append(daily_return)
            
        # Calculate Sharpe ratio
        if len(daily_returns) > 1:
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            if std_return > 0:
                sharpe_ratio = mean_return / std_return * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
            
        # Calculate Sortino ratio
        if len(daily_returns) > 1:
            negative_returns = [r for r in daily_returns if r < 0]
            if len(negative_returns) > 0:
                downside_deviation = np.std(negative_returns)
                if downside_deviation > 0:
                    sortino_ratio = mean_return / downside_deviation * np.sqrt(252)  # Annualized
                else:
                    sortino_ratio = 0.0
            else:
                sortino_ratio = 0.0
        else:
            sortino_ratio = 0.0
            
        # Calculate max drawdown
        max_drawdown = 0.0
        peak = initial_capital
        
        for point in equity_curve:
            if point["equity"] > peak:
                peak = point["equity"]
            else:
                drawdown = (peak - point["equity"]) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    
        # Calculate win rate
        if len(trades) > 0:
            winning_trades = [t for t in trades if t["profit_loss"] > 0]
            win_rate = len(winning_trades) / len(trades) * 100
        else:
            winning_trades = []
            win_rate = 0.0
            
        # Calculate profit factor
        if len(trades) > 0:
            gross_profit = sum([t["profit_loss"] for t in trades if t["profit_loss"] > 0])
            gross_loss = sum([abs(t["profit_loss"]) for t in trades if t["profit_loss"] < 0])
            
            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss
            else:
                profit_factor = float('inf') if gross_profit > 0 else 0.0
        else:
            profit_factor = 0.0
            
        # Calculate average profit per trade
        if len(winning_trades) > 0:
            avg_profit = sum([t["profit_loss"] for t in winning_trades]) / len(winning_trades)
        else:
            avg_profit = 0.0
            
        # Calculate average loss per trade
        losing_trades = [t for t in trades if t["profit_loss"] < 0]
        if len(losing_trades) > 0:
            avg_loss = sum([t["profit_loss"] for t in losing_trades]) / len(losing_trades)
        else:
            avg_loss = 0.0
            
        # Calculate risk-adjusted return
        risk_adjusted_return = total_return * (1 - max_drawdown)
        
        # Return metrics
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown * 100,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_profit_per_trade": avg_profit,
            "avg_loss_per_trade": avg_loss,
            "risk_adjusted_return": risk_adjusted_return,
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades)
        }
        
    def generate_charts(self, results: Dict) -> None:
        """
        Generate charts for backtest results.
        
        Args:
            results: Dictionary with backtest results
        """
        # Create figure for equity curve
        plt.figure(figsize=(12, 6))
        
        # Extract dates and equity values
        dates = [point["date"] for point in results["equity_curve"]]
        equity = [point["equity"] for point in results["equity_curve"]]
        
        # Plot equity curve
        plt.plot(dates, equity, label="Equity Curve")
        
        # Add initial capital line
        plt.axhline(y=results["initial_capital"], color="r", linestyle="--", label="Initial Capital")
        
        # Add title and labels
        plt.title(f"Equity Curve - {results['strategy']} on {', '.join(results['symbols'])}")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.legend()
        plt.grid(True)
        
        # Save figure
        plt.savefig(f"{self.results_dir}/charts/equity_curve.png")
        plt.close()
        
        # Create figure for drawdown
        plt.figure(figsize=(12, 6))
        
        # Extract dates and drawdown values
        dates = [point["date"] for point in results["drawdowns"]]
        drawdowns = [point["drawdown"] * 100 for point in results["drawdowns"]]
        
        # Plot drawdown
        plt.plot(dates, drawdowns)
        
        # Add max drawdown line
        plt.axhline(y=results["performance"]["max_drawdown"], color="r", linestyle="--", label=f"Max Drawdown: {results['performance']['max_drawdown']:.2f}%")
        
        # Add title and labels
        plt.title(f"Drawdown - {results['strategy']} on {', '.join(results['symbols'])}")
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%)")
        plt.legend()
        plt.grid(True)
        
        # Invert y-axis
        plt.gca().invert_yaxis()
        
        # Save figure
        plt.savefig(f"{self.results_dir}/charts/drawdown.png")
        plt.close()
        
        # Create figure for trade distribution
        if results["trades"]:
            plt.figure(figsize=(12, 6))
            
            # Extract profit/loss values
            profits = [trade["profit_pct"] * 100 for trade in results["trades"]]
            
            # Plot histogram
            plt.hist(profits, bins=20)
            
            # Add title and labels
            plt.title(f"Trade Distribution - {results['strategy']} on {', '.join(results['symbols'])}")
            plt.xlabel("Profit/Loss (%)")
            plt.ylabel("Frequency")
            plt.grid(True)
            
            # Save figure
            plt.savefig(f"{self.results_dir}/charts/trade_distribution.png")
            plt.close()
            
    def generate_walk_forward_charts(self, results: Dict) -> None:
        """
        Generate charts for walk-forward optimization results.
        
        Args:
            results: Dictionary with walk-forward optimization results
        """
        # Create figure for equity curve
        plt.figure(figsize=(12, 6))
        
        # Extract dates and equity values
        dates = [point["date"] for point in results["equity_curve"]]
        equity = [point["equity"] for point in results["equity_curve"]]
        
        # Plot equity curve
        plt.plot(dates, equity, label="Equity Curve")
        
        # Add window boundaries
        for window in results["windows"]:
            in_sample_end = pd.to_datetime(window["in_sample_end"])
            out_sample_start = pd.to_datetime(window["out_sample_start"])
            out_sample_end = pd.to_datetime(window["out_sample_end"])
            
            plt.axvline(x=in_sample_end, color="g", linestyle="--", alpha=0.5)
            plt.axvline(x=out_sample_end, color="r", linestyle="--", alpha=0.5)
            
        # Add title and labels
        plt.title(f"Walk-Forward Equity Curve - {results['strategy']} on {', '.join(results['symbols'])}")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.legend()
        plt.grid(True)
        
        # Save figure
        plt.savefig(f"{self.results_dir}/charts/walk_forward_equity_curve.png")
        plt.close()
        
        # Create figure for window performance
        plt.figure(figsize=(12, 6))
        
        # Extract window performance
        window_numbers = [window["window"] for window in results["windows"]]
        in_sample_returns = [window["in_sample_performance"]["total_return"] for window in results["windows"]]
        out_sample_returns = [window["out_sample_performance"]["total_return"] for window in results["windows"]]
        
        # Plot window performance
        plt.bar(np.array(window_numbers) - 0.2, in_sample_returns, width=0.4, label="In-Sample Return")
        plt.bar(np.array(window_numbers) + 0.2, out_sample_returns, width=0.4, label="Out-of-Sample Return")
        
        # Add title and labels
        plt.title(f"Window Performance - {results['strategy']} on {', '.join(results['symbols'])}")
        plt.xlabel("Window")
        plt.ylabel("Return (%)")
        plt.legend()
        plt.grid(True)
        
        # Save figure
        plt.savefig(f"{self.results_dir}/charts/walk_forward_window_performance.png")
        plt.close()
        
    def save_results(self, results: Dict) -> None:
        """
        Save backtest results.
        
        Args:
            results: Dictionary with backtest results
        """
        # Create filename
        filename = f"{self.results_dir}/{results['strategy']}_{'-'.join(results['symbols'])}_{results['interval']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert datetime objects to strings
        results_copy = results.copy()
        
        for i, point in enumerate(results_copy["equity_curve"]):
            results_copy["equity_curve"][i]["date"] = point["date"].isoformat()
            
        for i, point in enumerate(results_copy["drawdowns"]):
            results_copy["drawdowns"][i]["date"] = point["date"].isoformat()
            
        for i, trade in enumerate(results_copy["trades"]):
            results_copy["trades"][i]["entry_date"] = trade["entry_date"].isoformat()
            results_copy["trades"][i]["exit_date"] = trade["exit_date"].isoformat()
            
        # Save to file
        with open(filename, "w") as f:
            json.dump(results_copy, f, indent=2)
            
        logger.info(f"Saved backtest results to {filename}")
        
    def save_optimization_results(self, results: Dict) -> None:
        """
        Save optimization results.
        
        Args:
            results: Dictionary with optimization results
        """
        # Create filename
        filename = f"{self.results_dir}/{results['strategy']}_{'-'.join(results['symbols'])}_{results['interval']}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Save to file
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Saved optimization results to {filename}")
        
    def save_walk_forward_results(self, results: Dict) -> None:
        """
        Save walk-forward optimization results.
        
        Args:
            results: Dictionary with walk-forward optimization results
        """
        # Create filename
        filename = f"{self.results_dir}/{results['strategy']}_{'-'.join(results['symbols'])}_{results['interval']}_walk_forward_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert datetime objects to strings
        results_copy = results.copy()
        
        for i, point in enumerate(results_copy["equity_curve"]):
            results_copy["equity_curve"][i]["date"] = point["date"].isoformat()
            
        for i, point in enumerate(results_copy["drawdowns"]):
            results_copy["drawdowns"][i]["date"] = point["date"].isoformat()
            
        for i, trade in enumerate(results_copy["trades"]):
            results_copy["trades"][i]["entry_date"] = trade["entry_date"].isoformat()
            results_copy["trades"][i]["exit_date"] = trade["exit_date"].isoformat()
            
        # Save to file
        with open(filename, "w") as f:
            json.dump(results_copy, f, indent=2)
            
        logger.info(f"Saved walk-forward optimization results to {filename}")
        
    def interval_to_hours(self, interval: str) -> int:
        """
        Convert interval string to hours.
        
        Args:
            interval: Interval string (e.g., "1h", "4h", "1d")
            
        Returns:
            Number of hours
        """
        if interval.endswith("m"):
            return int(interval[:-1]) / 60
        elif interval.endswith("h"):
            return int(interval[:-1])
        elif interval.endswith("d"):
            return int(interval[:-1]) * 24
        elif interval.endswith("w"):
            return int(interval[:-1]) * 24 * 7
        else:
            return 1  # Default to 1 hour

def triple_confluence_strategy(data: pd.DataFrame, params: Dict = None) -> Dict:
    """
    Triple Confluence Strategy.
    
    Args:
        data: DataFrame with price data
        params: Strategy parameters
        
    Returns:
        Dictionary with signal
    """
    # Default parameters
    if params is None:
        params = {
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_period": 20,
            "bb_std": 2
        }
        
    # Get latest data point
    latest = data.iloc[-1]
    
    # Check if we have enough data
    if len(data) < max(params["rsi_period"], params["macd_slow"] + params["macd_signal"], params["bb_period"]):
        return {"action": "hold", "confidence": 0.0}
        
    # Check RSI
    rsi = latest["rsi"]
    rsi_signal = "buy" if rsi < params["rsi_oversold"] else "sell" if rsi > params["rsi_overbought"] else "hold"
    
    # Check MACD
    macd = latest["macd"]
    macd_signal = latest["macd_signal"]
    macd_hist = latest["macd_hist"]
    
    # MACD crossover
    prev_macd_hist = data.iloc[-2]["macd_hist"]
    macd_cross_signal = "buy" if prev_macd_hist < 0 and macd_hist > 0 else "sell" if prev_macd_hist > 0 and macd_hist < 0 else "hold"
    
    # Check Bollinger Bands
    bb_lower = latest["bb_lower"]
    bb_upper = latest["bb_upper"]
    close = latest["close"]
    
    bb_signal = "buy" if close < bb_lower else "sell" if close > bb_upper else "hold"
    
    # Combine signals
    signals = [rsi_signal, macd_cross_signal, bb_signal]
    buy_count = signals.count("buy")
    sell_count = signals.count("sell")
    
    # Calculate confidence
    confidence = 0.0
    
    if buy_count > sell_count:
        confidence = buy_count / len(signals)
        action = "buy"
    elif sell_count > buy_count:
        confidence = sell_count / len(signals)
        action = "sell"
    else:
        action = "hold"
        
    return {"action": action, "confidence": confidence}

def oracle_update_strategy(data: pd.DataFrame, params: Dict = None) -> Dict:
    """
    Oracle Update Strategy.
    
    Args:
        data: DataFrame with price data
        params: Strategy parameters
        
    Returns:
        Dictionary with signal
    """
    # Default parameters
    if params is None:
        params = {
            "lookback": 5,
            "threshold": 0.001,
            "oracle_delay": 3  # seconds
        }
        
    # Get latest data point
    latest = data.iloc[-1]
    
    # Check if we have enough data
    if len(data) < params["lookback"]:
        return {"action": "hold", "confidence": 0.0}
        
    # Calculate price changes
    price_changes = data["close"].pct_change(1).dropna().tail(params["lookback"])
    
    # Calculate average price change
    avg_price_change = price_changes.mean()
    
    # Determine signal
    if avg_price_change > params["threshold"]:
        action = "buy"
        confidence = min(1.0, avg_price_change / params["threshold"])
    elif avg_price_change < -params["threshold"]:
        action = "sell"
        confidence = min(1.0, abs(avg_price_change) / params["threshold"])
    else:
        action = "hold"
        confidence = 0.0
        
    return {"action": action, "confidence": confidence}

def master_omni_overlord_strategy(data: pd.DataFrame, params: Dict = None) -> Dict:
    """
    Master Omni Overlord Strategy.
    
    Args:
        data: DataFrame with price data
        params: Strategy parameters
        
    Returns:
        Dictionary with signal
    """
    # Default parameters
    if params is None:
        params = {
            "triple_confluence_weight": 0.6,
            "oracle_update_weight": 0.4,
            "confidence_threshold": 0.6,
            "market_regime_lookback": 20,
            "trend_threshold": 0.1,
            "volatility_threshold": 0.02
        }
        
    # Check if we have enough data
    if len(data) < params["market_regime_lookback"]:
        return {"action": "hold", "confidence": 0.0}
        
    # Detect market regime
    returns = data["close"].pct_change(1).dropna().tail(params["market_regime_lookback"])
    trend = returns.mean() * params["market_regime_lookback"]
    volatility = returns.std()
    
    # Determine market regime
    if abs(trend) > params["trend_threshold"]:
        if trend > 0:
            market_regime = "bullish"
            # Adjust weights for bullish market
            triple_confluence_weight = params["triple_confluence_weight"] * 0.8
            oracle_update_weight = params["oracle_update_weight"] * 1.2
        else:
            market_regime = "bearish"
            # Adjust weights for bearish market
            triple_confluence_weight = params["triple_confluence_weight"] * 1.2
            oracle_update_weight = params["oracle_update_weight"] * 0.8
    elif volatility > params["volatility_threshold"]:
        market_regime = "volatile"
        # Adjust weights for volatile market
        triple_confluence_weight = params["triple_confluence_weight"] * 0.7
        oracle_update_weight = params["oracle_update_weight"] * 1.3
    else:
        market_regime = "ranging"
        # Adjust weights for ranging market
        triple_confluence_weight = params["triple_confluence_weight"] * 1.1
        oracle_update_weight = params["oracle_update_weight"] * 0.9
        
    # Normalize weights
    total_weight = triple_confluence_weight + oracle_update_weight
    triple_confluence_weight /= total_weight
    oracle_update_weight /= total_weight
    
    # Get signals from sub-strategies
    triple_confluence_signal = triple_confluence_strategy(data)
    oracle_update_signal = oracle_update_strategy(data)
    
    # Combine signals
    if triple_confluence_signal["action"] == oracle_update_signal["action"]:
        action = triple_confluence_signal["action"]
        confidence = (triple_confluence_signal["confidence"] * triple_confluence_weight + 
                     oracle_update_signal["confidence"] * oracle_update_weight)
    else:
        # Weighted decision
        triple_score = (1 if triple_confluence_signal["action"] == "buy" else 
                       -1 if triple_confluence_signal["action"] == "sell" else 0) * triple_confluence_signal["confidence"]
        oracle_score = (1 if oracle_update_signal["action"] == "buy" else 
                       -1 if oracle_update_signal["action"] == "sell" else 0) * oracle_update_signal["confidence"]
        
        combined_score = triple_score * triple_confluence_weight + oracle_score * oracle_update_weight
        
        if combined_score > params["confidence_threshold"] / 2:
            action = "buy"
            confidence = combined_score
        elif combined_score < -params["confidence_threshold"] / 2:
            action = "sell"
            confidence = abs(combined_score)
        else:
            action = "hold"
            confidence = 0.0
            
    # Apply confidence threshold
    if confidence < params["confidence_threshold"]:
        action = "hold"
        
    return {
        "action": action, 
        "confidence": confidence,
        "market_regime": market_regime,
        "triple_confluence_weight": triple_confluence_weight,
        "oracle_update_weight": oracle_update_weight,
        "triple_confluence_signal": triple_confluence_signal,
        "oracle_update_signal": oracle_update_signal
    }

async def main():
    """
    Main function to run the enhanced backtester.
    """
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Backtester")
    parser.add_argument("--strategy", type=str, default="master_omni_overlord", help="Strategy to backtest")
    parser.add_argument("--symbols", type=str, default="BTC-USD-PERP,ETH-USD-PERP,SOL-USD-PERP", help="Comma-separated list of symbols")
    parser.add_argument("--interval", type=str, default="1h", help="Interval to backtest on")
    parser.add_argument("--days", type=int, default=30, help="Number of days to backtest")
    parser.add_argument("--initial_capital", type=float, default=10000.0, help="Initial capital")
    parser.add_argument("--position_size", type=float, default=0.1, help="Position size")
    parser.add_argument("--optimize", action="store_true", help="Run parameter optimization")
    parser.add_argument("--walk_forward", action="store_true", help="Run walk-forward optimization")
    args = parser.parse_args()
    
    # Create backtester
    backtester = EnhancedBacktester()
    
    # Get strategy function
    if args.strategy == "triple_confluence":
        strategy = triple_confluence_strategy
    elif args.strategy == "oracle_update":
        strategy = oracle_update_strategy
    elif args.strategy == "master_omni_overlord":
        strategy = master_omni_overlord_strategy
    else:
        logger.error(f"Unknown strategy: {args.strategy}")
        return
        
    # Get symbols
    symbols = args.symbols.split(",")
    
    # Calculate start and end dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Run backtest or optimization
    if args.optimize:
        # Define parameter grid
        if args.strategy == "triple_confluence":
            param_grid = {
                "rsi_period": [7, 14, 21],
                "rsi_overbought": [65, 70, 75],
                "rsi_oversold": [25, 30, 35],
                "macd_fast": [8, 12, 16],
                "macd_slow": [21, 26, 30],
                "macd_signal": [7, 9, 11],
                "bb_period": [15, 20, 25],
                "bb_std": [1.5, 2.0, 2.5]
            }
        elif args.strategy == "oracle_update":
            param_grid = {
                "lookback": [3, 5, 7],
                "threshold": [0.0005, 0.001, 0.002],
                "oracle_delay": [2, 3, 4]
            }
        elif args.strategy == "master_omni_overlord":
            param_grid = {
                "triple_confluence_weight": [0.4, 0.6, 0.8],
                "oracle_update_weight": [0.2, 0.4, 0.6],
                "confidence_threshold": [0.4, 0.6, 0.8],
                "market_regime_lookback": [10, 20, 30],
                "trend_threshold": [0.05, 0.1, 0.15],
                "volatility_threshold": [0.01, 0.02, 0.03]
            }
            
        if args.walk_forward:
            # Run walk-forward optimization
            results = backtester.run_walk_forward_optimization(
                strategy=strategy,
                symbols=symbols,
                interval=args.interval,
                param_grid=param_grid,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                window_size=60,
                step_size=30,
                initial_capital=args.initial_capital,
                position_sizing="percent",
                position_size=args.position_size
            )
        else:
            # Run parameter optimization
            results = backtester.run_parameter_optimization(
                strategy=strategy,
                symbols=symbols,
                interval=args.interval,
                param_grid=param_grid,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                initial_capital=args.initial_capital,
                position_sizing="percent",
                position_size=args.position_size
            )
    else:
        # Run backtest
        results = backtester.run_backtest(
            strategy=strategy,
            symbols=symbols,
            interval=args.interval,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            initial_capital=args.initial_capital,
            position_sizing="percent",
            position_size=args.position_size
        )
        
    # Print results
    print(f"Strategy: {args.strategy}")
    print(f"Symbols: {args.symbols}")
    print(f"Interval: {args.interval}")
    print(f"Initial Capital: ${args.initial_capital:.2f}")
    print(f"Final Equity: ${results['final_equity']:.2f}")
    print(f"Total Return: {results['performance']['total_return']:.2f}%")
    print(f"Annualized Return: {results['performance']['annualized_return']:.2f}%")
    print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['performance']['max_drawdown']:.2f}%")
    print(f"Win Rate: {results['performance']['win_rate']:.2f}%")
    print(f"Profit Factor: {results['performance']['profit_factor']:.2f}")
    print(f"Total Trades: {results['performance']['total_trades']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
