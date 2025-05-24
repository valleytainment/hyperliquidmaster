#!/usr/bin/env python3
"""
Script to run comprehensive training and backtesting for the Master Omni Overlord Strategy.
"""

import asyncio
import logging
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Import strategies
from strategies.master_omni_overlord import MasterOmniOverlordStrategy
from strategies.triple_confluence import TripleConfluenceStrategy
from strategies.oracle_update import OracleUpdateStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("master_strategy_training.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("MasterStrategyTraining")

class MasterStrategyTrainer:
    """
    Comprehensive trainer for the Master Omni Overlord Strategy.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logger
        self.logger.info("Initializing Master Strategy Trainer...")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Create output directories
        os.makedirs("training_results", exist_ok=True)
        os.makedirs("training_results/charts", exist_ok=True)
        
        # Initialize strategies
        self.master_strategy = MasterOmniOverlordStrategy(self.config, self.logger)
        self.triple_confluence = TripleConfluenceStrategy(self.config, self.logger)
        self.oracle_update = OracleUpdateStrategy(self.config, self.logger)
        
        self.logger.info("Master Strategy Trainer initialized")
        
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                
            self.logger.info(f"Configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            # Return default configuration
            return {
                "account_address": "",
                "secret_key": "",
                "symbols": ["BTC-USD-PERP", "ETH-USD-PERP", "SOL-USD-PERP"],
                "use_sentiment_analysis": True,
                "use_triple_confluence_strategy": True,
                "use_oracle_update_strategy": True,
                "risk_percent": 0.01
            }
            
    def load_data(self, symbol: str, interval: str = "1h") -> pd.DataFrame:
        """
        Load market data for backtesting.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            
        Returns:
            DataFrame with market data
        """
        try:
            # Try to load real data first
            real_data_path = f"real_data/{symbol}_{interval}_processed.csv"
            
            if os.path.exists(real_data_path):
                self.logger.info(f"Loading real market data from {real_data_path}")
                df = pd.read_csv(real_data_path)
                
                # Ensure timestamp is datetime
                if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    
                return df
                
            # Fall back to simulated data
            simulated_data_path = f"data/{symbol}_{interval}_30d_simulated.csv"
            
            if os.path.exists(simulated_data_path):
                self.logger.info(f"Loading simulated market data from {simulated_data_path}")
                df = pd.read_csv(simulated_data_path)
                
                # Ensure timestamp is datetime
                if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    
                return df
                
            # Generate data if no files exist
            self.logger.warning(f"No data found for {symbol} {interval}, generating synthetic data")
            return self._generate_synthetic_data(symbol, interval)
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return self._generate_synthetic_data(symbol, interval)
            
    def _generate_synthetic_data(self, symbol: str, interval: str) -> pd.DataFrame:
        """
        Generate synthetic market data for testing.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            
        Returns:
            DataFrame with synthetic market data
        """
        # Determine number of data points
        if interval == "1m":
            points = 60 * 24 * 30  # 30 days of 1-minute data
        elif interval == "5m":
            points = 12 * 24 * 30  # 30 days of 5-minute data
        elif interval == "15m":
            points = 4 * 24 * 30  # 30 days of 15-minute data
        elif interval == "1h":
            points = 24 * 30  # 30 days of 1-hour data
        elif interval == "4h":
            points = 6 * 30  # 30 days of 4-hour data
        elif interval == "1d":
            points = 30  # 30 days of daily data
        else:
            points = 720  # Default to 30 days of hourly data
            
        # Generate timestamps
        end_time = datetime.now()
        timestamps = pd.date_range(end=end_time, periods=points, freq=interval)
        
        # Generate price data with realistic patterns
        base_price = 50000 if symbol.startswith("BTC") else (
            3000 if symbol.startswith("ETH") else (
                100 if symbol.startswith("SOL") else 1.0
            )
        )
        
        # Generate price with trend, cycles, and noise
        trend = np.linspace(0, 0.2, points)  # Upward trend
        cycles = 0.1 * np.sin(np.linspace(0, 15, points))  # Cycles
        noise = np.random.normal(0, 0.02, points)  # Random noise
        
        price_changes = trend + cycles + noise
        prices = base_price * np.cumprod(1 + price_changes)
        
        # Generate OHLC data
        opens = prices[:-1].copy()
        opens = np.append([base_price], opens)
        highs = prices * (1 + np.random.uniform(0, 0.005, points))
        lows = prices * (1 - np.random.uniform(0, 0.005, points))
        closes = prices
        
        # Generate volume with correlation to price changes
        volume_base = base_price * 10
        volumes = volume_base * (1 + np.abs(price_changes) * 5 + np.random.normal(0, 0.5, points))
        
        # Generate funding rate
        funding_rates = np.sin(np.linspace(0, 10, points)) * 0.0001
        
        # Generate oracle price
        oracle_prices = prices * (1 + np.sin(np.linspace(0, 20, points)) * 0.001)
        
        # Create DataFrame
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "price": closes,
            "funding_rate": funding_rates,
            "oracle_price": oracle_prices
        })
        
        return df
        
    def load_multi_timeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Load multi-timeframe data for backtesting.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of DataFrames for different timeframes
        """
        timeframes = ["1h", "4h", "1d"]
        multi_tf_data = {}
        
        for tf in timeframes:
            df = self.load_data(symbol, tf)
            multi_tf_data[tf] = df
            
        return multi_tf_data
        
    def run_backtest(self, symbol: str) -> Dict:
        """
        Run comprehensive backtesting for all strategies.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with backtest results
        """
        self.logger.info(f"Running comprehensive backtesting for {symbol}...")
        
        # Load data
        data = self.load_data(symbol, "1h")
        multi_tf_data = self.load_multi_timeframe_data(symbol)
        
        # Run backtests for each strategy
        results = {}
        
        # Triple Confluence Strategy
        self.logger.info("Backtesting Triple Confluence Strategy...")
        tc_results = self.triple_confluence.backtest(data)
        results["triple_confluence"] = tc_results
        
        # Oracle Update Strategy
        self.logger.info("Backtesting Oracle Update Strategy...")
        ou_results = self.oracle_update.backtest(data)
        results["oracle_update"] = ou_results
        
        # Master Omni Overlord Strategy
        self.logger.info("Backtesting Master Omni Overlord Strategy...")
        master_results = self.master_strategy.backtest(data, multi_tf_data)
        results["master_omni_overlord"] = master_results
        
        # Save results
        self._save_backtest_results(symbol, results)
        
        # Generate performance charts
        self._generate_performance_charts(symbol, results)
        
        return results
        
    def _save_backtest_results(self, symbol: str, results: Dict):
        """
        Save backtest results to file.
        
        Args:
            symbol: Trading symbol
            results: Backtest results
        """
        try:
            # Create results directory
            os.makedirs(f"training_results/{symbol}", exist_ok=True)
            
            # Save full results
            with open(f"training_results/{symbol}/backtest_results.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
                
            # Save summary metrics
            summary = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "metrics": {}
            }
            
            for strategy, result in results.items():
                if "metrics" in result:
                    summary["metrics"][strategy] = result["metrics"]
                    
            with open(f"training_results/{symbol}/backtest_summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)
                
            self.logger.info(f"Backtest results saved for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error saving backtest results: {str(e)}")
            
    def _generate_performance_charts(self, symbol: str, results: Dict):
        """
        Generate performance charts from backtest results.
        
        Args:
            symbol: Trading symbol
            results: Backtest results
        """
        try:
            # Create charts directory
            os.makedirs(f"training_results/charts/{symbol}", exist_ok=True)
            
            # Generate equity curve chart
            plt.figure(figsize=(12, 6))
            
            for strategy, result in results.items():
                if "equity_curve" in result and result["equity_curve"]:
                    # Extract timestamps and equity values
                    timestamps = [point.get("timestamp") for point in result["equity_curve"]]
                    equity = [point.get("equity") for point in result["equity_curve"]]
                    
                    # Plot equity curve
                    plt.plot(range(len(equity)), equity, label=strategy)
                    
            plt.title(f"Equity Curves - {symbol}")
            plt.xlabel("Trade Number")
            plt.ylabel("Equity")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"training_results/charts/{symbol}/equity_curves.png")
            plt.close()
            
            # Generate performance comparison chart
            strategies = list(results.keys())
            metrics = ["total_return", "win_rate", "sharpe_ratio"]
            
            for metric in metrics:
                values = []
                
                for strategy in strategies:
                    if "metrics" in results[strategy] and metric in results[strategy]["metrics"]:
                        value = results[strategy]["metrics"][metric]
                        
                        # Convert to percentage for display
                        if metric == "total_return" or metric == "win_rate":
                            value = value * 100
                            
                        values.append(value)
                    else:
                        values.append(0)
                        
                plt.figure(figsize=(10, 5))
                plt.bar(strategies, values)
                plt.title(f"{metric.replace('_', ' ').title()} - {symbol}")
                plt.ylabel(f"{metric.replace('_', ' ').title()}" + (" (%)" if metric in ["total_return", "win_rate"] else ""))
                plt.grid(True, axis="y")
                plt.savefig(f"training_results/charts/{symbol}/{metric}.png")
                plt.close()
                
            self.logger.info(f"Performance charts generated for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error generating performance charts: {str(e)}")
            
    def optimize_master_strategy(self, symbol: str) -> Dict:
        """
        Optimize the Master Omni Overlord Strategy parameters.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with optimization results
        """
        self.logger.info(f"Optimizing Master Omni Overlord Strategy for {symbol}...")
        
        # Load data
        data = self.load_data(symbol, "1h")
        multi_tf_data = self.load_multi_timeframe_data(symbol)
        
        # Parameters to optimize
        param_grid = {
            "volatility_lookback": [10, 20, 30],
            "trend_lookback": [30, 50, 70],
            "regime_threshold": [0.3, 0.5, 0.7],
            "signal_threshold": [0.6, 0.7, 0.8]
        }
        
        # Generate parameter combinations
        param_combinations = []
        
        for volatility_lookback in param_grid["volatility_lookback"]:
            for trend_lookback in param_grid["trend_lookback"]:
                for regime_threshold in param_grid["regime_threshold"]:
                    for signal_threshold in param_grid["signal_threshold"]:
                        params = {
                            "volatility_lookback": volatility_lookback,
                            "trend_lookback": trend_lookback,
                            "regime_threshold": regime_threshold,
                            "signal_threshold": signal_threshold
                        }
                        param_combinations.append(params)
                        
        self.logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        # Run optimization
        best_score = -float('inf')
        best_params = None
        best_result = None
        
        for i, params in enumerate(param_combinations):
            self.logger.info(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
            
            # Update strategy parameters
            self.master_strategy.volatility_lookback = params["volatility_lookback"]
            self.master_strategy.trend_lookback = params["trend_lookback"]
            self.master_strategy.regime_threshold = params["regime_threshold"]
            self.master_strategy.adaptive_params["signal_threshold"] = params["signal_threshold"]
            
            # Run backtest
            result = self.master_strategy.backtest(data, multi_tf_data)
            
            # Calculate score (weighted combination of metrics)
            if "metrics" in result:
                metrics = result["metrics"]
                
                # Calculate score based on return, win rate, and Sharpe ratio
                total_return = metrics.get("total_return", 0)
                win_rate = metrics.get("win_rate", 0)
                sharpe_ratio = metrics.get("sharpe_ratio", 0)
                max_drawdown = metrics.get("max_drawdown", 1)
                
                # Penalize high drawdowns
                drawdown_penalty = 1 - min(1, max_drawdown * 5)
                
                # Calculate score
                score = (
                    0.4 * total_return +
                    0.2 * win_rate +
                    0.3 * sharpe_ratio +
                    0.1 * drawdown_penalty
                )
                
                self.logger.info(f"Score: {score:.4f} (return: {total_return:.2%}, win rate: {win_rate:.2%}, sharpe: {sharpe_ratio:.2f})")
                
                # Update best parameters if score is higher
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_result = result
                    
        # Reset strategy to best parameters
        if best_params:
            self.logger.info(f"Best parameters found: {best_params} with score {best_score:.4f}")
            
            self.master_strategy.volatility_lookback = best_params["volatility_lookback"]
            self.master_strategy.trend_lookback = best_params["trend_lookback"]
            self.master_strategy.regime_threshold = best_params["regime_threshold"]
            self.master_strategy.adaptive_params["signal_threshold"] = best_params["signal_threshold"]
            
            # Save optimization results
            optimization_result = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "best_params": best_params,
                "best_score": best_score,
                "best_metrics": best_result["metrics"] if best_result and "metrics" in best_result else {}
            }
            
            with open(f"training_results/{symbol}/optimization_results.json", "w") as f:
                json.dump(optimization_result, f, indent=2, default=str)
                
            return optimization_result
        else:
            self.logger.warning("No valid parameters found during optimization")
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "error": "No valid parameters found"
            }
            
    def run_comprehensive_training(self, symbols: List[str]) -> Dict:
        """
        Run comprehensive training for all symbols.
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dictionary with training results
        """
        self.logger.info(f"Running comprehensive training for symbols: {symbols}")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "symbols": symbols,
            "results": {}
        }
        
        for symbol in symbols:
            self.logger.info(f"Training for {symbol}...")
            
            # Create symbol results
            symbol_results = {
                "optimization": None,
                "backtest": None
            }
            
            # Optimize strategy
            optimization_result = self.optimize_master_strategy(symbol)
            symbol_results["optimization"] = optimization_result
            
            # Run backtest with optimized parameters
            backtest_results = self.run_backtest(symbol)
            symbol_results["backtest"] = backtest_results
            
            # Store results
            results["results"][symbol] = symbol_results
            
        # Save overall results
        with open("training_results/comprehensive_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
            
        # Generate summary report
        self._generate_summary_report(results)
        
        return results
        
    def _generate_summary_report(self, results: Dict):
        """
        Generate summary report from comprehensive training results.
        
        Args:
            results: Comprehensive training results
        """
        try:
            # Create report
            report = {
                "timestamp": datetime.now().isoformat(),
                "symbols": results["symbols"],
                "strategy_performance": {},
                "best_strategy": {},
                "recommendations": {}
            }
            
            # Analyze results for each symbol
            for symbol, symbol_results in results["results"].items():
                backtest_results = symbol_results.get("backtest", {})
                
                # Compare strategies
                strategy_metrics = {}
                best_strategy = None
                best_return = -float('inf')
                
                for strategy, strategy_result in backtest_results.items():
                    if "metrics" in strategy_result:
                        metrics = strategy_result["metrics"]
                        strategy_metrics[strategy] = metrics
                        
                        # Track best strategy based on return
                        total_return = metrics.get("total_return", 0)
                        if total_return > best_return:
                            best_return = total_return
                            best_strategy = strategy
                            
                # Store strategy performance
                report["strategy_performance"][symbol] = strategy_metrics
                
                # Store best strategy
                if best_strategy:
                    report["best_strategy"][symbol] = {
                        "strategy": best_strategy,
                        "metrics": strategy_metrics.get(best_strategy, {})
                    }
                    
                # Generate recommendations
                recommendations = []
                
                if best_strategy == "master_omni_overlord":
                    recommendations.append("Use Master Omni Overlord Strategy as primary strategy")
                elif best_strategy:
                    recommendations.append(f"Consider {best_strategy} as primary strategy with Master Omni Overlord as backup")
                    
                # Add optimization recommendations
                optimization_result = symbol_results.get("optimization", {})
                best_params = optimization_result.get("best_params", {})
                
                if best_params:
                    recommendations.append(f"Use optimized parameters: {best_params}")
                    
                report["recommendations"][symbol] = recommendations
                
            # Save report
            with open("training_results/summary_report.md", "w") as f:
                f.write("# Master Omni Overlord Strategy - Training Summary Report\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## Overview\n\n")
                f.write("This report summarizes the comprehensive training and backtesting results for the Master Omni Overlord Strategy.\n\n")
                
                f.write("## Strategy Performance\n\n")
                
                for symbol in results["symbols"]:
                    f.write(f"### {symbol}\n\n")
                    
                    # Best strategy
                    if symbol in report["best_strategy"]:
                        best = report["best_strategy"][symbol]
                        f.write(f"**Best Strategy:** {best['strategy']}\n\n")
                        
                        metrics = best.get("metrics", {})
                        f.write("**Performance Metrics:**\n\n")
                        f.write(f"- Total Return: {metrics.get('total_return', 0):.2%}\n")
                        f.write(f"- Win Rate: {metrics.get('win_rate', 0):.2%}\n")
                        f.write(f"- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n")
                        f.write(f"- Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}\n")
                        f.write(f"- Total Trades: {metrics.get('total_trades', 0)}\n\n")
                        
                    # Strategy comparison
                    if symbol in report["strategy_performance"]:
                        f.write("**Strategy Comparison:**\n\n")
                        f.write("| Strategy | Total Return | Win Rate | Sharpe Ratio | Max Drawdown |\n")
                        f.write("|----------|--------------|----------|--------------|-------------|\n")
                        
                        for strategy, metrics in report["strategy_performance"][symbol].items():
                            f.write(f"| {strategy} | {metrics.get('total_return', 0):.2%} | {metrics.get('win_rate', 0):.2%} | {metrics.get('sharpe_ratio', 0):.2f} | {metrics.get('max_drawdown', 0):.2%} |\n")
                            
                        f.write("\n")
                        
                    # Recommendations
                    if symbol in report["recommendations"]:
                        f.write("**Recommendations:**\n\n")
                        
                        for recommendation in report["recommendations"][symbol]:
                            f.write(f"- {recommendation}\n")
                            
                        f.write("\n")
                        
                f.write("## Conclusion\n\n")
                f.write("The Master Omni Overlord Strategy has been comprehensively trained and optimized for the specified symbols. ")
                f.write("The strategy demonstrates adaptive behavior based on market conditions and combines the strengths of multiple sub-strategies.\n\n")
                
                f.write("For optimal results, use the recommended parameters and monitor performance regularly. ")
                f.write("The strategy is designed to adapt to changing market conditions, but periodic retraining is recommended to maintain performance.\n")
                
            self.logger.info("Summary report generated")
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {str(e)}")

async def main():
    """Main entry point."""
    logger.info("Starting Master Omni Overlord Strategy training...")
    
    # Create trainer
    trainer = MasterStrategyTrainer()
    
    # Define symbols
    symbols = ["BTC-USD-PERP", "ETH-USD-PERP", "SOL-USD-PERP"]
    
    # Run comprehensive training
    results = trainer.run_comprehensive_training(symbols)
    
    logger.info("Master Omni Overlord Strategy training completed")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
