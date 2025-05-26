"""
Backtesting module for the hyperliquidmaster package.

This module provides functionality for backtesting trading strategies
using historical data.
"""

import os
import csv
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

from hyperliquidmaster.config import BotSettings
from hyperliquidmaster.strategies.base_strategy import BaseStrategy
from hyperliquidmaster.risk import RiskManager

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Engine for backtesting trading strategies.
    
    This class provides functionality for backtesting trading strategies
    using historical price data.
    """
    
    def __init__(self, 
                settings: BotSettings, 
                strategy: BaseStrategy,
                risk_manager: Optional[RiskManager] = None):
        """
        Initialize the backtest engine.
        
        Args:
            settings: Bot configuration settings
            strategy: Trading strategy to backtest
            risk_manager: Risk manager (optional)
        """
        self.settings = settings
        self.strategy = strategy
        self.risk_manager = risk_manager or RiskManager(settings)
        
        # Backtest parameters
        self.initial_capital = 10000.0  # Default initial capital
        self.commission_rate = self.settings.taker_fee  # Use taker fee as commission
        
        # Backtest results
        self.trades = []
        self.equity_curve = []
        self.performance_metrics = {}
    
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load historical data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with historical data
        """
        # Check file exists
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        # Load data based on file extension
        if path.suffix.lower() == '.csv':
            # Try to load CSV with different formats
            try:
                df = pd.read_csv(path, parse_dates=['timestamp'])
            except:
                try:
                    df = pd.read_csv(path, parse_dates=[0])  # Assume first column is timestamp
                except:
                    df = pd.read_csv(path)  # Last resort, no parsing
        elif path.suffix.lower() == '.json':
            df = pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        logger.info(f"Loaded {len(df)} data points from {path}")
        return df
    
    def run_backtest(self, 
                    data: pd.DataFrame, 
                    symbol: str,
                    initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Run backtest on historical data.
        
        Args:
            data: DataFrame with historical data
            symbol: Trading symbol
            initial_capital: Initial capital for backtest
            
        Returns:
            Dictionary with backtest results
        """
        self.initial_capital = initial_capital
        
        # Reset results
        self.trades = []
        self.equity_curve = []
        self.performance_metrics = {}
        
        # Initialize backtest state
        equity = initial_capital
        position = 0
        entry_price = 0
        entry_time = None
        
        # Add equity curve starting point
        self.equity_curve.append({
            'timestamp': data.iloc[0]['timestamp'],
            'equity': equity,
            'position': position
        })
        
        # Run backtest
        for i in range(len(data)):
            row = data.iloc[i]
            
            # Create market data for strategy
            market_data = {
                'symbol': symbol,
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'last_price': row['close']
            }
            
            # Update strategy state
            self.strategy.update_state(symbol, market_data)
            
            # Generate signal if we don't have a position
            if position == 0:
                signal = self.strategy.generate_signal(symbol, market_data)
                
                if signal and signal.get('action') in ['buy', 'sell']:
                    # Calculate position size
                    price = row['close']
                    stop_loss = signal.get('stop_loss')
                    
                    if self.risk_manager:
                        size = self.risk_manager.calculate_position_size(
                            equity, price, stop_loss
                        )
                    else:
                        # Default to 1% risk
                        size = (equity * 0.01) / price
                    
                    # Enter position
                    position = size if signal['action'] == 'buy' else -size
                    entry_price = price
                    entry_time = row['timestamp']
                    
                    # Log trade
                    self.trades.append({
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'position': position,
                        'action': signal['action'],
                        'signal_strength': signal.get('strength', 0),
                        'stop_loss': stop_loss
                    })
                    
                    logger.debug(f"Entered {signal['action']} position at {entry_price}")
            
            # Check for exit if we have a position
            elif position != 0:
                exit_signal = False
                exit_reason = ""
                
                # Check for stop loss
                last_trade = self.trades[-1]
                stop_loss = last_trade.get('stop_loss')
                
                if stop_loss:
                    if position > 0 and row['low'] <= stop_loss:  # Long position stop
                        exit_signal = True
                        exit_reason = "stop_loss"
                        exit_price = stop_loss
                    elif position < 0 and row['high'] >= stop_loss:  # Short position stop
                        exit_signal = True
                        exit_reason = "stop_loss"
                        exit_price = stop_loss
                
                # Check for strategy exit signal
                if not exit_signal:
                    signal = self.strategy.generate_signal(symbol, market_data)
                    
                    if signal and signal.get('action') == 'close':
                        exit_signal = True
                        exit_reason = "strategy"
                        exit_price = row['close']
                
                # Exit position if signaled
                if exit_signal:
                    # Calculate PnL
                    if position > 0:  # Long position
                        pnl = (exit_price - entry_price) * abs(position)
                    else:  # Short position
                        pnl = (entry_price - exit_price) * abs(position)
                    
                    # Subtract commission
                    commission = abs(position) * exit_price * self.commission_rate
                    pnl -= commission
                    
                    # Update equity
                    equity += pnl
                    
                    # Update last trade
                    last_trade.update({
                        'exit_time': row['timestamp'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'exit_reason': exit_reason
                    })
                    
                    # Reset position
                    position = 0
                    entry_price = 0
                    entry_time = None
                    
                    logger.debug(f"Exited position at {exit_price}, PnL: {pnl:.2f}")
            
            # Update equity curve
            self.equity_curve.append({
                'timestamp': row['timestamp'],
                'equity': equity,
                'position': position
            })
        
        # Close any open position at the end
        if position != 0:
            last_price = data.iloc[-1]['close']
            
            # Calculate PnL
            if position > 0:  # Long position
                pnl = (last_price - entry_price) * abs(position)
            else:  # Short position
                pnl = (entry_price - last_price) * abs(position)
            
            # Subtract commission
            commission = abs(position) * last_price * self.commission_rate
            pnl -= commission
            
            # Update equity
            equity += pnl
            
            # Update last trade
            self.trades[-1].update({
                'exit_time': data.iloc[-1]['timestamp'],
                'exit_price': last_price,
                'pnl': pnl,
                'exit_reason': "end_of_data"
            })
            
            # Update final equity curve point
            self.equity_curve[-1]['equity'] = equity
            self.equity_curve[-1]['position'] = 0
            
            logger.debug(f"Closed position at end of data, PnL: {pnl:.2f}")
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
        return self.get_results()
    
    def _calculate_performance_metrics(self) -> None:
        """Calculate performance metrics from backtest results."""
        if not self.trades or not self.equity_curve:
            logger.warning("No trades or equity curve data to calculate metrics")
            return
        
        # Convert equity curve to DataFrame for easier analysis
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Basic metrics
        initial_equity = self.initial_capital
        final_equity = self.equity_curve[-1]['equity']
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Calculate daily returns
        equity_df['daily_return'] = equity_df['equity'].pct_change()
        
        # Trading metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.get('pnl', 0) > 0])
        losing_trades = len([t for t in self.trades if t.get('pnl', 0) <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate average win and loss
        avg_win = np.mean([t['pnl'] for t in self.trades if t.get('pnl', 0) > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in self.trades if t.get('pnl', 0) <= 0]) if losing_trades > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum([t['pnl'] for t in self.trades if t.get('pnl', 0) > 0])
        gross_loss = abs(sum([t['pnl'] for t in self.trades if t.get('pnl', 0) <= 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = abs(equity_df['drawdown'].min())
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        if len(equity_df) > 1:
            daily_returns = equity_df['daily_return'].dropna()
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Store metrics
        self.performance_metrics = {
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio
        }
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get backtest results.
        
        Returns:
            Dictionary with backtest results
        """
        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'performance_metrics': self.performance_metrics
        }
    
    def save_results(self, file_path: Union[str, Path]) -> None:
        """
        Save backtest results to file.
        
        Args:
            file_path: Path to save results
        """
        path = Path(file_path)
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save results based on file extension
        if path.suffix.lower() == '.json':
            with open(path, 'w') as f:
                json.dump(self.get_results(), f, indent=2, default=str)
        elif path.suffix.lower() == '.csv':
            # Save trades to CSV
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(path.with_name(f"{path.stem}_trades.csv"), index=False)
            
            # Save equity curve to CSV
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.to_csv(path.with_name(f"{path.stem}_equity.csv"), index=False)
            
            # Save metrics to CSV
            metrics_df = pd.DataFrame([self.performance_metrics])
            metrics_df.to_csv(path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        logger.info(f"Saved backtest results to {path}")
    
    def plot_results(self, save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Plot backtest results.
        
        Args:
            save_path: Path to save plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
            
            # Convert equity curve to DataFrame
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            equity_df.set_index('timestamp', inplace=True)
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot equity curve
            ax1.plot(equity_df.index, equity_df['equity'])
            ax1.set_title('Equity Curve')
            ax1.set_ylabel('Equity')
            ax1.grid(True)
            
            # Plot drawdown
            equity_df['peak'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
            
            ax2.fill_between(equity_df.index, equity_df['drawdown'], 0, color='red', alpha=0.3)
            ax2.set_title('Drawdown (%)')
            ax2.set_ylabel('Drawdown %')
            ax2.grid(True)
            
            # Add key metrics as text
            metrics = self.performance_metrics
            metrics_text = (
                f"Total Return: {metrics['total_return_pct']:.2f}%\n"
                f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%\n"
                f"Win Rate: {metrics['win_rate']*100:.2f}%\n"
                f"Profit Factor: {metrics['profit_factor']:.2f}"
            )
            
            # Add text box with metrics
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax1.text(0.02, 0.05, metrics_text, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='bottom', bbox=props)
            
            plt.tight_layout()
            
            # Save or show plot
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Saved backtest plot to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib not available, cannot plot results")
        except Exception as e:
            logger.error(f"Error plotting results: {e}")
