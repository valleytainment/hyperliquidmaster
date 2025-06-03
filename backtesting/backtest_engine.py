"""
Comprehensive Backtesting Framework for Hyperliquid Trading Bot
Provides historical strategy testing with detailed performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
from pathlib import Path

from ..strategies.base_strategy import BaseStrategy, TradingSignal, SignalType, MarketData
from ..utils.logger import get_logger, TradingLogger

logger = get_logger(__name__)


@dataclass
class BacktestTrade:
    """Individual trade record for backtesting"""
    entry_time: datetime
    exit_time: Optional[datetime]
    coin: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: Optional[float]
    size: float
    pnl: Optional[float]
    pnl_percentage: Optional[float]
    fees: float
    duration: Optional[timedelta]
    exit_reason: str
    signal_confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'coin': self.coin,
            'side': self.side,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'size': self.size,
            'pnl': self.pnl,
            'pnl_percentage': self.pnl_percentage,
            'fees': self.fees,
            'duration': str(self.duration) if self.duration else None,
            'exit_reason': self.exit_reason,
            'signal_confidence': self.signal_confidence
        }


@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    # Basic metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # PnL metrics
    total_pnl: float
    total_pnl_percentage: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_percentage: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Trade metrics
    avg_trade_pnl: float
    avg_winning_trade: float
    avg_losing_trade: float
    largest_winning_trade: float
    largest_losing_trade: float
    avg_trade_duration: timedelta
    
    # Portfolio metrics
    initial_capital: float
    final_capital: float
    max_capital: float
    min_capital: float
    
    # Additional metrics
    total_fees: float
    trades_per_day: float
    exposure_time: float  # Percentage of time in market
    
    # Trade list
    trades: List[BacktestTrade]
    
    # Equity curve
    equity_curve: pd.DataFrame
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        
        # Convert timedelta objects
        if self.avg_trade_duration:
            result['avg_trade_duration'] = str(self.avg_trade_duration)
        
        # Convert trades to dict
        result['trades'] = [trade.to_dict() for trade in self.trades]
        
        # Convert equity curve to dict
        result['equity_curve'] = self.equity_curve.to_dict('records') if not self.equity_curve.empty else []
        
        return result


class BacktestEngine:
    """Advanced backtesting engine with realistic market simulation"""
    
    def __init__(self, initial_capital: float = 10000.0, 
                 commission_rate: float = 0.0005,  # 0.05% per trade
                 slippage_rate: float = 0.0001):   # 0.01% slippage
        """
        Initialize backtest engine
        
        Args:
            initial_capital: Starting capital in USD
            commission_rate: Commission rate per trade (0.0005 = 0.05%)
            slippage_rate: Slippage rate per trade (0.0001 = 0.01%)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        # Backtest state
        self.current_capital = initial_capital
        self.positions = {}  # coin -> position info
        self.trades = []
        self.equity_curve = []
        
        # Performance tracking
        self.max_capital = initial_capital
        self.min_capital = initial_capital
        self.peak_capital = initial_capital
        self.max_drawdown = 0.0
        
        logger.info(f"Backtest engine initialized with ${initial_capital:,.2f} capital")
    
    def calculate_position_size(self, signal: TradingSignal, current_price: float) -> float:
        """
        Calculate position size based on available capital and risk management
        
        Args:
            signal: Trading signal
            current_price: Current market price
            
        Returns:
            Position size in USD
        """
        # Use signal size if specified, otherwise use available capital percentage
        if signal.size:
            return min(signal.size, self.current_capital * 0.95)  # Max 95% of capital
        
        # Default to 10% of available capital per trade
        default_size = self.current_capital * 0.1
        return min(default_size, self.current_capital * 0.95)
    
    def calculate_fees(self, trade_value: float) -> float:
        """Calculate trading fees"""
        commission = trade_value * self.commission_rate
        slippage = trade_value * self.slippage_rate
        return commission + slippage
    
    def execute_signal(self, signal: TradingSignal, current_data: MarketData) -> bool:
        """
        Execute a trading signal in the backtest
        
        Args:
            signal: Trading signal to execute
            current_data: Current market data
            
        Returns:
            True if signal was executed successfully
        """
        coin = signal.coin
        current_price = current_data.close
        current_time = current_data.timestamp
        
        # Handle entry signals
        if signal.signal_type in [SignalType.LONG, SignalType.SHORT]:
            # Check if already have position
            if coin in self.positions and self.positions[coin].get('size', 0) != 0:
                return False
            
            # Calculate position size
            position_size = self.calculate_position_size(signal, current_price)
            
            # Check if we have enough capital
            if position_size > self.current_capital * 0.95:
                return False
            
            # Calculate fees
            fees = self.calculate_fees(position_size)
            
            # Apply slippage to entry price
            if signal.signal_type == SignalType.LONG:
                entry_price = current_price * (1 + self.slippage_rate)
                side = 'long'
            else:
                entry_price = current_price * (1 - self.slippage_rate)
                side = 'short'
            
            # Create position
            self.positions[coin] = {
                'side': side,
                'size': position_size,
                'entry_price': entry_price,
                'entry_time': current_time,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'signal_confidence': signal.confidence,
                'fees_paid': fees
            }
            
            # Update capital (subtract fees)
            self.current_capital -= fees
            
            logger.debug(f"Opened {side} position: {coin} size=${position_size:.2f} @ ${entry_price:.2f}")
            return True
        
        # Handle exit signals
        elif signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
            if coin not in self.positions or self.positions[coin].get('size', 0) == 0:
                return False
            
            position = self.positions[coin]
            
            # Apply slippage to exit price
            if position['side'] == 'long':
                exit_price = current_price * (1 - self.slippage_rate)
            else:
                exit_price = current_price * (1 + self.slippage_rate)
            
            # Calculate PnL
            entry_price = position['entry_price']
            position_size = position['size']
            
            if position['side'] == 'long':
                pnl = (exit_price - entry_price) * (position_size / entry_price)
            else:  # short
                pnl = (entry_price - exit_price) * (position_size / entry_price)
            
            # Calculate exit fees
            exit_fees = self.calculate_fees(position_size)
            total_fees = position['fees_paid'] + exit_fees
            
            # Net PnL after fees
            net_pnl = pnl - total_fees
            pnl_percentage = (net_pnl / position_size) * 100
            
            # Create trade record
            trade = BacktestTrade(
                entry_time=position['entry_time'],
                exit_time=current_time,
                coin=coin,
                side=position['side'],
                entry_price=entry_price,
                exit_price=exit_price,
                size=position_size,
                pnl=net_pnl,
                pnl_percentage=pnl_percentage,
                fees=total_fees,
                duration=current_time - position['entry_time'],
                exit_reason=signal.metadata.get('exit_reason', 'signal'),
                signal_confidence=position['signal_confidence']
            )
            
            self.trades.append(trade)
            
            # Update capital
            self.current_capital += position_size + net_pnl
            
            # Clear position
            self.positions[coin] = {'size': 0}
            
            logger.debug(f"Closed {position['side']} position: {coin} PnL=${net_pnl:.2f} ({pnl_percentage:.2f}%)")
            return True
        
        return False
    
    def check_stop_loss_take_profit(self, current_data: MarketData) -> List[TradingSignal]:
        """
        Check for stop loss and take profit triggers
        
        Args:
            current_data: Current market data
            
        Returns:
            List of exit signals triggered by SL/TP
        """
        signals = []
        coin = current_data.coin
        current_price = current_data.close
        
        if coin not in self.positions or self.positions[coin].get('size', 0) == 0:
            return signals
        
        position = self.positions[coin]
        
        # Check stop loss
        if position.get('stop_loss'):
            if ((position['side'] == 'long' and current_price <= position['stop_loss']) or
                (position['side'] == 'short' and current_price >= position['stop_loss'])):
                
                signal_type = SignalType.CLOSE_LONG if position['side'] == 'long' else SignalType.CLOSE_SHORT
                signals.append(TradingSignal(
                    signal_type=signal_type,
                    coin=coin,
                    confidence=1.0,
                    metadata={'exit_reason': 'stop_loss'}
                ))
        
        # Check take profit
        if position.get('take_profit'):
            if ((position['side'] == 'long' and current_price >= position['take_profit']) or
                (position['side'] == 'short' and current_price <= position['take_profit'])):
                
                signal_type = SignalType.CLOSE_LONG if position['side'] == 'long' else SignalType.CLOSE_SHORT
                signals.append(TradingSignal(
                    signal_type=signal_type,
                    coin=coin,
                    confidence=1.0,
                    metadata={'exit_reason': 'take_profit'}
                ))
        
        return signals
    
    def update_equity_curve(self, current_time: datetime):
        """Update equity curve with current portfolio value"""
        # Calculate unrealized PnL
        unrealized_pnl = 0.0
        for coin, position in self.positions.items():
            if position.get('size', 0) != 0:
                # This would need current market price, simplified for now
                unrealized_pnl += 0  # Would calculate based on current price vs entry
        
        total_equity = self.current_capital + unrealized_pnl
        
        # Update max/min tracking
        self.max_capital = max(self.max_capital, total_equity)
        self.min_capital = min(self.min_capital, total_equity)
        
        # Update drawdown tracking
        if total_equity > self.peak_capital:
            self.peak_capital = total_equity
        else:
            current_drawdown = (self.peak_capital - total_equity) / self.peak_capital
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Add to equity curve
        self.equity_curve.append({
            'timestamp': current_time,
            'equity': total_equity,
            'cash': self.current_capital,
            'unrealized_pnl': unrealized_pnl,
            'drawdown': (self.peak_capital - total_equity) / self.peak_capital
        })
    
    def run_backtest(self, strategy: BaseStrategy, market_data: Dict[str, List[MarketData]], 
                    start_date: datetime = None, end_date: datetime = None) -> BacktestResults:
        """
        Run backtest for a strategy
        
        Args:
            strategy: Trading strategy to test
            market_data: Dictionary of coin -> list of MarketData
            start_date: Start date for backtest (optional)
            end_date: End date for backtest (optional)
            
        Returns:
            BacktestResults object with comprehensive metrics
        """
        logger.info(f"Starting backtest for strategy: {strategy.name}")
        
        # Reset backtest state
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.max_capital = self.initial_capital
        self.min_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.max_drawdown = 0.0
        
        # Get all timestamps and sort
        all_timestamps = set()
        for coin_data in market_data.values():
            for data_point in coin_data:
                if start_date and data_point.timestamp < start_date:
                    continue
                if end_date and data_point.timestamp > end_date:
                    continue
                all_timestamps.add(data_point.timestamp)
        
        sorted_timestamps = sorted(all_timestamps)
        
        if not sorted_timestamps:
            logger.error("No market data available for backtest period")
            return self._create_empty_results()
        
        logger.info(f"Backtesting from {sorted_timestamps[0]} to {sorted_timestamps[-1]}")
        
        # Process each timestamp
        for i, timestamp in enumerate(sorted_timestamps):
            # Update strategy with current market data for each coin
            for coin, coin_data in market_data.items():
                # Get historical data up to current timestamp
                historical_data = [d for d in coin_data if d.timestamp <= timestamp]
                
                if len(historical_data) < 50:  # Need minimum data for indicators
                    continue
                
                # Update strategy's market data
                for data_point in historical_data:
                    strategy.update_market_data(coin, data_point)
                
                # Get current data point
                current_data = next((d for d in coin_data if d.timestamp == timestamp), None)
                if not current_data:
                    continue
                
                # Check stop loss / take profit first
                sl_tp_signals = self.check_stop_loss_take_profit(current_data)
                for signal in sl_tp_signals:
                    self.execute_signal(signal, current_data)
                
                # Generate strategy signal
                try:
                    signal = await strategy.generate_signal(coin, historical_data)
                    
                    if signal.signal_type != SignalType.NONE:
                        self.execute_signal(signal, current_data)
                        
                except Exception as e:
                    logger.warning(f"Error generating signal for {coin}: {e}")
            
            # Update equity curve
            self.update_equity_curve(timestamp)
            
            # Progress logging
            if i % 1000 == 0:
                progress = (i / len(sorted_timestamps)) * 100
                logger.info(f"Backtest progress: {progress:.1f}% - Equity: ${self.current_capital:,.2f}")
        
        # Close any remaining positions at the end
        for coin, position in self.positions.items():
            if position.get('size', 0) != 0:
                # Create exit signal
                signal_type = SignalType.CLOSE_LONG if position['side'] == 'long' else SignalType.CLOSE_SHORT
                exit_signal = TradingSignal(
                    signal_type=signal_type,
                    coin=coin,
                    confidence=1.0,
                    metadata={'exit_reason': 'backtest_end'}
                )
                
                # Get last data point for this coin
                last_data = None
                for data_point in market_data[coin]:
                    if data_point.timestamp <= sorted_timestamps[-1]:
                        last_data = data_point
                
                if last_data:
                    self.execute_signal(exit_signal, last_data)
        
        # Calculate final results
        results = self._calculate_results(sorted_timestamps[0], sorted_timestamps[-1])
        
        logger.info(f"Backtest completed. Total PnL: ${results.total_pnl:.2f} ({results.total_pnl_percentage:.2f}%)")
        logger.info(f"Win Rate: {results.win_rate:.1f}% | Max Drawdown: {results.max_drawdown_percentage:.2f}%")
        
        return results
    
    def _calculate_results(self, start_date: datetime, end_date: datetime) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        
        if not self.trades:
            return self._create_empty_results()
        
        # Basic trade statistics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # PnL calculations
        total_pnl = sum(t.pnl for t in self.trades)
        total_pnl_percentage = (total_pnl / self.initial_capital) * 100
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Trade metrics
        avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0
        avg_winning_trade = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_losing_trade = -gross_loss / losing_trades if losing_trades > 0 else 0
        largest_winning_trade = max((t.pnl for t in self.trades if t.pnl > 0), default=0)
        largest_losing_trade = min((t.pnl for t in self.trades if t.pnl < 0), default=0)
        
        # Duration metrics
        durations = [t.duration for t in self.trades if t.duration]
        avg_trade_duration = sum(durations, timedelta()) / len(durations) if durations else timedelta()
        
        # Portfolio metrics
        final_capital = self.current_capital
        
        # Risk metrics
        max_drawdown_percentage = self.max_drawdown * 100
        
        # Calculate Sharpe ratio (simplified)
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            returns = equity_df['equity'].pct_change().dropna()
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
            
            # Calmar ratio
            calmar_ratio = (total_pnl_percentage / 100) / (max_drawdown_percentage / 100) if max_drawdown_percentage > 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
            calmar_ratio = 0
        
        # Additional metrics
        total_fees = sum(t.fees for t in self.trades)
        backtest_days = (end_date - start_date).days
        trades_per_day = total_trades / backtest_days if backtest_days > 0 else 0
        
        # Exposure time (simplified - time with open positions)
        total_position_time = sum((t.duration for t in self.trades if t.duration), timedelta())
        total_backtest_time = end_date - start_date
        exposure_time = (total_position_time.total_seconds() / total_backtest_time.total_seconds()) * 100 if total_backtest_time.total_seconds() > 0 else 0
        
        # Create equity curve DataFrame
        equity_df = pd.DataFrame(self.equity_curve) if self.equity_curve else pd.DataFrame()
        
        return BacktestResults(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_percentage=total_pnl_percentage,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=profit_factor,
            max_drawdown=self.max_drawdown,
            max_drawdown_percentage=max_drawdown_percentage,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            avg_trade_pnl=avg_trade_pnl,
            avg_winning_trade=avg_winning_trade,
            avg_losing_trade=avg_losing_trade,
            largest_winning_trade=largest_winning_trade,
            largest_losing_trade=largest_losing_trade,
            avg_trade_duration=avg_trade_duration,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            max_capital=self.max_capital,
            min_capital=self.min_capital,
            total_fees=total_fees,
            trades_per_day=trades_per_day,
            exposure_time=exposure_time,
            trades=self.trades,
            equity_curve=equity_df
        )
    
    def _create_empty_results(self) -> BacktestResults:
        """Create empty results for failed backtests"""
        return BacktestResults(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            total_pnl=0,
            total_pnl_percentage=0,
            gross_profit=0,
            gross_loss=0,
            profit_factor=0,
            max_drawdown=0,
            max_drawdown_percentage=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            avg_trade_pnl=0,
            avg_winning_trade=0,
            avg_losing_trade=0,
            largest_winning_trade=0,
            largest_losing_trade=0,
            avg_trade_duration=timedelta(),
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            max_capital=self.initial_capital,
            min_capital=self.initial_capital,
            total_fees=0,
            trades_per_day=0,
            exposure_time=0,
            trades=[],
            equity_curve=pd.DataFrame()
        )
    
    def save_results(self, results: BacktestResults, filepath: str):
        """Save backtest results to file"""
        try:
            results_dict = results.to_dict()
            
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(results_dict, f, indent=2, default=str)
            
            logger.info(f"Backtest results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save backtest results: {e}")
    
    def load_results(self, filepath: str) -> Optional[BacktestResults]:
        """Load backtest results from file"""
        try:
            with open(filepath, 'r') as f:
                results_dict = json.load(f)
            
            # Convert back to BacktestResults object (simplified)
            logger.info(f"Backtest results loaded from {filepath}")
            return results_dict  # Return dict for now, could reconstruct full object
            
        except Exception as e:
            logger.error(f"Failed to load backtest results: {e}")
            return None

