"""
Base Strategy Framework for Hyperliquid Trading Bot
Provides the foundation for all trading strategies
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from enum import Enum

from ..utils.logger import get_logger, TradingLogger
from ..utils.config_manager import StrategyConfig

logger = get_logger(__name__)
trading_logger = TradingLogger(__name__)


class SignalType(Enum):
    """Trading signal types"""
    NONE = "NONE"
    LONG = "LONG"
    SHORT = "SHORT"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    HOLD = "HOLD"


class OrderType(Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass
class TradingSignal:
    """Trading signal data structure"""
    signal_type: SignalType
    coin: str
    confidence: float  # 0.0 to 1.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    size: Optional[float] = None
    order_type: OrderType = OrderType.MARKET
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MarketData:
    """Market data structure"""
    coin: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    # Additional market data
    bid: Optional[float] = None
    ask: Optional[float] = None
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'coin': self.coin,
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'bid': self.bid,
            'ask': self.ask,
            'funding_rate': self.funding_rate,
            'open_interest': self.open_interest
        }


class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, config: StrategyConfig, api_client=None):
        """
        Initialize base strategy
        
        Args:
            name: Strategy name
            config: Strategy configuration
            api_client: API client for trading
        """
        self.name = name
        self.config = config
        self.api_client = api_client
        
        # Strategy state
        self.is_running = False
        self.positions = {}  # coin -> position info
        self.signals_history = []
        self.performance_metrics = {}
        
        # Data storage
        self.market_data = {}  # coin -> list of MarketData
        self.indicators = {}  # coin -> dict of indicators
        
        # Risk management
        self.max_positions = config.max_positions
        self.position_size = config.position_size
        self.stop_loss_pct = config.stop_loss
        self.take_profit_pct = config.take_profit
        
        logger.info(f"Strategy {name} initialized with config: {config}")
    
    @abstractmethod
    async def generate_signal(self, coin: str, market_data: List[MarketData]) -> TradingSignal:
        """
        Generate trading signal based on market data
        
        Args:
            coin: Trading pair
            market_data: Historical market data
            
        Returns:
            TradingSignal object
        """
        pass
    
    @abstractmethod
    def calculate_indicators(self, coin: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for the strategy
        
        Args:
            coin: Trading pair
            data: OHLCV data as DataFrame
            
        Returns:
            Dictionary of calculated indicators
        """
        pass
    
    def update_market_data(self, coin: str, data: MarketData) -> None:
        """Update market data for a coin"""
        if coin not in self.market_data:
            self.market_data[coin] = []
        
        self.market_data[coin].append(data)
        
        # Keep only last N candles (configurable)
        max_candles = getattr(self.config, 'max_candles', 1000)
        if len(self.market_data[coin]) > max_candles:
            self.market_data[coin] = self.market_data[coin][-max_candles:]
    
    def get_market_dataframe(self, coin: str, periods: int = None) -> pd.DataFrame:
        """
        Get market data as pandas DataFrame
        
        Args:
            coin: Trading pair
            periods: Number of periods to return (None for all)
            
        Returns:
            DataFrame with OHLCV data
        """
        if coin not in self.market_data or not self.market_data[coin]:
            return pd.DataFrame()
        
        data = self.market_data[coin]
        if periods:
            data = data[-periods:]
        
        df = pd.DataFrame([d.to_dict() for d in data])
        df.set_index('timestamp', inplace=True)
        return df
    
    def calculate_position_size(self, coin: str, signal: TradingSignal) -> float:
        """
        Calculate position size based on risk management rules
        
        Args:
            coin: Trading pair
            signal: Trading signal
            
        Returns:
            Position size in USD
        """
        base_size = self.position_size
        
        # Adjust size based on confidence
        confidence_multiplier = signal.confidence
        adjusted_size = base_size * confidence_multiplier
        
        # Apply maximum position size limit
        max_size = getattr(self.config, 'max_position_size', base_size * 2)
        adjusted_size = min(adjusted_size, max_size)
        
        # Check available capital (if API client available)
        if self.api_client:
            try:
                account_state = self.api_client.get_account_state()
                available_capital = account_state.get('account_value', 0) * 0.1  # Use 10% max
                adjusted_size = min(adjusted_size, available_capital)
            except Exception as e:
                logger.warning(f"Could not check available capital: {e}")
        
        return max(adjusted_size, 10.0)  # Minimum $10 position
    
    def calculate_stop_loss(self, entry_price: float, signal_type: SignalType) -> float:
        """Calculate stop loss price"""
        if signal_type == SignalType.LONG:
            return entry_price * (1 - self.stop_loss_pct / 100)
        elif signal_type == SignalType.SHORT:
            return entry_price * (1 + self.stop_loss_pct / 100)
        return entry_price
    
    def calculate_take_profit(self, entry_price: float, signal_type: SignalType) -> float:
        """Calculate take profit price"""
        if signal_type == SignalType.LONG:
            return entry_price * (1 + self.take_profit_pct / 100)
        elif signal_type == SignalType.SHORT:
            return entry_price * (1 - self.take_profit_pct / 100)
        return entry_price
    
    def should_enter_position(self, coin: str, signal: TradingSignal) -> bool:
        """
        Check if we should enter a new position
        
        Args:
            coin: Trading pair
            signal: Trading signal
            
        Returns:
            True if should enter position
        """
        # Check if already have position in this coin
        if coin in self.positions:
            current_position = self.positions[coin]
            if current_position.get('size', 0) != 0:
                logger.debug(f"Already have position in {coin}")
                return False
        
        # Check maximum number of positions
        active_positions = sum(1 for pos in self.positions.values() if pos.get('size', 0) != 0)
        if active_positions >= self.max_positions:
            logger.debug(f"Maximum positions ({self.max_positions}) reached")
            return False
        
        # Check signal confidence threshold
        min_confidence = getattr(self.config, 'min_confidence', 0.6)
        if signal.confidence < min_confidence:
            logger.debug(f"Signal confidence {signal.confidence} below threshold {min_confidence}")
            return False
        
        return True
    
    def should_exit_position(self, coin: str, signal: TradingSignal) -> bool:
        """
        Check if we should exit an existing position
        
        Args:
            coin: Trading pair
            signal: Trading signal
            
        Returns:
            True if should exit position
        """
        if coin not in self.positions:
            return False
        
        position = self.positions[coin]
        if position.get('size', 0) == 0:
            return False
        
        # Check for exit signals
        if signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
            return True
        
        # Check for opposite signals
        current_side = 'long' if position.get('size', 0) > 0 else 'short'
        if current_side == 'long' and signal.signal_type == SignalType.SHORT:
            return True
        elif current_side == 'short' and signal.signal_type == SignalType.LONG:
            return True
        
        return False
    
    async def execute_signal(self, signal: TradingSignal) -> bool:
        """
        Execute a trading signal
        
        Args:
            signal: Trading signal to execute
            
        Returns:
            True if execution successful
        """
        if not self.api_client:
            logger.warning("No API client available for signal execution")
            return False
        
        try:
            coin = signal.coin
            
            # Check if should enter new position
            if signal.signal_type in [SignalType.LONG, SignalType.SHORT]:
                if not self.should_enter_position(coin, signal):
                    return False
                
                # Calculate position size
                size = signal.size or self.calculate_position_size(coin, signal)
                
                # Place order
                is_buy = signal.signal_type == SignalType.LONG
                
                if signal.order_type == OrderType.MARKET:
                    result = self.api_client.place_market_order(
                        coin=coin,
                        is_buy=is_buy,
                        sz=size
                    )
                else:
                    price = signal.entry_price or self.get_current_price(coin)
                    result = self.api_client.place_limit_order(
                        coin=coin,
                        is_buy=is_buy,
                        sz=size,
                        limit_px=price
                    )
                
                if result.get('status') == 'ok':
                    # Update position tracking
                    self.positions[coin] = {
                        'size': size if is_buy else -size,
                        'entry_price': signal.entry_price or self.get_current_price(coin),
                        'entry_time': datetime.now(),
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'signal': signal
                    }
                    
                    trading_logger.log_signal(self.name, coin, signal.signal_type.value, signal.confidence)
                    logger.info(f"Position opened: {coin} {signal.signal_type.value} size={size}")
                    return True
                else:
                    logger.error(f"Failed to execute signal: {result}")
                    return False
            
            # Check if should exit position
            elif signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
                if not self.should_exit_position(coin, signal):
                    return False
                
                # Close position
                result = self.api_client.close_position(coin, percentage=100.0)
                
                if result.get('status') == 'ok':
                    # Calculate PnL
                    position = self.positions.get(coin, {})
                    entry_price = position.get('entry_price', 0)
                    current_price = self.get_current_price(coin)
                    size = position.get('size', 0)
                    
                    if size > 0:  # Long position
                        pnl = (current_price - entry_price) * abs(size) / entry_price
                    else:  # Short position
                        pnl = (entry_price - current_price) * abs(size) / entry_price
                    
                    trading_logger.log_trade(
                        action="CLOSE",
                        coin=coin,
                        side="SELL" if size > 0 else "BUY",
                        size=abs(size),
                        price=current_price,
                        pnl=pnl
                    )
                    
                    # Clear position
                    self.positions[coin] = {'size': 0}
                    
                    logger.info(f"Position closed: {coin} PnL=${pnl:.2f}")
                    return True
                else:
                    logger.error(f"Failed to close position: {result}")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return False
    
    def get_current_price(self, coin: str) -> float:
        """Get current price for a coin"""
        if coin in self.market_data and self.market_data[coin]:
            return self.market_data[coin][-1].close
        
        # Fallback to API if available
        if self.api_client:
            try:
                market_data = self.api_client.get_market_data(coin)
                return market_data.get('price', 0.0)
            except Exception as e:
                logger.warning(f"Could not get current price for {coin}: {e}")
        
        return 0.0
    
    def add_signal_to_history(self, signal: TradingSignal) -> None:
        """Add signal to history"""
        self.signals_history.append(signal)
        
        # Keep only last N signals
        max_signals = 1000
        if len(self.signals_history) > max_signals:
            self.signals_history = self.signals_history[-max_signals:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate and return performance metrics"""
        if not self.signals_history:
            return {}
        
        # Calculate basic metrics
        total_signals = len(self.signals_history)
        long_signals = sum(1 for s in self.signals_history if s.signal_type == SignalType.LONG)
        short_signals = sum(1 for s in self.signals_history if s.signal_type == SignalType.SHORT)
        
        # Calculate average confidence
        avg_confidence = np.mean([s.confidence for s in self.signals_history])
        
        metrics = {
            'total_signals': total_signals,
            'long_signals': long_signals,
            'short_signals': short_signals,
            'long_ratio': long_signals / total_signals if total_signals > 0 else 0,
            'short_ratio': short_signals / total_signals if total_signals > 0 else 0,
            'avg_confidence': avg_confidence,
            'active_positions': len([p for p in self.positions.values() if p.get('size', 0) != 0])
        }
        
        self.performance_metrics = metrics
        return metrics
    
    async def start(self) -> None:
        """Start the strategy"""
        self.is_running = True
        logger.info(f"Strategy {self.name} started")
    
    async def stop(self) -> None:
        """Stop the strategy"""
        self.is_running = False
        logger.info(f"Strategy {self.name} stopped")
    
    def reset(self) -> None:
        """Reset strategy state"""
        self.positions = {}
        self.signals_history = []
        self.performance_metrics = {}
        self.market_data = {}
        self.indicators = {}
        logger.info(f"Strategy {self.name} reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get strategy status"""
        return {
            'name': self.name,
            'is_running': self.is_running,
            'positions': len([p for p in self.positions.values() if p.get('size', 0) != 0]),
            'signals_generated': len(self.signals_history),
            'performance': self.get_performance_metrics()
        }
    
    def __str__(self) -> str:
        """String representation"""
        return f"Strategy({self.name}, running={self.is_running}, positions={len(self.positions)})"

