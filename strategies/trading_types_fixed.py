"""
Trading Types and Data Structures for Hyperliquid Master
Complete implementation of all trading signals and market data types
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime
import json

class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"

class OrderType(Enum):
    """Order types for trading"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class TradingMode(Enum):
    """Trading modes"""
    SPOT = "spot"
    PERP = "perp"
    BOTH = "both"

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    high_24h: float
    low_24h: float
    change_24h: float
    bid: float
    ask: float
    spread: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'price': self.price,
            'volume': self.volume,
            'timestamp': self.timestamp.isoformat(),
            'high_24h': self.high_24h,
            'low_24h': self.low_24h,
            'change_24h': self.change_24h,
            'bid': self.bid,
            'ask': self.ask,
            'spread': self.spread
        }

@dataclass
class TradingSignal:
    """Trading signal with comprehensive information"""
    signal_type: SignalType
    symbol: str
    confidence: float
    price: float
    timestamp: datetime
    strategy_name: str
    reasoning: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    order_type: OrderType = OrderType.MARKET
    trading_mode: TradingMode = TradingMode.PERP
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Post-initialization validation"""
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.price <= 0:
            raise ValueError("Price must be positive")
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'signal_type': self.signal_type.value,
            'symbol': self.symbol,
            'confidence': self.confidence,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'strategy_name': self.strategy_name,
            'reasoning': self.reasoning,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size,
            'order_type': self.order_type.value,
            'trading_mode': self.trading_mode.value,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingSignal':
        """Create from dictionary"""
        return cls(
            signal_type=SignalType(data['signal_type']),
            symbol=data['symbol'],
            confidence=data['confidence'],
            price=data['price'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            strategy_name=data['strategy_name'],
            reasoning=data['reasoning'],
            stop_loss=data.get('stop_loss'),
            take_profit=data.get('take_profit'),
            position_size=data.get('position_size'),
            order_type=OrderType(data.get('order_type', 'market')),
            trading_mode=TradingMode(data.get('trading_mode', 'perp')),
            metadata=data.get('metadata', {})
        )
    
    def is_buy_signal(self) -> bool:
        """Check if this is a buy signal"""
        return self.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]
    
    def is_sell_signal(self) -> bool:
        """Check if this is a sell signal"""
        return self.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]
    
    def is_strong_signal(self) -> bool:
        """Check if this is a strong signal"""
        return self.signal_type in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]

@dataclass
class Position:
    """Trading position information"""
    symbol: str
    side: str  # "long" or "short"
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime
    trading_mode: TradingMode
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'size': self.size,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'timestamp': self.timestamp.isoformat(),
            'trading_mode': self.trading_mode.value,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }
    
    def get_pnl_percentage(self) -> float:
        """Get PnL as percentage"""
        if self.entry_price == 0:
            return 0.0
        return (self.unrealized_pnl / (self.entry_price * abs(self.size))) * 100

@dataclass
class TradeExecution:
    """Trade execution result"""
    success: bool
    order_id: Optional[str]
    symbol: str
    side: str
    size: float
    price: float
    timestamp: datetime
    error_message: Optional[str] = None
    fees: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'success': self.success,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'size': self.size,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'error_message': self.error_message,
            'fees': self.fees
        }

@dataclass
class TradingMetrics:
    """Trading performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    current_streak: int
    max_winning_streak: int
    max_losing_streak: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl': self.total_pnl,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'profit_factor': self.profit_factor,
            'current_streak': self.current_streak,
            'max_winning_streak': self.max_winning_streak,
            'max_losing_streak': self.max_losing_streak
        }

class TradingState(Enum):
    """Trading bot states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"

@dataclass
class AutomationConfig:
    """Automation configuration"""
    enabled: bool
    trading_mode: TradingMode
    max_positions: int
    position_size_usd: float
    stop_loss_pct: float
    take_profit_pct: float
    max_daily_loss: float
    max_drawdown: float
    symbols: List[str]
    strategies: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'enabled': self.enabled,
            'trading_mode': self.trading_mode.value,
            'max_positions': self.max_positions,
            'position_size_usd': self.position_size_usd,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'max_daily_loss': self.max_daily_loss,
            'max_drawdown': self.max_drawdown,
            'symbols': self.symbols,
            'strategies': self.strategies
        }

# Utility functions
def create_buy_signal(symbol: str, price: float, confidence: float, 
                     strategy_name: str, reasoning: str, 
                     trading_mode: TradingMode = TradingMode.PERP) -> TradingSignal:
    """Create a buy signal"""
    return TradingSignal(
        signal_type=SignalType.STRONG_BUY if confidence > 0.8 else SignalType.BUY,
        symbol=symbol,
        confidence=confidence,
        price=price,
        timestamp=datetime.now(),
        strategy_name=strategy_name,
        reasoning=reasoning,
        trading_mode=trading_mode
    )

def create_sell_signal(symbol: str, price: float, confidence: float,
                      strategy_name: str, reasoning: str,
                      trading_mode: TradingMode = TradingMode.PERP) -> TradingSignal:
    """Create a sell signal"""
    return TradingSignal(
        signal_type=SignalType.STRONG_SELL if confidence > 0.8 else SignalType.SELL,
        symbol=symbol,
        confidence=confidence,
        price=price,
        timestamp=datetime.now(),
        strategy_name=strategy_name,
        reasoning=reasoning,
        trading_mode=trading_mode
    )

def create_hold_signal(symbol: str, price: float, strategy_name: str, 
                      reasoning: str) -> TradingSignal:
    """Create a hold signal"""
    return TradingSignal(
        signal_type=SignalType.HOLD,
        symbol=symbol,
        confidence=0.5,
        price=price,
        timestamp=datetime.now(),
        strategy_name=strategy_name,
        reasoning=reasoning
    )

