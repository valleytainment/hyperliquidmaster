"""
Common types for trading strategies
"""

from enum import Enum
from typing import Dict, Any, Optional, List


class SignalType(Enum):
    """Signal type enum"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    NEUTRAL = "neutral"


class OrderType(Enum):
    """Order type enum"""
    LIMIT = "limit"
    MARKET = "market"


class TradingSignal:
    """Trading signal class"""
    
    def __init__(self, signal_type, confidence=1.0, reason=None, coin=None, price=None, size=None):
        """
        Initialize trading signal
        
        Args:
            signal_type: Signal type (buy, sell, neutral)
            confidence: Signal confidence (0.0 to 1.0)
            reason: Signal reason (optional)
            coin: Coin symbol (legacy)
            price: Target price (optional)
            size: Position size (optional)
        """
        self.signal_type = signal_type
        self.confidence = confidence
        self.reason = reason
        self.coin = coin
        self.price = price
        self.size = size
    
    def __str__(self):
        """String representation"""
        if self.coin:
            return f"{self.coin} {self.signal_type.value} (confidence: {self.confidence:.2f})"
        return f"{self.signal_type.value} (confidence: {self.confidence:.2f})"


class MarketData:
    """Market data class"""
    
    def __init__(self, coin=None, timeframe=None, candles=None, symbol=None, price=None, volume=None, timestamp=None):
        """
        Initialize market data
        
        Args:
            coin: Coin symbol (legacy)
            timeframe: Timeframe (e.g., 1m, 5m, 15m, 1h, 4h, 1d)
            candles: List of candles
            symbol: Symbol (new format)
            price: Current price
            volume: Current volume
            timestamp: Timestamp
        """
        # Support both old and new formats
        self.coin = coin or (symbol.split('-')[0] if symbol else None)
        self.symbol = symbol or coin
        self.timeframe = timeframe
        self.candles = candles or []
        self.price = price
        self.volume = volume
        self.timestamp = timestamp
    
    def __str__(self):
        """String representation"""
        if self.price is not None:
            return f"{self.symbol or self.coin} ${self.price:.4f}"
        return f"{self.coin or self.symbol} {self.timeframe} ({len(self.candles)} candles)"

