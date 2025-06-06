"""
Common types for trading strategies
"""

from enum import Enum
from typing import Dict, Any, Optional, List


class SignalType(Enum):
    """Signal type enum"""
    BUY = "buy"
    SELL = "sell"
    NEUTRAL = "neutral"


class TradingSignal:
    """Trading signal class"""
    
    def __init__(self, coin, signal_type, confidence=1.0, price=None, size=None, reason=None):
        """
        Initialize trading signal
        
        Args:
            coin: Coin symbol
            signal_type: Signal type (buy, sell, neutral)
            confidence: Signal confidence (0.0 to 1.0)
            price: Target price (optional)
            size: Position size (optional)
            reason: Signal reason (optional)
        """
        self.coin = coin
        self.signal_type = signal_type
        self.confidence = confidence
        self.price = price
        self.size = size
        self.reason = reason
    
    def __str__(self):
        """String representation"""
        return f"{self.coin} {self.signal_type.value} (confidence: {self.confidence:.2f})"


class MarketData:
    """Market data class"""
    
    def __init__(self, coin, timeframe, candles):
        """
        Initialize market data
        
        Args:
            coin: Coin symbol
            timeframe: Timeframe (e.g., 1m, 5m, 15m, 1h, 4h, 1d)
            candles: List of candles
        """
        self.coin = coin
        self.timeframe = timeframe
        self.candles = candles
    
    def __str__(self):
        """String representation"""
        return f"{self.coin} {self.timeframe} ({len(self.candles)} candles)"

