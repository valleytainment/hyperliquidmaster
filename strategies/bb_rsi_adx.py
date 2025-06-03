"""
Bollinger Bands + RSI + ADX Strategy
Advanced trend-following strategy combining multiple technical indicators
Ported and enhanced from the analyzed repositories
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from strategies.base_strategy import BaseStrategy, TradingSignal, SignalType, MarketData, OrderType
from utils.logger import get_logger

logger = get_logger(__name__)


class BBRSIADXStrategy(BaseStrategy):
    """
    Bollinger Bands + RSI + ADX Strategy
    
    This strategy combines:
    - Bollinger Bands for volatility and mean reversion
    - RSI for momentum and overbought/oversold conditions
    - ADX for trend strength confirmation
    
    Entry Conditions:
    LONG: Price touches lower BB, RSI < oversold, ADX > threshold (strong trend)
    SHORT: Price touches upper BB, RSI > overbought, ADX > threshold (strong trend)
    
    Exit Conditions:
    - Price crosses middle BB (opposite direction)
    - RSI reaches opposite extreme
    - Stop loss or take profit hit
    """
    
    def __init__(self, name: str = "BB_RSI_ADX", config=None, api_client=None):
        """Initialize BB RSI ADX strategy"""
        super().__init__(name, config, api_client)
        
        # Strategy parameters from config
        indicators_config = config.indicators if config else {}
        
        # Bollinger Bands parameters
        bb_config = indicators_config.get('bollinger_bands', {})
        self.bb_period = bb_config.get('period', 20)
        self.bb_std_dev = bb_config.get('std_dev', 2.0)
        
        # RSI parameters
        rsi_config = indicators_config.get('rsi', {})
        self.rsi_period = rsi_config.get('period', 14)
        self.rsi_overbought = rsi_config.get('overbought', 75)
        self.rsi_oversold = rsi_config.get('oversold', 25)
        
        # ADX parameters
        adx_config = indicators_config.get('adx', {})
        self.adx_period = adx_config.get('period', 14)
        self.adx_threshold = adx_config.get('threshold', 25)
        
        # Additional parameters
        self.min_candles = max(self.bb_period, self.rsi_period, self.adx_period) + 10
        
        logger.info(f"BB RSI ADX Strategy initialized with BB({self.bb_period}, {self.bb_std_dev}), "
                   f"RSI({self.rsi_period}, {self.rsi_oversold}, {self.rsi_overbought}), "
                   f"ADX({self.adx_period}, {self.adx_threshold})")
    
    def calculate_bollinger_bands(self, data: pd.Series) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = data.rolling(window=self.bb_period).mean()
        std = data.rolling(window=self.bb_period).std()
        
        upper_band = sma + (std * self.bb_std_dev)
        lower_band = sma - (std * self.bb_std_dev)
        
        return {
            'bb_upper': upper_band,
            'bb_middle': sma,
            'bb_lower': lower_band,
            'bb_width': (upper_band - lower_band) / sma,
            'bb_position': (data - lower_band) / (upper_band - lower_band)
        }
    
    def calculate_rsi(self, data: pd.Series) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """Calculate ADX (Average Directional Index)"""
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        dm_plus = high.diff()
        dm_minus = low.diff() * -1
        
        dm_plus[dm_plus < 0] = 0
        dm_minus[dm_minus < 0] = 0
        
        # When both DM+ and DM- are positive, only the larger one is kept
        dm_plus[(dm_plus < dm_minus)] = 0
        dm_minus[(dm_minus < dm_plus)] = 0
        
        # Smoothed True Range and Directional Movement
        atr = tr.rolling(window=self.adx_period).mean()
        di_plus = (dm_plus.rolling(window=self.adx_period).mean() / atr) * 100
        di_minus = (dm_minus.rolling(window=self.adx_period).mean() / atr) * 100
        
        # Directional Index
        dx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100
        adx = dx.rolling(window=self.adx_period).mean()
        
        return {
            'adx': adx,
            'di_plus': di_plus,
            'di_minus': di_minus,
            'atr': atr
        }
    
    def calculate_indicators(self, coin: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all technical indicators"""
        if len(data) < self.min_candles:
            return {}
        
        close = data['close']
        high = data['high']
        low = data['low']
        
        # Calculate Bollinger Bands
        bb_indicators = self.calculate_bollinger_bands(close)
        
        # Calculate RSI
        rsi = self.calculate_rsi(close)
        
        # Calculate ADX
        adx_indicators = self.calculate_adx(high, low, close)
        
        # Combine all indicators
        indicators = {
            **bb_indicators,
            'rsi': rsi,
            **adx_indicators
        }
        
        # Store indicators for this coin
        self.indicators[coin] = indicators
        
        return indicators
    
    def detect_bb_touch(self, price: float, bb_upper: float, bb_lower: float, 
                       bb_middle: float, tolerance: float = 0.001) -> str:
        """
        Detect if price is touching Bollinger Bands
        
        Args:
            price: Current price
            bb_upper: Upper Bollinger Band
            bb_lower: Lower Bollinger Band
            bb_middle: Middle Bollinger Band (SMA)
            tolerance: Touch tolerance (0.1% by default)
            
        Returns:
            'upper', 'lower', 'middle', or 'none'
        """
        upper_touch = abs(price - bb_upper) / bb_upper <= tolerance
        lower_touch = abs(price - bb_lower) / bb_lower <= tolerance
        middle_touch = abs(price - bb_middle) / bb_middle <= tolerance
        
        if upper_touch:
            return 'upper'
        elif lower_touch:
            return 'lower'
        elif middle_touch:
            return 'middle'
        else:
            return 'none'
    
    def calculate_signal_confidence(self, indicators: Dict[str, Any], 
                                  signal_type: SignalType) -> float:
        """
        Calculate signal confidence based on indicator alignment
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_type: Type of signal being generated
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.0
        
        # Get latest values
        rsi_current = indicators['rsi'].iloc[-1]
        adx_current = indicators['adx'].iloc[-1]
        bb_position = indicators['bb_position'].iloc[-1]
        bb_width = indicators['bb_width'].iloc[-1]
        
        if signal_type == SignalType.LONG:
            # RSI oversold condition
            if rsi_current <= self.rsi_oversold:
                confidence += 0.3
            elif rsi_current <= self.rsi_oversold + 10:
                confidence += 0.2
            
            # Bollinger Band position (lower is better for long)
            if bb_position <= 0.1:  # Very close to lower band
                confidence += 0.3
            elif bb_position <= 0.2:
                confidence += 0.2
            
            # ADX trend strength
            if adx_current >= self.adx_threshold + 10:
                confidence += 0.2
            elif adx_current >= self.adx_threshold:
                confidence += 0.1
            
            # Bollinger Band width (higher volatility)
            if bb_width >= 0.05:  # 5% width
                confidence += 0.1
            
            # DI+ vs DI- for trend direction
            if 'di_plus' in indicators and 'di_minus' in indicators:
                di_plus = indicators['di_plus'].iloc[-1]
                di_minus = indicators['di_minus'].iloc[-1]
                if di_plus > di_minus:
                    confidence += 0.1
        
        elif signal_type == SignalType.SHORT:
            # RSI overbought condition
            if rsi_current >= self.rsi_overbought:
                confidence += 0.3
            elif rsi_current >= self.rsi_overbought - 10:
                confidence += 0.2
            
            # Bollinger Band position (upper is better for short)
            if bb_position >= 0.9:  # Very close to upper band
                confidence += 0.3
            elif bb_position >= 0.8:
                confidence += 0.2
            
            # ADX trend strength
            if adx_current >= self.adx_threshold + 10:
                confidence += 0.2
            elif adx_current >= self.adx_threshold:
                confidence += 0.1
            
            # Bollinger Band width (higher volatility)
            if bb_width >= 0.05:  # 5% width
                confidence += 0.1
            
            # DI+ vs DI- for trend direction
            if 'di_plus' in indicators and 'di_minus' in indicators:
                di_plus = indicators['di_plus'].iloc[-1]
                di_minus = indicators['di_minus'].iloc[-1]
                if di_minus > di_plus:
                    confidence += 0.1
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    async def generate_signal(self, coin: str, market_data: List[MarketData]) -> TradingSignal:
        """Generate trading signal based on BB + RSI + ADX analysis"""
        
        # Convert market data to DataFrame
        if len(market_data) < self.min_candles:
            return TradingSignal(
                signal_type=SignalType.NONE,
                coin=coin,
                confidence=0.0,
                metadata={'reason': 'insufficient_data'}
            )
        
        df = pd.DataFrame([d.to_dict() for d in market_data])
        df.set_index('timestamp', inplace=True)
        
        # Calculate indicators
        indicators = self.calculate_indicators(coin, df)
        if not indicators:
            return TradingSignal(
                signal_type=SignalType.NONE,
                coin=coin,
                confidence=0.0,
                metadata={'reason': 'indicator_calculation_failed'}
            )
        
        # Get current values
        current_price = df['close'].iloc[-1]
        current_rsi = indicators['rsi'].iloc[-1]
        current_adx = indicators['adx'].iloc[-1]
        current_bb_upper = indicators['bb_upper'].iloc[-1]
        current_bb_lower = indicators['bb_lower'].iloc[-1]
        current_bb_middle = indicators['bb_middle'].iloc[-1]
        
        # Check for NaN values
        if pd.isna(current_rsi) or pd.isna(current_adx):
            return TradingSignal(
                signal_type=SignalType.NONE,
                coin=coin,
                confidence=0.0,
                metadata={'reason': 'nan_indicators'}
            )
        
        # Detect Bollinger Band touches
        bb_touch = self.detect_bb_touch(current_price, current_bb_upper, 
                                       current_bb_lower, current_bb_middle)
        
        signal_type = SignalType.NONE
        metadata = {
            'rsi': current_rsi,
            'adx': current_adx,
            'bb_touch': bb_touch,
            'bb_position': indicators['bb_position'].iloc[-1],
            'price': current_price
        }
        
        # Check for existing position to determine exit signals
        existing_position = self.positions.get(coin, {})
        position_size = existing_position.get('size', 0)
        
        # Exit signal logic
        if position_size != 0:
            if position_size > 0:  # Long position
                # Exit long if price crosses middle BB upward or RSI overbought
                if (bb_touch == 'middle' and current_price > current_bb_middle) or \
                   current_rsi >= self.rsi_overbought:
                    signal_type = SignalType.CLOSE_LONG
                    metadata['exit_reason'] = 'bb_middle_cross' if bb_touch == 'middle' else 'rsi_overbought'
            
            elif position_size < 0:  # Short position
                # Exit short if price crosses middle BB downward or RSI oversold
                if (bb_touch == 'middle' and current_price < current_bb_middle) or \
                   current_rsi <= self.rsi_oversold:
                    signal_type = SignalType.CLOSE_SHORT
                    metadata['exit_reason'] = 'bb_middle_cross' if bb_touch == 'middle' else 'rsi_oversold'
        
        # Entry signal logic (only if no existing position)
        else:
            # Long signal conditions
            if (bb_touch == 'lower' and 
                current_rsi <= self.rsi_oversold and 
                current_adx >= self.adx_threshold):
                signal_type = SignalType.LONG
                metadata['entry_reason'] = 'bb_lower_rsi_oversold_adx_strong'
            
            # Short signal conditions
            elif (bb_touch == 'upper' and 
                  current_rsi >= self.rsi_overbought and 
                  current_adx >= self.adx_threshold):
                signal_type = SignalType.SHORT
                metadata['entry_reason'] = 'bb_upper_rsi_overbought_adx_strong'
        
        # Calculate confidence
        confidence = 0.0
        if signal_type != SignalType.NONE:
            if signal_type in [SignalType.LONG, SignalType.SHORT]:
                confidence = self.calculate_signal_confidence(indicators, signal_type)
            else:  # Exit signals
                confidence = 0.8  # High confidence for exit signals
        
        # Create signal
        signal = TradingSignal(
            signal_type=signal_type,
            coin=coin,
            confidence=confidence,
            metadata=metadata
        )
        
        # Add stop loss and take profit for entry signals
        if signal_type in [SignalType.LONG, SignalType.SHORT]:
            signal.entry_price = current_price
            signal.stop_loss = self.calculate_stop_loss(current_price, signal_type)
            signal.take_profit = self.calculate_take_profit(current_price, signal_type)
            signal.size = self.calculate_position_size(coin, signal)
        
        # Add to signal history
        self.add_signal_to_history(signal)
        
        # Log signal if not NONE
        if signal_type != SignalType.NONE:
            logger.info(f"BB RSI ADX Signal: {coin} {signal_type.value} "
                       f"(confidence: {confidence:.2f}, RSI: {current_rsi:.1f}, "
                       f"ADX: {current_adx:.1f}, BB: {bb_touch})")
        
        return signal
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and current state"""
        return {
            'name': self.name,
            'type': 'BB_RSI_ADX',
            'parameters': {
                'bb_period': self.bb_period,
                'bb_std_dev': self.bb_std_dev,
                'rsi_period': self.rsi_period,
                'rsi_overbought': self.rsi_overbought,
                'rsi_oversold': self.rsi_oversold,
                'adx_period': self.adx_period,
                'adx_threshold': self.adx_threshold
            },
            'status': self.get_status(),
            'description': 'Bollinger Bands + RSI + ADX trend-following strategy'
        }

