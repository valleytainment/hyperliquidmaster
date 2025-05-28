"""
Master Strategy for Hyperliquid Trading Bot

This module provides the master trading strategy implementation for the Hyperliquid Trading Bot,
integrating multiple signal generators and technical indicators for comprehensive trading decisions.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure logging
logger = logging.getLogger(__name__)

class MasterOmniOverlordStrategy:
    """
    Master Omni Overlord Strategy for Hyperliquid Trading Bot.
    
    Integrates multiple signal generators and technical indicators
    for comprehensive trading decisions across different market conditions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the master strategy.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.data = None
        self.signals = []
        
        # Default parameters
        self.trend_period = self.config.get("trend_period", 50)
        self.volatility_period = self.config.get("volatility_period", 20)
        self.signal_threshold = self.config.get("signal_threshold", 0.7)
        
        # Strategy state
        self.current_position = 0  # -1 (short), 0 (neutral), 1 (long)
        self.position_entry_price = 0
        self.position_entry_time = None
        
        logger.info("MasterOmniOverlordStrategy initialized")
    
    def set_data(self, data: pd.DataFrame):
        """
        Set data for strategy execution.
        
        Args:
            data: DataFrame with OHLCV data
        """
        self.data = data
        logger.debug(f"Data set with shape: {data.shape}")
    
    def detect_market_condition(self) -> str:
        """
        Detect current market condition.
        
        Returns:
            Market condition: 'trending', 'ranging', 'volatile', or 'unknown'
        """
        if self.data is None or len(self.data) < self.trend_period:
            logger.warning("Insufficient data for market condition detection")
            return "unknown"
        
        try:
            # Calculate ADX for trend strength
            high = self.data['high']
            low = self.data['low']
            close = self.data['close']
            
            # True Range
            tr1 = abs(high - low)
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            
            # Directional Movement
            plus_dm = high.diff()
            minus_dm = low.diff()
            plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm.abs()), 0)
            minus_dm = minus_dm.abs().where((minus_dm < 0) & (minus_dm.abs() > plus_dm), 0)
            
            # Directional Indicators
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
            
            # Average Directional Index
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=14).mean()
            
            # Volatility (using ATR/Price ratio)
            volatility = atr / close * 100
            
            # Latest values
            latest_adx = adx.iloc[-1]
            latest_volatility = volatility.iloc[-1]
            
            # Determine market condition
            if latest_adx > 25:
                return "trending"
            elif latest_volatility > 5:
                return "volatile"
            else:
                return "ranging"
        except Exception as e:
            logger.error(f"Error detecting market condition: {e}")
            return "unknown"
    
    def generate_signal(self) -> int:
        """
        Generate trading signal based on current market condition and data.
        
        Returns:
            Signal value: 1 (buy), -1 (sell), or 0 (neutral)
        """
        if self.data is None or len(self.data) == 0:
            logger.warning("No data for signal generation")
            return 0
        
        try:
            # Detect market condition
            market_condition = self.detect_market_condition()
            logger.debug(f"Detected market condition: {market_condition}")
            
            # Generate signal based on market condition
            if market_condition == "trending":
                return self._generate_trend_following_signal()
            elif market_condition == "ranging":
                return self._generate_mean_reversion_signal()
            elif market_condition == "volatile":
                return self._generate_volatility_breakout_signal()
            else:
                return 0  # Neutral signal for unknown market condition
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return 0
    
    def _generate_trend_following_signal(self) -> int:
        """
        Generate trend following signal.
        
        Returns:
            Signal value: 1 (buy), -1 (sell), or 0 (neutral)
        """
        try:
            # Calculate EMAs
            ema_fast = self.data['close'].ewm(span=20, adjust=False).mean()
            ema_slow = self.data['close'].ewm(span=50, adjust=False).mean()
            
            # Calculate MACD
            ema12 = self.data['close'].ewm(span=12, adjust=False).mean()
            ema26 = self.data['close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal_line = macd.ewm(span=9, adjust=False).mean()
            
            # Latest values
            latest_ema_fast = ema_fast.iloc[-1]
            latest_ema_slow = ema_slow.iloc[-1]
            latest_macd = macd.iloc[-1]
            latest_signal = signal_line.iloc[-1]
            
            # Generate signal
            if latest_ema_fast > latest_ema_slow and latest_macd > latest_signal:
                return 1  # Buy signal
            elif latest_ema_fast < latest_ema_slow and latest_macd < latest_signal:
                return -1  # Sell signal
            
            return 0  # Neutral signal
        except Exception as e:
            logger.error(f"Error generating trend following signal: {e}")
            return 0
    
    def _generate_mean_reversion_signal(self) -> int:
        """
        Generate mean reversion signal.
        
        Returns:
            Signal value: 1 (buy), -1 (sell), or 0 (neutral)
        """
        try:
            # Calculate Bollinger Bands
            rolling_mean = self.data['close'].rolling(window=20).mean()
            rolling_std = self.data['close'].rolling(window=20).std()
            
            upper_band = rolling_mean + (rolling_std * 2)
            lower_band = rolling_mean - (rolling_std * 2)
            
            # Calculate RSI
            delta = self.data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Latest values
            latest_close = self.data['close'].iloc[-1]
            latest_upper = upper_band.iloc[-1]
            latest_lower = lower_band.iloc[-1]
            latest_rsi = rsi.iloc[-1]
            
            # Generate signal
            if latest_close < latest_lower and latest_rsi < 30:
                return 1  # Buy signal
            elif latest_close > latest_upper and latest_rsi > 70:
                return -1  # Sell signal
            
            return 0  # Neutral signal
        except Exception as e:
            logger.error(f"Error generating mean reversion signal: {e}")
            return 0
    
    def _generate_volatility_breakout_signal(self) -> int:
        """
        Generate volatility breakout signal.
        
        Returns:
            Signal value: 1 (buy), -1 (sell), or 0 (neutral)
        """
        try:
            # Calculate ATR
            high = self.data['high']
            low = self.data['low']
            close = self.data['close']
            
            tr1 = abs(high - low)
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            
            # Calculate Donchian Channels
            high_channel = high.rolling(window=20).max()
            low_channel = low.rolling(window=20).min()
            
            # Latest values
            latest_close = close.iloc[-1]
            latest_high_channel = high_channel.iloc[-1]
            latest_low_channel = low_channel.iloc[-1]
            latest_atr = atr.iloc[-1]
            
            # Generate signal
            if latest_close > latest_high_channel - (0.5 * latest_atr):
                return 1  # Buy signal
            elif latest_close < latest_low_channel + (0.5 * latest_atr):
                return -1  # Sell signal
            
            return 0  # Neutral signal
        except Exception as e:
            logger.error(f"Error generating volatility breakout signal: {e}")
            return 0
    
    def update_position(self, signal: int, price: float, timestamp: Any):
        """
        Update current position based on signal.
        
        Args:
            signal: Signal value (1, -1, or 0)
            price: Current price
            timestamp: Current timestamp
        """
        if signal == 0:
            return
        
        # Check if signal is strong enough to change position
        if signal == 1 and self.current_position <= 0:
            # Enter long position
            self.current_position = 1
            self.position_entry_price = price
            self.position_entry_time = timestamp
            
            logger.info(f"Entered long position at {price}")
        elif signal == -1 and self.current_position >= 0:
            # Enter short position
            self.current_position = -1
            self.position_entry_price = price
            self.position_entry_time = timestamp
            
            logger.info(f"Entered short position at {price}")
    
    def calculate_pnl(self, current_price: float) -> float:
        """
        Calculate profit/loss for current position.
        
        Args:
            current_price: Current price
        
        Returns:
            Profit/loss percentage
        """
        if self.current_position == 0 or self.position_entry_price == 0:
            return 0
        
        if self.current_position == 1:
            # Long position
            return (current_price - self.position_entry_price) / self.position_entry_price * 100
        else:
            # Short position
            return (self.position_entry_price - current_price) / self.position_entry_price * 100
    
    def should_exit_position(self, current_price: float, current_time: Any) -> bool:
        """
        Check if current position should be exited.
        
        Args:
            current_price: Current price
            current_time: Current timestamp
        
        Returns:
            True if position should be exited, False otherwise
        """
        if self.current_position == 0:
            return False
        
        # Calculate P&L
        pnl = self.calculate_pnl(current_price)
        
        # Check stop loss (-5%)
        if pnl < -5:
            logger.info(f"Exiting position due to stop loss: {pnl:.2f}%")
            return True
        
        # Check take profit (10%)
        if pnl > 10:
            logger.info(f"Exiting position due to take profit: {pnl:.2f}%")
            return True
        
        # Check time-based exit (48 hours)
        if current_time is not None and self.position_entry_time is not None:
            time_diff = current_time - self.position_entry_time
            if hasattr(time_diff, 'total_seconds') and time_diff.total_seconds() > 48 * 3600:
                logger.info(f"Exiting position due to time limit: {time_diff}")
                return True
        
        return False
    
    def exit_position(self, current_price: float):
        """
        Exit current position.
        
        Args:
            current_price: Current price
        """
        if self.current_position == 0:
            return
        
        # Calculate P&L
        pnl = self.calculate_pnl(current_price)
        
        logger.info(f"Exited {'long' if self.current_position == 1 else 'short'} position at {current_price} with P&L: {pnl:.2f}%")
        
        # Reset position
        self.current_position = 0
        self.position_entry_price = 0
        self.position_entry_time = None
    
    def execute(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Execute strategy on data.
        
        Args:
            data: DataFrame with OHLCV data (optional, uses self.data if None)
        
        Returns:
            Dictionary with execution results
        """
        if data is not None:
            self.set_data(data)
        
        if self.data is None or len(self.data) == 0:
            logger.warning("No data for strategy execution")
            return {
                'signal': 0,
                'position': 0,
                'market_condition': 'unknown',
                'pnl': 0
            }
        
        try:
            # Get latest price and timestamp
            latest_price = self.data['close'].iloc[-1]
            latest_timestamp = self.data.index[-1] if hasattr(self.data, 'index') else None
            
            # Check if current position should be exited
            if self.should_exit_position(latest_price, latest_timestamp):
                self.exit_position(latest_price)
            
            # Generate signal
            signal = self.generate_signal()
            
            # Update position
            self.update_position(signal, latest_price, latest_timestamp)
            
            # Calculate P&L
            pnl = self.calculate_pnl(latest_price)
            
            # Detect market condition
            market_condition = self.detect_market_condition()
            
            return {
                'signal': signal,
                'position': self.current_position,
                'market_condition': market_condition,
                'pnl': pnl
            }
        except Exception as e:
            logger.error(f"Error executing strategy: {e}")
            return {
                'signal': 0,
                'position': self.current_position,
                'market_condition': 'unknown',
                'pnl': 0
            }
