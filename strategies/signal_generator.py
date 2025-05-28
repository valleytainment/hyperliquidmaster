"""
Signal Generator for Hyperliquid Trading Bot

This module provides robust signal generation capabilities for the Hyperliquid Trading Bot,
with support for multiple technical indicators, timeframes, and adaptive parameters.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure logging
logger = logging.getLogger(__name__)

class RobustSignalGenerator:
    """
    Robust signal generator for Hyperliquid Trading Bot.
    
    Generates trading signals based on multiple technical indicators,
    with support for different timeframes and adaptive parameters.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the signal generator.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.data = None
        self.signals = []
        
        # Default parameters
        self.rsi_period = self.config.get("rsi_period", 14)
        self.rsi_overbought = self.config.get("rsi_overbought", 70)
        self.rsi_oversold = self.config.get("rsi_oversold", 30)
        
        self.macd_fast = self.config.get("macd_fast", 12)
        self.macd_slow = self.config.get("macd_slow", 26)
        self.macd_signal = self.config.get("macd_signal", 9)
        
        self.bb_period = self.config.get("bb_period", 20)
        self.bb_std = self.config.get("bb_std", 2)
        
        logger.info("RobustSignalGenerator initialized")
    
    def set_data(self, data: pd.DataFrame):
        """
        Set data for signal generation.
        
        Args:
            data: DataFrame with OHLCV data
        """
        self.data = data
        logger.debug(f"Data set with shape: {data.shape}")
    
    def generate_signals(self) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on technical indicators.
        
        Returns:
            List of signal dictionaries
        """
        if self.data is None or len(self.data) == 0:
            logger.warning("No data for signal generation")
            return []
        
        # Reset signals
        self.signals = []
        
        # Generate signals from different indicators
        self._generate_rsi_signals()
        self._generate_macd_signals()
        self._generate_bollinger_signals()
        
        # Generate master signal
        self._generate_master_signal()
        
        logger.info(f"Generated {len(self.signals)} signals")
        return self.signals
    
    def _generate_rsi_signals(self):
        """Generate signals based on RSI indicator."""
        try:
            # Calculate RSI
            delta = self.data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=self.rsi_period).mean()
            avg_loss = loss.rolling(window=self.rsi_period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Generate signals
            for i in range(self.rsi_period, len(self.data)):
                # Skip if NaN
                if pd.isna(rsi.iloc[i]):
                    continue
                
                # Check for oversold condition (buy signal)
                if rsi.iloc[i] < self.rsi_oversold and rsi.iloc[i-1] >= self.rsi_oversold:
                    self.signals.append({
                        'timestamp': self.data.index[i],
                        'indicator': 'RSI',
                        'signal': 1,  # Buy
                        'price': self.data['close'].iloc[i],
                        'strength': (self.rsi_oversold - rsi.iloc[i]) / self.rsi_oversold
                    })
                
                # Check for overbought condition (sell signal)
                elif rsi.iloc[i] > self.rsi_overbought and rsi.iloc[i-1] <= self.rsi_overbought:
                    self.signals.append({
                        'timestamp': self.data.index[i],
                        'indicator': 'RSI',
                        'signal': -1,  # Sell
                        'price': self.data['close'].iloc[i],
                        'strength': (rsi.iloc[i] - self.rsi_overbought) / (100 - self.rsi_overbought)
                    })
            
            logger.debug(f"Generated {len(self.signals)} RSI signals")
        except Exception as e:
            logger.error(f"Error generating RSI signals: {e}")
    
    def _generate_macd_signals(self):
        """Generate signals based on MACD indicator."""
        try:
            # Calculate MACD
            exp1 = self.data['close'].ewm(span=self.macd_fast, adjust=False).mean()
            exp2 = self.data['close'].ewm(span=self.macd_slow, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
            histogram = macd - signal
            
            # Generate signals
            for i in range(self.macd_slow + self.macd_signal, len(self.data)):
                # Skip if NaN
                if pd.isna(histogram.iloc[i]) or pd.isna(histogram.iloc[i-1]):
                    continue
                
                # Check for bullish crossover (buy signal)
                if histogram.iloc[i] > 0 and histogram.iloc[i-1] <= 0:
                    self.signals.append({
                        'timestamp': self.data.index[i],
                        'indicator': 'MACD',
                        'signal': 1,  # Buy
                        'price': self.data['close'].iloc[i],
                        'strength': abs(histogram.iloc[i]) / self.data['close'].iloc[i] * 100
                    })
                
                # Check for bearish crossover (sell signal)
                elif histogram.iloc[i] < 0 and histogram.iloc[i-1] >= 0:
                    self.signals.append({
                        'timestamp': self.data.index[i],
                        'indicator': 'MACD',
                        'signal': -1,  # Sell
                        'price': self.data['close'].iloc[i],
                        'strength': abs(histogram.iloc[i]) / self.data['close'].iloc[i] * 100
                    })
            
            logger.debug(f"Generated {len(self.signals) - self._count_indicator_signals('RSI')} MACD signals")
        except Exception as e:
            logger.error(f"Error generating MACD signals: {e}")
    
    def _generate_bollinger_signals(self):
        """Generate signals based on Bollinger Bands indicator."""
        try:
            # Calculate Bollinger Bands
            rolling_mean = self.data['close'].rolling(window=self.bb_period).mean()
            rolling_std = self.data['close'].rolling(window=self.bb_period).std()
            
            upper_band = rolling_mean + (rolling_std * self.bb_std)
            lower_band = rolling_mean - (rolling_std * self.bb_std)
            
            # Generate signals
            for i in range(self.bb_period, len(self.data)):
                # Skip if NaN
                if pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]):
                    continue
                
                # Check for price below lower band (buy signal)
                if self.data['close'].iloc[i] < lower_band.iloc[i] and self.data['close'].iloc[i-1] >= lower_band.iloc[i-1]:
                    self.signals.append({
                        'timestamp': self.data.index[i],
                        'indicator': 'BB',
                        'signal': 1,  # Buy
                        'price': self.data['close'].iloc[i],
                        'strength': (lower_band.iloc[i] - self.data['close'].iloc[i]) / lower_band.iloc[i]
                    })
                
                # Check for price above upper band (sell signal)
                elif self.data['close'].iloc[i] > upper_band.iloc[i] and self.data['close'].iloc[i-1] <= upper_band.iloc[i-1]:
                    self.signals.append({
                        'timestamp': self.data.index[i],
                        'indicator': 'BB',
                        'signal': -1,  # Sell
                        'price': self.data['close'].iloc[i],
                        'strength': (self.data['close'].iloc[i] - upper_band.iloc[i]) / upper_band.iloc[i]
                    })
            
            logger.debug(f"Generated {len(self.signals) - self._count_indicator_signals('RSI') - self._count_indicator_signals('MACD')} BB signals")
        except Exception as e:
            logger.error(f"Error generating Bollinger Bands signals: {e}")
    
    def _generate_master_signal(self):
        """Generate master signal based on all indicators."""
        try:
            # Group signals by timestamp
            signals_by_timestamp = {}
            for signal in self.signals:
                timestamp = signal['timestamp']
                if timestamp not in signals_by_timestamp:
                    signals_by_timestamp[timestamp] = []
                signals_by_timestamp[timestamp].append(signal)
            
            # Generate master signal for each timestamp
            for timestamp, signals in signals_by_timestamp.items():
                # Calculate weighted average signal
                total_signal = 0
                total_weight = 0
                
                for signal in signals:
                    weight = signal['strength']
                    total_signal += signal['signal'] * weight
                    total_weight += weight
                
                if total_weight > 0:
                    avg_signal = total_signal / total_weight
                    
                    # Determine master signal
                    master_signal = 0
                    if avg_signal > 0.5:
                        master_signal = 1  # Buy
                    elif avg_signal < -0.5:
                        master_signal = -1  # Sell
                    
                    # Add master signal
                    self.signals.append({
                        'timestamp': timestamp,
                        'indicator': 'MASTER',
                        'signal': master_signal,
                        'price': signals[0]['price'],
                        'strength': abs(avg_signal)
                    })
            
            logger.debug(f"Generated {self._count_indicator_signals('MASTER')} master signals")
        except Exception as e:
            logger.error(f"Error generating master signals: {e}")
    
    def _count_indicator_signals(self, indicator: str) -> int:
        """
        Count signals for a specific indicator.
        
        Args:
            indicator: Indicator name
        
        Returns:
            Number of signals for the indicator
        """
        return sum(1 for signal in self.signals if signal['indicator'] == indicator)
    
    def generate_master_signal(self, data: Optional[pd.DataFrame] = None) -> int:
        """
        Generate a single master signal for the latest data point.
        
        Args:
            data: DataFrame with OHLCV data (optional, uses self.data if None)
        
        Returns:
            Signal value: 1 (buy), -1 (sell), or 0 (neutral)
        """
        if data is not None:
            self.set_data(data)
        
        if self.data is None or len(self.data) == 0:
            logger.warning("No data for master signal generation")
            return 0
        
        # Generate signals
        self.generate_signals()
        
        # Get latest master signal
        master_signals = [s for s in self.signals if s['indicator'] == 'MASTER']
        
        if not master_signals:
            return 0
        
        # Sort by timestamp (descending) and get the latest
        latest_signal = sorted(master_signals, key=lambda s: s['timestamp'], reverse=True)[0]
        
        return latest_signal['signal']
    
    def check_rsi_signal(self, data: Optional[pd.DataFrame] = None) -> int:
        """
        Check for RSI signal in the latest data point.
        
        Args:
            data: DataFrame with OHLCV data (optional, uses self.data if None)
        
        Returns:
            Signal value: 1 (buy), -1 (sell), or 0 (neutral)
        """
        if data is not None:
            self.set_data(data)
        
        if self.data is None or len(self.data) == 0:
            logger.warning("No data for RSI signal check")
            return 0
        
        try:
            # Calculate RSI
            delta = self.data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=self.rsi_period).mean()
            avg_loss = loss.rolling(window=self.rsi_period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Check latest RSI value
            latest_rsi = rsi.iloc[-1]
            
            if pd.isna(latest_rsi):
                return 0
            
            if latest_rsi < self.rsi_oversold:
                return 1  # Buy signal
            elif latest_rsi > self.rsi_overbought:
                return -1  # Sell signal
            
            return 0  # Neutral
        except Exception as e:
            logger.error(f"Error checking RSI signal: {e}")
            return 0
