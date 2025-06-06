"""
Bollinger Bands, RSI, and ADX Strategy for Hyperliquid Trading Bot
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

from utils.logger import get_logger
from strategies.base_strategy import BaseStrategy
from strategies.trading_types_fixed import TradingSignal, SignalType, MarketData

logger = get_logger(__name__)


class BB_RSI_ADX(BaseStrategy):
    """
    Bollinger Bands, RSI, and ADX Strategy for Hyperliquid Trading Bot
    """
    
    def __init__(self, api=None, risk_manager=None, max_positions=3):
        """
        Initialize the strategy
        
        Args:
            api: API instance
            risk_manager: Risk manager instance
            max_positions: Maximum number of positions
        """
        super().__init__(api, risk_manager, max_positions)
        
        # Initialize parameters
        self.parameters = self.get_default_parameters()
        
        logger.info(f"BB RSI ADX Strategy initialized with BB({self.parameters['bb_period']}, {self.parameters['bb_std']}), RSI({self.parameters['rsi_period']}, {self.parameters['rsi_oversold']}, {self.parameters['rsi_overbought']}), ADX({self.parameters['adx_period']}, {self.parameters['adx_threshold']})")
    
    @classmethod
    def get_default_parameters(cls):
        """
        Get default parameters
        
        Returns:
            dict: Default parameters
        """
        return {
            "max_positions": 3,
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_oversold": 25,
            "rsi_overbought": 75,
            "adx_period": 14,
            "adx_threshold": 25
        }
    
    def execute(self):
        """
        Execute the strategy
        """
        try:
            # Update positions and orders
            self.update_positions()
            self.update_orders()
            
            # Get available coins
            coins = self._get_available_coins()
            
            # Check each coin
            for coin in coins:
                # Skip if we already have a position
                if coin in self.positions:
                    continue
                
                # Skip if we have reached max positions
                if len(self.positions) >= self.max_positions:
                    break
                
                # Get market data
                market_data = self._get_market_data(coin)
                
                # Generate signal
                signal = self._generate_signal(market_data)
                
                # Execute signal
                if signal and signal.signal_type != SignalType.NEUTRAL:
                    self._execute_signal(signal)
            
            # Check existing positions
            for coin in list(self.positions.keys()):
                # Get market data
                market_data = self._get_market_data(coin)
                
                # Generate signal
                signal = self._generate_signal(market_data)
                
                # Execute signal
                if signal and signal.signal_type != SignalType.NEUTRAL:
                    self._execute_signal(signal)
        except Exception as e:
            logger.error(f"Failed to execute strategy: {e}")
    
    def _get_available_coins(self):
        """
        Get available coins
        
        Returns:
            list: Available coins
        """
        try:
            if not self.api:
                logger.warning("No API instance available")
                return []
            
            # Get markets
            markets = self.api.get_markets()
            
            if not markets:
                logger.warning("No markets available")
                return []
            
            # Extract coin names
            coins = []
            for market in markets:
                if "name" in market:
                    coins.append(market["name"])
            
            return coins
        except Exception as e:
            logger.error(f"Failed to get available coins: {e}")
            return []
    
    def _get_market_data(self, coin):
        """
        Get market data
        
        Args:
            coin: Coin to get data for
        
        Returns:
            MarketData: Market data
        """
        try:
            if not self.api:
                logger.warning("No API instance available")
                return None
            
            # Get candles
            candles = self.api.get_candles(coin, "1h", 100)
            
            if not candles:
                logger.warning(f"No candles available for {coin}")
                return None
            
            # Create market data
            market_data = MarketData(coin, "1h", candles)
            
            return market_data
        except Exception as e:
            logger.error(f"Failed to get market data for {coin}: {e}")
            return None
    
    def _generate_signal(self, market_data):
        """
        Generate trading signal
        
        Args:
            market_data: Market data
        
        Returns:
            TradingSignal: Trading signal
        """
        try:
            if not market_data or not market_data.candles:
                return None
            
            # Extract parameters
            bb_period = self.parameters["bb_period"]
            bb_std = self.parameters["bb_std"]
            rsi_period = self.parameters["rsi_period"]
            rsi_oversold = self.parameters["rsi_oversold"]
            rsi_overbought = self.parameters["rsi_overbought"]
            adx_period = self.parameters["adx_period"]
            adx_threshold = self.parameters["adx_threshold"]
            
            # Convert candles to DataFrame
            df = pd.DataFrame(market_data.candles)
            
            # Calculate indicators
            df = self._calculate_bollinger_bands(df, bb_period, bb_std)
            df = self._calculate_rsi(df, rsi_period)
            df = self._calculate_adx(df, adx_period)
            
            # Get latest values
            latest = df.iloc[-1]
            
            # Generate signal
            signal_type = SignalType.NEUTRAL
            confidence = 0.0
            reason = ""
            
            # Check if price is below lower band and RSI is oversold
            if latest["close"] < latest["bb_lower"] and latest["rsi"] < rsi_oversold and latest["adx"] > adx_threshold:
                signal_type = SignalType.BUY
                confidence = min(1.0, (rsi_oversold - latest["rsi"]) / rsi_oversold)
                reason = f"Price below lower band, RSI oversold ({latest['rsi']:.2f}), ADX strong ({latest['adx']:.2f})"
            
            # Check if price is above upper band and RSI is overbought
            elif latest["close"] > latest["bb_upper"] and latest["rsi"] > rsi_overbought and latest["adx"] > adx_threshold:
                signal_type = SignalType.SELL
                confidence = min(1.0, (latest["rsi"] - rsi_overbought) / (100 - rsi_overbought))
                reason = f"Price above upper band, RSI overbought ({latest['rsi']:.2f}), ADX strong ({latest['adx']:.2f})"
            
            # Create signal
            if signal_type != SignalType.NEUTRAL:
                return TradingSignal(
                    market_data.coin,
                    signal_type,
                    confidence,
                    latest["close"],
                    None,
                    reason
                )
            
            return None
        except Exception as e:
            logger.error(f"Failed to generate signal: {e}")
            return None
    
    def _execute_signal(self, signal):
        """
        Execute trading signal
        
        Args:
            signal: Trading signal
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not signal:
                return False
            
            # Check if signal is within risk limits
            if not self.risk_manager.check_signal(signal):
                logger.warning(f"Signal rejected by risk manager: {signal}")
                return False
            
            # Calculate position size
            account_value = self.get_account_value()
            position_size = account_value * 0.1  # 10% of account value
            
            # Adjust position size
            position_size = self.risk_manager.adjust_position_size(
                signal.coin,
                signal.signal_type.value,
                position_size,
                signal.price
            )
            
            # Execute signal
            if signal.signal_type == SignalType.BUY:
                # Place buy order
                result = self.place_order(
                    signal.coin,
                    "buy",
                    position_size,
                    signal.price
                )
                
                if result:
                    logger.info(f"Executed buy signal for {signal.coin} at {signal.price}")
                    return True
            
            elif signal.signal_type == SignalType.SELL:
                # Check if we have a position
                if signal.coin in self.positions:
                    # Close position
                    result = self.close_position(signal.coin)
                    
                    if result:
                        logger.info(f"Executed sell signal for {signal.coin} at {signal.price}")
                        return True
                else:
                    # Place sell order
                    result = self.place_order(
                        signal.coin,
                        "sell",
                        position_size,
                        signal.price
                    )
                    
                    if result:
                        logger.info(f"Executed sell signal for {signal.coin} at {signal.price}")
                        return True
            
            return False
        except Exception as e:
            logger.error(f"Failed to execute signal: {e}")
            return False
    
    def _calculate_bollinger_bands(self, df, period=20, std=2.0):
        """
        Calculate Bollinger Bands
        
        Args:
            df: DataFrame with candles
            period: Period for moving average
            std: Standard deviation multiplier
        
        Returns:
            DataFrame: DataFrame with Bollinger Bands
        """
        try:
            # Calculate moving average
            df["bb_middle"] = df["close"].rolling(window=period).mean()
            
            # Calculate standard deviation
            df["bb_std"] = df["close"].rolling(window=period).std()
            
            # Calculate upper and lower bands
            df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * std)
            df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * std)
            
            return df
        except Exception as e:
            logger.error(f"Failed to calculate Bollinger Bands: {e}")
            return df
    
    def _calculate_rsi(self, df, period=14):
        """
        Calculate RSI
        
        Args:
            df: DataFrame with candles
            period: Period for RSI
        
        Returns:
            DataFrame: DataFrame with RSI
        """
        try:
            # Calculate price change
            df["price_change"] = df["close"].diff()
            
            # Calculate gains and losses
            df["gain"] = df["price_change"].apply(lambda x: x if x > 0 else 0)
            df["loss"] = df["price_change"].apply(lambda x: -x if x < 0 else 0)
            
            # Calculate average gains and losses
            df["avg_gain"] = df["gain"].rolling(window=period).mean()
            df["avg_loss"] = df["loss"].rolling(window=period).mean()
            
            # Calculate RS
            df["rs"] = df["avg_gain"] / df["avg_loss"]
            
            # Calculate RSI
            df["rsi"] = 100 - (100 / (1 + df["rs"]))
            
            return df
        except Exception as e:
            logger.error(f"Failed to calculate RSI: {e}")
            return df
    
    def _calculate_adx(self, df, period=14):
        """
        Calculate ADX
        
        Args:
            df: DataFrame with candles
            period: Period for ADX
        
        Returns:
            DataFrame: DataFrame with ADX
        """
        try:
            # Calculate True Range
            df["tr1"] = abs(df["high"] - df["low"])
            df["tr2"] = abs(df["high"] - df["close"].shift())
            df["tr3"] = abs(df["low"] - df["close"].shift())
            df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
            
            # Calculate Directional Movement
            df["up_move"] = df["high"] - df["high"].shift()
            df["down_move"] = df["low"].shift() - df["low"]
            
            df["plus_dm"] = np.where(
                (df["up_move"] > df["down_move"]) & (df["up_move"] > 0),
                df["up_move"],
                0
            )
            
            df["minus_dm"] = np.where(
                (df["down_move"] > df["up_move"]) & (df["down_move"] > 0),
                df["down_move"],
                0
            )
            
            # Calculate Smoothed Averages
            df["atr"] = df["tr"].rolling(window=period).mean()
            df["plus_di"] = 100 * (df["plus_dm"].rolling(window=period).mean() / df["atr"])
            df["minus_di"] = 100 * (df["minus_dm"].rolling(window=period).mean() / df["atr"])
            
            # Calculate Directional Index
            df["dx"] = 100 * (abs(df["plus_di"] - df["minus_di"]) / (df["plus_di"] + df["minus_di"]))
            
            # Calculate ADX
            df["adx"] = df["dx"].rolling(window=period).mean()
            
            return df
        except Exception as e:
            logger.error(f"Failed to calculate ADX: {e}")
            return df

