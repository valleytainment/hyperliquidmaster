"""
Hull Moving Average Suite Strategy for Hyperliquid Trading Bot
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

from utils.logger import get_logger
from strategies.base_strategy import BaseStrategy
from strategies.trading_types import TradingSignal, SignalType, MarketData, OrderType

logger = get_logger(__name__)


class HullSuiteStrategy(BaseStrategy):
    """
    Hull Moving Average Suite Strategy for Hyperliquid Trading Bot
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
        
        logger.info(f"Hull Suite Strategy initialized with Hull MA({self.parameters['hull_period']}), ATR({self.parameters['atr_period']}, {self.parameters['atr_multiplier']})")
    
    @classmethod
    def get_default_parameters(cls):
        """
        Get default parameters
        
        Returns:
            dict: Default parameters
        """
        return {
            "max_positions": 3,
            "hull_period": 34,
            "atr_period": 14,
            "atr_multiplier": 2.0
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
            hull_period = self.parameters["hull_period"]
            atr_period = self.parameters["atr_period"]
            atr_multiplier = self.parameters["atr_multiplier"]
            
            # Convert candles to DataFrame
            df = pd.DataFrame(market_data.candles)
            
            # Calculate indicators
            df = self._calculate_hull_ma(df, hull_period)
            df = self._calculate_atr(df, atr_period)
            
            # Get latest values
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Generate signal
            signal_type = SignalType.NEUTRAL
            confidence = 0.0
            reason = ""
            
            # Check for buy signal
            if previous["close"] < previous["hull_ma"] and latest["close"] > latest["hull_ma"]:
                signal_type = SignalType.BUY
                confidence = 0.8
                reason = f"Price crossed above Hull MA ({latest['hull_ma']:.2f})"
            
            # Check for sell signal
            elif previous["close"] > previous["hull_ma"] and latest["close"] < latest["hull_ma"]:
                signal_type = SignalType.SELL
                confidence = 0.8
                reason = f"Price crossed below Hull MA ({latest['hull_ma']:.2f})"
            
            # Check for stop loss
            elif market_data.coin in self.positions:
                position = self.positions[market_data.coin]
                
                if position["size"] > 0:  # Long position
                    stop_price = latest["hull_ma"] - (latest["atr"] * atr_multiplier)
                    
                    if latest["close"] < stop_price:
                        signal_type = SignalType.SELL
                        confidence = 0.9
                        reason = f"Stop loss triggered at {stop_price:.2f}"
                
                elif position["size"] < 0:  # Short position
                    stop_price = latest["hull_ma"] + (latest["atr"] * atr_multiplier)
                    
                    if latest["close"] > stop_price:
                        signal_type = SignalType.BUY
                        confidence = 0.9
                        reason = f"Stop loss triggered at {stop_price:.2f}"
            
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
    
    def _calculate_hull_ma(self, df, period=34):
        """
        Calculate Hull Moving Average
        
        Args:
            df: DataFrame with candles
            period: Period for Hull MA
        
        Returns:
            DataFrame: DataFrame with Hull MA
        """
        try:
            # Calculate WMA with period/2
            half_period = int(period / 2)
            df["wma_half"] = self._calculate_wma(df["close"], half_period)
            
            # Calculate WMA with period
            df["wma_full"] = self._calculate_wma(df["close"], period)
            
            # Calculate raw Hull MA
            df["hull_raw"] = 2 * df["wma_half"] - df["wma_full"]
            
            # Calculate Hull MA
            sqrt_period = int(np.sqrt(period))
            df["hull_ma"] = self._calculate_wma(df["hull_raw"], sqrt_period)
            
            return df
        except Exception as e:
            logger.error(f"Failed to calculate Hull MA: {e}")
            return df
    
    def _calculate_wma(self, series, period):
        """
        Calculate Weighted Moving Average
        
        Args:
            series: Series to calculate WMA for
            period: Period for WMA
        
        Returns:
            Series: WMA series
        """
        try:
            weights = np.arange(1, period + 1)
            return series.rolling(period).apply(lambda x: np.sum(weights * x) / weights.sum(), raw=True)
        except Exception as e:
            logger.error(f"Failed to calculate WMA: {e}")
            return series
    
    def _calculate_atr(self, df, period=14):
        """
        Calculate Average True Range
        
        Args:
            df: DataFrame with candles
            period: Period for ATR
        
        Returns:
            DataFrame: DataFrame with ATR
        """
        try:
            # Calculate True Range
            df["tr1"] = abs(df["high"] - df["low"])
            df["tr2"] = abs(df["high"] - df["close"].shift())
            df["tr3"] = abs(df["low"] - df["close"].shift())
            df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
            
            # Calculate ATR
            df["atr"] = df["tr"].rolling(window=period).mean()
            
            return df
        except Exception as e:
            logger.error(f"Failed to calculate ATR: {e}")
            return df

