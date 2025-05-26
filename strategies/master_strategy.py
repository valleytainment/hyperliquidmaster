"""
Master Strategy for Hyperliquid Trading Bot

This module provides a comprehensive trading strategy that combines multiple
signal generators and adapts to different market conditions.

Classes:
    MasterStrategy: Implements the master trading strategy
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/master_strategy.log")
    ]
)
logger = logging.getLogger(__name__)

class MasterStrategy:
    """
    Master trading strategy for Hyperliquid trading bot.
    
    This class combines multiple signal generators and adapts to different
    market conditions to generate robust trading signals.
    
    Attributes:
        symbol (str): Trading pair symbol
        timeframe (str): Timeframe for analysis
        signal_generator: Signal generator for technical analysis
        error_handler: Error handler for exception management
        config (dict): Configuration parameters
        data: Market data for analysis
    """
    
    def __init__(self, symbol, timeframe, signal_generator, error_handler, config=None):
        """
        Initialize the master strategy.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe for analysis
            signal_generator: Signal generator for technical analysis
            error_handler: Error handler for exception management
            config (dict, optional): Configuration parameters
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.signal_generator = signal_generator
        self.error_handler = error_handler
        self.data = None
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Default configuration
        self.config = {
            "risk_level": 0.01,  # 1% risk per trade
            "take_profit_multiplier": 2.0,  # Take profit at 2x risk
            "stop_loss_multiplier": 1.0,  # Stop loss at 1x risk
            "use_volatility_filters": True,
            "use_trend_filters": True,
            "use_volume_filters": True,
            "use_regime_detection": True,
            "strategy_weights": {
                "triple_confluence": 0.5,
                "oracle_update": 0.5
            },
            "adaptive_parameters": {
                "signal_threshold": 0.5,
                "trend_filter_strength": 0.2,
                "mean_reversion_factor": 0.1,
                "volatility_adjustment": 1.0
            }
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        logger.info(f"Initializing Master Strategy for {symbol} ({timeframe})")
    
    def set_data(self, data):
        """
        Set market data for analysis.
        
        Args:
            data: Market data (DataFrame)
        """
        self.data = data
        logger.info(f"Set data for {self.symbol} with {len(data)} candles")
    
    def generate_signal(self, candle_data=None):
        """
        Generate a trading signal.
        
        Args:
            candle_data (dict, optional): Current candle data
        
        Returns:
            int: Signal (-1 for sell, 0 for neutral, 1 for buy)
        """
        try:
            logger.info(f"Generating signal for {self.symbol} using Master Strategy")
            
            # Check if we have data
            if self.data is None and candle_data is None:
                logger.warning("No data for signal generation")
                return 0
            
            # Detect market regime if enabled
            if self.config["use_regime_detection"]:
                regime = self.signal_generator.detect_market_regime(self.data if candle_data is None else candle_data)
                logger.info(f"Market regime detected: {regime}")
                
                # Adjust strategy weights based on regime
                if regime == 'trending':
                    # In trending markets, favor trend-following strategies
                    self.config["strategy_weights"]["triple_confluence"] = 0.7
                    self.config["strategy_weights"]["oracle_update"] = 0.3
                elif regime == 'ranging':
                    # In ranging markets, favor mean-reversion strategies
                    self.config["strategy_weights"]["triple_confluence"] = 0.4
                    self.config["strategy_weights"]["oracle_update"] = 0.6
                elif regime == 'volatile':
                    # In volatile markets, favor more conservative strategies
                    self.config["strategy_weights"]["triple_confluence"] = 0.3
                    self.config["strategy_weights"]["oracle_update"] = 0.7
                
                logger.info(f"Adjusted strategy weights: Triple Confluence={self.config['strategy_weights']['triple_confluence']:.2f}, Oracle Update={self.config['strategy_weights']['oracle_update']:.2f}")
                
                # Adjust adaptive parameters based on regime
                if regime == 'trending':
                    self.config["adaptive_parameters"]["signal_threshold"] = 0.4
                    self.config["adaptive_parameters"]["trend_filter_strength"] = 0.3
                    self.config["adaptive_parameters"]["mean_reversion_factor"] = 0.05
                    self.config["adaptive_parameters"]["volatility_adjustment"] = 0.8
                elif regime == 'ranging':
                    self.config["adaptive_parameters"]["signal_threshold"] = 0.6
                    self.config["adaptive_parameters"]["trend_filter_strength"] = 0.1
                    self.config["adaptive_parameters"]["mean_reversion_factor"] = 0.3
                    self.config["adaptive_parameters"]["volatility_adjustment"] = 1.0
                elif regime == 'volatile':
                    self.config["adaptive_parameters"]["signal_threshold"] = 0.8
                    self.config["adaptive_parameters"]["trend_filter_strength"] = 0.4
                    self.config["adaptive_parameters"]["mean_reversion_factor"] = 0.2
                    self.config["adaptive_parameters"]["volatility_adjustment"] = 1.2
                
                logger.info(f"Adjusted adaptive parameters: {self.config['adaptive_parameters']}")
            
            # Generate signal using signal generator
            signal = self.signal_generator.generate_signal(candle_data if candle_data is not None else self.data, self.timeframe)
            
            # Apply additional filters if enabled
            if signal != 0:
                # Volatility filter
                if self.config["use_volatility_filters"] and not self._check_volatility_filter(signal):
                    logger.info("Signal rejected by volatility filter")
                    signal = 0
                
                # Trend filter
                if self.config["use_trend_filters"] and not self._check_trend_filter(signal):
                    logger.info("Signal rejected by trend filter")
                    signal = 0
                
                # Volume filter
                if self.config["use_volume_filters"] and not self._check_volume_filter(signal):
                    logger.info("Signal rejected by volume filter")
                    signal = 0
            
            logger.info(f"Generated signal for {self.symbol}: {signal}")
            
            return signal
        
        except Exception as e:
            error_context = {"symbol": self.symbol, "timeframe": self.timeframe}
            self.error_handler.handle_error("generate_signal", str(e), context=error_context)
            return 0
    
    def _check_volatility_filter(self, signal):
        """
        Check if the signal passes the volatility filter.
        
        Args:
            signal (int): Signal to check
        
        Returns:
            bool: True if the signal passes the filter, False otherwise
        """
        try:
            # Get volatility data
            if self.data is None:
                return True
            
            # Calculate historical volatility
            if len(self.data) < 20:
                return True
            
            close_prices = self.data['close'].values if isinstance(self.data, pd.DataFrame) else self.data.close.values
            returns = np.diff(close_prices) / close_prices[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
            
            # Adjust volatility threshold based on adaptive parameters
            volatility_threshold = 0.5 * self.config["adaptive_parameters"]["volatility_adjustment"]
            
            # Check if volatility is within acceptable range
            if signal > 0:  # Buy signal
                # For buy signals, we want volatility to be lower
                return volatility < volatility_threshold
            else:  # Sell signal
                # For sell signals, we want volatility to be higher
                return volatility > volatility_threshold * 0.8
        
        except Exception as e:
            error_context = {"signal": signal}
            self.error_handler.handle_error("_check_volatility_filter", str(e), context=error_context)
            return True
    
    def _check_trend_filter(self, signal):
        """
        Check if the signal passes the trend filter.
        
        Args:
            signal (int): Signal to check
        
        Returns:
            bool: True if the signal passes the filter, False otherwise
        """
        try:
            # Get trend data
            if self.data is None:
                return True
            
            # Calculate trend
            if len(self.data) < 50:
                return True
            
            close_prices = self.data['close'].values if isinstance(self.data, pd.DataFrame) else self.data.close.values
            
            # Calculate short-term and long-term moving averages
            short_ma = np.mean(close_prices[-20:])
            long_ma = np.mean(close_prices[-50:])
            
            # Calculate trend direction
            trend_direction = 1 if short_ma > long_ma else -1
            
            # Adjust trend filter strength based on adaptive parameters
            trend_filter_strength = self.config["adaptive_parameters"]["trend_filter_strength"]
            
            # Check if signal aligns with trend
            if trend_filter_strength > 0.5:
                # Strong trend filter: signal must align with trend
                return signal * trend_direction > 0
            else:
                # Weak trend filter: allow counter-trend signals with higher threshold
                if signal * trend_direction > 0:
                    return True
                else:
                    # Counter-trend signal needs to be stronger
                    return abs(signal) > 0.7
        
        except Exception as e:
            error_context = {"signal": signal}
            self.error_handler.handle_error("_check_trend_filter", str(e), context=error_context)
            return True
    
    def _check_volume_filter(self, signal):
        """
        Check if the signal passes the volume filter.
        
        Args:
            signal (int): Signal to check
        
        Returns:
            bool: True if the signal passes the filter, False otherwise
        """
        try:
            # Get volume data
            if self.data is None:
                return True
            
            # Check if volume data is available
            if 'volume' not in self.data and not hasattr(self.data, 'volume'):
                return True
            
            # Calculate volume metrics
            if len(self.data) < 20:
                return True
            
            volumes = self.data['volume'].values if isinstance(self.data, pd.DataFrame) else self.data.volume.values
            
            # Calculate average volume
            avg_volume = np.mean(volumes[-20:])
            current_volume = volumes[-1]
            
            # Check if volume is sufficient
            volume_ratio = current_volume / avg_volume
            
            # Volume should be above average for valid signals
            return volume_ratio > 0.8
        
        except Exception as e:
            error_context = {"signal": signal}
            self.error_handler.handle_error("_check_volume_filter", str(e), context=error_context)
            return True
    
    def calculate_position_size(self, account_balance, current_price):
        """
        Calculate position size based on risk level.
        
        Args:
            account_balance (float): Account balance
            current_price (float): Current price
        
        Returns:
            float: Position size in base currency
        """
        try:
            # Calculate risk amount
            risk_amount = account_balance * self.config["risk_level"]
            
            # Calculate stop loss distance (1% of current price by default)
            stop_loss_distance = current_price * 0.01
            
            # Calculate position size
            position_size = risk_amount / stop_loss_distance
            
            # Adjust for volatility if enabled
            if self.config["use_volatility_filters"] and self.data is not None and len(self.data) >= 20:
                close_prices = self.data['close'].values if isinstance(self.data, pd.DataFrame) else self.data.close.values
                returns = np.diff(close_prices) / close_prices[:-1]
                volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
                
                # Adjust position size based on volatility
                volatility_adjustment = self.config["adaptive_parameters"]["volatility_adjustment"]
                position_size *= (1.0 / (volatility * 10 * volatility_adjustment))
            
            return position_size
        
        except Exception as e:
            error_context = {"account_balance": account_balance, "current_price": current_price}
            self.error_handler.handle_error("calculate_position_size", str(e), context=error_context)
            return 0.0
    
    def calculate_take_profit(self, entry_price, position_type):
        """
        Calculate take profit level.
        
        Args:
            entry_price (float): Entry price
            position_type (str): Position type ('long' or 'short')
        
        Returns:
            float: Take profit price
        """
        try:
            # Calculate take profit distance
            take_profit_distance = entry_price * 0.01 * self.config["take_profit_multiplier"]
            
            # Calculate take profit level
            if position_type == 'long':
                take_profit = entry_price + take_profit_distance
            else:
                take_profit = entry_price - take_profit_distance
            
            return take_profit
        
        except Exception as e:
            error_context = {"entry_price": entry_price, "position_type": position_type}
            self.error_handler.handle_error("calculate_take_profit", str(e), context=error_context)
            return entry_price
    
    def calculate_stop_loss(self, entry_price, position_type):
        """
        Calculate stop loss level.
        
        Args:
            entry_price (float): Entry price
            position_type (str): Position type ('long' or 'short')
        
        Returns:
            float: Stop loss price
        """
        try:
            # Calculate stop loss distance
            stop_loss_distance = entry_price * 0.01 * self.config["stop_loss_multiplier"]
            
            # Calculate stop loss level
            if position_type == 'long':
                stop_loss = entry_price - stop_loss_distance
            else:
                stop_loss = entry_price + stop_loss_distance
            
            return stop_loss
        
        except Exception as e:
            error_context = {"entry_price": entry_price, "position_type": position_type}
            self.error_handler.handle_error("calculate_stop_loss", str(e), context=error_context)
            return entry_price
    
    def update_config(self, config):
        """
        Update configuration parameters.
        
        Args:
            config (dict): New configuration parameters
        """
        self.config.update(config)
        logger.info(f"Updated master strategy config: {config}")
    
    def get_strategy_info(self):
        """
        Get strategy information.
        
        Returns:
            dict: Strategy information
        """
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "config": self.config,
            "data_length": len(self.data) if self.data is not None else 0,
            "timestamp": datetime.now().isoformat()
        }
