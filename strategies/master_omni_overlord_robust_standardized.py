"""
Master Omni Overlord Trading Strategy with Robust Signal Generation - Standardized

This module implements the ultimate trading strategy that combines multiple advanced techniques
and adapts to market conditions for maximum performance on Hyperliquid, with robust handling
of limited data and edge cases. Constructor signature has been standardized for seamless integration.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

# Import robust signal generator
from strategies.robust_signal_generator_fixed_updated import RobustSignalGenerator

class MasterOmniOverlordRobustStrategy:
    """
    Master Omni Overlord Robust Strategy - The ultimate trading strategy that combines
    multiple advanced techniques and adapts to market conditions with robust handling
    of limited data and edge cases.
    """
    
    def __init__(self, 
                 symbol: str = "BTC", 
                 timeframe: str = "1h", 
                 signal_generator = None, 
                 error_handler = None, 
                 config: Dict = None,
                 data_accumulator = None, 
                 exchange_adapter = None,
                 logger = None):
        """
        Initialize the Master Omni Overlord Robust Strategy with standardized constructor.
        
        Args:
            symbol: Trading symbol (e.g., "BTC", "ETH", "XRP")
            timeframe: Trading timeframe (e.g., "1m", "5m", "1h", "1d")
            signal_generator: Signal generator instance
            error_handler: Error handler instance
            config: Configuration dictionary with strategy parameters
            data_accumulator: Data accumulator instance
            exchange_adapter: Exchange adapter instance
            logger: Optional logger instance
        """
        # Setup logging
        self.logger = logger or self._setup_logger()
        self.logger.info(f"Initializing Master Omni Overlord Robust Strategy for {symbol} on {timeframe} timeframe...")
        
        # Store basic parameters
        self.symbol = symbol
        self.timeframe = timeframe
        self.signal_generator = signal_generator
        self.error_handler = error_handler
        self.data_accumulator = data_accumulator
        self.exchange_adapter = exchange_adapter
        
        # Store configuration
        self.config = config or {}
        
        # Extract strategy parameters from config
        self.risk_level = self.config.get("risk_level", 0.05)
        self.max_position_size = self.config.get("max_position_size", 1.0)
        self.take_profit_multiplier = self.config.get("take_profit_multiplier", 3.0)
        self.stop_loss_multiplier = self.config.get("stop_loss_multiplier", 2.0)
        self.trailing_stop_activation = self.config.get("trailing_stop_activation", 0.02)
        self.trailing_stop_distance = self.config.get("trailing_stop_distance", 0.01)
        
        # Extract feature flags from config
        self.use_dynamic_sizing = self.config.get("use_dynamic_sizing", True)
        self.use_adaptive_exits = self.config.get("use_adaptive_exits", True)
        self.use_volatility_filters = self.config.get("use_volatility_filters", True)
        self.use_trend_filters = self.config.get("use_trend_filters", True)
        self.use_volume_filters = self.config.get("use_volume_filters", True)
        self.use_sentiment_data = self.config.get("use_sentiment_data", False)
        self.use_order_book_data = self.config.get("use_order_book_data", True)
        self.use_funding_rate_data = self.config.get("use_funding_rate_data", True)
        self.use_multi_timeframe_confirmation = self.config.get("use_multi_timeframe_confirmation", True)
        self.use_regime_detection = self.config.get("use_regime_detection", True)
        self.use_position_scaling = self.config.get("use_position_scaling", True)
        self.use_dynamic_indicators = self.config.get("use_dynamic_indicators", True)
        self.use_ml_confirmation = self.config.get("use_ml_confirmation", False)
        self.use_adaptive_parameters = self.config.get("use_adaptive_parameters", True)
        
        # Strategy weights - will be dynamically adjusted
        self.strategy_weights = {
            "triple_confluence": 0.6,
            "oracle_update": 0.4
        }
        
        # Adaptive parameters - will be dynamically adjusted
        self.adaptive_params = {
            "signal_threshold": 0.7,
            "trend_filter_strength": 0.5,
            "mean_reversion_factor": 0.3,
            "volatility_adjustment": 1.0
        }
        
        # Initialize data storage
        self.data = None
        self.additional_data = {}
        
        self.logger.info("Master Omni Overlord Robust Strategy initialized successfully")
        
    def _setup_logger(self):
        """
        Setup logger.
        
        Returns:
            Logger instance
        """
        logger = logging.getLogger("MasterOmniOverlordRobustStrategy")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def set_data(self, data):
        """
        Set data for the strategy.
        
        Args:
            data: DataFrame with price data
        """
        self.data = data
        self.logger.info(f"Data set for {self.symbol} on {self.timeframe} timeframe with {len(data)} candles")
    
    def set_additional_data(self, additional_data):
        """
        Set additional data for the strategy.
        
        Args:
            additional_data: Dictionary with additional data
        """
        self.additional_data = additional_data
        self.logger.info(f"Additional data set for {self.symbol} with {len(additional_data)} data sources")
        
    def detect_market_regime(self, df=None):
        """
        Detect market regime (trending, ranging, volatile) with robust handling of limited data.
        
        Args:
            df: DataFrame with price data (optional, uses self.data if None)
            
        Returns:
            Market regime as string: 'trending_up', 'trending_down', 'ranging', 'volatile', or 'unknown'
        """
        try:
            # Use provided data or self.data
            df = df if df is not None else self.data
            
            # Check if data is available
            if df is None or df.empty:
                self.logger.warning("No data for market regime detection")
                return 'unknown'
            
            # Use signal generator if available
            if self.signal_generator:
                regime = self.signal_generator.detect_market_regime(df)
            else:
                # Default to unknown if no signal generator
                regime = 'unknown'
            
            self.logger.info(f"Market regime detected: {regime}")
            return regime
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            # Use error handler if available
            if self.error_handler:
                self.error_handler.handle_error("detect_market_regime", str(e))
            return 'unknown'
            
    def adjust_strategy_weights(self, regime: str) -> None:
        """
        Adjust strategy weights based on market regime.
        
        Args:
            regime: Market regime
        """
        try:
            if regime == 'trending_up' or regime == 'trending_down':
                # In trending markets, favor Triple Confluence
                self.strategy_weights = {
                    "triple_confluence": 0.7,
                    "oracle_update": 0.3
                }
            elif regime == 'volatile':
                # In volatile markets, favor Oracle Update
                self.strategy_weights = {
                    "triple_confluence": 0.3,
                    "oracle_update": 0.7
                }
            else:
                # In ranging or unknown markets, use balanced weights
                self.strategy_weights = {
                    "triple_confluence": 0.5,
                    "oracle_update": 0.5
                }
                
            self.logger.info(f"Adjusted strategy weights: Triple Confluence={self.strategy_weights['triple_confluence']:.2f}, Oracle Update={self.strategy_weights['oracle_update']:.2f}")
        except Exception as e:
            self.logger.error(f"Error adjusting strategy weights: {str(e)}")
            # Use error handler if available
            if self.error_handler:
                self.error_handler.handle_error("adjust_strategy_weights", str(e))
            
    def adjust_adaptive_parameters(self, regime: str) -> None:
        """
        Adjust adaptive parameters based on market regime.
        
        Args:
            regime: Market regime
        """
        try:
            if regime == 'trending_up' or regime == 'trending_down':
                # In trending markets, lower threshold and increase trend filter
                self.adaptive_params = {
                    "signal_threshold": 0.6,
                    "trend_filter_strength": 0.7,
                    "mean_reversion_factor": 0.2,
                    "volatility_adjustment": 0.8
                }
            elif regime == 'volatile':
                # In volatile markets, increase threshold and volatility adjustment
                self.adaptive_params = {
                    "signal_threshold": 0.8,
                    "trend_filter_strength": 0.4,
                    "mean_reversion_factor": 0.2,
                    "volatility_adjustment": 1.2
                }
            elif regime == 'ranging':
                # In ranging markets, increase mean reversion
                self.adaptive_params = {
                    "signal_threshold": 0.7,
                    "trend_filter_strength": 0.3,
                    "mean_reversion_factor": 0.6,
                    "volatility_adjustment": 1.0
                }
            else:
                # In unknown markets, use balanced parameters
                self.adaptive_params = {
                    "signal_threshold": 0.7,
                    "trend_filter_strength": 0.5,
                    "mean_reversion_factor": 0.3,
                    "volatility_adjustment": 1.0
                }
                
            self.logger.info(f"Adjusted adaptive parameters: {self.adaptive_params}")
        except Exception as e:
            self.logger.error(f"Error adjusting adaptive parameters: {str(e)}")
            # Use error handler if available
            if self.error_handler:
                self.error_handler.handle_error("adjust_adaptive_parameters", str(e))
            
    def calculate_position_size(self, balance, price):
        """
        Calculate position size based on risk parameters.
        
        Args:
            balance: Account balance
            price: Current price
            
        Returns:
            Position size
        """
        try:
            # Default risk parameters
            risk_amount = balance * self.risk_level
            
            # Calculate position size
            position_size = risk_amount / price
            
            # Apply max position size limit
            max_size = balance * self.max_position_size / price
            position_size = min(position_size, max_size)
            
            self.logger.info(f"Calculated position size for {self.symbol}: {position_size:.6f} units (risk: {self.risk_level:.2%}, price: {price:.2f})")
            return position_size
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            # Use error handler if available
            if self.error_handler:
                self.error_handler.handle_error("calculate_position_size", str(e))
            return 0.0
            
    def generate_signal(self, candle):
        """
        Generate trading signal for the current candle.
        
        Args:
            candle: Current candle data
            
        Returns:
            Signal: 1 for buy, -1 for sell, 0 for neutral
        """
        try:
            self.logger.info(f"Generating signal for {self.symbol} using Master Omni Overlord Strategy")
            
            # Check if data is available
            if self.data is None or self.data.empty:
                self.logger.warning("No data for signal generation")
                return 0
            
            # Detect market regime
            regime = self.detect_market_regime()
            
            # Adjust strategy weights and parameters based on regime
            self.adjust_strategy_weights(regime)
            self.adjust_adaptive_parameters(regime)
            
            # Generate master signal using robust signal generator
            if self.signal_generator:
                signal = self.signal_generator.generate_signal(self.data, self.timeframe)
            else:
                # Default to neutral if no signal generator
                signal = 0
            
            self.logger.info(f"Generated signal for {self.symbol}: {signal}")
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            # Use error handler if available
            if self.error_handler:
                self.error_handler.handle_error("generate_signal", str(e))
            return 0
