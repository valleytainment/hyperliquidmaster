"""
Master Omni Overlord Trading Strategy with Robust Signal Generation

This module implements the ultimate trading strategy that combines multiple advanced techniques
and adapts to market conditions for maximum performance on Hyperliquid, with robust handling
of limited data and edge cases.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

# Import robust signal generator
from strategies.robust_signal_generator import RobustSignalGenerator

class MasterOmniOverlordRobustStrategy:
    """
    Master Omni Overlord Robust Strategy - The ultimate trading strategy that combines
    multiple advanced techniques and adapts to market conditions with robust handling
    of limited data and edge cases.
    """
    
    def __init__(self, config: Dict, logger=None, error_handler=None, data_accumulator=None, exchange_adapter=None):
        """
        Initialize the Master Omni Overlord Robust Strategy.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
            error_handler: Optional error handler instance
            data_accumulator: Optional data accumulator instance
            exchange_adapter: Optional exchange adapter instance
        """
        # Setup logging
        self.logger = logger or self._setup_logger()
        self.logger.info("Initializing Master Omni Overlord Robust Strategy...")
        
        # Store configuration and components
        self.config = config
        self.error_handler = error_handler
        self.data_accumulator = data_accumulator
        self.exchange_adapter = exchange_adapter
        
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
        
    def detect_market_regime(self, market_data: Dict) -> str:
        """
        Detect market regime (trending, ranging, volatile) with robust handling of limited data.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Market regime as string: 'trending', 'ranging', 'volatile', or 'unknown'
        """
        try:
            # Check if historical data is available
            if 'historical_data' not in market_data or market_data['historical_data'] is None:
                self.logger.warning("No historical data for market regime detection")
                return 'unknown'
                
            # Get historical data
            df = market_data['historical_data']
            
            # Check if market regime is already calculated
            if 'indicators' in market_data and 'market_regime' in market_data['indicators']:
                regime = market_data['indicators']['market_regime']
                if regime != 'unknown':
                    return regime
                    
            # Use robust market regime detection
            regime = RobustSignalGenerator.detect_market_regime(df)
            
            self.logger.info(f"Market regime detected: {regime}")
            return regime
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            # Use error handler if available
            if self.error_handler:
                self.error_handler.handle_error(
                    message=f"Error detecting market regime: {str(e)}",
                    category="SIGNAL",
                    severity="ERROR"
                )
            return 'unknown'
            
    def adjust_strategy_weights(self, regime: str) -> None:
        """
        Adjust strategy weights based on market regime.
        
        Args:
            regime: Market regime
        """
        try:
            if regime == 'trending':
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
                self.error_handler.handle_error(
                    message=f"Error adjusting strategy weights: {str(e)}",
                    category="SIGNAL",
                    severity="ERROR"
                )
            
    def adjust_adaptive_parameters(self, regime: str) -> None:
        """
        Adjust adaptive parameters based on market regime.
        
        Args:
            regime: Market regime
        """
        try:
            if regime == 'trending':
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
                self.error_handler.handle_error(
                    message=f"Error adjusting adaptive parameters: {str(e)}",
                    category="SIGNAL",
                    severity="ERROR"
                )
            
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            symbol: Symbol to calculate position size for
            entry_price: Entry price
            stop_loss: Stop loss price
            
        Returns:
            Position size
        """
        try:
            # Default risk parameters
            risk_pct = 0.01  # 1% risk per trade
            account_size = 10000.0  # Default account size
            
            # Override with config if available
            if 'risk' in self.config:
                risk_pct = self.config.get('risk', {}).get('risk_per_trade', risk_pct)
                account_size = self.config.get('risk', {}).get('account_size', account_size)
                
            # Calculate risk amount
            risk_amount = account_size * risk_pct
            
            # Calculate stop distance
            if entry_price <= 0 or stop_loss <= 0:
                self.logger.warning(f"Invalid entry price or stop loss for {symbol}")
                return 0.0
                
            stop_distance_pct = abs(entry_price - stop_loss) / entry_price
            
            if stop_distance_pct <= 0:
                self.logger.warning(f"Zero stop distance for {symbol}")
                return 0.0
                
            # Calculate position size
            position_size = risk_amount / (entry_price * stop_distance_pct)
            
            self.logger.info(f"Calculated position size for {symbol}: {position_size:.2f} units (risk: {risk_pct:.2%}, stop distance: {stop_distance_pct:.2%})")
            return position_size
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            # Use error handler if available
            if self.error_handler:
                self.error_handler.handle_error(
                    message=f"Error calculating position size: {str(e)}",
                    category="SIGNAL",
                    severity="ERROR"
                )
            return 0.0
            
    def generate_signal(self, symbol: str, market_data: Dict, order_book: Dict, positions: Dict = None) -> Dict:
        """
        Generate trading signal for a symbol.
        
        Args:
            symbol: Symbol to generate signal for
            market_data: Market data dictionary
            order_book: Order book dictionary
            positions: Current positions dictionary
            
        Returns:
            Signal dictionary
        """
        try:
            self.logger.info(f"Generating signal for {symbol} using Master Omni Overlord Strategy")
            
            # Detect market regime
            regime = self.detect_market_regime(market_data)
            
            # Adjust strategy weights and parameters based on regime
            self.adjust_strategy_weights(regime)
            self.adjust_adaptive_parameters(regime)
            
            # Generate master signal using robust signal generator
            signal = RobustSignalGenerator.generate_master_signal(
                df=market_data.get('historical_data'),
                order_book=order_book
            )
            
            # Extract signal components
            action = signal.get("action", "none")
            signal_strength = signal.get("signal", 0.0)
            entry_price = signal.get("entry_price", market_data.get("last_price", 0.0))
            stop_loss = signal.get("stop_loss", 0.0)
            take_profit = signal.get("take_profit", 0.0)
            
            # Apply adaptive threshold
            signal_threshold = self.adaptive_params["signal_threshold"]
            
            # Determine final trading decision
            if abs(signal_strength) < signal_threshold:
                action = "none"
                
            # Calculate position size
            if action != "none":
                position_size = self.calculate_position_size(symbol, entry_price, stop_loss)
            else:
                position_size = 0.0
                
            # Prepare signal result
            signal_result = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "action": action,
                "quantity": position_size,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "signal_strength": signal_strength,
                "market_regime": regime,
                "strategy_weights": self.strategy_weights.copy(),
                "adaptive_params": self.adaptive_params.copy(),
                "components": signal.get("components", {})
            }
            
            self.logger.info(f"Generated signal for {symbol}: {action} with quantity {position_size}")
            return signal_result
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            # Use error handler if available
            if self.error_handler:
                self.error_handler.handle_error(
                    message=f"Error generating signal: {str(e)}",
                    category="SIGNAL",
                    severity="ERROR"
                )
            return {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "action": "none",
                "quantity": 0.0,
                "entry_price": 0.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "signal_strength": 0.0,
                "error": str(e)
            }
