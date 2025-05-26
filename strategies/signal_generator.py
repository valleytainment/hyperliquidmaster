"""
Signal Generator for Hyperliquid Trading Bot

This module provides a robust signal generator that combines multiple
technical indicators and market data to generate trading signals.

Classes:
    SignalGenerator: Generates trading signals based on technical analysis
"""

import logging
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/signal_generator.log")
    ]
)
logger = logging.getLogger(__name__)

class SignalGenerator:
    """
    Robust signal generator for trading signals.
    
    This class combines multiple technical indicators and market data
    to generate trading signals with adaptive parameters based on
    market conditions.
    
    Attributes:
        technical_indicators: Technical indicator calculator
        error_handler: Error handler for exception management
        config: Configuration parameters for signal generation
    """
    
    def __init__(self, technical_indicators, error_handler, config=None):
        """
        Initialize the signal generator.
        
        Args:
            technical_indicators: Technical indicator calculator
            error_handler: Error handler for exception management
            config (dict, optional): Configuration parameters
        """
        self.technical_indicators = technical_indicators
        self.error_handler = error_handler
        
        # Default configuration
        self.config = {
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "macd_threshold": 0.0,
            "bb_threshold": 0.8,
            "vwma_fast_period": 20,
            "vwma_slow_period": 50,
            "signal_threshold": 0.5,
            "use_adaptive_parameters": True
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        logger.info("Signal generator initialized")
    
    def generate_signal(self, data, timeframe="1h"):
        """
        Generate a trading signal based on technical analysis.
        
        Args:
            data: Market data (DataFrame or dict)
            timeframe (str): Timeframe for analysis
        
        Returns:
            int: Signal (-1 for sell, 0 for neutral, 1 for buy)
        """
        try:
            # Convert dict to DataFrame if needed
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            
            # Check if data is empty
            if data.empty:
                logger.warning("No data for signal generation")
                return 0
            
            # Calculate buy and sell scores
            buy_score = self._calculate_buy_score(data, timeframe)
            sell_score = self._calculate_sell_score(data, timeframe)
            
            # Generate signal based on scores
            signal = self._generate_master_signal(buy_score, sell_score)
            
            logger.info(f"Generated master signal: {signal} (buy_score: {buy_score:.2f}, sell_score: {sell_score:.2f})")
            
            return signal
        
        except Exception as e:
            error_context = {"timeframe": timeframe, "data_shape": data.shape if hasattr(data, "shape") else "dict"}
            self.error_handler.handle_error("generate_signal", str(e), context=error_context)
            return 0
    
    def _calculate_buy_score(self, data, timeframe):
        """
        Calculate the buy score based on technical indicators.
        
        Args:
            data: Market data
            timeframe: Timeframe for analysis
        
        Returns:
            float: Buy score between 0 and 1
        """
        try:
            scores = []
            
            # RSI oversold condition
            rsi_score = self._check_rsi_signal(data, is_buy=True)
            scores.append(rsi_score)
            
            # MACD bullish crossover
            macd_score = self._check_macd_signal(data, is_buy=True)
            scores.append(macd_score)
            
            # Bollinger Bands lower band touch
            bb_score = self._check_bb_signal(data, is_buy=True)
            scores.append(bb_score)
            
            # VWMA bullish crossover
            vwma_score = self._check_vwma_signal(data, is_buy=True)
            scores.append(vwma_score)
            
            # Calculate weighted average
            weights = [0.3, 0.2, 0.2, 0.3]  # Adjust weights based on importance
            buy_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            
            return buy_score
        
        except Exception as e:
            error_context = {"timeframe": timeframe, "data_shape": data.shape if hasattr(data, "shape") else "dict"}
            self.error_handler.handle_error("_calculate_buy_score", str(e), context=error_context)
            return 0.0
    
    def _calculate_sell_score(self, data, timeframe):
        """
        Calculate the sell score based on technical indicators.
        
        Args:
            data: Market data
            timeframe: Timeframe for analysis
        
        Returns:
            float: Sell score between 0 and 1
        """
        try:
            scores = []
            
            # RSI overbought condition
            rsi_score = self._check_rsi_signal(data, is_buy=False)
            scores.append(rsi_score)
            
            # MACD bearish crossover
            macd_score = self._check_macd_signal(data, is_buy=False)
            scores.append(macd_score)
            
            # Bollinger Bands upper band touch
            bb_score = self._check_bb_signal(data, is_buy=False)
            scores.append(bb_score)
            
            # VWMA bearish crossover
            vwma_score = self._check_vwma_signal(data, is_buy=False)
            scores.append(vwma_score)
            
            # Calculate weighted average
            weights = [0.3, 0.2, 0.2, 0.3]  # Adjust weights based on importance
            sell_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            
            return sell_score
        
        except Exception as e:
            error_context = {"timeframe": timeframe, "data_shape": data.shape if hasattr(data, "shape") else "dict"}
            self.error_handler.handle_error("_calculate_sell_score", str(e), context=error_context)
            return 0.0
    
    def _check_rsi_signal(self, data, is_buy=True):
        """
        Check RSI signal.
        
        Args:
            data: Market data
            is_buy (bool): Whether to check for buy signal
        
        Returns:
            float: Signal strength between 0 and 1
        """
        try:
            # Get RSI value
            rsi_values = None
            
            # Handle different data types
            if hasattr(data, 'rsi'):
                # If RSI is already in the data
                if isinstance(data.rsi, (int, float)):
                    rsi_values = [data.rsi]
                elif isinstance(data.rsi, pd.Series):
                    rsi_values = data.rsi.values
                else:
                    rsi_values = [data.rsi]
            else:
                # Calculate RSI if not available
                try:
                    if hasattr(data, 'close') and isinstance(data.close, pd.Series):
                        rsi_values = self.technical_indicators.calculate_rsi(data.close.values)
                    elif 'close' in data:
                        if isinstance(data['close'], (list, np.ndarray)):
                            rsi_values = self.technical_indicators.calculate_rsi(data['close'])
                        else:
                            rsi_values = [50]  # Default if we can't calculate
                    else:
                        rsi_values = [50]  # Default if no close data
                except:
                    rsi_values = [50]  # Default if calculation fails
            
            # Ensure we have a value
            if not rsi_values or len(rsi_values) == 0:
                return 0.0
            
            rsi = rsi_values[-1]  # Use the most recent value
            
            # Adaptive thresholds based on market volatility
            rsi_overbought = self.config["rsi_overbought"]
            rsi_oversold = self.config["rsi_oversold"]
            
            if is_buy:
                # Buy signal: RSI below oversold threshold
                if rsi <= rsi_oversold:
                    # Stronger signal the lower it goes
                    strength = min(1.0, (rsi_oversold - rsi) / 10.0 + 0.5)
                    return strength
                return 0.0
            else:
                # Sell signal: RSI above overbought threshold
                if rsi >= rsi_overbought:
                    # Stronger signal the higher it goes
                    strength = min(1.0, (rsi - rsi_overbought) / 10.0 + 0.5)
                    return strength
                return 0.0
        
        except Exception as e:
            error_context = {"is_buy": is_buy, "data_type": type(data).__name__}
            self.error_handler.handle_error("_check_rsi_signal", str(e), context=error_context)
            return 0.0
    
    def _check_macd_signal(self, data, is_buy=True):
        """
        Check MACD signal.
        
        Args:
            data: Market data
            is_buy (bool): Whether to check for buy signal
        
        Returns:
            float: Signal strength between 0 and 1
        """
        try:
            # Get MACD values
            macd_line = None
            signal_line = None
            
            # Handle different data types
            if hasattr(data, 'macd_line') and hasattr(data, 'signal_line'):
                # If MACD is already in the data
                if isinstance(data.macd_line, (int, float)):
                    macd_line = [data.macd_line]
                    signal_line = [data.signal_line]
                elif isinstance(data.macd_line, pd.Series):
                    macd_line = data.macd_line.values
                    signal_line = data.signal_line.values
                else:
                    macd_line = [data.macd_line]
                    signal_line = [data.signal_line]
            else:
                # Calculate MACD if not available
                try:
                    if hasattr(data, 'close') and isinstance(data.close, pd.Series):
                        macd_result = self.technical_indicators.calculate_macd(data.close.values)
                        macd_line = macd_result['macd']
                        signal_line = macd_result['signal']
                    elif 'close' in data:
                        if isinstance(data['close'], (list, np.ndarray)):
                            macd_result = self.technical_indicators.calculate_macd(data['close'])
                            macd_line = macd_result['macd']
                            signal_line = macd_result['signal']
                        else:
                            macd_line = [0]
                            signal_line = [0]
                    else:
                        macd_line = [0]
                        signal_line = [0]
                except:
                    macd_line = [0]
                    signal_line = [0]
            
            # Ensure we have values
            if not macd_line or not signal_line or len(macd_line) == 0 or len(signal_line) == 0:
                return 0.0
            
            # Get the most recent values
            macd = macd_line[-1]
            signal = signal_line[-1]
            
            # Calculate the difference
            diff = macd - signal
            
            # Threshold for signal strength
            threshold = self.config["macd_threshold"]
            
            if is_buy:
                # Buy signal: MACD above signal line
                if diff > threshold:
                    # Stronger signal the larger the difference
                    strength = min(1.0, diff / 0.5)
                    return strength
                return 0.0
            else:
                # Sell signal: MACD below signal line
                if diff < -threshold:
                    # Stronger signal the larger the negative difference
                    strength = min(1.0, -diff / 0.5)
                    return strength
                return 0.0
        
        except Exception as e:
            error_context = {"is_buy": is_buy, "data_type": type(data).__name__}
            self.error_handler.handle_error("_check_macd_signal", str(e), context=error_context)
            return 0.0
    
    def _check_bb_signal(self, data, is_buy=True):
        """
        Check Bollinger Bands signal.
        
        Args:
            data: Market data
            is_buy (bool): Whether to check for buy signal
        
        Returns:
            float: Signal strength between 0 and 1
        """
        try:
            # Get Bollinger Bands values
            upper_band = None
            lower_band = None
            middle_band = None
            close = None
            
            # Handle different data types
            if hasattr(data, 'upper_band') and hasattr(data, 'lower_band') and hasattr(data, 'middle_band'):
                # If Bollinger Bands are already in the data
                if isinstance(data.upper_band, (int, float)):
                    upper_band = [data.upper_band]
                    lower_band = [data.lower_band]
                    middle_band = [data.middle_band]
                elif isinstance(data.upper_band, pd.Series):
                    upper_band = data.upper_band.values
                    lower_band = data.lower_band.values
                    middle_band = data.middle_band.values
                else:
                    upper_band = [data.upper_band]
                    lower_band = [data.lower_band]
                    middle_band = [data.middle_band]
                
                if hasattr(data, 'close'):
                    if isinstance(data.close, (int, float)):
                        close = [data.close]
                    elif isinstance(data.close, pd.Series):
                        close = data.close.values
                    else:
                        close = [data.close]
                else:
                    close = middle_band  # Use middle band as fallback
            else:
                # Calculate Bollinger Bands if not available
                try:
                    if hasattr(data, 'close') and isinstance(data.close, pd.Series):
                        close = data.close.values
                        bb_result = self.technical_indicators.calculate_bollinger_bands(close)
                        upper_band = bb_result['upper']
                        lower_band = bb_result['lower']
                        middle_band = bb_result['middle']
                    elif 'close' in data:
                        if isinstance(data['close'], (list, np.ndarray)):
                            close = data['close']
                            bb_result = self.technical_indicators.calculate_bollinger_bands(close)
                            upper_band = bb_result['upper']
                            lower_band = bb_result['lower']
                            middle_band = bb_result['middle']
                        else:
                            close = [data['close']]
                            upper_band = [data['close'] * 1.02]
                            lower_band = [data['close'] * 0.98]
                            middle_band = [data['close']]
                    else:
                        close = [0]
                        upper_band = [0]
                        lower_band = [0]
                        middle_band = [0]
                except:
                    close = [0]
                    upper_band = [0]
                    lower_band = [0]
                    middle_band = [0]
            
            # Ensure we have values
            if (not upper_band or not lower_band or not middle_band or not close or
                len(upper_band) == 0 or len(lower_band) == 0 or len(middle_band) == 0 or len(close) == 0):
                return 0.0
            
            # Get the most recent values
            upper = upper_band[-1]
            lower = lower_band[-1]
            middle = middle_band[-1]
            price = close[-1]
            
            # Calculate the position within the bands
            band_width = upper - lower
            if band_width == 0:
                return 0.0
            
            position = (price - lower) / band_width
            
            # Threshold for signal strength
            threshold = self.config["bb_threshold"]
            
            if is_buy:
                # Buy signal: Price near lower band
                if position < threshold:
                    # Stronger signal the closer to the lower band
                    strength = min(1.0, (threshold - position) / threshold)
                    return strength
                return 0.0
            else:
                # Sell signal: Price near upper band
                if position > (1 - threshold):
                    # Stronger signal the closer to the upper band
                    strength = min(1.0, (position - (1 - threshold)) / threshold)
                    return strength
                return 0.0
        
        except Exception as e:
            error_context = {"is_buy": is_buy, "data_type": type(data).__name__}
            self.error_handler.handle_error("_check_bb_signal", str(e), context=error_context)
            return 0.0
    
    def _check_vwma_signal(self, data, is_buy=True):
        """
        Check VWMA signal.
        
        Args:
            data: Market data
            is_buy (bool): Whether to check for buy signal
        
        Returns:
            float: Signal strength between 0 and 1
        """
        try:
            # Get VWMA values
            vwma_fast = None
            vwma_slow = None
            
            # Handle different data types
            if hasattr(data, 'vwma_fast') and hasattr(data, 'vwma_slow'):
                # If VWMA is already in the data
                if isinstance(data.vwma_fast, (int, float)):
                    vwma_fast = [data.vwma_fast]
                    vwma_slow = [data.vwma_slow]
                elif isinstance(data.vwma_fast, pd.Series):
                    vwma_fast = data.vwma_fast.values
                    vwma_slow = data.vwma_slow.values
                else:
                    vwma_fast = [data.vwma_fast]
                    vwma_slow = [data.vwma_slow]
            else:
                # Calculate VWMA if not available
                try:
                    if (hasattr(data, 'close') and hasattr(data, 'volume') and 
                        isinstance(data.close, pd.Series) and isinstance(data.volume, pd.Series)):
                        close = data.close.values
                        volume = data.volume.values
                        vwma_fast = self.technical_indicators.calculate_vwma(close, volume, self.config["vwma_fast_period"])
                        vwma_slow = self.technical_indicators.calculate_vwma(close, volume, self.config["vwma_slow_period"])
                    elif 'close' in data and 'volume' in data:
                        if isinstance(data['close'], (list, np.ndarray)) and isinstance(data['volume'], (list, np.ndarray)):
                            close = data['close']
                            volume = data['volume']
                            vwma_fast = self.technical_indicators.calculate_vwma(close, volume, self.config["vwma_fast_period"])
                            vwma_slow = self.technical_indicators.calculate_vwma(close, volume, self.config["vwma_slow_period"])
                        else:
                            vwma_fast = [data['close']]
                            vwma_slow = [data['close']]
                    else:
                        vwma_fast = [0]
                        vwma_slow = [0]
                except:
                    vwma_fast = [0]
                    vwma_slow = [0]
            
            # Ensure we have values
            if not vwma_fast or not vwma_slow or len(vwma_fast) == 0 or len(vwma_slow) == 0:
                return 0.0
            
            # Get the most recent values
            fast = vwma_fast[-1]
            slow = vwma_slow[-1]
            
            # Calculate the difference
            if slow == 0:
                return 0.0
            
            diff_percent = (fast - slow) / slow * 100
            
            if is_buy:
                # Buy signal: Fast VWMA above slow VWMA
                if diff_percent > 0:
                    # Stronger signal the larger the difference
                    strength = min(1.0, diff_percent / 2.0)
                    return strength
                return 0.0
            else:
                # Sell signal: Fast VWMA below slow VWMA
                if diff_percent < 0:
                    # Stronger signal the larger the negative difference
                    strength = min(1.0, -diff_percent / 2.0)
                    return strength
                return 0.0
        
        except Exception as e:
            error_context = {"is_buy": is_buy, "data_type": type(data).__name__}
            self.error_handler.handle_error("_check_vwma_signal", str(e), context=error_context)
            return 0.0
    
    def _generate_master_signal(self, buy_score, sell_score):
        """
        Generate master signal based on buy and sell scores.
        
        Args:
            buy_score (float): Buy score between 0 and 1
            sell_score (float): Sell score between 0 and 1
        
        Returns:
            int: Signal (-1 for sell, 0 for neutral, 1 for buy)
        """
        # Threshold for signal generation
        threshold = self.config["signal_threshold"]
        
        # Generate signal
        if buy_score > threshold and buy_score > sell_score:
            return 1  # Buy signal
        elif sell_score > threshold and sell_score > buy_score:
            return -1  # Sell signal
        else:
            return 0  # Neutral signal
    
    def update_config(self, config):
        """
        Update configuration parameters.
        
        Args:
            config (dict): New configuration parameters
        """
        self.config.update(config)
        logger.info(f"Updated signal generator config: {config}")
    
    def detect_market_regime(self, data):
        """
        Detect market regime (trending, ranging, volatile).
        
        Args:
            data: Market data
        
        Returns:
            str: Market regime ('trending', 'ranging', 'volatile')
        """
        try:
            # Calculate volatility
            if hasattr(data, 'close') and isinstance(data.close, pd.Series):
                close = data.close.values
            elif 'close' in data and isinstance(data['close'], (list, np.ndarray)):
                close = data['close']
            else:
                return 'unknown'
            
            # Calculate returns
            returns = np.diff(close) / close[:-1]
            
            # Calculate volatility (standard deviation of returns)
            volatility = np.std(returns)
            
            # Calculate trend strength
            if len(close) >= 20:
                # Use linear regression slope
                x = np.arange(len(close))
                slope, _, _, _, _ = np.polyfit(x, close, 1, full=True)
                trend_strength = abs(slope[0]) / np.mean(close)
            else:
                trend_strength = 0
            
            # Determine market regime
            high_volatility_threshold = 0.02
            high_trend_threshold = 0.001
            
            if volatility > high_volatility_threshold:
                regime = 'volatile'
            elif trend_strength > high_trend_threshold:
                regime = 'trending'
            else:
                regime = 'ranging'
            
            logger.info(f"Market regime detected: {regime}")
            
            return regime
        
        except Exception as e:
            error_context = {"data_type": type(data).__name__}
            self.error_handler.handle_error("detect_market_regime", str(e), context=error_context)
            return 'unknown'
    
    def adjust_parameters_for_regime(self, regime):
        """
        Adjust parameters based on market regime.
        
        Args:
            regime (str): Market regime ('trending', 'ranging', 'volatile')
        
        Returns:
            dict: Adjusted parameters
        """
        if not self.config["use_adaptive_parameters"]:
            return self.config
        
        # Create a copy of the current config
        adjusted_config = self.config.copy()
        
        if regime == 'trending':
            # In trending markets, favor trend-following indicators
            adjusted_config["signal_threshold"] = 0.4
            adjusted_config["vwma_fast_period"] = 10
            adjusted_config["vwma_slow_period"] = 30
        elif regime == 'ranging':
            # In ranging markets, favor mean-reversion indicators
            adjusted_config["signal_threshold"] = 0.6
            adjusted_config["rsi_overbought"] = 65
            adjusted_config["rsi_oversold"] = 35
        elif regime == 'volatile':
            # In volatile markets, require stronger signals
            adjusted_config["signal_threshold"] = 0.8
            adjusted_config["rsi_overbought"] = 75
            adjusted_config["rsi_oversold"] = 25
        
        logger.info(f"Adjusted parameters for {regime} market regime")
        
        return adjusted_config
