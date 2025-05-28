#!/usr/bin/env python3
"""
Headless Testing Script for Hyperliquid Trading Bot

This script tests the core functionality of the trading bot components without requiring
a display, validating the integration with strategy components and data pipelines.
"""

import os
import sys
import json
import logging
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("headless_test.log")
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import strategy components
from strategies.robust_signal_generator_fixed_updated import RobustSignalGenerator
from strategies.master_omni_overlord_robust_standardized import MasterOmniOverlordRobustStrategy
from strategies.advanced_technical_indicators_fixed import AdvancedTechnicalIndicators
from core.error_handling import ErrorHandler
from core.api_rate_limiter import APIRateLimiter
from core.enhanced_mock_data_provider import EnhancedMockDataProvider

def main():
    """
    Main entry point for headless testing.
    """
    logger.info("Starting headless tests for Hyperliquid Trading Bot")
    
    # Remove any command line arguments that would interfere with unittest
    # This is crucial to prevent argument parsing errors
    test_args = [sys.argv[0]]
    sys.argv = test_args
    
    # Run tests
    unittest.main(module=__name__, exit=False)
    
    logger.info("Headless tests completed")

class HeadlessTest(unittest.TestCase):
    """
    Test case for headless testing.
    """
    
    def setUp(self):
        """
        Set up test case.
        """
        logger.info("Setting up test case")
        
        # Create components
        self.error_handler = ErrorHandler()
        self.technical_indicators = AdvancedTechnicalIndicators()
        self.api_rate_limiter = APIRateLimiter()
        self.mock_data_provider = EnhancedMockDataProvider()
        
        # Create signal generator
        self.signal_generator = RobustSignalGenerator(
            technical_indicators=self.technical_indicators,
            error_handler=self.error_handler
        )
        
        # Create strategy configuration
        self.config = {
            "risk_level": 0.02,
            "take_profit_multiplier": 3.0,
            "stop_loss_multiplier": 2.0,
            "use_volatility_filters": True,
            "use_trend_filters": True,
            "use_volume_filters": True,
            "use_regime_detection": True
        }
        
        # Create strategy instance
        self.strategy = MasterOmniOverlordRobustStrategy(
            symbol="XRP",
            timeframe="1h",
            signal_generator=self.signal_generator,
            error_handler=self.error_handler,
            config=self.config
        )
        
        # Create data directory
        os.makedirs("enhanced_data/1h", exist_ok=True)
        
        # Create mock data
        self._create_mock_data()
    
    def _create_mock_data(self):
        """
        Create mock data for testing.
        """
        logger.info("Creating mock data")
        
        # Create mock data
        data = []
        
        # Generate 100 candles
        start_time = datetime.now() - timedelta(days=5)
        
        for i in range(100):
            timestamp = int((start_time + timedelta(hours=i)).timestamp())
            
            # Generate random price data
            open_price = 0.5 + np.random.random() * 0.1
            high_price = open_price + np.random.random() * 0.02
            low_price = open_price - np.random.random() * 0.02
            close_price = open_price + (np.random.random() - 0.5) * 0.03
            volume = np.random.random() * 1000000
            
            # Add candle
            data.append({
                "timestamp": timestamp,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume
            })
        
        # Save data
        with open("enhanced_data/1h/XRP.json", "w") as f:
            json.dump(data, f)
        
        logger.info("Mock data created")
    
    def test_data_loading(self):
        """
        Test data loading functionality.
        """
        logger.info("Testing data loading")
        
        # Load data
        file_path = "enhanced_data/1h/XRP.json"
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.set_index("datetime")
        
        # Verify data
        self.assertGreater(len(df), 0)
        self.assertIn("close", df.columns)
        self.assertIn("volume", df.columns)
        
        logger.info(f"Loaded {len(df)} candles")
        
        # Set data on strategy
        self.strategy.set_data(df)
        
        logger.info("Data loading test passed")
    
    def test_signal_generation(self):
        """
        Test signal generation functionality.
        """
        logger.info("Testing signal generation")
        
        # Load data
        file_path = "enhanced_data/1h/XRP.json"
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.set_index("datetime")
        
        # Set data on strategy
        self.strategy.set_data(df)
        
        # Generate signals
        signals = []
        
        for i in range(50, len(df)):
            # Get current candle data
            candle_data = df.iloc[i].to_dict()
            
            # Generate signal
            signal = self.strategy.generate_signal(candle_data)
            
            # Store signal
            signals.append(signal)
        
        # Verify signals
        self.assertGreater(len(signals), 0)
        
        # Count signal types
        buy_signals = sum(1 for s in signals if s > 0)
        sell_signals = sum(1 for s in signals if s < 0)
        neutral_signals = sum(1 for s in signals if s == 0)
        
        logger.info(f"Generated {len(signals)} signals")
        logger.info(f"Buy signals: {buy_signals}")
        logger.info(f"Sell signals: {sell_signals}")
        logger.info(f"Neutral signals: {neutral_signals}")
        
        logger.info("Signal generation test passed")
    
    def test_mock_data_mode(self):
        """
        Test mock data mode functionality.
        """
        logger.info("Testing mock data mode")
        
        # Get mock data
        data = self.mock_data_provider.get_klines("XRP", "1h")
        
        # Verify data
        self.assertGreater(len(data), 0)
        self.assertIn("timestamp", data[0])
        self.assertIn("close", data[0])
        self.assertIn("volume", data[0])
        
        logger.info(f"Got {len(data)} candles from mock data provider")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.set_index("datetime")
        
        # Set data on strategy
        self.strategy.set_data(df)
        
        # Generate signal
        candle_data = df.iloc[-1].to_dict()
        signal = self.strategy.generate_signal(candle_data)
        
        # Verify signal
        self.assertIn(signal, [-1, 0, 1])
        
        logger.info(f"Generated signal: {signal}")
        logger.info("Mock data mode test passed")
    
    def test_api_rate_limiter(self):
        """
        Test API rate limiter functionality.
        """
        logger.info("Testing API rate limiter")
        
        # Check initial status
        status = self.api_rate_limiter.get_status()
        
        self.assertIn("is_limited", status)
        self.assertIn("cooldown_remaining", status)
        
        logger.info(f"Initial status: {status}")
        
        # Record calls
        for _ in range(10):
            self.api_rate_limiter.record_call("klines")
        
        # Check status after calls
        status = self.api_rate_limiter.get_status()
        
        logger.info(f"Status after calls: {status}")
        
        # Reset rate limiter
        self.api_rate_limiter.reset()
        
        # Check status after reset
        status = self.api_rate_limiter.get_status()
        
        logger.info(f"Status after reset: {status}")
        
        logger.info("API rate limiter test passed")
    
    def test_error_handling(self):
        """
        Test error handling functionality.
        """
        logger.info("Testing error handling")
        
        # Create test error
        try:
            # Raise test exception
            raise ValueError("Test error")
        except Exception as e:
            # Handle error
            self.error_handler.handle_error("test_function", str(e))
        
        logger.info("Error handling test passed")
    
    def test_integration(self):
        """
        Test integration of all components.
        """
        logger.info("Testing integration of all components")
        
        # Load data
        file_path = "enhanced_data/1h/XRP.json"
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.set_index("datetime")
        
        # Set data on strategy
        self.strategy.set_data(df)
        
        # Generate signals
        signals = []
        
        for i in range(50, len(df)):
            # Get current candle data
            candle_data = df.iloc[i].to_dict()
            
            # Generate signal
            signal = self.strategy.generate_signal(candle_data)
            
            # Store signal
            signals.append(signal)
        
        # Simulate trades
        position = 0
        entry_price = 0
        entry_time = None
        trades = []
        
        for i, signal in enumerate(signals):
            current_price = df["close"].iloc[i + 50]
            current_time = df.index[i + 50]
            
            # Process signal
            if position == 0 and signal > 0:
                # Enter long position
                position = 1
                entry_price = current_price
                entry_time = current_time
            
            elif position == 0 and signal < 0:
                # Enter short position
                position = -1
                entry_price = current_price
                entry_time = current_time
            
            elif position > 0 and signal < 0:
                # Exit long position
                pnl = (current_price - entry_price) / entry_price
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": current_time,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "position": "long",
                    "pnl": pnl
                })
                position = 0
            
            elif position < 0 and signal > 0:
                # Exit short position
                pnl = (entry_price - current_price) / entry_price
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": current_time,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "position": "short",
                    "pnl": pnl
                })
                position = 0
        
        # Close any open position at the end
        if position != 0:
            current_price = df["close"].iloc[-1]
            current_time = df.index[-1]
            
            if position > 0:
                # Exit long position
                pnl = (current_price - entry_price) / entry_price
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": current_time,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "position": "long",
                    "pnl": pnl
                })
            
            elif position < 0:
                # Exit short position
                pnl = (entry_price - current_price) / entry_price
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": current_time,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "position": "short",
                    "pnl": pnl
                })
        
        # Calculate performance metrics
        if trades:
            # Calculate win rate
            winning_trades = sum(1 for trade in trades if trade["pnl"] > 0)
            win_rate = winning_trades / len(trades)
            
            # Calculate profit factor
            gross_profit = sum(trade["pnl"] for trade in trades if trade["pnl"] > 0)
            gross_loss = abs(sum(trade["pnl"] for trade in trades if trade["pnl"] < 0))
            
            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss
            else:
                profit_factor = float('inf') if gross_profit > 0 else 0.0
            
            # Calculate total return
            total_return = sum(trade["pnl"] for trade in trades)
            
            logger.info(f"Total trades: {len(trades)}")
            logger.info(f"Win rate: {win_rate:.2%}")
            logger.info(f"Profit factor: {profit_factor:.2f}")
            logger.info(f"Total return: {total_return:.2%}")
        
        logger.info("Integration test passed")

if __name__ == "__main__":
    main()
