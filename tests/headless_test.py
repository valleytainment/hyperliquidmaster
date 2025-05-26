"""
Headless GUI Test for Hyperliquid Trading Bot

This module provides a headless testing framework for the Hyperliquid trading bot GUI,
allowing validation of GUI components without requiring a display.

Classes:
    HeadlessGUITest: Headless testing framework for GUI components
"""

import os
import sys
import unittest
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core modules
from core.api_rate_limiter import APIRateLimiter
from core.error_handling import ErrorHandler

# Import data modules
from data.mock_data_provider import MockDataProvider

# Import strategy modules
from strategies.signal_generator import SignalGenerator
from strategies.master_strategy import MasterStrategy

# Import utility modules
from utils.technical_indicators import TechnicalIndicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/headless_test.log")
    ]
)
logger = logging.getLogger(__name__)

class HeadlessGUITest(unittest.TestCase):
    """
    Headless testing framework for GUI components.
    
    This class provides methods for testing GUI components without requiring a display.
    """
    
    def setUp(self):
        """
        Set up the test environment.
        """
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Initialize components
        self.api_rate_limiter = APIRateLimiter()
        self.error_handler = ErrorHandler()
        self.mock_data_provider = MockDataProvider()
        self.technical_indicators = TechnicalIndicators()
        
        # Initialize signal generator and strategy
        self.signal_generator = SignalGenerator(
            self.technical_indicators,
            self.error_handler
        )
        
        self.master_strategy = MasterStrategy(
            "XRP",
            "1h",
            self.signal_generator,
            self.error_handler
        )
        
        logger.info("Test environment set up")
    
    def tearDown(self):
        """
        Clean up after the test.
        """
        # Reset components
        self.api_rate_limiter.reset()
        
        logger.info("Test environment cleaned up")
    
    def test_api_rate_limiter(self):
        """
        Test the API rate limiter.
        """
        logger.info("Testing API rate limiter")
        
        # Test initial state
        self.assertFalse(self.api_rate_limiter.is_mock_mode_active())
        
        # Test rate limiting
        for i in range(100):
            self.api_rate_limiter.record_call("market_data")
        
        # Check if mock mode is active
        self.assertTrue(self.api_rate_limiter.is_mock_mode_active())
        
        # Test status
        status = self.api_rate_limiter.get_status()
        self.assertTrue(status["is_limited"])
        self.assertTrue(status["mock_mode_active"])
        
        logger.info("API rate limiter test passed")
    
    def test_mock_data_provider(self):
        """
        Test the mock data provider.
        """
        logger.info("Testing mock data provider")
        
        # Test klines
        klines = self.mock_data_provider.get_klines("XRP", "1h", limit=100)
        self.assertEqual(len(klines), 100)
        self.assertIn("timestamp", klines[0])
        self.assertIn("open", klines[0])
        self.assertIn("high", klines[0])
        self.assertIn("low", klines[0])
        self.assertIn("close", klines[0])
        self.assertIn("volume", klines[0])
        
        # Test market data
        market_data = self.mock_data_provider.get_market_data("XRP")
        self.assertIn("price", market_data)
        self.assertIn("funding_rate", market_data)
        
        # Test order book
        order_book = self.mock_data_provider.get_order_book("XRP")
        self.assertIn("bids", order_book)
        self.assertIn("asks", order_book)
        
        logger.info("Mock data provider test passed")
    
    def test_technical_indicators(self):
        """
        Test the technical indicators.
        """
        logger.info("Testing technical indicators")
        
        # Generate sample data
        import numpy as np
        prices = np.random.normal(100, 10, 100)
        volumes = np.random.gamma(2.0, 1000.0, 100)
        
        # Test RSI
        rsi = self.technical_indicators.calculate_rsi(prices)
        self.assertEqual(len(rsi), len(prices))
        self.assertTrue(all(0 <= r <= 100 for r in rsi))
        
        # Test MACD
        macd = self.technical_indicators.calculate_macd(prices)
        self.assertEqual(len(macd["macd"]), len(prices))
        self.assertEqual(len(macd["signal"]), len(prices))
        self.assertEqual(len(macd["histogram"]), len(prices))
        
        # Test Bollinger Bands
        bb = self.technical_indicators.calculate_bollinger_bands(prices)
        self.assertEqual(len(bb["upper"]), len(prices))
        self.assertEqual(len(bb["middle"]), len(prices))
        self.assertEqual(len(bb["lower"]), len(prices))
        
        # Test VWMA
        vwma = self.technical_indicators.calculate_vwma(prices, volumes)
        self.assertEqual(len(vwma), len(prices))
        
        logger.info("Technical indicators test passed")
    
    def test_signal_generator(self):
        """
        Test the signal generator.
        """
        logger.info("Testing signal generator")
        
        # Get sample data
        klines = self.mock_data_provider.get_klines("XRP", "1h", limit=100)
        
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame(klines)
        
        # Generate signal
        signal = self.signal_generator.generate_signal(df)
        
        # Check signal
        self.assertIn(signal, [-1, 0, 1])
        
        logger.info("Signal generator test passed")
    
    def test_master_strategy(self):
        """
        Test the master strategy.
        """
        logger.info("Testing master strategy")
        
        # Get sample data
        klines = self.mock_data_provider.get_klines("XRP", "1h", limit=100)
        
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame(klines)
        
        # Set data on strategy
        self.master_strategy.set_data(df)
        
        # Generate signal
        signal = self.master_strategy.generate_signal()
        
        # Check signal
        self.assertIn(signal, [-1, 0, 1])
        
        # Test position size calculation
        position_size = self.master_strategy.calculate_position_size(10000, 0.5)
        self.assertGreater(position_size, 0)
        
        # Test take profit calculation
        take_profit = self.master_strategy.calculate_take_profit(0.5, "long")
        self.assertGreater(take_profit, 0.5)
        
        # Test stop loss calculation
        stop_loss = self.master_strategy.calculate_stop_loss(0.5, "long")
        self.assertLess(stop_loss, 0.5)
        
        logger.info("Master strategy test passed")
    
    def test_integration(self):
        """
        Test the integration of all components.
        """
        logger.info("Testing integration")
        
        # Get sample data
        klines = self.mock_data_provider.get_klines("XRP", "1h", limit=100)
        
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame(klines)
        
        # Set data on strategy
        self.master_strategy.set_data(df)
        
        # Generate signals
        signals = []
        
        for i in range(50, len(df)):
            # Get current candle data
            candle_data = df.iloc[i].to_dict()
            
            # Generate signal
            signal = self.master_strategy.generate_signal(candle_data)
            
            # Store signal
            signals.append(signal)
        
        # Simulate trades
        position = 0
        entry_price = 0
        entry_time = None
        trades = []
        
        for i, signal in enumerate(signals):
            current_price = df["close"].iloc[i + 50]
            current_time = df.index[i + 50] if hasattr(df, "index") else i + 50
            
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
            current_time = df.index[-1] if hasattr(df, "index") else len(df) - 1
            
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

def main():
    """
    Main function.
    """
    logger.info("Starting headless GUI test")
    
    # Run tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    logger.info("Headless GUI test completed")

if __name__ == "__main__":
    main()
