#!/usr/bin/env python3
"""
Extended Testing with Real Market Data

This script performs comprehensive testing of the Hyperliquid trading bot
with real market data to validate all enhancements and integrations.
"""

import os
import sys
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np

# Import custom modules
from core.hyperliquid_adapter import HyperliquidAdapter
from strategies.master_omni_overlord_robust import MasterOmniOverlordRobustStrategy
from strategies.robust_signal_generator import RobustSignalGenerator
from strategies.advanced_technical_indicators import AdvancedTechnicalIndicators
from enhanced_historical_data_accumulator import EnhancedHistoricalDataAccumulator
from order_book_handler import OrderBookHandler
from api_rate_limiter import APIRateLimiter
from error_handling import (
    ErrorHandler, ErrorSeverity, ErrorCategory, TradingError,
    APIError, DataError, CalculationError, SignalError, OrderError,
    with_error_handling, handle_error
)

# Configure logging
os.makedirs("logs", exist_ok=True)
os.makedirs("logs/test_errors", exist_ok=True)
os.makedirs("test_results", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("logs/extended_testing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
TEST_DURATION = 3600  # Test duration in seconds (1 hour)
DATA_FETCH_INTERVAL = 10  # Data fetch interval in seconds
SIGNAL_GENERATION_INTERVAL = 30  # Signal generation interval in seconds
SYMBOLS = ["BTC", "ETH", "SOL"]  # Symbols to test
TIMEFRAMES = ["1m", "5m", "15m", "1h"]  # Timeframes to test

class ExtendedTesting:
    """
    Performs extended testing of the Hyperliquid trading bot with real market data.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the testing environment.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize error handler
        self.error_handler = ErrorHandler(
            log_dir="logs/test_errors",
            max_retries=3,
            notification_callback=self._handle_error_notification
        )
        
        # Initialize components
        self.api_rate_limiter = APIRateLimiter()
        self.exchange_adapter = HyperliquidAdapter(self.config)
        self.order_book_handler = OrderBookHandler()
        
        # Initialize data accumulator with correct parameter names
        self.data_accumulator = EnhancedHistoricalDataAccumulator(
            symbols=SYMBOLS,
            timeframes=TIMEFRAMES,
            max_data_points=1000  # Changed from max_bars to max_data_points
        )
        
        # Initialize signal generator
        self.signal_generator = RobustSignalGenerator()
        
        # Initialize strategy with required parameters
        self.strategy = MasterOmniOverlordRobustStrategy(
            config=self.config,
            logger=logger,
            error_handler=self.error_handler
        )
        
        # Initialize data structures
        self.market_data = {}
        self.signals = {}
        self.order_books = {}
        self.performance_metrics = {
            "api_calls": 0,
            "api_errors": 0,
            "signal_generations": 0,
            "signal_errors": 0,
            "data_points": 0,
            "data_errors": 0,
            "start_time": None,
            "end_time": None,
            "signals_by_type": {
                "buy": 0,
                "sell": 0,
                "neutral": 0
            }
        }
        
        # Initialize threading components
        self.running = False
        self.data_thread = None
        self.signal_thread = None
        self.lock = threading.RLock()
        
    def _load_config(self) -> Dict:
        """
        Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    return json.load(f)
            else:
                logger.warning(f"Configuration file not found: {self.config_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
            
    def _handle_error_notification(self, error: TradingError) -> None:
        """
        Handle error notification.
        
        Args:
            error: Error to handle
        """
        logger.error(f"Error notification: {error}")
        
    def start(self) -> None:
        """
        Start the testing process.
        """
        if self.running:
            return
            
        logger.info("Starting extended testing...")
        self.running = True
        self.performance_metrics["start_time"] = datetime.now()
        
        # Start data thread
        self.data_thread = threading.Thread(target=self._data_thread_func)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        # Start signal thread
        self.signal_thread = threading.Thread(target=self._signal_thread_func)
        self.signal_thread.daemon = True
        self.signal_thread.start()
        
        # Wait for test duration
        time.sleep(TEST_DURATION)
        
        # Stop testing
        self.stop()
        
    def stop(self) -> None:
        """
        Stop the testing process.
        """
        if not self.running:
            return
            
        logger.info("Stopping extended testing...")
        self.running = False
        self.performance_metrics["end_time"] = datetime.now()
        
        # Wait for threads to finish
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=5.0)
            
        if self.signal_thread and self.signal_thread.is_alive():
            self.signal_thread.join(timeout=5.0)
            
        # Generate test report
        self._generate_report()
        
        logger.info("Extended testing completed")
        
    def _data_thread_func(self) -> None:
        """
        Data thread function.
        """
        logger.info("Data thread started")
        
        while self.running:
            try:
                # Fetch market data for all symbols
                for symbol in SYMBOLS:
                    try:
                        # Fetch market data
                        market_data = self.api_rate_limiter.execute_with_rate_limit(
                            endpoint="market_data",
                            params={"symbol": symbol}
                        )
                        
                        # Increment API call counter
                        self.performance_metrics["api_calls"] += 1
                        
                        # Fetch order book
                        order_book = self.api_rate_limiter.execute_with_rate_limit(
                            endpoint="order_book",
                            params={"symbol": symbol}
                        )
                        
                        # Increment API call counter
                        self.performance_metrics["api_calls"] += 1
                        
                        # Process order book
                        processed_order_book = self.order_book_handler.process_order_book(order_book)
                        
                        # Update data
                        with self.lock:
                            self.market_data[symbol] = market_data
                            self.order_books[symbol] = processed_order_book
                            
                            # Add data to accumulator
                            self.data_accumulator.add_data_point(
                                symbol=symbol,
                                timestamp=datetime.now(),
                                data=market_data
                            )
                            
                            # Increment data point counter
                            self.performance_metrics["data_points"] += 1
                            
                        logger.info(f"Fetched data for {symbol}: {market_data.get('price', 0)}")
                    except Exception as e:
                        logger.error(f"Error fetching data for {symbol}: {str(e)}")
                        self.performance_metrics["data_errors"] += 1
                        self.performance_metrics["api_errors"] += 1
                        
                # Sleep to avoid excessive API calls
                time.sleep(DATA_FETCH_INTERVAL)
            except Exception as e:
                logger.error(f"Error in data thread: {str(e)}")
                self.performance_metrics["data_errors"] += 1
                time.sleep(DATA_FETCH_INTERVAL * 2)  # Sleep longer on error
                
        logger.info("Data thread stopped")
        
    def _signal_thread_func(self) -> None:
        """
        Signal thread function.
        """
        logger.info("Signal thread started")
        
        # Wait for initial data
        time.sleep(DATA_FETCH_INTERVAL * 2)
        
        while self.running:
            try:
                # Generate signals for all symbols
                for symbol in SYMBOLS:
                    try:
                        # Get historical data
                        historical_data = self.data_accumulator.get_data(
                            symbol=symbol,
                            timeframe="5m",
                            limit=100
                        )
                        
                        if historical_data is None or len(historical_data) < 10:
                            logger.warning(f"Insufficient historical data for {symbol}")
                            continue
                            
                        # Get order book
                        order_book = self.order_books.get(symbol, None)
                        
                        # Generate signal
                        signal = self.signal_generator.generate_master_signal(
                            df=historical_data,
                            order_book=order_book
                        )
                        
                        # Update signal
                        with self.lock:
                            self.signals[symbol] = signal
                            
                            # Increment signal counter
                            self.performance_metrics["signal_generations"] += 1
                            
                            # Update signal type counter
                            signal_type = signal.get("signal", "neutral")
                            self.performance_metrics["signals_by_type"][signal_type] += 1
                            
                        logger.info(f"Generated signal for {symbol}: {signal.get('signal', 'neutral')}")
                    except Exception as e:
                        logger.error(f"Error generating signal for {symbol}: {str(e)}")
                        self.performance_metrics["signal_errors"] += 1
                        
                # Sleep to avoid excessive processing
                time.sleep(SIGNAL_GENERATION_INTERVAL)
            except Exception as e:
                logger.error(f"Error in signal thread: {str(e)}")
                self.performance_metrics["signal_errors"] += 1
                time.sleep(SIGNAL_GENERATION_INTERVAL * 2)  # Sleep longer on error
                
        logger.info("Signal thread stopped")
        
    def _generate_report(self) -> None:
        """
        Generate test report.
        """
        logger.info("Generating test report...")
        
        # Calculate test duration
        start_time = self.performance_metrics.get("start_time", datetime.now())
        end_time = self.performance_metrics.get("end_time", datetime.now())
        duration = (end_time - start_time).total_seconds()
        
        # Calculate success rates
        api_success_rate = 1.0 - (self.performance_metrics.get("api_errors", 0) / max(1, self.performance_metrics.get("api_calls", 1)))
        signal_success_rate = 1.0 - (self.performance_metrics.get("signal_errors", 0) / max(1, self.performance_metrics.get("signal_generations", 1)))
        data_success_rate = 1.0 - (self.performance_metrics.get("data_errors", 0) / max(1, self.performance_metrics.get("data_points", 1)))
        
        # Calculate signal distribution
        signal_total = sum(self.performance_metrics.get("signals_by_type", {}).values())
        signal_distribution = {
            k: v / max(1, signal_total) for k, v in self.performance_metrics.get("signals_by_type", {}).items()
        }
        
        # Create report
        report = {
            "test_duration": duration,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "api_calls": self.performance_metrics.get("api_calls", 0),
            "api_errors": self.performance_metrics.get("api_errors", 0),
            "api_success_rate": api_success_rate,
            "signal_generations": self.performance_metrics.get("signal_generations", 0),
            "signal_errors": self.performance_metrics.get("signal_errors", 0),
            "signal_success_rate": signal_success_rate,
            "data_points": self.performance_metrics.get("data_points", 0),
            "data_errors": self.performance_metrics.get("data_errors", 0),
            "data_success_rate": data_success_rate,
            "signal_distribution": signal_distribution,
            "signals": self.signals,
            "error_statistics": self.error_handler.get_statistics()
        }
        
        # Save report to file
        report_path = f"test_results/extended_test_report_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
                
            logger.info(f"Test report saved to {report_path}")
        except Exception as e:
            logger.error(f"Error saving test report: {str(e)}")
            
        # Log report summary
        logger.info("Test Report Summary:")
        logger.info(f"Test Duration: {duration:.2f} seconds")
        logger.info(f"API Calls: {report['api_calls']} (Success Rate: {report['api_success_rate']:.2%})")
        logger.info(f"Signal Generations: {report['signal_generations']} (Success Rate: {report['signal_success_rate']:.2%})")
        logger.info(f"Data Points: {report['data_points']} (Success Rate: {report['data_success_rate']:.2%})")
        logger.info(f"Signal Distribution: Buy: {report['signal_distribution'].get('buy', 0):.2%}, Sell: {report['signal_distribution'].get('sell', 0):.2%}, Neutral: {report['signal_distribution'].get('neutral', 0):.2%}")
        
        # Print final success rate
        overall_success_rate = (api_success_rate + signal_success_rate + data_success_rate) / 3
        logger.info(f"Overall Success Rate: {overall_success_rate:.2%}")
        
        # Save success rate to file
        success_rate_path = "test_results/success_rate.txt"
        
        try:
            with open(success_rate_path, "w") as f:
                f.write(f"Overall Success Rate: {overall_success_rate:.2%}\n")
                f.write(f"API Success Rate: {api_success_rate:.2%}\n")
                f.write(f"Signal Success Rate: {signal_success_rate:.2%}\n")
                f.write(f"Data Success Rate: {data_success_rate:.2%}\n")
                
            logger.info(f"Success rate saved to {success_rate_path}")
        except Exception as e:
            logger.error(f"Error saving success rate: {str(e)}")

def main():
    """
    Main function.
    """
    logger.info("Starting extended testing...")
    
    # Create testing instance
    testing = ExtendedTesting()
    
    # Start testing
    testing.start()
    
    logger.info("Extended testing completed")

if __name__ == "__main__":
    main()
