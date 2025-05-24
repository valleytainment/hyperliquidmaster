#!/usr/bin/env python3
"""
Extended Testing with Mock Data Integration

This script performs extended testing of the Hyperliquid trading bot
with mock data integration to ensure high success rate even when
API rate limits are encountered.
"""

import os
import sys
import time
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from core.hyperliquid_adapter import HyperliquidAdapter
from api_rate_limiter import APIRateLimiter
from enhanced_historical_data_accumulator import EnhancedHistoricalDataAccumulator
from strategies.master_omni_overlord_robust import MasterOmniOverlordRobustStrategy
from error_handling_fixed import ErrorHandler
from strategies.advanced_technical_indicators_fixed import AdvancedTechnicalIndicators
from strategies.robust_signal_generator_fixed import RobustSignalGenerator

class ExtendedTesting:
    """
    Extended testing of the Hyperliquid trading bot with mock data integration.
    """
    
    def __init__(self, use_mock_data: bool = True):
        """
        Initialize extended testing.
        
        Args:
            use_mock_data: Whether to use mock data for testing
        """
        # Create directories for test results and logs
        os.makedirs("test_results", exist_ok=True)
        os.makedirs("logs/test_errors", exist_ok=True)
        
        # Initialize error handler with correct parameters
        self.error_handler = ErrorHandler(
            log_dir="logs/test_errors",
            max_retries=3
        )
        
        # Initialize API rate limiter with mock data integration
        self.rate_limiter = APIRateLimiter(
            rate_limits={
                "default": 60,
                "market_data": 120,
                "order_book": 60,
                "historical_data": 30,
                "user_state": 20
            },
            cooldown_file="rate_limit_cooldown.json",
            use_mock_data=use_mock_data
        )
        
        # Initialize exchange adapter
        self.exchange = HyperliquidAdapter()
        
        # Initialize historical data accumulator
        self.data_accumulator = EnhancedHistoricalDataAccumulator(
            symbols=["BTC", "ETH", "SOL"],
            timeframes=["1m", "5m", "15m", "1h", "4h", "1d"],
            max_data_points=1000,
            data_dir="historical_data"
        )
        
        # Initialize technical indicators
        self.indicators = AdvancedTechnicalIndicators()
        
        # Initialize signal generator
        self.signal_generator = RobustSignalGenerator()
        
        # Initialize strategy
        self.strategy = MasterOmniOverlordRobustStrategy(
            config={
                "symbols": ["BTC", "ETH", "SOL"],
                "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
                "risk_level": "medium",
                "max_position_size": 0.1,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.05,
                "trailing_stop_pct": 0.01
            },
            logger=logger,
            error_handler=self.error_handler,
            data_accumulator=self.data_accumulator,
            exchange_adapter=self.exchange,
            signal_generator=self.signal_generator
        )
        
        logger.info("Initializing Master Omni Overlord Robust Strategy...")
    
    def test_api_rate_limiting(self):
        """
        Test API rate limiting with mock data integration.
        """
        logger.info("Testing API rate limiting...")
        
        # Test market data endpoint
        for symbol in ["BTC", "ETH", "SOL"]:
            for i in range(5):
                result = self.rate_limiter.execute_with_rate_limit(
                    "market_data",
                    {"symbol": symbol}
                )
                
                if "error" in result:
                    logger.error(f"Error getting market data for {symbol}: {result['error']}")
                else:
                    logger.info(f"Market data for {symbol}: price={result.get('price')}, funding_rate={result.get('funding_rate')}")
                
                # Add small delay to avoid overwhelming logs
                time.sleep(0.5)
        
        # Test order book endpoint
        for symbol in ["BTC", "ETH", "SOL"]:
            result = self.rate_limiter.execute_with_rate_limit(
                "order_book",
                {"symbol": symbol, "depth": 5}
            )
            
            if "error" in result:
                logger.error(f"Error getting order book for {symbol}: {result['error']}")
            else:
                logger.info(f"Order book for {symbol}: {len(result.get('bids', []))} bids, {len(result.get('asks', []))} asks")
            
            # Add small delay to avoid overwhelming logs
            time.sleep(0.5)
        
        # Test historical data endpoint
        for symbol in ["BTC"]:
            for timeframe in ["1m", "1h"]:
                result = self.rate_limiter.execute_with_rate_limit(
                    "historical_data",
                    {"symbol": symbol, "timeframe": timeframe, "limit": 10}
                )
                
                if "error" in result:
                    logger.error(f"Error getting historical data for {symbol} ({timeframe}): {result['error']}")
                else:
                    logger.info(f"Historical data for {symbol} ({timeframe}): {len(result.get('candles', []))} candles")
                
                # Add small delay to avoid overwhelming logs
                time.sleep(0.5)
        
        # Test cooldown mechanism
        logger.info("Testing cooldown mechanism...")
        self.rate_limiter.force_cooldown("market_data", 1)  # 1 minute cooldown
        
        # This should use mock data due to cooldown
        result = self.rate_limiter.execute_with_rate_limit(
            "market_data",
            {"symbol": "BTC"}
        )
        
        if "error" in result:
            logger.error(f"Error getting market data during cooldown: {result['error']}")
        else:
            logger.info(f"Market data during cooldown: price={result.get('price')}, funding_rate={result.get('funding_rate')}")
        
        # Clear cooldown
        self.rate_limiter.clear_cooldown("market_data")
        logger.info("Cleared cooldown for market_data")
    
    def test_historical_data_accumulation(self):
        """
        Test historical data accumulation with mock data integration.
        """
        logger.info("Testing historical data accumulation...")
        
        # Accumulate historical data
        for symbol in ["BTC", "ETH", "SOL"]:
            for tf in ["1m", "5m", "1h"]:
                # Get historical data
                result = self.rate_limiter.execute_with_rate_limit(
                    "historical_data",
                    {"symbol": symbol, "timeframe": tf, "limit": 100}
                )
                
                if "error" in result:
                    logger.error(f"Error getting historical data for {symbol} ({tf}): {result['error']}")
                    continue
                
                candles = result.get("candles", [])
                logger.info(f"Got {len(candles)} candles for {symbol} ({tf})")
                
                # Add data to accumulator - FIXED: removed timeframe parameter
                for candle in candles:
                    try:
                        # Convert timestamp to seconds if it's in milliseconds
                        timestamp = candle["timestamp"]
                        if timestamp > 253402300799:  # Max timestamp for year 9999
                            timestamp = timestamp / 1000
                            
                        self.data_accumulator.add_data_point(
                            symbol=symbol,
                            timestamp=timestamp,
                            data={
                                "open": candle["open"],
                                "high": candle["high"],
                                "low": candle["low"],
                                "close": candle["close"],
                                "volume": candle["volume"]
                            }
                        )
                    except Exception as e:
                        logger.error(f"Error adding data point for {symbol}: {str(e)}")
                
                # Add small delay to avoid overwhelming logs
                time.sleep(0.5)
        
        # Verify data accumulation
        for symbol in ["BTC", "ETH", "SOL"]:
            for tf in ["1m", "5m", "1h"]:
                try:
                    data = self.data_accumulator.get_data(symbol, tf)
                    # FIXED: Handle pandas DataFrame or None properly
                    data_count = 0
                    if data is not None:
                        if hasattr(data, 'empty'):  # It's a pandas DataFrame
                            if not data.empty:
                                data_count = len(data)
                        elif isinstance(data, list):  # It's a list
                            data_count = len(data)
                            
                    logger.info(f"Accumulated {data_count} data points for {symbol} ({tf})")
                except Exception as e:
                    logger.error(f"Error getting data for {symbol} ({tf}): {str(e)}")
    
    def test_technical_indicators(self):
        """
        Test technical indicators with accumulated data.
        """
        logger.info("Testing technical indicators...")
        
        # Get data for indicators
        symbol = "BTC"
        timeframe = "1h"
        try:
            data = self.data_accumulator.get_data(symbol, timeframe)
            
            # FIXED: Handle pandas DataFrame or None properly
            if data is None or (hasattr(data, 'empty') and data.empty) or (isinstance(data, list) and len(data) == 0):
                logger.error(f"No data available for {symbol} ({timeframe})")
                return
            
            # Calculate indicators
            try:
                # Convert data to format expected by indicators
                if hasattr(data, 'to_dict'):  # It's a pandas DataFrame
                    # Convert DataFrame to dict of lists
                    data_dict = {col: data[col].tolist() for col in data.columns if col in ['open', 'high', 'low', 'close', 'volume']}
                    ohlcv = data_dict
                else:  # It's a list of dicts
                    ohlcv = {
                        "open": [d["open"] for d in data],
                        "high": [d["high"] for d in data],
                        "low": [d["low"] for d in data],
                        "close": [d["close"] for d in data],
                        "volume": [d["volume"] for d in data]
                    }
                
                # Calculate various indicators
                rsi = self.indicators.calculate_rsi(ohlcv["close"])
                macd = self.indicators.calculate_macd(ohlcv["close"])
                bollinger = self.indicators.calculate_bollinger_bands(ohlcv["close"])
                
                logger.info(f"RSI (last 5 values): {rsi[-5:] if len(rsi) >= 5 else rsi}")
                logger.info(f"MACD (last 5 values): {[round(m, 4) for m in macd['macd'][-5:]] if len(macd['macd']) >= 5 else macd['macd']}")
                logger.info(f"Bollinger Bands (last value): Upper={bollinger['upper'][-1] if bollinger['upper'] else 'N/A'}, Middle={bollinger['middle'][-1] if bollinger['middle'] else 'N/A'}, Lower={bollinger['lower'][-1] if bollinger['lower'] else 'N/A'}")
                
                # Test advanced indicators
                supertrend = self.indicators.calculate_supertrend(ohlcv)
                ichimoku = self.indicators.calculate_ichimoku_cloud(ohlcv)
                
                logger.info(f"SuperTrend (last 5 values): {supertrend[-5:] if len(supertrend) >= 5 else supertrend}")
                logger.info(f"Ichimoku Cloud (last value): Tenkan={ichimoku['tenkan'][-1] if ichimoku['tenkan'] else 'N/A'}, Kijun={ichimoku['kijun'][-1] if ichimoku['kijun'] else 'N/A'}")
                
                # Test pattern recognition
                patterns = self.indicators.detect_candlestick_patterns(ohlcv)
                logger.info(f"Detected patterns: {patterns[-5:] if len(patterns) >= 5 else patterns}")
                
                # Test market regime detection
                regime = self.indicators.detect_market_regime(ohlcv)
                logger.info(f"Market regime: {regime}")
            except Exception as e:
                logger.error(f"Error calculating indicators: {str(e)}")
        except Exception as e:
            logger.error(f"Error getting data for {symbol} ({timeframe}): {str(e)}")
    
    def test_strategy_signal_generation(self):
        """
        Test strategy signal generation with accumulated data.
        """
        logger.info("Testing strategy signal generation...")
        
        # Get order book data
        order_book = {}
        for symbol in ["BTC", "ETH", "SOL"]:
            result = self.rate_limiter.execute_with_rate_limit(
                "order_book",
                {"symbol": symbol, "depth": 10}
            )
            
            if "error" in result:
                logger.error(f"Error getting order book for {symbol}: {result['error']}")
                order_book[symbol] = {"bids": [], "asks": []}
            else:
                order_book[symbol] = {
                    "bids": result.get("bids", []),
                    "asks": result.get("asks", [])
                }
        
        # Generate signals
        for symbol in ["BTC", "ETH", "SOL"]:
            try:
                # Get market data
                market_data = self.rate_limiter.execute_with_rate_limit(
                    "market_data",
                    {"symbol": symbol}
                )
                
                if "error" in market_data:
                    logger.error(f"Error getting market data for {symbol}: {market_data['error']}")
                    continue
                
                # Generate signal using RobustSignalGenerator directly
                signal = self.signal_generator.generate_master_signal(
                    market_data=market_data,
                    order_book=order_book[symbol]
                )
                
                logger.info(f"Signal for {symbol}: {signal['action']} (strength: {signal['strength']:.2f})")
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {str(e)}")
    
    def test_error_handling(self):
        """
        Test error handling with various error scenarios.
        """
        logger.info("Testing error handling...")
        
        # Test API error handling
        try:
            # Force an API error by using an invalid symbol
            result = self.rate_limiter.execute_with_rate_limit(
                "market_data",
                {"symbol": "INVALID_SYMBOL"}
            )
            
            if "error" in result:
                logger.info(f"Successfully handled API error: {result['error']}")
                
                # Log error through error handler
                self.error_handler.log_error(
                    "API",
                    f"Invalid symbol: INVALID_SYMBOL",
                    {"endpoint": "market_data", "params": {"symbol": "INVALID_SYMBOL"}}
                )
            else:
                logger.warning("Expected API error but got success")
        except Exception as e:
            logger.error(f"Unexpected exception during API error test: {str(e)}")
        
        # Test data error handling
        try:
            # Force a data error by using invalid data
            self.data_accumulator.add_data_point(
                symbol="BTC",
                timestamp=int(time.time() * 1000),
                data={"invalid": "data"}
            )
            
            logger.info("Successfully handled data error")
        except Exception as e:
            self.error_handler.log_error(
                "DATA",
                f"Invalid data format: {str(e)}",
                {"symbol": "BTC"}
            )
            logger.info(f"Successfully caught and logged data error: {str(e)}")
        
        # Test calculation error handling
        try:
            # Force a calculation error by using invalid input
            self.indicators.calculate_rsi(None)
            
            logger.warning("Expected calculation error but got success")
        except Exception as e:
            self.error_handler.log_error(
                "CALCULATION",
                f"Invalid input for RSI calculation: {str(e)}",
                {"indicator": "RSI"}
            )
            logger.info(f"Successfully caught and logged calculation error: {str(e)}")
    
    def run_all_tests(self):
        """
        Run all tests.
        """
        logger.info("Running all tests...")
        
        # Run tests
        self.test_api_rate_limiting()
        self.test_historical_data_accumulation()
        self.test_technical_indicators()
        self.test_strategy_signal_generation()
        self.test_error_handling()
        
        logger.info("All tests completed")
    
    def save_results(self):
        """
        Save test results.
        """
        logger.info("Saving test results...")
        
        # Get rate limiter status
        rate_limiter_status = self.rate_limiter.get_status()
        
        # Save results
        results = {
            "timestamp": int(time.time()),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rate_limiter_status": rate_limiter_status,
            "data_accumulator_stats": {
                "symbols": self.data_accumulator.symbols,
                "timeframes": self.data_accumulator.timeframes
            }
        }
        
        # Save to file
        with open(os.path.join("test_results", f"test_results_{int(time.time())}.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info("Test results saved")

def main():
    """
    Main function.
    """
    logger.info("Starting extended testing with mock data integration...")
    
    # Initialize testing
    testing = ExtendedTesting(use_mock_data=True)
    
    # Run all tests
    testing.run_all_tests()
    
    # Save results
    testing.save_results()
    
    logger.info("Extended testing completed")

if __name__ == "__main__":
    main()
