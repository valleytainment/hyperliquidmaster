#!/usr/bin/env python3
"""
Headless Test Runner for Hyperliquid Trading Bot

This module provides functionality to test the Hyperliquid trading bot
in a headless environment without requiring a GUI.
"""

import os
import sys
import time
import json
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Import core components
from core.hyperliquid_adapter import HyperliquidAdapter
from strategies.master_omni_overlord_robust import MasterOmniOverlordRobustStrategy
from historical_data_accumulator import HistoricalDataAccumulator
from order_book_handler import OrderBookHandler
from api_rate_limiter import APIRateLimiter
from json_serialization import JSONEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class HeadlessTestRunner:
    """
    Headless test runner for the Hyperliquid trading bot.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the headless test runner.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.strategy = MasterOmniOverlordRobustStrategy(config=self.config)
        self.data_accumulator = HistoricalDataAccumulator()
        self.order_book_handler = OrderBookHandler()
        self.api_rate_limiter = APIRateLimiter()
        self.adapter = HyperliquidAdapter()
        self.test_results = {}
        
    def _load_config(self) -> Dict:
        """
        Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Config file {self.config_path} not found, using default config")
                return {
                    "symbols": ["BTC", "ETH", "SOL"],
                    "risk": {
                        "account_size": 10000,
                        "risk_per_trade": 0.01
                    },
                    "strategy": {
                        "signal_threshold": 0.7,
                        "trend_filter_strength": 0.5,
                        "mean_reversion_factor": 0.3
                    }
                }
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return {}
            
    def _save_results(self, results: Dict, filename: str = "test_results.json") -> None:
        """
        Save test results to file.
        
        Args:
            results: Test results dictionary
            filename: Output filename
        """
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, cls=JSONEncoder)
                
            logger.info(f"Test results saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving test results: {str(e)}")
            
    def test_market_data_retrieval(self) -> Dict:
        """
        Test market data retrieval functionality.
        
        Returns:
            Test results dictionary
        """
        logger.info("Testing market data retrieval")
        
        results = {
            "success": True,
            "symbols_tested": 0,
            "symbols_succeeded": 0,
            "errors": 0,
            "data": {}
        }
        
        symbols = self.config.get("symbols", ["BTC", "ETH", "SOL"])
        
        for symbol in symbols:
            try:
                # Use rate limiter to avoid hitting API limits
                self.api_rate_limiter.execute_with_rate_limit(lambda: None, "test_market_data", {"symbol": symbol})
                
                # Get market data
                market_data = asyncio.run(self.adapter.get_market_data(symbol))
                
                if market_data and "last_price" in market_data:
                    logger.info(f"Successfully retrieved market data for {symbol}: price={market_data['last_price']}")
                    results["symbols_succeeded"] += 1
                    results["data"][symbol] = {
                        "price": market_data["last_price"],
                        "funding_rate": market_data.get("funding_rate", 0.0)
                    }
                else:
                    logger.error(f"Failed to retrieve market data for {symbol}")
                    results["errors"] += 1
                    results["success"] = False
            except Exception as e:
                logger.error(f"Error retrieving market data for {symbol}: {str(e)}")
                results["errors"] += 1
                results["success"] = False
                
            results["symbols_tested"] += 1
            
            # Sleep to avoid rate limiting
            time.sleep(1)
            
        return results
        
    def test_order_book_retrieval(self) -> Dict:
        """
        Test order book retrieval functionality.
        
        Returns:
            Test results dictionary
        """
        logger.info("Testing order book retrieval")
        
        results = {
            "success": True,
            "symbols_tested": 0,
            "symbols_succeeded": 0,
            "errors": 0,
            "data": {}
        }
        
        symbols = self.config.get("symbols", ["BTC", "ETH", "SOL"])
        
        for symbol in symbols:
            try:
                # Use rate limiter to avoid hitting API limits
                self.api_rate_limiter.execute_with_rate_limit(lambda: None, "test_order_book", {"symbol": symbol})
                
                # Get order book
                order_book = self.order_book_handler.get_order_book(symbol)
                
                if order_book and "bids" in order_book and "asks" in order_book:
                    bid_count = len(order_book["bids"])
                    ask_count = len(order_book["asks"])
                    
                    logger.info(f"Successfully retrieved order book for {symbol}: {bid_count} bids, {ask_count} asks")
                    results["symbols_succeeded"] += 1
                    results["data"][symbol] = {
                        "bid_count": bid_count,
                        "ask_count": ask_count
                    }
                else:
                    logger.error(f"Failed to retrieve order book for {symbol}")
                    results["errors"] += 1
                    results["success"] = False
            except Exception as e:
                logger.error(f"Error retrieving order book for {symbol}: {str(e)}")
                results["errors"] += 1
                results["success"] = False
                
            results["symbols_tested"] += 1
            
            # Sleep to avoid rate limiting
            time.sleep(1)
            
        return results
        
    def test_historical_data_accumulation(self) -> Dict:
        """
        Test historical data accumulation functionality.
        
        Returns:
            Test results dictionary
        """
        logger.info("Testing historical data accumulation")
        
        results = {
            "success": True,
            "symbols_tested": 0,
            "symbols_succeeded": 0,
            "errors": 0,
            "data": {}
        }
        
        symbols = self.config.get("symbols", ["BTC", "ETH", "SOL"])
        
        for symbol in symbols:
            try:
                # Use rate limiter to avoid hitting API limits
                self.api_rate_limiter.execute_with_rate_limit(lambda: None, "test_historical_data", {"symbol": symbol})
                
                # Get market data
                market_data = asyncio.run(self.adapter.get_market_data(symbol))
                
                if market_data and "last_price" in market_data:
                    # Add data point
                    self.data_accumulator.add_data_point(
                        symbol=symbol,
                        market_data=market_data
                    )
                    
                    # Get historical data
                    historical_data = self.data_accumulator.get_historical_data(symbol)
                    
                    if historical_data is not None:
                        data_points = len(historical_data)
                        logger.info(f"Successfully accumulated historical data for {symbol}: {data_points} data points")
                        results["symbols_succeeded"] += 1
                        results["data"][symbol] = {
                            "data_points": data_points
                        }
                    else:
                        logger.error(f"Failed to accumulate historical data for {symbol}")
                        results["errors"] += 1
                        results["success"] = False
                else:
                    logger.error(f"Failed to retrieve market data for {symbol}")
                    results["errors"] += 1
                    results["success"] = False
            except Exception as e:
                logger.error(f"Error accumulating historical data for {symbol}: {str(e)}")
                results["errors"] += 1
                results["success"] = False
                
            results["symbols_tested"] += 1
            
            # Sleep to avoid rate limiting
            time.sleep(1)
            
        return results
        
    def test_signal_generation(self) -> Dict:
        """
        Test signal generation functionality.
        
        Returns:
            Test results dictionary
        """
        logger.info("Testing signal generation")
        
        results = {
            "success": True,
            "symbols_tested": 0,
            "symbols_succeeded": 0,
            "errors": 0,
            "data": {}
        }
        
        symbols = self.config.get("symbols", ["BTC", "ETH", "SOL"])
        
        for symbol in symbols:
            try:
                # Use rate limiter to avoid hitting API limits
                self.api_rate_limiter.execute_with_rate_limit(lambda: None, "test_signal_generation", {"symbol": symbol})
                
                # Get market data
                market_data = asyncio.run(self.adapter.get_market_data(symbol))
                
                # Get order book
                order_book = self.order_book_handler.get_order_book(symbol)
                
                # Process order book
                processed_order_book = self.order_book_handler.process_order_book(symbol, order_book)
                
                # Add order book metrics to market data
                market_data["order_book_metrics"] = processed_order_book["metrics"] if processed_order_book else {}
                
                # Generate signal
                signal = self.strategy.generate_signal(symbol, market_data, order_book=processed_order_book)
                
                if signal:
                    logger.info(f"Successfully generated signal for {symbol}: action={signal['action']}, strength={signal.get('signal_strength', 0)}")
                    results["symbols_succeeded"] += 1
                    results["data"][symbol] = {
                        "action": signal["action"],
                        "signal_strength": signal.get("signal_strength", 0),
                        "entry_price": signal.get("entry_price", 0),
                        "stop_loss": signal.get("stop_loss", 0),
                        "take_profit": signal.get("take_profit", 0)
                    }
                else:
                    logger.error(f"Failed to generate signal for {symbol}")
                    results["errors"] += 1
                    results["success"] = False
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {str(e)}")
                results["errors"] += 1
                results["success"] = False
                
            results["symbols_tested"] += 1
            
            # Sleep to avoid rate limiting
            time.sleep(1)
            
        return results
        
    def test_continuous_operation(self, duration_seconds: int = 60) -> Dict:
        """
        Test continuous operation functionality.
        
        Args:
            duration_seconds: Test duration in seconds
            
        Returns:
            Test results dictionary
        """
        logger.info("Testing continuous operation")
        logger.info(f"Running continuous operation test for {duration_seconds} seconds")
        
        results = {
            "success": True,
            "duration_seconds": duration_seconds,
            "data_collection_count": 0,
            "signal_generation_count": 0,
            "errors": 0,
            "signals": {}
        }
        
        symbols = self.config.get("symbols", ["BTC", "ETH", "SOL"])
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        # Intervals
        data_collection_interval = 3  # seconds
        signal_generation_interval = 3  # seconds
        
        # Last execution times
        last_data_collection = 0
        last_signal_generation = 0
        
        while time.time() < end_time:
            try:
                current_time = time.time()
                
                # Collect data
                if current_time - last_data_collection >= data_collection_interval:
                    try:
                        for symbol in symbols:
                            try:
                                # Use rate limiter to avoid hitting API limits
                                self.api_rate_limiter.execute_with_rate_limit(lambda: None, "continuous_data_collection", {"symbol": symbol})
                                
                                # Get market data
                                market_data = asyncio.run(self.adapter.get_market_data(symbol))
                                
                                # Add data point
                                if market_data and "last_price" in market_data:
                                    self.data_accumulator.add_data_point(
                                        symbol=symbol,
                                        market_data=market_data
                                    )
                                    
                                    logger.info(f"Collected data for {symbol}: price={market_data['last_price']}")
                                    results["data_collection_count"] += 1
                            except Exception as e:
                                logger.error(f"Error collecting data for {symbol}: {str(e)}")
                                results["errors"] += 1
                                
                            # Sleep to avoid rate limiting
                            time.sleep(0.5)
                    except Exception as e:
                        logger.error(f"Error in data collection cycle: {str(e)}")
                        results["errors"] += 1
                        
                    last_data_collection = current_time
                    
                # Generate signals
                if current_time - last_signal_generation >= signal_generation_interval:
                    try:
                        for symbol in symbols:
                            try:
                                # Use rate limiter to avoid hitting API limits
                                self.api_rate_limiter.execute_with_rate_limit(lambda: None, "continuous_signal_generation", {"symbol": symbol})
                                
                                # Get market data
                                market_data = asyncio.run(self.adapter.get_market_data(symbol))
                                
                                # Get order book
                                order_book = self.order_book_handler.get_order_book(symbol)
                                
                                # Process order book
                                processed_order_book = self.order_book_handler.process_order_book(symbol, order_book)
                                
                                # Add order book metrics to market data
                                market_data["order_book_metrics"] = processed_order_book["metrics"] if processed_order_book else {}
                                
                                # Generate signal
                                signal = self.strategy.generate_signal(symbol, market_data, order_book=processed_order_book)
                                
                                if signal:
                                    logger.info(f"Generated signal for {symbol}: action={signal['action']}, strength={signal.get('signal_strength', 0)}")
                                    results["signal_generation_count"] += 1
                                    
                                    # Store the latest signal for each symbol
                                    results["signals"][symbol] = {
                                        "action": signal["action"],
                                        "signal_strength": signal.get("signal_strength", 0),
                                        "entry_price": signal.get("entry_price", 0),
                                        "stop_loss": signal.get("stop_loss", 0),
                                        "take_profit": signal.get("take_profit", 0),
                                        "timestamp": datetime.now().isoformat()
                                    }
                            except Exception as e:
                                logger.error(f"Error generating signal for {symbol}: {str(e)}")
                                results["errors"] += 1
                                
                            # Sleep to avoid rate limiting
                            time.sleep(0.5)
                    except Exception as e:
                        logger.error(f"Error in signal generation cycle: {str(e)}")
                        results["errors"] += 1
                        
                    last_signal_generation = current_time
                    
                # Sleep to avoid high CPU usage
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in continuous operation test: {str(e)}")
                results["errors"] += 1
                
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        results["actual_duration_seconds"] = elapsed_time
        
        logger.info(f"Continuous operation test completed in {elapsed_time:.2f} seconds")
        logger.info(f"Data collection count: {results['data_collection_count']}")
        logger.info(f"Signal generation count: {results['signal_generation_count']}")
        logger.info(f"Errors: {results['errors']}")
        
        return results
        
    def run_all_tests(self) -> Dict:
        """
        Run all tests.
        
        Returns:
            Test results dictionary
        """
        logger.info("Running all tests")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "tests": {}
        }
        
        # Test market data retrieval
        try:
            results["tests"]["market_data_retrieval"] = self.test_market_data_retrieval()
            if not results["tests"]["market_data_retrieval"]["success"]:
                results["success"] = False
        except Exception as e:
            logger.error(f"Error running market data retrieval test: {str(e)}")
            results["tests"]["market_data_retrieval"] = {"success": False, "error": str(e)}
            results["success"] = False
            
        # Test order book retrieval
        try:
            results["tests"]["order_book_retrieval"] = self.test_order_book_retrieval()
            if not results["tests"]["order_book_retrieval"]["success"]:
                results["success"] = False
        except Exception as e:
            logger.error(f"Error running order book retrieval test: {str(e)}")
            results["tests"]["order_book_retrieval"] = {"success": False, "error": str(e)}
            results["success"] = False
            
        # Test historical data accumulation
        try:
            results["tests"]["historical_data_accumulation"] = self.test_historical_data_accumulation()
            if not results["tests"]["historical_data_accumulation"]["success"]:
                results["success"] = False
        except Exception as e:
            logger.error(f"Error running historical data accumulation test: {str(e)}")
            results["tests"]["historical_data_accumulation"] = {"success": False, "error": str(e)}
            results["success"] = False
            
        # Test signal generation
        try:
            results["tests"]["signal_generation"] = self.test_signal_generation()
            if not results["tests"]["signal_generation"]["success"]:
                results["success"] = False
        except Exception as e:
            logger.error(f"Error running signal generation test: {str(e)}")
            results["tests"]["signal_generation"] = {"success": False, "error": str(e)}
            results["success"] = False
            
        # Test continuous operation
        try:
            results["tests"]["continuous_operation"] = self.test_continuous_operation(duration_seconds=60)
            if not results["tests"]["continuous_operation"]["success"]:
                results["success"] = False
        except Exception as e:
            logger.error(f"Error running continuous operation test: {str(e)}")
            results["tests"]["continuous_operation"] = {"success": False, "error": str(e)}
            results["success"] = False
            
        # Save results
        self._save_results(results)
        
        return results
        
if __name__ == "__main__":
    runner = HeadlessTestRunner()
    results = runner.run_all_tests()
    
    if results["success"]:
        logger.info("All tests passed!")
        sys.exit(0)
    else:
        logger.error("Some tests failed!")
        sys.exit(1)
