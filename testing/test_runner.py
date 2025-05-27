"""
Test Runner for HyperLiquid Trading Bot

This module provides a comprehensive test suite to validate the functionality
and performance of the trading bot with all enhancements.
"""

import os
import time
import logging
import asyncio
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from testing.backtest_engine import BacktestEngine
from strategies.master_omni_overlord import MasterOmniOverlordStrategy
from strategies.triple_confluence import TripleConfluenceStrategy
from strategies.oracle_update import OracleUpdateStrategy
from core.hyperliquid_adapter import HyperliquidAdapter
from core.enhanced_connection_manager import EnhancedConnectionManager
from core.settings_manager import SettingsManager
from core.advanced_cache import CacheManager
from sentiment.sentiment_analyzer import SentimentAnalyzer

class TestRunner:
    """
    Test runner for validating trading bot functionality and performance.
    """
    
    def __init__(self, config_path: str, logger=None):
        """
        Initialize test runner.
        
        Args:
            config_path: Path to configuration file
            logger: Optional logger instance
        """
        # Setup logging
        self.logger = logger or self._setup_logger()
        
        # Store configuration path
        self.config_path = config_path
        
        # Initialize settings manager
        self.settings_manager = SettingsManager(config_path, self.logger)
        
        # Load configuration
        self.config = self.settings_manager.get_settings()
        
        # Initialize test results
        self.test_results = {}
        
        # Output directory
        self.output_dir = self.config.get("test_output_dir", "test_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger.info("Test runner initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """
        Set up the logger.
        
        Returns:
            Configured logger
        """
        logger = logging.getLogger("TestRunner")
        logger.setLevel(logging.INFO)
        
        # Check if handlers already exist
        if not logger.handlers:
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            console_handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(console_handler)
            
            # Create file handler
            os.makedirs("logs", exist_ok=True)
            file_handler = logging.FileHandler("logs/test_runner.log")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(file_handler)
            
        return logger
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all tests.
        
        Returns:
            Dictionary with test results
        """
        start_time = time.time()
        self.logger.info("Starting all tests")
        
        # Run unit tests
        unit_test_results = await self.run_unit_tests()
        
        # Run integration tests
        integration_test_results = await self.run_integration_tests()
        
        # Run backtests
        backtest_results = await self.run_backtests()
        
        # Combine results
        all_results = {
            "unit_tests": unit_test_results,
            "integration_tests": integration_test_results,
            "backtests": backtest_results,
            "duration": time.time() - start_time
        }
        
        # Calculate overall success
        all_results["success"] = (
            unit_test_results.get("success", False) and
            integration_test_results.get("success", False) and
            backtest_results.get("success", False)
        )
        
        self.logger.info(f"All tests completed in {all_results['duration']:.2f} seconds")
        self.logger.info(f"Overall success: {all_results['success']}")
        
        # Store results
        self.test_results = all_results
        
        return all_results
    
    async def run_unit_tests(self) -> Dict[str, Any]:
        """
        Run unit tests.
        
        Returns:
            Dictionary with unit test results
        """
        start_time = time.time()
        self.logger.info("Starting unit tests")
        
        # Initialize results
        results = {
            "tests": [],
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "duration": 0
        }
        
        # Run tests for each component
        try:
            # Test connection manager
            connection_manager_results = await self._test_connection_manager()
            results["tests"].append(connection_manager_results)
            
            # Test settings manager
            settings_manager_results = await self._test_settings_manager()
            results["tests"].append(settings_manager_results)
            
            # Test cache manager
            cache_manager_results = await self._test_cache_manager()
            results["tests"].append(cache_manager_results)
            
            # Test sentiment analyzer
            sentiment_analyzer_results = await self._test_sentiment_analyzer()
            results["tests"].append(sentiment_analyzer_results)
            
            # Test strategy components
            strategy_results = await self._test_strategies()
            results["tests"].append(strategy_results)
            
            # Calculate summary
            for test in results["tests"]:
                results["passed"] += test.get("passed", 0)
                results["failed"] += test.get("failed", 0)
                results["errors"] += test.get("errors", 0)
            
            results["duration"] = time.time() - start_time
            results["success"] = results["failed"] == 0 and results["errors"] == 0
            
            self.logger.info(f"Unit tests completed in {results['duration']:.2f} seconds")
            self.logger.info(f"Passed: {results['passed']}, Failed: {results['failed']}, Errors: {results['errors']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running unit tests: {e}")
            results["errors"] += 1
            results["duration"] = time.time() - start_time
            results["success"] = False
            return results
    
    async def run_integration_tests(self) -> Dict[str, Any]:
        """
        Run integration tests.
        
        Returns:
            Dictionary with integration test results
        """
        start_time = time.time()
        self.logger.info("Starting integration tests")
        
        # Initialize results
        results = {
            "tests": [],
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "duration": 0
        }
        
        # Run tests for integrated components
        try:
            # Test adapter with connection manager
            adapter_connection_results = await self._test_adapter_with_connection_manager()
            results["tests"].append(adapter_connection_results)
            
            # Test strategy with sentiment analyzer
            strategy_sentiment_results = await self._test_strategy_with_sentiment()
            results["tests"].append(strategy_sentiment_results)
            
            # Test full trading pipeline
            trading_pipeline_results = await self._test_trading_pipeline()
            results["tests"].append(trading_pipeline_results)
            
            # Calculate summary
            for test in results["tests"]:
                results["passed"] += test.get("passed", 0)
                results["failed"] += test.get("failed", 0)
                results["errors"] += test.get("errors", 0)
            
            results["duration"] = time.time() - start_time
            results["success"] = results["failed"] == 0 and results["errors"] == 0
            
            self.logger.info(f"Integration tests completed in {results['duration']:.2f} seconds")
            self.logger.info(f"Passed: {results['passed']}, Failed: {results['failed']}, Errors: {results['errors']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running integration tests: {e}")
            results["errors"] += 1
            results["duration"] = time.time() - start_time
            results["success"] = False
            return results
    
    async def run_backtests(self) -> Dict[str, Any]:
        """
        Run backtests.
        
        Returns:
            Dictionary with backtest results
        """
        start_time = time.time()
        self.logger.info("Starting backtests")
        
        # Initialize results
        results = {
            "tests": [],
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "duration": 0,
            "best_strategy": None,
            "best_params": None,
            "best_return": 0
        }
        
        try:
            # Initialize backtest engine
            backtest_engine = BacktestEngine(self.config, self.logger)
            
            # Load historical data
            symbols = self.config.get("symbols", ["BTC", "ETH", "SOL"])
            start_date = "2023-01-01"
            end_date = "2023-12-31"
            timeframe = "1h"
            
            data = {}
            for symbol in symbols:
                df = backtest_engine.load_historical_data(symbol, start_date, end_date, timeframe)
                if not df.empty:
                    data[symbol] = df
            
            if not data:
                self.logger.error("No historical data loaded")
                results["errors"] += 1
                results["duration"] = time.time() - start_time
                results["success"] = False
                return results
            
            # Test each strategy
            strategies = [
                {
                    "name": "MasterOmniOverlord",
                    "func": self._master_omni_overlord_strategy,
                    "params": {
                        "signal_threshold": 0.7,
                        "trend_filter_strength": 0.5,
                        "mean_reversion_factor": 0.3,
                        "volatility_adjustment": 1.0
                    }
                },
                {
                    "name": "TripleConfluence",
                    "func": self._triple_confluence_strategy,
                    "params": {
                        "ma_short": 20,
                        "ma_medium": 50,
                        "ma_long": 200,
                        "rsi_period": 14,
                        "rsi_overbought": 70,
                        "rsi_oversold": 30
                    }
                },
                {
                    "name": "OracleUpdate",
                    "func": self._oracle_update_strategy,
                    "params": {
                        "update_threshold": 0.5,
                        "momentum_factor": 0.7,
                        "reversal_threshold": 0.3
                    }
                }
            ]
            
            for strategy in strategies:
                self.logger.info(f"Testing {strategy['name']} strategy")
                
                # Run backtest
                backtest_result = backtest_engine.run_backtest(
                    strategy["func"],
                    data,
                    strategy["params"]
                )
                
                # Save plot
                plot_filename = os.path.join(self.output_dir, f"{strategy['name']}_backtest.png")
                backtest_engine.plot_results(plot_filename)
                
                # Save results
                results_filename = os.path.join(self.output_dir, f"{strategy['name']}_backtest.json")
                backtest_engine.save_results(results_filename)
                
                # Check if strategy is profitable
                metrics = backtest_result["metrics"]
                is_profitable = metrics["total_return"] > 0 and metrics["sharpe_ratio"] > 0
                
                # Add to results
                strategy_result = {
                    "name": strategy["name"],
                    "metrics": metrics,
                    "profitable": is_profitable,
                    "plot_file": plot_filename,
                    "results_file": results_filename
                }
                
                results["tests"].append(strategy_result)
                
                if is_profitable:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                
                # Check if this is the best strategy
                if metrics["total_return"] > results["best_return"]:
                    results["best_strategy"] = strategy["name"]
                    results["best_params"] = strategy["params"]
                    results["best_return"] = metrics["total_return"]
            
            # Optimize best strategy
            if results["best_strategy"]:
                self.logger.info(f"Optimizing {results['best_strategy']} strategy")
                
                # Get best strategy
                best_strategy = next((s for s in strategies if s["name"] == results["best_strategy"]), None)
                
                if best_strategy:
                    # Define parameter grid
                    param_grid = {}
                    for param, value in best_strategy["params"].items():
                        if isinstance(value, float):
                            param_grid[param] = [value * 0.8, value, value * 1.2]
                        elif isinstance(value, int):
                            param_grid[param] = [max(1, value - 5), value, value + 5]
                    
                    # Run optimization
                    optimization_result = backtest_engine.optimize_strategy(
                        best_strategy["func"],
                        data,
                        param_grid,
                        best_strategy["params"]
                    )
                    
                    # Save optimization results
                    results["optimization"] = {
                        "strategy": results["best_strategy"],
                        "best_params": optimization_result["best_params"],
                        "best_metrics": optimization_result["best_metrics"]
                    }
                    
                    # Update best params
                    results["best_params"] = optimization_result["best_params"]
                    results["best_return"] = optimization_result["best_metrics"]["total_return"]
            
            results["duration"] = time.time() - start_time
            results["success"] = results["passed"] > 0
            
            self.logger.info(f"Backtests completed in {results['duration']:.2f} seconds")
            self.logger.info(f"Passed: {results['passed']}, Failed: {results['failed']}, Errors: {results['errors']}")
            
            if results["best_strategy"]:
                self.logger.info(f"Best strategy: {results['best_strategy']} with {results['best_return']:.2f}% return")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running backtests: {e}")
            results["errors"] += 1
            results["duration"] = time.time() - start_time
            results["success"] = False
            return results
    
    async def _test_connection_manager(self) -> Dict[str, Any]:
        """
        Test connection manager.
        
        Returns:
            Dictionary with test results
        """
        self.logger.info("Testing connection manager")
        
        results = {
            "component": "ConnectionManager",
            "tests": [],
            "passed": 0,
            "failed": 0,
            "errors": 0
        }
        
        try:
            # Initialize connection manager
            connection_manager = EnhancedConnectionManager(self.logger)
            
            # Test reset_state
            try:
                connection_manager.reset_state()
                results["tests"].append({
                    "name": "reset_state",
                    "result": "passed"
                })
                results["passed"] += 1
            except Exception as e:
                results["tests"].append({
                    "name": "reset_state",
                    "result": "failed",
                    "error": str(e)
                })
                results["failed"] += 1
            
            # Test ensure_connection with mock functions
            try:
                # Mock functions
                connect_success = lambda: True
                connect_fail = lambda: False
                test_success = lambda: True
                test_fail = lambda: False
                
                # Test successful connection
                result = connection_manager.ensure_connection(connect_success, test_success)
                if result:
                    results["tests"].append({
                        "name": "ensure_connection_success",
                        "result": "passed"
                    })
                    results["passed"] += 1
                else:
                    results["tests"].append({
                        "name": "ensure_connection_success",
                        "result": "failed",
                        "error": "Connection should succeed but failed"
                    })
                    results["failed"] += 1
                
                # Test failed connection
                result = connection_manager.ensure_connection(connect_fail, test_fail)
                if not result:
                    results["tests"].append({
                        "name": "ensure_connection_fail",
                        "result": "passed"
                    })
                    results["passed"] += 1
                else:
                    results["tests"].append({
                        "name": "ensure_connection_fail",
                        "result": "failed",
                        "error": "Connection should fail but succeeded"
                    })
                    results["failed"] += 1
                
            except Exception as e:
                results["tests"].append({
                    "name": "ensure_connection",
                    "result": "error",
                    "error": str(e)
                })
                results["errors"] += 1
            
            # Test safe_api_call with mock functions
            try:
                # Mock functions
                api_success = lambda: {"success": True}
                api_fail = lambda: {"error": "API error"}
                
                # Test successful API call
                result = connection_manager.safe_api_call(api_success)
                if result.get("success"):
                    results["tests"].append({
                        "name": "safe_api_call_success",
                        "result": "passed"
                    })
                    results["passed"] += 1
                else:
                    results["tests"].append({
                        "name": "safe_api_call_success",
                        "result": "failed",
                        "error": "API call should succeed but failed"
                    })
                    results["failed"] += 1
                
                # Test failed API call
                result = connection_manager.safe_api_call(lambda: (_ for _ in ()).throw(Exception("Test error")))
                if "error" in result:
                    results["tests"].append({
                        "name": "safe_api_call_fail",
                        "result": "passed"
                    })
                    results["passed"] += 1
                else:
                    results["tests"].append({
                        "name": "safe_api_call_fail",
                        "result": "failed",
                        "error": "API call should fail but succeeded"
                    })
                    results["failed"] += 1
                
            except Exception as e:
                results["tests"].append({
                    "name": "safe_api_call",
                    "result": "error",
                    "error": str(e)
                })
                results["errors"] += 1
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error testing connection manager: {e}")
            results["errors"] += 1
            return results
    
    async def _test_settings_manager(self) -> Dict[str, Any]:
        """
        Test settings manager.
        
        Returns:
            Dictionary with test results
        """
        self.logger.info("Testing settings manager")
        
        results = {
            "component": "SettingsManager",
            "tests": [],
            "passed": 0,
            "failed": 0,
            "errors": 0
        }
        
        try:
            # Create temporary config file
            temp_config_path = os.path.join(self.output_dir, "temp_config.json")
            
            # Initialize settings manager
            settings_manager = SettingsManager(temp_config_path, self.logger)
            
            # Test get_settings
            try:
                settings = settings_manager.get_settings()
                if settings:
                    results["tests"].append({
                        "name": "get_settings",
                        "result": "passed"
                    })
                    results["passed"] += 1
                else:
                    results["tests"].append({
                        "name": "get_settings",
                        "result": "failed",
                        "error": "Failed to get settings"
                    })
                    results["failed"] += 1
            except Exception as e:
                results["tests"].append({
                    "name": "get_settings",
                    "result": "error",
                    "error": str(e)
                })
                results["errors"] += 1
            
            # Test update_settings
            try:
                new_settings = {
                    "test_setting": "test_value"
                }
                result = settings_manager.update_settings(new_settings)
                if result:
                    # Verify setting was updated
                    settings = settings_manager.get_settings()
                    if settings.get("test_setting") == "test_value":
                        results["tests"].append({
                            "name": "update_settings",
                            "result": "passed"
                        })
                        results["passed"] += 1
                    else:
                        results["tests"].append({
                            "name": "update_settings",
                            "result": "failed",
                            "error": "Setting was not updated correctly"
                        })
                        results["failed"] += 1
                else:
                    results["tests"].append({
                        "name": "update_settings",
                        "result": "failed",
                        "error": "Failed to update settings"
                    })
                    results["failed"] += 1
            except Exception as e:
                results["tests"].append({
                    "name": "update_settings",
                    "result": "error",
                    "error": str(e)
                })
                results["errors"] += 1
            
            # Test create_backup
            try:
                result = settings_manager.create_backup()
                if result:
                    results["tests"].append({
                        "name": "create_backup",
                        "result": "passed"
                    })
                    results["passed"] += 1
                else:
                    results["tests"].append({
                        "name": "create_backup",
                        "result": "failed",
                        "error": "Failed to create backup"
                    })
                    results["failed"] += 1
            except Exception as e:
                results["tests"].append({
                    "name": "create_backup",
                    "result": "error",
                    "error": str(e)
                })
                results["errors"] += 1
            
            # Test get_latest_backup
            try:
                backup_path = settings_manager.get_latest_backup()
                if backup_path:
                    results["tests"].append({
                        "name": "get_latest_backup",
                        "result": "passed"
                    })
                    results["passed"] += 1
                else:
                    results["tests"].append({
                        "name": "get_latest_backup",
                        "result": "failed",
                        "error": "Failed to get latest backup"
                    })
                    results["failed"] += 1
            except Exception as e:
                results["tests"].append({
                    "name": "get_latest_backup",
                    "result": "error",
                    "error": str(e)
                })
                results["errors"] += 1
            
            # Clean up
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error testing settings manager: {e}")
            results["errors"] += 1
            return results
    
    async def _test_cache_manager(self) -> Dict[str, Any]:
        """
        Test cache manager.
        
        Returns:
            Dictionary with test results
        """
        self.logger.info("Testing cache manager")
        
        results = {
            "component": "CacheManager",
            "tests": [],
            "passed": 0,
            "failed": 0,
            "errors": 0
        }
        
        try:
            # Initialize cache manager
            from core.advanced_cache import CacheManager
            cache_manager = CacheManager(self.logger)
            
            # Test get_cache
            try:
                cache = cache_manager.get_cache("test_cache")
                if cache:
                    results["tests"].append({
                        "name": "get_cache",
                        "result": "passed"
                    })
                    results["passed"] += 1
                else:
                    results["tests"].append({
                        "name": "get_cache",
                        "result": "failed",
                        "error": "Failed to get cache"
                    })
                    results["failed"] += 1
            except Exception as e:
                results["tests"].append({
                    "name": "get_cache",
                    "result": "error",
                    "error": str(e)
                })
                results["errors"] += 1
            
            # Test cache set and get
            try:
                cache = cache_manager.get_cache("test_cache")
                
                # Set value
                cache.set("test_key", "test_value")
                
                # Get value
                value = cache.get("test_key")
                
                if value == "test_value":
                    results["tests"].append({
                        "name": "cache_set_get",
                        "result": "passed"
                    })
                    results["passed"] += 1
                else:
                    results["tests"].append({
                        "name": "cache_set_get",
                        "result": "failed",
                        "error": f"Expected 'test_value', got '{value}'"
                    })
                    results["failed"] += 1
            except Exception as e:
                results["tests"].append({
                    "name": "cache_set_get",
                    "result": "error",
                    "error": str(e)
                })
                results["errors"] += 1
            
            # Test cache get_or_set
            try:
                cache = cache_manager.get_cache("test_cache")
                
                # Get or set with function
                value = cache.get_or_set("test_key2", lambda: "test_value2")
                
                if value == "test_value2":
                    # Verify it's cached
                    value2 = cache.get("test_key2")
                    if value2 == "test_value2":
                        results["tests"].append({
                            "name": "cache_get_or_set",
                            "result": "passed"
                        })
                        results["passed"] += 1
                    else:
                        results["tests"].append({
                            "name": "cache_get_or_set",
                            "result": "failed",
                            "error": "Value was not cached"
                        })
                        results["failed"] += 1
                else:
                    results["tests"].append({
                        "name": "cache_get_or_set",
                        "result": "failed",
                        "error": f"Expected 'test_value2', got '{value}'"
                    })
                    results["failed"] += 1
            except Exception as e:
                results["tests"].append({
                    "name": "cache_get_or_set",
                    "result": "error",
                    "error": str(e)
                })
                results["errors"] += 1
            
            # Test cache delete
            try:
                cache = cache_manager.get_cache("test_cache")
                
                # Set value
                cache.set("test_key3", "test_value3")
                
                # Delete value
                result = cache.delete("test_key3")
                
                if result:
                    # Verify it's deleted
                    value = cache.get("test_key3")
                    if value is None:
                        results["tests"].append({
                            "name": "cache_delete",
                            "result": "passed"
                        })
                        results["passed"] += 1
                    else:
                        results["tests"].append({
                            "name": "cache_delete",
                            "result": "failed",
                            "error": "Value was not deleted"
                        })
                        results["failed"] += 1
                else:
                    results["tests"].append({
                        "name": "cache_delete",
                        "result": "failed",
                        "error": "Delete returned False"
                    })
                    results["failed"] += 1
            except Exception as e:
                results["tests"].append({
                    "name": "cache_delete",
                    "result": "error",
                    "error": str(e)
                })
                results["errors"] += 1
            
            # Test cache clear
            try:
                cache = cache_manager.get_cache("test_cache")
                
                # Set values
                cache.set("test_key4", "test_value4")
                cache.set("test_key5", "test_value5")
                
                # Clear cache
                cache.clear()
                
                # Verify it's cleared
                value1 = cache.get("test_key4")
                value2 = cache.get("test_key5")
                
                if value1 is None and value2 is None:
                    results["tests"].append({
                        "name": "cache_clear",
                        "result": "passed"
                    })
                    results["passed"] += 1
                else:
                    results["tests"].append({
                        "name": "cache_clear",
                        "result": "failed",
                        "error": "Cache was not cleared"
                    })
                    results["failed"] += 1
            except Exception as e:
                results["tests"].append({
                    "name": "cache_clear",
                    "result": "error",
                    "error": str(e)
                })
                results["errors"] += 1
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error testing cache manager: {e}")
            results["errors"] += 1
            return results
    
    async def _test_sentiment_analyzer(self) -> Dict[str, Any]:
        """
        Test sentiment analyzer.
        
        Returns:
            Dictionary with test results
        """
        self.logger.info("Testing sentiment analyzer")
        
        results = {
            "component": "SentimentAnalyzer",
            "tests": [],
            "passed": 0,
            "failed": 0,
            "errors": 0
        }
        
        try:
            # Initialize sentiment analyzer
            sentiment_analyzer = SentimentAnalyzer(self.config, self.logger)
            
            # Test initialization
            try:
                await sentiment_analyzer.initialize()
                results["tests"].append({
                    "name": "initialize",
                    "result": "passed"
                })
                results["passed"] += 1
            except Exception as e:
                results["tests"].append({
                    "name": "initialize",
                    "result": "error",
                    "error": str(e)
                })
                results["errors"] += 1
            
            # Test analyze_sentiment
            try:
                sentiment = await sentiment_analyzer.analyze_sentiment("BTC", 24)
                
                if sentiment and "sentiment_score" in sentiment:
                    results["tests"].append({
                        "name": "analyze_sentiment",
                        "result": "passed",
                        "sentiment_score": sentiment["sentiment_score"],
                        "sentiment_label": sentiment["sentiment_label"]
                    })
                    results["passed"] += 1
                else:
                    results["tests"].append({
                        "name": "analyze_sentiment",
                        "result": "failed",
                        "error": "Failed to analyze sentiment"
                    })
                    results["failed"] += 1
            except Exception as e:
                results["tests"].append({
                    "name": "analyze_sentiment",
                    "result": "error",
                    "error": str(e)
                })
                results["errors"] += 1
            
            # Test sentiment caching
            try:
                # First call should have cached the result
                start_time = time.time()
                sentiment = await sentiment_analyzer.analyze_sentiment("BTC", 24)
                duration = time.time() - start_time
                
                if duration < 0.1:  # Should be very fast if cached
                    results["tests"].append({
                        "name": "sentiment_caching",
                        "result": "passed",
                        "duration": duration
                    })
                    results["passed"] += 1
                else:
                    results["tests"].append({
                        "name": "sentiment_caching",
                        "result": "failed",
                        "error": f"Cache lookup took too long: {duration:.2f}s"
                    })
                    results["failed"] += 1
            except Exception as e:
                results["tests"].append({
                    "name": "sentiment_caching",
                    "result": "error",
                    "error": str(e)
                })
                results["errors"] += 1
            
            # Clean up
            await sentiment_analyzer.close()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error testing sentiment analyzer: {e}")
            results["errors"] += 1
            return results
    
    async def _test_strategies(self) -> Dict[str, Any]:
        """
        Test strategy components.
        
        Returns:
            Dictionary with test results
        """
        self.logger.info("Testing strategy components")
        
        results = {
            "component": "Strategies",
            "tests": [],
            "passed": 0,
            "failed": 0,
            "errors": 0
        }
        
        try:
            # Test MasterOmniOverlordStrategy
            try:
                strategy = MasterOmniOverlordStrategy(self.config, self.logger)
                
                # Test initialization
                if strategy:
                    results["tests"].append({
                        "name": "master_omni_overlord_init",
                        "result": "passed"
                    })
                    results["passed"] += 1
                else:
                    results["tests"].append({
                        "name": "master_omni_overlord_init",
                        "result": "failed",
                        "error": "Failed to initialize strategy"
                    })
                    results["failed"] += 1
                
                # Test detect_market_regime with mock data
                df = pd.DataFrame({
                    "close": [100 + i for i in range(100)],
                    "high": [105 + i for i in range(100)],
                    "low": [95 + i for i in range(100)]
                })
                
                regime = strategy.detect_market_regime(df)
                
                if regime in ["trending", "ranging", "volatile", "unknown"]:
                    results["tests"].append({
                        "name": "detect_market_regime",
                        "result": "passed",
                        "regime": regime
                    })
                    results["passed"] += 1
                else:
                    results["tests"].append({
                        "name": "detect_market_regime",
                        "result": "failed",
                        "error": f"Invalid regime: {regime}"
                    })
                    results["failed"] += 1
                
            except Exception as e:
                results["tests"].append({
                    "name": "master_omni_overlord",
                    "result": "error",
                    "error": str(e)
                })
                results["errors"] += 1
            
            # Test TripleConfluenceStrategy
            try:
                strategy = TripleConfluenceStrategy(self.config, self.logger)
                
                # Test initialization
                if strategy:
                    results["tests"].append({
                        "name": "triple_confluence_init",
                        "result": "passed"
                    })
                    results["passed"] += 1
                else:
                    results["tests"].append({
                        "name": "triple_confluence_init",
                        "result": "failed",
                        "error": "Failed to initialize strategy"
                    })
                    results["failed"] += 1
                
            except Exception as e:
                results["tests"].append({
                    "name": "triple_confluence",
                    "result": "error",
                    "error": str(e)
                })
                results["errors"] += 1
            
            # Test OracleUpdateStrategy
            try:
                strategy = OracleUpdateStrategy(self.config, self.logger)
                
                # Test initialization
                if strategy:
                    results["tests"].append({
                        "name": "oracle_update_init",
                        "result": "passed"
                    })
                    results["passed"] += 1
                else:
                    results["tests"].append({
                        "name": "oracle_update_init",
                        "result": "failed",
                        "error": "Failed to initialize strategy"
                    })
                    results["failed"] += 1
                
            except Exception as e:
                results["tests"].append({
                    "name": "oracle_update",
                    "result": "error",
                    "error": str(e)
                })
                results["errors"] += 1
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error testing strategies: {e}")
            results["errors"] += 1
            return results
    
    async def _test_adapter_with_connection_manager(self) -> Dict[str, Any]:
        """
        Test adapter with connection manager.
        
        Returns:
            Dictionary with test results
        """
        self.logger.info("Testing adapter with connection manager")
        
        results = {
            "component": "AdapterWithConnectionManager",
            "tests": [],
            "passed": 0,
            "failed": 0,
            "errors": 0
        }
        
        try:
            # Initialize adapter
            adapter = HyperliquidAdapter(self.config_path)
            
            # Test is_connected attribute
            try:
                if hasattr(adapter, 'is_connected'):
                    results["tests"].append({
                        "name": "is_connected_attribute",
                        "result": "passed",
                        "value": adapter.is_connected
                    })
                    results["passed"] += 1
                else:
                    results["tests"].append({
                        "name": "is_connected_attribute",
                        "result": "failed",
                        "error": "is_connected attribute not found"
                    })
                    results["failed"] += 1
            except Exception as e:
                results["tests"].append({
                    "name": "is_connected_attribute",
                    "result": "error",
                    "error": str(e)
                })
                results["errors"] += 1
            
            # Test connection manager
            try:
                if hasattr(adapter, 'connection_manager'):
                    results["tests"].append({
                        "name": "connection_manager_attribute",
                        "result": "passed"
                    })
                    results["passed"] += 1
                else:
                    results["tests"].append({
                        "name": "connection_manager_attribute",
                        "result": "failed",
                        "error": "connection_manager attribute not found"
                    })
                    results["failed"] += 1
            except Exception as e:
                results["tests"].append({
                    "name": "connection_manager_attribute",
                    "result": "error",
                    "error": str(e)
                })
                results["errors"] += 1
            
            # Test test_connection method
            try:
                result = adapter.test_connection()
                
                results["tests"].append({
                    "name": "test_connection",
                    "result": "passed",
                    "connection_result": result
                })
                results["passed"] += 1
            except Exception as e:
                results["tests"].append({
                    "name": "test_connection",
                    "result": "error",
                    "error": str(e)
                })
                results["errors"] += 1
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error testing adapter with connection manager: {e}")
            results["errors"] += 1
            return results
    
    async def _test_strategy_with_sentiment(self) -> Dict[str, Any]:
        """
        Test strategy with sentiment analyzer.
        
        Returns:
            Dictionary with test results
        """
        self.logger.info("Testing strategy with sentiment analyzer")
        
        results = {
            "component": "StrategyWithSentiment",
            "tests": [],
            "passed": 0,
            "failed": 0,
            "errors": 0
        }
        
        try:
            # Initialize sentiment analyzer
            sentiment_analyzer = SentimentAnalyzer(self.config, self.logger)
            await sentiment_analyzer.initialize()
            
            # Initialize strategy
            strategy = MasterOmniOverlordStrategy(self.config, self.logger)
            
            # Get sentiment for BTC
            sentiment = await sentiment_analyzer.analyze_sentiment("BTC", 24)
            
            # Test if sentiment affects strategy
            try:
                # Create mock market data with sentiment
                market_data = {
                    "price": 50000.0,
                    "sentiment": sentiment
                }
                
                # Adjust strategy based on sentiment
                if hasattr(strategy, 'adjust_strategy_weights'):
                    # Store original weights
                    original_weights = strategy.strategy_weights.copy()
                    
                    # Modify market state based on sentiment
                    strategy.market_state["sentiment"] = sentiment["sentiment_label"]
                    
                    # Adjust weights
                    strategy.adjust_strategy_weights()
                    
                    # Check if weights changed
                    weights_changed = original_weights != strategy.strategy_weights
                    
                    if weights_changed:
                        results["tests"].append({
                            "name": "sentiment_affects_strategy",
                            "result": "passed",
                            "original_weights": original_weights,
                            "new_weights": strategy.strategy_weights
                        })
                        results["passed"] += 1
                    else:
                        results["tests"].append({
                            "name": "sentiment_affects_strategy",
                            "result": "failed",
                            "error": "Sentiment did not affect strategy weights"
                        })
                        results["failed"] += 1
                else:
                    results["tests"].append({
                        "name": "sentiment_affects_strategy",
                        "result": "failed",
                        "error": "Strategy does not have adjust_strategy_weights method"
                    })
                    results["failed"] += 1
            except Exception as e:
                results["tests"].append({
                    "name": "sentiment_affects_strategy",
                    "result": "error",
                    "error": str(e)
                })
                results["errors"] += 1
            
            # Clean up
            await sentiment_analyzer.close()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error testing strategy with sentiment: {e}")
            results["errors"] += 1
            return results
    
    async def _test_trading_pipeline(self) -> Dict[str, Any]:
        """
        Test full trading pipeline.
        
        Returns:
            Dictionary with test results
        """
        self.logger.info("Testing full trading pipeline")
        
        results = {
            "component": "TradingPipeline",
            "tests": [],
            "passed": 0,
            "failed": 0,
            "errors": 0
        }
        
        try:
            # This is a simplified test of the trading pipeline
            # In a real test, we would initialize the full trading bot
            # and run it through a complete cycle
            
            # For now, we'll just test that the key components are available
            
            # Test HyperliquidAdapter
            try:
                adapter = HyperliquidAdapter(self.config_path)
                if adapter:
                    results["tests"].append({
                        "name": "adapter_init",
                        "result": "passed"
                    })
                    results["passed"] += 1
                else:
                    results["tests"].append({
                        "name": "adapter_init",
                        "result": "failed",
                        "error": "Failed to initialize adapter"
                    })
                    results["failed"] += 1
            except Exception as e:
                results["tests"].append({
                    "name": "adapter_init",
                    "result": "error",
                    "error": str(e)
                })
                results["errors"] += 1
            
            # Test MasterOmniOverlordStrategy
            try:
                strategy = MasterOmniOverlordStrategy(self.config, self.logger)
                if strategy:
                    results["tests"].append({
                        "name": "strategy_init",
                        "result": "passed"
                    })
                    results["passed"] += 1
                else:
                    results["tests"].append({
                        "name": "strategy_init",
                        "result": "failed",
                        "error": "Failed to initialize strategy"
                    })
                    results["failed"] += 1
            except Exception as e:
                results["tests"].append({
                    "name": "strategy_init",
                    "result": "error",
                    "error": str(e)
                })
                results["errors"] += 1
            
            # Test SentimentAnalyzer
            try:
                sentiment_analyzer = SentimentAnalyzer(self.config, self.logger)
                await sentiment_analyzer.initialize()
                if sentiment_analyzer:
                    results["tests"].append({
                        "name": "sentiment_analyzer_init",
                        "result": "passed"
                    })
                    results["passed"] += 1
                else:
                    results["tests"].append({
                        "name": "sentiment_analyzer_init",
                        "result": "failed",
                        "error": "Failed to initialize sentiment analyzer"
                    })
                    results["failed"] += 1
                await sentiment_analyzer.close()
            except Exception as e:
                results["tests"].append({
                    "name": "sentiment_analyzer_init",
                    "result": "error",
                    "error": str(e)
                })
                results["errors"] += 1
            
            # Test CacheManager
            try:
                cache_manager = CacheManager(self.logger)
                if cache_manager:
                    results["tests"].append({
                        "name": "cache_manager_init",
                        "result": "passed"
                    })
                    results["passed"] += 1
                else:
                    results["tests"].append({
                        "name": "cache_manager_init",
                        "result": "failed",
                        "error": "Failed to initialize cache manager"
                    })
                    results["failed"] += 1
            except Exception as e:
                results["tests"].append({
                    "name": "cache_manager_init",
                    "result": "error",
                    "error": str(e)
                })
                results["errors"] += 1
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error testing trading pipeline: {e}")
            results["errors"] += 1
            return results
    
    def _master_omni_overlord_strategy(self, data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Master Omni Overlord strategy for backtesting.
        
        Args:
            data: Dictionary of DataFrames with historical data for each symbol
            params: Strategy parameters
            
        Returns:
            Dictionary of signals for each symbol
        """
        signals = {}
        
        for symbol, df in data.items():
            if df.empty:
                continue
                
            # Get latest data point
            latest = df.iloc[-1]
            
            # Get parameters
            signal_threshold = params.get("signal_threshold", 0.7)
            trend_filter_strength = params.get("trend_filter_strength", 0.5)
            mean_reversion_factor = params.get("mean_reversion_factor", 0.3)
            volatility_adjustment = params.get("volatility_adjustment", 1.0)
            
            # Calculate simple moving averages
            df["sma_20"] = df["close"].rolling(20).mean()
            df["sma_50"] = df["close"].rolling(50).mean()
            df["sma_200"] = df["close"].rolling(200).mean()
            
            # Calculate RSI
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            df["rsi"] = 100 - (100 / (1 + rs))
            
            # Get latest indicators
            latest_close = latest["close"]
            latest_sma_20 = df["sma_20"].iloc[-1]
            latest_sma_50 = df["sma_50"].iloc[-1]
            latest_sma_200 = df["sma_200"].iloc[-1]
            latest_rsi = df["rsi"].iloc[-1]
            
            # Check if we have enough data
            if pd.isna(latest_sma_200) or pd.isna(latest_rsi):
                continue
                
            # Calculate trend signal
            trend_signal = 0
            
            # Short-term trend
            if latest_sma_20 > latest_sma_50:
                trend_signal += 0.5
            else:
                trend_signal -= 0.5
                
            # Long-term trend
            if latest_sma_50 > latest_sma_200:
                trend_signal += 0.5
            else:
                trend_signal -= 0.5
                
            # Apply trend filter strength
            trend_signal *= trend_filter_strength
            
            # Calculate mean reversion signal
            mean_reversion_signal = 0
            
            # RSI mean reversion
            if latest_rsi > 70:
                mean_reversion_signal -= 1
            elif latest_rsi < 30:
                mean_reversion_signal += 1
                
            # Apply mean reversion factor
            mean_reversion_signal *= mean_reversion_factor
            
            # Calculate final signal
            final_signal = trend_signal + mean_reversion_signal
            
            # Apply volatility adjustment
            final_signal *= volatility_adjustment
            
            # Determine action
            if final_signal > signal_threshold:
                action = "buy"
                size = 1.0  # Full position
            elif final_signal < -signal_threshold:
                action = "sell"
                size = 1.0  # Full position
            else:
                action = "hold"
                size = 0
                
            # Create signal
            signals[symbol] = {
                "action": action,
                "size": size,
                "price": latest_close,
                "signal_strength": final_signal,
                "trend_signal": trend_signal,
                "mean_reversion_signal": mean_reversion_signal
            }
            
        return signals
    
    def _triple_confluence_strategy(self, data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Triple Confluence strategy for backtesting.
        
        Args:
            data: Dictionary of DataFrames with historical data for each symbol
            params: Strategy parameters
            
        Returns:
            Dictionary of signals for each symbol
        """
        signals = {}
        
        for symbol, df in data.items():
            if df.empty:
                continue
                
            # Get latest data point
            latest = df.iloc[-1]
            
            # Get parameters
            ma_short = params.get("ma_short", 20)
            ma_medium = params.get("ma_medium", 50)
            ma_long = params.get("ma_long", 200)
            rsi_period = params.get("rsi_period", 14)
            rsi_overbought = params.get("rsi_overbought", 70)
            rsi_oversold = params.get("rsi_oversold", 30)
            
            # Calculate moving averages
            df[f"ma_short"] = df["close"].rolling(ma_short).mean()
            df[f"ma_medium"] = df["close"].rolling(ma_medium).mean()
            df[f"ma_long"] = df["close"].rolling(ma_long).mean()
            
            # Calculate RSI
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(rsi_period).mean()
            avg_loss = loss.rolling(rsi_period).mean()
            rs = avg_gain / avg_loss
            df["rsi"] = 100 - (100 / (1 + rs))
            
            # Get latest indicators
            latest_close = latest["close"]
            latest_ma_short = df[f"ma_short"].iloc[-1]
            latest_ma_medium = df[f"ma_medium"].iloc[-1]
            latest_ma_long = df[f"ma_long"].iloc[-1]
            latest_rsi = df["rsi"].iloc[-1]
            
            # Check if we have enough data
            if pd.isna(latest_ma_long) or pd.isna(latest_rsi):
                continue
                
            # Check for triple confluence
            buy_signals = 0
            sell_signals = 0
            
            # Moving average alignment
            if latest_ma_short > latest_ma_medium > latest_ma_long:
                buy_signals += 1
            elif latest_ma_short < latest_ma_medium < latest_ma_long:
                sell_signals += 1
                
            # Price relative to moving averages
            if latest_close > latest_ma_short:
                buy_signals += 1
            elif latest_close < latest_ma_short:
                sell_signals += 1
                
            # RSI
            if latest_rsi < rsi_oversold:
                buy_signals += 1
            elif latest_rsi > rsi_overbought:
                sell_signals += 1
                
            # Determine action
            if buy_signals >= 2:
                action = "buy"
                size = 1.0  # Full position
            elif sell_signals >= 2:
                action = "sell"
                size = 1.0  # Full position
            else:
                action = "hold"
                size = 0
                
            # Create signal
            signals[symbol] = {
                "action": action,
                "size": size,
                "price": latest_close,
                "buy_signals": buy_signals,
                "sell_signals": sell_signals
            }
            
        return signals
    
    def _oracle_update_strategy(self, data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Oracle Update strategy for backtesting.
        
        Args:
            data: Dictionary of DataFrames with historical data for each symbol
            params: Strategy parameters
            
        Returns:
            Dictionary of signals for each symbol
        """
        signals = {}
        
        for symbol, df in data.items():
            if df.empty:
                continue
                
            # Get latest data point
            latest = df.iloc[-1]
            
            # Get parameters
            update_threshold = params.get("update_threshold", 0.5)
            momentum_factor = params.get("momentum_factor", 0.7)
            reversal_threshold = params.get("reversal_threshold", 0.3)
            
            # Calculate price changes
            df["pct_change"] = df["close"].pct_change()
            df["pct_change_5"] = df["close"].pct_change(5)
            df["pct_change_20"] = df["close"].pct_change(20)
            
            # Calculate momentum
            df["momentum"] = df["pct_change_5"] * momentum_factor
            
            # Get latest indicators
            latest_close = latest["close"]
            latest_pct_change = df["pct_change"].iloc[-1]
            latest_pct_change_5 = df["pct_change_5"].iloc[-1]
            latest_pct_change_20 = df["pct_change_20"].iloc[-1]
            latest_momentum = df["momentum"].iloc[-1]
            
            # Check if we have enough data
            if pd.isna(latest_pct_change_20) or pd.isna(latest_momentum):
                continue
                
            # Calculate oracle signal
            oracle_signal = 0
            
            # Recent price change
            if latest_pct_change > 0:
                oracle_signal += 0.3
            else:
                oracle_signal -= 0.3
                
            # Medium-term price change
            if latest_pct_change_5 > 0:
                oracle_signal += 0.3
            else:
                oracle_signal -= 0.3
                
            # Long-term price change
            if latest_pct_change_20 > 0:
                oracle_signal += 0.4
            else:
                oracle_signal -= 0.4
                
            # Apply momentum
            oracle_signal += latest_momentum
            
            # Check for reversal
            if latest_pct_change_5 > 0 and latest_pct_change < 0 and abs(latest_pct_change) > reversal_threshold:
                oracle_signal -= 0.5  # Potential reversal from up to down
            elif latest_pct_change_5 < 0 and latest_pct_change > 0 and abs(latest_pct_change) > reversal_threshold:
                oracle_signal += 0.5  # Potential reversal from down to up
                
            # Determine action
            if oracle_signal > update_threshold:
                action = "buy"
                size = 1.0  # Full position
            elif oracle_signal < -update_threshold:
                action = "sell"
                size = 1.0  # Full position
            else:
                action = "hold"
                size = 0
                
            # Create signal
            signals[symbol] = {
                "action": action,
                "size": size,
                "price": latest_close,
                "oracle_signal": oracle_signal,
                "momentum": latest_momentum
            }
            
        return signals
    
    def save_test_results(self, filename: str = None) -> str:
        """
        Save test results to file.
        
        Args:
            filename: Optional filename to save results
            
        Returns:
            Path to saved file
        """
        if not self.test_results:
            self.logger.warning("No test results to save")
            return None
            
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"test_results_{timestamp}.json")
            
        # Save to file
        with open(filename, "w") as f:
            import json
            json.dump(self.test_results, f, indent=4, default=str)
            
        self.logger.info(f"Test results saved to {filename}")
        return filename
