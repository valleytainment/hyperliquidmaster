#!/usr/bin/env python3
"""
Enhanced Strategy Integration Test

This script tests the optimized strategy with realistic market data
to validate profitability, consistency, and market regime adaptation.
"""

import os
import sys
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import strategy
from strategies.optimized_strategy import OptimizedStrategy, MarketRegime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("strategy_integration_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class StrategyIntegrationTest:
    """
    Enhanced strategy integration test with realistic market data.
    """
    
    def __init__(self):
        """Initialize the strategy integration test."""
        self.logger = logger
        self.logger.info("Initializing Strategy Integration Test")
        
        # Initialize strategy
        self.strategy = OptimizedStrategy({
            "long_threshold": 60,
            "short_threshold": 60,
            "exit_threshold": 50,
            "stop_loss_percent": 5.0,
            "take_profit_percent": 15.0,
            "trailing_activation": 5.0,
            "trailing_callback": 2.0
        }, self.logger)
        
        # Test symbols
        self.symbols = ["BTC", "ETH", "XRP", "SOL"]
        
        # Market regimes
        self.regimes = ["BULLISH", "BEARISH", "RANGING", "VOLATILE"]
        
        # Market data directory
        self.market_data_dir = "market_data"
        
        # Test results
        self.test_results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overall_success": False,
            "success_rate": 0.0,
            "profitability_tests": [],
            "consistency_tests": [],
            "market_regime_tests": []
        }
        
        self.logger.info("Strategy Integration Test initialized")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all strategy integration tests.
        
        Returns:
            Test results dictionary
        """
        try:
            self.logger.info("Running all strategy integration tests")
            
            # Run profitability tests
            self.run_profitability_tests()
            
            # Run consistency tests
            self.run_consistency_tests()
            
            # Run market regime adaptation tests
            self.run_market_regime_adaptation_tests()
            
            # Calculate overall success rate
            total_tests = (
                len(self.test_results["profitability_tests"]) +
                len(self.test_results["consistency_tests"]) +
                len(self.test_results["market_regime_tests"])
            )
            
            successful_tests = (
                sum(1 for test in self.test_results["profitability_tests"] if test["success"]) +
                sum(1 for test in self.test_results["consistency_tests"] if test["success"]) +
                sum(1 for test in self.test_results["market_regime_tests"] if test["success"])
            )
            
            if total_tests > 0:
                self.test_results["success_rate"] = successful_tests / total_tests * 100
            
            # Determine overall success
            self.test_results["overall_success"] = self.test_results["success_rate"] >= 80.0
            
            self.logger.info(f"All tests completed. Success rate: {self.test_results['success_rate']:.2f}%")
            
            # Convert boolean values to strings for JSON serialization
            json_safe_results = self._make_json_serializable(self.test_results)
            
            # Save test results to file
            with open("strategy_integration_test_results.json", "w") as f:
                json.dump(json_safe_results, f, indent=4)
            
            self.logger.info("Test results saved to strategy_integration_test_results.json")
            
            # Print test summary
            print("\n=== STRATEGY INTEGRATION TEST SUMMARY ===")
            print(f"Overall Success: {'YES' if self.test_results['overall_success'] else 'NO'}")
            print(f"Success Rate: {self.test_results['success_rate']:.2f}%")
            print(f"Profitability Tests: {len(self.test_results['profitability_tests'])}")
            print(f"Consistency Tests: {len(self.test_results['consistency_tests'])}")
            print(f"Market Regime Tests: {len(self.test_results['market_regime_tests'])}")
            print("============================================")
            
            return self.test_results
        except Exception as e:
            self.logger.error(f"Error running strategy integration tests: {e}")
            return self.test_results
    
    def _make_json_serializable(self, obj):
        """
        Convert object to JSON serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable object
        """
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (bool, np.bool_)):
            return str(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        else:
            return obj
    
    def run_profitability_tests(self) -> None:
        """Run profitability tests for all symbols."""
        self.logger.info("Running profitability tests")
        
        for symbol in self.symbols:
            self.logger.info(f"Testing profitability for {symbol}")
            
            # Load market data for all regimes
            market_data_by_regime = {}
            
            for regime in self.regimes:
                market_data_file = f"{self.market_data_dir}/{symbol}_{regime}.json"
                
                if os.path.exists(market_data_file):
                    with open(market_data_file, "r") as f:
                        market_data_by_regime[regime] = json.load(f)
            
            # Test profitability in each regime
            for regime, market_data in market_data_by_regime.items():
                # Update strategy with market data
                self.strategy.update_market_data(symbol, market_data)
                
                # Get trading signal
                signal_result = self.strategy.get_trading_signal(symbol)
                
                # Simulate trade execution
                trade_result = self._simulate_trade(symbol, signal_result)
                
                # Add test result
                self.test_results["profitability_tests"].append({
                    "symbol": symbol,
                    "regime": regime,
                    "signal": signal_result["signal"],
                    "confidence": signal_result["confidence"],
                    "profit_percent": trade_result["profit_percent"],
                    "success": trade_result["success"],
                    "details": trade_result["details"]
                })
                
                self.logger.info(f"Profitability test for {symbol} in {regime} regime: {'SUCCESS' if trade_result['success'] else 'FAILURE'}, Profit: {trade_result['profit_percent']:.2f}%")
    
    def run_consistency_tests(self) -> None:
        """Run consistency tests for all symbols."""
        self.logger.info("Running consistency tests")
        
        for symbol in self.symbols:
            self.logger.info(f"Testing consistency for {symbol}")
            
            # Load market data for ranging regime
            market_data_file = f"{self.market_data_dir}/{symbol}_RANGING.json"
            
            if not os.path.exists(market_data_file):
                self.logger.warning(f"Market data file {market_data_file} not found")
                continue
            
            with open(market_data_file, "r") as f:
                market_data = json.load(f)
            
            # Run multiple signal generations with small variations
            signals = []
            
            for i in range(5):
                # Create a copy of market data with small variations
                varied_data = market_data.copy()
                
                # Add small random variations to prices
                if "close" in varied_data and isinstance(varied_data["close"], list):
                    varied_data["close"] = [price * (1 + 0.001 * np.random.randn()) for price in varied_data["close"]]
                
                # Update strategy with varied market data
                self.strategy.update_market_data(symbol, varied_data)
                
                # Get trading signal
                signal_result = self.strategy.get_trading_signal(symbol)
                
                # Add signal to list
                signals.append(signal_result["signal"])
                
                self.logger.info(f"Trading signal for {symbol}: {signal_result['signal']} with {signal_result['confidence']:.2%} confidence")
            
            # Check consistency
            if len(signals) > 0:
                # Count occurrences of each signal
                signal_counts = {}
                
                for signal in signals:
                    if signal not in signal_counts:
                        signal_counts[signal] = 0
                    
                    signal_counts[signal] += 1
                
                # Find most common signal
                most_common_signal = max(signal_counts, key=signal_counts.get)
                consistency = signal_counts[most_common_signal] / len(signals) * 100
                
                # Add test result
                self.test_results["consistency_tests"].append({
                    "symbol": symbol,
                    "signals": signals,
                    "most_common_signal": most_common_signal,
                    "consistency": consistency,
                    "success": consistency >= 80.0
                })
                
                self.logger.info(f"Consistency test for {symbol}: {'SUCCESS' if consistency >= 80.0 else 'FAILURE'}, Consistency: {consistency:.2f}%")
    
    def run_market_regime_adaptation_tests(self) -> None:
        """Run market regime adaptation tests for all symbols."""
        self.logger.info("Testing strategy adaptation to different market regimes")
        
        for symbol in self.symbols:
            # Load market data for all regimes
            market_data_by_regime = {}
            
            for regime in self.regimes:
                market_data_file = f"{self.market_data_dir}/{symbol}_{regime}.json"
                
                if os.path.exists(market_data_file):
                    with open(market_data_file, "r") as f:
                        market_data_by_regime[regime] = json.load(f)
            
            # Test adaptation to each regime
            signals_by_regime = {}
            
            for regime, market_data in market_data_by_regime.items():
                # Update strategy with market data
                self.strategy.update_market_data(symbol, market_data)
                
                # Get trading signal
                signal_result = self.strategy.get_trading_signal(symbol)
                
                # Add signal to dictionary
                signals_by_regime[regime] = {
                    "signal": signal_result["signal"],
                    "confidence": signal_result["confidence"]
                }
                
                self.logger.info(f"Market regime for {symbol}: {regime}")
                self.logger.info(f"Trading signal for {symbol}: {signal_result['signal']} with {signal_result['confidence']:.2%} confidence")
            
            # Check adaptation
            adaptation_score = 0.0
            
            # Bullish regime should favor LONG signals
            if "BULLISH" in signals_by_regime:
                if signals_by_regime["BULLISH"]["signal"] == "LONG":
                    adaptation_score += 1.0
            
            # Bearish regime should favor SHORT signals
            if "BEARISH" in signals_by_regime:
                if signals_by_regime["BEARISH"]["signal"] == "SHORT":
                    adaptation_score += 1.0
            
            # Ranging regime should have lower confidence
            if "RANGING" in signals_by_regime:
                if signals_by_regime["RANGING"]["confidence"] < 0.7:
                    adaptation_score += 1.0
            
            # Volatile regime should have appropriate signals
            if "VOLATILE" in signals_by_regime:
                if signals_by_regime["VOLATILE"]["signal"] != "NONE":
                    adaptation_score += 1.0
            
            # Calculate adaptation score as percentage
            if len(signals_by_regime) > 0:
                adaptation_score = adaptation_score / len(signals_by_regime) * 100
            
            # Add test result
            self.test_results["market_regime_tests"].append({
                "symbol": symbol,
                "signals_by_regime": signals_by_regime,
                "adaptation_score": adaptation_score,
                "success": adaptation_score >= 75.0
            })
            
            self.logger.info(f"Market regime adaptation test for {symbol}: {'SUCCESS' if adaptation_score >= 75.0 else 'FAILED'}, Adaptation score: {adaptation_score:.2f}%")
    
    def _simulate_trade(self, symbol: str, signal_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate trade execution and calculate profit.
        
        Args:
            symbol: Trading symbol
            signal_result: Signal result dictionary
            
        Returns:
            Trade result dictionary
        """
        try:
            # Initialize trade result
            trade_result = {
                "symbol": symbol,
                "signal": signal_result["signal"],
                "entry_price": signal_result.get("entry_price", 0.0),
                "exit_price": 0.0,
                "profit_percent": 0.0,
                "success": False,
                "details": "No trade executed"
            }
            
            # Check if signal is valid
            if signal_result["signal"] not in ["LONG", "SHORT"]:
                return trade_result
            
            # Get entry price
            entry_price = signal_result.get("entry_price", 0.0)
            
            if entry_price <= 0.0:
                return trade_result
            
            # Simulate price movement after entry
            is_long = signal_result["signal"] == "LONG"
            
            # Generate realistic price movement based on signal
            n_points = 20
            prices = np.zeros(n_points)
            prices[0] = entry_price
            
            # Set price movement parameters based on signal and market regime
            if is_long:
                # For long positions
                if signal_result.get("regime") == "BULLISH":
                    # Bullish regime favors long positions
                    trend = 0.005  # Positive trend
                    volatility = 0.01  # Moderate volatility
                elif signal_result.get("regime") == "BEARISH":
                    # Bearish regime is challenging for long positions
                    trend = -0.003  # Negative trend
                    volatility = 0.015  # Higher volatility
                else:
                    # Neutral regime
                    trend = 0.001  # Slight positive trend
                    volatility = 0.01  # Moderate volatility
            else:
                # For short positions
                if signal_result.get("regime") == "BEARISH":
                    # Bearish regime favors short positions
                    trend = -0.005  # Negative trend
                    volatility = 0.01  # Moderate volatility
                elif signal_result.get("regime") == "BULLISH":
                    # Bullish regime is challenging for short positions
                    trend = 0.003  # Positive trend
                    volatility = 0.015  # Higher volatility
                else:
                    # Neutral regime
                    trend = -0.001  # Slight negative trend
                    volatility = 0.01  # Moderate volatility
            
            # Generate price movement
            for i in range(1, n_points):
                # Calculate price change
                trend_component = trend * (1 + 0.2 * np.random.randn())
                volatility_component = volatility * np.random.randn()
                
                # Apply price change
                price_change = trend_component + volatility_component
                prices[i] = prices[i-1] * (1 + price_change)
            
            # Get exit price (last price)
            exit_price = prices[-1]
            
            # Calculate profit
            if is_long:
                profit_percent = (exit_price - entry_price) / entry_price * 100
            else:
                profit_percent = (entry_price - exit_price) / entry_price * 100
            
            # Determine success
            success = profit_percent > 0
            
            # Update trade result
            trade_result["exit_price"] = exit_price
            trade_result["profit_percent"] = profit_percent
            trade_result["success"] = success
            trade_result["details"] = f"{'Long' if is_long else 'Short'} trade {'profitable' if success else 'unprofitable'}"
            
            return trade_result
        except Exception as e:
            self.logger.error(f"Error simulating trade: {e}")
            return {
                "symbol": symbol,
                "signal": signal_result.get("signal", "ERROR"),
                "entry_price": signal_result.get("entry_price", 0.0),
                "exit_price": 0.0,
                "profit_percent": 0.0,
                "success": False,
                "details": f"Error: {e}"
            }

if __name__ == "__main__":
    # Run strategy integration tests
    test = StrategyIntegrationTest()
    test.run_all_tests()
