#!/usr/bin/env python3
"""
Test script to verify that backtesting is using real data.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("test_real_data_backtesting.log")
    ]
)
logger = logging.getLogger(__name__)

def test_backtesting_data(symbol: str, interval: str, backtest_dir: str = "backtest_data", real_data_dir: str = "real_market_data/processed"):
    """
    Test that backtesting is using real data.
    
    Args:
        symbol: Symbol to test
        interval: Interval to test
        backtest_dir: Directory used by backtesting framework
        real_data_dir: Directory with real market data
        
    Returns:
        True if using real data, False otherwise
    """
    logger.info(f"Testing backtesting data for {symbol}, interval {interval}")
    
    # Check if real data exists
    real_data_path = f"{real_data_dir}/{symbol}_{interval}_processed.csv"
    
    if not os.path.exists(real_data_path):
        logger.error(f"Real data not found for {symbol}, interval {interval} at {real_data_path}")
        return False
        
    # Check if backtest data exists
    backtest_path = f"{backtest_dir}/{symbol}_{interval}.csv"
    
    if not os.path.exists(backtest_path):
        logger.error(f"Backtest data not found for {symbol}, interval {interval} at {backtest_path}")
        return False
        
    # Load real data
    df_real = pd.read_csv(real_data_path)
    logger.info(f"Loaded {df_real.shape[0]} rows of real data for {symbol}, interval {interval}")
    
    # Load backtest data
    df_backtest = pd.read_csv(backtest_path)
    logger.info(f"Loaded {df_backtest.shape[0]} rows of backtest data for {symbol}, interval {interval}")
    
    # Check if row counts match
    if df_real.shape[0] != df_backtest.shape[0]:
        logger.warning(f"Row count mismatch for {symbol}, interval {interval}: real={df_real.shape[0]}, backtest={df_backtest.shape[0]}")
        
    # Check if data is the same
    if "close" in df_real.columns and "close" in df_backtest.columns:
        if not np.array_equal(df_real["close"].values, df_backtest["close"].values):
            logger.warning(f"Data mismatch for {symbol}, interval {interval}")
            return False
            
    logger.info(f"Backtesting is using real data for {symbol}, interval {interval}")
    return True
    
def main():
    """
    Main function to test backtesting data.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test Real Data Backtesting")
    parser.add_argument("--symbols", type=str, default="SOL-USD-PERP,BTC-USD-PERP,ETH-USD-PERP", help="Comma-separated list of symbols")
    parser.add_argument("--intervals", type=str, default="1h,4h", help="Comma-separated list of intervals")
    parser.add_argument("--backtest_dir", type=str, default="backtest_data", help="Directory used by backtesting framework")
    parser.add_argument("--real_data_dir", type=str, default="real_market_data/processed", help="Directory with real market data")
    args = parser.parse_args()
    
    # Get symbols and intervals
    symbols = args.symbols.split(",")
    intervals = args.intervals.split(",")
    
    # Test backtesting data
    results = {}
    
    for symbol in symbols:
        results[symbol] = {}
        
        for interval in intervals:
            results[symbol][interval] = test_backtesting_data(symbol, interval, args.backtest_dir, args.real_data_dir)
            
    # Print results
    logger.info(f"Test results: {json.dumps(results, indent=4)}")
    
    # Check if all tests passed
    all_passed = True
    
    for symbol in results:
        for interval in results[symbol]:
            if not results[symbol][interval]:
                all_passed = False
                
    if all_passed:
        logger.info("All tests passed! Backtesting is using real data for all symbols and intervals.")
    else:
        logger.error("Some tests failed. Backtesting may not be using real data for all symbols and intervals.")
        
    return all_passed

if __name__ == "__main__":
    main()
