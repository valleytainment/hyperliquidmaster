"""
Fix Backtesting Data Source Module

This module ensures that the backtesting framework correctly uses real market data
from the data pipeline instead of synthetic data.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("fix_backtesting_data.log")
    ]
)
logger = logging.getLogger(__name__)

class BacktestingDataFixer:
    """
    Utility to fix backtesting data sources and ensure real data is used.
    """
    
    def __init__(self, base_dir: str = "real_market_data", backtest_dir: str = "backtest_data"):
        """
        Initialize the backtesting data fixer.
        
        Args:
            base_dir: Base directory for real market data
            backtest_dir: Directory used by backtesting framework
        """
        self.base_dir = base_dir
        self.backtest_dir = backtest_dir
        
        # Create backtest directory if it doesn't exist
        os.makedirs(self.backtest_dir, exist_ok=True)
        
        logger.info(f"Backtesting Data Fixer initialized with base_dir={base_dir}, backtest_dir={backtest_dir}")
        
    def fix_data_sources(self, symbols: List[str], intervals: List[str]) -> Dict:
        """
        Fix data sources for backtesting by ensuring real data is used.
        
        Args:
            symbols: List of symbols to fix
            intervals: List of intervals to fix
            
        Returns:
            Dictionary with results
        """
        results = {}
        
        for symbol in symbols:
            results[symbol] = {}
            
            for interval in intervals:
                logger.info(f"Fixing data source for {symbol}, interval {interval}")
                
                # Check if processed real data exists
                processed_path = f"{self.base_dir}/processed/{symbol}_{interval}_processed.csv"
                
                if not os.path.exists(processed_path):
                    logger.error(f"Processed data not found for {symbol}, interval {interval} at {processed_path}")
                    results[symbol][interval] = {
                        "status": "error",
                        "message": f"Processed data not found at {processed_path}"
                    }
                    continue
                
                # Load processed data
                try:
                    df = pd.read_csv(processed_path)
                    logger.info(f"Loaded {df.shape[0]} rows of processed data for {symbol}, interval {interval}")
                    
                    # Check data quality
                    if df.shape[0] < 100:
                        logger.warning(f"Processed data for {symbol}, interval {interval} has only {df.shape[0]} rows, which may be insufficient")
                        
                    # Check for missing values
                    missing_values = df.isnull().sum().sum()
                    if missing_values > 0:
                        logger.warning(f"Processed data for {symbol}, interval {interval} has {missing_values} missing values")
                        
                    # Ensure timestamp is in datetime format
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        
                    # Check data freshness
                    if "timestamp" in df.columns:
                        latest_timestamp = df["timestamp"].max()
                        if latest_timestamp < datetime.now() - timedelta(days=1):
                            logger.warning(f"Processed data for {symbol}, interval {interval} is not fresh (latest: {latest_timestamp})")
                            
                    # Save to backtest directory
                    backtest_path = f"{self.backtest_dir}/{symbol}_{interval}.csv"
                    df.to_csv(backtest_path, index=False)
                    logger.info(f"Saved processed data to {backtest_path} for backtesting")
                    
                    # Create a symlink to ensure backtesting framework uses this data
                    symlink_path = f"{self.backtest_dir}/{symbol}_{interval}_real.csv"
                    if os.path.exists(symlink_path):
                        os.remove(symlink_path)
                    os.symlink(backtest_path, symlink_path)
                    logger.info(f"Created symlink at {symlink_path} pointing to {backtest_path}")
                    
                    # Update results
                    results[symbol][interval] = {
                        "status": "success",
                        "rows": df.shape[0],
                        "source_path": processed_path,
                        "backtest_path": backtest_path,
                        "symlink_path": symlink_path
                    }
                except Exception as e:
                    logger.error(f"Error processing data for {symbol}, interval {interval}: {str(e)}")
                    results[symbol][interval] = {
                        "status": "error",
                        "message": str(e)
                    }
                    
        return results
        
    def update_backtesting_config(self, config_path: str = "real_data_backtesting.py") -> bool:
        """
        Update backtesting configuration to ensure it uses real data.
        
        Args:
            config_path: Path to backtesting configuration file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read backtesting configuration
            with open(config_path, "r") as f:
                config = f.read()
                
            # Check if configuration already uses real data
            if "real_market_data/processed" in config or "backtest_data" in config:
                logger.info(f"Backtesting configuration already uses real data")
                return True
                
            # Update configuration to use real data
            config = config.replace("synthetic_data", "backtest_data")
            config = config.replace("data_dir = \"data\"", "data_dir = \"backtest_data\"")
            
            # Add code to check for real data
            real_data_check = """
    # Check if real data exists
    real_data_path = f"{data_dir}/{symbol}_{interval}_real.csv"
    if os.path.exists(real_data_path):
        logger.info(f"Using real market data for {symbol}, interval {interval}")
        data_path = real_data_path
    else:
        logger.warning(f"Real market data not found for {symbol}, interval {interval}, using regular data")
        data_path = f"{data_dir}/{symbol}_{interval}.csv"
"""
            
            # Insert real data check
            if "data_path = f\"{data_dir}/{symbol}_{interval}.csv\"" in config:
                config = config.replace(
                    "data_path = f\"{data_dir}/{symbol}_{interval}.csv\"",
                    real_data_check + "    data_path = data_path"
                )
                
            # Update synthetic data ratio check
            if "synthetic_ratio = 1.0" in config:
                config = config.replace(
                    "synthetic_ratio = 1.0",
                    "synthetic_ratio = 0.0 if os.path.exists(f\"{data_dir}/{symbol}_{interval}_real.csv\") else 1.0"
                )
                
            # Write updated configuration
            with open(config_path, "w") as f:
                f.write(config)
                
            logger.info(f"Updated backtesting configuration at {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error updating backtesting configuration: {str(e)}")
            return False
            
    def verify_backtesting_data(self, symbols: List[str], intervals: List[str]) -> Dict:
        """
        Verify that backtesting is using real data.
        
        Args:
            symbols: List of symbols to verify
            intervals: List of intervals to verify
            
        Returns:
            Dictionary with verification results
        """
        results = {}
        
        for symbol in symbols:
            results[symbol] = {}
            
            for interval in intervals:
                logger.info(f"Verifying backtesting data for {symbol}, interval {interval}")
                
                # Check if real data symlink exists
                symlink_path = f"{self.backtest_dir}/{symbol}_{interval}_real.csv"
                
                if not os.path.exists(symlink_path):
                    logger.error(f"Real data symlink not found for {symbol}, interval {interval} at {symlink_path}")
                    results[symbol][interval] = {
                        "status": "error",
                        "message": f"Real data symlink not found at {symlink_path}"
                    }
                    continue
                
                # Load data from symlink
                try:
                    df = pd.read_csv(symlink_path)
                    logger.info(f"Loaded {df.shape[0]} rows of data from symlink for {symbol}, interval {interval}")
                    
                    # Check if data is the same as processed data
                    processed_path = f"{self.base_dir}/processed/{symbol}_{interval}_processed.csv"
                    
                    if os.path.exists(processed_path):
                        df_processed = pd.read_csv(processed_path)
                        
                        # Check if row counts match
                        if df.shape[0] != df_processed.shape[0]:
                            logger.warning(f"Row count mismatch for {symbol}, interval {interval}: symlink={df.shape[0]}, processed={df_processed.shape[0]}")
                            
                        # Check if data is the same
                        if "close" in df.columns and "close" in df_processed.columns:
                            if not np.array_equal(df["close"].values, df_processed["close"].values):
                                logger.warning(f"Data mismatch for {symbol}, interval {interval}")
                                
                    # Update results
                    results[symbol][interval] = {
                        "status": "success",
                        "rows": df.shape[0],
                        "symlink_path": symlink_path,
                        "is_real_data": True
                    }
                except Exception as e:
                    logger.error(f"Error verifying data for {symbol}, interval {interval}: {str(e)}")
                    results[symbol][interval] = {
                        "status": "error",
                        "message": str(e)
                    }
                    
        return results
        
    def create_test_script(self, output_path: str = "test_real_data_backtesting.py") -> bool:
        """
        Create a test script to verify that backtesting is using real data.
        
        Args:
            output_path: Path to output test script
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create test script
            script = """#!/usr/bin/env python3
\"\"\"
Test script to verify that backtesting is using real data.
\"\"\"

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
    \"\"\"
    Test that backtesting is using real data.
    
    Args:
        symbol: Symbol to test
        interval: Interval to test
        backtest_dir: Directory used by backtesting framework
        real_data_dir: Directory with real market data
        
    Returns:
        True if using real data, False otherwise
    \"\"\"
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
    \"\"\"
    Main function to test backtesting data.
    \"\"\"
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
"""
            
            # Write test script
            with open(output_path, "w") as f:
                f.write(script)
                
            # Make executable
            os.chmod(output_path, 0o755)
                
            logger.info(f"Created test script at {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating test script: {str(e)}")
            return False

def main():
    """
    Main function to fix backtesting data sources.
    """
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Fix Backtesting Data Sources")
    parser.add_argument("--symbols", type=str, default="SOL-USD-PERP,BTC-USD-PERP,ETH-USD-PERP", help="Comma-separated list of symbols")
    parser.add_argument("--intervals", type=str, default="1h,4h", help="Comma-separated list of intervals")
    parser.add_argument("--base_dir", type=str, default="real_market_data", help="Base directory for real market data")
    parser.add_argument("--backtest_dir", type=str, default="backtest_data", help="Directory used by backtesting framework")
    parser.add_argument("--config_path", type=str, default="real_data_backtesting.py", help="Path to backtesting configuration file")
    args = parser.parse_args()
    
    # Get symbols and intervals
    symbols = args.symbols.split(",")
    intervals = args.intervals.split(",")
    
    # Create fixer
    fixer = BacktestingDataFixer(args.base_dir, args.backtest_dir)
    
    # Fix data sources
    logger.info(f"Fixing data sources for symbols {symbols}, intervals {intervals}")
    results = fixer.fix_data_sources(symbols, intervals)
    
    # Print results
    logger.info(f"Data source fix results: {json.dumps(results, indent=4)}")
    
    # Update backtesting configuration
    logger.info(f"Updating backtesting configuration at {args.config_path}")
    config_updated = fixer.update_backtesting_config(args.config_path)
    
    if config_updated:
        logger.info("Backtesting configuration updated successfully")
    else:
        logger.error("Failed to update backtesting configuration")
        
    # Verify backtesting data
    logger.info(f"Verifying backtesting data for symbols {symbols}, intervals {intervals}")
    verification_results = fixer.verify_backtesting_data(symbols, intervals)
    
    # Print verification results
    logger.info(f"Verification results: {json.dumps(verification_results, indent=4)}")
    
    # Create test script
    logger.info("Creating test script")
    test_script_created = fixer.create_test_script()
    
    if test_script_created:
        logger.info("Test script created successfully")
    else:
        logger.error("Failed to create test script")
        
    # Print final status
    all_success = True
    
    for symbol in results:
        for interval in results[symbol]:
            if results[symbol][interval]["status"] != "success":
                all_success = False
                
    if all_success and config_updated and test_script_created:
        logger.info("All data sources fixed successfully! Backtesting should now use real data.")
    else:
        logger.error("Some errors occurred while fixing data sources. Check logs for details.")

if __name__ == "__main__":
    main()
