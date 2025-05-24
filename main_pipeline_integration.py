"""
Main Pipeline Integration Module

This module integrates the specialized SOL data resolver into the main data pipeline,
ensuring that real market data is used for all trading decisions and backtesting.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

# Import specialized resolvers
from sol_data_resolver import SOLDataResolver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("main_pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

class MainDataPipeline:
    """
    Main data pipeline that integrates specialized resolvers for different tokens.
    """
    
    def __init__(self, base_dir: str = "real_market_data"):
        """
        Initialize the main data pipeline.
        
        Args:
            base_dir: Base directory for data storage
        """
        self.base_dir = base_dir
        
        # Create directories
        os.makedirs(f"{self.base_dir}/raw", exist_ok=True)
        os.makedirs(f"{self.base_dir}/processed", exist_ok=True)
        
        # Initialize specialized resolvers
        self.sol_resolver = SOLDataResolver(base_dir=base_dir)
        
        logger.info("Main data pipeline initialized")
        
    async def fetch_data(self, symbol: str, interval: str, days: int = 30) -> Tuple[pd.DataFrame, Dict]:
        """
        Fetch data for a symbol and interval, using specialized resolvers when available.
        
        Args:
            symbol: Symbol to fetch data for
            interval: Interval to fetch data for
            days: Number of days to fetch
            
        Returns:
            Tuple of (DataFrame with data, Dictionary with data quality metrics)
        """
        logger.info(f"Fetching data for {symbol}, interval {interval}")
        
        # Use specialized resolver for SOL-USD-PERP
        if symbol == "SOL-USD-PERP":
            logger.info(f"Using specialized SOL resolver for {symbol}, interval {interval}")
            return await self.sol_resolver.fetch_sol_data(interval, days)
        
        # For other symbols, use the generic resolver (to be implemented)
        logger.warning(f"No specialized resolver available for {symbol}, using generic resolver")
        
        # TODO: Implement generic resolver for other symbols
        # For now, return None to indicate that specialized resolver is required
        return None, {"error": "No specialized resolver available for this symbol"}
        
    async def fetch_all_data(self, symbols: List[str], intervals: List[str], days: int = 30) -> Dict:
        """
        Fetch data for multiple symbols and intervals.
        
        Args:
            symbols: List of symbols to fetch data for
            intervals: List of intervals to fetch data for
            days: Number of days to fetch
            
        Returns:
            Dictionary with data and quality metrics for each symbol and interval
        """
        logger.info(f"Fetching data for symbols {symbols}, intervals {intervals}")
        
        results = {}
        
        for symbol in symbols:
            results[symbol] = {}
            
            for interval in intervals:
                logger.info(f"Fetching data for {symbol}, interval {interval}")
                
                df, quality = await self.fetch_data(symbol, interval, days)
                
                if df is not None:
                    results[symbol][interval] = {
                        "data": df,
                        "quality": quality
                    }
                else:
                    results[symbol][interval] = {
                        "data": None,
                        "quality": quality
                    }
                    
        return results
        
    def get_data_quality_summary(self, results: Dict) -> Dict:
        """
        Get a summary of data quality for all symbols and intervals.
        
        Args:
            results: Dictionary with data and quality metrics for each symbol and interval
            
        Returns:
            Dictionary with data quality summary
        """
        summary = {}
        
        for symbol in results:
            summary[symbol] = {}
            
            for interval in results[symbol]:
                quality = results[symbol][interval]["quality"]
                
                if "error" in quality:
                    summary[symbol][interval] = {
                        "status": "error",
                        "error": quality["error"]
                    }
                else:
                    summary[symbol][interval] = {
                        "status": "success",
                        "source": quality["source"],
                        "is_synthetic": quality["is_synthetic"],
                        "synthetic_ratio": quality["synthetic_ratio"],
                        "data_points": quality["data_points"]
                    }
                    
        return summary
        
    def save_data_quality_summary(self, summary: Dict, filename: str = "data_quality_summary.json"):
        """
        Save data quality summary to a file.
        
        Args:
            summary: Dictionary with data quality summary
            filename: Filename to save to
        """
        # Convert to serializable format
        serializable_summary = {}
        
        for symbol in summary:
            serializable_summary[symbol] = {}
            
            for interval in summary[symbol]:
                serializable_summary[symbol][interval] = {}
                
                for key, value in summary[symbol][interval].items():
                    # Convert numpy types to Python types
                    if isinstance(value, (np.int64, np.float64)):
                        serializable_summary[symbol][interval][key] = float(value)
                    else:
                        serializable_summary[symbol][interval][key] = value
                        
        # Save to file
        with open(filename, "w") as f:
            json.dump(serializable_summary, f, indent=4)
            
        logger.info(f"Data quality summary saved to {filename}")

async def main():
    """
    Main function to test the main data pipeline.
    """
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Main Data Pipeline")
    parser.add_argument("--symbols", type=str, default="SOL-USD-PERP", help="Comma-separated list of symbols")
    parser.add_argument("--intervals", type=str, default="1h,4h", help="Comma-separated list of intervals")
    parser.add_argument("--days", type=int, default=30, help="Number of days to fetch")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Create pipeline
    pipeline = MainDataPipeline()
    
    # Get symbols and intervals
    symbols = args.symbols.split(",")
    intervals = args.intervals.split(",")
    
    # Fetch data
    results = await pipeline.fetch_all_data(symbols, intervals, args.days)
    
    # Get data quality summary
    summary = pipeline.get_data_quality_summary(results)
    
    # Print summary
    logger.info(f"Data quality summary:\n{json.dumps(summary, indent=4)}")
    
    # Save summary
    pipeline.save_data_quality_summary(summary)
    
    # Print data shapes
    for symbol in results:
        for interval in results[symbol]:
            if results[symbol][interval]["data"] is not None:
                df = results[symbol][interval]["data"]
                logger.info(f"Data shape for {symbol}, interval {interval}: {df.shape}")
                
                # Save data to CSV
                df.to_csv(f"{pipeline.base_dir}/processed/{symbol}_{interval}_processed.csv", index=False)
                logger.info(f"Data saved to {pipeline.base_dir}/processed/{symbol}_{interval}_processed.csv")

if __name__ == "__main__":
    asyncio.run(main())
