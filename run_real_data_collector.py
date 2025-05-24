#!/usr/bin/env python3
"""
Script to run the Hyperliquid real market data collector.
"""

import asyncio
import logging
from hyperliquid_data_collector import HyperliquidDataCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("real_data_collection.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("RealDataCollection")

async def main():
    """Main entry point for real data collection."""
    logger.info("Starting real market data collection from Hyperliquid...")
    
    # Create data collector
    collector = HyperliquidDataCollector()
    
    # Define symbols and intervals
    symbols = ["BTC-USD-PERP", "ETH-USD-PERP", "SOL-USD-PERP"]
    intervals = ["1h", "4h"]
    
    # Collect all real market data (reduced stream duration for testing)
    logger.info(f"Collecting data for symbols: {symbols} with intervals: {intervals}")
    results = await collector.collect_all_real_data(symbols, intervals, stream_duration=300)
    
    # Process collected data
    logger.info("Processing collected real market data...")
    processing_results = collector.process_collected_data(symbols, intervals)
    
    logger.info(f"Data collection completed with results: {results}")
    logger.info(f"Data processing completed with results: {processing_results}")
    
    return results, processing_results

if __name__ == "__main__":
    asyncio.run(main())
