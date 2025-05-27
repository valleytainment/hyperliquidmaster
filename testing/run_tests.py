"""
Test Runner Script for HyperLiquid Trading Bot

This script runs all tests and backtests for the HyperLiquid trading bot.
"""

import os
import sys
import asyncio
import logging
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from testing.test_runner import TestRunner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/test_run.log")
    ]
)

logger = logging.getLogger("TestRunScript")

async def main():
    """Run all tests and backtests."""
    logger.info("Starting test run")
    
    # Create output directory
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize test runner
    config_path = "config.json"
    test_runner = TestRunner(config_path, logger)
    
    # Run all tests
    results = await test_runner.run_all_tests()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"test_results_{timestamp}.json")
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4, default=str)
    
    logger.info(f"Test results saved to {results_file}")
    
    # Print summary
    logger.info("Test run completed")
    logger.info(f"Overall success: {results.get('success', False)}")
    
    unit_tests = results.get("unit_tests", {})
    integration_tests = results.get("integration_tests", {})
    backtests = results.get("backtests", {})
    
    logger.info(f"Unit tests: {unit_tests.get('passed', 0)} passed, {unit_tests.get('failed', 0)} failed, {unit_tests.get('errors', 0)} errors")
    logger.info(f"Integration tests: {integration_tests.get('passed', 0)} passed, {integration_tests.get('failed', 0)} failed, {integration_tests.get('errors', 0)} errors")
    logger.info(f"Backtests: {backtests.get('passed', 0)} passed, {backtests.get('failed', 0)} failed, {backtests.get('errors', 0)} errors")
    
    if backtests.get("best_strategy"):
        logger.info(f"Best strategy: {backtests.get('best_strategy')} with {backtests.get('best_return', 0):.2f}% return")
    
    return results

if __name__ == "__main__":
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Run tests
    asyncio.run(main())
