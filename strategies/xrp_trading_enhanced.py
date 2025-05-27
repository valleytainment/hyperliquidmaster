#!/usr/bin/env python3
"""
Enhanced XRP Trading Test Script with Improved Connection Handling

This script tests live XRP trading with the HyperliquidMaster system,
including buy and sell orders, and evaluates profit building capabilities.
"""

import os
import sys
import json
import time
import logging
import traceback
from typing import Dict, Any, List, Optional
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("xrp_trading_enhanced.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import required modules
from core.hyperliquid_adapter import HyperliquidAdapter

class XRPTradingTest:
    """Test class for XRP trading and profit evaluation with enhanced connection handling"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the test"""
        self.config_path = config_path
        self.adapter = HyperliquidAdapter(config_path)
        self.symbol = "XRP"
        self.test_sizes = [0.1, 0.2, 0.5]  # Different position sizes to test
        self.max_retries = 5  # Maximum number of retries for API operations
        self.retry_delay = 5  # Delay between retries in seconds
        self.test_results = {
            "trades": [],
            "success_count": 0,
            "total_trades": 0,
            "total_pnl": 0.0,
            "execution_times": [],
            "initial_balance": 0.0,
            "final_balance": 0.0,
            "actual_pnl": 0.0,
            "score": 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize the test with enhanced error handling"""
        logger.info("Initializing XRP trading test...")
        logger.info("Initializing HyperliquidAdapter...")
        
        # Initialize adapter with retries
        for attempt in range(self.max_retries):
            try:
                success = await self.adapter.initialize()
                if success:
                    logger.info("HyperliquidAdapter initialized successfully")
                    
                    # Get initial account balance with retries
                    for balance_attempt in range(self.max_retries):
                        try:
                            account_info = self.adapter.get_account_info()
                            if "error" not in account_info:
                                self.test_results["initial_balance"] = account_info.get("equity", 0.0)
                                logger.info(f"Initial account balance: ${self.test_results['initial_balance']:.2f}")
                                return True
                            else:
                                logger.warning(f"Error getting account info (attempt {balance_attempt+1}/{self.max_retries}): {account_info['error']}")
                                if balance_attempt < self.max_retries - 1:
                                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                                    await asyncio.sleep(self.retry_delay)
                        except Exception as e:
                            logger.warning(f"Exception getting account info (attempt {balance_attempt+1}/{self.max_retries}): {e}")
                            if balance_attempt < self.max_retries - 1:
                                logger.info(f"Retrying in {self.retry_delay} seconds...")
                                await asyncio.sleep(self.retry_delay)
                    
                    # If we get here, all balance attempts failed
                    logger.error("Failed to get account info after maximum retries")
                    return False
                else:
                    logger.warning(f"Failed to initialize adapter (attempt {attempt+1}/{self.max_retries})")
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying in {self.retry_delay} seconds...")
                        await asyncio.sleep(self.retry_delay)
            except Exception as e:
                logger.warning(f"Exception initializing adapter (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    await asyncio.sleep(self.retry_delay)
        
        # If we get here, all initialization attempts failed
        logger.error("Failed to initialize adapter after maximum retries")
        return False
    
    async def execute_buy_trade(self, size: float) -> Dict[str, Any]:
        """Execute a buy trade for XRP with enhanced error handling"""
        logger.info(f"Executing buy trade for {self.symbol} with size {size}")
        
        try:
            # Get current market price with retries
            for attempt in range(self.max_retries):
                try:
                    market_data = self.adapter.get_market_data(self.symbol)
                    if "error" not in market_data:
                        price = market_data.get("price", 0.0)
                        if price > 0:
                            break
                        else:
                            logger.warning(f"Invalid market price for {self.symbol} (attempt {attempt+1}/{self.max_retries})")
                    else:
                        logger.warning(f"Error getting market data (attempt {attempt+1}/{self.max_retries}): {market_data['error']}")
                    
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying in {self.retry_delay} seconds...")
                        await asyncio.sleep(self.retry_delay)
                except Exception as e:
                    logger.warning(f"Exception getting market data (attempt {attempt+1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying in {self.retry_delay} seconds...")
                        await asyncio.sleep(self.retry_delay)
            
            # Check if we got a valid price
            if price <= 0:
                error_msg = f"Failed to get valid market price for {self.symbol} after maximum retries"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            
            # Place buy order with retries
            for attempt in range(self.max_retries):
                try:
                    start_time = time.time()
                    result = self.adapter.place_order(
                        symbol=self.symbol,
                        is_buy=True,
                        size=size,
                        price=price,
                        order_type="LIMIT"
                    )
                    execution_time = time.time() - start_time
                    
                    if "error" not in result:
                        # Record trade
                        trade_data = {
                            "symbol": self.symbol,
                            "side": "buy",
                            "size": size,
                            "price": price,
                            "execution_time": execution_time,
                            "order_id": result.get("data", {}).get("id", "Unknown")
                        }
                        
                        logger.info(f"Buy trade executed successfully: {trade_data}")
                        return {"success": True, "trade": trade_data, "execution_time": execution_time}
                    else:
                        logger.warning(f"Error placing order (attempt {attempt+1}/{self.max_retries}): {result['error']}")
                        if attempt < self.max_retries - 1:
                            logger.info(f"Retrying in {self.retry_delay} seconds...")
                            await asyncio.sleep(self.retry_delay)
                except Exception as e:
                    logger.warning(f"Exception placing order (attempt {attempt+1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying in {self.retry_delay} seconds...")
                        await asyncio.sleep(self.retry_delay)
            
            # If we get here, all order attempts failed
            error_msg = "Failed to place buy order after maximum retries"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        except Exception as e:
            error_msg = f"Error executing buy trade: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {"success": False, "error": error_msg}
    
    async def execute_sell_trade(self, size: float) -> Dict[str, Any]:
        """Execute a sell trade for XRP with enhanced error handling"""
        logger.info(f"Executing sell trade for {self.symbol} with size {size}")
        
        try:
            # Get current market price with retries
            for attempt in range(self.max_retries):
                try:
                    market_data = self.adapter.get_market_data(self.symbol)
                    if "error" not in market_data:
                        price = market_data.get("price", 0.0)
                        if price > 0:
                            break
                        else:
                            logger.warning(f"Invalid market price for {self.symbol} (attempt {attempt+1}/{self.max_retries})")
                    else:
                        logger.warning(f"Error getting market data (attempt {attempt+1}/{self.max_retries}): {market_data['error']}")
                    
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying in {self.retry_delay} seconds...")
                        await asyncio.sleep(self.retry_delay)
                except Exception as e:
                    logger.warning(f"Exception getting market data (attempt {attempt+1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying in {self.retry_delay} seconds...")
                        await asyncio.sleep(self.retry_delay)
            
            # Check if we got a valid price
            if price <= 0:
                error_msg = f"Failed to get valid market price for {self.symbol} after maximum retries"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            
            # Place sell order with retries
            for attempt in range(self.max_retries):
                try:
                    start_time = time.time()
                    result = self.adapter.place_order(
                        symbol=self.symbol,
                        is_buy=False,
                        size=size,
                        price=price,
                        order_type="LIMIT"
                    )
                    execution_time = time.time() - start_time
                    
                    if "error" not in result:
                        # Record trade
                        trade_data = {
                            "symbol": self.symbol,
                            "side": "sell",
                            "size": size,
                            "price": price,
                            "execution_time": execution_time,
                            "order_id": result.get("data", {}).get("id", "Unknown")
                        }
                        
                        logger.info(f"Sell trade executed successfully: {trade_data}")
                        return {"success": True, "trade": trade_data, "execution_time": execution_time}
                    else:
                        logger.warning(f"Error placing order (attempt {attempt+1}/{self.max_retries}): {result['error']}")
                        if attempt < self.max_retries - 1:
                            logger.info(f"Retrying in {self.retry_delay} seconds...")
                            await asyncio.sleep(self.retry_delay)
                except Exception as e:
                    logger.warning(f"Exception placing order (attempt {attempt+1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying in {self.retry_delay} seconds...")
                        await asyncio.sleep(self.retry_delay)
            
            # If we get here, all order attempts failed
            error_msg = "Failed to place sell order after maximum retries"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        except Exception as e:
            error_msg = f"Error executing sell trade: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {"success": False, "error": error_msg}
    
    async def execute_trade_cycle(self, size: float) -> Dict[str, Any]:
        """Execute a complete trade cycle (buy then sell) with enhanced error handling"""
        logger.info(f"Executing trade cycle for {self.symbol} with size {size}")
        
        # Execute buy trade
        buy_result = await self.execute_buy_trade(size)
        if not buy_result.get("success", False):
            error_msg = f"Trade cycle failed at buy step: {buy_result.get('error', 'Unknown error')}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        # Wait for a short time to simulate holding period
        logger.info(f"Waiting for {self.retry_delay} seconds before selling...")
        await asyncio.sleep(self.retry_delay)
        
        # Execute sell trade
        sell_result = await self.execute_sell_trade(size)
        if not sell_result.get("success", False):
            error_msg = f"Trade cycle failed at sell step: {sell_result.get('error', 'Unknown error')}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        # Calculate PnL
        buy_price = buy_result.get("trade", {}).get("price", 0.0)
        sell_price = sell_result.get("trade", {}).get("price", 0.0)
        pnl = (sell_price - buy_price) * size
        
        # Record cycle
        cycle_data = {
            "buy": buy_result.get("trade", {}),
            "sell": sell_result.get("trade", {}),
            "pnl": pnl,
            "execution_time": buy_result.get("execution_time", 0.0) + sell_result.get("execution_time", 0.0)
        }
        
        logger.info(f"Trade cycle completed successfully: PnL=${pnl:.4f}")
        return {"success": True, "cycle": cycle_data}
    
    async def run_test(self) -> Dict[str, Any]:
        """Run the XRP trading test with enhanced error handling"""
        logger.info("Starting XRP trading test...")
        
        try:
            # Initialize test
            init_success = await self.initialize()
            
            if not init_success:
                logger.error("Failed to initialize test")
                return self._calculate_score()
            
            # Execute trade cycles with different sizes
            for size in self.test_sizes:
                self.test_results["total_trades"] += 1
                
                # Execute trade cycle
                cycle_result = await self.execute_trade_cycle(size)
                
                if cycle_result.get("success", False):
                    self.test_results["success_count"] += 1
                    cycle_data = cycle_result.get("cycle", {})
                    
                    # Record trade data
                    self.test_results["trades"].append(cycle_data)
                    self.test_results["total_pnl"] += cycle_data.get("pnl", 0.0)
                    self.test_results["execution_times"].append(cycle_data.get("execution_time", 0.0))
                
                # Wait between cycles
                logger.info(f"Waiting for {self.retry_delay} seconds before next cycle...")
                await asyncio.sleep(self.retry_delay)
            
            # Get final account balance with retries
            for attempt in range(self.max_retries):
                try:
                    account_info = self.adapter.get_account_info()
                    if "error" not in account_info:
                        self.test_results["final_balance"] = account_info.get("equity", 0.0)
                        self.test_results["actual_pnl"] = self.test_results["final_balance"] - self.test_results["initial_balance"]
                        break
                    else:
                        logger.warning(f"Error getting final account info (attempt {attempt+1}/{self.max_retries}): {account_info['error']}")
                        if attempt < self.max_retries - 1:
                            logger.info(f"Retrying in {self.retry_delay} seconds...")
                            await asyncio.sleep(self.retry_delay)
                except Exception as e:
                    logger.warning(f"Exception getting final account info (attempt {attempt+1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying in {self.retry_delay} seconds...")
                        await asyncio.sleep(self.retry_delay)
            
            logger.info(f"Final account balance: ${self.test_results['final_balance']:.2f}")
            logger.info(f"Actual P&L from account balance: ${self.test_results['actual_pnl']:.2f}")
            
            # Calculate score
            return self._calculate_score()
        
        except Exception as e:
            logger.error(f"Error running test: {e}")
            logger.error(traceback.format_exc())
            return self._calculate_score()
    
    def _calculate_score(self) -> Dict[str, Any]:
        """Calculate the performance score"""
        # Calculate success rate
        success_rate = 0.0
        if self.test_results["total_trades"] > 0:
            success_rate = self.test_results["success_count"] / self.test_results["total_trades"] * 100
        
        # Calculate average execution time
        avg_execution_time = 0.0
        if len(self.test_results["execution_times"]) > 0:
            avg_execution_time = sum(self.test_results["execution_times"]) / len(self.test_results["execution_times"])
        
        # Calculate score components
        reliability_score = success_rate / 100 * 40  # 40% weight for reliability
        profit_score = 0.0
        if self.test_results["total_pnl"] > 0:
            profit_score = min(40, self.test_results["total_pnl"] * 10)  # 40% weight for profit
        
        speed_score = 0.0
        if avg_execution_time > 0:
            speed_score = min(20, 20 / (avg_execution_time + 1))  # 20% weight for speed
        
        # Calculate total score
        total_score = reliability_score + profit_score + speed_score
        
        # Update test results
        self.test_results["success_rate"] = success_rate
        self.test_results["avg_execution_time"] = avg_execution_time
        self.test_results["score"] = total_score
        
        # Log score
        logger.info(f"XRP trading test completed with score: {total_score:.1f}/100")
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"Total P&L: ${self.test_results['total_pnl']:.4f}")
        logger.info(f"Average execution time: {avg_execution_time:.3f}s")
        
        # Save results to file
        with open("xrp_test_results.json", "w") as f:
            json.dump(self.test_results, f, indent=2)
        
        logger.info("Test results saved to xrp_test_results.json")
        
        # Print summary
        print("=" * 50)
        print("XRP TRADING TEST RESULTS")
        print("=" * 50)
        print(f"Score: {total_score:.1f}/100")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total P&L: ${self.test_results['total_pnl']:.4f}")
        print(f"Average Execution Time: {avg_execution_time:.3f}s")
        print(f"Total Trades: {self.test_results['total_trades']}")
        print("=" * 50)
        
        return self.test_results

async def main():
    """Main entry point for the XRP trading test"""
    # Run the test
    test = XRPTradingTest()
    results = await test.run_test()
    return results

if __name__ == "__main__":
    # Create and run asyncio event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        results = loop.run_until_complete(main())
    finally:
        loop.close()
