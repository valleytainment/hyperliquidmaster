#!/usr/bin/env python3
"""
Debug script to inspect Exchange.order response structure
"""

import sys
import json
import logging
from typing import Any, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("exchange_order_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import required modules
import eth_account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info

def debug_exchange_order():
    """Test Exchange.order and inspect response structure"""
    try:
        # Load credentials
        account_address = "0x306D29F56EA1345c7E6F1ff27657ba05cEE15D4F"
        secret_key = "43ba46de58067dd1ef3794c653bf3b11fa78866623cc515a5aff5f4be31fd3b8"
        
        # Create account
        account = eth_account.Account.from_key(secret_key)
        
        # Initialize Exchange
        exchange = Exchange(
            wallet=account,
            account_address=account_address
        )
        
        # Initialize Info
        info = Info()
        
        # Get available assets
        logger.info("Getting available assets...")
        meta_assets = info.meta_and_asset_ctxs()
        
        # Check if meta_assets is a list with at least one element
        if isinstance(meta_assets, list) and len(meta_assets) > 0:
            meta = meta_assets[0]
            if "universe" in meta:
                universe = meta["universe"]
                logger.info(f"Available assets: {[asset.get('name', 'Unknown') for asset in universe]}")
        
        # Place test order for XRP
        logger.info("Placing test order for XRP...")
        result = exchange.order(
            name="XRP",
            is_buy=True,
            sz=0.1,
            limit_px=0.5,
            order_type="limit"
        )
        
        # Inspect result
        logger.info(f"Result type: {type(result)}")
        logger.info(f"Result content: {result}")
        
        # Try to access result as different types
        logger.info("Attempting to access result in different ways:")
        
        # As dictionary
        if isinstance(result, dict):
            logger.info("Result is a dictionary")
            logger.info(f"Keys: {result.keys()}")
            for key, value in result.items():
                logger.info(f"  {key}: {value} (type: {type(value)})")
        
        # As list
        elif isinstance(result, list):
            logger.info("Result is a list")
            logger.info(f"Length: {len(result)}")
            for i, item in enumerate(result):
                logger.info(f"  Item {i}: {item} (type: {type(item)})")
        
        # As string
        elif isinstance(result, str):
            logger.info("Result is a string")
            logger.info(f"Length: {len(result)}")
            # Try to parse as JSON
            try:
                parsed = json.loads(result)
                logger.info(f"Parsed JSON: {parsed}")
            except json.JSONDecodeError:
                logger.info("Not valid JSON")
        
        # Other type
        else:
            logger.info(f"Result is of type {type(result)}")
            logger.info(f"Dir(result): {dir(result)}")
        
        return {
            "success": True,
            "result_type": str(type(result)),
            "result": result
        }
    
    except Exception as e:
        logger.error(f"Error in debug_exchange_order: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    result = debug_exchange_order()
    print("\n" + "="*50)
    print("EXCHANGE ORDER DEBUG RESULTS")
    print("="*50)
    if result["success"]:
        print(f"Result Type: {result['result_type']}")
        print(f"Result: {result['result']}")
    else:
        print(f"Error: {result['error']}")
    print("="*50)
    print("See exchange_order_debug.log for more details")
