"""
Hyperliquid API Integration with Rate Limiting

This module provides a rate-limited interface to the Hyperliquid API.
"""

import os
import sys
import json
import logging
import time
import asyncio
import traceback
import requests
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

from api_rate_limiter import APIRateLimiter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class HyperliquidAPIClient:
    """
    Rate-limited client for Hyperliquid API.
    """
    
    def __init__(self, base_url: str = "https://api.hyperliquid.xyz"):
        """
        Initialize the Hyperliquid API client.
        
        Args:
            base_url: Base URL for Hyperliquid API
        """
        self.base_url = base_url
        self.rate_limiter = APIRateLimiter(
            requests_per_minute=30,  # Conservative limit
            cache_ttl_seconds=2,     # Short cache TTL for market data
            max_retries=5,
            retry_delay=1.0
        )
        
        # Initialize cache for meta data with longer TTL
        self.meta_cache = {}
        self.meta_cache_ttl = 60  # 60 seconds
        self.meta_cache_timestamp = 0
        
    def get_meta(self) -> Dict:
        """
        Get meta data with caching and rate limiting.
        
        Returns:
            Meta data dictionary
        """
        # Check if we have a valid cached meta data
        current_time = time.time()
        if self.meta_cache and current_time - self.meta_cache_timestamp < self.meta_cache_ttl:
            logger.debug("Using cached meta data")
            return self.meta_cache
            
        # Define the function to execute
        def execute_request():
            payload = {"type": "meta"}
            response = requests.post(f"{self.base_url}/info", json=payload)
            
            if response.status_code != 200:
                raise Exception(f"Failed to get meta data: {response.status_code}")
                
            return response.json()
            
        try:
            # Execute with rate limiting
            result = self.rate_limiter.execute_with_rate_limit(
                func=execute_request,
                endpoint="/info",
                params={"type": "meta"},
                use_cache=False  # We're handling caching manually for meta
            )
            
            # Update cache
            self.meta_cache = result
            self.meta_cache_timestamp = current_time
            
            return result
        except Exception as e:
            logger.error(f"Error getting meta data: {str(e)}")
            
            # Return cached data if available, even if expired
            if self.meta_cache:
                logger.warning("Returning stale meta data due to API error")
                return self.meta_cache
                
            return {}
            
    def get_all_mids(self) -> Dict:
        """
        Get all mid prices with rate limiting.
        
        Returns:
            Dictionary of mid prices
        """
        def execute_request():
            payload = {"type": "allMids"}
            response = requests.post(f"{self.base_url}/info", json=payload)
            
            if response.status_code != 200:
                raise Exception(f"Failed to get all mids: {response.status_code}")
                
            return response.json()
            
        try:
            return self.rate_limiter.execute_with_rate_limit(
                func=execute_request,
                endpoint="/info",
                params={"type": "allMids"},
                use_cache=True
            )
        except Exception as e:
            logger.error(f"Error getting all mids: {str(e)}")
            return {}
            
    def get_order_book(self, symbol: str) -> Dict:
        """
        Get order book with rate limiting.
        
        Args:
            symbol: Symbol to get order book for
            
        Returns:
            Order book dictionary
        """
        def execute_request():
            payload = {"type": "l2Book", "coin": symbol}
            response = requests.post(f"{self.base_url}/info", json=payload)
            
            if response.status_code != 200:
                raise Exception(f"Failed to get order book: {response.status_code}")
                
            return response.json()
            
        try:
            return self.rate_limiter.execute_with_rate_limit(
                func=execute_request,
                endpoint="/info",
                params={"type": "l2Book", "coin": symbol},
                use_cache=True
            )
        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {str(e)}")
            return {}
            
    def get_market_data(self, symbol: str) -> Dict:
        """
        Get comprehensive market data for a symbol.
        
        Args:
            symbol: Symbol to get market data for
            
        Returns:
            Market data dictionary
        """
        try:
            # Get meta data
            meta = self.get_meta()
            
            # Find asset index
            asset_index = None
            for i, asset in enumerate(meta.get("universe", [])):
                if asset.get("name") == symbol:
                    asset_index = i
                    break
                    
            if asset_index is None:
                logger.warning(f"Asset not found in meta data: {symbol}")
                return {}
                
            # Get all mids (prices)
            all_mids = self.get_all_mids()
            
            # Try different key formats - the API might use symbol name or index-based keys
            price = None
            
            # First try symbol name directly (e.g., "BTC")
            if symbol in all_mids:
                price = float(all_mids[symbol])
            else:
                # Try index-based key (e.g., "@0")
                price_key = f"@{asset_index}"
                
                if price_key in all_mids:
                    price = float(all_mids[price_key])
                    
            if not price or price <= 0:
                logger.warning(f"Price not found for {symbol}")
                return {}
                
            # Get order book (with lower priority)
            order_book = None
            try:
                order_book = self.get_order_book(symbol)
            except Exception as e:
                logger.warning(f"Could not get order book for {symbol}: {str(e)}")
                
            # Create market data dictionary
            result = {
                "symbol": symbol,
                "last_price": price,
                "bid_price": price * 0.999,  # Approximate
                "ask_price": price * 1.001,  # Approximate
                "funding_rate": 0.0,  # Default since funding rate endpoint is not reliable
                "timestamp": datetime.now().timestamp(),
                "order_book": order_book
            }
            
            logger.info(f"Retrieved market data for {symbol}: price={price}")
            return result
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return {}

# Singleton instance
api_client = HyperliquidAPIClient()

def get_market_data(symbol: str) -> Dict:
    """
    Get market data for a symbol using the singleton client.
    
    Args:
        symbol: Symbol to get market data for
        
    Returns:
        Market data dictionary
    """
    return api_client.get_market_data(symbol)
    
def get_meta() -> Dict:
    """
    Get meta data using the singleton client.
    
    Returns:
        Meta data dictionary
    """
    return api_client.get_meta()
    
def get_all_mids() -> Dict:
    """
    Get all mid prices using the singleton client.
    
    Returns:
        Dictionary of mid prices
    """
    return api_client.get_all_mids()
    
def get_order_book(symbol: str) -> Dict:
    """
    Get order book using the singleton client.
    
    Args:
        symbol: Symbol to get order book for
        
    Returns:
        Order book dictionary
    """
    return api_client.get_order_book(symbol)
