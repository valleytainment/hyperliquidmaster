"""
Hyperliquid Exchange Adapter Module

This module provides an adapter for interacting with the Hyperliquid exchange API.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hyperliquid_adapter.log")
    ]
)
logger = logging.getLogger(__name__)

try:
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange
    from hyperliquid.utils import constants
    import web3
    from eth_account import Account
except ImportError:
    logger.error("Required packages missing. Install hyperliquid-python-sdk and web3/eth_account.")
    raise ImportError("Required packages missing. Install hyperliquid-python-sdk and web3/eth_account.")

class HyperliquidAdapter:
    """
    Adapter for interacting with the Hyperliquid exchange API.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize Hyperliquid adapter.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize API clients
        self.info = Info(base_url=self.config.get("base_url", "https://api.hyperliquid.xyz"))
        
        # Initialize account if private key is provided
        if "private_key" in self.config and self.config["private_key"]:
            self.account = Account.from_key(self.config["private_key"])
            self.exchange = Exchange(
                base_url=self.config.get("base_url", "https://api.hyperliquid.xyz"),
                wallet=self.account
            )
        else:
            self.account = None
            self.exchange = None
            
        logger.info("Hyperliquid adapter initialized")
        
    def _load_config(self) -> Dict:
        """
        Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
                
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
            
    async def get_market_data(self, symbol: str) -> Dict:
        """
        Get market data for a symbol.
        
        Args:
            symbol: Symbol to get market data for
            
        Returns:
            Market data dictionary
        """
        try:
            # Get meta data to find asset index
            meta = self._get_meta()
            asset_index = None
            
            for i, asset in enumerate(meta.get("universe", [])):
                if asset.get("name") == symbol:
                    asset_index = i
                    break
                    
            if asset_index is None:
                logger.warning(f"Asset not found in meta data: {symbol}")
                return {}
                
            # Get all mids (prices) with rate limiting
            max_retries = 5
            retry_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    payload = {"type": "allMids"}
                    response = requests.post(
                        f"{self.config.get('base_url', 'https://api.hyperliquid.xyz')}/info",
                        json=payload
                    )
                    
                    if response.status_code == 429 and attempt < max_retries - 1:
                        # Rate limit hit, apply exponential backoff with jitter
                        sleep_time = retry_delay * (2 ** attempt) * (0.5 + random.random())
                        logger.warning(f"Rate limit hit, retrying in {sleep_time:.2f} seconds (attempt {attempt+1}/{max_retries})")
                        time.sleep(sleep_time)
                        continue
                        
                    if response.status_code != 200:
                        logger.warning(f"Failed to get all mids: {response.status_code}")
                        return {}
                        
                    all_mids = response.json()
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        sleep_time = retry_delay * (2 ** attempt) * (0.5 + random.random())
                        logger.warning(f"Rate limit hit, retrying in {sleep_time:.2f} seconds (attempt {attempt+1}/{max_retries})")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"Error getting all mids: {str(e)}")
                        return {}
            
            # Try different key formats - the API might use symbol name or index-based keys
            # First try symbol name directly (e.g., "BTC")
            if symbol in all_mids:
                price_key = symbol
            else:
                # Try index-based key (e.g., "@0")
                price_key = f"@{asset_index}"
                
                # If that doesn't exist either, return empty
                if price_key not in all_mids:
                    logger.warning(f"Price not found for {symbol} (tried keys: {symbol}, {price_key})")
                    return {}
                
            price = float(all_mids[price_key])
            
            if price <= 0:
                logger.warning(f"Invalid price for {symbol}: {price}")
                return {}
            
            # Get funding rates (using fallback since direct API doesn't work)
            funding_rate = 0.0  # Default to 0 since funding endpoints return 422
            
            # Create market data dictionary
            result = {
                "symbol": symbol,
                "last_price": price,
                "bid_price": price * 0.999,  # Approximate
                "ask_price": price * 1.001,  # Approximate
                "funding_rate": funding_rate,
                "timestamp": datetime.now().timestamp()
            }
            
            logger.info(f"Retrieved market data for {symbol}: price={price}, funding_rate={funding_rate}")
            return result
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
            
    async def get_order_book(self, symbol: str) -> Dict:
        """
        Get order book for a symbol.
        
        Args:
            symbol: Symbol to get order book for
            
        Returns:
            Order book dictionary
        """
        try:
            # Get meta data
            meta = self._get_meta()
            
            # Find asset index
            asset_index = None
            
            for i, asset in enumerate(meta.get("universe", [])):
                if asset.get("name") == symbol:
                    asset_index = i
                    break
                    
            if asset_index is None:
                logger.warning(f"Asset not found: {symbol}")
                return {}
                
            # Get order book
            order_book = self._get_order_book(asset_index)
            
            if not order_book:
                logger.warning(f"Order book not available for {symbol}")
                return {}
                
            # Create order book dictionary
            result = {
                "symbol": symbol,
                "bids": order_book.get("bids", []),
                "asks": order_book.get("asks", []),
                "timestamp": datetime.now().timestamp()
            }
            
            return result
        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
            
    async def get_positions(self) -> Dict:
        """
        Get positions.
        
        Returns:
            Positions dictionary
        """
        try:
            if not self.exchange:
                logger.warning("Exchange client not initialized (private key not provided)")
                return {}
                
            # Get user state
            user_state = await self.exchange.user_state()
            
            if not user_state:
                logger.warning("User state not available")
                return {}
                
            # Get meta data
            meta = self._get_meta()
            
            # Create positions dictionary
            positions = {}
            
            for position in user_state.get("assetPositions", []):
                asset_index = position.get("coin")
                
                if asset_index is None or asset_index >= len(meta.get("universe", [])):
                    continue
                    
                symbol = meta["universe"][asset_index].get("name")
                
                if not symbol:
                    continue
                    
                positions[symbol] = {
                    "symbol": symbol,
                    "size": position.get("position", {}).get("size", 0.0),
                    "entry_price": position.get("position", {}).get("entryPx", 0.0),
                    "liquidation_price": position.get("position", {}).get("liquidationPx", 0.0),
                    "unrealized_pnl": position.get("position", {}).get("unrealizedPnl", 0.0),
                    "leverage": position.get("position", {}).get("leverage", 0.0)
                }
                
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
            
    async def place_order(self, symbol: str, side: str, quantity: float, price: float, 
                         order_type: str = "limit", reduce_only: bool = False) -> Dict:
        """
        Place an order.
        
        Args:
            symbol: Symbol to place order for
            side: Order side (buy/sell)
            quantity: Order quantity
            price: Order price
            order_type: Order type (limit/market)
            reduce_only: Whether order is reduce-only
            
        Returns:
            Order response dictionary
        """
        try:
            if not self.exchange:
                logger.warning("Exchange client not initialized (private key not provided)")
                return {}
                
            # Get meta data
            meta = self._get_meta()
            
            # Find asset index
            asset_index = None
            
            for i, asset in enumerate(meta.get("universe", [])):
                if asset.get("name") == symbol:
                    asset_index = i
                    break
                    
            if asset_index is None:
                logger.warning(f"Asset not found: {symbol}")
                return {}
                
            # Create order
            order = {
                "coin": asset_index,
                "is_buy": side.lower() == "buy",
                "sz": quantity,
                "limit_px": price,
                "reduce_only": reduce_only,
                "order_type": {
                    "limit": {"tif": "Gtc"}
                } if order_type.lower() == "limit" else {
                    "market": {}
                }
            }
            
            # Place order
            response = await self.exchange.place_order(order)
            
            if not response:
                logger.warning(f"Order placement failed for {symbol} {side} {quantity} @ {price}")
                return {}
                
            # Create order response dictionary
            result = {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "order_type": order_type,
                "reduce_only": reduce_only,
                "order_id": response.get("order_id"),
                "status": response.get("status"),
                "timestamp": datetime.now().timestamp()
            }
            
            logger.info(f"Placed order for {symbol} {side} {quantity} @ {price}: {result}")
            return result
        except Exception as e:
            logger.error(f"Error placing order for {symbol} {side} {quantity} @ {price}: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
            
    async def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """
        Cancel an order.
        
        Args:
            symbol: Symbol to cancel order for
            order_id: Order ID to cancel
            
        Returns:
            Cancel response dictionary
        """
        try:
            if not self.exchange:
                logger.warning("Exchange client not initialized (private key not provided)")
                return {}
                
            # Get meta data
            meta = self._get_meta()
            
            # Find asset index
            asset_index = None
            
            for i, asset in enumerate(meta.get("universe", [])):
                if asset.get("name") == symbol:
                    asset_index = i
                    break
                    
            if asset_index is None:
                logger.warning(f"Asset not found: {symbol}")
                return {}
                
            # Cancel order
            response = await self.exchange.cancel_order(asset_index, order_id)
            
            if not response:
                logger.warning(f"Order cancellation failed for {symbol} {order_id}")
                return {}
                
            # Create cancel response dictionary
            result = {
                "symbol": symbol,
                "order_id": order_id,
                "status": response.get("status"),
                "timestamp": datetime.now().timestamp()
            }
            
            logger.info(f"Cancelled order for {symbol} {order_id}: {result}")
            return result
        except Exception as e:
            logger.error(f"Error cancelling order for {symbol} {order_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
            
    async def get_orders(self) -> Dict:
        """
        Get orders.
        
        Returns:
            Orders dictionary
        """
        try:
            if not self.exchange:
                logger.warning("Exchange client not initialized (private key not provided)")
                return {}
                
            # Get user state
            user_state = await self.exchange.user_state()
            
            if not user_state:
                logger.warning("User state not available")
                return {}
                
            # Get meta data
            meta = self._get_meta()
            
            # Create orders dictionary
            orders = {}
            
            for order in user_state.get("openOrders", []):
                asset_index = order.get("coin")
                
                if asset_index is None or asset_index >= len(meta.get("universe", [])):
                    continue
                    
                symbol = meta["universe"][asset_index].get("name")
                
                if not symbol:
                    continue
                    
                order_id = order.get("oid")
                
                if not order_id:
                    continue
                    
                orders[order_id] = {
                    "symbol": symbol,
                    "order_id": order_id,
                    "side": "buy" if order.get("is_buy") else "sell",
                    "quantity": order.get("sz", 0.0),
                    "price": order.get("limit_px", 0.0),
                    "reduce_only": order.get("reduce_only", False),
                    "timestamp": order.get("timestamp", datetime.now().timestamp())
                }
                
            return orders
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
            
    def _get_meta(self) -> Dict:
        """
        Get meta data.
        
        Returns:
            Meta data dictionary
        """
        try:
            # Implement rate limiting with exponential backoff
            max_retries = 5
            retry_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    return self.info.meta()
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        # Rate limit hit, apply exponential backoff with jitter
                        sleep_time = retry_delay * (2 ** attempt) * (0.5 + random.random())
                        logger.warning(f"Rate limit hit, retrying in {sleep_time:.2f} seconds (attempt {attempt+1}/{max_retries})")
                        time.sleep(sleep_time)
                    else:
                        raise
            
            # If we get here, all retries failed
            logger.error("All retry attempts failed for meta data")
            return {}
        except Exception as e:
            logger.error(f"Error getting meta data: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
            
    def _get_all_mids(self) -> List:
        """
        Get all mid prices.
        
        Returns:
            List of mid prices
        """
        try:
            return self.info.all_mids()
        except Exception as e:
            logger.error(f"Error getting all mids: {str(e)}")
            logger.error(traceback.format_exc())
            return []
            
    def _get_funding_rates(self) -> Dict:
        """
        Get funding rates.
        
        Returns:
            Dictionary of funding rates
        """
        try:
            # The SDK doesn't have a funding_rates method, so we'll implement our own
            # by making a direct API call to get funding info
            payload = {"type": "fundingInfo"}
            response = requests.post(
                f"{self.config.get('base_url', 'https://api.hyperliquid.xyz')}/info",
                json=payload
            )
            
            if response.status_code != 200:
                logger.warning(f"Failed to get funding info: {response.status_code}")
                return {}
                
            data = response.json()
            
            # Create a dictionary mapping asset indices to funding rates
            funding_rates = {}
            
            if isinstance(data, list):
                for i, item in enumerate(data):
                    funding_rate = item.get("fundingRate", 0.0)
                    funding_rates[i] = funding_rate
                    
            return funding_rates
        except Exception as e:
            logger.error(f"Error getting funding rates: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
            
    def _get_order_book(self, asset_index: int) -> Dict:
        """
        Get order book.
        
        Args:
            asset_index: Asset index
            
        Returns:
            Order book dictionary
        """
        try:
            # Get meta data to find the symbol name
            meta = self._get_meta()
            
            if not meta or "universe" not in meta or asset_index >= len(meta["universe"]):
                logger.warning(f"Invalid asset index: {asset_index}")
                return {}
                
            # Get symbol name from asset index
            symbol_name = meta["universe"][asset_index].get("name")
            
            if not symbol_name:
                logger.warning(f"Symbol name not found for asset index: {asset_index}")
                return {}
                
            # Use symbol name instead of asset index with rate limiting
            max_retries = 5
            retry_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    payload = {"type": "l2Book", "coin": symbol_name}
                    response = requests.post(
                        f"{self.config.get('base_url', 'https://api.hyperliquid.xyz')}/info",
                        json=payload
                    )
                    
                    if response.status_code == 429 and attempt < max_retries - 1:
                        # Rate limit hit, apply exponential backoff with jitter
                        sleep_time = retry_delay * (2 ** attempt) * (0.5 + random.random())
                        logger.warning(f"Rate limit hit, retrying in {sleep_time:.2f} seconds (attempt {attempt+1}/{max_retries})")
                        time.sleep(sleep_time)
                        continue
                        
                    if response.status_code != 200:
                        logger.warning(f"Failed to get order book: {response.status_code}")
                        return {}
                        
                    return response.json()
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        sleep_time = retry_delay * (2 ** attempt) * (0.5 + random.random())
                        logger.warning(f"Rate limit hit, retrying in {sleep_time:.2f} seconds (attempt {attempt+1}/{max_retries})")
                        time.sleep(sleep_time)
                    else:
                        raise
            
            # If we get here, all retries failed
            logger.error(f"All retry attempts failed for order book for {symbol_name}")
            return {}
        except Exception as e:
            logger.error(f"Error getting order book for asset index {asset_index}: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

async def main():
    """
    Main function to test Hyperliquid adapter.
    """
    try:
        # Create adapter
        adapter = HyperliquidAdapter()
        
        # Get market data
        market_data = await adapter.get_market_data("BTC-USD-PERP")
        logger.info(f"Market data: {json.dumps(market_data, indent=4)}")
        
        # Get order book
        order_book = await adapter.get_order_book("BTC-USD-PERP")
        logger.info(f"Order book: {json.dumps(order_book, indent=4)}")
        
        # Get positions
        positions = await adapter.get_positions()
        logger.info(f"Positions: {json.dumps(positions, indent=4)}")
        
        # Get orders
        orders = await adapter.get_orders()
        logger.info(f"Orders: {json.dumps(orders, indent=4)}")
        
    except Exception as e:
        logger.error(f"Error testing Hyperliquid adapter: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())
