"""
Modified HyperliquidAdapter to use the new Pydantic-based configuration.

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

from hyperliquidmaster.config import BotSettings

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
    
    def __init__(self, settings: BotSettings):
        """
        Initialize Hyperliquid adapter.
        
        Args:
            settings: Bot configuration settings
        """
        self.settings = settings
        
        # Initialize API clients
        self.info = Info(base_url=self.settings.base_url)
        
        # Initialize account if private key is provided
        if self.settings.secret_key:
            self.account = Account.from_key(self.settings.secret_key)
            self.exchange = Exchange(
                base_url=self.settings.base_url,
                private_key=self.settings.secret_key
            )
            logger.info("Initialized exchange client with account")
        else:
            self.account = None
            self.exchange = None
            logger.warning("No private key provided, trading functionality will be limited")
        
        # Cache for market data
        self.market_data_cache = {}
        self.last_update_time = {}
        
        logger.info("Hyperliquid adapter initialized")
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get market data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Market data dictionary
        """
        try:
            # Check cache first
            current_time = time.time()
            if (symbol in self.market_data_cache and 
                symbol in self.last_update_time and 
                current_time - self.last_update_time[symbol] < 5):  # 5 second cache
                return self.market_data_cache[symbol]
            
            # Fetch market data
            meta = await self.info.meta()
            markets = meta.get("universe", [])
            
            # Find the requested symbol
            market_data = {}
            for market in markets:
                if market.get("name") == symbol:
                    # Get current price
                    candles = await self.info.candles(symbol)
                    if candles and len(candles) > 0:
                        latest_candle = candles[-1]
                        market_data["last_price"] = float(latest_candle[4])  # Close price
                    
                    # Get funding rate
                    funding = await self.info.funding_info(symbol)
                    if funding:
                        market_data["funding_rate"] = float(funding.get("fundingRate", 0))
                    
                    # Get other market info
                    market_data["base_asset"] = symbol
                    market_data["quote_asset"] = "USD"
                    market_data["timestamp"] = int(time.time() * 1000)
                    
                    # Update cache
                    self.market_data_cache[symbol] = market_data
                    self.last_update_time[symbol] = current_time
                    
                    return market_data
            
            logger.warning(f"Symbol {symbol} not found in markets")
            return {}
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return {}
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Account information dictionary
        """
        if not self.exchange:
            logger.error("Exchange client not initialized, cannot get account info")
            return {}
        
        try:
            # Get account information
            account_info = await self.exchange.user_state()
            
            # Process and return relevant information
            return {
                "account_id": self.settings.account_address,
                "equity": float(account_info.get("crossMarginSummary", {}).get("accountValue", 0)),
                "available_balance": float(account_info.get("crossMarginSummary", {}).get("accountValue", 0)),
                "positions": self._process_positions(account_info.get("assetPositions", [])),
                "timestamp": int(time.time() * 1000)
            }
            
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return {}
    
    def _process_positions(self, positions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Process positions data.
        
        Args:
            positions: List of position data from API
            
        Returns:
            Processed positions dictionary
        """
        result = {}
        
        for position in positions:
            symbol = position.get("name", "")
            if symbol:
                result[symbol] = {
                    "symbol": symbol,
                    "size": float(position.get("position", {}).get("size", 0)),
                    "entry_price": float(position.get("position", {}).get("entryPx", 0)),
                    "mark_price": float(position.get("position", {}).get("markPx", 0)),
                    "liquidation_price": float(position.get("position", {}).get("liquidationPx", 0)),
                    "unrealized_pnl": float(position.get("position", {}).get("unrealizedPnl", 0)),
                    "leverage": float(position.get("position", {}).get("leverage", 0)),
                    "side": "long" if float(position.get("position", {}).get("size", 0)) > 0 else "short"
                }
        
        return result
    
    async def place_order(self, 
                         symbol: str, 
                         side: str, 
                         order_type: str, 
                         quantity: float, 
                         price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place an order.
        
        Args:
            symbol: Trading symbol
            side: Order side ("buy" or "sell")
            order_type: Order type ("market" or "limit")
            quantity: Order quantity
            price: Order price (required for limit orders)
            
        Returns:
            Order response dictionary
        """
        if not self.exchange:
            logger.error("Exchange client not initialized, cannot place order")
            return {"success": False, "error": "Exchange client not initialized"}
        
        try:
            # Validate inputs
            if side.lower() not in ["buy", "sell"]:
                return {"success": False, "error": "Invalid side, must be 'buy' or 'sell'"}
            
            if order_type.lower() not in ["market", "limit"]:
                return {"success": False, "error": "Invalid order type, must be 'market' or 'limit'"}
            
            if order_type.lower() == "limit" and price is None:
                return {"success": False, "error": "Price is required for limit orders"}
            
            # Prepare order
            order_side = "B" if side.lower() == "buy" else "A"
            order_data = {
                "coin": symbol,
                "is_buy": side.lower() == "buy",
                "sz": abs(quantity),
                "limit_px": price if order_type.lower() == "limit" else 0,
                "order_type": {"limit": {"tif": "Gtc"}} if order_type.lower() == "limit" else {"market": {}},
                "reduce_only": False
            }
            
            # Place order
            response = await self.exchange.order(order_data)
            
            # Process response
            if "status" in response and response["status"] == "ok":
                logger.info(f"Order placed successfully: {symbol} {side} {quantity} @ {price}")
                return {
                    "success": True,
                    "order_id": response.get("data", {}).get("oid", ""),
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "order_type": order_type,
                    "timestamp": int(time.time() * 1000)
                }
            else:
                logger.error(f"Order placement failed: {response}")
                return {
                    "success": False,
                    "error": str(response),
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "order_type": order_type,
                    "timestamp": int(time.time() * 1000)
                }
                
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            symbol: Trading symbol
            order_id: Order ID to cancel
            
        Returns:
            Cancel response dictionary
        """
        if not self.exchange:
            logger.error("Exchange client not initialized, cannot cancel order")
            return {"success": False, "error": "Exchange client not initialized"}
        
        try:
            # Prepare cancel request
            cancel_data = {
                "coin": symbol,
                "oid": order_id
            }
            
            # Cancel order
            response = await self.exchange.cancel_order(cancel_data)
            
            # Process response
            if "status" in response and response["status"] == "ok":
                logger.info(f"Order cancelled successfully: {symbol} {order_id}")
                return {
                    "success": True,
                    "order_id": order_id,
                    "symbol": symbol,
                    "timestamp": int(time.time() * 1000)
                }
            else:
                logger.error(f"Order cancellation failed: {response}")
                return {
                    "success": False,
                    "error": str(response),
                    "order_id": order_id,
                    "symbol": symbol,
                    "timestamp": int(time.time() * 1000)
                }
                
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_order_book(self, symbol: str, depth: int = 10) -> Dict[str, Any]:
        """
        Get order book for a symbol.
        
        Args:
            symbol: Trading symbol
            depth: Order book depth
            
        Returns:
            Order book dictionary
        """
        try:
            # Get order book
            order_book = await self.info.l2_snapshot(symbol)
            
            # Process and return relevant information
            if order_book:
                return {
                    "symbol": symbol,
                    "bids": [{"price": float(bid[0]), "quantity": float(bid[1])} for bid in order_book.get("bids", [])[:depth]],
                    "asks": [{"price": float(ask[0]), "quantity": float(ask[1])} for ask in order_book.get("asks", [])[:depth]],
                    "timestamp": int(time.time() * 1000)
                }
            else:
                return {"symbol": symbol, "bids": [], "asks": [], "timestamp": int(time.time() * 1000)}
                
        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {str(e)}")
            return {"symbol": symbol, "bids": [], "asks": [], "timestamp": int(time.time() * 1000)}
    
    async def get_recent_trades(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent trades for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Maximum number of trades to return
            
        Returns:
            List of recent trades
        """
        try:
            # Get recent trades
            trades = await self.info.recent_trades(symbol)
            
            # Process and return relevant information
            result = []
            for trade in trades[:limit]:
                result.append({
                    "symbol": symbol,
                    "id": trade.get("tid", ""),
                    "price": float(trade.get("px", 0)),
                    "quantity": float(trade.get("sz", 0)),
                    "side": "buy" if trade.get("side", "") == "B" else "sell",
                    "timestamp": int(trade.get("time", 0))
                })
            
            return result
                
        except Exception as e:
            logger.error(f"Error getting recent trades for {symbol}: {str(e)}")
            return []
