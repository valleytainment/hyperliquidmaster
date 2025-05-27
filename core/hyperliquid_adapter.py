"""
HyperliquidAdapter module for interfacing with the Hyperliquid exchange API.
This is a real implementation for live trading.
"""

import json
import os
from typing import Dict, List, Optional, Any, Union

from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

class HyperliquidAdapter:
    """
    Adapter for the Hyperliquid exchange API.
    Provides methods for trading, account management, and market data.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the adapter with configuration from the provided path.
        
        Args:
            config_path: Path to the configuration file containing API keys
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize API clients
        self.base_url = constants.MAINNET_API_URL  # Use mainnet by default
        self.info = Info(self.base_url, skip_ws=True)
        
        # Only initialize exchange if we have valid credentials
        self.exchange = None
        if self._has_valid_credentials():
            self._initialize_exchange()
    
    def _load_config(self) -> Dict:
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                # Create default config
                default_config = {
                    "account_address": "",
                    "secret_key": "",
                    "symbols": ["BTC", "ETH", "SOL"],
                    "theme": "dark"
                }
                
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                
                return default_config
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def _has_valid_credentials(self) -> bool:
        """Check if valid API credentials are available."""
        return (
            "account_address" in self.config and 
            "secret_key" in self.config and 
            self.config["account_address"] and 
            self.config["secret_key"]
        )
    
    def _initialize_exchange(self):
        """Initialize the exchange client with credentials."""
        try:
            self.exchange = Exchange(
                self.base_url,
                self.config["account_address"],
                self.config["secret_key"]
            )
        except Exception as e:
            print(f"Error initializing exchange: {e}")
            self.exchange = None
    
    def reload_config(self):
        """Reload configuration and reinitialize if needed."""
        self.config = self._load_config()
        if self._has_valid_credentials():
            self._initialize_exchange()
    
    def get_account_info(self) -> Dict:
        """
        Get account information including equity and positions.
        
        Returns:
            Dict containing account information
        """
        try:
            if not self._has_valid_credentials():
                return {"error": "No valid API credentials"}
            
            user_state = self.info.user_state(self.config["account_address"])
            
            # Extract relevant information
            account_info = {
                "equity": 0.0,
                "available_balance": 0.0,
                "positions": []
            }
            
            if "crossMarginSummary" in user_state:
                margin_summary = user_state["crossMarginSummary"]
                account_info["equity"] = float(margin_summary.get("equity", 0))
                account_info["available_balance"] = float(margin_summary.get("availableBalance", 0))
            
            if "assetPositions" in user_state:
                for position in user_state["assetPositions"]:
                    if "position" in position:
                        account_info["positions"].append(position["position"])
            
            return account_info
        except Exception as e:
            return {"error": f"Error getting account info: {e}"}
    
    def place_order(self, symbol: str, side: bool, size: float, price: Optional[float] = None, 
                   order_type: str = "LIMIT") -> Dict:
        """
        Place an order on the exchange.
        
        Args:
            symbol: Trading symbol (e.g., "BTC")
            side: True for buy, False for sell
            size: Order size
            price: Order price (required for limit orders)
            order_type: Order type ("LIMIT" or "MARKET")
            
        Returns:
            Dict containing order result
        """
        try:
            if not self.exchange:
                return {"error": "Exchange not initialized"}
            
            # Convert order type to Hyperliquid format
            if order_type == "MARKET":
                order_spec = {"market": {}}
            else:
                # Default to GTC limit order
                order_spec = {"limit": {"tif": "Gtc"}}
            
            # Place the order
            result = self.exchange.order(symbol, side, size, price if price else 0, order_spec)
            return result
        except Exception as e:
            return {"error": f"Error placing order: {e}"}
    
    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """
        Cancel an order by ID.
        
        Args:
            symbol: Trading symbol
            order_id: Order ID to cancel
            
        Returns:
            Dict containing cancel result
        """
        try:
            if not self.exchange:
                return {"error": "Exchange not initialized"}
            
            result = self.exchange.cancel(symbol, order_id)
            return result
        except Exception as e:
            return {"error": f"Error canceling order: {e}"}
    
    def get_positions(self) -> List[Dict]:
        """
        Get current positions.
        
        Returns:
            List of position dictionaries
        """
        try:
            account_info = self.get_account_info()
            return account_info.get("positions", [])
        except Exception as e:
            print(f"Error getting positions: {e}")
            return []
    
    def close_position(self, symbol: str, size_percentage: float = 100.0) -> Dict:
        """
        Close a position for a symbol.
        
        Args:
            symbol: Trading symbol
            size_percentage: Percentage of position to close (default: 100%)
            
        Returns:
            Dict containing close result
        """
        try:
            if not self.exchange:
                return {"error": "Exchange not initialized"}
            
            # Get current position
            positions = self.get_positions()
            position = None
            
            for pos in positions:
                if pos.get("coin") == symbol:
                    position = pos
                    break
            
            if not position:
                return {"error": f"No open position for {symbol}"}
            
            # Determine side and size
            position_size = float(position.get("szi", 0))
            if position_size == 0:
                return {"error": f"Position size is zero for {symbol}"}
            
            is_long = position_size > 0
            close_size = abs(position_size) * (size_percentage / 100.0)
            
            # Place order in opposite direction
            return self.place_order(symbol, not is_long, close_size, None, "MARKET")
        except Exception as e:
            return {"error": f"Error closing position: {e}"}
    
    def get_market_data(self, symbol: str) -> Dict:
        """
        Get market data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict containing market data
        """
        try:
            # Get market data from info endpoint
            meta_and_asset = self.info.meta_and_asset()
            
            # Find the asset in the response
            asset_info = None
            for asset in meta_and_asset.get("universe", []):
                if asset.get("name") == symbol:
                    asset_info = asset
                    break
            
            if not asset_info:
                return {"error": f"Symbol {symbol} not found"}
            
            # Get order book for the symbol
            order_book = self.info.l2_snapshot(symbol)
            
            # Construct market data
            market_data = {
                "symbol": symbol,
                "price": float(asset_info.get("oraclePrice", 0)),
                "funding_rate": float(asset_info.get("funding", {}).get("fundingRate", 0)),
                "open_interest": float(asset_info.get("openInterest", {}).get("szi", 0)),
                "order_book": order_book
            }
            
            return market_data
        except Exception as e:
            return {"error": f"Error getting market data: {e}"}
    
    def get_order_status(self, order_id: str) -> Dict:
        """
        Get status of an order by ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            Dict containing order status
        """
        try:
            if not self._has_valid_credentials():
                return {"error": "No valid API credentials"}
            
            order_status = self.info.query_order_by_oid(self.config["account_address"], order_id)
            return order_status
        except Exception as e:
            return {"error": f"Error getting order status: {e}"}
