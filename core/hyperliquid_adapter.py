"""
HyperliquidAdapter module for connecting to the Hyperliquid exchange API.
Provides direct integration with the Hyperliquid exchange for trading operations.
"""

import os
import json
import time
import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple

# Import eth_account for LocalAccount creation
import eth_account
from eth_account.signers.local import LocalAccount

# Import Hyperliquid SDK - fixed import paths
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info

class HyperliquidAdapter:
    """
    Adapter for the Hyperliquid exchange API.
    Provides methods for interacting with the exchange.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the Hyperliquid adapter.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.logger = logging.getLogger("HyperliquidAdapter")
        self.exchange = None
        self.info = None
        self.account_address = ""
        self.secret_key = ""
        self.is_connected = False
        self.api_url = "https://api.hyperliquid.xyz"
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize API if keys are available
        self._init_api()
    
    def _load_config(self) -> Dict:
        """
        Load configuration from file.
        
        Returns:
            Dict containing the configuration
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self.config = self._load_config()
        self._init_api()
    
    def _init_api(self) -> None:
        """Initialize the Hyperliquid API with current configuration."""
        try:
            # Get API keys from config
            self.account_address = self.config.get("account_address", "")
            self.secret_key = self.config.get("secret_key", "")
            self.api_url = self.config.get("api_url", "https://api.hyperliquid.xyz")
            
            # Check if keys are available
            if not self.account_address or not self.secret_key:
                self.is_connected = False
                return
            
            # Initialize API
            try:
                # Create LocalAccount from secret key
                account: LocalAccount = eth_account.Account.from_key(self.secret_key)
                
                # Initialize Info and Exchange
                self.info = Info(base_url=self.api_url)
                self.exchange = Exchange(
                    wallet=account,
                    base_url=self.api_url,
                    account_address=self.account_address
                )
                self.is_connected = True
                self.logger.info("API initialized successfully")
            except Exception as e:
                self.logger.error(f"Error initializing exchange: {e}")
                self.is_connected = False
        except Exception as e:
            self.logger.error(f"Error initializing API: {e}")
            self.is_connected = False
    
    def set_api_keys(self, account_address: str, secret_key: str) -> bool:
        """
        Set API keys and initialize the API.
        
        Args:
            account_address: The account address
            secret_key: The secret key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update config
            self.config["account_address"] = account_address
            self.config["secret_key"] = secret_key
            
            # Save config
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            # Initialize API
            self._init_api()
            
            return True
        except Exception as e:
            self.logger.error(f"Error setting API keys: {e}")
            return False
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to the exchange.
        
        Returns:
            Dict containing the result of the test
        """
        try:
            if not self.is_connected:
                return {"error": "Not connected to exchange"}
            
            # Test connection by getting user state
            user_state = self.info.user_state(self.account_address)
            
            if "error" in user_state:
                self.logger.error(f"Connection test failed: {user_state['error']}")
                return {"error": f"Connection test failed: {user_state['error']}"}
            
            return {"success": "Connection test successful"}
        except Exception as e:
            self.logger.error(f"Error testing connection: {e}")
            return {"error": f"Error testing connection: {e}"}
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dict containing account information
        """
        try:
            if not self.is_connected:
                return {"error": "Not connected to exchange"}
            
            # Get user state
            user_state = self.info.user_state(self.account_address)
            
            if "error" in user_state:
                self.logger.error(f"Error getting account info: {user_state['error']}")
                return {"error": f"Error getting account info: {user_state['error']}"}
            
            # Extract account info
            margin_summary = user_state.get("marginSummary", {})
            account_info = {
                "equity": float(margin_summary.get("accountValue", 0)),
                "margin": float(margin_summary.get("totalMargin", 0)),
                "free_margin": float(margin_summary.get("marginAvailable", 0)),
                "margin_ratio": float(margin_summary.get("marginRatio", 0)),
                "unrealized_pnl": float(margin_summary.get("unrealizedPnl", 0)),
                "realized_pnl": float(margin_summary.get("realizedPnl", 0)),
                "total_pnl": float(margin_summary.get("totalPnl", 0)),
                "wallet_balance": float(margin_summary.get("walletBalance", 0))
            }
            
            return account_info
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {"error": f"Error getting account info: {e}"}
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get market data for a symbol.
        
        Args:
            symbol: The symbol to get market data for
            
        Returns:
            Dict containing market data
        """
        try:
            if not self.info:
                return {"error": "Info API not initialized"}
            
            if not symbol or symbol.strip() == "":
                return {"error": "Symbol cannot be empty"}
            
            # Get market data
            try:
                meta_and_asset = self.info.meta_and_asset()
                
                # Find the symbol in the meta data
                symbol_data = None
                for asset in meta_and_asset.get("universe", []):
                    if asset.get("name") == symbol:
                        symbol_data = asset
                        break
                
                if not symbol_data:
                    self.logger.error(f"Symbol {symbol} not found")
                    return {"error": f"Symbol {symbol} not found"}
                
                # Get market data for the symbol
                market_data = self.info.market_data(symbol)
                
                # Extract market data
                return {
                    "symbol": symbol,
                    "price": float(market_data.get("midPrice", 0)),
                    "mark_price": float(market_data.get("markPrice", 0)),
                    "index_price": float(market_data.get("indexPrice", 0)),
                    "funding_rate": float(market_data.get("fundingRate", 0)),
                    "open_interest": float(market_data.get("openInterest", 0)),
                    "volume_24h": float(market_data.get("volume24h", 0)),
                    "price_change_24h": float(market_data.get("priceChange24h", 0))
                }
            except Exception as e:
                self.logger.error(f"Error getting market data: {e}")
                return {"error": f"Error getting market data: {e}"}
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return {"error": f"Error getting market data: {e}"}
    
    def get_positions(self) -> Dict[str, Any]:
        """
        Get current positions.
        
        Returns:
            Dict containing positions
        """
        try:
            if not self.is_connected:
                return {"error": "Not connected to exchange"}
            
            # Get user state
            user_state = self.info.user_state(self.account_address)
            
            if "error" in user_state:
                self.logger.error(f"Error getting positions: {user_state['error']}")
                return {"error": f"Error getting positions: {user_state['error']}"}
            
            # Extract positions
            positions = user_state.get("assetPositions", [])
            
            return {"success": True, "data": positions}
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {"error": f"Error getting positions: {e}"}
    
    def get_orders(self) -> Dict[str, Any]:
        """
        Get current orders.
        
        Returns:
            Dict containing orders
        """
        try:
            if not self.is_connected:
                return {"error": "Not connected to exchange"}
            
            # Get open orders
            orders = self.info.open_orders(self.account_address)
            
            if "error" in orders:
                self.logger.error(f"Error getting orders: {orders['error']}")
                return {"error": f"Error getting orders: {orders['error']}"}
            
            return {"success": True, "data": orders}
        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            return {"error": f"Error getting orders: {e}"}
    
    def place_order(self, symbol: str, is_buy: bool, size: float, price: float, order_type: str = "LIMIT") -> Dict[str, Any]:
        """
        Place an order.
        
        Args:
            symbol: The symbol to place an order for
            is_buy: Whether the order is a buy order
            size: The size of the order
            price: The price of the order
            order_type: The type of order (LIMIT, MARKET)
            
        Returns:
            Dict containing the result of the operation
        """
        try:
            if not self.is_connected:
                return {"error": "Not connected to exchange"}
            
            # Place order
            try:
                # Prepare order type
                order_spec = {"limit": {"tif": "Gtc"}} if order_type == "LIMIT" else "market"
                
                # Place order
                order_result = self.exchange.order(
                    name=symbol,
                    is_buy=is_buy,
                    sz=size,
                    limit_px=price,
                    order_type=order_spec
                )
                
                if order_result.get("status") != "ok":
                    self.logger.error(f"Error placing order: {order_result}")
                    return {"error": f"Error placing order: {order_result}"}
                
                return {"success": True, "data": order_result}
            except Exception as e:
                self.logger.error(f"Error placing order: {e}")
                return {"error": f"Error placing order: {e}"}
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {"error": f"Error placing order: {e}"}
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            order_id: The ID of the order to cancel
            
        Returns:
            Dict containing the result of the operation
        """
        try:
            if not self.is_connected:
                return {"error": "Not connected to exchange"}
            
            # Cancel order
            try:
                cancel_result = self.exchange.cancel(oid=order_id)
                
                if cancel_result.get("status") != "ok":
                    self.logger.error(f"Error canceling order: {cancel_result}")
                    return {"error": f"Error canceling order: {cancel_result}"}
                
                return {"success": True, "data": cancel_result}
            except Exception as e:
                self.logger.error(f"Error canceling order: {e}")
                return {"error": f"Error canceling order: {e}"}
        except Exception as e:
            self.logger.error(f"Error canceling order: {e}")
            return {"error": f"Error canceling order: {e}"}
    
    def cancel_all_orders(self, symbol: str = None) -> Dict[str, Any]:
        """
        Cancel all orders.
        
        Args:
            symbol: The symbol to cancel orders for (optional)
            
        Returns:
            Dict containing the result of the operation
        """
        try:
            if not self.is_connected:
                return {"error": "Not connected to exchange"}
            
            # Cancel all orders
            try:
                cancel_result = self.exchange.cancel_all(coin=symbol)
                
                if cancel_result.get("status") != "ok":
                    self.logger.error(f"Error canceling all orders: {cancel_result}")
                    return {"error": f"Error canceling all orders: {cancel_result}"}
                
                return {"success": True, "data": cancel_result}
            except Exception as e:
                self.logger.error(f"Error canceling all orders: {e}")
                return {"error": f"Error canceling all orders: {e}"}
        except Exception as e:
            self.logger.error(f"Error canceling all orders: {e}")
            return {"error": f"Error canceling all orders: {e}"}
    
    def close_position(self, symbol: str, size_percentage: float = 100.0) -> Dict[str, Any]:
        """
        Close a position.
        
        Args:
            symbol: The symbol to close the position for
            size_percentage: The percentage of the position to close (0-100)
            
        Returns:
            Dict containing the result of the operation
        """
        try:
            if not self.is_connected:
                return {"error": "Not connected to exchange"}
            
            # Get positions
            positions_result = self.get_positions()
            
            if "error" in positions_result:
                return positions_result
            
            positions = positions_result.get("data", [])
            
            # Find the position for the symbol
            position = None
            for pos in positions:
                if pos.get("coin") == symbol:
                    position = pos
                    break
            
            if not position:
                self.logger.error(f"No position found for {symbol}")
                return {"error": f"No position found for {symbol}"}
            
            # Calculate size to close
            position_data = position.get("position", {})
            size = float(position_data.get("szi", 0))
            if size == 0:
                self.logger.error(f"Position size is 0 for {symbol}")
                return {"error": f"Position size is 0 for {symbol}"}
            
            close_size = size * (size_percentage / 100.0)
            is_long = size > 0
            
            # Get current market price
            market_data = self.get_market_data(symbol)
            
            if "error" in market_data:
                return market_data
            
            price = market_data["price"]
            
            # Place order to close position
            return self.place_order(
                symbol=symbol,
                is_buy=not is_long,  # Opposite of position direction
                size=abs(close_size),
                price=price,
                order_type="LIMIT"
            )
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return {"error": f"Error closing position: {e}"}
    
    def get_available_symbols(self) -> List[str]:
        """
        Get a list of available trading symbols.
        
        Returns:
            List of available symbols
        """
        try:
            if not self.info:
                return []
            
            # Get meta data
            try:
                meta_and_asset = self.info.meta_and_asset()
                
                # Extract symbols
                symbols = []
                for asset in meta_and_asset.get("universe", []):
                    symbol = asset.get("name")
                    if symbol:
                        symbols.append(symbol)
                
                return symbols
            except Exception as e:
                self.logger.error(f"Error getting available symbols: {e}")
                return []
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []
