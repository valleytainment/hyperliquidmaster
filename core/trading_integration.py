"""
Trading integration module for connecting the GUI with the Hyperliquid exchange.
Provides a unified interface for all trading operations.
"""

import os
import json
import time
import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple

# Import Hyperliquid SDK
try:
    from hyperliquid import HyperliquidApi, Info, Exchange
except ImportError:
    # Mock implementation for testing
    class HyperliquidApi:
        def __init__(self, *args, **kwargs):
            pass
    
    class Info:
        def __init__(self, *args, **kwargs):
            pass
        
        def meta_and_asset(self, *args, **kwargs):
            return {"error": "Mock implementation"}
    
    class Exchange:
        def __init__(self, *args, **kwargs):
            pass
        
        def user_state(self, *args, **kwargs):
            return {"error": "Mock implementation"}
        
        def place_order(self, *args, **kwargs):
            return {"error": "Mock implementation"}
        
        def cancel_order(self, *args, **kwargs):
            return {"error": "Mock implementation"}
        
        def cancel_all(self, *args, **kwargs):
            return {"error": "Mock implementation"}

class TradingIntegration:
    """
    Integrates the GUI with the Hyperliquid exchange.
    Provides a unified interface for all trading operations.
    """
    
    def __init__(self, config_path: str, logger: logging.Logger):
        """
        Initialize the trading integration.
        
        Args:
            config_path: Path to the configuration file
            logger: Logger instance for logging
        """
        self.config_path = config_path
        self.logger = logger
        self.is_connected = False
        self.exchange = None
        self.info = None
        self.account_address = ""
        self.secret_key = ""
        
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
            
            # Check if keys are available
            if not self.account_address or not self.secret_key:
                self.is_connected = False
                return
            
            # Initialize API
            api_url = self.config.get("api_url", "https://api.hyperliquid.xyz")
            
            try:
                self.info = Info(base_url=api_url)
                self.exchange = Exchange(
                    base_url=api_url,
                    wallet_address=self.account_address,
                    private_key=self.secret_key
                )
                self.is_connected = True
                self.logger.info("API initialized successfully")
            except Exception as e:
                self.logger.error(f"Error initializing exchange: {e}")
                self.is_connected = False
        except Exception as e:
            self.logger.error(f"Error initializing API: {e}")
            self.is_connected = False
    
    def set_api_keys(self, account_address: str, secret_key: str) -> Dict[str, Any]:
        """
        Set API keys and initialize the API.
        
        Args:
            account_address: The account address
            secret_key: The secret key
            
        Returns:
            Dict containing the result of the operation
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
            
            self.logger.info("API keys updated successfully")
            return {"success": True, "message": "API keys updated successfully"}
        except Exception as e:
            self.logger.error(f"Error setting API keys: {e}")
            return {"success": False, "message": f"Error setting API keys: {e}"}
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to the exchange.
        
        Returns:
            Dict containing the result of the test
        """
        try:
            if not self.is_connected:
                return {"success": False, "message": "Not connected to exchange"}
            
            # Test connection by getting user state
            user_state = self.exchange.user_state()
            
            if "error" in user_state:
                self.logger.error(f"Connection test failed: {user_state['error']}")
                return {"success": False, "message": f"Connection test failed: {user_state['error']}"}
            
            return {"success": True, "message": "Connection test successful"}
        except Exception as e:
            self.logger.error(f"Error testing connection: {e}")
            return {"success": False, "message": f"Error testing connection: {e}"}
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dict containing account information
        """
        try:
            if not self.is_connected:
                return {"success": False, "message": "Not connected to exchange", "data": {}}
            
            # Get user state
            user_state = self.exchange.user_state()
            
            if "error" in user_state:
                self.logger.error(f"Error getting account info: {user_state['error']}")
                return {"success": False, "message": f"Error getting account info: {user_state['error']}", "data": {}}
            
            # Extract account info
            account_info = {
                "equity": float(user_state.get("crossMarginSummary", {}).get("equity", 0)),
                "margin": float(user_state.get("crossMarginSummary", {}).get("margin", 0)),
                "free_margin": float(user_state.get("crossMarginSummary", {}).get("freeMargin", 0)),
                "margin_ratio": float(user_state.get("crossMarginSummary", {}).get("marginRatio", 0)),
                "unrealized_pnl": float(user_state.get("crossMarginSummary", {}).get("unrealizedPnl", 0)),
                "realized_pnl": float(user_state.get("crossMarginSummary", {}).get("realizedPnl", 0)),
                "total_pnl": float(user_state.get("crossMarginSummary", {}).get("totalPnl", 0)),
                "wallet_balance": float(user_state.get("crossMarginSummary", {}).get("walletBalance", 0))
            }
            
            return {"success": True, "message": "Account info retrieved successfully", "data": account_info}
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {"success": False, "message": f"Error getting account info: {e}", "data": {}}
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get market data for a symbol.
        
        Args:
            symbol: The symbol to get market data for
            
        Returns:
            Dict containing market data
        """
        try:
            if not self.is_connected:
                return {"success": False, "message": "Not connected to exchange", "data": {}}
            
            # Get market data
            try:
                meta_and_asset = self.info.meta_and_asset()
                
                # Find the symbol in the meta data
                symbol_data = None
                for asset in meta_and_asset.get("assetCtxs", []):
                    if asset.get("name") == symbol:
                        symbol_data = asset
                        break
                
                if not symbol_data:
                    self.logger.error(f"Symbol {symbol} not found")
                    return {"success": False, "message": f"Symbol {symbol} not found", "data": {}}
                
                # Extract market data
                market_data = {
                    "symbol": symbol,
                    "price": float(symbol_data.get("oraclePx", 0)),
                    "mark_price": float(symbol_data.get("markPx", 0)),
                    "index_price": float(symbol_data.get("indexPx", 0)),
                    "funding_rate": float(symbol_data.get("fundingRate", 0)),
                    "open_interest": float(symbol_data.get("openInterest", 0)),
                    "volume_24h": float(symbol_data.get("dailyVolume", 0)),
                    "price_change_24h": float(symbol_data.get("dailyPxChg", 0))
                }
                
                return {"success": True, "message": "Market data retrieved successfully", "data": market_data}
            except Exception as e:
                self.logger.error(f"Error getting market data: {e}")
                return {"success": False, "message": f"Error getting market data: {e}", "data": {}}
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return {"success": False, "message": f"Error getting market data: {e}", "data": {}}
    
    def get_positions(self) -> Dict[str, Any]:
        """
        Get current positions.
        
        Returns:
            Dict containing positions
        """
        try:
            if not self.is_connected:
                return {"success": False, "message": "Not connected to exchange", "data": []}
            
            # Get user state
            user_state = self.exchange.user_state()
            
            if "error" in user_state:
                self.logger.error(f"Error getting positions: {user_state['error']}")
                return {"success": False, "message": f"Error getting positions: {user_state['error']}", "data": []}
            
            # Extract positions
            positions = user_state.get("assetPositions", [])
            
            return {"success": True, "message": "Positions retrieved successfully", "data": positions}
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {"success": False, "message": f"Error getting positions: {e}", "data": []}
    
    def get_orders(self) -> Dict[str, Any]:
        """
        Get current orders.
        
        Returns:
            Dict containing orders
        """
        try:
            if not self.is_connected:
                return {"success": False, "message": "Not connected to exchange", "data": []}
            
            # Get user state
            user_state = self.exchange.user_state()
            
            if "error" in user_state:
                self.logger.error(f"Error getting orders: {user_state['error']}")
                return {"success": False, "message": f"Error getting orders: {user_state['error']}", "data": []}
            
            # Extract orders
            orders = user_state.get("openOrders", [])
            
            return {"success": True, "message": "Orders retrieved successfully", "data": orders}
        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            return {"success": False, "message": f"Error getting orders: {e}", "data": []}
    
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
                return {"success": False, "message": "Not connected to exchange", "data": {}}
            
            # Prepare order
            side = "B" if is_buy else "A"
            
            # Place order
            try:
                order_result = self.exchange.place_order(
                    coin=symbol,
                    is_buy=is_buy,
                    sz=size,
                    limit_px=price,
                    order_type=order_type
                )
                
                if "error" in order_result:
                    self.logger.error(f"Error placing order: {order_result['error']}")
                    return {"success": False, "message": f"Error placing order: {order_result['error']}", "data": {}}
                
                return {"success": True, "message": "Order placed successfully", "data": order_result}
            except Exception as e:
                self.logger.error(f"Error placing order: {e}")
                return {"success": False, "message": f"Error placing order: {e}", "data": {}}
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {"success": False, "message": f"Error placing order: {e}", "data": {}}
    
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
                return {"success": False, "message": "Not connected to exchange", "data": {}}
            
            # Cancel order
            try:
                cancel_result = self.exchange.cancel_order(order_id=order_id)
                
                if "error" in cancel_result:
                    self.logger.error(f"Error canceling order: {cancel_result['error']}")
                    return {"success": False, "message": f"Error canceling order: {cancel_result['error']}", "data": {}}
                
                return {"success": True, "message": "Order canceled successfully", "data": cancel_result}
            except Exception as e:
                self.logger.error(f"Error canceling order: {e}")
                return {"success": False, "message": f"Error canceling order: {e}", "data": {}}
        except Exception as e:
            self.logger.error(f"Error canceling order: {e}")
            return {"success": False, "message": f"Error canceling order: {e}", "data": {}}
    
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
                return {"success": False, "message": "Not connected to exchange", "data": {}}
            
            # Cancel all orders
            try:
                cancel_result = self.exchange.cancel_all(coin=symbol)
                
                if "error" in cancel_result:
                    self.logger.error(f"Error canceling all orders: {cancel_result['error']}")
                    return {"success": False, "message": f"Error canceling all orders: {cancel_result['error']}", "data": {}}
                
                return {"success": True, "message": "All orders canceled successfully", "data": cancel_result}
            except Exception as e:
                self.logger.error(f"Error canceling all orders: {e}")
                return {"success": False, "message": f"Error canceling all orders: {e}", "data": {}}
        except Exception as e:
            self.logger.error(f"Error canceling all orders: {e}")
            return {"success": False, "message": f"Error canceling all orders: {e}", "data": {}}
    
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
                return {"success": False, "message": "Not connected to exchange", "data": {}}
            
            # Get positions
            positions_result = self.get_positions()
            
            if not positions_result["success"]:
                return positions_result
            
            # Find the position for the symbol
            position = None
            for pos in positions_result["data"]:
                if pos.get("coin") == symbol:
                    position = pos
                    break
            
            if not position:
                self.logger.error(f"No position found for {symbol}")
                return {"success": False, "message": f"No position found for {symbol}", "data": {}}
            
            # Calculate size to close
            size = float(position.get("szi", 0))
            if size == 0:
                self.logger.error(f"Position size is 0 for {symbol}")
                return {"success": False, "message": f"Position size is 0 for {symbol}", "data": {}}
            
            close_size = size * (size_percentage / 100.0)
            is_long = size > 0
            
            # Get current market price
            market_data_result = self.get_market_data(symbol)
            
            if not market_data_result["success"]:
                return market_data_result
            
            price = market_data_result["data"]["price"]
            
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
            return {"success": False, "message": f"Error closing position: {e}", "data": {}}
