"""
Trading integration module for the HyperliquidMaster trading bot.
Provides a unified interface for interacting with the Hyperliquid exchange.
"""

import os
import json
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Union, Tuple

from core.hyperliquid_adapter import HyperliquidAdapter
from core.error_handler import ErrorHandler

class TradingIntegration:
    """
    Trading integration class for the HyperliquidMaster trading bot.
    Provides a unified interface for interacting with the Hyperliquid exchange.
    """
    
    def __init__(self, config_path: str, logger: logging.Logger = None):
        """
        Initialize the trading integration.
        
        Args:
            config_path: Path to the configuration file
            logger: Logger instance
        """
        self.config_path = config_path
        self.logger = logger or logging.getLogger("TradingIntegration")
        self.adapter = HyperliquidAdapter(config_path)
        self.error_handler = ErrorHandler(self.logger)
        self.is_connected = self.adapter.is_connected
        
        # Connection state tracking
        self.connection_lock = threading.Lock()
        self.last_connection_check = 0
        self.connection_check_interval = 30  # seconds
        
        # SDK compatibility
        self.sdk_version = self._detect_sdk_version()
    
    def _detect_sdk_version(self) -> str:
        """
        Detect the version of the Hyperliquid SDK.
        
        Returns:
            String representing the SDK version
        """
        try:
            # Try to import the SDK version
            try:
                from hyperliquid import __version__
                return __version__
            except (ImportError, AttributeError):
                # If version is not available, try to detect based on API structure
                if hasattr(self.adapter.info, "meta_and_asset"):
                    return "1.0+"
                else:
                    return "unknown"
        except Exception as e:
            self.logger.warning(f"Error detecting SDK version: {e}")
            return "unknown"
    
    def _ensure_connection(self) -> bool:
        """
        Ensure that the connection to the exchange is active.
        
        Returns:
            True if connected, False otherwise
        """
        with self.connection_lock:
            current_time = time.time()
            
            # Check if we need to verify connection
            if current_time - self.last_connection_check > self.connection_check_interval:
                self.last_connection_check = current_time
                
                # Update connection status from adapter
                self.is_connected = self.adapter.is_connected
                
                # If not connected, try to reconnect
                if not self.is_connected:
                    self.logger.warning("Connection lost, attempting to reconnect")
                    result = self.test_connection()
                    self.is_connected = result.get("success", False)
            
            return self.is_connected
    
    def _safe_api_call(self, api_func, *args, **kwargs) -> Dict[str, Any]:
        """
        Safely call an API function with error handling.
        
        Args:
            api_func: Function to call
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the API call
        """
        try:
            # Ensure connection before making API call
            if not self._ensure_connection():
                return {"success": False, "message": "Not connected to exchange"}
            
            # Call the API function
            result = api_func(*args, **kwargs)
            
            # Handle error responses
            if isinstance(result, dict) and "error" in result:
                error_msg = result["error"]
                self.logger.error(f"API error: {error_msg}")
                
                # Check if it's a connection error
                if "connect" in str(error_msg).lower() or "timeout" in str(error_msg).lower():
                    self.is_connected = False
                
                return {"success": False, "message": error_msg}
            
            return {"success": True, "data": result}
        except Exception as e:
            # Handle exceptions
            error_info = self.error_handler.handle_error(e, {
                "function": api_func.__name__ if hasattr(api_func, "__name__") else "unknown",
                "args": args,
                "kwargs": kwargs
            })
            
            # Check if it's a connection error
            if error_info["retry_recommended"]:
                self.is_connected = False
            
            return {"success": False, "message": str(e), "error_info": error_info}
    
    def _safe_get_attribute(self, obj: Any, attr_name: str, default: Any = None) -> Any:
        """
        Safely get an attribute from an object.
        
        Args:
            obj: Object to get attribute from
            attr_name: Name of the attribute
            default: Default value if attribute doesn't exist
            
        Returns:
            Attribute value or default
        """
        try:
            if obj is None:
                return default
            
            if hasattr(obj, attr_name):
                return getattr(obj, attr_name)
            elif isinstance(obj, dict) and attr_name in obj:
                return obj[attr_name]
            else:
                return default
        except Exception as e:
            self.logger.warning(f"Error getting attribute {attr_name}: {e}")
            return default
    
    def _safe_get_nested(self, data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
        """
        Safely get a nested value from a dictionary.
        
        Args:
            data: Dictionary to get value from
            keys: List of keys to traverse
            default: Default value if key doesn't exist
            
        Returns:
            Nested value or default
        """
        if not isinstance(data, dict):
            return default
        
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def _safe_float_convert(self, value: Any, default: float = 0.0) -> float:
        """
        Safely convert a value to float.
        
        Args:
            value: Value to convert
            default: Default value if conversion fails
            
        Returns:
            Float value or default
        """
        try:
            if value is None:
                return default
            return float(value)
        except (ValueError, TypeError):
            return default
    
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
            result = self.adapter.set_api_keys(account_address, secret_key)
            self.is_connected = self.adapter.is_connected
            
            if result:
                return {"success": True, "message": "API keys set successfully"}
            else:
                return {"success": False, "message": "Failed to set API keys"}
        except Exception as e:
            error_info = self.error_handler.handle_error(e, {
                "function": "set_api_keys",
                "account_address": account_address[:5] + "..." if account_address else None
            })
            error_msg = f"Error setting API keys: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg, "error_info": error_info}
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to the exchange.
        
        Returns:
            Dict containing the result of the test
        """
        try:
            result = self.adapter.test_connection()
            
            if "error" in result:
                self.is_connected = False
                return {"success": False, "message": result["error"]}
            else:
                self.is_connected = True
                self.last_connection_check = time.time()
                return {"success": True, "message": "Connection test successful"}
        except Exception as e:
            error_info = self.error_handler.handle_error(e, {"function": "test_connection"})
            error_msg = f"Error testing connection: {e}"
            self.logger.error(error_msg)
            self.is_connected = False
            return {"success": False, "message": error_msg, "error_info": error_info}
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dict containing account information
        """
        try:
            if not self._ensure_connection():
                return {"success": False, "message": "Not connected to exchange"}
            
            result = self.adapter.get_account_info()
            
            if "error" in result:
                return {"success": False, "message": result["error"]}
            else:
                # Ensure all expected fields are present with fallbacks
                account_info = {
                    "equity": self._safe_get_attribute(result, "equity", 0.0),
                    "margin": self._safe_get_attribute(result, "margin", 0.0),
                    "free_margin": self._safe_get_attribute(result, "free_margin", 0.0),
                    "margin_ratio": self._safe_get_attribute(result, "margin_ratio", 0.0),
                    "unrealized_pnl": self._safe_get_attribute(result, "unrealized_pnl", 0.0),
                    "realized_pnl": self._safe_get_attribute(result, "realized_pnl", 0.0),
                    "total_pnl": self._safe_get_attribute(result, "total_pnl", 0.0),
                    "wallet_balance": self._safe_get_attribute(result, "wallet_balance", 0.0)
                }
                return {"success": True, "data": account_info}
        except Exception as e:
            error_info = self.error_handler.handle_error(e, {"function": "get_account_info"})
            error_msg = f"Error getting account info: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg, "error_info": error_info}
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get market data for a symbol.
        
        Args:
            symbol: The symbol to get market data for
            
        Returns:
            Dict containing market data
        """
        try:
            if not self._ensure_connection():
                return {"success": False, "message": "Not connected to exchange"}
            
            if not symbol or symbol.strip() == "":
                return {"success": False, "message": "Symbol cannot be empty"}
            
            result = self.adapter.get_market_data(symbol)
            
            if "error" in result:
                return {"success": False, "message": result["error"]}
            else:
                # Ensure all expected fields are present with fallbacks
                market_data = {
                    "symbol": symbol,
                    "price": self._safe_get_attribute(result, "price", 0.0),
                    "mark_price": self._safe_get_attribute(result, "mark_price", 0.0),
                    "index_price": self._safe_get_attribute(result, "index_price", 0.0),
                    "funding_rate": self._safe_get_attribute(result, "funding_rate", 0.0),
                    "open_interest": self._safe_get_attribute(result, "open_interest", 0.0),
                    "volume_24h": self._safe_get_attribute(result, "volume_24h", 0.0),
                    "price_change_24h": self._safe_get_attribute(result, "price_change_24h", 0.0)
                }
                return {"success": True, "data": market_data}
        except Exception as e:
            error_info = self.error_handler.handle_error(e, {
                "function": "get_market_data",
                "symbol": symbol
            })
            error_msg = f"Error getting market data for {symbol}: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg, "error_info": error_info}
    
    def get_positions(self) -> Dict[str, Any]:
        """
        Get current positions.
        
        Returns:
            Dict containing positions
        """
        try:
            if not self._ensure_connection():
                return {"success": False, "message": "Not connected to exchange"}
            
            result = self.adapter.get_positions()
            
            if "error" in result:
                return {"success": False, "message": result["error"]}
            else:
                # Process positions to ensure consistent format
                positions = self._safe_get_attribute(result, "data", [])
                processed_positions = []
                
                for pos in positions:
                    # Handle different SDK versions and formats
                    processed_pos = {
                        "symbol": self._safe_get_attribute(pos, "coin", 
                                 self._safe_get_attribute(pos, "name", "Unknown")),
                        "size": self._safe_float_convert(self._safe_get_attribute(pos, "szi", 
                                self._safe_get_attribute(pos, "size", 0.0))),
                        "entry_price": self._safe_float_convert(self._safe_get_attribute(pos, "entryPx", 
                                      self._safe_get_attribute(pos, "entry_price", 0.0))),
                        "mark_price": self._safe_float_convert(self._safe_get_attribute(pos, "markPx", 
                                     self._safe_get_attribute(pos, "mark_price", 0.0))),
                        "liquidation_price": self._safe_float_convert(self._safe_get_attribute(pos, "liqPx", 
                                           self._safe_get_attribute(pos, "liquidation_price", 0.0))),
                        "unrealized_pnl": self._safe_float_convert(self._safe_get_attribute(pos, "unrealizedPnl", 
                                         self._safe_get_attribute(pos, "unrealized_pnl", 0.0))),
                        "leverage": self._safe_float_convert(self._safe_get_attribute(pos, "leverage", 1.0))
                    }
                    
                    # Calculate PnL percentage
                    if processed_pos["entry_price"] > 0 and processed_pos["size"] != 0:
                        price_diff = processed_pos["mark_price"] - processed_pos["entry_price"]
                        direction = 1 if processed_pos["size"] > 0 else -1
                        processed_pos["pnl_percentage"] = direction * price_diff / processed_pos["entry_price"] * 100
                    else:
                        processed_pos["pnl_percentage"] = 0.0
                    
                    processed_positions.append(processed_pos)
                
                return {"success": True, "data": processed_positions}
        except Exception as e:
            error_info = self.error_handler.handle_error(e, {"function": "get_positions"})
            error_msg = f"Error getting positions: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg, "error_info": error_info}
    
    def get_orders(self) -> Dict[str, Any]:
        """
        Get current orders.
        
        Returns:
            Dict containing orders
        """
        try:
            if not self._ensure_connection():
                return {"success": False, "message": "Not connected to exchange"}
            
            result = self.adapter.get_orders()
            
            if "error" in result:
                return {"success": False, "message": result["error"]}
            else:
                # Process orders to ensure consistent format
                orders = self._safe_get_attribute(result, "data", [])
                processed_orders = []
                
                for order in orders:
                    # Handle different SDK versions and formats
                    processed_order = {
                        "id": self._safe_get_attribute(order, "oid", 
                              self._safe_get_attribute(order, "order_id", "Unknown")),
                        "symbol": self._safe_get_attribute(order, "coin", 
                                 self._safe_get_attribute(order, "name", "Unknown")),
                        "size": self._safe_float_convert(self._safe_get_attribute(order, "sz", 
                               self._safe_get_attribute(order, "size", 0.0))),
                        "price": self._safe_float_convert(self._safe_get_attribute(order, "px", 
                                self._safe_get_attribute(order, "price", 0.0))),
                        "side": "buy" if self._safe_get_attribute(order, "side", "") == "B" else "sell",
                        "type": self._safe_get_attribute(order, "orderType", 
                               self._safe_get_attribute(order, "type", "limit")),
                        "status": self._safe_get_attribute(order, "status", "open"),
                        "time": self._safe_get_attribute(order, "timestamp", 
                               self._safe_get_attribute(order, "time", 0))
                    }
                    
                    processed_orders.append(processed_order)
                
                return {"success": True, "data": processed_orders}
        except Exception as e:
            error_info = self.error_handler.handle_error(e, {"function": "get_orders"})
            error_msg = f"Error getting orders: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg, "error_info": error_info}
    
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
            if not self._ensure_connection():
                return {"success": False, "message": "Not connected to exchange"}
            
            # Validate inputs
            if not symbol or symbol.strip() == "":
                return {"success": False, "message": "Symbol cannot be empty"}
            
            if size <= 0:
                return {"success": False, "message": "Size must be greater than 0"}
            
            if order_type == "LIMIT" and price <= 0:
                return {"success": False, "message": "Price must be greater than 0 for LIMIT orders"}
            
            result = self.adapter.place_order(symbol, is_buy, size, price, order_type)
            
            if "error" in result:
                return {"success": False, "message": result["error"]}
            else:
                return {"success": True, "data": result.get("data", {})}
        except Exception as e:
            error_info = self.error_handler.handle_error(e, {
                "function": "place_order",
                "symbol": symbol,
                "is_buy": is_buy,
                "size": size,
                "price": price,
                "order_type": order_type
            })
            error_msg = f"Error placing order: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg, "error_info": error_info}
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            order_id: The ID of the order to cancel
            
        Returns:
            Dict containing the result of the operation
        """
        try:
            if not self._ensure_connection():
                return {"success": False, "message": "Not connected to exchange"}
            
            if not order_id:
                return {"success": False, "message": "Order ID cannot be empty"}
            
            result = self.adapter.cancel_order(order_id)
            
            if "error" in result:
                return {"success": False, "message": result["error"]}
            else:
                return {"success": True, "data": result.get("data", {})}
        except Exception as e:
            error_info = self.error_handler.handle_error(e, {
                "function": "cancel_order",
                "order_id": order_id
            })
            error_msg = f"Error canceling order: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg, "error_info": error_info}
    
    def cancel_all_orders(self, symbol: str = None) -> Dict[str, Any]:
        """
        Cancel all orders.
        
        Args:
            symbol: The symbol to cancel orders for (optional)
            
        Returns:
            Dict containing the result of the operation
        """
        try:
            if not self._ensure_connection():
                return {"success": False, "message": "Not connected to exchange"}
            
            result = self.adapter.cancel_all_orders(symbol)
            
            if "error" in result:
                return {"success": False, "message": result["error"]}
            else:
                return {"success": True, "data": result.get("data", {})}
        except Exception as e:
            error_info = self.error_handler.handle_error(e, {
                "function": "cancel_all_orders",
                "symbol": symbol
            })
            error_msg = f"Error canceling all orders: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg, "error_info": error_info}
    
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
            if not self._ensure_connection():
                return {"success": False, "message": "Not connected to exchange"}
            
            if not symbol or symbol.strip() == "":
                return {"success": False, "message": "Symbol cannot be empty"}
            
            if size_percentage <= 0 or size_percentage > 100:
                return {"success": False, "message": "Size percentage must be between 0 and 100"}
            
            result = self.adapter.close_position(symbol, size_percentage)
            
            if "error" in result:
                return {"success": False, "message": result["error"]}
            else:
                return {"success": True, "data": result.get("data", {})}
        except Exception as e:
            error_info = self.error_handler.handle_error(e, {
                "function": "close_position",
                "symbol": symbol,
                "size_percentage": size_percentage
            })
            error_msg = f"Error closing position: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg, "error_info": error_info}
    
    def get_available_symbols(self) -> List[str]:
        """
        Get a list of available trading symbols.
        
        Returns:
            List of available symbols
        """
        try:
            if not self._ensure_connection():
                return []
            
            symbols = self.adapter.get_available_symbols()
            
            # Ensure we always return a list, even if the adapter returns None
            if symbols is None:
                return []
            
            return symbols
        except Exception as e:
            error_info = self.error_handler.handle_error(e, {"function": "get_available_symbols"})
            self.logger.error(f"Error getting available symbols: {e}")
            return []
