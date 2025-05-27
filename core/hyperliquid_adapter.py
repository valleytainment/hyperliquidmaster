"""
HyperliquidAdapter module for connecting to the Hyperliquid exchange API.
Provides direct integration with the Hyperliquid exchange for trading operations.
"""

import os
import json
import time
import logging
import traceback
import threading
from typing import Dict, List, Any, Optional, Tuple, Union

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
        
        # Connection state tracking
        self.last_connection_attempt = 0
        self.connection_attempts = 0
        self.max_connection_attempts = 5
        self.connection_backoff_base = 2  # seconds
        self.connection_lock = threading.Lock()
        self.connection_health_check_interval = 60  # seconds
        self.last_health_check = 0
        
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
                self.logger.warning(f"Config file not found at {self.config_path}, using empty config")
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
        with self.connection_lock:
            try:
                # Get API keys from config
                self.account_address = self.config.get("account_address", "")
                self.secret_key = self.config.get("secret_key", "")
                self.api_url = self.config.get("api_url", "https://api.hyperliquid.xyz")
                
                # Check if keys are available
                if not self.account_address or not self.secret_key:
                    self.logger.warning("API keys not found in config")
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
                    
                    # Test connection
                    self._test_api_connection()
                    
                    # Reset connection attempts on successful connection
                    self.connection_attempts = 0
                    self.last_connection_attempt = time.time()
                    self.last_health_check = time.time()
                    
                    self.logger.info("API initialized successfully")
                except Exception as e:
                    self.logger.error(f"Error initializing exchange: {e}")
                    self.is_connected = False
                    self.connection_attempts += 1
            except Exception as e:
                self.logger.error(f"Error initializing API: {e}")
                self.is_connected = False
                self.connection_attempts += 1
    
    def _test_api_connection(self) -> bool:
        """
        Test the API connection by making a simple request.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Make a simple request to test connection
            if self.info:
                # Try to get meta data
                meta = self.info.meta()
                if meta and not isinstance(meta, dict) or not "error" in meta:
                    self.is_connected = True
                    return True
            
            self.is_connected = False
            return False
        except Exception as e:
            self.logger.error(f"API connection test failed: {e}")
            self.is_connected = False
            return False
    
    def _ensure_connection(self) -> bool:
        """
        Ensure that the API is connected, attempting to reconnect if necessary.
        
        Returns:
            True if connected, False otherwise
        """
        with self.connection_lock:
            # Check if already connected
            if self.is_connected:
                # Check if health check is due
                current_time = time.time()
                if current_time - self.last_health_check > self.connection_health_check_interval:
                    self.last_health_check = current_time
                    if not self._test_api_connection():
                        self.logger.warning("Health check failed, attempting to reconnect")
                        return self._reconnect()
                return True
            
            # Not connected, attempt to reconnect
            return self._reconnect()
    
    def _reconnect(self) -> bool:
        """
        Attempt to reconnect to the API with exponential backoff.
        
        Returns:
            True if reconnection is successful, False otherwise
        """
        # Check if max attempts reached
        if self.connection_attempts >= self.max_connection_attempts:
            self.logger.error(f"Max connection attempts ({self.max_connection_attempts}) reached")
            return False
        
        # Calculate backoff time
        current_time = time.time()
        backoff_time = self.connection_backoff_base ** self.connection_attempts
        time_since_last_attempt = current_time - self.last_connection_attempt
        
        # Wait for backoff if necessary
        if time_since_last_attempt < backoff_time:
            wait_time = backoff_time - time_since_last_attempt
            self.logger.info(f"Waiting {wait_time:.2f}s before reconnection attempt")
            time.sleep(wait_time)
        
        # Attempt to reconnect
        self.logger.info(f"Attempting to reconnect (attempt {self.connection_attempts + 1}/{self.max_connection_attempts})")
        self.last_connection_attempt = time.time()
        self._init_api()
        
        return self.is_connected
    
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
            
            # Reset connection state
            self.connection_attempts = 0
            
            # Initialize API
            self._init_api()
            
            return self.is_connected
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
            if not self._ensure_connection():
                return {"error": "Not connected to exchange"}
            
            # Test connection by getting user state
            try:
                user_state = self._safe_api_call(lambda: self.info.user_state(self.account_address))
                
                if isinstance(user_state, dict) and "error" in user_state:
                    self.logger.error(f"Connection test failed: {user_state['error']}")
                    return {"error": f"Connection test failed: {user_state['error']}"}
                
                return {"success": "Connection test successful"}
            except Exception as e:
                self.logger.error(f"Connection test failed: {e}")
                return {"error": f"Connection test failed: {e}"}
        except Exception as e:
            self.logger.error(f"Error testing connection: {e}")
            return {"error": f"Error testing connection: {e}"}
    
    def _safe_api_call(self, api_func, max_retries=3, retry_delay=1):
        """
        Safely call an API function with retry logic.
        
        Args:
            api_func: Function to call
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            
        Returns:
            Result of the API call
        """
        retries = 0
        last_error = None
        
        while retries <= max_retries:
            try:
                if not self.is_connected and not self._ensure_connection():
                    return {"error": "Not connected to exchange"}
                
                result = api_func()
                
                # Check if result is an error
                if isinstance(result, dict) and "error" in result:
                    # Check if it's a connection error
                    error_msg = str(result["error"]).lower()
                    if "connect" in error_msg or "timeout" in error_msg or "network" in error_msg:
                        self.is_connected = False
                        if not self._ensure_connection():
                            return {"error": f"Connection error: {result['error']}"}
                        retries += 1
                        time.sleep(retry_delay * retries)
                        continue
                
                return result
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                
                # Check if it's a connection error
                if "connect" in error_msg or "timeout" in error_msg or "network" in error_msg:
                    self.is_connected = False
                    if not self._ensure_connection():
                        return {"error": f"Connection error: {e}"}
                
                retries += 1
                if retries <= max_retries:
                    self.logger.warning(f"API call failed, retrying ({retries}/{max_retries}): {e}")
                    time.sleep(retry_delay * retries)
                else:
                    self.logger.error(f"API call failed after {max_retries} retries: {e}")
                    return {"error": f"API call failed: {e}"}
        
        return {"error": f"API call failed: {last_error}"}
    
    def _safe_get_attribute(self, obj, attr_name, default=None):
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
            if hasattr(obj, attr_name):
                return getattr(obj, attr_name)
            elif isinstance(obj, dict) and attr_name in obj:
                return obj[attr_name]
            else:
                return default
        except Exception as e:
            self.logger.warning(f"Error getting attribute {attr_name}: {e}")
            return default
    
    def _safe_get_nested(self, data, keys, default=None):
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
    
    def _safe_float_convert(self, value, default=0.0):
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
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dict containing account information
        """
        try:
            if not self._ensure_connection():
                return {"error": "Not connected to exchange"}
            
            # Get user state
            user_state = self._safe_api_call(lambda: self.info.user_state(self.account_address))
            
            if isinstance(user_state, dict) and "error" in user_state:
                self.logger.error(f"Error getting account info: {user_state['error']}")
                return {"error": f"Error getting account info: {user_state['error']}"}
            
            # Extract account info with safe access
            margin_summary = self._safe_get_attribute(user_state, "marginSummary", {})
            account_info = {
                "equity": self._safe_float_convert(self._safe_get_attribute(margin_summary, "accountValue", 0)),
                "margin": self._safe_float_convert(self._safe_get_attribute(margin_summary, "totalMargin", 0)),
                "free_margin": self._safe_float_convert(self._safe_get_attribute(margin_summary, "marginAvailable", 0)),
                "margin_ratio": self._safe_float_convert(self._safe_get_attribute(margin_summary, "marginRatio", 0)),
                "unrealized_pnl": self._safe_float_convert(self._safe_get_attribute(margin_summary, "unrealizedPnl", 0)),
                "realized_pnl": self._safe_float_convert(self._safe_get_attribute(margin_summary, "realizedPnl", 0)),
                "total_pnl": self._safe_float_convert(self._safe_get_attribute(margin_summary, "totalPnl", 0)),
                "wallet_balance": self._safe_float_convert(self._safe_get_attribute(margin_summary, "walletBalance", 0))
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
            if not self._ensure_connection():
                return {"error": "Not connected to exchange"}
            
            if not symbol or symbol.strip() == "":
                return {"error": "Symbol cannot be empty"}
            
            # Get market data
            try:
                # Get meta data
                meta_and_asset = self._safe_api_call(lambda: self.info.meta_and_asset())
                
                if isinstance(meta_and_asset, dict) and "error" in meta_and_asset:
                    self.logger.error(f"Error getting meta data: {meta_and_asset['error']}")
                    return {"error": f"Error getting meta data: {meta_and_asset['error']}"}
                
                # Find the symbol in the meta data
                symbol_data = None
                universe = self._safe_get_attribute(meta_and_asset, "universe", [])
                
                for asset in universe:
                    if self._safe_get_attribute(asset, "name") == symbol:
                        symbol_data = asset
                        break
                
                if not symbol_data:
                    self.logger.error(f"Symbol {symbol} not found")
                    return {"error": f"Symbol {symbol} not found"}
                
                # Get market data for the symbol
                market_data = self._safe_api_call(lambda: self.info.market_data(symbol))
                
                if isinstance(market_data, dict) and "error" in market_data:
                    self.logger.error(f"Error getting market data: {market_data['error']}")
                    return {"error": f"Error getting market data: {market_data['error']}"}
                
                # Extract market data with safe access
                return {
                    "symbol": symbol,
                    "price": self._safe_float_convert(self._safe_get_attribute(market_data, "midPrice", 0)),
                    "mark_price": self._safe_float_convert(self._safe_get_attribute(market_data, "markPrice", 0)),
                    "index_price": self._safe_float_convert(self._safe_get_attribute(market_data, "indexPrice", 0)),
                    "funding_rate": self._safe_float_convert(self._safe_get_attribute(market_data, "fundingRate", 0)),
                    "open_interest": self._safe_float_convert(self._safe_get_attribute(market_data, "openInterest", 0)),
                    "volume_24h": self._safe_float_convert(self._safe_get_attribute(market_data, "volume24h", 0)),
                    "price_change_24h": self._safe_float_convert(self._safe_get_attribute(market_data, "priceChange24h", 0))
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
            if not self._ensure_connection():
                return {"error": "Not connected to exchange"}
            
            # Get user state
            user_state = self._safe_api_call(lambda: self.info.user_state(self.account_address))
            
            if isinstance(user_state, dict) and "error" in user_state:
                self.logger.error(f"Error getting positions: {user_state['error']}")
                return {"error": f"Error getting positions: {user_state['error']}"}
            
            # Extract positions with safe access
            positions = self._safe_get_attribute(user_state, "assetPositions", [])
            
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
            if not self._ensure_connection():
                return {"error": "Not connected to exchange"}
            
            # Get open orders
            orders = self._safe_api_call(lambda: self.info.open_orders(self.account_address))
            
            if isinstance(orders, dict) and "error" in orders:
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
            if not self._ensure_connection():
                return {"error": "Not connected to exchange"}
            
            # Place order
            try:
                # Prepare order type
                order_spec = {"limit": {"tif": "Gtc"}} if order_type == "LIMIT" else "market"
                
                # Place order
                order_result = self._safe_api_call(lambda: self.exchange.order(
                    name=symbol,
                    is_buy=is_buy,
                    sz=size,
                    limit_px=price,
                    order_type=order_spec
                ))
                
                if isinstance(order_result, dict) and "error" in order_result:
                    self.logger.error(f"Error placing order: {order_result['error']}")
                    return {"error": f"Error placing order: {order_result['error']}"}
                
                if self._safe_get_attribute(order_result, "status") != "ok":
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
            if not self._ensure_connection():
                return {"error": "Not connected to exchange"}
            
            # Cancel order
            try:
                cancel_result = self._safe_api_call(lambda: self.exchange.cancel(oid=order_id))
                
                if isinstance(cancel_result, dict) and "error" in cancel_result:
                    self.logger.error(f"Error canceling order: {cancel_result['error']}")
                    return {"error": f"Error canceling order: {cancel_result['error']}"}
                
                if self._safe_get_attribute(cancel_result, "status") != "ok":
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
            if not self._ensure_connection():
                return {"error": "Not connected to exchange"}
            
            # Cancel all orders
            try:
                cancel_result = self._safe_api_call(lambda: self.exchange.cancel_all(coin=symbol))
                
                if isinstance(cancel_result, dict) and "error" in cancel_result:
                    self.logger.error(f"Error canceling all orders: {cancel_result['error']}")
                    return {"error": f"Error canceling all orders: {cancel_result['error']}"}
                
                if self._safe_get_attribute(cancel_result, "status") != "ok":
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
            if not self._ensure_connection():
                return {"error": "Not connected to exchange"}
            
            # Get positions
            positions_result = self.get_positions()
            
            if "error" in positions_result:
                return positions_result
            
            positions = positions_result.get("data", [])
            
            # Find the position for the symbol
            position = None
            for pos in positions:
                if self._safe_get_attribute(pos, "coin") == symbol:
                    position = pos
                    break
            
            if not position:
                self.logger.error(f"No position found for {symbol}")
                return {"error": f"No position found for {symbol}"}
            
            # Calculate size to close
            size = self._safe_float_convert(self._safe_get_attribute(position, "szi", 0))
            is_long = size > 0
            size = abs(size)
            
            if size_percentage < 100:
                size = size * size_percentage / 100
            
            # Place order to close position
            return self.place_order(
                symbol=symbol,
                is_buy=not is_long,  # Opposite direction to close
                size=size,
                price=0,  # Market order
                order_type="MARKET"
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
            if not self._ensure_connection():
                return []
            
            # Get meta data
            meta_and_asset = self._safe_api_call(lambda: self.info.meta_and_asset())
            
            if isinstance(meta_and_asset, dict) and "error" in meta_and_asset:
                self.logger.error(f"Error getting meta data: {meta_and_asset['error']}")
                return []
            
            # Extract symbols
            symbols = []
            universe = self._safe_get_attribute(meta_and_asset, "universe", [])
            
            for asset in universe:
                symbol = self._safe_get_attribute(asset, "name")
                if symbol:
                    symbols.append(symbol)
            
            return symbols
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []
