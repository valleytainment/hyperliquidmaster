"""
Enhanced HyperliquidAdapter module for connecting to the Hyperliquid exchange API.
Provides robust connection management and error handling for trading operations.
"""

import os
import json
import time
import logging
import traceback
import threading
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union

# Import eth_account for LocalAccount creation
import eth_account
from eth_account.signers.local import LocalAccount

# Import Hyperliquid SDK - fixed import paths
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils.signing import OrderType

# Import enhanced connection and settings managers
from core.enhanced_connection_manager import EnhancedConnectionManager
from core.settings_manager import SettingsManager

class HyperliquidAdapter:
    """
    Enhanced adapter for the Hyperliquid exchange API.
    Provides methods for interacting with the exchange with robust connection management.
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
        self.api_url = "https://api.hyperliquid.xyz"
        
        # Initialize enhanced connection manager
        self.connection_manager = EnhancedConnectionManager(self.logger)
        
        # Initialize settings manager
        self.settings_manager = SettingsManager(config_path, self.logger)
        
        # Load configuration
        self.config = self.settings_manager.get_settings()
        
        # Initialize API if keys are available
        self._init_api()
    
    async def initialize(self) -> bool:
        """
        Initialize the adapter asynchronously.
        This method is called by the main trading bot.
        
        Returns:
            True if initialization is successful, False otherwise
        """
        self.logger.info("Initializing HyperliquidAdapter...")
        
        # Reload config to ensure we have the latest settings
        self.reload_config()
        
        # Ensure connection
        if not self.connection_manager.ensure_connection(
            connect_func=self._init_api,
            test_func=self._test_api_connection
        ):
            self.logger.error("Failed to initialize adapter: Could not establish connection")
            return False
        
        # Test connection
        test_result = self.test_connection()
        if "error" in test_result:
            self.logger.error(f"Failed to initialize adapter: {test_result['error']}")
            return False
        
        self.logger.info("HyperliquidAdapter initialized successfully")
        return True
    
    def reload_config(self) -> None:
        """Reload configuration from settings manager."""
        self.config = self.settings_manager.get_settings()
        self._init_api()
    
    def _init_api(self) -> bool:
        """
        Initialize the Hyperliquid API with current configuration.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get API keys from config
            self.account_address = self.config.get("account_address", "")
            self.secret_key = self.config.get("secret_key", "")
            self.api_url = self.config.get("api_url", "https://api.hyperliquid.xyz")
            
            # Check if keys are available
            if not self.account_address or not self.secret_key:
                self.logger.warning("API keys not found in config")
                return False
            
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
                return self._test_api_connection()
            except Exception as e:
                self.logger.error(f"Error initializing exchange: {e}")
                return False
        except Exception as e:
            self.logger.error(f"Error initializing API: {e}")
            return False
    
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
                    return True
            
            return False
        except Exception as e:
            self.logger.error(f"API connection test failed: {e}")
            return False
    
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
            new_settings = self.config.copy()
            new_settings["account_address"] = account_address
            new_settings["secret_key"] = secret_key
            
            # Save config using settings manager
            if not self.settings_manager.update_settings(new_settings):
                self.logger.error("Failed to save API keys to settings")
                return False
            
            # Reload config
            self.reload_config()
            
            # Reset connection state
            self.connection_manager.reset_state()
            
            # Initialize API
            return self._init_api()
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
            if not self.connection_manager.ensure_connection(
                connect_func=self._init_api,
                test_func=self._test_api_connection
            ):
                return {"error": "Not connected to exchange"}
            
            # Test connection by getting user state
            try:
                user_state = self.connection_manager.safe_api_call(
                    lambda: self.info.user_state(self.account_address),
                    connect_func=self._init_api,
                    test_func=self._test_api_connection
                )
                
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
            if not self.connection_manager.ensure_connection(
                connect_func=self._init_api,
                test_func=self._test_api_connection
            ):
                return {"error": "Not connected to exchange"}
            
            # Get user state
            user_state = self.connection_manager.safe_api_call(
                lambda: self.info.user_state(self.account_address),
                connect_func=self._init_api,
                test_func=self._test_api_connection
            )
            
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
            if not self.connection_manager.ensure_connection(
                connect_func=self._init_api,
                test_func=self._test_api_connection
            ):
                return {"error": "Not connected to exchange"}
            
            if not symbol:
                return {"error": "Symbol cannot be empty"}
            
            # Get meta and asset contexts which contains all market data
            meta_and_assets = self.connection_manager.safe_api_call(
                lambda: self.info.meta_and_asset_ctxs(),
                connect_func=self._init_api,
                test_func=self._test_api_connection
            )
            
            if isinstance(meta_and_assets, dict) and "error" in meta_and_assets:
                self.logger.error(f"Error getting meta data: {meta_and_assets['error']}")
                return {"error": f"Error getting meta data: {meta_and_assets['error']}"}
            
            # Find asset data for the specified symbol
            asset_data = None
            asset_index = -1
            
            # meta_and_assets is a list where first item is meta data and second item is list of assets
            if len(meta_and_assets) >= 2 and isinstance(meta_and_assets[1], list):
                assets = meta_and_assets[1]
                
                # Debug log the first few assets to understand structure
                if len(assets) > 0:
                    self.logger.debug(f"First asset data: {assets[0]}")
                
                # First try to find by exact name match
                for i, asset in enumerate(assets):
                    # Get name from asset metadata in first element if available
                    if len(meta_and_assets) >= 1 and isinstance(meta_and_assets[0], dict):
                        meta = meta_and_assets[0]
                        universe = self._safe_get_attribute(meta, "universe", [])
                        if i < len(universe):
                            asset_name = self._safe_get_attribute(universe[i], "name", "")
                            if asset_name.upper() == symbol.upper():
                                asset_data = asset
                                asset_index = i
                                break
            
            if not asset_data:
                self.logger.warning(f"Symbol {symbol} not found in asset data")
                # Return default data structure with zeros
                return {
                    "symbol": symbol,
                    "price": 0.0,
                    "mark_price": 0.0,
                    "index_price": 0.0,
                    "funding_rate": 0.0,
                    "open_interest": 0.0,
                    "volume_24h": 0.0,
                    "price_change_24h": 0.0
                }
            
            # Extract market data from asset context with robust fallbacks
            # Use midPx as primary price source, with fallbacks to markPx and oraclePx
            mid_price = self._safe_float_convert(self._safe_get_attribute(asset_data, "midPx", 0.0))
            mark_price = self._safe_float_convert(self._safe_get_attribute(asset_data, "markPx", mid_price))
            oracle_price = self._safe_float_convert(self._safe_get_attribute(asset_data, "oraclePx", mid_price or mark_price))
            
            # Determine the best price to use (midPx > markPx > oraclePx > 0.0)
            price = mid_price
            if price == 0.0:
                price = mark_price
            if price == 0.0:
                price = oracle_price
            
            # Get other market data with fallbacks
            funding_rate = self._safe_float_convert(self._safe_get_attribute(asset_data, "funding", 0.0))
            open_interest = self._safe_float_convert(self._safe_get_attribute(asset_data, "openInterest", 0.0))
            volume_24h = self._safe_float_convert(self._safe_get_attribute(asset_data, "dayNtlVlm", 0.0))
            
            # Calculate price change (if previous day price is available)
            prev_day_price = self._safe_float_convert(self._safe_get_attribute(asset_data, "prevDayPx", 0.0))
            price_change_24h = 0.0
            if prev_day_price > 0 and price > 0:
                price_change_24h = (price - prev_day_price) / prev_day_price * 100
            
            # Construct market data with price field explicitly set
            market_data = {
                "symbol": symbol,
                "price": price,  # Explicitly set price field
                "mark_price": mark_price,
                "index_price": oracle_price,
                "funding_rate": funding_rate,
                "open_interest": open_interest,
                "volume_24h": volume_24h,
                "price_change_24h": price_change_24h
            }
            
            # Log successful market data retrieval
            self.logger.debug(f"Market data for {symbol}: price={price}, funding_rate={funding_rate}")
            
            return market_data
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Return default data structure with zeros on error
            return {
                "symbol": symbol,
                "price": 0.0,
                "mark_price": 0.0,
                "index_price": 0.0,
                "funding_rate": 0.0,
                "open_interest": 0.0,
                "volume_24h": 0.0,
                "price_change_24h": 0.0
            }
    
    async def fetch_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch market data for a symbol asynchronously.
        This method is called by the main trading bot.
        
        Args:
            symbol: The symbol to get market data for
            
        Returns:
            Dict containing market data
        """
        return self.get_market_data(symbol)
    
    def get_positions(self) -> Dict[str, Any]:
        """
        Get current positions.
        
        Returns:
            Dict containing positions
        """
        try:
            if not self.connection_manager.ensure_connection(
                connect_func=self._init_api,
                test_func=self._test_api_connection
            ):
                return {"error": "Not connected to exchange"}
            
            # Get user state
            user_state = self.connection_manager.safe_api_call(
                lambda: self.info.user_state(self.account_address),
                connect_func=self._init_api,
                test_func=self._test_api_connection
            )
            
            if isinstance(user_state, dict) and "error" in user_state:
                self.logger.error(f"Error getting positions: {user_state['error']}")
                return {"error": f"Error getting positions: {user_state['error']}"}
            
            # Extract positions
            positions_data = self._safe_get_attribute(user_state, "assetPositions", [])
            positions = []
            
            for pos in positions_data:
                # Skip positions with zero size
                size = self._safe_float_convert(self._safe_get_attribute(pos, "position", 0.0))
                if size == 0:
                    continue
                
                # Extract position data
                coin = self._safe_get_attribute(pos, "coin", "Unknown")
                entry_price = self._safe_float_convert(self._safe_get_attribute(pos, "entryPx", 0.0))
                mark_price = self._safe_float_convert(self._safe_get_attribute(pos, "markPx", 0.0))
                liquidation_price = self._safe_float_convert(self._safe_get_attribute(pos, "liquidationPx", 0.0))
                unrealized_pnl = self._safe_float_convert(self._safe_get_attribute(pos, "unrealizedPnl", 0.0))
                
                # Calculate PnL percentage
                pnl_percentage = 0.0
                if entry_price > 0 and size != 0:
                    price_diff = mark_price - entry_price
                    direction = 1 if size > 0 else -1
                    pnl_percentage = direction * price_diff / entry_price * 100
                
                # Construct position
                position = {
                    "symbol": coin,
                    "size": size,
                    "entry_price": entry_price,
                    "mark_price": mark_price,
                    "liquidation_price": liquidation_price,
                    "unrealized_pnl": unrealized_pnl,
                    "pnl_percentage": pnl_percentage
                }
                
                positions.append(position)
            
            return {"data": positions}
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {"error": f"Error getting positions: {e}"}
    
    async def get_user_positions(self) -> Dict[str, Any]:
        """
        Get user positions asynchronously.
        This method is called by the main trading bot.
        
        Returns:
            Dict containing positions
        """
        positions_result = self.get_positions()
        
        if "error" in positions_result:
            return {}
        
        # Convert list of positions to dict keyed by symbol
        positions_dict = {}
        for pos in positions_result.get("data", []):
            symbol = pos.get("symbol", "Unknown")
            positions_dict[symbol] = pos
        
        return positions_dict
    
    def get_orders(self) -> Dict[str, Any]:
        """
        Get current orders.
        
        Returns:
            Dict containing orders
        """
        try:
            if not self.connection_manager.ensure_connection(
                connect_func=self._init_api,
                test_func=self._test_api_connection
            ):
                return {"error": "Not connected to exchange"}
            
            # Get user state
            user_state = self.connection_manager.safe_api_call(
                lambda: self.info.user_state(self.account_address),
                connect_func=self._init_api,
                test_func=self._test_api_connection
            )
            
            if isinstance(user_state, dict) and "error" in user_state:
                self.logger.error(f"Error getting orders: {user_state['error']}")
                return {"error": f"Error getting orders: {user_state['error']}"}
            
            # Extract orders
            orders_data = self._safe_get_attribute(user_state, "openOrders", [])
            orders = []
            
            for order in orders_data:
                # Extract order data
                coin = self._safe_get_attribute(order, "coin", "Unknown")
                order_id = self._safe_get_attribute(order, "oid", "Unknown")
                size = self._safe_float_convert(self._safe_get_attribute(order, "sz", 0.0))
                price = self._safe_float_convert(self._safe_get_attribute(order, "limitPx", 0.0))
                side = "buy" if self._safe_get_attribute(order, "side", "") == "B" else "sell"
                order_type = self._safe_get_attribute(order, "orderType", "limit")
                
                # Construct order
                order_data = {
                    "id": order_id,
                    "symbol": coin,
                    "size": size,
                    "price": price,
                    "side": side,
                    "type": order_type,
                    "status": "open",
                    "time": int(time.time() * 1000)  # Current time in milliseconds
                }
                
                orders.append(order_data)
            
            return {"data": orders}
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
            if not self.connection_manager.ensure_connection(
                connect_func=self._init_api,
                test_func=self._test_api_connection
            ):
                return {"error": "Not connected to exchange"}
            
            if not symbol:
                return {"error": "Symbol cannot be empty"}
            
            if size <= 0:
                return {"error": "Size must be greater than 0"}
            
            if order_type.upper() == "LIMIT" and price <= 0:
                return {"error": "Price must be greater than 0 for LIMIT orders"}
            
            # Prepare order
            side = "B" if is_buy else "S"
            
            # Convert string order_type to proper OrderType format
            # OrderType is a TypedDict that expects {'limit': 'limit'} or similar format
            order_type_lower = order_type.lower()
            order_type_dict = {order_type_lower: order_type_lower}
            
            # Place order - FIXED: Use 'name' instead of 'coin' parameter and proper OrderType format
            result = self.connection_manager.safe_api_call(
                lambda: self.exchange.order(
                    name=symbol,  # Changed from 'coin' to 'name' to match SDK signature
                    is_buy=is_buy,
                    sz=size,
                    limit_px=price,
                    order_type=order_type_dict  # Changed from string to dict format
                ),
                connect_func=self._init_api,
                test_func=self._test_api_connection
            )
            
            if isinstance(result, dict) and "error" in result:
                self.logger.error(f"Error placing order: {result['error']}")
                return {"error": f"Error placing order: {result['error']}"}
            
            # Extract order ID - handle both string and dict responses
            order_id = "Unknown"
            if isinstance(result, dict) and "oid" in result:
                order_id = result["oid"]
            elif isinstance(result, dict) and "order_id" in result:
                order_id = result["order_id"]
            elif isinstance(result, str):
                order_id = result
            
            # Construct response
            response = {
                "data": {
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": "buy" if is_buy else "sell",
                    "size": size,
                    "price": price,
                    "type": order_type.lower(),
                    "status": "open",
                    "time": int(time.time() * 1000)  # Current time in milliseconds
                }
            }
            
            return response
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {"error": f"Error placing order: {e}"}
    
    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            symbol: The symbol of the order
            order_id: The ID of the order
            
        Returns:
            Dict containing the result of the operation
        """
        try:
            if not self.connection_manager.ensure_connection(
                connect_func=self._init_api,
                test_func=self._test_api_connection
            ):
                return {"error": "Not connected to exchange"}
            
            if not symbol:
                return {"error": "Symbol cannot be empty"}
            
            if not order_id:
                return {"error": "Order ID cannot be empty"}
            
            # Cancel order - FIXED: Use 'name' instead of 'coin' parameter
            result = self.connection_manager.safe_api_call(
                lambda: self.exchange.cancel_order(
                    name=symbol,  # Changed from 'coin' to 'name' to match SDK signature
                    oid=order_id
                ),
                connect_func=self._init_api,
                test_func=self._test_api_connection
            )
            
            if isinstance(result, dict) and "error" in result:
                self.logger.error(f"Error canceling order: {result['error']}")
                return {"error": f"Error canceling order: {result['error']}"}
            
            # Construct response
            response = {
                "data": {
                    "order_id": order_id,
                    "symbol": symbol,
                    "status": "canceled",
                    "time": int(time.time() * 1000)  # Current time in milliseconds
                }
            }
            
            return response
        except Exception as e:
            self.logger.error(f"Error canceling order: {e}")
            return {"error": f"Error canceling order: {e}"}
    
    def cancel_all_orders(self, symbol: str = None) -> Dict[str, Any]:
        """
        Cancel all orders.
        
        Args:
            symbol: Optional symbol to cancel orders for
            
        Returns:
            Dict containing the result of the operation
        """
        try:
            if not self.connection_manager.ensure_connection(
                connect_func=self._init_api,
                test_func=self._test_api_connection
            ):
                return {"error": "Not connected to exchange"}
            
            # Get current orders
            orders_result = self.get_orders()
            
            if "error" in orders_result:
                return {"error": f"Error getting orders: {orders_result['error']}"}
            
            orders = orders_result.get("data", [])
            
            # Filter orders by symbol if provided
            if symbol:
                orders = [order for order in orders if order.get("symbol") == symbol]
            
            # Cancel each order
            canceled_orders = []
            for order in orders:
                order_symbol = order.get("symbol")
                order_id = order.get("id")
                
                if order_symbol and order_id:
                    cancel_result = self.cancel_order(order_symbol, order_id)
                    
                    if "error" not in cancel_result:
                        canceled_orders.append(cancel_result.get("data", {}))
            
            # Construct response
            response = {
                "data": {
                    "canceled_orders": canceled_orders,
                    "count": len(canceled_orders),
                    "time": int(time.time() * 1000)  # Current time in milliseconds
                }
            }
            
            return response
        except Exception as e:
            self.logger.error(f"Error canceling all orders: {e}")
            return {"error": f"Error canceling all orders: {e}"}
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get connection statistics.
        
        Returns:
            Dict containing connection statistics
        """
        return self.connection_manager.get_connection_stats()
    
    def get_settings_backups(self) -> List[Dict[str, Any]]:
        """
        Get list of available settings backups.
        
        Returns:
            List of dicts containing backup information
        """
        return self.settings_manager.list_backups()
    
    def restore_settings_backup(self, backup_index: int = 0) -> bool:
        """
        Restore settings from a backup.
        
        Args:
            backup_index: Index of the backup to restore (0 = most recent)
            
        Returns:
            True if successful, False otherwise
        """
        return self.settings_manager.restore_backup(backup_index)
    
    def force_settings_backup(self) -> bool:
        """
        Force creation of a settings backup.
        
        Returns:
            True if successful, False otherwise
        """
        return self.settings_manager.force_backup()
