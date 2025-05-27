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
        
        # Initialize connection status
        self.is_connected = False
        
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
            self.is_connected = False
            return False
        
        # Test connection
        test_result = self.test_connection()
        if "error" in test_result:
            self.logger.error(f"Failed to initialize adapter: {test_result['error']}")
            self.is_connected = False
            return False
        
        self.logger.info("HyperliquidAdapter initialized successfully")
        self.is_connected = True
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
                self.is_connected = False
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
                connection_success = self._test_api_connection()
                self.is_connected = connection_success
                return connection_success
            except Exception as e:
                self.logger.error(f"Error initializing exchange: {e}")
                self.is_connected = False
                return False
        except Exception as e:
            self.logger.error(f"Error initializing API: {e}")
            self.is_connected = False
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
                    self.is_connected = True
                    return True
            
            self.is_connected = False
            return False
        except Exception as e:
            self.logger.error(f"API connection test failed: {e}")
            self.is_connected = False
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
                self.is_connected = False
                return False
            
            # Reload config
            self.reload_config()
            
            # Reset connection state
            self.connection_manager.reset_state()
            
            # Initialize API
            connection_success = self._init_api()
            self.is_connected = connection_success
            return connection_success
        except Exception as e:
            self.logger.error(f"Error setting API keys: {e}")
            self.is_connected = False
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
                self.is_connected = False
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
                    self.is_connected = False
                    return {"error": f"Connection test failed: {user_state['error']}"}
                
                self.is_connected = True
                return {"success": "Connection test successful"}
            except Exception as e:
                self.logger.error(f"Connection test failed: {e}")
                self.is_connected = False
                return {"error": f"Connection test failed: {e}"}
        except Exception as e:
            self.logger.error(f"Error testing connection: {e}")
            self.is_connected = False
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
                self.is_connected = False
                return {"error": "Not connected to exchange"}
            
            # Get user state
            user_state = self.connection_manager.safe_api_call(
                lambda: self.info.user_state(self.account_address),
                connect_func=self._init_api,
                test_func=self._test_api_connection
            )
            
            if isinstance(user_state, dict) and "error" in user_state:
                self.logger.error(f"Error getting account info: {user_state['error']}")
                self.is_connected = False
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
            
            self.is_connected = True
            return account_info
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            self.is_connected = False
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
                self.is_connected = False
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
                self.is_connected = False
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
            
            # Extract other market data
            funding_rate = self._safe_float_convert(self._safe_get_attribute(asset_data, "fundingRate", 0.0))
            open_interest = self._safe_float_convert(self._safe_get_attribute(asset_data, "openInterest", 0.0))
            
            # Get 24h volume and price change if available
            volume_24h = self._safe_float_convert(self._safe_get_attribute(asset_data, "volume24h", 0.0))
            price_change_24h = self._safe_float_convert(self._safe_get_attribute(asset_data, "priceChange24h", 0.0))
            
            # Construct market data
            market_data = {
                "symbol": symbol,
                "price": price,
                "mark_price": mark_price,
                "index_price": oracle_price,
                "funding_rate": funding_rate,
                "open_interest": open_interest,
                "volume_24h": volume_24h,
                "price_change_24h": price_change_24h
            }
            
            self.is_connected = True
            return market_data
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            self.is_connected = False
            return {"error": f"Error getting market data for {symbol}: {e}"}
    
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
                self.is_connected = False
                return {"error": "Not connected to exchange"}
            
            # Get user state
            user_state = self.connection_manager.safe_api_call(
                lambda: self.info.user_state(self.account_address),
                connect_func=self._init_api,
                test_func=self._test_api_connection
            )
            
            if isinstance(user_state, dict) and "error" in user_state:
                self.logger.error(f"Error getting positions: {user_state['error']}")
                self.is_connected = False
                return {"error": f"Error getting positions: {user_state['error']}"}
            
            # Extract positions from user state
            positions = []
            
            # Get assetPositions from user state
            asset_positions = self._safe_get_attribute(user_state, "assetPositions", [])
            
            # Process each position
            for pos in asset_positions:
                # Get position details
                coin = self._safe_get_attribute(pos, "coin", "")
                position_size = self._safe_float_convert(self._safe_get_attribute(pos, "position", 0.0))
                entry_price = self._safe_float_convert(self._safe_get_attribute(pos, "entryPx", 0.0))
                
                # Skip positions with zero size
                if position_size == 0.0:
                    continue
                
                # Get additional position details
                liquidation_price = self._safe_float_convert(self._safe_get_attribute(pos, "liquidationPx", 0.0))
                unrealized_pnl = self._safe_float_convert(self._safe_get_attribute(pos, "unrealizedPnl", 0.0))
                
                # Create position object
                position = {
                    "symbol": coin,
                    "size": position_size,
                    "entry_price": entry_price,
                    "liquidation_price": liquidation_price,
                    "unrealized_pnl": unrealized_pnl
                }
                
                positions.append(position)
            
            self.is_connected = True
            return {"data": positions}
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            self.is_connected = False
            return {"error": f"Error getting positions: {e}"}
    
    def place_order(self, symbol: str, size: float, price: Optional[float] = None, is_buy: bool = True, reduce_only: bool = False) -> Dict[str, Any]:
        """
        Place an order on the exchange.
        
        Args:
            symbol: Symbol to place order for
            size: Size of the order
            price: Price of the order (None for market orders)
            is_buy: True for buy orders, False for sell orders
            reduce_only: True to only reduce position, not open new ones
            
        Returns:
            Dict containing the result of the order placement
        """
        try:
            if not self.connection_manager.ensure_connection(
                connect_func=self._init_api,
                test_func=self._test_api_connection
            ):
                self.is_connected = False
                return {"error": "Not connected to exchange"}
            
            # Validate inputs
            if not symbol:
                return {"error": "Symbol cannot be empty"}
            
            if size <= 0:
                return {"error": "Size must be positive"}
            
            # Determine order type
            order_type = OrderType.LIMIT if price is not None else OrderType.MARKET
            
            # Adjust size sign based on direction
            signed_size = size if is_buy else -size
            
            # Place order
            order_result = self.connection_manager.safe_api_call(
                lambda: self.exchange.order(
                    coin=symbol,
                    is_buy=is_buy,
                    sz=abs(signed_size),
                    limit_px=price if price is not None else None,
                    order_type=order_type,
                    reduce_only=reduce_only
                ),
                connect_func=self._init_api,
                test_func=self._test_api_connection
            )
            
            if isinstance(order_result, dict) and "error" in order_result:
                self.logger.error(f"Error placing order: {order_result['error']}")
                return {"error": f"Error placing order: {order_result['error']}"}
            
            self.is_connected = True
            return {"success": "Order placed successfully", "data": order_result}
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            self.is_connected = False
            return {"error": f"Error placing order: {e}"}
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order on the exchange.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            Dict containing the result of the cancellation
        """
        try:
            if not self.connection_manager.ensure_connection(
                connect_func=self._init_api,
                test_func=self._test_api_connection
            ):
                self.is_connected = False
                return {"error": "Not connected to exchange"}
            
            # Validate inputs
            if not order_id:
                return {"error": "Order ID cannot be empty"}
            
            # Cancel order
            cancel_result = self.connection_manager.safe_api_call(
                lambda: self.exchange.cancel_order(order_id),
                connect_func=self._init_api,
                test_func=self._test_api_connection
            )
            
            if isinstance(cancel_result, dict) and "error" in cancel_result:
                self.logger.error(f"Error cancelling order: {cancel_result['error']}")
                return {"error": f"Error cancelling order: {cancel_result['error']}"}
            
            self.is_connected = True
            return {"success": "Order cancelled successfully", "data": cancel_result}
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            self.is_connected = False
            return {"error": f"Error cancelling order: {e}"}
    
    def get_open_orders(self) -> Dict[str, Any]:
        """
        Get open orders.
        
        Returns:
            Dict containing open orders
        """
        try:
            if not self.connection_manager.ensure_connection(
                connect_func=self._init_api,
                test_func=self._test_api_connection
            ):
                self.is_connected = False
                return {"error": "Not connected to exchange"}
            
            # Get open orders
            open_orders = self.connection_manager.safe_api_call(
                lambda: self.info.open_orders(self.account_address),
                connect_func=self._init_api,
                test_func=self._test_api_connection
            )
            
            if isinstance(open_orders, dict) and "error" in open_orders:
                self.logger.error(f"Error getting open orders: {open_orders['error']}")
                self.is_connected = False
                return {"error": f"Error getting open orders: {open_orders['error']}"}
            
            self.is_connected = True
            return {"data": open_orders}
        except Exception as e:
            self.logger.error(f"Error getting open orders: {e}")
            self.is_connected = False
            return {"error": f"Error getting open orders: {e}"}
