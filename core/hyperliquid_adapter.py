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
    
    def __init__(self, use_testnet=False, connection_manager=None, config_path=None):
        """
        Initialize the Hyperliquid adapter.
        
        Args:
            use_testnet: Whether to use the testnet API
            connection_manager: Optional connection manager for enhanced resilience
            config_path: Path to the configuration file (optional)
        """
        self.logger = logging.getLogger("HyperliquidAdapter")
        self.exchange = None
        self.info = None
        self.account_address = ""
        self.secret_key = ""
        self.is_connected = False
        self.use_testnet = use_testnet
        self.connection_manager = connection_manager
        
        # Set API URL based on testnet flag
        self.api_url = "https://api.hyperliquid-testnet.xyz" if use_testnet else "https://api.hyperliquid.xyz"
        
        # Set config path
        self.config_path = config_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.json")
        
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
            
            # Override API URL if specified in config
            if "api_url" in self.config:
                self.api_url = self.config["api_url"]
            
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
    
    def connect(self) -> bool:
        """
        Connect to the exchange.
        
        Returns:
            True if connected, False otherwise
        """
        try:
            # Initialize API
            self._init_api()
            
            # Test connection
            if self.is_connected:
                result = self.test_connection()
                return "success" in result
            
            return False
        except Exception as e:
            self.logger.error(f"Error connecting to exchange: {e}")
            return False
    
    def ensure_connection(self) -> bool:
        """
        Ensure connection to the exchange, reconnecting if necessary.
        
        Returns:
            True if connected, False otherwise
        """
        try:
            # Check if connection manager is available
            if self.connection_manager:
                # Use connection manager to check and restore connection
                if not self.connection_manager.is_connected():
                    self.logger.info("Connection lost, attempting to reconnect")
                    self.connection_manager.reconnect()
                    
                    # Update connection status
                    self.is_connected = self.connection_manager.is_connected()
                    
                    return self.is_connected
                
                return True
            
            # Fallback to simple connection check
            if not self.is_connected:
                self.logger.info("Not connected, attempting to connect")
                return self.connect()
            
            # Test connection
            result = self.test_connection()
            if "error" in result:
                self.logger.info("Connection test failed, attempting to reconnect")
                return self.connect()
            
            return True
        except Exception as e:
            self.logger.error(f"Error ensuring connection: {e}")
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
            
            if isinstance(user_state, dict) and "error" in user_state:
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
            # Ensure connection
            if not self.ensure_connection():
                return {"error": "Not connected to exchange"}
            
            # Get user state
            user_state = self.info.user_state(self.account_address)
            
            if isinstance(user_state, dict) and "error" in user_state:
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
            # Ensure connection
            if not self.ensure_connection():
                return {"error": "Not connected to exchange"}
            
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
            # Ensure connection
            if not self.ensure_connection():
                return {"error": "Not connected to exchange"}
            
            # Get user state
            user_state = self.info.user_state(self.account_address)
            
            if isinstance(user_state, dict) and "error" in user_state:
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
            # Ensure connection
            if not self.ensure_connection():
                return {"error": "Not connected to exchange"}
            
            # Get open orders
            orders = self.info.open_orders(self.account_address)
            
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
            # Ensure connection
            if not self.ensure_connection():
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
            # Ensure connection
            if not self.ensure_connection():
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
            # Ensure connection
            if not self.ensure_connection():
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
            # Ensure connection
            if not self.ensure_connection():
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
            size = float(position.get("szi", 0))
            is_long = size > 0
            
            if size == 0:
                self.logger.error(f"Position size is zero for {symbol}")
                return {"error": f"Position size is zero for {symbol}"}
            
            # Calculate size to close
            size_to_close = abs(size) * size_percentage / 100.0
            
            # Place order to close position
            return self.place_order(
                symbol=symbol,
                is_buy=not is_long,
                size=size_to_close,
                price=0,
                order_type="MARKET"
            )
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return {"error": f"Error closing position: {e}"}
    
    def close_all_positions(self) -> Dict[str, Any]:
        """
        Close all positions.
        
        Returns:
            Dict containing the result of the operation
        """
        try:
            # Ensure connection
            if not self.ensure_connection():
                return {"error": "Not connected to exchange"}
            
            # Get positions
            positions_result = self.get_positions()
            
            if "error" in positions_result:
                return positions_result
            
            positions = positions_result.get("data", [])
            
            # Close each position
            results = []
            for position in positions:
                symbol = position.get("coin")
                size = float(position.get("szi", 0))
                
                if size == 0:
                    continue
                
                result = self.close_position(symbol)
                results.append({
                    "symbol": symbol,
                    "result": result
                })
            
            return {"success": True, "data": results}
        except Exception as e:
            self.logger.error(f"Error closing all positions: {e}")
            return {"error": f"Error closing all positions: {e}"}
    
    def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Get funding rate for a symbol.
        
        Args:
            symbol: The symbol to get funding rate for
            
        Returns:
            Dict containing funding rate information
        """
        try:
            # Ensure connection
            if not self.ensure_connection():
                return {"error": "Not connected to exchange"}
            
            # Get market data
            market_data = self.get_market_data(symbol)
            
            if "error" in market_data:
                return market_data
            
            # Extract funding rate
            funding_rate = market_data.get("funding_rate", 0)
            
            return {"success": True, "funding_rate": funding_rate}
        except Exception as e:
            self.logger.error(f"Error getting funding rate: {e}")
            return {"error": f"Error getting funding rate: {e}"}
    
    def get_leverage(self, symbol: str) -> Dict[str, Any]:
        """
        Get leverage for a symbol.
        
        Args:
            symbol: The symbol to get leverage for
            
        Returns:
            Dict containing leverage information
        """
        try:
            # Ensure connection
            if not self.ensure_connection():
                return {"error": "Not connected to exchange"}
            
            # Get user state
            user_state = self.info.user_state(self.account_address)
            
            if isinstance(user_state, dict) and "error" in user_state:
                self.logger.error(f"Error getting leverage: {user_state['error']}")
                return {"error": f"Error getting leverage: {user_state['error']}"}
            
            # Extract leverage
            leverage = 1.0  # Default leverage
            
            # Find the symbol in the user state
            for position in user_state.get("assetPositions", []):
                if position.get("coin") == symbol:
                    leverage = float(position.get("leverage", 1.0))
                    break
            
            return {"success": True, "leverage": leverage}
        except Exception as e:
            self.logger.error(f"Error getting leverage: {e}")
            return {"error": f"Error getting leverage: {e}"}
    
    def set_leverage(self, symbol: str, leverage: float) -> Dict[str, Any]:
        """
        Set leverage for a symbol.
        
        Args:
            symbol: The symbol to set leverage for
            leverage: The leverage to set
            
        Returns:
            Dict containing the result of the operation
        """
        try:
            # Ensure connection
            if not self.ensure_connection():
                return {"error": "Not connected to exchange"}
            
            # Set leverage
            try:
                leverage_result = self.exchange.update_leverage(
                    name=symbol,
                    leverage=leverage
                )
                
                if leverage_result.get("status") != "ok":
                    self.logger.error(f"Error setting leverage: {leverage_result}")
                    return {"error": f"Error setting leverage: {leverage_result}"}
                
                return {"success": True, "data": leverage_result}
            except Exception as e:
                self.logger.error(f"Error setting leverage: {e}")
                return {"error": f"Error setting leverage: {e}"}
        except Exception as e:
            self.logger.error(f"Error setting leverage: {e}")
            return {"error": f"Error setting leverage: {e}"}
    
    def get_order_history(self, symbol: str = None, limit: int = 100) -> Dict[str, Any]:
        """
        Get order history.
        
        Args:
            symbol: The symbol to get order history for (optional)
            limit: The maximum number of orders to return
            
        Returns:
            Dict containing order history
        """
        try:
            # Ensure connection
            if not self.ensure_connection():
                return {"error": "Not connected to exchange"}
            
            # Get order history
            try:
                order_history = self.info.order_history(
                    address=self.account_address,
                    coin=symbol,
                    limit=limit
                )
                
                if isinstance(order_history, dict) and "error" in order_history:
                    self.logger.error(f"Error getting order history: {order_history['error']}")
                    return {"error": f"Error getting order history: {order_history['error']}"}
                
                return {"success": True, "data": order_history}
            except Exception as e:
                self.logger.error(f"Error getting order history: {e}")
                return {"error": f"Error getting order history: {e}"}
        except Exception as e:
            self.logger.error(f"Error getting order history: {e}")
            return {"error": f"Error getting order history: {e}"}
    
    def get_trade_history(self, symbol: str = None, limit: int = 100) -> Dict[str, Any]:
        """
        Get trade history.
        
        Args:
            symbol: The symbol to get trade history for (optional)
            limit: The maximum number of trades to return
            
        Returns:
            Dict containing trade history
        """
        try:
            # Ensure connection
            if not self.ensure_connection():
                return {"error": "Not connected to exchange"}
            
            # Get trade history
            try:
                trade_history = self.info.fill_history(
                    address=self.account_address,
                    coin=symbol,
                    limit=limit
                )
                
                if isinstance(trade_history, dict) and "error" in trade_history:
                    self.logger.error(f"Error getting trade history: {trade_history['error']}")
                    return {"error": f"Error getting trade history: {trade_history['error']}"}
                
                return {"success": True, "data": trade_history}
            except Exception as e:
                self.logger.error(f"Error getting trade history: {e}")
                return {"error": f"Error getting trade history: {e}"}
        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}")
            return {"error": f"Error getting trade history: {e}"}
    
    def get_available_symbols(self) -> Dict[str, Any]:
        """
        Get available symbols.
        
        Returns:
            Dict containing available symbols
        """
        try:
            # Ensure connection
            if not self.ensure_connection():
                return {"error": "Not connected to exchange"}
            
            # Get meta and asset
            try:
                meta_and_asset = self.info.meta_and_asset()
                
                if isinstance(meta_and_asset, dict) and "error" in meta_and_asset:
                    self.logger.error(f"Error getting available symbols: {meta_and_asset['error']}")
                    return {"error": f"Error getting available symbols: {meta_and_asset['error']}"}
                
                # Extract symbols
                symbols = []
                for asset in meta_and_asset.get("universe", []):
                    symbols.append(asset.get("name"))
                
                return {"success": True, "data": symbols}
            except Exception as e:
                self.logger.error(f"Error getting available symbols: {e}")
                return {"error": f"Error getting available symbols: {e}"}
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return {"error": f"Error getting available symbols: {e}"}
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get ticker for a symbol.
        
        Args:
            symbol: The symbol to get ticker for
            
        Returns:
            Dict containing ticker information
        """
        try:
            # Ensure connection
            if not self.ensure_connection():
                return {"error": "Not connected to exchange"}
            
            # Get market data
            market_data = self.get_market_data(symbol)
            
            if "error" in market_data:
                return market_data
            
            # Extract ticker
            ticker = {
                "symbol": symbol,
                "last_price": market_data.get("price", 0),
                "mark_price": market_data.get("mark_price", 0),
                "index_price": market_data.get("index_price", 0),
                "funding_rate": market_data.get("funding_rate", 0),
                "volume_24h": market_data.get("volume_24h", 0),
                "price_change_24h": market_data.get("price_change_24h", 0)
            }
            
            return {"success": True, "data": ticker}
        except Exception as e:
            self.logger.error(f"Error getting ticker: {e}")
            return {"error": f"Error getting ticker: {e}"}
    
    def get_orderbook(self, symbol: str) -> Dict[str, Any]:
        """
        Get orderbook for a symbol.
        
        Args:
            symbol: The symbol to get orderbook for
            
        Returns:
            Dict containing orderbook information
        """
        try:
            # Ensure connection
            if not self.ensure_connection():
                return {"error": "Not connected to exchange"}
            
            # Get orderbook
            try:
                orderbook = self.info.l2_snapshot(symbol)
                
                if isinstance(orderbook, dict) and "error" in orderbook:
                    self.logger.error(f"Error getting orderbook: {orderbook['error']}")
                    return {"error": f"Error getting orderbook: {orderbook['error']}"}
                
                return {"success": True, "data": orderbook}
            except Exception as e:
                self.logger.error(f"Error getting orderbook: {e}")
                return {"error": f"Error getting orderbook: {e}"}
        except Exception as e:
            self.logger.error(f"Error getting orderbook: {e}")
            return {"error": f"Error getting orderbook: {e}"}
    
    def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> Dict[str, Any]:
        """
        Get klines for a symbol.
        
        Args:
            symbol: The symbol to get klines for
            interval: The interval of the klines (1m, 5m, 15m, 1h, 4h, 1d)
            limit: The maximum number of klines to return
            
        Returns:
            Dict containing klines information
        """
        try:
            # Ensure connection
            if not self.ensure_connection():
                return {"error": "Not connected to exchange"}
            
            # Map interval to resolution
            resolution_map = {
                "1m": 60,
                "5m": 300,
                "15m": 900,
                "1h": 3600,
                "4h": 14400,
                "1d": 86400
            }
            
            resolution = resolution_map.get(interval, 3600)
            
            # Get klines
            try:
                klines = self.info.candles(
                    coin=symbol,
                    resolution=resolution,
                    limit=limit
                )
                
                if isinstance(klines, dict) and "error" in klines:
                    self.logger.error(f"Error getting klines: {klines['error']}")
                    return {"error": f"Error getting klines: {klines['error']}"}
                
                return {"success": True, "data": klines}
            except Exception as e:
                self.logger.error(f"Error getting klines: {e}")
                return {"error": f"Error getting klines: {e}"}
        except Exception as e:
            self.logger.error(f"Error getting klines: {e}")
            return {"error": f"Error getting klines: {e}"}
