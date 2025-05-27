"""
Trading integration module for the HyperliquidMaster trading bot.
Provides a unified interface for interacting with the Hyperliquid exchange.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional

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
            error_msg = f"Error setting API keys: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg}
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to the exchange.
        
        Returns:
            Dict containing the result of the test
        """
        try:
            result = self.adapter.test_connection()
            
            if "error" in result:
                return {"success": False, "message": result["error"]}
            else:
                return {"success": True, "message": "Connection test successful"}
        except Exception as e:
            error_msg = f"Error testing connection: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg}
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dict containing account information
        """
        try:
            result = self.adapter.get_account_info()
            
            if "error" in result:
                return {"success": False, "message": result["error"]}
            else:
                return {"success": True, "data": result}
        except Exception as e:
            error_msg = f"Error getting account info: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg}
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get market data for a symbol.
        
        Args:
            symbol: The symbol to get market data for
            
        Returns:
            Dict containing market data
        """
        try:
            result = self.adapter.get_market_data(symbol)
            
            if "error" in result:
                return {"success": False, "message": result["error"]}
            else:
                return {"success": True, "data": result}
        except Exception as e:
            error_msg = f"Error getting market data: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg}
    
    def get_positions(self) -> Dict[str, Any]:
        """
        Get current positions.
        
        Returns:
            Dict containing positions
        """
        try:
            result = self.adapter.get_positions()
            
            if "error" in result:
                return {"success": False, "message": result["error"]}
            else:
                return result
        except Exception as e:
            error_msg = f"Error getting positions: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg}
    
    def get_orders(self) -> Dict[str, Any]:
        """
        Get current orders.
        
        Returns:
            Dict containing orders
        """
        try:
            result = self.adapter.get_orders()
            
            if "error" in result:
                return {"success": False, "message": result["error"]}
            else:
                return result
        except Exception as e:
            error_msg = f"Error getting orders: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg}
    
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
            result = self.adapter.place_order(symbol, is_buy, size, price, order_type)
            
            if "error" in result:
                return {"success": False, "message": result["error"]}
            else:
                return result
        except Exception as e:
            error_msg = f"Error placing order: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg}
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            order_id: The ID of the order to cancel
            
        Returns:
            Dict containing the result of the operation
        """
        try:
            result = self.adapter.cancel_order(order_id)
            
            if "error" in result:
                return {"success": False, "message": result["error"]}
            else:
                return result
        except Exception as e:
            error_msg = f"Error canceling order: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg}
    
    def cancel_all_orders(self, symbol: str = None) -> Dict[str, Any]:
        """
        Cancel all orders.
        
        Args:
            symbol: The symbol to cancel orders for (optional)
            
        Returns:
            Dict containing the result of the operation
        """
        try:
            result = self.adapter.cancel_all_orders(symbol)
            
            if "error" in result:
                return {"success": False, "message": result["error"]}
            else:
                return result
        except Exception as e:
            error_msg = f"Error canceling all orders: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg}
    
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
            result = self.adapter.close_position(symbol, size_percentage)
            
            if "error" in result:
                return {"success": False, "message": result["error"]}
            else:
                return result
        except Exception as e:
            error_msg = f"Error closing position: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg}
    
    def get_available_symbols(self) -> List[str]:
        """
        Get a list of available trading symbols.
        
        Returns:
            List of available symbols
        """
        try:
            return self.adapter.get_available_symbols()
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []
