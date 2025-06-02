"""
Trading Integration module for HyperliquidMaster.

This module provides integration between the GUI and the core trading components.
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional

from .hyperliquid_adapter import HyperliquidAdapter
from .enhanced_connection_manager import ConnectionManager
from .trading_mode import TradingModeManager
from .enhanced_risk_manager import RiskManager
from .advanced_order_manager import OrderManager
from .position_manager_wrapper import PositionManagerWrapper

class TradingIntegration:
    """
    Integrates all trading components for use by the GUI.
    
    This class provides a unified interface for the GUI to interact with
    all trading components.
    """
    
    def __init__(self, config_path: str, logger: logging.Logger,
                 connection_manager: Optional[ConnectionManager] = None,
                 mode_manager: Optional[TradingModeManager] = None,
                 risk_manager: Optional[RiskManager] = None,
                 order_manager: Optional[OrderManager] = None,
                 position_manager: Optional[PositionManagerWrapper] = None):
        """
        Initialize the trading integration.
        
        Args:
            config_path: Path to the configuration file
            logger: Logger instance
            connection_manager: Optional connection manager
            mode_manager: Optional trading mode manager
            risk_manager: Optional risk manager
            order_manager: Optional order manager
            position_manager: Optional position manager
        """
        self.logger = logger
        self.config_path = config_path
        
        # Initialize connection manager if not provided
        if connection_manager is None:
            self.connection_manager = ConnectionManager(logger)
        else:
            self.connection_manager = connection_manager
        
        # Initialize adapter
        self.adapter = HyperliquidAdapter(
            connection_manager=self.connection_manager,
            config_path=config_path
        )
        
        # Set connection status
        self.is_connected = self.adapter.is_connected
        
        # Initialize trading mode manager if not provided
        if mode_manager is None:
            self.config = self._load_config()
            self.mode_manager = TradingModeManager(self.config, logger)
        else:
            self.mode_manager = mode_manager
            self.config = self.mode_manager.config
        
        # Initialize risk manager if not provided
        if risk_manager is None:
            self.risk_manager = RiskManager(self.config, logger)
        else:
            self.risk_manager = risk_manager
        
        # Initialize order manager if not provided
        if order_manager is None:
            self.order_manager = OrderManager(self.adapter)
        else:
            self.order_manager = order_manager
        
        # Initialize position manager if not provided
        if position_manager is None:
            self.position_manager = PositionManagerWrapper(self.config, logger)
        else:
            self.position_manager = position_manager
    
    def _load_config(self) -> Dict[str, Any]:
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
    
    def connect(self) -> bool:
        """
        Connect to the exchange.
        
        Returns:
            True if connected, False otherwise
        """
        try:
            # Connect to exchange
            result = self.adapter.connect()
            
            # Update connection status
            self.is_connected = self.adapter.is_connected
            
            return result
        except Exception as e:
            self.logger.error(f"Error connecting to exchange: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from the exchange.
        
        Returns:
            True if disconnected, False otherwise
        """
        try:
            # Disconnect from exchange
            self.is_connected = False
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting from exchange: {e}")
            return False
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get connection status.
        
        Returns:
            Dict containing connection status
        """
        try:
            # Update connection status
            self.is_connected = self.adapter.is_connected
            
            return {
                "connected": self.is_connected,
                "exchange": "Hyperliquid",
                "mode": self.mode_manager.get_current_mode().name
            }
        except Exception as e:
            self.logger.error(f"Error getting connection status: {e}")
            return {
                "connected": False,
                "exchange": "Hyperliquid",
                "error": str(e)
            }
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dict containing account information
        """
        try:
            # Get account info
            return self.adapter.get_account_info()
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {"error": f"Error getting account info: {e}"}
    
    def get_all_available_tokens(self) -> Dict[str, Any]:
        """
        Get all available tokens from the exchange.
        
        Returns:
            Dict containing list of available tokens
        """
        try:
            # Get all available tokens
            return self.adapter.get_all_available_tokens()
        except Exception as e:
            self.logger.error(f"Error getting available tokens: {e}")
            return {"error": f"Error getting available tokens: {e}", "tokens": []}
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get market data for a symbol.
        
        Args:
            symbol: The symbol to get market data for
            
        Returns:
            Dict containing market data
        """
        try:
            # Get market data
            return self.adapter.get_market_data(symbol)
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
            # Get positions
            return self.adapter.get_positions()
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
            # Get orders
            return self.adapter.get_orders()
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
            # Check if trading is allowed in current mode
            if not self.mode_manager.is_trading_allowed():
                return {"error": f"Trading not allowed in {self.mode_manager.get_current_mode().name} mode"}
            
            # Place order
            return self.adapter.place_order(symbol, is_buy, size, price, order_type)
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
            # Cancel order
            return self.adapter.cancel_order(order_id)
        except Exception as e:
            self.logger.error(f"Error canceling order: {e}")
            return {"error": f"Error canceling order: {e}"}
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancel all orders.
        
        Args:
            symbol: The symbol to cancel orders for (optional)
            
        Returns:
            Dict containing the result of the operation
        """
        try:
            # Cancel all orders
            return self.adapter.cancel_all_orders(symbol)
        except Exception as e:
            self.logger.error(f"Error canceling all orders: {e}")
            return {"error": f"Error canceling all orders: {e}"}
    
    def close_position(self, symbol: str, is_long: bool) -> Dict[str, Any]:
        """
        Close a position.
        
        Args:
            symbol: The symbol to close position for
            is_long: Whether the position is long
            
        Returns:
            Dict containing the result of the operation
        """
        try:
            # Close position
            return self.position_manager.close_position(symbol, is_long)
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return {"error": f"Error closing position: {e}"}
    
    def close_all_positions(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Close all positions.
        
        Args:
            symbol: The symbol to close positions for (optional)
            
        Returns:
            Dict containing the result of the operation
        """
        try:
            # Close all positions
            return self.position_manager.close_all_positions(symbol)
        except Exception as e:
            self.logger.error(f"Error closing all positions: {e}")
            return {"error": f"Error closing all positions: {e}"}
    
    def set_trading_mode(self, mode_name: str) -> Dict[str, Any]:
        """
        Set trading mode.
        
        Args:
            mode_name: The name of the trading mode to set
            
        Returns:
            Dict containing the result of the operation
        """
        try:
            # Set trading mode
            from .trading_mode import TradingMode
            
            # Convert mode name to enum
            try:
                mode = TradingMode[mode_name]
            except (KeyError, ValueError):
                return {"error": f"Invalid trading mode: {mode_name}"}
            
            # Set mode
            result = self.mode_manager.set_mode(mode)
            
            if result:
                return {"success": True, "message": f"Trading mode set to {mode_name}"}
            else:
                return {"error": f"Error setting trading mode to {mode_name}"}
        except Exception as e:
            self.logger.error(f"Error setting trading mode: {e}")
            return {"error": f"Error setting trading mode: {e}"}
    
    def get_trading_mode(self) -> Dict[str, Any]:
        """
        Get current trading mode.
        
        Returns:
            Dict containing the current trading mode
        """
        try:
            # Get trading mode
            mode = self.mode_manager.get_current_mode()
            
            return {
                "mode": mode.name,
                "description": self.mode_manager.get_mode_description(mode),
                "trading_allowed": self.mode_manager.is_trading_allowed()
            }
        except Exception as e:
            self.logger.error(f"Error getting trading mode: {e}")
            return {"error": f"Error getting trading mode: {e}"}
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get risk metrics.
        
        Returns:
            Dict containing risk metrics
        """
        try:
            # Get risk metrics
            return self.risk_manager.get_risk_metrics()
        except Exception as e:
            self.logger.error(f"Error getting risk metrics: {e}")
            return {"error": f"Error getting risk metrics: {e}"}
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss_price: float) -> Dict[str, Any]:
        """
        Calculate position size based on risk parameters.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_price: Stop loss price
            
        Returns:
            Dict containing calculated position size
        """
        try:
            # Get account info for equity
            account_info = self.get_account_info()
            
            if "error" in account_info:
                return {"error": account_info["error"]}
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                symbol=symbol,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                account_equity=account_info.get("equity", 0)
            )
            
            return {"success": True, "position_size": position_size}
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return {"error": f"Error calculating position size: {e}"}
    
    def validate_risk_reward_ratio(self, entry_price: float, stop_loss_price: float, take_profit_price: float) -> Dict[str, Any]:
        """
        Validate if a trade setup meets the minimum risk-reward ratio.
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            
        Returns:
            Dict containing validation result
        """
        try:
            # Calculate risk-reward ratio
            ratio = self.risk_manager.calculate_risk_reward_ratio(
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price
            )
            
            # Validate ratio
            is_valid = self.risk_manager.validate_risk_reward_ratio(
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price
            )
            
            return {
                "success": True,
                "ratio": ratio,
                "is_valid": is_valid
            }
        except Exception as e:
            self.logger.error(f"Error validating risk-reward ratio: {e}")
            return {"error": f"Error validating risk-reward ratio: {e}"}
    
    def check_drawdown_protection(self) -> Dict[str, Any]:
        """
        Check if trading should be paused due to drawdown protection.
        
        Returns:
            Dict containing drawdown protection status
        """
        try:
            # Check drawdown protection
            trading_allowed = self.risk_manager.check_drawdown_protection()
            
            return {
                "success": True,
                "trading_allowed": trading_allowed
            }
        except Exception as e:
            self.logger.error(f"Error checking drawdown protection: {e}")
            return {"error": f"Error checking drawdown protection: {e}"}
