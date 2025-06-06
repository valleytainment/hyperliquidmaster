"""
Base Strategy for Hyperliquid Trading Bot
"""

import os
import sys
import time
import logging
from typing import Dict, Any, Optional, List
import json

from utils.logger import get_logger
from risk_management.risk_manager import RiskManager

# Re-export classes from trading_types.py
from strategies.trading_types import TradingSignal, SignalType, MarketData

# Add OrderType enum that was missing
from enum import Enum

class OrderType(Enum):
    """Order type enum"""
    LIMIT = "limit"
    MARKET = "market"

logger = get_logger(__name__)


class BaseStrategy:
    """
    Base Strategy for Hyperliquid Trading Bot
    """
    
    def __init__(self, api=None, risk_manager=None, max_positions=3):
        """
        Initialize the base strategy
        
        Args:
            api: API instance
            risk_manager: Risk manager instance
            max_positions: Maximum number of positions
        """
        self.name = self.__class__.__name__
        self.api = api
        self.risk_manager = risk_manager or RiskManager()
        self.max_positions = max_positions
        
        # Initialize state
        self.positions = {}
        self.orders = {}
        self.performance = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0
        }
        
        # Initialize parameters
        self.parameters = self.get_default_parameters()
        
        logger.info(f"Strategy '{self.name}' initialized with max_positions: {max_positions}")
    
    @classmethod
    def get_default_parameters(cls):
        """
        Get default parameters
        
        Returns:
            dict: Default parameters
        """
        return {
            "max_positions": 3
        }
    
    def update_parameters(self, parameters):
        """
        Update parameters
        
        Args:
            parameters: Dictionary of parameters to update
        """
        for key, value in parameters.items():
            if key in self.parameters:
                self.parameters[key] = value
        
        # Update max positions
        if "max_positions" in parameters:
            self.max_positions = parameters["max_positions"]
        
        logger.info(f"Updated parameters for strategy '{self.name}'")
    
    def get_parameters(self):
        """
        Get parameters
        
        Returns:
            dict: Parameters
        """
        return self.parameters
    
    def execute(self):
        """
        Execute the strategy
        
        This method should be overridden by subclasses
        """
        raise NotImplementedError("Subclasses must implement execute()")
    
    def get_positions(self):
        """
        Get positions
        
        Returns:
            dict: Positions
        """
        return self.positions
    
    def get_orders(self):
        """
        Get orders
        
        Returns:
            dict: Orders
        """
        return self.orders
    
    def get_performance(self):
        """
        Get performance
        
        Returns:
            dict: Performance
        """
        return self.performance
    
    def reset(self):
        """
        Reset the strategy
        """
        # Reset positions
        self.positions = {}
        
        # Reset orders
        self.orders = {}
        
        # Reset performance
        self.performance = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0
        }
        
        logger.info(f"Reset strategy '{self.name}'")
    
    def update_positions(self):
        """
        Update positions
        """
        try:
            if not self.api:
                logger.warning("No API instance available")
                return
            
            # Get account state
            account_state = self.api.get_account_state()
            
            if not account_state:
                logger.warning("No account state available")
                return
            
            # Update positions
            if "assetPositions" in account_state:
                positions = account_state["assetPositions"]
                
                # Clear existing positions
                self.positions = {}
                
                # Update positions
                for pos in positions:
                    if "coin" in pos and "position" in pos:
                        coin = pos["coin"]
                        position = float(pos["position"])
                        
                        self.positions[coin] = {
                            "size": position,
                            "entry_price": 0.0,  # Placeholder
                            "mark_price": 0.0,  # Placeholder
                            "pnl": 0.0  # Placeholder
                        }
        except Exception as e:
            logger.error(f"Failed to update positions: {e}")
    
    def update_orders(self):
        """
        Update orders
        """
        try:
            if not self.api:
                logger.warning("No API instance available")
                return
            
            # Get open orders
            orders = self.api.get_open_orders()
            
            if not orders:
                # Clear existing orders
                self.orders = {}
                return
            
            # Clear existing orders
            self.orders = {}
            
            # Update orders
            for order in orders:
                if "coin" in order and "side" in order and "size" in order and "price" in order:
                    coin = order["coin"]
                    side = order["side"]
                    size = float(order["size"])
                    price = float(order["price"])
                    
                    if coin not in self.orders:
                        self.orders[coin] = []
                    
                    self.orders[coin].append({
                        "side": side,
                        "size": size,
                        "price": price,
                        "status": "open"
                    })
        except Exception as e:
            logger.error(f"Failed to update orders: {e}")
    
    def update_performance(self, trade_result):
        """
        Update performance
        
        Args:
            trade_result: Dictionary with trade result
                - pnl: Profit/loss
                - win: True if winning trade, False otherwise
        """
        try:
            # Update total trades
            self.performance["total_trades"] += 1
            
            # Update winning/losing trades
            if trade_result["win"]:
                self.performance["winning_trades"] += 1
            else:
                self.performance["losing_trades"] += 1
            
            # Update total PnL
            self.performance["total_pnl"] += trade_result["pnl"]
            
            # Update win rate
            if self.performance["total_trades"] > 0:
                self.performance["win_rate"] = self.performance["winning_trades"] / self.performance["total_trades"]
            
            # Update max drawdown (placeholder)
            # This would require more sophisticated tracking
        except Exception as e:
            logger.error(f"Failed to update performance: {e}")
    
    def place_order(self, coin, side, size, price=None, order_type="limit"):
        """
        Place an order
        
        Args:
            coin: Coin to trade
            side: Order side (buy or sell)
            size: Order size
            price: Order price (optional for market orders)
            order_type: Order type (limit or market)
        
        Returns:
            dict: Order result
        """
        try:
            if not self.api:
                logger.warning("No API instance available")
                return None
            
            # Check risk limits
            if not self.risk_manager.check_order(coin, side, size, price):
                logger.warning(f"Order rejected by risk manager: {coin} {side} {size} @ {price}")
                return None
            
            # Place order
            if order_type == "limit" and price is not None:
                result = self.api.place_limit_order(coin, side, size, price)
            else:
                result = self.api.place_market_order(coin, side, size)
            
            # Update orders
            self.update_orders()
            
            return result
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None
    
    def cancel_order(self, order_id):
        """
        Cancel an order
        
        Args:
            order_id: Order ID
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.api:
                logger.warning("No API instance available")
                return False
            
            # Cancel order
            result = self.api.cancel_order(order_id)
            
            # Update orders
            self.update_orders()
            
            return result
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
    
    def cancel_all_orders(self, coin=None):
        """
        Cancel all orders
        
        Args:
            coin: Coin to cancel orders for (optional)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.api:
                logger.warning("No API instance available")
                return False
            
            # Cancel orders
            result = self.api.cancel_all_orders(coin)
            
            # Update orders
            self.update_orders()
            
            return result
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return False
    
    def close_position(self, coin):
        """
        Close position
        
        Args:
            coin: Coin to close position for
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.api:
                logger.warning("No API instance available")
                return False
            
            # Check if position exists
            if coin not in self.positions:
                logger.warning(f"No position found for {coin}")
                return False
            
            # Get position size
            position = self.positions[coin]
            size = position["size"]
            
            # Determine side
            side = "sell" if size > 0 else "buy"
            
            # Close position
            result = self.api.place_market_order(coin, side, abs(size))
            
            # Update positions
            self.update_positions()
            
            return result is not None
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False
    
    def close_all_positions(self):
        """
        Close all positions
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.api:
                logger.warning("No API instance available")
                return False
            
            # Close each position
            success = True
            for coin in list(self.positions.keys()):
                if not self.close_position(coin):
                    success = False
            
            return success
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return False
    
    def get_market_data(self, coin, timeframe="1h", limit=100):
        """
        Get market data
        
        Args:
            coin: Coin to get data for
            timeframe: Timeframe (e.g., 1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to get
        
        Returns:
            list: List of candles
        """
        try:
            if not self.api:
                logger.warning("No API instance available")
                return []
            
            # Get market data
            return self.api.get_candles(coin, timeframe, limit)
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return []
    
    def get_ticker(self, coin):
        """
        Get ticker
        
        Args:
            coin: Coin to get ticker for
        
        Returns:
            dict: Ticker data
        """
        try:
            if not self.api:
                logger.warning("No API instance available")
                return None
            
            # Get ticker
            return self.api.get_ticker(coin)
        except Exception as e:
            logger.error(f"Failed to get ticker: {e}")
            return None
    
    def get_orderbook(self, coin, depth=10):
        """
        Get orderbook
        
        Args:
            coin: Coin to get orderbook for
            depth: Orderbook depth
        
        Returns:
            dict: Orderbook data
        """
        try:
            if not self.api:
                logger.warning("No API instance available")
                return None
            
            # Get orderbook
            return self.api.get_orderbook(coin, depth)
        except Exception as e:
            logger.error(f"Failed to get orderbook: {e}")
            return None
    
    def get_account_state(self):
        """
        Get account state
        
        Returns:
            dict: Account state
        """
        try:
            if not self.api:
                logger.warning("No API instance available")
                return None
            
            # Get account state
            return self.api.get_account_state()
        except Exception as e:
            logger.error(f"Failed to get account state: {e}")
            return None
    
    def get_account_value(self):
        """
        Get account value
        
        Returns:
            float: Account value
        """
        try:
            if not self.api:
                logger.warning("No API instance available")
                return 0.0
            
            # Get account state
            account_state = self.api.get_account_state()
            
            if not account_state:
                logger.warning("No account state available")
                return 0.0
            
            # Get account value
            if "marginSummary" in account_state and "accountValue" in account_state["marginSummary"]:
                return float(account_state["marginSummary"]["accountValue"])
            
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get account value: {e}")
            return 0.0

