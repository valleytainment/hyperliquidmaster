"""
Advanced Order Manager for HyperliquidMaster

This module provides advanced order types and execution strategies
for the HyperliquidMaster trading bot.
"""

import logging
import time
import threading
import math
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

class AdvancedOrderManager:
    """
    Manages advanced order types and execution strategies.
    
    This class provides functionality for executing advanced order types
    such as TWAP, VWAP, Iceberg, and Scale orders.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the advanced order manager.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.active_orders = {}
        self.order_threads = {}
        self.stop_flags = {}
        self.order_lock = threading.RLock()
        
        self.logger.info("Advanced order manager initialized")
    
    def execute_twap_order(self, symbol: str, is_buy: bool, is_long: bool, size: float, 
                          duration_minutes: int, slices: int = 5, 
                          stop_loss_price: Optional[float] = None, 
                          take_profit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute a Time-Weighted Average Price (TWAP) order.
        
        Args:
            symbol: Trading symbol
            is_buy: Whether this is a buy order
            is_long: Whether this is a long position
            size: Total position size
            duration_minutes: Duration in minutes
            slices: Number of order slices
            stop_loss_price: Optional stop loss price
            take_profit_price: Optional take profit price
            
        Returns:
            Dictionary containing order result information
        """
        try:
            self.logger.info(f"Executing TWAP order for {symbol}: {'BUY' if is_buy else 'SELL'} {'LONG' if is_long else 'SHORT'}, Size: {size}, Duration: {duration_minutes}min, Slices: {slices}")
            
            # Generate unique order ID
            order_id = f"twap_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Calculate slice size
            slice_size = size / slices
            
            # Calculate time interval between slices
            interval_seconds = (duration_minutes * 60) / slices
            
            # Create order information
            order_info = {
                "id": order_id,
                "symbol": symbol,
                "is_buy": is_buy,
                "is_long": is_long,
                "total_size": size,
                "slice_size": slice_size,
                "slices": slices,
                "duration_minutes": duration_minutes,
                "interval_seconds": interval_seconds,
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
                "start_time": datetime.now(),
                "end_time": datetime.now() + timedelta(minutes=duration_minutes),
                "completed_slices": 0,
                "executed_size": 0.0,
                "average_price": 0.0,
                "status": "active",
                "order_ids": []
            }
            
            # Store order information
            with self.order_lock:
                self.active_orders[order_id] = order_info
                self.stop_flags[order_id] = threading.Event()
            
            # Start order execution thread
            thread = threading.Thread(target=self._execute_twap_thread, args=(order_id,))
            thread.daemon = True
            thread.start()
            
            with self.order_lock:
                self.order_threads[order_id] = thread
            
            return {
                "success": True,
                "order_id": order_id,
                "message": f"TWAP order started: {slices} slices over {duration_minutes} minutes"
            }
        except Exception as e:
            self.logger.error(f"Error executing TWAP order: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _execute_twap_thread(self, order_id: str) -> None:
        """
        Thread function for executing TWAP order.
        
        Args:
            order_id: Order ID
        """
        try:
            order_info = self.active_orders[order_id]
            stop_flag = self.stop_flags[order_id]
            
            self.logger.info(f"Starting TWAP execution thread for order {order_id}")
            
            total_executed = 0.0
            total_value = 0.0
            
            for i in range(order_info["slices"]):
                # Check if order should be stopped
                if stop_flag.is_set():
                    self.logger.info(f"TWAP order {order_id} stopped after {i} slices")
                    break
                
                # Calculate slice size (last slice may be adjusted for rounding errors)
                remaining = order_info["total_size"] - total_executed
                slices_left = order_info["slices"] - i
                
                if slices_left == 1:
                    # Last slice, use remaining size
                    slice_size = remaining
                else:
                    slice_size = order_info["slice_size"]
                
                # Execute slice
                result = self._execute_order_slice(
                    symbol=order_info["symbol"],
                    is_buy=order_info["is_buy"],
                    is_long=order_info["is_long"],
                    size=slice_size
                )
                
                if result.get("success", False):
                    # Update order information
                    with self.order_lock:
                        order_info["completed_slices"] += 1
                        order_info["executed_size"] += slice_size
                        order_info["order_ids"].append(result.get("order_id", "unknown"))
                        
                        # Update average price
                        slice_price = result.get("price", 0.0)
                        if slice_price > 0:
                            total_executed += slice_size
                            total_value += slice_size * slice_price
                            
                            if total_executed > 0:
                                order_info["average_price"] = total_value / total_executed
                    
                    self.logger.info(f"TWAP slice {i+1}/{order_info['slices']} executed: Size {slice_size}, Price {slice_price}")
                else:
                    self.logger.error(f"Error executing TWAP slice {i+1}/{order_info['slices']}: {result.get('error', 'Unknown error')}")
                
                # Wait for next slice
                if i < order_info["slices"] - 1:
                    wait_time = order_info["interval_seconds"]
                    self.logger.debug(f"Waiting {wait_time:.2f} seconds for next TWAP slice")
                    
                    # Wait with periodic checks for stop flag
                    wait_start = time.time()
                    while time.time() - wait_start < wait_time:
                        if stop_flag.is_set():
                            self.logger.info(f"TWAP order {order_id} stopped during wait period")
                            break
                        time.sleep(min(1.0, wait_time - (time.time() - wait_start)))
                    
                    if stop_flag.is_set():
                        break
            
            # Set stop loss and take profit if specified
            if order_info["stop_loss_price"] or order_info["take_profit_price"]:
                self._set_stop_loss_take_profit(
                    symbol=order_info["symbol"],
                    is_long=order_info["is_long"],
                    stop_loss_price=order_info["stop_loss_price"],
                    take_profit_price=order_info["take_profit_price"]
                )
            
            # Update order status
            with self.order_lock:
                if order_info["completed_slices"] == order_info["slices"]:
                    order_info["status"] = "completed"
                elif stop_flag.is_set():
                    order_info["status"] = "stopped"
                else:
                    order_info["status"] = "partial"
            
            self.logger.info(f"TWAP order {order_id} {order_info['status']}: {order_info['completed_slices']}/{order_info['slices']} slices, Average price: {order_info['average_price']}")
        except Exception as e:
            self.logger.error(f"Error in TWAP execution thread for order {order_id}: {e}")
            
            # Update order status
            with self.order_lock:
                if order_id in self.active_orders:
                    self.active_orders[order_id]["status"] = "error"
    
    def execute_scale_order(self, symbol: str, is_buy: bool, is_long: bool, total_size: float,
                           price_range_percent: float, num_orders: int,
                           base_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute a Scale order (multiple orders at different price levels).
        
        Args:
            symbol: Trading symbol
            is_buy: Whether this is a buy order
            is_long: Whether this is a long position
            total_size: Total position size
            price_range_percent: Price range as percentage
            num_orders: Number of orders
            base_price: Base price (optional, uses current price if not provided)
            
        Returns:
            Dictionary containing order result information
        """
        try:
            self.logger.info(f"Executing Scale order for {symbol}: {'BUY' if is_buy else 'SELL'} {'LONG' if is_long else 'SHORT'}, Size: {total_size}, Range: {price_range_percent}%, Orders: {num_orders}")
            
            # Get current price if base price not provided
            if base_price is None:
                base_price = self._get_current_price(symbol)
                
                if base_price is None:
                    return {
                        "success": False,
                        "error": "Failed to get current price"
                    }
            
            # Generate unique order ID
            order_id = f"scale_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Calculate price range
            price_range = base_price * (price_range_percent / 100.0)
            
            # Calculate price step
            price_step = price_range / (num_orders - 1) if num_orders > 1 else 0
            
            # Calculate order size
            order_size = total_size / num_orders
            
            # Create order information
            order_info = {
                "id": order_id,
                "symbol": symbol,
                "is_buy": is_buy,
                "is_long": is_long,
                "total_size": total_size,
                "order_size": order_size,
                "base_price": base_price,
                "price_range_percent": price_range_percent,
                "price_range": price_range,
                "price_step": price_step,
                "num_orders": num_orders,
                "start_time": datetime.now(),
                "completed_orders": 0,
                "executed_size": 0.0,
                "average_price": 0.0,
                "status": "active",
                "order_ids": []
            }
            
            # Store order information
            with self.order_lock:
                self.active_orders[order_id] = order_info
                self.stop_flags[order_id] = threading.Event()
            
            # Start order execution thread
            thread = threading.Thread(target=self._execute_scale_thread, args=(order_id,))
            thread.daemon = True
            thread.start()
            
            with self.order_lock:
                self.order_threads[order_id] = thread
            
            return {
                "success": True,
                "order_id": order_id,
                "message": f"Scale order started: {num_orders} orders over {price_range_percent}% range"
            }
        except Exception as e:
            self.logger.error(f"Error executing Scale order: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _execute_scale_thread(self, order_id: str) -> None:
        """
        Thread function for executing Scale order.
        
        Args:
            order_id: Order ID
        """
        try:
            order_info = self.active_orders[order_id]
            stop_flag = self.stop_flags[order_id]
            
            self.logger.info(f"Starting Scale execution thread for order {order_id}")
            
            total_executed = 0.0
            total_value = 0.0
            
            # Calculate prices for each order
            prices = []
            
            if order_info["is_buy"]:
                # For buy orders, prices decrease from base price
                start_price = order_info["base_price"]
                for i in range(order_info["num_orders"]):
                    price = start_price - (i * order_info["price_step"])
                    prices.append(price)
            else:
                # For sell orders, prices increase from base price
                start_price = order_info["base_price"]
                for i in range(order_info["num_orders"]):
                    price = start_price + (i * order_info["price_step"])
                    prices.append(price)
            
            # Execute orders
            for i, price in enumerate(prices):
                # Check if order should be stopped
                if stop_flag.is_set():
                    self.logger.info(f"Scale order {order_id} stopped after {i} orders")
                    break
                
                # Calculate order size (last order may be adjusted for rounding errors)
                remaining = order_info["total_size"] - total_executed
                orders_left = order_info["num_orders"] - i
                
                if orders_left == 1:
                    # Last order, use remaining size
                    order_size = remaining
                else:
                    order_size = order_info["order_size"]
                
                # Execute order
                result = self._execute_limit_order(
                    symbol=order_info["symbol"],
                    is_buy=order_info["is_buy"],
                    is_long=order_info["is_long"],
                    size=order_size,
                    price=price
                )
                
                if result.get("success", False):
                    # Update order information
                    with self.order_lock:
                        order_info["completed_orders"] += 1
                        order_info["executed_size"] += order_size
                        order_info["order_ids"].append(result.get("order_id", "unknown"))
                        
                        # Update average price
                        total_executed += order_size
                        total_value += order_size * price
                        
                        if total_executed > 0:
                            order_info["average_price"] = total_value / total_executed
                    
                    self.logger.info(f"Scale order {i+1}/{order_info['num_orders']} executed: Size {order_size}, Price {price}")
                else:
                    self.logger.error(f"Error executing Scale order {i+1}/{order_info['num_orders']}: {result.get('error', 'Unknown error')}")
            
            # Update order status
            with self.order_lock:
                if order_info["completed_orders"] == order_info["num_orders"]:
                    order_info["status"] = "completed"
                elif stop_flag.is_set():
                    order_info["status"] = "stopped"
                else:
                    order_info["status"] = "partial"
            
            self.logger.info(f"Scale order {order_id} {order_info['status']}: {order_info['completed_orders']}/{order_info['num_orders']} orders, Average price: {order_info['average_price']}")
        except Exception as e:
            self.logger.error(f"Error in Scale execution thread for order {order_id}: {e}")
            
            # Update order status
            with self.order_lock:
                if order_id in self.active_orders:
                    self.active_orders[order_id]["status"] = "error"
    
    def _execute_order_slice(self, symbol: str, is_buy: bool, is_long: bool, size: float) -> Dict[str, Any]:
        """
        Execute a single order slice.
        
        Args:
            symbol: Trading symbol
            is_buy: Whether this is a buy order
            is_long: Whether this is a long position
            size: Order size
            
        Returns:
            Dictionary containing order result information
        """
        # This is a placeholder method that would be implemented with actual exchange API calls
        # For testing purposes, we simulate a successful order execution
        time.sleep(0.1)  # Simulate API call delay
        
        # Simulate current price
        price = self._get_current_price(symbol)
        
        if price is None:
            return {
                "success": False,
                "error": "Failed to get current price"
            }
        
        # Simulate order execution
        order_id = f"order_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        return {
            "success": True,
            "order_id": order_id,
            "price": price,
            "size": size
        }
    
    def _execute_limit_order(self, symbol: str, is_buy: bool, is_long: bool, size: float, price: float) -> Dict[str, Any]:
        """
        Execute a limit order.
        
        Args:
            symbol: Trading symbol
            is_buy: Whether this is a buy order
            is_long: Whether this is a long position
            size: Order size
            price: Limit price
            
        Returns:
            Dictionary containing order result information
        """
        # This is a placeholder method that would be implemented with actual exchange API calls
        # For testing purposes, we simulate a successful order execution
        time.sleep(0.1)  # Simulate API call delay
        
        # Simulate order execution
        order_id = f"limit_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        return {
            "success": True,
            "order_id": order_id,
            "price": price,
            "size": size
        }
    
    def _set_stop_loss_take_profit(self, symbol: str, is_long: bool, stop_loss_price: Optional[float], take_profit_price: Optional[float]) -> Dict[str, Any]:
        """
        Set stop loss and take profit orders.
        
        Args:
            symbol: Trading symbol
            is_long: Whether this is a long position
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            
        Returns:
            Dictionary containing order result information
        """
        # This is a placeholder method that would be implemented with actual exchange API calls
        # For testing purposes, we simulate successful order placement
        time.sleep(0.1)  # Simulate API call delay
        
        result = {
            "success": True,
            "orders": []
        }
        
        # Place stop loss order
        if stop_loss_price is not None:
            stop_loss_id = f"sl_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            result["orders"].append({
                "type": "stop_loss",
                "order_id": stop_loss_id,
                "price": stop_loss_price
            })
        
        # Place take profit order
        if take_profit_price is not None:
            take_profit_id = f"tp_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            result["orders"].append({
                "type": "take_profit",
                "order_id": take_profit_id,
                "price": take_profit_price
            })
        
        return result
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price or None if error
        """
        # This is a placeholder method that would be implemented with actual exchange API calls
        # For testing purposes, we simulate a price
        
        # Use a simple deterministic price based on symbol and current time
        # This ensures consistent prices for testing while still having some variation
        base_price = sum(ord(c) for c in symbol) / 10.0
        time_factor = int(time.time() / 60) % 100  # Changes every minute, cycles every 100 minutes
        
        return base_price + (time_factor / 100.0)
    
    def stop_order(self, order_id: str) -> Dict[str, Any]:
        """
        Stop an active order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Dictionary containing result information
        """
        try:
            with self.order_lock:
                if order_id not in self.active_orders:
                    return {
                        "success": False,
                        "error": f"Order {order_id} not found"
                    }
                
                # Set stop flag
                self.stop_flags[order_id].set()
                
                # Wait for thread to complete
                if order_id in self.order_threads:
                    thread = self.order_threads[order_id]
                    if thread.is_alive():
                        thread.join(timeout=5.0)
                
                # Update order status
                self.active_orders[order_id]["status"] = "stopped"
            
            self.logger.info(f"Order {order_id} stopped")
            
            return {
                "success": True,
                "message": f"Order {order_id} stopped"
            }
        except Exception as e:
            self.logger.error(f"Error stopping order {order_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get status of an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Dictionary containing order status information
        """
        try:
            with self.order_lock:
                if order_id not in self.active_orders:
                    return {
                        "success": False,
                        "error": f"Order {order_id} not found"
                    }
                
                # Return order information
                return {
                    "success": True,
                    "order": self.active_orders[order_id]
                }
        except Exception as e:
            self.logger.error(f"Error getting order status for {order_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_active_orders(self) -> Dict[str, Any]:
        """
        Get all active orders.
        
        Returns:
            Dictionary containing active orders information
        """
        try:
            with self.order_lock:
                # Filter active orders
                active = {
                    order_id: order_info
                    for order_id, order_info in self.active_orders.items()
                    if order_info["status"] in ["active", "partial"]
                }
                
                return {
                    "success": True,
                    "orders": active
                }
        except Exception as e:
            self.logger.error(f"Error getting active orders: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Define OrderManager as an alias for AdvancedOrderManager to maintain compatibility
class OrderManager:
    """
    OrderManager class for compatibility with test scripts.
    This is a simplified wrapper around AdvancedOrderManager.
    """
    
    def __init__(self, adapter):
        """
        Initialize the order manager.
        
        Args:
            adapter: HyperliquidAdapter instance
        """
        self.adapter = adapter
        self.logger = logging.getLogger("OrderManager")
        
        # Create config for AdvancedOrderManager
        config = {
            "adapter": adapter
        }
        
        # Create AdvancedOrderManager instance
        self.advanced_manager = AdvancedOrderManager(config, self.logger)
    
    def create_order(self, symbol: str, is_buy: bool, size: float, price: float, order_type: str = "LIMIT") -> Dict[str, Any]:
        """
        Create an order.
        
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
            # Determine if this is a long or short position
            is_long = is_buy
            
            if order_type == "MARKET":
                # Execute market order
                return self._execute_order_slice(symbol, is_buy, is_long, size)
            else:
                # Execute limit order
                return self._execute_limit_order(symbol, is_buy, is_long, size, price)
        except Exception as e:
            self.logger.error(f"Error creating order: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_twap_order(self, symbol: str, is_buy: bool, size: float, duration_minutes: int, price: float = None,
                         stop_loss_price: Optional[float] = None, take_profit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Create a TWAP order.
        
        Args:
            symbol: The symbol to place an order for
            is_buy: Whether the order is a buy order
            size: The size of the order
            duration_minutes: The duration of the TWAP order in minutes
            price: The base price for the order (optional)
            stop_loss_price: Optional stop loss price
            take_profit_price: Optional take profit price
            
        Returns:
            Dict containing the result of the operation
        """
        # Determine if this is a long or short position
        is_long = is_buy
        
        # Execute TWAP order
        return self.advanced_manager.execute_twap_order(
            symbol=symbol,
            is_buy=is_buy,
            is_long=is_long,
            size=size,
            duration_minutes=duration_minutes,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price
        )
    
    def create_scale_order(self, symbol: str, is_buy: bool, size: float, price: float,
                          scale_range_percent: float, num_orders: int) -> Dict[str, Any]:
        """
        Create a Scale order.
        
        Args:
            symbol: The symbol to place an order for
            is_buy: Whether the order is a buy order
            size: The size of the order
            price: The base price for the order
            scale_range_percent: The price range as percentage
            num_orders: The number of orders
            
        Returns:
            Dict containing the result of the operation
        """
        # Determine if this is a long or short position
        is_long = is_buy
        
        # Execute Scale order
        return self.advanced_manager.execute_scale_order(
            symbol=symbol,
            is_buy=is_buy,
            is_long=is_long,
            total_size=size,
            price_range_percent=scale_range_percent,
            num_orders=num_orders,
            base_price=price
        )
    
    def _execute_order_slice(self, symbol: str, is_buy: bool, is_long: bool, size: float) -> Dict[str, Any]:
        """
        Execute a single order slice.
        
        Args:
            symbol: Trading symbol
            is_buy: Whether this is a buy order
            is_long: Whether this is a long position
            size: Order size
            
        Returns:
            Dictionary containing order result information
        """
        return self.advanced_manager._execute_order_slice(symbol, is_buy, is_long, size)
    
    def _execute_limit_order(self, symbol: str, is_buy: bool, is_long: bool, size: float, price: float) -> Dict[str, Any]:
        """
        Execute a limit order.
        
        Args:
            symbol: Trading symbol
            is_buy: Whether this is a buy order
            is_long: Whether this is a long position
            size: Order size
            price: Limit price
            
        Returns:
            Dictionary containing order result information
        """
        return self.advanced_manager._execute_limit_order(symbol, is_buy, is_long, size, price)
