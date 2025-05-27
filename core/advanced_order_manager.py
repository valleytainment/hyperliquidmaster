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
    
    def execute_iceberg_order(self, symbol: str, is_buy: bool, is_long: bool, total_size: float,
                             visible_size: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute an Iceberg order (large order split into smaller visible portions).
        
        Args:
            symbol: Trading symbol
            is_buy: Whether this is a buy order
            is_long: Whether this is a long position
            total_size: Total position size
            visible_size: Visible portion size
            price: Limit price (optional, uses market order if not provided)
            
        Returns:
            Dictionary containing order result information
        """
        try:
            self.logger.info(f"Executing Iceberg order for {symbol}: {'BUY' if is_buy else 'SELL'} {'LONG' if is_long else 'SHORT'}, Total Size: {total_size}, Visible Size: {visible_size}")
            
            # Generate unique order ID
            order_id = f"iceberg_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Calculate number of slices
            num_slices = math.ceil(total_size / visible_size)
            
            # Create order information
            order_info = {
                "id": order_id,
                "symbol": symbol,
                "is_buy": is_buy,
                "is_long": is_long,
                "total_size": total_size,
                "visible_size": visible_size,
                "price": price,
                "num_slices": num_slices,
                "start_time": datetime.now(),
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
            thread = threading.Thread(target=self._execute_iceberg_thread, args=(order_id,))
            thread.daemon = True
            thread.start()
            
            with self.order_lock:
                self.order_threads[order_id] = thread
            
            return {
                "success": True,
                "order_id": order_id,
                "message": f"Iceberg order started: {total_size} total size, {visible_size} visible size"
            }
        except Exception as e:
            self.logger.error(f"Error executing Iceberg order: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _execute_iceberg_thread(self, order_id: str) -> None:
        """
        Thread function for executing Iceberg order.
        
        Args:
            order_id: Order ID
        """
        try:
            order_info = self.active_orders[order_id]
            stop_flag = self.stop_flags[order_id]
            
            self.logger.info(f"Starting Iceberg execution thread for order {order_id}")
            
            total_executed = 0.0
            total_value = 0.0
            
            for i in range(order_info["num_slices"]):
                # Check if order should be stopped
                if stop_flag.is_set():
                    self.logger.info(f"Iceberg order {order_id} stopped after {i} slices")
                    break
                
                # Calculate slice size (last slice may be adjusted for rounding errors)
                remaining = order_info["total_size"] - total_executed
                
                if remaining <= order_info["visible_size"]:
                    # Last slice, use remaining size
                    slice_size = remaining
                else:
                    slice_size = order_info["visible_size"]
                
                # Execute slice
                if order_info["price"] is not None:
                    # Limit order
                    result = self._execute_limit_order(
                        symbol=order_info["symbol"],
                        is_buy=order_info["is_buy"],
                        is_long=order_info["is_long"],
                        size=slice_size,
                        price=order_info["price"]
                    )
                else:
                    # Market order
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
                    
                    self.logger.info(f"Iceberg slice {i+1}/{order_info['num_slices']} executed: Size {slice_size}, Price {slice_price}")
                    
                    # Wait for order to be filled before next slice
                    if i < order_info["num_slices"] - 1:
                        self._wait_for_order_fill(result.get("order_id", "unknown"))
                else:
                    self.logger.error(f"Error executing Iceberg slice {i+1}/{order_info['num_slices']}: {result.get('error', 'Unknown error')}")
                    
                    # Wait before retrying
                    time.sleep(5)
                    
                    # Retry this slice
                    i -= 1
                    continue
            
            # Update order status
            with self.order_lock:
                if order_info["completed_slices"] == order_info["num_slices"]:
                    order_info["status"] = "completed"
                elif stop_flag.is_set():
                    order_info["status"] = "stopped"
                else:
                    order_info["status"] = "partial"
            
            self.logger.info(f"Iceberg order {order_id} {order_info['status']}: {order_info['completed_slices']}/{order_info['num_slices']} slices, Average price: {order_info['average_price']}")
        except Exception as e:
            self.logger.error(f"Error in Iceberg execution thread for order {order_id}: {e}")
            
            # Update order status
            with self.order_lock:
                if order_id in self.active_orders:
                    self.active_orders[order_id]["status"] = "error"
    
    def stop_advanced_order(self, order_id: str) -> Dict[str, Any]:
        """
        Stop an active advanced order.
        
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
                if order_id in self.stop_flags:
                    self.stop_flags[order_id].set()
                
                # Wait for thread to finish
                if order_id in self.order_threads and self.order_threads[order_id].is_alive():
                    self.order_threads[order_id].join(timeout=5.0)
                
                # Update order status
                if self.active_orders[order_id]["status"] == "active":
                    self.active_orders[order_id]["status"] = "stopped"
                
                self.logger.info(f"Advanced order {order_id} stopped")
                
                return {
                    "success": True,
                    "order_id": order_id,
                    "status": self.active_orders[order_id]["status"],
                    "message": f"Order {order_id} stopped"
                }
        except Exception as e:
            self.logger.error(f"Error stopping advanced order {order_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_advanced_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get status of an advanced order.
        
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
                
                return {
                    "success": True,
                    "order_id": order_id,
                    "status": self.active_orders[order_id]
                }
        except Exception as e:
            self.logger.error(f"Error getting advanced order status for {order_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_all_advanced_orders(self) -> Dict[str, Any]:
        """
        Get status of all advanced orders.
        
        Returns:
            Dictionary containing all order status information
        """
        try:
            with self.order_lock:
                return {
                    "success": True,
                    "orders": self.active_orders
                }
        except Exception as e:
            self.logger.error(f"Error getting all advanced orders: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _execute_order_slice(self, symbol: str, is_buy: bool, is_long: bool, size: float) -> Dict[str, Any]:
        """
        Execute a single order slice (market order).
        
        Args:
            symbol: Trading symbol
            is_buy: Whether this is a buy order
            is_long: Whether this is a long position
            size: Order size
            
        Returns:
            Dictionary containing order result information
        """
        # This is a placeholder for actual order execution
        # In a real implementation, this would call the exchange API
        
        # Simulate order execution
        time.sleep(0.5)
        
        # Get current price
        price = self._get_current_price(symbol)
        
        if price is None:
            return {
                "success": False,
                "error": "Failed to get current price"
            }
        
        # Generate order ID
        order_id = f"order_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        self.logger.info(f"Executed order slice: {symbol} {'BUY' if is_buy else 'SELL'} {'LONG' if is_long else 'SHORT'}, Size: {size}, Price: {price}")
        
        return {
            "success": True,
            "order_id": order_id,
            "symbol": symbol,
            "is_buy": is_buy,
            "is_long": is_long,
            "size": size,
            "price": price,
            "type": "market"
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
        # This is a placeholder for actual limit order execution
        # In a real implementation, this would call the exchange API
        
        # Simulate order execution
        time.sleep(0.5)
        
        # Generate order ID
        order_id = f"limit_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        self.logger.info(f"Executed limit order: {symbol} {'BUY' if is_buy else 'SELL'} {'LONG' if is_long else 'SHORT'}, Size: {size}, Price: {price}")
        
        return {
            "success": True,
            "order_id": order_id,
            "symbol": symbol,
            "is_buy": is_buy,
            "is_long": is_long,
            "size": size,
            "price": price,
            "type": "limit"
        }
    
    def _set_stop_loss_take_profit(self, symbol: str, is_long: bool, stop_loss_price: Optional[float], take_profit_price: Optional[float]) -> Dict[str, Any]:
        """
        Set stop loss and take profit for a position.
        
        Args:
            symbol: Trading symbol
            is_long: Whether this is a long position
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            
        Returns:
            Dictionary containing result information
        """
        # This is a placeholder for actual stop loss and take profit setting
        # In a real implementation, this would call the exchange API
        
        # Simulate setting stop loss and take profit
        time.sleep(0.5)
        
        self.logger.info(f"Set stop loss and take profit: {symbol} {'LONG' if is_long else 'SHORT'}, SL: {stop_loss_price}, TP: {take_profit_price}")
        
        return {
            "success": True,
            "symbol": symbol,
            "is_long": is_long,
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price
        }
    
    def _wait_for_order_fill(self, order_id: str, timeout: float = 60.0) -> bool:
        """
        Wait for an order to be filled.
        
        Args:
            order_id: Order ID
            timeout: Timeout in seconds
            
        Returns:
            True if order filled, False if timeout
        """
        # This is a placeholder for actual order fill checking
        # In a real implementation, this would check the exchange API
        
        # Simulate waiting for order fill
        time.sleep(2.0)
        
        self.logger.debug(f"Order {order_id} filled")
        
        return True
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price, or None if failed
        """
        # This is a placeholder for actual price fetching
        # In a real implementation, this would call the exchange API
        
        # Simulate price fetching
        time.sleep(0.1)
        
        # Return dummy prices for testing
        prices = {
            "BTC": 50000.0,
            "ETH": 3000.0,
            "XRP": 0.5,
            "SOL": 100.0,
            "DOGE": 0.1,
            "AVAX": 30.0,
            "LINK": 15.0,
            "MATIC": 1.0
        }
        
        return prices.get(symbol, 1.0)
