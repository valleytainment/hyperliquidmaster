"""
Advanced Order Types Module for HyperliquidMaster
-------------------------------------------------
Implements sophisticated order types based on expert recommendations
including TWAP, Scale Orders, and enhanced stop orders.
"""

import logging
import time
import math
import threading
from typing import Dict, Any, Optional, Tuple, List, Union, Callable

class AdvancedOrderManager:
    """
    Advanced order management system for crypto derivatives trading.
    Implements TWAP, Scale Orders, and enhanced stop orders.
    """
    
    def __init__(self, adapter=None, logger=None):
        """
        Initialize the advanced order manager.
        
        Args:
            adapter: HyperliquidAdapter instance for order execution
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger("AdvancedOrderManager")
        self.adapter = adapter
        
        # Active order tracking
        self.active_orders = {}
        self.order_threads = {}
        
        # TWAP settings
        self.default_twap_interval = 30  # seconds
        self.default_twap_slippage = 0.03  # 3%
        
        self.logger.info("Advanced Order Manager initialized")
    
    def set_adapter(self, adapter) -> None:
        """
        Set the HyperliquidAdapter instance.
        
        Args:
            adapter: HyperliquidAdapter instance
        """
        self.adapter = adapter
    
    def execute_twap_order(self, 
                          symbol: str, 
                          is_buy: bool, 
                          total_size: float, 
                          duration_seconds: int, 
                          max_slippage: Optional[float] = None,
                          interval_seconds: Optional[int] = None,
                          callback: Optional[Callable] = None) -> str:
        """
        Execute a Time-Weighted Average Price (TWAP) order.
        
        Args:
            symbol: Trading pair symbol
            is_buy: True for buy orders, False for sell orders
            total_size: Total position size to execute
            duration_seconds: Total duration for execution in seconds
            max_slippage: Maximum allowed slippage from current price (default: 3%)
            interval_seconds: Time between sub-orders in seconds (default: 30s)
            callback: Optional callback function to call after completion
            
        Returns:
            Order ID for tracking
        """
        if not self.adapter:
            self.logger.error("No adapter set for order execution")
            return ""
        
        # Use default values if not provided
        interval = interval_seconds if interval_seconds is not None else self.default_twap_interval
        slippage = max_slippage if max_slippage is not None else self.default_twap_slippage
        
        # Calculate number of sub-orders
        num_intervals = max(1, math.floor(duration_seconds / interval))
        size_per_interval = total_size / num_intervals
        
        # Generate unique order ID
        order_id = f"twap_{symbol}_{int(time.time())}"
        
        # Store order details
        self.active_orders[order_id] = {
            "type": "TWAP",
            "symbol": symbol,
            "is_buy": is_buy,
            "total_size": total_size,
            "executed_size": 0.0,
            "remaining_size": total_size,
            "intervals": num_intervals,
            "interval_size": size_per_interval,
            "interval_seconds": interval,
            "max_slippage": slippage,
            "start_time": time.time(),
            "end_time": time.time() + duration_seconds,
            "status": "active",
            "sub_orders": [],
            "callback": callback
        }
        
        # Start execution thread
        thread = threading.Thread(
            target=self._execute_twap_thread,
            args=(order_id,),
            daemon=True
        )
        self.order_threads[order_id] = thread
        thread.start()
        
        self.logger.info(f"TWAP order {order_id} started: {symbol}, size: {total_size}, intervals: {num_intervals}")
        return order_id
    
    def _execute_twap_thread(self, order_id: str) -> None:
        """
        Thread function to execute TWAP order.
        
        Args:
            order_id: TWAP order ID
        """
        if order_id not in self.active_orders:
            self.logger.error(f"TWAP order {order_id} not found")
            return
        
        order = self.active_orders[order_id]
        
        try:
            intervals_completed = 0
            
            while intervals_completed < order["intervals"] and order["status"] == "active":
                # Get current market price
                market_data = self.adapter.get_market_data(order["symbol"])
                current_price = market_data.get("price", 0.0)
                
                if current_price <= 0:
                    self.logger.warning(f"Invalid price for {order['symbol']}, retrying in 5 seconds")
                    time.sleep(5)
                    continue
                
                # Calculate limit price with slippage
                if order["is_buy"]:
                    limit_price = current_price * (1 + order["max_slippage"])
                else:
                    limit_price = current_price * (1 - order["max_slippage"])
                
                # Place sub-order
                sub_order_result = self.adapter.place_order(
                    symbol=order["symbol"],
                    is_buy=order["is_buy"],
                    size=order["interval_size"],
                    price=limit_price,
                    order_type="LIMIT"
                )
                
                # Track sub-order
                if "error" not in sub_order_result:
                    sub_order_id = sub_order_result.get("data", {}).get("order_id", "unknown")
                    order["sub_orders"].append({
                        "id": sub_order_id,
                        "size": order["interval_size"],
                        "price": limit_price,
                        "time": time.time()
                    })
                    
                    order["executed_size"] += order["interval_size"]
                    order["remaining_size"] -= order["interval_size"]
                    
                    self.logger.info(f"TWAP sub-order placed: {sub_order_id}, {intervals_completed+1}/{order['intervals']}")
                else:
                    self.logger.error(f"TWAP sub-order failed: {sub_order_result.get('error')}")
                
                intervals_completed += 1
                
                # Sleep until next interval if not the last one
                if intervals_completed < order["intervals"]:
                    time.sleep(order["interval_seconds"])
            
            # Update order status
            order["status"] = "completed"
            self.logger.info(f"TWAP order {order_id} completed: {intervals_completed}/{order['intervals']} intervals")
            
            # Call callback if provided
            if order["callback"] and callable(order["callback"]):
                order["callback"](order_id, order)
                
        except Exception as e:
            self.logger.error(f"Error executing TWAP order {order_id}: {e}")
            order["status"] = "error"
            
            # Call callback if provided
            if order["callback"] and callable(order["callback"]):
                order["callback"](order_id, order)
    
    def cancel_twap_order(self, order_id: str) -> bool:
        """
        Cancel an active TWAP order.
        
        Args:
            order_id: TWAP order ID
            
        Returns:
            True if canceled successfully, False otherwise
        """
        if order_id not in self.active_orders:
            self.logger.error(f"TWAP order {order_id} not found")
            return False
        
        order = self.active_orders[order_id]
        
        # Update status to trigger thread termination
        order["status"] = "canceled"
        
        # Cancel any open sub-orders
        for sub_order in order["sub_orders"]:
            if sub_order.get("status", "") != "filled":
                self.adapter.cancel_order(order["symbol"], sub_order["id"])
        
        self.logger.info(f"TWAP order {order_id} canceled")
        return True
    
    def execute_scale_orders(self, 
                            symbol: str, 
                            is_buy: bool, 
                            total_size: float, 
                            price_range: Tuple[float, float], 
                            num_orders: int,
                            distribution: str = "linear") -> List[Dict[str, Any]]:
        """
        Execute multiple limit orders distributed across a price range.
        
        Args:
            symbol: Trading pair symbol
            is_buy: True for buy orders, False for sell orders
            total_size: Total position size to be distributed
            price_range: Tuple of (start_price, end_price)
            num_orders: Number of orders to create
            distribution: Distribution type (linear, exponential)
            
        Returns:
            List of order results
        """
        if not self.adapter:
            self.logger.error("No adapter set for order execution")
            return []
        
        start_price, end_price = price_range
        
        # Ensure start_price < end_price for buy orders and start_price > end_price for sell orders
        if is_buy and start_price > end_price:
            start_price, end_price = end_price, start_price
        elif not is_buy and start_price < end_price:
            start_price, end_price = end_price, start_price
        
        # Calculate sizes and prices based on distribution
        sizes = []
        prices = []
        
        if distribution == "exponential":
            # Exponential distribution - more size at better prices
            base = 1.5  # Exponential base
            total_weight = sum(base ** i for i in range(num_orders))
            
            for i in range(num_orders):
                weight = base ** i
                size = (weight / total_weight) * total_size
                sizes.append(size)
                
                # Calculate price
                progress = i / (num_orders - 1) if num_orders > 1 else 0
                price = start_price + progress * (end_price - start_price)
                prices.append(price)
                
            # Reverse sizes for sell orders (more size at higher prices)
            if not is_buy:
                sizes.reverse()
        else:
            # Linear distribution - equal size for all orders
            size_per_order = total_size / num_orders
            sizes = [size_per_order] * num_orders
            
            # Calculate prices
            for i in range(num_orders):
                progress = i / (num_orders - 1) if num_orders > 1 else 0
                price = start_price + progress * (end_price - start_price)
                prices.append(price)
        
        # Place orders
        order_results = []
        for i in range(num_orders):
            result = self.adapter.place_order(
                symbol=symbol,
                is_buy=is_buy,
                size=sizes[i],
                price=prices[i],
                order_type="LIMIT"
            )
            
            order_results.append(result)
            
            # Small delay between orders to avoid rate limiting
            time.sleep(0.1)
        
        self.logger.info(f"Placed {num_orders} scale orders for {symbol}, total size: {total_size}")
        return order_results
    
    def place_stop_limit_order(self, 
                              symbol: str, 
                              is_buy: bool, 
                              size: float, 
                              stop_price: float, 
                              limit_price: float,
                              reduce_only: bool = False) -> Dict[str, Any]:
        """
        Place a stop-limit order.
        
        Args:
            symbol: Trading pair symbol
            is_buy: True for buy orders, False for sell orders
            size: Order size
            stop_price: Price at which the order is triggered
            limit_price: Limit price for execution after trigger
            reduce_only: Whether the order should only reduce position
            
        Returns:
            Order result
        """
        if not self.adapter:
            self.logger.error("No adapter set for order execution")
            return {"error": "No adapter set"}
        
        # Generate unique order ID
        order_id = f"stop_limit_{symbol}_{int(time.time())}"
        
        # Store order details
        self.active_orders[order_id] = {
            "type": "STOP_LIMIT",
            "symbol": symbol,
            "is_buy": is_buy,
            "size": size,
            "stop_price": stop_price,
            "limit_price": limit_price,
            "reduce_only": reduce_only,
            "status": "pending",
            "start_time": time.time(),
            "exchange_order_id": None
        }
        
        # Start monitoring thread
        thread = threading.Thread(
            target=self._monitor_stop_order,
            args=(order_id,),
            daemon=True
        )
        self.order_threads[order_id] = thread
        thread.start()
        
        self.logger.info(f"Stop-limit order {order_id} created: {symbol}, size: {size}, stop: {stop_price}, limit: {limit_price}")
        return {"order_id": order_id, "status": "pending"}
    
    def _monitor_stop_order(self, order_id: str) -> None:
        """
        Thread function to monitor and execute stop order.
        
        Args:
            order_id: Stop order ID
        """
        if order_id not in self.active_orders:
            self.logger.error(f"Stop order {order_id} not found")
            return
        
        order = self.active_orders[order_id]
        
        try:
            triggered = False
            
            while not triggered and order["status"] == "pending":
                # Get current market price
                market_data = self.adapter.get_market_data(order["symbol"])
                current_price = market_data.get("price", 0.0)
                
                if current_price <= 0:
                    self.logger.warning(f"Invalid price for {order['symbol']}, retrying in 5 seconds")
                    time.sleep(5)
                    continue
                
                # Check if stop price is triggered
                if (order["is_buy"] and current_price >= order["stop_price"]) or \
                   (not order["is_buy"] and current_price <= order["stop_price"]):
                    triggered = True
                    
                    # Place limit order
                    result = self.adapter.place_order(
                        symbol=order["symbol"],
                        is_buy=order["is_buy"],
                        size=order["size"],
                        price=order["limit_price"],
                        order_type="LIMIT"
                    )
                    
                    if "error" not in result:
                        order["status"] = "triggered"
                        order["exchange_order_id"] = result.get("data", {}).get("order_id", "unknown")
                        self.logger.info(f"Stop-limit order {order_id} triggered at {current_price}, limit order placed: {order['exchange_order_id']}")
                    else:
                        order["status"] = "error"
                        self.logger.error(f"Failed to place limit order after stop trigger: {result.get('error')}")
                else:
                    # Sleep before checking again
                    time.sleep(1)
            
        except Exception as e:
            self.logger.error(f"Error monitoring stop order {order_id}: {e}")
            order["status"] = "error"
    
    def cancel_stop_order(self, order_id: str) -> bool:
        """
        Cancel a pending stop order.
        
        Args:
            order_id: Stop order ID
            
        Returns:
            True if canceled successfully, False otherwise
        """
        if order_id not in self.active_orders:
            self.logger.error(f"Stop order {order_id} not found")
            return False
        
        order = self.active_orders[order_id]
        
        # Cancel only if not yet triggered
        if order["status"] == "pending":
            order["status"] = "canceled"
            self.logger.info(f"Stop order {order_id} canceled")
            return True
        elif order["status"] == "triggered" and order["exchange_order_id"]:
            # Cancel the limit order if already triggered
            result = self.adapter.cancel_order(order["symbol"], order["exchange_order_id"])
            if "error" not in result:
                order["status"] = "canceled"
                self.logger.info(f"Triggered stop-limit order {order_id} canceled")
                return True
            else:
                self.logger.error(f"Failed to cancel triggered stop-limit order: {result.get('error')}")
                return False
        else:
            self.logger.warning(f"Cannot cancel stop order {order_id} with status {order['status']}")
            return False
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get the status of an advanced order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status details
        """
        if order_id not in self.active_orders:
            return {"error": f"Order {order_id} not found"}
        
        return self.active_orders[order_id]
    
    def cleanup_completed_orders(self, max_age_hours: int = 24) -> int:
        """
        Clean up completed, canceled, or error orders older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours to keep completed orders
            
        Returns:
            Number of orders cleaned up
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        orders_to_remove = []
        
        for order_id, order in self.active_orders.items():
            if order["status"] in ["completed", "canceled", "error"]:
                order_age = current_time - order.get("start_time", current_time)
                if order_age > max_age_seconds:
                    orders_to_remove.append(order_id)
        
        # Remove orders
        for order_id in orders_to_remove:
            del self.active_orders[order_id]
            
            # Clean up thread reference if exists
            if order_id in self.order_threads:
                del self.order_threads[order_id]
        
        self.logger.info(f"Cleaned up {len(orders_to_remove)} old orders")
        return len(orders_to_remove)
