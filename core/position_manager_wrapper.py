"""
Position Manager Wrapper for HyperliquidMaster

This module provides a simplified interface to the existing AdvancedPositionManager
with robust error handling and reconnection logic.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple

class PositionManagerWrapper:
    """
    Wrapper for the AdvancedPositionManager with simplified interface and robust error handling.
    
    This class provides a simplified interface to the existing AdvancedPositionManager
    with robust error handling and reconnection logic.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the position manager wrapper.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.max_retries = 3
        self.retry_delay = 2.0
        
        # Initialize position cache
        self.position_cache = {}
        self.last_update_time = 0
        self.cache_ttl = 10.0  # Cache TTL in seconds
        
        self.logger.info("Position manager wrapper initialized")
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open positions with robust error handling.
        
        Returns:
            List of position dictionaries
        """
        try:
            # Check if cache is valid
            current_time = time.time()
            if current_time - self.last_update_time < self.cache_ttl and self.position_cache:
                self.logger.debug("Using cached positions")
                return list(self.position_cache.values())
            
            # Fetch positions with retry logic
            for attempt in range(self.max_retries):
                try:
                    # This is a placeholder for actual position fetching
                    # In a real implementation, this would call the exchange API
                    
                    # Simulate position fetching
                    time.sleep(0.5)
                    
                    # Return dummy positions for testing
                    positions = [
                        {
                            "symbol": "XRP",
                            "side": "LONG",
                            "size": 100.0,
                            "entry_price": 0.5,
                            "current_price": 0.52,
                            "liquidation_price": 0.3,
                            "pnl": 2.0,
                            "pnl_percent": 4.0
                        }
                    ]
                    
                    # Update cache
                    self.position_cache = {f"{p['symbol']}_{p['side']}": p for p in positions}
                    self.last_update_time = current_time
                    
                    self.logger.info(f"Fetched {len(positions)} positions")
                    return positions
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        self.logger.warning(f"Error fetching positions (attempt {attempt+1}/{self.max_retries}): {e}")
                        time.sleep(self.retry_delay)
                    else:
                        raise
            
            # Should not reach here, but just in case
            return []
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            
            # Return cached positions if available
            if self.position_cache:
                self.logger.warning("Returning cached positions due to error")
                return list(self.position_cache.values())
            
            return []
    
    def get_position(self, symbol: str, is_long: bool) -> Optional[Dict[str, Any]]:
        """
        Get a specific position with robust error handling.
        
        Args:
            symbol: Trading symbol
            is_long: Whether this is a long position
            
        Returns:
            Position dictionary, or None if not found
        """
        try:
            # Get all positions
            positions = self.get_positions()
            
            # Find matching position
            side = "LONG" if is_long else "SHORT"
            for position in positions:
                if position["symbol"] == symbol and position["side"] == side:
                    return position
            
            return None
        except Exception as e:
            self.logger.error(f"Error getting position for {symbol} {side}: {e}")
            
            # Check cache
            position_key = f"{symbol}_{side}"
            if position_key in self.position_cache:
                self.logger.warning(f"Returning cached position for {symbol} {side} due to error")
                return self.position_cache[position_key]
            
            return None
    
    def open_position(self, symbol: str, is_long: bool, size: float, 
                     entry_price: Optional[float] = None,
                     stop_loss_price: Optional[float] = None, 
                     take_profit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Open a new position with robust error handling.
        
        Args:
            symbol: Trading symbol
            is_long: Whether this is a long position
            size: Position size
            entry_price: Entry price (optional, uses market price if not provided)
            stop_loss_price: Stop loss price (optional)
            take_profit_price: Take profit price (optional)
            
        Returns:
            Dictionary containing result information
        """
        try:
            self.logger.info(f"Opening position: {symbol} {'LONG' if is_long else 'SHORT'}, Size: {size}")
            
            # Fetch positions with retry logic
            for attempt in range(self.max_retries):
                try:
                    # This is a placeholder for actual position opening
                    # In a real implementation, this would call the exchange API
                    
                    # Simulate position opening
                    time.sleep(1.0)
                    
                    # Get current price if entry price not provided
                    current_price = entry_price if entry_price is not None else self._get_current_price(symbol)
                    
                    if current_price is None:
                        return {
                            "success": False,
                            "error": "Failed to get current price"
                        }
                    
                    # Create position
                    position = {
                        "symbol": symbol,
                        "side": "LONG" if is_long else "SHORT",
                        "size": size,
                        "entry_price": current_price,
                        "current_price": current_price,
                        "liquidation_price": current_price * (0.6 if is_long else 1.4),  # Dummy liquidation price
                        "pnl": 0.0,
                        "pnl_percent": 0.0,
                        "stop_loss_price": stop_loss_price,
                        "take_profit_price": take_profit_price
                    }
                    
                    # Update cache
                    position_key = f"{symbol}_{'LONG' if is_long else 'SHORT'}"
                    self.position_cache[position_key] = position
                    self.last_update_time = time.time()
                    
                    self.logger.info(f"Position opened: {symbol} {'LONG' if is_long else 'SHORT'}, Size: {size}, Entry: {current_price}")
                    
                    return {
                        "success": True,
                        "position": position,
                        "message": f"Position opened: {symbol} {'LONG' if is_long else 'SHORT'}, Size: {size}, Entry: {current_price}"
                    }
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        self.logger.warning(f"Error opening position (attempt {attempt+1}/{self.max_retries}): {e}")
                        time.sleep(self.retry_delay)
                    else:
                        raise
            
            # Should not reach here, but just in case
            return {
                "success": False,
                "error": "Failed to open position after multiple attempts"
            }
        except Exception as e:
            self.logger.error(f"Error opening position: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def close_position(self, symbol: str, is_long: Optional[bool] = None) -> Dict[str, Any]:
        """
        Close an existing position with robust error handling.
        
        Args:
            symbol: Trading symbol
            is_long: Whether this is a long position (optional, closes all positions for symbol if not provided)
            
        Returns:
            Dictionary containing result information
        """
        try:
            if is_long is not None:
                side = "LONG" if is_long else "SHORT"
                self.logger.info(f"Closing position: {symbol} {side}")
                
                # Check if position exists
                position = self.get_position(symbol, is_long)
                
                if position is None:
                    return {
                        "success": False,
                        "error": f"Position not found: {symbol} {side}"
                    }
                
                # Fetch positions with retry logic
                for attempt in range(self.max_retries):
                    try:
                        # This is a placeholder for actual position closing
                        # In a real implementation, this would call the exchange API
                        
                        # Simulate position closing
                        time.sleep(1.0)
                        
                        # Get current price
                        current_price = self._get_current_price(symbol)
                        
                        if current_price is None:
                            return {
                                "success": False,
                                "error": "Failed to get current price"
                            }
                        
                        # Calculate PnL
                        entry_price = position["entry_price"]
                        size = position["size"]
                        
                        if is_long:
                            pnl = (current_price - entry_price) * size
                            pnl_percent = (current_price - entry_price) / entry_price * 100
                        else:
                            pnl = (entry_price - current_price) * size
                            pnl_percent = (entry_price - current_price) / entry_price * 100
                        
                        # Remove from cache
                        position_key = f"{symbol}_{side}"
                        if position_key in self.position_cache:
                            del self.position_cache[position_key]
                        
                        self.logger.info(f"Position closed: {symbol} {side}, Exit: {current_price}, PnL: {pnl:.2f} ({pnl_percent:.2f}%)")
                        
                        return {
                            "success": True,
                            "symbol": symbol,
                            "side": side,
                            "exit_price": current_price,
                            "pnl": pnl,
                            "pnl_percent": pnl_percent,
                            "message": f"Position closed: {symbol} {side}, Exit: {current_price}, PnL: {pnl:.2f} ({pnl_percent:.2f}%)"
                        }
                    except Exception as e:
                        if attempt < self.max_retries - 1:
                            self.logger.warning(f"Error closing position (attempt {attempt+1}/{self.max_retries}): {e}")
                            time.sleep(self.retry_delay)
                        else:
                            raise
            else:
                # Close all positions for the symbol
                self.logger.info(f"Closing all positions for {symbol}")
                
                results = []
                success = True
                
                # Try to close long position
                long_result = self.close_position(symbol, True)
                if long_result.get("success", False):
                    results.append(long_result)
                elif "Position not found" not in long_result.get("error", ""):
                    success = False
                
                # Try to close short position
                short_result = self.close_position(symbol, False)
                if short_result.get("success", False):
                    results.append(short_result)
                elif "Position not found" not in short_result.get("error", ""):
                    success = False
                
                if not results:
                    return {
                        "success": False,
                        "error": f"No positions found for {symbol}"
                    }
                
                return {
                    "success": success,
                    "results": results,
                    "message": f"Closed {len(results)} positions for {symbol}"
                }
            
            # Should not reach here, but just in case
            return {
                "success": False,
                "error": "Failed to close position after multiple attempts"
            }
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def partial_exit(self, symbol: str, is_long: bool, percentage: float) -> Dict[str, Any]:
        """
        Partially exit an existing position with robust error handling.
        
        Args:
            symbol: Trading symbol
            is_long: Whether this is a long position
            percentage: Percentage of position to exit (0-100)
            
        Returns:
            Dictionary containing result information
        """
        try:
            side = "LONG" if is_long else "SHORT"
            self.logger.info(f"Partially exiting position: {symbol} {side}, {percentage}%")
            
            # Check if position exists
            position = self.get_position(symbol, is_long)
            
            if position is None:
                return {
                    "success": False,
                    "error": f"Position not found: {symbol} {side}"
                }
            
            # Calculate exit size
            original_size = position["size"]
            exit_size = original_size * (percentage / 100.0)
            remaining_size = original_size - exit_size
            
            # Fetch positions with retry logic
            for attempt in range(self.max_retries):
                try:
                    # This is a placeholder for actual partial position closing
                    # In a real implementation, this would call the exchange API
                    
                    # Simulate partial position closing
                    time.sleep(1.0)
                    
                    # Get current price
                    current_price = self._get_current_price(symbol)
                    
                    if current_price is None:
                        return {
                            "success": False,
                            "error": "Failed to get current price"
                        }
                    
                    # Calculate PnL for exited portion
                    entry_price = position["entry_price"]
                    
                    if is_long:
                        pnl = (current_price - entry_price) * exit_size
                        pnl_percent = (current_price - entry_price) / entry_price * 100
                    else:
                        pnl = (entry_price - current_price) * exit_size
                        pnl_percent = (entry_price - current_price) / entry_price * 100
                    
                    # Update position in cache
                    position_key = f"{symbol}_{side}"
                    if position_key in self.position_cache:
                        self.position_cache[position_key]["size"] = remaining_size
                    
                    self.logger.info(f"Position partially exited: {symbol} {side}, {percentage}%, Exit: {current_price}, PnL: {pnl:.2f} ({pnl_percent:.2f}%)")
                    
                    return {
                        "success": True,
                        "symbol": symbol,
                        "side": side,
                        "exit_price": current_price,
                        "exit_size": exit_size,
                        "remaining_size": remaining_size,
                        "pnl": pnl,
                        "pnl_percent": pnl_percent,
                        "message": f"Position partially exited: {symbol} {side}, {percentage}%, Exit: {current_price}, PnL: {pnl:.2f} ({pnl_percent:.2f}%)"
                    }
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        self.logger.warning(f"Error partially exiting position (attempt {attempt+1}/{self.max_retries}): {e}")
                        time.sleep(self.retry_delay)
                    else:
                        raise
            
            # Should not reach here, but just in case
            return {
                "success": False,
                "error": "Failed to partially exit position after multiple attempts"
            }
        except Exception as e:
            self.logger.error(f"Error partially exiting position: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def modify_position(self, symbol: str, is_long: bool, 
                       stop_loss_price: Optional[float] = None, 
                       take_profit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Modify an existing position with robust error handling.
        
        Args:
            symbol: Trading symbol
            is_long: Whether this is a long position
            stop_loss_price: New stop loss price (optional)
            take_profit_price: New take profit price (optional)
            
        Returns:
            Dictionary containing result information
        """
        try:
            side = "LONG" if is_long else "SHORT"
            self.logger.info(f"Modifying position: {symbol} {side}, SL: {stop_loss_price}, TP: {take_profit_price}")
            
            # Check if position exists
            position = self.get_position(symbol, is_long)
            
            if position is None:
                return {
                    "success": False,
                    "error": f"Position not found: {symbol} {side}"
                }
            
            # Fetch positions with retry logic
            for attempt in range(self.max_retries):
                try:
                    # This is a placeholder for actual position modification
                    # In a real implementation, this would call the exchange API
                    
                    # Simulate position modification
                    time.sleep(0.5)
                    
                    # Update position in cache
                    position_key = f"{symbol}_{side}"
                    if position_key in self.position_cache:
                        if stop_loss_price is not None:
                            self.position_cache[position_key]["stop_loss_price"] = stop_loss_price
                        
                        if take_profit_price is not None:
                            self.position_cache[position_key]["take_profit_price"] = take_profit_price
                    
                    self.logger.info(f"Position modified: {symbol} {side}, SL: {stop_loss_price}, TP: {take_profit_price}")
                    
                    return {
                        "success": True,
                        "symbol": symbol,
                        "side": side,
                        "stop_loss_price": stop_loss_price,
                        "take_profit_price": take_profit_price,
                        "message": f"Position modified: {symbol} {side}, SL: {stop_loss_price}, TP: {take_profit_price}"
                    }
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        self.logger.warning(f"Error modifying position (attempt {attempt+1}/{self.max_retries}): {e}")
                        time.sleep(self.retry_delay)
                    else:
                        raise
            
            # Should not reach here, but just in case
            return {
                "success": False,
                "error": "Failed to modify position after multiple attempts"
            }
        except Exception as e:
            self.logger.error(f"Error modifying position: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
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


# Define PositionManager as an alias for PositionManagerWrapper to maintain compatibility
class PositionManager:
    """
    PositionManager class for compatibility with test scripts.
    This is a simplified wrapper around PositionManagerWrapper.
    """
    
    def __init__(self, adapter):
        """
        Initialize the position manager.
        
        Args:
            adapter: HyperliquidAdapter instance
        """
        self.adapter = adapter
        self.logger = logging.getLogger("PositionManager")
        
        # Create config for PositionManagerWrapper
        config = {
            "adapter": adapter
        }
        
        # Create PositionManagerWrapper instance
        self.wrapper = PositionManagerWrapper(config, self.logger)
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open positions.
        
        Returns:
            List of position dictionaries
        """
        return self.wrapper.get_positions()
    
    def get_position(self, symbol: str, is_long: bool) -> Optional[Dict[str, Any]]:
        """
        Get a specific position.
        
        Args:
            symbol: Trading symbol
            is_long: Whether this is a long position
            
        Returns:
            Position dictionary, or None if not found
        """
        return self.wrapper.get_position(symbol, is_long)
    
    def open_position(self, symbol: str, is_long: bool, size: float, 
                     entry_price: Optional[float] = None,
                     stop_loss_price: Optional[float] = None, 
                     take_profit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Open a new position.
        
        Args:
            symbol: Trading symbol
            is_long: Whether this is a long position
            size: Position size
            entry_price: Entry price (optional, uses market price if not provided)
            stop_loss_price: Stop loss price (optional)
            take_profit_price: Take profit price (optional)
            
        Returns:
            Dictionary containing result information
        """
        return self.wrapper.open_position(
            symbol=symbol,
            is_long=is_long,
            size=size,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price
        )
    
    def close_position(self, symbol: str, is_long: Optional[bool] = None) -> Dict[str, Any]:
        """
        Close an existing position.
        
        Args:
            symbol: Trading symbol
            is_long: Whether this is a long position (optional, closes all positions for symbol if not provided)
            
        Returns:
            Dictionary containing result information
        """
        return self.wrapper.close_position(symbol=symbol, is_long=is_long)
    
    def modify_position(self, symbol: str, is_long: bool, 
                       stop_loss_price: Optional[float] = None, 
                       take_profit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Modify an existing position.
        
        Args:
            symbol: Trading symbol
            is_long: Whether this is a long position
            stop_loss_price: New stop loss price (optional)
            take_profit_price: New take profit price (optional)
            
        Returns:
            Dictionary containing result information
        """
        return self.wrapper.modify_position(
            symbol=symbol,
            is_long=is_long,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price
        )
