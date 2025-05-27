"""
Position Manager for the Hyperliquid trading bot.
Provides a simplified interface for the AdvancedPositionManager.
"""

import logging
import asyncio
from typing import Dict, Any, Optional

from core.position_manager import AdvancedPositionManager

class PositionManager:
    """
    Position manager that provides a simplified interface to the AdvancedPositionManager.
    Handles opening and closing positions for both long and short trades.
    """
    
    def __init__(self, adapter, logger=None):
        """
        Initialize the position manager with the trading adapter.
        
        Args:
            adapter: The HyperliquidAdapter instance
            logger: Optional logger instance
        """
        self.adapter = adapter
        self.logger = logger or logging.getLogger("PositionManager")
        self.advanced_manager = AdvancedPositionManager(adapter, logger)
    
    async def open_position(self, symbol: str, size: float, is_long: bool, 
                           price: Optional[float] = None, reduce_only: bool = False) -> Dict[str, Any]:
        """
        Open a position.
        
        Args:
            symbol: Trading symbol
            size: Position size (positive for long, negative for short)
            is_long: Whether this is a long position
            price: Optional limit price (None for market orders)
            reduce_only: Whether this order should only reduce position
            
        Returns:
            Dict containing the result of the operation
        """
        try:
            self.logger.info(f"Opening {'long' if is_long else 'short'} position for {symbol} with size {size}")
            
            # Ensure size has correct sign
            if is_long and size < 0:
                size = abs(size)
            elif not is_long and size > 0:
                size = -abs(size)
            
            # Create signal dict for the advanced manager
            signal = {
                "signal": "BUY" if is_long else "SELL",
                "price": price or 0.0,
                "confidence": 0.8,  # Default high confidence
                "stop_loss": price * 0.95 if is_long else price * 1.05 if price else 0.0,
            }
            
            # Override the calculated position size
            original_calculate_position_size = self.advanced_manager.calculate_position_size
            
            def override_size(*args, **kwargs):
                return abs(size)
            
            # Apply the override
            self.advanced_manager.calculate_position_size = override_size
            
            # Place the order
            result = self.advanced_manager.open_position(symbol, signal)
            
            # Restore the original method
            self.advanced_manager.calculate_position_size = original_calculate_position_size
            
            # Process result
            if "error" in result:
                self.logger.error(f"Error opening position: {result['error']}")
                return {"success": False, "message": result["error"]}
            
            self.logger.info(f"Position opened successfully: {result}")
            return {
                "success": True,
                "order_id": result.get("order", {}).get("order_id", "unknown"),
                "action": result.get("action", "OPEN"),
                "direction": result.get("direction", "LONG" if is_long else "SHORT"),
                "size": result.get("size", size)
            }
            
        except Exception as e:
            self.logger.error(f"Error opening position: {e}")
            return {"success": False, "message": f"Error opening position: {e}"}
    
    async def close_position(self, symbol: str, is_long: bool, 
                            price: Optional[float] = None, percentage: float = 100.0) -> Dict[str, Any]:
        """
        Close a position.
        
        Args:
            symbol: Trading symbol
            is_long: Whether this is a long position to close
            price: Optional limit price (None for market orders)
            percentage: Percentage of position to close (0-100)
            
        Returns:
            Dict containing the result of the operation
        """
        try:
            self.logger.info(f"Closing {'long' if is_long else 'short'} position for {symbol} ({percentage}%)")
            
            # Get current positions to check if we have an open position
            current_positions = self.advanced_manager.get_current_positions()
            
            if symbol not in current_positions:
                self.logger.warning(f"No open position found for {symbol}")
                return {"success": False, "message": f"No open position found for {symbol}"}
            
            # Check if position direction matches
            position = current_positions[symbol]
            position_size = position.get("size", 0.0)
            position_is_long = position_size > 0
            
            if position_is_long != is_long:
                self.logger.warning(f"Position direction mismatch for {symbol}: trying to close {'long' if is_long else 'short'} but position is {'long' if position_is_long else 'short'}")
                return {"success": False, "message": f"Position direction mismatch for {symbol}"}
            
            # Close the position
            result = self.advanced_manager.close_position(symbol, percentage)
            
            # Process result
            if "error" in result:
                self.logger.error(f"Error closing position: {result['error']}")
                return {"success": False, "message": result["error"]}
            
            self.logger.info(f"Position closed successfully: {result}")
            return {
                "success": True,
                "order_id": result.get("order", {}).get("order_id", "unknown"),
                "action": result.get("action", "CLOSE"),
                "percentage": percentage
            }
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return {"success": False, "message": f"Error closing position: {e}"}
    
    async def get_positions(self) -> Dict[str, Any]:
        """
        Get current positions.
        
        Returns:
            Dict containing current positions
        """
        try:
            positions = self.advanced_manager.get_current_positions()
            return {"success": True, "data": positions}
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {"success": False, "message": f"Error getting positions: {e}"}
    
    async def get_position_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict containing position information
        """
        try:
            positions = self.advanced_manager.get_current_positions()
            
            if symbol in positions:
                return {"success": True, "data": positions[symbol]}
            else:
                return {"success": True, "data": None}
                
        except Exception as e:
            self.logger.error(f"Error getting position for {symbol}: {e}")
            return {"success": False, "message": f"Error getting position for {symbol}: {e}"}
