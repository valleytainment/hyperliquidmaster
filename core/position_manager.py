"""
Advanced Position Manager for the Hyperliquid trading bot.
Handles sophisticated position management for both long and short trades.
"""

import time
import logging
import json
from typing import Dict, Any, List, Optional, Tuple

class AdvancedPositionManager:
    """
    Advanced position manager with robust handling for long and short positions.
    Implements sophisticated entry, exit, scaling, and risk management strategies.
    """
    
    def __init__(self, adapter, logger=None):
        """Initialize the position manager with the trading adapter"""
        self.adapter = adapter
        self.logger = logger or logging.getLogger("AdvancedPositionManager")
        
        # Configuration parameters
        self.config = {
            # Position management
            "max_positions": 5,                # Maximum number of concurrent positions
            "max_position_size": 0.2,          # Maximum position size as fraction of account
            "default_risk_per_trade": 0.02,    # Default risk per trade as fraction of account
            
            # Entry strategies
            "allow_scaling_in": True,          # Allow scaling into positions
            "max_scale_in_attempts": 3,        # Maximum number of scale-in attempts
            "scale_in_threshold": 0.02,        # Price movement threshold for scaling in (2%)
            
            # Exit strategies
            "allow_partial_exits": True,       # Allow partial position exits
            "partial_exit_levels": [0.25, 0.5, 0.25],  # Exit portions at different levels
            "trailing_stop_activation": 1.5,   # Activate trailing stop at this multiple of risk
            "trailing_stop_distance": 0.5,     # Trailing stop distance as multiple of ATR
            
            # Position reversal
            "allow_position_reversal": True,   # Allow reversing positions
            "reversal_confirmation_threshold": 0.75,  # Confidence threshold for reversals
            
            # Risk management
            "max_drawdown_per_position": 0.05, # Maximum drawdown per position
            "max_total_drawdown": 0.1,         # Maximum total drawdown
            "correlation_limit": 0.7,          # Maximum correlation between positions
            
            # Emergency protocols
            "emergency_close_threshold": 0.15, # Emergency close at this drawdown level
            "market_circuit_breaker": 0.1,     # Pause trading if market moves this much in short time
        }
        
        # State tracking
        self.state = {
            "positions": {},                   # Current positions
            "position_history": {},            # Historical position data
            "total_pnl": 0.0,                  # Total realized P&L
            "drawdown": 0.0,                   # Current drawdown
            "max_drawdown": 0.0,               # Maximum historical drawdown
            "last_position_update": 0,         # Timestamp of last position update
            "trading_paused": False,           # Whether trading is paused
            "emergency_mode": False,           # Whether emergency mode is active
        }
    
    def get_current_positions(self) -> Dict[str, Any]:
        """
        Get current positions from the adapter.
        
        Returns:
            Dictionary of current positions
        """
        try:
            positions_result = self.adapter.get_positions()
            
            if "error" in positions_result:
                self.logger.error(f"Error getting positions: {positions_result['error']}")
                return {}
            
            # Convert list of positions to dict keyed by symbol
            positions_dict = {}
            for pos in positions_result.get("data", []):
                symbol = pos.get("symbol", "Unknown")
                positions_dict[symbol] = pos
            
            # Update state
            self.state["positions"] = positions_dict
            self.state["last_position_update"] = time.time()
            
            return positions_dict
            
        except Exception as e:
            self.logger.error(f"Error getting current positions: {e}")
            return {}
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information from the adapter.
        
        Returns:
            Dictionary with account information
        """
        try:
            account_info = self.adapter.get_account_info()
            
            if "error" in account_info:
                self.logger.error(f"Error getting account info: {account_info['error']}")
                return {}
            
            return account_info
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}
    
    def calculate_position_size(self, signal: Dict[str, Any], account_info: Dict[str, Any]) -> float:
        """
        Calculate optimal position size based on signal and account information.
        
        Args:
            signal: Signal dictionary with price and stop_loss
            account_info: Account information dictionary
            
        Returns:
            Position size in base currency
        """
        try:
            # Extract signal data
            price = signal.get("price", 0.0)
            stop_loss = signal.get("stop_loss", 0.0)
            confidence = signal.get("confidence", 0.5)
            
            if price <= 0 or stop_loss <= 0:
                self.logger.warning("Invalid price or stop loss for position sizing")
                return 0.0
            
            # Extract account data
            equity = account_info.get("equity", 0.0)
            
            if equity <= 0:
                self.logger.warning("Invalid account equity for position sizing")
                return 0.0
            
            # Calculate risk amount based on confidence
            risk_multiplier = min(1.5, max(0.5, confidence))
            risk_per_trade = self.config["default_risk_per_trade"] * risk_multiplier
            risk_amount = equity * risk_per_trade
            
            # Calculate risk per unit
            risk_per_unit = abs(price - stop_loss)
            
            if risk_per_unit <= 0:
                self.logger.warning("Invalid risk per unit for position sizing")
                return 0.0
            
            # Calculate position size
            position_size = risk_amount / risk_per_unit
            
            # Apply maximum position size limit
            max_size = equity * self.config["max_position_size"] / price
            position_size = min(position_size, max_size)
            
            # Adjust for existing positions
            current_positions = self.get_current_positions()
            position_count = len(current_positions)
            
            if position_count >= self.config["max_positions"]:
                self.logger.warning(f"Maximum positions ({self.config['max_positions']}) reached")
                return 0.0
            
            # Reduce position size as we approach max positions
            if position_count > 0:
                position_factor = 1.0 - (position_count / self.config["max_positions"])
                position_size *= position_factor
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def open_position(self, symbol: str, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Open a new position based on signal.
        
        Args:
            symbol: Trading symbol
            signal: Signal dictionary
            
        Returns:
            Dictionary with result of operation
        """
        try:
            # Check if trading is paused
            if self.state["trading_paused"]:
                return {"error": "Trading is currently paused"}
            
            # Check if emergency mode is active
            if self.state["emergency_mode"]:
                return {"error": "Emergency mode is active, no new positions allowed"}
            
            # Get account info
            account_info = self.get_account_info()
            if not account_info:
                return {"error": "Could not get account information"}
            
            # Get current positions
            current_positions = self.get_current_positions()
            
            # Check if we already have a position in this symbol
            if symbol in current_positions:
                existing_position = current_positions[symbol]
                existing_size = existing_position.get("size", 0.0)
                
                # Check if position direction matches signal
                is_long_signal = signal.get("signal") == "BUY"
                is_long_position = existing_size > 0
                
                if is_long_signal == is_long_position:
                    # Same direction - check if scaling in is allowed
                    if not self.config["allow_scaling_in"]:
                        return {"error": f"Already have a {signal['signal']} position in {symbol} and scaling in is disabled"}
                    
                    # Check scale-in history
                    position_history = self.state["position_history"].get(symbol, {})
                    scale_in_count = position_history.get("scale_in_count", 0)
                    
                    if scale_in_count >= self.config["max_scale_in_attempts"]:
                        return {"error": f"Maximum scale-in attempts ({self.config['max_scale_in_attempts']}) reached for {symbol}"}
                    
                    # Calculate additional position size (smaller than initial)
                    additional_size = self.calculate_position_size(signal, account_info) * 0.5
                    
                    # Place order
                    result = self.adapter.place_order(
                        symbol=symbol,
                        is_buy=is_long_signal,
                        size=additional_size,
                        price=signal.get("price", 0.0),
                        order_type="LIMIT"
                    )
                    
                    if "error" in result:
                        return {"error": f"Error scaling in to position: {result['error']}"}
                    
                    # Update position history
                    if symbol not in self.state["position_history"]:
                        self.state["position_history"][symbol] = {}
                    
                    self.state["position_history"][symbol]["scale_in_count"] = scale_in_count + 1
                    self.state["position_history"][symbol]["last_scale_in"] = time.time()
                    
                    return {
                        "action": "SCALE_IN",
                        "symbol": symbol,
                        "direction": "LONG" if is_long_signal else "SHORT",
                        "size": additional_size,
                        "order": result.get("data", {})
                    }
                    
                else:
                    # Opposite direction - check if position reversal is allowed
                    if not self.config["allow_position_reversal"]:
                        return {"error": f"Already have a {'LONG' if is_long_position else 'SHORT'} position in {symbol} and position reversal is disabled"}
                    
                    # Check confidence threshold for reversal
                    if signal.get("confidence", 0.0) < self.config["reversal_confirmation_threshold"]:
                        return {"error": f"Signal confidence ({signal.get('confidence', 0.0):.2f}) below threshold for position reversal"}
                    
                    # Close existing position
                    close_result = self.close_position(symbol)
                    
                    if "error" in close_result:
                        return {"error": f"Error closing existing position for reversal: {close_result['error']}"}
                    
                    # Calculate new position size
                    position_size = self.calculate_position_size(signal, account_info)
                    
                    # Place order for new position
                    result = self.adapter.place_order(
                        symbol=symbol,
                        is_buy=is_long_signal,
                        size=position_size,
                        price=signal.get("price", 0.0),
                        order_type="LIMIT"
                    )
                    
                    if "error" in result:
                        return {"error": f"Error opening reversed position: {result['error']}"}
                    
                    # Reset position history
                    if symbol not in self.state["position_history"]:
                        self.state["position_history"][symbol] = {}
                    
                    self.state["position_history"][symbol]["scale_in_count"] = 0
                    self.state["position_history"][symbol]["entry_time"] = time.time()
                    self.state["position_history"][symbol]["signal"] = signal
                    
                    return {
                        "action": "REVERSE",
                        "symbol": symbol,
                        "direction": "LONG" if is_long_signal else "SHORT",
                        "size": position_size,
                        "order": result.get("data", {})
                    }
            
            # No existing position - open new one
            # Calculate position size
            position_size = self.calculate_position_size(signal, account_info)
            
            if position_size <= 0:
                return {"error": "Invalid position size calculation"}
            
            # Determine direction
            is_buy = signal.get("signal") == "BUY"
            
            # Place order
            result = self.adapter.place_order(
                symbol=symbol,
                is_buy=is_buy,
                size=position_size,
                price=signal.get("price", 0.0),
                order_type="LIMIT"
            )
            
            if "error" in result:
                return {"error": f"Error opening position: {result['error']}"}
            
            # Initialize position history
            if symbol not in self.state["position_history"]:
                self.state["position_history"][symbol] = {}
            
            self.state["position_history"][symbol]["scale_in_count"] = 0
            self.state["position_history"][symbol]["entry_time"] = time.time()
            self.state["position_history"][symbol]["signal"] = signal
            
            return {
                "action": "OPEN",
                "symbol": symbol,
                "direction": "LONG" if is_buy else "SHORT",
                "size": position_size,
                "order": result.get("data", {})
            }
            
        except Exception as e:
            self.logger.error(f"Error opening position: {e}")
            return {"error": f"Error opening position: {e}"}
    
    def close_position(self, symbol: str, percentage: float = 100.0) -> Dict[str, Any]:
        """
        Close a position.
        
        Args:
            symbol: Trading symbol
            percentage: Percentage of position to close (0-100)
            
        Returns:
            Dictionary with result of operation
        """
        try:
            # Validate percentage
            if percentage <= 0 or percentage > 100:
                return {"error": "Percentage must be between 0 and 100"}
            
            # Get current positions
            current_positions = self.get_current_positions()
            
            # Check if we have a position in this symbol
            if symbol not in current_positions:
                return {"error": f"No position found for {symbol}"}
            
            # Close position
            result = self.adapter.close_position(symbol, percentage)
            
            if "error" in result:
                return {"error": f"Error closing position: {result['error']}"}
            
            # Update position history
            if symbol in self.state["position_history"] and percentage >= 100:
                # Calculate P&L
                position = current_positions[symbol]
                unrealized_pnl = position.get("unrealized_pnl", 0.0)
                
                # Update total P&L
                self.state["total_pnl"] += unrealized_pnl
                
                # Record closed position
                close_time = time.time()
                entry_time = self.state["position_history"][symbol].get("entry_time", close_time)
                duration = close_time - entry_time
                
                self.state["position_history"][symbol]["close_time"] = close_time
                self.state["position_history"][symbol]["duration"] = duration
                self.state["position_history"][symbol]["pnl"] = unrealized_pnl
                
                # Remove from active positions if fully closed
                if percentage >= 100:
                    if symbol in self.state["positions"]:
                        del self.state["positions"][symbol]
            
            return {
                "action": "CLOSE",
                "symbol": symbol,
                "percentage": percentage,
                "order": result.get("data", {})
            }
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return {"error": f"Error closing position: {e}"}
    
    def update_stop_loss(self, symbol: str, new_stop_price: float) -> Dict[str, Any]:
        """
        Update stop loss for a position.
        
        Args:
            symbol: Trading symbol
            new_stop_price: New stop loss price
            
        Returns:
            Dictionary with result of operation
        """
        try:
            # Get current positions
            current_positions = self.get_current_positions()
            
            # Check if we have a position in this symbol
            if symbol not in current_positions:
                return {"error": f"No position found for {symbol}"}
            
            position = current_positions[symbol]
            position_size = position.get("size", 0.0)
            
            if position_size == 0:
                return {"error": f"Position size for {symbol} is 0"}
            
            # Determine if long or short
            is_long = position_size > 0
            
            # Validate stop price
            current_price = position.get("mark_price", 0.0)
            
            if current_price <= 0:
                return {"error": f"Invalid current price for {symbol}"}
            
            if (is_long and new_stop_price >= current_price) or (not is_long and new_stop_price <= current_price):
                return {"error": f"Invalid stop price: must be below current price for long positions and above for short positions"}
            
            # Update position history
            if symbol in self.state["position_history"]:
                self.state["position_history"][symbol]["stop_loss"] = new_stop_price
                self.state["position_history"][symbol]["stop_updated_time"] = time.time()
            
            return {
                "action": "UPDATE_STOP",
                "symbol": symbol,
                "stop_price": new_stop_price
            }
            
        except Exception as e:
            self.logger.error(f"Error updating stop loss: {e}")
            return {"error": f"Error updating stop loss: {e}"}
    
    def apply_trailing_stop(self, symbol: str) -> Dict[str, Any]:
        """
        Apply trailing stop to a position.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with result of operation
        """
        try:
            # Get current positions
            current_positions = self.get_current_positions()
            
            # Check if we have a position in this symbol
            if symbol not in current_positions:
                return {"error": f"No position found for {symbol}"}
            
            position = current_positions[symbol]
            position_size = position.get("size", 0.0)
            
            if position_size == 0:
                return {"error": f"Position size for {symbol} is 0"}
            
            # Determine if long or short
            is_long = position_size > 0
            
            # Get current price and position data
            current_price = position.get("mark_price", 0.0)
            entry_price = position.get("entry_price", 0.0)
            
            if current_price <= 0 or entry_price <= 0:
                return {"error": f"Invalid price data for {symbol}"}
            
            # Get position history
            position_history = self.state["position_history"].get(symbol, {})
            original_stop = position_history.get("stop_loss", 0.0)
            
            if original_stop <= 0:
                # No stop loss set, use signal data
                signal = position_history.get("signal", {})
                original_stop = signal.get("stop_loss", 0.0)
                
                if original_stop <= 0:
                    return {"error": f"No stop loss data found for {symbol}"}
            
            # Calculate price movement
            price_movement = 0.0
            if is_long:
                price_movement = current_price - entry_price
            else:
                price_movement = entry_price - current_price
            
            # Calculate risk (distance from entry to original stop)
            risk = abs(entry_price - original_stop)
            
            if risk <= 0:
                return {"error": f"Invalid risk calculation for {symbol}"}
            
            # Check if trailing stop should be activated
            activation_threshold = risk * self.config["trailing_stop_activation"]
            
            if price_movement < activation_threshold:
                return {"message": f"Price movement not sufficient to activate trailing stop for {symbol}"}
            
            # Calculate new stop loss
            atr = position_history.get("signal", {}).get("atr", risk * 0.1)
            trailing_distance = atr * self.config["trailing_stop_distance"]
            
            new_stop = 0.0
            if is_long:
                new_stop = current_price - trailing_distance
                # Ensure new stop is higher than original
                new_stop = max(new_stop, original_stop)
            else:
                new_stop = current_price + trailing_distance
                # Ensure new stop is lower than original
                new_stop = min(new_stop, original_stop)
            
            # Update stop loss
            return self.update_stop_loss(symbol, new_stop)
            
        except Exception as e:
            self.logger.error(f"Error applying trailing stop: {e}")
            return {"error": f"Error applying trailing stop: {e}"}
    
    def check_stop_loss(self, symbol: str) -> Dict[str, Any]:
        """
        Check if stop loss has been hit for a position.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with result of operation
        """
        try:
            # Get current positions
            current_positions = self.get_current_positions()
            
            # Check if we have a position in this symbol
            if symbol not in current_positions:
                return {"triggered": False, "message": f"No position found for {symbol}"}
            
            position = current_positions[symbol]
            position_size = position.get("size", 0.0)
            
            if position_size == 0:
                return {"triggered": False, "message": f"Position size for {symbol} is 0"}
            
            # Determine if long or short
            is_long = position_size > 0
            
            # Get current price
            current_price = position.get("mark_price", 0.0)
            
            if current_price <= 0:
                return {"triggered": False, "message": f"Invalid current price for {symbol}"}
            
            # Get stop loss price
            position_history = self.state["position_history"].get(symbol, {})
            stop_loss = position_history.get("stop_loss", 0.0)
            
            if stop_loss <= 0:
                # No stop loss set, use signal data
                signal = position_history.get("signal", {})
                stop_loss = signal.get("stop_loss", 0.0)
                
                if stop_loss <= 0:
                    return {"triggered": False, "message": f"No stop loss data found for {symbol}"}
            
            # Check if stop loss has been hit
            stop_triggered = False
            if is_long and current_price <= stop_loss:
                stop_triggered = True
            elif not is_long and current_price >= stop_loss:
                stop_triggered = True
            
            if stop_triggered:
                # Close position
                close_result = self.close_position(symbol)
                
                if "error" in close_result:
                    return {"triggered": True, "message": f"Stop loss triggered but error closing position: {close_result['error']}"}
                
                return {"triggered": True, "message": f"Stop loss triggered for {symbol}", "close_result": close_result}
            
            return {"triggered": False, "message": f"Stop loss not triggered for {symbol}"}
            
        except Exception as e:
            self.logger.error(f"Error checking stop loss: {e}")
            return {"triggered": False, "error": f"Error checking stop loss: {e}"}
    
    def check_take_profit(self, symbol: str) -> Dict[str, Any]:
        """
        Check if take profit has been hit for a position.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with result of operation
        """
        try:
            # Get current positions
            current_positions = self.get_current_positions()
            
            # Check if we have a position in this symbol
            if symbol not in current_positions:
                return {"triggered": False, "message": f"No position found for {symbol}"}
            
            position = current_positions[symbol]
            position_size = position.get("size", 0.0)
            
            if position_size == 0:
                return {"triggered": False, "message": f"Position size for {symbol} is 0"}
            
            # Determine if long or short
            is_long = position_size > 0
            
            # Get current price
            current_price = position.get("mark_price", 0.0)
            
            if current_price <= 0:
                return {"triggered": False, "message": f"Invalid current price for {symbol}"}
            
            # Get take profit price
            position_history = self.state["position_history"].get(symbol, {})
            signal = position_history.get("signal", {})
            take_profit = signal.get("take_profit", 0.0)
            
            if take_profit <= 0:
                return {"triggered": False, "message": f"No take profit data found for {symbol}"}
            
            # Check if take profit has been hit
            tp_triggered = False
            if is_long and current_price >= take_profit:
                tp_triggered = True
            elif not is_long and current_price <= take_profit:
                tp_triggered = True
            
            if tp_triggered:
                # Check if partial exits are enabled
                if self.config["allow_partial_exits"]:
                    # Get partial exit history
                    partial_exits = position_history.get("partial_exits", 0)
                    
                    # If we haven't done all partial exits yet
                    if partial_exits < len(self.config["partial_exit_levels"]):
                        # Get exit percentage for this level
                        exit_percentage = self.config["partial_exit_levels"][partial_exits] * 100
                        
                        # Close partial position
                        close_result = self.close_position(symbol, exit_percentage)
                        
                        if "error" in close_result:
                            return {"triggered": True, "message": f"Take profit triggered but error closing partial position: {close_result['error']}"}
                        
                        # Update partial exit history
                        self.state["position_history"][symbol]["partial_exits"] = partial_exits + 1
                        
                        # Move stop loss to entry (break even) after first partial exit
                        if partial_exits == 0:
                            entry_price = position.get("entry_price", 0.0)
                            if entry_price > 0:
                                self.update_stop_loss(symbol, entry_price)
                        
                        return {
                            "triggered": True, 
                            "message": f"Partial take profit ({exit_percentage}%) triggered for {symbol}", 
                            "close_result": close_result
                        }
                    
                # If partial exits are disabled or all partial exits done, close full position
                close_result = self.close_position(symbol)
                
                if "error" in close_result:
                    return {"triggered": True, "message": f"Take profit triggered but error closing position: {close_result['error']}"}
                
                return {"triggered": True, "message": f"Take profit triggered for {symbol}", "close_result": close_result}
            
            return {"triggered": False, "message": f"Take profit not triggered for {symbol}"}
            
        except Exception as e:
            self.logger.error(f"Error checking take profit: {e}")
            return {"triggered": False, "error": f"Error checking take profit: {e}"}
    
    def manage_positions(self) -> Dict[str, Any]:
        """
        Manage all open positions.
        
        Returns:
            Dictionary with results of position management
        """
        try:
            # Get current positions
            current_positions = self.get_current_positions()
            
            if not current_positions:
                return {"message": "No open positions to manage"}
            
            results = {}
            
            # Check each position
            for symbol, position in current_positions.items():
                position_result = {}
                
                # Check stop loss
                stop_result = self.check_stop_loss(symbol)
                position_result["stop_check"] = stop_result
                
                # If stop loss not triggered, check take profit
                if not stop_result.get("triggered", False):
                    tp_result = self.check_take_profit(symbol)
                    position_result["tp_check"] = tp_result
                    
                    # If take profit not triggered, apply trailing stop
                    if not tp_result.get("triggered", False):
                        trail_result = self.apply_trailing_stop(symbol)
                        position_result["trailing_stop"] = trail_result
                
                results[symbol] = position_result
            
            return {"results": results}
            
        except Exception as e:
            self.logger.error(f"Error managing positions: {e}")
            return {"error": f"Error managing positions: {e}"}
    
    def emergency_close_all(self) -> Dict[str, Any]:
        """
        Emergency close all positions.
        
        Returns:
            Dictionary with results of emergency close
        """
        try:
            # Get current positions
            current_positions = self.get_current_positions()
            
            if not current_positions:
                return {"message": "No open positions to close"}
            
            results = {}
            
            # Close each position
            for symbol in current_positions:
                close_result = self.close_position(symbol)
                results[symbol] = close_result
            
            # Set emergency mode
            self.state["emergency_mode"] = True
            self.state["trading_paused"] = True
            
            return {"results": results, "message": "Emergency close completed"}
            
        except Exception as e:
            self.logger.error(f"Error in emergency close: {e}")
            return {"error": f"Error in emergency close: {e}"}
    
    def check_drawdown(self) -> Dict[str, Any]:
        """
        Check drawdown and take action if necessary.
        
        Returns:
            Dictionary with drawdown check results
        """
        try:
            # Get account info
            account_info = self.get_account_info()
            
            if not account_info:
                return {"error": "Could not get account information"}
            
            # Calculate drawdown
            equity = account_info.get("equity", 0.0)
            peak_equity = account_info.get("peak_equity", equity)
            
            if peak_equity <= 0:
                return {"error": "Invalid peak equity value"}
            
            drawdown = (peak_equity - equity) / peak_equity
            
            # Update state
            self.state["drawdown"] = drawdown
            self.state["max_drawdown"] = max(self.state["max_drawdown"], drawdown)
            
            # Check emergency threshold
            if drawdown >= self.config["emergency_close_threshold"]:
                # Trigger emergency close
                emergency_result = self.emergency_close_all()
                
                return {
                    "drawdown": drawdown,
                    "max_drawdown": self.state["max_drawdown"],
                    "emergency_triggered": True,
                    "emergency_result": emergency_result
                }
            
            return {
                "drawdown": drawdown,
                "max_drawdown": self.state["max_drawdown"],
                "emergency_triggered": False
            }
            
        except Exception as e:
            self.logger.error(f"Error checking drawdown: {e}")
            return {"error": f"Error checking drawdown: {e}"}
    
    def get_position_stats(self) -> Dict[str, Any]:
        """
        Get position statistics.
        
        Returns:
            Dictionary with position statistics
        """
        try:
            # Get current positions
            current_positions = self.get_current_positions()
            
            # Calculate statistics
            total_positions = len(current_positions)
            long_positions = sum(1 for pos in current_positions.values() if pos.get("size", 0.0) > 0)
            short_positions = sum(1 for pos in current_positions.values() if pos.get("size", 0.0) < 0)
            
            # Calculate unrealized P&L
            total_unrealized_pnl = sum(pos.get("unrealized_pnl", 0.0) for pos in current_positions.values())
            
            # Calculate average holding time
            current_time = time.time()
            holding_times = []
            
            for symbol in current_positions:
                if symbol in self.state["position_history"]:
                    entry_time = self.state["position_history"][symbol].get("entry_time", current_time)
                    holding_time = current_time - entry_time
                    holding_times.append(holding_time)
            
            avg_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0
            
            return {
                "total_positions": total_positions,
                "long_positions": long_positions,
                "short_positions": short_positions,
                "total_unrealized_pnl": total_unrealized_pnl,
                "total_realized_pnl": self.state["total_pnl"],
                "avg_holding_time": avg_holding_time,
                "drawdown": self.state["drawdown"],
                "max_drawdown": self.state["max_drawdown"],
                "trading_paused": self.state["trading_paused"],
                "emergency_mode": self.state["emergency_mode"]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting position stats: {e}")
            return {"error": f"Error getting position stats: {e}"}
    
    def save_state(self, file_path: str) -> Dict[str, Any]:
        """
        Save position manager state to file.
        
        Args:
            file_path: Path to save state file
            
        Returns:
            Dictionary with result of operation
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.state, f, indent=2)
            
            return {"message": f"State saved to {file_path}"}
            
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
            return {"error": f"Error saving state: {e}"}
    
    def load_state(self, file_path: str) -> Dict[str, Any]:
        """
        Load position manager state from file.
        
        Args:
            file_path: Path to state file
            
        Returns:
            Dictionary with result of operation
        """
        try:
            with open(file_path, 'r') as f:
                self.state = json.load(f)
            
            return {"message": f"State loaded from {file_path}"}
            
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            return {"error": f"Error loading state: {e}"}
