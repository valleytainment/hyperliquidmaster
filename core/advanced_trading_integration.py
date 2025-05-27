"""
Integration Module for Advanced Trading Features
-----------------------------------------------
Integrates enhanced risk management and advanced order types
into the main HyperliquidMaster trading workflow.
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple, List, Union

# Import enhanced modules
from core.enhanced_risk_manager import EnhancedRiskManager
from core.advanced_order_manager import AdvancedOrderManager
from core.trading_mode import TradingModeManager, TradingMode

class AdvancedTradingIntegration:
    """
    Integration layer for advanced trading features.
    Connects enhanced risk management and advanced order types
    to the main trading workflow.
    """
    
    def __init__(self, adapter=None, mode_manager=None, logger=None):
        """
        Initialize the advanced trading integration.
        
        Args:
            adapter: HyperliquidAdapter instance
            mode_manager: TradingModeManager instance
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger("AdvancedTradingIntegration")
        self.adapter = adapter
        self.mode_manager = mode_manager
        
        # Initialize enhanced modules
        self.risk_manager = EnhancedRiskManager(self.logger)
        self.order_manager = AdvancedOrderManager(self.adapter, self.logger)
        
        # Default settings
        self.use_volatility_sizing = True
        self.enforce_risk_reward_ratio = True
        self.enable_drawdown_protection = True
        self.enable_psychological_safeguards = True
        
        self.logger.info("Advanced Trading Integration initialized")
    
    def set_adapter(self, adapter) -> None:
        """
        Set the HyperliquidAdapter instance.
        
        Args:
            adapter: HyperliquidAdapter instance
        """
        self.adapter = adapter
        self.order_manager.set_adapter(adapter)
    
    def set_mode_manager(self, mode_manager) -> None:
        """
        Set the TradingModeManager instance.
        
        Args:
            mode_manager: TradingModeManager instance
        """
        self.mode_manager = mode_manager
    
    def update_account_metrics(self) -> None:
        """Update account metrics for risk management."""
        if not self.adapter:
            self.logger.error("No adapter set for account metrics update")
            return
        
        # Get account info
        account_info = self.adapter.get_account_info()
        
        if "error" in account_info:
            self.logger.error(f"Error getting account info: {account_info['error']}")
            return
        
        # Update risk manager with current equity
        equity = account_info.get("equity", 0.0)
        self.risk_manager.set_account_equity(equity)
        
        # Check if trading should be allowed
        allowed, reason = self.risk_manager.should_allow_trade()
        if not allowed and self.enable_drawdown_protection:
            self.logger.warning(f"Trading restricted: {reason}")
            
            # If using mode manager, switch to monitor-only mode
            if self.mode_manager:
                current_mode = self.mode_manager.get_current_mode()
                if current_mode != TradingMode.MONITOR_ONLY:
                    self.logger.warning(f"Switching from {current_mode} to MONITOR_ONLY mode due to risk limits")
                    self.mode_manager.set_mode(TradingMode.MONITOR_ONLY)
    
    def calculate_position_size(self, 
                               symbol: str, 
                               entry_price: float, 
                               stop_loss_price: float) -> float:
        """
        Calculate appropriate position size based on risk parameters.
        
        Args:
            symbol: Trading pair symbol
            entry_price: Planned entry price
            stop_loss_price: Planned stop loss price
            
        Returns:
            Recommended position size
        """
        if not self.adapter:
            self.logger.error("No adapter set for position size calculation")
            return 0.0
        
        # Get account info
        account_info = self.adapter.get_account_info()
        
        if "error" in account_info:
            self.logger.error(f"Error getting account info: {account_info['error']}")
            return 0.0
        
        equity = account_info.get("equity", 0.0)
        
        # Get volatility factor if enabled
        volatility_factor = None
        if self.use_volatility_sizing:
            # Get market data for volatility calculation
            market_data = self.adapter.get_market_data(symbol)
            
            if "error" not in market_data:
                # Use price change as simple volatility measure
                # In a real implementation, this would use ATR or other volatility metrics
                volatility_factor = abs(market_data.get("price_change_24h", 0.0)) / 100.0
        
        # Get risk percentage based on trading mode
        risk_percentage = 1.0  # Default 1% rule
        
        if self.mode_manager:
            current_mode = self.mode_manager.get_current_mode()
            
            # Adjust risk based on mode
            if current_mode == TradingMode.CONSERVATIVE:
                risk_percentage = 0.5  # 0.5% for conservative mode
            elif current_mode == TradingMode.AGGRESSIVE:
                risk_percentage = 2.0  # 2% for aggressive mode
            elif current_mode == TradingMode.MONITOR_ONLY:
                risk_percentage = 0.0  # 0% for monitor-only mode
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            account_equity=equity,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            volatility_factor=volatility_factor,
            risk_percentage=risk_percentage
        )
        
        return position_size
    
    def validate_trade_setup(self, 
                            entry_price: float, 
                            stop_loss_price: float, 
                            take_profit_price: float) -> bool:
        """
        Validate if a trade setup meets risk management criteria.
        
        Args:
            entry_price: Planned entry price
            stop_loss_price: Planned stop loss price
            take_profit_price: Planned take profit price
            
        Returns:
            True if valid, False otherwise
        """
        if not self.enforce_risk_reward_ratio:
            return True
        
        # Get minimum RRR based on trading mode
        min_rrr = 2.0  # Default 1:2 risk-reward ratio
        
        if self.mode_manager:
            current_mode = self.mode_manager.get_current_mode()
            
            # Adjust RRR based on mode
            if current_mode == TradingMode.CONSERVATIVE:
                min_rrr = 3.0  # 1:3 for conservative mode
            elif current_mode == TradingMode.AGGRESSIVE:
                min_rrr = 1.5  # 1:1.5 for aggressive mode
        
        # Validate risk-reward ratio
        is_valid, ratio = self.risk_manager.validate_risk_reward_ratio(
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            min_ratio=min_rrr
        )
        
        return is_valid
    
    def execute_entry_with_advanced_orders(self, 
                                         symbol: str, 
                                         is_long: bool, 
                                         size: float, 
                                         entry_price: float, 
                                         stop_loss_price: float, 
                                         take_profit_price: float,
                                         use_scale_entry: bool = False,
                                         scale_range_percent: float = 1.0,
                                         scale_orders: int = 3) -> Dict[str, Any]:
        """
        Execute an entry with advanced order types.
        
        Args:
            symbol: Trading pair symbol
            is_long: True for long position, False for short
            size: Position size
            entry_price: Target entry price
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            use_scale_entry: Whether to use scale orders for entry
            scale_range_percent: Percentage range for scale orders
            scale_orders: Number of scale orders
            
        Returns:
            Dict containing entry results
        """
        if not self.adapter:
            self.logger.error("No adapter set for order execution")
            return {"error": "No adapter set"}
        
        # Check if trading is allowed
        allowed, reason = self.risk_manager.should_allow_trade()
        if not allowed and self.enable_drawdown_protection:
            self.logger.warning(f"Trade rejected: {reason}")
            return {"error": reason}
        
        # Validate risk-reward ratio
        if self.enforce_risk_reward_ratio:
            is_valid = self.validate_trade_setup(entry_price, stop_loss_price, take_profit_price)
            if not is_valid:
                self.logger.warning("Trade rejected: Does not meet risk-reward requirements")
                return {"error": "Does not meet risk-reward requirements"}
        
        # Execute entry
        entry_orders = []
        
        if use_scale_entry:
            # Calculate scale range
            range_amount = entry_price * (scale_range_percent / 100.0)
            
            if is_long:
                # For long positions, scale down from entry price
                price_range = (entry_price - range_amount, entry_price)
            else:
                # For short positions, scale up from entry price
                price_range = (entry_price, entry_price + range_amount)
            
            # Execute scale orders
            entry_results = self.order_manager.execute_scale_orders(
                symbol=symbol,
                is_buy=is_long,
                total_size=size,
                price_range=price_range,
                num_orders=scale_orders
            )
            
            entry_orders.extend(entry_results)
        else:
            # Execute single entry order
            entry_result = self.adapter.place_order(
                symbol=symbol,
                is_buy=is_long,
                size=size,
                price=entry_price,
                order_type="LIMIT"
            )
            
            entry_orders.append(entry_result)
        
        # Place stop loss order
        stop_result = self.order_manager.place_stop_limit_order(
            symbol=symbol,
            is_buy=not is_long,  # Opposite direction for stop loss
            size=size,
            stop_price=stop_loss_price,
            limit_price=stop_loss_price * (0.99 if is_long else 1.01),  # Slight adjustment for execution
            reduce_only=True
        )
        
        # Place take profit order
        take_profit_result = self.adapter.place_order(
            symbol=symbol,
            is_buy=not is_long,  # Opposite direction for take profit
            size=size,
            price=take_profit_price,
            order_type="LIMIT"
        )
        
        # Record trade setup
        trade_setup = {
            "symbol": symbol,
            "is_long": is_long,
            "size": size,
            "entry_price": entry_price,
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "entry_orders": entry_orders,
            "stop_loss_order": stop_result,
            "take_profit_order": take_profit_result,
            "time": int(time.time() * 1000)
        }
        
        self.logger.info(f"Advanced entry executed for {symbol}: {'LONG' if is_long else 'SHORT'}, size: {size}")
        return {"success": True, "trade_setup": trade_setup}
    
    def execute_twap_entry(self, 
                          symbol: str, 
                          is_long: bool, 
                          size: float, 
                          duration_minutes: int,
                          stop_loss_price: float, 
                          take_profit_price: float) -> Dict[str, Any]:
        """
        Execute a TWAP entry.
        
        Args:
            symbol: Trading pair symbol
            is_long: True for long position, False for short
            size: Position size
            duration_minutes: Duration for TWAP execution in minutes
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            
        Returns:
            Dict containing entry results
        """
        if not self.adapter:
            self.logger.error("No adapter set for order execution")
            return {"error": "No adapter set"}
        
        # Check if trading is allowed
        allowed, reason = self.risk_manager.should_allow_trade()
        if not allowed and self.enable_drawdown_protection:
            self.logger.warning(f"Trade rejected: {reason}")
            return {"error": reason}
        
        # Get current market price for validation
        market_data = self.adapter.get_market_data(symbol)
        if "error" in market_data:
            self.logger.error(f"Error getting market data: {market_data['error']}")
            return {"error": f"Error getting market data: {market_data['error']}"}
        
        current_price = market_data.get("price", 0.0)
        
        # Validate risk-reward ratio using current price as entry
        if self.enforce_risk_reward_ratio:
            is_valid = self.validate_trade_setup(current_price, stop_loss_price, take_profit_price)
            if not is_valid:
                self.logger.warning("Trade rejected: Does not meet risk-reward requirements")
                return {"error": "Does not meet risk-reward requirements"}
        
        # Execute TWAP entry
        twap_order_id = self.order_manager.execute_twap_order(
            symbol=symbol,
            is_buy=is_long,
            total_size=size,
            duration_seconds=duration_minutes * 60,
            max_slippage=0.03,  # 3% max slippage
            interval_seconds=30  # 30 second intervals
        )
        
        # Place stop loss order
        stop_result = self.order_manager.place_stop_limit_order(
            symbol=symbol,
            is_buy=not is_long,  # Opposite direction for stop loss
            size=size,
            stop_price=stop_loss_price,
            limit_price=stop_loss_price * (0.99 if is_long else 1.01),  # Slight adjustment for execution
            reduce_only=True
        )
        
        # Place take profit order
        take_profit_result = self.adapter.place_order(
            symbol=symbol,
            is_buy=not is_long,  # Opposite direction for take profit
            size=size,
            price=take_profit_price,
            order_type="LIMIT"
        )
        
        # Record trade setup
        trade_setup = {
            "symbol": symbol,
            "is_long": is_long,
            "size": size,
            "entry_type": "TWAP",
            "duration_minutes": duration_minutes,
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "twap_order_id": twap_order_id,
            "stop_loss_order": stop_result,
            "take_profit_order": take_profit_result,
            "time": int(time.time() * 1000)
        }
        
        self.logger.info(f"TWAP entry executed for {symbol}: {'LONG' if is_long else 'SHORT'}, size: {size}, duration: {duration_minutes}m")
        return {"success": True, "trade_setup": trade_setup}
    
    def record_trade_result(self, 
                           symbol: str, 
                           is_win: bool, 
                           profit: float, 
                           trade_details: Dict[str, Any]) -> None:
        """
        Record a trade result for performance tracking.
        
        Args:
            symbol: Trading pair symbol
            is_win: Whether the trade was profitable
            profit: Profit amount (positive or negative)
            trade_details: Additional trade details
        """
        # Create trade result record
        trade_result = {
            "symbol": symbol,
            "is_win": is_win,
            "profit": profit,
            "details": trade_details,
            "timestamp": int(time.time())
        }
        
        # Record in risk manager
        self.risk_manager.record_trade_result(trade_result)
        
        # Log trade result
        result_str = "WIN" if is_win else "LOSS"
        profit_str = f"+{profit:.2f}" if profit >= 0 else f"{profit:.2f}"
        self.logger.info(f"Trade result recorded: {symbol} {result_str} {profit_str}")
        
        # Get updated performance metrics
        metrics = self.risk_manager.get_performance_metrics()
        
        # Log key metrics
        self.logger.info(f"Performance metrics - Win rate: {metrics['win_rate']:.1f}%, Profit factor: {metrics['profit_factor']:.2f}")
        
        # Check for streak warnings if psychological safeguards are enabled
        if self.enable_psychological_safeguards:
            streak = metrics.get("current_streak", {})
            streak_type = streak.get("type")
            streak_count = streak.get("count", 0)
            
            if streak_type == "win" and streak_count >= 5:
                self.logger.warning(f"WINNING STREAK ALERT: {streak_count} consecutive wins. "
                                   f"Be cautious of overconfidence and maintain strict risk management.")
            
            if streak_type == "loss" and streak_count >= 3:
                self.logger.warning(f"LOSING STREAK ALERT: {streak_count} consecutive losses. "
                                   f"Consider reducing position size or taking a break.")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report.
        
        Returns:
            Dict containing performance metrics
        """
        # Get basic metrics from risk manager
        metrics = self.risk_manager.get_performance_metrics()
        
        # Add additional information
        if self.mode_manager:
            metrics["current_mode"] = self.mode_manager.get_current_mode()
        
        # Add risk settings
        metrics["risk_settings"] = {
            "max_risk_percentage": self.risk_manager.max_risk_percentage,
            "min_risk_reward_ratio": self.risk_manager.min_risk_reward_ratio,
            "use_volatility_adjustment": self.use_volatility_sizing,
            "max_daily_drawdown": self.risk_manager.max_daily_drawdown,
            "max_total_drawdown": self.risk_manager.max_total_drawdown
        }
        
        return metrics
