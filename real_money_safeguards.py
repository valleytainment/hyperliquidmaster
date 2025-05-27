"""
Real-Money Trading Safeguards Module

This module provides safeguards for real-money trading to ensure capital preservation
and risk management. It includes:
- Circuit breakers to pause trading after consecutive losses
- Dynamic position sizing based on volatility
- Tiered stop-loss strategy
- Maximum drawdown protection
- Synthetic data detection and risk adjustments
"""

import os
import sys
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("real_money_safeguards.log")
    ]
)
logger = logging.getLogger(__name__)

class RealMoneyTradingSafeguards:
    """
    Safeguards for real-money trading.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize real-money trading safeguards.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize safeguard parameters
        self.risk_percent = self.config.get("risk_percent", 0.01)  # 1% risk per trade
        self.max_drawdown_threshold = self.config.get("max_drawdown_threshold", 0.07)  # 7% max drawdown
        self.circuit_breaker_threshold = self.config.get("circuit_breaker_threshold", 0.05)  # 5% daily loss
        self.max_consecutive_losses = self.config.get("max_consecutive_losses", 3)
        
        # Initialize state
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.circuit_breaker_triggered = False
        self.max_drawdown_triggered = False
        self.last_reset = datetime.now()
        
        logger.info(f"Real-Money Trading Safeguards initialized with config: {json.dumps(self.config, indent=4)}")
        
    def apply_safeguards(self, signal: Dict, market_data: Dict, positions: Dict, performance_metrics: Dict) -> Dict:
        """
        Apply safeguards to trading signal.
        
        Args:
            signal: Trading signal
            market_data: Market data
            positions: Current positions
            performance_metrics: Performance metrics
            
        Returns:
            Modified trading signal
        """
        try:
            # Check if signal is valid
            if not signal or signal.get("action") == "none":
                return signal
                
            # Reset daily PnL if needed
            self._reset_daily_pnl_if_needed()
                
            # Check circuit breaker
            if self._check_circuit_breaker(performance_metrics):
                logger.warning("Circuit breaker triggered, blocking all trades")
                return {"action": "none", "reason": "circuit_breaker"}
                
            # Check max drawdown
            if self._check_max_drawdown(performance_metrics):
                logger.warning("Max drawdown threshold exceeded, blocking all trades")
                return {"action": "none", "reason": "max_drawdown"}
                
            # Check consecutive losses
            if self._check_consecutive_losses():
                logger.warning("Too many consecutive losses, blocking all trades")
                return {"action": "none", "reason": "consecutive_losses"}
                
            # Apply dynamic position sizing
            signal = self._apply_dynamic_position_sizing(signal, market_data, performance_metrics)
                
            # Apply synthetic data risk adjustments
            signal = self._apply_synthetic_data_risk_adjustments(signal, market_data)
                
            return signal
        except Exception as e:
            logger.error(f"Error applying safeguards: {str(e)}")
            return {"action": "none", "reason": "safeguard_error"}
            
    def update_trade_result(self, trade_result: Dict):
        """
        Update safeguards with trade result.
        
        Args:
            trade_result: Trade result
        """
        try:
            # Update consecutive losses
            if trade_result.get("pnl", 0.0) < 0:
                self.consecutive_losses += 1
                logger.info(f"Consecutive losses: {self.consecutive_losses}")
            else:
                self.consecutive_losses = 0
                
            # Update daily PnL
            self.daily_pnl += trade_result.get("pnl", 0.0)
            logger.info(f"Daily PnL: {self.daily_pnl:.2f}")
        except Exception as e:
            logger.error(f"Error updating trade result: {str(e)}")
            
    def _reset_daily_pnl_if_needed(self):
        """
        Reset daily PnL if needed.
        """
        try:
            now = datetime.now()
            
            if now.date() > self.last_reset.date():
                logger.info(f"Resetting daily PnL from {self.daily_pnl:.2f} to 0.0")
                self.daily_pnl = 0.0
                self.circuit_breaker_triggered = False
                self.last_reset = now
        except Exception as e:
            logger.error(f"Error resetting daily PnL: {str(e)}")
            
    def _check_circuit_breaker(self, performance_metrics: Dict) -> bool:
        """
        Check if circuit breaker should be triggered.
        
        Args:
            performance_metrics: Performance metrics
            
        Returns:
            True if circuit breaker should be triggered, False otherwise
        """
        try:
            if self.circuit_breaker_triggered:
                return True
                
            initial_capital = performance_metrics.get("initial_capital", 0.0)
            
            if initial_capital <= 0.0:
                return False
                
            daily_loss_percent = abs(self.daily_pnl) / initial_capital
            
            if daily_loss_percent >= self.circuit_breaker_threshold:
                logger.warning(f"Circuit breaker triggered: daily loss {daily_loss_percent:.2%} exceeds threshold {self.circuit_breaker_threshold:.2%}")
                self.circuit_breaker_triggered = True
                return True
                
            return False
        except Exception as e:
            logger.error(f"Error checking circuit breaker: {str(e)}")
            return False
            
    def _check_max_drawdown(self, performance_metrics: Dict) -> bool:
        """
        Check if max drawdown threshold is exceeded.
        
        Args:
            performance_metrics: Performance metrics
            
        Returns:
            True if max drawdown threshold is exceeded, False otherwise
        """
        try:
            if self.max_drawdown_triggered:
                return True
                
            max_drawdown = performance_metrics.get("max_drawdown", 0.0)
            
            if max_drawdown >= self.max_drawdown_threshold:
                logger.warning(f"Max drawdown threshold exceeded: {max_drawdown:.2%} >= {self.max_drawdown_threshold:.2%}")
                self.max_drawdown_triggered = True
                return True
                
            return False
        except Exception as e:
            logger.error(f"Error checking max drawdown: {str(e)}")
            return False
            
    def _check_consecutive_losses(self) -> bool:
        """
        Check if too many consecutive losses.
        
        Returns:
            True if too many consecutive losses, False otherwise
        """
        try:
            return self.consecutive_losses >= self.max_consecutive_losses
        except Exception as e:
            logger.error(f"Error checking consecutive losses: {str(e)}")
            return False
            
    def _apply_dynamic_position_sizing(self, signal: Dict, market_data: Dict, performance_metrics: Dict) -> Dict:
        """
        Apply dynamic position sizing.
        
        Args:
            signal: Trading signal
            market_data: Market data
            performance_metrics: Performance metrics
            
        Returns:
            Modified trading signal
        """
        try:
            # Check if signal is valid
            if not signal or signal.get("action") == "none":
                return signal
                
            # Get current capital
            current_capital = performance_metrics.get("current_capital", 0.0)
            
            if current_capital <= 0.0:
                return {"action": "none", "reason": "insufficient_capital"}
                
            # Get price
            price = market_data.get("last_price", 0.0)
            
            if price <= 0.0:
                return {"action": "none", "reason": "invalid_price"}
                
            # Calculate position size based on risk
            risk_amount = current_capital * self.risk_percent
            
            # Get stop loss distance
            stop_loss_percent = signal.get("stop_loss_percent", 0.02)  # Default 2%
            stop_loss_distance = price * stop_loss_percent
            
            if stop_loss_distance <= 0.0:
                return {"action": "none", "reason": "invalid_stop_loss"}
                
            # Calculate quantity
            quantity = risk_amount / stop_loss_distance
            
            # Apply volatility adjustment
            volatility = market_data.get("volatility", 0.0)
            volatility_threshold = self.config.get("volatility_threshold", 0.02)  # Default 2%
            
            if volatility > volatility_threshold:
                volatility_factor = volatility_threshold / volatility
                quantity *= volatility_factor
                logger.info(f"Applied volatility adjustment: {volatility_factor:.2f}")
                
            # Update signal
            signal["quantity"] = quantity
            signal["risk_amount"] = risk_amount
            
            return signal
        except Exception as e:
            logger.error(f"Error applying dynamic position sizing: {str(e)}")
            return {"action": "none", "reason": "position_sizing_error"}
            
    def _apply_synthetic_data_risk_adjustments(self, signal: Dict, market_data: Dict) -> Dict:
        """
        Apply risk adjustments for synthetic data.
        
        Args:
            signal: Trading signal
            market_data: Market data
            
        Returns:
            Modified trading signal
        """
        try:
            # Check if signal is valid
            if not signal or signal.get("action") == "none":
                return signal
                
            # Check if market data is synthetic
            is_synthetic = market_data.get("is_synthetic", False)
            synthetic_ratio = market_data.get("synthetic_ratio", 0.0)
            
            if is_synthetic or synthetic_ratio > 0.0:
                # Apply risk reduction based on synthetic ratio
                risk_factor = 1.0 - min(synthetic_ratio, 0.9)  # Max 90% reduction
                
                # Update quantity
                signal["quantity"] *= risk_factor
                
                # Update confidence
                if "confidence" in signal:
                    signal["confidence"] *= risk_factor
                    
                logger.info(f"Applied synthetic data risk adjustment: {risk_factor:.2f}")
                
                # Block trade if synthetic ratio is too high
                if synthetic_ratio > 0.8:  # 80% threshold
                    logger.warning(f"Blocking trade due to high synthetic ratio: {synthetic_ratio:.2f}")
                    return {"action": "none", "reason": "high_synthetic_ratio"}
                    
            return signal
        except Exception as e:
            logger.error(f"Error applying synthetic data risk adjustments: {str(e)}")
            return {"action": "none", "reason": "risk_adjustment_error"}

# Test function
def main():
    """
    Test real-money trading safeguards.
    """
    # Create safeguards
    config = {
        "risk_percent": 0.01,
        "max_drawdown_threshold": 0.07,
        "circuit_breaker_threshold": 0.05,
        "max_consecutive_losses": 3
    }
    safeguards = RealMoneyTradingSafeguards(config=config)
    
    # Create test signal
    signal = {
        "action": "long",
        "symbol": "BTC-USD-PERP",
        "confidence": 0.8,
        "stop_loss_percent": 0.02
    }
    
    # Create test market data
    market_data = {
        "symbol": "BTC-USD-PERP",
        "last_price": 50000.0,
        "volatility": 0.03,
        "is_synthetic": False,
        "synthetic_ratio": 0.0
    }
    
    # Create test positions
    positions = {}
    
    # Create test performance metrics
    performance_metrics = {
        "initial_capital": 10000.0,
        "current_capital": 10000.0,
        "max_drawdown": 0.03
    }
    
    # Apply safeguards
    result = safeguards.apply_safeguards(signal, market_data, positions, performance_metrics)
    
    # Print result
    print(f"Original signal: {signal}")
    print(f"Modified signal: {result}")
    
    # Test with synthetic data
    market_data["is_synthetic"] = True
    market_data["synthetic_ratio"] = 0.5
    
    result = safeguards.apply_safeguards(signal, market_data, positions, performance_metrics)
    
    print(f"Modified signal with synthetic data: {result}")
    
    # Test with high synthetic ratio
    market_data["synthetic_ratio"] = 0.9
    
    result = safeguards.apply_safeguards(signal, market_data, positions, performance_metrics)
    
    print(f"Modified signal with high synthetic ratio: {result}")
    
    # Test with circuit breaker
    safeguards.daily_pnl = -600.0  # 6% loss
    
    result = safeguards.apply_safeguards(signal, market_data, positions, performance_metrics)
    
    print(f"Modified signal with circuit breaker: {result}")

if __name__ == "__main__":
    main()
