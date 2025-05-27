"""
Enhanced Risk Management Module for HyperliquidMaster

This module provides comprehensive risk management functionality
with position sizing, drawdown protection, and risk metrics.
"""

import logging
import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

class RiskManager:
    """
    Manages risk parameters and enforces risk management rules.
    
    This class provides functionality to manage risk parameters,
    calculate position sizes, and enforce risk management rules.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the risk manager.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.risk_metrics_file = "risk_metrics.json"
        
        # Initialize risk parameters
        self.max_drawdown = config.get("max_drawdown", 0.1)  # 10% max drawdown
        self.daily_loss_limit = config.get("daily_loss_limit", 0.05)  # 5% daily loss limit
        self.max_position_size = config.get("max_position_size", 0.1)  # 10% max position size
        self.max_open_positions = config.get("max_open_positions", 5)  # Max 5 open positions
        
        # Initialize risk metrics
        self.risk_metrics = self._load_risk_metrics()
        
        # Initialize trading state
        self.trading_paused = False
        self.pause_reason = None
        
        # Check if trading should be paused
        self._check_trading_pause()
        
        self.logger.info("Risk manager initialized")
    
    def _load_risk_metrics(self) -> Dict[str, Any]:
        """
        Load risk metrics from file or create default metrics.
        
        Returns:
            Dictionary of risk metrics
        """
        if os.path.exists(self.risk_metrics_file):
            try:
                with open(self.risk_metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                self.logger.info(f"Risk metrics loaded from {self.risk_metrics_file}")
                return metrics
            except Exception as e:
                self.logger.error(f"Error loading risk metrics: {e}")
                return self._create_default_metrics()
        else:
            return self._create_default_metrics()
    
    def _create_default_metrics(self) -> Dict[str, Any]:
        """
        Create default risk metrics.
        
        Returns:
            Dictionary of default risk metrics
        """
        metrics = {
            "account_equity": 1000.0,  # Initial account equity
            "starting_equity": 1000.0,  # Starting equity for drawdown calculation
            "peak_equity": 1000.0,  # Peak equity for drawdown calculation
            "current_drawdown": 0.0,  # Current drawdown percentage
            "max_drawdown_reached": 0.0,  # Maximum drawdown reached
            "daily_pnl": 0.0,  # Daily profit/loss
            "daily_pnl_percentage": 0.0,  # Daily profit/loss percentage
            "daily_reset_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Time of last daily reset
            "total_trades": 0,  # Total number of trades
            "winning_trades": 0,  # Number of winning trades
            "losing_trades": 0,  # Number of losing trades
            "win_rate": 0.0,  # Win rate percentage
            "average_win": 0.0,  # Average win amount
            "average_loss": 0.0,  # Average loss amount
            "profit_factor": 0.0,  # Profit factor (gross profit / gross loss)
            "consecutive_wins": 0,  # Number of consecutive winning trades
            "consecutive_losses": 0,  # Number of consecutive losing trades
            "max_consecutive_wins": 0,  # Maximum number of consecutive winning trades
            "max_consecutive_losses": 0,  # Maximum number of consecutive losing trades
            "open_positions": []  # List of open positions
        }
        
        # Save default metrics
        self._save_risk_metrics(metrics)
        
        return metrics
    
    def _save_risk_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Save risk metrics to file.
        
        Args:
            metrics: Dictionary of risk metrics
        """
        try:
            with open(self.risk_metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            self.logger.debug(f"Risk metrics saved to {self.risk_metrics_file}")
        except Exception as e:
            self.logger.error(f"Error saving risk metrics: {e}")
    
    def _check_trading_pause(self) -> None:
        """
        Check if trading should be paused based on risk metrics.
        """
        # Check max drawdown
        if self.risk_metrics["current_drawdown"] >= self.max_drawdown:
            self.trading_paused = True
            self.pause_reason = f"Max drawdown reached: {self.risk_metrics['current_drawdown']:.2%} >= {self.max_drawdown:.2%}"
            self.logger.warning(self.pause_reason)
            return
        
        # Check daily loss limit
        if self.risk_metrics["daily_pnl_percentage"] <= -self.daily_loss_limit:
            self.trading_paused = True
            self.pause_reason = f"Daily loss limit reached: {self.risk_metrics['daily_pnl_percentage']:.2%} <= -{self.daily_loss_limit:.2%}"
            self.logger.warning(self.pause_reason)
            return
        
        # Check consecutive losses
        if self.risk_metrics["consecutive_losses"] >= 5:
            self.trading_paused = True
            self.pause_reason = f"Too many consecutive losses: {self.risk_metrics['consecutive_losses']} >= 5"
            self.logger.warning(self.pause_reason)
            return
        
        # Trading not paused
        self.trading_paused = False
        self.pause_reason = None
    
    def update_account_equity(self, equity: float) -> None:
        """
        Update account equity and recalculate risk metrics.
        
        Args:
            equity: Current account equity
        """
        try:
            # Get previous equity
            previous_equity = self.risk_metrics["account_equity"]
            
            # Update equity
            self.risk_metrics["account_equity"] = equity
            
            # Update peak equity if new equity is higher
            if equity > self.risk_metrics["peak_equity"]:
                self.risk_metrics["peak_equity"] = equity
            
            # Calculate current drawdown
            if self.risk_metrics["peak_equity"] > 0:
                self.risk_metrics["current_drawdown"] = 1.0 - (equity / self.risk_metrics["peak_equity"])
                
                # Update max drawdown reached if current drawdown is higher
                if self.risk_metrics["current_drawdown"] > self.risk_metrics["max_drawdown_reached"]:
                    self.risk_metrics["max_drawdown_reached"] = self.risk_metrics["current_drawdown"]
            
            # Update daily PnL
            daily_pnl = equity - previous_equity
            self.risk_metrics["daily_pnl"] += daily_pnl
            
            # Calculate daily PnL percentage
            if self.risk_metrics["starting_equity"] > 0:
                self.risk_metrics["daily_pnl_percentage"] = self.risk_metrics["daily_pnl"] / self.risk_metrics["starting_equity"]
            
            # Check if daily reset is needed
            self._check_daily_reset()
            
            # Save updated metrics
            self._save_risk_metrics(self.risk_metrics)
            
            # Check if trading should be paused
            self._check_trading_pause()
            
            self.logger.info(f"Account equity updated: {equity:.2f}, Drawdown: {self.risk_metrics['current_drawdown']:.2%}, Daily PnL: {self.risk_metrics['daily_pnl']:.2f} ({self.risk_metrics['daily_pnl_percentage']:.2%})")
        except Exception as e:
            self.logger.error(f"Error updating account equity: {e}")
    
    def _check_daily_reset(self) -> None:
        """
        Check if daily reset is needed and reset daily metrics if necessary.
        """
        try:
            # Parse last reset time
            last_reset_time = datetime.strptime(self.risk_metrics["daily_reset_time"], "%Y-%m-%d %H:%M:%S")
            
            # Check if a day has passed since last reset
            if datetime.now() - last_reset_time > timedelta(days=1):
                # Reset daily metrics
                self.risk_metrics["daily_pnl"] = 0.0
                self.risk_metrics["daily_pnl_percentage"] = 0.0
                self.risk_metrics["daily_reset_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.risk_metrics["starting_equity"] = self.risk_metrics["account_equity"]
                
                self.logger.info("Daily risk metrics reset")
        except Exception as e:
            self.logger.error(f"Error checking daily reset: {e}")
    
    def update_trade_metrics(self, trade_result: Dict[str, Any]) -> None:
        """
        Update trade metrics based on a completed trade.
        
        Args:
            trade_result: Dictionary containing trade result information
        """
        try:
            # Extract trade information
            pnl = trade_result.get("pnl", 0.0)
            is_win = pnl > 0
            
            # Update trade counts
            self.risk_metrics["total_trades"] += 1
            
            if is_win:
                self.risk_metrics["winning_trades"] += 1
                self.risk_metrics["consecutive_wins"] += 1
                self.risk_metrics["consecutive_losses"] = 0
                
                # Update max consecutive wins
                if self.risk_metrics["consecutive_wins"] > self.risk_metrics["max_consecutive_wins"]:
                    self.risk_metrics["max_consecutive_wins"] = self.risk_metrics["consecutive_wins"]
            else:
                self.risk_metrics["losing_trades"] += 1
                self.risk_metrics["consecutive_losses"] += 1
                self.risk_metrics["consecutive_wins"] = 0
                
                # Update max consecutive losses
                if self.risk_metrics["consecutive_losses"] > self.risk_metrics["max_consecutive_losses"]:
                    self.risk_metrics["max_consecutive_losses"] = self.risk_metrics["consecutive_losses"]
            
            # Calculate win rate
            if self.risk_metrics["total_trades"] > 0:
                self.risk_metrics["win_rate"] = self.risk_metrics["winning_trades"] / self.risk_metrics["total_trades"]
            
            # Update average win/loss
            if is_win and self.risk_metrics["winning_trades"] > 0:
                self.risk_metrics["average_win"] = ((self.risk_metrics["average_win"] * (self.risk_metrics["winning_trades"] - 1)) + pnl) / self.risk_metrics["winning_trades"]
            elif not is_win and self.risk_metrics["losing_trades"] > 0:
                self.risk_metrics["average_loss"] = ((self.risk_metrics["average_loss"] * (self.risk_metrics["losing_trades"] - 1)) + pnl) / self.risk_metrics["losing_trades"]
            
            # Calculate profit factor
            gross_profit = self.risk_metrics["average_win"] * self.risk_metrics["winning_trades"]
            gross_loss = abs(self.risk_metrics["average_loss"] * self.risk_metrics["losing_trades"])
            
            if gross_loss > 0:
                self.risk_metrics["profit_factor"] = gross_profit / gross_loss
            
            # Save updated metrics
            self._save_risk_metrics(self.risk_metrics)
            
            # Check if trading should be paused
            self._check_trading_pause()
            
            self.logger.info(f"Trade metrics updated: Win Rate: {self.risk_metrics['win_rate']:.2%}, Profit Factor: {self.risk_metrics['profit_factor']:.2f}")
        except Exception as e:
            self.logger.error(f"Error updating trade metrics: {e}")
    
    def update_open_positions(self, positions: List[Dict[str, Any]]) -> None:
        """
        Update open positions list.
        
        Args:
            positions: List of open positions
        """
        try:
            # Update open positions
            self.risk_metrics["open_positions"] = positions
            
            # Save updated metrics
            self._save_risk_metrics(self.risk_metrics)
            
            self.logger.debug(f"Open positions updated: {len(positions)} positions")
        except Exception as e:
            self.logger.error(f"Error updating open positions: {e}")
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss_price: float, account_equity: Optional[float] = None) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_price: Stop loss price
            account_equity: Account equity (optional, uses stored equity if not provided)
            
        Returns:
            Position size
        """
        try:
            # Get account equity
            equity = account_equity or self.risk_metrics["account_equity"]
            
            # Get risk level from config
            risk_level = self.config.get("risk_level", 0.01)  # Default 1% risk per trade
            
            # Calculate risk amount
            risk_amount = equity * risk_level
            
            # Calculate price difference
            price_diff = abs(entry_price - stop_loss_price)
            
            if price_diff == 0:
                self.logger.warning("Stop loss price is equal to entry price, using default position size")
                return 0.01  # Default small position size
            
            # Calculate position size
            position_size = risk_amount / price_diff
            
            # Apply volatility adjustment if available
            volatility = self._get_symbol_volatility(symbol)
            if volatility > 0:
                # Reduce position size for high volatility
                volatility_factor = 1.0 / (1.0 + volatility)
                position_size *= volatility_factor
                self.logger.debug(f"Applied volatility adjustment: {volatility_factor:.2f}")
            
            # Limit position size to max position size
            max_size = equity * self.max_position_size
            if position_size > max_size:
                self.logger.warning(f"Position size {position_size:.4f} exceeds max position size {max_size:.4f}, limiting")
                position_size = max_size
            
            # Check if trading is paused
            if self.trading_paused:
                self.logger.warning(f"Trading is paused: {self.pause_reason}, returning zero position size")
                return 0.0
            
            # Check if max open positions reached
            if len(self.risk_metrics["open_positions"]) >= self.max_open_positions:
                self.logger.warning(f"Max open positions reached: {len(self.risk_metrics['open_positions'])} >= {self.max_open_positions}, returning zero position size")
                return 0.0
            
            self.logger.info(f"Calculated position size: {position_size:.4f} for {symbol} with {risk_level:.2%} risk")
            return position_size
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.01  # Default small position size
            
    def calculate_risk_reward_ratio(self, entry_price: float, stop_loss_price: float, take_profit_price: float) -> float:
        """
        Calculate risk-reward ratio for a trade setup.
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            
        Returns:
            Risk-reward ratio (reward/risk)
        """
        try:
            # Calculate risk and reward
            risk = abs(entry_price - stop_loss_price)
            reward = abs(entry_price - take_profit_price)
            
            if risk == 0:
                self.logger.warning("Risk is zero, cannot calculate risk-reward ratio")
                return 0.0
                
            # Calculate ratio
            ratio = reward / risk
            
            self.logger.debug(f"Calculated risk-reward ratio: {ratio:.2f}")
            return ratio
        except Exception as e:
            self.logger.error(f"Error calculating risk-reward ratio: {e}")
            return 0.0
            
    def validate_risk_reward_ratio(self, entry_price: float, stop_loss_price: float, take_profit_price: float) -> bool:
        """
        Validate if a trade setup meets the minimum risk-reward ratio.
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Calculate risk-reward ratio
            ratio = self.calculate_risk_reward_ratio(entry_price, stop_loss_price, take_profit_price)
            
            # Get minimum ratio from config
            min_ratio = self.config.get("min_risk_reward_ratio", 2.0)
            
            # Validate ratio
            is_valid = ratio >= min_ratio
            
            if not is_valid:
                self.logger.warning(f"Trade setup does not meet minimum risk-reward ratio: {ratio:.2f} < {min_ratio:.2f}")
            else:
                self.logger.debug(f"Trade setup meets minimum risk-reward ratio: {ratio:.2f} >= {min_ratio:.2f}")
                
            return is_valid
        except Exception as e:
            self.logger.error(f"Error validating risk-reward ratio: {e}")
            return False
            
    def check_drawdown_protection(self) -> bool:
        """
        Check if trading should be paused due to drawdown protection.
        
        Returns:
            True if trading should continue, False if it should be paused
        """
        try:
            # Check current drawdown against max allowed
            current_drawdown = self.risk_metrics["current_drawdown"]
            max_drawdown = self.max_drawdown
            
            if current_drawdown >= max_drawdown:
                self.logger.warning(f"Drawdown protection triggered: {current_drawdown:.2%} >= {max_drawdown:.2%}")
                return False
                
            # Check daily loss against max allowed
            daily_loss = -self.risk_metrics["daily_pnl_percentage"]
            max_daily_loss = self.daily_loss_limit
            
            if daily_loss >= max_daily_loss:
                self.logger.warning(f"Daily loss protection triggered: {daily_loss:.2%} >= {max_daily_loss:.2%}")
                return False
                
            # Check consecutive losses
            consecutive_losses = self.risk_metrics["consecutive_losses"]
            max_consecutive_losses = 5  # Default value
            
            if consecutive_losses >= max_consecutive_losses:
                self.logger.warning(f"Consecutive loss protection triggered: {consecutive_losses} >= {max_consecutive_losses}")
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Error checking drawdown protection: {e}")
            return True  # Default to allowing trading in case of error
    
    def _get_symbol_volatility(self, symbol: str) -> float:
        """
        Get volatility for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Volatility value (0.0 if not available)
        """
        # In a real implementation, this would get volatility from market data
        # For now, return default values for common symbols
        volatility_map = {
            "BTC": 0.03,  # 3% daily volatility
            "ETH": 0.04,  # 4% daily volatility
            "XRP": 0.05,  # 5% daily volatility
            "SOL": 0.06,  # 6% daily volatility
            "DOGE": 0.08,  # 8% daily volatility
            "AVAX": 0.07,  # 7% daily volatility
            "LINK": 0.05,  # 5% daily volatility
            "MATIC": 0.06   # 6% daily volatility
        }
        
        return volatility_map.get(symbol, 0.05)  # Default 5% volatility
    
    def is_trading_allowed(self) -> bool:
        """
        Check if trading is allowed based on risk parameters.
        
        Returns:
            True if trading is allowed, False otherwise
        """
        return not self.trading_paused
    
    def get_trading_pause_reason(self) -> Optional[str]:
        """
        Get the reason why trading is paused.
        
        Returns:
            Reason string, or None if trading is not paused
        """
        return self.pause_reason
    
    def reset_trading_pause(self) -> None:
        """
        Reset trading pause state.
        """
        self.trading_paused = False
        self.pause_reason = None
        self.logger.info("Trading pause reset")
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get current risk metrics.
        
        Returns:
            Dictionary of risk metrics
        """
        return self.risk_metrics
    
    def get_max_drawdown(self) -> float:
        """
        Get maximum drawdown limit.
        
        Returns:
            Maximum drawdown limit as a decimal (0.1 = 10%)
        """
        return self.max_drawdown
    
    def set_max_drawdown(self, max_drawdown: float) -> None:
        """
        Set maximum drawdown limit.
        
        Args:
            max_drawdown: Maximum drawdown limit as a decimal (0.1 = 10%)
        """
        self.max_drawdown = max_drawdown
        self.config["max_drawdown"] = max_drawdown
        self.logger.info(f"Max drawdown set to {max_drawdown:.2%}")
        
        # Check if trading should be paused
        self._check_trading_pause()
    
    def get_daily_loss_limit(self) -> float:
        """
        Get daily loss limit.
        
        Returns:
            Daily loss limit as a decimal (0.05 = 5%)
        """
        return self.daily_loss_limit
    
    def set_daily_loss_limit(self, daily_loss_limit: float) -> None:
        """
        Set daily loss limit.
        
        Args:
            daily_loss_limit: Daily loss limit as a decimal (0.05 = 5%)
        """
        self.daily_loss_limit = daily_loss_limit
        self.config["daily_loss_limit"] = daily_loss_limit
        self.logger.info(f"Daily loss limit set to {daily_loss_limit:.2%}")
        
        # Check if trading should be paused
        self._check_trading_pause()
    
    def get_max_position_size(self) -> float:
        """
        Get maximum position size.
        
        Returns:
            Maximum position size as a decimal (0.1 = 10%)
        """
        return self.max_position_size
    
    def set_max_position_size(self, max_position_size: float) -> None:
        """
        Set maximum position size.
        
        Args:
            max_position_size: Maximum position size as a decimal (0.1 = 10%)
        """
        self.max_position_size = max_position_size
        self.config["max_position_size"] = max_position_size
        self.logger.info(f"Max position size set to {max_position_size:.2%}")
    
    def get_max_open_positions(self) -> int:
        """
        Get maximum number of open positions.
        
        Returns:
            Maximum number of open positions
        """
        return self.max_open_positions
    
    def set_max_open_positions(self, max_open_positions: int) -> None:
        """
        Set maximum number of open positions.
        
        Args:
            max_open_positions: Maximum number of open positions
        """
        self.max_open_positions = max_open_positions
        self.config["max_open_positions"] = max_open_positions
        self.logger.info(f"Max open positions set to {max_open_positions}")
    
    def get_risk_report(self) -> str:
        """
        Get a formatted risk report.
        
        Returns:
            Formatted risk report string
        """
        try:
            metrics = self.risk_metrics
            
            report = "=== RISK MANAGEMENT REPORT ===\n\n"
            
            # Account metrics
            report += "Account Metrics:\n"
            report += f"- Account Equity: ${metrics['account_equity']:.2f}\n"
            report += f"- Peak Equity: ${metrics['peak_equity']:.2f}\n"
            report += f"- Current Drawdown: {metrics['current_drawdown']:.2%}\n"
            report += f"- Max Drawdown Reached: {metrics['max_drawdown_reached']:.2%}\n"
            report += f"- Daily PnL: ${metrics['daily_pnl']:.2f} ({metrics['daily_pnl_percentage']:.2%})\n\n"
            
            # Trade metrics
            report += "Trade Metrics:\n"
            report += f"- Total Trades: {metrics['total_trades']}\n"
            report += f"- Win Rate: {metrics['win_rate']:.2%}\n"
            report += f"- Average Win: ${metrics['average_win']:.2f}\n"
            report += f"- Average Loss: ${metrics['average_loss']:.2f}\n"
            report += f"- Profit Factor: {metrics['profit_factor']:.2f}\n"
            report += f"- Consecutive Wins: {metrics['consecutive_wins']}\n"
            report += f"- Consecutive Losses: {metrics['consecutive_losses']}\n"
            report += f"- Max Consecutive Wins: {metrics['max_consecutive_wins']}\n"
            report += f"- Max Consecutive Losses: {metrics['max_consecutive_losses']}\n\n"
            
            # Risk parameters
            report += "Risk Parameters:\n"
            report += f"- Max Drawdown: {self.max_drawdown:.2%}\n"
            report += f"- Daily Loss Limit: {self.daily_loss_limit:.2%}\n"
            report += f"- Max Position Size: {self.max_position_size:.2%}\n"
            report += f"- Max Open Positions: {self.max_open_positions}\n\n"
            
            # Trading status
            report += "Trading Status:\n"
            report += f"- Trading Allowed: {'No' if self.trading_paused else 'Yes'}\n"
            if self.trading_paused:
                report += f"- Pause Reason: {self.pause_reason}\n"
            
            return report
        except Exception as e:
            self.logger.error(f"Error generating risk report: {e}")
            return "Error generating risk report"
