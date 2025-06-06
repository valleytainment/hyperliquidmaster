#!/usr/bin/env python3
"""
Enhanced Trading Engine with Real-time Wallet Equity and Price Display
---------------------------------------------------------------------
Integrates all missing features from master_bot.py including:
• Real-time wallet equity monitoring
• Live token price feeds
• Comprehensive position tracking
• Advanced order management
• Circuit breaker protection
"""

import os
import time
import math
import json
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

try:
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange
    from eth_account import Account
    from eth_account.signers.local import LocalAccount
except ImportError as e:
    logging.error(f"Required packages missing: {e}")
    raise

from strategies.base_strategy import BaseStrategy, TradingSignal, SignalType, MarketData
from strategies.enhanced_neural_strategy import EnhancedNeuralStrategy
from strategies.bb_rsi_adx import BBRSIADXStrategy
from strategies.hull_suite import HullSuiteStrategy
from risk_management.risk_manager import RiskManager

logger = logging.getLogger(__name__)

class EnhancedTradingEngine:
    """Enhanced Trading Engine with real-time wallet equity and price display"""
    
    def __init__(self, config: dict):
        self.config = config
        self.running = False
        self.automation_running = False
        
        # API configuration
        self.account_address = config.get("account_address", "")
        self.secret_key = config.get("secret_key", "")
        self.api_url = config.get("api_url", "https://api.hyperliquid.xyz")
        
        # Trading configuration
        self.current_symbol = config.get("trade_symbol", "BTC-USD-PERP")
        self.trade_mode = config.get("trade_mode", "perp").lower()
        self.starting_capital = config.get("starting_capital", 100.0)
        
        # Initialize wallet and API clients
        self.wallet: Optional[LocalAccount] = None
        self.exchange: Optional[Exchange] = None
        self.info_client: Optional[Info] = None
        
        # Initialize components
        self.risk_manager = None
        self.strategies = {}
        self.active_strategy = None
        
        # Real-time data tracking
        self.current_equity = 0.0
        self.current_price = 0.0
        self.current_volume = 0.0
        self.price_history = []
        self.equity_history = []
        
        # Position tracking
        self.positions = {}
        self.trade_history = []
        self.last_trade_time = 0
        self.circuit_breaker_triggered = False
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        
        # Threading
        self.trading_thread = None
        self.price_update_thread = None
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        logger.info("Enhanced Trading Engine initialized")
    
    def initialize(self):
        """Initialize all components"""
        try:
            # Initialize wallet and API
            self._initialize_api()
            
            # Initialize risk manager
            self._initialize_risk_manager()
            
            # Initialize strategies
            self._initialize_strategies()
            
            # Set default strategy
            self.active_strategy = self.strategies.get("enhanced_neural")
            
            # Start real-time data updates
            self._start_real_time_updates()
            
            logger.info("Enhanced trading engine initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced trading engine: {e}")
            return False
    
    def _initialize_api(self):
        """Initialize Hyperliquid API connection"""
        try:
            if self.secret_key:
                self.wallet = Account.from_key(self.secret_key)
                logger.info(f"Wallet initialized: {self.wallet.address}")
            else:
                logger.warning("No private key provided - read-only mode")
            
            # Initialize exchange client
            self.exchange = Exchange(
                wallet=self.wallet,
                base_url=self.api_url,
                account_address=self.account_address
            )
            
            # Initialize info client
            self.info_client = Info(self.api_url)
            
            logger.info("API clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize API: {e}")
            raise
    
    def _initialize_risk_manager(self):
        """Initialize risk management system"""
        try:
            risk_config = {
                "max_position_size": self.config.get("max_position_size", 0.1),
                "max_daily_loss": self.config.get("max_daily_loss", 0.05),
                "stop_loss_pct": self.config.get("stop_loss_pct", 0.02),
                "take_profit_pct": self.config.get("take_profit_pct", 0.04),
                "max_positions": self.config.get("max_positions", 3),
                "circuit_breaker_threshold": self.config.get("circuit_breaker_threshold", 0.1)
            }
            
            self.risk_manager = RiskManager(risk_config)
            logger.info("Risk manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize risk manager: {e}")
            raise
    
    def _initialize_strategies(self):
        """Initialize trading strategies"""
        try:
            # Enhanced Neural Network Strategy
            self.strategies["enhanced_neural"] = EnhancedNeuralStrategy(
                api=self,
                risk_manager=self.risk_manager,
                config=self.config
            )
            
            # BB RSI ADX Strategy
            self.strategies["bb_rsi_adx"] = BBRSIADXStrategy(
                api=self,
                risk_manager=self.risk_manager
            )
            
            # Hull Suite Strategy
            self.strategies["hull_suite"] = HullSuiteStrategy(
                api=self,
                risk_manager=self.risk_manager
            )
            
            logger.info(f"Initialized {len(self.strategies)} trading strategies")
            
        except Exception as e:
            logger.error(f"Failed to initialize strategies: {e}")
            raise
    
    def _start_real_time_updates(self):
        """Start real-time data update threads"""
        try:
            # Start price update thread
            self.price_update_thread = threading.Thread(
                target=self._price_update_loop, 
                daemon=True
            )
            self.price_update_thread.start()
            
            logger.info("Real-time data updates started")
            
        except Exception as e:
            logger.error(f"Failed to start real-time updates: {e}")
    
    def _price_update_loop(self):
        """Continuous price and equity update loop"""
        while True:
            try:
                # Update current equity
                self.current_equity = self.get_equity()
                
                # Update current price and volume
                price_data = self.fetch_price_volume()
                if price_data:
                    self.current_price = price_data["price"]
                    self.current_volume = price_data["volume"]
                    
                    # Store price history
                    self.price_history.append({
                        "timestamp": datetime.now(),
                        "price": self.current_price,
                        "volume": self.current_volume
                    })
                    
                    # Limit history size
                    if len(self.price_history) > 1000:
                        self.price_history = self.price_history[-500:]
                
                # Store equity history
                self.equity_history.append({
                    "timestamp": datetime.now(),
                    "equity": self.current_equity
                })
                
                # Limit equity history size
                if len(self.equity_history) > 1000:
                    self.equity_history = self.equity_history[-500:]
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Sleep for update interval
                time.sleep(self.config.get("price_update_interval", 2))
                
            except Exception as e:
                logger.error(f"Error in price update loop: {e}")
                time.sleep(5)
    
    def connect(self):
        """Connect to the trading API"""
        try:
            if not self.exchange or not self.info_client:
                self._initialize_api()
            
            # Test connection by getting equity
            equity = self.get_equity()
            logger.info(f"Connected successfully - Equity: ${equity:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the trading API"""
        try:
            if self.automation_running:
                self.stop_automation()
            
            self.exchange = None
            self.info_client = None
            logger.info("Disconnected from API")
            return True
            
        except Exception as e:
            logger.error(f"Disconnect failed: {e}")
            return False
    
    def get_equity(self) -> float:
        """Get current account equity with enhanced error handling"""
        try:
            if not self.info_client:
                return 0.0
            
            if self.trade_mode == "spot":
                state = self.info_client.spot_clearinghouse_state(self.account_address)
                for balance in state.get("balances", []):
                    if balance.get("coin", "").upper() in ["USDC", "USD"]:
                        return float(balance.get("total", 0))
                return 0.0
            else:
                state = self.info_client.user_state(self.account_address)
                
                # Try multiple equity fields
                equity = state.get("portfolioStats", {}).get("equity", None)
                if equity is not None:
                    return float(equity)
                
                # Fallback to cross margin summary
                cross_val = state.get("crossMarginSummary", {}).get("accountValue", None)
                if cross_val is not None:
                    return float(cross_val)
                
                # Fallback to margin summary
                margin_val = state.get("marginSummary", {}).get("accountValue", None)
                if margin_val is not None:
                    return float(margin_val)
                
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error getting equity: {e}")
            return self.current_equity  # Return cached value
    
    def fetch_price_volume(self, symbol: str = None) -> Optional[Dict]:
        """Fetch current price and volume for symbol with enhanced data"""
        try:
            if not self.info_client:
                return None
            
            symbol = symbol or self.current_symbol
            
            # Get market data
            if self.trade_mode == "spot":
                # Spot market data
                markets = self.info_client.spot_meta()
                for market in markets.get("tokens", []):
                    if market.get("name") == symbol.replace("-USD-SPOT", ""):
                        # Get real spot price data
                        ticker = self.info_client.spot_clearinghouse_state(self.account_address)
                        return {
                            "price": float(market.get("midPx", 0)),
                            "volume": float(market.get("dayNtlVlm", 0)),
                            "high_24h": float(market.get("dayHigh", 0)),
                            "low_24h": float(market.get("dayLow", 0)),
                            "change_24h": float(market.get("dayChange", 0))
                        }
            else:
                # Perp market data
                markets = self.info_client.meta()
                for market in markets.get("universe", []):
                    if market.get("name") == symbol.replace("-USD-PERP", ""):
                        # Get real perp price data
                        return {
                            "price": float(market.get("midPx", 0)),
                            "volume": float(market.get("dayNtlVlm", 0)),
                            "high_24h": float(market.get("dayHigh", 0)),
                            "low_24h": float(market.get("dayLow", 0)),
                            "change_24h": float(market.get("dayChange", 0)),
                            "funding_rate": float(market.get("funding", 0)),
                            "open_interest": float(market.get("openInterest", 0))
                        }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error fetching price/volume: {e}")
            return None
    
    def get_user_positions(self) -> List[Dict]:
        """Get current user positions with enhanced details"""
        try:
            if not self.info_client:
                return []
            
            positions = []
            state = self.info_client.user_state(self.account_address)
            
            for asset_pos in state.get("assetPositions", []):
                position = asset_pos.get("position", {})
                coin = position.get("coin", "")
                size = float(position.get("szi", 0))
                
                if size != 0:
                    # Get current price for P&L calculation
                    current_price = self.current_price
                    entry_price = float(position.get("entryPx", 0))
                    
                    # Calculate unrealized P&L
                    if size > 0:  # Long position
                        unrealized_pnl = size * (current_price - entry_price)
                    else:  # Short position
                        unrealized_pnl = abs(size) * (entry_price - current_price)
                    
                    positions.append({
                        "symbol": f"{coin}-USD-PERP",
                        "side": 1 if size > 0 else 2,
                        "size": abs(size),
                        "entryPrice": entry_price,
                        "currentPrice": current_price,
                        "unrealizedPnl": unrealized_pnl,
                        "pnlPercent": (unrealized_pnl / (entry_price * abs(size))) * 100 if entry_price > 0 else 0,
                        "leverage": position.get("leverage", {}).get("value", 1),
                        "marginUsed": float(position.get("marginUsed", 0))
                    })
            
            return positions
            
        except Exception as e:
            logger.warning(f"Error getting positions: {e}")
            return []
    
    def get_real_time_data(self) -> Dict:
        """Get comprehensive real-time trading data"""
        try:
            positions = self.get_user_positions()
            
            # Calculate total unrealized P&L
            total_unrealized_pnl = sum(pos["unrealizedPnl"] for pos in positions)
            
            # Get recent price changes
            price_change_1m = 0.0
            price_change_5m = 0.0
            price_change_1h = 0.0
            
            if len(self.price_history) > 1:
                current_time = datetime.now()
                
                # 1 minute change
                one_min_ago = [p for p in self.price_history 
                              if (current_time - p["timestamp"]).total_seconds() >= 60]
                if one_min_ago:
                    old_price = one_min_ago[-1]["price"]
                    price_change_1m = ((self.current_price - old_price) / old_price) * 100
                
                # 5 minute change
                five_min_ago = [p for p in self.price_history 
                               if (current_time - p["timestamp"]).total_seconds() >= 300]
                if five_min_ago:
                    old_price = five_min_ago[-1]["price"]
                    price_change_5m = ((self.current_price - old_price) / old_price) * 100
                
                # 1 hour change
                one_hour_ago = [p for p in self.price_history 
                               if (current_time - p["timestamp"]).total_seconds() >= 3600]
                if one_hour_ago:
                    old_price = one_hour_ago[-1]["price"]
                    price_change_1h = ((self.current_price - old_price) / old_price) * 100
            
            return {
                "timestamp": datetime.now(),
                "equity": self.current_equity,
                "price": self.current_price,
                "volume": self.current_volume,
                "positions": positions,
                "total_unrealized_pnl": total_unrealized_pnl,
                "price_changes": {
                    "1m": price_change_1m,
                    "5m": price_change_5m,
                    "1h": price_change_1h
                },
                "performance": self.get_performance_metrics(),
                "automation_status": self.automation_running,
                "circuit_breaker": self.circuit_breaker_triggered
            }
            
        except Exception as e:
            logger.error(f"Error getting real-time data: {e}")
            return {}
    
    def execute_manual_trade(self, side: str, size: float, symbol: str = None) -> Dict:
        """Execute manual trade with enhanced validation"""
        try:
            symbol = symbol or self.current_symbol
            coin = self._parse_base_coin(symbol)
            
            logger.info(f"Executing manual {side} trade: {size} {coin}")
            
            # Enhanced risk checks
            if not self.risk_manager.can_open_position(size, self.current_equity):
                return {"success": False, "error": "Risk limits exceeded"}
            
            # Check circuit breaker
            if self.circuit_breaker_triggered:
                return {"success": False, "error": "Circuit breaker is active"}
            
            # Execute order
            is_buy = side.upper() == "BUY"
            result = self.exchange.market_open(coin, is_buy, size)
            
            # Log trade with enhanced details
            trade_record = {
                "timestamp": datetime.now(),
                "symbol": symbol,
                "side": side,
                "size": size,
                "price": self.current_price,
                "equity_before": self.current_equity,
                "type": "manual",
                "result": result
            }
            self.trade_history.append(trade_record)
            self.total_trades += 1
            
            logger.info(f"Manual trade executed: {result}")
            return {"success": True, "result": result, "trade_record": trade_record}
            
        except Exception as e:
            logger.error(f"Manual trade execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def close_all_positions(self) -> Dict:
        """Close all open positions with enhanced tracking"""
        try:
            positions = self.get_user_positions()
            results = []
            total_pnl = 0.0
            
            for position in positions:
                coin = self._parse_base_coin(position["symbol"])
                size = position["size"]
                is_long = position["side"] == 1
                
                # Close position (opposite side)
                result = self.exchange.market_close(coin, not is_long, size)
                results.append(result)
                
                # Track P&L
                total_pnl += position["unrealizedPnl"]
                
                logger.info(f"Closed position: {coin} {size} - P&L: ${position['unrealizedPnl']:.2f}")
            
            # Update performance metrics
            self.total_pnl += total_pnl
            if total_pnl > 0:
                self.winning_trades += len(positions)
            
            return {
                "success": True, 
                "results": results, 
                "total_pnl": total_pnl,
                "positions_closed": len(positions)
            }
            
        except Exception as e:
            logger.error(f"Failed to close positions: {e}")
            return {"success": False, "error": str(e)}
    
    def start_automation(self, mode: str, strategy_name: str):
        """Start 24/7 automated trading with enhanced monitoring"""
        try:
            if self.automation_running:
                logger.warning("Automation already running")
                return
            
            # Set trading mode
            self.trade_mode = mode.lower()
            
            # Set active strategy
            if strategy_name.lower().replace(" ", "_") in self.strategies:
                self.active_strategy = self.strategies[strategy_name.lower().replace(" ", "_")]
            else:
                self.active_strategy = self.strategies["enhanced_neural"]
            
            # Reset circuit breaker
            self.circuit_breaker_triggered = False
            
            # Start automation
            self.automation_running = True
            self.trading_thread = threading.Thread(target=self._automation_loop, daemon=True)
            self.trading_thread.start()
            
            logger.info(f"Started 24/7 automation - Mode: {mode}, Strategy: {strategy_name}")
            
        except Exception as e:
            logger.error(f"Failed to start automation: {e}")
            self.automation_running = False
    
    def stop_automation(self):
        """Stop automated trading"""
        try:
            self.automation_running = False
            
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=5)
            
            logger.info("Automation stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop automation: {e}")
    
    def _automation_loop(self):
        """Enhanced automation trading loop"""
        logger.info("Starting enhanced automation loop")
        
        while self.automation_running:
            try:
                # Check circuit breaker
                if self._check_circuit_breaker():
                    logger.warning("Circuit breaker triggered - stopping automation")
                    break
                
                # Get current market data
                price_data = self.fetch_price_volume()
                if not price_data:
                    time.sleep(2)
                    continue
                
                # Create market data object
                market_data = MarketData(
                    symbol=self.current_symbol,
                    price=price_data["price"],
                    volume=price_data["volume"],
                    timestamp=datetime.now()
                )
                
                # Generate trading signal
                if self.active_strategy:
                    signal = self.active_strategy.generate_signal(market_data)
                    
                    if signal.signal_type != SignalType.HOLD:
                        self._execute_signal(signal, market_data)
                
                # Manage existing positions
                self._manage_positions()
                
                # Sleep before next iteration
                time.sleep(self.config.get("poll_interval_seconds", 2))
                
            except Exception as e:
                logger.error(f"Error in automation loop: {e}")
                time.sleep(5)
        
        logger.info("Enhanced automation loop ended")
    
    def _execute_signal(self, signal: TradingSignal, market_data: MarketData):
        """Execute trading signal with enhanced validation"""
        try:
            # Check minimum trade interval
            min_interval = self.config.get("min_trade_interval", 60)
            if time.time() - self.last_trade_time < min_interval:
                return
            
            # Calculate position size
            if self.config.get("use_manual_entry_size", True):
                position_size = self.config.get("manual_entry_size", 20.0) / market_data.price
            else:
                risk_pct = self.config.get("risk_percent", 0.01)
                position_value = self.current_equity * risk_pct
                position_size = position_value / market_data.price
            
            if position_size <= 0:
                return
            
            # Check risk limits
            if not self.risk_manager.can_open_position(position_size, self.current_equity):
                logger.warning("Risk limits prevent trade execution")
                return
            
            # Execute trade
            side = "BUY" if signal.signal_type == SignalType.BUY else "SELL"
            result = self.execute_manual_trade(side, position_size, self.current_symbol)
            
            if result["success"]:
                self.last_trade_time = time.time()
                logger.info(f"Executed {side} signal: {position_size:.4f} @ {market_data.price:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to execute signal: {e}")
    
    def _manage_positions(self):
        """Enhanced position management with trailing stops"""
        try:
            positions = self.get_user_positions()
            
            for position in positions:
                self._manage_single_position(position)
                
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
    
    def _manage_single_position(self, position: Dict):
        """Manage a single position with enhanced features"""
        try:
            symbol = position["symbol"]
            side = position["side"]
            size = position["size"]
            entry_price = position["entryPrice"]
            current_price = self.current_price
            
            # Calculate P&L percentage
            if side == 1:  # Long position
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # Short position
                pnl_pct = (entry_price - current_price) / entry_price
            
            # Check stop loss
            stop_loss_pct = self.config.get("stop_loss_pct", 0.02)
            if pnl_pct <= -stop_loss_pct:
                self._close_position(position, "Stop Loss")
                return
            
            # Check take profit
            take_profit_pct = self.config.get("take_profit_pct", 0.04)
            if pnl_pct >= take_profit_pct:
                self._close_position(position, "Take Profit")
                return
            
            # Check trailing stop if enabled
            if self.config.get("use_trailing_stop", False):
                self._check_trailing_stop(position, current_price, pnl_pct)
                
        except Exception as e:
            logger.error(f"Error managing position: {e}")
    
    def _close_position(self, position: Dict, reason: str):
        """Close a specific position with enhanced tracking"""
        try:
            coin = self._parse_base_coin(position["symbol"])
            size = position["size"]
            is_long = position["side"] == 1
            
            # Close position (opposite side)
            result = self.exchange.market_close(coin, not is_long, size)
            
            # Calculate realized P&L
            realized_pnl = position["unrealizedPnl"]
            self.total_pnl += realized_pnl
            
            if realized_pnl > 0:
                self.winning_trades += 1
            
            # Record trade closure
            trade_record = {
                "timestamp": datetime.now(),
                "symbol": position["symbol"],
                "side": "CLOSE",
                "size": size,
                "price": self.current_price,
                "entry_price": position["entryPrice"],
                "realized_pnl": realized_pnl,
                "reason": reason,
                "type": "automated",
                "result": result
            }
            self.trade_history.append(trade_record)
            
            logger.info(f"Closed position: {coin} {size} - P&L: ${realized_pnl:.2f} ({reason})")
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
    
    def _check_trailing_stop(self, position: Dict, current_price: float, pnl_pct: float):
        """Check and update trailing stop"""
        # Implementation for trailing stop logic
        pass
    
    def _check_circuit_breaker(self) -> bool:
        """Enhanced circuit breaker with multiple conditions"""
        try:
            if self.circuit_breaker_triggered:
                return True
            
            # Check daily loss limit
            if self.starting_capital > 0:
                daily_loss_pct = (self.starting_capital - self.current_equity) / self.starting_capital
                threshold = self.config.get("circuit_breaker_threshold", 0.1)
                
                if daily_loss_pct >= threshold:
                    self.circuit_breaker_triggered = True
                    logger.warning(f"Circuit breaker triggered - Daily loss: {daily_loss_pct:.2%}")
                    return True
            
            # Check maximum drawdown
            if self.peak_equity > 0:
                current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
                max_drawdown_threshold = self.config.get("max_drawdown_threshold", 0.15)
                
                if current_drawdown >= max_drawdown_threshold:
                    self.circuit_breaker_triggered = True
                    logger.warning(f"Circuit breaker triggered - Max drawdown: {current_drawdown:.2%}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking circuit breaker: {e}")
            return False
    
    def _update_performance_metrics(self):
        """Enhanced performance metrics tracking"""
        try:
            # Update peak equity and drawdown
            if self.current_equity > self.peak_equity:
                self.peak_equity = self.current_equity
            
            if self.peak_equity > 0:
                current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
                if current_drawdown > self.max_drawdown:
                    self.max_drawdown = current_drawdown
                
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _parse_base_coin(self, symbol: str) -> str:
        """Parse base coin from symbol"""
        symbol = symbol.upper()
        if symbol.endswith("-USD-PERP") or symbol.endswith("-USD-SPOT"):
            return symbol[:-9]
        return symbol
    
    def get_performance_metrics(self) -> Dict:
        """Get enhanced performance metrics"""
        try:
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            total_return = ((self.current_equity - self.starting_capital) / self.starting_capital * 100) if self.starting_capital > 0 else 0
            
            return {
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "win_rate": win_rate,
                "total_pnl": self.total_pnl,
                "total_return": total_return,
                "max_drawdown": self.max_drawdown * 100,
                "current_equity": self.current_equity,
                "starting_capital": self.starting_capital,
                "peak_equity": self.peak_equity,
                "current_drawdown": ((self.peak_equity - self.current_equity) / self.peak_equity * 100) if self.peak_equity > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def shutdown(self):
        """Enhanced shutdown with cleanup"""
        try:
            # Stop automation
            if self.automation_running:
                self.stop_automation()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info("Enhanced trading engine shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


class EnhancedProductionTradingBot:
    """Enhanced Production Trading Bot with real-time features"""
    
    def __init__(self, config: dict = None):
        self.config = config or self._default_config()
        
        # Initialize enhanced trading engine
        self.engine = EnhancedTradingEngine(self.config)
        
        # Initialize GUI reference
        self.gui = None
        
        # API reference for backward compatibility
        self.api = self.engine
        
        logger.info("Enhanced Production Trading Bot initialized")
    
    def _default_config(self) -> dict:
        """Enhanced default configuration"""
        return {
            "account_address": "0x306D29F56EA1345c7E6F1ff27657ba05cEE15D4F",
            "secret_key": "43ba46de58067dd1ef3794c653bf3b11fa78866623cc515a5aff5f4be31fd3b8",
            "api_url": "https://api.hyperliquid.xyz",
            "trade_symbol": "BTC-USD-PERP",
            "trade_mode": "perp",
            "starting_capital": 100.0,
            "poll_interval_seconds": 2,
            "price_update_interval": 1,
            "min_trade_interval": 60,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "max_position_size": 0.1,
            "max_positions": 3,
            "circuit_breaker_threshold": 0.1,
            "max_drawdown_threshold": 0.15,
            "use_manual_entry_size": True,
            "manual_entry_size": 20.0,
            "use_trailing_stop": True,
            "trail_start_profit": 0.005,
            "trail_offset": 0.0025
        }
    
    def initialize(self):
        """Initialize the enhanced trading bot"""
        return self.engine.initialize()
    
    def connect(self):
        """Connect to trading API"""
        return self.engine.connect()
    
    def disconnect(self):
        """Disconnect from trading API"""
        return self.engine.disconnect()
    
    def get_real_time_data(self):
        """Get real-time trading data"""
        return self.engine.get_real_time_data()
    
    def start_automation(self, mode: str, strategy: str):
        """Start automated trading"""
        self.engine.start_automation(mode, strategy)
    
    def stop_automation(self):
        """Stop automated trading"""
        self.engine.stop_automation()
    
    def execute_manual_trade(self, side: str, size: float, symbol: str = None):
        """Execute manual trade"""
        return self.engine.execute_manual_trade(side, size, symbol)
    
    def close_all_positions(self):
        """Close all positions"""
        return self.engine.close_all_positions()
    
    def get_performance_metrics(self):
        """Get performance metrics"""
        return self.engine.get_performance_metrics()
    
    def shutdown(self):
        """Shutdown the bot"""
        self.engine.shutdown()

