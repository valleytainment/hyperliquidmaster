#!/usr/bin/env python3
"""
ULTIMATE COMPREHENSIVE TRADING ENGINE
===================================
Complete trading engine with ALL features from provided files:
• Real-time wallet equity monitoring and display
• Live token price feeds with technical indicators
• Advanced order execution with all order types
• 24/7 automation with multiple strategies
• Risk management and circuit breakers
• Performance analytics and reporting
• Auto-connection with default wallet credentials
"""

import asyncio
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import requests
import websocket
import logging

# Import our core modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enhanced_api import EnhancedHyperliquidAPI
from strategies.enhanced_neural_strategy import EnhancedNeuralStrategy
from strategies.bb_rsi_adx import BBRSIADXStrategy
from strategies.hull_suite import HullSuiteStrategy
from utils.logger import get_logger, TradingLogger
from utils.config_manager import ConfigManager
from utils.security import SecurityManager

logger = get_logger(__name__)
trading_logger = TradingLogger(__name__)


@dataclass
class RealTimeData:
    """Real-time data structure"""
    timestamp: datetime
    equity: float
    available_balance: float
    margin_used: float
    total_pnl: float
    daily_pnl: float
    positions: List[Dict]
    orders: List[Dict]
    price_data: Dict[str, float]
    performance_metrics: Dict[str, float]


@dataclass
class TradingConfig:
    """Trading configuration"""
    starting_capital: float = 100.0
    manual_entry_size: float = 20.0
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    circuit_breaker_threshold: float = 0.10
    max_daily_loss: float = 0.05
    max_drawdown: float = 0.15
    use_trailing_stop: bool = True
    use_partial_profits: bool = False
    use_dynamic_sizing: bool = False
    use_multi_timeframe: bool = True
    trading_mode: str = "perpetual"
    strategy: str = "enhanced_neural"


class UltimateComprehensiveTradingEngine:
    """Ultimate comprehensive trading engine with all features"""
    
    def __init__(self):
        """Initialize the ultimate trading engine"""
        self.config = TradingConfig()
        self.config_manager = ConfigManager()
        self.security_manager = SecurityManager()
        
        # Default credentials for auto-connection
        self.default_credentials = {
            "account_address": "0x306D29F56EA1345c7E6F1ff27657ba05cEE15D4F",
            "private_key": "43ba46de58067dd1ef3794c653bf3b11fa78866623cc515a5aff5f4be31fd3b8"
        }
        
        # API and connection
        self.api = None
        self.is_connected = False
        self.is_initialized = False
        
        # Trading state
        self.automation_running = False
        self.positions = {}
        self.orders = {}
        self.trade_history = []
        self.equity_history = []
        self.price_history = {}
        
        # Performance tracking
        self.performance_metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "daily_pnl": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "calmar_ratio": 0.0
        }
        
        # Strategies
        self.strategies = {}
        self.current_strategy = None
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.stop_flag = threading.Event()
        
        # Real-time data
        self.real_time_data = None
        self.last_equity_update = 0
        self.last_price_update = 0
        
        # Circuit breaker
        self.circuit_breaker_triggered = False
        self.daily_start_equity = 0
        
        logger.info("🚀 Ultimate Comprehensive Trading Engine initialized")
    
    def initialize(self) -> bool:
        """Initialize the trading engine with default credentials"""
        try:
            logger.info("Initializing trading engine with default credentials...")
            
            # Initialize API with default credentials
            self.api = EnhancedHyperliquidAPI(testnet=False)
            
            # Authenticate with default credentials
            if not self.api.authenticate(
                private_key=self.default_credentials["private_key"],
                wallet_address=self.default_credentials["account_address"]
            ):
                logger.error("Failed to authenticate with default credentials")
                return False
            
            # Initialize strategies
            self._initialize_strategies()
            
            # Load configuration
            self._load_configuration()
            
            self.is_initialized = True
            logger.info("✅ Trading engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize trading engine: {e}")
            return False
    
    def connect(self) -> bool:
        """Connect to the API and start real-time data feeds"""
        try:
            if not self.is_initialized:
                logger.error("Trading engine not initialized")
                return False
            
            logger.info("Connecting to Hyperliquid API...")
            
            # Test connection
            account_info = asyncio.run(self.api.get_account_info())
            
            if not account_info:
                logger.error("Failed to get account info")
                return False
            
            self.is_connected = True
            
            # Initialize real-time data
            self._initialize_real_time_data(account_info)
            
            # Start background processes
            self._start_background_processes()
            
            logger.info(f"✅ Connected to Hyperliquid (Mainnet)")
            logger.info(f"📊 Account: {self.default_credentials['account_address']}")
            logger.info(f"💰 Initial Equity: ${account_info.get('equity', 0):.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect: {e}")
            self.is_connected = False
            return False
    
    def _initialize_strategies(self):
        """Initialize all trading strategies"""
        try:
            # Enhanced Neural Strategy
            self.strategies["enhanced_neural"] = EnhancedNeuralStrategy()
            
            # BB RSI ADX Strategy
            self.strategies["bb_rsi_adx"] = BBRSIADXStrategy()
            
            # Hull Suite Strategy
            self.strategies["hull_suite"] = HullSuiteStrategy()
            
            # Set default strategy
            self.current_strategy = self.strategies["enhanced_neural"]
            
            logger.info(f"✅ Initialized {len(self.strategies)} strategies")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize strategies: {e}")
    
    def _load_configuration(self):
        """Load configuration from config manager"""
        try:
            config_data = self.config_manager.get_config()
            
            # Load trading configuration
            trading_config = config_data.get("trading", {})
            automation_config = config_data.get("automation", {})
            
            # Update config with loaded values
            if trading_config:
                for key, value in trading_config.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
            
            if automation_config:
                for key, value in automation_config.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
            
            logger.info("✅ Configuration loaded")
            
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
    
    def _initialize_real_time_data(self, account_info: Dict):
        """Initialize real-time data structure"""
        try:
            equity = account_info.get("equity", 0)
            available_balance = account_info.get("available_balance", 0)
            margin_used = account_info.get("margin_used", 0)
            
            self.real_time_data = RealTimeData(
                timestamp=datetime.now(),
                equity=equity,
                available_balance=available_balance,
                margin_used=margin_used,
                total_pnl=0.0,
                daily_pnl=0.0,
                positions=[],
                orders=[],
                price_data={},
                performance_metrics=self.performance_metrics.copy()
            )
            
            # Set daily start equity for circuit breaker
            self.daily_start_equity = equity
            
            # Initialize equity history
            self.equity_history.append({
                "timestamp": datetime.now(),
                "equity": equity
            })
            
            logger.info(f"✅ Real-time data initialized - Equity: ${equity:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize real-time data: {e}")
    
    def _start_background_processes(self):
        """Start background processes for real-time updates"""
        try:
            # Start real-time data update loop
            self.executor.submit(self._real_time_update_loop)
            
            # Start price feed updates
            self.executor.submit(self._price_feed_loop)
            
            # Start performance calculation loop
            self.executor.submit(self._performance_calculation_loop)
            
            logger.info("✅ Background processes started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start background processes: {e}")
    
    def _real_time_update_loop(self):
        """Real-time data update loop"""
        while not self.stop_flag.is_set() and self.is_connected:
            try:
                # Get account info
                account_info = asyncio.run(self.api.get_account_info())
                
                if account_info:
                    # Update real-time data
                    self.real_time_data.timestamp = datetime.now()
                    self.real_time_data.equity = account_info.get("equity", 0)
                    self.real_time_data.available_balance = account_info.get("available_balance", 0)
                    self.real_time_data.margin_used = account_info.get("margin_used", 0)
                    
                    # Update equity history
                    self.equity_history.append({
                        "timestamp": datetime.now(),
                        "equity": self.real_time_data.equity
                    })
                    
                    # Limit history size
                    if len(self.equity_history) > 1000:
                        self.equity_history = self.equity_history[-500:]
                    
                    # Get positions
                    positions = asyncio.run(self.api.get_positions())
                    if positions:
                        self.real_time_data.positions = positions
                        self.positions = {pos.get("symbol", ""): pos for pos in positions}
                    
                    # Get orders
                    orders = asyncio.run(self.api.get_open_orders())
                    if orders:
                        self.real_time_data.orders = orders
                        self.orders = {order.get("id", ""): order for order in orders}
                    
                    # Calculate P&L
                    self._calculate_pnl()
                    
                    # Check circuit breaker
                    self._check_circuit_breaker()
                    
                    self.last_equity_update = time.time()
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Real-time update error: {e}")
                time.sleep(5)
    
    def _price_feed_loop(self):
        """Price feed update loop"""
        symbols = ["BTC-USD-PERP", "ETH-USD-PERP", "SOL-USD-PERP", "AVAX-USD-PERP"]
        
        while not self.stop_flag.is_set() and self.is_connected:
            try:
                for symbol in symbols:
                    # Get current price
                    price_data = asyncio.run(self.api.get_current_price(symbol))
                    
                    if price_data:
                        price = price_data.get("price", 0)
                        
                        # Update price data
                        self.real_time_data.price_data[symbol] = price
                        
                        # Update price history
                        if symbol not in self.price_history:
                            self.price_history[symbol] = []
                        
                        self.price_history[symbol].append({
                            "timestamp": datetime.now(),
                            "price": price
                        })
                        
                        # Limit history size
                        if len(self.price_history[symbol]) > 1000:
                            self.price_history[symbol] = self.price_history[symbol][-500:]
                
                self.last_price_update = time.time()
                time.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                logger.error(f"Price feed error: {e}")
                time.sleep(10)
    
    def _performance_calculation_loop(self):
        """Performance metrics calculation loop"""
        while not self.stop_flag.is_set() and self.is_connected:
            try:
                self._calculate_performance_metrics()
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Performance calculation error: {e}")
                time.sleep(30)
    
    def _calculate_pnl(self):
        """Calculate P&L from positions"""
        try:
            total_pnl = 0.0
            
            for position in self.real_time_data.positions:
                pnl = position.get("unrealized_pnl", 0)
                total_pnl += pnl
            
            self.real_time_data.total_pnl = total_pnl
            
            # Calculate daily P&L
            if self.daily_start_equity > 0:
                self.real_time_data.daily_pnl = self.real_time_data.equity - self.daily_start_equity
            
        except Exception as e:
            logger.error(f"P&L calculation error: {e}")
    
    def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        try:
            if len(self.trade_history) == 0:
                return
            
            # Calculate win rate
            winning_trades = sum(1 for trade in self.trade_history if trade.get("pnl", 0) > 0)
            total_trades = len(self.trade_history)
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # Calculate profit factor
            gross_profit = sum(trade.get("pnl", 0) for trade in self.trade_history if trade.get("pnl", 0) > 0)
            gross_loss = abs(sum(trade.get("pnl", 0) for trade in self.trade_history if trade.get("pnl", 0) < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Calculate max drawdown
            if len(self.equity_history) > 1:
                equities = [e["equity"] for e in self.equity_history]
                peak = equities[0]
                max_drawdown = 0
                
                for equity in equities:
                    if equity > peak:
                        peak = equity
                    drawdown = (peak - equity) / peak if peak > 0 else 0
                    max_drawdown = max(max_drawdown, drawdown)
                
                self.performance_metrics["max_drawdown"] = max_drawdown * 100
            
            # Update performance metrics
            self.performance_metrics.update({
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": total_trades - winning_trades,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "total_pnl": self.real_time_data.total_pnl
            })
            
            # Update real-time data
            self.real_time_data.performance_metrics = self.performance_metrics.copy()
            
        except Exception as e:
            logger.error(f"Performance metrics calculation error: {e}")
    
    def _check_circuit_breaker(self):
        """Check circuit breaker conditions"""
        try:
            if self.daily_start_equity <= 0:
                return
            
            # Check daily loss threshold
            daily_loss_pct = abs(self.real_time_data.daily_pnl) / self.daily_start_equity
            
            if daily_loss_pct >= self.config.circuit_breaker_threshold:
                if not self.circuit_breaker_triggered:
                    self.circuit_breaker_triggered = True
                    logger.warning(f"🚨 CIRCUIT BREAKER TRIGGERED - Daily loss: {daily_loss_pct*100:.2f}%")
                    
                    # Stop automation if running
                    if self.automation_running:
                        self.stop_automation()
                    
                    # Close all positions
                    asyncio.run(self._emergency_close_all_positions())
            
        except Exception as e:
            logger.error(f"Circuit breaker check error: {e}")
    
    async def _emergency_close_all_positions(self):
        """Emergency close all positions"""
        try:
            logger.warning("🚨 Emergency closing all positions...")
            
            for position in self.real_time_data.positions:
                symbol = position.get("symbol", "")
                size = abs(position.get("size", 0))
                side = "sell" if position.get("side", "") == "long" else "buy"
                
                if size > 0:
                    # Place market order to close position
                    await self.api.place_order(
                        symbol=symbol,
                        side=side,
                        size=size,
                        order_type="market"
                    )
            
            logger.warning("🚨 Emergency position closure completed")
            
        except Exception as e:
            logger.error(f"Emergency position closure error: {e}")
    
    def start_automation(self, mode: str = "perpetual", strategy: str = "enhanced_neural") -> bool:
        """Start 24/7 automation"""
        try:
            if not self.is_connected:
                logger.error("Not connected to API")
                return False
            
            if self.automation_running:
                logger.warning("Automation already running")
                return False
            
            if self.circuit_breaker_triggered:
                logger.error("Circuit breaker triggered - cannot start automation")
                return False
            
            # Set trading mode and strategy
            self.config.trading_mode = mode.lower()
            self.config.strategy = strategy.lower()
            
            # Set current strategy
            if strategy.lower() in self.strategies:
                self.current_strategy = self.strategies[strategy.lower()]
            else:
                logger.error(f"Strategy '{strategy}' not found")
                return False
            
            # Start automation loop
            self.automation_running = True
            self.executor.submit(self._automation_loop)
            
            logger.info(f"🚀 Automation started - Mode: {mode}, Strategy: {strategy}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start automation: {e}")
            return False
    
    def stop_automation(self) -> bool:
        """Stop automation"""
        try:
            self.automation_running = False
            logger.info("⏹️ Automation stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop automation: {e}")
            return False
    
    def _automation_loop(self):
        """Main automation loop"""
        logger.info("🤖 Automation loop started")
        
        while self.automation_running and not self.stop_flag.is_set():
            try:
                if self.circuit_breaker_triggered:
                    logger.warning("Circuit breaker triggered - stopping automation")
                    break
                
                # Get market data for strategy
                market_data = self._get_market_data_for_strategy()
                
                if market_data and self.current_strategy:
                    # Generate trading signal
                    signal = self.current_strategy.generate_signal(market_data)
                    
                    if signal and signal.signal_type.value != "hold":
                        # Execute trade based on signal
                        asyncio.run(self._execute_automated_trade(signal))
                
                # Sleep between iterations
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Automation loop error: {e}")
                time.sleep(60)
        
        logger.info("🤖 Automation loop ended")
    
    def _get_market_data_for_strategy(self) -> Optional[Dict]:
        """Get market data for strategy analysis"""
        try:
            # Get primary trading symbol
            symbol = "BTC-USD-PERP"  # Default symbol
            
            # Get historical data
            historical_data = asyncio.run(self.api.get_historical_data(symbol, "1h", 100))
            
            if not historical_data:
                return None
            
            # Convert to format expected by strategy
            market_data = {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "price": self.real_time_data.price_data.get(symbol, 0),
                "historical_data": historical_data,
                "volume": 0,  # Would be populated from real data
                "high_24h": 0,
                "low_24h": 0,
                "change_24h": 0
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return None
    
    async def _execute_automated_trade(self, signal):
        """Execute automated trade based on signal"""
        try:
            symbol = signal.symbol
            side = signal.signal_type.value  # "buy" or "sell"
            size = self.config.manual_entry_size
            
            # Check available balance
            if self.real_time_data.available_balance < size:
                logger.warning(f"Insufficient balance for trade: ${size:.2f}")
                return
            
            # Place order
            result = await self.api.place_order(
                symbol=symbol,
                side=side,
                size=size,
                order_type="market"
            )
            
            if result.get("success", False):
                # Record trade
                trade_record = {
                    "timestamp": datetime.now(),
                    "symbol": symbol,
                    "side": side,
                    "size": size,
                    "price": signal.price,
                    "signal_strength": signal.confidence,
                    "strategy": self.config.strategy,
                    "automated": True
                }
                
                self.trade_history.append(trade_record)
                
                logger.info(f"✅ Automated trade executed: {side.upper()} {symbol} ${size:.2f}")
                
                # Set stop loss and take profit if enabled
                if self.config.use_trailing_stop:
                    await self._set_stop_loss_take_profit(symbol, side, signal.price)
            else:
                logger.error(f"❌ Automated trade failed: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"Automated trade execution error: {e}")
    
    async def _set_stop_loss_take_profit(self, symbol: str, side: str, entry_price: float):
        """Set stop loss and take profit orders"""
        try:
            if side == "buy":
                # Long position
                stop_price = entry_price * (1 - self.config.stop_loss_pct)
                take_profit_price = entry_price * (1 + self.config.take_profit_pct)
                stop_side = "sell"
            else:
                # Short position
                stop_price = entry_price * (1 + self.config.stop_loss_pct)
                take_profit_price = entry_price * (1 - self.config.take_profit_pct)
                stop_side = "buy"
            
            # Place stop loss order
            await self.api.place_order(
                symbol=symbol,
                side=stop_side,
                size=self.config.manual_entry_size,
                order_type="stop_loss",
                stop_price=stop_price
            )
            
            # Place take profit order
            await self.api.place_order(
                symbol=symbol,
                side=stop_side,
                size=self.config.manual_entry_size,
                order_type="take_profit",
                limit_price=take_profit_price
            )
            
            logger.info(f"✅ Stop loss and take profit set for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to set stop loss/take profit: {e}")
    
    def execute_manual_trade(self, side: str, size: float, symbol: str = "BTC-USD-PERP") -> Dict:
        """Execute manual trade"""
        try:
            if not self.is_connected:
                return {"success": False, "error": "Not connected to API"}
            
            # Check available balance
            if self.real_time_data.available_balance < size:
                return {"success": False, "error": f"Insufficient balance: ${self.real_time_data.available_balance:.2f}"}
            
            # Execute trade
            result = asyncio.run(self.api.place_order(
                symbol=symbol,
                side=side.lower(),
                size=size,
                order_type="market"
            ))
            
            if result.get("success", False):
                # Record trade
                trade_record = {
                    "timestamp": datetime.now(),
                    "symbol": symbol,
                    "side": side.lower(),
                    "size": size,
                    "price": self.real_time_data.price_data.get(symbol, 0),
                    "strategy": "manual",
                    "automated": False
                }
                
                self.trade_history.append(trade_record)
                
                logger.info(f"✅ Manual trade executed: {side.upper()} {symbol} ${size:.2f}")
                return {"success": True, "trade_id": result.get("order_id", "")}
            else:
                error_msg = result.get("error", "Unknown error")
                logger.error(f"❌ Manual trade failed: {error_msg}")
                return {"success": False, "error": error_msg}
            
        except Exception as e:
            logger.error(f"Manual trade execution error: {e}")
            return {"success": False, "error": str(e)}
    
    def close_all_positions(self) -> Dict:
        """Close all open positions"""
        try:
            if not self.is_connected:
                return {"success": False, "error": "Not connected to API"}
            
            positions_closed = 0
            total_pnl = 0.0
            
            for position in self.real_time_data.positions:
                symbol = position.get("symbol", "")
                size = abs(position.get("size", 0))
                side = "sell" if position.get("side", "") == "long" else "buy"
                pnl = position.get("unrealized_pnl", 0)
                
                if size > 0:
                    # Place market order to close position
                    result = asyncio.run(self.api.place_order(
                        symbol=symbol,
                        side=side,
                        size=size,
                        order_type="market"
                    ))
                    
                    if result.get("success", False):
                        positions_closed += 1
                        total_pnl += pnl
                        
                        # Record trade
                        trade_record = {
                            "timestamp": datetime.now(),
                            "symbol": symbol,
                            "side": side,
                            "size": size,
                            "price": self.real_time_data.price_data.get(symbol, 0),
                            "pnl": pnl,
                            "strategy": "manual_close",
                            "automated": False
                        }
                        
                        self.trade_history.append(trade_record)
            
            logger.info(f"✅ Closed {positions_closed} positions, Total P&L: ${total_pnl:.2f}")
            
            return {
                "success": True,
                "positions_closed": positions_closed,
                "total_pnl": total_pnl
            }
            
        except Exception as e:
            logger.error(f"Close all positions error: {e}")
            return {"success": False, "error": str(e)}
    
    def get_real_time_data(self) -> Optional[Dict]:
        """Get current real-time data"""
        try:
            if not self.real_time_data:
                return None
            
            # Convert to dictionary
            data = asdict(self.real_time_data)
            
            # Add additional computed fields
            data["equity_change_24h"] = self.real_time_data.daily_pnl
            data["equity_change_pct"] = (self.real_time_data.daily_pnl / self.daily_start_equity * 100) if self.daily_start_equity > 0 else 0
            data["margin_usage_pct"] = (self.real_time_data.margin_used / self.real_time_data.equity * 100) if self.real_time_data.equity > 0 else 0
            data["automation_running"] = self.automation_running
            data["circuit_breaker_triggered"] = self.circuit_breaker_triggered
            data["last_equity_update"] = self.last_equity_update
            data["last_price_update"] = self.last_price_update
            
            # Add current price for main symbol
            data["price"] = self.real_time_data.price_data.get("BTC-USD-PERP", 0)
            
            return data
            
        except Exception as e:
            logger.error(f"Get real-time data error: {e}")
            return None
    
    def get_equity_history(self) -> List[Dict]:
        """Get equity history"""
        return self.equity_history.copy()
    
    def get_price_history(self, symbol: str) -> List[Dict]:
        """Get price history for symbol"""
        return self.price_history.get(symbol, []).copy()
    
    def get_trade_history(self) -> List[Dict]:
        """Get trade history"""
        return self.trade_history.copy()
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        return self.performance_metrics.copy()
    
    def update_config(self, new_config: Dict):
        """Update trading configuration"""
        try:
            for key, value in new_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            # Save to config manager
            self.config_manager.update_config({"trading": asdict(self.config)})
            
            logger.info("✅ Configuration updated")
            
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker"""
        self.circuit_breaker_triggered = False
        self.daily_start_equity = self.real_time_data.equity if self.real_time_data else 0
        logger.info("🔄 Circuit breaker reset")
    
    def get_available_symbols(self) -> List[str]:
        """Get available trading symbols"""
        return ["BTC-USD-PERP", "ETH-USD-PERP", "SOL-USD-PERP", "AVAX-USD-PERP", 
                "MATIC-USD-PERP", "LINK-USD-PERP", "UNI-USD-PERP", "AAVE-USD-PERP"]
    
    def shutdown(self):
        """Shutdown the trading engine"""
        try:
            logger.info("🔄 Shutting down trading engine...")
            
            # Stop automation
            if self.automation_running:
                self.stop_automation()
            
            # Set stop flag
            self.stop_flag.set()
            
            # Shutdown executor
            self.executor.shutdown(wait=False)
            
            # Disconnect API
            self.is_connected = False
            
            logger.info("✅ Trading engine shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")


# Convenience class for backward compatibility
class EnhancedProductionTradingBot(UltimateComprehensiveTradingEngine):
    """Alias for backward compatibility"""
    pass


def main():
    """Main entry point for testing"""
    try:
        # Initialize trading engine
        engine = UltimateComprehensiveTradingEngine()
        
        # Initialize and connect
        if engine.initialize():
            if engine.connect():
                print("✅ Trading engine connected successfully!")
                
                # Get real-time data
                data = engine.get_real_time_data()
                if data:
                    print(f"💰 Current Equity: ${data['equity']:.2f}")
                    print(f"📊 Available Balance: ${data['available_balance']:.2f}")
                    print(f"📈 Total P&L: ${data['total_pnl']:.2f}")
                
                # Test manual trade (commented out for safety)
                # result = engine.execute_manual_trade("buy", 20.0, "BTC-USD-PERP")
                # print(f"Trade result: {result}")
                
                # Keep running for a bit to test real-time updates
                time.sleep(10)
                
                # Shutdown
                engine.shutdown()
            else:
                print("❌ Failed to connect")
        else:
            print("❌ Failed to initialize")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()

