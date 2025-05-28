#!/usr/bin/env python3
"""
Enhanced Hyperliquid Trading Bot GUI

A professional-grade trading interface with advanced features:
- Tabbed interface for all major sections
- Real-time charts with technical indicators
- Order book visualization with depth chart
- Position tracking and trade history
- Comprehensive settings and configuration
- Proper error handling and validation
- Light/dark theme support
- Responsive design with scrollbars

This implementation follows MVVM architecture pattern with clear separation of concerns.
"""

import os
import sys
import json
import time
import logging
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

# Import core components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.api_rate_limiter_enhanced import APIRateLimiter
from core.error_handling_fixed_updated import ErrorHandler
from core.enhanced_mock_data_provider import EnhancedMockDataProvider
from strategies.robust_signal_generator_fixed_updated import RobustSignalGenerator
from strategies.master_omni_overlord_robust_standardized import MasterOmniOverlordRobustStrategy
from strategies.advanced_technical_indicators_fixed import AdvancedTechnicalIndicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("gui_main_enhanced.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
CONFIG_FILE = os.path.join(CONFIG_DIR, "gui_config.json")
VERSION = "3.0.0"

#############################################################################
# Models - Data structures and business logic
#############################################################################

class MarketData:
    """Model for market data"""
    
    def __init__(self):
        self.symbols = {}
        self.current_symbol = ""
        self.current_timeframe = ""
        self.ohlcv_data = {}
        self.orderbook = {}
        self.trades = []
    
    def update_ohlcv(self, symbol: str, timeframe: str, data: pd.DataFrame) -> None:
        """Update OHLCV data for a symbol and timeframe"""
        if symbol not in self.ohlcv_data:
            self.ohlcv_data[symbol] = {}
        self.ohlcv_data[symbol][timeframe] = data
    
    def update_orderbook(self, symbol: str, orderbook: Dict) -> None:
        """Update orderbook for a symbol"""
        self.orderbook[symbol] = orderbook
    
    def update_trades(self, symbol: str, trades: List) -> None:
        """Update recent trades for a symbol"""
        self.trades = trades
    
    def get_ohlcv(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get OHLCV data for a symbol and timeframe"""
        if symbol in self.ohlcv_data and timeframe in self.ohlcv_data[symbol]:
            return self.ohlcv_data[symbol][timeframe]
        return None
    
    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        """Get orderbook for a symbol"""
        return self.orderbook.get(symbol)
    
    def get_trades(self) -> List:
        """Get recent trades"""
        return self.trades


class PositionData:
    """Model for position data"""
    
    def __init__(self):
        self.positions = {}
        self.orders = []
        self.trade_history = []
        self.account_balance = 0.0
    
    def update_positions(self, positions: Dict) -> None:
        """Update positions"""
        self.positions = positions
    
    def update_orders(self, orders: List) -> None:
        """Update open orders"""
        self.orders = orders
    
    def update_trade_history(self, trades: List) -> None:
        """Update trade history"""
        self.trade_history = trades
    
    def update_account_balance(self, balance: float) -> None:
        """Update account balance"""
        self.account_balance = balance
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for a symbol"""
        return self.positions.get(symbol)
    
    def get_orders(self) -> List:
        """Get open orders"""
        return self.orders
    
    def get_trade_history(self) -> List:
        """Get trade history"""
        return self.trade_history
    
    def get_account_balance(self) -> float:
        """Get account balance"""
        return self.account_balance


class ConfigData:
    """Model for configuration data"""
    
    def __init__(self):
        self.config = self._load_default_config()
        self._load_from_file()
    
    def _load_default_config(self) -> Dict:
        """Load default configuration"""
        return {
            "theme": "dark",
            "symbols": ["BTC", "ETH", "XRP", "SOL", "DOGE"],
            "default_symbol": "XRP",
            "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
            "default_timeframe": "1h",
            "risk_level": 0.02,
            "use_mock_data": False,
            "api_key": "",
            "api_secret": "",
            "chart_indicators": ["MA", "RSI", "MACD", "BB"],
            "order_types": ["Market", "Limit", "Stop", "Stop Limit"],
            "default_order_type": "Limit",
            "default_quantity": 1.0,
            "auto_refresh_interval": 5000,
            "show_trade_confirmations": True,
            "advanced_mode": False
        }
    
    def _load_from_file(self) -> None:
        """Load configuration from file"""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, "r") as f:
                    loaded_config = json.load(f)
                    # Update config with loaded values, keeping defaults for missing keys
                    for key, value in loaded_config.items():
                        self.config[key] = value
                logger.info("Configuration loaded from file")
            else:
                # Create config directory if it doesn't exist
                os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
                # Save default config
                self.save()
                logger.info("Default configuration created")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
    
    def save(self) -> None:
        """Save configuration to file"""
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=4)
            logger.info("Configuration saved to file")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self.config[key] = value
    
    def get_all(self) -> Dict:
        """Get all configuration values"""
        return self.config


#############################################################################
# ViewModels - Business logic that connects Models and Views
#############################################################################

class MarketViewModel:
    """ViewModel for market data"""
    
    def __init__(self, market_data: MarketData, config_data: ConfigData, error_handler: ErrorHandler):
        self.market_data = market_data
        self.config_data = config_data
        self.error_handler = error_handler
        self.api_rate_limiter = APIRateLimiter()
        self.mock_data_provider = EnhancedMockDataProvider()
        self.technical_indicators = AdvancedTechnicalIndicators()
        self.using_mock_data = config_data.get("use_mock_data", False)
        self.data_callbacks = []
    
    def add_data_callback(self, callback: Callable) -> None:
        """Add callback for data updates"""
        if callback not in self.data_callbacks:
            self.data_callbacks.append(callback)
    
    def remove_data_callback(self, callback: Callable) -> None:
        """Remove callback for data updates"""
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)
    
    def _notify_data_callbacks(self) -> None:
        """Notify all callbacks of data updates"""
        for callback in self.data_callbacks:
            try:
                callback()
            except Exception as e:
                self.error_handler.handle_error(f"Error in data callback: {str(e)}")
    
    def fetch_market_data(self, symbol: str, timeframe: str) -> None:
        """Fetch market data for a symbol and timeframe"""
        try:
            # Check if we should use mock data
            if self.using_mock_data or self.api_rate_limiter.is_rate_limited():
                if not self.using_mock_data:
                    logger.warning("API rate limited. Using mock data.")
                    self.using_mock_data = True
                
                # Get mock data
                ohlcv = self.mock_data_provider.get_klines(symbol, timeframe)
                orderbook = self.mock_data_provider.get_orderbook(symbol)
                trades = self.mock_data_provider.get_recent_trades(symbol)
                
                # Update market data
                self.market_data.update_ohlcv(symbol, timeframe, ohlcv)
                self.market_data.update_orderbook(symbol, orderbook)
                self.market_data.update_trades(symbol, trades)
                
                # Calculate indicators
                self._calculate_indicators(symbol, timeframe)
                
                # Notify callbacks
                self._notify_data_callbacks()
            else:
                # TODO: Implement real API calls
                # For now, use mock data
                ohlcv = self.mock_data_provider.get_klines(symbol, timeframe)
                orderbook = self.mock_data_provider.get_orderbook(symbol)
                trades = self.mock_data_provider.get_recent_trades(symbol)
                
                # Update market data
                self.market_data.update_ohlcv(symbol, timeframe, ohlcv)
                self.market_data.update_orderbook(symbol, orderbook)
                self.market_data.update_trades(symbol, trades)
                
                # Calculate indicators
                self._calculate_indicators(symbol, timeframe)
                
                # Notify callbacks
                self._notify_data_callbacks()
        except Exception as e:
            self.error_handler.handle_error(f"Error fetching market data: {str(e)}")
    
    def _calculate_indicators(self, symbol: str, timeframe: str) -> None:
        """Calculate technical indicators for a symbol and timeframe"""
        try:
            ohlcv = self.market_data.get_ohlcv(symbol, timeframe)
            if ohlcv is not None and not ohlcv.empty:
                # Calculate indicators
                ohlcv['ma_fast'] = self.technical_indicators.calculate_sma(ohlcv['close'], 20)
                ohlcv['ma_slow'] = self.technical_indicators.calculate_sma(ohlcv['close'], 50)
                ohlcv['rsi'] = self.technical_indicators.calculate_rsi(ohlcv['close'], 14)
                
                macd, signal, hist = self.technical_indicators.calculate_macd(ohlcv['close'])
                ohlcv['macd'] = macd
                ohlcv['macd_signal'] = signal
                ohlcv['macd_hist'] = hist
                
                upper, middle, lower = self.technical_indicators.calculate_bollinger_bands(ohlcv['close'], 20, 2)
                ohlcv['bb_upper'] = upper
                ohlcv['bb_middle'] = middle
                ohlcv['bb_lower'] = lower
                
                # Update market data
                self.market_data.update_ohlcv(symbol, timeframe, ohlcv)
        except Exception as e:
            self.error_handler.handle_error(f"Error calculating indicators: {str(e)}")
    
    def get_ohlcv_with_indicators(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get OHLCV data with indicators for a symbol and timeframe"""
        return self.market_data.get_ohlcv(symbol, timeframe)
    
    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        """Get orderbook for a symbol"""
        return self.market_data.get_orderbook(symbol)
    
    def get_trades(self) -> List:
        """Get recent trades"""
        return self.market_data.get_trades()
    
    def toggle_mock_data(self, use_mock: bool) -> None:
        """Toggle mock data usage"""
        self.using_mock_data = use_mock
        self.config_data.set("use_mock_data", use_mock)
        self.config_data.save()


class TradingViewModel:
    """ViewModel for trading operations"""
    
    def __init__(self, position_data: PositionData, config_data: ConfigData, error_handler: ErrorHandler):
        self.position_data = position_data
        self.config_data = config_data
        self.error_handler = error_handler
        self.api_rate_limiter = APIRateLimiter()
        self.mock_data_provider = EnhancedMockDataProvider()
        self.using_mock_data = config_data.get("use_mock_data", False)
        self.trading_callbacks = []
    
    def add_trading_callback(self, callback: Callable) -> None:
        """Add callback for trading updates"""
        if callback not in self.trading_callbacks:
            self.trading_callbacks.append(callback)
    
    def remove_trading_callback(self, callback: Callable) -> None:
        """Remove callback for trading updates"""
        if callback in self.trading_callbacks:
            self.trading_callbacks.remove(callback)
    
    def _notify_trading_callbacks(self) -> None:
        """Notify all callbacks of trading updates"""
        for callback in self.trading_callbacks:
            try:
                callback()
            except Exception as e:
                self.error_handler.handle_error(f"Error in trading callback: {str(e)}")
    
    def fetch_account_data(self) -> None:
        """Fetch account data"""
        try:
            # Check if we should use mock data
            if self.using_mock_data or self.api_rate_limiter.is_rate_limited():
                if not self.using_mock_data:
                    logger.warning("API rate limited. Using mock data.")
                    self.using_mock_data = True
                
                # Get mock data
                positions = self.mock_data_provider.get_positions()
                orders = self.mock_data_provider.get_open_orders()
                trades = self.mock_data_provider.get_trade_history()
                balance = self.mock_data_provider.get_account_balance()
                
                # Update position data
                self.position_data.update_positions(positions)
                self.position_data.update_orders(orders)
                self.position_data.update_trade_history(trades)
                self.position_data.update_account_balance(balance)
                
                # Notify callbacks
                self._notify_trading_callbacks()
            else:
                # TODO: Implement real API calls
                # For now, use mock data
                positions = self.mock_data_provider.get_positions()
                orders = self.mock_data_provider.get_open_orders()
                trades = self.mock_data_provider.get_trade_history()
                balance = self.mock_data_provider.get_account_balance()
                
                # Update position data
                self.position_data.update_positions(positions)
                self.position_data.update_orders(orders)
                self.position_data.update_trade_history(trades)
                self.position_data.update_account_balance(balance)
                
                # Notify callbacks
                self._notify_trading_callbacks()
        except Exception as e:
            self.error_handler.handle_error(f"Error fetching account data: {str(e)}")
    
    def place_order(self, symbol: str, order_type: str, side: str, quantity: float, price: Optional[float] = None, 
                   stop_price: Optional[float] = None) -> bool:
        """Place an order"""
        try:
            # Validate inputs
            if not symbol:
                raise ValueError("Symbol is required")
            if not order_type:
                raise ValueError("Order type is required")
            if side not in ["buy", "sell"]:
                raise ValueError("Side must be 'buy' or 'sell'")
            if quantity <= 0:
                raise ValueError("Quantity must be greater than 0")
            if order_type in ["limit", "stop_limit"] and (price is None or price <= 0):
                raise ValueError("Price is required for limit orders")
            if order_type in ["stop", "stop_limit"] and (stop_price is None or stop_price <= 0):
                raise ValueError("Stop price is required for stop orders")
            
            # Check if we should use mock data
            if self.using_mock_data or self.api_rate_limiter.is_rate_limited():
                if not self.using_mock_data:
                    logger.warning("API rate limited. Using mock data.")
                    self.using_mock_data = True
                
                # Place mock order
                success = self.mock_data_provider.place_order(
                    symbol, order_type, side, quantity, price, stop_price
                )
                
                if success:
                    # Refresh account data
                    self.fetch_account_data()
                    return True
                else:
                    raise Exception("Failed to place mock order")
            else:
                # TODO: Implement real API calls
                # For now, use mock data
                success = self.mock_data_provider.place_order(
                    symbol, order_type, side, quantity, price, stop_price
                )
                
                if success:
                    # Refresh account data
                    self.fetch_account_data()
                    return True
                else:
                    raise Exception("Failed to place order")
        except Exception as e:
            self.error_handler.handle_error(f"Error placing order: {str(e)}")
            return False
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            # Check if we should use mock data
            if self.using_mock_data or self.api_rate_limiter.is_rate_limited():
                if not self.using_mock_data:
                    logger.warning("API rate limited. Using mock data.")
                    self.using_mock_data = True
                
                # Cancel mock order
                success = self.mock_data_provider.cancel_order(order_id)
                
                if success:
                    # Refresh account data
                    self.fetch_account_data()
                    return True
                else:
                    raise Exception("Failed to cancel mock order")
            else:
                # TODO: Implement real API calls
                # For now, use mock data
                success = self.mock_data_provider.cancel_order(order_id)
                
                if success:
                    # Refresh account data
                    self.fetch_account_data()
                    return True
                else:
                    raise Exception("Failed to cancel order")
        except Exception as e:
            self.error_handler.handle_error(f"Error canceling order: {str(e)}")
            return False
    
    def close_position(self, symbol: str) -> bool:
        """Close a position"""
        try:
            # Check if we should use mock data
            if self.using_mock_data or self.api_rate_limiter.is_rate_limited():
                if not self.using_mock_data:
                    logger.warning("API rate limited. Using mock data.")
                    self.using_mock_data = True
                
                # Close mock position
                success = self.mock_data_provider.close_position(symbol)
                
                if success:
                    # Refresh account data
                    self.fetch_account_data()
                    return True
                else:
                    raise Exception("Failed to close mock position")
            else:
                # TODO: Implement real API calls
                # For now, use mock data
                success = self.mock_data_provider.close_position(symbol)
                
                if success:
                    # Refresh account data
                    self.fetch_account_data()
                    return True
                else:
                    raise Exception("Failed to close position")
        except Exception as e:
            self.error_handler.handle_error(f"Error closing position: {str(e)}")
            return False
    
    def get_positions(self) -> Dict:
        """Get positions"""
        return self.position_data.positions
    
    def get_orders(self) -> List:
        """Get open orders"""
        return self.position_data.orders
    
    def get_trade_history(self) -> List:
        """Get trade history"""
        return self.position_data.trade_history
    
    def get_account_balance(self) -> float:
        """Get account balance"""
        return self.position_data.account_balance


#############################################################################
# Views - UI Components
#############################################################################

class ScrollableFrame(ttk.Frame):
    """A scrollable frame that can contain other widgets"""
    
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        
        # Create a canvas and scrollbar
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configure canvas
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack widgets
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Bind mouse wheel
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class StatusBar(ttk.Frame):
    """Status bar for displaying status messages"""
    
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        
        # Create status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self, textvariable=self.status_var)
        self.status_label.pack(side="left", padx=5)
        
        # Create API status label
        self.api_status_var = tk.StringVar(value="API: OK")
        self.api_status_label = ttk.Label(self, textvariable=self.api_status_var)
        self.api_status_label.pack(side="right", padx=5)
        
        # Create mock data indicator
        self.mock_data_var = tk.StringVar(value="")
        self.mock_data_label = ttk.Label(self, textvariable=self.mock_data_var)
        self.mock_data_label.pack(side="right", padx=5)
    
    def set_status(self, status: str) -> None:
        """Set status message"""
        self.status_var.set(status)
    
    def set_api_status(self, status: str, is_error: bool = False) -> None:
        """Set API status message"""
        self.api_status_var.set(f"API: {status}")
        if is_error:
            self.api_status_label.configure(foreground="red")
        else:
            self.api_status_label.configure(foreground="")
    
    def set_mock_data(self, using_mock: bool) -> None:
        """Set mock data indicator"""
        if using_mock:
            self.mock_data_var.set("Using Mock Data")
            self.mock_data_label.configure(foreground="orange")
        else:
            self.mock_data_var.set("")


class ChartView(ttk.Frame):
    """View for displaying price charts and indicators"""
    
    def __init__(self, parent, market_vm: MarketViewModel, config_data: ConfigData, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.market_vm = market_vm
        self.config_data = config_data
        
        # Add callback for data updates
        self.market_vm.add_data_callback(self.update_chart)
        
        # Create chart controls
        self._create_controls()
        
        # Create chart
        self._create_chart()
    
    def _create_controls(self) -> None:
        """Create chart controls"""
        controls_frame = ttk.Frame(self)
        controls_frame.pack(fill="x", padx=5, pady=5)
        
        # Symbol selection
        ttk.Label(controls_frame, text="Symbol:").pack(side="left", padx=(0, 5))
        
        self.symbol_var = tk.StringVar(value=self.config_data.get("default_symbol"))
        symbol_combo = ttk.Combobox(
            controls_frame, 
            textvariable=self.symbol_var, 
            values=self.config_data.get("symbols"),
            width=8
        )
        symbol_combo.pack(side="left", padx=(0, 10))
        symbol_combo.bind("<<ComboboxSelected>>", self._on_symbol_change)
        
        # Timeframe selection
        ttk.Label(controls_frame, text="Timeframe:").pack(side="left", padx=(0, 5))
        
        self.timeframe_var = tk.StringVar(value=self.config_data.get("default_timeframe"))
        timeframe_combo = ttk.Combobox(
            controls_frame, 
            textvariable=self.timeframe_var, 
            values=self.config_data.get("timeframes"),
            width=8
        )
        timeframe_combo.pack(side="left", padx=(0, 10))
        timeframe_combo.bind("<<ComboboxSelected>>", self._on_timeframe_change)
        
        # Indicator selection
        ttk.Label(controls_frame, text="Indicators:").pack(side="left", padx=(0, 5))
        
        self.ma_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls_frame, text="MA", variable=self.ma_var, command=self.update_chart).pack(side="left")
        
        self.rsi_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls_frame, text="RSI", variable=self.rsi_var, command=self.update_chart).pack(side="left")
        
        self.macd_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls_frame, text="MACD", variable=self.macd_var, command=self.update_chart).pack(side="left")
        
        self.bb_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls_frame, text="BB", variable=self.bb_var, command=self.update_chart).pack(side="left")
        
        # Refresh button
        refresh_button = ttk.Button(controls_frame, text="Refresh", command=self._refresh_data)
        refresh_button.pack(side="right")
    
    def _create_chart(self) -> None:
        """Create chart"""
        # Create chart frame
        chart_frame = ttk.Frame(self)
        chart_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create figure with subplots
        self.fig = Figure(figsize=(10, 8), dpi=100)
        
        # Price chart (60% height)
        self.price_ax = self.fig.add_subplot(5, 1, (1, 3))
        
        # Indicator charts (20% height each)
        self.ind1_ax = self.fig.add_subplot(5, 1, 4, sharex=self.price_ax)
        self.ind2_ax = self.fig.add_subplot(5, 1, 5, sharex=self.price_ax)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Adjust layout
        self.fig.tight_layout()
        
        # Initial data fetch
        self._refresh_data()
    
    def _on_symbol_change(self, event) -> None:
        """Handle symbol change"""
        self._refresh_data()
    
    def _on_timeframe_change(self, event) -> None:
        """Handle timeframe change"""
        self._refresh_data()
    
    def _refresh_data(self) -> None:
        """Refresh chart data"""
        symbol = self.symbol_var.get()
        timeframe = self.timeframe_var.get()
        self.market_vm.fetch_market_data(symbol, timeframe)
    
    def update_chart(self) -> None:
        """Update chart with latest data"""
        try:
            symbol = self.symbol_var.get()
            timeframe = self.timeframe_var.get()
            data = self.market_vm.get_ohlcv_with_indicators(symbol, timeframe)
            
            if data is None or data.empty:
                logger.warning(f"No data available for {symbol} {timeframe}")
                return
            
            # Clear axes
            self.price_ax.clear()
            self.ind1_ax.clear()
            self.ind2_ax.clear()
            
            # Plot price
            self.price_ax.set_title(f"{symbol} {timeframe}")
            self.price_ax.plot(data.index, data['close'], label='Close')
            
            # Plot indicators on price chart
            if self.ma_var.get() and 'ma_fast' in data.columns and 'ma_slow' in data.columns:
                self.price_ax.plot(data.index, data['ma_fast'], label='MA (20)', linestyle='--')
                self.price_ax.plot(data.index, data['ma_slow'], label='MA (50)', linestyle='--')
            
            if self.bb_var.get() and 'bb_upper' in data.columns and 'bb_middle' in data.columns and 'bb_lower' in data.columns:
                self.price_ax.plot(data.index, data['bb_upper'], label='BB Upper', linestyle=':')
                self.price_ax.plot(data.index, data['bb_middle'], label='BB Middle', linestyle=':')
                self.price_ax.plot(data.index, data['bb_lower'], label='BB Lower', linestyle=':')
            
            # Plot RSI
            if self.rsi_var.get() and 'rsi' in data.columns:
                self.ind1_ax.set_title('RSI (14)')
                self.ind1_ax.plot(data.index, data['rsi'], label='RSI')
                self.ind1_ax.axhline(y=70, color='r', linestyle='-')
                self.ind1_ax.axhline(y=30, color='g', linestyle='-')
                self.ind1_ax.set_ylim(0, 100)
            
            # Plot MACD
            if self.macd_var.get() and 'macd' in data.columns and 'macd_signal' in data.columns and 'macd_hist' in data.columns:
                self.ind2_ax.set_title('MACD')
                self.ind2_ax.plot(data.index, data['macd'], label='MACD')
                self.ind2_ax.plot(data.index, data['macd_signal'], label='Signal')
                self.ind2_ax.bar(data.index, data['macd_hist'], label='Histogram')
                self.ind2_ax.axhline(y=0, color='k', linestyle='-')
            
            # Add legends
            self.price_ax.legend(loc='upper left')
            self.ind1_ax.legend(loc='upper left')
            self.ind2_ax.legend(loc='upper left')
            
            # Format x-axis
            self.price_ax.tick_params(labelbottom=False)
            self.ind1_ax.tick_params(labelbottom=False)
            
            # Adjust layout
            self.fig.tight_layout()
            
            # Draw canvas
            self.canvas.draw()
        except Exception as e:
            logger.error(f"Error updating chart: {str(e)}")


class OrderBookView(ttk.Frame):
    """View for displaying order book and depth chart"""
    
    def __init__(self, parent, market_vm: MarketViewModel, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.market_vm = market_vm
        
        # Add callback for data updates
        self.market_vm.add_data_callback(self.update_orderbook)
        
        # Create order book display
        self._create_orderbook_display()
    
    def _create_orderbook_display(self) -> None:
        """Create order book display"""
        # Create main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create depth chart frame (top 40%)
        depth_frame = ttk.LabelFrame(main_frame, text="Depth Chart")
        depth_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create figure for depth chart
        self.depth_fig = Figure(figsize=(8, 3), dpi=100)
        self.depth_ax = self.depth_fig.add_subplot(111)
        
        # Create canvas for depth chart
        self.depth_canvas = FigureCanvasTkAgg(self.depth_fig, master=depth_frame)
        self.depth_canvas.draw()
        self.depth_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Create order book frame (bottom 60%)
        orderbook_frame = ttk.LabelFrame(main_frame, text="Order Book")
        orderbook_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create order book table frame
        table_frame = ttk.Frame(orderbook_frame)
        table_frame.pack(fill="both", expand=True)
        
        # Create asks table (left)
        asks_frame = ttk.Frame(table_frame)
        asks_frame.pack(side="left", fill="both", expand=True)
        
        ttk.Label(asks_frame, text="Asks", anchor="center").pack(fill="x")
        
        self.asks_tree = ttk.Treeview(
            asks_frame,
            columns=("price", "quantity", "total"),
            show="headings",
            height=10
        )
        
        self.asks_tree.heading("price", text="Price")
        self.asks_tree.heading("quantity", text="Quantity")
        self.asks_tree.heading("total", text="Total")
        
        self.asks_tree.column("price", width=100, anchor="e")
        self.asks_tree.column("quantity", width=100, anchor="e")
        self.asks_tree.column("total", width=100, anchor="e")
        
        asks_scrollbar = ttk.Scrollbar(asks_frame, orient="vertical", command=self.asks_tree.yview)
        self.asks_tree.configure(yscrollcommand=asks_scrollbar.set)
        
        asks_scrollbar.pack(side="right", fill="y")
        self.asks_tree.pack(side="left", fill="both", expand=True)
        
        # Create bids table (right)
        bids_frame = ttk.Frame(table_frame)
        bids_frame.pack(side="right", fill="both", expand=True)
        
        ttk.Label(bids_frame, text="Bids", anchor="center").pack(fill="x")
        
        self.bids_tree = ttk.Treeview(
            bids_frame,
            columns=("price", "quantity", "total"),
            show="headings",
            height=10
        )
        
        self.bids_tree.heading("price", text="Price")
        self.bids_tree.heading("quantity", text="Quantity")
        self.bids_tree.heading("total", text="Total")
        
        self.bids_tree.column("price", width=100, anchor="e")
        self.bids_tree.column("quantity", width=100, anchor="e")
        self.bids_tree.column("total", width=100, anchor="e")
        
        bids_scrollbar = ttk.Scrollbar(bids_frame, orient="vertical", command=self.bids_tree.yview)
        self.bids_tree.configure(yscrollcommand=bids_scrollbar.set)
        
        bids_scrollbar.pack(side="right", fill="y")
        self.bids_tree.pack(side="left", fill="both", expand=True)
    
    def update_orderbook(self) -> None:
        """Update order book display"""
        try:
            symbol = self.market_vm.config_data.get("default_symbol")
            orderbook = self.market_vm.get_orderbook(symbol)
            
            if orderbook is None:
                logger.warning(f"No orderbook data available for {symbol}")
                return
            
            # Clear trees
            for item in self.asks_tree.get_children():
                self.asks_tree.delete(item)
            
            for item in self.bids_tree.get_children():
                self.bids_tree.delete(item)
            
            # Update asks
            asks = orderbook.get("asks", [])
            asks_total = 0
            for i, (price, quantity) in enumerate(asks[:20]):  # Show top 20 asks
                asks_total += quantity
                self.asks_tree.insert("", "end", values=(f"{price:.2f}", f"{quantity:.4f}", f"{asks_total:.4f}"))
            
            # Update bids
            bids = orderbook.get("bids", [])
            bids_total = 0
            for i, (price, quantity) in enumerate(bids[:20]):  # Show top 20 bids
                bids_total += quantity
                self.bids_tree.insert("", "end", values=(f"{price:.2f}", f"{quantity:.4f}", f"{bids_total:.4f}"))
            
            # Update depth chart
            self._update_depth_chart(asks, bids)
        except Exception as e:
            logger.error(f"Error updating orderbook: {str(e)}")
    
    def _update_depth_chart(self, asks: List, bids: List) -> None:
        """Update depth chart"""
        try:
            # Clear axis
            self.depth_ax.clear()
            
            # Prepare data
            ask_prices = [price for price, _ in asks[:20]]
            ask_quantities = [sum(quantity for _, quantity in asks[:i+1]) for i in range(min(20, len(asks)))]
            
            bid_prices = [price for price, _ in bids[:20]]
            bid_quantities = [sum(quantity for _, quantity in bids[:i+1]) for i in range(min(20, len(bids)))]
            
            # Plot depth chart
            if ask_prices and ask_quantities:
                self.depth_ax.plot(ask_prices, ask_quantities, 'r-', label='Asks')
                self.depth_ax.fill_between(ask_prices, ask_quantities, alpha=0.3, color='r')
            
            if bid_prices and bid_quantities:
                self.depth_ax.plot(bid_prices, bid_quantities, 'g-', label='Bids')
                self.depth_ax.fill_between(bid_prices, bid_quantities, alpha=0.3, color='g')
            
            # Add labels and legend
            self.depth_ax.set_xlabel('Price')
            self.depth_ax.set_ylabel('Cumulative Quantity')
            self.depth_ax.legend(loc='upper left')
            
            # Draw canvas
            self.depth_fig.tight_layout()
            self.depth_canvas.draw()
        except Exception as e:
            logger.error(f"Error updating depth chart: {str(e)}")


class PositionsView(ttk.Frame):
    """View for displaying positions and orders"""
    
    def __init__(self, parent, trading_vm: TradingViewModel, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.trading_vm = trading_vm
        
        # Add callback for trading updates
        self.trading_vm.add_trading_callback(self.update_positions)
        
        # Create positions display
        self._create_positions_display()
    
    def _create_positions_display(self) -> None:
        """Create positions display"""
        # Create notebook for positions, orders, and history
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create positions tab
        positions_frame = ttk.Frame(self.notebook)
        self.notebook.add(positions_frame, text="Positions")
        
        # Create positions table
        positions_tree_frame = ttk.Frame(positions_frame)
        positions_tree_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.positions_tree = ttk.Treeview(
            positions_tree_frame,
            columns=("symbol", "side", "size", "entry_price", "mark_price", "pnl", "pnl_pct", "liquidation"),
            show="headings",
            height=10
        )
        
        self.positions_tree.heading("symbol", text="Symbol")
        self.positions_tree.heading("side", text="Side")
        self.positions_tree.heading("size", text="Size")
        self.positions_tree.heading("entry_price", text="Entry Price")
        self.positions_tree.heading("mark_price", text="Mark Price")
        self.positions_tree.heading("pnl", text="PnL")
        self.positions_tree.heading("pnl_pct", text="PnL %")
        self.positions_tree.heading("liquidation", text="Liquidation")
        
        self.positions_tree.column("symbol", width=80, anchor="center")
        self.positions_tree.column("side", width=80, anchor="center")
        self.positions_tree.column("size", width=80, anchor="e")
        self.positions_tree.column("entry_price", width=100, anchor="e")
        self.positions_tree.column("mark_price", width=100, anchor="e")
        self.positions_tree.column("pnl", width=100, anchor="e")
        self.positions_tree.column("pnl_pct", width=80, anchor="e")
        self.positions_tree.column("liquidation", width=100, anchor="e")
        
        positions_scrollbar = ttk.Scrollbar(positions_tree_frame, orient="vertical", command=self.positions_tree.yview)
        self.positions_tree.configure(yscrollcommand=positions_scrollbar.set)
        
        positions_scrollbar.pack(side="right", fill="y")
        self.positions_tree.pack(side="left", fill="both", expand=True)
        
        # Add close position button
        close_button = ttk.Button(positions_frame, text="Close Position", command=self._close_position)
        close_button.pack(pady=5)
        
        # Create orders tab
        orders_frame = ttk.Frame(self.notebook)
        self.notebook.add(orders_frame, text="Orders")
        
        # Create orders table
        orders_tree_frame = ttk.Frame(orders_frame)
        orders_tree_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.orders_tree = ttk.Treeview(
            orders_tree_frame,
            columns=("id", "symbol", "type", "side", "price", "quantity", "filled", "status", "time"),
            show="headings",
            height=10
        )
        
        self.orders_tree.heading("id", text="ID")
        self.orders_tree.heading("symbol", text="Symbol")
        self.orders_tree.heading("type", text="Type")
        self.orders_tree.heading("side", text="Side")
        self.orders_tree.heading("price", text="Price")
        self.orders_tree.heading("quantity", text="Quantity")
        self.orders_tree.heading("filled", text="Filled")
        self.orders_tree.heading("status", text="Status")
        self.orders_tree.heading("time", text="Time")
        
        self.orders_tree.column("id", width=80, anchor="w")
        self.orders_tree.column("symbol", width=80, anchor="center")
        self.orders_tree.column("type", width=80, anchor="center")
        self.orders_tree.column("side", width=80, anchor="center")
        self.orders_tree.column("price", width=100, anchor="e")
        self.orders_tree.column("quantity", width=80, anchor="e")
        self.orders_tree.column("filled", width=80, anchor="e")
        self.orders_tree.column("status", width=80, anchor="center")
        self.orders_tree.column("time", width=150, anchor="w")
        
        orders_scrollbar = ttk.Scrollbar(orders_tree_frame, orient="vertical", command=self.orders_tree.yview)
        self.orders_tree.configure(yscrollcommand=orders_scrollbar.set)
        
        orders_scrollbar.pack(side="right", fill="y")
        self.orders_tree.pack(side="left", fill="both", expand=True)
        
        # Add cancel order button
        cancel_button = ttk.Button(orders_frame, text="Cancel Order", command=self._cancel_order)
        cancel_button.pack(pady=5)
        
        # Create history tab
        history_frame = ttk.Frame(self.notebook)
        self.notebook.add(history_frame, text="Trade History")
        
        # Create history table
        history_tree_frame = ttk.Frame(history_frame)
        history_tree_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.history_tree = ttk.Treeview(
            history_tree_frame,
            columns=("id", "symbol", "side", "price", "quantity", "fee", "time"),
            show="headings",
            height=10
        )
        
        self.history_tree.heading("id", text="ID")
        self.history_tree.heading("symbol", text="Symbol")
        self.history_tree.heading("side", text="Side")
        self.history_tree.heading("price", text="Price")
        self.history_tree.heading("quantity", text="Quantity")
        self.history_tree.heading("fee", text="Fee")
        self.history_tree.heading("time", text="Time")
        
        self.history_tree.column("id", width=80, anchor="w")
        self.history_tree.column("symbol", width=80, anchor="center")
        self.history_tree.column("side", width=80, anchor="center")
        self.history_tree.column("price", width=100, anchor="e")
        self.history_tree.column("quantity", width=80, anchor="e")
        self.history_tree.column("fee", width=80, anchor="e")
        self.history_tree.column("time", width=150, anchor="w")
        
        history_scrollbar = ttk.Scrollbar(history_tree_frame, orient="vertical", command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=history_scrollbar.set)
        
        history_scrollbar.pack(side="right", fill="y")
        self.history_tree.pack(side="left", fill="both", expand=True)
        
        # Initial data fetch
        self.trading_vm.fetch_account_data()
    
    def update_positions(self) -> None:
        """Update positions display"""
        try:
            # Update positions
            positions = self.trading_vm.get_positions()
            
            # Clear tree
            for item in self.positions_tree.get_children():
                self.positions_tree.delete(item)
            
            # Add positions
            for symbol, position in positions.items():
                side = "Long" if position["size"] > 0 else "Short"
                pnl = position["unrealized_pnl"]
                pnl_pct = position["unrealized_pnl_pct"]
                
                # Format values
                size = f"{position['size']:.4f}"
                entry_price = f"{position['entry_price']:.2f}"
                mark_price = f"{position['mark_price']:.2f}"
                pnl_str = f"{pnl:.2f}"
                pnl_pct_str = f"{pnl_pct:.2f}%"
                liquidation = f"{position['liquidation_price']:.2f}"
                
                # Set row color based on PnL
                tag = "profit" if pnl > 0 else "loss"
                
                self.positions_tree.insert(
                    "", "end", 
                    values=(symbol, side, size, entry_price, mark_price, pnl_str, pnl_pct_str, liquidation),
                    tags=(tag,)
                )
            
            # Configure tags
            self.positions_tree.tag_configure("profit", foreground="green")
            self.positions_tree.tag_configure("loss", foreground="red")
            
            # Update orders
            orders = self.trading_vm.get_orders()
            
            # Clear tree
            for item in self.orders_tree.get_children():
                self.orders_tree.delete(item)
            
            # Add orders
            for order in orders:
                order_id = order["id"]
                symbol = order["symbol"]
                order_type = order["type"]
                side = order["side"]
                price = f"{order['price']:.2f}" if order["price"] else "Market"
                quantity = f"{order['quantity']:.4f}"
                filled = f"{order['filled']:.4f}"
                status = order["status"]
                time_str = datetime.fromtimestamp(order["time"]).strftime("%Y-%m-%d %H:%M:%S")
                
                self.orders_tree.insert(
                    "", "end", 
                    values=(order_id, symbol, order_type, side, price, quantity, filled, status, time_str)
                )
            
            # Update trade history
            history = self.trading_vm.get_trade_history()
            
            # Clear tree
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
            
            # Add trade history
            for trade in history:
                trade_id = trade["id"]
                symbol = trade["symbol"]
                side = trade["side"]
                price = f"{trade['price']:.2f}"
                quantity = f"{trade['quantity']:.4f}"
                fee = f"{trade['fee']:.4f}"
                time_str = datetime.fromtimestamp(trade["time"]).strftime("%Y-%m-%d %H:%M:%S")
                
                # Set row color based on side
                tag = "buy" if side.lower() == "buy" else "sell"
                
                self.history_tree.insert(
                    "", "end", 
                    values=(trade_id, symbol, side, price, quantity, fee, time_str),
                    tags=(tag,)
                )
            
            # Configure tags
            self.history_tree.tag_configure("buy", foreground="green")
            self.history_tree.tag_configure("sell", foreground="red")
        except Exception as e:
            logger.error(f"Error updating positions: {str(e)}")
    
    def _close_position(self) -> None:
        """Close selected position"""
        try:
            selected = self.positions_tree.selection()
            if not selected:
                messagebox.showwarning("Warning", "No position selected")
                return
            
            # Get symbol from selected position
            values = self.positions_tree.item(selected[0], "values")
            symbol = values[0]
            
            # Confirm close
            if messagebox.askyesno("Confirm", f"Close position for {symbol}?"):
                # Close position
                success = self.trading_vm.close_position(symbol)
                
                if success:
                    messagebox.showinfo("Success", f"Position for {symbol} closed")
                else:
                    messagebox.showerror("Error", f"Failed to close position for {symbol}")
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            messagebox.showerror("Error", f"Error closing position: {str(e)}")
    
    def _cancel_order(self) -> None:
        """Cancel selected order"""
        try:
            selected = self.orders_tree.selection()
            if not selected:
                messagebox.showwarning("Warning", "No order selected")
                return
            
            # Get order ID from selected order
            values = self.orders_tree.item(selected[0], "values")
            order_id = values[0]
            
            # Confirm cancel
            if messagebox.askyesno("Confirm", f"Cancel order {order_id}?"):
                # Cancel order
                success = self.trading_vm.cancel_order(order_id)
                
                if success:
                    messagebox.showinfo("Success", f"Order {order_id} cancelled")
                else:
                    messagebox.showerror("Error", f"Failed to cancel order {order_id}")
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            messagebox.showerror("Error", f"Error cancelling order: {str(e)}")


class TradeView(ttk.Frame):
    """View for placing trades"""
    
    def __init__(self, parent, trading_vm: TradingViewModel, market_vm: MarketViewModel, config_data: ConfigData, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.trading_vm = trading_vm
        self.market_vm = market_vm
        self.config_data = config_data
        
        # Create trade form
        self._create_trade_form()
    
    def _create_trade_form(self) -> None:
        """Create trade form"""
        # Create main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create form frame
        form_frame = ttk.LabelFrame(main_frame, text="Place Order")
        form_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create scrollable frame for form
        form_scroll = ScrollableFrame(form_frame)
        form_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Symbol selection
        symbol_frame = ttk.Frame(form_scroll.scrollable_frame)
        symbol_frame.pack(fill="x", pady=5)
        
        ttk.Label(symbol_frame, text="Symbol:").pack(side="left")
        
        self.symbol_var = tk.StringVar(value=self.config_data.get("default_symbol"))
        symbol_combo = ttk.Combobox(
            symbol_frame, 
            textvariable=self.symbol_var, 
            values=self.config_data.get("symbols"),
            width=10
        )
        symbol_combo.pack(side="left", padx=5)
        
        # Order type selection
        type_frame = ttk.Frame(form_scroll.scrollable_frame)
        type_frame.pack(fill="x", pady=5)
        
        ttk.Label(type_frame, text="Order Type:").pack(side="left")
        
        self.order_type_var = tk.StringVar(value=self.config_data.get("default_order_type"))
        order_type_combo = ttk.Combobox(
            type_frame, 
            textvariable=self.order_type_var, 
            values=self.config_data.get("order_types"),
            width=10
        )
        order_type_combo.pack(side="left", padx=5)
        order_type_combo.bind("<<ComboboxSelected>>", self._on_order_type_change)
        
        # Side selection
        side_frame = ttk.Frame(form_scroll.scrollable_frame)
        side_frame.pack(fill="x", pady=5)
        
        ttk.Label(side_frame, text="Side:").pack(side="left")
        
        self.side_var = tk.StringVar(value="buy")
        
        buy_radio = ttk.Radiobutton(side_frame, text="Buy", variable=self.side_var, value="buy")
        buy_radio.pack(side="left", padx=5)
        
        sell_radio = ttk.Radiobutton(side_frame, text="Sell", variable=self.side_var, value="sell")
        sell_radio.pack(side="left", padx=5)
        
        # Quantity
        quantity_frame = ttk.Frame(form_scroll.scrollable_frame)
        quantity_frame.pack(fill="x", pady=5)
        
        ttk.Label(quantity_frame, text="Quantity:").pack(side="left")
        
        self.quantity_var = tk.StringVar(value=str(self.config_data.get("default_quantity")))
        quantity_entry = ttk.Entry(quantity_frame, textvariable=self.quantity_var, width=10)
        quantity_entry.pack(side="left", padx=5)
        
        # Quantity validation
        vcmd = (self.register(self._validate_float), '%P')
        quantity_entry.configure(validate="key", validatecommand=vcmd)
        
        # Quantity buttons
        quantity_buttons_frame = ttk.Frame(quantity_frame)
        quantity_buttons_frame.pack(side="left")
        
        ttk.Button(quantity_buttons_frame, text="25%", width=5, 
                  command=lambda: self._set_quantity_percent(0.25)).pack(side="left", padx=2)
        ttk.Button(quantity_buttons_frame, text="50%", width=5,
                  command=lambda: self._set_quantity_percent(0.5)).pack(side="left", padx=2)
        ttk.Button(quantity_buttons_frame, text="75%", width=5,
                  command=lambda: self._set_quantity_percent(0.75)).pack(side="left", padx=2)
        ttk.Button(quantity_buttons_frame, text="100%", width=5,
                  command=lambda: self._set_quantity_percent(1.0)).pack(side="left", padx=2)
        
        # Price
        self.price_frame = ttk.Frame(form_scroll.scrollable_frame)
        self.price_frame.pack(fill="x", pady=5)
        
        ttk.Label(self.price_frame, text="Price:").pack(side="left")
        
        self.price_var = tk.StringVar(value="0.0")
        self.price_entry = ttk.Entry(self.price_frame, textvariable=self.price_var, width=10)
        self.price_entry.pack(side="left", padx=5)
        
        # Price validation
        self.price_entry.configure(validate="key", validatecommand=vcmd)
        
        # Stop price
        self.stop_price_frame = ttk.Frame(form_scroll.scrollable_frame)
        self.stop_price_frame.pack(fill="x", pady=5)
        
        ttk.Label(self.stop_price_frame, text="Stop Price:").pack(side="left")
        
        self.stop_price_var = tk.StringVar(value="0.0")
        self.stop_price_entry = ttk.Entry(self.stop_price_frame, textvariable=self.stop_price_var, width=10)
        self.stop_price_entry.pack(side="left", padx=5)
        
        # Stop price validation
        self.stop_price_entry.configure(validate="key", validatecommand=vcmd)
        
        # Update form based on initial order type
        self._on_order_type_change(None)
        
        # Submit button
        submit_frame = ttk.Frame(form_scroll.scrollable_frame)
        submit_frame.pack(fill="x", pady=10)
        
        self.submit_button = ttk.Button(submit_frame, text="Place Order", command=self._submit_order)
        self.submit_button.pack(side="left", padx=5)
        
        # Reset button
        self.reset_button = ttk.Button(submit_frame, text="Reset", command=self._reset_form)
        self.reset_button.pack(side="left", padx=5)
        
        # Error message
        self.error_var = tk.StringVar(value="")
        self.error_label = ttk.Label(form_scroll.scrollable_frame, textvariable=self.error_var, foreground="red")
        self.error_label.pack(fill="x", pady=5)
    
    def _validate_float(self, value) -> bool:
        """Validate float input"""
        if value == "":
            return True
        
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def _on_order_type_change(self, event) -> None:
        """Handle order type change"""
        order_type = self.order_type_var.get().lower()
        
        # Show/hide price field
        if order_type in ["limit", "stop_limit"]:
            self.price_frame.pack(fill="x", pady=5)
        else:
            self.price_frame.pack_forget()
        
        # Show/hide stop price field
        if order_type in ["stop", "stop_limit"]:
            self.stop_price_frame.pack(fill="x", pady=5)
        else:
            self.stop_price_frame.pack_forget()
    
    def _set_quantity_percent(self, percent: float) -> None:
        """Set quantity to percent of available balance"""
        try:
            balance = self.trading_vm.get_account_balance()
            symbol = self.symbol_var.get()
            
            # Get current price (simplified)
            price = 1000.0  # Placeholder, should get from market data
            
            # Calculate quantity
            quantity = balance * percent / price
            
            # Update quantity field
            self.quantity_var.set(f"{quantity:.4f}")
        except Exception as e:
            logger.error(f"Error setting quantity: {str(e)}")
            self.error_var.set(f"Error: {str(e)}")
    
    def _reset_form(self) -> None:
        """Reset form to defaults"""
        self.symbol_var.set(self.config_data.get("default_symbol"))
        self.order_type_var.set(self.config_data.get("default_order_type"))
        self.side_var.set("buy")
        self.quantity_var.set(str(self.config_data.get("default_quantity")))
        self.price_var.set("0.0")
        self.stop_price_var.set("0.0")
        self.error_var.set("")
        
        # Update form based on order type
        self._on_order_type_change(None)
    
    def _submit_order(self) -> None:
        """Submit order"""
        try:
            # Clear error message
            self.error_var.set("")
            
            # Get form values
            symbol = self.symbol_var.get()
            order_type = self.order_type_var.get().lower()
            side = self.side_var.get()
            
            # Validate quantity
            try:
                quantity = float(self.quantity_var.get())
                if quantity <= 0:
                    raise ValueError("Quantity must be greater than 0")
            except ValueError:
                self.error_var.set("Error: Invalid quantity")
                return
            
            # Validate price for limit orders
            price = None
            if order_type in ["limit", "stop_limit"]:
                try:
                    price = float(self.price_var.get())
                    if price <= 0:
                        raise ValueError("Price must be greater than 0")
                except ValueError:
                    self.error_var.set("Error: Invalid price")
                    return
            
            # Validate stop price for stop orders
            stop_price = None
            if order_type in ["stop", "stop_limit"]:
                try:
                    stop_price = float(self.stop_price_var.get())
                    if stop_price <= 0:
                        raise ValueError("Stop price must be greater than 0")
                except ValueError:
                    self.error_var.set("Error: Invalid stop price")
                    return
            
            # Confirm order
            message = f"Place {side} {order_type} order for {quantity} {symbol}"
            if price:
                message += f" at price {price}"
            if stop_price:
                message += f" with stop price {stop_price}"
            
            if messagebox.askyesno("Confirm Order", f"{message}?"):
                # Place order
                success = self.trading_vm.place_order(
                    symbol, order_type, side, quantity, price, stop_price
                )
                
                if success:
                    messagebox.showinfo("Success", "Order placed successfully")
                    self._reset_form()
                else:
                    self.error_var.set("Error: Failed to place order")
        except Exception as e:
            logger.error(f"Error submitting order: {str(e)}")
            self.error_var.set(f"Error: {str(e)}")


class SettingsView(ttk.Frame):
    """View for settings and configuration"""
    
    def __init__(self, parent, config_data: ConfigData, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.config_data = config_data
        
        # Create settings form
        self._create_settings_form()
    
    def _create_settings_form(self) -> None:
        """Create settings form"""
        # Create main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create notebook for settings categories
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create general settings tab
        general_frame = ttk.Frame(notebook)
        notebook.add(general_frame, text="General")
        
        # Create scrollable frame for general settings
        general_scroll = ScrollableFrame(general_frame)
        general_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Theme selection
        theme_frame = ttk.Frame(general_scroll.scrollable_frame)
        theme_frame.pack(fill="x", pady=5)
        
        ttk.Label(theme_frame, text="Theme:").pack(side="left")
        
        self.theme_var = tk.StringVar(value=self.config_data.get("theme"))
        theme_combo = ttk.Combobox(
            theme_frame, 
            textvariable=self.theme_var, 
            values=["light", "dark"],
            width=10
        )
        theme_combo.pack(side="left", padx=5)
        
        # Auto refresh interval
        refresh_frame = ttk.Frame(general_scroll.scrollable_frame)
        refresh_frame.pack(fill="x", pady=5)
        
        ttk.Label(refresh_frame, text="Auto Refresh Interval (ms):").pack(side="left")
        
        self.refresh_var = tk.StringVar(value=str(self.config_data.get("auto_refresh_interval")))
        refresh_entry = ttk.Entry(refresh_frame, textvariable=self.refresh_var, width=10)
        refresh_entry.pack(side="left", padx=5)
        
        # Show trade confirmations
        confirm_frame = ttk.Frame(general_scroll.scrollable_frame)
        confirm_frame.pack(fill="x", pady=5)
        
        self.confirm_var = tk.BooleanVar(value=self.config_data.get("show_trade_confirmations"))
        confirm_check = ttk.Checkbutton(
            confirm_frame, 
            text="Show Trade Confirmations", 
            variable=self.confirm_var
        )
        confirm_check.pack(side="left")
        
        # Advanced mode
        advanced_frame = ttk.Frame(general_scroll.scrollable_frame)
        advanced_frame.pack(fill="x", pady=5)
        
        self.advanced_var = tk.BooleanVar(value=self.config_data.get("advanced_mode"))
        advanced_check = ttk.Checkbutton(
            advanced_frame, 
            text="Advanced Mode", 
            variable=self.advanced_var
        )
        advanced_check.pack(side="left")
        
        # Create API settings tab
        api_frame = ttk.Frame(notebook)
        notebook.add(api_frame, text="API")
        
        # Create scrollable frame for API settings
        api_scroll = ScrollableFrame(api_frame)
        api_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # API key
        key_frame = ttk.Frame(api_scroll.scrollable_frame)
        key_frame.pack(fill="x", pady=5)
        
        ttk.Label(key_frame, text="API Key:").pack(side="left")
        
        self.key_var = tk.StringVar(value=self.config_data.get("api_key"))
        key_entry = ttk.Entry(key_frame, textvariable=self.key_var, width=40, show="*")
        key_entry.pack(side="left", padx=5)
        
        # API secret
        secret_frame = ttk.Frame(api_scroll.scrollable_frame)
        secret_frame.pack(fill="x", pady=5)
        
        ttk.Label(secret_frame, text="API Secret:").pack(side="left")
        
        self.secret_var = tk.StringVar(value=self.config_data.get("api_secret"))
        secret_entry = ttk.Entry(secret_frame, textvariable=self.secret_var, width=40, show="*")
        secret_entry.pack(side="left", padx=5)
        
        # Use mock data
        mock_frame = ttk.Frame(api_scroll.scrollable_frame)
        mock_frame.pack(fill="x", pady=5)
        
        self.mock_var = tk.BooleanVar(value=self.config_data.get("use_mock_data"))
        mock_check = ttk.Checkbutton(
            mock_frame, 
            text="Use Mock Data", 
            variable=self.mock_var
        )
        mock_check.pack(side="left")
        
        # Create trading settings tab
        trading_frame = ttk.Frame(notebook)
        notebook.add(trading_frame, text="Trading")
        
        # Create scrollable frame for trading settings
        trading_scroll = ScrollableFrame(trading_frame)
        trading_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Default symbol
        default_symbol_frame = ttk.Frame(trading_scroll.scrollable_frame)
        default_symbol_frame.pack(fill="x", pady=5)
        
        ttk.Label(default_symbol_frame, text="Default Symbol:").pack(side="left")
        
        self.default_symbol_var = tk.StringVar(value=self.config_data.get("default_symbol"))
        default_symbol_combo = ttk.Combobox(
            default_symbol_frame, 
            textvariable=self.default_symbol_var, 
            values=self.config_data.get("symbols"),
            width=10
        )
        default_symbol_combo.pack(side="left", padx=5)
        
        # Default timeframe
        default_timeframe_frame = ttk.Frame(trading_scroll.scrollable_frame)
        default_timeframe_frame.pack(fill="x", pady=5)
        
        ttk.Label(default_timeframe_frame, text="Default Timeframe:").pack(side="left")
        
        self.default_timeframe_var = tk.StringVar(value=self.config_data.get("default_timeframe"))
        default_timeframe_combo = ttk.Combobox(
            default_timeframe_frame, 
            textvariable=self.default_timeframe_var, 
            values=self.config_data.get("timeframes"),
            width=10
        )
        default_timeframe_combo.pack(side="left", padx=5)
        
        # Default order type
        default_order_type_frame = ttk.Frame(trading_scroll.scrollable_frame)
        default_order_type_frame.pack(fill="x", pady=5)
        
        ttk.Label(default_order_type_frame, text="Default Order Type:").pack(side="left")
        
        self.default_order_type_var = tk.StringVar(value=self.config_data.get("default_order_type"))
        default_order_type_combo = ttk.Combobox(
            default_order_type_frame, 
            textvariable=self.default_order_type_var, 
            values=self.config_data.get("order_types"),
            width=10
        )
        default_order_type_combo.pack(side="left", padx=5)
        
        # Default quantity
        default_quantity_frame = ttk.Frame(trading_scroll.scrollable_frame)
        default_quantity_frame.pack(fill="x", pady=5)
        
        ttk.Label(default_quantity_frame, text="Default Quantity:").pack(side="left")
        
        self.default_quantity_var = tk.StringVar(value=str(self.config_data.get("default_quantity")))
        default_quantity_entry = ttk.Entry(default_quantity_frame, textvariable=self.default_quantity_var, width=10)
        default_quantity_entry.pack(side="left", padx=5)
        
        # Risk level
        risk_frame = ttk.Frame(trading_scroll.scrollable_frame)
        risk_frame.pack(fill="x", pady=5)
        
        ttk.Label(risk_frame, text="Risk Level:").pack(side="left")
        
        self.risk_var = tk.StringVar(value=str(self.config_data.get("risk_level")))
        risk_entry = ttk.Entry(risk_frame, textvariable=self.risk_var, width=10)
        risk_entry.pack(side="left", padx=5)
        
        # Save button
        save_frame = ttk.Frame(main_frame)
        save_frame.pack(fill="x", pady=10)
        
        save_button = ttk.Button(save_frame, text="Save Settings", command=self._save_settings)
        save_button.pack(side="left", padx=5)
        
        # Reset button
        reset_button = ttk.Button(save_frame, text="Reset to Defaults", command=self._reset_settings)
        reset_button.pack(side="left", padx=5)
        
        # Status message
        self.status_var = tk.StringVar(value="")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.pack(fill="x", pady=5)
    
    def _save_settings(self) -> None:
        """Save settings"""
        try:
            # Update config data
            self.config_data.set("theme", self.theme_var.get())
            self.config_data.set("auto_refresh_interval", int(self.refresh_var.get()))
            self.config_data.set("show_trade_confirmations", self.confirm_var.get())
            self.config_data.set("advanced_mode", self.advanced_var.get())
            self.config_data.set("api_key", self.key_var.get())
            self.config_data.set("api_secret", self.secret_var.get())
            self.config_data.set("use_mock_data", self.mock_var.get())
            self.config_data.set("default_symbol", self.default_symbol_var.get())
            self.config_data.set("default_timeframe", self.default_timeframe_var.get())
            self.config_data.set("default_order_type", self.default_order_type_var.get())
            self.config_data.set("default_quantity", float(self.default_quantity_var.get()))
            self.config_data.set("risk_level", float(self.risk_var.get()))
            
            # Save config
            self.config_data.save()
            
            # Update status
            self.status_var.set("Settings saved successfully")
            
            # Show message
            messagebox.showinfo("Success", "Settings saved successfully. Some changes may require a restart to take effect.")
        except Exception as e:
            logger.error(f"Error saving settings: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Error saving settings: {str(e)}")
    
    def _reset_settings(self) -> None:
        """Reset settings to defaults"""
        try:
            if messagebox.askyesno("Confirm", "Reset all settings to defaults?"):
                # Reset config data
                self.config_data = ConfigData()
                
                # Update form
                self.theme_var.set(self.config_data.get("theme"))
                self.refresh_var.set(str(self.config_data.get("auto_refresh_interval")))
                self.confirm_var.set(self.config_data.get("show_trade_confirmations"))
                self.advanced_var.set(self.config_data.get("advanced_mode"))
                self.key_var.set(self.config_data.get("api_key"))
                self.secret_var.set(self.config_data.get("api_secret"))
                self.mock_var.set(self.config_data.get("use_mock_data"))
                self.default_symbol_var.set(self.config_data.get("default_symbol"))
                self.default_timeframe_var.set(self.config_data.get("default_timeframe"))
                self.default_order_type_var.set(self.config_data.get("default_order_type"))
                self.default_quantity_var.set(str(self.config_data.get("default_quantity")))
                self.risk_var.set(str(self.config_data.get("risk_level")))
                
                # Update status
                self.status_var.set("Settings reset to defaults")
        except Exception as e:
            logger.error(f"Error resetting settings: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Error resetting settings: {str(e)}")


class LogView(ttk.Frame):
    """View for logs and messages"""
    
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        
        # Create log display
        self._create_log_display()
        
        # Set up log queue
        self.log_queue = queue.Queue()
        self._setup_logger()
        
        # Start log consumer
        self.log_consumer_running = True
        self.log_consumer_thread = threading.Thread(target=self._consume_logs, daemon=True)
        self.log_consumer_thread.start()
    
    def _create_log_display(self) -> None:
        """Create log display"""
        # Create main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create log text
        self.log_text = scrolledtext.ScrolledText(main_frame, height=10, wrap=tk.WORD)
        self.log_text.pack(fill="both", expand=True)
        self.log_text.config(state=tk.DISABLED)
        
        # Create control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill="x", pady=5)
        
        # Create clear button
        clear_button = ttk.Button(control_frame, text="Clear Logs", command=self._clear_logs)
        clear_button.pack(side="left", padx=5)
        
        # Create log level filter
        ttk.Label(control_frame, text="Log Level:").pack(side="left", padx=(10, 5))
        
        self.log_level_var = tk.StringVar(value="INFO")
        log_level_combo = ttk.Combobox(
            control_frame, 
            textvariable=self.log_level_var, 
            values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            width=10
        )
        log_level_combo.pack(side="left", padx=5)
        log_level_combo.bind("<<ComboboxSelected>>", self._on_log_level_change)
    
    def _setup_logger(self) -> None:
        """Set up logger with queue handler"""
        # Create queue handler
        queue_handler = logging.Handler()
        queue_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        queue_handler.setFormatter(formatter)
        
        # Override emit method to put log records in queue
        def emit(record):
            self.log_queue.put(record)
        
        queue_handler.emit = emit
        
        # Add handler to logger
        logger.addHandler(queue_handler)
    
    def _consume_logs(self) -> None:
        """Consume logs from queue"""
        while self.log_consumer_running:
            try:
                # Get log record from queue
                record = self.log_queue.get(block=True, timeout=0.1)
                
                # Format record
                message = record.getMessage()
                level = record.levelname
                
                # Add log to text widget
                self._add_log(f"[{level}] {message}")
                
                # Mark as done
                self.log_queue.task_done()
            except queue.Empty:
                # No logs in queue, continue
                continue
            except Exception as e:
                print(f"Error consuming logs: {str(e)}")
    
    def _add_log(self, message: str) -> None:
        """Add log message to text widget"""
        try:
            # Get log level
            level = self.log_level_var.get()
            
            # Check if message should be displayed based on log level
            if level == "DEBUG":
                # Show all logs
                pass
            elif level == "INFO" and ("[DEBUG]" in message):
                # Skip debug logs
                return
            elif level == "WARNING" and any(x in message for x in ["[DEBUG]", "[INFO]"]):
                # Skip debug and info logs
                return
            elif level == "ERROR" and any(x in message for x in ["[DEBUG]", "[INFO]", "[WARNING]"]):
                # Skip debug, info, and warning logs
                return
            elif level == "CRITICAL" and any(x in message for x in ["[DEBUG]", "[INFO]", "[WARNING]", "[ERROR]"]):
                # Skip all but critical logs
                return
            
            # Enable text widget for editing
            self.log_text.config(state=tk.NORMAL)
            
            # Add message with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_text.insert(tk.END, f"{timestamp} {message}\n")
            
            # Apply tag based on log level
            if "[ERROR]" in message or "[CRITICAL]" in message:
                self.log_text.tag_add("error", f"end-{len(message)+12}c", "end-1c")
            elif "[WARNING]" in message:
                self.log_text.tag_add("warning", f"end-{len(message)+12}c", "end-1c")
            
            # Configure tags
            self.log_text.tag_configure("error", foreground="red")
            self.log_text.tag_configure("warning", foreground="orange")
            
            # Scroll to end
            self.log_text.see(tk.END)
            
            # Disable text widget
            self.log_text.config(state=tk.DISABLED)
        except Exception as e:
            print(f"Error adding log: {str(e)}")
    
    def _clear_logs(self) -> None:
        """Clear logs"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def _on_log_level_change(self, event) -> None:
        """Handle log level change"""
        # Clear logs
        self._clear_logs()
        
        # Add message about log level change
        level = self.log_level_var.get()
        self._add_log(f"[INFO] Log level changed to {level}")


#############################################################################
# Main Application
#############################################################################

class EnhancedTradingBotGUI:
    """
    Main GUI class for the Enhanced Hyperliquid Trading Bot.
    Provides a comprehensive interface for trading and monitoring.
    """
    
    def __init__(self, root):
        """
        Initialize the GUI and all components.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title(f"Enhanced Hyperliquid Trading Bot v{VERSION}")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Initialize models
        self.market_data = MarketData()
        self.position_data = PositionData()
        self.config_data = ConfigData()
        
        # Initialize error handler
        self.error_handler = ErrorHandler()
        
        # Initialize view models
        self.market_vm = MarketViewModel(self.market_data, self.config_data, self.error_handler)
        self.trading_vm = TradingViewModel(self.position_data, self.config_data, self.error_handler)
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create status bar
        self.status_bar = StatusBar(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create chart tab
        chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(chart_frame, text="Chart")
        self.chart_view = ChartView(chart_frame, self.market_vm, self.config_data)
        
        # Create order book tab
        orderbook_frame = ttk.Frame(self.notebook)
        self.notebook.add(orderbook_frame, text="Order Book")
        self.orderbook_view = OrderBookView(orderbook_frame, self.market_vm)
        
        # Create positions tab
        positions_frame = ttk.Frame(self.notebook)
        self.notebook.add(positions_frame, text="Positions")
        self.positions_view = PositionsView(positions_frame, self.trading_vm)
        
        # Create trade tab
        trade_frame = ttk.Frame(self.notebook)
        self.notebook.add(trade_frame, text="Trade")
        self.trade_view = TradeView(trade_frame, self.trading_vm, self.market_vm, self.config_data)
        
        # Create settings tab
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")
        self.settings_view = SettingsView(settings_frame, self.config_data)
        
        # Create logs tab
        logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(logs_frame, text="Logs")
        self.log_view = LogView(logs_frame)
        
        # Set up auto refresh
        self._setup_auto_refresh()
        
        # Set up API rate limiter check
        self._setup_rate_limiter_check()
        
        # Log startup
        logger.info(f"Enhanced Hyperliquid Trading Bot v{VERSION} started")
        self.status_bar.set_status("Ready")
    
    def _setup_auto_refresh(self) -> None:
        """Set up auto refresh"""
        # Get refresh interval
        interval = self.config_data.get("auto_refresh_interval", 5000)
        
        # Define refresh function
        def refresh():
            try:
                # Get current tab
                current_tab = self.notebook.index(self.notebook.select())
                
                # Refresh data based on current tab
                if current_tab == 0:  # Chart tab
                    self.market_vm.fetch_market_data(
                        self.config_data.get("default_symbol"),
                        self.config_data.get("default_timeframe")
                    )
                elif current_tab == 1:  # Order Book tab
                    self.market_vm.fetch_market_data(
                        self.config_data.get("default_symbol"),
                        self.config_data.get("default_timeframe")
                    )
                elif current_tab == 2:  # Positions tab
                    self.trading_vm.fetch_account_data()
                
                # Schedule next refresh
                self.root.after(interval, refresh)
            except Exception as e:
                logger.error(f"Error in auto refresh: {str(e)}")
                # Schedule next refresh even if error
                self.root.after(interval, refresh)
        
        # Start refresh
        self.root.after(interval, refresh)
    
    def _setup_rate_limiter_check(self) -> None:
        """Set up API rate limiter check"""
        # Define check function
        def check_rate_limiter():
            try:
                # Get rate limiter status
                status = self.market_vm.api_rate_limiter.get_status()
                
                # Update status bar
                if status["is_limited"]:
                    self.status_bar.set_api_status(f"Limited ({status['cooldown_remaining']}s)", True)
                    self.status_bar.set_mock_data(True)
                else:
                    self.status_bar.set_api_status("OK")
                    self.status_bar.set_mock_data(self.market_vm.using_mock_data)
                
                # Schedule next check
                self.root.after(5000, check_rate_limiter)
            except Exception as e:
                logger.error(f"Error checking rate limiter: {str(e)}")
                # Schedule next check even if error
                self.root.after(5000, check_rate_limiter)
        
        # Start check
        self.root.after(5000, check_rate_limiter)


def main():
    """Main function"""
    # Create root window
    root = tk.Tk()
    
    # Create GUI
    app = EnhancedTradingBotGUI(root)
    
    # Start main loop
    root.mainloop()


if __name__ == "__main__":
    main()
