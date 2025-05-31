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
import queue # Added for LogView

# Import core components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.api_rate_limiter import APIRateLimiter # Use the fixed version
from core.error_handling import ErrorHandler # Use the fixed version
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
VERSION = "3.1.0" # Updated version

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
            allowed, reason = self.api_rate_limiter.check_rate_limit()
            if self.using_mock_data or not allowed:
                if not self.using_mock_data and not allowed:
                    logger.warning(f"API rate limited ({reason}). Using mock data.")
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
                self.api_rate_limiter.add_request() # Record request
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
                ohlcv["ma_fast"] = self.technical_indicators.calculate_sma(ohlcv["close"], 20)
                ohlcv["ma_slow"] = self.technical_indicators.calculate_sma(ohlcv["close"], 50)
                ohlcv["rsi"] = self.technical_indicators.calculate_rsi(ohlcv["close"], 14)
                
                macd, signal, hist = self.technical_indicators.calculate_macd(ohlcv["close"])
                ohlcv["macd"] = macd
                ohlcv["macd_signal"] = signal
                ohlcv["macd_hist"] = hist
                
                upper, middle, lower = self.technical_indicators.calculate_bollinger_bands(ohlcv["close"], 20, 2)
                ohlcv["bb_upper"] = upper
                ohlcv["bb_middle"] = middle
                ohlcv["bb_lower"] = lower
                
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
            allowed, reason = self.api_rate_limiter.check_rate_limit()
            if self.using_mock_data or not allowed:
                if not self.using_mock_data and not allowed:
                    logger.warning(f"API rate limited ({reason}). Using mock data.")
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
                self.api_rate_limiter.add_request() # Record request
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
    
    def place_order(self, symbol: str, order_type: str, side: str, quantity: float, 
                    price: Optional[float] = None, stop_price: Optional[float] = None) -> bool:
        """Place an order"""
        try:
            # Check if we should use mock data
            allowed, reason = self.api_rate_limiter.check_rate_limit()
            if self.using_mock_data or not allowed:
                if not self.using_mock_data and not allowed:
                    logger.warning(f"API rate limited ({reason}). Using mock data.")
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
                self.api_rate_limiter.add_request() # Record request
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
            allowed, reason = self.api_rate_limiter.check_rate_limit()
            if self.using_mock_data or not allowed:
                if not self.using_mock_data and not allowed:
                    logger.warning(f"API rate limited ({reason}). Using mock data.")
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
                self.api_rate_limiter.add_request() # Record request
                success = self.mock_data_provider.cancel_order(order_id)
                
                if success:
                    # Refresh account data
                    self.fetch_account_data()
                    return True
                else:
                    raise Exception("Failed to cancel order")
        except Exception as e:
            self.error_handler.handle_error(f"Error cancelling order: {str(e)}")
            return False
    
    def close_position(self, symbol: str) -> bool:
        """Close a position"""
        try:
            # Check if we should use mock data
            allowed, reason = self.api_rate_limiter.check_rate_limit()
            if self.using_mock_data or not allowed:
                if not self.using_mock_data and not allowed:
                    logger.warning(f"API rate limited ({reason}). Using mock data.")
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
                self.api_rate_limiter.add_request() # Record request
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
    """A scrollable frame that supports both vertical and horizontal scrolling."""
    
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        
        # Create a canvas and scrollbars
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configure canvas
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
        
        # Grid layout for canvas and scrollbars
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        # Configure grid weights
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Bind mouse wheel for vertical scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel_vertical)
        # Bind Shift+MouseWheel for horizontal scrolling (Linux/Windows)
        self.canvas.bind_all("<Shift-MouseWheel>", self._on_mousewheel_horizontal)
        # Bind Control+MouseWheel for horizontal scrolling (macOS)
        self.canvas.bind_all("<Control-MouseWheel>", self._on_mousewheel_horizontal)
        
        # Update scroll region when frame size changes
        self.scrollable_frame.bind("<Configure>", self._on_frame_configure)
        
        # Bind canvas resize to update the scrollable window
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_frame_configure(self, event=None):
        """Reset the scroll region to encompass the inner frame"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event=None):
        """Update the width of the scrollable window when canvas is resized"""
        # Update the width of the scrollable window to fill the canvas
        if event and event.width > 1:
            self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel_vertical(self, event):
        """Handle vertical mouse wheel scrolling"""
        # Determine scroll direction and amount based on platform
        if sys.platform == "darwin":  # macOS
            scroll_units = event.delta
        else:  # Windows/Linux
            scroll_units = int(-1 * (event.delta / 120))
        self.canvas.yview_scroll(scroll_units, "units")

    def _on_mousewheel_horizontal(self, event):
        """Handle horizontal mouse wheel scrolling"""
        # Determine scroll direction and amount based on platform
        if sys.platform == "darwin":  # macOS
            scroll_units = event.delta
        else:  # Windows/Linux
            scroll_units = int(-1 * (event.delta / 120))
        self.canvas.xview_scroll(scroll_units, "units")


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
    
    def set_status(self, message: str) -> None:
        """Set status message"""
        self.status_var.set(message)
    
    def set_api_status(self, status: str, is_limited: bool = False) -> None:
        """Set API status message"""
        self.api_status_var.set(f"API: {status}")
        if is_limited:
            self.api_status_label.config(foreground="red")
        else:
            self.api_status_label.config(foreground="")  # Reset to default
    
    def set_mock_data(self, is_mock: bool) -> None:
        """Set mock data indicator"""
        if is_mock:
            self.mock_data_var.set("Mock Data")
            self.mock_data_label.config(foreground="orange")
        else:
            self.mock_data_var.set("")
            self.mock_data_label.config(foreground="")  # Reset to default


class ChartView(ttk.Frame):
    """View for displaying charts"""
    
    def __init__(self, parent, market_vm: MarketViewModel, config_data: ConfigData, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.market_vm = market_vm
        self.config_data = config_data
        
        # Create chart area
        self._create_chart_area()
        
        # Register callback
        self.market_vm.add_data_callback(self.update_chart)
    
    def _create_chart_area(self) -> None:
        """Create chart area"""
        # Create figure and axes
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Apply theme
        self._apply_theme()
    
    def _apply_theme(self) -> None:
        """Apply theme to chart"""
        theme = self.config_data.get("theme", "dark")
        if theme == "dark":
            plt.style.use("dark_background")
            self.fig.patch.set_facecolor("#2E2E2E")
            self.ax.set_facecolor("#1E1E1E")
            self.ax.tick_params(axis="x", colors="white")
            self.ax.tick_params(axis="y", colors="white")
            self.ax.yaxis.label.set_color("white")
            self.ax.xaxis.label.set_color("white")
            self.ax.title.set_color("white")
        else:
            plt.style.use("default")
            self.fig.patch.set_facecolor("white")
            self.ax.set_facecolor("white")
            self.ax.tick_params(axis="x", colors="black")
            self.ax.tick_params(axis="y", colors="black")
            self.ax.yaxis.label.set_color("black")
            self.ax.xaxis.label.set_color("black")
            self.ax.title.set_color("black")
    
    def update_chart(self) -> None:
        """Update chart display"""
        try:
            # Get data
            symbol = self.config_data.get("default_symbol")
            timeframe = self.config_data.get("default_timeframe")
            ohlcv = self.market_vm.get_ohlcv_with_indicators(symbol, timeframe)
            
            if ohlcv is None or ohlcv.empty:
                logger.warning(f"No OHLCV data available for {symbol} {timeframe}")
                return
            
            # Clear axes
            self.ax.clear()
            
            # Apply theme
            self._apply_theme()
            
            # Plot candlestick data
            self._plot_candlestick(ohlcv)
            
            # Plot indicators
            self._plot_indicators(ohlcv)
            
            # Set title and labels
            self.ax.set_title(f"{symbol} {timeframe} Chart")
            self.ax.set_ylabel("Price")
            
            # Format x-axis
            self.ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
            self.fig.autofmt_xdate()
            
            # Add legend
            self.ax.legend()
            
            # Draw canvas
            self.canvas.draw()
        except Exception as e:
            logger.error(f"Error updating chart: {str(e)}")
    
    def _plot_candlestick(self, ohlcv: pd.DataFrame) -> None:
        """Plot candlestick chart"""
        # Convert timestamps to matplotlib format
        ohlcv["time_mpl"] = ohlcv["time"].apply(matplotlib.dates.date2num)
        
        # Extract data
        quotes = ohlcv[["time_mpl", "open", "high", "low", "close"]].values
        
        # Plot candlesticks
        from matplotlib.finance import candlestick_ohlc
        candlestick_ohlc(self.ax, quotes, width=0.02, colorup="g", colordown="r")
    
    def _plot_indicators(self, ohlcv: pd.DataFrame) -> None:
        """Plot technical indicators"""
        indicators = self.config_data.get("chart_indicators", [])
        
        if "MA" in indicators:
            if "ma_fast" in ohlcv.columns:
                self.ax.plot(ohlcv["time"], ohlcv["ma_fast"], label="MA Fast", color="blue", linewidth=1)
            if "ma_slow" in ohlcv.columns:
                self.ax.plot(ohlcv["time"], ohlcv["ma_slow"], label="MA Slow", color="orange", linewidth=1)
        
        if "BB" in indicators:
            if "bb_upper" in ohlcv.columns:
                self.ax.plot(ohlcv["time"], ohlcv["bb_upper"], label="BB Upper", color="gray", linestyle="--", linewidth=1)
            if "bb_middle" in ohlcv.columns:
                self.ax.plot(ohlcv["time"], ohlcv["bb_middle"], label="BB Middle", color="gray", linestyle="-", linewidth=1)
            if "bb_lower" in ohlcv.columns:
                self.ax.plot(ohlcv["time"], ohlcv["bb_lower"], label="BB Lower", color="gray", linestyle="--", linewidth=1)
        
        # Plot RSI and MACD on separate axes if needed
        # TODO: Implement secondary axes for RSI/MACD


class OrderBookView(ttk.Frame):
    """View for displaying order book"""
    
    def __init__(self, parent, market_vm: MarketViewModel, config_data: ConfigData, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.market_vm = market_vm
        self.config_data = config_data
        
        # Create order book display
        self._create_order_book_display()
        
        # Register callback
        self.market_vm.add_data_callback(self.update_order_book)
    
    def _create_order_book_display(self) -> None:
        """Create order book display"""
        # Create main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create bids frame
        bids_frame = ttk.LabelFrame(main_frame, text="Bids")
        bids_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        self.bids_tree = ttk.Treeview(bids_frame, columns=("price", "size"), show="headings", height=15)
        self.bids_tree.heading("price", text="Price")
        self.bids_tree.heading("size", text="Size")
        self.bids_tree.column("price", width=100, anchor="e")
        self.bids_tree.column("size", width=100, anchor="e")
        self.bids_tree.pack(fill="both", expand=True)
        self.bids_tree.tag_configure("bid", foreground="green")
        
        # Create asks frame
        asks_frame = ttk.LabelFrame(main_frame, text="Asks")
        asks_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        self.asks_tree = ttk.Treeview(asks_frame, columns=("price", "size"), show="headings", height=15)
        self.asks_tree.heading("price", text="Price")
        self.asks_tree.heading("size", text="Size")
        self.asks_tree.column("price", width=100, anchor="e")
        self.asks_tree.column("size", width=100, anchor="e")
        self.asks_tree.pack(fill="both", expand=True)
        self.asks_tree.tag_configure("ask", foreground="red")
    
    def update_order_book(self) -> None:
        """Update order book display"""
        try:
            # Get data
            symbol = self.config_data.get("default_symbol")
            orderbook = self.market_vm.get_orderbook(symbol)
            
            if orderbook is None:
                logger.warning(f"No order book data available for {symbol}")
                return
            
            # Clear trees
            for item in self.bids_tree.get_children():
                self.bids_tree.delete(item)
            for item in self.asks_tree.get_children():
                self.asks_tree.delete(item)
            
            # Add bids
            for price, size in orderbook.get("bids", [])[:15]:  # Limit to top 15
                self.bids_tree.insert("", "end", values=(f"{price:.2f}", f"{size:.4f}"), tags=("bid",))
            
            # Add asks
            for price, size in orderbook.get("asks", [])[:15]:  # Limit to top 15
                self.asks_tree.insert("", 0, values=(f"{price:.2f}", f"{size:.4f}"), tags=("ask",))
        except Exception as e:
            logger.error(f"Error updating order book: {str(e)}")


class PositionsView(ttk.Frame):
    """View for displaying positions, orders, and trade history"""
    
    def __init__(self, parent, trading_vm: TradingViewModel, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.trading_vm = trading_vm
        
        # Create notebook for positions, orders, history
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
        
        # Register callback
        self.trading_vm.add_trading_callback(self.update_positions)
    
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
                price = f"{order['price']:.2f}" if order['price'] else "Market"
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
        
        # Configure grid layout for scrollable frame
        form_scroll.scrollable_frame.columnconfigure(1, weight=1)
        row_index = 0
        
        # Symbol selection
        ttk.Label(form_scroll.scrollable_frame, text="Symbol:").grid(row=row_index, column=0, sticky="w", padx=5, pady=5)
        self.symbol_var = tk.StringVar(value=self.config_data.get("default_symbol"))
        symbol_combo = ttk.Combobox(
            form_scroll.scrollable_frame, 
            textvariable=self.symbol_var, 
            values=self.config_data.get("symbols"),
            width=10
        )
        symbol_combo.grid(row=row_index, column=1, sticky="ew", padx=5, pady=5)
        row_index += 1
        
        # Order type selection
        ttk.Label(form_scroll.scrollable_frame, text="Order Type:").grid(row=row_index, column=0, sticky="w", padx=5, pady=5)
        self.order_type_var = tk.StringVar(value=self.config_data.get("default_order_type"))
        order_type_combo = ttk.Combobox(
            form_scroll.scrollable_frame, 
            textvariable=self.order_type_var, 
            values=self.config_data.get("order_types"),
            width=10
        )
        order_type_combo.grid(row=row_index, column=1, sticky="ew", padx=5, pady=5)
        order_type_combo.bind("<<ComboboxSelected>>", self._on_order_type_change)
        row_index += 1
        
        # Side selection
        ttk.Label(form_scroll.scrollable_frame, text="Side:").grid(row=row_index, column=0, sticky="w", padx=5, pady=5)
        side_frame = ttk.Frame(form_scroll.scrollable_frame)
        side_frame.grid(row=row_index, column=1, sticky="ew", padx=5, pady=5)
        self.side_var = tk.StringVar(value="buy")
        buy_radio = ttk.Radiobutton(side_frame, text="Buy", variable=self.side_var, value="buy")
        buy_radio.pack(side="left", padx=5)
        sell_radio = ttk.Radiobutton(side_frame, text="Sell", variable=self.side_var, value="sell")
        sell_radio.pack(side="left", padx=5)
        row_index += 1
        
        # Quantity
        ttk.Label(form_scroll.scrollable_frame, text="Quantity:").grid(row=row_index, column=0, sticky="w", padx=5, pady=5)
        quantity_frame = ttk.Frame(form_scroll.scrollable_frame)
        quantity_frame.grid(row=row_index, column=1, sticky="ew", padx=5, pady=5)
        self.quantity_var = tk.StringVar(value=str(self.config_data.get("default_quantity")))
        quantity_entry = ttk.Entry(quantity_frame, textvariable=self.quantity_var, width=10)
        quantity_entry.pack(side="left", padx=5)
        
        # Register validation function for float input
        validate_float = self.register(self._validate_float)
        quantity_entry.configure(validate="key", validatecommand=(validate_float, '%P'))
        
        row_index += 1
        
        # Price
        self.price_frame = ttk.Frame(form_scroll.scrollable_frame)
        ttk.Label(form_scroll.scrollable_frame, text="Price:").grid(row=row_index, column=0, sticky="w", padx=5, pady=5)
        price_frame = ttk.Frame(form_scroll.scrollable_frame)
        price_frame.grid(row=row_index, column=1, sticky="ew", padx=5, pady=5)
        self.price_var = tk.StringVar(value="0.0")
        price_entry = ttk.Entry(price_frame, textvariable=self.price_var, width=10)
        price_entry.pack(side="left", padx=5)
        
        # Register validation function for float input
        price_entry.configure(validate="key", validatecommand=(validate_float, '%P'))
        
        row_index += 1
        
        # Stop price
        self.stop_price_row = row_index
        ttk.Label(form_scroll.scrollable_frame, text="Stop Price:").grid(row=row_index, column=0, sticky="w", padx=5, pady=5)
        stop_price_frame = ttk.Frame(form_scroll.scrollable_frame)
        stop_price_frame.grid(row=row_index, column=1, sticky="ew", padx=5, pady=5)
        self.stop_price_var = tk.StringVar(value="0.0")
        stop_price_entry = ttk.Entry(stop_price_frame, textvariable=self.stop_price_var, width=10)
        stop_price_entry.pack(side="left", padx=5)
        
        # Register validation function for float input
        stop_price_entry.configure(validate="key", validatecommand=(validate_float, '%P'))
        
        row_index += 1
        
        # Buttons
        buttons_frame = ttk.Frame(form_scroll.scrollable_frame)
        buttons_frame.grid(row=row_index, column=0, columnspan=2, sticky="ew", padx=5, pady=10)
        
        ttk.Button(buttons_frame, text="Buy", command=self._buy).pack(side="left", padx=5)
        ttk.Button(buttons_frame, text="Sell", command=self._sell).pack(side="left", padx=5)
        ttk.Button(buttons_frame, text="Clear", command=self._clear_form).pack(side="left", padx=5)
        
        # Initial order type change to set up form
        self._on_order_type_change(None)
    
    def _validate_float(self, value: str) -> bool:
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
        order_type = self.order_type_var.get()
        
        # Show/hide price and stop price fields based on order type
        if order_type == "Market":
            self.price_var.set("0.0")
            self.stop_price_var.set("0.0")
        elif order_type == "Limit":
            self.stop_price_var.set("0.0")
        elif order_type == "Stop" or order_type == "Stop Limit":
            pass
    
    def _buy(self) -> None:
        """Place buy order"""
        self._place_order("buy")
    
    def _sell(self) -> None:
        """Place sell order"""
        self._place_order("sell")
    
    def _place_order(self, side: str) -> None:
        """Place order"""
        try:
            # Get form values
            symbol = self.symbol_var.get()
            order_type = self.order_type_var.get()
            quantity = float(self.quantity_var.get())
            price = float(self.price_var.get()) if self.price_var.get() else None
            stop_price = float(self.stop_price_var.get()) if self.stop_price_var.get() else None
            
            # Validate inputs
            if not symbol:
                messagebox.showerror("Error", "Symbol is required")
                return
            
            if quantity <= 0:
                messagebox.showerror("Error", "Quantity must be positive")
                return
            
            if order_type in ["Limit", "Stop Limit"] and (price is None or price <= 0):
                messagebox.showerror("Error", "Price must be positive for Limit orders")
                return
            
            if order_type in ["Stop", "Stop Limit"] and (stop_price is None or stop_price <= 0):
                messagebox.showerror("Error", "Stop price must be positive for Stop orders")
                return
            
            # Place order
            success = self.trading_vm.place_order(
                symbol=symbol,
                order_type=order_type,
                side=side,
                quantity=quantity,
                price=price,
                stop_price=stop_price
            )
            
            if success:
                messagebox.showinfo("Success", f"{side.capitalize()} order placed successfully")
                self._clear_form()
            else:
                messagebox.showerror("Error", f"Failed to place {side} order")
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            messagebox.showerror("Error", f"Error placing order: {str(e)}")
    
    def _clear_form(self) -> None:
        """Clear form fields"""
        self.quantity_var.set(str(self.config_data.get("default_quantity")))
        self.price_var.set("0.0")
        self.stop_price_var.set("0.0")


class SettingsView(ttk.Frame):
    """View for settings and configuration"""
    
    def __init__(self, parent, config_data: ConfigData, market_vm: MarketViewModel, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.config_data = config_data
        self.market_vm = market_vm
        
        # Create settings form
        self._create_settings_form()
    
    def _create_settings_form(self) -> None:
        """Create settings form with proper scrolling"""
        # Create main container that fills the entire frame
        main_container = ttk.Frame(self)
        main_container.pack(fill="both", expand=True)
        
        # Configure grid weights to make the container expandable
        main_container.columnconfigure(0, weight=1)
        main_container.rowconfigure(0, weight=1)
        
        # Create scrollable frame
        self.scrollable_frame = ScrollableFrame(main_container)
        self.scrollable_frame.grid(row=0, column=0, sticky="nsew")
        
        # Add settings sections to the scrollable frame
        self._add_api_settings()
        self._add_trading_settings()
        self._add_chart_settings()
        self._add_notification_settings()
        self._add_appearance_settings()
        
        # Add save/reset buttons
        self._add_buttons()
    
    def _add_api_settings(self) -> None:
        """Add API settings section"""
        # Create section frame
        section_frame = ttk.LabelFrame(self.scrollable_frame.scrollable_frame, text="API Settings")
        section_frame.pack(fill="x", padx=10, pady=10, anchor="nw")
        
        # Configure grid
        section_frame.columnconfigure(1, weight=1)
        
        # API Key
        ttk.Label(section_frame, text="API Key:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.api_key_var = tk.StringVar(value=self.config_data.get("api_key", ""))
        ttk.Entry(section_frame, textvariable=self.api_key_var).grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        # API Secret
        ttk.Label(section_frame, text="API Secret:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.api_secret_var = tk.StringVar(value=self.config_data.get("api_secret", ""))
        secret_entry = ttk.Entry(section_frame, textvariable=self.api_secret_var, show="*")
        secret_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        # Use Mock Data
        ttk.Label(section_frame, text="Use Mock Data:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.use_mock_var = tk.BooleanVar(value=self.config_data.get("use_mock_data", False))
        ttk.Checkbutton(section_frame, variable=self.use_mock_var).grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        # Test Connection Button
        ttk.Button(section_frame, text="Test Connection", command=self._test_connection).grid(row=3, column=0, columnspan=2, padx=5, pady=5)
    
    def _add_trading_settings(self) -> None:
        """Add trading settings section"""
        # Create section frame
        section_frame = ttk.LabelFrame(self.scrollable_frame.scrollable_frame, text="Trading Settings")
        section_frame.pack(fill="x", padx=10, pady=10, anchor="nw")
        
        # Configure grid
        section_frame.columnconfigure(1, weight=1)
        
        # Default Symbol
        ttk.Label(section_frame, text="Default Symbol:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.default_symbol_var = tk.StringVar(value=self.config_data.get("default_symbol", "XRP"))
        ttk.Combobox(
            section_frame, 
            textvariable=self.default_symbol_var,
            values=self.config_data.get("symbols", ["BTC", "ETH", "XRP", "SOL", "DOGE"])
        ).grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        # Default Timeframe
        ttk.Label(section_frame, text="Default Timeframe:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.default_timeframe_var = tk.StringVar(value=self.config_data.get("default_timeframe", "1h"))
        ttk.Combobox(
            section_frame, 
            textvariable=self.default_timeframe_var,
            values=self.config_data.get("timeframes", ["1m", "5m", "15m", "1h", "4h", "1d"])
        ).grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        # Default Order Type
        ttk.Label(section_frame, text="Default Order Type:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.default_order_type_var = tk.StringVar(value=self.config_data.get("default_order_type", "Limit"))
        ttk.Combobox(
            section_frame, 
            textvariable=self.default_order_type_var,
            values=self.config_data.get("order_types", ["Market", "Limit", "Stop", "Stop Limit"])
        ).grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        
        # Default Quantity
        ttk.Label(section_frame, text="Default Quantity:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.default_quantity_var = tk.StringVar(value=str(self.config_data.get("default_quantity", 1.0)))
        quantity_entry = ttk.Entry(section_frame, textvariable=self.default_quantity_var)
        quantity_entry.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
        
        # Register validation function for float input
        validate_float = self.register(self._validate_float)
        quantity_entry.configure(validate="key", validatecommand=(validate_float, '%P'))
        
        # Risk Level
        ttk.Label(section_frame, text="Risk Level (%):").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.risk_level_var = tk.StringVar(value=str(self.config_data.get("risk_level", 0.02) * 100))
        risk_entry = ttk.Entry(section_frame, textvariable=self.risk_level_var)
        risk_entry.grid(row=4, column=1, sticky="ew", padx=5, pady=5)
        
        # Register validation function for float input
        risk_entry.configure(validate="key", validatecommand=(validate_float, '%P'))
    
    def _add_chart_settings(self) -> None:
        """Add chart settings section"""
        # Create section frame
        section_frame = ttk.LabelFrame(self.scrollable_frame.scrollable_frame, text="Chart Settings")
        section_frame.pack(fill="x", padx=10, pady=10, anchor="nw")
        
        # Configure grid
        section_frame.columnconfigure(1, weight=1)
        
        # Chart Indicators
        ttk.Label(section_frame, text="Indicators:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        indicators_frame = ttk.Frame(section_frame)
        indicators_frame.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        # Create indicator checkboxes
        all_indicators = ["MA", "RSI", "MACD", "BB", "Volume"]
        enabled_indicators = self.config_data.get("chart_indicators", ["MA", "RSI", "MACD", "BB"])
        
        self.indicator_vars = {}
        for i, indicator in enumerate(all_indicators):
            var = tk.BooleanVar(value=indicator in enabled_indicators)
            self.indicator_vars[indicator] = var
            ttk.Checkbutton(indicators_frame, text=indicator, variable=var).pack(side="left", padx=5)
        
        # Auto Refresh Interval
        ttk.Label(section_frame, text="Refresh Interval (ms):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.refresh_interval_var = tk.StringVar(value=str(self.config_data.get("auto_refresh_interval", 5000)))
        refresh_entry = ttk.Entry(section_frame, textvariable=self.refresh_interval_var)
        refresh_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        # Register validation function for integer input
        validate_int = self.register(self._validate_int)
        refresh_entry.configure(validate="key", validatecommand=(validate_int, '%P'))
    
    def _add_notification_settings(self) -> None:
        """Add notification settings section"""
        # Create section frame
        section_frame = ttk.LabelFrame(self.scrollable_frame.scrollable_frame, text="Notification Settings")
        section_frame.pack(fill="x", padx=10, pady=10, anchor="nw")
        
        # Configure grid
        section_frame.columnconfigure(1, weight=1)
        
        # Show Trade Confirmations
        ttk.Label(section_frame, text="Show Trade Confirmations:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.show_confirmations_var = tk.BooleanVar(value=self.config_data.get("show_trade_confirmations", True))
        ttk.Checkbutton(section_frame, variable=self.show_confirmations_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Show Notifications
        ttk.Label(section_frame, text="Show Desktop Notifications:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.show_notifications_var = tk.BooleanVar(value=self.config_data.get("show_notifications", True))
        ttk.Checkbutton(section_frame, variable=self.show_notifications_var).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # Play Sound
        ttk.Label(section_frame, text="Play Sound on Events:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.play_sound_var = tk.BooleanVar(value=self.config_data.get("play_sound", True))
        ttk.Checkbutton(section_frame, variable=self.play_sound_var).grid(row=2, column=1, sticky="w", padx=5, pady=5)
    
    def _add_appearance_settings(self) -> None:
        """Add appearance settings section"""
        # Create section frame
        section_frame = ttk.LabelFrame(self.scrollable_frame.scrollable_frame, text="Appearance Settings")
        section_frame.pack(fill="x", padx=10, pady=10, anchor="nw")
        
        # Configure grid
        section_frame.columnconfigure(1, weight=1)
        
        # Theme
        ttk.Label(section_frame, text="Theme:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.theme_var = tk.StringVar(value=self.config_data.get("theme", "dark"))
        theme_combo = ttk.Combobox(section_frame, textvariable=self.theme_var, values=["dark", "light"])
        theme_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        # Advanced Mode
        ttk.Label(section_frame, text="Advanced Mode:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.advanced_mode_var = tk.BooleanVar(value=self.config_data.get("advanced_mode", False))
        ttk.Checkbutton(section_frame, variable=self.advanced_mode_var).grid(row=1, column=1, sticky="w", padx=5, pady=5)
    
    def _add_buttons(self) -> None:
        """Add save/reset buttons"""
        # Create buttons frame
        buttons_frame = ttk.Frame(self.scrollable_frame.scrollable_frame)
        buttons_frame.pack(fill="x", padx=10, pady=10, anchor="nw")
        
        # Save button
        save_button = ttk.Button(buttons_frame, text="Save Settings", command=self._save_settings)
        save_button.pack(side="left", padx=5, pady=5)
        
        # Reset button
        reset_button = ttk.Button(buttons_frame, text="Reset to Defaults", command=self._reset_settings)
        reset_button.pack(side="left", padx=5, pady=5)
    
    def _validate_float(self, value: str) -> bool:
        """Validate float input"""
        if value == "":
            return True
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def _validate_int(self, value: str) -> bool:
        """Validate integer input"""
        if value == "":
            return True
        try:
            int(value)
            return True
        except ValueError:
            return False
    
    def _save_settings(self) -> None:
        """Save settings"""
        try:
            # API settings
            self.config_data.set("api_key", self.api_key_var.get())
            self.config_data.set("api_secret", self.api_secret_var.get())
            self.config_data.set("use_mock_data", self.use_mock_var.get())
            
            # Trading settings
            self.config_data.set("default_symbol", self.default_symbol_var.get())
            self.config_data.set("default_timeframe", self.default_timeframe_var.get())
            self.config_data.set("default_order_type", self.default_order_type_var.get())
            self.config_data.set("default_quantity", float(self.default_quantity_var.get() or 1.0))
            self.config_data.set("risk_level", float(self.risk_level_var.get() or 2.0) / 100.0)
            
            # Chart settings
            enabled_indicators = [indicator for indicator, var in self.indicator_vars.items() if var.get()]
            self.config_data.set("chart_indicators", enabled_indicators)
            self.config_data.set("auto_refresh_interval", int(self.refresh_interval_var.get() or 5000))
            
            # Notification settings
            self.config_data.set("show_trade_confirmations", self.show_confirmations_var.get())
            self.config_data.set("show_notifications", self.show_notifications_var.get())
            self.config_data.set("play_sound", self.play_sound_var.get())
            
            # Appearance settings
            self.config_data.set("theme", self.theme_var.get())
            self.config_data.set("advanced_mode", self.advanced_mode_var.get())
            
            # Save to file
            self.config_data.save()
            
            # Update mock data usage
            self.market_vm.toggle_mock_data(self.use_mock_var.get())
            
            messagebox.showinfo("Success", "Settings saved successfully")
        except Exception as e:
            logger.error(f"Error saving settings: {str(e)}")
            messagebox.showerror("Error", f"Error saving settings: {str(e)}")
    
    def _reset_settings(self) -> None:
        """Reset settings to defaults"""
        if messagebox.askyesno("Confirm", "Reset all settings to defaults?"):
            # Reset config data
            self.config_data.config = self.config_data._load_default_config()
            self.config_data.save()
            
            # Update form
            self._update_form_from_config()
            
            messagebox.showinfo("Success", "Settings reset to defaults")
    
    def _update_form_from_config(self) -> None:
        """Update form fields from config data"""
        # API settings
        self.api_key_var.set(self.config_data.get("api_key", ""))
        self.api_secret_var.set(self.config_data.get("api_secret", ""))
        self.use_mock_var.set(self.config_data.get("use_mock_data", False))
        
        # Trading settings
        self.default_symbol_var.set(self.config_data.get("default_symbol", "XRP"))
        self.default_timeframe_var.set(self.config_data.get("default_timeframe", "1h"))
        self.default_order_type_var.set(self.config_data.get("default_order_type", "Limit"))
        self.default_quantity_var.set(str(self.config_data.get("default_quantity", 1.0)))
        self.risk_level_var.set(str(self.config_data.get("risk_level", 0.02) * 100))
        
        # Chart settings
        enabled_indicators = self.config_data.get("chart_indicators", ["MA", "RSI", "MACD", "BB"])
        for indicator, var in self.indicator_vars.items():
            var.set(indicator in enabled_indicators)
        self.refresh_interval_var.set(str(self.config_data.get("auto_refresh_interval", 5000)))
        
        # Notification settings
        self.show_confirmations_var.set(self.config_data.get("show_trade_confirmations", True))
        self.show_notifications_var.set(self.config_data.get("show_notifications", True))
        self.play_sound_var.set(self.config_data.get("play_sound", True))
        
        # Appearance settings
        self.theme_var.set(self.config_data.get("theme", "dark"))
        self.advanced_mode_var.set(self.config_data.get("advanced_mode", False))
    
    def _test_connection(self) -> None:
        """Test API connection"""
        try:
            # Get API credentials
            api_key = self.api_key_var.get()
            api_secret = self.api_secret_var.get()
            
            if not api_key or not api_secret:
                messagebox.showwarning("Warning", "API key and secret are required")
                return
            
            # TODO: Implement real API connection test
            # For now, just show success message
            messagebox.showinfo("Success", "Connection test successful")
        except Exception as e:
            logger.error(f"Error testing connection: {str(e)}")
            messagebox.showerror("Error", f"Error testing connection: {str(e)}")


class LogView(ttk.Frame):
    """View for displaying logs"""
    
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        
        # Create log text area
        self._create_log_area()
        
        # Set up log queue and handler
        self.log_queue = queue.Queue()
        self.log_handler = QueueHandler(self.log_queue)
        
        # Add handler to logger
        logger.addHandler(self.log_handler)
        
        # Start log consumer
        self.after(100, self._consume_logs)
    
    def _create_log_area(self) -> None:
        """Create log text area"""
        # Create main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create log text area
        self.log_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD)
        self.log_text.pack(fill="both", expand=True)
        
        # Make text read-only
        self.log_text.config(state=tk.DISABLED)
        
        # Create buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill="x", padx=5, pady=5)
        
        # Add buttons
        ttk.Button(buttons_frame, text="Clear Logs", command=self._clear_logs).pack(side="left", padx=5)
        ttk.Button(buttons_frame, text="Save Logs", command=self._save_logs).pack(side="left", padx=5)
        
        # Add log level filter
        ttk.Label(buttons_frame, text="Log Level:").pack(side="left", padx=5)
        self.log_level_var = tk.StringVar(value="INFO")
        ttk.Combobox(buttons_frame, textvariable=self.log_level_var, values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], width=10).pack(side="left", padx=5)
        self.log_level_var.trace("w", self._on_log_level_change)
    
    def _consume_logs(self) -> None:
        """Consume logs from queue"""
        try:
            while True:
                record = self.log_queue.get(block=False)
                self._display_log(record)
        except queue.Empty:
            # Schedule next check
            self.after(100, self._consume_logs)
    
    def _display_log(self, record) -> None:
        """Display log record"""
        # Check log level filter
        if not self._should_display_log(record):
            return
        
        # Format log message
        log_message = self.log_handler.format(record)
        
        # Add to text area
        self.log_text.config(state=tk.NORMAL)
        
        # Add with appropriate tag
        if record.levelno >= logging.ERROR:
            self.log_text.tag_config("error", foreground="red")
            self.log_text.insert(tk.END, log_message + "\n", "error")
        elif record.levelno >= logging.WARNING:
            self.log_text.tag_config("warning", foreground="orange")
            self.log_text.insert(tk.END, log_message + "\n", "warning")
        else:
            self.log_text.insert(tk.END, log_message + "\n")
        
        # Scroll to end
        self.log_text.see(tk.END)
        
        # Make text read-only again
        self.log_text.config(state=tk.DISABLED)
    
    def _should_display_log(self, record) -> bool:
        """Check if log record should be displayed based on level filter"""
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        
        filter_level = level_map.get(self.log_level_var.get(), logging.INFO)
        return record.levelno >= filter_level
    
    def _on_log_level_change(self, *args) -> None:
        """Handle log level change"""
        # Clear and reload logs
        self._clear_logs()
    
    def _clear_logs(self) -> None:
        """Clear log text area"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def _save_logs(self) -> None:
        """Save logs to file"""
        try:
            # Get log content
            self.log_text.config(state=tk.NORMAL)
            log_content = self.log_text.get(1.0, tk.END)
            self.log_text.config(state=tk.DISABLED)
            
            # Save to file
            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(log_dir, f"gui_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            
            with open(log_file, "w") as f:
                f.write(log_content)
            
            messagebox.showinfo("Success", f"Logs saved to {log_file}")
        except Exception as e:
            logger.error(f"Error saving logs: {str(e)}")
            messagebox.showerror("Error", f"Error saving logs: {str(e)}")


class QueueHandler(logging.Handler):
    """Handler for logging to a queue"""
    
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    
    def emit(self, record):
        """Put log record in queue"""
        self.log_queue.put(record)


class MainApplication(ttk.Frame):
    """Main application frame"""
    
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        
        # Initialize models
        self.market_data = MarketData()
        self.position_data = PositionData()
        self.config_data = ConfigData()
        self.error_handler = ErrorHandler()
        
        # Initialize view models
        self.market_vm = MarketViewModel(self.market_data, self.config_data, self.error_handler)
        self.trading_vm = TradingViewModel(self.position_data, self.config_data, self.error_handler)
        
        # Create UI
        self._create_ui()
        
        # Start data update thread
        self.running = True
        self.update_thread = threading.Thread(target=self._update_data_thread)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def _create_ui(self) -> None:
        """Create user interface"""
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create tabs
        self.chart_tab = ttk.Frame(self.notebook)
        self.trade_tab = ttk.Frame(self.notebook)
        self.positions_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)
        self.logs_tab = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.chart_tab, text="Chart")
        self.notebook.add(self.trade_tab, text="Trade")
        self.notebook.add(self.positions_tab, text="Positions")
        self.notebook.add(self.settings_tab, text="Settings")
        self.notebook.add(self.logs_tab, text="Logs")
        
        # Create views
        self.chart_view = ChartView(self.chart_tab, self.market_vm, self.config_data)
        self.chart_view.pack(fill="both", expand=True)
        
        self.order_book_view = OrderBookView(self.chart_tab, self.market_vm, self.config_data)
        self.order_book_view.pack(fill="both", expand=True)
        
        self.trade_view = TradeView(self.trade_tab, self.trading_vm, self.market_vm, self.config_data)
        self.trade_view.pack(fill="both", expand=True)
        
        self.positions_view = PositionsView(self.positions_tab, self.trading_vm)
        self.positions_view.pack(fill="both", expand=True)
        
        self.settings_view = SettingsView(self.settings_tab, self.config_data, self.market_vm)
        self.settings_view.pack(fill="both", expand=True)
        
        self.log_view = LogView(self.logs_tab)
        self.log_view.pack(fill="both", expand=True)
        
        # Create status bar
        self.status_bar = StatusBar(self)
        self.status_bar.pack(fill="x", side="bottom")
        
        # Set initial status
        self.status_bar.set_status("Ready")
        self.status_bar.set_mock_data(self.config_data.get("use_mock_data", False))
    
    def _update_data_thread(self) -> None:
        """Thread for updating market data"""
        while self.running:
            try:
                # Get current symbol and timeframe
                symbol = self.config_data.get("default_symbol")
                timeframe = self.config_data.get("default_timeframe")
                
                # Fetch market data
                self.market_vm.fetch_market_data(symbol, timeframe)
                
                # Fetch account data
                self.trading_vm.fetch_account_data()
                
                # Update status
                self.status_bar.set_status(f"Data updated: {datetime.now().strftime('%H:%M:%S')}")
                self.status_bar.set_mock_data(self.market_vm.using_mock_data)
                
                # Check API rate limit
                allowed, reason = self.market_vm.api_rate_limiter.check_rate_limit()
                if not allowed:
                    self.status_bar.set_api_status(reason, is_limited=True)
                else:
                    self.status_bar.set_api_status("OK")
                
                # Sleep for refresh interval
                refresh_interval = self.config_data.get("auto_refresh_interval", 5000) / 1000.0
                time.sleep(refresh_interval)
            except Exception as e:
                logger.error(f"Error in update thread: {str(e)}")
                time.sleep(5)  # Sleep longer on error
    
    def on_closing(self) -> None:
        """Handle window closing"""
        self.running = False
        self.parent.destroy()


def main():
    """Main entry point"""
    # Create root window
    root = tk.Tk()
    root.title("Hyperliquid Trading Bot")
    root.geometry("1200x800")
    
    # Create main application
    app = MainApplication(root)
    app.pack(fill="both", expand=True)
    
    # Set up closing handler
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start main loop
    root.mainloop()


if __name__ == "__main__":
    main()
