"""
Tabbed GUI Main Application for Hyperliquid Trading Bot

This module provides a tabbed GUI interface for the Hyperliquid Trading Bot,
with support for trading, monitoring, configuration, and visualization.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import logging
import threading
import time
import os
import json
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np

# Import from local modules
from core.api_rate_limiter import APIRateLimiter
from core.error_handling import ErrorHandler
from data.enhanced_mock_data_provider import EnhancedMockDataProvider
from strategies.signal_generator import RobustSignalGenerator
from strategies.master_strategy import MasterOmniOverlordStrategy
from utils.technical_indicators import calculate_indicators
from gui.order_book_visualizer import OrderBookVisualizer
from gui.chart_visualizer import ChartVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/gui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TabbedGUI(tk.Tk):
    """
    Tabbed GUI for Hyperliquid Trading Bot.
    
    Provides a comprehensive interface with tabs for:
    - Trading: Execute trades and monitor positions
    - Market Data: View real-time market data and charts
    - Configuration: Configure bot settings
    - Logs: View application logs
    """
    
    def __init__(self):
        """Initialize the GUI."""
        super().__init__()
        
        self.title("Hyperliquid Trading Bot")
        self.geometry("1200x800")
        
        # Initialize components
        self.api_rate_limiter = APIRateLimiter()
        self.error_handler = ErrorHandler()
        self.mock_data_provider = EnhancedMockDataProvider()
        self.signal_generator = RobustSignalGenerator()
        self.strategy = MasterOmniOverlordStrategy()
        
        # Initialize data
        self.market_data = {}
        self.positions = {}
        self.orders = {}
        self.config = self._load_config()
        
        # Initialize UI
        self._init_ui()
        
        # Start background threads
        self.running = True
        self.update_thread = threading.Thread(target=self._update_data_thread)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        logger.info("GUI initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        config_path = os.path.join("config", "mode_settings.json")
        
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    return json.load(f)
            else:
                logger.warning(f"Config file not found: {config_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _save_config(self):
        """Save configuration to file."""
        config_path = os.path.join("config", "mode_settings.json")
        
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Config saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.trading_tab = ttk.Frame(self.notebook)
        self.market_data_tab = ttk.Frame(self.notebook)
        self.config_tab = ttk.Frame(self.notebook)
        self.logs_tab = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.trading_tab, text="Trading")
        self.notebook.add(self.market_data_tab, text="Market Data")
        self.notebook.add(self.config_tab, text="Configuration")
        self.notebook.add(self.logs_tab, text="Logs")
        
        # Initialize tab contents
        self._init_trading_tab()
        self._init_market_data_tab()
        self._init_config_tab()
        self._init_logs_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _init_trading_tab(self):
        """Initialize the trading tab."""
        # Create frames
        control_frame = ttk.LabelFrame(self.trading_tab, text="Trading Controls")
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        positions_frame = ttk.LabelFrame(self.trading_tab, text="Positions")
        positions_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        orders_frame = ttk.LabelFrame(self.trading_tab, text="Orders")
        orders_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Trading controls
        controls = ttk.Frame(control_frame)
        controls.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(controls, text="Symbol:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.symbol_var = tk.StringVar(value="XRP-PERP")
        ttk.Combobox(controls, textvariable=self.symbol_var, values=["XRP-PERP", "BTC-PERP", "ETH-PERP"]).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(controls, text="Quantity:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.quantity_var = tk.StringVar(value="1.0")
        ttk.Entry(controls, textvariable=self.quantity_var, width=10).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(controls, text="Price:").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        self.price_var = tk.StringVar(value="0.0")
        ttk.Entry(controls, textvariable=self.price_var, width=10).grid(row=0, column=5, padx=5, pady=5, sticky=tk.W)
        
        ttk.Button(controls, text="Buy", command=self._buy).grid(row=0, column=6, padx=5, pady=5)
        ttk.Button(controls, text="Sell", command=self._sell).grid(row=0, column=7, padx=5, pady=5)
        ttk.Button(controls, text="Cancel All", command=self._cancel_all).grid(row=0, column=8, padx=5, pady=5)
        
        # Positions table
        positions_frame_inner = ttk.Frame(positions_frame)
        positions_frame_inner.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        positions_columns = ("Symbol", "Side", "Quantity", "Entry Price", "Current Price", "PnL", "PnL %")
        self.positions_tree = ttk.Treeview(positions_frame_inner, columns=positions_columns, show="headings")
        
        for col in positions_columns:
            self.positions_tree.heading(col, text=col)
            self.positions_tree.column(col, width=100, anchor=tk.CENTER)
        
        positions_scrollbar = ttk.Scrollbar(positions_frame_inner, orient=tk.VERTICAL, command=self.positions_tree.yview)
        self.positions_tree.configure(yscrollcommand=positions_scrollbar.set)
        
        self.positions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        positions_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Orders table
        orders_frame_inner = ttk.Frame(orders_frame)
        orders_frame_inner.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        orders_columns = ("ID", "Symbol", "Side", "Type", "Quantity", "Price", "Status", "Time")
        self.orders_tree = ttk.Treeview(orders_frame_inner, columns=orders_columns, show="headings")
        
        for col in orders_columns:
            self.orders_tree.heading(col, text=col)
            self.orders_tree.column(col, width=100, anchor=tk.CENTER)
        
        orders_scrollbar = ttk.Scrollbar(orders_frame_inner, orient=tk.VERTICAL, command=self.orders_tree.yview)
        self.orders_tree.configure(yscrollcommand=orders_scrollbar.set)
        
        self.orders_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        orders_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _init_market_data_tab(self):
        """Initialize the market data tab."""
        # Create frames
        chart_frame = ttk.LabelFrame(self.market_data_tab, text="Price Chart")
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        order_book_frame = ttk.LabelFrame(self.market_data_tab, text="Order Book")
        order_book_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Chart
        self.chart_visualizer = ChartVisualizer(chart_frame)
        
        # Order book
        self.order_book_visualizer = OrderBookVisualizer(order_book_frame)
    
    def _init_config_tab(self):
        """Initialize the configuration tab."""
        # Create frames
        settings_frame = ttk.LabelFrame(self.config_tab, text="Settings")
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create scrollable frame for settings
        settings_canvas = tk.Canvas(settings_frame)
        settings_scrollbar = ttk.Scrollbar(settings_frame, orient=tk.VERTICAL, command=settings_canvas.yview)
        settings_scrollable_frame = ttk.Frame(settings_canvas)
        
        settings_scrollable_frame.bind(
            "<Configure>",
            lambda e: settings_canvas.configure(scrollregion=settings_canvas.bbox("all"))
        )
        
        settings_canvas.create_window((0, 0), window=settings_scrollable_frame, anchor=tk.NW)
        settings_canvas.configure(yscrollcommand=settings_scrollbar.set)
        
        settings_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        settings_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # API settings
        api_frame = ttk.LabelFrame(settings_scrollable_frame, text="API Settings")
        api_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(api_frame, text="API Key:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.api_key_var = tk.StringVar(value=self.config.get("api_key", ""))
        ttk.Entry(api_frame, textvariable=self.api_key_var, width=40).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(api_frame, text="API Secret:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.api_secret_var = tk.StringVar(value=self.config.get("api_secret", ""))
        ttk.Entry(api_frame, textvariable=self.api_secret_var, width=40, show="*").grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(api_frame, text="Use Mock Data:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.use_mock_var = tk.BooleanVar(value=self.config.get("use_mock", False))
        ttk.Checkbutton(api_frame, variable=self.use_mock_var).grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Trading settings
        trading_frame = ttk.LabelFrame(settings_scrollable_frame, text="Trading Settings")
        trading_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(trading_frame, text="Trading Mode:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.trading_mode_var = tk.StringVar(value=self.config.get("trading_mode", "Paper"))
        ttk.Combobox(trading_frame, textvariable=self.trading_mode_var, values=["Paper", "Live", "Monitor"]).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(trading_frame, text="Risk Level:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.risk_level_var = tk.StringVar(value=self.config.get("risk_level", "Conservative"))
        ttk.Combobox(trading_frame, textvariable=self.risk_level_var, values=["Conservative", "Moderate", "Aggressive"]).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(trading_frame, text="Max Position Size:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.max_position_var = tk.StringVar(value=str(self.config.get("max_position_size", 1.0)))
        ttk.Entry(trading_frame, textvariable=self.max_position_var, width=10).grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(trading_frame, text="Stop Loss %:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.stop_loss_var = tk.StringVar(value=str(self.config.get("stop_loss_percent", 5.0)))
        ttk.Entry(trading_frame, textvariable=self.stop_loss_var, width=10).grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(trading_frame, text="Take Profit %:").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        self.take_profit_var = tk.StringVar(value=str(self.config.get("take_profit_percent", 10.0)))
        ttk.Entry(trading_frame, textvariable=self.take_profit_var, width=10).grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Strategy settings
        strategy_frame = ttk.LabelFrame(settings_scrollable_frame, text="Strategy Settings")
        strategy_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(strategy_frame, text="Strategy:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.strategy_var = tk.StringVar(value=self.config.get("strategy", "MasterOmniOverlord"))
        ttk.Combobox(strategy_frame, textvariable=self.strategy_var, values=["MasterOmniOverlord", "RobustSignal", "TrendFollowing", "MeanReversion"]).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(strategy_frame, text="Timeframe:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.timeframe_var = tk.StringVar(value=self.config.get("timeframe", "1h"))
        ttk.Combobox(strategy_frame, textvariable=self.timeframe_var, values=["1m", "5m", "15m", "1h", "4h", "1d"]).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Buttons
        buttons_frame = ttk.Frame(settings_scrollable_frame)
        buttons_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(buttons_frame, text="Save", command=self._save_settings).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(buttons_frame, text="Reset", command=self._reset_settings).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(buttons_frame, text="Test Connection", command=self._test_connection).pack(side=tk.LEFT, padx=5, pady=5)
    
    def _init_logs_tab(self):
        """Initialize the logs tab."""
        # Create log text area
        log_frame = ttk.Frame(self.logs_tab)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Add log handler
        self.log_handler = GUILogHandler(self.log_text)
        logging.getLogger().addHandler(self.log_handler)
    
    def _buy(self):
        """Execute buy order."""
        try:
            symbol = self.symbol_var.get()
            quantity = float(self.quantity_var.get())
            price = float(self.price_var.get()) if self.price_var.get() else None
            
            # Validate inputs
            if not symbol:
                messagebox.showerror("Error", "Symbol is required")
                return
            
            if quantity <= 0:
                messagebox.showerror("Error", "Quantity must be positive")
                return
            
            # Execute order
            order_id = f"order_{int(time.time())}"
            order_type = "MARKET" if price is None else "LIMIT"
            
            self.orders[order_id] = {
                "id": order_id,
                "symbol": symbol,
                "side": "BUY",
                "type": order_type,
                "quantity": quantity,
                "price": price,
                "status": "PENDING",
                "time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Update UI
            self._update_orders_table()
            
            # Log
            logger.info(f"Buy order placed: {symbol}, {quantity}, {price}")
            self.status_var.set(f"Buy order placed: {symbol}, {quantity}, {price}")
            
            # Simulate order execution
            threading.Thread(target=self._simulate_order_execution, args=(order_id,)).start()
        except Exception as e:
            logger.error(f"Error placing buy order: {e}")
            messagebox.showerror("Error", f"Error placing buy order: {e}")
    
    def _sell(self):
        """Execute sell order."""
        try:
            symbol = self.symbol_var.get()
            quantity = float(self.quantity_var.get())
            price = float(self.price_var.get()) if self.price_var.get() else None
            
            # Validate inputs
            if not symbol:
                messagebox.showerror("Error", "Symbol is required")
                return
            
            if quantity <= 0:
                messagebox.showerror("Error", "Quantity must be positive")
                return
            
            # Execute order
            order_id = f"order_{int(time.time())}"
            order_type = "MARKET" if price is None else "LIMIT"
            
            self.orders[order_id] = {
                "id": order_id,
                "symbol": symbol,
                "side": "SELL",
                "type": order_type,
                "quantity": quantity,
                "price": price,
                "status": "PENDING",
                "time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Update UI
            self._update_orders_table()
            
            # Log
            logger.info(f"Sell order placed: {symbol}, {quantity}, {price}")
            self.status_var.set(f"Sell order placed: {symbol}, {quantity}, {price}")
            
            # Simulate order execution
            threading.Thread(target=self._simulate_order_execution, args=(order_id,)).start()
        except Exception as e:
            logger.error(f"Error placing sell order: {e}")
            messagebox.showerror("Error", f"Error placing sell order: {e}")
    
    def _cancel_all(self):
        """Cancel all open orders."""
        try:
            # Cancel orders
            for order_id, order in list(self.orders.items()):
                if order["status"] in ["PENDING", "OPEN"]:
                    order["status"] = "CANCELED"
            
            # Update UI
            self._update_orders_table()
            
            # Log
            logger.info("All orders canceled")
            self.status_var.set("All orders canceled")
        except Exception as e:
            logger.error(f"Error canceling orders: {e}")
            messagebox.showerror("Error", f"Error canceling orders: {e}")
    
    def _simulate_order_execution(self, order_id):
        """
        Simulate order execution.
        
        Args:
            order_id: Order ID
        """
        try:
            # Get order
            order = self.orders.get(order_id)
            if not order:
                return
            
            # Update status
            order["status"] = "OPEN"
            self._update_orders_table()
            
            # Simulate execution delay
            time.sleep(2)
            
            # Execute order
            order["status"] = "FILLED"
            self._update_orders_table()
            
            # Update position
            symbol = order["symbol"]
            side = order["side"]
            quantity = order["quantity"]
            price = order["price"] or self._get_market_price(symbol)
            
            if symbol not in self.positions:
                self.positions[symbol] = {
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "entry_price": price,
                    "current_price": price
                }
            else:
                position = self.positions[symbol]
                
                if side == position["side"]:
                    # Increase position
                    new_quantity = position["quantity"] + quantity
                    new_entry_price = (position["entry_price"] * position["quantity"] + price * quantity) / new_quantity
                    
                    position["quantity"] = new_quantity
                    position["entry_price"] = new_entry_price
                else:
                    # Decrease position
                    new_quantity = position["quantity"] - quantity
                    
                    if new_quantity <= 0:
                        # Close position
                        del self.positions[symbol]
                    else:
                        # Update position
                        position["quantity"] = new_quantity
            
            # Update UI
            self._update_positions_table()
            
            # Log
            logger.info(f"Order executed: {order_id}")
            self.status_var.set(f"Order executed: {order_id}")
        except Exception as e:
            logger.error(f"Error simulating order execution: {e}")
    
    def _get_market_price(self, symbol):
        """
        Get market price for symbol.
        
        Args:
            symbol: Symbol
        
        Returns:
            Market price
        """
        # Simulate market price
        if symbol == "XRP-PERP":
            return 0.5 + 0.1 * np.random.random()
        elif symbol == "BTC-PERP":
            return 50000 + 1000 * np.random.random()
        elif symbol == "ETH-PERP":
            return 3000 + 100 * np.random.random()
        else:
            return 1.0
    
    def _update_positions_table(self):
        """Update positions table."""
        # Clear table
        for item in self.positions_tree.get_children():
            self.positions_tree.delete(item)
        
        # Add positions
        for position in self.positions.values():
            symbol = position["symbol"]
            side = position["side"]
            quantity = position["quantity"]
            entry_price = position["entry_price"]
            current_price = position["current_price"]
            
            # Calculate PnL
            if side == "BUY":
                pnl = (current_price - entry_price) * quantity
                pnl_percent = (current_price - entry_price) / entry_price * 100
            else:
                pnl = (entry_price - current_price) * quantity
                pnl_percent = (entry_price - current_price) / entry_price * 100
            
            # Add to table
            self.positions_tree.insert(
                "",
                tk.END,
                values=(
                    symbol,
                    side,
                    f"{quantity:.4f}",
                    f"{entry_price:.4f}",
                    f"{current_price:.4f}",
                    f"{pnl:.4f}",
                    f"{pnl_percent:.2f}%"
                )
            )
    
    def _update_orders_table(self):
        """Update orders table."""
        # Clear table
        for item in self.orders_tree.get_children():
            self.orders_tree.delete(item)
        
        # Add orders
        for order in self.orders.values():
            # Add to table
            self.orders_tree.insert(
                "",
                tk.END,
                values=(
                    order["id"],
                    order["symbol"],
                    order["side"],
                    order["type"],
                    f"{order['quantity']:.4f}",
                    f"{order['price']:.4f}" if order["price"] else "MARKET",
                    order["status"],
                    order["time"]
                )
            )
    
    def _save_settings(self):
        """Save settings."""
        try:
            # Update config
            self.config["api_key"] = self.api_key_var.get()
            self.config["api_secret"] = self.api_secret_var.get()
            self.config["use_mock"] = self.use_mock_var.get()
            self.config["trading_mode"] = self.trading_mode_var.get()
            self.config["risk_level"] = self.risk_level_var.get()
            self.config["max_position_size"] = float(self.max_position_var.get())
            self.config["stop_loss_percent"] = float(self.stop_loss_var.get())
            self.config["take_profit_percent"] = float(self.take_profit_var.get())
            self.config["strategy"] = self.strategy_var.get()
            self.config["timeframe"] = self.timeframe_var.get()
            
            # Save config
            self._save_config()
            
            # Log
            logger.info("Settings saved")
            self.status_var.set("Settings saved")
            
            # Show message
            messagebox.showinfo("Success", "Settings saved successfully")
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            messagebox.showerror("Error", f"Error saving settings: {e}")
    
    def _reset_settings(self):
        """Reset settings to defaults."""
        try:
            # Confirm
            if not messagebox.askyesno("Confirm", "Are you sure you want to reset all settings to defaults?"):
                return
            
            # Reset config
            self.config = {
                "api_key": "",
                "api_secret": "",
                "use_mock": True,
                "trading_mode": "Paper",
                "risk_level": "Conservative",
                "max_position_size": 1.0,
                "stop_loss_percent": 5.0,
                "take_profit_percent": 10.0,
                "strategy": "MasterOmniOverlord",
                "timeframe": "1h"
            }
            
            # Update UI
            self.api_key_var.set(self.config["api_key"])
            self.api_secret_var.set(self.config["api_secret"])
            self.use_mock_var.set(self.config["use_mock"])
            self.trading_mode_var.set(self.config["trading_mode"])
            self.risk_level_var.set(self.config["risk_level"])
            self.max_position_var.set(str(self.config["max_position_size"]))
            self.stop_loss_var.set(str(self.config["stop_loss_percent"]))
            self.take_profit_var.set(str(self.config["take_profit_percent"]))
            self.strategy_var.set(self.config["strategy"])
            self.timeframe_var.set(self.config["timeframe"])
            
            # Save config
            self._save_config()
            
            # Log
            logger.info("Settings reset to defaults")
            self.status_var.set("Settings reset to defaults")
            
            # Show message
            messagebox.showinfo("Success", "Settings reset to defaults")
        except Exception as e:
            logger.error(f"Error resetting settings: {e}")
            messagebox.showerror("Error", f"Error resetting settings: {e}")
    
    def _test_connection(self):
        """Test API connection."""
        try:
            # Get API key and secret
            api_key = self.api_key_var.get()
            api_secret = self.api_secret_var.get()
            
            # Validate inputs
            if not api_key or not api_secret:
                messagebox.showerror("Error", "API key and secret are required")
                return
            
            # Test connection
            self.status_var.set("Testing connection...")
            
            # Simulate connection test
            time.sleep(1)
            
            # Show result
            messagebox.showinfo("Success", "Connection successful")
            self.status_var.set("Connection successful")
            
            # Log
            logger.info("Connection test successful")
        except Exception as e:
            logger.error(f"Error testing connection: {e}")
            messagebox.showerror("Error", f"Error testing connection: {e}")
            self.status_var.set("Connection failed")
    
    def _update_data_thread(self):
        """Background thread for updating data."""
        while self.running:
            try:
                # Update market data
                self._update_market_data()
                
                # Update positions
                self._update_positions()
                
                # Update UI
                self._update_positions_table()
                
                # Update chart and order book
                self._update_chart_and_order_book()
                
                # Sleep
                time.sleep(5)
            except Exception as e:
                logger.error(f"Error in update thread: {e}")
    
    def _update_market_data(self):
        """Update market data."""
        try:
            # Get symbols
            symbols = ["XRP-PERP", "BTC-PERP", "ETH-PERP"]
            
            # Update market data
            for symbol in symbols:
                # Simulate market data
                price = self._get_market_price(symbol)
                
                # Update market data
                if symbol not in self.market_data:
                    self.market_data[symbol] = {
                        "price": price,
                        "24h_change": 0.0,
                        "24h_volume": 0.0,
                        "bid": price * 0.999,
                        "ask": price * 1.001
                    }
                else:
                    # Update price with small random change
                    old_price = self.market_data[symbol]["price"]
                    change_percent = 0.002 * (np.random.random() - 0.5)
                    new_price = old_price * (1 + change_percent)
                    
                    self.market_data[symbol]["price"] = new_price
                    self.market_data[symbol]["24h_change"] = (new_price - old_price) / old_price * 100
                    self.market_data[symbol]["24h_volume"] += new_price * np.random.random() * 100
                    self.market_data[symbol]["bid"] = new_price * 0.999
                    self.market_data[symbol]["ask"] = new_price * 1.001
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    def _update_positions(self):
        """Update positions."""
        try:
            # Update positions
            for symbol, position in list(self.positions.items()):
                # Get market price
                market_price = self.market_data.get(symbol, {}).get("price", self._get_market_price(symbol))
                
                # Update position
                position["current_price"] = market_price
                
                # Check stop loss and take profit
                side = position["side"]
                entry_price = position["entry_price"]
                
                if side == "BUY":
                    pnl_percent = (market_price - entry_price) / entry_price * 100
                else:
                    pnl_percent = (entry_price - market_price) / entry_price * 100
                
                # Check stop loss
                stop_loss_percent = float(self.stop_loss_var.get())
                if pnl_percent < -stop_loss_percent:
                    # Close position
                    logger.info(f"Stop loss triggered for {symbol}: {pnl_percent:.2f}%")
                    del self.positions[symbol]
                    continue
                
                # Check take profit
                take_profit_percent = float(self.take_profit_var.get())
                if pnl_percent > take_profit_percent:
                    # Close position
                    logger.info(f"Take profit triggered for {symbol}: {pnl_percent:.2f}%")
                    del self.positions[symbol]
                    continue
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def _update_chart_and_order_book(self):
        """Update chart and order book."""
        try:
            # Get symbol
            symbol = self.symbol_var.get()
            
            # Get market data
            market_data = self.market_data.get(symbol)
            
            if market_data:
                # Update chart
                self.chart_visualizer.update_chart(symbol, market_data["price"])
                
                # Update order book
                self.order_book_visualizer.update_order_book(symbol, market_data["bid"], market_data["ask"])
        except Exception as e:
            logger.error(f"Error updating chart and order book: {e}")
    
    def on_closing(self):
        """Handle window closing."""
        try:
            # Stop background thread
            self.running = False
            
            # Save config
            self._save_config()
            
            # Destroy window
            self.destroy()
        except Exception as e:
            logger.error(f"Error closing window: {e}")
            self.destroy()


class GUILogHandler(logging.Handler):
    """
    Log handler for GUI.
    
    Redirects log messages to the GUI log text area.
    """
    
    def __init__(self, text_widget):
        """
        Initialize the log handler.
        
        Args:
            text_widget: Text widget to display logs
        """
        super().__init__()
        self.text_widget = text_widget
        
        # Configure formatter
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    def emit(self, record):
        """
        Emit a log record.
        
        Args:
            record: Log record
        """
        try:
            # Format record
            msg = self.format(record)
            
            # Add to text widget
            def _add_to_text():
                self.text_widget.configure(state=tk.NORMAL)
                self.text_widget.insert(tk.END, msg + "\n")
                self.text_widget.configure(state=tk.DISABLED)
                self.text_widget.see(tk.END)
            
            # Schedule on main thread
            self.text_widget.after(0, _add_to_text)
        except Exception:
            self.handleError(record)


if __name__ == "__main__":
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Create GUI
    app = TabbedGUI()
    
    # Set closing handler
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start main loop
    app.mainloop()
