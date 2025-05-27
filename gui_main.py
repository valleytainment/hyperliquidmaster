#!/usr/bin/env python3
"""
Enhanced Hyperliquid Trading Bot with GUI
-----------------------------------------
This module integrates the enhanced trading bot backend with a comprehensive
Tkinter GUI interface for monitoring and controlling the trading system.

Features:
- Real-time price charts and visualization
- Configuration controls for trading parameters
- Live monitoring of positions and performance metrics
- Manual trading controls (buy/sell buttons)
- Technical indicator visualization
- API key management
- Modern themed interface with light/dark mode support
- Live trading integration
- Scrollable interface for all content
- Dedicated positions tab
"""

import os
import sys
import time
import json
import queue
import logging
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Import core components
from core.trading_integration import TradingIntegration
from core.api_key_manager import ApiKeyManager
from core.error_handler import ErrorHandler

# Import GUI style manager
from gui_style import (
    GUIStyleManager, 
    create_header_label, 
    create_subheader_label, 
    create_scrollable_frame,
    create_scrollable_text
)

# Constants
CONFIG_FILE = "config.json"
LOG_FILE = "enhanced_bot_gui.log"
VERSION = "2.0.0"

###############################################################################
# Logging Setup
###############################################################################
class QueueLoggingHandler(logging.Handler):
    """Custom logging handler that puts logs into a queue for GUI display."""
    
    def __init__(self, log_queue: queue.Queue):
        """Initialize with a queue to store log messages."""
        super().__init__()
        self.log_queue = log_queue
        
    def emit(self, record):
        """Put formatted log message into the queue."""
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception:
            self.handleError(record)

###############################################################################
# Enhanced Trading Bot with GUI
###############################################################################
class EnhancedTradingBotGUI:
    """
    Main GUI class for the Enhanced Hyperliquid Trading Bot.
    Provides a comprehensive interface for trading and monitoring.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the GUI and all components.
        
        Args:
            config_path: Path to the configuration file
        """
        # Initialize GUI components
        self.root = tk.Tk()
        self.root.title(f"Enhanced Hyperliquid Trading Bot v{VERSION}")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Store config path
        self.config_path = config_path
        
        # Setup logging with queue for GUI display
        self.log_queue = queue.Queue()
        self.logger = self._setup_logger()
        
        # Initialize error handler
        self.error_handler = ErrorHandler(self.logger)
        
        # Initialize API key manager
        self.api_key_manager = ApiKeyManager(config_path)
        
        # Initialize trading integration
        self.trading = TradingIntegration(config_path, self.logger)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize style manager
        theme = self.config.get("theme", "dark")
        self.style_manager = GUIStyleManager(self.root, theme)
        
        # Runtime variables
        self.running = False
        self.thread = None
        self.warmup_done = False
        self.warmup_start = None
        self.warmup_duration = 20.0
        self.start_equity = 0.0
        self.last_trade_time = 0
        self.market_data = {}
        self.positions = {}
        self.hist_data = pd.DataFrame()
        self.orders = []
        
        # Create GUI components
        self._create_gui()
        
        # Start log consumer
        self.log_consumer_running = True
        self.log_consumer_thread = threading.Thread(target=self._consume_logs, daemon=True)
        self.log_consumer_thread.start()
        
        # Update connection status
        self._update_connection_status()
        
        # Schedule periodic updates
        self._schedule_updates()
        
        # Log startup
        self.logger.info(f"Enhanced Hyperliquid Trading Bot v{VERSION} started")
    
    def _setup_logger(self) -> logging.Logger:
        """
        Set up the logger with queue handler for GUI display.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger("EnhancedTradingBot")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(LOG_FILE, mode="a")
        file_handler.setLevel(logging.INFO)
        
        # Create queue handler for GUI
        queue_handler = QueueLoggingHandler(self.log_queue)
        queue_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        queue_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.addHandler(queue_handler)
        
        return logger
    
    def _load_config(self) -> Dict:
        """
        Load configuration from file.
        
        Returns:
            Dict containing the configuration
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                # Create default config
                default_config = {
                    "account_address": "",
                    "secret_key": "",
                    "symbols": ["BTC", "ETH", "SOL"],
                    "theme": "dark",
                    "trade_symbol": "BTC",
                    "fast_ma": 5,
                    "slow_ma": 15,
                    "rsi_period": 14,
                    "macd_fast": 12,
                    "macd_slow": 26,
                    "macd_signal": 9,
                    "boll_period": 20,
                    "boll_stddev": 2.0,
                    "stop_loss_pct": 0.005,
                    "take_profit_pct": 0.01,
                    "use_trailing_stop": True,
                    "trail_start_profit": 0.005,
                    "trail_offset": 0.0025,
                    "use_partial_tp": True,
                    "min_trade_interval": 60,
                    "risk_percent": 0.01,
                    "use_manual_entry_size": True,
                    "manual_entry_size": 1.0,
                    "use_manual_close_size": False,
                    "position_close_size": 100.0,
                    "nn_lookback_bars": 100,
                    "nn_hidden_size": 64,
                    "nn_lr": 0.001,
                    "synergy_conf_threshold": 0.7,
                    "circuit_breaker_threshold": 0.05,
                    "taker_fee": 0.00042,
                    "api_url": "https://api.hyperliquid.xyz",
                    "poll_interval_seconds": 5.0,
                    "micro_poll_interval": 2.0
                }
                
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                
                return default_config
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}
    
    def _create_gui(self):
        """Create the GUI components."""
        # Create main frame with scrolling capability
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create top frame for controls
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(10, 10), padx=10)
        
        # Create symbol selection
        ttk.Label(top_frame, text="Symbol:").pack(side=tk.LEFT, padx=(0, 5))
        self.symbol_var = tk.StringVar(value=self.config.get("trade_symbol", "BTC"))
        symbol_combo = ttk.Combobox(top_frame, textvariable=self.symbol_var, 
                                    values=["BTC", "ETH", "SOL"])
        symbol_combo.pack(side=tk.LEFT, padx=(0, 10))
        symbol_combo.bind("<<ComboboxSelected>>", self._on_symbol_change)
        
        # Create start/stop buttons
        self.start_button = ttk.Button(top_frame, text="Start Bot", command=self.start_bot)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_button = ttk.Button(top_frame, text="Stop Bot", command=self.stop_bot, style="Error.TButton", state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Create manual trade buttons
        ttk.Label(top_frame, text="Manual:").pack(side=tk.LEFT, padx=(10, 5))
        
        self.buy_button = ttk.Button(top_frame, text="Buy", command=self._manual_buy, style="Success.TButton")
        self.buy_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.sell_button = ttk.Button(top_frame, text="Sell", command=self._manual_sell, style="Warning.TButton")
        self.sell_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.close_button = ttk.Button(top_frame, text="Close Position", command=self._manual_close)
        self.close_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Create size entry
        ttk.Label(top_frame, text="Size:").pack(side=tk.LEFT, padx=(10, 5))
        self.size_var = tk.StringVar(value=str(self.config.get("manual_entry_size", 1.0)))
        size_entry = ttk.Entry(top_frame, textvariable=self.size_var, width=8)
        size_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        # Add theme toggle button
        self.theme_button = ttk.Button(top_frame, text="Toggle Theme", command=self._toggle_theme)
        self.theme_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Add connection status indicator
        self.connection_status_var = tk.StringVar(value="Disconnected")
        self.connection_status_label = ttk.Label(top_frame, textvariable=self.connection_status_var)
        self.connection_status_label.pack(side=tk.RIGHT, padx=(0, 10))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Create chart tab
        chart_frame = ttk.Frame(notebook)
        notebook.add(chart_frame, text="Price Chart")
        
        # Create chart
        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.ax1 = self.fig.add_subplot(211)  # Price chart
        self.ax2 = self.fig.add_subplot(212, sharex=self.ax1)  # Volume chart
        
        # Apply matplotlib style
        plt.rcParams.update(self.style_manager.get_matplotlib_style())
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create indicators tab
        indicators_frame = ttk.Frame(notebook)
        notebook.add(indicators_frame, text="Indicators")
        
        # Create indicators chart
        self.ind_fig = Figure(figsize=(12, 8), dpi=100)
        self.ind_ax1 = self.ind_fig.add_subplot(411)  # RSI
        self.ind_ax2 = self.ind_fig.add_subplot(412, sharex=self.ind_ax1)  # MACD
        self.ind_ax3 = self.ind_fig.add_subplot(413, sharex=self.ind_ax1)  # Bollinger Bands
        self.ind_ax4 = self.ind_fig.add_subplot(414, sharex=self.ind_ax1)  # ADX
        
        self.ind_canvas = FigureCanvasTkAgg(self.ind_fig, master=indicators_frame)
        self.ind_canvas.draw()
        self.ind_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create positions tab
        positions_frame = ttk.Frame(notebook)
        notebook.add(positions_frame, text="Positions")
        
        # Create positions view
        positions_container = ttk.Frame(positions_frame)
        positions_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create positions table
        positions_header_frame = ttk.Frame(positions_container)
        positions_header_frame.pack(fill=tk.X, pady=(0, 5))
        
        create_header_label(positions_header_frame, "Open Positions").pack(side=tk.LEFT)
        self.refresh_positions_button = ttk.Button(
            positions_header_frame, 
            text="Refresh", 
            command=self._refresh_positions
        )
        self.refresh_positions_button.pack(side=tk.RIGHT)
        
        # Create treeview for positions
        positions_tree_frame = ttk.Frame(positions_container)
        positions_tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.positions_tree = ttk.Treeview(
            positions_tree_frame,
            columns=("symbol", "size", "entry_price", "current_price", "pnl", "pnl_pct"),
            show="headings"
        )
        
        # Configure columns
        self.positions_tree.heading("symbol", text="Symbol")
        self.positions_tree.heading("size", text="Size")
        self.positions_tree.heading("entry_price", text="Entry Price")
        self.positions_tree.heading("current_price", text="Current Price")
        self.positions_tree.heading("pnl", text="PnL")
        self.positions_tree.heading("pnl_pct", text="PnL %")
        
        self.positions_tree.column("symbol", width=100)
        self.positions_tree.column("size", width=100)
        self.positions_tree.column("entry_price", width=150)
        self.positions_tree.column("current_price", width=150)
        self.positions_tree.column("pnl", width=150)
        self.positions_tree.column("pnl_pct", width=150)
        
        # Add scrollbar to treeview
        positions_scrollbar = ttk.Scrollbar(positions_tree_frame, orient="vertical", command=self.positions_tree.yview)
        self.positions_tree.configure(yscrollcommand=positions_scrollbar.set)
        
        positions_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.positions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create orders tab
        orders_frame = ttk.Frame(notebook)
        notebook.add(orders_frame, text="Orders")
        
        # Create orders view
        orders_container = ttk.Frame(orders_frame)
        orders_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create orders table header
        orders_header_frame = ttk.Frame(orders_container)
        orders_header_frame.pack(fill=tk.X, pady=(0, 5))
        
        create_header_label(orders_header_frame, "Open Orders").pack(side=tk.LEFT)
        self.refresh_orders_button = ttk.Button(
            orders_header_frame, 
            text="Refresh", 
            command=self._refresh_orders
        )
        self.refresh_orders_button.pack(side=tk.RIGHT)
        
        # Create treeview for orders
        orders_tree_frame = ttk.Frame(orders_container)
        orders_tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.orders_tree = ttk.Treeview(
            orders_tree_frame,
            columns=("id", "symbol", "side", "size", "price", "type", "status", "time"),
            show="headings"
        )
        
        # Configure columns
        self.orders_tree.heading("id", text="Order ID")
        self.orders_tree.heading("symbol", text="Symbol")
        self.orders_tree.heading("side", text="Side")
        self.orders_tree.heading("size", text="Size")
        self.orders_tree.heading("price", text="Price")
        self.orders_tree.heading("type", text="Type")
        self.orders_tree.heading("status", text="Status")
        self.orders_tree.heading("time", text="Time")
        
        self.orders_tree.column("id", width=100)
        self.orders_tree.column("symbol", width=80)
        self.orders_tree.column("side", width=80)
        self.orders_tree.column("size", width=100)
        self.orders_tree.column("price", width=100)
        self.orders_tree.column("type", width=100)
        self.orders_tree.column("status", width=100)
        self.orders_tree.column("time", width=150)
        
        # Add scrollbar to treeview
        orders_scrollbar = ttk.Scrollbar(orders_tree_frame, orient="vertical", command=self.orders_tree.yview)
        self.orders_tree.configure(yscrollcommand=orders_scrollbar.set)
        
        orders_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.orders_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create settings tab with scrollable content
        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text="Settings")
        
        # Create scrollable container for settings
        settings_canvas, settings_scrollable_frame = create_scrollable_frame(settings_frame)
        
        # Create settings grid
        settings_grid = ttk.Frame(settings_scrollable_frame)
        settings_grid.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create settings controls
        row = 0
        
        # API Key Management
        create_header_label(settings_grid, "API Key Management").grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        row += 1
        
        # Account Address
        ttk.Label(settings_grid, text="Account Address:").grid(row=row, column=0, sticky=tk.W)
        self.account_address_var = tk.StringVar(value=str(self.config.get("account_address", "")))
        ttk.Entry(settings_grid, textvariable=self.account_address_var, width=40).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        # Secret Key
        ttk.Label(settings_grid, text="Secret Key:").grid(row=row, column=0, sticky=tk.W)
        self.secret_key_var = tk.StringVar(value=str(self.config.get("secret_key", "")))
        secret_key_entry = ttk.Entry(settings_grid, textvariable=self.secret_key_var, width=40, show="*")
        secret_key_entry.grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        # Show/Hide Secret Key
        self.show_secret_key_var = tk.BooleanVar(value=False)
        show_secret_key_cb = ttk.Checkbutton(
            settings_grid, 
            text="Show Secret Key", 
            variable=self.show_secret_key_var,
            command=lambda: secret_key_entry.config(show="" if self.show_secret_key_var.get() else "*")
        )
        show_secret_key_cb.grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        
        # Test Connection button
        test_connection_button = ttk.Button(settings_grid, text="Test Connection", command=self._test_connection)
        test_connection_button.grid(row=row, column=0, columnspan=2, pady=(5, 10))
        row += 1
        
        # Technical indicators settings
        create_header_label(settings_grid, "Technical Indicators").grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 10))
        row += 1
        
        # Fast MA
        ttk.Label(settings_grid, text="Fast MA Period:").grid(row=row, column=0, sticky=tk.W)
        self.fast_ma_var = tk.StringVar(value=str(self.config.get("fast_ma", 5)))
        ttk.Entry(settings_grid, textvariable=self.fast_ma_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        # Slow MA
        ttk.Label(settings_grid, text="Slow MA Period:").grid(row=row, column=0, sticky=tk.W)
        self.slow_ma_var = tk.StringVar(value=str(self.config.get("slow_ma", 15)))
        ttk.Entry(settings_grid, textvariable=self.slow_ma_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        # RSI
        ttk.Label(settings_grid, text="RSI Period:").grid(row=row, column=0, sticky=tk.W)
        self.rsi_period_var = tk.StringVar(value=str(self.config.get("rsi_period", 14)))
        ttk.Entry(settings_grid, textvariable=self.rsi_period_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        # MACD
        ttk.Label(settings_grid, text="MACD Fast:").grid(row=row, column=0, sticky=tk.W)
        self.macd_fast_var = tk.StringVar(value=str(self.config.get("macd_fast", 12)))
        ttk.Entry(settings_grid, textvariable=self.macd_fast_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(settings_grid, text="MACD Slow:").grid(row=row, column=0, sticky=tk.W)
        self.macd_slow_var = tk.StringVar(value=str(self.config.get("macd_slow", 26)))
        ttk.Entry(settings_grid, textvariable=self.macd_slow_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(settings_grid, text="MACD Signal:").grid(row=row, column=0, sticky=tk.W)
        self.macd_signal_var = tk.StringVar(value=str(self.config.get("macd_signal", 9)))
        ttk.Entry(settings_grid, textvariable=self.macd_signal_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        # Bollinger Bands
        ttk.Label(settings_grid, text="Bollinger Period:").grid(row=row, column=0, sticky=tk.W)
        self.boll_period_var = tk.StringVar(value=str(self.config.get("boll_period", 20)))
        ttk.Entry(settings_grid, textvariable=self.boll_period_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(settings_grid, text="Bollinger StdDev:").grid(row=row, column=0, sticky=tk.W)
        self.boll_stddev_var = tk.StringVar(value=str(self.config.get("boll_stddev", 2.0)))
        ttk.Entry(settings_grid, textvariable=self.boll_stddev_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        # Risk management settings
        create_header_label(settings_grid, "Risk Management").grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 10))
        row += 1
        
        # Stop loss
        ttk.Label(settings_grid, text="Stop Loss %:").grid(row=row, column=0, sticky=tk.W)
        self.stop_loss_var = tk.StringVar(value=str(self.config.get("stop_loss_pct", 0.005)))
        ttk.Entry(settings_grid, textvariable=self.stop_loss_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        # Take profit
        ttk.Label(settings_grid, text="Take Profit %:").grid(row=row, column=0, sticky=tk.W)
        self.take_profit_var = tk.StringVar(value=str(self.config.get("take_profit_pct", 0.01)))
        ttk.Entry(settings_grid, textvariable=self.take_profit_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        # Trailing stop
        self.use_trailing_stop_var = tk.BooleanVar(value=self.config.get("use_trailing_stop", True))
        ttk.Checkbutton(settings_grid, text="Use Trailing Stop", variable=self.use_trailing_stop_var).grid(
            row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        
        ttk.Label(settings_grid, text="Trail Start Profit:").grid(row=row, column=0, sticky=tk.W)
        self.trail_start_var = tk.StringVar(value=str(self.config.get("trail_start_profit", 0.005)))
        ttk.Entry(settings_grid, textvariable=self.trail_start_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(settings_grid, text="Trail Offset:").grid(row=row, column=0, sticky=tk.W)
        self.trail_offset_var = tk.StringVar(value=str(self.config.get("trail_offset", 0.0025)))
        ttk.Entry(settings_grid, textvariable=self.trail_offset_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        # Partial take profit
        self.use_partial_tp_var = tk.BooleanVar(value=self.config.get("use_partial_tp", True))
        ttk.Checkbutton(settings_grid, text="Use Partial Take Profit", variable=self.use_partial_tp_var).grid(
            row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        
        # Order settings
        create_header_label(settings_grid, "Order Settings").grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 10))
        row += 1
        
        # Min trade interval
        ttk.Label(settings_grid, text="Min Trade Interval (s):").grid(row=row, column=0, sticky=tk.W)
        self.min_trade_interval_var = tk.StringVar(value=str(self.config.get("min_trade_interval", 60)))
        ttk.Entry(settings_grid, textvariable=self.min_trade_interval_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        # Risk percent
        ttk.Label(settings_grid, text="Risk Percent:").grid(row=row, column=0, sticky=tk.W)
        self.risk_percent_var = tk.StringVar(value=str(self.config.get("risk_percent", 0.01)))
        ttk.Entry(settings_grid, textvariable=self.risk_percent_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        # Manual entry size
        self.use_manual_entry_size_var = tk.BooleanVar(value=self.config.get("use_manual_entry_size", True))
        ttk.Checkbutton(settings_grid, text="Use Manual Entry Size", variable=self.use_manual_entry_size_var).grid(
            row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        
        ttk.Label(settings_grid, text="Manual Entry Size:").grid(row=row, column=0, sticky=tk.W)
        self.manual_entry_size_var = tk.StringVar(value=str(self.config.get("manual_entry_size", 1.0)))
        ttk.Entry(settings_grid, textvariable=self.manual_entry_size_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        # Manual close size
        self.use_manual_close_size_var = tk.BooleanVar(value=self.config.get("use_manual_close_size", False))
        ttk.Checkbutton(settings_grid, text="Use Manual Close Size", variable=self.use_manual_close_size_var).grid(
            row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        
        ttk.Label(settings_grid, text="Position Close Size %:").grid(row=row, column=0, sticky=tk.W)
        self.position_close_size_var = tk.StringVar(value=str(self.config.get("position_close_size", 100.0)))
        ttk.Entry(settings_grid, textvariable=self.position_close_size_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        # Advanced settings
        create_header_label(settings_grid, "Advanced Settings").grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 10))
        row += 1
        
        # Neural network settings
        ttk.Label(settings_grid, text="NN Lookback Bars:").grid(row=row, column=0, sticky=tk.W)
        self.nn_lookback_bars_var = tk.StringVar(value=str(self.config.get("nn_lookback_bars", 100)))
        ttk.Entry(settings_grid, textvariable=self.nn_lookback_bars_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(settings_grid, text="NN Hidden Size:").grid(row=row, column=0, sticky=tk.W)
        self.nn_hidden_size_var = tk.StringVar(value=str(self.config.get("nn_hidden_size", 64)))
        ttk.Entry(settings_grid, textvariable=self.nn_hidden_size_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(settings_grid, text="NN Learning Rate:").grid(row=row, column=0, sticky=tk.W)
        self.nn_lr_var = tk.StringVar(value=str(self.config.get("nn_lr", 0.001)))
        ttk.Entry(settings_grid, textvariable=self.nn_lr_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        # Strategy settings
        ttk.Label(settings_grid, text="Synergy Conf Threshold:").grid(row=row, column=0, sticky=tk.W)
        self.synergy_conf_threshold_var = tk.StringVar(value=str(self.config.get("synergy_conf_threshold", 0.7)))
        ttk.Entry(settings_grid, textvariable=self.synergy_conf_threshold_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(settings_grid, text="Circuit Breaker Threshold:").grid(row=row, column=0, sticky=tk.W)
        self.circuit_breaker_threshold_var = tk.StringVar(value=str(self.config.get("circuit_breaker_threshold", 0.05)))
        ttk.Entry(settings_grid, textvariable=self.circuit_breaker_threshold_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        # Fee settings
        ttk.Label(settings_grid, text="Taker Fee:").grid(row=row, column=0, sticky=tk.W)
        self.taker_fee_var = tk.StringVar(value=str(self.config.get("taker_fee", 0.00042)))
        ttk.Entry(settings_grid, textvariable=self.taker_fee_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        # API settings
        create_header_label(settings_grid, "API Settings").grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 10))
        row += 1
        
        # API URL
        ttk.Label(settings_grid, text="API URL:").grid(row=row, column=0, sticky=tk.W)
        self.api_url_var = tk.StringVar(value=str(self.config.get("api_url", "https://api.hyperliquid.xyz")))
        ttk.Entry(settings_grid, textvariable=self.api_url_var, width=30).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        # Poll interval
        ttk.Label(settings_grid, text="Poll Interval (s):").grid(row=row, column=0, sticky=tk.W)
        self.poll_interval_var = tk.StringVar(value=str(self.config.get("poll_interval_seconds", 5.0)))
        ttk.Entry(settings_grid, textvariable=self.poll_interval_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        # Micro poll interval
        ttk.Label(settings_grid, text="Micro Poll Interval (s):").grid(row=row, column=0, sticky=tk.W)
        self.micro_poll_interval_var = tk.StringVar(value=str(self.config.get("micro_poll_interval", 2.0)))
        ttk.Entry(settings_grid, textvariable=self.micro_poll_interval_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        # Save settings button
        save_button = ttk.Button(settings_grid, text="Save Settings", command=self._save_settings)
        save_button.grid(row=row, column=0, columnspan=2, pady=(10, 0))
        row += 1
        
        # Create logs tab with scrollable text
        logs_frame = ttk.Frame(notebook)
        notebook.add(logs_frame, text="Logs")
        
        # Create log text widget with scrollbar
        log_frame, self.log_text = create_scrollable_text(logs_frame, wrap=tk.WORD, height=20)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(0, 5), padx=10)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT)
        
        # Add version info to status bar
        version_label = ttk.Label(status_frame, text=f"v{VERSION}")
        version_label.pack(side=tk.RIGHT)
        
        # Initialize charts
        self._init_charts()
    
    def _toggle_theme(self):
        """Toggle between light and dark themes."""
        try:
            current_theme = self.style_manager.theme_name
            new_theme = "light" if current_theme == "dark" else "dark"
            
            # Switch theme
            self.style_manager.switch_theme(new_theme)
            
            # Update matplotlib style
            plt.rcParams.update(self.style_manager.get_matplotlib_style())
            
            # Redraw charts
            self._update_charts()
            
            # Save theme preference
            self.config["theme"] = new_theme
            self.style_manager.save_theme_preference(self.config_path)
            
            self.logger.info(f"Switched to {new_theme} theme")
        except Exception as e:
            self.logger.error(f"Error toggling theme: {e}")
            messagebox.showerror("Error", f"Error toggling theme: {e}")
    
    def _update_connection_status(self):
        """Update the connection status indicator."""
        try:
            if self.trading.is_connected:
                self.connection_status_var.set("Connected")
                self.connection_status_label.configure(foreground=self.style_manager.get_color("success_color"))
            else:
                self.connection_status_var.set("Disconnected")
                self.connection_status_label.configure(foreground=self.style_manager.get_color("error_color"))
        except Exception as e:
            self.logger.error(f"Error updating connection status: {e}")
    
    def _test_connection(self):
        """Test the connection to the exchange."""
        try:
            # First save the API keys
            account_address = self.account_address_var.get()
            secret_key = self.secret_key_var.get()
            
            if not account_address or not secret_key:
                messagebox.showerror("Error", "Please enter both Account Address and Secret Key")
                return
            
            # Update API keys
            result = self.trading.set_api_keys(account_address, secret_key)
            
            if not result["success"]:
                messagebox.showerror("Error", f"Failed to update API keys: {result['message']}")
                return
            
            # Test connection
            self.logger.info("Testing connection to exchange...")
            result = self.trading.test_connection()
            
            if result["success"]:
                messagebox.showinfo("Success", "Connection successful!")
                self.logger.info("Connection test successful")
            else:
                messagebox.showerror("Error", f"Connection failed: {result['message']}")
                self.logger.error(f"Connection test failed: {result['message']}")
            
            # Update connection status
            self._update_connection_status()
        except Exception as e:
            self.logger.error(f"Error testing connection: {e}")
            messagebox.showerror("Error", f"Error testing connection: {e}")
    
    def _init_charts(self):
        """Initialize charts."""
        try:
            # Price chart
            self.ax1.clear()
            self.ax1.set_title("Price")
            self.ax1.set_ylabel("Price")
            self.ax1.grid(True)
            
            # Volume chart
            self.ax2.clear()
            self.ax2.set_title("Volume")
            self.ax2.set_ylabel("Volume")
            self.ax2.set_xlabel("Time")
            self.ax2.grid(True)
            
            self.canvas.draw()
            
            # Indicators chart
            self.ind_ax1.clear()
            self.ind_ax1.set_title("RSI")
            self.ind_ax1.axhline(y=70, color='r', linestyle='-')
            self.ind_ax1.axhline(y=30, color='g', linestyle='-')
            self.ind_ax1.set_ylim(0, 100)
            self.ind_ax1.grid(True)
            
            self.ind_ax2.clear()
            self.ind_ax2.set_title("MACD")
            self.ind_ax2.grid(True)
            
            self.ind_ax3.clear()
            self.ind_ax3.set_title("Bollinger Bands")
            self.ind_ax3.grid(True)
            
            self.ind_ax4.clear()
            self.ind_ax4.set_title("ADX")
            self.ind_ax4.grid(True)
            
            self.ind_canvas.draw()
        except Exception as e:
            self.logger.error(f"Error initializing charts: {e}")
    
    def _consume_logs(self):
        """Consume logs from the queue and display them in the log text widget."""
        while self.log_consumer_running:
            try:
                # Get log message from queue (with timeout to allow checking running flag)
                try:
                    message = self.log_queue.get(block=True, timeout=0.1)
                    self.log_text.insert(tk.END, message + "\n")
                    self.log_text.see(tk.END)
                    self.log_queue.task_done()
                except queue.Empty:
                    continue
            except Exception as e:
                print(f"Error in log consumer: {e}")
                time.sleep(1)
    
    def _on_symbol_change(self, event):
        """Handle symbol change event."""
        try:
            symbol = self.symbol_var.get()
            self.logger.info(f"Symbol changed to {symbol}")
            
            # Update config
            self.config["trade_symbol"] = symbol
            
            # Reset charts
            self._init_charts()
            
            # Update status
            self.status_var.set(f"Symbol changed to {symbol}")
        except Exception as e:
            self.logger.error(f"Error handling symbol change: {e}")
    
    def _save_settings(self):
        """Save settings to config file."""
        try:
            # Update config with values from GUI
            self.config["trade_symbol"] = self.symbol_var.get()
            
            # API Key Management
            self.config["account_address"] = self.account_address_var.get()
            self.config["secret_key"] = self.secret_key_var.get()
            
            # Technical indicators
            self.config["fast_ma"] = int(self.fast_ma_var.get())
            self.config["slow_ma"] = int(self.slow_ma_var.get())
            self.config["rsi_period"] = int(self.rsi_period_var.get())
            self.config["macd_fast"] = int(self.macd_fast_var.get())
            self.config["macd_slow"] = int(self.macd_slow_var.get())
            self.config["macd_signal"] = int(self.macd_signal_var.get())
            self.config["boll_period"] = int(self.boll_period_var.get())
            self.config["boll_stddev"] = float(self.boll_stddev_var.get())
            
            # Risk management
            self.config["stop_loss_pct"] = float(self.stop_loss_var.get())
            self.config["take_profit_pct"] = float(self.take_profit_var.get())
            self.config["use_trailing_stop"] = self.use_trailing_stop_var.get()
            self.config["trail_start_profit"] = float(self.trail_start_var.get())
            self.config["trail_offset"] = float(self.trail_offset_var.get())
            self.config["use_partial_tp"] = self.use_partial_tp_var.get()
            
            # Order settings
            self.config["min_trade_interval"] = int(self.min_trade_interval_var.get())
            self.config["risk_percent"] = float(self.risk_percent_var.get())
            self.config["use_manual_entry_size"] = self.use_manual_entry_size_var.get()
            self.config["manual_entry_size"] = float(self.manual_entry_size_var.get())
            self.config["use_manual_close_size"] = self.use_manual_close_size_var.get()
            self.config["position_close_size"] = float(self.position_close_size_var.get())
            
            # Advanced settings
            self.config["nn_lookback_bars"] = int(self.nn_lookback_bars_var.get())
            self.config["nn_hidden_size"] = int(self.nn_hidden_size_var.get())
            self.config["nn_lr"] = float(self.nn_lr_var.get())
            self.config["synergy_conf_threshold"] = float(self.synergy_conf_threshold_var.get())
            self.config["circuit_breaker_threshold"] = float(self.circuit_breaker_threshold_var.get())
            self.config["taker_fee"] = float(self.taker_fee_var.get())
            
            # API settings
            self.config["api_url"] = self.api_url_var.get()
            self.config["poll_interval_seconds"] = float(self.poll_interval_var.get())
            self.config["micro_poll_interval"] = float(self.micro_poll_interval_var.get())
            
            # Save to file
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            # Update trading integration with new settings
            self.trading.reload_config()
            
            # Update connection status
            self._update_connection_status()
            
            self.logger.info("Settings saved to config.json")
            messagebox.showinfo("Settings", "Settings saved successfully!")
            
            # Update size variable for manual trading
            self.size_var.set(str(self.config["manual_entry_size"]))
            
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            messagebox.showerror("Error", f"Error saving settings: {e}")
    
    def _manual_buy(self):
        """Execute manual buy order."""
        try:
            if not self.trading.is_connected:
                messagebox.showerror("Error", "Not connected to exchange")
                return
            
            size = float(self.size_var.get())
            symbol = self.symbol_var.get()
            
            self.logger.info(f"Manual BUY order: {symbol}, size={size}")
            
            # Get current market price
            market_data = self.trading.get_market_data(symbol)
            if not market_data["success"]:
                messagebox.showerror("Error", f"Failed to get market data: {market_data['message']}")
                return
            
            price = market_data["data"]["price"]
            
            # Execute order
            result = self.trading.place_order(symbol, True, size, price, "LIMIT")
            
            if result["success"]:
                self.logger.info(f"Manual BUY order executed: {symbol}, size={size}, price={price}")
                messagebox.showinfo("Order Placed", f"BUY order for {size} {symbol} at ${price} placed successfully")
                
                # Update status
                self.status_var.set(f"Manual BUY order executed: {symbol}, size={size}")
                
                # Refresh orders
                self._refresh_orders()
            else:
                self.logger.error(f"Error executing manual BUY order: {result['message']}")
                messagebox.showerror("Error", f"Error executing manual BUY order: {result['message']}")
            
        except Exception as e:
            self.logger.error(f"Error executing manual BUY order: {e}")
            messagebox.showerror("Error", f"Error executing manual BUY order: {e}")
    
    def _manual_sell(self):
        """Execute manual sell order."""
        try:
            if not self.trading.is_connected:
                messagebox.showerror("Error", "Not connected to exchange")
                return
            
            size = float(self.size_var.get())
            symbol = self.symbol_var.get()
            
            self.logger.info(f"Manual SELL order: {symbol}, size={size}")
            
            # Get current market price
            market_data = self.trading.get_market_data(symbol)
            if not market_data["success"]:
                messagebox.showerror("Error", f"Failed to get market data: {market_data['message']}")
                return
            
            price = market_data["data"]["price"]
            
            # Execute order
            result = self.trading.place_order(symbol, False, size, price, "LIMIT")
            
            if result["success"]:
                self.logger.info(f"Manual SELL order executed: {symbol}, size={size}, price={price}")
                messagebox.showinfo("Order Placed", f"SELL order for {size} {symbol} at ${price} placed successfully")
                
                # Update status
                self.status_var.set(f"Manual SELL order executed: {symbol}, size={size}")
                
                # Refresh orders
                self._refresh_orders()
            else:
                self.logger.error(f"Error executing manual SELL order: {result['message']}")
                messagebox.showerror("Error", f"Error executing manual SELL order: {result['message']}")
            
        except Exception as e:
            self.logger.error(f"Error executing manual SELL order: {e}")
            messagebox.showerror("Error", f"Error executing manual SELL order: {e}")
    
    def _manual_close(self):
        """Execute manual close position order."""
        try:
            if not self.trading.is_connected:
                messagebox.showerror("Error", "Not connected to exchange")
                return
            
            symbol = self.symbol_var.get()
            
            # Get position close size percentage
            size_percentage = 100.0
            if self.use_manual_close_size_var.get():
                size_percentage = float(self.position_close_size_var.get())
            
            self.logger.info(f"Manual CLOSE position: {symbol}, {size_percentage}%")
            
            # Execute close
            result = self.trading.close_position(symbol, size_percentage)
            
            if result["success"]:
                self.logger.info(f"Manual CLOSE position executed: {symbol}, {size_percentage}%")
                messagebox.showinfo("Position Closed", f"Closed {size_percentage}% of {symbol} position successfully")
                
                # Update status
                self.status_var.set(f"Manual CLOSE position executed: {symbol}, {size_percentage}%")
                
                # Refresh positions
                self._refresh_positions()
            else:
                self.logger.error(f"Error executing manual CLOSE position: {result['message']}")
                messagebox.showerror("Error", f"Error executing manual CLOSE position: {result['message']}")
            
        except Exception as e:
            self.logger.error(f"Error executing manual CLOSE position: {e}")
            messagebox.showerror("Error", f"Error executing manual CLOSE position: {e}")
    
    def _refresh_positions(self):
        """Refresh the positions display."""
        try:
            # Clear existing positions
            for item in self.positions_tree.get_children():
                self.positions_tree.delete(item)
            
            # Get positions from trading integration
            positions_result = self.trading.get_positions()
            
            if positions_result["success"]:
                positions = positions_result["data"]
                
                for position in positions:
                    symbol = position.get("coin", "")
                    if symbol:
                        size = float(position.get("szi", 0))
                        entry_price = float(position.get("entryPx", 0))
                        
                        # Get current price
                        market_data_result = self.trading.get_market_data(symbol)
                        if market_data_result["success"]:
                            current_price = market_data_result["data"]["price"]
                            
                            # Calculate PnL
                            pnl = size * (current_price - entry_price)
                            pnl_pct = (current_price / entry_price - 1) * 100 if entry_price > 0 else 0
                            
                            # Add to treeview
                            self.positions_tree.insert(
                                "", "end",
                                values=(
                                    symbol,
                                    f"{size:.4f}",
                                    f"${entry_price:.2f}",
                                    f"${current_price:.2f}",
                                    f"${pnl:.2f}",
                                    f"{pnl_pct:.2f}%"
                                )
                            )
            
            self.logger.info("Positions refreshed")
        except Exception as e:
            self.logger.error(f"Error refreshing positions: {e}")
    
    def _refresh_orders(self):
        """Refresh the orders display."""
        try:
            # Clear existing orders
            for item in self.orders_tree.get_children():
                self.orders_tree.delete(item)
            
            # Get orders from trading integration
            # This is a mock implementation since the actual API call would depend on the exchange
            # In a real implementation, you would call the trading integration to get orders
            
            # For now, just display some sample orders
            sample_orders = [
                {
                    "id": "12345",
                    "symbol": self.symbol_var.get(),
                    "side": "BUY",
                    "size": 1.0,
                    "price": 50000.0,
                    "type": "LIMIT",
                    "status": "OPEN",
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                {
                    "id": "12346",
                    "symbol": self.symbol_var.get(),
                    "side": "SELL",
                    "size": 0.5,
                    "price": 55000.0,
                    "type": "LIMIT",
                    "status": "OPEN",
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            ]
            
            for order in sample_orders:
                self.orders_tree.insert(
                    "", "end",
                    values=(
                        order["id"],
                        order["symbol"],
                        order["side"],
                        f"{order['size']:.4f}",
                        f"${order['price']:.2f}",
                        order["type"],
                        order["status"],
                        order["time"]
                    )
                )
            
            self.logger.info("Orders refreshed")
        except Exception as e:
            self.logger.error(f"Error refreshing orders: {e}")
    
    def start_bot(self):
        """Start the trading bot."""
        try:
            if not self.trading.is_connected:
                messagebox.showerror("Error", "Not connected to exchange")
                return
            
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self.trading_loop, daemon=True)
                self.thread.start()
                
                # Update buttons
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                
                # Update status
                self.status_var.set("Bot started")
                
                self.logger.info("Bot started")
        except Exception as e:
            self.logger.error(f"Error starting bot: {e}")
            messagebox.showerror("Error", f"Error starting bot: {e}")
    
    def stop_bot(self):
        """Stop the trading bot."""
        try:
            if self.running:
                self.running = False
                
                if self.thread:
                    self.thread.join(timeout=3.0)
                
                # Update buttons
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
                
                # Update status
                self.status_var.set("Bot stopped")
                
                self.logger.info("Bot stopped")
        except Exception as e:
            self.logger.error(f"Error stopping bot: {e}")
            messagebox.showerror("Error", f"Error stopping bot: {e}")
    
    def trading_loop(self):
        """Main trading loop."""
        try:
            self.logger.info(f"Trading loop started for {self.symbol_var.get()}")
            self.warmup_start = time.time()
            
            # Get initial account info
            account_info = self.trading.get_account_info()
            if account_info["success"]:
                self.start_equity = account_info["data"]["equity"]
            
            while self.running:
                try:
                    time.sleep(float(self.config.get("micro_poll_interval", 2)))
                    
                    if not self.warmup_done:
                        remain = self.warmup_duration - (time.time() - self.warmup_start)
                        if remain > 0:
                            self.logger.info(f"Warmup: {remain:.1f}s left to gather initial data.")
                            self.status_var.set(f"Warmup: {remain:.1f}s left")
                            continue
                        else:
                            self.warmup_done = True
                            self.logger.info("Warmup complete")
                    
                    # Get current symbol
                    symbol = self.symbol_var.get()
                    
                    # Fetch market data
                    market_data_result = self.trading.get_market_data(symbol)
                    if not market_data_result["success"]:
                        self.logger.error(f"Error fetching market data: {market_data_result['message']}")
                        time.sleep(5)
                        continue
                    
                    market_data = market_data_result["data"]
                    px = market_data["price"]
                    
                    # Simulate volume for now
                    volx = 10.0 + np.random.normal(0, 2)
                    
                    now_str = datetime.utcnow().isoformat()
                    
                    # Update historical data
                    if self.hist_data.empty:
                        columns = ["time", "price", "volume", "vol_ma", "fast_ma", "slow_ma", "rsi",
                                "macd_hist", "bb_high", "bb_low", "stoch_k", "stoch_d", "adx", "atr"]
                    else:
                        columns = self.hist_data.columns
                    
                    ncols = len(columns)
                    new_row = pd.DataFrame([[now_str, px, volx] + [np.nan]*(ncols-3)], columns=columns)
                    
                    if not self.hist_data.empty:
                        new_row = new_row.astype(self.hist_data.dtypes.to_dict())
                    
                    self.hist_data = pd.concat([self.hist_data, new_row], ignore_index=True)
                    
                    if len(self.hist_data) > 2000:
                        self.hist_data = self.hist_data.iloc[-2000:]
                    
                    # Compute indicators
                    self._compute_indicators()
                    
                    # Update charts
                    self._update_charts()
                    
                    # Update positions
                    self._update_positions()
                    
                    # Update status
                    self.status_var.set(f"{symbol}: ${px:.2f}")
                    
                except Exception as e:
                    self.logger.error(f"Error in trading loop iteration: {e}")
                    time.sleep(5)
        except Exception as e:
            self.logger.error(f"Error in trading loop: {e}")
    
    def _update_positions(self):
        """Update positions display."""
        try:
            # Get positions from trading integration
            positions_result = self.trading.get_positions()
            
            if positions_result["success"]:
                self.positions = {}
                
                for position in positions_result["data"]:
                    symbol = position.get("coin", "")
                    if symbol:
                        size = float(position.get("szi", 0))
                        entry_price = float(position.get("entryPx", 0))
                        
                        # Get current price
                        market_data_result = self.trading.get_market_data(symbol)
                        if market_data_result["success"]:
                            current_price = market_data_result["data"]["price"]
                            
                            # Calculate PnL
                            pnl = size * (current_price - entry_price)
                            pnl_pct = (current_price / entry_price - 1) * 100 if entry_price > 0 else 0
                            
                            self.positions[symbol] = {
                                "size": size,
                                "entry_price": entry_price,
                                "current_price": current_price,
                                "pnl": pnl,
                                "pnl_pct": pnl_pct
                            }
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    def _compute_indicators(self):
        """Compute technical indicators."""
        if len(self.hist_data) < 30:
            return
        
        try:
            # Get configuration values
            fast_ma = int(self.config.get("fast_ma", 5))
            slow_ma = int(self.config.get("slow_ma", 15))
            rsi_period = int(self.config.get("rsi_period", 14))
            macd_fast = int(self.config.get("macd_fast", 12))
            macd_slow = int(self.config.get("macd_slow", 26))
            macd_signal = int(self.config.get("macd_signal", 9))
            boll_period = int(self.config.get("boll_period", 20))
            boll_stddev = float(self.config.get("boll_stddev", 2.0))
            
            # Compute volume MA
            self.hist_data["vol_ma"] = self.hist_data["volume"].rolling(window=20).mean()
            
            # Compute fast and slow MAs
            self.hist_data["fast_ma"] = self.hist_data["price"].rolling(window=fast_ma).mean()
            self.hist_data["slow_ma"] = self.hist_data["price"].rolling(window=slow_ma).mean()
            
            # Compute RSI
            delta = self.hist_data["price"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=rsi_period).mean()
            avg_loss = loss.rolling(window=rsi_period).mean()
            rs = avg_gain / avg_loss.replace(0, 0.001)
            self.hist_data["rsi"] = 100 - (100 / (1 + rs))
            
            # Compute MACD
            ema_fast = self.hist_data["price"].ewm(span=macd_fast, adjust=False).mean()
            ema_slow = self.hist_data["price"].ewm(span=macd_slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()
            self.hist_data["macd_line"] = macd_line
            self.hist_data["macd_signal"] = signal_line
            self.hist_data["macd_hist"] = macd_line - signal_line
            
            # Compute Bollinger Bands
            self.hist_data["bb_middle"] = self.hist_data["price"].rolling(window=boll_period).mean()
            std = self.hist_data["price"].rolling(window=boll_period).std()
            self.hist_data["bb_high"] = self.hist_data["bb_middle"] + (std * boll_stddev)
            self.hist_data["bb_low"] = self.hist_data["bb_middle"] - (std * boll_stddev)
            
            # Compute Stochastic Oscillator
            low_min = self.hist_data["price"].rolling(window=14).min()
            high_max = self.hist_data["price"].rolling(window=14).max()
            self.hist_data["stoch_k"] = 100 * ((self.hist_data["price"] - low_min) / (high_max - low_min).replace(0, 0.001))
            self.hist_data["stoch_d"] = self.hist_data["stoch_k"].rolling(window=3).mean()
            
            # Compute ADX
            high = self.hist_data["price"] * 1.001  # Dummy high values
            low = self.hist_data["price"] * 0.999   # Dummy low values
            
            # True Range
            tr1 = high - low
            tr2 = abs(high - self.hist_data["price"].shift(1))
            tr3 = abs(low - self.hist_data["price"].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            self.hist_data["atr"] = tr.rolling(window=14).mean()
            
            # Directional Movement
            plus_dm = high - high.shift(1)
            minus_dm = low.shift(1) - low
            plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
            minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
            
            # Smoothed Directional Indicators
            smoothed_plus_dm = plus_dm.rolling(window=14).sum()
            smoothed_minus_dm = minus_dm.rolling(window=14).sum()
            smoothed_tr = tr.rolling(window=14).sum()
            
            # Directional Indicators
            plus_di = 100 * (smoothed_plus_dm / smoothed_tr.replace(0, 0.001))
            minus_di = 100 * (smoothed_minus_dm / smoothed_tr.replace(0, 0.001))
            
            # ADX
            dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 0.001))
            self.hist_data["adx"] = dx.rolling(window=14).mean()
            
        except Exception as e:
            self.logger.error(f"Error computing indicators: {e}")
    
    def _update_charts(self):
        """Update charts with latest data."""
        if len(self.hist_data) < 5:
            return
        
        try:
            # Get last N data points
            n = min(100, len(self.hist_data))
            data = self.hist_data.iloc[-n:]
            
            # Update price chart
            self.ax1.clear()
            self.ax1.set_title(f"{self.symbol_var.get()} Price")
            self.ax1.plot(range(len(data)), data["price"], label="Price", color=self.style_manager.get_color("chart_line"))
            
            if "fast_ma" in data.columns and not data["fast_ma"].isna().all():
                self.ax1.plot(range(len(data)), data["fast_ma"], label=f"Fast MA ({self.config.get('fast_ma', 5)})", color=self.style_manager.get_color("accent_color"))
            
            if "slow_ma" in data.columns and not data["slow_ma"].isna().all():
                self.ax1.plot(range(len(data)), data["slow_ma"], label=f"Slow MA ({self.config.get('slow_ma', 15)})", color=self.style_manager.get_color("secondary_accent"))
            
            if "bb_high" in data.columns and "bb_low" in data.columns and not data["bb_high"].isna().all():
                self.ax1.plot(range(len(data)), data["bb_high"], "--", label="BB Upper", color=self.style_manager.get_color("error_color"))
                self.ax1.plot(range(len(data)), data["bb_low"], "--", label="BB Lower", color=self.style_manager.get_color("success_color"))
            
            self.ax1.legend(loc="upper left")
            self.ax1.grid(True)
            
            # Update volume chart
            self.ax2.clear()
            self.ax2.set_title("Volume")
            self.ax2.bar(range(len(data)), data["volume"], label="Volume", color=self.style_manager.get_color("tertiary_accent"))
            
            if "vol_ma" in data.columns and not data["vol_ma"].isna().all():
                self.ax2.plot(range(len(data)), data["vol_ma"], "-", label="Volume MA (20)", color=self.style_manager.get_color("error_color"))
            
            self.ax2.legend(loc="upper left")
            self.ax2.grid(True)
            
            # Update canvas
            self.canvas.draw()
            
            # Update indicators chart
            # RSI
            self.ind_ax1.clear()
            self.ind_ax1.set_title("RSI")
            
            if "rsi" in data.columns and not data["rsi"].isna().all():
                self.ind_ax1.plot(range(len(data)), data["rsi"], "-", label="RSI", color=self.style_manager.get_color("accent_color"))
                self.ind_ax1.axhline(y=70, color=self.style_manager.get_color("error_color"), linestyle='-')
                self.ind_ax1.axhline(y=30, color=self.style_manager.get_color("success_color"), linestyle='-')
                self.ind_ax1.set_ylim(0, 100)
            
            self.ind_ax1.grid(True)
            self.ind_ax1.legend(loc="upper left")
            
            # MACD
            self.ind_ax2.clear()
            self.ind_ax2.set_title("MACD")
            
            if "macd_line" in data.columns and "macd_signal" in data.columns and not data["macd_line"].isna().all():
                self.ind_ax2.plot(range(len(data)), data["macd_line"], "-", label="MACD", color=self.style_manager.get_color("accent_color"))
                self.ind_ax2.plot(range(len(data)), data["macd_signal"], "-", label="Signal", color=self.style_manager.get_color("secondary_accent"))
                
                # Plot histogram
                if "macd_hist" in data.columns:
                    for i in range(len(data)):
                        if not np.isnan(data["macd_hist"].iloc[i]):
                            color = self.style_manager.get_color("success_color") if data["macd_hist"].iloc[i] >= 0 else self.style_manager.get_color("error_color")
                            self.ind_ax2.bar(i, data["macd_hist"].iloc[i], color=color, width=0.8)
            
            self.ind_ax2.grid(True)
            self.ind_ax2.legend(loc="upper left")
            
            # Bollinger Bands
            self.ind_ax3.clear()
            self.ind_ax3.set_title("Price with Bollinger Bands")
            
            self.ind_ax3.plot(range(len(data)), data["price"], "-", label="Price", color=self.style_manager.get_color("chart_line"))
            
            if "bb_middle" in data.columns and not data["bb_middle"].isna().all():
                self.ind_ax3.plot(range(len(data)), data["bb_middle"], "-", label="BB Middle", color=self.style_manager.get_color("text_secondary"))
                self.ind_ax3.plot(range(len(data)), data["bb_high"], "--", label="BB Upper", color=self.style_manager.get_color("error_color"))
                self.ind_ax3.plot(range(len(data)), data["bb_low"], "--", label="BB Lower", color=self.style_manager.get_color("success_color"))
            
            self.ind_ax3.grid(True)
            self.ind_ax3.legend(loc="upper left")
            
            # ADX
            self.ind_ax4.clear()
            self.ind_ax4.set_title("ADX")
            
            if "adx" in data.columns and not data["adx"].isna().all():
                self.ind_ax4.plot(range(len(data)), data["adx"], "-", label="ADX", color=self.style_manager.get_color("accent_color"))
                self.ind_ax4.axhline(y=25, color=self.style_manager.get_color("error_color"), linestyle='--')
            
            self.ind_ax4.grid(True)
            self.ind_ax4.legend(loc="upper left")
            
            # Update canvas
            self.ind_canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error updating charts: {e}")
    
    def _schedule_updates(self):
        """Schedule periodic updates."""
        try:
            # Schedule position updates
            self.root.after(10000, self._scheduled_position_update)
            
            # Schedule order updates
            self.root.after(15000, self._scheduled_order_update)
        except Exception as e:
            self.logger.error(f"Error scheduling updates: {e}")
    
    def _scheduled_position_update(self):
        """Scheduled position update."""
        try:
            if self.trading.is_connected:
                self._refresh_positions()
            
            # Reschedule
            self.root.after(10000, self._scheduled_position_update)
        except Exception as e:
            self.logger.error(f"Error in scheduled position update: {e}")
            # Reschedule even on error
            self.root.after(10000, self._scheduled_position_update)
    
    def _scheduled_order_update(self):
        """Scheduled order update."""
        try:
            if self.trading.is_connected:
                self._refresh_orders()
            
            # Reschedule
            self.root.after(15000, self._scheduled_order_update)
        except Exception as e:
            self.logger.error(f"Error in scheduled order update: {e}")
            # Reschedule even on error
            self.root.after(15000, self._scheduled_order_update)
    
    def on_closing(self):
        """Handle window closing event."""
        try:
            if messagebox.askokcancel("Quit", "Do you want to quit?"):
                self.running = False
                self.log_consumer_running = False
                
                if self.thread:
                    self.thread.join(timeout=1.0)
                
                if self.log_consumer_thread:
                    self.log_consumer_thread.join(timeout=1.0)
                
                self.root.destroy()
        except Exception as e:
            print(f"Error during closing: {e}")
            self.root.destroy()

###############################################################################
# Main Entry Point
###############################################################################
def main():
    """Main entry point for the application."""
    try:
        app = EnhancedTradingBotGUI()
        app.root.mainloop()
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
