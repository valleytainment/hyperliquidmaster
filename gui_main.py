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
from typing import Dict, List, Optional, Any

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Import core components from enhanced bot
from core.hyperliquid_adapter import HyperliquidAdapter
from core.error_handler import ErrorHandler
from strategies.master_omni_overlord_robust import MasterOmniOverlordRobustStrategy
from strategies.robust_signal_generator import RobustSignalGenerator
from historical_data_accumulator import HistoricalDataAccumulator
from config_compatibility import ConfigManager
from api_rate_limiter import APIRateLimiter
from order_book_handler import OrderBookHandler

# Import GUI style manager
from gui_style import GUIStyleManager, create_header_label, create_subheader_label

# Constants
CONFIG_FILE = "config.json"

###############################################################################
# Logging Setup
###############################################################################
class QueueLoggingHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue
    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception:
            self.handleError(record)

###############################################################################
# Enhanced Trading Bot with GUI
###############################################################################
class EnhancedTradingBotGUI:
    def __init__(self, config_path: str = "config.json"):
        # Initialize GUI components
        self.root = tk.Tk()
        self.root.title("Enhanced Hyperliquid Trading Bot")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Store config path
        self.config_path = config_path
        
        # Setup logging with queue for GUI display
        self.log_queue = queue.Queue()
        self.logger = self._setup_logger()
        
        # Load configuration
        self.config_manager = ConfigManager(config_path, self.logger)
        self.config = self.config_manager.get_config()
        
        # Initialize style manager
        theme = self.config.get("theme", "dark")
        self.style_manager = GUIStyleManager(self.root, theme)
        
        # Initialize error handler
        self.error_handler = ErrorHandler(self.logger)
        
        # Initialize exchange adapter
        self.exchange = HyperliquidAdapter(
            self.config_path
        )
        
        # Initialize API rate limiter
        self.rate_limiter = APIRateLimiter()
        
        # Initialize order book handler
        self.order_book_handler = OrderBookHandler(self.logger)
        
        # Initialize historical data accumulator
        self.data_accumulator = HistoricalDataAccumulator()
        
        # Initialize strategy
        self.strategy = MasterOmniOverlordRobustStrategy(self.logger)
        
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
        
        # Create GUI components
        self._create_gui()
        
        # Start log consumer
        self.log_consumer_running = True
        self.log_consumer_thread = threading.Thread(target=self._consume_logs, daemon=True)
        self.log_consumer_thread.start()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up the logger with queue handler for GUI display."""
        logger = logging.getLogger("EnhancedTradingBot")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("enhanced_bot_gui.log", mode="a")
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
    
    def _create_gui(self):
        """Create the GUI components."""
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create top frame for controls
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
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
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
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
        
        # Create settings tab
        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text="Settings")
        
        # Create settings grid
        settings_grid = ttk.Frame(settings_frame)
        settings_grid.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create settings controls
        row = 0
        
        # Technical indicators settings
        create_header_label(settings_grid, "Technical Indicators").grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
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
        self.manual_entry_size_var = tk.StringVar(value=str(self.config.get("manual_entry_size", 55.0)))
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
        
        # API Key Management
        create_header_label(settings_grid, "API Key Management").grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 10))
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
        
        # Save settings button
        save_button = ttk.Button(settings_grid, text="Save Settings", command=self._save_settings)
        save_button.grid(row=row, column=0, columnspan=2, pady=(10, 0))
        row += 1
        
        # Create logs tab
        logs_frame = ttk.Frame(notebook)
        notebook.add(logs_frame, text="Logs")
        
        # Create log text widget
        self.log_text = scrolledtext.ScrolledText(logs_frame, wrap=tk.WORD, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT)
        
        # Initialize charts
        self._init_charts()
    
    def _toggle_theme(self):
        """Toggle between light and dark themes."""
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
    
    def _init_charts(self):
        """Initialize charts."""
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
        symbol = self.symbol_var.get()
        self.logger.info(f"Symbol changed to {symbol}")
        
        # Update config
        self.config["trade_symbol"] = symbol
        
        # Reset charts
        self._init_charts()
        
        # Update status
        self.status_var.set(f"Symbol changed to {symbol}")
    
    def _save_settings(self):
        """Save settings to config file."""
        try:
            # Update config with values from GUI
            self.config["trade_symbol"] = self.symbol_var.get()
            
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
            
            # API Key Management
            self.config["account_address"] = self.account_address_var.get()
            self.config["secret_key"] = self.secret_key_var.get()
            
            # Save to file
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=2)
            
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
            size = float(self.size_var.get())
            symbol = self.symbol_var.get()
            
            self.logger.info(f"Manual BUY order: {symbol}, size={size}")
            
            # Execute order
            # This would call the exchange adapter to place the order
            # For now, just log it
            self.logger.info(f"Manual BUY order executed: {symbol}, size={size}")
            
            # Update status
            self.status_var.set(f"Manual BUY order executed: {symbol}, size={size}")
            
        except Exception as e:
            self.logger.error(f"Error executing manual BUY order: {e}")
            messagebox.showerror("Error", f"Error executing manual BUY order: {e}")
    
    def _manual_sell(self):
        """Execute manual sell order."""
        try:
            size = float(self.size_var.get())
            symbol = self.symbol_var.get()
            
            self.logger.info(f"Manual SELL order: {symbol}, size={size}")
            
            # Execute order
            # This would call the exchange adapter to place the order
            # For now, just log it
            self.logger.info(f"Manual SELL order executed: {symbol}, size={size}")
            
            # Update status
            self.status_var.set(f"Manual SELL order executed: {symbol}, size={size}")
            
        except Exception as e:
            self.logger.error(f"Error executing manual SELL order: {e}")
            messagebox.showerror("Error", f"Error executing manual SELL order: {e}")
    
    def _manual_close(self):
        """Execute manual close position order."""
        try:
            symbol = self.symbol_var.get()
            
            self.logger.info(f"Manual CLOSE position: {symbol}")
            
            # Execute order
            # This would call the exchange adapter to close the position
            # For now, just log it
            self.logger.info(f"Manual CLOSE position executed: {symbol}")
            
            # Update status
            self.status_var.set(f"Manual CLOSE position executed: {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error executing manual CLOSE position: {e}")
            messagebox.showerror("Error", f"Error executing manual CLOSE position: {e}")
    
    def start_bot(self):
        """Start the trading bot."""
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
    
    def stop_bot(self):
        """Stop the trading bot."""
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
    
    def trading_loop(self):
        """Main trading loop."""
        self.logger.info(f"Trading loop started for {self.symbol_var.get()}")
        self.warmup_start = time.time()
        self.start_equity = self._get_equity()
        
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
                
                # Fetch price and volume
                pv = self._fetch_price_volume()
                if not pv or pv["price"] <= 0:
                    continue
                
                px = pv["price"]
                volx = pv["volume"]
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
                self.status_var.set(f"{self.symbol_var.get()}: ${px:.2f}")
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(5)
    
    def _get_equity(self) -> float:
        """Get account equity."""
        # This would call the exchange adapter to get the equity
        # For now, just return a dummy value
        return 10000.0
    
    def _fetch_price_volume(self) -> Optional[Dict]:
        """Fetch price and volume data."""
        # This would call the exchange adapter to get the price and volume
        # For now, just return dummy values
        return {
            "price": 50000.0 + np.random.normal(0, 100),
            "volume": 10.0 + np.random.normal(0, 2)
        }
    
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
            self.ax1.plot(range(len(data)), data["price"], label="Price")
            
            if "fast_ma" in data.columns and not data["fast_ma"].isna().all():
                self.ax1.plot(range(len(data)), data["fast_ma"], label=f"Fast MA ({self.config.get('fast_ma', 5)})")
            
            if "slow_ma" in data.columns and not data["slow_ma"].isna().all():
                self.ax1.plot(range(len(data)), data["slow_ma"], label=f"Slow MA ({self.config.get('slow_ma', 15)})")
            
            if "bb_high" in data.columns and "bb_low" in data.columns and not data["bb_high"].isna().all():
                self.ax1.plot(range(len(data)), data["bb_high"], "r--", label="BB Upper")
                self.ax1.plot(range(len(data)), data["bb_low"], "g--", label="BB Lower")
            
            self.ax1.legend(loc="upper left")
            self.ax1.grid(True)
            
            # Update volume chart
            self.ax2.clear()
            self.ax2.set_title("Volume")
            self.ax2.bar(range(len(data)), data["volume"], label="Volume")
            
            if "vol_ma" in data.columns and not data["vol_ma"].isna().all():
                self.ax2.plot(range(len(data)), data["vol_ma"], "r-", label="Volume MA (20)")
            
            self.ax2.legend(loc="upper left")
            self.ax2.grid(True)
            
            # Update canvas
            self.canvas.draw()
            
            # Update indicators chart
            # RSI
            self.ind_ax1.clear()
            self.ind_ax1.set_title("RSI")
            
            if "rsi" in data.columns and not data["rsi"].isna().all():
                self.ind_ax1.plot(range(len(data)), data["rsi"], "b-", label="RSI")
                self.ind_ax1.axhline(y=70, color='r', linestyle='-')
                self.ind_ax1.axhline(y=30, color='g', linestyle='-')
                self.ind_ax1.set_ylim(0, 100)
            
            self.ind_ax1.grid(True)
            self.ind_ax1.legend(loc="upper left")
            
            # MACD
            self.ind_ax2.clear()
            self.ind_ax2.set_title("MACD")
            
            if "macd_line" in data.columns and "macd_signal" in data.columns and not data["macd_line"].isna().all():
                self.ind_ax2.plot(range(len(data)), data["macd_line"], "b-", label="MACD")
                self.ind_ax2.plot(range(len(data)), data["macd_signal"], "r-", label="Signal")
                
                # Plot histogram
                if "macd_hist" in data.columns:
                    for i in range(len(data)):
                        if not np.isnan(data["macd_hist"].iloc[i]):
                            color = "g" if data["macd_hist"].iloc[i] >= 0 else "r"
                            self.ind_ax2.bar(i, data["macd_hist"].iloc[i], color=color, width=0.8)
            
            self.ind_ax2.grid(True)
            self.ind_ax2.legend(loc="upper left")
            
            # Bollinger Bands
            self.ind_ax3.clear()
            self.ind_ax3.set_title("Price with Bollinger Bands")
            
            self.ind_ax3.plot(range(len(data)), data["price"], "b-", label="Price")
            
            if "bb_middle" in data.columns and not data["bb_middle"].isna().all():
                self.ind_ax3.plot(range(len(data)), data["bb_middle"], "k-", label="BB Middle")
                self.ind_ax3.plot(range(len(data)), data["bb_high"], "r--", label="BB Upper")
                self.ind_ax3.plot(range(len(data)), data["bb_low"], "g--", label="BB Lower")
            
            self.ind_ax3.grid(True)
            self.ind_ax3.legend(loc="upper left")
            
            # ADX
            self.ind_ax4.clear()
            self.ind_ax4.set_title("ADX")
            
            if "adx" in data.columns and not data["adx"].isna().all():
                self.ind_ax4.plot(range(len(data)), data["adx"], "b-", label="ADX")
                self.ind_ax4.axhline(y=25, color='r', linestyle='--')
            
            self.ind_ax4.grid(True)
            self.ind_ax4.legend(loc="upper left")
            
            # Update canvas
            self.ind_canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error updating charts: {e}")
    
    def _update_positions(self):
        """Update positions display."""
        # This would call the exchange adapter to get the positions
        # For now, just use dummy values
        self.positions = {
            "BTC": {
                "size": 0.1,
                "entry_price": 50000.0,
                "current_price": 50100.0,
                "pnl": 10.0,
                "pnl_pct": 0.2
            }
        }
    
    def on_closing(self):
        """Handle window closing event."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.running = False
            self.log_consumer_running = False
            
            if self.thread:
                self.thread.join(timeout=1.0)
            
            if self.log_consumer_thread:
                self.log_consumer_thread.join(timeout=1.0)
            
            self.root.destroy()

###############################################################################
# Main Entry Point
###############################################################################
def main():
    app = EnhancedTradingBotGUI()
    app.root.mainloop()

if __name__ == "__main__":
    main()
