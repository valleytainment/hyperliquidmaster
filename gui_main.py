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
        
        self.stop_button = ttk.Button(top_frame, text="Stop Bot", command=self.stop_bot, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Create manual trade buttons
        ttk.Label(top_frame, text="Manual:").pack(side=tk.LEFT, padx=(10, 5))
        
        self.buy_button = ttk.Button(top_frame, text="Buy", command=self._manual_buy)
        self.buy_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.sell_button = ttk.Button(top_frame, text="Sell", command=self._manual_sell)
        self.sell_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.close_button = ttk.Button(top_frame, text="Close Position", command=self._manual_close)
        self.close_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Create size entry
        ttk.Label(top_frame, text="Size:").pack(side=tk.LEFT, padx=(10, 5))
        self.size_var = tk.StringVar(value=str(self.config.get("manual_entry_size", 1.0)))
        size_entry = ttk.Entry(top_frame, textvariable=self.size_var, width=8)
        size_entry.pack(side=tk.LEFT, padx=(0, 5))
        
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
        ttk.Label(settings_grid, text="Technical Indicators", font=("TkDefaultFont", 12, "bold")).grid(
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
        ttk.Label(settings_grid, text="Risk Management", font=("TkDefaultFont", 12, "bold")).grid(
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
        ttk.Label(settings_grid, text="Order Settings", font=("TkDefaultFont", 12, "bold")).grid(
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
        self.use_manual_close_size_var = tk.BooleanVar(value=self.config.get("use_manual_close_size", True))
        ttk.Checkbutton(settings_grid, text="Use Manual Close Size", variable=self.use_manual_close_size_var).grid(
            row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        
        ttk.Label(settings_grid, text="Position Close Size:").grid(row=row, column=0, sticky=tk.W)
        self.position_close_size_var = tk.StringVar(value=str(self.config.get("position_close_size", 10.0)))
        ttk.Entry(settings_grid, textvariable=self.position_close_size_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        # Save settings button
        ttk.Button(settings_grid, text="Save Settings", command=self._save_settings).grid(
            row=row, column=0, columnspan=2, pady=(10, 0))
        
        # Create second column for advanced settings
        row = 0
        col_offset = 3
        
        # Advanced settings
        ttk.Label(settings_grid, text="Advanced Settings", font=("TkDefaultFont", 12, "bold")).grid(
            row=row, column=col_offset, columnspan=2, sticky=tk.W, pady=(0, 10))
        row += 1
        
        # Neural network settings
        ttk.Label(settings_grid, text="NN Lookback Bars:").grid(row=row, column=col_offset, sticky=tk.W)
        self.nn_lookback_bars_var = tk.StringVar(value=str(self.config.get("nn_lookback_bars", 30)))
        ttk.Entry(settings_grid, textvariable=self.nn_lookback_bars_var, width=8).grid(row=row, column=col_offset+1, sticky=tk.W)
        row += 1
        
        ttk.Label(settings_grid, text="NN Hidden Size:").grid(row=row, column=col_offset, sticky=tk.W)
        self.nn_hidden_size_var = tk.StringVar(value=str(self.config.get("nn_hidden_size", 64)))
        ttk.Entry(settings_grid, textvariable=self.nn_hidden_size_var, width=8).grid(row=row, column=col_offset+1, sticky=tk.W)
        row += 1
        
        ttk.Label(settings_grid, text="NN Learning Rate:").grid(row=row, column=col_offset, sticky=tk.W)
        self.nn_lr_var = tk.StringVar(value=str(self.config.get("nn_lr", 0.0003)))
        ttk.Entry(settings_grid, textvariable=self.nn_lr_var, width=8).grid(row=row, column=col_offset+1, sticky=tk.W)
        row += 1
        
        # Synergy confidence threshold
        ttk.Label(settings_grid, text="Synergy Threshold:").grid(row=row, column=col_offset, sticky=tk.W)
        self.synergy_conf_threshold_var = tk.StringVar(value=str(self.config.get("synergy_conf_threshold", 0.8)))
        ttk.Entry(settings_grid, textvariable=self.synergy_conf_threshold_var, width=8).grid(row=row, column=col_offset+1, sticky=tk.W)
        row += 1
        
        # Circuit breaker threshold
        ttk.Label(settings_grid, text="Circuit Breaker %:").grid(row=row, column=col_offset, sticky=tk.W)
        self.circuit_breaker_threshold_var = tk.StringVar(value=str(self.config.get("circuit_breaker_threshold", 0.05)))
        ttk.Entry(settings_grid, textvariable=self.circuit_breaker_threshold_var, width=8).grid(row=row, column=col_offset+1, sticky=tk.W)
        row += 1
        
        # Taker fee
        ttk.Label(settings_grid, text="Taker Fee:").grid(row=row, column=col_offset, sticky=tk.W)
        self.taker_fee_var = tk.StringVar(value=str(self.config.get("taker_fee", 0.00042)))
        ttk.Entry(settings_grid, textvariable=self.taker_fee_var, width=8).grid(row=row, column=col_offset+1, sticky=tk.W)
        row += 1
        
        # API settings
        ttk.Label(settings_grid, text="API Settings", font=("TkDefaultFont", 12, "bold")).grid(
            row=row, column=col_offset, columnspan=2, sticky=tk.W, pady=(10, 10))
        row += 1
        
        # API URL
        ttk.Label(settings_grid, text="API URL:").grid(row=row, column=col_offset, sticky=tk.W)
        self.api_url_var = tk.StringVar(value=str(self.config.get("api_url", "https://api.hyperliquid.xyz")))
        ttk.Entry(settings_grid, textvariable=self.api_url_var, width=30).grid(row=row, column=col_offset+1, sticky=tk.W)
        row += 1
        
        # Poll interval
        ttk.Label(settings_grid, text="Poll Interval (s):").grid(row=row, column=col_offset, sticky=tk.W)
        self.poll_interval_var = tk.StringVar(value=str(self.config.get("poll_interval_seconds", 2)))
        ttk.Entry(settings_grid, textvariable=self.poll_interval_var, width=8).grid(row=row, column=col_offset+1, sticky=tk.W)
        row += 1
        
        # Micro poll interval
        ttk.Label(settings_grid, text="Micro Poll Interval (s):").grid(row=row, column=col_offset, sticky=tk.W)
        self.micro_poll_interval_var = tk.StringVar(value=str(self.config.get("micro_poll_interval", 2)))
        ttk.Entry(settings_grid, textvariable=self.micro_poll_interval_var, width=8).grid(row=row, column=col_offset+1, sticky=tk.W)
        row += 1
        
        # Create positions tab
        positions_frame = ttk.Frame(notebook)
        notebook.add(positions_frame, text="Positions")
        
        # Create positions treeview
        columns = ("symbol", "side", "size", "entry_price", "current_price", "pnl", "pnl_pct")
        self.positions_tree = ttk.Treeview(positions_frame, columns=columns, show="headings")
        
        # Define headings
        self.positions_tree.heading("symbol", text="Symbol")
        self.positions_tree.heading("side", text="Side")
        self.positions_tree.heading("size", text="Size")
        self.positions_tree.heading("entry_price", text="Entry Price")
        self.positions_tree.heading("current_price", text="Current Price")
        self.positions_tree.heading("pnl", text="PnL")
        self.positions_tree.heading("pnl_pct", text="PnL %")
        
        # Define columns
        self.positions_tree.column("symbol", width=100)
        self.positions_tree.column("side", width=100)
        self.positions_tree.column("size", width=100)
        self.positions_tree.column("entry_price", width=100)
        self.positions_tree.column("current_price", width=100)
        self.positions_tree.column("pnl", width=100)
        self.positions_tree.column("pnl_pct", width=100)
        
        # Add scrollbar
        positions_scrollbar = ttk.Scrollbar(positions_frame, orient=tk.VERTICAL, command=self.positions_tree.yview)
        self.positions_tree.configure(yscrollcommand=positions_scrollbar.set)
        
        # Pack treeview and scrollbar
        self.positions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        positions_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create log tab
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="Logs")
        
        # Create log text widget
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Create status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
        
        # Initialize charts
        self._init_charts()
    
    def _init_charts(self):
        """Initialize empty charts."""
        # Price chart
        self.ax1.clear()
        self.ax1.set_title("Price Chart")
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
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 0.001)
            self.hist_data["adx"] = dx.rolling(window=14).mean()
            
        except Exception as e:
            self.logger.error(f"Error computing indicators: {e}")
    
    def _update_charts(self):
        """Update charts with latest data."""
        if len(self.hist_data) < 30:
            return
        
        try:
            # Get last 100 data points for display
            display_data = self.hist_data.tail(100).copy()
            
            # Convert time to datetime for better display
            display_data["time"] = pd.to_datetime(display_data["time"])
            
            # Update price chart
            self.ax1.clear()
            self.ax1.set_title("Price Chart")
            self.ax1.set_ylabel("Price")
            self.ax1.grid(True)
            
            # Plot price
            self.ax1.plot(display_data["time"], display_data["price"], label="Price")
            
            # Plot MAs if available
            if "fast_ma" in display_data.columns and not display_data["fast_ma"].isna().all():
                self.ax1.plot(display_data["time"], display_data["fast_ma"], label=f"Fast MA ({self.config.get('fast_ma', 5)})")
            
            if "slow_ma" in display_data.columns and not display_data["slow_ma"].isna().all():
                self.ax1.plot(display_data["time"], display_data["slow_ma"], label=f"Slow MA ({self.config.get('slow_ma', 15)})")
            
            # Plot Bollinger Bands if available
            if "bb_high" in display_data.columns and not display_data["bb_high"].isna().all():
                self.ax1.plot(display_data["time"], display_data["bb_high"], 'r--', label="BB Upper")
                self.ax1.plot(display_data["time"], display_data["bb_low"], 'g--', label="BB Lower")
            
            self.ax1.legend(loc="upper left")
            
            # Format x-axis
            self.ax1.tick_params(axis='x', rotation=45)
            
            # Update volume chart
            self.ax2.clear()
            self.ax2.set_title("Volume")
            self.ax2.set_ylabel("Volume")
            self.ax2.set_xlabel("Time")
            self.ax2.grid(True)
            
            # Plot volume
            self.ax2.bar(display_data["time"], display_data["volume"], label="Volume", alpha=0.5)
            
            # Plot volume MA if available
            if "vol_ma" in display_data.columns and not display_data["vol_ma"].isna().all():
                self.ax2.plot(display_data["time"], display_data["vol_ma"], 'r', label="Volume MA")
            
            self.ax2.legend(loc="upper left")
            
            # Format x-axis
            self.ax2.tick_params(axis='x', rotation=45)
            
            # Adjust layout and draw
            self.fig.tight_layout()
            self.canvas.draw()
            
            # Update indicators chart
            # RSI
            self.ind_ax1.clear()
            self.ind_ax1.set_title("RSI")
            self.ind_ax1.axhline(y=70, color='r', linestyle='-')
            self.ind_ax1.axhline(y=30, color='g', linestyle='-')
            self.ind_ax1.set_ylim(0, 100)
            self.ind_ax1.grid(True)
            
            if "rsi" in display_data.columns and not display_data["rsi"].isna().all():
                self.ind_ax1.plot(display_data["time"], display_data["rsi"], label="RSI")
            
            # MACD
            self.ind_ax2.clear()
            self.ind_ax2.set_title("MACD")
            self.ind_ax2.grid(True)
            
            if "macd_line" in display_data.columns and not display_data["macd_line"].isna().all():
                self.ind_ax2.plot(display_data["time"], display_data["macd_line"], label="MACD")
                self.ind_ax2.plot(display_data["time"], display_data["macd_signal"], label="Signal")
                self.ind_ax2.bar(display_data["time"], display_data["macd_hist"], label="Histogram", alpha=0.5)
                self.ind_ax2.axhline(y=0, color='k', linestyle='-')
                self.ind_ax2.legend(loc="upper left")
            
            # Bollinger Bands
            self.ind_ax3.clear()
            self.ind_ax3.set_title("Bollinger Bands")
            self.ind_ax3.grid(True)
            
            if "bb_high" in display_data.columns and not display_data["bb_high"].isna().all():
                self.ind_ax3.plot(display_data["time"], display_data["price"], label="Price")
                self.ind_ax3.plot(display_data["time"], display_data["bb_middle"], label="Middle")
                self.ind_ax3.plot(display_data["time"], display_data["bb_high"], 'r--', label="Upper")
                self.ind_ax3.plot(display_data["time"], display_data["bb_low"], 'g--', label="Lower")
                self.ind_ax3.legend(loc="upper left")
            
            # ADX
            self.ind_ax4.clear()
            self.ind_ax4.set_title("ADX")
            self.ind_ax4.grid(True)
            
            if "adx" in display_data.columns and not display_data["adx"].isna().all():
                self.ind_ax4.plot(display_data["time"], display_data["adx"], label="ADX")
                self.ind_ax4.axhline(y=25, color='r', linestyle='--')
                self.ind_ax4.legend(loc="upper left")
            
            # Format x-axis
            for ax in [self.ind_ax1, self.ind_ax2, self.ind_ax3, self.ind_ax4]:
                ax.tick_params(axis='x', rotation=45)
            
            # Adjust layout and draw
            self.ind_fig.tight_layout()
            self.ind_canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error updating charts: {e}")
    
    def _update_positions(self):
        """Update positions display."""
        try:
            # Clear existing items
            for item in self.positions_tree.get_children():
                self.positions_tree.delete(item)
            
            # Get positions
            positions = self._get_positions()
            
            # Add positions to treeview
            for symbol, pos in positions.items():
                side = "LONG" if pos["side"] == 1 else "SHORT"
                size = pos["size"]
                entry_price = pos["entry_price"]
                current_price = pos["current_price"]
                
                # Calculate PnL
                if side == "LONG":
                    pnl = size * (current_price - entry_price)
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                else:
                    pnl = size * (entry_price - current_price)
                    pnl_pct = (entry_price - current_price) / entry_price * 100
                
                # Add to treeview
                self.positions_tree.insert("", "end", values=(
                    symbol, side, f"{size:.4f}", f"${entry_price:.2f}", 
                    f"${current_price:.2f}", f"${pnl:.2f}", f"{pnl_pct:.2f}%"
                ))
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    def _get_positions(self) -> Dict:
        """Get current positions."""
        # This would call the exchange adapter to get the positions
        # For now, just return dummy values
        return {
            "BTC": {
                "side": 1,  # 1 for long, 2 for short
                "size": 0.1,
                "entry_price": 50000.0,
                "current_price": 50100.0
            },
            "ETH": {
                "side": 2,  # 1 for long, 2 for short
                "size": 1.0,
                "entry_price": 3000.0,
                "current_price": 2950.0
            }
        }
    
    def on_closing(self):
        """Handle window closing event."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            # Stop bot if running
            if self.running:
                self.stop_bot()
            
            # Stop log consumer
            self.log_consumer_running = False
            if self.log_consumer_thread:
                self.log_consumer_thread.join(timeout=1.0)
            
            # Destroy window
            self.root.destroy()
    
    def run(self):
        """Run the GUI application."""
        self.root.mainloop()

def main():
    """Main entry point."""
    # Get configuration path from command line arguments
    config_path = "config.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # Create and run the GUI application
    app = EnhancedTradingBotGUI(config_path)
    app.run()

if __name__ == "__main__":
    main()
