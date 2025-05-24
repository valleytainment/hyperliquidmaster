#!/usr/bin/env python3
"""
Enhanced GUI for Hyperliquid Trading Bot

This module provides an optimized GUI with responsive layout, real-time data visualization,
error visualization, market regime display, enhanced performance metrics, and improved threading.
"""

import sys
import os
import logging
import threading
import time
import json
import queue
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better performance
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Import custom modules
from core.hyperliquid_adapter import HyperliquidAdapter
from strategies.master_omni_overlord_robust import MasterOmniOverlordRobustStrategy
from strategies.robust_signal_generator import RobustSignalGenerator
from strategies.advanced_technical_indicators import AdvancedTechnicalIndicators
from enhanced_historical_data_accumulator import EnhancedHistoricalDataAccumulator
from order_book_handler import OrderBookHandler
from api_rate_limiter import APIRateLimiter
from error_handling import (
    ErrorHandler, ErrorSeverity, ErrorCategory, TradingError,
    APIError, DataError, CalculationError, SignalError, OrderError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("logs/gui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
UPDATE_INTERVAL = 1000  # GUI update interval in milliseconds
CHART_UPDATE_INTERVAL = 5000  # Chart update interval in milliseconds
MAX_CHART_POINTS = 100  # Maximum number of points to display on charts
ERROR_COLORS = {
    ErrorSeverity.INFO: "#4CAF50",  # Green
    ErrorSeverity.WARNING: "#FFC107",  # Yellow
    ErrorSeverity.ERROR: "#FF5722",  # Orange
    ErrorSeverity.CRITICAL: "#F44336",  # Red
    ErrorSeverity.FATAL: "#9C27B0"  # Purple
}
MARKET_REGIME_COLORS = {
    "trending_up": "#4CAF50",  # Green
    "trending_down": "#F44336",  # Red
    "ranging": "#2196F3",  # Blue
    "volatile": "#FF9800",  # Orange
    "unknown": "#9E9E9E"  # Gray
}

class EnhancedGUI:
    """
    Enhanced GUI for Hyperliquid Trading Bot with responsive layout,
    real-time data visualization, and improved threading.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the GUI.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.error_handler = ErrorHandler(
            log_dir="logs",
            max_retries=3,
            notification_callback=self._handle_error_notification
        )
        self.api_rate_limiter = APIRateLimiter()
        self.exchange_adapter = None
        self.order_book_handler = None
        self.data_accumulator = None
        self.signal_generator = None
        self.strategy = None
        
        # Initialize data structures
        self.symbols = self.config.get("symbols", ["BTC", "ETH", "SOL"])
        self.current_symbol = self.symbols[0] if self.symbols else "BTC"
        self.market_data = {}
        self.signals = {}
        self.order_book = {}
        self.performance_metrics = {}
        self.error_log = []
        self.max_error_log = 100
        
        # Initialize threading components
        self.running = False
        self.data_thread = None
        self.signal_thread = None
        self.update_queue = queue.Queue()
        self.lock = threading.RLock()
        
        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("Hyperliquid Trading Bot")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Set theme
        self._set_theme()
        
        # Create GUI components
        self._create_menu()
        self._create_layout()
        self._create_status_bar()
        
        # Initialize charts
        self._init_charts()
        
        # Initialize components
        self._init_components()
        
    def _load_config(self) -> Dict:
        """
        Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    return json.load(f)
            else:
                logger.warning(f"Configuration file not found: {self.config_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
            
    def _save_config(self) -> None:
        """
        Save configuration to file.
        """
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            
    def _set_theme(self) -> None:
        """
        Set GUI theme.
        """
        style = ttk.Style()
        
        # Use system theme as base
        if sys.platform == "win32":
            style.theme_use("vista")
        elif sys.platform == "darwin":
            style.theme_use("aqua")
        else:
            style.theme_use("clam")
            
        # Configure colors
        style.configure("TFrame", background="#f5f5f5")
        style.configure("TLabel", background="#f5f5f5")
        style.configure("TButton", background="#e0e0e0")
        style.configure("TNotebook", background="#f5f5f5")
        style.configure("TNotebook.Tab", background="#e0e0e0", padding=[10, 2])
        
        # Configure special styles
        style.configure("Green.TButton", foreground="#4CAF50")
        style.configure("Red.TButton", foreground="#F44336")
        style.configure("Bold.TLabel", font=("TkDefaultFont", 10, "bold"))
        style.configure("Header.TLabel", font=("TkDefaultFont", 12, "bold"))
        
    def _create_menu(self) -> None:
        """
        Create menu bar.
        """
        menu_bar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Load Configuration", command=self._load_config_dialog)
        file_menu.add_command(label="Save Configuration", command=self._save_config_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        # Trading menu
        trading_menu = tk.Menu(menu_bar, tearoff=0)
        trading_menu.add_command(label="Start Trading", command=self.start)
        trading_menu.add_command(label="Stop Trading", command=self.stop)
        trading_menu.add_separator()
        trading_menu.add_command(label="View Performance", command=self._show_performance)
        menu_bar.add_cascade(label="Trading", menu=trading_menu)
        
        # Tools menu
        tools_menu = tk.Menu(menu_bar, tearoff=0)
        tools_menu.add_command(label="Clear Error Log", command=self._clear_error_log)
        tools_menu.add_command(label="View Error Statistics", command=self._show_error_statistics)
        tools_menu.add_separator()
        tools_menu.add_command(label="Test API Connection", command=self._test_api_connection)
        menu_bar.add_cascade(label="Tools", menu=tools_menu)
        
        # Help menu
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Documentation", command=self._show_documentation)
        help_menu.add_command(label="About", command=self._show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menu_bar)
        
    def _create_layout(self) -> None:
        """
        Create main layout.
        """
        # Main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure grid
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=3)
        self.main_frame.rowconfigure(0, weight=1)
        
        # Left panel (controls and info)
        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Right panel (charts and data)
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Configure left panel grid
        self.left_panel.columnconfigure(0, weight=1)
        self.left_panel.rowconfigure(0, weight=0)  # Control panel
        self.left_panel.rowconfigure(1, weight=1)  # Info panel
        self.left_panel.rowconfigure(2, weight=1)  # Error panel
        
        # Configure right panel grid
        self.right_panel.columnconfigure(0, weight=1)
        self.right_panel.rowconfigure(0, weight=3)  # Price chart
        self.right_panel.rowconfigure(1, weight=2)  # Indicator charts
        self.right_panel.rowconfigure(2, weight=1)  # Order book
        
        # Create control panel
        self._create_control_panel()
        
        # Create info panel
        self._create_info_panel()
        
        # Create error panel
        self._create_error_panel()
        
        # Create chart panel
        self._create_chart_panel()
        
        # Create indicator panel
        self._create_indicator_panel()
        
        # Create order book panel
        self._create_order_book_panel()
        
    def _create_control_panel(self) -> None:
        """
        Create control panel.
        """
        # Control panel frame
        self.control_panel = ttk.LabelFrame(self.left_panel, text="Controls")
        self.control_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure grid
        self.control_panel.columnconfigure(0, weight=1)
        self.control_panel.columnconfigure(1, weight=1)
        
        # Symbol selection
        ttk.Label(self.control_panel, text="Symbol:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.symbol_var = tk.StringVar(value=self.current_symbol)
        self.symbol_combo = ttk.Combobox(self.control_panel, textvariable=self.symbol_var, values=self.symbols)
        self.symbol_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.symbol_combo.bind("<<ComboboxSelected>>", self._on_symbol_change)
        
        # Strategy selection
        ttk.Label(self.control_panel, text="Strategy:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.strategy_var = tk.StringVar(value="Master Omni Overlord")
        self.strategy_combo = ttk.Combobox(self.control_panel, textvariable=self.strategy_var, 
                                         values=["Master Omni Overlord"])
        self.strategy_combo.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        # Start/Stop buttons
        self.start_button = ttk.Button(self.control_panel, text="Start", command=self.start, style="Green.TButton")
        self.start_button.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        self.stop_button = ttk.Button(self.control_panel, text="Stop", command=self.stop, style="Red.TButton")
        self.stop_button.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        self.stop_button.config(state=tk.DISABLED)
        
    def _create_info_panel(self) -> None:
        """
        Create information panel.
        """
        # Info panel frame
        self.info_panel = ttk.LabelFrame(self.left_panel, text="Information")
        self.info_panel.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure grid
        self.info_panel.columnconfigure(0, weight=1)
        self.info_panel.columnconfigure(1, weight=1)
        
        # Market regime
        ttk.Label(self.info_panel, text="Market Regime:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.market_regime_var = tk.StringVar(value="Unknown")
        self.market_regime_label = ttk.Label(self.info_panel, textvariable=self.market_regime_var)
        self.market_regime_label.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Current price
        ttk.Label(self.info_panel, text="Current Price:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.price_var = tk.StringVar(value="0.00")
        ttk.Label(self.info_panel, textvariable=self.price_var).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # 24h change
        ttk.Label(self.info_panel, text="24h Change:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.change_var = tk.StringVar(value="0.00%")
        self.change_label = ttk.Label(self.info_panel, textvariable=self.change_var)
        self.change_label.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        # Volume
        ttk.Label(self.info_panel, text="24h Volume:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.volume_var = tk.StringVar(value="0.00")
        ttk.Label(self.info_panel, textvariable=self.volume_var).grid(row=3, column=1, sticky="w", padx=5, pady=5)
        
        # Signal
        ttk.Label(self.info_panel, text="Current Signal:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.signal_var = tk.StringVar(value="Neutral")
        self.signal_label = ttk.Label(self.info_panel, textvariable=self.signal_var)
        self.signal_label.grid(row=4, column=1, sticky="w", padx=5, pady=5)
        
        # Signal strength
        ttk.Label(self.info_panel, text="Signal Strength:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.signal_strength_var = tk.StringVar(value="0.00")
        ttk.Label(self.info_panel, textvariable=self.signal_strength_var).grid(row=5, column=1, sticky="w", padx=5, pady=5)
        
        # Signal confidence
        ttk.Label(self.info_panel, text="Signal Confidence:").grid(row=6, column=0, sticky="w", padx=5, pady=5)
        self.signal_confidence_var = tk.StringVar(value="0.00")
        ttk.Label(self.info_panel, textvariable=self.signal_confidence_var).grid(row=6, column=1, sticky="w", padx=5, pady=5)
        
        # Last update
        ttk.Label(self.info_panel, text="Last Update:").grid(row=7, column=0, sticky="w", padx=5, pady=5)
        self.last_update_var = tk.StringVar(value="Never")
        ttk.Label(self.info_panel, textvariable=self.last_update_var).grid(row=7, column=1, sticky="w", padx=5, pady=5)
        
        # Add a separator
        ttk.Separator(self.info_panel, orient=tk.HORIZONTAL).grid(row=8, column=0, columnspan=2, sticky="ew", padx=5, pady=10)
        
        # Signal reasoning
        ttk.Label(self.info_panel, text="Signal Reasoning:", style="Bold.TLabel").grid(row=9, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        self.reasoning_text = scrolledtext.ScrolledText(self.info_panel, wrap=tk.WORD, height=8)
        self.reasoning_text.grid(row=10, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        self.reasoning_text.config(state=tk.DISABLED)
        
    def _create_error_panel(self) -> None:
        """
        Create error panel.
        """
        # Error panel frame
        self.error_panel = ttk.LabelFrame(self.left_panel, text="Error Log")
        self.error_panel.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure grid
        self.error_panel.columnconfigure(0, weight=1)
        self.error_panel.rowconfigure(0, weight=1)
        
        # Error log
        self.error_log_text = scrolledtext.ScrolledText(self.error_panel, wrap=tk.WORD)
        self.error_log_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.error_log_text.config(state=tk.DISABLED)
        
        # Add tags for error severity colors
        self.error_log_text.tag_config(ErrorSeverity.INFO, foreground=ERROR_COLORS[ErrorSeverity.INFO])
        self.error_log_text.tag_config(ErrorSeverity.WARNING, foreground=ERROR_COLORS[ErrorSeverity.WARNING])
        self.error_log_text.tag_config(ErrorSeverity.ERROR, foreground=ERROR_COLORS[ErrorSeverity.ERROR])
        self.error_log_text.tag_config(ErrorSeverity.CRITICAL, foreground=ERROR_COLORS[ErrorSeverity.CRITICAL])
        self.error_log_text.tag_config(ErrorSeverity.FATAL, foreground=ERROR_COLORS[ErrorSeverity.FATAL])
        
    def _create_chart_panel(self) -> None:
        """
        Create chart panel.
        """
        # Chart panel frame
        self.chart_panel = ttk.LabelFrame(self.right_panel, text="Price Chart")
        self.chart_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure grid
        self.chart_panel.columnconfigure(0, weight=1)
        self.chart_panel.rowconfigure(0, weight=1)
        
        # Create figure and canvas
        self.price_figure = Figure(figsize=(8, 4), dpi=100)
        self.price_canvas = FigureCanvasTkAgg(self.price_figure, master=self.chart_panel)
        self.price_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Add toolbar
        self.price_toolbar = NavigationToolbar2Tk(self.price_canvas, self.chart_panel)
        self.price_toolbar.update()
        self.price_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
    def _create_indicator_panel(self) -> None:
        """
        Create indicator panel.
        """
        # Indicator panel frame
        self.indicator_panel = ttk.LabelFrame(self.right_panel, text="Technical Indicators")
        self.indicator_panel.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure grid
        self.indicator_panel.columnconfigure(0, weight=1)
        self.indicator_panel.rowconfigure(0, weight=1)
        
        # Create notebook for indicators
        self.indicator_notebook = ttk.Notebook(self.indicator_panel)
        self.indicator_notebook.grid(row=0, column=0, sticky="nsew")
        
        # Create tabs for different indicators
        self.momentum_frame = ttk.Frame(self.indicator_notebook)
        self.trend_frame = ttk.Frame(self.indicator_notebook)
        self.volatility_frame = ttk.Frame(self.indicator_notebook)
        self.volume_frame = ttk.Frame(self.indicator_notebook)
        
        self.indicator_notebook.add(self.momentum_frame, text="Momentum")
        self.indicator_notebook.add(self.trend_frame, text="Trend")
        self.indicator_notebook.add(self.volatility_frame, text="Volatility")
        self.indicator_notebook.add(self.volume_frame, text="Volume")
        
        # Configure frames
        for frame in [self.momentum_frame, self.trend_frame, self.volatility_frame, self.volume_frame]:
            frame.columnconfigure(0, weight=1)
            frame.rowconfigure(0, weight=1)
            
        # Create figures and canvases
        self.momentum_figure = Figure(figsize=(8, 3), dpi=100)
        self.momentum_canvas = FigureCanvasTkAgg(self.momentum_figure, master=self.momentum_frame)
        self.momentum_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        self.trend_figure = Figure(figsize=(8, 3), dpi=100)
        self.trend_canvas = FigureCanvasTkAgg(self.trend_figure, master=self.trend_frame)
        self.trend_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        self.volatility_figure = Figure(figsize=(8, 3), dpi=100)
        self.volatility_canvas = FigureCanvasTkAgg(self.volatility_figure, master=self.volatility_frame)
        self.volatility_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        self.volume_figure = Figure(figsize=(8, 3), dpi=100)
        self.volume_canvas = FigureCanvasTkAgg(self.volume_figure, master=self.volume_frame)
        self.volume_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
    def _create_order_book_panel(self) -> None:
        """
        Create order book panel.
        """
        # Order book panel frame
        self.order_book_panel = ttk.LabelFrame(self.right_panel, text="Order Book")
        self.order_book_panel.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure grid
        self.order_book_panel.columnconfigure(0, weight=1)
        self.order_book_panel.columnconfigure(1, weight=1)
        self.order_book_panel.rowconfigure(0, weight=1)
        
        # Create frames for bids and asks
        self.bids_frame = ttk.LabelFrame(self.order_book_panel, text="Bids")
        self.bids_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.asks_frame = ttk.LabelFrame(self.order_book_panel, text="Asks")
        self.asks_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Configure frames
        for frame in [self.bids_frame, self.asks_frame]:
            frame.columnconfigure(0, weight=1)
            frame.columnconfigure(1, weight=1)
            frame.rowconfigure(0, weight=0)  # Header
            frame.rowconfigure(1, weight=1)  # Content
            
        # Create headers
        ttk.Label(self.bids_frame, text="Price", style="Bold.TLabel").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Label(self.bids_frame, text="Size", style="Bold.TLabel").grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(self.asks_frame, text="Price", style="Bold.TLabel").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Label(self.asks_frame, text="Size", style="Bold.TLabel").grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Create treeviews for bids and asks
        self.bids_tree = ttk.Treeview(self.bids_frame, columns=("price", "size"), show="headings", height=10)
        self.bids_tree.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        
        self.asks_tree = ttk.Treeview(self.asks_frame, columns=("price", "size"), show="headings", height=10)
        self.asks_tree.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        
        # Configure treeviews
        for tree in [self.bids_tree, self.asks_tree]:
            tree.heading("price", text="Price")
            tree.heading("size", text="Size")
            tree.column("price", width=100)
            tree.column("size", width=100)
            
            # Add scrollbars
            scrollbar = ttk.Scrollbar(tree, orient=tk.VERTICAL, command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
    def _create_status_bar(self) -> None:
        """
        Create status bar.
        """
        # Status bar frame
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
        
        # Configure grid
        self.status_bar.columnconfigure(0, weight=1)
        self.status_bar.columnconfigure(1, weight=0)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.status_bar, textvariable=self.status_var)
        self.status_label.grid(row=0, column=0, sticky="w")
        
        # Connection status
        self.connection_var = tk.StringVar(value="Disconnected")
        self.connection_label = ttk.Label(self.status_bar, textvariable=self.connection_var)
        self.connection_label.grid(row=0, column=1, sticky="e")
        
    def _init_charts(self) -> None:
        """
        Initialize charts.
        """
        # Price chart
        self.price_ax = self.price_figure.add_subplot(111)
        self.price_ax.set_title(f"{self.current_symbol} Price")
        self.price_ax.set_xlabel("Time")
        self.price_ax.set_ylabel("Price")
        self.price_ax.grid(True)
        
        # Momentum indicators
        self.momentum_ax = self.momentum_figure.add_subplot(111)
        self.momentum_ax.set_title("Momentum Indicators")
        self.momentum_ax.set_xlabel("Time")
        self.momentum_ax.set_ylabel("Value")
        self.momentum_ax.grid(True)
        
        # Trend indicators
        self.trend_ax = self.trend_figure.add_subplot(111)
        self.trend_ax.set_title("Trend Indicators")
        self.trend_ax.set_xlabel("Time")
        self.trend_ax.set_ylabel("Value")
        self.trend_ax.grid(True)
        
        # Volatility indicators
        self.volatility_ax = self.volatility_figure.add_subplot(111)
        self.volatility_ax.set_title("Volatility Indicators")
        self.volatility_ax.set_xlabel("Time")
        self.volatility_ax.set_ylabel("Value")
        self.volatility_ax.grid(True)
        
        # Volume indicators
        self.volume_ax = self.volume_figure.add_subplot(111)
        self.volume_ax.set_title("Volume Indicators")
        self.volume_ax.set_xlabel("Time")
        self.volume_ax.set_ylabel("Value")
        self.volume_ax.grid(True)
        
        # Draw empty charts
        self.price_canvas.draw()
        self.momentum_canvas.draw()
        self.trend_canvas.draw()
        self.volatility_canvas.draw()
        self.volume_canvas.draw()
        
    def _init_components(self) -> None:
        """
        Initialize trading components.
        """
        try:
            # Initialize exchange adapter
            self.exchange_adapter = HyperliquidAdapter(self.config)
            
            # Initialize order book handler
            self.order_book_handler = OrderBookHandler()
            
            # Initialize data accumulator
            self.data_accumulator = EnhancedHistoricalDataAccumulator(
                symbols=self.symbols,
                timeframes=["1m", "5m", "15m", "1h", "4h", "1d"],
                max_data_points=1000
            )
            
            # Initialize signal generator
            self.signal_generator = RobustSignalGenerator()
            
            # Initialize strategy
            self.strategy = MasterOmniOverlordRobustStrategy(
                config=self.config,
                logger=logger,
                error_handler=self.error_handler
            )
            
            # Update connection status
            self.connection_var.set("Connected")
            self.status_var.set("Ready")
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            self.connection_var.set("Error")
            self.status_var.set(f"Error: {str(e)}")
            self._log_error(TradingError(
                message=f"Error initializing components: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM
            ))
            
    def start(self) -> None:
        """
        Start trading.
        """
        if self.running:
            return
            
        try:
            # Update status
            self.status_var.set("Starting...")
            self.running = True
            
            # Update buttons
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            # Start data thread
            self.data_thread = threading.Thread(target=self._data_thread_func)
            self.data_thread.daemon = True
            self.data_thread.start()
            
            # Start signal thread
            self.signal_thread = threading.Thread(target=self._signal_thread_func)
            self.signal_thread.daemon = True
            self.signal_thread.start()
            
            # Schedule GUI updates
            self.root.after(UPDATE_INTERVAL, self._update_gui)
            self.root.after(CHART_UPDATE_INTERVAL, self._update_charts)
            
            # Update status
            self.status_var.set("Running")
            logger.info("Trading started")
        except Exception as e:
            logger.error(f"Error starting trading: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            self.running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self._log_error(TradingError(
                message=f"Error starting trading: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM
            ))
            
    def stop(self) -> None:
        """
        Stop trading.
        """
        if not self.running:
            return
            
        try:
            # Update status
            self.status_var.set("Stopping...")
            self.running = False
            
            # Update buttons
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            
            # Wait for threads to finish
            if self.data_thread and self.data_thread.is_alive():
                self.data_thread.join(timeout=1.0)
                
            if self.signal_thread and self.signal_thread.is_alive():
                self.signal_thread.join(timeout=1.0)
                
            # Update status
            self.status_var.set("Stopped")
            logger.info("Trading stopped")
        except Exception as e:
            logger.error(f"Error stopping trading: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            self._log_error(TradingError(
                message=f"Error stopping trading: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM
            ))
            
    def _data_thread_func(self) -> None:
        """
        Data thread function.
        """
        logger.info("Data thread started")
        
        while self.running:
            try:
                # Fetch market data for all symbols
                for symbol in self.symbols:
                    # Fetch market data
                    market_data = self.api_rate_limiter.execute_with_rate_limit(
                        endpoint="market_data",
                        params={"symbol": symbol}
                    )
                    
                    # Fetch order book
                    order_book = self.api_rate_limiter.execute_with_rate_limit(
                        endpoint="order_book",
                        params={"symbol": symbol}
                    )
                    
                    # Process order book
                    processed_order_book = self.order_book_handler.process_order_book(order_book)
                    
                    # Update data
                    with self.lock:
                        self.market_data[symbol] = market_data
                        self.order_book[symbol] = processed_order_book
                        
                        # Add data to accumulator
                        self.data_accumulator.add_data_point(
                            symbol=symbol,
                            timestamp=datetime.now(),
                            data=market_data
                        )
                        
                    # Queue update
                    self.update_queue.put(("market_data", symbol))
                    
                # Sleep to avoid excessive API calls
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"Error in data thread: {str(e)}")
                self._log_error(TradingError(
                    message=f"Error in data thread: {str(e)}",
                    severity=ErrorSeverity.ERROR,
                    category=ErrorCategory.DATA
                ))
                time.sleep(5.0)  # Sleep longer on error
                
        logger.info("Data thread stopped")
        
    def _signal_thread_func(self) -> None:
        """
        Signal thread function.
        """
        logger.info("Signal thread started")
        
        while self.running:
            try:
                # Generate signals for all symbols
                for symbol in self.symbols:
                    # Get historical data
                    historical_data = self.data_accumulator.get_data(
                        symbol=symbol,
                        timeframe="5m",
                        limit=100
                    )
                    
                    if historical_data is None or len(historical_data) < 10:
                        logger.warning(f"Insufficient historical data for {symbol}")
                        continue
                        
                    # Get order book
                    order_book = self.order_book.get(symbol, None)
                    
                    # Generate signal
                    signal = self.signal_generator.generate_master_signal(
                        df=historical_data,
                        order_book=order_book
                    )
                    
                    # Update signal
                    with self.lock:
                        self.signals[symbol] = signal
                        
                    # Queue update
                    self.update_queue.put(("signal", symbol))
                    
                # Sleep to avoid excessive processing
                time.sleep(5.0)
            except Exception as e:
                logger.error(f"Error in signal thread: {str(e)}")
                self._log_error(TradingError(
                    message=f"Error in signal thread: {str(e)}",
                    severity=ErrorSeverity.ERROR,
                    category=ErrorCategory.SIGNAL
                ))
                time.sleep(10.0)  # Sleep longer on error
                
        logger.info("Signal thread stopped")
        
    def _update_gui(self) -> None:
        """
        Update GUI with latest data.
        """
        if not self.running:
            return
            
        try:
            # Process updates from queue
            while not self.update_queue.empty():
                update_type, symbol = self.update_queue.get_nowait()
                
                if update_type == "market_data" and symbol == self.current_symbol:
                    self._update_market_data()
                elif update_type == "signal" and symbol == self.current_symbol:
                    self._update_signal_data()
                    
            # Update last update time
            self.last_update_var.set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            # Schedule next update
            self.root.after(UPDATE_INTERVAL, self._update_gui)
        except Exception as e:
            logger.error(f"Error updating GUI: {str(e)}")
            self._log_error(TradingError(
                message=f"Error updating GUI: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM
            ))
            
    def _update_market_data(self) -> None:
        """
        Update market data display.
        """
        with self.lock:
            market_data = self.market_data.get(self.current_symbol, {})
            
            if not market_data:
                return
                
            # Update price
            price = market_data.get("price", 0.0)
            self.price_var.set(f"{price:.2f}")
            
            # Update change
            change = market_data.get("change_24h", 0.0)
            self.change_var.set(f"{change:.2f}%")
            
            # Update change label color
            if change > 0:
                self.change_label.config(foreground="#4CAF50")  # Green
            elif change < 0:
                self.change_label.config(foreground="#F44336")  # Red
            else:
                self.change_label.config(foreground="#000000")  # Black
                
            # Update volume
            volume = market_data.get("volume_24h", 0.0)
            self.volume_var.set(f"{volume:.2f}")
            
            # Update order book
            self._update_order_book()
            
    def _update_signal_data(self) -> None:
        """
        Update signal data display.
        """
        with self.lock:
            signal_data = self.signals.get(self.current_symbol, {})
            
            if not signal_data:
                return
                
            # Update signal
            signal = signal_data.get("signal", "neutral")
            self.signal_var.set(signal.capitalize())
            
            # Update signal label color
            if signal == "buy":
                self.signal_label.config(foreground="#4CAF50")  # Green
            elif signal == "sell":
                self.signal_label.config(foreground="#F44336")  # Red
            else:
                self.signal_label.config(foreground="#000000")  # Black
                
            # Update signal strength
            strength = signal_data.get("strength", 0.0)
            self.signal_strength_var.set(f"{strength:.2f}")
            
            # Update signal confidence
            confidence = signal_data.get("confidence", 0.0)
            self.signal_confidence_var.set(f"{confidence:.2f}")
            
            # Update market regime
            market_regime = signal_data.get("market_regime", "unknown")
            self.market_regime_var.set(market_regime.replace("_", " ").capitalize())
            
            # Update market regime label color
            self.market_regime_label.config(
                foreground=MARKET_REGIME_COLORS.get(market_regime, "#000000")
            )
            
            # Update reasoning
            reasoning = signal_data.get("reasoning", "")
            self.reasoning_text.config(state=tk.NORMAL)
            self.reasoning_text.delete(1.0, tk.END)
            self.reasoning_text.insert(tk.END, reasoning)
            self.reasoning_text.config(state=tk.DISABLED)
            
    def _update_order_book(self) -> None:
        """
        Update order book display.
        """
        with self.lock:
            order_book = self.order_book.get(self.current_symbol, {})
            
            if not order_book:
                return
                
            # Clear treeviews
            self.bids_tree.delete(*self.bids_tree.get_children())
            self.asks_tree.delete(*self.asks_tree.get_children())
            
            # Update bids
            bids = order_book.get("bids", [])
            for i, (price, size) in enumerate(bids[:10]):  # Show top 10 bids
                self.bids_tree.insert("", tk.END, values=(f"{price:.2f}", f"{size:.4f}"))
                
            # Update asks
            asks = order_book.get("asks", [])
            for i, (price, size) in enumerate(asks[:10]):  # Show top 10 asks
                self.asks_tree.insert("", tk.END, values=(f"{price:.2f}", f"{size:.4f}"))
                
    def _update_charts(self) -> None:
        """
        Update charts with latest data.
        """
        if not self.running:
            return
            
        try:
            # Get historical data
            with self.lock:
                historical_data = self.data_accumulator.get_data(
                    symbol=self.current_symbol,
                    timeframe="5m",
                    limit=MAX_CHART_POINTS
                )
                
            if historical_data is None or len(historical_data) < 2:
                logger.warning(f"Insufficient historical data for charts")
                # Schedule next update
                self.root.after(CHART_UPDATE_INTERVAL, self._update_charts)
                return
                
            # Update price chart
            self._update_price_chart(historical_data)
            
            # Update indicator charts
            self._update_indicator_charts(historical_data)
            
            # Schedule next update
            self.root.after(CHART_UPDATE_INTERVAL, self._update_charts)
        except Exception as e:
            logger.error(f"Error updating charts: {str(e)}")
            self._log_error(TradingError(
                message=f"Error updating charts: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM
            ))
            
    def _update_price_chart(self, df: pd.DataFrame) -> None:
        """
        Update price chart.
        
        Args:
            df: Historical data DataFrame
        """
        # Clear previous plot
        self.price_ax.clear()
        
        # Set title and labels
        self.price_ax.set_title(f"{self.current_symbol} Price")
        self.price_ax.set_xlabel("Time")
        self.price_ax.set_ylabel("Price")
        
        # Plot price
        timestamps = pd.to_datetime(df.index)
        self.price_ax.plot(timestamps, df["close"], label="Close", color="#1976D2")
        
        # Plot moving averages if available
        if "ema_50" in df.columns:
            self.price_ax.plot(timestamps, df["ema_50"], label="EMA 50", color="#FF9800", alpha=0.7)
            
        if "ema_200" in df.columns:
            self.price_ax.plot(timestamps, df["ema_200"], label="EMA 200", color="#F44336", alpha=0.7)
            
        # Plot Bollinger Bands if available
        if all(col in df.columns for col in ["bb_upper", "bb_middle", "bb_lower"]):
            self.price_ax.plot(timestamps, df["bb_upper"], label="BB Upper", color="#4CAF50", linestyle="--", alpha=0.5)
            self.price_ax.plot(timestamps, df["bb_middle"], label="BB Middle", color="#2196F3", linestyle="--", alpha=0.5)
            self.price_ax.plot(timestamps, df["bb_lower"], label="BB Lower", color="#F44336", linestyle="--", alpha=0.5)
            
        # Format x-axis
        self.price_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        self.price_ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        
        # Add grid and legend
        self.price_ax.grid(True, alpha=0.3)
        self.price_ax.legend(loc="upper left")
        
        # Rotate x-axis labels
        plt.setp(self.price_ax.get_xticklabels(), rotation=45, ha="right")
        
        # Update canvas
        self.price_figure.tight_layout()
        self.price_canvas.draw()
        
    def _update_indicator_charts(self, df: pd.DataFrame) -> None:
        """
        Update indicator charts.
        
        Args:
            df: Historical data DataFrame
        """
        timestamps = pd.to_datetime(df.index)
        
        # Update momentum indicators
        self.momentum_ax.clear()
        self.momentum_ax.set_title("Momentum Indicators")
        self.momentum_ax.set_xlabel("Time")
        self.momentum_ax.set_ylabel("Value")
        
        # Plot RSI if available
        if "rsi_14" in df.columns:
            self.momentum_ax.plot(timestamps, df["rsi_14"], label="RSI (14)", color="#1976D2")
            self.momentum_ax.axhline(y=70, color="#F44336", linestyle="--", alpha=0.5)
            self.momentum_ax.axhline(y=30, color="#4CAF50", linestyle="--", alpha=0.5)
            self.momentum_ax.axhline(y=50, color="#9E9E9E", linestyle="--", alpha=0.3)
            
        # Plot Stochastic if available
        if "stoch_k" in df.columns and "stoch_d" in df.columns:
            self.momentum_ax.plot(timestamps, df["stoch_k"], label="Stoch %K", color="#FF9800")
            self.momentum_ax.plot(timestamps, df["stoch_d"], label="Stoch %D", color="#F44336")
            
        self.momentum_ax.set_ylim(0, 100)
        self.momentum_ax.grid(True, alpha=0.3)
        self.momentum_ax.legend(loc="upper left")
        
        # Update trend indicators
        self.trend_ax.clear()
        self.trend_ax.set_title("Trend Indicators")
        self.trend_ax.set_xlabel("Time")
        self.trend_ax.set_ylabel("Value")
        
        # Plot MACD if available
        if all(col in df.columns for col in ["macd_line", "macd_signal", "macd_histogram"]):
            self.trend_ax.plot(timestamps, df["macd_line"], label="MACD Line", color="#1976D2")
            self.trend_ax.plot(timestamps, df["macd_signal"], label="Signal Line", color="#F44336")
            self.trend_ax.bar(timestamps, df["macd_histogram"], label="Histogram", color="#4CAF50", alpha=0.5)
            
        # Plot ADX if available
        if "adx" in df.columns:
            self.trend_ax.plot(timestamps, df["adx"], label="ADX", color="#9C27B0")
            self.trend_ax.axhline(y=25, color="#9E9E9E", linestyle="--", alpha=0.3)
            
        self.trend_ax.grid(True, alpha=0.3)
        self.trend_ax.legend(loc="upper left")
        
        # Update volatility indicators
        self.volatility_ax.clear()
        self.volatility_ax.set_title("Volatility Indicators")
        self.volatility_ax.set_xlabel("Time")
        self.volatility_ax.set_ylabel("Value")
        
        # Plot ATR if available
        if "atr" in df.columns:
            self.volatility_ax.plot(timestamps, df["atr"], label="ATR", color="#1976D2")
            
        # Plot Bollinger Band Width if available
        if "bb_width" in df.columns:
            self.volatility_ax.plot(timestamps, df["bb_width"], label="BB Width", color="#F44336")
            
        self.volatility_ax.grid(True, alpha=0.3)
        self.volatility_ax.legend(loc="upper left")
        
        # Update volume indicators
        self.volume_ax.clear()
        self.volume_ax.set_title("Volume Indicators")
        self.volume_ax.set_xlabel("Time")
        self.volume_ax.set_ylabel("Value")
        
        # Plot volume if available
        if "volume" in df.columns:
            self.volume_ax.bar(timestamps, df["volume"], label="Volume", color="#1976D2", alpha=0.5)
            
        # Plot OBV if available
        if "obv" in df.columns:
            # Normalize OBV for better visualization
            obv = df["obv"]
            obv_norm = (obv - obv.min()) / (obv.max() - obv.min()) * df["volume"].max()
            self.volume_ax.plot(timestamps, obv_norm, label="OBV (Normalized)", color="#F44336")
            
        self.volume_ax.grid(True, alpha=0.3)
        self.volume_ax.legend(loc="upper left")
        
        # Update canvases
        for fig, canvas in [
            (self.momentum_figure, self.momentum_canvas),
            (self.trend_figure, self.trend_canvas),
            (self.volatility_figure, self.volatility_canvas),
            (self.volume_figure, self.volume_canvas)
        ]:
            fig.tight_layout()
            canvas.draw()
            
    def _on_symbol_change(self, event) -> None:
        """
        Handle symbol change event.
        
        Args:
            event: Event object
        """
        self.current_symbol = self.symbol_var.get()
        logger.info(f"Symbol changed to {self.current_symbol}")
        
        # Update displays
        self._update_market_data()
        self._update_signal_data()
        
        # Update charts
        self._update_charts()
        
    def _on_close(self) -> None:
        """
        Handle window close event.
        """
        if self.running:
            if messagebox.askyesno("Confirm Exit", "Trading is still running. Are you sure you want to exit?"):
                self.stop()
                self.root.destroy()
        else:
            self.root.destroy()
            
    def _load_config_dialog(self) -> None:
        """
        Show load configuration dialog.
        """
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, "r") as f:
                    self.config = json.load(f)
                    
                self.config_path = filename
                self._save_config()
                
                messagebox.showinfo("Success", "Configuration loaded successfully")
                logger.info(f"Configuration loaded from {filename}")
                
                # Reinitialize components
                self._init_components()
            except Exception as e:
                messagebox.showerror("Error", f"Error loading configuration: {str(e)}")
                logger.error(f"Error loading configuration: {str(e)}")
                
    def _save_config_dialog(self) -> None:
        """
        Show save configuration dialog.
        """
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if filename:
            try:
                self.config_path = filename
                self._save_config()
                
                messagebox.showinfo("Success", "Configuration saved successfully")
                logger.info(f"Configuration saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving configuration: {str(e)}")
                logger.error(f"Error saving configuration: {str(e)}")
                
    def _show_performance(self) -> None:
        """
        Show performance metrics.
        """
        # Create performance window
        performance_window = tk.Toplevel(self.root)
        performance_window.title("Performance Metrics")
        performance_window.geometry("600x400")
        performance_window.minsize(400, 300)
        
        # Create performance text
        performance_text = scrolledtext.ScrolledText(performance_window, wrap=tk.WORD)
        performance_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add performance metrics
        performance_text.insert(tk.END, "Performance Metrics\n")
        performance_text.insert(tk.END, "===================\n\n")
        
        # Add signal statistics
        performance_text.insert(tk.END, "Signal Statistics\n")
        performance_text.insert(tk.END, "----------------\n")
        
        for symbol in self.symbols:
            signal_data = self.signals.get(symbol, {})
            signal = signal_data.get("signal", "neutral")
            strength = signal_data.get("strength", 0.0)
            confidence = signal_data.get("confidence", 0.0)
            
            performance_text.insert(tk.END, f"{symbol}: {signal.capitalize()} (Strength: {strength:.2f}, Confidence: {confidence:.2f})\n")
            
        performance_text.insert(tk.END, "\n")
        
        # Add API statistics
        performance_text.insert(tk.END, "API Statistics\n")
        performance_text.insert(tk.END, "-------------\n")
        
        api_stats = self.api_rate_limiter.get_statistics()
        performance_text.insert(tk.END, f"Total Requests: {api_stats.get('total_requests', 0)}\n")
        performance_text.insert(tk.END, f"Rate Limited Requests: {api_stats.get('rate_limited', 0)}\n")
        performance_text.insert(tk.END, f"Cache Hits: {api_stats.get('cache_hits', 0)}\n")
        performance_text.insert(tk.END, f"Cache Efficiency: {api_stats.get('cache_efficiency', 0.0):.2f}%\n")
        
        performance_text.insert(tk.END, "\n")
        
        # Add error statistics
        performance_text.insert(tk.END, "Error Statistics\n")
        performance_text.insert(tk.END, "--------------\n")
        
        error_stats = self.error_handler.get_statistics()
        error_counts = error_stats.get("counts", {})
        
        for category, count in error_counts.items():
            performance_text.insert(tk.END, f"{category}: {count}\n")
            
        # Make text read-only
        performance_text.config(state=tk.DISABLED)
        
    def _clear_error_log(self) -> None:
        """
        Clear error log.
        """
        self.error_log = []
        self.error_log_text.config(state=tk.NORMAL)
        self.error_log_text.delete(1.0, tk.END)
        self.error_log_text.config(state=tk.DISABLED)
        
        # Clear error handler statistics
        self.error_handler.clear_statistics()
        
        logger.info("Error log cleared")
        
    def _show_error_statistics(self) -> None:
        """
        Show error statistics.
        """
        # Create statistics window
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Error Statistics")
        stats_window.geometry("600x400")
        stats_window.minsize(400, 300)
        
        # Create statistics text
        stats_text = scrolledtext.ScrolledText(stats_window, wrap=tk.WORD)
        stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add statistics
        stats_text.insert(tk.END, "Error Statistics\n")
        stats_text.insert(tk.END, "===============\n\n")
        
        # Add error counts
        stats_text.insert(tk.END, "Error Counts by Category\n")
        stats_text.insert(tk.END, "-----------------------\n")
        
        error_stats = self.error_handler.get_statistics()
        error_counts = error_stats.get("counts", {})
        
        for category, count in error_counts.items():
            stats_text.insert(tk.END, f"{category}: {count}\n")
            
        stats_text.insert(tk.END, "\n")
        
        # Add recent errors
        stats_text.insert(tk.END, "Recent Errors\n")
        stats_text.insert(tk.END, "------------\n")
        
        recent_errors = self.error_handler.get_recent_errors(count=10)
        
        for error in recent_errors:
            timestamp = error.get("timestamp", datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
            severity = error.get("severity", ErrorSeverity.INFO)
            category = error.get("category", ErrorCategory.UNKNOWN)
            message = error.get("message", "Unknown error")
            
            stats_text.insert(tk.END, f"{timestamp} [{severity}] {category}: {message}\n")
            
        # Make text read-only
        stats_text.config(state=tk.DISABLED)
        
    def _test_api_connection(self) -> None:
        """
        Test API connection.
        """
        try:
            self.status_var.set("Testing API connection...")
            
            # Test connection
            result = self.exchange_adapter.test_connection()
            
            if result:
                messagebox.showinfo("Success", "API connection successful")
                self.connection_var.set("Connected")
                self.status_var.set("Ready")
            else:
                messagebox.showerror("Error", "API connection failed")
                self.connection_var.set("Disconnected")
                self.status_var.set("API connection failed")
        except Exception as e:
            messagebox.showerror("Error", f"API connection error: {str(e)}")
            self.connection_var.set("Error")
            self.status_var.set(f"Error: {str(e)}")
            self._log_error(TradingError(
                message=f"API connection error: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.API
            ))
            
    def _show_documentation(self) -> None:
        """
        Show documentation.
        """
        # Create documentation window
        doc_window = tk.Toplevel(self.root)
        doc_window.title("Documentation")
        doc_window.geometry("800x600")
        doc_window.minsize(600, 400)
        
        # Create documentation text
        doc_text = scrolledtext.ScrolledText(doc_window, wrap=tk.WORD)
        doc_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add documentation
        doc_text.insert(tk.END, "Hyperliquid Trading Bot Documentation\n")
        doc_text.insert(tk.END, "===================================\n\n")
        
        doc_text.insert(tk.END, "Overview\n")
        doc_text.insert(tk.END, "--------\n")
        doc_text.insert(tk.END, "This trading bot uses the Master Omni Overlord strategy to generate trading signals for Hyperliquid markets. "
                              "The strategy combines multiple technical indicators, order book analysis, and market regime detection "
                              "to generate high-confidence trading signals.\n\n")
        
        doc_text.insert(tk.END, "Getting Started\n")
        doc_text.insert(tk.END, "--------------\n")
        doc_text.insert(tk.END, "1. Configure your API keys in the configuration file\n")
        doc_text.insert(tk.END, "2. Select a symbol from the dropdown menu\n")
        doc_text.insert(tk.END, "3. Click the Start button to begin trading\n")
        doc_text.insert(tk.END, "4. Monitor signals and performance in the information panel\n")
        doc_text.insert(tk.END, "5. Click the Stop button to stop trading\n\n")
        
        doc_text.insert(tk.END, "Features\n")
        doc_text.insert(tk.END, "--------\n")
        doc_text.insert(tk.END, "- Real-time market data and order book visualization\n")
        doc_text.insert(tk.END, "- Advanced technical indicators with adaptive parameters\n")
        doc_text.insert(tk.END, "- Market regime detection for strategy adaptation\n")
        doc_text.insert(tk.END, "- Comprehensive error handling and logging\n")
        doc_text.insert(tk.END, "- Performance metrics and statistics\n\n")
        
        doc_text.insert(tk.END, "Technical Indicators\n")
        doc_text.insert(tk.END, "-------------------\n")
        doc_text.insert(tk.END, "The bot uses the following technical indicators:\n\n")
        doc_text.insert(tk.END, "- Trend Indicators: MACD, ADX, Parabolic SAR, SuperTrend\n")
        doc_text.insert(tk.END, "- Momentum Indicators: RSI, Stochastic, CCI\n")
        doc_text.insert(tk.END, "- Volatility Indicators: Bollinger Bands, ATR, Keltner Channels\n")
        doc_text.insert(tk.END, "- Volume Indicators: OBV, VWAP, CMF\n\n")
        
        doc_text.insert(tk.END, "Market Regimes\n")
        doc_text.insert(tk.END, "--------------\n")
        doc_text.insert(tk.END, "The bot detects the following market regimes:\n\n")
        doc_text.insert(tk.END, "- Trending Up: Strong upward trend\n")
        doc_text.insert(tk.END, "- Trending Down: Strong downward trend\n")
        doc_text.insert(tk.END, "- Ranging: Price moving within a range\n")
        doc_text.insert(tk.END, "- Volatile: High volatility with unpredictable movements\n\n")
        
        doc_text.insert(tk.END, "Error Handling\n")
        doc_text.insert(tk.END, "--------------\n")
        doc_text.insert(tk.END, "The bot uses a comprehensive error handling system with the following features:\n\n")
        doc_text.insert(tk.END, "- Specialized error handlers for different error types\n")
        doc_text.insert(tk.END, "- Automatic recovery mechanisms with exponential backoff\n")
        doc_text.insert(tk.END, "- Detailed error logging with severity levels\n")
        doc_text.insert(tk.END, "- Error statistics and analysis\n")
        doc_text.insert(tk.END, "- Graceful degradation with fallback mechanisms\n\n")
        
        # Make text read-only
        doc_text.config(state=tk.DISABLED)
        
    def _show_about(self) -> None:
        """
        Show about dialog.
        """
        messagebox.showinfo(
            "About",
            "Hyperliquid Trading Bot\n\n"
            "Version: 1.0.0\n"
            "Author: Manus AI\n\n"
            "A high-performance trading bot for Hyperliquid markets."
        )
        
    def _log_error(self, error: TradingError) -> None:
        """
        Log an error to the error log.
        
        Args:
            error: Error to log
        """
        # Add error to log
        self.error_log.append(error)
        
        # Trim log if needed
        if len(self.error_log) > self.max_error_log:
            self.error_log = self.error_log[-self.max_error_log:]
            
        # Update error log text
        self.error_log_text.config(state=tk.NORMAL)
        
        # Add timestamp and error message
        timestamp = error.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        self.error_log_text.insert(tk.END, f"{timestamp} [{error.severity}] {error.category}: {error.message}\n", error.severity)
        
        # Scroll to end
        self.error_log_text.see(tk.END)
        self.error_log_text.config(state=tk.DISABLED)
        
    def _handle_error_notification(self, error: TradingError) -> None:
        """
        Handle error notification.
        
        Args:
            error: Error to handle
        """
        # Log error
        self._log_error(error)
        
        # Show notification for critical and fatal errors
        if error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
            self.update_queue.put(("error_notification", error))
            
    def run(self) -> None:
        """
        Run the GUI.
        """
        self.root.mainloop()

if __name__ == "__main__":
    # Create GUI
    gui = EnhancedGUI()
    
    # Run GUI
    gui.run()
