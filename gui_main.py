#!/usr/bin/env python3
"""
Enhanced GUI for the HyperliquidMaster trading bot.
Provides a user-friendly interface for trading on the Hyperliquid exchange.
Includes comprehensive settings, advanced trading modes, and robust error handling.
"""

import os
import sys
import json
import time
import logging
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, scrolledtext
from typing import Dict, List, Any, Optional, Tuple
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import custom modules
from gui_style import GUIStyleManager
from core.trading_integration import TradingIntegration
from core.error_handler import ErrorHandler
from core.trading_mode import TradingModeManager, TradingMode
from core.enhanced_connection_manager import ConnectionManager
from core.settings_manager import SettingsManager
from core.enhanced_risk_manager import RiskManager
from core.advanced_order_manager import AdvancedOrderManager
from core.position_manager_wrapper import PositionManagerWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("hyperliquid_bot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

class HyperliquidMasterGUI:
    """
    Main GUI class for the HyperliquidMaster trading bot.
    """
    
    def __init__(self, root: tk.Tk):
        """
        Initialize the GUI.
        
        Args:
            root: The root Tk instance
        """
        self.root = root
        self.root.title("Enhanced Hyperliquid Trading Bot v2.1.0")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Initialize logger
        self.logger = logging.getLogger("HyperliquidMasterGUI")
        
        # Initialize config and settings
        self.config_path = "config.json"
        self.settings_manager = SettingsManager(self.config_path, self.logger)
        self.config = self.settings_manager.load_settings()
        
        # Initialize style manager
        self.style_manager = GUIStyleManager(self.root, self.logger)
        
        # Initialize error handler
        self.error_handler = ErrorHandler(self.logger)
        
        # Initialize connection manager
        self.connection_manager = ConnectionManager(self.logger)
        
        # Initialize trading mode manager
        self.mode_manager = TradingModeManager(self.config, self.logger)
        
        # Initialize risk manager
        self.risk_manager = RiskManager(self.config, self.logger)
        
        # Initialize order manager
        self.order_manager = AdvancedOrderManager(self.config, self.logger)
        
        # Initialize position manager
        self.position_manager = PositionManagerWrapper(self.config, self.logger)
        
        # Initialize trading integration
        self.trading = TradingIntegration(
            self.config_path, 
            self.logger,
            self.connection_manager,
            self.mode_manager,
            self.risk_manager,
            self.order_manager,
            self.position_manager
        )
        
        # Initialize variables
        self.is_bot_running = False
        self.selected_symbol = tk.StringVar(value=self.config.get("symbol", "XRP"))
        self.account_address = tk.StringVar(value=self.config.get("account_address", ""))
        self.secret_key = tk.StringVar(value=self.config.get("secret_key", ""))
        self.show_secret_key = tk.BooleanVar(value=False)
        self.position_size = tk.StringVar(value=self.config.get("position_size", "0.01"))
        self.stop_loss = tk.StringVar(value=self.config.get("stop_loss", "1.0"))
        self.take_profit = tk.StringVar(value=self.config.get("take_profit", "2.0"))
        self.risk_level = tk.DoubleVar(value=self.config.get("risk_level", 0.02))
        self.use_volatility_filters = tk.BooleanVar(value=self.config.get("use_volatility_filters", True))
        self.use_trend_filters = tk.BooleanVar(value=self.config.get("use_trend_filters", True))
        self.use_volume_filters = tk.BooleanVar(value=self.config.get("use_volume_filters", True))
        self.tp_multiplier = tk.DoubleVar(value=self.config.get("tp_multiplier", 2.0))
        self.sl_multiplier = tk.DoubleVar(value=self.config.get("sl_multiplier", 1.0))
        self.use_mock_data = tk.BooleanVar(value=self.config.get("use_mock_data", False))
        self.available_symbols = ["BTC", "ETH", "XRP", "SOL", "DOGE", "AVAX", "LINK", "MATIC"]
        self.timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        self.selected_timeframe = tk.StringVar(value=self.config.get("timeframe", "1h"))
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.dashboard_tab = ttk.Frame(self.notebook)
        self.positions_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)
        self.strategy_tab = ttk.Frame(self.notebook)
        self.advanced_tab = ttk.Frame(self.notebook)
        self.logs_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.dashboard_tab, text="Dashboard")
        self.notebook.add(self.positions_tab, text="Positions")
        self.notebook.add(self.settings_tab, text="Settings")
        self.notebook.add(self.strategy_tab, text="Strategy")
        self.notebook.add(self.advanced_tab, text="Advanced")
        self.notebook.add(self.logs_tab, text="Logs")
        
        # Create theme toggle button
        self.theme_button = tk.Button(self.root, text="Toggle Theme", command=self._toggle_theme)
        self.theme_button.place(relx=0.95, rely=0.02, anchor="ne")
        self.style_manager.style_button(self.theme_button)
        
        # Initialize tabs
        self._init_dashboard_tab()
        self._init_positions_tab()
        self._init_settings_tab()
        self._init_strategy_tab()
        self._init_advanced_tab()
        self._init_logs_tab()
        
        # Start update loops
        self._start_update_loops()
        
        # Log startup
        self.logger.info("Enhanced Hyperliquid Trading Bot v2.1.0 started")
    
    def _save_config(self) -> None:
        """Save configuration to file."""
        try:
            config = {
                "symbol": self.selected_symbol.get(),
                "account_address": self.account_address.get(),
                "secret_key": self.secret_key.get(),
                "position_size": self.position_size.get(),
                "stop_loss": self.stop_loss.get(),
                "take_profit": self.take_profit.get(),
                "risk_level": self.risk_level.get(),
                "use_volatility_filters": self.use_volatility_filters.get(),
                "use_trend_filters": self.use_trend_filters.get(),
                "use_volume_filters": self.use_volume_filters.get(),
                "tp_multiplier": self.tp_multiplier.get(),
                "sl_multiplier": self.sl_multiplier.get(),
                "use_mock_data": self.use_mock_data.get(),
                "timeframe": self.selected_timeframe.get(),
                "trading_mode": self.mode_manager.get_current_mode().name
            }
            
            self.settings_manager.save_settings(config)
            self.logger.info("Configuration saved successfully")
            messagebox.showinfo("Settings", "Settings saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            messagebox.showerror("Error", f"Error saving settings: {e}")
    
    def _toggle_theme(self) -> None:
        """Toggle between dark and light themes."""
        try:
            self.style_manager.toggle_theme()
        except Exception as e:
            self.logger.error(f"Error toggling theme: {e}")
    
    def _init_dashboard_tab(self) -> None:
        """Initialize the dashboard tab."""
        # Create frames
        top_frame = ttk.Frame(self.dashboard_tab)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        middle_frame = ttk.Frame(self.dashboard_tab)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        bottom_frame = ttk.Frame(self.dashboard_tab)
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create symbol selection
        symbol_label = ttk.Label(top_frame, text="Symbol:")
        symbol_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.symbol_combobox = ttk.Combobox(top_frame, textvariable=self.selected_symbol, values=self.available_symbols, width=10)
        self.symbol_combobox.pack(side=tk.LEFT, padx=(0, 10))
        self.symbol_combobox.bind("<<ComboboxSelected>>", self._on_symbol_selected)
        
        # Create timeframe selection
        timeframe_label = ttk.Label(top_frame, text="Timeframe:")
        timeframe_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.timeframe_combobox = ttk.Combobox(top_frame, textvariable=self.selected_timeframe, values=self.timeframes, width=10)
        self.timeframe_combobox.pack(side=tk.LEFT, padx=(0, 10))
        self.timeframe_combobox.bind("<<ComboboxSelected>>", self._on_timeframe_selected)
        
        # Create refresh button
        refresh_button = tk.Button(top_frame, text="Refresh", command=self._refresh_dashboard)
        refresh_button.pack(side=tk.LEFT, padx=(0, 10))
        self.style_manager.style_button(refresh_button)
        
        # Create trading mode indicator
        mode_label = ttk.Label(top_frame, text="Mode:")
        mode_label.pack(side=tk.LEFT, padx=(10, 5))
        
        self.mode_indicator = tk.Label(top_frame, text=self.mode_manager.get_current_mode().name, 
                                      bg=self._get_mode_color(), fg="white", padx=5, pady=2)
        self.mode_indicator.pack(side=tk.LEFT)
        
        # Create connection status indicator
        self.connection_status = tk.Label(top_frame, text="Not Connected", bg="red", fg="white", padx=5, pady=2)
        self.connection_status.pack(side=tk.RIGHT)
        
        # Create chart frame
        self.chart_frame = ttk.Frame(middle_frame)
        self.chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create chart
        self._create_chart()
        
        # Create trading controls
        controls_frame = ttk.Frame(bottom_frame)
        controls_frame.pack(fill=tk.X)
        
        # Create position size input
        size_label = ttk.Label(controls_frame, text="Size:")
        size_label.grid(row=0, column=0, padx=(0, 5), pady=5)
        
        size_entry = tk.Entry(controls_frame, textvariable=self.position_size, width=10)
        size_entry.grid(row=0, column=1, padx=(0, 10), pady=5)
        self.style_manager.style_entry(size_entry)
        
        # Create stop loss input
        sl_label = ttk.Label(controls_frame, text="Stop Loss %:")
        sl_label.grid(row=0, column=2, padx=(0, 5), pady=5)
        
        sl_entry = tk.Entry(controls_frame, textvariable=self.stop_loss, width=10)
        sl_entry.grid(row=0, column=3, padx=(0, 10), pady=5)
        self.style_manager.style_entry(sl_entry)
        
        # Create take profit input
        tp_label = ttk.Label(controls_frame, text="Take Profit %:")
        tp_label.grid(row=0, column=4, padx=(0, 5), pady=5)
        
        tp_entry = tk.Entry(controls_frame, textvariable=self.take_profit, width=10)
        tp_entry.grid(row=0, column=5, padx=(0, 10), pady=5)
        self.style_manager.style_entry(tp_entry)
        
        # Create buy long button
        buy_long_button = tk.Button(controls_frame, text="BUY LONG", command=lambda: self._place_order(True, True))
        buy_long_button.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        self.style_manager.style_button(buy_long_button, "success")
        
        # Create sell long button
        sell_long_button = tk.Button(controls_frame, text="SELL LONG", command=lambda: self._place_order(False, True))
        sell_long_button.grid(row=1, column=2, columnspan=2, padx=5, pady=5, sticky="ew")
        self.style_manager.style_button(sell_long_button, "error")
        
        # Create buy short button
        buy_short_button = tk.Button(controls_frame, text="BUY SHORT", command=lambda: self._place_order(True, False))
        buy_short_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        self.style_manager.style_button(buy_short_button, "success")
        
        # Create sell short button
        sell_short_button = tk.Button(controls_frame, text="SELL SHORT", command=lambda: self._place_order(False, False))
        sell_short_button.grid(row=2, column=2, columnspan=2, padx=5, pady=5, sticky="ew")
        self.style_manager.style_button(sell_short_button, "error")
        
        # Create bot controls
        bot_frame = ttk.Frame(bottom_frame)
        bot_frame.pack(fill=tk.X, pady=10)
        
        # Create start/stop bot button
        self.bot_button = tk.Button(bot_frame, text="Start Bot", command=self._toggle_bot)
        self.bot_button.pack(fill=tk.X)
        self.style_manager.style_button(self.bot_button, "success")
    
    def _init_positions_tab(self) -> None:
        """Initialize the positions tab."""
        # Create frames
        top_frame = ttk.Frame(self.positions_tab)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create refresh button
        refresh_button = tk.Button(top_frame, text="Refresh Positions", command=self._refresh_positions)
        refresh_button.pack(side=tk.LEFT)
        self.style_manager.style_button(refresh_button)
        
        # Create positions treeview
        columns = ("symbol", "side", "size", "entry_price", "current_price", "liquidation_price", "pnl", "pnl_percent")
        self.positions_tree = ttk.Treeview(self.positions_tab, columns=columns, show="headings")
        
        # Define headings
        self.positions_tree.heading("symbol", text="Symbol")
        self.positions_tree.heading("side", text="Side")
        self.positions_tree.heading("size", text="Size")
        self.positions_tree.heading("entry_price", text="Entry Price")
        self.positions_tree.heading("current_price", text="Current Price")
        self.positions_tree.heading("liquidation_price", text="Liquidation Price")
        self.positions_tree.heading("pnl", text="PnL")
        self.positions_tree.heading("pnl_percent", text="PnL %")
        
        # Define columns
        self.positions_tree.column("symbol", width=80)
        self.positions_tree.column("side", width=80)
        self.positions_tree.column("size", width=80)
        self.positions_tree.column("entry_price", width=100)
        self.positions_tree.column("current_price", width=100)
        self.positions_tree.column("liquidation_price", width=120)
        self.positions_tree.column("pnl", width=100)
        self.positions_tree.column("pnl_percent", width=100)
        
        # Add scrollbar
        positions_scrollbar = ttk.Scrollbar(self.positions_tab, orient="vertical", command=self.positions_tree.yview)
        self.positions_tree.configure(yscrollcommand=positions_scrollbar.set)
        
        # Pack widgets
        self.positions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        positions_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=(0, 10))
        
        # Create context menu
        self.positions_context_menu = tk.Menu(self.positions_tree, tearoff=0)
        self.positions_context_menu.add_command(label="Close Position", command=self._close_selected_position)
        self.positions_context_menu.add_command(label="Modify Stop Loss", command=self._modify_stop_loss)
        self.positions_context_menu.add_command(label="Modify Take Profit", command=self._modify_take_profit)
        self.positions_context_menu.add_command(label="Add Trailing Stop", command=self._add_trailing_stop)
        
        # Bind right-click to show context menu
        self.positions_tree.bind("<Button-3>", self._show_positions_context_menu)
    
    def _init_settings_tab(self) -> None:
        """Initialize the settings tab."""
        # Create scrollable frame
        container, settings_frame = self.style_manager.create_scrollable_frame(self.settings_tab)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create API key management section
        api_frame = ttk.Frame(settings_frame)
        api_frame.pack(fill=tk.X, pady=(0, 20))
        
        api_title = ttk.Label(api_frame, text="API Key Management", style="Header.TLabel")
        api_title.pack(anchor=tk.W, pady=(0, 10))
        
        # Create account address input
        addr_frame = ttk.Frame(api_frame)
        addr_frame.pack(fill=tk.X, pady=5)
        
        addr_label = ttk.Label(addr_frame, text="Account Address:")
        addr_label.pack(side=tk.LEFT, padx=(0, 5))
        
        addr_entry = tk.Entry(addr_frame, textvariable=self.account_address, width=50)
        addr_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.style_manager.style_entry(addr_entry)
        
        # Create secret key input
        key_frame = ttk.Frame(api_frame)
        key_frame.pack(fill=tk.X, pady=5)
        
        key_label = ttk.Label(key_frame, text="Secret Key:")
        key_label.pack(side=tk.LEFT, padx=(0, 5))
        
        key_entry = tk.Entry(key_frame, textvariable=self.secret_key, width=50, show="*")
        key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.style_manager.style_entry(key_entry)
        
        # Create show/hide secret key checkbox
        show_key_check = ttk.Checkbutton(key_frame, text="Show", variable=self.show_secret_key, command=self._toggle_show_secret_key)
        show_key_check.pack(side=tk.LEFT, padx=(5, 0))
        
        # Create API key buttons
        api_buttons_frame = ttk.Frame(api_frame)
        api_buttons_frame.pack(fill=tk.X, pady=5)
        
        save_api_button = tk.Button(api_buttons_frame, text="Save API Keys", command=self._save_api_keys)
        save_api_button.pack(side=tk.LEFT, padx=(0, 5))
        self.style_manager.style_button(save_api_button, "success")
        
        test_api_button = tk.Button(api_buttons_frame, text="Test Connection", command=self._test_connection)
        test_api_button.pack(side=tk.LEFT)
        self.style_manager.style_button(test_api_button)
        
        # Create trading settings section
        trading_frame = ttk.Frame(settings_frame)
        trading_frame.pack(fill=tk.X, pady=(0, 20))
        
        trading_title = ttk.Label(trading_frame, text="Trading Settings", style="Header.TLabel")
        trading_title.pack(anchor=tk.W, pady=(0, 10))
        
        # Create default position size input
        default_size_frame = ttk.Frame(trading_frame)
        default_size_frame.pack(fill=tk.X, pady=5)
        
        default_size_label = ttk.Label(default_size_frame, text="Default Position Size:")
        default_size_label.pack(side=tk.LEFT, padx=(0, 5))
        
        default_size_entry = tk.Entry(default_size_frame, textvariable=self.position_size, width=10)
        default_size_entry.pack(side=tk.LEFT)
        self.style_manager.style_entry(default_size_entry)
        
        # Create default stop loss input
        default_sl_frame = ttk.Frame(trading_frame)
        default_sl_frame.pack(fill=tk.X, pady=5)
        
        default_sl_label = ttk.Label(default_sl_frame, text="Default Stop Loss %:")
        default_sl_label.pack(side=tk.LEFT, padx=(0, 5))
        
        default_sl_entry = tk.Entry(default_sl_frame, textvariable=self.stop_loss, width=10)
        default_sl_entry.pack(side=tk.LEFT)
        self.style_manager.style_entry(default_sl_entry)
        
        # Create default take profit input
        default_tp_frame = ttk.Frame(trading_frame)
        default_tp_frame.pack(fill=tk.X, pady=5)
        
        default_tp_label = ttk.Label(default_tp_frame, text="Default Take Profit %:")
        default_tp_label.pack(side=tk.LEFT, padx=(0, 5))
        
        default_tp_entry = tk.Entry(default_tp_frame, textvariable=self.take_profit, width=10)
        default_tp_entry.pack(side=tk.LEFT)
        self.style_manager.style_entry(default_tp_entry)
        
        # Create trading mode section
        mode_frame = ttk.Frame(settings_frame)
        mode_frame.pack(fill=tk.X, pady=(0, 20))
        
        mode_title = ttk.Label(mode_frame, text="Trading Mode", style="Header.TLabel")
        mode_title.pack(anchor=tk.W, pady=(0, 10))
        
        # Create mode selection
        modes_frame = ttk.Frame(mode_frame)
        modes_frame.pack(fill=tk.X, pady=5)
        
        # Create radio buttons for each mode
        self.mode_var = tk.StringVar(value=self.mode_manager.get_current_mode().name)
        
        for i, mode in enumerate(TradingMode):
            mode_radio = ttk.Radiobutton(
                modes_frame, 
                text=mode.name, 
                variable=self.mode_var, 
                value=mode.name,
                command=self._on_mode_changed
            )
            mode_radio.grid(row=i//3, column=i%3, sticky=tk.W, padx=5, pady=2)
        
        # Create mode description
        mode_desc_frame = ttk.Frame(mode_frame)
        mode_desc_frame.pack(fill=tk.X, pady=5)
        
        mode_desc_label = ttk.Label(mode_desc_frame, text="Mode Description:")
        mode_desc_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.mode_desc_text = tk.Text(mode_desc_frame, height=4, wrap=tk.WORD)
        self.mode_desc_text.pack(fill=tk.X)
        self.style_manager.style_text(self.mode_desc_text)
        self.mode_desc_text.config(state=tk.DISABLED)
        
        # Update mode description
        self._update_mode_description()
        
        # Create save settings button
        save_settings_button = tk.Button(settings_frame, text="Save All Settings", command=self._save_config)
        save_settings_button.pack(anchor=tk.W, pady=5)
        self.style_manager.style_button(save_settings_button, "success")
    
    def _init_strategy_tab(self) -> None:
        """Initialize the strategy tab."""
        # Create scrollable frame
        container, strategy_frame = self.style_manager.create_scrollable_frame(self.strategy_tab)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create strategy settings section
        strategy_settings_frame = ttk.Frame(strategy_frame)
        strategy_settings_frame.pack(fill=tk.X, pady=(0, 20))
        
        strategy_title = ttk.Label(strategy_settings_frame, text="Strategy Settings", style="Header.TLabel")
        strategy_title.pack(anchor=tk.W, pady=(0, 10))
        
        # Create risk level input
        risk_frame = ttk.Frame(strategy_settings_frame)
        risk_frame.pack(fill=tk.X, pady=5)
        
        risk_label = ttk.Label(risk_frame, text="Risk Level (%):")
        risk_label.pack(side=tk.LEFT, padx=(0, 5))
        
        risk_spinbox = ttk.Spinbox(risk_frame, from_=0.1, to=5.0, increment=0.1, textvariable=self.risk_level, width=10)
        risk_spinbox.pack(side=tk.LEFT)
        
        # Create take profit multiplier input
        tp_mult_frame = ttk.Frame(strategy_settings_frame)
        tp_mult_frame.pack(fill=tk.X, pady=5)
        
        tp_mult_label = ttk.Label(tp_mult_frame, text="Take Profit Multiplier:")
        tp_mult_label.pack(side=tk.LEFT, padx=(0, 5))
        
        tp_mult_spinbox = ttk.Spinbox(tp_mult_frame, from_=1.0, to=10.0, increment=0.1, textvariable=self.tp_multiplier, width=10)
        tp_mult_spinbox.pack(side=tk.LEFT)
        
        # Create stop loss multiplier input
        sl_mult_frame = ttk.Frame(strategy_settings_frame)
        sl_mult_frame.pack(fill=tk.X, pady=5)
        
        sl_mult_label = ttk.Label(sl_mult_frame, text="Stop Loss Multiplier:")
        sl_mult_label.pack(side=tk.LEFT, padx=(0, 5))
        
        sl_mult_spinbox = ttk.Spinbox(sl_mult_frame, from_=0.5, to=5.0, increment=0.1, textvariable=self.sl_multiplier, width=10)
        sl_mult_spinbox.pack(side=tk.LEFT)
        
        # Create filter checkboxes
        filters_frame = ttk.Frame(strategy_settings_frame)
        filters_frame.pack(fill=tk.X, pady=5)
        
        volatility_check = ttk.Checkbutton(filters_frame, text="Use Volatility Filters", variable=self.use_volatility_filters)
        volatility_check.grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        
        trend_check = ttk.Checkbutton(filters_frame, text="Use Trend Filters", variable=self.use_trend_filters)
        trend_check.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        volume_check = ttk.Checkbutton(filters_frame, text="Use Volume Filters", variable=self.use_volume_filters)
        volume_check.grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        
        # Create mock data checkbox
        mock_data_check = ttk.Checkbutton(filters_frame, text="Use Mock Data (for testing)", variable=self.use_mock_data)
        mock_data_check.grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # Create strategy visualization section
        visualization_frame = ttk.Frame(strategy_frame)
        visualization_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        visualization_title = ttk.Label(visualization_frame, text="Strategy Visualization", style="Header.TLabel")
        visualization_title.pack(anchor=tk.W, pady=(0, 10))
        
        # Create strategy chart
        self.strategy_chart_frame = ttk.Frame(visualization_frame)
        self.strategy_chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create strategy chart
        self._create_strategy_chart()
        
        # Create save strategy button
        save_strategy_button = tk.Button(strategy_frame, text="Save Strategy Settings", command=self._save_strategy_settings)
        save_strategy_button.pack(anchor=tk.W, pady=5)
        self.style_manager.style_button(save_strategy_button, "success")
    
    def _init_advanced_tab(self) -> None:
        """Initialize the advanced tab."""
        # Create scrollable frame
        container, advanced_frame = self.style_manager.create_scrollable_frame(self.advanced_tab)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create advanced order types section
        order_types_frame = ttk.Frame(advanced_frame)
        order_types_frame.pack(fill=tk.X, pady=(0, 20))
        
        order_types_title = ttk.Label(order_types_frame, text="Advanced Order Types", style="Header.TLabel")
        order_types_title.pack(anchor=tk.W, pady=(0, 10))
        
        # Create TWAP order section
        twap_frame = ttk.LabelFrame(order_types_frame, text="TWAP Orders")
        twap_frame.pack(fill=tk.X, pady=5)
        
        # TWAP symbol
        twap_symbol_frame = ttk.Frame(twap_frame)
        twap_symbol_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(twap_symbol_frame, text="Symbol:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.twap_symbol_var = tk.StringVar(value=self.selected_symbol.get())
        twap_symbol_combo = ttk.Combobox(twap_symbol_frame, textvariable=self.twap_symbol_var, values=self.available_symbols, width=10)
        twap_symbol_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        # TWAP side
        ttk.Label(twap_symbol_frame, text="Side:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.twap_side_var = tk.StringVar(value="BUY")
        twap_side_combo = ttk.Combobox(twap_symbol_frame, textvariable=self.twap_side_var, values=["BUY", "SELL"], width=10)
        twap_side_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        # TWAP position type
        ttk.Label(twap_symbol_frame, text="Position:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.twap_position_var = tk.StringVar(value="LONG")
        twap_position_combo = ttk.Combobox(twap_symbol_frame, textvariable=self.twap_position_var, values=["LONG", "SHORT"], width=10)
        twap_position_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        # TWAP parameters
        twap_params_frame = ttk.Frame(twap_frame)
        twap_params_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(twap_params_frame, text="Size:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.twap_size_var = tk.StringVar(value="0.01")
        twap_size_entry = tk.Entry(twap_params_frame, textvariable=self.twap_size_var, width=10)
        twap_size_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.style_manager.style_entry(twap_size_entry)
        
        ttk.Label(twap_params_frame, text="Duration (min):").pack(side=tk.LEFT, padx=(0, 5))
        
        self.twap_duration_var = tk.StringVar(value="10")
        twap_duration_entry = tk.Entry(twap_params_frame, textvariable=self.twap_duration_var, width=10)
        twap_duration_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.style_manager.style_entry(twap_duration_entry)
        
        ttk.Label(twap_params_frame, text="Slices:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.twap_slices_var = tk.StringVar(value="5")
        twap_slices_entry = tk.Entry(twap_params_frame, textvariable=self.twap_slices_var, width=10)
        twap_slices_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.style_manager.style_entry(twap_slices_entry)
        
        # TWAP execute button
        twap_execute_button = tk.Button(twap_frame, text="Execute TWAP Order", command=self._execute_twap_order)
        twap_execute_button.pack(anchor=tk.W, pady=5)
        self.style_manager.style_button(twap_execute_button, "success")
        
        # Create Scale order section
        scale_frame = ttk.LabelFrame(order_types_frame, text="Scale Orders")
        scale_frame.pack(fill=tk.X, pady=5)
        
        # Scale symbol
        scale_symbol_frame = ttk.Frame(scale_frame)
        scale_symbol_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(scale_symbol_frame, text="Symbol:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.scale_symbol_var = tk.StringVar(value=self.selected_symbol.get())
        scale_symbol_combo = ttk.Combobox(scale_symbol_frame, textvariable=self.scale_symbol_var, values=self.available_symbols, width=10)
        scale_symbol_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        # Scale side
        ttk.Label(scale_symbol_frame, text="Side:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.scale_side_var = tk.StringVar(value="BUY")
        scale_side_combo = ttk.Combobox(scale_symbol_frame, textvariable=self.scale_side_var, values=["BUY", "SELL"], width=10)
        scale_side_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        # Scale position type
        ttk.Label(scale_symbol_frame, text="Position:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.scale_position_var = tk.StringVar(value="LONG")
        scale_position_combo = ttk.Combobox(scale_symbol_frame, textvariable=self.scale_position_var, values=["LONG", "SHORT"], width=10)
        scale_position_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        # Scale parameters
        scale_params_frame = ttk.Frame(scale_frame)
        scale_params_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(scale_params_frame, text="Total Size:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.scale_size_var = tk.StringVar(value="0.01")
        scale_size_entry = tk.Entry(scale_params_frame, textvariable=self.scale_size_var, width=10)
        scale_size_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.style_manager.style_entry(scale_size_entry)
        
        ttk.Label(scale_params_frame, text="Price Range (%):").pack(side=tk.LEFT, padx=(0, 5))
        
        self.scale_range_var = tk.StringVar(value="1.0")
        scale_range_entry = tk.Entry(scale_params_frame, textvariable=self.scale_range_var, width=10)
        scale_range_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.style_manager.style_entry(scale_range_entry)
        
        ttk.Label(scale_params_frame, text="Orders:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.scale_orders_var = tk.StringVar(value="5")
        scale_orders_entry = tk.Entry(scale_params_frame, textvariable=self.scale_orders_var, width=10)
        scale_orders_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.style_manager.style_entry(scale_orders_entry)
        
        # Scale execute button
        scale_execute_button = tk.Button(scale_frame, text="Execute Scale Order", command=self._execute_scale_order)
        scale_execute_button.pack(anchor=tk.W, pady=5)
        self.style_manager.style_button(scale_execute_button, "success")
        
        # Create risk management section
        risk_frame = ttk.Frame(advanced_frame)
        risk_frame.pack(fill=tk.X, pady=(0, 20))
        
        risk_title = ttk.Label(risk_frame, text="Risk Management", style="Header.TLabel")
        risk_title.pack(anchor=tk.W, pady=(0, 10))
        
        # Create risk parameters
        risk_params_frame = ttk.Frame(risk_frame)
        risk_params_frame.pack(fill=tk.X, pady=5)
        
        # Max drawdown
        ttk.Label(risk_params_frame, text="Max Drawdown (%):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        
        self.max_drawdown_var = tk.StringVar(value=str(self.risk_manager.get_max_drawdown() * 100))
        max_drawdown_entry = tk.Entry(risk_params_frame, textvariable=self.max_drawdown_var, width=10)
        max_drawdown_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        self.style_manager.style_entry(max_drawdown_entry)
        
        # Daily loss limit
        ttk.Label(risk_params_frame, text="Daily Loss Limit (%):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        
        self.daily_loss_var = tk.StringVar(value=str(self.risk_manager.get_daily_loss_limit() * 100))
        daily_loss_entry = tk.Entry(risk_params_frame, textvariable=self.daily_loss_var, width=10)
        daily_loss_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        self.style_manager.style_entry(daily_loss_entry)
        
        # Max position size
        ttk.Label(risk_params_frame, text="Max Position Size (%):").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        
        self.max_position_var = tk.StringVar(value=str(self.risk_manager.get_max_position_size() * 100))
        max_position_entry = tk.Entry(risk_params_frame, textvariable=self.max_position_var, width=10)
        max_position_entry.grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        self.style_manager.style_entry(max_position_entry)
        
        # Max open positions
        ttk.Label(risk_params_frame, text="Max Open Positions:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        
        self.max_positions_var = tk.StringVar(value=str(self.risk_manager.get_max_open_positions()))
        max_positions_entry = tk.Entry(risk_params_frame, textvariable=self.max_positions_var, width=10)
        max_positions_entry.grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
        self.style_manager.style_entry(max_positions_entry)
        
        # Save risk settings button
        save_risk_button = tk.Button(risk_frame, text="Save Risk Settings", command=self._save_risk_settings)
        save_risk_button.pack(anchor=tk.W, pady=5)
        self.style_manager.style_button(save_risk_button, "success")
        
        # Create connection management section
        connection_frame = ttk.Frame(advanced_frame)
        connection_frame.pack(fill=tk.X, pady=(0, 20))
        
        connection_title = ttk.Label(connection_frame, text="Connection Management", style="Header.TLabel")
        connection_title.pack(anchor=tk.W, pady=(0, 10))
        
        # Connection status
        connection_status_frame = ttk.Frame(connection_frame)
        connection_status_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(connection_status_frame, text="Connection Status:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.connection_status_var = tk.StringVar(value="Disconnected")
        connection_status_label = ttk.Label(connection_status_frame, textvariable=self.connection_status_var)
        connection_status_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Connection buttons
        connection_buttons_frame = ttk.Frame(connection_frame)
        connection_buttons_frame.pack(fill=tk.X, pady=5)
        
        reconnect_button = tk.Button(connection_buttons_frame, text="Force Reconnect", command=self._force_reconnect)
        reconnect_button.pack(side=tk.LEFT, padx=(0, 5))
        self.style_manager.style_button(reconnect_button)
        
        reset_button = tk.Button(connection_buttons_frame, text="Reset Connection", command=self._reset_connection)
        reset_button.pack(side=tk.LEFT, padx=(0, 5))
        self.style_manager.style_button(reset_button)
        
        test_button = tk.Button(connection_buttons_frame, text="Test Connection", command=self._test_connection)
        test_button.pack(side=tk.LEFT)
        self.style_manager.style_button(test_button)
    
    def _init_logs_tab(self) -> None:
        """Initialize the logs tab."""
        # Create log text widget
        self.log_text = scrolledtext.ScrolledText(self.logs_tab, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.log_text.config(state=tk.DISABLED)
        
        # Add log handler
        self.log_handler = GUILogHandler(self.log_text)
        logging.getLogger().addHandler(self.log_handler)
    
    def _create_chart(self) -> None:
        """Create price chart."""
        # Create figure and axis
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Update chart
        self._update_chart()
    
    def _create_strategy_chart(self) -> None:
        """Create strategy chart."""
        # Create figure and axes
        self.strategy_fig = Figure(figsize=(5, 4), dpi=100)
        self.strategy_ax1 = self.strategy_fig.add_subplot(211)
        self.strategy_ax2 = self.strategy_fig.add_subplot(212, sharex=self.strategy_ax1)
        
        # Create canvas
        self.strategy_canvas = FigureCanvasTkAgg(self.strategy_fig, master=self.strategy_chart_frame)
        self.strategy_canvas.draw()
        self.strategy_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Update chart
        self._update_strategy_chart()
    
    def _update_chart(self) -> None:
        """Update price chart with latest data."""
        try:
            # Clear previous data
            self.ax.clear()
            
            # Get price data
            symbol = self.selected_symbol.get()
            timeframe = self.selected_timeframe.get()
            
            # Get data from trading integration
            data = self.trading.get_price_data(symbol, timeframe)
            
            if data is not None and len(data) > 0:
                # Plot price data
                self.ax.plot(data.index, data['close'], label='Close Price')
                
                # Add moving averages
                self.ax.plot(data.index, data['ma20'], label='MA20')
                self.ax.plot(data.index, data['ma50'], label='MA50')
                
                # Add buy/sell signals if available
                if 'buy_signal' in data.columns:
                    buy_signals = data[data['buy_signal'] == 1]
                    self.ax.scatter(buy_signals.index, buy_signals['close'], color='green', marker='^', s=100, label='Buy')
                
                if 'sell_signal' in data.columns:
                    sell_signals = data[data['sell_signal'] == 1]
                    self.ax.scatter(sell_signals.index, sell_signals['close'], color='red', marker='v', s=100, label='Sell')
                
                # Set labels and title
                self.ax.set_title(f'{symbol} - {timeframe}')
                self.ax.set_xlabel('Date')
                self.ax.set_ylabel('Price')
                self.ax.legend()
                self.ax.grid(True)
                
                # Format x-axis dates
                self.fig.autofmt_xdate()
                
                # Adjust layout
                self.fig.tight_layout()
            else:
                self.ax.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center', transform=self.ax.transAxes)
            
            # Redraw canvas
            self.canvas.draw()
        except Exception as e:
            self.logger.error(f"Error updating chart: {e}")
    
    def _update_strategy_chart(self) -> None:
        """Update strategy chart with latest data."""
        try:
            # Clear previous data
            self.strategy_ax1.clear()
            self.strategy_ax2.clear()
            
            # Get price data
            symbol = self.selected_symbol.get()
            timeframe = self.selected_timeframe.get()
            
            # Get data from trading integration
            data = self.trading.get_strategy_data(symbol, timeframe)
            
            if data is not None and len(data) > 0:
                # Plot price data on top chart
                self.strategy_ax1.plot(data.index, data['close'], label='Close Price')
                
                # Add moving averages
                self.strategy_ax1.plot(data.index, data['ma20'], label='MA20')
                self.strategy_ax1.plot(data.index, data['ma50'], label='MA50')
                
                # Add buy/sell signals if available
                if 'buy_signal' in data.columns:
                    buy_signals = data[data['buy_signal'] == 1]
                    self.strategy_ax1.scatter(buy_signals.index, buy_signals['close'], color='green', marker='^', s=100, label='Buy')
                
                if 'sell_signal' in data.columns:
                    sell_signals = data[data['sell_signal'] == 1]
                    self.strategy_ax1.scatter(sell_signals.index, sell_signals['close'], color='red', marker='v', s=100, label='Sell')
                
                # Plot indicators on bottom chart
                if 'rsi' in data.columns:
                    self.strategy_ax2.plot(data.index, data['rsi'], label='RSI')
                    self.strategy_ax2.axhline(y=70, color='r', linestyle='-', alpha=0.3)
                    self.strategy_ax2.axhline(y=30, color='g', linestyle='-', alpha=0.3)
                
                # Set labels and title
                self.strategy_ax1.set_title(f'{symbol} - {timeframe} Strategy Analysis')
                self.strategy_ax2.set_xlabel('Date')
                self.strategy_ax1.set_ylabel('Price')
                self.strategy_ax2.set_ylabel('Indicators')
                self.strategy_ax1.legend()
                self.strategy_ax2.legend()
                self.strategy_ax1.grid(True)
                self.strategy_ax2.grid(True)
                
                # Format x-axis dates
                self.strategy_fig.autofmt_xdate()
                
                # Adjust layout
                self.strategy_fig.tight_layout()
            else:
                self.strategy_ax1.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center', transform=self.strategy_ax1.transAxes)
            
            # Redraw canvas
            self.strategy_canvas.draw()
        except Exception as e:
            self.logger.error(f"Error updating strategy chart: {e}")
    
    def _start_update_loops(self) -> None:
        """Start update loops for various components."""
        # Start connection status update loop
        self._update_connection_status()
        
        # Start positions update loop
        self._update_positions()
        
        # Start chart update loop
        self.root.after(30000, self._update_chart)
        
        # Start strategy chart update loop
        self.root.after(30000, self._update_strategy_chart)
    
    def _update_connection_status(self) -> None:
        """Update connection status indicator."""
        try:
            is_connected = self.connection_manager.is_connected()
            
            if is_connected:
                self.connection_status.config(text="Connected", bg="green")
                self.connection_status_var.set("Connected")
            else:
                self.connection_status.config(text="Disconnected", bg="red")
                self.connection_status_var.set("Disconnected")
                
                # Try to reconnect if not connected
                if self.is_bot_running:
                    self.connection_manager.reconnect()
            
            # Schedule next update
            self.root.after(5000, self._update_connection_status)
        except Exception as e:
            self.logger.error(f"Error updating connection status: {e}")
            self.root.after(10000, self._update_connection_status)
    
    def _update_positions(self) -> None:
        """Update positions display."""
        try:
            # Clear existing items
            for item in self.positions_tree.get_children():
                self.positions_tree.delete(item)
            
            # Get positions from trading integration
            positions = self.trading.get_positions()
            
            if positions:
                for position in positions:
                    # Format values
                    pnl = position.get('pnl', 0)
                    pnl_percent = position.get('pnl_percent', 0)
                    
                    # Add color tags based on PnL
                    if pnl > 0:
                        tag = 'profit'
                    elif pnl < 0:
                        tag = 'loss'
                    else:
                        tag = ''
                    
                    # Insert position into treeview
                    self.positions_tree.insert(
                        '', 'end', 
                        values=(
                            position.get('symbol', ''),
                            position.get('side', ''),
                            position.get('size', ''),
                            position.get('entry_price', ''),
                            position.get('current_price', ''),
                            position.get('liquidation_price', ''),
                            f"{pnl:.2f}",
                            f"{pnl_percent:.2f}%"
                        ),
                        tags=(tag,)
                    )
            
            # Configure tags
            self.positions_tree.tag_configure('profit', background='#d4edda')
            self.positions_tree.tag_configure('loss', background='#f8d7da')
            
            # Schedule next update
            self.root.after(10000, self._update_positions)
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
            self.root.after(15000, self._update_positions)
    
    def _on_symbol_selected(self, event=None) -> None:
        """Handle symbol selection event."""
        try:
            # Update chart
            self._update_chart()
            
            # Update strategy chart
            self._update_strategy_chart()
        except Exception as e:
            self.logger.error(f"Error handling symbol selection: {e}")
    
    def _on_timeframe_selected(self, event=None) -> None:
        """Handle timeframe selection event."""
        try:
            # Update chart
            self._update_chart()
            
            # Update strategy chart
            self._update_strategy_chart()
        except Exception as e:
            self.logger.error(f"Error handling timeframe selection: {e}")
    
    def _on_mode_changed(self, event=None) -> None:
        """Handle trading mode change event."""
        try:
            # Get selected mode
            mode_name = self.mode_var.get()
            mode = TradingMode[mode_name]
            
            # Set mode
            self.mode_manager.set_mode(mode)
            
            # Update mode indicator
            self.mode_indicator.config(text=mode_name, bg=self._get_mode_color())
            
            # Update mode description
            self._update_mode_description()
            
            self.logger.info(f"Trading mode changed to {mode_name}")
        except Exception as e:
            self.logger.error(f"Error changing trading mode: {e}")
    
    def _update_mode_description(self) -> None:
        """Update mode description text."""
        try:
            # Get current mode
            mode = self.mode_manager.get_current_mode()
            
            # Get description
            description = self.mode_manager.get_mode_description(mode)
            
            # Update text widget
            self.mode_desc_text.config(state=tk.NORMAL)
            self.mode_desc_text.delete(1.0, tk.END)
            self.mode_desc_text.insert(tk.END, description)
            self.mode_desc_text.config(state=tk.DISABLED)
        except Exception as e:
            self.logger.error(f"Error updating mode description: {e}")
    
    def _get_mode_color(self) -> str:
        """Get color for current trading mode."""
        mode = self.mode_manager.get_current_mode()
        
        if mode == TradingMode.PAPER_TRADING:
            return "#007bff"  # Blue
        elif mode == TradingMode.LIVE_TRADING:
            return "#28a745"  # Green
        elif mode == TradingMode.MONITOR_ONLY:
            return "#6c757d"  # Gray
        elif mode == TradingMode.AGGRESSIVE:
            return "#dc3545"  # Red
        elif mode == TradingMode.CONSERVATIVE:
            return "#17a2b8"  # Cyan
        else:
            return "#6c757d"  # Gray
    
    def _toggle_show_secret_key(self) -> None:
        """Toggle showing/hiding secret key."""
        try:
            for widget in self.root.winfo_children():
                if isinstance(widget, tk.Entry) and widget.cget("show") == "*":
                    widget.config(show="" if self.show_secret_key.get() else "*")
        except Exception as e:
            self.logger.error(f"Error toggling secret key visibility: {e}")
    
    def _save_api_keys(self) -> None:
        """Save API keys to config."""
        try:
            # Update config
            self.config["account_address"] = self.account_address.get()
            self.config["secret_key"] = self.secret_key.get()
            
            # Save config
            self.settings_manager.save_settings(self.config)
            
            # Update trading integration
            self.trading.update_api_keys(self.account_address.get(), self.secret_key.get())
            
            self.logger.info("API keys saved successfully")
            messagebox.showinfo("API Keys", "API keys saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving API keys: {e}")
            messagebox.showerror("Error", f"Error saving API keys: {e}")
    
    def _test_connection(self) -> None:
        """Test connection to exchange."""
        try:
            # Test connection
            result = self.connection_manager.test_connection()
            
            if result:
                self.logger.info("Connection test successful")
                messagebox.showinfo("Connection Test", "Connection test successful")
            else:
                self.logger.warning("Connection test failed")
                messagebox.showwarning("Connection Test", "Connection test failed")
        except Exception as e:
            self.logger.error(f"Error testing connection: {e}")
            messagebox.showerror("Error", f"Error testing connection: {e}")
    
    def _force_reconnect(self) -> None:
        """Force reconnection to exchange."""
        try:
            # Force reconnect
            self.connection_manager.reconnect(force=True)
            
            self.logger.info("Forced reconnection initiated")
            messagebox.showinfo("Reconnect", "Forced reconnection initiated")
        except Exception as e:
            self.logger.error(f"Error forcing reconnection: {e}")
            messagebox.showerror("Error", f"Error forcing reconnection: {e}")
    
    def _reset_connection(self) -> None:
        """Reset connection to exchange."""
        try:
            # Reset connection
            self.connection_manager.reset()
            
            self.logger.info("Connection reset successful")
            messagebox.showinfo("Reset Connection", "Connection reset successful")
        except Exception as e:
            self.logger.error(f"Error resetting connection: {e}")
            messagebox.showerror("Error", f"Error resetting connection: {e}")
    
    def _refresh_dashboard(self) -> None:
        """Refresh dashboard data."""
        try:
            # Update chart
            self._update_chart()
            
            # Update positions
            self._update_positions()
            
            self.logger.info("Dashboard refreshed")
        except Exception as e:
            self.logger.error(f"Error refreshing dashboard: {e}")
    
    def _refresh_positions(self) -> None:
        """Refresh positions data."""
        try:
            # Update positions
            self._update_positions()
            
            self.logger.info("Positions refreshed")
        except Exception as e:
            self.logger.error(f"Error refreshing positions: {e}")
    
    def _place_order(self, is_buy: bool, is_long: bool) -> None:
        """
        Place an order.
        
        Args:
            is_buy: Whether this is a buy order
            is_long: Whether this is a long position
        """
        try:
            # Get order parameters
            symbol = self.selected_symbol.get()
            size = float(self.position_size.get())
            stop_loss_percent = float(self.stop_loss.get())
            take_profit_percent = float(self.take_profit.get())
            
            # Place order
            result = self.trading.place_order(
                symbol=symbol,
                is_buy=is_buy,
                is_long=is_long,
                size=size,
                stop_loss_percent=stop_loss_percent,
                take_profit_percent=take_profit_percent
            )
            
            if result.get('success', False):
                self.logger.info(f"Order placed successfully: {result.get('order_id')}")
                messagebox.showinfo("Order Placed", f"Order placed successfully: {result.get('order_id')}")
                
                # Refresh positions
                self._refresh_positions()
            else:
                self.logger.warning(f"Order placement failed: {result.get('error')}")
                messagebox.showwarning("Order Failed", f"Order placement failed: {result.get('error')}")
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            messagebox.showerror("Error", f"Error placing order: {e}")
    
    def _close_selected_position(self) -> None:
        """Close the selected position."""
        try:
            # Get selected position
            selected_item = self.positions_tree.selection()
            
            if not selected_item:
                messagebox.showinfo("No Selection", "Please select a position to close")
                return
            
            # Get position details
            values = self.positions_tree.item(selected_item, 'values')
            symbol = values[0]
            side = values[1]
            
            # Confirm close
            if messagebox.askyesno("Close Position", f"Are you sure you want to close the {side} position for {symbol}?"):
                # Close position
                result = self.trading.close_position(symbol, side == "LONG")
                
                if result.get('success', False):
                    self.logger.info(f"Position closed successfully: {symbol} {side}")
                    messagebox.showinfo("Position Closed", f"Position closed successfully: {symbol} {side}")
                    
                    # Refresh positions
                    self._refresh_positions()
                else:
                    self.logger.warning(f"Position close failed: {result.get('error')}")
                    messagebox.showwarning("Close Failed", f"Position close failed: {result.get('error')}")
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            messagebox.showerror("Error", f"Error closing position: {e}")
    
    def _modify_stop_loss(self) -> None:
        """Modify stop loss for selected position."""
        try:
            # Get selected position
            selected_item = self.positions_tree.selection()
            
            if not selected_item:
                messagebox.showinfo("No Selection", "Please select a position to modify")
                return
            
            # Get position details
            values = self.positions_tree.item(selected_item, 'values')
            symbol = values[0]
            side = values[1]
            current_price = float(values[4])
            
            # Ask for new stop loss
            new_stop_loss = simpledialog.askfloat(
                "Modify Stop Loss", 
                f"Enter new stop loss price for {symbol} {side}:",
                minvalue=0.0001,
                maxvalue=current_price * 2
            )
            
            if new_stop_loss is not None:
                # Modify stop loss
                result = self.trading.modify_stop_loss(symbol, side == "LONG", new_stop_loss)
                
                if result.get('success', False):
                    self.logger.info(f"Stop loss modified successfully: {symbol} {side} to {new_stop_loss}")
                    messagebox.showinfo("Stop Loss Modified", f"Stop loss modified successfully: {symbol} {side} to {new_stop_loss}")
                    
                    # Refresh positions
                    self._refresh_positions()
                else:
                    self.logger.warning(f"Stop loss modification failed: {result.get('error')}")
                    messagebox.showwarning("Modification Failed", f"Stop loss modification failed: {result.get('error')}")
        except Exception as e:
            self.logger.error(f"Error modifying stop loss: {e}")
            messagebox.showerror("Error", f"Error modifying stop loss: {e}")
    
    def _modify_take_profit(self) -> None:
        """Modify take profit for selected position."""
        try:
            # Get selected position
            selected_item = self.positions_tree.selection()
            
            if not selected_item:
                messagebox.showinfo("No Selection", "Please select a position to modify")
                return
            
            # Get position details
            values = self.positions_tree.item(selected_item, 'values')
            symbol = values[0]
            side = values[1]
            current_price = float(values[4])
            
            # Ask for new take profit
            new_take_profit = simpledialog.askfloat(
                "Modify Take Profit", 
                f"Enter new take profit price for {symbol} {side}:",
                minvalue=0.0001,
                maxvalue=current_price * 2
            )
            
            if new_take_profit is not None:
                # Modify take profit
                result = self.trading.modify_take_profit(symbol, side == "LONG", new_take_profit)
                
                if result.get('success', False):
                    self.logger.info(f"Take profit modified successfully: {symbol} {side} to {new_take_profit}")
                    messagebox.showinfo("Take Profit Modified", f"Take profit modified successfully: {symbol} {side} to {new_take_profit}")
                    
                    # Refresh positions
                    self._refresh_positions()
                else:
                    self.logger.warning(f"Take profit modification failed: {result.get('error')}")
                    messagebox.showwarning("Modification Failed", f"Take profit modification failed: {result.get('error')}")
        except Exception as e:
            self.logger.error(f"Error modifying take profit: {e}")
            messagebox.showerror("Error", f"Error modifying take profit: {e}")
    
    def _add_trailing_stop(self) -> None:
        """Add trailing stop for selected position."""
        try:
            # Get selected position
            selected_item = self.positions_tree.selection()
            
            if not selected_item:
                messagebox.showinfo("No Selection", "Please select a position to modify")
                return
            
            # Get position details
            values = self.positions_tree.item(selected_item, 'values')
            symbol = values[0]
            side = values[1]
            
            # Ask for trailing stop parameters
            activation_percent = simpledialog.askfloat(
                "Add Trailing Stop", 
                f"Enter activation percent for {symbol} {side} trailing stop:",
                minvalue=0.1,
                maxvalue=10.0
            )
            
            if activation_percent is not None:
                callback_percent = simpledialog.askfloat(
                    "Add Trailing Stop", 
                    f"Enter callback percent for {symbol} {side} trailing stop:",
                    minvalue=0.1,
                    maxvalue=5.0
                )
                
                if callback_percent is not None:
                    # Add trailing stop
                    result = self.trading.add_trailing_stop(
                        symbol, 
                        side == "LONG", 
                        activation_percent / 100, 
                        callback_percent / 100
                    )
                    
                    if result.get('success', False):
                        self.logger.info(f"Trailing stop added successfully: {symbol} {side}")
                        messagebox.showinfo("Trailing Stop Added", f"Trailing stop added successfully: {symbol} {side}")
                        
                        # Refresh positions
                        self._refresh_positions()
                    else:
                        self.logger.warning(f"Trailing stop addition failed: {result.get('error')}")
                        messagebox.showwarning("Addition Failed", f"Trailing stop addition failed: {result.get('error')}")
        except Exception as e:
            self.logger.error(f"Error adding trailing stop: {e}")
            messagebox.showerror("Error", f"Error adding trailing stop: {e}")
    
    def _show_positions_context_menu(self, event) -> None:
        """Show context menu for positions."""
        try:
            # Get item under cursor
            item = self.positions_tree.identify_row(event.y)
            
            if item:
                # Select the item
                self.positions_tree.selection_set(item)
                
                # Show context menu
                self.positions_context_menu.post(event.x_root, event.y_root)
        except Exception as e:
            self.logger.error(f"Error showing positions context menu: {e}")
    
    def _toggle_bot(self) -> None:
        """Toggle bot running state."""
        try:
            if self.is_bot_running:
                # Stop bot
                self.is_bot_running = False
                self.bot_button.config(text="Start Bot")
                self.style_manager.style_button(self.bot_button, "success")
                
                # Stop trading
                self.trading.stop_trading()
                
                self.logger.info("Bot stopped")
            else:
                # Start bot
                self.is_bot_running = True
                self.bot_button.config(text="Stop Bot")
                self.style_manager.style_button(self.bot_button, "error")
                
                # Start trading
                self.trading.start_trading(
                    symbol=self.selected_symbol.get(),
                    timeframe=self.selected_timeframe.get()
                )
                
                self.logger.info("Bot started")
        except Exception as e:
            self.logger.error(f"Error toggling bot: {e}")
            messagebox.showerror("Error", f"Error toggling bot: {e}")
    
    def _save_strategy_settings(self) -> None:
        """Save strategy settings."""
        try:
            # Update config
            self.config["risk_level"] = self.risk_level.get()
            self.config["tp_multiplier"] = self.tp_multiplier.get()
            self.config["sl_multiplier"] = self.sl_multiplier.get()
            self.config["use_volatility_filters"] = self.use_volatility_filters.get()
            self.config["use_trend_filters"] = self.use_trend_filters.get()
            self.config["use_volume_filters"] = self.use_volume_filters.get()
            self.config["use_mock_data"] = self.use_mock_data.get()
            
            # Save config
            self.settings_manager.save_settings(self.config)
            
            # Update trading integration
            self.trading.update_strategy_settings(
                risk_level=self.risk_level.get(),
                tp_multiplier=self.tp_multiplier.get(),
                sl_multiplier=self.sl_multiplier.get(),
                use_volatility_filters=self.use_volatility_filters.get(),
                use_trend_filters=self.use_trend_filters.get(),
                use_volume_filters=self.use_volume_filters.get(),
                use_mock_data=self.use_mock_data.get()
            )
            
            self.logger.info("Strategy settings saved successfully")
            messagebox.showinfo("Strategy Settings", "Strategy settings saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving strategy settings: {e}")
            messagebox.showerror("Error", f"Error saving strategy settings: {e}")
    
    def _save_risk_settings(self) -> None:
        """Save risk settings."""
        try:
            # Get values
            max_drawdown = float(self.max_drawdown_var.get()) / 100
            daily_loss_limit = float(self.daily_loss_var.get()) / 100
            max_position_size = float(self.max_position_var.get()) / 100
            max_open_positions = int(self.max_positions_var.get())
            
            # Update risk manager
            self.risk_manager.set_max_drawdown(max_drawdown)
            self.risk_manager.set_daily_loss_limit(daily_loss_limit)
            self.risk_manager.set_max_position_size(max_position_size)
            self.risk_manager.set_max_open_positions(max_open_positions)
            
            # Update config
            self.config["max_drawdown"] = max_drawdown
            self.config["daily_loss_limit"] = daily_loss_limit
            self.config["max_position_size"] = max_position_size
            self.config["max_open_positions"] = max_open_positions
            
            # Save config
            self.settings_manager.save_settings(self.config)
            
            self.logger.info("Risk settings saved successfully")
            messagebox.showinfo("Risk Settings", "Risk settings saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving risk settings: {e}")
            messagebox.showerror("Error", f"Error saving risk settings: {e}")
    
    def _execute_twap_order(self) -> None:
        """Execute TWAP order."""
        try:
            # Get parameters
            symbol = self.twap_symbol_var.get()
            is_buy = self.twap_side_var.get() == "BUY"
            is_long = self.twap_position_var.get() == "LONG"
            size = float(self.twap_size_var.get())
            duration = int(self.twap_duration_var.get())
            slices = int(self.twap_slices_var.get())
            
            # Execute TWAP order
            result = self.order_manager.execute_twap_order(
                symbol=symbol,
                is_buy=is_buy,
                is_long=is_long,
                size=size,
                duration_minutes=duration,
                slices=slices
            )
            
            if result.get('success', False):
                self.logger.info(f"TWAP order executed successfully: {result.get('order_id')}")
                messagebox.showinfo("TWAP Order", f"TWAP order executed successfully: {result.get('order_id')}")
            else:
                self.logger.warning(f"TWAP order execution failed: {result.get('error')}")
                messagebox.showwarning("TWAP Order Failed", f"TWAP order execution failed: {result.get('error')}")
        except Exception as e:
            self.logger.error(f"Error executing TWAP order: {e}")
            messagebox.showerror("Error", f"Error executing TWAP order: {e}")
    
    def _execute_scale_order(self) -> None:
        """Execute scale order."""
        try:
            # Get parameters
            symbol = self.scale_symbol_var.get()
            is_buy = self.scale_side_var.get() == "BUY"
            is_long = self.scale_position_var.get() == "LONG"
            size = float(self.scale_size_var.get())
            price_range = float(self.scale_range_var.get())
            num_orders = int(self.scale_orders_var.get())
            
            # Execute scale order
            result = self.order_manager.execute_scale_order(
                symbol=symbol,
                is_buy=is_buy,
                is_long=is_long,
                total_size=size,
                price_range_percent=price_range,
                num_orders=num_orders
            )
            
            if result.get('success', False):
                self.logger.info(f"Scale order executed successfully: {result.get('order_ids')}")
                messagebox.showinfo("Scale Order", f"Scale order executed successfully: {len(result.get('order_ids', []))} orders placed")
            else:
                self.logger.warning(f"Scale order execution failed: {result.get('error')}")
                messagebox.showwarning("Scale Order Failed", f"Scale order execution failed: {result.get('error')}")
        except Exception as e:
            self.logger.error(f"Error executing scale order: {e}")
            messagebox.showerror("Error", f"Error executing scale order: {e}")


class GUILogHandler(logging.Handler):
    """
    Custom logging handler for GUI log display.
    """
    
    def __init__(self, text_widget):
        """
        Initialize the handler.
        
        Args:
            text_widget: The text widget to display logs
        """
        super().__init__()
        self.text_widget = text_widget
        
        # Configure formatter
        self.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        
        # Configure level
        self.setLevel(logging.INFO)
        
        # Configure colors
        self.colors = {
            logging.DEBUG: 'gray',
            logging.INFO: 'black',
            logging.WARNING: 'orange',
            logging.ERROR: 'red',
            logging.CRITICAL: 'red'
        }
    
    def emit(self, record):
        """
        Emit a log record.
        
        Args:
            record: The log record
        """
        # Format the record
        msg = self.format(record)
        
        # Get color
        color = self.colors.get(record.levelno, 'black')
        
        # Insert into text widget
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.insert(tk.END, msg + '\n', color)
        self.text_widget.tag_config(color, foreground=color)
        self.text_widget.see(tk.END)
        self.text_widget.config(state=tk.DISABLED)


def main():
    """Main function."""
    # Create root window
    root = tk.Tk()
    
    # Create GUI
    app = HyperliquidMasterGUI(root)
    
    # Start main loop
    root.mainloop()


if __name__ == "__main__":
    main()
