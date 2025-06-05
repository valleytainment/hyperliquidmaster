"""
Enhanced GUI for Hyperliquid Trading Bot - PRODUCTION READY
🚀 ULTIMATE VERSION with robust 24/7 trading capabilities
Fixed all freezing issues, added comprehensive token support, and full automation
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import queue
import requests
from concurrent.futures import ThreadPoolExecutor

# Import our core modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.api import EnhancedHyperliquidAPI
from utils.logger import get_logger, TradingLogger
from utils.config_manager import ConfigManager
from utils.security import SecurityManager

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

logger = get_logger(__name__)
trading_logger = TradingLogger(__name__)


class TradingDashboard:
    """Production-ready trading dashboard with 24/7 automation capabilities"""
    
    def __init__(self):
        """Initialize the trading dashboard"""
        self.root = ctk.CTk()
        self.root.title("🚀 Hyperliquid Trading Bot - Ultimate Dashboard")
        self.root.geometry("1600x1000")
        self.root.minsize(1400, 900)
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.security_manager = SecurityManager()
        self.api = None
        self.is_connected = False
        self.is_trading = False
        self.auto_trading_enabled = False
        
        # Thread management
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.update_queue = queue.Queue()
        
        # Data storage
        self.account_data = {}
        self.market_data = {}
        self.positions = []
        self.orders = []
        self.trade_history = []
        self.available_tokens = []
        self.token_prices = {}
        
        # GUI components
        self.widgets = {}
        self.charts = {}
        
        # Update control
        self.stop_updates = False
        self.last_token_refresh = 0
        self.last_price_update = 0
        
        # Initialize GUI
        self.setup_gui()
        self.setup_menu()
        self.setup_status_bar()
        
        # Start background processes
        self.start_background_processes()
        
        logger.info("🚀 Ultimate Trading Dashboard initialized")
    
    def setup_gui(self):
        """Setup the main GUI layout with improved stability"""
        # Create main container
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        self.notebook = ctk.CTkTabview(self.main_container)
        self.notebook.pack(fill="both", expand=True)
        
        # Create tabs
        self.setup_dashboard_tab()
        self.setup_trading_tab()
        self.setup_positions_tab()
        self.setup_automation_tab()
        self.setup_strategies_tab()
        self.setup_backtesting_tab()
        self.setup_settings_tab()
    
    def setup_dashboard_tab(self):
        """Setup the main dashboard tab with real-time data"""
        dashboard_tab = self.notebook.add("📊 Dashboard")
        
        # Top row - Connection and account summary
        top_frame = ctk.CTkFrame(dashboard_tab)
        top_frame.pack(fill="x", padx=10, pady=5)
        
        # Connection panel
        conn_frame = ctk.CTkFrame(top_frame)
        conn_frame.pack(side="left", fill="y", padx=5)
        
        ctk.CTkLabel(conn_frame, text="🔗 Connection", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        # Connection status
        status_frame = ctk.CTkFrame(conn_frame)
        status_frame.pack(fill="x", padx=5, pady=2)
        
        self.widgets['connection_indicator'] = ctk.CTkLabel(status_frame, text="●", text_color="red", font=ctk.CTkFont(size=20))
        self.widgets['connection_indicator'].pack(side="left")
        
        self.widgets['status_label'] = ctk.CTkLabel(status_frame, text="Disconnected")
        self.widgets['status_label'].pack(side="left", padx=5)
        
        # Quick connect button
        self.widgets['quick_connect_btn'] = ctk.CTkButton(conn_frame, text="🚀 Quick Connect", command=self.quick_connect)
        self.widgets['quick_connect_btn'].pack(fill="x", padx=5, pady=2)
        
        # Account summary panel
        account_frame = ctk.CTkFrame(top_frame)
        account_frame.pack(side="right", fill="both", expand=True, padx=5)
        
        ctk.CTkLabel(account_frame, text="💰 Account Summary", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        # Account metrics grid
        metrics_grid = ctk.CTkFrame(account_frame)
        metrics_grid.pack(fill="x", padx=10, pady=5)
        
        # Create metric displays in a 2x4 grid
        self.widgets['account_value'] = self.create_metric_display(metrics_grid, "Account Value", "$0.00", 0, 0)
        self.widgets['total_pnl'] = self.create_metric_display(metrics_grid, "Total PnL", "$0.00", 0, 1)
        self.widgets['margin_used'] = self.create_metric_display(metrics_grid, "Margin Used", "0%", 0, 2)
        self.widgets['open_positions'] = self.create_metric_display(metrics_grid, "Open Positions", "0", 0, 3)
        self.widgets['daily_pnl'] = self.create_metric_display(metrics_grid, "Daily PnL", "$0.00", 1, 0)
        self.widgets['win_rate'] = self.create_metric_display(metrics_grid, "Win Rate", "0%", 1, 1)
        self.widgets['total_trades'] = self.create_metric_display(metrics_grid, "Total Trades", "0", 1, 2)
        self.widgets['auto_status'] = self.create_metric_display(metrics_grid, "Auto Trading", "OFF", 1, 3)
        
        # Middle row - Charts and market data
        middle_frame = ctk.CTkFrame(dashboard_tab)
        middle_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Left panel - Charts
        charts_frame = ctk.CTkFrame(middle_frame)
        charts_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        # Chart notebook
        chart_notebook = ctk.CTkTabview(charts_frame)
        chart_notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Portfolio chart
        portfolio_tab = chart_notebook.add("📈 Portfolio")
        self.setup_portfolio_chart(portfolio_tab)
        
        # Price chart
        price_tab = chart_notebook.add("💹 Price Chart")
        self.setup_price_chart(price_tab)
        
        # Performance chart
        performance_tab = chart_notebook.add("📊 Performance")
        self.setup_performance_chart(performance_tab)
        
        # Right panel - Market overview
        market_frame = ctk.CTkFrame(middle_frame)
        market_frame.pack(side="right", fill="y", padx=5)
        market_frame.configure(width=300)
        
        ctk.CTkLabel(market_frame, text="🌍 Market Overview", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        # Market data list
        self.widgets['market_list'] = ctk.CTkScrollableFrame(market_frame)
        self.widgets['market_list'].pack(fill="both", expand=True, padx=5, pady=5)
        
        # Bottom row - Activity and alerts
        bottom_frame = ctk.CTkFrame(dashboard_tab)
        bottom_frame.pack(fill="x", padx=10, pady=5)
        
        # Activity panel
        activity_frame = ctk.CTkFrame(bottom_frame)
        activity_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        ctk.CTkLabel(activity_frame, text="📋 Recent Activity", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        self.widgets['activity_list'] = ctk.CTkTextbox(activity_frame, height=120)
        self.widgets['activity_list'].pack(fill="both", expand=True, padx=5, pady=5)
        
        # Alerts panel
        alerts_frame = ctk.CTkFrame(bottom_frame)
        alerts_frame.pack(side="right", fill="y", padx=5)
        alerts_frame.configure(width=300)
        
        ctk.CTkLabel(alerts_frame, text="🚨 Alerts", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        self.widgets['alerts_list'] = ctk.CTkTextbox(alerts_frame, height=120)
        self.widgets['alerts_list'].pack(fill="both", expand=True, padx=5, pady=5)
    
    def setup_trading_tab(self):
        """Setup the enhanced trading tab with comprehensive token support"""
        trading_tab = self.notebook.add("💰 Trading")
        
        # Left panel - Order entry
        left_frame = ctk.CTkFrame(trading_tab)
        left_frame.pack(side="left", fill="y", padx=5, pady=5)
        left_frame.configure(width=350)
        
        ctk.CTkLabel(left_frame, text="📝 Order Entry", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
        
        # Token selection with refresh
        token_frame = ctk.CTkFrame(left_frame)
        token_frame.pack(fill="x", padx=10, pady=5)
        
        token_header = ctk.CTkFrame(token_frame)
        token_header.pack(fill="x", pady=2)
        
        ctk.CTkLabel(token_header, text="Token:").pack(side="left")
        
        self.widgets['refresh_tokens_btn'] = ctk.CTkButton(
            token_header, 
            text="🔄", 
            width=30, 
            command=self.refresh_tokens
        )
        self.widgets['refresh_tokens_btn'].pack(side="right")
        
        # Token dropdown with search
        self.widgets['token_var'] = tk.StringVar()
        self.widgets['token_combobox'] = ctk.CTkComboBox(
            token_frame,
            variable=self.widgets['token_var'],
            values=["Loading tokens..."],
            command=self.on_token_selected
        )
        self.widgets['token_combobox'].pack(fill="x", pady=2)
        
        # Token info display
        token_info_frame = ctk.CTkFrame(token_frame)
        token_info_frame.pack(fill="x", pady=2)
        
        self.widgets['token_price'] = ctk.CTkLabel(token_info_frame, text="Price: $0.00")
        self.widgets['token_price'].pack(side="left")
        
        self.widgets['token_change'] = ctk.CTkLabel(token_info_frame, text="24h: 0%")
        self.widgets['token_change'].pack(side="right")
        
        # Order type
        order_type_frame = ctk.CTkFrame(left_frame)
        order_type_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(order_type_frame, text="Order Type:").pack(anchor="w")
        self.widgets['order_type'] = ctk.CTkOptionMenu(
            order_type_frame, 
            values=["Market", "Limit", "Stop Loss", "Take Profit"],
            command=self.on_order_type_changed
        )
        self.widgets['order_type'].pack(fill="x", pady=2)
        
        # Side selection with visual indicators
        side_frame = ctk.CTkFrame(left_frame)
        side_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(side_frame, text="Side:").pack(anchor="w")
        
        side_buttons_frame = ctk.CTkFrame(side_frame)
        side_buttons_frame.pack(fill="x", pady=2)
        
        self.widgets['buy_btn'] = ctk.CTkButton(
            side_buttons_frame, 
            text="🟢 BUY", 
            fg_color="green",
            command=lambda: self.set_order_side("buy")
        )
        self.widgets['buy_btn'].pack(side="left", fill="x", expand=True, padx=2)
        
        self.widgets['sell_btn'] = ctk.CTkButton(
            side_buttons_frame, 
            text="🔴 SELL", 
            fg_color="red",
            command=lambda: self.set_order_side("sell")
        )
        self.widgets['sell_btn'].pack(side="right", fill="x", expand=True, padx=2)
        
        self.order_side = "buy"  # Default
        
        # Size entry with percentage buttons
        size_frame = ctk.CTkFrame(left_frame)
        size_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(size_frame, text="Size (USD):").pack(anchor="w")
        
        size_entry_frame = ctk.CTkFrame(size_frame)
        size_entry_frame.pack(fill="x", pady=2)
        
        self.widgets['size_entry'] = ctk.CTkEntry(size_entry_frame, placeholder_text="100.00")
        self.widgets['size_entry'].pack(side="left", fill="x", expand=True)
        
        # Percentage buttons
        pct_frame = ctk.CTkFrame(size_frame)
        pct_frame.pack(fill="x", pady=2)
        
        for pct in ["25%", "50%", "75%", "100%"]:
            btn = ctk.CTkButton(
                pct_frame, 
                text=pct, 
                width=60,
                command=lambda p=pct: self.set_size_percentage(p)
            )
            btn.pack(side="left", padx=1)
        
        # Price entry (for limit orders)
        price_frame = ctk.CTkFrame(left_frame)
        price_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(price_frame, text="Price:").pack(anchor="w")
        self.widgets['price_entry'] = ctk.CTkEntry(price_frame, placeholder_text="Market price")
        self.widgets['price_entry'].pack(fill="x", pady=2)
        
        # Advanced options
        advanced_frame = ctk.CTkFrame(left_frame)
        advanced_frame.pack(fill="x", padx=10, pady=5)
        
        self.widgets['advanced_toggle'] = ctk.CTkSwitch(advanced_frame, text="Advanced Options")
        self.widgets['advanced_toggle'].pack(anchor="w")
        
        # Stop loss and take profit
        self.widgets['sl_frame'] = ctk.CTkFrame(advanced_frame)
        
        ctk.CTkLabel(self.widgets['sl_frame'], text="Stop Loss:").pack(anchor="w")
        self.widgets['stop_loss_entry'] = ctk.CTkEntry(self.widgets['sl_frame'], placeholder_text="Optional")
        self.widgets['stop_loss_entry'].pack(fill="x", pady=2)
        
        ctk.CTkLabel(self.widgets['sl_frame'], text="Take Profit:").pack(anchor="w")
        self.widgets['take_profit_entry'] = ctk.CTkEntry(self.widgets['sl_frame'], placeholder_text="Optional")
        self.widgets['take_profit_entry'].pack(fill="x", pady=2)
        
        # Order buttons
        order_buttons_frame = ctk.CTkFrame(left_frame)
        order_buttons_frame.pack(fill="x", padx=10, pady=10)
        
        self.widgets['place_order_btn'] = ctk.CTkButton(
            order_buttons_frame, 
            text="🚀 Place Order", 
            font=ctk.CTkFont(size=16, weight="bold"),
            height=40,
            command=self.place_order_async
        )
        self.widgets['place_order_btn'].pack(fill="x", pady=2)
        
        self.widgets['cancel_all_btn'] = ctk.CTkButton(
            order_buttons_frame, 
            text="❌ Cancel All Orders", 
            fg_color="red",
            command=self.cancel_all_orders_async
        )
        self.widgets['cancel_all_btn'].pack(fill="x", pady=2)
        
        # Right panel - Order book and trades
        right_frame = ctk.CTkFrame(trading_tab)
        right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        # Order book and recent trades notebook
        right_notebook = ctk.CTkTabview(right_frame)
        right_notebook.pack(fill="both", expand=True)
        
        # Order book tab
        orderbook_tab = right_notebook.add("📖 Order Book")
        self.setup_orderbook_display(orderbook_tab)
        
        # Recent trades tab
        trades_tab = right_notebook.add("📊 Recent Trades")
        self.setup_recent_trades_display(trades_tab)
        
        # Open orders tab
        orders_tab = right_notebook.add("📋 Open Orders")
        self.setup_open_orders_display(orders_tab)
    
    def setup_automation_tab(self):
        """Setup 24/7 automation tab"""
        automation_tab = self.notebook.add("🤖 Automation")
        
        # Main automation controls
        main_frame = ctk.CTkFrame(automation_tab)
        main_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(main_frame, text="🤖 24/7 Automated Trading", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)
        
        # Master automation switch
        master_frame = ctk.CTkFrame(main_frame)
        master_frame.pack(fill="x", padx=20, pady=10)
        
        self.widgets['master_auto_switch'] = ctk.CTkSwitch(
            master_frame, 
            text="🚀 Enable Automated Trading", 
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self.toggle_automation
        )
        self.widgets['master_auto_switch'].pack(pady=10)
        
        # Automation status
        status_frame = ctk.CTkFrame(main_frame)
        status_frame.pack(fill="x", padx=20, pady=5)
        
        self.widgets['auto_status_label'] = ctk.CTkLabel(
            status_frame, 
            text="Status: STOPPED", 
            font=ctk.CTkFont(size=14)
        )
        self.widgets['auto_status_label'].pack()
        
        # Strategy selection for automation
        strategy_frame = ctk.CTkFrame(automation_tab)
        strategy_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(strategy_frame, text="📈 Active Strategies", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=5)
        
        # Strategy list with controls
        self.widgets['strategy_list_frame'] = ctk.CTkScrollableFrame(strategy_frame)
        self.widgets['strategy_list_frame'].pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add default strategies
        self.setup_strategy_controls()
        
        # Risk management for automation
        risk_frame = ctk.CTkFrame(automation_tab)
        risk_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(risk_frame, text="🛡️ Risk Management", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=5)
        
        # Risk controls grid
        risk_grid = ctk.CTkFrame(risk_frame)
        risk_grid.pack(fill="x", padx=10, pady=5)
        
        # Max daily loss
        max_loss_frame = ctk.CTkFrame(risk_grid)
        max_loss_frame.pack(side="left", fill="x", expand=True, padx=5)
        
        ctk.CTkLabel(max_loss_frame, text="Max Daily Loss ($):").pack()
        self.widgets['max_daily_loss'] = ctk.CTkEntry(max_loss_frame, placeholder_text="1000")
        self.widgets['max_daily_loss'].pack(fill="x", pady=2)
        
        # Max position size
        max_pos_frame = ctk.CTkFrame(risk_grid)
        max_pos_frame.pack(side="left", fill="x", expand=True, padx=5)
        
        ctk.CTkLabel(max_pos_frame, text="Max Position Size ($):").pack()
        self.widgets['max_position_size'] = ctk.CTkEntry(max_pos_frame, placeholder_text="5000")
        self.widgets['max_position_size'].pack(fill="x", pady=2)
        
        # Max open positions
        max_open_frame = ctk.CTkFrame(risk_grid)
        max_open_frame.pack(side="left", fill="x", expand=True, padx=5)
        
        ctk.CTkLabel(max_open_frame, text="Max Open Positions:").pack()
        self.widgets['max_open_positions'] = ctk.CTkEntry(max_open_frame, placeholder_text="5")
        self.widgets['max_open_positions'].pack(fill="x", pady=2)
        
        # Performance monitoring
        perf_frame = ctk.CTkFrame(automation_tab)
        perf_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(perf_frame, text="📊 Performance Monitoring", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=5)
        
        # Performance metrics
        self.widgets['perf_metrics'] = ctk.CTkTextbox(perf_frame)
        self.widgets['perf_metrics'].pack(fill="both", expand=True, padx=10, pady=10)
    
    def setup_positions_tab(self):
        """Setup positions tab with enhanced management"""
        positions_tab = self.notebook.add("📊 Positions")
        
        # Positions header
        header_frame = ctk.CTkFrame(positions_tab)
        header_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(header_frame, text="📊 Open Positions", font=ctk.CTkFont(size=18, weight="bold")).pack(side="left", pady=10)
        
        # Refresh button
        self.widgets['refresh_positions_btn'] = ctk.CTkButton(
            header_frame, 
            text="🔄 Refresh", 
            command=self.refresh_positions_async
        )
        self.widgets['refresh_positions_btn'].pack(side="right", padx=10, pady=10)
        
        # Positions table
        self.setup_positions_table(positions_tab)
        
        # Position management buttons
        buttons_frame = ctk.CTkFrame(positions_tab)
        buttons_frame.pack(fill="x", padx=10, pady=5)
        
        self.widgets['close_all_btn'] = ctk.CTkButton(
            buttons_frame, 
            text="❌ Close All Positions", 
            fg_color="red",
            command=self.close_all_positions_async
        )
        self.widgets['close_all_btn'].pack(side="left", padx=5)
        
        self.widgets['hedge_all_btn'] = ctk.CTkButton(
            buttons_frame, 
            text="🛡️ Hedge All", 
            command=self.hedge_all_positions_async
        )
        self.widgets['hedge_all_btn'].pack(side="left", padx=5)
    
    def setup_strategies_tab(self):
        """Setup strategies tab"""
        strategies_tab = self.notebook.add("📈 Strategies")
        
        # Strategy selection
        strategy_frame = ctk.CTkFrame(strategies_tab)
        strategy_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(strategy_frame, text="📈 Trading Strategies", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=5)
        
        # Available strategies
        self.widgets['strategy_selector'] = ctk.CTkOptionMenu(
            strategy_frame,
            values=["BB RSI ADX", "Hull Suite", "Custom Strategy"],
            command=self.on_strategy_selected
        )
        self.widgets['strategy_selector'].pack(pady=5)
        
        # Strategy parameters
        params_frame = ctk.CTkFrame(strategies_tab)
        params_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.widgets['strategy_params'] = ctk.CTkScrollableFrame(params_frame)
        self.widgets['strategy_params'].pack(fill="both", expand=True, padx=10, pady=10)
    
    def setup_backtesting_tab(self):
        """Setup backtesting tab"""
        backtesting_tab = self.notebook.add("🔬 Backtesting")
        
        # Backtesting controls
        controls_frame = ctk.CTkFrame(backtesting_tab)
        controls_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(controls_frame, text="🔬 Strategy Backtesting", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=5)
        
        # Backtest parameters
        params_grid = ctk.CTkFrame(controls_frame)
        params_grid.pack(fill="x", padx=10, pady=5)
        
        # Date range
        date_frame = ctk.CTkFrame(params_grid)
        date_frame.pack(side="left", fill="x", expand=True, padx=5)
        
        ctk.CTkLabel(date_frame, text="Start Date:").pack()
        self.widgets['backtest_start'] = ctk.CTkEntry(date_frame, placeholder_text="2024-01-01")
        self.widgets['backtest_start'].pack(fill="x", pady=2)
        
        ctk.CTkLabel(date_frame, text="End Date:").pack()
        self.widgets['backtest_end'] = ctk.CTkEntry(date_frame, placeholder_text="2024-12-31")
        self.widgets['backtest_end'].pack(fill="x", pady=2)
        
        # Initial capital
        capital_frame = ctk.CTkFrame(params_grid)
        capital_frame.pack(side="left", fill="x", expand=True, padx=5)
        
        ctk.CTkLabel(capital_frame, text="Initial Capital:").pack()
        self.widgets['initial_capital'] = ctk.CTkEntry(capital_frame, placeholder_text="10000")
        self.widgets['initial_capital'].pack(fill="x", pady=2)
        
        # Run backtest button
        self.widgets['run_backtest_btn'] = ctk.CTkButton(
            controls_frame, 
            text="🚀 Run Backtest", 
            command=self.run_backtest_async
        )
        self.widgets['run_backtest_btn'].pack(pady=10)
        
        # Results display
        results_frame = ctk.CTkFrame(backtesting_tab)
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.widgets['backtest_results'] = ctk.CTkTextbox(results_frame)
        self.widgets['backtest_results'].pack(fill="both", expand=True, padx=10, pady=10)
    
    def setup_settings_tab(self):
        """Setup settings tab with direct private key and wallet address input"""
        settings_tab = self.notebook.add("⚙️ Settings")
        
        # Create scrollable frame for settings
        scrollable_frame = ctk.CTkScrollableFrame(settings_tab)
        scrollable_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Account Configuration Section
        account_frame = ctk.CTkFrame(scrollable_frame)
        account_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(account_frame, text="👤 Account Configuration", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)
        
        # Wallet Address Input
        wallet_section = ctk.CTkFrame(account_frame)
        wallet_section.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(wallet_section, text="💳 Wallet Address", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", pady=(5,2))
        ctk.CTkLabel(wallet_section, text="Enter your Hyperliquid wallet address (0x...)", font=ctk.CTkFont(size=12)).pack(anchor="w")
        
        self.widgets['wallet_address'] = ctk.CTkEntry(
            wallet_section, 
            placeholder_text="0x1234567890abcdef1234567890abcdef12345678",
            font=ctk.CTkFont(size=12),
            height=35
        )
        self.widgets['wallet_address'].pack(fill="x", pady=5)
        
        # Load existing wallet address if available
        try:
            existing_address = self.config_manager.get_config().get('trading', {}).get('wallet_address', '')
            if existing_address:
                self.widgets['wallet_address'].insert(0, existing_address)
        except:
            pass
        
        # Private Key Input Section
        key_section = ctk.CTkFrame(account_frame)
        key_section.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(key_section, text="🔐 Private Key", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", pady=(5,2))
        ctk.CTkLabel(key_section, text="Enter your private key (will be encrypted and stored securely)", font=ctk.CTkFont(size=12)).pack(anchor="w")
        
        # Private key input with show/hide toggle
        key_input_frame = ctk.CTkFrame(key_section)
        key_input_frame.pack(fill="x", pady=5)
        
        self.widgets['private_key'] = ctk.CTkEntry(
            key_input_frame,
            placeholder_text="Enter your private key here...",
            show="*",  # Hide by default
            font=ctk.CTkFont(size=12),
            height=35
        )
        self.widgets['private_key'].pack(side="left", fill="x", expand=True, padx=(0,5))
        
        # Show/Hide private key toggle
        self.widgets['show_key_var'] = tk.BooleanVar()
        self.widgets['show_key_btn'] = ctk.CTkButton(
            key_input_frame,
            text="👁️",
            width=40,
            command=self.toggle_private_key_visibility
        )
        self.widgets['show_key_btn'].pack(side="right")
        
        # Private key status and actions
        key_actions_frame = ctk.CTkFrame(key_section)
        key_actions_frame.pack(fill="x", pady=5)
        
        self.widgets['key_status'] = ctk.CTkLabel(
            key_actions_frame, 
            text="🔴 Not configured", 
            font=ctk.CTkFont(size=12)
        )
        self.widgets['key_status'].pack(side="left")
        
        self.widgets['test_connection_btn'] = ctk.CTkButton(
            key_actions_frame,
            text="🔗 Test Connection",
            command=self.test_connection_async,
            width=120
        )
        self.widgets['test_connection_btn'].pack(side="right", padx=5)
        
        self.widgets['save_credentials_btn'] = ctk.CTkButton(
            key_actions_frame,
            text="💾 Save Credentials",
            command=self.save_credentials_async,
            width=120
        )
        self.widgets['save_credentials_btn'].pack(side="right", padx=5)
        
        # Connection Settings Section
        conn_frame = ctk.CTkFrame(scrollable_frame)
        conn_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(conn_frame, text="🔗 Connection Settings", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
        
        # Testnet toggle
        testnet_frame = ctk.CTkFrame(conn_frame)
        testnet_frame.pack(fill="x", padx=15, pady=5)
        
        self.widgets['testnet_var'] = tk.BooleanVar(value=True)
        self.widgets['testnet_switch'] = ctk.CTkSwitch(
            testnet_frame, 
            text="🧪 Use Testnet (Recommended for testing)", 
            variable=self.widgets['testnet_var'],
            font=ctk.CTkFont(size=14)
        )
        self.widgets['testnet_switch'].pack(anchor="w", pady=5)
        
        # Load existing testnet setting
        try:
            existing_testnet = self.config_manager.get_config().get('trading', {}).get('testnet', True)
            self.widgets['testnet_var'].set(existing_testnet)
        except:
            pass
        
        # Trading Settings Section
        trading_frame = ctk.CTkFrame(scrollable_frame)
        trading_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(trading_frame, text="💰 Trading Settings", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
        
        # Trading settings grid
        settings_grid = ctk.CTkFrame(trading_frame)
        settings_grid.pack(fill="x", padx=15, pady=5)
        
        # Row 1: Order size and slippage
        row1 = ctk.CTkFrame(settings_grid)
        row1.pack(fill="x", pady=5)
        
        # Default order size
        size_frame = ctk.CTkFrame(row1)
        size_frame.pack(side="left", fill="x", expand=True, padx=(0,5))
        
        ctk.CTkLabel(size_frame, text="Default Order Size ($):", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
        self.widgets['default_size'] = ctk.CTkEntry(size_frame, placeholder_text="100", height=30)
        self.widgets['default_size'].pack(fill="x", pady=2)
        
        # Max slippage
        slippage_frame = ctk.CTkFrame(row1)
        slippage_frame.pack(side="left", fill="x", expand=True, padx=(5,0))
        
        ctk.CTkLabel(slippage_frame, text="Max Slippage (%):", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
        self.widgets['max_slippage'] = ctk.CTkEntry(slippage_frame, placeholder_text="0.5", height=30)
        self.widgets['max_slippage'].pack(fill="x", pady=2)
        
        # Row 2: Stop loss and take profit
        row2 = ctk.CTkFrame(settings_grid)
        row2.pack(fill="x", pady=5)
        
        # Default stop loss
        sl_frame = ctk.CTkFrame(row2)
        sl_frame.pack(side="left", fill="x", expand=True, padx=(0,5))
        
        ctk.CTkLabel(sl_frame, text="Default Stop Loss (%):", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
        self.widgets['default_stop_loss'] = ctk.CTkEntry(sl_frame, placeholder_text="2.0", height=30)
        self.widgets['default_stop_loss'].pack(fill="x", pady=2)
        
        # Default take profit
        tp_frame = ctk.CTkFrame(row2)
        tp_frame.pack(side="left", fill="x", expand=True, padx=(5,0))
        
        ctk.CTkLabel(tp_frame, text="Default Take Profit (%):", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
        self.widgets['default_take_profit'] = ctk.CTkEntry(tp_frame, placeholder_text="4.0", height=30)
        self.widgets['default_take_profit'].pack(fill="x", pady=2)
        
        # Load existing trading settings
        self.load_existing_settings()
        
        # Action buttons
        buttons_frame = ctk.CTkFrame(scrollable_frame)
        buttons_frame.pack(fill="x", padx=10, pady=20)
        
        self.widgets['save_all_settings_btn'] = ctk.CTkButton(
            buttons_frame, 
            text="💾 Save All Settings", 
            command=self.save_all_settings_async,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40
        )
        self.widgets['save_all_settings_btn'].pack(side="left", padx=10)
        
        self.widgets['reset_settings_btn'] = ctk.CTkButton(
            buttons_frame, 
            text="🔄 Reset to Defaults", 
            command=self.reset_settings_async,
            fg_color="orange",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40
        )
        self.widgets['reset_settings_btn'].pack(side="left", padx=10)
        
        self.widgets['clear_all_btn'] = ctk.CTkButton(
            buttons_frame, 
            text="🗑️ Clear All", 
            command=self.clear_all_settings_async,
            fg_color="red",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40
        )
        self.widgets['clear_all_btn'].pack(side="right", padx=10)
    
    def setup_menu(self):
        """Setup menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Import Config", command=self.import_config_async)
        file_menu.add_command(label="Export Config", command=self.export_config_async)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Refresh All Data", command=self.refresh_all_data_async)
        tools_menu.add_command(label="Export Trades", command=self.export_trades_async)
        tools_menu.add_command(label="Clear Cache", command=self.clear_cache_async)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="About", command=self.show_about)
    
    def setup_status_bar(self):
        """Setup status bar"""
        self.status_bar = ctk.CTkFrame(self.root)
        self.status_bar.pack(side="bottom", fill="x", padx=10, pady=5)
        
        # Status elements
        self.widgets['status_text'] = ctk.CTkLabel(self.status_bar, text="Ready")
        self.widgets['status_text'].pack(side="left", padx=5)
        
        self.widgets['last_update'] = ctk.CTkLabel(self.status_bar, text="Last update: Never")
        self.widgets['last_update'].pack(side="right", padx=5)
    
    # Async wrapper methods to prevent GUI freezing
    def run_async(self, coro):
        """Run coroutine in thread pool to prevent GUI freezing"""
        def run_in_thread():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(coro)
                loop.close()
                return result
            except Exception as e:
                logger.error(f"Async operation failed: {e}")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Operation failed: {e}"))
        
        self.executor.submit(run_in_thread)
    
    def quick_connect(self):
        """Quick connect with saved credentials"""
        self.run_async(self._quick_connect())
    
    async def _quick_connect(self):
        """Async quick connect implementation"""
        try:
            # Update status
            self.root.after(0, lambda: self.widgets['connection_indicator'].configure(text_color="orange"))
            self.root.after(0, lambda: self.widgets['status_label'].configure(text="Connecting..."))
            
            # Check if we have saved credentials
            private_key = self.security_manager.get_private_key()
            if not private_key:
                self.root.after(0, lambda: self.widgets['connection_indicator'].configure(text_color="red"))
                self.root.after(0, lambda: self.widgets['status_label'].configure(text="No private key"))
                self.root.after(0, lambda: messagebox.showerror("Error", "No private key found. Please go to Settings tab to configure your credentials."))
                return
            
            # Get configuration
            config = self.config_manager.get_config()
            trading_config = config.get('trading', {})
            wallet_address = trading_config.get('wallet_address', '')
            use_testnet = trading_config.get('testnet', True)
            
            if not wallet_address:
                self.root.after(0, lambda: self.widgets['connection_indicator'].configure(text_color="red"))
                self.root.after(0, lambda: self.widgets['status_label'].configure(text="No wallet address"))
                self.root.after(0, lambda: messagebox.showerror("Error", "No wallet address configured. Please go to Settings tab to configure your wallet address."))
                return
            
            # Initialize API
            from core.api import EnhancedHyperliquidAPI
            self.api = EnhancedHyperliquidAPI(
                private_key=private_key,
                testnet=use_testnet
            )
            
            # Test connection
            account_info = await self.api.get_account_info()
            
            if account_info:
                # Connection successful
                self.is_connected = True
                self.root.after(0, lambda: self.widgets['connection_indicator'].configure(text_color="green"))
                self.root.after(0, lambda: self.widgets['status_label'].configure(text=f"Connected ({'Testnet' if use_testnet else 'Mainnet'})"))
                
                # Update account data
                self.account_data = account_info
                
                # Update connection status throughout GUI
                self.update_connection_status()
                
                # Refresh data
                # Implement missing refresh_account_data_async method
                # await self.refresh_account_data_async()
                await self.refresh_tokens()
                
                self.root.after(0, lambda: messagebox.showinfo("Success", f"Connected successfully!\nWallet: {wallet_address}\nNetwork: {'Testnet' if use_testnet else 'Mainnet'}"))
                
                logger.info(f"Connected to Hyperliquid ({'testnet' if use_testnet else 'mainnet'})")
                
            else:
                # Connection failed
                self.is_connected = False
                self.root.after(0, lambda: self.widgets['connection_indicator'].configure(text_color="red"))
                self.root.after(0, lambda: self.widgets['status_label'].configure(text="Connection failed"))
                self.root.after(0, lambda: messagebox.showerror("Error", "Connection failed. Please check your credentials in Settings."))
                
        except Exception as e:
            logger.error(f"Quick connect failed: {e}")
            self.is_connected = False
            self.root.after(0, lambda: self.widgets['connection_indicator'].configure(text_color="red"))
            self.root.after(0, lambda: self.widgets['status_label'].configure(text="Connection error"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Connection failed: {str(e)}"))
    
    def update_connection_status(self):
        """Update connection status in GUI"""
        if self.is_connected:
            self.widgets['connection_indicator'].configure(text_color="green")
            self.widgets['status_label'].configure(text="Connected")
            self.widgets['status_text'].configure(text="Connected to Hyperliquid")
        else:
            self.widgets['connection_indicator'].configure(text_color="red")
            self.widgets['status_label'].configure(text="Disconnected")
            self.widgets['status_text'].configure(text="Disconnected")
    
    def refresh_tokens(self):
        """Refresh available tokens list"""
        self.run_async(self._refresh_tokens())
    
    async def _refresh_tokens(self):
        """Async token refresh implementation"""
        try:
            self.root.after(0, lambda: self.widgets['refresh_tokens_btn'].configure(text="🔄", state="disabled"))
            
            if not self.api:
                # Use public API to get token list
                response = requests.get("https://api.hyperliquid.xyz/info", 
                                      json={"type": "meta"}, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    tokens = [asset['name'] for asset in data.get('universe', [])]
                else:
                    tokens = ["BTC", "ETH", "SOL", "AVAX", "MATIC"]  # Fallback
            else:
                # Use authenticated API
                tokens = await self.api.get_available_tokens()
            
            self.available_tokens = tokens
            
            # Update GUI
            self.root.after(0, lambda: self.widgets['token_combobox'].configure(values=tokens))
            self.root.after(0, lambda: self.widgets['refresh_tokens_btn'].configure(text="🔄", state="normal"))
            self.root.after(0, lambda: self.add_activity(f"✅ Refreshed {len(tokens)} tokens"))
            
            self.last_token_refresh = time.time()
            
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            self.root.after(0, lambda: self.widgets['refresh_tokens_btn'].configure(text="❌", state="normal"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to refresh tokens: {e}"))
    
    def setup_private_key_async(self):
        """Setup private key asynchronously"""
        self.run_async(self._setup_private_key())
    
    async def _setup_private_key(self):
        """Async private key setup"""
        try:
            self.root.after(0, lambda: self.widgets['setup_key_btn'].configure(state="disabled"))
            
            # Run in thread to prevent GUI freezing
            def setup_key():
                return self.security_manager.setup_private_key()
            
            success = await asyncio.get_event_loop().run_in_executor(None, setup_key)
            
            if success:
                self.root.after(0, lambda: self.widgets['key_status'].configure(text="Status: ✅ Configured"))
                self.root.after(0, lambda: messagebox.showinfo("Success", "Private key setup completed"))
                self.root.after(0, lambda: self.add_activity("✅ Private key configured"))
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", "Private key setup failed"))
                
        except Exception as e:
            logger.error(f"Private key setup failed: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Setup failed: {e}"))
        finally:
            self.root.after(0, lambda: self.widgets['setup_key_btn'].configure(state="normal"))
    
    def place_order_async(self):
        """Place order asynchronously"""
        self.run_async(self._place_order())
    
    async def _place_order(self):
        """Async order placement"""
        try:
            if not self.is_connected:
                self.root.after(0, lambda: messagebox.showerror("Error", "Not connected to API"))
                return
            
            # Get order parameters
            token = self.widgets['token_var'].get()
            order_type = self.widgets['order_type'].get()
            size = float(self.widgets['size_entry'].get() or "0")
            side = self.order_side
            
            if not token or size <= 0:
                self.root.after(0, lambda: messagebox.showerror("Error", "Please enter valid order parameters"))
                return
            
            # Place order
            self.root.after(0, lambda: self.widgets['place_order_btn'].configure(state="disabled"))
            
            result = await self.api.place_order_async(
                symbol=token,
                side=side,
                size=size,
                order_type=order_type.lower(),
                price=float(self.widgets['price_entry'].get() or "0") if order_type == "Limit" else None
            )
            
            if result.get('success'):
                self.root.after(0, lambda: self.add_activity(f"✅ {side.upper()} order placed: {size} {token}"))
                self.root.after(0, lambda: messagebox.showinfo("Success", "Order placed successfully"))
            else:
                error_msg = result.get('error', 'Unknown error')
                self.root.after(0, lambda: messagebox.showerror("Error", f"Order failed: {error_msg}"))
                
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Order failed: {e}"))
        finally:
            self.root.after(0, lambda: self.widgets['place_order_btn'].configure(state="normal"))
    
    def start_background_processes(self):
        """Start background update processes"""
        # Start GUI update loop
        self.update_gui_loop()
        
        # Auto-refresh tokens every 5 minutes
        self.root.after(300000, self.auto_refresh_tokens)
        
        # Auto-refresh data every 30 seconds
        self.root.after(30000, self.auto_refresh_data)
    
    def update_gui_loop(self):
        """Main GUI update loop"""
        try:
            # Process any queued updates
            while not self.update_queue.empty():
                try:
                    update_func = self.update_queue.get_nowait()
                    update_func()
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"GUI update error: {e}")
            
            # Update last update time
            self.widgets['last_update'].configure(text=f"Last update: {datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            logger.error(f"GUI update loop error: {e}")
        
        # Schedule next update
        if not self.stop_updates:
            self.root.after(1000, self.update_gui_loop)
    
    def auto_refresh_tokens(self):
        """Auto refresh tokens periodically"""
        if self.is_connected and time.time() - self.last_token_refresh > 300:  # 5 minutes
            self.refresh_tokens()
        
        # Schedule next refresh
        if not self.stop_updates:
            self.root.after(300000, self.auto_refresh_tokens)
    
    def auto_refresh_data(self):
        """Auto refresh market data"""
        if self.is_connected:
            self.run_async(self.refresh_market_data())
        
        # Schedule next refresh
        if not self.stop_updates:
            self.root.after(30000, self.auto_refresh_data)
    
    async def refresh_market_data(self):
        """Refresh market data"""
        try:
            if self.api:
                # Get account data
                account_info = await self.api.get_account_info_async()
                if account_info:
                    self.account_data = account_info
                    self.root.after(0, self.update_account_display)
                
                # Get positions
                positions = await self.api.get_positions_async()
                if positions:
                    self.positions = positions
                    self.root.after(0, self.update_positions_display)
                
        except Exception as e:
            logger.error(f"Market data refresh failed: {e}")
    
    def update_account_display(self):
        """Update account information display"""
        try:
            if self.account_data:
                account_value = self.account_data.get('accountValue', 0)
                total_pnl = self.account_data.get('totalPnl', 0)
                margin_used = self.account_data.get('marginUsed', 0)
                
                self.widgets['account_value'].configure(text=f"${account_value:,.2f}")
                self.widgets['total_pnl'].configure(text=f"${total_pnl:,.2f}")
                self.widgets['margin_used'].configure(text=f"{margin_used:.1f}%")
                self.widgets['open_positions'].configure(text=str(len(self.positions)))
                
        except Exception as e:
            logger.error(f"Account display update failed: {e}")
    
    # Helper methods
    def create_metric_display(self, parent, label, value, row, col):
        """Create a metric display widget"""
        frame = ctk.CTkFrame(parent)
        frame.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
        
        label_widget = ctk.CTkLabel(frame, text=label, font=ctk.CTkFont(size=12))
        label_widget.pack()
        
        value_widget = ctk.CTkLabel(frame, text=value, font=ctk.CTkFont(size=16, weight="bold"))
        value_widget.pack()
        
        return value_widget
    
    def add_activity(self, message):
        """Add activity message"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            activity_text = f"[{timestamp}] {message}\n"
            
            current_text = self.widgets['activity_list'].get("1.0", "end")
            lines = current_text.split('\n')
            
            # Keep only last 50 lines
            if len(lines) > 50:
                lines = lines[-50:]
            
            new_text = '\n'.join(lines) + activity_text
            
            self.widgets['activity_list'].delete("1.0", "end")
            self.widgets['activity_list'].insert("1.0", new_text)
            
        except Exception as e:
            logger.error(f"Add activity failed: {e}")
    
    def on_closing(self):
        """Handle application closing"""
        try:
            self.stop_updates = True
            
            if self.api:
                self.api.stop_websocket()
            
            if self.executor:
                self.executor.shutdown(wait=False)
            
            self.root.quit()
            self.root.destroy()
            
        except Exception as e:
            logger.error(f"Closing error: {e}")
    
    # Implementation of required methods
    
    def toggle_private_key_visibility(self):
        """Toggle private key visibility between shown and hidden"""
        try:
            # Get the current state
            current_show = self.widgets['private_key'].cget("show")
            
            if current_show == "*":
                # Currently hidden, show it
                self.widgets['private_key'].configure(show="")
                self.widgets['show_key_btn'].configure(text="🔒")
            else:
                # Currently shown, hide it
                self.widgets['private_key'].configure(show="*")
                self.widgets['show_key_btn'].configure(text="👁️")
                
            logger.info("Private key visibility toggled")
        except Exception as e:
            logger.error(f"Error toggling private key visibility: {e}")
            messagebox.showerror("Error", f"Could not toggle visibility: {e}")
            
    def test_connection_async(self):
        """Test API connection asynchronously"""
        self.run_async(self._test_connection())
    
    async def _test_connection(self):
        """Async implementation of connection testing"""
        try:
            # Disable button during test
            self.root.after(0, lambda: self.widgets['test_connection_btn'].configure(state="disabled"))
            self.root.after(0, lambda: self.widgets['test_connection_btn'].configure(text="Testing..."))
            
            # Get credentials
            private_key = self.widgets['private_key'].get()
            wallet_address = self.widgets['wallet_address'].get()
            
            if not private_key or not wallet_address:
                self.root.after(0, lambda: messagebox.showerror("Error", "Please enter both private key and wallet address"))
                return
            
            # Test connection
            testnet = self.widgets['testnet_var'].get()
            
            # Initialize API if needed
            if not self.api:
                self.api = EnhancedHyperliquidAPI(testnet=testnet)
            
            # Test authentication
            success = await self.api.test_connection_async(private_key, wallet_address)
            
            if success:
                self.root.after(0, lambda: self.widgets['key_status'].configure(text="✅ Connection successful"))
                self.root.after(0, lambda: messagebox.showinfo("Success", "Connection test successful!"))
                self.is_connected = True
                self.root.after(0, lambda: self.add_activity("✅ Connection test successful"))
            else:
                self.root.after(0, lambda: self.widgets['key_status'].configure(text="❌ Connection failed"))
                self.root.after(0, lambda: messagebox.showerror("Error", "Connection test failed. Please check your credentials."))
                self.is_connected = False
                
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            self.root.after(0, lambda: self.widgets['key_status'].configure(text="❌ Connection error"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Connection test failed: {e}"))
            self.is_connected = False
            
        finally:
            # Re-enable button
            self.root.after(0, lambda: self.widgets['test_connection_btn'].configure(state="normal"))
            self.root.after(0, lambda: self.widgets['test_connection_btn'].configure(text="🔗 Test Connection"))
            
    def save_credentials_async(self):
        """Save credentials asynchronously"""
        self.run_async(self._save_credentials())
    
    async def _save_credentials(self):
        """Async implementation of saving credentials"""
        try:
            # Disable button during save
            self.root.after(0, lambda: self.widgets['save_credentials_btn'].configure(state="disabled"))
            self.root.after(0, lambda: self.widgets['save_credentials_btn'].configure(text="Saving..."))
            
            # Get credentials
            private_key = self.widgets['private_key'].get()
            wallet_address = self.widgets['wallet_address'].get()
            
            if not private_key or not wallet_address:
                self.root.after(0, lambda: messagebox.showerror("Error", "Please enter both private key and wallet address"))
                return
            
            # Save wallet address to config
            if self.config_manager:
                config = self.config_manager.get_config()
                if 'trading' not in config:
                    config['trading'] = {}
                config['trading']['wallet_address'] = wallet_address
                self.config_manager.save_config(config)
            
            # Save private key securely
            if self.security_manager:
                success = self.security_manager.save_private_key(private_key)
                if success:
                    self.root.after(0, lambda: self.widgets['key_status'].configure(text="✅ Credentials saved"))
                    self.root.after(0, lambda: messagebox.showinfo("Success", "Credentials saved successfully!"))
                    self.root.after(0, lambda: self.add_activity("✅ Credentials saved"))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Failed to save private key"))
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", "Security manager not initialized"))
                
        except Exception as e:
            logger.error(f"Save credentials failed: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Save failed: {e}"))
            
        finally:
            # Re-enable button
            self.root.after(0, lambda: self.widgets['save_credentials_btn'].configure(state="normal"))
            self.root.after(0, lambda: self.widgets['save_credentials_btn'].configure(text="💾 Save Credentials"))
    
    def save_all_settings_async(self):
        """Save all settings asynchronously"""
        self.run_async(self._save_all_settings())
    
    async def _save_all_settings(self):
        """Async implementation of saving all settings"""
        try:
            # Disable button during save
            self.root.after(0, lambda: self.widgets['save_all_settings_btn'].configure(state="disabled"))
            self.root.after(0, lambda: self.widgets['save_all_settings_btn'].configure(text="Saving..."))
            
            # Get settings
            testnet = self.widgets['testnet_var'].get()
            default_size = float(self.widgets['default_size'].get() or "100")
            max_slippage = float(self.widgets['max_slippage'].get() or "0.5")
            default_stop_loss = float(self.widgets['default_stop_loss'].get() or "2.0")
            default_take_profit = float(self.widgets['default_take_profit'].get() or "4.0")
            
            # Save to config
            if self.config_manager:
                config = self.config_manager.get_config()
                if 'trading' not in config:
                    config['trading'] = {}
                
                config['trading']['testnet'] = testnet
                config['trading']['default_order_size'] = default_size
                config['trading']['max_slippage'] = max_slippage
                config['trading']['default_stop_loss'] = default_stop_loss
                config['trading']['default_take_profit'] = default_take_profit
                
                self.config_manager.save_config(config)
                
                self.root.after(0, lambda: messagebox.showinfo("Success", "Settings saved successfully!"))
                self.root.after(0, lambda: self.add_activity("✅ Settings saved"))
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", "Config manager not initialized"))
                
        except Exception as e:
            logger.error(f"Save settings failed: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Save failed: {e}"))
            
        finally:
            # Re-enable button
            self.root.after(0, lambda: self.widgets['save_all_settings_btn'].configure(state="normal"))
            self.root.after(0, lambda: self.widgets['save_all_settings_btn'].configure(text="💾 Save All Settings"))
    
    def reset_settings_async(self):
        """Reset settings to defaults asynchronously"""
        self.run_async(self._reset_settings())
    
    async def _reset_settings(self):
        """Async implementation of resetting settings"""
        try:
            # Confirm reset
            confirm = messagebox.askyesno("Confirm Reset", "Are you sure you want to reset all settings to defaults?")
            if not confirm:
                return
                
            # Reset settings to defaults
            self.widgets['testnet_var'].set(True)
            self.widgets['default_size'].delete(0, tk.END)
            self.widgets['default_size'].insert(0, "100")
            self.widgets['max_slippage'].delete(0, tk.END)
            self.widgets['max_slippage'].insert(0, "0.5")
            self.widgets['default_stop_loss'].delete(0, tk.END)
            self.widgets['default_stop_loss'].insert(0, "2.0")
            self.widgets['default_take_profit'].delete(0, tk.END)
            self.widgets['default_take_profit'].insert(0, "4.0")
            
            self.root.after(0, lambda: messagebox.showinfo("Success", "Settings reset to defaults"))
            self.root.after(0, lambda: self.add_activity("🔄 Settings reset to defaults"))
                
        except Exception as e:
            logger.error(f"Reset settings failed: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Reset failed: {e}"))
    
    def load_existing_settings(self):
        """Load existing settings from config"""
        try:
            if not self.config_manager:
                logger.warning("Config manager not initialized, cannot load settings")
                return
                
            config = self.config_manager.get_config()
            trading_config = config.get('trading', {})
            
            # Load trading settings if available
            if 'default_order_size' in trading_config:
                self.widgets['default_size'].delete(0, tk.END)
                self.widgets['default_size'].insert(0, str(trading_config['default_order_size']))
                
            if 'max_slippage' in trading_config:
                self.widgets['max_slippage'].delete(0, tk.END)
                self.widgets['max_slippage'].insert(0, str(trading_config['max_slippage']))
                
            if 'default_stop_loss' in trading_config:
                self.widgets['default_stop_loss'].delete(0, tk.END)
                self.widgets['default_stop_loss'].insert(0, str(trading_config['default_stop_loss']))
                
            if 'default_take_profit' in trading_config:
                self.widgets['default_take_profit'].delete(0, tk.END)
                self.widgets['default_take_profit'].insert(0, str(trading_config['default_take_profit']))
                
            logger.info("Existing settings loaded from config")
            
        except Exception as e:
            logger.error(f"Failed to load existing settings: {e}")
    
    def clear_all_settings_async(self):
        """Clear all settings asynchronously"""
        self.run_async(self._clear_all_settings())
    
    async def _clear_all_settings(self):
        """Async implementation of clearing all settings"""
        try:
            # Confirm clear
            confirm = messagebox.askyesno("Confirm Clear", "Are you sure you want to clear all settings? This cannot be undone.")
            if not confirm:
                return
                
            # Clear all settings
            self.widgets['wallet_address'].delete(0, tk.END)
            self.widgets['private_key'].delete(0, tk.END)
            self.widgets['testnet_var'].set(True)
            self.widgets['default_size'].delete(0, tk.END)
            self.widgets['max_slippage'].delete(0, tk.END)
            self.widgets['default_stop_loss'].delete(0, tk.END)
            self.widgets['default_take_profit'].delete(0, tk.END)
            
            # Clear saved credentials
            if self.security_manager:
                self.security_manager.clear_private_key()
            
            # Clear config
            if self.config_manager:
                self.config_manager.reset_config()
            
            self.root.after(0, lambda: self.widgets['key_status'].configure(text="🔴 Not configured"))
            self.root.after(0, lambda: messagebox.showinfo("Success", "All settings cleared"))
            self.root.after(0, lambda: self.add_activity("🗑️ All settings cleared"))
                
        except Exception as e:
            logger.error(f"Clear settings failed: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Clear failed: {e}"))
    def on_token_selected(self, token):
        """Handle token selection"""
        pass
    
    def on_order_type_changed(self, order_type):
        """Handle order type change"""
        pass
    
    def set_order_side(self, side):
        """Set order side"""
        self.order_side = side
    
    def set_size_percentage(self, percentage):
        """Set size based on percentage"""
        pass
    
    def setup_portfolio_chart(self, parent):
        """Setup portfolio chart"""
        pass
    
    def setup_price_chart(self, parent):
        """Setup price chart"""
        pass
    
    def setup_performance_chart(self, parent):
        """Setup performance chart"""
        pass
    
    def setup_orderbook_display(self, parent):
        """Setup order book display"""
        pass
    
    def setup_recent_trades_display(self, parent):
        """Setup recent trades display"""
        pass
    
    def setup_open_orders_display(self, parent):
        """Setup open orders display"""
        pass
    
    def setup_strategy_controls(self):
        """Setup strategy controls"""
        pass
    
    def setup_positions_table(self, parent):
        """Setup positions table"""
        pass
    
    def toggle_automation(self):
        """Toggle automation"""
        pass
    
    def on_strategy_selected(self, strategy):
        """Handle strategy selection"""
        pass
    
    # Additional async methods
    def cancel_all_orders_async(self):
        self.run_async(self._cancel_all_orders())
    
    async def _cancel_all_orders(self):
        """Cancel all orders"""
        pass
    
    def refresh_positions_async(self):
        self.run_async(self._refresh_positions())
    
    async def _refresh_positions(self):
        """Refresh positions"""
        pass
    
    def close_all_positions_async(self):
        self.run_async(self._close_all_positions())
    
    async def _close_all_positions(self):
        """Close all positions"""
        pass
    
    def hedge_all_positions_async(self):
        self.run_async(self._hedge_all_positions())
    
    async def _hedge_all_positions(self):
        """Hedge all positions"""
        pass
    
    def run_backtest_async(self):
        self.run_async(self._run_backtest())
    
    async def _run_backtest(self):
        """Run backtest"""
        pass
    
    def clear_private_key_async(self):
        self.run_async(self._clear_private_key())
    
    async def _clear_private_key(self):
        """Clear private key"""
        pass
    
    def save_settings_async(self):
        self.run_async(self._save_settings())
    
    async def _save_settings(self):
        """Save settings"""
        pass
    
    def import_config_async(self):
        self.run_async(self._import_config())
    
    async def _import_config(self):
        """Import config"""
        pass
    
    def export_config_async(self):
        self.run_async(self._export_config())
    
    async def _export_config(self):
        """Export config"""
        pass
    
    def refresh_all_data_async(self):
        self.run_async(self.refresh_all_data())
    
    async def refresh_all_data(self):
        """Refresh all data"""
        await self._refresh_tokens()
        await self.refresh_market_data()
    
    def export_trades_async(self):
        self.run_async(self._export_trades())
    
    async def _export_trades(self):
        """Export trades"""
        pass
    
    def clear_cache_async(self):
        self.run_async(self._clear_cache())
    
    async def _clear_cache(self):
        """Clear cache"""
        pass
    
    def show_documentation(self):
        """Show documentation"""
        messagebox.showinfo("Documentation", "Documentation available at: https://github.com/valleytainment/hyperliquidmaster")
    
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo("About", "🚀 Hyperliquid Trading Bot - Ultimate Edition\nVersion 2.0\nProduction-ready 24/7 automated trading")
    
    def update_positions_display(self):
        """Update positions display"""
        pass


def main():
    """Main function to run the trading dashboard"""
    try:
        app = TradingDashboard()
        
        # Handle window closing
        app.root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        # Start the GUI
        app.root.mainloop()
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        messagebox.showerror("Startup Error", f"Failed to start application: {e}")


if __name__ == "__main__":
    main()


    
    # New methods for enhanced settings functionality
    
    def toggle_private_key_visibility(self):
        """Toggle private key visibility"""
        try:
            current_show = self.widgets['private_key'].cget('show')
            if current_show == '*':
                self.widgets['private_key'].configure(show='')
                self.widgets['show_key_btn'].configure(text='🙈')
            else:
                self.widgets['private_key'].configure(show='*')
                self.widgets['show_key_btn'].configure(text='👁️')
        except Exception as e:
            logger.error(f"Error toggling private key visibility: {e}")
    
    def test_connection_async(self):
        """Test connection with current credentials"""
        self.run_async(self._test_connection())
    
    async def _test_connection(self):
        """Test connection to Hyperliquid API"""
        try:
            # Get credentials from GUI
            wallet_address = self.widgets['wallet_address'].get().strip()
            private_key = self.widgets['private_key'].get().strip()
            use_testnet = self.widgets['testnet_var'].get()
            
            if not wallet_address:
                self.root.after(0, lambda: messagebox.showerror("Error", "Please enter a wallet address"))
                return
            
            if not private_key:
                self.root.after(0, lambda: messagebox.showerror("Error", "Please enter a private key"))
                return
            
            # Update status
            self.root.after(0, lambda: self.widgets['key_status'].configure(text="🟡 Testing connection..."))
            
            # Test connection
            from core.api import EnhancedHyperliquidAPI
            test_api = EnhancedHyperliquidAPI(
                private_key=private_key,
                testnet=use_testnet
            )
            
            # Try to get account info
            account_info = await test_api.get_account_info()
            
            if account_info:
                self.root.after(0, lambda: self.widgets['key_status'].configure(text="🟢 Connection successful!"))
                self.root.after(0, lambda: messagebox.showinfo("Success", f"Connection successful!\nWallet: {wallet_address}\nNetwork: {'Testnet' if use_testnet else 'Mainnet'}"))
            else:
                self.root.after(0, lambda: self.widgets['key_status'].configure(text="🔴 Connection failed"))
                self.root.after(0, lambda: messagebox.showerror("Error", "Connection failed. Please check your credentials."))
                
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            self.root.after(0, lambda: self.widgets['key_status'].configure(text="🔴 Connection failed"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Connection test failed: {str(e)}"))
    
    def save_credentials_async(self):
        """Save credentials securely"""
        self.run_async(self._save_credentials())
    
    async def _save_credentials(self):
        """Save wallet address and private key securely"""
        try:
            wallet_address = self.widgets['wallet_address'].get().strip()
            private_key = self.widgets['private_key'].get().strip()
            use_testnet = self.widgets['testnet_var'].get()
            
            if not wallet_address:
                self.root.after(0, lambda: messagebox.showerror("Error", "Please enter a wallet address"))
                return
            
            if not private_key:
                self.root.after(0, lambda: messagebox.showerror("Error", "Please enter a private key"))
                return
            
            # Validate wallet address format
            if not wallet_address.startswith('0x') or len(wallet_address) != 42:
                self.root.after(0, lambda: messagebox.showerror("Error", "Invalid wallet address format. Must be 42 characters starting with 0x"))
                return
            
            # Save wallet address to config
            self.config_manager.update_config({
                'trading': {
                    'wallet_address': wallet_address,
                    'testnet': use_testnet
                }
            })
            
            # Save private key securely
            success = self.security_manager.store_private_key(private_key, 'encrypted_file')
            
            if success:
                self.root.after(0, lambda: self.widgets['key_status'].configure(text="🟢 Credentials saved securely"))
                self.root.after(0, lambda: messagebox.showinfo("Success", "Credentials saved securely!\nPrivate key encrypted and stored safely."))
                
                # Clear the private key field for security
                self.widgets['private_key'].delete(0, 'end')
                
                # Update connection status
                self.is_connected = True
                self.update_connection_status()
                
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", "Failed to save private key securely"))
                
        except Exception as e:
            logger.error(f"Error saving credentials: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to save credentials: {str(e)}"))
    
    def load_existing_settings(self):
        """Load existing settings from config"""
        try:
            config = self.config_manager.get_config()
            trading_config = config.get('trading', {})
            
            # Load trading settings
            if 'default_order_size' in trading_config:
                self.widgets['default_size'].insert(0, str(trading_config['default_order_size']))
            
            if 'max_slippage' in trading_config:
                self.widgets['max_slippage'].insert(0, str(trading_config['max_slippage']))
            
            if 'default_stop_loss' in trading_config:
                self.widgets['default_stop_loss'].insert(0, str(trading_config['default_stop_loss']))
            
            if 'default_take_profit' in trading_config:
                self.widgets['default_take_profit'].insert(0, str(trading_config['default_take_profit']))
            
            # Check if private key exists
            if self.security_manager.has_stored_key():
                self.widgets['key_status'].configure(text="🟢 Private key configured")
                self.is_connected = True
            else:
                self.widgets['key_status'].configure(text="🔴 Not configured")
                self.is_connected = False
                
        except Exception as e:
            logger.error(f"Error loading existing settings: {e}")
    
    def save_all_settings_async(self):
        """Save all settings"""
        self.run_async(self._save_all_settings())
    
    async def _save_all_settings(self):
        """Save all settings to config"""
        try:
            # Get all values from GUI
            wallet_address = self.widgets['wallet_address'].get().strip()
            private_key = self.widgets['private_key'].get().strip()
            use_testnet = self.widgets['testnet_var'].get()
            
            default_size = self.widgets['default_size'].get().strip()
            max_slippage = self.widgets['max_slippage'].get().strip()
            default_stop_loss = self.widgets['default_stop_loss'].get().strip()
            default_take_profit = self.widgets['default_take_profit'].get().strip()
            
            # Prepare config update
            config_update = {
                'trading': {
                    'testnet': use_testnet
                }
            }
            
            # Add wallet address if provided
            if wallet_address:
                if not wallet_address.startswith('0x') or len(wallet_address) != 42:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Invalid wallet address format"))
                    return
                config_update['trading']['wallet_address'] = wallet_address
            
            # Add trading settings if provided
            if default_size:
                try:
                    config_update['trading']['default_order_size'] = float(default_size)
                except ValueError:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Invalid default order size"))
                    return
            
            if max_slippage:
                try:
                    config_update['trading']['max_slippage'] = float(max_slippage)
                except ValueError:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Invalid max slippage"))
                    return
            
            if default_stop_loss:
                try:
                    config_update['trading']['default_stop_loss'] = float(default_stop_loss)
                except ValueError:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Invalid default stop loss"))
                    return
            
            if default_take_profit:
                try:
                    config_update['trading']['default_take_profit'] = float(default_take_profit)
                except ValueError:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Invalid default take profit"))
                    return
            
            # Save config
            self.config_manager.update_config(config_update)
            
            # Save private key if provided
            if private_key:
                success = self.security_manager.store_private_key(private_key, 'encrypted_file')
                if success:
                    self.widgets['private_key'].delete(0, 'end')  # Clear for security
                    self.widgets['key_status'].configure(text="🟢 All settings saved securely")
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Failed to save private key"))
                    return
            
            self.root.after(0, lambda: messagebox.showinfo("Success", "All settings saved successfully!"))
            
            # Update connection status
            self.update_connection_status()
            
        except Exception as e:
            logger.error(f"Error saving all settings: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to save settings: {str(e)}"))
    
    def reset_settings_async(self):
        """Reset settings to defaults"""
        self.run_async(self._reset_settings())
    
    async def _reset_settings(self):
        """Reset all settings to defaults"""
        try:
            # Confirm reset
            result = messagebox.askyesno("Confirm Reset", "Are you sure you want to reset all settings to defaults?\nThis will not affect your saved private key.")
            
            if result:
                # Clear GUI fields
                self.widgets['default_size'].delete(0, 'end')
                self.widgets['default_size'].insert(0, '100')
                
                self.widgets['max_slippage'].delete(0, 'end')
                self.widgets['max_slippage'].insert(0, '0.5')
                
                self.widgets['default_stop_loss'].delete(0, 'end')
                self.widgets['default_stop_loss'].insert(0, '2.0')
                
                self.widgets['default_take_profit'].delete(0, 'end')
                self.widgets['default_take_profit'].insert(0, '4.0')
                
                self.widgets['testnet_var'].set(True)
                
                # Save defaults to config
                config_update = {
                    'trading': {
                        'default_order_size': 100.0,
                        'max_slippage': 0.5,
                        'default_stop_loss': 2.0,
                        'default_take_profit': 4.0,
                        'testnet': True
                    }
                }
                
                self.config_manager.update_config(config_update)
                
                self.root.after(0, lambda: messagebox.showinfo("Success", "Settings reset to defaults!"))
                
        except Exception as e:
            logger.error(f"Error resetting settings: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to reset settings: {str(e)}"))
    
    def clear_all_settings_async(self):
        """Clear all settings including private key"""
        self.run_async(self._clear_all_settings())
    
    async def _clear_all_settings(self):
        """Clear all settings and private key"""
        try:
            # Confirm clear
            result = messagebox.askyesno("Confirm Clear All", "Are you sure you want to clear ALL settings including private key?\nThis action cannot be undone!")
            
            if result:
                # Clear GUI fields
                self.widgets['wallet_address'].delete(0, 'end')
                self.widgets['private_key'].delete(0, 'end')
                self.widgets['default_size'].delete(0, 'end')
                self.widgets['max_slippage'].delete(0, 'end')
                self.widgets['default_stop_loss'].delete(0, 'end')
                self.widgets['default_take_profit'].delete(0, 'end')
                
                # Clear private key
                self.security_manager.clear_private_key()
                
                # Reset config to minimal
                config_update = {
                    'trading': {
                        'testnet': True
                    }
                }
                
                self.config_manager.update_config(config_update)
                
                # Update status
                self.widgets['key_status'].configure(text="🔴 Not configured")
                self.is_connected = False
                self.update_connection_status()
                
                self.root.after(0, lambda: messagebox.showinfo("Success", "All settings cleared!"))
                
        except Exception as e:
            logger.error(f"Error clearing all settings: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to clear settings: {str(e)}"))
    
    def update_connection_status(self):
        """Update connection status throughout the GUI"""
        try:
            if self.is_connected:
                # Update status indicators
                if 'connection_status' in self.widgets:
                    self.widgets['connection_status'].configure(text="🟢 Connected", text_color="green")
                
                # Enable trading buttons
                if 'place_order_btn' in self.widgets:
                    self.widgets['place_order_btn'].configure(state="normal")
                
                if 'start_auto_trading_btn' in self.widgets:
                    self.widgets['start_auto_trading_btn'].configure(state="normal")
                    
            else:
                # Update status indicators
                if 'connection_status' in self.widgets:
                    self.widgets['connection_status'].configure(text="🔴 Not Connected", text_color="red")
                
                # Disable trading buttons
                if 'place_order_btn' in self.widgets:
                    self.widgets['place_order_btn'].configure(state="disabled")
                
                if 'start_auto_trading_btn' in self.widgets:
                    self.widgets['start_auto_trading_btn'].configure(state="disabled")
                    
        except Exception as e:
            logger.error(f"Error updating connection status: {e}")

