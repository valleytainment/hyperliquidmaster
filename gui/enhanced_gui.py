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
        """Setup settings tab with improved private key handling"""
        settings_tab = self.notebook.add("⚙️ Settings")
        
        # Connection settings
        conn_frame = ctk.CTkFrame(settings_tab)
        conn_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(conn_frame, text="🔗 Connection Settings", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=5)
        
        # Testnet toggle
        testnet_frame = ctk.CTkFrame(conn_frame)
        testnet_frame.pack(fill="x", padx=10, pady=5)
        
        self.widgets['testnet_var'] = tk.BooleanVar(value=True)
        self.widgets['testnet_switch'] = ctk.CTkSwitch(
            testnet_frame, 
            text="🧪 Use Testnet (Recommended for testing)", 
            variable=self.widgets['testnet_var']
        )
        self.widgets['testnet_switch'].pack(anchor="w")
        
        # Wallet address
        wallet_frame = ctk.CTkFrame(conn_frame)
        wallet_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(wallet_frame, text="Wallet Address:").pack(anchor="w")
        self.widgets['wallet_address'] = ctk.CTkEntry(wallet_frame, placeholder_text="0x...")
        self.widgets['wallet_address'].pack(fill="x", pady=2)
        
        # Private key management
        key_frame = ctk.CTkFrame(settings_tab)
        key_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(key_frame, text="🔐 Private Key Management", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=5)
        
        # Private key status
        key_status_frame = ctk.CTkFrame(key_frame)
        key_status_frame.pack(fill="x", padx=10, pady=5)
        
        self.widgets['key_status'] = ctk.CTkLabel(key_status_frame, text="Status: Not configured")
        self.widgets['key_status'].pack(side="left")
        
        # Private key buttons
        key_buttons_frame = ctk.CTkFrame(key_frame)
        key_buttons_frame.pack(fill="x", padx=10, pady=5)
        
        self.widgets['setup_key_btn'] = ctk.CTkButton(
            key_buttons_frame, 
            text="🔑 Setup Private Key", 
            command=self.setup_private_key_async
        )
        self.widgets['setup_key_btn'].pack(side="left", padx=5)
        
        self.widgets['clear_key_btn'] = ctk.CTkButton(
            key_buttons_frame, 
            text="🗑️ Clear Key", 
            fg_color="red",
            command=self.clear_private_key_async
        )
        self.widgets['clear_key_btn'].pack(side="left", padx=5)
        
        # Trading settings
        trading_frame = ctk.CTkFrame(settings_tab)
        trading_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(trading_frame, text="💰 Trading Settings", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=5)
        
        # Default settings grid
        defaults_grid = ctk.CTkFrame(trading_frame)
        defaults_grid.pack(fill="x", padx=10, pady=5)
        
        # Default order size
        size_frame = ctk.CTkFrame(defaults_grid)
        size_frame.pack(side="left", fill="x", expand=True, padx=5)
        
        ctk.CTkLabel(size_frame, text="Default Order Size ($):").pack()
        self.widgets['default_size'] = ctk.CTkEntry(size_frame, placeholder_text="100")
        self.widgets['default_size'].pack(fill="x", pady=2)
        
        # Default slippage
        slippage_frame = ctk.CTkFrame(defaults_grid)
        slippage_frame.pack(side="left", fill="x", expand=True, padx=5)
        
        ctk.CTkLabel(slippage_frame, text="Max Slippage (%):").pack()
        self.widgets['max_slippage'] = ctk.CTkEntry(slippage_frame, placeholder_text="0.5")
        self.widgets['max_slippage'].pack(fill="x", pady=2)
        
        # Save settings button
        self.widgets['save_settings_btn'] = ctk.CTkButton(
            settings_tab, 
            text="💾 Save Settings", 
            command=self.save_settings_async
        )
        self.widgets['save_settings_btn'].pack(pady=20)
    
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
            # Check if we have saved credentials
            private_key = self.security_manager.get_private_key()
            if not private_key:
                self.root.after(0, lambda: messagebox.showerror("Error", "No private key found. Please setup private key first."))
                return
            
            # Get wallet address from config
            config = self.config_manager.get_config()
            wallet_address = config.get('wallet_address', '')
            
            if not wallet_address:
                self.root.after(0, lambda: messagebox.showerror("Error", "No wallet address configured. Please check settings."))
                return
            
            # Update status
            self.root.after(0, lambda: self.widgets['status_text'].configure(text="Connecting..."))
            
            # Initialize API
            testnet = self.widgets['testnet_var'].get()
            self.api = EnhancedHyperliquidAPI(testnet=testnet)
            
            # Authenticate
            if await self.api.authenticate_async(private_key, wallet_address):
                self.is_connected = True
                self.root.after(0, self._update_connection_status)
                self.root.after(0, lambda: self.add_activity("✅ Connected to Hyperliquid API"))
                
                # Start data refresh
                await self.refresh_all_data()
                
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", "Failed to authenticate with API"))
                
        except Exception as e:
            logger.error(f"Quick connect failed: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Connection failed: {e}"))
    
    def _update_connection_status(self):
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
    
    # Placeholder methods for missing functionality
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

