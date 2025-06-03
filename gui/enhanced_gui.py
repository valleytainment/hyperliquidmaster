"""
Enhanced GUI for Hyperliquid Trading Bot
Modern interface combining features from all analyzed repositories
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
    """Main trading dashboard with comprehensive functionality"""
    
    def __init__(self):
        """Initialize the trading dashboard"""
        self.root = ctk.CTk()
        self.root.title("Hyperliquid Trading Bot - Enhanced Dashboard")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.security_manager = SecurityManager()
        self.api = None
        self.is_connected = False
        self.is_trading = False
        
        # Data storage
        self.account_data = {}
        self.market_data = {}
        self.positions = []
        self.orders = []
        self.trade_history = []
        
        # GUI components
        self.widgets = {}
        self.charts = {}
        
        # Update threads
        self.update_thread = None
        self.stop_updates = False
        
        # Initialize GUI
        self.setup_gui()
        self.setup_menu()
        self.setup_status_bar()
        
        # Start update loop
        self.start_update_loop()
        
        logger.info("Trading dashboard initialized")
    
    def setup_gui(self):
        """Setup the main GUI layout"""
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
        self.setup_strategies_tab()
        self.setup_backtesting_tab()
        self.setup_settings_tab()
    
    def setup_dashboard_tab(self):
        """Setup the main dashboard tab"""
        dashboard_tab = self.notebook.add("Dashboard")
        
        # Top row - Account summary
        account_frame = ctk.CTkFrame(dashboard_tab)
        account_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(account_frame, text="Account Summary", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=5)
        
        # Account metrics
        metrics_frame = ctk.CTkFrame(account_frame)
        metrics_frame.pack(fill="x", padx=10, pady=5)
        
        # Create metric displays
        self.widgets['account_value'] = self.create_metric_display(metrics_frame, "Account Value", "$0.00", 0, 0)
        self.widgets['total_pnl'] = self.create_metric_display(metrics_frame, "Total PnL", "$0.00", 0, 1)
        self.widgets['margin_used'] = self.create_metric_display(metrics_frame, "Margin Used", "0%", 0, 2)
        self.widgets['open_positions'] = self.create_metric_display(metrics_frame, "Open Positions", "0", 0, 3)
        
        # Middle row - Charts
        charts_frame = ctk.CTkFrame(dashboard_tab)
        charts_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create chart notebook
        chart_notebook = ctk.CTkTabview(charts_frame)
        chart_notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Portfolio chart
        portfolio_tab = chart_notebook.add("Portfolio")
        self.setup_portfolio_chart(portfolio_tab)
        
        # Price chart
        price_tab = chart_notebook.add("Price Chart")
        self.setup_price_chart(price_tab)
        
        # Performance chart
        performance_tab = chart_notebook.add("Performance")
        self.setup_performance_chart(performance_tab)
        
        # Bottom row - Recent activity
        activity_frame = ctk.CTkFrame(dashboard_tab)
        activity_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(activity_frame, text="Recent Activity", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        # Activity list
        self.widgets['activity_list'] = ctk.CTkTextbox(activity_frame, height=150)
        self.widgets['activity_list'].pack(fill="x", padx=10, pady=5)
    
    def setup_trading_tab(self):
        """Setup the trading tab"""
        trading_tab = self.notebook.add("Trading")
        
        # Left panel - Order entry
        left_frame = ctk.CTkFrame(trading_tab)
        left_frame.pack(side="left", fill="y", padx=5, pady=5)
        
        ctk.CTkLabel(left_frame, text="Order Entry", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
        
        # Symbol selection
        symbol_frame = ctk.CTkFrame(left_frame)
        symbol_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(symbol_frame, text="Symbol:").pack(anchor="w")
        self.widgets['symbol_entry'] = ctk.CTkEntry(symbol_frame, placeholder_text="BTC")
        self.widgets['symbol_entry'].pack(fill="x", pady=2)
        
        # Order type
        order_type_frame = ctk.CTkFrame(left_frame)
        order_type_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(order_type_frame, text="Order Type:").pack(anchor="w")
        self.widgets['order_type'] = ctk.CTkOptionMenu(order_type_frame, values=["Market", "Limit", "Stop"])
        self.widgets['order_type'].pack(fill="x", pady=2)
        
        # Side selection
        side_frame = ctk.CTkFrame(left_frame)
        side_frame.pack(fill="x", padx=10, pady=5)
        
        self.widgets['side_var'] = tk.StringVar(value="Buy")
        buy_radio = ctk.CTkRadioButton(side_frame, text="Buy", variable=self.widgets['side_var'], value="Buy")
        sell_radio = ctk.CTkRadioButton(side_frame, text="Sell", variable=self.widgets['side_var'], value="Sell")
        buy_radio.pack(side="left", padx=5)
        sell_radio.pack(side="left", padx=5)
        
        # Size entry
        size_frame = ctk.CTkFrame(left_frame)
        size_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(size_frame, text="Size (USD):").pack(anchor="w")
        self.widgets['size_entry'] = ctk.CTkEntry(size_frame, placeholder_text="100.00")
        self.widgets['size_entry'].pack(fill="x", pady=2)
        
        # Price entry (for limit orders)
        price_frame = ctk.CTkFrame(left_frame)
        price_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(price_frame, text="Price:").pack(anchor="w")
        self.widgets['price_entry'] = ctk.CTkEntry(price_frame, placeholder_text="Current market price")
        self.widgets['price_entry'].pack(fill="x", pady=2)
        
        # Leverage
        leverage_frame = ctk.CTkFrame(left_frame)
        leverage_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(leverage_frame, text="Leverage:").pack(anchor="w")
        self.widgets['leverage_slider'] = ctk.CTkSlider(leverage_frame, from_=1, to=50, number_of_steps=49)
        self.widgets['leverage_slider'].set(1)
        self.widgets['leverage_slider'].pack(fill="x", pady=2)
        
        self.widgets['leverage_label'] = ctk.CTkLabel(leverage_frame, text="1x")
        self.widgets['leverage_label'].pack()
        
        # Bind slider update
        self.widgets['leverage_slider'].configure(command=self.update_leverage_label)
        
        # Order buttons
        button_frame = ctk.CTkFrame(left_frame)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        self.widgets['place_order_btn'] = ctk.CTkButton(button_frame, text="Place Order", command=self.place_order)
        self.widgets['place_order_btn'].pack(fill="x", pady=2)
        
        self.widgets['cancel_all_btn'] = ctk.CTkButton(button_frame, text="Cancel All Orders", command=self.cancel_all_orders)
        self.widgets['cancel_all_btn'].pack(fill="x", pady=2)
        
        # Right panel - Market data and order book
        right_frame = ctk.CTkFrame(trading_tab)
        right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        # Market data
        market_frame = ctk.CTkFrame(right_frame)
        market_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(market_frame, text="Market Data", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        # Price display
        self.widgets['current_price'] = ctk.CTkLabel(market_frame, text="$0.00", font=ctk.CTkFont(size=24, weight="bold"))
        self.widgets['current_price'].pack(pady=5)
        
        # Market stats
        stats_frame = ctk.CTkFrame(market_frame)
        stats_frame.pack(fill="x", padx=10, pady=5)
        
        self.widgets['volume_24h'] = self.create_stat_display(stats_frame, "24h Volume", "$0", 0, 0)
        self.widgets['funding_rate'] = self.create_stat_display(stats_frame, "Funding Rate", "0%", 0, 1)
        
        # Order book
        orderbook_frame = ctk.CTkFrame(right_frame)
        orderbook_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        ctk.CTkLabel(orderbook_frame, text="Order Book", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        # Order book display (simplified)
        self.widgets['orderbook'] = ctk.CTkTextbox(orderbook_frame, height=300)
        self.widgets['orderbook'].pack(fill="both", expand=True, padx=10, pady=5)
    
    def setup_positions_tab(self):
        """Setup the positions tab"""
        positions_tab = self.notebook.add("Positions")
        
        # Positions table
        positions_frame = ctk.CTkFrame(positions_tab)
        positions_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(positions_frame, text="Open Positions", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
        
        # Create treeview for positions
        columns = ("Symbol", "Side", "Size", "Entry Price", "Current Price", "PnL", "PnL%", "Actions")
        self.widgets['positions_tree'] = ttk.Treeview(positions_frame, columns=columns, show="headings", height=10)
        
        # Configure columns
        for col in columns:
            self.widgets['positions_tree'].heading(col, text=col)
            self.widgets['positions_tree'].column(col, width=120, anchor="center")
        
        # Scrollbar for positions
        positions_scrollbar = ttk.Scrollbar(positions_frame, orient="vertical", command=self.widgets['positions_tree'].yview)
        self.widgets['positions_tree'].configure(yscrollcommand=positions_scrollbar.set)
        
        # Pack positions table
        self.widgets['positions_tree'].pack(side="left", fill="both", expand=True, padx=10, pady=5)
        positions_scrollbar.pack(side="right", fill="y")
        
        # Position actions
        actions_frame = ctk.CTkFrame(positions_tab)
        actions_frame.pack(fill="x", padx=10, pady=5)
        
        self.widgets['close_position_btn'] = ctk.CTkButton(actions_frame, text="Close Selected Position", command=self.close_selected_position)
        self.widgets['close_position_btn'].pack(side="left", padx=5)
        
        self.widgets['close_all_btn'] = ctk.CTkButton(actions_frame, text="Close All Positions", command=self.close_all_positions)
        self.widgets['close_all_btn'].pack(side="left", padx=5)
        
        # Orders section
        orders_frame = ctk.CTkFrame(positions_tab)
        orders_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(orders_frame, text="Open Orders", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
        
        # Create treeview for orders
        order_columns = ("Symbol", "Side", "Type", "Size", "Price", "Status", "Time", "Actions")
        self.widgets['orders_tree'] = ttk.Treeview(orders_frame, columns=order_columns, show="headings", height=8)
        
        # Configure order columns
        for col in order_columns:
            self.widgets['orders_tree'].heading(col, text=col)
            self.widgets['orders_tree'].column(col, width=100, anchor="center")
        
        # Scrollbar for orders
        orders_scrollbar = ttk.Scrollbar(orders_frame, orient="vertical", command=self.widgets['orders_tree'].yview)
        self.widgets['orders_tree'].configure(yscrollcommand=orders_scrollbar.set)
        
        # Pack orders table
        self.widgets['orders_tree'].pack(side="left", fill="both", expand=True, padx=10, pady=5)
        orders_scrollbar.pack(side="right", fill="y")
    
    def setup_strategies_tab(self):
        """Setup the strategies tab"""
        strategies_tab = self.notebook.add("Strategies")
        
        # Strategy list
        strategy_list_frame = ctk.CTkFrame(strategies_tab)
        strategy_list_frame.pack(side="left", fill="y", padx=5, pady=5)
        
        ctk.CTkLabel(strategy_list_frame, text="Available Strategies", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        # Strategy buttons
        strategies = ["BB RSI ADX", "Hull Suite", "Scalping", "Grid Trading", "DCA Bot"]
        self.widgets['strategy_buttons'] = {}
        
        for strategy in strategies:
            btn = ctk.CTkButton(strategy_list_frame, text=strategy, command=lambda s=strategy: self.select_strategy(s))
            btn.pack(fill="x", padx=10, pady=2)
            self.widgets['strategy_buttons'][strategy] = btn
        
        # Strategy configuration
        config_frame = ctk.CTkFrame(strategies_tab)
        config_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(config_frame, text="Strategy Configuration", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        # Configuration area
        self.widgets['strategy_config'] = ctk.CTkTextbox(config_frame, height=400)
        self.widgets['strategy_config'].pack(fill="both", expand=True, padx=10, pady=5)
        
        # Strategy controls
        controls_frame = ctk.CTkFrame(config_frame)
        controls_frame.pack(fill="x", padx=10, pady=5)
        
        self.widgets['start_strategy_btn'] = ctk.CTkButton(controls_frame, text="Start Strategy", command=self.start_strategy)
        self.widgets['start_strategy_btn'].pack(side="left", padx=5)
        
        self.widgets['stop_strategy_btn'] = ctk.CTkButton(controls_frame, text="Stop Strategy", command=self.stop_strategy)
        self.widgets['stop_strategy_btn'].pack(side="left", padx=5)
        
        self.widgets['save_config_btn'] = ctk.CTkButton(controls_frame, text="Save Config", command=self.save_strategy_config)
        self.widgets['save_config_btn'].pack(side="left", padx=5)
    
    def setup_backtesting_tab(self):
        """Setup the backtesting tab"""
        backtesting_tab = self.notebook.add("Backtesting")
        
        # Backtest configuration
        config_frame = ctk.CTkFrame(backtesting_tab)
        config_frame.pack(side="left", fill="y", padx=5, pady=5)
        
        ctk.CTkLabel(config_frame, text="Backtest Configuration", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        # Strategy selection
        strategy_frame = ctk.CTkFrame(config_frame)
        strategy_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(strategy_frame, text="Strategy:").pack(anchor="w")
        self.widgets['backtest_strategy'] = ctk.CTkOptionMenu(strategy_frame, values=["BB RSI ADX", "Hull Suite", "Scalping"])
        self.widgets['backtest_strategy'].pack(fill="x", pady=2)
        
        # Time range
        time_frame = ctk.CTkFrame(config_frame)
        time_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(time_frame, text="Time Range:").pack(anchor="w")
        self.widgets['backtest_timeframe'] = ctk.CTkOptionMenu(time_frame, values=["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"])
        self.widgets['backtest_timeframe'].pack(fill="x", pady=2)
        
        # Initial capital
        capital_frame = ctk.CTkFrame(config_frame)
        capital_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(capital_frame, text="Initial Capital:").pack(anchor="w")
        self.widgets['initial_capital'] = ctk.CTkEntry(capital_frame, placeholder_text="10000")
        self.widgets['initial_capital'].pack(fill="x", pady=2)
        
        # Run backtest button
        self.widgets['run_backtest_btn'] = ctk.CTkButton(config_frame, text="Run Backtest", command=self.run_backtest)
        self.widgets['run_backtest_btn'].pack(fill="x", padx=10, pady=10)
        
        # Results area
        results_frame = ctk.CTkFrame(backtesting_tab)
        results_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(results_frame, text="Backtest Results", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        # Results display
        self.widgets['backtest_results'] = ctk.CTkTextbox(results_frame, height=300)
        self.widgets['backtest_results'].pack(fill="both", expand=True, padx=10, pady=5)
        
        # Performance chart
        self.setup_backtest_chart(results_frame)
    
    def setup_settings_tab(self):
        """Setup the settings tab"""
        settings_tab = self.notebook.add("Settings")
        
        # Connection settings
        connection_frame = ctk.CTkFrame(settings_tab)
        connection_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(connection_frame, text="Connection Settings", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        # API settings
        api_frame = ctk.CTkFrame(connection_frame)
        api_frame.pack(fill="x", padx=10, pady=5)
        
        # Testnet toggle
        self.widgets['testnet_var'] = tk.BooleanVar()
        testnet_check = ctk.CTkCheckBox(api_frame, text="Use Testnet", variable=self.widgets['testnet_var'])
        testnet_check.pack(anchor="w", padx=10, pady=5)
        
        # Wallet address
        wallet_frame = ctk.CTkFrame(api_frame)
        wallet_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(wallet_frame, text="Wallet Address:").pack(anchor="w")
        self.widgets['wallet_address'] = ctk.CTkEntry(wallet_frame, placeholder_text="0x...")
        self.widgets['wallet_address'].pack(fill="x", pady=2)
        
        # Connection buttons
        conn_buttons_frame = ctk.CTkFrame(api_frame)
        conn_buttons_frame.pack(fill="x", padx=10, pady=5)
        
        self.widgets['connect_btn'] = ctk.CTkButton(conn_buttons_frame, text="Connect", command=self.connect_api)
        self.widgets['connect_btn'].pack(side="left", padx=5)
        
        self.widgets['disconnect_btn'] = ctk.CTkButton(conn_buttons_frame, text="Disconnect", command=self.disconnect_api)
        self.widgets['disconnect_btn'].pack(side="left", padx=5)
        
        # Risk management settings
        risk_frame = ctk.CTkFrame(settings_tab)
        risk_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(risk_frame, text="Risk Management", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        # Risk parameters
        risk_params_frame = ctk.CTkFrame(risk_frame)
        risk_params_frame.pack(fill="x", padx=10, pady=5)
        
        # Max position size
        max_pos_frame = ctk.CTkFrame(risk_params_frame)
        max_pos_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(max_pos_frame, text="Max Position Size (USD):").pack(anchor="w")
        self.widgets['max_position_size'] = ctk.CTkEntry(max_pos_frame, placeholder_text="1000")
        self.widgets['max_position_size'].pack(fill="x", pady=2)
        
        # Stop loss
        stop_loss_frame = ctk.CTkFrame(risk_params_frame)
        stop_loss_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(stop_loss_frame, text="Default Stop Loss (%):").pack(anchor="w")
        self.widgets['stop_loss_pct'] = ctk.CTkEntry(stop_loss_frame, placeholder_text="2.0")
        self.widgets['stop_loss_pct'].pack(fill="x", pady=2)
        
        # Save settings button
        self.widgets['save_settings_btn'] = ctk.CTkButton(risk_frame, text="Save Settings", command=self.save_settings)
        self.widgets['save_settings_btn'].pack(pady=10)
    
    def setup_menu(self):
        """Setup the menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Import Config", command=self.import_config)
        file_menu.add_command(label="Export Config", command=self.export_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Setup Private Key", command=self.setup_private_key)
        tools_menu.add_command(label="Clear Credentials", command=self.clear_credentials)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def setup_status_bar(self):
        """Setup the status bar"""
        self.status_frame = ctk.CTkFrame(self.root)
        self.status_frame.pack(side="bottom", fill="x", padx=10, pady=5)
        
        self.widgets['status_label'] = ctk.CTkLabel(self.status_frame, text="Disconnected")
        self.widgets['status_label'].pack(side="left", padx=10)
        
        self.widgets['connection_indicator'] = ctk.CTkLabel(self.status_frame, text="●", text_color="red")
        self.widgets['connection_indicator'].pack(side="right", padx=10)
    
    def create_metric_display(self, parent, title, value, row, col):
        """Create a metric display widget"""
        frame = ctk.CTkFrame(parent)
        frame.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
        
        title_label = ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=12))
        title_label.pack(pady=2)
        
        value_label = ctk.CTkLabel(frame, text=value, font=ctk.CTkFont(size=16, weight="bold"))
        value_label.pack(pady=2)
        
        return value_label
    
    def create_stat_display(self, parent, title, value, row, col):
        """Create a stat display widget"""
        frame = ctk.CTkFrame(parent)
        frame.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
        
        title_label = ctk.CTkLabel(frame, text=title)
        title_label.pack()
        
        value_label = ctk.CTkLabel(frame, text=value, font=ctk.CTkFont(weight="bold"))
        value_label.pack()
        
        return value_label
    
    def setup_portfolio_chart(self, parent):
        """Setup portfolio allocation chart"""
        fig = Figure(figsize=(6, 4), dpi=100, facecolor='#2b2b2b')
        ax = fig.add_subplot(111, facecolor='#2b2b2b')
        
        # Sample data
        labels = ['BTC', 'ETH', 'SOL', 'AVAX']
        sizes = [40, 30, 20, 10]
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Portfolio Allocation', color='white')
        
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self.charts['portfolio'] = (fig, ax, canvas)
    
    def setup_price_chart(self, parent):
        """Setup price chart"""
        fig = Figure(figsize=(8, 4), dpi=100, facecolor='#2b2b2b')
        ax = fig.add_subplot(111, facecolor='#2b2b2b')
        
        # Sample price data
        x = pd.date_range(start='2024-01-01', periods=100, freq='H')
        y = np.cumsum(np.random.randn(100)) + 50000
        
        ax.plot(x, y, color='#4ecdc4', linewidth=2)
        ax.set_title('BTC Price Chart', color='white')
        ax.set_xlabel('Time', color='white')
        ax.set_ylabel('Price (USD)', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self.charts['price'] = (fig, ax, canvas)
    
    def setup_performance_chart(self, parent):
        """Setup performance chart"""
        fig = Figure(figsize=(8, 4), dpi=100, facecolor='#2b2b2b')
        ax = fig.add_subplot(111, facecolor='#2b2b2b')
        
        # Sample performance data
        x = pd.date_range(start='2024-01-01', periods=30, freq='D')
        y = np.cumsum(np.random.randn(30) * 0.02) + 1
        
        ax.plot(x, y, color='#96ceb4', linewidth=2)
        ax.axhline(y=1, color='white', linestyle='--', alpha=0.5)
        ax.set_title('Portfolio Performance', color='white')
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Cumulative Return', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self.charts['performance'] = (fig, ax, canvas)
    
    def setup_backtest_chart(self, parent):
        """Setup backtest results chart"""
        chart_frame = ctk.CTkFrame(parent)
        chart_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        fig = Figure(figsize=(8, 3), dpi=100, facecolor='#2b2b2b')
        ax = fig.add_subplot(111, facecolor='#2b2b2b')
        
        ax.set_title('Backtest Results', color='white')
        ax.set_xlabel('Time', color='white')
        ax.set_ylabel('Portfolio Value', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self.charts['backtest'] = (fig, ax, canvas)
    
    # Event handlers and methods
    def update_leverage_label(self, value):
        """Update leverage label"""
        leverage = int(float(value))
        self.widgets['leverage_label'].configure(text=f"{leverage}x")
    
    def connect_api(self):
        """Connect to Hyperliquid API"""
        try:
            testnet = self.widgets['testnet_var'].get()
            wallet_address = self.widgets['wallet_address'].get()
            
            if not wallet_address:
                messagebox.showerror("Error", "Please enter wallet address")
                return
            
            # Initialize API
            self.api = EnhancedHyperliquidAPI(testnet=testnet)
            
            # Get private key
            private_key = self.security_manager.get_private_key()
            if not private_key:
                messagebox.showerror("Error", "Private key not found. Please setup private key first.")
                return
            
            # Authenticate
            if self.api.authenticate(private_key, wallet_address):
                self.is_connected = True
                self.widgets['status_label'].configure(text="Connected")
                self.widgets['connection_indicator'].configure(text_color="green")
                self.add_activity("Connected to Hyperliquid API")
                logger.info("Successfully connected to Hyperliquid API")
            else:
                messagebox.showerror("Error", "Failed to authenticate with API")
                
        except Exception as e:
            messagebox.showerror("Error", f"Connection failed: {e}")
            logger.error(f"API connection failed: {e}")
    
    def disconnect_api(self):
        """Disconnect from API"""
        if self.api:
            self.api.stop_websocket()
            self.api = None
        
        self.is_connected = False
        self.widgets['status_label'].configure(text="Disconnected")
        self.widgets['connection_indicator'].configure(text_color="red")
        self.add_activity("Disconnected from API")
        logger.info("Disconnected from Hyperliquid API")
    
    def place_order(self):
        """Place a trading order"""
        if not self.is_connected:
            messagebox.showerror("Error", "Not connected to API")
            return
        
        try:
            symbol = self.widgets['symbol_entry'].get().upper()
            order_type = self.widgets['order_type'].get()
            side = self.widgets['side_var'].get()
            size = float(self.widgets['size_entry'].get())
            
            if order_type == "Market":
                result = self.api.place_market_order(
                    coin=symbol,
                    is_buy=(side == "Buy"),
                    sz=size
                )
            else:
                price = float(self.widgets['price_entry'].get())
                result = self.api.place_limit_order(
                    coin=symbol,
                    is_buy=(side == "Buy"),
                    sz=size,
                    limit_px=price
                )
            
            if result.get('status') == 'ok':
                self.add_activity(f"Order placed: {side} {size} {symbol}")
                messagebox.showinfo("Success", "Order placed successfully")
            else:
                messagebox.showerror("Error", f"Order failed: {result.get('error', 'Unknown error')}")
                
        except ValueError:
            messagebox.showerror("Error", "Invalid input values")
        except Exception as e:
            messagebox.showerror("Error", f"Order failed: {e}")
    
    def cancel_all_orders(self):
        """Cancel all open orders"""
        if not self.is_connected:
            messagebox.showerror("Error", "Not connected to API")
            return
        
        try:
            result = self.api.cancel_all_orders()
            if result.get('status') == 'ok':
                self.add_activity("All orders cancelled")
                messagebox.showinfo("Success", "All orders cancelled")
            else:
                messagebox.showerror("Error", f"Cancel failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Cancel failed: {e}")
    
    def close_selected_position(self):
        """Close selected position"""
        selection = self.widgets['positions_tree'].selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a position to close")
            return
        
        # Implementation for closing position
        messagebox.showinfo("Info", "Position close functionality will be implemented")
    
    def close_all_positions(self):
        """Close all positions"""
        if not self.is_connected:
            messagebox.showerror("Error", "Not connected to API")
            return
        
        # Implementation for closing all positions
        messagebox.showinfo("Info", "Close all positions functionality will be implemented")
    
    def select_strategy(self, strategy_name):
        """Select and configure a strategy"""
        self.widgets['strategy_config'].delete("1.0", "end")
        
        # Load strategy configuration
        config_text = f"Strategy: {strategy_name}\n\n"
        config_text += "Configuration parameters will be loaded here...\n"
        
        self.widgets['strategy_config'].insert("1.0", config_text)
    
    def start_strategy(self):
        """Start the selected strategy"""
        messagebox.showinfo("Info", "Strategy start functionality will be implemented")
    
    def stop_strategy(self):
        """Stop the running strategy"""
        messagebox.showinfo("Info", "Strategy stop functionality will be implemented")
    
    def save_strategy_config(self):
        """Save strategy configuration"""
        messagebox.showinfo("Info", "Strategy config save functionality will be implemented")
    
    def run_backtest(self):
        """Run backtest"""
        strategy = self.widgets['backtest_strategy'].get()
        timeframe = self.widgets['backtest_timeframe'].get()
        capital = self.widgets['initial_capital'].get()
        
        self.widgets['backtest_results'].delete("1.0", "end")
        self.widgets['backtest_results'].insert("1.0", f"Running backtest for {strategy}...\n")
        
        # Implementation for backtesting
        messagebox.showinfo("Info", "Backtest functionality will be implemented")
    
    def save_settings(self):
        """Save application settings"""
        try:
            # Update configuration
            self.config_manager.set('trading.wallet_address', self.widgets['wallet_address'].get())
            self.config_manager.set('trading.testnet', self.widgets['testnet_var'].get())
            self.config_manager.set('trading.max_position_size', float(self.widgets['max_position_size'].get() or 1000))
            self.config_manager.set('trading.stop_loss_percentage', float(self.widgets['stop_loss_pct'].get() or 2.0))
            
            self.config_manager.save_config()
            messagebox.showinfo("Success", "Settings saved successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")
    
    def setup_private_key(self):
        """Setup private key"""
        if self.security_manager.setup_private_key():
            messagebox.showinfo("Success", "Private key setup completed")
        else:
            messagebox.showerror("Error", "Private key setup failed")
    
    def clear_credentials(self):
        """Clear all stored credentials"""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all credentials?"):
            if self.security_manager.clear_all_credentials():
                messagebox.showinfo("Success", "All credentials cleared")
            else:
                messagebox.showerror("Error", "Failed to clear credentials")
    
    def import_config(self):
        """Import configuration from file"""
        file_path = filedialog.askopenfilename(
            title="Import Configuration",
            filetypes=[("YAML files", "*.yaml"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.config_manager.import_config(file_path)
                messagebox.showinfo("Success", "Configuration imported successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Import failed: {e}")
    
    def export_config(self):
        """Export configuration to file"""
        file_path = filedialog.asksaveasfilename(
            title="Export Configuration",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.config_manager.export_config(file_path)
                messagebox.showinfo("Success", "Configuration exported successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
Hyperliquid Trading Bot - Enhanced Dashboard

Version: 1.0.0
Author: AI Trading Bot Developer

Features:
- Real-time trading interface
- Multiple trading strategies
- Comprehensive backtesting
- Risk management tools
- Portfolio analytics

Built with Python, tkinter, and the Hyperliquid API.
        """
        messagebox.showinfo("About", about_text)
    
    def add_activity(self, message):
        """Add activity to the activity log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        activity_message = f"[{timestamp}] {message}\n"
        
        self.widgets['activity_list'].insert("end", activity_message)
        self.widgets['activity_list'].see("end")
    
    def update_account_data(self):
        """Update account data from API"""
        if not self.is_connected or not self.api:
            return
        
        try:
            account_state = self.api.get_account_state()
            
            # Update account metrics
            self.widgets['account_value'].configure(text=f"${account_state.get('account_value', 0):,.2f}")
            
            # Calculate total PnL
            total_pnl = sum(pos.get('unrealized_pnl', 0) for pos in account_state.get('positions', []))
            pnl_color = "green" if total_pnl >= 0 else "red"
            self.widgets['total_pnl'].configure(text=f"${total_pnl:,.2f}", text_color=pnl_color)
            
            # Update margin usage
            margin_used = account_state.get('total_margin_used', 0)
            account_value = account_state.get('account_value', 1)
            margin_pct = (margin_used / account_value) * 100 if account_value > 0 else 0
            self.widgets['margin_used'].configure(text=f"{margin_pct:.1f}%")
            
            # Update open positions count
            self.widgets['open_positions'].configure(text=str(len(account_state.get('positions', []))))
            
            # Update positions table
            self.update_positions_table(account_state.get('positions', []))
            
            # Update orders table
            self.update_orders_table(account_state.get('orders', []))
            
        except Exception as e:
            logger.error(f"Failed to update account data: {e}")
    
    def update_positions_table(self, positions):
        """Update positions table"""
        # Clear existing items
        for item in self.widgets['positions_tree'].get_children():
            self.widgets['positions_tree'].delete(item)
        
        # Add positions
        for pos in positions:
            pnl_pct = (pos.get('unrealized_pnl', 0) / abs(pos.get('position_value', 1))) * 100
            
            self.widgets['positions_tree'].insert("", "end", values=(
                pos.get('coin', ''),
                'Long' if pos.get('size', 0) > 0 else 'Short',
                f"{abs(pos.get('size', 0)):.4f}",
                f"${pos.get('entry_px', 0):.2f}",
                f"${pos.get('current_price', 0):.2f}",
                f"${pos.get('unrealized_pnl', 0):.2f}",
                f"{pnl_pct:.2f}%",
                "Close"
            ))
    
    def update_orders_table(self, orders):
        """Update orders table"""
        # Clear existing items
        for item in self.widgets['orders_tree'].get_children():
            self.widgets['orders_tree'].delete(item)
        
        # Add orders
        for order in orders:
            self.widgets['orders_tree'].insert("", "end", values=(
                order.get('coin', ''),
                order.get('side', ''),
                order.get('order_type', ''),
                f"{order.get('sz', 0):.4f}",
                f"${order.get('limit_px', 0):.2f}",
                "Open",
                datetime.fromtimestamp(order.get('timestamp', 0) / 1000).strftime("%H:%M:%S"),
                "Cancel"
            ))
    
    def start_update_loop(self):
        """Start the data update loop"""
        def update_loop():
            while not self.stop_updates:
                try:
                    if self.is_connected:
                        self.update_account_data()
                    time.sleep(5)  # Update every 5 seconds
                except Exception as e:
                    logger.error(f"Update loop error: {e}")
                    time.sleep(10)  # Wait longer on error
        
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
    
    def on_closing(self):
        """Handle application closing"""
        self.stop_updates = True
        
        if self.api:
            self.disconnect_api()
        
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


def main():
    """Main entry point"""
    try:
        app = TradingDashboard()
        app.run()
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        messagebox.showerror("Error", f"Application failed to start: {e}")


if __name__ == "__main__":
    main()

