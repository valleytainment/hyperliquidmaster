#!/usr/bin/env python3
"""
Ultimate Production GUI with Real-time Wallet Equity and Price Display
--------------------------------------------------------------------
Integrates all missing features from master_bot.py including:
• Real-time wallet balance monitoring
• Live token price feeds with charts
• Comprehensive position tracking
• Advanced automation controls
• Performance metrics dashboard
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import asyncio
import json
import time
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor

# Import our enhanced trading engine
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enhanced_trading_engine import EnhancedProductionTradingBot
from utils.logger import get_logger, TradingLogger
from utils.config_manager import ConfigManager
from utils.security import SecurityManager

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

logger = get_logger(__name__)
trading_logger = TradingLogger(__name__)


class UltimateProductionGUI:
    """Ultimate Production GUI with comprehensive real-time features"""
    
    def __init__(self):
        """Initialize the ultimate production GUI"""
        self.root = ctk.CTk()
        self.root.title("🚀 ULTIMATE HYPERLIQUID MASTER - Real-time Trading Dashboard")
        self.root.geometry("1400x900")
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.security_manager = SecurityManager()
        
        # Initialize trading bot with enhanced features
        self.trading_bot = EnhancedProductionTradingBot()
        
        # GUI state variables
        self.connected = False
        self.automation_running = False
        self.real_time_data = {}
        
        # Real-time update queues
        self.price_queue = queue.Queue()
        self.equity_queue = queue.Queue()
        self.log_queue = queue.Queue()
        
        # Chart data storage
        self.price_history = []
        self.equity_history = []
        self.pnl_history = []
        
        # Threading
        self.update_thread = None
        self.chart_update_thread = None
        
        # Create GUI
        self.create_gui()
        
        # Start real-time updates
        self.start_real_time_updates()
        
        logger.info("Ultimate Production GUI initialized")
    
    def create_gui(self):
        """Create the comprehensive GUI interface"""
        # Create main container with tabs
        self.tabview = ctk.CTkTabview(self.root)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_trading_tab()
        self.create_automation_tab()
        self.create_positions_tab()
        self.create_performance_tab()
        self.create_settings_tab()
        self.create_logs_tab()
    
    def create_dashboard_tab(self):
        """Create real-time dashboard tab"""
        dashboard_tab = self.tabview.add("📊 Dashboard")
        
        # Top row - Key metrics
        metrics_frame = ctk.CTkFrame(dashboard_tab)
        metrics_frame.pack(fill="x", padx=10, pady=5)
        
        # Wallet Equity Display
        equity_frame = ctk.CTkFrame(metrics_frame)
        equity_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(equity_frame, text="💰 WALLET EQUITY", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        self.equity_label = ctk.CTkLabel(equity_frame, text="$0.00", 
                                        font=ctk.CTkFont(size=24, weight="bold"),
                                        text_color="green")
        self.equity_label.pack(pady=5)
        
        self.equity_change_label = ctk.CTkLabel(equity_frame, text="(+$0.00 / +0.00%)", 
                                               font=ctk.CTkFont(size=12),
                                               text_color="gray")
        self.equity_change_label.pack()
        
        # Current Price Display
        price_frame = ctk.CTkFrame(metrics_frame)
        price_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(price_frame, text="📈 LIVE PRICE", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        self.symbol_label = ctk.CTkLabel(price_frame, text="BTC-USD-PERP", 
                                        font=ctk.CTkFont(size=14, weight="bold"))
        self.symbol_label.pack()
        
        self.price_label = ctk.CTkLabel(price_frame, text="$0.00", 
                                       font=ctk.CTkFont(size=24, weight="bold"),
                                       text_color="blue")
        self.price_label.pack(pady=5)
        
        self.price_change_label = ctk.CTkLabel(price_frame, text="(+0.00%)", 
                                              font=ctk.CTkFont(size=12),
                                              text_color="gray")
        self.price_change_label.pack()
        
        # Performance Metrics
        performance_frame = ctk.CTkFrame(metrics_frame)
        performance_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(performance_frame, text="📊 PERFORMANCE", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        self.total_pnl_label = ctk.CTkLabel(performance_frame, text="Total P&L: $0.00", 
                                           font=ctk.CTkFont(size=14))
        self.total_pnl_label.pack()
        
        self.win_rate_label = ctk.CTkLabel(performance_frame, text="Win Rate: 0%", 
                                          font=ctk.CTkFont(size=14))
        self.win_rate_label.pack()
        
        self.total_trades_label = ctk.CTkLabel(performance_frame, text="Total Trades: 0", 
                                              font=ctk.CTkFont(size=14))
        self.total_trades_label.pack()
        
        # Status and Connection
        status_frame = ctk.CTkFrame(metrics_frame)
        status_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(status_frame, text="🔗 STATUS", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        self.connection_status = ctk.CTkLabel(status_frame, text="❌ Disconnected", 
                                             font=ctk.CTkFont(size=14),
                                             text_color="red")
        self.connection_status.pack()
        
        self.automation_status = ctk.CTkLabel(status_frame, text="⏸️ Automation Off", 
                                             font=ctk.CTkFont(size=14),
                                             text_color="gray")
        self.automation_status.pack()
        
        self.circuit_breaker_status = ctk.CTkLabel(status_frame, text="🛡️ Circuit Breaker OK", 
                                                  font=ctk.CTkFont(size=14),
                                                  text_color="green")
        self.circuit_breaker_status.pack()
        
        # Charts section
        charts_frame = ctk.CTkFrame(dashboard_tab)
        charts_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Price Chart
        price_chart_frame = ctk.CTkFrame(charts_frame)
        price_chart_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(price_chart_frame, text="📈 LIVE PRICE CHART", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        self.price_fig, self.price_ax = plt.subplots(figsize=(6, 4), facecolor='#2b2b2b')
        self.price_ax.set_facecolor('#2b2b2b')
        self.price_canvas = FigureCanvasTkAgg(self.price_fig, price_chart_frame)
        self.price_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Equity Chart
        equity_chart_frame = ctk.CTkFrame(charts_frame)
        equity_chart_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(equity_chart_frame, text="💰 EQUITY CHART", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        self.equity_fig, self.equity_ax = plt.subplots(figsize=(6, 4), facecolor='#2b2b2b')
        self.equity_ax.set_facecolor('#2b2b2b')
        self.equity_canvas = FigureCanvasTkAgg(self.equity_fig, equity_chart_frame)
        self.equity_canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def create_trading_tab(self):
        """Create manual trading tab"""
        trading_tab = self.tabview.add("💹 Trading")
        
        # Connection controls
        connection_frame = ctk.CTkFrame(trading_tab)
        connection_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(connection_frame, text="🔗 API CONNECTION", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        conn_buttons_frame = ctk.CTkFrame(connection_frame)
        conn_buttons_frame.pack(pady=5)
        
        self.connect_btn = ctk.CTkButton(conn_buttons_frame, text="🔗 Connect", 
                                        command=self.connect_api, fg_color="green")
        self.connect_btn.pack(side="left", padx=5)
        
        self.disconnect_btn = ctk.CTkButton(conn_buttons_frame, text="❌ Disconnect", 
                                           command=self.disconnect_api, fg_color="red")
        self.disconnect_btn.pack(side="left", padx=5)
        
        # Symbol selection
        symbol_frame = ctk.CTkFrame(trading_tab)
        symbol_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(symbol_frame, text="📊 TRADING SYMBOL", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        symbol_controls_frame = ctk.CTkFrame(symbol_frame)
        symbol_controls_frame.pack(pady=5)
        
        ctk.CTkLabel(symbol_controls_frame, text="Symbol:").pack(side="left", padx=5)
        
        self.symbol_var = ctk.StringVar(value="BTC-USD-PERP")
        self.symbol_entry = ctk.CTkEntry(symbol_controls_frame, textvariable=self.symbol_var, width=150)
        self.symbol_entry.pack(side="left", padx=5)
        
        ctk.CTkLabel(symbol_controls_frame, text="Mode:").pack(side="left", padx=5)
        
        self.mode_var = ctk.StringVar(value="perp")
        self.mode_dropdown = ctk.CTkOptionMenu(symbol_controls_frame, variable=self.mode_var,
                                              values=["perp", "spot"])
        self.mode_dropdown.pack(side="left", padx=5)
        
        ctk.CTkButton(symbol_controls_frame, text="Set Symbol", 
                     command=self.set_symbol).pack(side="left", padx=5)
        
        # Manual trading controls
        manual_frame = ctk.CTkFrame(trading_tab)
        manual_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(manual_frame, text="🎯 MANUAL TRADING", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        manual_controls_frame = ctk.CTkFrame(manual_frame)
        manual_controls_frame.pack(pady=5)
        
        ctk.CTkLabel(manual_controls_frame, text="Size ($):").pack(side="left", padx=5)
        
        self.trade_size_var = ctk.StringVar(value="20.0")
        self.trade_size_entry = ctk.CTkEntry(manual_controls_frame, textvariable=self.trade_size_var, width=100)
        self.trade_size_entry.pack(side="left", padx=5)
        
        self.buy_btn = ctk.CTkButton(manual_controls_frame, text="🟢 BUY", 
                                    command=lambda: self.execute_manual_trade("BUY"), 
                                    fg_color="green", width=100)
        self.buy_btn.pack(side="left", padx=5)
        
        self.sell_btn = ctk.CTkButton(manual_controls_frame, text="🔴 SELL", 
                                     command=lambda: self.execute_manual_trade("SELL"), 
                                     fg_color="red", width=100)
        self.sell_btn.pack(side="left", padx=5)
        
        self.close_all_btn = ctk.CTkButton(manual_controls_frame, text="❌ Close All", 
                                          command=self.close_all_positions, 
                                          fg_color="orange", width=100)
        self.close_all_btn.pack(side="left", padx=5)
    
    def create_automation_tab(self):
        """Create 24/7 automation tab"""
        automation_tab = self.tabview.add("🤖 Automation")
        
        # Automation controls
        auto_frame = ctk.CTkFrame(automation_tab)
        auto_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(auto_frame, text="🤖 24/7 AUTOMATED TRADING", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        auto_controls_frame = ctk.CTkFrame(auto_frame)
        auto_controls_frame.pack(pady=5)
        
        ctk.CTkLabel(auto_controls_frame, text="Trading Mode:").pack(side="left", padx=5)
        
        self.auto_mode_var = ctk.StringVar(value="perp")
        self.auto_mode_dropdown = ctk.CTkOptionMenu(auto_controls_frame, variable=self.auto_mode_var,
                                                   values=["perp", "spot"])
        self.auto_mode_dropdown.pack(side="left", padx=5)
        
        ctk.CTkLabel(auto_controls_frame, text="Strategy:").pack(side="left", padx=5)
        
        self.strategy_var = ctk.StringVar(value="Enhanced Neural")
        self.strategy_dropdown = ctk.CTkOptionMenu(auto_controls_frame, variable=self.strategy_var,
                                                  values=["Enhanced Neural", "BB RSI ADX", "Hull Suite"])
        self.strategy_dropdown.pack(side="left", padx=5)
        
        self.start_auto_btn = ctk.CTkButton(auto_controls_frame, text="🚀 START AUTOMATION", 
                                           command=self.start_automation, 
                                           fg_color="green", width=150)
        self.start_auto_btn.pack(side="left", padx=10)
        
        self.stop_auto_btn = ctk.CTkButton(auto_controls_frame, text="⏹️ STOP AUTOMATION", 
                                          command=self.stop_automation, 
                                          fg_color="red", width=150)
        self.stop_auto_btn.pack(side="left", padx=5)
        
        # Automation settings
        settings_frame = ctk.CTkFrame(automation_tab)
        settings_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(settings_frame, text="⚙️ AUTOMATION SETTINGS", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        settings_grid = ctk.CTkFrame(settings_frame)
        settings_grid.pack(pady=5)
        
        # Starting capital
        ctk.CTkLabel(settings_grid, text="Starting Capital ($):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.starting_capital_var = ctk.StringVar(value="100.0")
        ctk.CTkEntry(settings_grid, textvariable=self.starting_capital_var, width=100).grid(row=0, column=1, padx=5, pady=5)
        
        # Position size
        ctk.CTkLabel(settings_grid, text="Position Size ($):").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.position_size_var = ctk.StringVar(value="20.0")
        ctk.CTkEntry(settings_grid, textvariable=self.position_size_var, width=100).grid(row=0, column=3, padx=5, pady=5)
        
        # Stop loss
        ctk.CTkLabel(settings_grid, text="Stop Loss (%):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.stop_loss_var = ctk.StringVar(value="2.0")
        ctk.CTkEntry(settings_grid, textvariable=self.stop_loss_var, width=100).grid(row=1, column=1, padx=5, pady=5)
        
        # Take profit
        ctk.CTkLabel(settings_grid, text="Take Profit (%):").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        self.take_profit_var = ctk.StringVar(value="4.0")
        ctk.CTkEntry(settings_grid, textvariable=self.take_profit_var, width=100).grid(row=1, column=3, padx=5, pady=5)
        
        # Circuit breaker
        ctk.CTkLabel(settings_grid, text="Circuit Breaker (%):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.circuit_breaker_var = ctk.StringVar(value="10.0")
        ctk.CTkEntry(settings_grid, textvariable=self.circuit_breaker_var, width=100).grid(row=2, column=1, padx=5, pady=5)
        
        # Trailing stop
        self.trailing_stop_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(settings_grid, text="Enable Trailing Stop", 
                       variable=self.trailing_stop_var).grid(row=2, column=2, columnspan=2, padx=5, pady=5, sticky="w")
        
        ctk.CTkButton(settings_grid, text="💾 Save Settings", 
                     command=self.save_automation_settings).grid(row=3, column=0, columnspan=4, pady=10)
    
    def create_positions_tab(self):
        """Create positions monitoring tab"""
        positions_tab = self.tabview.add("📋 Positions")
        
        # Positions header
        header_frame = ctk.CTkFrame(positions_tab)
        header_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(header_frame, text="📋 OPEN POSITIONS", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        # Positions table
        self.positions_frame = ctk.CTkScrollableFrame(positions_tab)
        self.positions_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create positions table headers
        headers = ["Symbol", "Side", "Size", "Entry Price", "Current Price", "P&L", "P&L %", "Actions"]
        for i, header in enumerate(headers):
            ctk.CTkLabel(self.positions_frame, text=header, 
                        font=ctk.CTkFont(weight="bold")).grid(row=0, column=i, padx=5, pady=5, sticky="w")
    
    def create_performance_tab(self):
        """Create performance analytics tab"""
        performance_tab = self.tabview.add("📈 Performance")
        
        # Performance metrics
        metrics_frame = ctk.CTkFrame(performance_tab)
        metrics_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(metrics_frame, text="📈 PERFORMANCE ANALYTICS", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        # Metrics grid
        metrics_grid = ctk.CTkFrame(metrics_frame)
        metrics_grid.pack(pady=5)
        
        # Performance labels
        self.perf_total_return = ctk.CTkLabel(metrics_grid, text="Total Return: 0.00%")
        self.perf_total_return.grid(row=0, column=0, padx=10, pady=5)
        
        self.perf_max_drawdown = ctk.CTkLabel(metrics_grid, text="Max Drawdown: 0.00%")
        self.perf_max_drawdown.grid(row=0, column=1, padx=10, pady=5)
        
        self.perf_sharpe_ratio = ctk.CTkLabel(metrics_grid, text="Sharpe Ratio: 0.00")
        self.perf_sharpe_ratio.grid(row=0, column=2, padx=10, pady=5)
        
        self.perf_profit_factor = ctk.CTkLabel(metrics_grid, text="Profit Factor: 0.00")
        self.perf_profit_factor.grid(row=1, column=0, padx=10, pady=5)
        
        self.perf_avg_win = ctk.CTkLabel(metrics_grid, text="Avg Win: $0.00")
        self.perf_avg_win.grid(row=1, column=1, padx=10, pady=5)
        
        self.perf_avg_loss = ctk.CTkLabel(metrics_grid, text="Avg Loss: $0.00")
        self.perf_avg_loss.grid(row=1, column=2, padx=10, pady=5)
        
        # Performance chart
        chart_frame = ctk.CTkFrame(performance_tab)
        chart_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        ctk.CTkLabel(chart_frame, text="📊 P&L CHART", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        self.pnl_fig, self.pnl_ax = plt.subplots(figsize=(10, 6), facecolor='#2b2b2b')
        self.pnl_ax.set_facecolor('#2b2b2b')
        self.pnl_canvas = FigureCanvasTkAgg(self.pnl_fig, chart_frame)
        self.pnl_canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def create_settings_tab(self):
        """Create settings tab"""
        settings_tab = self.tabview.add("⚙️ Settings")
        
        # API settings
        api_frame = ctk.CTkFrame(settings_tab)
        api_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(api_frame, text="🔑 API SETTINGS", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        api_grid = ctk.CTkFrame(api_frame)
        api_grid.pack(pady=5)
        
        ctk.CTkLabel(api_grid, text="Account Address:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.account_address_var = ctk.StringVar(value="0x306D29F56EA1345c7E6F1ff27657ba05cEE15D4F")
        ctk.CTkEntry(api_grid, textvariable=self.account_address_var, width=400).grid(row=0, column=1, padx=5, pady=5)
        
        ctk.CTkLabel(api_grid, text="Private Key:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.private_key_var = ctk.StringVar(value="43ba46de58067dd1ef3794c653bf3b11fa78866623cc515a5aff5f4be31fd3b8")
        private_key_entry = ctk.CTkEntry(api_grid, textvariable=self.private_key_var, width=400, show="*")
        private_key_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ctk.CTkButton(api_grid, text="💾 Save API Settings", 
                     command=self.save_api_settings).grid(row=2, column=0, columnspan=2, pady=10)
    
    def create_logs_tab(self):
        """Create logs tab"""
        logs_tab = self.tabview.add("📝 Logs")
        
        # Logs header
        header_frame = ctk.CTkFrame(logs_tab)
        header_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(header_frame, text="📝 TRADING LOGS", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        # Logs text area
        self.logs_text = ctk.CTkTextbox(logs_tab, height=400)
        self.logs_text.pack(fill="both", expand=True, padx=10, pady=5)
    
    def start_real_time_updates(self):
        """Start real-time data updates"""
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        self.chart_update_thread = threading.Thread(target=self._chart_update_loop, daemon=True)
        self.chart_update_thread.start()
        
        # Start GUI update timer
        self.root.after(1000, self.update_gui)
    
    def _update_loop(self):
        """Real-time data update loop"""
        while True:
            try:
                if self.connected:
                    # Get real-time data from trading bot
                    self.real_time_data = self.trading_bot.get_real_time_data()
                    
                    # Update price history
                    if self.real_time_data.get("price", 0) > 0:
                        self.price_history.append({
                            "timestamp": datetime.now(),
                            "price": self.real_time_data["price"]
                        })
                        
                        # Limit history
                        if len(self.price_history) > 100:
                            self.price_history = self.price_history[-50:]
                    
                    # Update equity history
                    if self.real_time_data.get("equity", 0) > 0:
                        self.equity_history.append({
                            "timestamp": datetime.now(),
                            "equity": self.real_time_data["equity"]
                        })
                        
                        # Limit history
                        if len(self.equity_history) > 100:
                            self.equity_history = self.equity_history[-50:]
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(5)
    
    def _chart_update_loop(self):
        """Chart update loop"""
        while True:
            try:
                if self.connected and len(self.price_history) > 1:
                    self.update_charts()
                
                time.sleep(5)  # Update charts every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in chart update loop: {e}")
                time.sleep(10)
    
    def update_gui(self):
        """Update GUI elements with real-time data"""
        try:
            if self.real_time_data:
                # Update equity display
                equity = self.real_time_data.get("equity", 0)
                self.equity_label.configure(text=f"${equity:.2f}")
                
                # Update price display
                price = self.real_time_data.get("price", 0)
                self.price_label.configure(text=f"${price:.2f}")
                
                # Update symbol
                symbol = self.symbol_var.get()
                self.symbol_label.configure(text=symbol)
                
                # Update performance metrics
                performance = self.real_time_data.get("performance", {})
                if performance:
                    self.total_pnl_label.configure(text=f"Total P&L: ${performance.get('total_pnl', 0):.2f}")
                    self.win_rate_label.configure(text=f"Win Rate: {performance.get('win_rate', 0):.1f}%")
                    self.total_trades_label.configure(text=f"Total Trades: {performance.get('total_trades', 0)}")
                
                # Update status indicators
                automation_status = self.real_time_data.get("automation_status", False)
                if automation_status:
                    self.automation_status.configure(text="▶️ Automation ON", text_color="green")
                else:
                    self.automation_status.configure(text="⏸️ Automation OFF", text_color="gray")
                
                circuit_breaker = self.real_time_data.get("circuit_breaker", False)
                if circuit_breaker:
                    self.circuit_breaker_status.configure(text="🚨 Circuit Breaker ACTIVE", text_color="red")
                else:
                    self.circuit_breaker_status.configure(text="🛡️ Circuit Breaker OK", text_color="green")
                
                # Update positions table
                self.update_positions_table()
            
            # Schedule next update
            self.root.after(1000, self.update_gui)
            
        except Exception as e:
            logger.error(f"Error updating GUI: {e}")
            self.root.after(1000, self.update_gui)
    
    def update_charts(self):
        """Update real-time charts"""
        try:
            # Update price chart
            if len(self.price_history) > 1:
                times = [p["timestamp"] for p in self.price_history]
                prices = [p["price"] for p in self.price_history]
                
                self.price_ax.clear()
                self.price_ax.plot(times, prices, color='#00ff00', linewidth=2)
                self.price_ax.set_title('Live Price', color='white')
                self.price_ax.tick_params(colors='white')
                self.price_ax.grid(True, alpha=0.3)
                self.price_canvas.draw()
            
            # Update equity chart
            if len(self.equity_history) > 1:
                times = [e["timestamp"] for e in self.equity_history]
                equities = [e["equity"] for e in self.equity_history]
                
                self.equity_ax.clear()
                self.equity_ax.plot(times, equities, color='#0080ff', linewidth=2)
                self.equity_ax.set_title('Wallet Equity', color='white')
                self.equity_ax.tick_params(colors='white')
                self.equity_ax.grid(True, alpha=0.3)
                self.equity_canvas.draw()
                
        except Exception as e:
            logger.error(f"Error updating charts: {e}")
    
    def update_positions_table(self):
        """Update positions table"""
        try:
            # Clear existing position widgets
            for widget in self.positions_frame.winfo_children():
                if int(widget.grid_info()["row"]) > 0:  # Keep headers
                    widget.destroy()
            
            # Get current positions
            positions = self.real_time_data.get("positions", [])
            
            for i, position in enumerate(positions, 1):
                # Symbol
                ctk.CTkLabel(self.positions_frame, text=position.get("symbol", "")).grid(row=i, column=0, padx=5, pady=2, sticky="w")
                
                # Side
                side_text = "LONG" if position.get("side") == 1 else "SHORT"
                side_color = "green" if position.get("side") == 1 else "red"
                ctk.CTkLabel(self.positions_frame, text=side_text, text_color=side_color).grid(row=i, column=1, padx=5, pady=2, sticky="w")
                
                # Size
                ctk.CTkLabel(self.positions_frame, text=f"{position.get('size', 0):.4f}").grid(row=i, column=2, padx=5, pady=2, sticky="w")
                
                # Entry Price
                ctk.CTkLabel(self.positions_frame, text=f"${position.get('entryPrice', 0):.2f}").grid(row=i, column=3, padx=5, pady=2, sticky="w")
                
                # Current Price
                ctk.CTkLabel(self.positions_frame, text=f"${position.get('currentPrice', 0):.2f}").grid(row=i, column=4, padx=5, pady=2, sticky="w")
                
                # P&L
                pnl = position.get("unrealizedPnl", 0)
                pnl_color = "green" if pnl >= 0 else "red"
                ctk.CTkLabel(self.positions_frame, text=f"${pnl:.2f}", text_color=pnl_color).grid(row=i, column=5, padx=5, pady=2, sticky="w")
                
                # P&L %
                pnl_pct = position.get("pnlPercent", 0)
                ctk.CTkLabel(self.positions_frame, text=f"{pnl_pct:.2f}%", text_color=pnl_color).grid(row=i, column=6, padx=5, pady=2, sticky="w")
                
                # Actions
                close_btn = ctk.CTkButton(self.positions_frame, text="Close", width=60, height=25,
                                         command=lambda p=position: self.close_position(p))
                close_btn.grid(row=i, column=7, padx=5, pady=2)
                
        except Exception as e:
            logger.error(f"Error updating positions table: {e}")
    
    # Event handlers
    def connect_api(self):
        """Connect to trading API"""
        try:
            if self.trading_bot.connect():
                self.connected = True
                self.connection_status.configure(text="✅ Connected", text_color="green")
                self.log_message("✅ Successfully connected to Hyperliquid API")
            else:
                self.log_message("❌ Failed to connect to API")
        except Exception as e:
            self.log_message(f"❌ Connection error: {e}")
    
    def disconnect_api(self):
        """Disconnect from trading API"""
        try:
            self.trading_bot.disconnect()
            self.connected = False
            self.connection_status.configure(text="❌ Disconnected", text_color="red")
            self.log_message("❌ Disconnected from API")
        except Exception as e:
            self.log_message(f"❌ Disconnect error: {e}")
    
    def set_symbol(self):
        """Set trading symbol"""
        try:
            symbol = self.symbol_var.get().strip().upper()
            mode = self.mode_var.get()
            
            # Update trading bot configuration
            self.trading_bot.config["trade_symbol"] = symbol
            self.trading_bot.config["trade_mode"] = mode
            
            self.log_message(f"📊 Symbol set to {symbol} ({mode.upper()} mode)")
        except Exception as e:
            self.log_message(f"❌ Error setting symbol: {e}")
    
    def execute_manual_trade(self, side: str):
        """Execute manual trade"""
        try:
            if not self.connected:
                messagebox.showerror("Error", "Not connected to API")
                return
            
            size_usd = float(self.trade_size_var.get())
            symbol = self.symbol_var.get()
            
            result = self.trading_bot.execute_manual_trade(side, size_usd, symbol)
            
            if result["success"]:
                self.log_message(f"✅ {side} order executed: ${size_usd}")
            else:
                self.log_message(f"❌ {side} order failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.log_message(f"❌ Trade execution error: {e}")
    
    def close_all_positions(self):
        """Close all open positions"""
        try:
            if not self.connected:
                messagebox.showerror("Error", "Not connected to API")
                return
            
            result = self.trading_bot.close_all_positions()
            
            if result["success"]:
                self.log_message(f"✅ Closed {result['positions_closed']} positions - Total P&L: ${result['total_pnl']:.2f}")
            else:
                self.log_message(f"❌ Failed to close positions: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.log_message(f"❌ Error closing positions: {e}")
    
    def close_position(self, position):
        """Close specific position"""
        try:
            # Implementation for closing specific position
            self.log_message(f"🔄 Closing position: {position.get('symbol', 'Unknown')}")
        except Exception as e:
            self.log_message(f"❌ Error closing position: {e}")
    
    def start_automation(self):
        """Start 24/7 automation"""
        try:
            if not self.connected:
                messagebox.showerror("Error", "Not connected to API")
                return
            
            mode = self.auto_mode_var.get()
            strategy = self.strategy_var.get()
            
            # Update settings
            self.update_automation_config()
            
            # Start automation
            self.trading_bot.start_automation(mode, strategy)
            self.automation_running = True
            
            self.log_message(f"🚀 Started 24/7 automation - Mode: {mode.upper()}, Strategy: {strategy}")
            
        except Exception as e:
            self.log_message(f"❌ Failed to start automation: {e}")
    
    def stop_automation(self):
        """Stop automation"""
        try:
            self.trading_bot.stop_automation()
            self.automation_running = False
            
            self.log_message("⏹️ Automation stopped")
            
        except Exception as e:
            self.log_message(f"❌ Failed to stop automation: {e}")
    
    def update_automation_config(self):
        """Update automation configuration"""
        try:
            self.trading_bot.config.update({
                "starting_capital": float(self.starting_capital_var.get()),
                "manual_entry_size": float(self.position_size_var.get()),
                "stop_loss_pct": float(self.stop_loss_var.get()) / 100,
                "take_profit_pct": float(self.take_profit_var.get()) / 100,
                "circuit_breaker_threshold": float(self.circuit_breaker_var.get()) / 100,
                "use_trailing_stop": self.trailing_stop_var.get()
            })
        except Exception as e:
            self.log_message(f"❌ Error updating config: {e}")
    
    def save_automation_settings(self):
        """Save automation settings"""
        try:
            self.update_automation_config()
            self.log_message("💾 Automation settings saved")
        except Exception as e:
            self.log_message(f"❌ Error saving settings: {e}")
    
    def save_api_settings(self):
        """Save API settings"""
        try:
            self.trading_bot.config.update({
                "account_address": self.account_address_var.get(),
                "secret_key": self.private_key_var.get()
            })
            self.log_message("💾 API settings saved")
        except Exception as e:
            self.log_message(f"❌ Error saving API settings: {e}")
    
    def log_message(self, message: str):
        """Add message to logs"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"
            
            self.logs_text.insert("end", log_entry)
            self.logs_text.see("end")
            
            logger.info(message)
            
        except Exception as e:
            logger.error(f"Error logging message: {e}")
    
    def run(self):
        """Run the GUI application"""
        try:
            # Initialize trading bot
            if self.trading_bot.initialize():
                self.log_message("🚀 Ultimate Production GUI started successfully")
                self.log_message("💡 Connect to API and start trading!")
            else:
                self.log_message("❌ Failed to initialize trading bot")
            
            # Start GUI main loop
            self.root.mainloop()
            
        except Exception as e:
            logger.error(f"Error running GUI: {e}")
        finally:
            # Cleanup
            try:
                self.trading_bot.shutdown()
            except:
                pass


def main():
    """Main entry point"""
    try:
        app = UltimateProductionGUI()
        app.run()
    except Exception as e:
        logger.error(f"Failed to start application: {e}")


if __name__ == "__main__":
    main()

