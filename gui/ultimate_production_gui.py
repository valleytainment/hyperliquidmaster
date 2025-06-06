#!/usr/bin/env python3
"""
Ultimate Production GUI - Complete Trading Interface
---------------------------------------------------
Features:
• Live wallet balance monitoring
• Real-time price feeds for selected tokens
• 24/7 automation controls with start/stop buttons
• Spot and Perp trading mode selection
• Neural network strategy integration
• Advanced risk management controls
• Real-time P&L tracking and charts
• Comprehensive settings and configuration
• Professional dark theme interface
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import queue
import json
import logging
from datetime import datetime
from typing import Dict, Optional, List
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

# Set matplotlib backend for GUI
import matplotlib
matplotlib.use('TkAgg')

logger = logging.getLogger(__name__)

class UltimateProductionGUI:
    """Ultimate Production GUI with all advanced features"""
    
    def __init__(self, trading_bot):
        self.trading_bot = trading_bot
        self.root = None
        self.running = False
        self.automation_running = False
        
        # GUI state
        self.current_symbol = "BTC-USD-PERP"
        self.current_mode = "perp"
        self.starting_capital = 100.0
        self.current_price = 0.0
        self.wallet_balance = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        # Data storage
        self.price_history = []
        self.pnl_history = []
        self.trade_history = []
        
        # Threading
        self.update_thread = None
        self.log_queue = queue.Queue()
        
        # GUI components
        self.widgets = {}
        self.charts = {}
        
        # Style configuration
        self.colors = {
            'bg_primary': '#1e1e1e',
            'bg_secondary': '#2d2d2d',
            'bg_tertiary': '#3d3d3d',
            'text_primary': '#ffffff',
            'text_secondary': '#cccccc',
            'accent_green': '#00ff88',
            'accent_red': '#ff4444',
            'accent_blue': '#4488ff',
            'accent_yellow': '#ffaa00'
        }
    
    def create_gui(self):
        """Create the main GUI interface"""
        self.root = tk.Tk()
        self.root.title("🚀 HYPERLIQUID MASTER - Ultimate Trading Bot")
        self.root.geometry("1400x900")
        self.root.configure(bg=self.colors['bg_primary'])
        
        # Configure styles
        self._configure_styles()
        
        # Create main layout
        self._create_main_layout()
        
        # Start update thread
        self._start_update_thread()
        
        logger.info("Ultimate Production GUI created successfully")
    
    def _configure_styles(self):
        """Configure ttk styles for dark theme"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles
        style.configure('Dark.TFrame', background=self.colors['bg_primary'])
        style.configure('Dark.TLabel', background=self.colors['bg_primary'], foreground=self.colors['text_primary'])
        style.configure('Dark.TButton', background=self.colors['bg_secondary'], foreground=self.colors['text_primary'])
        style.configure('Dark.TEntry', background=self.colors['bg_secondary'], foreground=self.colors['text_primary'])
        style.configure('Dark.TCombobox', background=self.colors['bg_secondary'], foreground=self.colors['text_primary'])
        
        # Special button styles
        style.configure('Green.TButton', background=self.colors['accent_green'], foreground='black')
        style.configure('Red.TButton', background=self.colors['accent_red'], foreground='white')
        style.configure('Blue.TButton', background=self.colors['accent_blue'], foreground='white')
    
    def _create_main_layout(self):
        """Create the main layout with all components"""
        # Main container
        main_frame = ttk.Frame(self.root, style='Dark.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top status bar
        self._create_status_bar(main_frame)
        
        # Main content area with tabs
        self._create_tabbed_interface(main_frame)
        
        # Bottom control panel
        self._create_control_panel(main_frame)
    
    def _create_status_bar(self, parent):
        """Create top status bar with key information"""
        status_frame = ttk.Frame(parent, style='Dark.TFrame')
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Connection status
        self.widgets['connection_status'] = ttk.Label(
            status_frame, text="🔴 Disconnected", style='Dark.TLabel', font=('Arial', 12, 'bold')
        )
        self.widgets['connection_status'].pack(side=tk.LEFT)
        
        # Wallet balance
        self.widgets['wallet_balance'] = ttk.Label(
            status_frame, text=f"💰 Balance: ${self.wallet_balance:.2f}", 
            style='Dark.TLabel', font=('Arial', 12, 'bold')
        )
        self.widgets['wallet_balance'].pack(side=tk.LEFT, padx=(20, 0))
        
        # Current price
        self.widgets['current_price'] = ttk.Label(
            status_frame, text=f"📈 {self.current_symbol}: ${self.current_price:.2f}", 
            style='Dark.TLabel', font=('Arial', 12, 'bold')
        )
        self.widgets['current_price'].pack(side=tk.LEFT, padx=(20, 0))
        
        # P&L
        self.widgets['pnl_display'] = ttk.Label(
            status_frame, text=f"💹 P&L: ${self.unrealized_pnl:.2f}", 
            style='Dark.TLabel', font=('Arial', 12, 'bold')
        )
        self.widgets['pnl_display'].pack(side=tk.LEFT, padx=(20, 0))
        
        # Automation status
        self.widgets['automation_status'] = ttk.Label(
            status_frame, text="🤖 Manual Mode", style='Dark.TLabel', font=('Arial', 12, 'bold')
        )
        self.widgets['automation_status'].pack(side=tk.RIGHT)
    
    def _create_tabbed_interface(self, parent):
        """Create tabbed interface with all features"""
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Trading tab
        trading_frame = ttk.Frame(notebook, style='Dark.TFrame')
        notebook.add(trading_frame, text="🚀 Trading")
        self._create_trading_tab(trading_frame)
        
        # Automation tab
        automation_frame = ttk.Frame(notebook, style='Dark.TFrame')
        notebook.add(automation_frame, text="🤖 Automation")
        self._create_automation_tab(automation_frame)
        
        # Positions tab
        positions_frame = ttk.Frame(notebook, style='Dark.TFrame')
        notebook.add(positions_frame, text="📊 Positions")
        self._create_positions_tab(positions_frame)
        
        # Settings tab
        settings_frame = ttk.Frame(notebook, style='Dark.TFrame')
        notebook.add(settings_frame, text="⚙️ Settings")
        self._create_settings_tab(settings_frame)
        
        # Logs tab
        logs_frame = ttk.Frame(notebook, style='Dark.TFrame')
        notebook.add(logs_frame, text="📝 Logs")
        self._create_logs_tab(logs_frame)
    
    def _create_trading_tab(self, parent):
        """Create manual trading controls"""
        # Symbol selection
        symbol_frame = ttk.LabelFrame(parent, text="Symbol & Mode", style='Dark.TFrame')
        symbol_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(symbol_frame, text="Symbol:", style='Dark.TLabel').grid(row=0, column=0, padx=5, pady=5)
        self.widgets['symbol_entry'] = ttk.Entry(symbol_frame, style='Dark.TEntry', width=20)
        self.widgets['symbol_entry'].insert(0, self.current_symbol)
        self.widgets['symbol_entry'].grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(symbol_frame, text="Mode:", style='Dark.TLabel').grid(row=0, column=2, padx=5, pady=5)
        self.widgets['mode_combo'] = ttk.Combobox(symbol_frame, values=["spot", "perp"], style='Dark.TCombobox', width=10)
        self.widgets['mode_combo'].set(self.current_mode)
        self.widgets['mode_combo'].grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Button(symbol_frame, text="Update", command=self._update_symbol, style='Blue.TButton').grid(row=0, column=4, padx=5, pady=5)
        
        # Live price display
        price_frame = ttk.LabelFrame(parent, text="Live Price Feed", style='Dark.TFrame')
        price_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.widgets['live_price'] = ttk.Label(
            price_frame, text=f"${self.current_price:.4f}", 
            style='Dark.TLabel', font=('Arial', 24, 'bold')
        )
        self.widgets['live_price'].pack(pady=10)
        
        # Manual trading controls
        manual_frame = ttk.LabelFrame(parent, text="Manual Trading", style='Dark.TFrame')
        manual_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Order size
        ttk.Label(manual_frame, text="Order Size ($):", style='Dark.TLabel').grid(row=0, column=0, padx=5, pady=5)
        self.widgets['order_size'] = ttk.Entry(manual_frame, style='Dark.TEntry', width=15)
        self.widgets['order_size'].insert(0, "20.00")
        self.widgets['order_size'].grid(row=0, column=1, padx=5, pady=5)
        
        # Buy/Sell buttons
        ttk.Button(manual_frame, text="🟢 BUY", command=lambda: self._manual_trade("BUY"), 
                  style='Green.TButton', width=15).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(manual_frame, text="🔴 SELL", command=lambda: self._manual_trade("SELL"), 
                  style='Red.TButton', width=15).grid(row=0, column=3, padx=5, pady=5)
        
        # Close position button
        ttk.Button(manual_frame, text="❌ Close Position", command=self._close_position, 
                  style='Red.TButton', width=20).grid(row=1, column=1, columnspan=2, padx=5, pady=5)
    
    def _create_automation_tab(self, parent):
        """Create 24/7 automation controls"""
        # Main automation controls
        auto_frame = ttk.LabelFrame(parent, text="24/7 Automation Controls", style='Dark.TFrame')
        auto_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Start/Stop buttons
        button_frame = ttk.Frame(auto_frame, style='Dark.TFrame')
        button_frame.pack(pady=20)
        
        self.widgets['start_button'] = ttk.Button(
            button_frame, text="🚀 START 24/7 TRADING", 
            command=self._start_automation, style='Green.TButton',
            width=25
        )
        self.widgets['start_button'].pack(side=tk.LEFT, padx=10)
        
        self.widgets['stop_button'] = ttk.Button(
            button_frame, text="⏹️ STOP TRADING", 
            command=self._stop_automation, style='Red.TButton',
            width=25, state=tk.DISABLED
        )
        self.widgets['stop_button'].pack(side=tk.LEFT, padx=10)
        
        # Mode selection for automation
        mode_frame = ttk.LabelFrame(parent, text="Automation Mode", style='Dark.TFrame')
        mode_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.widgets['auto_mode'] = tk.StringVar(value="perp")
        ttk.Radiobutton(mode_frame, text="🔄 Auto Spot Trading", variable=self.widgets['auto_mode'], 
                       value="spot", style='Dark.TLabel').pack(anchor=tk.W, padx=10, pady=5)
        ttk.Radiobutton(mode_frame, text="📈 Auto Perp Trading", variable=self.widgets['auto_mode'], 
                       value="perp", style='Dark.TLabel').pack(anchor=tk.W, padx=10, pady=5)
        
        # Strategy selection
        strategy_frame = ttk.LabelFrame(parent, text="Trading Strategy", style='Dark.TFrame')
        strategy_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.widgets['strategy_combo'] = ttk.Combobox(
            strategy_frame, 
            values=["Enhanced Neural Network", "BB RSI ADX", "Hull Suite"], 
            style='Dark.TCombobox', width=30
        )
        self.widgets['strategy_combo'].set("Enhanced Neural Network")
        self.widgets['strategy_combo'].pack(pady=10)
        
        # Performance metrics
        perf_frame = ttk.LabelFrame(parent, text="Performance Metrics", style='Dark.TFrame')
        perf_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        metrics_text = tk.Text(perf_frame, height=8, bg=self.colors['bg_secondary'], 
                              fg=self.colors['text_primary'], font=('Courier', 10))
        metrics_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.widgets['metrics_text'] = metrics_text
    
    def _create_positions_tab(self, parent):
        """Create positions monitoring and P&L tracking"""
        # Current positions
        pos_frame = ttk.LabelFrame(parent, text="Current Positions", style='Dark.TFrame')
        pos_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Position table
        columns = ("Symbol", "Side", "Size", "Entry Price", "Current Price", "Unrealized P&L", "%")
        self.widgets['position_tree'] = ttk.Treeview(pos_frame, columns=columns, show='headings', height=6)
        
        for col in columns:
            self.widgets['position_tree'].heading(col, text=col)
            self.widgets['position_tree'].column(col, width=120)
        
        self.widgets['position_tree'].pack(fill=tk.X, padx=5, pady=5)
        
        # P&L Chart
        chart_frame = ttk.LabelFrame(parent, text="P&L Chart", style='Dark.TFrame')
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib figure
        self.charts['pnl_fig'] = Figure(figsize=(12, 6), facecolor=self.colors['bg_secondary'])
        self.charts['pnl_ax'] = self.charts['pnl_fig'].add_subplot(111)
        self.charts['pnl_ax'].set_facecolor(self.colors['bg_tertiary'])
        self.charts['pnl_ax'].tick_params(colors=self.colors['text_primary'])
        
        self.charts['pnl_canvas'] = FigureCanvasTkAgg(self.charts['pnl_fig'], chart_frame)
        self.charts['pnl_canvas'].get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Trade history
        history_frame = ttk.LabelFrame(parent, text="Trade History", style='Dark.TFrame')
        history_frame.pack(fill=tk.X, padx=10, pady=10)
        
        hist_columns = ("Time", "Symbol", "Side", "Size", "Price", "P&L", "Status")
        self.widgets['history_tree'] = ttk.Treeview(history_frame, columns=hist_columns, show='headings', height=6)
        
        for col in hist_columns:
            self.widgets['history_tree'].heading(col, text=col)
            self.widgets['history_tree'].column(col, width=100)
        
        self.widgets['history_tree'].pack(fill=tk.X, padx=5, pady=5)
    
    def _create_settings_tab(self, parent):
        """Create comprehensive settings"""
        # Capital settings
        capital_frame = ttk.LabelFrame(parent, text="Capital Management", style='Dark.TFrame')
        capital_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(capital_frame, text="Starting Capital ($):", style='Dark.TLabel').grid(row=0, column=0, padx=5, pady=5)
        self.widgets['starting_capital'] = ttk.Entry(capital_frame, style='Dark.TEntry', width=15)
        self.widgets['starting_capital'].insert(0, str(self.starting_capital))
        self.widgets['starting_capital'].grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(capital_frame, text="Update Capital", command=self._update_capital, 
                  style='Blue.TButton').grid(row=0, column=2, padx=5, pady=5)
        
        # Risk management
        risk_frame = ttk.LabelFrame(parent, text="Risk Management", style='Dark.TFrame')
        risk_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Stop loss
        ttk.Label(risk_frame, text="Stop Loss (%):", style='Dark.TLabel').grid(row=0, column=0, padx=5, pady=5)
        self.widgets['stop_loss'] = ttk.Entry(risk_frame, style='Dark.TEntry', width=10)
        self.widgets['stop_loss'].insert(0, "2.0")
        self.widgets['stop_loss'].grid(row=0, column=1, padx=5, pady=5)
        
        # Take profit
        ttk.Label(risk_frame, text="Take Profit (%):", style='Dark.TLabel').grid(row=0, column=2, padx=5, pady=5)
        self.widgets['take_profit'] = ttk.Entry(risk_frame, style='Dark.TEntry', width=10)
        self.widgets['take_profit'].insert(0, "4.0")
        self.widgets['take_profit'].grid(row=0, column=3, padx=5, pady=5)
        
        # Position size
        ttk.Label(risk_frame, text="Position Size ($):", style='Dark.TLabel').grid(row=1, column=0, padx=5, pady=5)
        self.widgets['position_size'] = ttk.Entry(risk_frame, style='Dark.TEntry', width=10)
        self.widgets['position_size'].insert(0, "20.0")
        self.widgets['position_size'].grid(row=1, column=1, padx=5, pady=5)
        
        # Max positions
        ttk.Label(risk_frame, text="Max Positions:", style='Dark.TLabel').grid(row=1, column=2, padx=5, pady=5)
        self.widgets['max_positions'] = ttk.Entry(risk_frame, style='Dark.TEntry', width=10)
        self.widgets['max_positions'].insert(0, "3")
        self.widgets['max_positions'].grid(row=1, column=3, padx=5, pady=5)
        
        # API settings
        api_frame = ttk.LabelFrame(parent, text="API Configuration", style='Dark.TFrame')
        api_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(api_frame, text="Wallet Address:", style='Dark.TLabel').grid(row=0, column=0, padx=5, pady=5)
        self.widgets['wallet_address'] = ttk.Entry(api_frame, style='Dark.TEntry', width=50)
        self.widgets['wallet_address'].insert(0, "0x306D29F56EA1345c7E6F1ff27657ba05cEE15D4F")
        self.widgets['wallet_address'].grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(api_frame, text="Private Key:", style='Dark.TLabel').grid(row=1, column=0, padx=5, pady=5)
        self.widgets['private_key'] = ttk.Entry(api_frame, style='Dark.TEntry', width=50, show="*")
        self.widgets['private_key'].insert(0, "43ba46de58067dd1ef3794c653bf3b11fa78866623cc515a5aff5f4be31fd3b8")
        self.widgets['private_key'].grid(row=1, column=1, padx=5, pady=5)
        
        # Save settings button
        ttk.Button(api_frame, text="💾 Save Settings", command=self._save_settings, 
                  style='Blue.TButton').grid(row=2, column=1, padx=5, pady=10)
        
        # Neural network settings
        nn_frame = ttk.LabelFrame(parent, text="Neural Network Settings", style='Dark.TFrame')
        nn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(nn_frame, text="Confidence Threshold:", style='Dark.TLabel').grid(row=0, column=0, padx=5, pady=5)
        self.widgets['confidence_threshold'] = ttk.Entry(nn_frame, style='Dark.TEntry', width=10)
        self.widgets['confidence_threshold'].insert(0, "0.8")
        self.widgets['confidence_threshold'].grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(nn_frame, text="Lookback Bars:", style='Dark.TLabel').grid(row=0, column=2, padx=5, pady=5)
        self.widgets['lookback_bars'] = ttk.Entry(nn_frame, style='Dark.TEntry', width=10)
        self.widgets['lookback_bars'].insert(0, "30")
        self.widgets['lookback_bars'].grid(row=0, column=3, padx=5, pady=5)
    
    def _create_logs_tab(self, parent):
        """Create logs display"""
        log_frame = ttk.LabelFrame(parent, text="System Logs", style='Dark.TFrame')
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Log text area
        self.widgets['log_text'] = scrolledtext.ScrolledText(
            log_frame, 
            bg=self.colors['bg_secondary'], 
            fg=self.colors['text_primary'],
            font=('Courier', 9),
            wrap=tk.WORD
        )
        self.widgets['log_text'].pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Log controls
        log_controls = ttk.Frame(log_frame, style='Dark.TFrame')
        log_controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(log_controls, text="Clear Logs", command=self._clear_logs, 
                  style='Dark.TButton').pack(side=tk.LEFT)
        ttk.Button(log_controls, text="Save Logs", command=self._save_logs, 
                  style='Dark.TButton').pack(side=tk.LEFT, padx=5)
    
    def _create_control_panel(self, parent):
        """Create bottom control panel"""
        control_frame = ttk.Frame(parent, style='Dark.TFrame')
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Emergency stop
        ttk.Button(control_frame, text="🚨 EMERGENCY STOP", command=self._emergency_stop, 
                  style='Red.TButton', width=20).pack(side=tk.LEFT)
        
        # Connection controls
        ttk.Button(control_frame, text="🔌 Connect", command=self._connect, 
                  style='Green.TButton', width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="🔌 Disconnect", command=self._disconnect, 
                  style='Red.TButton', width=15).pack(side=tk.LEFT)
        
        # Status indicator
        self.widgets['status_indicator'] = ttk.Label(
            control_frame, text="● Ready", style='Dark.TLabel', font=('Arial', 12, 'bold')
        )
        self.widgets['status_indicator'].pack(side=tk.RIGHT)
    
    def _start_update_thread(self):
        """Start the GUI update thread"""
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
    
    def _update_loop(self):
        """Main update loop for GUI"""
        while self.running:
            try:
                # Update wallet balance
                self._update_wallet_balance()
                
                # Update current price
                self._update_current_price()
                
                # Update positions
                self._update_positions()
                
                # Update P&L chart
                self._update_pnl_chart()
                
                # Update logs
                self._update_logs()
                
                # Update status
                self._update_status()
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in GUI update loop: {e}")
                time.sleep(5)
    
    def _update_wallet_balance(self):
        """Update wallet balance display"""
        try:
            if hasattr(self.trading_bot, 'api') and self.trading_bot.api:
                balance = self.trading_bot.api.get_equity()
                self.wallet_balance = balance
                
                if self.widgets.get('wallet_balance'):
                    self.root.after(0, lambda: self.widgets['wallet_balance'].config(
                        text=f"💰 Balance: ${balance:.2f}"
                    ))
        except Exception as e:
            logger.error(f"Error updating wallet balance: {e}")
    
    def _update_current_price(self):
        """Update current price display"""
        try:
            if hasattr(self.trading_bot, 'api') and self.trading_bot.api:
                # Get current price for selected symbol
                price_data = self.trading_bot.api.fetch_price_volume(self.current_symbol)
                if price_data:
                    self.current_price = price_data.get('price', 0.0)
                    
                    if self.widgets.get('current_price'):
                        self.root.after(0, lambda: self.widgets['current_price'].config(
                            text=f"📈 {self.current_symbol}: ${self.current_price:.4f}"
                        ))
                    
                    if self.widgets.get('live_price'):
                        self.root.after(0, lambda: self.widgets['live_price'].config(
                            text=f"${self.current_price:.4f}"
                        ))
                    
                    # Store price history
                    self.price_history.append({
                        'time': datetime.now(),
                        'price': self.current_price
                    })
                    
                    # Limit history size
                    if len(self.price_history) > 1000:
                        self.price_history = self.price_history[-1000:]
        except Exception as e:
            logger.error(f"Error updating current price: {e}")
    
    def _update_positions(self):
        """Update positions display"""
        try:
            if hasattr(self.trading_bot, 'api') and self.trading_bot.api:
                positions = self.trading_bot.api.get_user_positions()
                
                if self.widgets.get('position_tree'):
                    # Clear existing items
                    for item in self.widgets['position_tree'].get_children():
                        self.widgets['position_tree'].delete(item)
                    
                    # Add current positions
                    total_unrealized = 0.0
                    for pos in positions:
                        symbol = pos.get('symbol', 'Unknown')
                        side = 'LONG' if pos.get('side') == 1 else 'SHORT'
                        size = pos.get('size', 0.0)
                        entry_price = pos.get('entryPrice', 0.0)
                        current_price = self.current_price
                        
                        # Calculate unrealized P&L
                        if side == 'LONG':
                            unrealized = size * (current_price - entry_price)
                        else:
                            unrealized = size * (entry_price - current_price)
                        
                        unrealized_pct = (unrealized / (size * entry_price)) * 100 if entry_price > 0 else 0
                        total_unrealized += unrealized
                        
                        self.widgets['position_tree'].insert('', 'end', values=(
                            symbol, side, f"{size:.4f}", f"${entry_price:.4f}", 
                            f"${current_price:.4f}", f"${unrealized:.2f}", f"{unrealized_pct:.2f}%"
                        ))
                    
                    self.unrealized_pnl = total_unrealized
                    
                    # Update P&L display
                    if self.widgets.get('pnl_display'):
                        self.root.after(0, lambda: self.widgets['pnl_display'].config(
                            text=f"💹 P&L: ${total_unrealized:.2f}"
                        ))
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def _update_pnl_chart(self):
        """Update P&L chart"""
        try:
            if len(self.pnl_history) > 1:
                times = [entry['time'] for entry in self.pnl_history[-100:]]
                pnls = [entry['pnl'] for entry in self.pnl_history[-100:]]
                
                self.charts['pnl_ax'].clear()
                self.charts['pnl_ax'].plot(times, pnls, color=self.colors['accent_green'], linewidth=2)
                self.charts['pnl_ax'].axhline(y=0, color=self.colors['accent_red'], linestyle='--', alpha=0.7)
                self.charts['pnl_ax'].set_title('P&L Over Time', color=self.colors['text_primary'])
                self.charts['pnl_ax'].set_facecolor(self.colors['bg_tertiary'])
                self.charts['pnl_ax'].tick_params(colors=self.colors['text_primary'])
                
                self.charts['pnl_canvas'].draw()
        except Exception as e:
            logger.error(f"Error updating P&L chart: {e}")
    
    def _update_logs(self):
        """Update logs display"""
        try:
            while not self.log_queue.empty():
                log_message = self.log_queue.get_nowait()
                if self.widgets.get('log_text'):
                    self.root.after(0, lambda msg=log_message: self._append_log(msg))
        except Exception as e:
            logger.error(f"Error updating logs: {e}")
    
    def _append_log(self, message):
        """Append log message to display"""
        if self.widgets.get('log_text'):
            self.widgets['log_text'].insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
            self.widgets['log_text'].see(tk.END)
    
    def _update_status(self):
        """Update connection and automation status"""
        try:
            # Update connection status
            if hasattr(self.trading_bot, 'api') and self.trading_bot.api:
                if self.widgets.get('connection_status'):
                    self.root.after(0, lambda: self.widgets['connection_status'].config(
                        text="🟢 Connected (Mainnet)", foreground=self.colors['accent_green']
                    ))
            else:
                if self.widgets.get('connection_status'):
                    self.root.after(0, lambda: self.widgets['connection_status'].config(
                        text="🔴 Disconnected", foreground=self.colors['accent_red']
                    ))
            
            # Update automation status
            if self.automation_running:
                if self.widgets.get('automation_status'):
                    self.root.after(0, lambda: self.widgets['automation_status'].config(
                        text="🤖 Auto Trading", foreground=self.colors['accent_green']
                    ))
            else:
                if self.widgets.get('automation_status'):
                    self.root.after(0, lambda: self.widgets['automation_status'].config(
                        text="🤖 Manual Mode", foreground=self.colors['text_secondary']
                    ))
        except Exception as e:
            logger.error(f"Error updating status: {e}")
    
    # Event handlers
    def _update_symbol(self):
        """Update trading symbol and mode"""
        self.current_symbol = self.widgets['symbol_entry'].get().strip()
        self.current_mode = self.widgets['mode_combo'].get().strip()
        self._append_log(f"Updated symbol to {self.current_symbol} ({self.current_mode})")
    
    def _manual_trade(self, side):
        """Execute manual trade"""
        try:
            order_size = float(self.widgets['order_size'].get())
            self._append_log(f"Executing manual {side} order: ${order_size}")
            
            # Execute trade through trading bot
            if hasattr(self.trading_bot, 'execute_manual_trade'):
                result = self.trading_bot.execute_manual_trade(side, order_size, self.current_symbol)
                self._append_log(f"Trade result: {result}")
            else:
                self._append_log("Manual trading not implemented in trading bot")
                
        except ValueError:
            messagebox.showerror("Error", "Invalid order size")
        except Exception as e:
            messagebox.showerror("Error", f"Trade execution failed: {e}")
    
    def _close_position(self):
        """Close current position"""
        try:
            self._append_log("Closing all positions...")
            if hasattr(self.trading_bot, 'close_all_positions'):
                result = self.trading_bot.close_all_positions()
                self._append_log(f"Close positions result: {result}")
            else:
                self._append_log("Close position not implemented in trading bot")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to close position: {e}")
    
    def _start_automation(self):
        """Start 24/7 automation"""
        try:
            self.automation_running = True
            mode = self.widgets['auto_mode'].get()
            strategy = self.widgets['strategy_combo'].get()
            
            self._append_log(f"Starting 24/7 automation - Mode: {mode}, Strategy: {strategy}")
            
            # Update button states
            self.widgets['start_button'].config(state=tk.DISABLED)
            self.widgets['stop_button'].config(state=tk.NORMAL)
            
            # Start automation in trading bot
            if hasattr(self.trading_bot, 'start_automation'):
                self.trading_bot.start_automation(mode, strategy)
            else:
                self._append_log("Automation not implemented in trading bot")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start automation: {e}")
            self.automation_running = False
    
    def _stop_automation(self):
        """Stop automation"""
        try:
            self.automation_running = False
            self._append_log("Stopping automation...")
            
            # Update button states
            self.widgets['start_button'].config(state=tk.NORMAL)
            self.widgets['stop_button'].config(state=tk.DISABLED)
            
            # Stop automation in trading bot
            if hasattr(self.trading_bot, 'stop_automation'):
                self.trading_bot.stop_automation()
            else:
                self._append_log("Stop automation not implemented in trading bot")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop automation: {e}")
    
    def _update_capital(self):
        """Update starting capital"""
        try:
            self.starting_capital = float(self.widgets['starting_capital'].get())
            self._append_log(f"Updated starting capital to ${self.starting_capital:.2f}")
        except ValueError:
            messagebox.showerror("Error", "Invalid capital amount")
    
    def _save_settings(self):
        """Save all settings"""
        try:
            settings = {
                'starting_capital': float(self.widgets['starting_capital'].get()),
                'stop_loss': float(self.widgets['stop_loss'].get()),
                'take_profit': float(self.widgets['take_profit'].get()),
                'position_size': float(self.widgets['position_size'].get()),
                'max_positions': int(self.widgets['max_positions'].get()),
                'wallet_address': self.widgets['wallet_address'].get(),
                'private_key': self.widgets['private_key'].get(),
                'confidence_threshold': float(self.widgets['confidence_threshold'].get()),
                'lookback_bars': int(self.widgets['lookback_bars'].get())
            }
            
            # Save to file
            with open('gui_settings.json', 'w') as f:
                json.dump(settings, f, indent=2)
            
            self._append_log("Settings saved successfully")
            messagebox.showinfo("Success", "Settings saved successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")
    
    def _clear_logs(self):
        """Clear log display"""
        if self.widgets.get('log_text'):
            self.widgets['log_text'].delete(1.0, tk.END)
    
    def _save_logs(self):
        """Save logs to file"""
        try:
            if self.widgets.get('log_text'):
                logs = self.widgets['log_text'].get(1.0, tk.END)
                filename = f"trading_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(filename, 'w') as f:
                    f.write(logs)
                self._append_log(f"Logs saved to {filename}")
                messagebox.showinfo("Success", f"Logs saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save logs: {e}")
    
    def _emergency_stop(self):
        """Emergency stop all operations"""
        try:
            self._append_log("🚨 EMERGENCY STOP ACTIVATED 🚨")
            
            # Stop automation
            self.automation_running = False
            
            # Close all positions
            self._close_position()
            
            # Update button states
            if self.widgets.get('start_button'):
                self.widgets['start_button'].config(state=tk.NORMAL)
            if self.widgets.get('stop_button'):
                self.widgets['stop_button'].config(state=tk.DISABLED)
            
            messagebox.showwarning("Emergency Stop", "All trading operations have been stopped!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Emergency stop failed: {e}")
    
    def _connect(self):
        """Connect to trading API"""
        try:
            self._append_log("Connecting to Hyperliquid API...")
            if hasattr(self.trading_bot, 'connect'):
                result = self.trading_bot.connect()
                self._append_log(f"Connection result: {result}")
            else:
                self._append_log("Connect method not implemented in trading bot")
        except Exception as e:
            messagebox.showerror("Error", f"Connection failed: {e}")
    
    def _disconnect(self):
        """Disconnect from trading API"""
        try:
            self._append_log("Disconnecting from API...")
            if hasattr(self.trading_bot, 'disconnect'):
                result = self.trading_bot.disconnect()
                self._append_log(f"Disconnect result: {result}")
            else:
                self._append_log("Disconnect method not implemented in trading bot")
        except Exception as e:
            messagebox.showerror("Error", f"Disconnect failed: {e}")
    
    def run(self):
        """Run the GUI"""
        if self.root is None:
            self.create_gui()
        
        # Auto-connect on startup
        self._connect()
        
        # Start the main loop
        self.root.mainloop()
    
    def on_closing(self):
        """Handle GUI closing"""
        self.running = False
        if self.automation_running:
            self._stop_automation()
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2)
        
        if self.root:
            self.root.destroy()
    
    def add_log_message(self, message):
        """Add log message to queue"""
        self.log_queue.put(message)


# Alias for backward compatibility
AutoConnectTradingDashboardV2 = UltimateProductionGUI

