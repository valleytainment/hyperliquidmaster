"""
Enhanced GUI for Hyperliquid Trading Bot with Auto-Connection and Improved Feedback
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import logging
from typing import Dict, Any, Optional, Callable
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')

from utils.logger import get_logger
from utils.config_manager import ConfigManager
from utils.security import SecurityManager
from core.connection_manager_enhanced import EnhancedConnectionManager

logger = get_logger(__name__)


class AutoConnectTradingDashboardV2:
    """
    Enhanced Trading Dashboard with Auto-Connection and Improved Feedback
    """
    
    def __init__(self, connection_manager=None):
        """
        Initialize the trading dashboard
        
        Args:
            connection_manager: Connection manager instance
        """
        self.root = tk.Tk()
        self.root.title("Hyperliquid Trading Bot")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Set icon
        try:
            self.root.iconbitmap("assets/icon.ico")
        except:
            pass
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.security_manager = SecurityManager()
        self.connection_manager = connection_manager or EnhancedConnectionManager()
        
        # Initialize variables
        self.private_key_visible = False
        self.auto_connect_on_generation = tk.BooleanVar(value=True)
        self.show_balance_history = tk.BooleanVar(value=True)
        self.show_position_chart = tk.BooleanVar(value=True)
        
        # Initialize widgets dictionary
        self.widgets = {}
        
        # Initialize data
        self.balance_history = []
        self.position_history = {}
        
        # Setup GUI
        self.setup_gui()
        
        # Setup periodic tasks
        self.setup_periodic_tasks()
        
        # Update connection status
        self.update_connection_status()
        
        logger.info("Trading Dashboard initialized")
    
    def setup_gui(self):
        """Setup the GUI"""
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        dashboard_tab = ttk.Frame(notebook)
        trading_tab = ttk.Frame(notebook)
        settings_tab = ttk.Frame(notebook)
        logs_tab = ttk.Frame(notebook)
        
        notebook.add(dashboard_tab, text="Dashboard")
        notebook.add(trading_tab, text="Trading")
        notebook.add(settings_tab, text="Settings")
        notebook.add(logs_tab, text="Logs")
        
        # Setup tabs
        self.setup_dashboard_tab(dashboard_tab)
        self.setup_trading_tab(trading_tab)
        self.setup_settings_tab(settings_tab)
        self.setup_logs_tab(logs_tab)
        
        # Setup status bar
        self.setup_status_bar(main_frame)
        
        # Store notebook
        self.widgets['notebook'] = notebook
    
    def setup_dashboard_tab(self, parent):
        """Setup the dashboard tab"""
        # Create frames
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        middle_frame = ttk.Frame(parent)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        bottom_frame = ttk.Frame(parent)
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Connection status section
        connection_frame = ttk.LabelFrame(top_frame, text="Connection Status")
        connection_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Connection status indicators
        status_frame = ttk.Frame(connection_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Connection status
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        connection_status = ttk.Label(status_frame, text="Connecting...", foreground="orange")
        connection_status.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        self.widgets['connection_status'] = connection_status
        
        # Wallet address
        ttk.Label(status_frame, text="Address:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        wallet_address = ttk.Label(status_frame, text="Not connected")
        wallet_address.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        self.widgets['wallet_address'] = wallet_address
        
        # Network
        ttk.Label(status_frame, text="Network:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        network = ttk.Label(status_frame, text="Mainnet")
        network.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        self.widgets['network'] = network
        
        # Using default
        ttk.Label(status_frame, text="Using Default:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        using_default = ttk.Label(status_frame, text="Yes")
        using_default.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        self.widgets['using_default'] = using_default
        
        # Connection buttons
        button_frame = ttk.Frame(connection_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        reconnect_btn = ttk.Button(button_frame, text="Reconnect", command=self.reconnect)
        reconnect_btn.pack(side=tk.LEFT, padx=5)
        
        test_connection_btn = ttk.Button(button_frame, text="Test Connection", command=self.test_connection)
        test_connection_btn.pack(side=tk.LEFT, padx=5)
        
        switch_network_btn = ttk.Button(button_frame, text="Switch Network", command=self.switch_network)
        switch_network_btn.pack(side=tk.LEFT, padx=5)
        
        # Account summary section
        account_frame = ttk.LabelFrame(top_frame, text="Account Summary")
        account_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Account summary indicators
        summary_frame = ttk.Frame(account_frame)
        summary_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Account value
        ttk.Label(summary_frame, text="Account Value:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        account_value = ttk.Label(summary_frame, text="$0.00")
        account_value.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        self.widgets['account_value'] = account_value
        
        # Margin used
        ttk.Label(summary_frame, text="Margin Used:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        margin_used = ttk.Label(summary_frame, text="$0.00")
        margin_used.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        self.widgets['margin_used'] = margin_used
        
        # Total position
        ttk.Label(summary_frame, text="Total Position:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        total_position = ttk.Label(summary_frame, text="$0.00")
        total_position.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        self.widgets['total_position'] = total_position
        
        # Charts section
        charts_frame = ttk.LabelFrame(middle_frame, text="Charts")
        charts_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create charts
        charts_notebook = ttk.Notebook(charts_frame)
        charts_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Balance history chart
        balance_frame = ttk.Frame(charts_notebook)
        charts_notebook.add(balance_frame, text="Balance History")
        
        # Create figure for balance history
        self.balance_fig = plt.Figure(figsize=(5, 4), dpi=100)
        self.balance_ax = self.balance_fig.add_subplot(111)
        self.balance_canvas = FigureCanvasTkAgg(self.balance_fig, balance_frame)
        self.balance_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Position chart
        position_frame = ttk.Frame(charts_notebook)
        charts_notebook.add(position_frame, text="Positions")
        
        # Create figure for positions
        self.position_fig = plt.Figure(figsize=(5, 4), dpi=100)
        self.position_ax = self.position_fig.add_subplot(111)
        self.position_canvas = FigureCanvasTkAgg(self.position_fig, position_frame)
        self.position_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Chart options
        options_frame = ttk.Frame(bottom_frame)
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Balance history checkbox
        balance_check = ttk.Checkbutton(options_frame, text="Show Balance History", variable=self.show_balance_history, command=self.update_charts)
        balance_check.pack(side=tk.LEFT, padx=5)
        
        # Position chart checkbox
        position_check = ttk.Checkbutton(options_frame, text="Show Position Chart", variable=self.show_position_chart, command=self.update_charts)
        position_check.pack(side=tk.LEFT, padx=5)
        
        # Refresh button
        refresh_btn = ttk.Button(options_frame, text="Refresh", command=self.refresh_dashboard)
        refresh_btn.pack(side=tk.RIGHT, padx=5)
    
    def setup_trading_tab(self, parent):
        """Setup the trading tab"""
        # Create frames
        left_frame = ttk.Frame(parent)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        right_frame = ttk.Frame(parent)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Markets section
        markets_frame = ttk.LabelFrame(left_frame, text="Markets")
        markets_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Markets treeview
        markets_tree = ttk.Treeview(markets_frame, columns=("Price", "24h Change", "Volume"), show="headings")
        markets_tree.heading("Price", text="Price")
        markets_tree.heading("24h Change", text="24h Change")
        markets_tree.heading("Volume", text="Volume")
        markets_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.widgets['markets_tree'] = markets_tree
        
        # Positions section
        positions_frame = ttk.LabelFrame(right_frame, text="Positions")
        positions_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Positions treeview
        positions_tree = ttk.Treeview(positions_frame, columns=("Size", "Entry Price", "Mark Price", "PnL"), show="headings")
        positions_tree.heading("Size", text="Size")
        positions_tree.heading("Entry Price", text="Entry Price")
        positions_tree.heading("Mark Price", text="Mark Price")
        positions_tree.heading("PnL", text="PnL")
        positions_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.widgets['positions_tree'] = positions_tree
        
        # Orders section
        orders_frame = ttk.LabelFrame(right_frame, text="Orders")
        orders_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Orders treeview
        orders_tree = ttk.Treeview(orders_frame, columns=("Side", "Size", "Price", "Status"), show="headings")
        orders_tree.heading("Side", text="Side")
        orders_tree.heading("Size", text="Size")
        orders_tree.heading("Price", text="Price")
        orders_tree.heading("Status", text="Status")
        orders_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.widgets['orders_tree'] = orders_tree
    
    def setup_settings_tab(self, parent):
        """Setup the settings tab"""
        # Create frames
        left_frame = ttk.Frame(parent)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        right_frame = ttk.Frame(parent)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Credentials section
        credentials_frame = ttk.LabelFrame(left_frame, text="Credentials")
        credentials_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Wallet address
        ttk.Label(credentials_frame, text="Wallet Address:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        wallet_address = ttk.Entry(credentials_frame, width=50)
        wallet_address.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.widgets['wallet_address_entry'] = wallet_address
        
        # Private key
        ttk.Label(credentials_frame, text="Private Key:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        private_key_frame = ttk.Frame(credentials_frame)
        private_key_frame.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        private_key = ttk.Entry(private_key_frame, width=50, show="*")
        private_key.pack(side=tk.LEFT)
        self.widgets['private_key'] = private_key
        
        show_key_btn = ttk.Button(private_key_frame, text="Show", command=self.toggle_private_key_visibility)
        show_key_btn.pack(side=tk.LEFT, padx=5)
        self.widgets['show_key_btn'] = show_key_btn
        
        # Buttons
        buttons_frame = ttk.Frame(credentials_frame)
        buttons_frame.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        save_btn = ttk.Button(buttons_frame, text="Save Credentials", command=self.save_credentials_async)
        save_btn.pack(side=tk.LEFT, padx=5)
        
        test_btn = ttk.Button(buttons_frame, text="Test Connection", command=self.test_connection)
        test_btn.pack(side=tk.LEFT, padx=5)
        
        generate_btn = ttk.Button(buttons_frame, text="Generate New Wallet", command=self.generate_wallet)
        generate_btn.pack(side=tk.LEFT, padx=5)
        
        default_btn = ttk.Button(buttons_frame, text="Use Default", command=self.use_default_credentials)
        default_btn.pack(side=tk.LEFT, padx=5)
        
        # Auto-connect checkbox
        auto_connect_check = ttk.Checkbutton(credentials_frame, text="Auto-connect on wallet generation", variable=self.auto_connect_on_generation)
        auto_connect_check.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Network section
        network_frame = ttk.LabelFrame(left_frame, text="Network")
        network_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Network selection
        self.network_var = tk.StringVar(value="mainnet")
        mainnet_radio = ttk.Radiobutton(network_frame, text="Mainnet", variable=self.network_var, value="mainnet", command=self.update_network)
        mainnet_radio.pack(anchor=tk.W, padx=5, pady=2)
        
        testnet_radio = ttk.Radiobutton(network_frame, text="Testnet", variable=self.network_var, value="testnet", command=self.update_network)
        testnet_radio.pack(anchor=tk.W, padx=5, pady=2)
        
        # Auto-reconnect section
        reconnect_frame = ttk.LabelFrame(left_frame, text="Auto-Reconnect")
        reconnect_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Auto-reconnect selection
        self.auto_reconnect_var = tk.BooleanVar(value=True)
        enabled_radio = ttk.Radiobutton(reconnect_frame, text="Enabled", variable=self.auto_reconnect_var, value=True, command=self.update_auto_reconnect)
        enabled_radio.pack(anchor=tk.W, padx=5, pady=2)
        
        disabled_radio = ttk.Radiobutton(reconnect_frame, text="Disabled", variable=self.auto_reconnect_var, value=False, command=self.update_auto_reconnect)
        disabled_radio.pack(anchor=tk.W, padx=5, pady=2)
        
        # Trading settings section
        trading_frame = ttk.LabelFrame(right_frame, text="Trading Settings")
        trading_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Max positions
        ttk.Label(trading_frame, text="Max Positions:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        max_positions = ttk.Spinbox(trading_frame, from_=1, to=10, width=5)
        max_positions.set(3)
        max_positions.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.widgets['max_positions'] = max_positions
        
        # Risk level
        ttk.Label(trading_frame, text="Risk Level:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        risk_level = ttk.Combobox(trading_frame, values=["Low", "Medium", "High"], width=10)
        risk_level.current(1)
        risk_level.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        self.widgets['risk_level'] = risk_level
        
        # Strategy selection
        ttk.Label(trading_frame, text="Strategy:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        strategy = ttk.Combobox(trading_frame, values=["BB_RSI_ADX", "Hull_Suite"], width=15)
        strategy.current(0)
        strategy.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        self.widgets['strategy'] = strategy
        
        # Save trading settings button
        save_trading_btn = ttk.Button(trading_frame, text="Save Trading Settings", command=self.save_trading_settings)
        save_trading_btn.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Load saved credentials
        self.load_credentials()
        
        # Load saved trading settings
        self.load_trading_settings()
    
    def setup_logs_tab(self, parent):
        """Setup the logs tab"""
        # Create log text widget
        log_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD)
        log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.widgets['log_text'] = log_text
        
        # Create log handler
        log_handler = LogHandler(log_text)
        log_handler.setLevel(logging.INFO)
        
        # Add handler to logger
        logger.addHandler(log_handler)
        
        # Add initial log message
        logger.info("Log initialized")
    
    def setup_status_bar(self, parent):
        """Setup the status bar"""
        status_bar = ttk.Frame(parent)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
        
        # Status label
        status_label = ttk.Label(status_bar, text="Ready")
        status_label.pack(side=tk.LEFT)
        self.widgets['status_label'] = status_label
        
        # Connection indicator
        connection_indicator = ttk.Label(status_bar, text="●", foreground="red")
        connection_indicator.pack(side=tk.RIGHT)
        self.widgets['connection_indicator'] = connection_indicator
        
        # Version label
        version_label = ttk.Label(status_bar, text="v1.0.0")
        version_label.pack(side=tk.RIGHT, padx=10)
    
    def setup_periodic_tasks(self):
        """Setup periodic tasks"""
        # Update connection status every 5 seconds
        self.root.after(5000, self.periodic_connection_check)
        
        # Update account info every 10 seconds
        self.root.after(10000, self.periodic_account_update)
        
        # Update charts every 30 seconds
        self.root.after(30000, self.periodic_chart_update)
    
    def periodic_connection_check(self):
        """Periodic connection check"""
        try:
            # Check connection health
            self.connection_manager.check_connection_health()
            
            # Update connection status
            self.update_connection_status()
        except Exception as e:
            logger.error(f"Error in periodic connection check: {e}")
        
        # Schedule next check
        self.root.after(5000, self.periodic_connection_check)
    
    def periodic_account_update(self):
        """Periodic account update"""
        try:
            # Update account info
            self.update_account_info()
            
            # Update trading info
            self.update_trading_info()
        except Exception as e:
            logger.error(f"Error in periodic account update: {e}")
        
        # Schedule next update
        self.root.after(10000, self.periodic_account_update)
    
    def periodic_chart_update(self):
        """Periodic chart update"""
        try:
            # Update charts
            self.update_charts()
        except Exception as e:
            logger.error(f"Error in periodic chart update: {e}")
        
        # Schedule next update
        self.root.after(30000, self.periodic_chart_update)
    
    def toggle_private_key_visibility(self):
        """Toggle private key visibility"""
        try:
            if 'private_key' in self.widgets and 'show_key_btn' in self.widgets:
                private_key = self.widgets['private_key']
                show_key_btn = self.widgets['show_key_btn']
                
                if self.private_key_visible:
                    private_key.config(show="*")
                    show_key_btn.config(text="Show")
                    self.private_key_visible = False
                else:
                    private_key.config(show="")
                    show_key_btn.config(text="Hide")
                    self.private_key_visible = True
        except Exception as e:
            logger.error(f"Error toggling private key visibility: {e}")
    
    def save_credentials_async(self):
        """Save credentials asynchronously"""
        try:
            # Get credentials
            wallet_address = self.widgets['wallet_address_entry'].get().strip()
            private_key = self.widgets['private_key'].get().strip()
            
            # Validate inputs
            if not wallet_address or not private_key:
                messagebox.showerror("Error", "Please enter both wallet address and private key")
                return
            
            # Show saving message
            self.update_status("Saving credentials...")
            
            # Save credentials in a separate thread
            threading.Thread(target=self._save_credentials, args=(wallet_address, private_key)).start()
        except Exception as e:
            logger.error(f"Save credentials failed: {e}")
            messagebox.showerror("Error", f"Failed to save credentials: {e}")
            self.update_status("Ready")
    
    def _save_credentials(self, wallet_address, private_key):
        """Save credentials"""
        try:
            # Save wallet address
            self.config_manager.set('trading.wallet_address', wallet_address)
            
            # Save private key
            self.security_manager.store_private_key(private_key)
            
            # Save config
            self.config_manager.save_config()
            
            # Connect with new credentials
            if self.connection_manager._connect_with_credentials(private_key, wallet_address):
                # Update UI
                self.root.after(0, lambda: self.update_connection_status())
                self.root.after(0, lambda: self.update_account_info())
                self.root.after(0, lambda: messagebox.showinfo("Success", "Credentials saved and connected successfully"))
                self.root.after(0, lambda: self.update_status("Ready"))
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", "Failed to connect with saved credentials"))
                self.root.after(0, lambda: self.update_status("Ready"))
        except Exception as e:
            logger.error(f"Save credentials failed: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to save credentials: {e}"))
            self.root.after(0, lambda: self.update_status("Ready"))
    
    def load_credentials(self):
        """Load saved credentials"""
        try:
            # Get wallet address
            wallet_address = self.config_manager.get('trading.wallet_address', '')
            
            # Get private key
            private_key = self.security_manager.get_private_key() or ''
            
            # Update UI
            if 'wallet_address_entry' in self.widgets:
                self.widgets['wallet_address_entry'].delete(0, tk.END)
                self.widgets['wallet_address_entry'].insert(0, wallet_address)
            
            if 'private_key' in self.widgets:
                self.widgets['private_key'].delete(0, tk.END)
                self.widgets['private_key'].insert(0, private_key)
            
            # Update network selection
            testnet = self.config_manager.get('trading.testnet', False)
            self.network_var.set("testnet" if testnet else "mainnet")
            
            # Update auto-reconnect selection
            auto_reconnect = self.config_manager.get('trading.auto_reconnect', True)
            self.auto_reconnect_var.set(auto_reconnect)
        except Exception as e:
            logger.error(f"Load credentials failed: {e}")
    
    def load_trading_settings(self):
        """Load saved trading settings"""
        try:
            # Get max positions
            max_positions = self.config_manager.get('trading.max_positions', 3)
            
            # Get risk level
            risk_level = self.config_manager.get('trading.risk_level', 'Medium')
            
            # Get strategy
            strategy = self.config_manager.get('trading.strategy', 'BB_RSI_ADX')
            
            # Update UI
            if 'max_positions' in self.widgets:
                self.widgets['max_positions'].delete(0, tk.END)
                self.widgets['max_positions'].insert(0, max_positions)
            
            if 'risk_level' in self.widgets:
                self.widgets['risk_level'].set(risk_level)
            
            if 'strategy' in self.widgets:
                self.widgets['strategy'].set(strategy)
        except Exception as e:
            logger.error(f"Load trading settings failed: {e}")
    
    def save_trading_settings(self):
        """Save trading settings"""
        try:
            # Get settings
            max_positions = int(self.widgets['max_positions'].get())
            risk_level = self.widgets['risk_level'].get()
            strategy = self.widgets['strategy'].get()
            
            # Save settings
            self.config_manager.set('trading.max_positions', max_positions)
            self.config_manager.set('trading.risk_level', risk_level)
            self.config_manager.set('trading.strategy', strategy)
            
            # Save config
            self.config_manager.save_config()
            
            # Show success message
            messagebox.showinfo("Success", "Trading settings saved successfully")
        except Exception as e:
            logger.error(f"Save trading settings failed: {e}")
            messagebox.showerror("Error", f"Failed to save trading settings: {e}")
    
    def update_connection_status(self):
        """Update connection status"""
        try:
            # Get connection status
            status = self.connection_manager.get_connection_status()
            
            # Update connection status
            if 'connection_status' in self.widgets:
                if status['connected']:
                    self.widgets['connection_status'].config(text="Connected", foreground="green")
                else:
                    self.widgets['connection_status'].config(text="Disconnected", foreground="red")
            
            # Update wallet address
            if 'wallet_address' in self.widgets:
                self.widgets['wallet_address'].config(text=status['address'] or "Not connected")
            
            # Update network
            if 'network' in self.widgets:
                self.widgets['network'].config(text=status['network'].capitalize())
            
            # Update using default
            if 'using_default' in self.widgets:
                self.widgets['using_default'].config(text="Yes" if status['using_default'] else "No")
            
            # Update connection indicator
            if 'connection_indicator' in self.widgets:
                if status['connected']:
                    self.widgets['connection_indicator'].config(text="●", foreground="green")
                else:
                    self.widgets['connection_indicator'].config(text="●", foreground="red")
        except Exception as e:
            logger.error(f"Update connection status failed: {e}")
    
    def update_account_info(self):
        """Update account information"""
        try:
            # Get account state
            account_state = self.connection_manager.get_account_state()
            
            if not account_state:
                return
            
            # Update account value
            if "marginSummary" in account_state and "accountValue" in account_state["marginSummary"]:
                account_value = float(account_state["marginSummary"]["accountValue"])
                if 'account_value' in self.widgets:
                    self.widgets['account_value'].config(text=f"${account_value:.2f}")
                
                # Add to balance history
                self.balance_history.append((time.time(), account_value))
                
                # Keep only last 100 points
                if len(self.balance_history) > 100:
                    self.balance_history = self.balance_history[-100:]
            
            # Update margin used
            if "marginSummary" in account_state and "totalMarginUsed" in account_state["marginSummary"]:
                margin_used = float(account_state["marginSummary"]["totalMarginUsed"])
                if 'margin_used' in self.widgets:
                    self.widgets['margin_used'].config(text=f"${margin_used:.2f}")
            
            # Update total position
            if "marginSummary" in account_state and "totalNtlPos" in account_state["marginSummary"]:
                total_position = float(account_state["marginSummary"]["totalNtlPos"])
                if 'total_position' in self.widgets:
                    self.widgets['total_position'].config(text=f"${total_position:.2f}")
            
            # Update positions
            if "assetPositions" in account_state:
                positions = account_state["assetPositions"]
                
                # Update position history
                for pos in positions:
                    if "coin" in pos and "position" in pos:
                        coin = pos["coin"]
                        position = float(pos["position"])
                        
                        if coin not in self.position_history:
                            self.position_history[coin] = []
                        
                        self.position_history[coin].append((time.time(), position))
                        
                        # Keep only last 100 points
                        if len(self.position_history[coin]) > 100:
                            self.position_history[coin] = self.position_history[coin][-100:]
        except Exception as e:
            logger.error(f"Update account info failed: {e}")
    
    def update_trading_info(self):
        """Update trading information"""
        try:
            # Clear existing data
            if 'markets_tree' in self.widgets:
                for item in self.widgets['markets_tree'].get_children():
                    self.widgets['markets_tree'].delete(item)
            
            if 'positions_tree' in self.widgets:
                for item in self.widgets['positions_tree'].get_children():
                    self.widgets['positions_tree'].delete(item)
            
            if 'orders_tree' in self.widgets:
                for item in self.widgets['orders_tree'].get_children():
                    self.widgets['orders_tree'].delete(item)
            
            # Get markets
            try:
                markets = self.connection_manager.api.get_markets()
                
                if markets and 'markets_tree' in self.widgets:
                    for market in markets:
                        if "name" in market and "markPrice" in market:
                            name = market["name"]
                            price = float(market["markPrice"])
                            change = 0.0  # Placeholder
                            volume = 0.0  # Placeholder
                            
                            self.widgets['markets_tree'].insert("", tk.END, text=name, values=(f"${price:.2f}", f"{change:.2f}%", f"${volume:.2f}"))
            except Exception as e:
                logger.error(f"Update markets failed: {e}")
            
            # Get positions
            try:
                account_state = self.connection_manager.get_account_state()
                
                if account_state and "assetPositions" in account_state and 'positions_tree' in self.widgets:
                    positions = account_state["assetPositions"]
                    
                    for pos in positions:
                        if "coin" in pos and "position" in pos:
                            coin = pos["coin"]
                            position = float(pos["position"])
                            entry_price = 0.0  # Placeholder
                            mark_price = 0.0  # Placeholder
                            pnl = 0.0  # Placeholder
                            
                            self.widgets['positions_tree'].insert("", tk.END, text=coin, values=(position, f"${entry_price:.2f}", f"${mark_price:.2f}", f"${pnl:.2f}"))
            except Exception as e:
                logger.error(f"Update positions failed: {e}")
            
            # Get orders
            try:
                if self.connection_manager.is_connected and 'orders_tree' in self.widgets:
                    orders = self.connection_manager.api.get_open_orders(self.connection_manager.current_address)
                    
                    if orders:
                        for order in orders:
                            if "coin" in order and "side" in order and "size" in order and "price" in order:
                                coin = order["coin"]
                                side = order["side"]
                                size = float(order["size"])
                                price = float(order["price"])
                                status = "Open"  # Placeholder
                                
                                self.widgets['orders_tree'].insert("", tk.END, text=coin, values=(side, size, f"${price:.2f}", status))
            except Exception as e:
                logger.error(f"Update orders failed: {e}")
        except Exception as e:
            logger.error(f"Update trading info failed: {e}")
    
    def update_charts(self):
        """Update charts"""
        try:
            # Update balance history chart
            if self.show_balance_history.get() and self.balance_history:
                self.balance_ax.clear()
                
                times = [t[0] for t in self.balance_history]
                values = [t[1] for t in self.balance_history]
                
                # Convert timestamps to relative times
                relative_times = [(t - times[0]) / 60 for t in times]  # Minutes
                
                self.balance_ax.plot(relative_times, values, 'b-')
                self.balance_ax.set_title('Balance History')
                self.balance_ax.set_xlabel('Time (minutes)')
                self.balance_ax.set_ylabel('Balance ($)')
                self.balance_ax.grid(True)
                
                self.balance_fig.tight_layout()
                self.balance_canvas.draw()
            
            # Update position chart
            if self.show_position_chart.get() and self.position_history:
                self.position_ax.clear()
                
                for coin, history in self.position_history.items():
                    if history:
                        times = [t[0] for t in history]
                        values = [t[1] for t in history]
                        
                        # Convert timestamps to relative times
                        relative_times = [(t - times[0]) / 60 for t in times]  # Minutes
                        
                        self.position_ax.plot(relative_times, values, label=coin)
                
                self.position_ax.set_title('Position History')
                self.position_ax.set_xlabel('Time (minutes)')
                self.position_ax.set_ylabel('Position Size')
                self.position_ax.grid(True)
                self.position_ax.legend()
                
                self.position_fig.tight_layout()
                self.position_canvas.draw()
        except Exception as e:
            logger.error(f"Update charts failed: {e}")
    
    def test_connection(self):
        """Test connection"""
        try:
            # Show testing message
            self.update_status("Testing connection...")
            
            # Get credentials
            wallet_address = self.widgets['wallet_address_entry'].get().strip()
            private_key = self.widgets['private_key'].get().strip()
            
            # Validate inputs
            if not wallet_address or not private_key:
                messagebox.showerror("Error", "Please enter both wallet address and private key")
                self.update_status("Ready")
                return
            
            # Test connection in a separate thread
            threading.Thread(target=self._test_connection, args=(wallet_address, private_key)).start()
        except Exception as e:
            logger.error(f"Test connection failed: {e}")
            messagebox.showerror("Error", f"Failed to test connection: {e}")
            self.update_status("Ready")
    
    def _test_connection(self, wallet_address, private_key):
        """Test connection"""
        try:
            # Test connection
            if self.connection_manager._connect_with_credentials(private_key, wallet_address):
                # Update UI
                self.root.after(0, lambda: self.update_connection_status())
                self.root.after(0, lambda: self.update_account_info())
                self.root.after(0, lambda: messagebox.showinfo("Success", "Connection test successful"))
                self.root.after(0, lambda: self.update_status("Ready"))
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", "Connection test failed"))
                self.root.after(0, lambda: self.update_status("Ready"))
        except Exception as e:
            logger.error(f"Test connection failed: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to test connection: {e}"))
            self.root.after(0, lambda: self.update_status("Ready"))
    
    def reconnect(self):
        """Reconnect to Hyperliquid"""
        try:
            # Show reconnecting message
            self.update_status("Reconnecting...")
            
            # Reconnect in a separate thread
            threading.Thread(target=self._reconnect).start()
        except Exception as e:
            logger.error(f"Reconnect failed: {e}")
            messagebox.showerror("Error", f"Failed to reconnect: {e}")
            self.update_status("Ready")
    
    def _reconnect(self):
        """Reconnect to Hyperliquid"""
        try:
            # Disconnect
            self.connection_manager.disconnect()
            
            # Reconnect
            if self.connection_manager.ensure_connection():
                # Update UI
                self.root.after(0, lambda: self.update_connection_status())
                self.root.after(0, lambda: self.update_account_info())
                self.root.after(0, lambda: messagebox.showinfo("Success", "Reconnected successfully"))
                self.root.after(0, lambda: self.update_status("Ready"))
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", "Failed to reconnect"))
                self.root.after(0, lambda: self.update_status("Ready"))
        except Exception as e:
            logger.error(f"Reconnect failed: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to reconnect: {e}"))
            self.root.after(0, lambda: self.update_status("Ready"))
    
    def generate_wallet(self):
        """Generate a new wallet"""
        try:
            # Show generating message
            self.update_status("Generating wallet...")
            
            # Generate wallet in a separate thread
            threading.Thread(target=self._generate_wallet).start()
        except Exception as e:
            logger.error(f"Generate wallet failed: {e}")
            messagebox.showerror("Error", f"Failed to generate wallet: {e}")
            self.update_status("Ready")
    
    def _generate_wallet(self):
        """Generate a new wallet"""
        try:
            # Generate wallet
            success, address, private_key = self.connection_manager.connect_with_new_wallet()
            
            if success:
                # Update UI
                self.root.after(0, lambda: self.widgets['wallet_address_entry'].delete(0, tk.END))
                self.root.after(0, lambda: self.widgets['wallet_address_entry'].insert(0, address))
                self.root.after(0, lambda: self.widgets['private_key'].delete(0, tk.END))
                self.root.after(0, lambda: self.widgets['private_key'].insert(0, private_key))
                self.root.after(0, lambda: self.update_connection_status())
                self.root.after(0, lambda: self.update_account_info())
                self.root.after(0, lambda: messagebox.showinfo("Success", f"New wallet generated and connected!\n\nAddress: {address}\n\nPrivate Key: {private_key}\n\nIMPORTANT: Save your private key securely!"))
                self.root.after(0, lambda: self.update_status("Ready"))
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", "Failed to generate and connect with new wallet"))
                self.root.after(0, lambda: self.update_status("Ready"))
        except Exception as e:
            logger.error(f"Generate wallet failed: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to generate wallet: {e}"))
            self.root.after(0, lambda: self.update_status("Ready"))
    
    def use_default_credentials(self):
        """Use default credentials"""
        try:
            # Show using default message
            self.update_status("Using default credentials...")
            
            # Use default credentials in a separate thread
            threading.Thread(target=self._use_default_credentials).start()
        except Exception as e:
            logger.error(f"Use default credentials failed: {e}")
            messagebox.showerror("Error", f"Failed to use default credentials: {e}")
            self.update_status("Ready")
    
    def _use_default_credentials(self):
        """Use default credentials"""
        try:
            # Use default credentials
            if self.connection_manager._connect_with_default_credentials():
                # Update UI
                self.root.after(0, lambda: self.widgets['wallet_address_entry'].delete(0, tk.END))
                self.root.after(0, lambda: self.widgets['wallet_address_entry'].insert(0, self.connection_manager.default_address))
                self.root.after(0, lambda: self.widgets['private_key'].delete(0, tk.END))
                self.root.after(0, lambda: self.widgets['private_key'].insert(0, self.connection_manager.default_private_key))
                self.root.after(0, lambda: self.update_connection_status())
                self.root.after(0, lambda: self.update_account_info())
                self.root.after(0, lambda: messagebox.showinfo("Success", "Connected with default credentials"))
                self.root.after(0, lambda: self.update_status("Ready"))
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", "Failed to connect with default credentials"))
                self.root.after(0, lambda: self.update_status("Ready"))
        except Exception as e:
            logger.error(f"Use default credentials failed: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to use default credentials: {e}"))
            self.root.after(0, lambda: self.update_status("Ready"))
    
    def switch_network(self):
        """Switch network"""
        try:
            # Get current network
            current_testnet = self.connection_manager.testnet
            
            # Switch network
            new_testnet = not current_testnet
            
            # Show switching message
            self.update_status(f"Switching to {'testnet' if new_testnet else 'mainnet'}...")
            
            # Switch network in a separate thread
            threading.Thread(target=self._switch_network, args=(new_testnet,)).start()
        except Exception as e:
            logger.error(f"Switch network failed: {e}")
            messagebox.showerror("Error", f"Failed to switch network: {e}")
            self.update_status("Ready")
    
    def _switch_network(self, testnet):
        """Switch network"""
        try:
            # Switch network
            if self.connection_manager.update_network(testnet):
                # Update UI
                self.root.after(0, lambda: self.network_var.set("testnet" if testnet else "mainnet"))
                self.root.after(0, lambda: self.update_connection_status())
                self.root.after(0, lambda: self.update_account_info())
                self.root.after(0, lambda: messagebox.showinfo("Success", f"Switched to {'testnet' if testnet else 'mainnet'}"))
                self.root.after(0, lambda: self.update_status("Ready"))
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", "Failed to switch network"))
                self.root.after(0, lambda: self.update_status("Ready"))
        except Exception as e:
            logger.error(f"Switch network failed: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to switch network: {e}"))
            self.root.after(0, lambda: self.update_status("Ready"))
    
    def update_network(self):
        """Update network based on radio button selection"""
        try:
            # Get selected network
            network = self.network_var.get()
            testnet = network == "testnet"
            
            # Update network
            self.connection_manager.update_network(testnet)
            
            # Update connection status
            self.update_connection_status()
        except Exception as e:
            logger.error(f"Update network failed: {e}")
    
    def update_auto_reconnect(self):
        """Update auto-reconnect based on radio button selection"""
        try:
            # Get selected auto-reconnect setting
            auto_reconnect = self.auto_reconnect_var.get()
            
            # Update auto-reconnect
            self.connection_manager.set_auto_reconnect(auto_reconnect)
        except Exception as e:
            logger.error(f"Update auto-reconnect failed: {e}")
    
    def refresh_dashboard(self):
        """Refresh dashboard"""
        try:
            # Show refreshing message
            self.update_status("Refreshing dashboard...")
            
            # Update connection status
            self.update_connection_status()
            
            # Update account info
            self.update_account_info()
            
            # Update trading info
            self.update_trading_info()
            
            # Update charts
            self.update_charts()
            
            # Show ready message
            self.update_status("Ready")
        except Exception as e:
            logger.error(f"Refresh dashboard failed: {e}")
            self.update_status("Ready")
    
    def update_status(self, message):
        """Update status bar message"""
        try:
            if 'status_label' in self.widgets:
                self.widgets['status_label'].config(text=message)
        except Exception as e:
            logger.error(f"Update status failed: {e}")
    
    def run(self):
        """Run the application"""
        try:
            # Show ready message
            self.update_status("Ready")
            
            # Start main loop
            self.root.mainloop()
        except Exception as e:
            logger.error(f"Run failed: {e}")


class LogHandler(logging.Handler):
    """Custom log handler for GUI"""
    
    def __init__(self, text_widget):
        """Initialize the log handler"""
        logging.Handler.__init__(self)
        self.text_widget = text_widget
        
        # Configure formatter
        formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', '%Y-%m-%d %H:%M:%S')
        self.setFormatter(formatter)
    
    def emit(self, record):
        """Emit a log record"""
        msg = self.format(record)
        
        # Add color based on level
        if record.levelno >= logging.ERROR:
            color = "red"
        elif record.levelno >= logging.WARNING:
            color = "orange"
        elif record.levelno >= logging.INFO:
            color = "blue"
        else:
            color = "black"
        
        # Schedule text insertion in main thread
        self.text_widget.after(0, self._insert_log, msg, color)
    
    def _insert_log(self, msg, color):
        """Insert log message"""
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.insert(tk.END, msg + "\n", color)
        self.text_widget.tag_config(color, foreground=color)
        self.text_widget.see(tk.END)
        self.text_widget.config(state=tk.DISABLED)

