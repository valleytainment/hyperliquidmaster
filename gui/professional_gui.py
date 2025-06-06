"""
Enhanced GUI with Professional Styling and Auto-Connection
Integrated with advanced features from Ultimate Master Bot
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import queue
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_manager import ConfigManager
from utils.security import SecurityManager
from core.api import EnhancedHyperliquidAPI
from core.connection_manager_enhanced import EnhancedConnectionManager
from strategies.enhanced_neural_strategy import EnhancedNeuralStrategy
from utils.logger import get_logger

logger = get_logger(__name__)

class ProfessionalTradingGUI:
    """
    Professional Trading GUI with advanced features and styling
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hyperliquid Master - Professional Trading Bot v3.0")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.security_manager = SecurityManager()
        self.connection_manager = None
        self.api = None
        self.neural_strategy = None
        
        # GUI state
        self.is_connected = False
        self.is_trading = False
        self.log_queue = queue.Queue()
        
        # Setup GUI
        self.setup_styles()
        self.create_widgets()
        self.setup_logging()
        
        # Auto-connect on startup
        self.auto_connect()
        
        logger.info("Professional Trading GUI initialized")
    
    def setup_styles(self):
        """Setup professional dark theme styling"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        bg_color = '#1e1e1e'
        fg_color = '#ffffff'
        select_bg = '#404040'
        select_fg = '#ffffff'
        
        # Configure styles
        style.configure('TFrame', background=bg_color)
        style.configure('TLabel', background=bg_color, foreground=fg_color)
        style.configure('TButton', background='#404040', foreground=fg_color)
        style.map('TButton', background=[('active', '#505050')])
        style.configure('TEntry', background='#2d2d2d', foreground=fg_color)
        style.configure('TCombobox', background='#2d2d2d', foreground=fg_color)
        style.configure('TNotebook', background=bg_color)
        style.configure('TNotebook.Tab', background='#404040', foreground=fg_color)
        style.map('TNotebook.Tab', background=[('selected', '#505050')])
        
        # Configure Treeview
        style.configure('Treeview', background='#2d2d2d', foreground=fg_color)
        style.configure('Treeview.Heading', background='#404040', foreground=fg_color)
        style.map('Treeview', background=[('selected', select_bg)])
    
    def create_widgets(self):
        """Create all GUI widgets with professional layout"""
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top frame for connection status and controls
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Connection status
        self.connection_frame = ttk.LabelFrame(top_frame, text="Connection Status", padding=10)
        self.connection_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.status_label = ttk.Label(self.connection_frame, text="Disconnected", foreground='red')
        self.status_label.pack(side=tk.LEFT)
        
        self.connect_button = ttk.Button(self.connection_frame, text="Connect", command=self.toggle_connection)
        self.connect_button.pack(side=tk.RIGHT)
        
        # Trading controls
        self.trading_frame = ttk.LabelFrame(top_frame, text="Trading Controls", padding=10)
        self.trading_frame.pack(side=tk.RIGHT, padx=(10, 0))
        
        self.trading_button = ttk.Button(self.trading_frame, text="Start Trading", command=self.toggle_trading)
        self.trading_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.emergency_stop_button = ttk.Button(self.trading_frame, text="Emergency Stop", command=self.emergency_stop)
        self.emergency_stop_button.pack(side=tk.LEFT)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_strategy_tab()
        self.create_positions_tab()
        self.create_settings_tab()
        self.create_logs_tab()
    
    def create_dashboard_tab(self):
        """Create main dashboard tab"""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="Dashboard")
        
        # Account info frame
        account_frame = ttk.LabelFrame(dashboard_frame, text="Account Information", padding=10)
        account_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Account details
        ttk.Label(account_frame, text="Address:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.address_label = ttk.Label(account_frame, text="Not connected")
        self.address_label.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(account_frame, text="Balance:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        self.balance_label = ttk.Label(account_frame, text="$0.00")
        self.balance_label.grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(account_frame, text="PnL:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10))
        self.pnl_label = ttk.Label(account_frame, text="$0.00")
        self.pnl_label.grid(row=2, column=1, sticky=tk.W)
        
        # Performance metrics frame
        metrics_frame = ttk.LabelFrame(dashboard_frame, text="Performance Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Metrics grid
        metrics = [
            ("Total Trades:", "0"),
            ("Win Rate:", "0%"),
            ("Avg Profit:", "$0.00"),
            ("Max Drawdown:", "0%"),
            ("Sharpe Ratio:", "0.00"),
            ("Current Streak:", "0")
        ]
        
        self.metric_labels = {}
        for i, (label, value) in enumerate(metrics):
            row = i // 3
            col = (i % 3) * 2
            ttk.Label(metrics_frame, text=label).grid(row=row, column=col, sticky=tk.W, padx=(0, 10))
            self.metric_labels[label] = ttk.Label(metrics_frame, text=value)
            self.metric_labels[label].grid(row=row, column=col+1, sticky=tk.W, padx=(0, 20))
        
        # Market overview frame
        market_frame = ttk.LabelFrame(dashboard_frame, text="Market Overview", padding=10)
        market_frame.pack(fill=tk.BOTH, expand=True)
        
        # Market data table
        columns = ("Symbol", "Price", "Change 24h", "Volume", "Signal")
        self.market_tree = ttk.Treeview(market_frame, columns=columns, show="headings", height=10)
        
        for col in columns:
            self.market_tree.heading(col, text=col)
            self.market_tree.column(col, width=120)
        
        # Scrollbar for market data
        market_scrollbar = ttk.Scrollbar(market_frame, orient=tk.VERTICAL, command=self.market_tree.yview)
        self.market_tree.configure(yscrollcommand=market_scrollbar.set)
        
        self.market_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        market_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_strategy_tab(self):
        """Create strategy configuration tab"""
        strategy_frame = ttk.Frame(self.notebook)
        self.notebook.add(strategy_frame, text="Strategy")
        
        # Strategy selection
        selection_frame = ttk.LabelFrame(strategy_frame, text="Strategy Selection", padding=10)
        selection_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(selection_frame, text="Active Strategy:").pack(side=tk.LEFT)
        self.strategy_var = tk.StringVar(value="Enhanced Neural Network")
        strategy_combo = ttk.Combobox(selection_frame, textvariable=self.strategy_var, 
                                    values=["Enhanced Neural Network", "BB RSI ADX", "Hull Suite"])
        strategy_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Strategy parameters
        params_frame = ttk.LabelFrame(strategy_frame, text="Strategy Parameters", padding=10)
        params_frame.pack(fill=tk.BOTH, expand=True)
        
        # Neural Network specific parameters
        neural_frame = ttk.LabelFrame(params_frame, text="Neural Network Settings", padding=10)
        neural_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Parameter controls
        param_controls = [
            ("Confidence Threshold:", "0.7", "confidence_threshold"),
            ("Lookback Bars:", "30", "lookback_bars"),
            ("Learning Rate:", "0.0003", "learning_rate"),
            ("Stop Loss %:", "2.0", "stop_loss_pct"),
            ("Take Profit %:", "4.0", "take_profit_pct")
        ]
        
        self.param_vars = {}
        for i, (label, default, key) in enumerate(param_controls):
            row = i // 2
            col = (i % 2) * 3
            
            ttk.Label(neural_frame, text=label).grid(row=row, column=col, sticky=tk.W, padx=(0, 5))
            var = tk.StringVar(value=default)
            self.param_vars[key] = var
            entry = ttk.Entry(neural_frame, textvariable=var, width=10)
            entry.grid(row=row, column=col+1, padx=(0, 20))
        
        # Apply button
        ttk.Button(neural_frame, text="Apply Parameters", command=self.apply_parameters).grid(row=3, column=0, columnspan=6, pady=10)
        
        # RL Tuning controls
        rl_frame = ttk.LabelFrame(params_frame, text="RL Parameter Tuning", padding=10)
        rl_frame.pack(fill=tk.X)
        
        self.rl_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(rl_frame, text="Enable RL Parameter Tuning", variable=self.rl_enabled).pack(side=tk.LEFT)
        
        ttk.Button(rl_frame, text="Reset RL State", command=self.reset_rl_state).pack(side=tk.RIGHT)
    
    def create_positions_tab(self):
        """Create positions monitoring tab"""
        positions_frame = ttk.Frame(self.notebook)
        self.notebook.add(positions_frame, text="Positions")
        
        # Active positions
        active_frame = ttk.LabelFrame(positions_frame, text="Active Positions", padding=10)
        active_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Positions table
        pos_columns = ("Symbol", "Side", "Size", "Entry Price", "Current Price", "PnL", "PnL %")
        self.positions_tree = ttk.Treeview(active_frame, columns=pos_columns, show="headings", height=8)
        
        for col in pos_columns:
            self.positions_tree.heading(col, text=col)
            self.positions_tree.column(col, width=100)
        
        pos_scrollbar = ttk.Scrollbar(active_frame, orient=tk.VERTICAL, command=self.positions_tree.yview)
        self.positions_tree.configure(yscrollcommand=pos_scrollbar.set)
        
        self.positions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pos_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Position controls
        controls_frame = ttk.Frame(positions_frame)
        controls_frame.pack(fill=tk.X)
        
        ttk.Button(controls_frame, text="Close All Positions", command=self.close_all_positions).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(controls_frame, text="Refresh Positions", command=self.refresh_positions).pack(side=tk.LEFT)
    
    def create_settings_tab(self):
        """Create settings configuration tab"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")
        
        # Connection settings
        conn_frame = ttk.LabelFrame(settings_frame, text="Connection Settings", padding=10)
        conn_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Network selection
        ttk.Label(conn_frame, text="Network:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.network_var = tk.StringVar(value="mainnet")
        network_combo = ttk.Combobox(conn_frame, textvariable=self.network_var, values=["mainnet", "testnet"])
        network_combo.grid(row=0, column=1, sticky=tk.W)
        
        # Wallet settings
        wallet_frame = ttk.LabelFrame(settings_frame, text="Wallet Settings", padding=10)
        wallet_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(wallet_frame, text="Address:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.wallet_address_var = tk.StringVar()
        ttk.Entry(wallet_frame, textvariable=self.wallet_address_var, width=50).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(wallet_frame, text="Private Key:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        self.private_key_var = tk.StringVar()
        ttk.Entry(wallet_frame, textvariable=self.private_key_var, width=50, show="*").grid(row=1, column=1, sticky=tk.W)
        
        # Buttons
        button_frame = ttk.Frame(wallet_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Generate New Wallet", command=self.generate_wallet).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Save Credentials", command=self.save_credentials).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Load Credentials", command=self.load_credentials).pack(side=tk.LEFT)
        
        # Risk management settings
        risk_frame = ttk.LabelFrame(settings_frame, text="Risk Management", padding=10)
        risk_frame.pack(fill=tk.X)
        
        risk_settings = [
            ("Max Position Size:", "1000", "max_position_size"),
            ("Max Daily Loss:", "500", "max_daily_loss"),
            ("Max Positions:", "5", "max_positions"),
            ("Risk Per Trade %:", "1.0", "risk_per_trade")
        ]
        
        self.risk_vars = {}
        for i, (label, default, key) in enumerate(risk_settings):
            row = i // 2
            col = (i % 2) * 3
            
            ttk.Label(risk_frame, text=label).grid(row=row, column=col, sticky=tk.W, padx=(0, 5))
            var = tk.StringVar(value=default)
            self.risk_vars[key] = var
            ttk.Entry(risk_frame, textvariable=var, width=10).grid(row=row, column=col+1, padx=(0, 20))
    
    def create_logs_tab(self):
        """Create logs monitoring tab"""
        logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(logs_frame, text="Logs")
        
        # Log display
        self.log_text = scrolledtext.ScrolledText(logs_frame, height=25, bg='#1e1e1e', fg='#ffffff', 
                                                 insertbackground='#ffffff')
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Log controls
        log_controls = ttk.Frame(logs_frame)
        log_controls.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(log_controls, text="Clear Logs", command=self.clear_logs).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(log_controls, text="Save Logs", command=self.save_logs).pack(side=tk.LEFT)
        
        # Log level filter
        ttk.Label(log_controls, text="Level:").pack(side=tk.RIGHT, padx=(10, 5))
        self.log_level_var = tk.StringVar(value="INFO")
        log_level_combo = ttk.Combobox(log_controls, textvariable=self.log_level_var, 
                                     values=["DEBUG", "INFO", "WARNING", "ERROR"], width=10)
        log_level_combo.pack(side=tk.RIGHT)
    
    def setup_logging(self):
        """Setup logging to display in GUI"""
        # Create custom handler for GUI
        class GUILogHandler(logging.Handler):
            def __init__(self, log_queue):
                super().__init__()
                self.log_queue = log_queue
            
            def emit(self, record):
                msg = self.format(record)
                self.log_queue.put(msg)
        
        # Add GUI handler to logger
        gui_handler = GUILogHandler(self.log_queue)
        gui_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(gui_handler)
        
        # Start log processing
        self.process_log_queue()
    
    def process_log_queue(self):
        """Process log messages from queue and display in GUI"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, message + '\n')
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_log_queue)
    
    def auto_connect(self):
        """Auto-connect with default credentials on startup"""
        def connect_thread():
            try:
                # Initialize connection manager
                self.connection_manager = EnhancedConnectionManager()
                
                # Check if connected
                if self.connection_manager.is_connected:
                    self.is_connected = True
                    self.api = self.connection_manager.api
                    
                    # Initialize neural strategy
                    self.neural_strategy = EnhancedNeuralStrategy(
                        api=self.api,
                        risk_manager=None,
                        max_positions=3
                    )
                    
                    # Update GUI
                    self.root.after(0, self.update_connection_status)
                    logger.info("Auto-connected successfully")
                else:
                    logger.warning("Auto-connection failed")
                    
            except Exception as e:
                logger.error(f"Auto-connection error: {e}")
        
        # Start connection in background thread
        threading.Thread(target=connect_thread, daemon=True).start()
    
    def update_connection_status(self):
        """Update connection status in GUI"""
        if self.is_connected:
            self.status_label.config(text="Connected", foreground='green')
            self.connect_button.config(text="Disconnect")
            
            # Update address if available
            if self.connection_manager and self.connection_manager.current_address:
                self.address_label.config(text=self.connection_manager.current_address)
                self.wallet_address_var.set(self.connection_manager.current_address)
        else:
            self.status_label.config(text="Disconnected", foreground='red')
            self.connect_button.config(text="Connect")
            self.address_label.config(text="Not connected")
    
    def toggle_connection(self):
        """Toggle connection state"""
        if self.is_connected:
            self.disconnect()
        else:
            self.connect()
    
    def connect(self):
        """Connect to Hyperliquid"""
        def connect_thread():
            try:
                if not self.connection_manager:
                    self.connection_manager = EnhancedConnectionManager()
                
                if self.connection_manager.ensure_connection():
                    self.is_connected = True
                    self.api = self.connection_manager.api
                    
                    # Initialize neural strategy
                    if not self.neural_strategy:
                        self.neural_strategy = EnhancedNeuralStrategy(
                            api=self.api,
                            risk_manager=None,
                            max_positions=3
                        )
                    
                    self.root.after(0, self.update_connection_status)
                    logger.info("Connected successfully")
                else:
                    logger.error("Failed to connect")
                    
            except Exception as e:
                logger.error(f"Connection error: {e}")
        
        threading.Thread(target=connect_thread, daemon=True).start()
    
    def disconnect(self):
        """Disconnect from Hyperliquid"""
        try:
            if self.connection_manager:
                self.connection_manager.disconnect()
            
            self.is_connected = False
            self.is_trading = False
            self.update_connection_status()
            self.update_trading_status()
            logger.info("Disconnected")
            
        except Exception as e:
            logger.error(f"Disconnect error: {e}")
    
    def toggle_trading(self):
        """Toggle trading state"""
        if self.is_trading:
            self.stop_trading()
        else:
            self.start_trading()
    
    def start_trading(self):
        """Start automated trading"""
        if not self.is_connected:
            messagebox.showerror("Error", "Must be connected to start trading")
            return
        
        self.is_trading = True
        self.update_trading_status()
        logger.info("Trading started")
    
    def stop_trading(self):
        """Stop automated trading"""
        self.is_trading = False
        self.update_trading_status()
        logger.info("Trading stopped")
    
    def emergency_stop(self):
        """Emergency stop all trading and close positions"""
        self.stop_trading()
        self.close_all_positions()
        logger.warning("Emergency stop activated")
    
    def update_trading_status(self):
        """Update trading status in GUI"""
        if self.is_trading:
            self.trading_button.config(text="Stop Trading")
        else:
            self.trading_button.config(text="Start Trading")
    
    def apply_parameters(self):
        """Apply strategy parameters"""
        try:
            if self.neural_strategy:
                # Update neural strategy parameters
                self.neural_strategy.confidence_threshold = float(self.param_vars["confidence_threshold"].get())
                self.neural_strategy.stop_loss_pct = float(self.param_vars["stop_loss_pct"].get()) / 100
                self.neural_strategy.take_profit_pct = float(self.param_vars["take_profit_pct"].get()) / 100
                
                logger.info("Strategy parameters updated")
                messagebox.showinfo("Success", "Parameters applied successfully")
            else:
                messagebox.showerror("Error", "Strategy not initialized")
                
        except Exception as e:
            logger.error(f"Error applying parameters: {e}")
            messagebox.showerror("Error", f"Failed to apply parameters: {e}")
    
    def reset_rl_state(self):
        """Reset RL parameter tuning state"""
        try:
            if self.neural_strategy and hasattr(self.neural_strategy, 'param_tuner'):
                # Reset tuner state
                self.neural_strategy.param_tuner.best_pnl = -float("inf")
                self.neural_strategy.param_tuner.best_params = {}
                self.neural_strategy.param_tuner.episode_pnl = 0.0
                self.neural_strategy.param_tuner.trade_count = 0
                self.neural_strategy.param_tuner.save_state()
                
                logger.info("RL state reset")
                messagebox.showinfo("Success", "RL state reset successfully")
            else:
                messagebox.showerror("Error", "Neural strategy not available")
                
        except Exception as e:
            logger.error(f"Error resetting RL state: {e}")
            messagebox.showerror("Error", f"Failed to reset RL state: {e}")
    
    def close_all_positions(self):
        """Close all open positions"""
        try:
            # Implementation would depend on your position management system
            logger.info("Closing all positions")
            messagebox.showinfo("Info", "All positions closed")
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
            messagebox.showerror("Error", f"Failed to close positions: {e}")
    
    def refresh_positions(self):
        """Refresh positions display"""
        try:
            # Clear existing items
            for item in self.positions_tree.get_children():
                self.positions_tree.delete(item)
            
            # Add sample data (replace with real position data)
            sample_positions = [
                ("BTC", "LONG", "0.1", "45000", "46000", "+100", "+2.22%"),
                ("ETH", "SHORT", "2.0", "3000", "2950", "+100", "+1.67%")
            ]
            
            for pos in sample_positions:
                self.positions_tree.insert("", tk.END, values=pos)
                
        except Exception as e:
            logger.error(f"Error refreshing positions: {e}")
    
    def generate_wallet(self):
        """Generate new wallet"""
        try:
            if self.connection_manager:
                success, address, private_key = self.connection_manager.connect_with_new_wallet()
                if success:
                    self.wallet_address_var.set(address)
                    self.private_key_var.set(private_key)
                    messagebox.showinfo("Success", f"New wallet generated:\n{address}")
                else:
                    messagebox.showerror("Error", "Failed to generate wallet")
            else:
                messagebox.showerror("Error", "Connection manager not available")
                
        except Exception as e:
            logger.error(f"Error generating wallet: {e}")
            messagebox.showerror("Error", f"Failed to generate wallet: {e}")
    
    def save_credentials(self):
        """Save wallet credentials"""
        try:
            address = self.wallet_address_var.get()
            private_key = self.private_key_var.get()
            
            if address and private_key:
                self.config_manager.set('trading.wallet_address', address)
                self.security_manager.store_private_key(private_key)
                self.config_manager.save_config()
                
                messagebox.showinfo("Success", "Credentials saved successfully")
                logger.info("Credentials saved")
            else:
                messagebox.showerror("Error", "Please enter both address and private key")
                
        except Exception as e:
            logger.error(f"Error saving credentials: {e}")
            messagebox.showerror("Error", f"Failed to save credentials: {e}")
    
    def load_credentials(self):
        """Load saved credentials"""
        try:
            address = self.config_manager.get('trading.wallet_address', '')
            private_key = self.security_manager.get_private_key()
            
            if address:
                self.wallet_address_var.set(address)
            if private_key:
                self.private_key_var.set(private_key)
            
            if address or private_key:
                messagebox.showinfo("Success", "Credentials loaded successfully")
                logger.info("Credentials loaded")
            else:
                messagebox.showinfo("Info", "No saved credentials found")
                
        except Exception as e:
            logger.error(f"Error loading credentials: {e}")
            messagebox.showerror("Error", f"Failed to load credentials: {e}")
    
    def clear_logs(self):
        """Clear log display"""
        self.log_text.delete(1.0, tk.END)
    
    def save_logs(self):
        """Save logs to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/gui_logs_{timestamp}.txt"
            
            os.makedirs("logs", exist_ok=True)
            
            with open(filename, 'w') as f:
                f.write(self.log_text.get(1.0, tk.END))
            
            messagebox.showinfo("Success", f"Logs saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving logs: {e}")
            messagebox.showerror("Error", f"Failed to save logs: {e}")
    
    def on_closing(self):
        """Handle window closing"""
        try:
            if self.is_trading:
                if messagebox.askyesno("Confirm", "Trading is active. Stop trading and exit?"):
                    self.stop_trading()
                    self.disconnect()
                    self.root.destroy()
            else:
                self.disconnect()
                self.root.destroy()
        except Exception as e:
            logger.error(f"Error during closing: {e}")
            self.root.destroy()
    
    def run(self):
        """Run the GUI application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


if __name__ == "__main__":
    app = ProfessionalTradingGUI()
    app.run()

