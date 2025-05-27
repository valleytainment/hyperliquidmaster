#!/usr/bin/env python3
"""
Enhanced GUI for the HyperliquidMaster trading bot.
Provides a user-friendly interface for trading on the Hyperliquid exchange.
"""

import os
import sys
import json
import time
import logging
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from typing import Dict, List, Any, Optional, Tuple
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import custom modules
from gui_style import GUIStyleManager
from core.trading_integration import TradingIntegration
from core.error_handler import ErrorHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
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
        self.root.title("Enhanced Hyperliquid Trading Bot v2.0.0")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Initialize logger
        self.logger = logging.getLogger("HyperliquidMasterGUI")
        
        # Initialize config
        self.config_path = "config.json"
        self.config = self._load_config()
        
        # Initialize style manager
        self.style_manager = GUIStyleManager(self.root, self.logger)
        
        # Initialize error handler
        self.error_handler = ErrorHandler(self.logger)
        
        # Initialize trading integration
        self.trading = TradingIntegration(self.config_path, self.logger)
        
        # Initialize variables
        self.is_bot_running = False
        self.selected_symbol = tk.StringVar(value=self.config.get("symbol", "BTC"))
        self.account_address = tk.StringVar(value=self.config.get("account_address", ""))
        self.secret_key = tk.StringVar(value=self.config.get("secret_key", ""))
        self.show_secret_key = tk.BooleanVar(value=False)
        self.position_size = tk.StringVar(value=self.config.get("position_size", "0.01"))
        self.stop_loss = tk.StringVar(value=self.config.get("stop_loss", "1.0"))
        self.take_profit = tk.StringVar(value=self.config.get("take_profit", "2.0"))
        self.available_symbols = []
        
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
        self.logs_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.dashboard_tab, text="Dashboard")
        self.notebook.add(self.positions_tab, text="Positions")
        self.notebook.add(self.settings_tab, text="Settings")
        self.notebook.add(self.logs_tab, text="Logs")
        
        # Create theme toggle button
        self.theme_button = tk.Button(self.root, text="Toggle Theme", command=self._toggle_theme)
        self.theme_button.place(relx=0.95, rely=0.02, anchor="ne")
        self.style_manager.style_button(self.theme_button)
        
        # Initialize tabs
        self._init_dashboard_tab()
        self._init_positions_tab()
        self._init_settings_tab()
        self._init_logs_tab()
        
        # Start update loops
        self._start_update_loops()
        
        # Log startup
        self.logger.info("Enhanced Hyperliquid Trading Bot v2.0.0 started")
    
    def _load_config(self) -> Dict:
        """
        Load configuration from file.
        
        Returns:
            Dict containing the configuration
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}
    
    def _save_config(self) -> None:
        """Save configuration to file."""
        try:
            config = {
                "symbol": self.selected_symbol.get(),
                "account_address": self.account_address.get(),
                "secret_key": self.secret_key.get(),
                "position_size": self.position_size.get(),
                "stop_loss": self.stop_loss.get(),
                "take_profit": self.take_profit.get()
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
    
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
        
        self.symbol_combobox = ttk.Combobox(top_frame, textvariable=self.selected_symbol, width=10)
        self.symbol_combobox.pack(side=tk.LEFT, padx=(0, 10))
        self.symbol_combobox.bind("<<ComboboxSelected>>", self._on_symbol_selected)
        
        # Create refresh button
        refresh_button = tk.Button(top_frame, text="Refresh", command=self._refresh_dashboard)
        refresh_button.pack(side=tk.LEFT, padx=(0, 10))
        self.style_manager.style_button(refresh_button)
        
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
        
        # Create buy button
        buy_button = tk.Button(controls_frame, text="BUY", command=lambda: self._place_order(True))
        buy_button.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        self.style_manager.style_button(buy_button, "success")
        
        # Create sell button
        sell_button = tk.Button(controls_frame, text="SELL", command=lambda: self._place_order(False))
        sell_button.grid(row=1, column=3, columnspan=3, padx=5, pady=5, sticky="ew")
        self.style_manager.style_button(sell_button, "error")
        
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
        columns = ("symbol", "size", "entry_price", "current_price", "pnl", "pnl_percent")
        self.positions_tree = ttk.Treeview(self.positions_tab, columns=columns, show="headings")
        
        # Define headings
        self.positions_tree.heading("symbol", text="Symbol")
        self.positions_tree.heading("size", text="Size")
        self.positions_tree.heading("entry_price", text="Entry Price")
        self.positions_tree.heading("current_price", text="Current Price")
        self.positions_tree.heading("pnl", text="PnL")
        self.positions_tree.heading("pnl_percent", text="PnL %")
        
        # Define columns
        self.positions_tree.column("symbol", width=100)
        self.positions_tree.column("size", width=100)
        self.positions_tree.column("entry_price", width=100)
        self.positions_tree.column("current_price", width=100)
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
        
        # Create save settings button
        save_settings_button = tk.Button(trading_frame, text="Save Settings", command=self._save_config)
        save_settings_button.pack(anchor=tk.W, pady=5)
        self.style_manager.style_button(save_settings_button, "success")
    
    def _init_logs_tab(self) -> None:
        """Initialize the logs tab."""
        # Create log text widget with scrollbar
        log_frame, self.log_text = self.style_manager.create_scrollable_text(self.logs_tab, height=20, width=80)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure log text
        self.log_text.config(state=tk.DISABLED)
        
        # Create custom log handler
        self.log_handler = LogTextHandler(self.log_text)
        self.log_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        self.log_handler.setFormatter(formatter)
        
        # Add handler to logger
        logging.getLogger().addHandler(self.log_handler)
    
    def _create_chart(self) -> None:
        """Create a chart for price data."""
        # Get chart colors
        chart_colors = self.style_manager.get_chart_colors()
        
        # Create figure and axis
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor=chart_colors["bg"])
        self.ax = self.fig.add_subplot(111)
        
        # Configure axis
        self.ax.set_facecolor(chart_colors["bg"])
        self.ax.tick_params(axis='x', colors=chart_colors["fg"])
        self.ax.tick_params(axis='y', colors=chart_colors["fg"])
        self.ax.spines['bottom'].set_color(chart_colors["grid"])
        self.ax.spines['top'].set_color(chart_colors["grid"])
        self.ax.spines['left'].set_color(chart_colors["grid"])
        self.ax.spines['right'].set_color(chart_colors["grid"])
        self.ax.grid(True, linestyle='--', alpha=0.7, color=chart_colors["grid"])
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _update_chart(self) -> None:
        """Update the chart with new data."""
        try:
            # Get chart colors
            chart_colors = self.style_manager.get_chart_colors()
            
            # Clear the axis
            self.ax.clear()
            
            # Configure axis
            self.ax.set_facecolor(chart_colors["bg"])
            self.ax.tick_params(axis='x', colors=chart_colors["fg"])
            self.ax.tick_params(axis='y', colors=chart_colors["fg"])
            self.ax.spines['bottom'].set_color(chart_colors["grid"])
            self.ax.spines['top'].set_color(chart_colors["grid"])
            self.ax.spines['left'].set_color(chart_colors["grid"])
            self.ax.spines['right'].set_color(chart_colors["grid"])
            self.ax.grid(True, linestyle='--', alpha=0.7, color=chart_colors["grid"])
            
            # Get symbol
            symbol = self.selected_symbol.get()
            
            # Plot dummy data for now (would be replaced with real data)
            import numpy as np
            x = np.linspace(0, 10, 100)
            y = np.sin(x) + np.random.normal(0, 0.1, 100)
            
            self.ax.plot(x, y, color=chart_colors["line1"], linewidth=2)
            self.ax.set_title(f"{symbol} Price", color=chart_colors["fg"])
            self.ax.set_xlabel("Time", color=chart_colors["fg"])
            self.ax.set_ylabel("Price", color=chart_colors["fg"])
            
            # Draw the canvas
            self.canvas.draw()
        except Exception as e:
            self.logger.error(f"Error updating chart: {e}")
    
    def _start_update_loops(self) -> None:
        """Start update loops for various components."""
        # Start positions update loop
        self._schedule_positions_update()
        
        # Start orders update loop
        self._schedule_orders_update()
        
        # Get available symbols
        self._get_available_symbols()
    
    def _schedule_positions_update(self) -> None:
        """Schedule positions update."""
        self._refresh_positions()
        self.root.after(10000, self._schedule_positions_update)  # Update every 10 seconds
    
    def _schedule_orders_update(self) -> None:
        """Schedule orders update."""
        self._refresh_orders()
        self.root.after(5000, self._schedule_orders_update)  # Update every 5 seconds
    
    def _refresh_dashboard(self) -> None:
        """Refresh the dashboard."""
        try:
            # Update chart
            self._update_chart()
            
            # Update connection status
            self._update_connection_status()
        except Exception as e:
            self.logger.error(f"Error refreshing dashboard: {e}")
    
    def _refresh_positions(self) -> None:
        """Refresh positions data."""
        try:
            # Clear existing items
            for item in self.positions_tree.get_children():
                self.positions_tree.delete(item)
            
            # Get positions
            positions_result = self.trading.get_positions()
            
            if not positions_result["success"]:
                self.logger.error(f"Error getting positions: {positions_result['message']}")
                return
            
            positions = positions_result["data"]
            
            # Add positions to treeview
            for position in positions:
                symbol = position.get("coin", "")
                size = float(position.get("szi", 0))
                entry_price = float(position.get("entryPx", 0))
                
                # Get current price
                market_data_result = self.trading.get_market_data(symbol)
                
                if not market_data_result["success"]:
                    current_price = 0
                else:
                    current_price = market_data_result["data"]["price"]
                
                # Calculate PnL
                if size != 0 and entry_price != 0 and current_price != 0:
                    pnl = size * (current_price - entry_price)
                    pnl_percent = (current_price / entry_price - 1) * 100 * (1 if size > 0 else -1)
                else:
                    pnl = 0
                    pnl_percent = 0
                
                # Format values
                size_str = f"{size:.4f}"
                entry_price_str = f"{entry_price:.2f}"
                current_price_str = f"{current_price:.2f}"
                pnl_str = f"{pnl:.2f}"
                pnl_percent_str = f"{pnl_percent:.2f}%"
                
                # Add to treeview
                self.positions_tree.insert("", tk.END, values=(symbol, size_str, entry_price_str, current_price_str, pnl_str, pnl_percent_str))
            
            self.logger.info("Positions refreshed")
        except Exception as e:
            self.logger.error(f"Error refreshing positions: {e}")
    
    def _refresh_orders(self) -> None:
        """Refresh orders data."""
        try:
            # Get orders
            orders_result = self.trading.get_orders()
            
            if not orders_result["success"]:
                self.logger.error(f"Error getting orders: {orders_result['message']}")
                return
            
            # Log orders count
            orders = orders_result["data"]
            self.logger.info("Orders refreshed")
        except Exception as e:
            self.logger.error(f"Error refreshing orders: {e}")
    
    def _update_connection_status(self) -> None:
        """Update the connection status indicator."""
        if self.trading.is_connected:
            self.connection_status.config(text="Connected", bg="green")
        else:
            self.connection_status.config(text="Not Connected", bg="red")
    
    def _get_available_symbols(self) -> None:
        """Get available trading symbols."""
        try:
            symbols = self.trading.get_available_symbols()
            
            if symbols:
                self.available_symbols = symbols
                self.symbol_combobox["values"] = symbols
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
    
    def _on_symbol_selected(self, event) -> None:
        """Handle symbol selection."""
        try:
            symbol = self.selected_symbol.get()
            self._update_chart()
            self._save_config()
        except Exception as e:
            self.logger.error(f"Error handling symbol selection: {e}")
    
    def _toggle_show_secret_key(self) -> None:
        """Toggle showing/hiding the secret key."""
        try:
            show = self.show_secret_key.get()
            
            # Find the secret key entry widget
            for widget in self.root.winfo_children():
                self._find_and_update_secret_key_entry(widget, show)
        except Exception as e:
            self.logger.error(f"Error toggling secret key visibility: {e}")
    
    def _find_and_update_secret_key_entry(self, widget, show) -> None:
        """
        Recursively find and update the secret key entry widget.
        
        Args:
            widget: The widget to search in
            show: Whether to show the secret key
        """
        if isinstance(widget, tk.Entry) and widget.cget("textvariable") == str(self.secret_key):
            widget.config(show="" if show else "*")
        
        for child in widget.winfo_children():
            self._find_and_update_secret_key_entry(child, show)
    
    def _save_api_keys(self) -> None:
        """Save API keys and initialize the API."""
        try:
            account_address = self.account_address.get()
            secret_key = self.secret_key.get()
            
            # Validate inputs
            if not account_address or not secret_key:
                messagebox.showerror("Error", "Account address and secret key are required")
                return
            
            # Save API keys
            result = self.trading.set_api_keys(account_address, secret_key)
            
            if result["success"]:
                messagebox.showinfo("Success", "API keys saved successfully")
                self._update_connection_status()
                self._test_connection()
            else:
                messagebox.showerror("Error", f"Failed to save API keys: {result['message']}")
        except Exception as e:
            self.logger.error(f"Error saving API keys: {e}")
            messagebox.showerror("Error", f"Failed to save API keys: {str(e)}")
    
    def _test_connection(self) -> None:
        """Test the connection to the exchange."""
        try:
            self.logger.info("Testing connection to exchange...")
            
            # Test connection
            result = self.trading.test_connection()
            
            if result["success"]:
                messagebox.showinfo("Success", "Connection test successful")
            else:
                messagebox.showerror("Error", f"Connection test failed: {result['message']}")
        except Exception as e:
            self.logger.error(f"Error testing connection: {e}")
            messagebox.showerror("Error", f"Connection test failed: {str(e)}")
    
    def _place_order(self, is_buy: bool) -> None:
        """
        Place an order.
        
        Args:
            is_buy: Whether the order is a buy order
        """
        try:
            # Get inputs
            symbol = self.selected_symbol.get()
            size_str = self.position_size.get()
            
            # Validate inputs
            if not symbol:
                messagebox.showerror("Error", "Symbol is required")
                return
            
            try:
                size = float(size_str)
            except ValueError:
                messagebox.showerror("Error", "Invalid position size")
                return
            
            # Get market data
            market_data_result = self.trading.get_market_data(symbol)
            
            if not market_data_result["success"]:
                messagebox.showerror("Error", f"Failed to get market data: {market_data_result['message']}")
                return
            
            price = market_data_result["data"]["price"]
            
            # Confirm order
            action = "BUY" if is_buy else "SELL"
            if not messagebox.askyesno("Confirm Order", f"Place {action} order for {size} {symbol} at {price}?"):
                return
            
            # Place order
            result = self.trading.place_order(symbol, is_buy, size, price)
            
            if result["success"]:
                messagebox.showinfo("Success", f"{action} order placed successfully")
            else:
                messagebox.showerror("Error", f"Failed to place order: {result['message']}")
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            messagebox.showerror("Error", f"Failed to place order: {str(e)}")
    
    def _close_selected_position(self) -> None:
        """Close the selected position."""
        try:
            # Get selected position
            selection = self.positions_tree.selection()
            
            if not selection:
                messagebox.showerror("Error", "No position selected")
                return
            
            # Get position data
            item = self.positions_tree.item(selection[0])
            values = item["values"]
            
            symbol = values[0]
            
            # Confirm close
            if not messagebox.askyesno("Confirm Close", f"Close position for {symbol}?"):
                return
            
            # Close position
            result = self.trading.close_position(symbol)
            
            if result["success"]:
                messagebox.showinfo("Success", f"Position for {symbol} closed successfully")
                self._refresh_positions()
            else:
                messagebox.showerror("Error", f"Failed to close position: {result['message']}")
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            messagebox.showerror("Error", f"Failed to close position: {str(e)}")
    
    def _show_positions_context_menu(self, event) -> None:
        """
        Show the positions context menu.
        
        Args:
            event: The event that triggered the menu
        """
        try:
            # Get item under cursor
            item = self.positions_tree.identify_row(event.y)
            
            if item:
                # Select the item
                self.positions_tree.selection_set(item)
                
                # Show the menu
                self.positions_context_menu.post(event.x_root, event.y_root)
        except Exception as e:
            self.logger.error(f"Error showing positions context menu: {e}")
    
    def _toggle_bot(self) -> None:
        """Toggle the bot on/off."""
        try:
            if self.is_bot_running:
                # Stop the bot
                self.is_bot_running = False
                self.bot_button.config(text="Start Bot")
                self.style_manager.style_button(self.bot_button, "success")
                self.logger.info("Bot stopped")
            else:
                # Start the bot
                self.is_bot_running = True
                self.bot_button.config(text="Stop Bot")
                self.style_manager.style_button(self.bot_button, "error")
                self.logger.info("Bot started")
                
                # Start trading loop in a separate thread
                threading.Thread(target=self._trading_loop, daemon=True).start()
        except Exception as e:
            self.logger.error(f"Error toggling bot: {e}")
    
    def _trading_loop(self) -> None:
        """Trading loop for the bot."""
        try:
            symbol = self.selected_symbol.get()
            self.logger.info(f"Trading loop started for {symbol}")
            
            # Get account info
            account_info_result = self.trading.get_account_info()
            
            if not account_info_result["success"]:
                self.logger.error(f"Error getting account info: {account_info_result['message']}")
                return
            
            # Warmup period
            warmup_time = 20  # seconds
            for i in range(warmup_time, 0, -2):
                if not self.is_bot_running:
                    return
                
                self.logger.info(f"Warmup: {i}.0s left to gather initial data.")
                time.sleep(2)
            
            self.logger.info("Warmup complete")
            
            # Main trading loop
            while self.is_bot_running:
                try:
                    # Get market data
                    market_data_result = self.trading.get_market_data(symbol)
                    
                    if not market_data_result["success"]:
                        self.logger.error(f"Error fetching market data: {market_data_result['message']}")
                        time.sleep(7)
                        continue
                    
                    # Trading logic would go here
                    
                    # Sleep to avoid excessive API calls
                    time.sleep(7)
                except Exception as e:
                    self.logger.error(f"Error in trading loop: {e}")
                    time.sleep(7)
        except Exception as e:
            self.logger.error(f"Error in trading loop: {e}")


class LogTextHandler(logging.Handler):
    """
    Custom logging handler that writes logs to a tkinter Text widget.
    """
    
    def __init__(self, text_widget: tk.Text):
        """
        Initialize the handler.
        
        Args:
            text_widget: The Text widget to write logs to
        """
        super().__init__()
        self.text_widget = text_widget
        self.queue = []
        self.text_widget.after(100, self._process_queue)
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record.
        
        Args:
            record: The log record to emit
        """
        self.queue.append(record)
    
    def _process_queue(self) -> None:
        """Process the queue of log records."""
        while self.queue:
            record = self.queue.pop(0)
            msg = self.format(record)
            
            self.text_widget.config(state=tk.NORMAL)
            self.text_widget.insert(tk.END, msg + "\n")
            
            # Apply color based on log level
            if record.levelno >= logging.ERROR:
                self.text_widget.tag_add("error", "end-1l linestart", "end-1l lineend")
                self.text_widget.tag_config("error", foreground="red")
            elif record.levelno >= logging.WARNING:
                self.text_widget.tag_add("warning", "end-1l linestart", "end-1l lineend")
                self.text_widget.tag_config("warning", foreground="orange")
            
            self.text_widget.config(state=tk.DISABLED)
            self.text_widget.see(tk.END)
        
        self.text_widget.after(100, self._process_queue)


if __name__ == "__main__":
    root = tk.Tk()
    app = HyperliquidMasterGUI(root)
    root.mainloop()
