"""
Enhanced GUI main module for the HyperliquidMaster trading bot.
Integrates the new settings management system for flawless API key handling.
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
from core.settings_manager import SettingsManager
from gui_settings_integration import GUISettingsIntegration

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
        
        # Initialize settings manager
        self.settings_manager = SettingsManager(self.config_path, self.logger)
        self.config = self.settings_manager.settings
        
        # Initialize style manager
        self.style_manager = GUIStyleManager(self.root, self.logger)
        
        # Initialize error handler
        self.error_handler = ErrorHandler(self.logger)
        
        # Initialize trading integration
        self.trading = TradingIntegration(self.config_path, self.logger)
        
        # Initialize settings integration
        self.settings_integration = GUISettingsIntegration(
            self.root, 
            self.settings_manager, 
            self.trading, 
            self.logger
        )
        
        # Initialize variables
        self.is_bot_running = False
        self.selected_symbol = tk.StringVar(value=self.config.get("symbol", "BTC"))
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
        self.logger.info("Applied dark theme")
        self.logger.info(f"Loaded settings from {self.config_path}")
        self.logger.info(f"Created settings backup at {self.settings_manager.backup_dir}/settings_{int(time.time())}.json")
    
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
        """Initialize the settings tab using the settings integration."""
        # Use the settings integration to create the settings tab
        self.settings_integration.create_settings_tab(self.settings_tab, self.style_manager)
        
        # Create trading settings section
        trading_frame = ttk.Frame(self.settings_tab)
        trading_frame.pack(fill=tk.X, padx=10, pady=(20, 10))
        
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
        save_settings_button = tk.Button(trading_frame, text="Save Trading Settings", command=self._save_trading_settings)
        save_settings_button.pack(anchor=tk.W, pady=5)
        self.style_manager.style_button(save_settings_button, "success")
    
    def _init_logs_tab(self) -> None:
        """Initialize the logs tab."""
        # Create log text widget
        self.log_text = tk.Text(self.logs_tab, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create log handler
        self.log_handler = TextHandler(self.log_text)
        self.log_handler.setLevel(logging.INFO)
        self.log_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        
        # Add handler to logger
        self.logger.addHandler(self.log_handler)
        
        # Create clear logs button
        clear_button = tk.Button(self.logs_tab, text="Clear Logs", command=self._clear_logs)
        clear_button.pack(anchor=tk.E, padx=10, pady=(0, 10))
        self.style_manager.style_button(clear_button)
    
    def _create_chart(self) -> None:
        """Create the price chart."""
        # Create figure and axis
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize with empty data
        self.ax.plot([], [])
        self.ax.set_title("Price Chart")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Price")
        self.fig.tight_layout()
    
    def _update_chart(self) -> None:
        """Update the price chart with new data."""
        try:
            # Get market data
            symbol = self.selected_symbol.get()
            result = self.trading.get_market_data(symbol)
            
            if not result.get("success", False):
                return
            
            # Get historical data (mock data for now)
            import numpy as np
            x = np.arange(100)
            y = np.sin(x / 10) * 100 + result["data"]["price"]
            
            # Clear previous plot
            self.ax.clear()
            
            # Plot new data
            self.ax.plot(x, y)
            
            # Set labels
            self.ax.set_title(f"{symbol} Price Chart")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Price")
            
            # Update canvas
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            self.logger.error(f"Error updating chart: {e}")
    
    def _on_symbol_selected(self, event) -> None:
        """Handle symbol selection."""
        try:
            symbol = self.selected_symbol.get()
            self.logger.info(f"Selected symbol: {symbol}")
            
            # Update chart
            self._update_chart()
            
            # Save to settings
            self._save_trading_settings()
        except Exception as e:
            self.logger.error(f"Error handling symbol selection: {e}")
    
    def _refresh_dashboard(self) -> None:
        """Refresh the dashboard."""
        try:
            # Update connection status
            self._update_connection_status()
            
            # Update chart
            self._update_chart()
            
            # Update available symbols
            self._update_available_symbols()
            
            self.logger.info("Dashboard refreshed")
        except Exception as e:
            self.logger.error(f"Error refreshing dashboard: {e}")
    
    def _update_connection_status(self) -> None:
        """Update the connection status indicator."""
        try:
            if self.trading.is_connected:
                self.connection_status.config(text="Connected", bg="green")
            else:
                self.connection_status.config(text="Not Connected", bg="red")
        except Exception as e:
            self.logger.error(f"Error updating connection status: {e}")
    
    def _update_available_symbols(self) -> None:
        """Update the available symbols list."""
        try:
            # Get available symbols (mock data for now)
            self.available_symbols = ["BTC", "ETH", "SOL", "AVAX", "MATIC"]
            
            # Update combobox
            self.symbol_combobox["values"] = self.available_symbols
            
            # Select current symbol if available
            current_symbol = self.selected_symbol.get()
            if current_symbol in self.available_symbols:
                self.symbol_combobox.set(current_symbol)
            elif self.available_symbols:
                self.symbol_combobox.set(self.available_symbols[0])
        except Exception as e:
            self.logger.error(f"Error updating available symbols: {e}")
    
    def _refresh_positions(self) -> None:
        """Refresh the positions list."""
        try:
            # Clear current positions
            for item in self.positions_tree.get_children():
                self.positions_tree.delete(item)
            
            # Get positions
            result = self.trading.get_positions()
            
            if not result.get("success", False):
                self.logger.error(f"Error getting positions: {result.get('message', 'Unknown error')}")
                return
            
            # Add positions to treeview
            positions = result.get("data", [])
            for pos in positions:
                # Format values
                symbol = pos.get("symbol", "Unknown")
                size = f"{pos.get('size', 0):.4f}"
                entry_price = f"${pos.get('entry_price', 0):.2f}"
                mark_price = f"${pos.get('mark_price', 0):.2f}"
                pnl = f"${pos.get('unrealized_pnl', 0):.2f}"
                pnl_percent = f"{pos.get('pnl_percentage', 0):.2f}%"
                
                # Add to treeview
                self.positions_tree.insert("", "end", values=(symbol, size, entry_price, mark_price, pnl, pnl_percent))
            
            self.logger.info("Positions refreshed")
        except Exception as e:
            self.logger.error(f"Error refreshing positions: {e}")
    
    def _show_positions_context_menu(self, event) -> None:
        """Show the positions context menu."""
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
    
    def _close_selected_position(self) -> None:
        """Close the selected position."""
        try:
            # Get selected item
            selected = self.positions_tree.selection()
            
            if not selected:
                return
            
            # Get position data
            item = selected[0]
            values = self.positions_tree.item(item, "values")
            symbol = values[0]
            
            # Confirm close
            if messagebox.askyesno("Confirm", f"Close position for {symbol}?"):
                # Close position
                result = self.trading.close_position(symbol)
                
                if result.get("success", False):
                    self.logger.info(f"Position closed for {symbol}")
                    messagebox.showinfo("Success", f"Position closed for {symbol}")
                    
                    # Refresh positions
                    self._refresh_positions()
                else:
                    self.logger.error(f"Error closing position: {result.get('message', 'Unknown error')}")
                    messagebox.showerror("Error", f"Error closing position: {result.get('message', 'Unknown error')}")
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    def _place_order(self, is_buy: bool) -> None:
        """Place a buy or sell order."""
        try:
            symbol = self.selected_symbol.get()
            size = float(self.position_size.get())
            stop_loss = float(self.stop_loss.get())
            take_profit = float(self.take_profit.get())
            
            # Confirm order
            order_type = "BUY" if is_buy else "SELL"
            if messagebox.askyesno("Confirm Order", f"Place {order_type} order for {size} {symbol}?"):
                # Place order
                result = self.trading.place_order(symbol, size, is_buy, stop_loss, take_profit)
                
                if result.get("success", False):
                    self.logger.info(f"{order_type} order placed for {size} {symbol}")
                    messagebox.showinfo("Success", f"{order_type} order placed for {size} {symbol}")
                    
                    # Refresh positions
                    self._refresh_positions()
                else:
                    self.logger.error(f"Error placing order: {result.get('message', 'Unknown error')}")
                    messagebox.showerror("Error", f"Error placing order: {result.get('message', 'Unknown error')}")
        except ValueError:
            self.logger.error("Invalid input values")
            messagebox.showerror("Error", "Invalid input values")
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
    
    def _toggle_bot(self) -> None:
        """Toggle the bot on/off."""
        try:
            if self.is_bot_running:
                # Stop bot
                self.is_bot_running = False
                self.bot_button.config(text="Start Bot")
                self.style_manager.style_button(self.bot_button, "success")
                self.logger.info("Bot stopped")
            else:
                # Start bot
                self.is_bot_running = True
                self.bot_button.config(text="Stop Bot")
                self.style_manager.style_button(self.bot_button, "error")
                self.logger.info("Bot started")
        except Exception as e:
            self.logger.error(f"Error toggling bot: {e}")
    
    def _save_trading_settings(self) -> None:
        """Save trading settings."""
        try:
            # Get values
            symbol = self.selected_symbol.get()
            position_size = self.position_size.get()
            stop_loss = self.stop_loss.get()
            take_profit = self.take_profit.get()
            
            # Update settings
            settings = {
                "symbol": symbol,
                "position_size": position_size,
                "stop_loss": stop_loss,
                "take_profit": take_profit
            }
            
            # Save settings
            result = self.settings_manager.update_settings(settings)
            
            if result:
                self.logger.info("Trading settings saved")
            else:
                self.logger.error("Error saving trading settings")
        except Exception as e:
            self.logger.error(f"Error saving trading settings: {e}")
    
    def _clear_logs(self) -> None:
        """Clear the logs."""
        try:
            self.log_text.config(state=tk.NORMAL)
            self.log_text.delete(1.0, tk.END)
            self.log_text.config(state=tk.DISABLED)
            self.logger.info("Logs cleared")
        except Exception as e:
            self.logger.error(f"Error clearing logs: {e}")
    
    def _start_update_loops(self) -> None:
        """Start the update loops."""
        try:
            # Start connection status update loop
            self._update_connection_status()
            self.root.after(5000, self._connection_status_loop)
            
            # Start positions update loop
            self._refresh_positions()
            self.root.after(10000, self._positions_update_loop)
            
            # Start chart update loop
            self._update_chart()
            self.root.after(15000, self._chart_update_loop)
        except Exception as e:
            self.logger.error(f"Error starting update loops: {e}")
    
    def _connection_status_loop(self) -> None:
        """Connection status update loop."""
        try:
            self._update_connection_status()
            self.root.after(5000, self._connection_status_loop)
        except Exception as e:
            self.logger.error(f"Error in connection status loop: {e}")
    
    def _positions_update_loop(self) -> None:
        """Positions update loop."""
        try:
            if self.notebook.index(self.notebook.select()) == 1:  # Positions tab is selected
                self._refresh_positions()
            self.root.after(10000, self._positions_update_loop)
        except Exception as e:
            self.logger.error(f"Error in positions update loop: {e}")
    
    def _chart_update_loop(self) -> None:
        """Chart update loop."""
        try:
            if self.notebook.index(self.notebook.select()) == 0:  # Dashboard tab is selected
                self._update_chart()
            self.root.after(15000, self._chart_update_loop)
        except Exception as e:
            self.logger.error(f"Error in chart update loop: {e}")


class TextHandler(logging.Handler):
    """Custom logging handler that writes to a tkinter Text widget."""
    
    def __init__(self, text_widget: tk.Text):
        """
        Initialize the handler.
        
        Args:
            text_widget: The Text widget to write to
        """
        super().__init__()
        self.text_widget = text_widget
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a record.
        
        Args:
            record: The log record
        """
        msg = self.format(record)
        
        def append():
            self.text_widget.configure(state=tk.NORMAL)
            self.text_widget.insert(tk.END, msg + "\n")
            self.text_widget.see(tk.END)
            self.text_widget.configure(state=tk.DISABLED)
        
        # Schedule append to be called in the main thread
        self.text_widget.after(0, append)


if __name__ == "__main__":
    root = tk.Tk()
    app = HyperliquidMasterGUI(root)
    root.mainloop()
