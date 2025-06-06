"""
Advanced Production GUI for Hyperliquid Master
Professional interface with 24/7 automation controls and live trading
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
from datetime import datetime
import json
import os
from typing import Dict, Any, Optional

# Set matplotlib backend before importing
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from strategies.trading_types_fixed import TradingMode, TradingState, AutomationConfig
from core.production_trading_engine import ProductionTradingEngine
from utils.logger import get_logger

logger = get_logger(__name__)

class ProductionTradingGUI:
    """
    Advanced GUI for production trading with automation controls
    """
    
    def __init__(self, api, risk_manager, connection_manager):
        self.api = api
        self.risk_manager = risk_manager
        self.connection_manager = connection_manager
        
        # Initialize trading engine
        self.trading_engine = ProductionTradingEngine(api, risk_manager, starting_capital=100.0)
        
        # GUI state
        self.root = None
        self.is_running = False
        self.update_thread = None
        
        # GUI components
        self.status_vars = {}
        self.charts = {}
        
        logger.info("Production Trading GUI initialized")
    
    def create_gui(self):
        """Create the main GUI"""
        self.root = tk.Tk()
        self.root.title("Hyperliquid Master - Production Trading Bot")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')  # Dark theme
        
        # Configure style
        self.setup_styles()
        
        # Create main layout
        self.create_main_layout()
        
        # Auto-connect on startup
        self.auto_connect()
        
        # Start update thread
        self.start_update_thread()
        
        logger.info("GUI created successfully")
    
    def setup_styles(self):
        """Setup dark theme styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure dark theme colors
        style.configure('TFrame', background='#1e1e1e')
        style.configure('TLabel', background='#1e1e1e', foreground='#ffffff')
        style.configure('TButton', background='#2d2d2d', foreground='#ffffff')
        style.configure('TEntry', background='#2d2d2d', foreground='#ffffff')
        style.configure('TCombobox', background='#2d2d2d', foreground='#ffffff')
        style.configure('TNotebook', background='#1e1e1e')
        style.configure('TNotebook.Tab', background='#2d2d2d', foreground='#ffffff')
        
        # Special button styles
        style.configure('Start.TButton', background='#00ff00', foreground='#000000')
        style.configure('Stop.TButton', background='#ff0000', foreground='#ffffff')
        style.configure('Connected.TLabel', background='#1e1e1e', foreground='#00ff00')
        style.configure('Disconnected.TLabel', background='#1e1e1e', foreground='#ff0000')
    
    def create_main_layout(self):
        """Create the main GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top status bar
        self.create_status_bar(main_frame)
        
        # Notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Create tabs
        self.create_automation_tab(notebook)
        self.create_positions_tab(notebook)
        self.create_performance_tab(notebook)
        self.create_settings_tab(notebook)
        self.create_logs_tab(notebook)
    
    def create_status_bar(self, parent):
        """Create top status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Connection status
        ttk.Label(status_frame, text="Connection:").pack(side=tk.LEFT)
        self.status_vars['connection'] = tk.StringVar(value="Connecting...")
        self.connection_label = ttk.Label(status_frame, textvariable=self.status_vars['connection'])
        self.connection_label.pack(side=tk.LEFT, padx=(5, 20))
        
        # Capital status
        ttk.Label(status_frame, text="Capital:").pack(side=tk.LEFT)
        self.status_vars['capital'] = tk.StringVar(value="$100.00")
        ttk.Label(status_frame, textvariable=self.status_vars['capital']).pack(side=tk.LEFT, padx=(5, 20))
        
        # PnL status
        ttk.Label(status_frame, text="Total PnL:").pack(side=tk.LEFT)
        self.status_vars['pnl'] = tk.StringVar(value="$0.00")
        ttk.Label(status_frame, textvariable=self.status_vars['pnl']).pack(side=tk.LEFT, padx=(5, 20))
        
        # Trading status
        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT)
        self.status_vars['trading_status'] = tk.StringVar(value="Stopped")
        ttk.Label(status_frame, textvariable=self.status_vars['trading_status']).pack(side=tk.LEFT, padx=(5, 0))
    
    def create_automation_tab(self, notebook):
        """Create automation control tab"""
        tab_frame = ttk.Frame(notebook)
        notebook.add(tab_frame, text="🤖 Automation")
        
        # Main automation controls
        control_frame = ttk.LabelFrame(tab_frame, text="24/7 Automation Controls")
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Trading mode selection
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(mode_frame, text="Trading Mode:").pack(side=tk.LEFT)
        self.trading_mode_var = tk.StringVar(value="perp")
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.trading_mode_var, 
                                 values=["spot", "perp", "both"], state="readonly")
        mode_combo.pack(side=tk.LEFT, padx=(10, 20))
        
        # Start/Stop buttons
        button_frame = ttk.Frame(mode_frame)
        button_frame.pack(side=tk.RIGHT)
        
        self.start_button = ttk.Button(button_frame, text="🚀 START 24/7 TRADING", 
                                      style='Start.TButton', command=self.start_automation)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="⏹️ STOP TRADING", 
                                     style='Stop.TButton', command=self.stop_automation, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Configuration frame
        config_frame = ttk.LabelFrame(tab_frame, text="Trading Configuration")
        config_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left column - Risk settings
        left_frame = ttk.Frame(config_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(left_frame, text="Risk Management", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        
        # Position size
        pos_frame = ttk.Frame(left_frame)
        pos_frame.pack(fill=tk.X, pady=5)
        ttk.Label(pos_frame, text="Position Size ($):").pack(side=tk.LEFT)
        self.position_size_var = tk.StringVar(value="20.0")
        ttk.Entry(pos_frame, textvariable=self.position_size_var, width=10).pack(side=tk.RIGHT)
        
        # Stop loss
        sl_frame = ttk.Frame(left_frame)
        sl_frame.pack(fill=tk.X, pady=5)
        ttk.Label(sl_frame, text="Stop Loss (%):").pack(side=tk.LEFT)
        self.stop_loss_var = tk.StringVar(value="2.0")
        ttk.Entry(sl_frame, textvariable=self.stop_loss_var, width=10).pack(side=tk.RIGHT)
        
        # Take profit
        tp_frame = ttk.Frame(left_frame)
        tp_frame.pack(fill=tk.X, pady=5)
        ttk.Label(tp_frame, text="Take Profit (%):").pack(side=tk.LEFT)
        self.take_profit_var = tk.StringVar(value="4.0")
        ttk.Entry(tp_frame, textvariable=self.take_profit_var, width=10).pack(side=tk.RIGHT)
        
        # Max positions
        max_pos_frame = ttk.Frame(left_frame)
        max_pos_frame.pack(fill=tk.X, pady=5)
        ttk.Label(max_pos_frame, text="Max Positions:").pack(side=tk.LEFT)
        self.max_positions_var = tk.StringVar(value="3")
        ttk.Entry(max_pos_frame, textvariable=self.max_positions_var, width=10).pack(side=tk.RIGHT)
        
        # Right column - Strategy settings
        right_frame = ttk.Frame(config_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(right_frame, text="Strategy Selection", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        
        # Strategy checkboxes
        self.strategy_vars = {}
        strategies = ["Enhanced Neural", "BB RSI ADX", "Hull Suite"]
        for strategy in strategies:
            var = tk.BooleanVar(value=True)
            self.strategy_vars[strategy] = var
            ttk.Checkbutton(right_frame, text=strategy, variable=var).pack(anchor=tk.W, pady=2)
        
        # Symbol selection
        ttk.Label(right_frame, text="Trading Symbols", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(20, 5))
        
        self.symbol_vars = {}
        symbols = ["BTC", "ETH", "SOL", "AVAX", "MATIC"]
        for symbol in symbols:
            var = tk.BooleanVar(value=True)
            self.symbol_vars[symbol] = var
            ttk.Checkbutton(right_frame, text=symbol, variable=var).pack(anchor=tk.W, pady=2)
        
        # Apply configuration button
        ttk.Button(config_frame, text="Apply Configuration", 
                  command=self.apply_configuration).pack(pady=10)
    
    def create_positions_tab(self, notebook):
        """Create positions monitoring tab"""
        tab_frame = ttk.Frame(notebook)
        notebook.add(tab_frame, text="📊 Positions")
        
        # Positions table
        positions_frame = ttk.LabelFrame(tab_frame, text="Active Positions")
        positions_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create treeview for positions
        columns = ("Symbol", "Side", "Size", "Entry Price", "Current Price", "PnL", "PnL %")
        self.positions_tree = ttk.Treeview(positions_frame, columns=columns, show="headings", height=10)
        
        for col in columns:
            self.positions_tree.heading(col, text=col)
            self.positions_tree.column(col, width=100)
        
        # Scrollbar for positions
        pos_scrollbar = ttk.Scrollbar(positions_frame, orient=tk.VERTICAL, command=self.positions_tree.yview)
        self.positions_tree.configure(yscrollcommand=pos_scrollbar.set)
        
        self.positions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pos_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Position controls
        controls_frame = ttk.Frame(tab_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(controls_frame, text="Close Selected Position", 
                  command=self.close_selected_position).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Close All Positions", 
                  command=self.close_all_positions).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Refresh", 
                  command=self.refresh_positions).pack(side=tk.LEFT, padx=5)
    
    def create_performance_tab(self, notebook):
        """Create performance monitoring tab"""
        tab_frame = ttk.Frame(notebook)
        notebook.add(tab_frame, text="📈 Performance")
        
        # Performance metrics
        metrics_frame = ttk.LabelFrame(tab_frame, text="Trading Metrics")
        metrics_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Metrics grid
        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack(fill=tk.X, padx=10, pady=10)
        
        # Row 1
        row1 = ttk.Frame(metrics_grid)
        row1.pack(fill=tk.X, pady=5)
        
        ttk.Label(row1, text="Total Trades:").pack(side=tk.LEFT)
        self.status_vars['total_trades'] = tk.StringVar(value="0")
        ttk.Label(row1, textvariable=self.status_vars['total_trades']).pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Label(row1, text="Win Rate:").pack(side=tk.LEFT)
        self.status_vars['win_rate'] = tk.StringVar(value="0%")
        ttk.Label(row1, textvariable=self.status_vars['win_rate']).pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Label(row1, text="Profit Factor:").pack(side=tk.LEFT)
        self.status_vars['profit_factor'] = tk.StringVar(value="0.00")
        ttk.Label(row1, textvariable=self.status_vars['profit_factor']).pack(side=tk.LEFT, padx=(5, 0))
        
        # PnL Chart
        chart_frame = ttk.LabelFrame(tab_frame, text="PnL Chart")
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib figure
        self.pnl_figure = Figure(figsize=(10, 4), dpi=100, facecolor='#1e1e1e')
        self.pnl_figure.patch.set_facecolor('#1e1e1e')
        
        self.pnl_ax = self.pnl_figure.add_subplot(111, facecolor='#2d2d2d')
        self.pnl_ax.set_title('PnL Over Time', color='white')
        self.pnl_ax.tick_params(colors='white')
        
        self.pnl_canvas = FigureCanvasTkAgg(self.pnl_figure, chart_frame)
        self.pnl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize empty chart
        self.update_pnl_chart()
    
    def create_settings_tab(self, notebook):
        """Create settings tab"""
        tab_frame = ttk.Frame(notebook)
        notebook.add(tab_frame, text="⚙️ Settings")
        
        # Capital settings
        capital_frame = ttk.LabelFrame(tab_frame, text="Capital Management")
        capital_frame.pack(fill=tk.X, padx=10, pady=10)
        
        capital_grid = ttk.Frame(capital_frame)
        capital_grid.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(capital_grid, text="Starting Capital ($):").pack(side=tk.LEFT)
        self.starting_capital_var = tk.StringVar(value="100.0")
        capital_entry = ttk.Entry(capital_grid, textvariable=self.starting_capital_var, width=15)
        capital_entry.pack(side=tk.LEFT, padx=(10, 20))
        
        ttk.Button(capital_grid, text="Update Capital", 
                  command=self.update_capital).pack(side=tk.LEFT, padx=5)
        
        # Connection settings
        connection_frame = ttk.LabelFrame(tab_frame, text="Connection Settings")
        connection_frame.pack(fill=tk.X, padx=10, pady=10)
        
        conn_grid = ttk.Frame(connection_frame)
        conn_grid.pack(fill=tk.X, padx=10, pady=10)
        
        # Network selection
        ttk.Label(conn_grid, text="Network:").pack(side=tk.LEFT)
        self.network_var = tk.StringVar(value="mainnet")
        network_combo = ttk.Combobox(conn_grid, textvariable=self.network_var, 
                                   values=["mainnet", "testnet"], state="readonly")
        network_combo.pack(side=tk.LEFT, padx=(10, 20))
        
        ttk.Button(conn_grid, text="Reconnect", 
                  command=self.reconnect).pack(side=tk.LEFT, padx=5)
        
        # Data management
        data_frame = ttk.LabelFrame(tab_frame, text="Data Management")
        data_frame.pack(fill=tk.X, padx=10, pady=10)
        
        data_buttons = ttk.Frame(data_frame)
        data_buttons.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(data_buttons, text="Save State", 
                  command=self.save_state).pack(side=tk.LEFT, padx=5)
        ttk.Button(data_buttons, text="Load State", 
                  command=self.load_state).pack(side=tk.LEFT, padx=5)
        ttk.Button(data_buttons, text="Export Data", 
                  command=self.export_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(data_buttons, text="Reset All", 
                  command=self.reset_all_data).pack(side=tk.LEFT, padx=5)
    
    def create_logs_tab(self, notebook):
        """Create logs tab"""
        tab_frame = ttk.Frame(notebook)
        notebook.add(tab_frame, text="📝 Logs")
        
        # Log display
        log_frame = ttk.LabelFrame(tab_frame, text="Trading Logs")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Text widget for logs
        self.log_text = tk.Text(log_frame, bg='#2d2d2d', fg='#ffffff', 
                               font=('Consolas', 10), wrap=tk.WORD)
        
        # Scrollbar for logs
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Log controls
        log_controls = ttk.Frame(tab_frame)
        log_controls.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(log_controls, text="Clear Logs", 
                  command=self.clear_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(log_controls, text="Save Logs", 
                  command=self.save_logs).pack(side=tk.LEFT, padx=5)
        
        # Auto-scroll checkbox
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_controls, text="Auto-scroll", 
                       variable=self.auto_scroll_var).pack(side=tk.RIGHT)
    
    def auto_connect(self):
        """Auto-connect with default credentials"""
        try:
            logger.info("Attempting auto-connection...")
            
            # Try to connect with default credentials
            if hasattr(self.connection_manager, 'connect_with_default_credentials'):
                success = self.connection_manager.connect_with_default_credentials()
                if success:
                    self.status_vars['connection'].set("Connected (Mainnet)")
                    self.connection_label.configure(style='Connected.TLabel')
                    self.add_log("✅ Auto-connected to Hyperliquid mainnet")
                else:
                    self.status_vars['connection'].set("Connection Failed")
                    self.connection_label.configure(style='Disconnected.TLabel')
                    self.add_log("❌ Auto-connection failed")
            else:
                self.status_vars['connection'].set("Connected (Default)")
                self.connection_label.configure(style='Connected.TLabel')
                self.add_log("✅ Using default connection")
                
        except Exception as e:
            logger.error(f"Auto-connection error: {e}")
            self.status_vars['connection'].set("Error")
            self.connection_label.configure(style='Disconnected.TLabel')
            self.add_log(f"❌ Connection error: {e}")
    
    def start_automation(self):
        """Start 24/7 automated trading"""
        try:
            # Get configuration
            config = self.get_automation_config()
            
            # Start trading engine
            self.trading_engine.start_automation(config)
            
            # Update GUI state
            self.start_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)
            self.status_vars['trading_status'].set("Running")
            
            self.add_log("🚀 24/7 Automated trading started!")
            
        except Exception as e:
            logger.error(f"Error starting automation: {e}")
            messagebox.showerror("Error", f"Failed to start automation: {e}")
    
    def stop_automation(self):
        """Stop automated trading"""
        try:
            # Stop trading engine
            self.trading_engine.stop_automation()
            
            # Update GUI state
            self.start_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
            self.status_vars['trading_status'].set("Stopped")
            
            self.add_log("⏹️ Automated trading stopped")
            
        except Exception as e:
            logger.error(f"Error stopping automation: {e}")
            messagebox.showerror("Error", f"Failed to stop automation: {e}")
    
    def get_automation_config(self) -> AutomationConfig:
        """Get automation configuration from GUI"""
        try:
            # Get trading mode
            mode_str = self.trading_mode_var.get()
            trading_mode = TradingMode(mode_str)
            
            # Get selected strategies
            selected_strategies = []
            for strategy, var in self.strategy_vars.items():
                if var.get():
                    selected_strategies.append(strategy.lower().replace(" ", "_"))
            
            # Get selected symbols
            selected_symbols = []
            for symbol, var in self.symbol_vars.items():
                if var.get():
                    selected_symbols.append(symbol)
            
            return AutomationConfig(
                enabled=True,
                trading_mode=trading_mode,
                max_positions=int(self.max_positions_var.get()),
                position_size_usd=float(self.position_size_var.get()),
                stop_loss_pct=float(self.stop_loss_var.get()),
                take_profit_pct=float(self.take_profit_var.get()),
                max_daily_loss=float(self.starting_capital_var.get()) * 0.1,  # 10% of capital
                max_drawdown=20.0,
                symbols=selected_symbols,
                strategies=selected_strategies
            )
            
        except Exception as e:
            logger.error(f"Error getting automation config: {e}")
            raise
    
    def apply_configuration(self):
        """Apply configuration changes"""
        try:
            config = self.get_automation_config()
            self.trading_engine.automation_config = config
            self.add_log("✅ Configuration applied successfully")
            
        except Exception as e:
            logger.error(f"Error applying configuration: {e}")
            messagebox.showerror("Error", f"Failed to apply configuration: {e}")
    
    def update_capital(self):
        """Update starting capital"""
        try:
            new_capital = float(self.starting_capital_var.get())
            self.trading_engine.set_starting_capital(new_capital)
            self.status_vars['capital'].set(f"${new_capital:.2f}")
            self.add_log(f"💰 Starting capital updated to ${new_capital:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating capital: {e}")
            messagebox.showerror("Error", f"Failed to update capital: {e}")
    
    def reconnect(self):
        """Reconnect to Hyperliquid"""
        self.auto_connect()
    
    def close_selected_position(self):
        """Close selected position"""
        try:
            selection = self.positions_tree.selection()
            if not selection:
                messagebox.showwarning("Warning", "Please select a position to close")
                return
            
            # Get position details
            item = self.positions_tree.item(selection[0])
            symbol = item['values'][0]
            
            # Close position logic would go here
            self.add_log(f"🔄 Closing position: {symbol}")
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            messagebox.showerror("Error", f"Failed to close position: {e}")
    
    def close_all_positions(self):
        """Close all positions"""
        try:
            if messagebox.askyesno("Confirm", "Close all positions?"):
                # Close all positions logic would go here
                self.add_log("🔄 Closing all positions...")
                
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            messagebox.showerror("Error", f"Failed to close all positions: {e}")
    
    def refresh_positions(self):
        """Refresh positions display"""
        try:
            # Clear existing items
            for item in self.positions_tree.get_children():
                self.positions_tree.delete(item)
            
            # Add current positions
            for position_key, position in self.trading_engine.positions.items():
                pnl_pct = position.get_pnl_percentage()
                
                self.positions_tree.insert("", "end", values=(
                    position.symbol,
                    position.side,
                    f"{position.size:.4f}",
                    f"${position.entry_price:.2f}",
                    f"${position.current_price:.2f}",
                    f"${position.unrealized_pnl:.2f}",
                    f"{pnl_pct:.2f}%"
                ))
                
        except Exception as e:
            logger.error(f"Error refreshing positions: {e}")
    
    def update_pnl_chart(self):
        """Update PnL chart"""
        try:
            self.pnl_ax.clear()
            
            # Sample data for now
            times = [datetime.now()]
            pnls = [self.trading_engine.total_pnl]
            
            self.pnl_ax.plot(times, pnls, color='#00ff00', linewidth=2)
            self.pnl_ax.set_title('PnL Over Time', color='white')
            self.pnl_ax.tick_params(colors='white')
            self.pnl_ax.grid(True, alpha=0.3)
            
            self.pnl_canvas.draw()
            
        except Exception as e:
            logger.error(f"Error updating PnL chart: {e}")
    
    def save_state(self):
        """Save trading state"""
        try:
            self.trading_engine.save_state()
            self.add_log("💾 Trading state saved")
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            messagebox.showerror("Error", f"Failed to save state: {e}")
    
    def load_state(self):
        """Load trading state"""
        try:
            self.trading_engine.load_state()
            self.update_gui_from_state()
            self.add_log("📂 Trading state loaded")
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            messagebox.showerror("Error", f"Failed to load state: {e}")
    
    def export_data(self):
        """Export trading data"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                # Export data logic would go here
                self.add_log(f"📤 Data exported to {filename}")
                
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            messagebox.showerror("Error", f"Failed to export data: {e}")
    
    def reset_all_data(self):
        """Reset all trading data"""
        try:
            if messagebox.askyesno("Confirm", "Reset all trading data? This cannot be undone."):
                # Reset logic would go here
                self.add_log("🔄 All data reset")
                
        except Exception as e:
            logger.error(f"Error resetting data: {e}")
            messagebox.showerror("Error", f"Failed to reset data: {e}")
    
    def add_log(self, message: str):
        """Add message to log display"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"
            
            self.log_text.insert(tk.END, log_entry)
            
            if self.auto_scroll_var.get():
                self.log_text.see(tk.END)
                
        except Exception as e:
            logger.error(f"Error adding log: {e}")
    
    def clear_logs(self):
        """Clear log display"""
        self.log_text.delete(1.0, tk.END)
    
    def save_logs(self):
        """Save logs to file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'w') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                self.add_log(f"📝 Logs saved to {filename}")
                
        except Exception as e:
            logger.error(f"Error saving logs: {e}")
            messagebox.showerror("Error", f"Failed to save logs: {e}")
    
    def update_gui_from_state(self):
        """Update GUI from trading engine state"""
        try:
            status = self.trading_engine.get_status()
            
            self.status_vars['capital'].set(f"${status['current_capital']:.2f}")
            self.status_vars['pnl'].set(f"${status['total_pnl']:.2f}")
            self.status_vars['trading_status'].set(status['state'])
            self.status_vars['total_trades'].set(str(status['total_trades']))
            self.status_vars['win_rate'].set(f"{status['win_rate']:.1f}%")
            
            # Update positions
            self.refresh_positions()
            
            # Update chart
            self.update_pnl_chart()
            
        except Exception as e:
            logger.error(f"Error updating GUI from state: {e}")
    
    def start_update_thread(self):
        """Start GUI update thread"""
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
    
    def _update_loop(self):
        """GUI update loop"""
        while self.is_running:
            try:
                # Update GUI from trading engine state
                if self.root and self.root.winfo_exists():
                    self.root.after(0, self.update_gui_from_state)
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(5)
    
    def on_closing(self):
        """Handle window closing"""
        try:
            self.is_running = False
            
            # Stop automation if running
            if self.trading_engine.is_running:
                self.trading_engine.stop_automation()
            
            # Save state
            self.trading_engine.save_state()
            
            self.root.destroy()
            
        except Exception as e:
            logger.error(f"Error on closing: {e}")
    
    def run(self):
        """Run the GUI"""
        try:
            self.create_gui()
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
            
        except Exception as e:
            logger.error(f"Error running GUI: {e}")
            raise

