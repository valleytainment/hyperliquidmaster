"""
Main GUI Application for Hyperliquid Trading Bot

This module provides a graphical user interface for the Hyperliquid trading bot,
with automatic mock data mode during API rate limits.

Classes:
    HyperliquidGUI: Main GUI application
"""

import os
import sys
import json
import time
import logging
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core modules
from core.api_rate_limiter import APIRateLimiter
from core.error_handling import ErrorHandler

# Import data modules
from data.mock_data_provider import MockDataProvider

# Import strategy modules
from strategies.signal_generator import SignalGenerator
from strategies.master_strategy import MasterStrategy

# Import utility modules
from utils.technical_indicators import TechnicalIndicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/gui.log")
    ]
)
logger = logging.getLogger(__name__)

class HyperliquidGUI:
    """
    Main GUI application for Hyperliquid trading bot.
    
    This class provides a graphical user interface for the Hyperliquid trading bot,
    with automatic mock data mode during API rate limits.
    
    Attributes:
        root: Tkinter root window
        api_rate_limiter: API rate limiter
        error_handler: Error handler
        mock_data_provider: Mock data provider
        technical_indicators: Technical indicators calculator
        signal_generator: Signal generator
        master_strategy: Master strategy
        is_running: Whether the bot is running
        update_interval: Update interval in seconds
    """
    
    def __init__(self):
        """
        Initialize the GUI application.
        """
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Initialize components
        self.api_rate_limiter = APIRateLimiter()
        self.error_handler = ErrorHandler()
        self.mock_data_provider = MockDataProvider()
        self.technical_indicators = TechnicalIndicators()
        
        # Initialize signal generator and strategy
        self.signal_generator = SignalGenerator(
            self.technical_indicators,
            self.error_handler
        )
        
        self.master_strategy = MasterStrategy(
            "XRP",
            "1h",
            self.signal_generator,
            self.error_handler
        )
        
        # Initialize state
        self.is_running = False
        self.update_interval = 5  # seconds
        
        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("Hyperliquid Trading Bot")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.dashboard_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)
        self.logs_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.dashboard_tab, text="Dashboard")
        self.notebook.add(self.settings_tab, text="Settings")
        self.notebook.add(self.logs_tab, text="Logs")
        
        # Initialize tabs
        self._init_dashboard_tab()
        self._init_settings_tab()
        self._init_logs_tab()
        
        # Initialize status bar
        self._init_status_bar()
        
        # Set up update timer
        self.update_timer = None
        
        logger.info("GUI initialized")
    
    def _init_dashboard_tab(self):
        """
        Initialize the dashboard tab.
        """
        # Create frames
        control_frame = ttk.LabelFrame(self.dashboard_tab, text="Control", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        status_frame = ttk.LabelFrame(self.dashboard_tab, text="Status", padding=10)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        signals_frame = ttk.LabelFrame(self.dashboard_tab, text="Signals", padding=10)
        signals_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control frame
        self.start_button = ttk.Button(control_frame, text="Start", command=self.start_bot)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_bot, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = ttk.Button(control_frame, text="Reset", command=self.reset_bot)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        # Status frame
        status_grid = ttk.Frame(status_frame)
        status_grid.pack(fill=tk.X)
        
        ttk.Label(status_grid, text="API Status:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.api_status_label = ttk.Label(status_grid, text="Idle")
        self.api_status_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(status_grid, text="Mock Mode:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.mock_mode_label = ttk.Label(status_grid, text="Inactive")
        self.mock_mode_label.grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(status_grid, text="Symbol:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.symbol_label = ttk.Label(status_grid, text="XRP")
        self.symbol_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(status_grid, text="Timeframe:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        self.timeframe_label = ttk.Label(status_grid, text="1h")
        self.timeframe_label.grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Signals frame
        self.signals_text = scrolledtext.ScrolledText(signals_frame, wrap=tk.WORD, height=10)
        self.signals_text.pack(fill=tk.BOTH, expand=True)
        self.signals_text.config(state=tk.DISABLED)
    
    def _init_settings_tab(self):
        """
        Initialize the settings tab.
        """
        # Create frames
        api_frame = ttk.LabelFrame(self.settings_tab, text="API Settings", padding=10)
        api_frame.pack(fill=tk.X, padx=5, pady=5)
        
        strategy_frame = ttk.LabelFrame(self.settings_tab, text="Strategy Settings", padding=10)
        strategy_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # API settings
        api_grid = ttk.Frame(api_frame)
        api_grid.pack(fill=tk.X)
        
        ttk.Label(api_grid, text="Update Interval (seconds):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.update_interval_var = tk.StringVar(value=str(self.update_interval))
        update_interval_entry = ttk.Entry(api_grid, textvariable=self.update_interval_var, width=10)
        update_interval_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Strategy settings
        strategy_grid = ttk.Frame(strategy_frame)
        strategy_grid.pack(fill=tk.X)
        
        ttk.Label(strategy_grid, text="Risk Level (%):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.risk_level_var = tk.StringVar(value=str(self.master_strategy.config["risk_level"] * 100))
        risk_level_entry = ttk.Entry(strategy_grid, textvariable=self.risk_level_var, width=10)
        risk_level_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(strategy_grid, text="Take Profit Multiplier:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.tp_multiplier_var = tk.StringVar(value=str(self.master_strategy.config["take_profit_multiplier"]))
        tp_multiplier_entry = ttk.Entry(strategy_grid, textvariable=self.tp_multiplier_var, width=10)
        tp_multiplier_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(strategy_grid, text="Stop Loss Multiplier:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.sl_multiplier_var = tk.StringVar(value=str(self.master_strategy.config["stop_loss_multiplier"]))
        sl_multiplier_entry = ttk.Entry(strategy_grid, textvariable=self.sl_multiplier_var, width=10)
        sl_multiplier_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Checkboxes for strategy options
        self.use_volatility_filters_var = tk.BooleanVar(value=self.master_strategy.config["use_volatility_filters"])
        volatility_check = ttk.Checkbutton(strategy_grid, text="Use Volatility Filters", variable=self.use_volatility_filters_var)
        volatility_check.grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        
        self.use_trend_filters_var = tk.BooleanVar(value=self.master_strategy.config["use_trend_filters"])
        trend_check = ttk.Checkbutton(strategy_grid, text="Use Trend Filters", variable=self.use_trend_filters_var)
        trend_check.grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        
        self.use_volume_filters_var = tk.BooleanVar(value=self.master_strategy.config["use_volume_filters"])
        volume_check = ttk.Checkbutton(strategy_grid, text="Use Volume Filters", variable=self.use_volume_filters_var)
        volume_check.grid(row=2, column=2, sticky=tk.W, padx=5, pady=2)
        
        # Save button
        save_button = ttk.Button(self.settings_tab, text="Save Settings", command=self.save_settings)
        save_button.pack(pady=10)
    
    def _init_logs_tab(self):
        """
        Initialize the logs tab.
        """
        # Create log text widget
        self.log_text = scrolledtext.ScrolledText(self.logs_tab, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)
        
        # Add log handler
        self.log_handler = GUILogHandler(self.log_text)
        logger.addHandler(self.log_handler)
    
    def _init_status_bar(self):
        """
        Initialize the status bar.
        """
        self.status_bar = ttk.Frame(self.root, relief=tk.SUNKEN, padding=(2, 2))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = ttk.Label(self.status_bar, text="Ready")
        self.status_label.pack(side=tk.LEFT)
        
        self.time_label = ttk.Label(self.status_bar, text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.time_label.pack(side=tk.RIGHT)
        
        # Update time every second
        self._update_time()
    
    def _update_time(self):
        """
        Update the time label.
        """
        self.time_label.config(text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.root.after(1000, self._update_time)
    
    def start_bot(self):
        """
        Start the trading bot.
        """
        if self.is_running:
            return
        
        logger.info("Starting trading bot")
        
        # Update UI
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Running")
        
        # Set running flag
        self.is_running = True
        
        # Start update timer
        self._schedule_update()
        
        # Log
        self._append_to_signals("Bot started at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    def stop_bot(self):
        """
        Stop the trading bot.
        """
        if not self.is_running:
            return
        
        logger.info("Stopping trading bot")
        
        # Update UI
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Stopped")
        
        # Set running flag
        self.is_running = False
        
        # Cancel update timer
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
            self.update_timer = None
        
        # Log
        self._append_to_signals("Bot stopped at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    def reset_bot(self):
        """
        Reset the trading bot.
        """
        logger.info("Resetting trading bot")
        
        # Stop the bot if running
        if self.is_running:
            self.stop_bot()
        
        # Reset components
        self.api_rate_limiter.reset()
        
        # Update UI
        self.api_status_label.config(text="Idle")
        self.mock_mode_label.config(text="Inactive")
        self.status_label.config(text="Ready")
        
        # Clear signals
        self.signals_text.config(state=tk.NORMAL)
        self.signals_text.delete(1.0, tk.END)
        self.signals_text.config(state=tk.DISABLED)
        
        # Log
        logger.info("Bot reset")
    
    def save_settings(self):
        """
        Save settings.
        """
        try:
            # Get values from UI
            self.update_interval = int(self.update_interval_var.get())
            
            # Update strategy config
            strategy_config = {
                "risk_level": float(self.risk_level_var.get()) / 100,
                "take_profit_multiplier": float(self.tp_multiplier_var.get()),
                "stop_loss_multiplier": float(self.sl_multiplier_var.get()),
                "use_volatility_filters": self.use_volatility_filters_var.get(),
                "use_trend_filters": self.use_trend_filters_var.get(),
                "use_volume_filters": self.use_volume_filters_var.get()
            }
            
            self.master_strategy.update_config(strategy_config)
            
            # Log
            logger.info("Settings saved")
            messagebox.showinfo("Settings", "Settings saved successfully")
        
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            messagebox.showerror("Error", f"Error saving settings: {e}")
    
    def _schedule_update(self):
        """
        Schedule the next update.
        """
        if self.is_running:
            self.update_timer = self.root.after(self.update_interval * 1000, self._update)
    
    def _update(self):
        """
        Update the bot state.
        """
        try:
            # Check API rate limiter status
            api_status = self.api_rate_limiter.get_status()
            
            # Update UI
            self.api_status_label.config(text="Rate Limited" if api_status["is_limited"] else "Normal")
            self.mock_mode_label.config(text="Active" if api_status["mock_mode_active"] else "Inactive")
            
            # Get market data
            if api_status["mock_mode_active"]:
                # Use mock data
                market_data = self.mock_data_provider.get_market_data("XRP")
                klines = self.mock_data_provider.get_klines("XRP", "1h", limit=100)
                
                # Convert to DataFrame
                df = pd.DataFrame(klines)
                
                # Set data on strategy
                self.master_strategy.set_data(df)
                
                # Log
                logger.info("Using mock data due to API rate limits")
            else:
                # Use real API data (not implemented in this example)
                # In a real implementation, this would call the Hyperliquid API
                # For now, we'll use mock data as a placeholder
                market_data = self.mock_data_provider.get_market_data("XRP")
                klines = self.mock_data_provider.get_klines("XRP", "1h", limit=100)
                
                # Convert to DataFrame
                df = pd.DataFrame(klines)
                
                # Set data on strategy
                self.master_strategy.set_data(df)
                
                # Record API call
                self.api_rate_limiter.record_call("market_data")
                self.api_rate_limiter.record_call("klines")
            
            # Generate signal
            signal = self.master_strategy.generate_signal()
            
            # Log signal
            signal_text = "NEUTRAL"
            if signal > 0:
                signal_text = "BUY"
            elif signal < 0:
                signal_text = "SELL"
            
            self._append_to_signals(f"[{datetime.now().strftime('%H:%M:%S')}] Signal: {signal_text} ({signal})")
            
            # Schedule next update
            self._schedule_update()
        
        except Exception as e:
            logger.error(f"Error in update: {e}")
            self._append_to_signals(f"Error: {e}")
            
            # Schedule next update
            self._schedule_update()
    
    def _append_to_signals(self, text):
        """
        Append text to the signals text widget.
        
        Args:
            text (str): Text to append
        """
        self.signals_text.config(state=tk.NORMAL)
        self.signals_text.insert(tk.END, text + "\n")
        self.signals_text.see(tk.END)
        self.signals_text.config(state=tk.DISABLED)
    
    def run(self):
        """
        Run the GUI application.
        """
        logger.info("Starting GUI application")
        self.root.mainloop()


class GUILogHandler(logging.Handler):
    """
    Custom log handler for the GUI.
    
    This class handles logging messages and displays them in the GUI.
    
    Attributes:
        text_widget: Text widget to display logs
    """
    
    def __init__(self, text_widget):
        """
        Initialize the log handler.
        
        Args:
            text_widget: Text widget to display logs
        """
        super().__init__()
        self.text_widget = text_widget
        
        # Set formatter
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        self.setFormatter(formatter)
    
    def emit(self, record):
        """
        Emit a log record.
        
        Args:
            record: Log record
        """
        msg = self.format(record)
        
        # Add to text widget
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.insert(tk.END, msg + "\n")
        self.text_widget.see(tk.END)
        self.text_widget.config(state=tk.DISABLED)


def main():
    """
    Main function.
    """
    # Create and run GUI
    gui = HyperliquidGUI()
    gui.run()


if __name__ == "__main__":
    main()
