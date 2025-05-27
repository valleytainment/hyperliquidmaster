#!/usr/bin/env python3
"""
GUI Main Application - Optimized and Fixed

This module provides a GUI for the Hyperliquid trading bot with optimized performance,
enhanced error handling, and mock data support during API rate limits.
"""

import os
import sys
import json
import time
import logging
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

# Import strategy components
from strategies.robust_signal_generator_fixed_updated import RobustSignalGenerator
from strategies.master_omni_overlord_robust_standardized import MasterOmniOverlordRobustStrategy
from strategies.advanced_technical_indicators_fixed import AdvancedTechnicalIndicators
from error_handling_fixed_updated import ErrorHandler
from api_rate_limiter_enhanced import APIRateLimiter
from enhanced_mock_data_provider import EnhancedMockDataProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("gui_main.log")
    ]
)
logger = logging.getLogger(__name__)

class HyperliquidTradingBotGUI:
    """
    GUI for the Hyperliquid trading bot with optimized performance and enhanced features.
    """
    
    def __init__(self, root):
        """
        Initialize the GUI.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Hyperliquid Trading Bot")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Set theme
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Configure colors
        self.bg_color = "#2E3440"
        self.fg_color = "#ECEFF4"
        self.accent_color = "#88C0D0"
        self.warning_color = "#EBCB8B"
        self.error_color = "#BF616A"
        self.success_color = "#A3BE8C"
        
        # Apply colors
        self.root.configure(bg=self.bg_color)
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("TLabel", background=self.bg_color, foreground=self.fg_color)
        self.style.configure("TButton", background=self.accent_color, foreground=self.bg_color)
        self.style.map("TButton",
            background=[("active", self.accent_color), ("disabled", "#4C566A")],
            foreground=[("active", self.bg_color), ("disabled", "#D8DEE9")]
        )
        
        # Initialize components
        self.error_handler = ErrorHandler()
        self.technical_indicators = AdvancedTechnicalIndicators()
        self.api_rate_limiter = APIRateLimiter()
        self.mock_data_provider = EnhancedMockDataProvider()
        
        # Initialize data
        self.symbols = ["BTC", "ETH", "XRP", "SOL", "DOGE"]
        self.timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        self.strategies = {}
        self.market_data = {}
        self.signals = {}
        self.positions = {}
        self.running = False
        self.using_mock_data = False
        
        # Create GUI elements
        self._create_menu()
        self._create_main_frame()
        
        # Initialize status
        self.update_status("Ready")
        
        # Check API rate limiter status
        self._check_rate_limiter_status()
    
    def _create_menu(self):
        """
        Create menu bar.
        """
        self.menu_bar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Load Configuration", command=self._load_config)
        file_menu.add_command(label="Save Configuration", command=self._save_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        
        # Tools menu
        tools_menu = tk.Menu(self.menu_bar, tearoff=0)
        tools_menu.add_command(label="Backtest", command=self._open_backtest_window)
        tools_menu.add_command(label="Optimization", command=self._open_optimization_window)
        tools_menu.add_command(label="Data Viewer", command=self._open_data_viewer)
        tools_menu.add_separator()
        tools_menu.add_command(label="Reset API Rate Limiter", command=self._reset_rate_limiter)
        self.menu_bar.add_cascade(label="Tools", menu=tools_menu)
        
        # Help menu
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        help_menu.add_command(label="Documentation", command=self._show_documentation)
        help_menu.add_command(label="About", command=self._show_about)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=self.menu_bar)
    
    def _create_main_frame(self):
        """
        Create main frame with all GUI elements.
        """
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create top frame for controls
        self._create_control_frame()
        
        # Create middle frame for charts
        self._create_chart_frame()
        
        # Create bottom frame for logs and signals
        self._create_log_frame()
    
    def _create_control_frame(self):
        """
        Create control frame with settings and buttons.
        """
        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Left side - Settings
        settings_frame = ttk.LabelFrame(control_frame, text="Settings", padding=10)
        settings_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Symbol selection
        symbol_frame = ttk.Frame(settings_frame)
        symbol_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(symbol_frame, text="Symbol:").pack(side=tk.LEFT)
        
        self.symbol_var = tk.StringVar(value="XRP")
        symbol_combo = ttk.Combobox(symbol_frame, textvariable=self.symbol_var, values=self.symbols, width=10)
        symbol_combo.pack(side=tk.LEFT, padx=(5, 10))
        
        # Timeframe selection
        ttk.Label(symbol_frame, text="Timeframe:").pack(side=tk.LEFT)
        
        self.timeframe_var = tk.StringVar(value="1h")
        timeframe_combo = ttk.Combobox(symbol_frame, textvariable=self.timeframe_var, values=self.timeframes, width=10)
        timeframe_combo.pack(side=tk.LEFT, padx=(5, 10))
        
        # Strategy parameters
        params_frame = ttk.Frame(settings_frame)
        params_frame.pack(fill=tk.X, pady=5)
        
        # Risk level
        ttk.Label(params_frame, text="Risk Level:").pack(side=tk.LEFT)
        
        self.risk_var = tk.DoubleVar(value=0.02)
        risk_spinbox = ttk.Spinbox(params_frame, from_=0.01, to=0.1, increment=0.01, textvariable=self.risk_var, width=5)
        risk_spinbox.pack(side=tk.LEFT, padx=(5, 10))
        
        # Mock data checkbox
        self.mock_data_var = tk.BooleanVar(value=False)
        mock_data_check = ttk.Checkbutton(params_frame, text="Use Mock Data", variable=self.mock_data_var)
        mock_data_check.pack(side=tk.LEFT, padx=(10, 0))
        
        # Right side - Controls
        controls_frame = ttk.LabelFrame(control_frame, text="Controls", padding=10)
        controls_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Buttons
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.pack(fill=tk.X)
        
        self.start_button = ttk.Button(buttons_frame, text="Start", command=self._start_bot)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(buttons_frame, text="Stop", command=self._stop_bot, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.refresh_button = ttk.Button(buttons_frame, text="Refresh Data", command=self._refresh_data)
        self.refresh_button.pack(side=tk.LEFT, padx=5)
        
        # Status
        status_frame = ttk.Frame(controls_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # API Rate Limiter status
        rate_limit_frame = ttk.Frame(controls_frame)
        rate_limit_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(rate_limit_frame, text="API Rate Limit:").pack(side=tk.LEFT)
        
        self.rate_limit_var = tk.StringVar(value="OK")
        self.rate_limit_label = ttk.Label(rate_limit_frame, textvariable=self.rate_limit_var)
        self.rate_limit_label.pack(side=tk.LEFT, padx=(5, 0))
    
    def _create_chart_frame(self):
        """
        Create chart frame with price and indicator charts.
        """
        self.chart_frame = ttk.LabelFrame(self.main_frame, text="Charts", padding=10)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create figure for charts
        self.fig = plt.figure(figsize=(10, 6), facecolor=self.bg_color)
        
        # Price chart
        self.price_ax = self.fig.add_subplot(2, 1, 1)
        self.price_ax.set_facecolor(self.bg_color)
        self.price_ax.tick_params(colors=self.fg_color)
        self.price_ax.spines['bottom'].set_color(self.fg_color)
        self.price_ax.spines['top'].set_color(self.fg_color)
        self.price_ax.spines['left'].set_color(self.fg_color)
        self.price_ax.spines['right'].set_color(self.fg_color)
        
        # Indicator chart
        self.indicator_ax = self.fig.add_subplot(2, 1, 2, sharex=self.price_ax)
        self.indicator_ax.set_facecolor(self.bg_color)
        self.indicator_ax.tick_params(colors=self.fg_color)
        self.indicator_ax.spines['bottom'].set_color(self.fg_color)
        self.indicator_ax.spines['top'].set_color(self.fg_color)
        self.indicator_ax.spines['left'].set_color(self.fg_color)
        self.indicator_ax.spines['right'].set_color(self.fg_color)
        
        # Add canvas to frame
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Adjust spacing
        self.fig.tight_layout()
    
    def _create_log_frame(self):
        """
        Create log frame with logs and signals.
        """
        log_frame = ttk.Frame(self.main_frame)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for logs and signals
        notebook = ttk.Notebook(log_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Logs tab
        logs_frame = ttk.Frame(notebook, padding=10)
        notebook.add(logs_frame, text="Logs")
        
        self.log_text = scrolledtext.ScrolledText(logs_frame, height=10, bg="#3B4252", fg=self.fg_color)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Signals tab
        signals_frame = ttk.Frame(notebook, padding=10)
        notebook.add(signals_frame, text="Signals")
        
        self.signals_text = scrolledtext.ScrolledText(signals_frame, height=10, bg="#3B4252", fg=self.fg_color)
        self.signals_text.pack(fill=tk.BOTH, expand=True)
        
        # Positions tab
        positions_frame = ttk.Frame(notebook, padding=10)
        notebook.add(positions_frame, text="Positions")
        
        self.positions_text = scrolledtext.ScrolledText(positions_frame, height=10, bg="#3B4252", fg=self.fg_color)
        self.positions_text.pack(fill=tk.BOTH, expand=True)
    
    def _check_rate_limiter_status(self):
        """
        Check API rate limiter status and update GUI.
        """
        try:
            status = self.api_rate_limiter.get_status()
            
            if status["is_limited"]:
                self.rate_limit_var.set(f"Limited (Cooldown: {status['cooldown_remaining']}s)")
                self.rate_limit_label.configure(foreground=self.error_color)
                
                # Enable mock data if rate limited
                if not self.using_mock_data and self.running:
                    self.using_mock_data = True
                    self.mock_data_var.set(True)
                    self.log("API rate limited. Switching to mock data mode.", "warning")
            else:
                self.rate_limit_var.set("OK")
                self.rate_limit_label.configure(foreground=self.success_color)
                
                # Disable mock data if no longer rate limited
                if self.using_mock_data and self.running and self.mock_data_var.get():
                    self.using_mock_data = False
                    self.log("API rate limit cooldown completed. Switching back to live data.", "info")
            
            # Schedule next check
            self.root.after(5000, self._check_rate_limiter_status)
        
        except Exception as e:
            self.log(f"Error checking rate limiter status: {str(e)}", "error")
            self.root.after(10000, self._check_rate_limiter_status)
    
    def _reset_rate_limiter(self):
        """
        Reset API rate limiter.
        """
        try:
            self.api_rate_limiter.reset()
            self.log("API rate limiter reset successfully.", "info")
            self._check_rate_limiter_status()
        except Exception as e:
            self.log(f"Error resetting rate limiter: {str(e)}", "error")
    
    def _load_config(self):
        """
        Load configuration from file.
        """
        try:
            if os.path.exists("config.json"):
                with open("config.json", "r") as f:
                    config = json.load(f)
                
                # Apply configuration
                if "symbol" in config:
                    self.symbol_var.set(config["symbol"])
                
                if "timeframe" in config:
                    self.timeframe_var.set(config["timeframe"])
                
                if "risk_level" in config:
                    self.risk_var.set(config["risk_level"])
                
                if "use_mock_data" in config:
                    self.mock_data_var.set(config["use_mock_data"])
                
                self.log("Configuration loaded successfully.", "info")
            else:
                self.log("No configuration file found.", "warning")
        except Exception as e:
            self.log(f"Error loading configuration: {str(e)}", "error")
    
    def _save_config(self):
        """
        Save configuration to file.
        """
        try:
            config = {
                "symbol": self.symbol_var.get(),
                "timeframe": self.timeframe_var.get(),
                "risk_level": self.risk_var.get(),
                "use_mock_data": self.mock_data_var.get()
            }
            
            with open("config.json", "w") as f:
                json.dump(config, f, indent=4)
            
            self.log("Configuration saved successfully.", "info")
        except Exception as e:
            self.log(f"Error saving configuration: {str(e)}", "error")
    
    def _open_backtest_window(self):
        """
        Open backtest window.
        """
        self.log("Opening backtest window...", "info")
        
        # Create backtest window
        backtest_window = tk.Toplevel(self.root)
        backtest_window.title("Backtest")
        backtest_window.geometry("800x600")
        backtest_window.minsize(800, 600)
        backtest_window.configure(bg=self.bg_color)
        
        # Create backtest frame
        backtest_frame = ttk.Frame(backtest_window, padding=10)
        backtest_frame.pack(fill=tk.BOTH, expand=True)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(backtest_frame, text="Backtest Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Symbol selection
        symbol_frame = ttk.Frame(settings_frame)
        symbol_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(symbol_frame, text="Symbol:").pack(side=tk.LEFT)
        
        symbol_var = tk.StringVar(value=self.symbol_var.get())
        symbol_combo = ttk.Combobox(symbol_frame, textvariable=symbol_var, values=self.symbols, width=10)
        symbol_combo.pack(side=tk.LEFT, padx=(5, 10))
        
        # Timeframe selection
        ttk.Label(symbol_frame, text="Timeframe:").pack(side=tk.LEFT)
        
        timeframe_var = tk.StringVar(value=self.timeframe_var.get())
        timeframe_combo = ttk.Combobox(symbol_frame, textvariable=timeframe_var, values=self.timeframes, width=10)
        timeframe_combo.pack(side=tk.LEFT, padx=(5, 10))
        
        # Date range
        date_frame = ttk.Frame(settings_frame)
        date_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(date_frame, text="Start Date:").pack(side=tk.LEFT)
        
        start_date_var = tk.StringVar(value=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"))
        start_date_entry = ttk.Entry(date_frame, textvariable=start_date_var, width=12)
        start_date_entry.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Label(date_frame, text="End Date:").pack(side=tk.LEFT)
        
        end_date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        end_date_entry = ttk.Entry(date_frame, textvariable=end_date_var, width=12)
        end_date_entry.pack(side=tk.LEFT, padx=(5, 10))
        
        # Strategy parameters
        params_frame = ttk.Frame(settings_frame)
        params_frame.pack(fill=tk.X, pady=5)
        
        # Risk level
        ttk.Label(params_frame, text="Risk Level:").pack(side=tk.LEFT)
        
        risk_var = tk.DoubleVar(value=self.risk_var.get())
        risk_spinbox = ttk.Spinbox(params_frame, from_=0.01, to=0.1, increment=0.01, textvariable=risk_var, width=5)
        risk_spinbox.pack(side=tk.LEFT, padx=(5, 10))
        
        # Run button
        run_button = ttk.Button(settings_frame, text="Run Backtest", command=lambda: self._run_backtest(
            symbol_var.get(),
            timeframe_var.get(),
            start_date_var.get(),
            end_date_var.get(),
            risk_var.get(),
            backtest_results_text
        ))
        run_button.pack(pady=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(backtest_frame, text="Backtest Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        backtest_results_text = scrolledtext.ScrolledText(results_frame, bg="#3B4252", fg=self.fg_color)
        backtest_results_text.pack(fill=tk.BOTH, expand=True)
    
    def _run_backtest(self, symbol, timeframe, start_date, end_date, risk_level, results_text):
        """
        Run backtest with specified parameters.
        
        Args:
            symbol: Symbol to backtest
            timeframe: Timeframe to backtest
            start_date: Start date for backtest
            end_date: End date for backtest
            risk_level: Risk level for backtest
            results_text: Text widget to display results
        """
        try:
            results_text.delete(1.0, tk.END)
            results_text.insert(tk.END, f"Running backtest for {symbol} on {timeframe} timeframe...\n")
            results_text.insert(tk.END, f"Date range: {start_date} to {end_date}\n")
            results_text.insert(tk.END, f"Risk level: {risk_level}\n\n")
            
            # Load data
            results_text.insert(tk.END, "Loading data...\n")
            
            # Convert dates to datetime
            start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Load data from enhanced data directory
            file_path = os.path.join("enhanced_data", timeframe, f"{symbol}.json")
            
            if not os.path.exists(file_path):
                results_text.insert(tk.END, f"Error: Enhanced data file not found: {file_path}\n")
                return
            
            # Load data from JSON file
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Convert timestamp to datetime if it exists as a column
            if "timestamp" in df.columns:
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
                # Filter by date range
                df = df[(df["datetime"] >= start_datetime) & (df["datetime"] <= end_datetime)]
                # Set datetime as index
                df = df.set_index("datetime")
            
            results_text.insert(tk.END, f"Loaded {len(df)} candles for {symbol} ({timeframe})\n\n")
            
            # Create strategy components
            signal_generator = RobustSignalGenerator(
                technical_indicators=self.technical_indicators,
                error_handler=self.error_handler
            )
            
            # Create strategy configuration
            config = {
                "risk_level": risk_level,
                "take_profit_multiplier": 3.0,
                "stop_loss_multiplier": 2.0,
                "use_volatility_filters": True,
                "use_trend_filters": True,
                "use_volume_filters": True,
                "use_regime_detection": True
            }
            
            # Create strategy instance
            strategy = MasterOmniOverlordRobustStrategy(
                symbol=symbol,
                timeframe=timeframe,
                signal_generator=signal_generator,
                error_handler=self.error_handler,
                config=config
            )
            
            # Set data on the strategy
            strategy.set_data(df)
            
            # Generate signals
            results_text.insert(tk.END, f"Generating signals for {len(df)} candles...\n")
            
            # Add signal column
            df["signal"] = 0
            
            # Process each candle
            for i in range(50, len(df)):  # Start after warmup period
                try:
                    # Get current candle data
                    candle_data = df.iloc[i].to_dict()
                    
                    # Generate signal
                    signal = strategy.generate_signal(candle_data)
                    
                    # Store signal
                    df.loc[df.index[i], "signal"] = signal
                    
                except Exception as e:
                    results_text.insert(tk.END, f"Error generating signal for candle {i}: {str(e)}\n")
            
            # Count signals
            buy_signals = (df["signal"] > 0).sum()
            sell_signals = (df["signal"] < 0).sum()
            neutral_signals = (df["signal"] == 0).sum()
            
            results_text.insert(tk.END, f"Signal counts: Buy={buy_signals}, Sell={sell_signals}, Neutral={neutral_signals}\n\n")
            
            # Simulate trades
            results_text.insert(tk.END, "Simulating trades...\n")
            
            position = 0
            entry_price = 0
            entry_time = None
            trades = []
            
            for i in range(len(df)):
                current_price = df["close"].iloc[i]
                current_time = df.index[i]
                current_signal = df["signal"].iloc[i]
                
                # Process signal
                if position == 0 and current_signal > 0:
                    # Enter long position
                    position = 1
                    entry_price = current_price
                    entry_time = current_time
                
                elif position == 0 and current_signal < 0:
                    # Enter short position
                    position = -1
                    entry_price = current_price
                    entry_time = current_time
                
                elif position > 0 and current_signal < 0:
                    # Exit long position
                    pnl = (current_price - entry_price) / entry_price
                    trades.append({
                        "entry_time": entry_time,
                        "exit_time": current_time,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "position": "long",
                        "pnl": pnl
                    })
                    position = 0
                
                elif position < 0 and current_signal > 0:
                    # Exit short position
                    pnl = (entry_price - current_price) / entry_price
                    trades.append({
                        "entry_time": entry_time,
                        "exit_time": current_time,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "position": "short",
                        "pnl": pnl
                    })
                    position = 0
            
            # Close any open position at the end
            if position != 0:
                current_price = df["close"].iloc[-1]
                current_time = df.index[-1]
                
                if position > 0:
                    # Exit long position
                    pnl = (current_price - entry_price) / entry_price
                    trades.append({
                        "entry_time": entry_time,
                        "exit_time": current_time,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "position": "long",
                        "pnl": pnl
                    })
                
                elif position < 0:
                    # Exit short position
                    pnl = (entry_price - current_price) / entry_price
                    trades.append({
                        "entry_time": entry_time,
                        "exit_time": current_time,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "position": "short",
                        "pnl": pnl
                    })
            
            # Calculate performance metrics
            results_text.insert(tk.END, f"Total trades: {len(trades)}\n")
            
            if trades:
                # Calculate win rate
                winning_trades = sum(1 for trade in trades if trade["pnl"] > 0)
                win_rate = winning_trades / len(trades)
                
                # Calculate profit factor
                gross_profit = sum(trade["pnl"] for trade in trades if trade["pnl"] > 0)
                gross_loss = abs(sum(trade["pnl"] for trade in trades if trade["pnl"] < 0))
                
                if gross_loss > 0:
                    profit_factor = gross_profit / gross_loss
                else:
                    profit_factor = float('inf') if gross_profit > 0 else 0.0
                
                # Calculate total return
                total_return = sum(trade["pnl"] for trade in trades)
                
                # Calculate equity curve
                equity = [1.0]
                for trade in trades:
                    equity.append(equity[-1] * (1 + trade["pnl"]))
                
                # Calculate max drawdown
                max_equity = equity[0]
                max_drawdown = 0.0
                
                for e in equity:
                    max_equity = max(max_equity, e)
                    drawdown = (max_equity - e) / max_equity
                    max_drawdown = max(max_drawdown, drawdown)
                
                # Calculate Sharpe ratio
                returns = [trade["pnl"] for trade in trades]
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                
                if std_return > 0:
                    sharpe_ratio = avg_return / std_return * np.sqrt(252)  # Annualized
                else:
                    sharpe_ratio = 0.0
                
                # Display metrics
                results_text.insert(tk.END, f"Win rate: {win_rate:.2%}\n")
                results_text.insert(tk.END, f"Profit factor: {profit_factor:.2f}\n")
                results_text.insert(tk.END, f"Total return: {total_return:.2%}\n")
                results_text.insert(tk.END, f"Max drawdown: {max_drawdown:.2%}\n")
                results_text.insert(tk.END, f"Sharpe ratio: {sharpe_ratio:.2f}\n\n")
                
                # Display trade details
                results_text.insert(tk.END, "Trade Details:\n")
                
                for i, trade in enumerate(trades):
                    results_text.insert(tk.END, f"Trade {i+1}: {trade['position']} from {trade['entry_time']} to {trade['exit_time']}, PnL: {trade['pnl']:.2%}\n")
            
            results_text.insert(tk.END, "\nBacktest completed successfully.\n")
        
        except Exception as e:
            results_text.insert(tk.END, f"Error running backtest: {str(e)}\n")
            import traceback
            results_text.insert(tk.END, traceback.format_exc())
    
    def _open_optimization_window(self):
        """
        Open optimization window.
        """
        self.log("Opening optimization window...", "info")
        
        # Create optimization window
        optimization_window = tk.Toplevel(self.root)
        optimization_window.title("Optimization")
        optimization_window.geometry("800x600")
        optimization_window.minsize(800, 600)
        optimization_window.configure(bg=self.bg_color)
        
        # Create optimization frame
        optimization_frame = ttk.Frame(optimization_window, padding=10)
        optimization_frame.pack(fill=tk.BOTH, expand=True)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(optimization_frame, text="Optimization Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Symbol selection
        symbol_frame = ttk.Frame(settings_frame)
        symbol_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(symbol_frame, text="Symbol:").pack(side=tk.LEFT)
        
        symbol_var = tk.StringVar(value=self.symbol_var.get())
        symbol_combo = ttk.Combobox(symbol_frame, textvariable=symbol_var, values=self.symbols, width=10)
        symbol_combo.pack(side=tk.LEFT, padx=(5, 10))
        
        # Timeframe selection
        ttk.Label(symbol_frame, text="Timeframe:").pack(side=tk.LEFT)
        
        timeframe_var = tk.StringVar(value=self.timeframe_var.get())
        timeframe_combo = ttk.Combobox(symbol_frame, textvariable=timeframe_var, values=self.timeframes, width=10)
        timeframe_combo.pack(side=tk.LEFT, padx=(5, 10))
        
        # Date range
        date_frame = ttk.Frame(settings_frame)
        date_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(date_frame, text="Start Date:").pack(side=tk.LEFT)
        
        start_date_var = tk.StringVar(value=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"))
        start_date_entry = ttk.Entry(date_frame, textvariable=start_date_var, width=12)
        start_date_entry.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Label(date_frame, text="End Date:").pack(side=tk.LEFT)
        
        end_date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        end_date_entry = ttk.Entry(date_frame, textvariable=end_date_var, width=12)
        end_date_entry.pack(side=tk.LEFT, padx=(5, 10))
        
        # Parameter ranges
        params_frame = ttk.LabelFrame(settings_frame, text="Parameter Ranges", padding=10)
        params_frame.pack(fill=tk.X, pady=5)
        
        # Risk level range
        risk_frame = ttk.Frame(params_frame)
        risk_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(risk_frame, text="Risk Level:").pack(side=tk.LEFT)
        
        risk_min_var = tk.DoubleVar(value=0.01)
        risk_min_spinbox = ttk.Spinbox(risk_frame, from_=0.01, to=0.1, increment=0.01, textvariable=risk_min_var, width=5)
        risk_min_spinbox.pack(side=tk.LEFT, padx=(5, 5))
        
        ttk.Label(risk_frame, text="to").pack(side=tk.LEFT)
        
        risk_max_var = tk.DoubleVar(value=0.05)
        risk_max_spinbox = ttk.Spinbox(risk_frame, from_=0.01, to=0.1, increment=0.01, textvariable=risk_max_var, width=5)
        risk_max_spinbox.pack(side=tk.LEFT, padx=(5, 5))
        
        ttk.Label(risk_frame, text="Step:").pack(side=tk.LEFT)
        
        risk_step_var = tk.DoubleVar(value=0.01)
        risk_step_spinbox = ttk.Spinbox(risk_frame, from_=0.01, to=0.1, increment=0.01, textvariable=risk_step_var, width=5)
        risk_step_spinbox.pack(side=tk.LEFT, padx=(5, 0))
        
        # Take profit multiplier range
        tp_frame = ttk.Frame(params_frame)
        tp_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(tp_frame, text="Take Profit:").pack(side=tk.LEFT)
        
        tp_min_var = tk.DoubleVar(value=2.0)
        tp_min_spinbox = ttk.Spinbox(tp_frame, from_=1.0, to=5.0, increment=0.5, textvariable=tp_min_var, width=5)
        tp_min_spinbox.pack(side=tk.LEFT, padx=(5, 5))
        
        ttk.Label(tp_frame, text="to").pack(side=tk.LEFT)
        
        tp_max_var = tk.DoubleVar(value=4.0)
        tp_max_spinbox = ttk.Spinbox(tp_frame, from_=1.0, to=5.0, increment=0.5, textvariable=tp_max_var, width=5)
        tp_max_spinbox.pack(side=tk.LEFT, padx=(5, 5))
        
        ttk.Label(tp_frame, text="Step:").pack(side=tk.LEFT)
        
        tp_step_var = tk.DoubleVar(value=0.5)
        tp_step_spinbox = ttk.Spinbox(tp_frame, from_=0.5, to=1.0, increment=0.5, textvariable=tp_step_var, width=5)
        tp_step_spinbox.pack(side=tk.LEFT, padx=(5, 0))
        
        # Stop loss multiplier range
        sl_frame = ttk.Frame(params_frame)
        sl_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(sl_frame, text="Stop Loss:").pack(side=tk.LEFT)
        
        sl_min_var = tk.DoubleVar(value=1.5)
        sl_min_spinbox = ttk.Spinbox(sl_frame, from_=1.0, to=3.0, increment=0.5, textvariable=sl_min_var, width=5)
        sl_min_spinbox.pack(side=tk.LEFT, padx=(5, 5))
        
        ttk.Label(sl_frame, text="to").pack(side=tk.LEFT)
        
        sl_max_var = tk.DoubleVar(value=2.5)
        sl_max_spinbox = ttk.Spinbox(sl_frame, from_=1.0, to=3.0, increment=0.5, textvariable=sl_max_var, width=5)
        sl_max_spinbox.pack(side=tk.LEFT, padx=(5, 5))
        
        ttk.Label(sl_frame, text="Step:").pack(side=tk.LEFT)
        
        sl_step_var = tk.DoubleVar(value=0.5)
        sl_step_spinbox = ttk.Spinbox(sl_frame, from_=0.5, to=1.0, increment=0.5, textvariable=sl_step_var, width=5)
        sl_step_spinbox.pack(side=tk.LEFT, padx=(5, 0))
        
        # Run button
        run_button = ttk.Button(settings_frame, text="Run Optimization", command=lambda: self._run_optimization(
            symbol_var.get(),
            timeframe_var.get(),
            start_date_var.get(),
            end_date_var.get(),
            risk_min_var.get(),
            risk_max_var.get(),
            risk_step_var.get(),
            tp_min_var.get(),
            tp_max_var.get(),
            tp_step_var.get(),
            sl_min_var.get(),
            sl_max_var.get(),
            sl_step_var.get(),
            optimization_results_text
        ))
        run_button.pack(pady=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(optimization_frame, text="Optimization Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        optimization_results_text = scrolledtext.ScrolledText(results_frame, bg="#3B4252", fg=self.fg_color)
        optimization_results_text.pack(fill=tk.BOTH, expand=True)
    
    def _run_optimization(self, symbol, timeframe, start_date, end_date, 
                         risk_min, risk_max, risk_step,
                         tp_min, tp_max, tp_step,
                         sl_min, sl_max, sl_step,
                         results_text):
        """
        Run optimization with specified parameters.
        
        Args:
            symbol: Symbol to optimize
            timeframe: Timeframe to optimize
            start_date: Start date for optimization
            end_date: End date for optimization
            risk_min: Minimum risk level
            risk_max: Maximum risk level
            risk_step: Risk level step
            tp_min: Minimum take profit multiplier
            tp_max: Maximum take profit multiplier
            tp_step: Take profit multiplier step
            sl_min: Minimum stop loss multiplier
            sl_max: Maximum stop loss multiplier
            sl_step: Stop loss multiplier step
            results_text: Text widget to display results
        """
        try:
            results_text.delete(1.0, tk.END)
            results_text.insert(tk.END, f"Running optimization for {symbol} on {timeframe} timeframe...\n")
            results_text.insert(tk.END, f"Date range: {start_date} to {end_date}\n")
            results_text.insert(tk.END, f"Risk level range: {risk_min} to {risk_max} (step: {risk_step})\n")
            results_text.insert(tk.END, f"Take profit range: {tp_min} to {tp_max} (step: {tp_step})\n")
            results_text.insert(tk.END, f"Stop loss range: {sl_min} to {sl_max} (step: {sl_step})\n\n")
            
            # Load data
            results_text.insert(tk.END, "Loading data...\n")
            
            # Convert dates to datetime
            start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Load data from enhanced data directory
            file_path = os.path.join("enhanced_data", timeframe, f"{symbol}.json")
            
            if not os.path.exists(file_path):
                results_text.insert(tk.END, f"Error: Enhanced data file not found: {file_path}\n")
                return
            
            # Load data from JSON file
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Convert timestamp to datetime if it exists as a column
            if "timestamp" in df.columns:
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
                # Filter by date range
                df = df[(df["datetime"] >= start_datetime) & (df["datetime"] <= end_datetime)]
                # Set datetime as index
                df = df.set_index("datetime")
            
            results_text.insert(tk.END, f"Loaded {len(df)} candles for {symbol} ({timeframe})\n\n")
            
            # Generate parameter combinations
            risk_values = np.arange(risk_min, risk_max + risk_step / 2, risk_step)
            tp_values = np.arange(tp_min, tp_max + tp_step / 2, tp_step)
            sl_values = np.arange(sl_min, sl_max + sl_step / 2, sl_step)
            
            param_combinations = []
            for risk in risk_values:
                for tp in tp_values:
                    for sl in sl_values:
                        param_combinations.append({
                            "risk_level": risk,
                            "take_profit_multiplier": tp,
                            "stop_loss_multiplier": sl,
                            "use_volatility_filters": True,
                            "use_trend_filters": True,
                            "use_volume_filters": True,
                            "use_regime_detection": True
                        })
            
            results_text.insert(tk.END, f"Generated {len(param_combinations)} parameter combinations\n\n")
            
            # Initialize results
            optimization_results = []
            
            # Create strategy components
            signal_generator = RobustSignalGenerator(
                technical_indicators=self.technical_indicators,
                error_handler=self.error_handler
            )
            
            # Test each parameter combination
            for i, params in enumerate(param_combinations):
                results_text.insert(tk.END, f"Testing parameter combination {i+1}/{len(param_combinations)}: {params}\n")
                results_text.see(tk.END)
                
                # Create strategy instance
                strategy = MasterOmniOverlordRobustStrategy(
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_generator=signal_generator,
                    error_handler=self.error_handler,
                    config=params
                )
                
                # Set data on the strategy
                strategy.set_data(df)
                
                # Add signal column
                df_copy = df.copy()
                df_copy["signal"] = 0
                
                # Process each candle
                for j in range(50, len(df_copy)):  # Start after warmup period
                    try:
                        # Get current candle data
                        candle_data = df_copy.iloc[j].to_dict()
                        
                        # Generate signal
                        signal = strategy.generate_signal(candle_data)
                        
                        # Store signal
                        df_copy.loc[df_copy.index[j], "signal"] = signal
                        
                    except Exception as e:
                        results_text.insert(tk.END, f"Error generating signal for candle {j}: {str(e)}\n")
                
                # Simulate trades
                position = 0
                entry_price = 0
                entry_time = None
                trades = []
                
                for j in range(len(df_copy)):
                    current_price = df_copy["close"].iloc[j]
                    current_time = df_copy.index[j]
                    current_signal = df_copy["signal"].iloc[j]
                    
                    # Process signal
                    if position == 0 and current_signal > 0:
                        # Enter long position
                        position = 1
                        entry_price = current_price
                        entry_time = current_time
                    
                    elif position == 0 and current_signal < 0:
                        # Enter short position
                        position = -1
                        entry_price = current_price
                        entry_time = current_time
                    
                    elif position > 0 and current_signal < 0:
                        # Exit long position
                        pnl = (current_price - entry_price) / entry_price
                        trades.append({
                            "entry_time": entry_time,
                            "exit_time": current_time,
                            "entry_price": entry_price,
                            "exit_price": current_price,
                            "position": "long",
                            "pnl": pnl
                        })
                        position = 0
                    
                    elif position < 0 and current_signal > 0:
                        # Exit short position
                        pnl = (entry_price - current_price) / entry_price
                        trades.append({
                            "entry_time": entry_time,
                            "exit_time": current_time,
                            "entry_price": entry_price,
                            "exit_price": current_price,
                            "position": "short",
                            "pnl": pnl
                        })
                        position = 0
                
                # Close any open position at the end
                if position != 0:
                    current_price = df_copy["close"].iloc[-1]
                    current_time = df_copy.index[-1]
                    
                    if position > 0:
                        # Exit long position
                        pnl = (current_price - entry_price) / entry_price
                        trades.append({
                            "entry_time": entry_time,
                            "exit_time": current_time,
                            "entry_price": entry_price,
                            "exit_price": current_price,
                            "position": "long",
                            "pnl": pnl
                        })
                    
                    elif position < 0:
                        # Exit short position
                        pnl = (entry_price - current_price) / entry_price
                        trades.append({
                            "entry_time": entry_time,
                            "exit_time": current_time,
                            "entry_price": entry_price,
                            "exit_price": current_price,
                            "position": "short",
                            "pnl": pnl
                        })
                
                # Calculate performance metrics
                metrics = {
                    "params": params,
                    "total_trades": len(trades),
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "total_return": 0.0,
                    "max_drawdown": 0.0,
                    "sharpe_ratio": 0.0
                }
                
                if trades:
                    # Calculate win rate
                    winning_trades = sum(1 for trade in trades if trade["pnl"] > 0)
                    metrics["win_rate"] = winning_trades / len(trades)
                    
                    # Calculate profit factor
                    gross_profit = sum(trade["pnl"] for trade in trades if trade["pnl"] > 0)
                    gross_loss = abs(sum(trade["pnl"] for trade in trades if trade["pnl"] < 0))
                    
                    if gross_loss > 0:
                        metrics["profit_factor"] = gross_profit / gross_loss
                    else:
                        metrics["profit_factor"] = float('inf') if gross_profit > 0 else 0.0
                    
                    # Calculate total return
                    metrics["total_return"] = sum(trade["pnl"] for trade in trades)
                    
                    # Calculate equity curve
                    equity = [1.0]
                    for trade in trades:
                        equity.append(equity[-1] * (1 + trade["pnl"]))
                    
                    # Calculate max drawdown
                    max_equity = equity[0]
                    max_drawdown = 0.0
                    
                    for e in equity:
                        max_equity = max(max_equity, e)
                        drawdown = (max_equity - e) / max_equity
                        max_drawdown = max(max_drawdown, drawdown)
                    
                    metrics["max_drawdown"] = max_drawdown
                    
                    # Calculate Sharpe ratio
                    returns = [trade["pnl"] for trade in trades]
                    avg_return = np.mean(returns)
                    std_return = np.std(returns)
                    
                    if std_return > 0:
                        metrics["sharpe_ratio"] = avg_return / std_return * np.sqrt(252)  # Annualized
                    else:
                        metrics["sharpe_ratio"] = 0.0
                
                # Store metrics
                optimization_results.append(metrics)
                
                # Display metrics
                results_text.insert(tk.END, f"  Total trades: {metrics['total_trades']}\n")
                results_text.insert(tk.END, f"  Win rate: {metrics['win_rate']:.2%}\n")
                results_text.insert(tk.END, f"  Profit factor: {metrics['profit_factor']:.2f}\n")
                results_text.insert(tk.END, f"  Total return: {metrics['total_return']:.2%}\n")
                results_text.insert(tk.END, f"  Max drawdown: {metrics['max_drawdown']:.2%}\n")
                results_text.insert(tk.END, f"  Sharpe ratio: {metrics['sharpe_ratio']:.2f}\n\n")
            
            # Sort results by Sharpe ratio
            optimization_results.sort(key=lambda x: x["sharpe_ratio"], reverse=True)
            
            # Display best results
            results_text.insert(tk.END, "Best Parameter Combinations:\n\n")
            
            for i, result in enumerate(optimization_results[:5]):
                results_text.insert(tk.END, f"Rank {i+1}:\n")
                results_text.insert(tk.END, f"  Parameters: {result['params']}\n")
                results_text.insert(tk.END, f"  Total trades: {result['total_trades']}\n")
                results_text.insert(tk.END, f"  Win rate: {result['win_rate']:.2%}\n")
                results_text.insert(tk.END, f"  Profit factor: {result['profit_factor']:.2f}\n")
                results_text.insert(tk.END, f"  Total return: {result['total_return']:.2%}\n")
                results_text.insert(tk.END, f"  Max drawdown: {result['max_drawdown']:.2%}\n")
                results_text.insert(tk.END, f"  Sharpe ratio: {result['sharpe_ratio']:.2f}\n\n")
            
            results_text.insert(tk.END, "Optimization completed successfully.\n")
            
            # Save results to file
            results_dir = "backtest_results/optimization"
            os.makedirs(results_dir, exist_ok=True)
            
            results_path = os.path.join(results_dir, f"{symbol}_{timeframe}_optimization.json")
            
            with open(results_path, "w") as f:
                json.dump(optimization_results, f, indent=4, default=str)
            
            results_text.insert(tk.END, f"Results saved to {results_path}\n")
        
        except Exception as e:
            results_text.insert(tk.END, f"Error running optimization: {str(e)}\n")
            import traceback
            results_text.insert(tk.END, traceback.format_exc())
    
    def _open_data_viewer(self):
        """
        Open data viewer window.
        """
        self.log("Opening data viewer...", "info")
        
        # Create data viewer window
        data_viewer = tk.Toplevel(self.root)
        data_viewer.title("Data Viewer")
        data_viewer.geometry("800x600")
        data_viewer.minsize(800, 600)
        data_viewer.configure(bg=self.bg_color)
        
        # Create data viewer frame
        data_frame = ttk.Frame(data_viewer, padding=10)
        data_frame.pack(fill=tk.BOTH, expand=True)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(data_frame, text="Data Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Symbol selection
        symbol_frame = ttk.Frame(settings_frame)
        symbol_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(symbol_frame, text="Symbol:").pack(side=tk.LEFT)
        
        symbol_var = tk.StringVar(value=self.symbol_var.get())
        symbol_combo = ttk.Combobox(symbol_frame, textvariable=symbol_var, values=self.symbols, width=10)
        symbol_combo.pack(side=tk.LEFT, padx=(5, 10))
        
        # Timeframe selection
        ttk.Label(symbol_frame, text="Timeframe:").pack(side=tk.LEFT)
        
        timeframe_var = tk.StringVar(value=self.timeframe_var.get())
        timeframe_combo = ttk.Combobox(symbol_frame, textvariable=timeframe_var, values=self.timeframes, width=10)
        timeframe_combo.pack(side=tk.LEFT, padx=(5, 10))
        
        # Load button
        load_button = ttk.Button(settings_frame, text="Load Data", command=lambda: self._load_data_viewer(
            symbol_var.get(),
            timeframe_var.get(),
            data_text
        ))
        load_button.pack(pady=10)
        
        # Data frame
        data_view_frame = ttk.LabelFrame(data_frame, text="Data View", padding=10)
        data_view_frame.pack(fill=tk.BOTH, expand=True)
        
        data_text = scrolledtext.ScrolledText(data_view_frame, bg="#3B4252", fg=self.fg_color)
        data_text.pack(fill=tk.BOTH, expand=True)
    
    def _load_data_viewer(self, symbol, timeframe, data_text):
        """
        Load data for data viewer.
        
        Args:
            symbol: Symbol to load data for
            timeframe: Timeframe to load data for
            data_text: Text widget to display data
        """
        try:
            data_text.delete(1.0, tk.END)
            data_text.insert(tk.END, f"Loading data for {symbol} on {timeframe} timeframe...\n\n")
            
            # Load data from enhanced data directory
            file_path = os.path.join("enhanced_data", timeframe, f"{symbol}.json")
            
            if not os.path.exists(file_path):
                data_text.insert(tk.END, f"Error: Enhanced data file not found: {file_path}\n")
                return
            
            # Load data from JSON file
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Convert timestamp to datetime if it exists as a column
            if "timestamp" in df.columns:
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
            
            # Display data
            data_text.insert(tk.END, f"Loaded {len(df)} candles for {symbol} ({timeframe})\n\n")
            
            # Display columns
            data_text.insert(tk.END, f"Columns: {', '.join(df.columns)}\n\n")
            
            # Display first 10 rows
            data_text.insert(tk.END, "First 10 rows:\n\n")
            data_text.insert(tk.END, df.head(10).to_string())
            
            # Display statistics
            data_text.insert(tk.END, "\n\nStatistics:\n\n")
            data_text.insert(tk.END, df.describe().to_string())
        
        except Exception as e:
            data_text.insert(tk.END, f"Error loading data: {str(e)}\n")
            import traceback
            data_text.insert(tk.END, traceback.format_exc())
    
    def _show_documentation(self):
        """
        Show documentation.
        """
        self.log("Showing documentation...", "info")
        
        # Create documentation window
        doc_window = tk.Toplevel(self.root)
        doc_window.title("Documentation")
        doc_window.geometry("800x600")
        doc_window.minsize(800, 600)
        doc_window.configure(bg=self.bg_color)
        
        # Create documentation frame
        doc_frame = ttk.Frame(doc_window, padding=10)
        doc_frame.pack(fill=tk.BOTH, expand=True)
        
        # Documentation text
        doc_text = scrolledtext.ScrolledText(doc_frame, bg="#3B4252", fg=self.fg_color)
        doc_text.pack(fill=tk.BOTH, expand=True)
        
        # Add documentation content
        doc_text.insert(tk.END, "Hyperliquid Trading Bot Documentation\n")
        doc_text.insert(tk.END, "================================\n\n")
        
        doc_text.insert(tk.END, "Overview\n")
        doc_text.insert(tk.END, "--------\n\n")
        doc_text.insert(tk.END, "The Hyperliquid Trading Bot is a powerful tool for automated trading on the Hyperliquid exchange. It features advanced technical analysis, robust error handling, and a user-friendly interface.\n\n")
        
        doc_text.insert(tk.END, "Features\n")
        doc_text.insert(tk.END, "--------\n\n")
        doc_text.insert(tk.END, "- Real-time market data monitoring\n")
        doc_text.insert(tk.END, "- Advanced technical analysis with multiple indicators\n")
        doc_text.insert(tk.END, "- Robust error handling and recovery\n")
        doc_text.insert(tk.END, "- Mock data mode for testing and development\n")
        doc_text.insert(tk.END, "- API rate limiting with cooldown periods\n")
        doc_text.insert(tk.END, "- Backtesting and optimization tools\n")
        doc_text.insert(tk.END, "- Customizable trading parameters\n\n")
        
        doc_text.insert(tk.END, "Getting Started\n")
        doc_text.insert(tk.END, "---------------\n\n")
        doc_text.insert(tk.END, "1. Select a symbol and timeframe from the dropdown menus\n")
        doc_text.insert(tk.END, "2. Adjust risk level and other parameters as needed\n")
        doc_text.insert(tk.END, "3. Click 'Start' to begin trading\n")
        doc_text.insert(tk.END, "4. Monitor signals and positions in the logs\n")
        doc_text.insert(tk.END, "5. Use the tools menu for backtesting and optimization\n\n")
        
        doc_text.insert(tk.END, "API Rate Limiting\n")
        doc_text.insert(tk.END, "----------------\n\n")
        doc_text.insert(tk.END, "The bot includes an API rate limiter to prevent exceeding Hyperliquid's rate limits. When rate limited, the bot will automatically switch to mock data mode if enabled. The rate limiter status is displayed in the control panel.\n\n")
        
        doc_text.insert(tk.END, "Mock Data Mode\n")
        doc_text.insert(tk.END, "-------------\n\n")
        doc_text.insert(tk.END, "Mock data mode allows the bot to operate without making actual API calls. This is useful for testing strategies or when API rate limits are reached. Enable mock data mode by checking the 'Use Mock Data' checkbox.\n\n")
        
        doc_text.insert(tk.END, "Backtesting\n")
        doc_text.insert(tk.END, "----------\n\n")
        doc_text.insert(tk.END, "The backtesting tool allows you to test your trading strategy on historical data. Access it from the Tools menu. Select a symbol, timeframe, date range, and parameters, then click 'Run Backtest' to see the results.\n\n")
        
        doc_text.insert(tk.END, "Optimization\n")
        doc_text.insert(tk.END, "-----------\n\n")
        doc_text.insert(tk.END, "The optimization tool helps you find the best parameters for your trading strategy. Access it from the Tools menu. Select a symbol, timeframe, date range, and parameter ranges, then click 'Run Optimization' to find the optimal parameters.\n\n")
    
    def _show_about(self):
        """
        Show about dialog.
        """
        messagebox.showinfo(
            "About",
            "Hyperliquid Trading Bot\n\n"
            "Version: 1.0.0\n\n"
            "A powerful trading bot for the Hyperliquid exchange with advanced technical analysis, "
            "robust error handling, and a user-friendly interface."
        )
    
    def _start_bot(self):
        """
        Start the trading bot.
        """
        if self.running:
            return
        
        try:
            # Update status
            self.update_status("Starting...")
            
            # Get parameters
            symbol = self.symbol_var.get()
            timeframe = self.timeframe_var.get()
            risk_level = self.risk_var.get()
            use_mock_data = self.mock_data_var.get()
            
            # Log start
            self.log(f"Starting bot for {symbol} on {timeframe} timeframe with risk level {risk_level}", "info")
            
            # Check if using mock data
            if use_mock_data:
                self.log("Using mock data mode", "info")
                self.using_mock_data = True
            
            # Create strategy components
            signal_generator = RobustSignalGenerator(
                technical_indicators=self.technical_indicators,
                error_handler=self.error_handler
            )
            
            # Create strategy configuration
            config = {
                "risk_level": risk_level,
                "take_profit_multiplier": 3.0,
                "stop_loss_multiplier": 2.0,
                "use_volatility_filters": True,
                "use_trend_filters": True,
                "use_volume_filters": True,
                "use_regime_detection": True
            }
            
            # Create strategy instance
            strategy = MasterOmniOverlordRobustStrategy(
                symbol=symbol,
                timeframe=timeframe,
                signal_generator=signal_generator,
                error_handler=self.error_handler,
                config=config
            )
            
            # Store strategy
            self.strategies[symbol] = strategy
            
            # Update UI
            self.start_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)
            
            # Set running flag
            self.running = True
            
            # Start update loop
            self._update_data()
            
            # Update status
            self.update_status("Running")
        
        except Exception as e:
            self.log(f"Error starting bot: {str(e)}", "error")
            self.update_status("Error")
    
    def _stop_bot(self):
        """
        Stop the trading bot.
        """
        if not self.running:
            return
        
        try:
            # Update status
            self.update_status("Stopping...")
            
            # Log stop
            self.log("Stopping bot", "info")
            
            # Update UI
            self.start_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
            
            # Set running flag
            self.running = False
            
            # Update status
            self.update_status("Stopped")
        
        except Exception as e:
            self.log(f"Error stopping bot: {str(e)}", "error")
            self.update_status("Error")
    
    def _refresh_data(self):
        """
        Refresh market data.
        """
        try:
            # Update status
            self.update_status("Refreshing data...")
            
            # Get parameters
            symbol = self.symbol_var.get()
            timeframe = self.timeframe_var.get()
            
            # Log refresh
            self.log(f"Refreshing data for {symbol} on {timeframe} timeframe", "info")
            
            # Fetch data
            self._fetch_data(symbol, timeframe)
            
            # Update charts
            self._update_charts()
            
            # Update status
            self.update_status("Data refreshed")
        
        except Exception as e:
            self.log(f"Error refreshing data: {str(e)}", "error")
            self.update_status("Error")
    
    def _update_data(self):
        """
        Update market data periodically.
        """
        if not self.running:
            return
        
        try:
            # Get parameters
            symbol = self.symbol_var.get()
            timeframe = self.timeframe_var.get()
            
            # Fetch data
            self._fetch_data(symbol, timeframe)
            
            # Generate signals
            self._generate_signals(symbol, timeframe)
            
            # Update charts
            self._update_charts()
            
            # Schedule next update
            update_interval = self._get_update_interval(timeframe)
            self.root.after(update_interval, self._update_data)
        
        except Exception as e:
            self.log(f"Error updating data: {str(e)}", "error")
            self.update_status("Error")
            
            # Schedule retry
            self.root.after(10000, self._update_data)
    
    def _fetch_data(self, symbol, timeframe):
        """
        Fetch market data for a symbol and timeframe.
        
        Args:
            symbol: Symbol to fetch data for
            timeframe: Timeframe to fetch data for
        """
        try:
            # Check if using mock data
            if self.using_mock_data:
                # Get mock data
                data = self.mock_data_provider.get_klines(symbol, timeframe)
            else:
                # Check if rate limited
                if self.api_rate_limiter.is_limited():
                    self.log("API rate limited. Using mock data.", "warning")
                    self.using_mock_data = True
                    data = self.mock_data_provider.get_klines(symbol, timeframe)
                else:
                    # Load data from enhanced data directory
                    file_path = os.path.join("enhanced_data", timeframe, f"{symbol}.json")
                    
                    if not os.path.exists(file_path):
                        self.log(f"Enhanced data file not found: {file_path}", "error")
                        return
                    
                    # Load data from JSON file
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    
                    # Record API call
                    self.api_rate_limiter.record_call("klines")
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Convert timestamp to datetime if it exists as a column
            if "timestamp" in df.columns:
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
                # Set datetime as index
                df = df.set_index("datetime")
            
            # Store data
            self.market_data[symbol] = df
            
            # Set data on strategy
            if symbol in self.strategies:
                self.strategies[symbol].set_data(df)
            
            self.log(f"Fetched {len(df)} candles for {symbol} ({timeframe})", "info")
        
        except Exception as e:
            self.log(f"Error fetching data: {str(e)}", "error")
            raise
    
    def _generate_signals(self, symbol, timeframe):
        """
        Generate trading signals for a symbol and timeframe.
        
        Args:
            symbol: Symbol to generate signals for
            timeframe: Timeframe to generate signals for
        """
        try:
            # Check if strategy exists
            if symbol not in self.strategies:
                self.log(f"No strategy found for {symbol}", "error")
                return
            
            # Get strategy
            strategy = self.strategies[symbol]
            
            # Get market data
            if symbol not in self.market_data:
                self.log(f"No market data found for {symbol}", "error")
                return
            
            df = self.market_data[symbol]
            
            # Generate signal
            candle_data = df.iloc[-1].to_dict()
            signal = strategy.generate_signal(candle_data)
            
            # Store signal
            if symbol not in self.signals:
                self.signals[symbol] = []
            
            self.signals[symbol].append({
                "timestamp": datetime.now(),
                "signal": signal
            })
            
            # Trim signals
            if len(self.signals[symbol]) > 100:
                self.signals[symbol] = self.signals[symbol][-100:]
            
            # Log signal
            signal_text = "BUY" if signal > 0 else "SELL" if signal < 0 else "NEUTRAL"
            self.log_signal(f"Signal for {symbol}: {signal_text} ({signal})")
            
            # Process signal
            self._process_signal(symbol, signal)
        
        except Exception as e:
            self.log(f"Error generating signals: {str(e)}", "error")
            raise
    
    def _process_signal(self, symbol, signal):
        """
        Process a trading signal.
        
        Args:
            symbol: Symbol to process signal for
            signal: Signal to process
        """
        try:
            # Check if position exists
            if symbol not in self.positions:
                self.positions[symbol] = {
                    "position": 0,
                    "entry_price": 0,
                    "entry_time": None
                }
            
            # Get position
            position = self.positions[symbol]
            
            # Get market data
            if symbol not in self.market_data:
                self.log(f"No market data found for {symbol}", "error")
                return
            
            df = self.market_data[symbol]
            current_price = df["close"].iloc[-1]
            
            # Process signal
            if position["position"] == 0 and signal > 0:
                # Enter long position
                position["position"] = 1
                position["entry_price"] = current_price
                position["entry_time"] = datetime.now()
                
                self.log_position(f"Entered LONG position for {symbol} at {current_price}")
            
            elif position["position"] == 0 and signal < 0:
                # Enter short position
                position["position"] = -1
                position["entry_price"] = current_price
                position["entry_time"] = datetime.now()
                
                self.log_position(f"Entered SHORT position for {symbol} at {current_price}")
            
            elif position["position"] > 0 and signal < 0:
                # Exit long position
                pnl = (current_price - position["entry_price"]) / position["entry_price"]
                
                self.log_position(f"Exited LONG position for {symbol} at {current_price} (PnL: {pnl:.2%})")
                
                position["position"] = 0
                position["entry_price"] = 0
                position["entry_time"] = None
            
            elif position["position"] < 0 and signal > 0:
                # Exit short position
                pnl = (position["entry_price"] - current_price) / position["entry_price"]
                
                self.log_position(f"Exited SHORT position for {symbol} at {current_price} (PnL: {pnl:.2%})")
                
                position["position"] = 0
                position["entry_price"] = 0
                position["entry_time"] = None
        
        except Exception as e:
            self.log(f"Error processing signal: {str(e)}", "error")
            raise
    
    def _update_charts(self):
        """
        Update charts with latest data.
        """
        try:
            # Get parameters
            symbol = self.symbol_var.get()
            
            # Check if market data exists
            if symbol not in self.market_data:
                return
            
            # Get market data
            df = self.market_data[symbol]
            
            # Clear charts
            self.price_ax.clear()
            self.indicator_ax.clear()
            
            # Set background color
            self.price_ax.set_facecolor(self.bg_color)
            self.indicator_ax.set_facecolor(self.bg_color)
            
            # Set text color
            self.price_ax.tick_params(colors=self.fg_color)
            self.indicator_ax.tick_params(colors=self.fg_color)
            
            # Set spine color
            for spine in self.price_ax.spines.values():
                spine.set_color(self.fg_color)
            
            for spine in self.indicator_ax.spines.values():
                spine.set_color(self.fg_color)
            
            # Plot price
            self.price_ax.plot(df.index[-100:], df["close"].iloc[-100:], label="Close Price", color=self.accent_color)
            
            # Plot Bollinger Bands if available
            if "bb_upper" in df.columns and "bb_middle" in df.columns and "bb_lower" in df.columns:
                self.price_ax.plot(df.index[-100:], df["bb_upper"].iloc[-100:], "r--", alpha=0.5, label="BB Upper")
                self.price_ax.plot(df.index[-100:], df["bb_middle"].iloc[-100:], "g-", alpha=0.5, label="BB Middle")
                self.price_ax.plot(df.index[-100:], df["bb_lower"].iloc[-100:], "r--", alpha=0.5, label="BB Lower")
            
            # Plot signals
            if symbol in self.signals:
                for signal_data in self.signals[symbol][-20:]:
                    signal = signal_data["signal"]
                    timestamp = signal_data["timestamp"]
                    
                    # Find closest index
                    closest_idx = df.index[df.index <= pd.Timestamp(timestamp)].max()
                    
                    if closest_idx is not None:
                        price = df.loc[closest_idx, "close"]
                        
                        if signal > 0:
                            self.price_ax.scatter(closest_idx, price, color="green", marker="^", s=100, label="Buy Signal")
                        elif signal < 0:
                            self.price_ax.scatter(closest_idx, price, color="red", marker="v", s=100, label="Sell Signal")
            
            # Plot RSI if available
            if "rsi" in df.columns:
                self.indicator_ax.plot(df.index[-100:], df["rsi"].iloc[-100:], label="RSI", color=self.accent_color)
                self.indicator_ax.axhline(y=70, color="r", linestyle="--", alpha=0.5)
                self.indicator_ax.axhline(y=30, color="g", linestyle="--", alpha=0.5)
                self.indicator_ax.set_ylim(0, 100)
            
            # Set labels
            self.price_ax.set_title(f"{symbol} Price Chart", color=self.fg_color)
            self.price_ax.set_ylabel("Price", color=self.fg_color)
            self.indicator_ax.set_ylabel("RSI", color=self.fg_color)
            self.indicator_ax.set_xlabel("Date", color=self.fg_color)
            
            # Add legend
            self.price_ax.legend(loc="upper left", facecolor=self.bg_color, edgecolor=self.fg_color, labelcolor=self.fg_color)
            self.indicator_ax.legend(loc="upper left", facecolor=self.bg_color, edgecolor=self.fg_color, labelcolor=self.fg_color)
            
            # Format x-axis
            self.price_ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m-%d %H:%M"))
            self.price_ax.tick_params(axis="x", rotation=45)
            
            # Adjust layout
            self.fig.tight_layout()
            
            # Draw canvas
            self.canvas.draw()
        
        except Exception as e:
            self.log(f"Error updating charts: {str(e)}", "error")
    
    def _get_update_interval(self, timeframe):
        """
        Get update interval for a timeframe.
        
        Args:
            timeframe: Timeframe to get update interval for
            
        Returns:
            Update interval in milliseconds
        """
        # Default interval
        interval = 60000  # 1 minute
        
        # Adjust interval based on timeframe
        if timeframe == "1m":
            interval = 10000  # 10 seconds
        elif timeframe == "5m":
            interval = 30000  # 30 seconds
        elif timeframe == "15m":
            interval = 60000  # 1 minute
        elif timeframe == "1h":
            interval = 300000  # 5 minutes
        elif timeframe == "4h":
            interval = 600000  # 10 minutes
        elif timeframe == "1d":
            interval = 1800000  # 30 minutes
        
        return interval
    
    def update_status(self, status):
        """
        Update status label.
        
        Args:
            status: Status text
        """
        self.status_var.set(status)
        
        # Update color based on status
        if status == "Running":
            self.status_label.configure(foreground=self.success_color)
        elif status == "Stopped":
            self.status_label.configure(foreground=self.warning_color)
        elif status == "Error":
            self.status_label.configure(foreground=self.error_color)
        else:
            self.status_label.configure(foreground=self.fg_color)
    
    def log(self, message, level="info"):
        """
        Log a message.
        
        Args:
            message: Message to log
            level: Log level
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format message
        formatted_message = f"{timestamp} [{level.upper()}] {message}\n"
        
        # Set color based on level
        if level == "error":
            tag = "error"
            color = self.error_color
        elif level == "warning":
            tag = "warning"
            color = self.warning_color
        elif level == "success":
            tag = "success"
            color = self.success_color
        else:
            tag = "info"
            color = self.fg_color
        
        # Add message to log
        self.log_text.insert(tk.END, formatted_message, tag)
        self.log_text.tag_config(tag, foreground=color)
        
        # Scroll to end
        self.log_text.see(tk.END)
        
        # Log to console
        print(formatted_message, end="")
    
    def log_signal(self, message):
        """
        Log a signal message.
        
        Args:
            message: Message to log
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format message
        formatted_message = f"{timestamp} [SIGNAL] {message}\n"
        
        # Add message to signals log
        self.signals_text.insert(tk.END, formatted_message)
        
        # Scroll to end
        self.signals_text.see(tk.END)
    
    def log_position(self, message):
        """
        Log a position message.
        
        Args:
            message: Message to log
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format message
        formatted_message = f"{timestamp} [POSITION] {message}\n"
        
        # Add message to positions log
        self.positions_text.insert(tk.END, formatted_message)
        
        # Scroll to end
        self.positions_text.see(tk.END)

def main():
    """
    Main function.
    """
    # Create root window
    root = tk.Tk()
    
    # Create GUI
    app = HyperliquidTradingBotGUI(root)
    
    # Run main loop
    root.mainloop()

if __name__ == "__main__":
    main()
