"""
Modified UI module with event-driven architecture for the hyperliquidmaster package.

This module provides the GUI components for the trading bot with event-driven communication.
"""

import os
import sys
import time
import json
import queue
import logging
import threading
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from hyperliquidmaster.config import BotSettings
from hyperliquidmaster.core.hyperliquid_adapter import HyperliquidAdapter
from hyperliquidmaster.core.error_handler import ErrorHandler
from hyperliquidmaster.risk import RiskManager
from hyperliquidmaster.events import event_bus

# Configure logging
logger = logging.getLogger(__name__)

# Define event types
EVENT_MARKET_DATA = "market_data"
EVENT_ORDER_BOOK = "order_book"
EVENT_ACCOUNT_INFO = "account_info"
EVENT_TRADE_SIGNAL = "trade_signal"
EVENT_ORDER_UPDATE = "order_update"
EVENT_ERROR = "error"
EVENT_STATUS = "status"

###############################################################################
# Logging Setup
###############################################################################
class QueueLoggingHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue
    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception:
            self.handleError(record)

###############################################################################
# Enhanced Trading Bot with GUI
###############################################################################
class EnhancedTradingBotGUI:
    def __init__(self, settings: BotSettings):
        """
        Initialize the GUI components.
        
        Args:
            settings: Bot configuration settings
        """
        # Store settings
        self.settings = settings
        
        # Initialize GUI components
        self.root = tk.Tk()
        self.root.title("Enhanced Hyperliquid Trading Bot")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Setup logging with queue for GUI display
        self.log_queue = queue.Queue()
        self.logger = self._setup_logger()
        
        # Initialize error handler
        self.error_handler = ErrorHandler(self.logger)
        
        # Runtime variables
        self.running = False
        self.thread = None
        self.market_data = {}
        self.positions = {}
        
        # Create GUI components
        self._create_gui()
        
        # Start log consumer
        self.log_consumer_running = True
        self.log_consumer_thread = threading.Thread(target=self._consume_logs, daemon=True)
        self.log_consumer_thread.start()
        
        # Subscribe to events
        self._subscribe_to_events()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up the logger with queue handler for GUI display."""
        logger = logging.getLogger("EnhancedTradingBot")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("enhanced_bot_gui.log", mode="a")
        file_handler.setLevel(logging.INFO)
        
        # Create queue handler for GUI
        queue_handler = QueueLoggingHandler(self.log_queue)
        queue_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        queue_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.addHandler(queue_handler)
        
        return logger
    
    def _subscribe_to_events(self):
        """Subscribe to events from the event bus."""
        event_bus.subscribe(EVENT_MARKET_DATA, self._on_market_data)
        event_bus.subscribe(EVENT_ORDER_BOOK, self._on_order_book)
        event_bus.subscribe(EVENT_ACCOUNT_INFO, self._on_account_info)
        event_bus.subscribe(EVENT_TRADE_SIGNAL, self._on_trade_signal)
        event_bus.subscribe(EVENT_ORDER_UPDATE, self._on_order_update)
        event_bus.subscribe(EVENT_ERROR, self._on_error)
        event_bus.subscribe(EVENT_STATUS, self._on_status)
        
        # Start event bus
        event_bus.start()
    
    def _create_gui(self):
        """Create the GUI components."""
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create top frame for controls
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create symbol selection
        ttk.Label(top_frame, text="Symbol:").pack(side=tk.LEFT, padx=(0, 5))
        self.symbol_var = tk.StringVar(value=self.settings.trade_symbol)
        symbol_combo = ttk.Combobox(top_frame, textvariable=self.symbol_var, 
                                    values=self.settings.symbols)
        symbol_combo.pack(side=tk.LEFT, padx=(0, 10))
        symbol_combo.bind("<<ComboboxSelected>>", self._on_symbol_change)
        
        # Create start/stop buttons
        self.start_button = ttk.Button(top_frame, text="Start Bot", command=self.start_bot)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_button = ttk.Button(top_frame, text="Stop Bot", command=self.stop_bot, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Create manual trade buttons
        ttk.Label(top_frame, text="Manual:").pack(side=tk.LEFT, padx=(10, 5))
        
        self.buy_button = ttk.Button(top_frame, text="Buy", command=self._manual_buy)
        self.buy_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.sell_button = ttk.Button(top_frame, text="Sell", command=self._manual_sell)
        self.sell_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.close_button = ttk.Button(top_frame, text="Close Position", command=self._manual_close)
        self.close_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Create size entry
        ttk.Label(top_frame, text="Size:").pack(side=tk.LEFT, padx=(10, 5))
        self.size_var = tk.StringVar(value="1.0")
        size_entry = ttk.Entry(top_frame, textvariable=self.size_var, width=8)
        size_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create chart tab
        chart_frame = ttk.Frame(notebook)
        notebook.add(chart_frame, text="Price Chart")
        
        # Create chart
        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.ax1 = self.fig.add_subplot(211)  # Price chart
        self.ax2 = self.fig.add_subplot(212, sharex=self.ax1)  # Volume chart
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create logs tab
        logs_frame = ttk.Frame(notebook)
        notebook.add(logs_frame, text="Logs")
        
        # Create log text area
        self.log_text = scrolledtext.ScrolledText(logs_frame, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Create status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT)
        
        # Initialize UI
        self._update_ui()
    
    def _consume_logs(self):
        """Consume logs from the queue and display them in the UI."""
        while self.log_consumer_running:
            try:
                # Get log message from queue with timeout
                message = self.log_queue.get(block=True, timeout=0.1)
                
                # Update log text area
                self.log_text.insert(tk.END, message + "\n")
                self.log_text.see(tk.END)
                
                # Mark as done
                self.log_queue.task_done()
            except queue.Empty:
                # No log message available, continue
                pass
            except Exception as e:
                print(f"Error in log consumer: {e}")
            
            # Sleep briefly to reduce CPU usage
            time.sleep(0.01)
    
    def _update_ui(self):
        """Update UI elements."""
        if not self.running:
            return
        
        try:
            # Update status
            self.status_var.set(f"Running - {self.symbol_var.get()}")
            
            # Schedule next update
            self.root.after(1000, self._update_ui)
        except Exception as e:
            self.logger.error(f"Error updating UI: {e}")
    
    def _on_symbol_change(self, event):
        """Handle symbol change event."""
        symbol = self.symbol_var.get()
        self.logger.info(f"Symbol changed to {symbol}")
        
        # Publish symbol change event
        event_bus.publish_sync(EVENT_STATUS, {
            "type": "symbol_change",
            "symbol": symbol
        })
    
    def _manual_buy(self):
        """Handle manual buy button click."""
        try:
            symbol = self.symbol_var.get()
            size = float(self.size_var.get())
            
            self.logger.info(f"Manual buy: {symbol} {size}")
            
            # Publish manual trade event
            event_bus.publish_sync(EVENT_TRADE_SIGNAL, {
                "action": "buy",
                "symbol": symbol,
                "size": size,
                "source": "manual"
            })
            
            messagebox.showinfo("Manual Trade", f"Buy order placed for {size} {symbol}")
        except Exception as e:
            self.logger.error(f"Error placing buy order: {e}")
            messagebox.showerror("Error", f"Failed to place buy order: {e}")
    
    def _manual_sell(self):
        """Handle manual sell button click."""
        try:
            symbol = self.symbol_var.get()
            size = float(self.size_var.get())
            
            self.logger.info(f"Manual sell: {symbol} {size}")
            
            # Publish manual trade event
            event_bus.publish_sync(EVENT_TRADE_SIGNAL, {
                "action": "sell",
                "symbol": symbol,
                "size": size,
                "source": "manual"
            })
            
            messagebox.showinfo("Manual Trade", f"Sell order placed for {size} {symbol}")
        except Exception as e:
            self.logger.error(f"Error placing sell order: {e}")
            messagebox.showerror("Error", f"Failed to place sell order: {e}")
    
    def _manual_close(self):
        """Handle manual close position button click."""
        try:
            symbol = self.symbol_var.get()
            
            self.logger.info(f"Manual close position: {symbol}")
            
            # Publish manual trade event
            event_bus.publish_sync(EVENT_TRADE_SIGNAL, {
                "action": "close",
                "symbol": symbol,
                "source": "manual"
            })
            
            messagebox.showinfo("Manual Trade", f"Position closed for {symbol}")
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            messagebox.showerror("Error", f"Failed to close position: {e}")
    
    def start_bot(self):
        """Start the trading bot."""
        if self.running:
            return
        
        try:
            self.logger.info("Starting bot")
            self.running = True
            
            # Update button states
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            # Publish start event
            event_bus.publish_sync(EVENT_STATUS, {
                "type": "start",
                "symbol": self.symbol_var.get()
            })
            
            # Start UI updates
            self._update_ui()
            
            self.logger.info("Bot started")
        except Exception as e:
            self.logger.error(f"Error starting bot: {e}")
            self.running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
    
    def stop_bot(self):
        """Stop the trading bot."""
        if not self.running:
            return
        
        try:
            self.logger.info("Stopping bot")
            self.running = False
            
            # Update button states
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            
            # Publish stop event
            event_bus.publish_sync(EVENT_STATUS, {
                "type": "stop"
            })
            
            self.logger.info("Bot stopped")
        except Exception as e:
            self.logger.error(f"Error stopping bot: {e}")
    
    def _on_market_data(self, data):
        """Handle market data event."""
        try:
            symbol = data.get("symbol")
            if symbol:
                self.market_data[symbol] = data
                
                # Update chart if it's the current symbol
                if symbol == self.symbol_var.get():
                    # TODO: Update chart with new data
                    pass
        except Exception as e:
            self.logger.error(f"Error handling market data: {e}")
    
    def _on_order_book(self, data):
        """Handle order book event."""
        # TODO: Update order book display
        pass
    
    def _on_account_info(self, data):
        """Handle account info event."""
        try:
            # Update positions
            self.positions = data.get("positions", {})
            
            # TODO: Update account info display
        except Exception as e:
            self.logger.error(f"Error handling account info: {e}")
    
    def _on_trade_signal(self, data):
        """Handle trade signal event."""
        try:
            action = data.get("action")
            symbol = data.get("symbol")
            source = data.get("source", "strategy")
            
            # Log signal
            self.logger.info(f"Trade signal: {action} {symbol} from {source}")
            
            # TODO: Update signal display
        except Exception as e:
            self.logger.error(f"Error handling trade signal: {e}")
    
    def _on_order_update(self, data):
        """Handle order update event."""
        try:
            order_id = data.get("order_id")
            status = data.get("status")
            
            # Log order update
            self.logger.info(f"Order update: {order_id} {status}")
            
            # TODO: Update order display
        except Exception as e:
            self.logger.error(f"Error handling order update: {e}")
    
    def _on_error(self, data):
        """Handle error event."""
        try:
            error_type = data.get("type")
            message = data.get("message")
            
            # Log error
            self.logger.error(f"Error event: {error_type} - {message}")
            
            # Show error message if critical
            if data.get("critical", False):
                messagebox.showerror("Error", message)
        except Exception as e:
            self.logger.error(f"Error handling error event: {e}")
    
    def _on_status(self, data):
        """Handle status event."""
        try:
            status_type = data.get("type")
            
            # Update status display
            if status_type == "running":
                self.status_var.set(f"Running - {data.get('symbol', '')}")
            elif status_type == "stopped":
                self.status_var.set("Stopped")
            elif status_type == "error":
                self.status_var.set(f"Error: {data.get('message', '')}")
        except Exception as e:
            self.logger.error(f"Error handling status event: {e}")
    
    def on_closing(self):
        """Handle window closing event."""
        if self.running:
            if messagebox.askokcancel("Quit", "The bot is still running. Do you want to stop it and quit?"):
                self.stop_bot()
                self._cleanup()
                self.root.destroy()
        else:
            self._cleanup()
            self.root.destroy()
    
    def _cleanup(self):
        """Clean up resources."""
        # Stop log consumer
        self.log_consumer_running = False
        if self.log_consumer_thread and self.log_consumer_thread.is_alive():
            self.log_consumer_thread.join(timeout=1.0)
        
        # Stop event bus
        event_bus.stop()

def launch_gui(settings: BotSettings):
    """
    Launch the GUI application.
    
    Args:
        settings: Bot configuration settings
    """
    try:
        app = EnhancedTradingBotGUI(settings)
        app.root.mainloop()
    except Exception as e:
        logger.error(f"Error launching GUI: {e}")
        raise
