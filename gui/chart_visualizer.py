"""
Real-time Chart Visualization Module for Hyperliquid Trading Bot

This module provides real-time chart visualization tools for price data, including:
- Candlestick charts with customizable timeframes
- Technical indicator overlays
- Volume profile visualization
- Trading signal visualization
- Multi-timeframe analysis

Features:
- Interactive charts with zoom and pan
- Customizable technical indicators
- Real-time updates with configurable refresh rate
- Multiple chart types (candlestick, line, area)
- Trading signal integration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import time
import threading
import datetime
from datetime import datetime, timedelta
import mplfinance as mpf

# Configure logging
logger = logging.getLogger(__name__)

class ChartVisualizer:
    """
    Real-time chart visualization tools for the Hyperliquid Trading Bot.
    
    Provides interactive charts for price data with technical indicators
    and trading signal visualization.
    """
    
    def __init__(self, parent_frame=None, symbol="XRP-PERP", 
                 timeframe: str = "1h", max_candles: int = 100,
                 refresh_rate: float = 5.0, theme: str = "dark"):
        """
        Initialize the chart visualizer.
        
        Args:
            parent_frame: Parent tkinter frame for visualization
            symbol: Trading symbol
            timeframe: Chart timeframe (e.g., '1m', '5m', '1h', '1d')
            max_candles: Maximum number of candles to display
            refresh_rate: Refresh rate in seconds
            theme: Visualization theme ('dark' or 'light')
        """
        self.parent_frame = parent_frame
        self.symbol = symbol
        self.timeframe = timeframe
        self.max_candles = max_candles
        self.refresh_rate = refresh_rate
        self.theme = theme
        
        # Initialize data structures
        self.ohlcv_data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.indicators = {}
        self.signals = []
        
        # Set up visualization components
        self.setup_visualization_components()
        
        # Set up update thread
        self.running = False
        self.update_thread = None
        
        logger.info(f"Chart visualizer initialized for {symbol} ({timeframe})")
    
    def setup_visualization_components(self):
        """Set up visualization components based on parent frame."""
        if self.parent_frame is None:
            return
        
        # Create main frame
        self.main_frame = ttk.Frame(self.parent_frame)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create top control frame
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create timeframe selector
        ttk.Label(self.control_frame, text="Timeframe:").pack(side=tk.LEFT, padx=5)
        self.timeframe_var = tk.StringVar(value=self.timeframe)
        timeframe_combo = ttk.Combobox(self.control_frame, textvariable=self.timeframe_var, 
                                      values=["1m", "5m", "15m", "1h", "4h", "1d"], width=5)
        timeframe_combo.pack(side=tk.LEFT, padx=5)
        timeframe_combo.bind("<<ComboboxSelected>>", self.on_timeframe_change)
        
        # Create indicator selector
        ttk.Label(self.control_frame, text="Indicators:").pack(side=tk.LEFT, padx=5)
        self.indicator_var = tk.StringVar(value="None")
        indicator_combo = ttk.Combobox(self.control_frame, textvariable=self.indicator_var, 
                                      values=["None", "MA", "EMA", "MACD", "RSI", "BB"], width=5)
        indicator_combo.pack(side=tk.LEFT, padx=5)
        indicator_combo.bind("<<ComboboxSelected>>", self.on_indicator_change)
        
        # Create chart type selector
        ttk.Label(self.control_frame, text="Chart Type:").pack(side=tk.LEFT, padx=5)
        self.chart_type_var = tk.StringVar(value="Candle")
        chart_type_combo = ttk.Combobox(self.control_frame, textvariable=self.chart_type_var, 
                                       values=["Candle", "Line", "OHLC"], width=5)
        chart_type_combo.pack(side=tk.LEFT, padx=5)
        chart_type_combo.bind("<<ComboboxSelected>>", self.on_chart_type_change)
        
        # Create refresh button
        refresh_button = ttk.Button(self.control_frame, text="Refresh", command=self.refresh_chart)
        refresh_button.pack(side=tk.RIGHT, padx=5)
        
        # Create chart frame
        self.chart_frame = ttk.Frame(self.main_frame)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create figure and canvas
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self.chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.chart_frame)
        self.toolbar.update()
        
        # Create subplots
        self.setup_subplots()
    
    def setup_subplots(self):
        """Set up chart subplots."""
        # Clear previous subplots
        self.figure.clear()
        
        # Create price subplot (top, 70% height)
        self.price_ax = self.figure.add_subplot(5, 1, (1, 3))
        
        # Create volume subplot (bottom, 15% height)
        self.volume_ax = self.figure.add_subplot(5, 1, 4, sharex=self.price_ax)
        
        # Create indicator subplot (bottom, 15% height)
        self.indicator_ax = self.figure.add_subplot(5, 1, 5, sharex=self.price_ax)
        
        # Set theme
        if self.theme == "dark":
            self.figure.patch.set_facecolor('#2d2d2d')
            
            for ax in [self.price_ax, self.volume_ax, self.indicator_ax]:
                ax.set_facecolor('#2d2d2d')
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
        
        # Set labels
        self.price_ax.set_title(f"{self.symbol} ({self.timeframe})")
        self.price_ax.set_ylabel("Price")
        self.volume_ax.set_ylabel("Volume")
        self.indicator_ax.set_ylabel("Indicator")
        self.indicator_ax.set_xlabel("Time")
        
        # Format x-axis
        self.price_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        self.price_ax.tick_params(axis='x', rotation=45)
        
        # Hide x-axis labels for price and volume subplots
        self.price_ax.tick_params(axis='x', labelbottom=False)
        self.volume_ax.tick_params(axis='x', labelbottom=False)
        
        # Adjust layout
        self.figure.tight_layout()
        self.figure.subplots_adjust(hspace=0.1)
        
        # Initial empty plot
        self.price_ax.plot([], [], 'w-', label="Close")
        self.price_ax.legend()
        
        self.canvas.draw()
    
    def on_timeframe_change(self, event):
        """Handle timeframe change event."""
        new_timeframe = self.timeframe_var.get()
        if new_timeframe != self.timeframe:
            self.timeframe = new_timeframe
            logger.info(f"Changed timeframe to {self.timeframe}")
            self.refresh_chart()
    
    def on_indicator_change(self, event):
        """Handle indicator change event."""
        indicator = self.indicator_var.get()
        logger.info(f"Changed indicator to {indicator}")
        self.refresh_chart()
    
    def on_chart_type_change(self, event):
        """Handle chart type change event."""
        chart_type = self.chart_type_var.get()
        logger.info(f"Changed chart type to {chart_type}")
        self.refresh_chart()
    
    def refresh_chart(self):
        """Manually refresh the chart."""
        self.update_chart()
    
    def update_ohlcv_data(self, data: pd.DataFrame):
        """
        Update OHLCV data.
        
        Args:
            data: DataFrame with OHLCV data
        """
        # Ensure data has required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Data must have '{column}' column")
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        # Limit to max candles
        if len(data) > self.max_candles:
            data = data.tail(self.max_candles)
        
        # Update instance variable
        self.ohlcv_data = data
        
        # Update chart
        self.update_chart()
        
        logger.debug(f"OHLCV data updated for {self.symbol} ({self.timeframe})")
    
    def update_chart(self):
        """Update chart visualization."""
        if self.ohlcv_data.empty:
            return
        
        # Clear previous plot
        self.price_ax.clear()
        self.volume_ax.clear()
        self.indicator_ax.clear()
        
        # Get data
        timestamps = self.ohlcv_data['timestamp']
        opens = self.ohlcv_data['open']
        highs = self.ohlcv_data['high']
        lows = self.ohlcv_data['low']
        closes = self.ohlcv_data['close']
        volumes = self.ohlcv_data['volume']
        
        # Plot based on chart type
        chart_type = self.chart_type_var.get()
        
        if chart_type == "Candle":
            # Plot candlestick chart
            width = 0.6
            width2 = width * 0.8
            
            up = closes >= opens
            down = closes < opens
            
            # Plot up candles
            self.price_ax.bar(timestamps[up], highs[up] - lows[up], width=width2, 
                             bottom=lows[up], color='g', alpha=0.5)
            self.price_ax.bar(timestamps[up], closes[up] - opens[up], width=width, 
                             bottom=opens[up], color='g')
            
            # Plot down candles
            self.price_ax.bar(timestamps[down], highs[down] - lows[down], width=width2, 
                             bottom=lows[down], color='r', alpha=0.5)
            self.price_ax.bar(timestamps[down], closes[down] - opens[down], width=width, 
                             bottom=opens[down], color='r')
        
        elif chart_type == "OHLC":
            # Plot OHLC chart
            width = 0.6
            
            for i, (timestamp, open_price, high, low, close) in enumerate(
                zip(timestamps, opens, highs, lows, closes)):
                # Plot high-low line
                self.price_ax.plot([timestamp, timestamp], [low, high], 'k-')
                
                # Plot open tick
                self.price_ax.plot([timestamp - width/2, timestamp], [open_price, open_price], 
                                  'k-', color='g' if close >= open_price else 'r')
                
                # Plot close tick
                self.price_ax.plot([timestamp, timestamp + width/2], [close, close], 
                                  'k-', color='g' if close >= open_price else 'r')
        
        else:  # Line chart
            # Plot line chart
            self.price_ax.plot(timestamps, closes, 'b-', label="Close")
        
        # Plot volume
        self.volume_ax.bar(timestamps, volumes, width=0.8, color='b', alpha=0.5)
        
        # Plot selected indicator
        indicator = self.indicator_var.get()
        
        if indicator == "MA":
            # Simple Moving Average
            ma_period = 20
            ma = self.calculate_ma(closes, ma_period)
            self.price_ax.plot(timestamps, ma, 'y-', label=f"MA({ma_period})")
        
        elif indicator == "EMA":
            # Exponential Moving Average
            ema_period = 20
            ema = self.calculate_ema(closes, ema_period)
            self.price_ax.plot(timestamps, ema, 'm-', label=f"EMA({ema_period})")
        
        elif indicator == "MACD":
            # MACD
            macd, signal, histogram = self.calculate_macd(closes)
            self.indicator_ax.plot(timestamps, macd, 'b-', label="MACD")
            self.indicator_ax.plot(timestamps, signal, 'r-', label="Signal")
            self.indicator_ax.bar(timestamps, histogram, width=0.6, color='g', alpha=0.5)
            self.indicator_ax.axhline(y=0, color='w', linestyle='-', alpha=0.3)
            self.indicator_ax.set_ylabel("MACD")
        
        elif indicator == "RSI":
            # RSI
            rsi = self.calculate_rsi(closes)
            self.indicator_ax.plot(timestamps, rsi, 'g-', label="RSI")
            self.indicator_ax.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            self.indicator_ax.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            self.indicator_ax.set_ylim(0, 100)
            self.indicator_ax.set_ylabel("RSI")
        
        elif indicator == "BB":
            # Bollinger Bands
            upper, middle, lower = self.calculate_bollinger_bands(closes)
            self.price_ax.plot(timestamps, upper, 'r--', label="Upper BB")
            self.price_ax.plot(timestamps, middle, 'y-', label="Middle BB")
            self.price_ax.plot(timestamps, lower, 'g--', label="Lower BB")
        
        # Plot trading signals if available
        self.plot_signals()
        
        # Set labels
        self.price_ax.set_title(f"{self.symbol} ({self.timeframe})")
        self.price_ax.set_ylabel("Price")
        self.volume_ax.set_ylabel("Volume")
        self.indicator_ax.set_xlabel("Time")
        
        # Add legend
        self.price_ax.legend(loc='upper left')
        if indicator in ["MACD", "RSI"]:
            self.indicator_ax.legend(loc='upper left')
        
        # Format x-axis
        self.price_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        self.price_ax.tick_params(axis='x', rotation=45)
        
        # Hide x-axis labels for price and volume subplots
        self.price_ax.tick_params(axis='x', labelbottom=False)
        self.volume_ax.tick_params(axis='x', labelbottom=False)
        
        # Adjust layout
        self.figure.tight_layout()
        self.figure.subplots_adjust(hspace=0.1)
        
        # Update canvas
        self.canvas.draw()
    
    def plot_signals(self):
        """Plot trading signals on the chart."""
        if not self.signals:
            return
        
        for signal in self.signals:
            timestamp = signal.get('timestamp')
            price = signal.get('price')
            direction = signal.get('direction', 0)
            
            if timestamp is None or price is None:
                continue
            
            # Convert timestamp to datetime if it's not already
            if not isinstance(timestamp, datetime):
                timestamp = pd.to_datetime(timestamp, unit='ms')
            
            # Plot signal marker
            if direction > 0:
                # Buy signal
                self.price_ax.scatter(timestamp, price, marker='^', color='g', s=100, zorder=5)
            elif direction < 0:
                # Sell signal
                self.price_ax.scatter(timestamp, price, marker='v', color='r', s=100, zorder=5)
    
    def add_signal(self, signal: Dict[str, Any]):
        """
        Add trading signal to the chart.
        
        Args:
            signal: Dictionary with signal information
        """
        self.signals.append(signal)
        self.update_chart()
    
    def clear_signals(self):
        """Clear all trading signals from the chart."""
        self.signals = []
        self.update_chart()
    
    def calculate_ma(self, data: np.ndarray, period: int = 20) -> np.ndarray:
        """
        Calculate Simple Moving Average.
        
        Args:
            data: Price data
            period: MA period
            
        Returns:
            ndarray with MA values
        """
        ma = np.zeros_like(data)
        
        for i in range(period - 1, len(data)):
            ma[i] = np.mean(data[i - period + 1:i + 1])
        
        return ma
    
    def calculate_ema(self, data: np.ndarray, period: int = 20) -> np.ndarray:
        """
        Calculate Exponential Moving Average.
        
        Args:
            data: Price data
            period: EMA period
            
        Returns:
            ndarray with EMA values
        """
        ema = np.zeros_like(data)
        ema[period - 1] = np.mean(data[:period])
        
        multiplier = 2.0 / (period + 1)
        
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]
        
        return ema
    
    def calculate_macd(self, data: np.ndarray, fast_period: int = 12, 
                      slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate MACD.
        
        Args:
            data: Price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal EMA period
            
        Returns:
            Tuple of (macd, signal, histogram)
        """
        ema_fast = self.calculate_ema(data, fast_period)
        ema_slow = self.calculate_ema(data, slow_period)
        
        macd = ema_fast - ema_slow
        signal = self.calculate_ema(macd, signal_period)
        histogram = macd - signal
        
        return macd, signal, histogram
    
    def calculate_rsi(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calculate RSI.
        
        Args:
            data: Price data
            period: RSI period
            
        Returns:
            ndarray with RSI values
        """
        rsi = np.zeros_like(data)
        
        # Calculate price changes
        deltas = np.diff(data)
        deltas = np.append(0, deltas)
        
        # Calculate gains and losses
        gains = np.zeros_like(deltas)
        losses = np.zeros_like(deltas)
        
        gains[deltas > 0] = deltas[deltas > 0]
        losses[deltas < 0] = -deltas[deltas < 0]
        
        # Calculate average gains and losses
        avg_gain = np.zeros_like(data)
        avg_loss = np.zeros_like(data)
        
        # Initialize
        avg_gain[period] = np.mean(gains[1:period+1])
        avg_loss[period] = np.mean(losses[1:period+1])
        
        # Calculate RSI
        for i in range(period + 1, len(data)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period
            
            if avg_loss[i] == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    def calculate_bollinger_bands(self, data: np.ndarray, period: int = 20, 
                                 std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: Price data
            period: SMA period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle_band = self.calculate_ma(data, period)
        
        # Calculate standard deviation
        rolling_std = np.zeros_like(data)
        
        for i in range(period - 1, len(data)):
            rolling_std[i] = np.std(data[i - period + 1:i + 1])
        
        # Calculate upper and lower bands
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    def start_updates(self, data_provider=None):
        """
        Start periodic updates of chart visualization.
        
        Args:
            data_provider: Function or object that provides OHLCV data
        """
        if self.running:
            return
        
        self.running = True
        self.data_provider = data_provider
        
        def update_loop():
            while self.running:
                try:
                    if self.data_provider is not None:
                        # Get data from provider
                        ohlcv_data = self.data_provider.get_ohlcv_data(self.symbol, self.timeframe)
                        if ohlcv_data is not None:
                            self.update_ohlcv_data(ohlcv_data)
                except Exception as e:
                    logger.error(f"Error updating chart: {e}")
                
                time.sleep(self.refresh_rate)
        
        self.update_thread = threading.Thread(target=update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        logger.info(f"Started chart updates for {self.symbol} ({self.timeframe})")
    
    def stop_updates(self):
        """Stop periodic updates of chart visualization."""
        self.running = False
        if self.update_thread is not None:
            self.update_thread.join(timeout=1.0)
            self.update_thread = None
        
        logger.info(f"Stopped chart updates for {self.symbol} ({self.timeframe})")
    
    def save_chart(self, filename: str = None):
        """
        Save current chart to file.
        
        Args:
            filename: Output filename (default: {symbol}_{timeframe}_{timestamp}.png)
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"{self.symbol}_{self.timeframe}_{timestamp}.png"
        
        self.figure.savefig(filename)
        logger.info(f"Saved chart to {filename}")


class MockOHLCVProvider:
    """
    Mock OHLCV data provider for testing.
    
    Generates realistic OHLCV data for testing visualization
    without connecting to an exchange.
    """
    
    def __init__(self, symbol="XRP-PERP", initial_price: float = 0.5, 
                 volatility: float = 0.01, volume_scale: float = 1000.0):
        """
        Initialize mock OHLCV provider.
        
        Args:
            symbol: Trading symbol
            initial_price: Initial price
            volatility: Price volatility
            volume_scale: Volume scale factor
        """
        self.symbol = symbol
        self.initial_price = initial_price
        self.volatility = volatility
        self.volume_scale = volume_scale
        
        # Initialize random state
        self.random_state = np.random.RandomState(42)
        
        # Initialize data cache
        self.data_cache = {}
        
        logger.info(f"Mock OHLCV provider initialized for {symbol}")
    
    def get_ohlcv_data(self, symbol: str = None, timeframe: str = "1h") -> pd.DataFrame:
        """
        Get mock OHLCV data.
        
        Args:
            symbol: Trading symbol (ignored in mock provider)
            timeframe: Chart timeframe (e.g., '1m', '5m', '1h', '1d')
            
        Returns:
            DataFrame with OHLCV data
        """
        # Check if data is already in cache
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        # Determine number of candles and time delta based on timeframe
        num_candles = 100
        
        if timeframe == "1m":
            time_delta = timedelta(minutes=1)
        elif timeframe == "5m":
            time_delta = timedelta(minutes=5)
        elif timeframe == "15m":
            time_delta = timedelta(minutes=15)
        elif timeframe == "1h":
            time_delta = timedelta(hours=1)
        elif timeframe == "4h":
            time_delta = timedelta(hours=4)
        elif timeframe == "1d":
            time_delta = timedelta(days=1)
        else:
            time_delta = timedelta(hours=1)
        
        # Generate timestamps
        end_time = datetime.now()
        timestamps = [end_time - (time_delta * i) for i in range(num_candles, 0, -1)]
        
        # Generate price data with random walk
        closes = np.zeros(num_candles)
        closes[0] = self.initial_price
        
        for i in range(1, num_candles):
            closes[i] = closes[i-1] * (1 + self.random_state.normal(0, self.volatility))
        
        # Generate OHLC data
        opens = np.zeros(num_candles)
        highs = np.zeros(num_candles)
        lows = np.zeros(num_candles)
        
        for i in range(num_candles):
            if i == 0:
                opens[i] = self.initial_price * (1 + self.random_state.normal(0, self.volatility * 0.5))
            else:
                opens[i] = closes[i-1] * (1 + self.random_state.normal(0, self.volatility * 0.5))
            
            price_range = abs(closes[i] - opens[i]) + (closes[i] * self.volatility)
            
            highs[i] = max(opens[i], closes[i]) + (price_range * self.random_state.random())
            lows[i] = min(opens[i], closes[i]) - (price_range * self.random_state.random())
        
        # Generate volume data
        volumes = self.random_state.exponential(1.0, num_candles) * self.volume_scale
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        # Cache data
        self.data_cache[cache_key] = data
        
        return data


# Example usage
if __name__ == "__main__":
    # Create mock data provider
    mock_provider = MockOHLCVProvider(symbol="XRP-PERP", initial_price=0.5)
    
    # Create tkinter window
    root = tk.Tk()
    root.title("Chart Visualization")
    root.geometry("1200x800")
    
    # Create chart visualizer
    visualizer = ChartVisualizer(root, symbol="XRP-PERP", theme="dark")
    
    # Start updates
    visualizer.start_updates(mock_provider)
    
    # Run tkinter main loop
    root.mainloop()
    
    # Stop updates when window is closed
    visualizer.stop_updates()
