"""
Order Book Visualization Module for Hyperliquid Trading Bot

This module provides visualization tools for the order book data, including:
- Depth chart visualization
- Order book table display
- Real-time order book updates
- Bid/ask spread analysis
- Volume profile visualization

Features:
- Customizable visualization parameters
- Real-time updates with configurable refresh rate
- Multiple visualization modes (depth chart, table, heatmap)
- Integration with trading signals
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import time
import threading

# Configure logging
logger = logging.getLogger(__name__)

class OrderBookVisualizer:
    """
    Order book visualization tools for the Hyperliquid Trading Bot.
    
    Provides multiple visualization methods for order book data,
    including depth charts, tables, and heatmaps.
    """
    
    def __init__(self, parent_frame=None, symbol="XRP-PERP", 
                 max_depth: int = 20, refresh_rate: float = 1.0,
                 theme: str = "dark"):
        """
        Initialize the order book visualizer.
        
        Args:
            parent_frame: Parent tkinter frame for visualization
            symbol: Trading symbol
            max_depth: Maximum depth of order book to display
            refresh_rate: Refresh rate in seconds
            theme: Visualization theme ('dark' or 'light')
        """
        self.parent_frame = parent_frame
        self.symbol = symbol
        self.max_depth = max_depth
        self.refresh_rate = refresh_rate
        self.theme = theme
        
        # Initialize data structures
        self.bids = pd.DataFrame(columns=['price', 'size', 'total'])
        self.asks = pd.DataFrame(columns=['price', 'size', 'total'])
        
        # Set up visualization components
        self.setup_visualization_components()
        
        # Set up update thread
        self.running = False
        self.update_thread = None
        
        logger.info(f"Order book visualizer initialized for {symbol}")
    
    def setup_visualization_components(self):
        """Set up visualization components based on parent frame."""
        if self.parent_frame is None:
            return
        
        # Create main frame
        self.main_frame = ttk.Frame(self.parent_frame)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.tab_control = ttk.Notebook(self.main_frame)
        
        # Depth chart tab
        self.depth_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.depth_tab, text="Depth Chart")
        
        # Order book tab
        self.book_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.book_tab, text="Order Book")
        
        # Heatmap tab
        self.heatmap_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.heatmap_tab, text="Heatmap")
        
        self.tab_control.pack(fill=tk.BOTH, expand=True)
        
        # Set up depth chart
        self.setup_depth_chart()
        
        # Set up order book table
        self.setup_order_book_table()
        
        # Set up heatmap
        self.setup_heatmap()
    
    def setup_depth_chart(self):
        """Set up depth chart visualization."""
        if self.depth_tab is None:
            return
        
        # Create figure and canvas
        self.depth_figure = Figure(figsize=(8, 4), dpi=100)
        self.depth_canvas = FigureCanvasTkAgg(self.depth_figure, self.depth_tab)
        self.depth_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create subplot
        self.depth_ax = self.depth_figure.add_subplot(111)
        
        # Set theme
        if self.theme == "dark":
            self.depth_figure.patch.set_facecolor('#2d2d2d')
            self.depth_ax.set_facecolor('#2d2d2d')
            self.depth_ax.tick_params(colors='white')
            self.depth_ax.spines['bottom'].set_color('white')
            self.depth_ax.spines['top'].set_color('white')
            self.depth_ax.spines['left'].set_color('white')
            self.depth_ax.spines['right'].set_color('white')
            self.depth_ax.xaxis.label.set_color('white')
            self.depth_ax.yaxis.label.set_color('white')
            self.depth_ax.title.set_color('white')
        
        # Set labels
        self.depth_ax.set_title(f"{self.symbol} Order Book Depth")
        self.depth_ax.set_xlabel("Price")
        self.depth_ax.set_ylabel("Cumulative Size")
        
        # Initial plot
        self.depth_ax.plot([], [], 'g-', label="Bids")
        self.depth_ax.plot([], [], 'r-', label="Asks")
        self.depth_ax.legend()
        
        self.depth_figure.tight_layout()
        self.depth_canvas.draw()
    
    def setup_order_book_table(self):
        """Set up order book table visualization."""
        if self.book_tab is None:
            return
        
        # Create frames for bids and asks
        self.book_frame = ttk.Frame(self.book_tab)
        self.book_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create bid frame (left side)
        self.bid_frame = ttk.Frame(self.book_frame)
        self.bid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create ask frame (right side)
        self.ask_frame = ttk.Frame(self.book_frame)
        self.ask_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create bid table
        bid_label = ttk.Label(self.bid_frame, text="Bids", anchor=tk.CENTER)
        bid_label.pack(fill=tk.X)
        
        self.bid_tree = ttk.Treeview(self.bid_frame, columns=("price", "size", "total"), show="headings")
        self.bid_tree.heading("price", text="Price")
        self.bid_tree.heading("size", text="Size")
        self.bid_tree.heading("total", text="Total")
        
        self.bid_tree.column("price", width=100, anchor=tk.E)
        self.bid_tree.column("size", width=100, anchor=tk.E)
        self.bid_tree.column("total", width=100, anchor=tk.E)
        
        # Add scrollbar to bid table
        bid_scrollbar = ttk.Scrollbar(self.bid_frame, orient=tk.VERTICAL, command=self.bid_tree.yview)
        self.bid_tree.configure(yscrollcommand=bid_scrollbar.set)
        
        bid_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.bid_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create ask table
        ask_label = ttk.Label(self.ask_frame, text="Asks", anchor=tk.CENTER)
        ask_label.pack(fill=tk.X)
        
        self.ask_tree = ttk.Treeview(self.ask_frame, columns=("price", "size", "total"), show="headings")
        self.ask_tree.heading("price", text="Price")
        self.ask_tree.heading("size", text="Size")
        self.ask_tree.heading("total", text="Total")
        
        self.ask_tree.column("price", width=100, anchor=tk.E)
        self.ask_tree.column("size", width=100, anchor=tk.E)
        self.ask_tree.column("total", width=100, anchor=tk.E)
        
        # Add scrollbar to ask table
        ask_scrollbar = ttk.Scrollbar(self.ask_frame, orient=tk.VERTICAL, command=self.ask_tree.yview)
        self.ask_tree.configure(yscrollcommand=ask_scrollbar.set)
        
        ask_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.ask_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Set tag configurations for coloring
        self.bid_tree.tag_configure('bid', background='#1e3f1e')
        self.ask_tree.tag_configure('ask', background='#3f1e1e')
    
    def setup_heatmap(self):
        """Set up order book heatmap visualization."""
        if self.heatmap_tab is None:
            return
        
        # Create figure and canvas
        self.heatmap_figure = Figure(figsize=(8, 6), dpi=100)
        self.heatmap_canvas = FigureCanvasTkAgg(self.heatmap_figure, self.heatmap_tab)
        self.heatmap_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create subplot
        self.heatmap_ax = self.heatmap_figure.add_subplot(111)
        
        # Set theme
        if self.theme == "dark":
            self.heatmap_figure.patch.set_facecolor('#2d2d2d')
            self.heatmap_ax.set_facecolor('#2d2d2d')
            self.heatmap_ax.tick_params(colors='white')
            self.heatmap_ax.spines['bottom'].set_color('white')
            self.heatmap_ax.spines['top'].set_color('white')
            self.heatmap_ax.spines['left'].set_color('white')
            self.heatmap_ax.spines['right'].set_color('white')
            self.heatmap_ax.xaxis.label.set_color('white')
            self.heatmap_ax.yaxis.label.set_color('white')
            self.heatmap_ax.title.set_color('white')
        
        # Set labels
        self.heatmap_ax.set_title(f"{self.symbol} Order Book Heatmap")
        self.heatmap_ax.set_xlabel("Price Levels")
        self.heatmap_ax.set_ylabel("Order Size")
        
        # Initial empty heatmap
        self.heatmap = self.heatmap_ax.imshow(
            np.zeros((2, self.max_depth)),
            aspect='auto',
            cmap='RdYlGn',
            interpolation='nearest'
        )
        
        # Add colorbar
        self.heatmap_figure.colorbar(self.heatmap, ax=self.heatmap_ax, label="Order Size")
        
        self.heatmap_figure.tight_layout()
        self.heatmap_canvas.draw()
    
    def update_order_book(self, bids: List[List[float]], asks: List[List[float]]):
        """
        Update order book data.
        
        Args:
            bids: List of [price, size] pairs for bids
            asks: List of [price, size] pairs for asks
        """
        # Convert to DataFrames
        bids_df = pd.DataFrame(bids, columns=['price', 'size'])
        asks_df = pd.DataFrame(asks, columns=['price', 'size'])
        
        # Sort bids (descending) and asks (ascending)
        bids_df = bids_df.sort_values('price', ascending=False)
        asks_df = asks_df.sort_values('price', ascending=True)
        
        # Limit to max depth
        bids_df = bids_df.head(self.max_depth)
        asks_df = asks_df.head(self.max_depth)
        
        # Calculate cumulative sizes
        bids_df['total'] = bids_df['size'].cumsum()
        asks_df['total'] = asks_df['size'].cumsum()
        
        # Update instance variables
        self.bids = bids_df
        self.asks = asks_df
        
        # Update visualizations
        self.update_depth_chart()
        self.update_order_book_table()
        self.update_heatmap()
        
        logger.debug(f"Order book updated for {self.symbol}")
    
    def update_depth_chart(self):
        """Update depth chart visualization."""
        if not hasattr(self, 'depth_ax'):
            return
        
        # Clear previous plot
        self.depth_ax.clear()
        
        # Plot bids and asks
        if not self.bids.empty:
            self.depth_ax.plot(self.bids['price'], self.bids['total'], 'g-', label="Bids")
        
        if not self.asks.empty:
            self.depth_ax.plot(self.asks['price'], self.asks['total'], 'r-', label="Asks")
        
        # Set labels
        self.depth_ax.set_title(f"{self.symbol} Order Book Depth")
        self.depth_ax.set_xlabel("Price")
        self.depth_ax.set_ylabel("Cumulative Size")
        
        # Add legend
        self.depth_ax.legend()
        
        # Add mid price line if both bids and asks exist
        if not self.bids.empty and not self.asks.empty:
            mid_price = (self.bids['price'].iloc[0] + self.asks['price'].iloc[0]) / 2
            self.depth_ax.axvline(x=mid_price, color='y', linestyle='--', alpha=0.5)
            self.depth_ax.text(mid_price, 0, f"Mid: {mid_price:.2f}", 
                              color='y', ha='center', va='bottom')
        
        # Update canvas
        self.depth_figure.tight_layout()
        self.depth_canvas.draw()
    
    def update_order_book_table(self):
        """Update order book table visualization."""
        if not hasattr(self, 'bid_tree') or not hasattr(self, 'ask_tree'):
            return
        
        # Clear previous data
        for item in self.bid_tree.get_children():
            self.bid_tree.delete(item)
        
        for item in self.ask_tree.get_children():
            self.ask_tree.delete(item)
        
        # Add bids
        for _, row in self.bids.iterrows():
            self.bid_tree.insert("", tk.END, values=(
                f"{row['price']:.4f}",
                f"{row['size']:.4f}",
                f"{row['total']:.4f}"
            ), tags=('bid',))
        
        # Add asks
        for _, row in self.asks.iterrows():
            self.ask_tree.insert("", tk.END, values=(
                f"{row['price']:.4f}",
                f"{row['size']:.4f}",
                f"{row['total']:.4f}"
            ), tags=('ask',))
    
    def update_heatmap(self):
        """Update order book heatmap visualization."""
        if not hasattr(self, 'heatmap_ax') or not hasattr(self, 'heatmap'):
            return
        
        # Create heatmap data
        heatmap_data = np.zeros((2, self.max_depth))
        
        # Fill bid data (row 0)
        for i, (_, row) in enumerate(self.bids.iterrows()):
            if i < self.max_depth:
                heatmap_data[0, i] = row['size']
        
        # Fill ask data (row 1)
        for i, (_, row) in enumerate(self.asks.iterrows()):
            if i < self.max_depth:
                heatmap_data[1, i] = row['size']
        
        # Update heatmap
        self.heatmap.set_data(heatmap_data)
        
        # Update colorbar scale
        self.heatmap.set_clim(0, np.max(heatmap_data) if np.max(heatmap_data) > 0 else 1)
        
        # Update x-axis labels
        if not self.bids.empty and not self.asks.empty:
            bid_prices = self.bids['price'].values
            ask_prices = self.asks['price'].values
            
            # Create price labels (show only a few for readability)
            num_labels = 5
            bid_indices = np.linspace(0, len(bid_prices) - 1, num_labels, dtype=int)
            ask_indices = np.linspace(0, len(ask_prices) - 1, num_labels, dtype=int)
            
            bid_labels = [f"{bid_prices[i]:.2f}" for i in bid_indices]
            ask_labels = [f"{ask_prices[i]:.2f}" for i in ask_indices]
            
            # Set x-axis tick positions and labels
            tick_positions = np.concatenate([bid_indices, ask_indices + self.max_depth])
            tick_labels = bid_labels + ask_labels
            
            self.heatmap_ax.set_xticks(tick_positions)
            self.heatmap_ax.set_xticklabels(tick_labels, rotation=45)
        
        # Set y-axis labels
        self.heatmap_ax.set_yticks([0, 1])
        self.heatmap_ax.set_yticklabels(['Bids', 'Asks'])
        
        # Update canvas
        self.heatmap_figure.tight_layout()
        self.heatmap_canvas.draw()
    
    def start_updates(self, data_provider=None):
        """
        Start periodic updates of order book visualization.
        
        Args:
            data_provider: Function or object that provides order book data
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
                        order_book = self.data_provider.get_order_book(self.symbol)
                        if order_book is not None:
                            bids = order_book.get('bids', [])
                            asks = order_book.get('asks', [])
                            self.update_order_book(bids, asks)
                except Exception as e:
                    logger.error(f"Error updating order book: {e}")
                
                time.sleep(self.refresh_rate)
        
        self.update_thread = threading.Thread(target=update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        logger.info(f"Started order book updates for {self.symbol}")
    
    def stop_updates(self):
        """Stop periodic updates of order book visualization."""
        self.running = False
        if self.update_thread is not None:
            self.update_thread.join(timeout=1.0)
            self.update_thread = None
        
        logger.info(f"Stopped order book updates for {self.symbol}")
    
    def calculate_spread(self) -> Tuple[float, float]:
        """
        Calculate bid-ask spread.
        
        Returns:
            Tuple of (absolute_spread, percentage_spread)
        """
        if self.bids.empty or self.asks.empty:
            return 0.0, 0.0
        
        best_bid = self.bids['price'].iloc[0]
        best_ask = self.asks['price'].iloc[0]
        
        absolute_spread = best_ask - best_bid
        percentage_spread = (absolute_spread / best_bid) * 100.0
        
        return absolute_spread, percentage_spread
    
    def calculate_market_depth(self, price_range_pct: float = 1.0) -> Tuple[float, float]:
        """
        Calculate market depth within a given price range.
        
        Args:
            price_range_pct: Price range as percentage of mid price
            
        Returns:
            Tuple of (bid_depth, ask_depth)
        """
        if self.bids.empty or self.asks.empty:
            return 0.0, 0.0
        
        best_bid = self.bids['price'].iloc[0]
        best_ask = self.asks['price'].iloc[0]
        mid_price = (best_bid + best_ask) / 2
        
        price_range = mid_price * (price_range_pct / 100.0)
        
        bid_min = mid_price - price_range
        ask_max = mid_price + price_range
        
        bid_depth = self.bids[self.bids['price'] >= bid_min]['size'].sum()
        ask_depth = self.asks[self.asks['price'] <= ask_max]['size'].sum()
        
        return bid_depth, ask_depth
    
    def calculate_imbalance(self) -> float:
        """
        Calculate order book imbalance.
        
        Returns:
            Imbalance ratio (-1.0 to 1.0, negative means more asks, positive means more bids)
        """
        if self.bids.empty or self.asks.empty:
            return 0.0
        
        bid_total = self.bids['size'].sum()
        ask_total = self.asks['size'].sum()
        
        if bid_total + ask_total == 0:
            return 0.0
        
        imbalance = (bid_total - ask_total) / (bid_total + ask_total)
        
        return imbalance
    
    def get_order_book_snapshot(self) -> Dict[str, Any]:
        """
        Get a snapshot of current order book metrics.
        
        Returns:
            Dictionary with order book metrics
        """
        absolute_spread, percentage_spread = self.calculate_spread()
        bid_depth, ask_depth = self.calculate_market_depth()
        imbalance = self.calculate_imbalance()
        
        if not self.bids.empty and not self.asks.empty:
            best_bid = self.bids['price'].iloc[0]
            best_ask = self.asks['price'].iloc[0]
            mid_price = (best_bid + best_ask) / 2
        else:
            best_bid = 0.0
            best_ask = 0.0
            mid_price = 0.0
        
        return {
            'symbol': self.symbol,
            'timestamp': time.time(),
            'best_bid': best_bid,
            'best_ask': best_ask,
            'mid_price': mid_price,
            'absolute_spread': absolute_spread,
            'percentage_spread': percentage_spread,
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'imbalance': imbalance,
            'bid_levels': len(self.bids),
            'ask_levels': len(self.asks)
        }
    
    def generate_trading_signal(self) -> Dict[str, Any]:
        """
        Generate trading signal based on order book analysis.
        
        Returns:
            Dictionary with trading signal information
        """
        snapshot = self.get_order_book_snapshot()
        
        # Default signal (neutral)
        signal = {
            'symbol': self.symbol,
            'timestamp': time.time(),
            'direction': 0,  # -1 for sell, 0 for neutral, 1 for buy
            'strength': 0.0,  # 0.0 to 1.0
            'reason': "Neutral"
        }
        
        # Check for significant imbalance
        imbalance = snapshot['imbalance']
        if imbalance > 0.3:
            signal['direction'] = 1
            signal['strength'] = min(abs(imbalance), 1.0)
            signal['reason'] = "Strong bid imbalance"
        elif imbalance < -0.3:
            signal['direction'] = -1
            signal['strength'] = min(abs(imbalance), 1.0)
            signal['reason'] = "Strong ask imbalance"
        
        # Check for tight spread
        percentage_spread = snapshot['percentage_spread']
        if percentage_spread < 0.05:
            # Tight spread, increase signal strength
            signal['strength'] = min(signal['strength'] + 0.2, 1.0)
            signal['reason'] += ", tight spread"
        
        return signal
    
    def save_visualization(self, filename: str = None):
        """
        Save current visualization to file.
        
        Args:
            filename: Output filename (default: {symbol}_orderbook_{timestamp}.png)
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"{self.symbol}_orderbook_{timestamp}.png"
        
        # Save depth chart
        if hasattr(self, 'depth_figure'):
            self.depth_figure.savefig(f"depth_{filename}")
            logger.info(f"Saved depth chart to depth_{filename}")
        
        # Save heatmap
        if hasattr(self, 'heatmap_figure'):
            self.heatmap_figure.savefig(f"heatmap_{filename}")
            logger.info(f"Saved heatmap to heatmap_{filename}")
        
        # Save snapshot data
        snapshot = self.get_order_book_snapshot()
        snapshot_df = pd.DataFrame([snapshot])
        snapshot_df.to_csv(f"snapshot_{filename.replace('.png', '.csv')}", index=False)
        logger.info(f"Saved snapshot data to snapshot_{filename.replace('.png', '.csv')}")


class MockOrderBookProvider:
    """
    Mock order book data provider for testing.
    
    Generates realistic order book data for testing visualization
    without connecting to an exchange.
    """
    
    def __init__(self, symbol="XRP-PERP", mid_price: float = 1000.0, 
                 volatility: float = 0.01, depth: int = 20):
        """
        Initialize mock order book provider.
        
        Args:
            symbol: Trading symbol
            mid_price: Initial mid price
            volatility: Price volatility
            depth: Order book depth
        """
        self.symbol = symbol
        self.mid_price = mid_price
        self.volatility = volatility
        self.depth = depth
        
        # Initialize random state
        self.random_state = np.random.RandomState(42)
        
        logger.info(f"Mock order book provider initialized for {symbol}")
    
    def get_order_book(self, symbol: str = None) -> Dict[str, List[List[float]]]:
        """
        Get mock order book data.
        
        Args:
            symbol: Trading symbol (ignored in mock provider)
            
        Returns:
            Dictionary with 'bids' and 'asks' lists
        """
        # Update mid price with random walk
        self.mid_price *= (1 + self.random_state.normal(0, self.volatility))
        
        # Generate bid prices (descending from mid price)
        bid_prices = np.array([
            self.mid_price * (1 - self.random_state.exponential(0.001) * i)
            for i in range(1, self.depth + 1)
        ])
        
        # Generate ask prices (ascending from mid price)
        ask_prices = np.array([
            self.mid_price * (1 + self.random_state.exponential(0.001) * i)
            for i in range(1, self.depth + 1)
        ])
        
        # Generate sizes with more volume near the mid price
        bid_sizes = np.array([
            self.random_state.exponential(10) * (self.depth - i + 1) / self.depth
            for i in range(1, self.depth + 1)
        ])
        
        ask_sizes = np.array([
            self.random_state.exponential(10) * (self.depth - i + 1) / self.depth
            for i in range(1, self.depth + 1)
        ])
        
        # Create order book
        bids = [[float(price), float(size)] for price, size in zip(bid_prices, bid_sizes)]
        asks = [[float(price), float(size)] for price, size in zip(ask_prices, ask_sizes)]
        
        return {
            'bids': bids,
            'asks': asks
        }


# Example usage
if __name__ == "__main__":
    # Create mock data provider
    mock_provider = MockOrderBookProvider(symbol="XRP-PERP", mid_price=0.5)
    
    # Create tkinter window
    root = tk.Tk()
    root.title("Order Book Visualization")
    root.geometry("1200x800")
    
    # Create order book visualizer
    visualizer = OrderBookVisualizer(root, symbol="XRP-PERP", theme="dark")
    
    # Start updates
    visualizer.start_updates(mock_provider)
    
    # Run tkinter main loop
    root.mainloop()
    
    # Stop updates when window is closed
    visualizer.stop_updates()
