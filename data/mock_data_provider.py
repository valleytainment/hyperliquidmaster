"""
Enhanced Mock Data Provider

This module provides a mock data provider for the Hyperliquid trading bot.
It generates realistic synthetic market data for testing and development,
and serves as a fallback during API rate limit periods.

Classes:
    MockDataProvider: Generates and provides synthetic market data
"""

import os
import json
import random
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/mock_data_provider.log")
    ]
)
logger = logging.getLogger(__name__)

class MockDataProvider:
    """
    Enhanced mock data provider for synthetic market data.
    
    This class generates realistic synthetic market data for testing and
    development, and serves as a fallback during API rate limit periods.
    
    Attributes:
        assets (list): List of supported assets
        base_prices (dict): Base prices for each asset
        volatilities (dict): Volatility levels for each asset
        trends (dict): Current trend directions for each asset
        data_dir (str): Directory for storing synthetic data
    """
    
    def __init__(self, data_dir="mock_data"):
        """
        Initialize the mock data provider.
        
        Args:
            data_dir (str): Directory for storing synthetic data
        """
        # Create data directory if it doesn't exist
        self.data_dir = data_dir
        os.makedirs(f"{data_dir}/candles", exist_ok=True)
        os.makedirs(f"{data_dir}/order_book", exist_ok=True)
        os.makedirs(f"{data_dir}/market_data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Define supported assets
        self.assets = ["BTC", "ETH", "SOL", "XRP", "AVAX", "MATIC", "LINK", "DOGE", "ADA", "DOT"]
        
        # Initialize base prices and volatilities
        self.base_prices = {
            "BTC": 60000.0,
            "ETH": 4000.0,
            "SOL": 130.0,
            "XRP": 0.5,
            "AVAX": 35.0,
            "MATIC": 1.2,
            "LINK": 15.0,
            "DOGE": 0.1,
            "ADA": 0.5,
            "DOT": 8.0
        }
        
        self.volatilities = {
            "BTC": 0.02,
            "ETH": 0.025,
            "SOL": 0.04,
            "XRP": 0.035,
            "AVAX": 0.045,
            "MATIC": 0.05,
            "LINK": 0.03,
            "DOGE": 0.06,
            "ADA": 0.035,
            "DOT": 0.04
        }
        
        # Initialize trends (1 for uptrend, 0 for sideways, -1 for downtrend)
        self.trends = {asset: random.choice([-1, 0, 1]) for asset in self.assets}
        
        logger.info(f"Mock data provider initialized with {len(self.assets)} assets")
    
    def get_klines(self, symbol, timeframe, limit=100):
        """
        Get synthetic klines (candlestick data) for the specified symbol and timeframe.
        
        Args:
            symbol (str): Trading pair symbol (e.g., "BTC")
            timeframe (str): Timeframe (e.g., "1m", "5m", "1h", "4h", "1d")
            limit (int): Number of klines to return
        
        Returns:
            list: List of klines (OHLCV data)
        """
        # Check if symbol is supported
        if symbol not in self.assets:
            logger.warning(f"Unsupported symbol: {symbol}")
            return []
        
        # Check if cached data exists
        cache_file = f"{self.data_dir}/candles/{symbol}_{timeframe}.json"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    klines = json.load(f)
                
                # Return the requested number of klines
                return klines[-limit:]
            except Exception as e:
                logger.error(f"Error loading cached klines: {e}")
        
        # Generate synthetic klines
        klines = self._generate_klines(symbol, timeframe, limit)
        
        # Cache the klines
        try:
            with open(cache_file, "w") as f:
                json.dump(klines, f)
        except Exception as e:
            logger.error(f"Error caching klines: {e}")
        
        return klines
    
    def _generate_klines(self, symbol, timeframe, limit):
        """
        Generate synthetic klines for the specified symbol and timeframe.
        
        Args:
            symbol (str): Trading pair symbol (e.g., "BTC")
            timeframe (str): Timeframe (e.g., "1m", "5m", "1h", "4h", "1d")
            limit (int): Number of klines to generate
        
        Returns:
            list: List of synthetic klines
        """
        # Get base price and volatility
        base_price = self.base_prices.get(symbol, 100.0)
        volatility = self.volatilities.get(symbol, 0.03)
        trend = self.trends.get(symbol, 0)
        
        # Convert timeframe to minutes
        timeframe_minutes = self._timeframe_to_minutes(timeframe)
        
        # Generate timestamps
        end_time = datetime.now()
        timestamps = []
        for i in range(limit):
            timestamp = int((end_time - timedelta(minutes=timeframe_minutes * (limit - i - 1))).timestamp())
            timestamps.append(timestamp)
        
        # Generate prices with random walk and trend bias
        prices = []
        current_price = base_price
        
        for i in range(limit):
            # Add trend bias
            trend_factor = 0.002 * trend
            
            # Generate random price movement
            price_change = np.random.normal(trend_factor, volatility)
            current_price *= (1 + price_change)
            
            # Ensure price doesn't go too far from base price
            if current_price < base_price * 0.5:
                current_price = base_price * 0.5
            elif current_price > base_price * 2.0:
                current_price = base_price * 2.0
            
            prices.append(current_price)
        
        # Generate OHLCV data
        klines = []
        for i in range(limit):
            # Generate open, high, low, close prices
            close = prices[i]
            
            # For the first candle, open is close of previous day with small change
            if i == 0:
                open_price = close * (1 + np.random.normal(0, 0.005))
            else:
                open_price = klines[i-1]["close"]
            
            # High and low are derived from open and close
            price_range = abs(close - open_price)
            high = max(open_price, close) + abs(np.random.normal(0, 1)) * price_range
            low = min(open_price, close) - abs(np.random.normal(0, 1)) * price_range
            
            # Generate volume
            volume = base_price * np.random.gamma(2.0, 1000.0)
            
            # Create kline
            kline = {
                "timestamp": timestamps[i],
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume
            }
            
            klines.append(kline)
        
        return klines
    
    def _timeframe_to_minutes(self, timeframe):
        """
        Convert timeframe string to minutes.
        
        Args:
            timeframe (str): Timeframe (e.g., "1m", "5m", "1h", "4h", "1d")
        
        Returns:
            int: Timeframe in minutes
        """
        if timeframe.endswith("m"):
            return int(timeframe[:-1])
        elif timeframe.endswith("h"):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith("d"):
            return int(timeframe[:-1]) * 60 * 24
        else:
            return 60  # Default to 1 hour
    
    def get_order_book(self, symbol, limit=10):
        """
        Get synthetic order book for the specified symbol.
        
        Args:
            symbol (str): Trading pair symbol (e.g., "BTC")
            limit (int): Number of levels to return
        
        Returns:
            dict: Order book with bids and asks
        """
        # Check if symbol is supported
        if symbol not in self.assets:
            logger.warning(f"Unsupported symbol: {symbol}")
            return {"bids": [], "asks": []}
        
        # Get current price
        current_price = self._get_current_price(symbol)
        
        # Generate bids (buy orders)
        bids = []
        for i in range(limit):
            price = current_price * (1 - 0.001 * (i + 1) * np.random.uniform(0.8, 1.2))
            size = np.random.gamma(2.0, current_price / 10.0)
            bids.append([price, size])
        
        # Generate asks (sell orders)
        asks = []
        for i in range(limit):
            price = current_price * (1 + 0.001 * (i + 1) * np.random.uniform(0.8, 1.2))
            size = np.random.gamma(2.0, current_price / 10.0)
            asks.append([price, size])
        
        # Sort bids and asks
        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])
        
        return {
            "bids": bids,
            "asks": asks
        }
    
    def get_market_data(self, symbol):
        """
        Get synthetic market data for the specified symbol.
        
        Args:
            symbol (str): Trading pair symbol (e.g., "BTC")
        
        Returns:
            dict: Market data including price and funding rate
        """
        # Check if symbol is supported
        if symbol not in self.assets:
            logger.warning(f"Unsupported symbol: {symbol}")
            return {"price": 0.0, "funding_rate": 0.0}
        
        # Get current price
        current_price = self._get_current_price(symbol)
        
        # Generate funding rate (typically between -0.01% and 0.01%)
        funding_rate = np.random.normal(0, 0.0001)
        
        return {
            "price": current_price,
            "funding_rate": funding_rate
        }
    
    def _get_current_price(self, symbol):
        """
        Get current synthetic price for the specified symbol.
        
        Args:
            symbol (str): Trading pair symbol (e.g., "BTC")
        
        Returns:
            float: Current synthetic price
        """
        # Get base price and volatility
        base_price = self.base_prices.get(symbol, 100.0)
        volatility = self.volatilities.get(symbol, 0.03)
        
        # Generate random price
        price = base_price * (1 + np.random.normal(0, volatility))
        
        return price
    
    def get_trades(self, symbol, limit=50):
        """
        Get synthetic trades for the specified symbol.
        
        Args:
            symbol (str): Trading pair symbol (e.g., "BTC")
            limit (int): Number of trades to return
        
        Returns:
            list: List of synthetic trades
        """
        # Check if symbol is supported
        if symbol not in self.assets:
            logger.warning(f"Unsupported symbol: {symbol}")
            return []
        
        # Get current price
        current_price = self._get_current_price(symbol)
        
        # Generate trades
        trades = []
        end_time = datetime.now()
        
        for i in range(limit):
            # Generate timestamp
            timestamp = int((end_time - timedelta(seconds=i * np.random.uniform(1, 10))).timestamp())
            
            # Generate price
            price = current_price * (1 + np.random.normal(0, 0.001))
            
            # Generate size
            size = np.random.gamma(2.0, current_price / 100.0)
            
            # Generate side
            side = "buy" if np.random.random() > 0.5 else "sell"
            
            # Create trade
            trade = {
                "timestamp": timestamp,
                "price": price,
                "size": size,
                "side": side
            }
            
            trades.append(trade)
        
        return trades
    
    def update_trend(self, symbol, trend):
        """
        Update the trend for the specified symbol.
        
        Args:
            symbol (str): Trading pair symbol (e.g., "BTC")
            trend (int): Trend direction (1 for uptrend, 0 for sideways, -1 for downtrend)
        """
        if symbol in self.assets:
            self.trends[symbol] = trend
            logger.info(f"Updated trend for {symbol} to {trend}")
    
    def generate_all_timeframes(self, symbol, days=30):
        """
        Generate synthetic data for all timeframes for the specified symbol.
        
        Args:
            symbol (str): Trading pair symbol (e.g., "BTC")
            days (int): Number of days of data to generate
        
        Returns:
            dict: Dictionary of timeframes and their klines
        """
        # Check if symbol is supported
        if symbol not in self.assets:
            logger.warning(f"Unsupported symbol: {symbol}")
            return {}
        
        # Define timeframes
        timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        
        # Calculate limits for each timeframe
        limits = {
            "1m": days * 24 * 60,
            "5m": days * 24 * 12,
            "15m": days * 24 * 4,
            "1h": days * 24,
            "4h": days * 6,
            "1d": days
        }
        
        # Generate klines for each timeframe
        result = {}
        for tf in timeframes:
            logger.info(f"Generating {tf} klines for {symbol}")
            klines = self.get_klines(symbol, tf, limit=limits[tf])
            result[tf] = klines
            
            # Save to file
            cache_file = f"{self.data_dir}/candles/{symbol}_{tf}_{days}d.csv"
            try:
                df = pd.DataFrame(klines)
                df.to_csv(cache_file, index=False)
                logger.info(f"Saved {len(klines)} {tf} klines for {symbol} to {cache_file}")
            except Exception as e:
                logger.error(f"Error saving klines to CSV: {e}")
        
        return result
