"""
Mock Data Provider for Hyperliquid Trading Bot

This module provides mock market data for development and testing when
API rate limits are encountered. It generates realistic price movements
and order book dynamics based on historical patterns.
"""

import os
import json
import time
import random
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class MockDataProvider:
    """
    Provides mock market data for development and testing.
    """
    
    def __init__(self, data_dir: str = "mock_data", seed: Optional[int] = None):
        """
        Initialize the mock data provider.
        
        Args:
            data_dir: Directory to store mock data
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize base prices for common assets
        self.base_prices = {
            "BTC": 65000.0,
            "ETH": 3500.0,
            "SOL": 150.0,
            "AVAX": 35.0,
            "MATIC": 1.2,
            "LINK": 18.0,
            "DOGE": 0.15,
            "SHIB": 0.00002,
            "ADA": 0.55,
            "DOT": 8.5,
            "UNI": 10.0,
            "AAVE": 95.0,
            "SNX": 3.2,
            "CRV": 0.65,
            "MKR": 2200.0,
            "COMP": 65.0,
            "YFI": 12000.0,
            "SUSHI": 1.5,
            "1INCH": 0.85,
            "BAL": 6.5,
            "LDO": 2.8,
            "APE": 1.7,
            "OP": 3.2,
            "ARB": 1.4,
            "GMX": 45.0,
            "PERP": 1.2,
            "DYDX": 3.5,
            "INJ": 28.0,
            "NEAR": 5.8,
            "FTM": 0.75
        }
        
        # Initialize price history
        self.price_history = {}
        for symbol, price in self.base_prices.items():
            self.price_history[symbol] = self._generate_price_history(symbol, price)
        
        # Initialize order books
        self.order_books = {}
        for symbol in self.base_prices.keys():
            self.order_books[symbol] = self._generate_order_book(symbol)
        
        # Initialize funding rates
        self.funding_rates = {}
        for symbol in self.base_prices.keys():
            self.funding_rates[symbol] = self._generate_funding_rate(symbol)
        
        # Save initial data
        self._save_data()
        
        logger.info(f"Mock data provider initialized with {len(self.base_prices)} assets")
    
    def _save_data(self):
        """
        Save mock data to disk.
        """
        # Save price history
        with open(os.path.join(self.data_dir, "price_history.json"), "w") as f:
            json.dump(self.price_history, f)
        
        # Save order books
        with open(os.path.join(self.data_dir, "order_books.json"), "w") as f:
            json.dump(self.order_books, f)
        
        # Save funding rates
        with open(os.path.join(self.data_dir, "funding_rates.json"), "w") as f:
            json.dump(self.funding_rates, f)
    
    def _load_data(self):
        """
        Load mock data from disk.
        """
        # Load price history
        try:
            with open(os.path.join(self.data_dir, "price_history.json"), "r") as f:
                self.price_history = json.load(f)
        except FileNotFoundError:
            logger.warning("Price history file not found, using generated data")
        
        # Load order books
        try:
            with open(os.path.join(self.data_dir, "order_books.json"), "r") as f:
                self.order_books = json.load(f)
        except FileNotFoundError:
            logger.warning("Order books file not found, using generated data")
        
        # Load funding rates
        try:
            with open(os.path.join(self.data_dir, "funding_rates.json"), "r") as f:
                self.funding_rates = json.load(f)
        except FileNotFoundError:
            logger.warning("Funding rates file not found, using generated data")
    
    def _generate_price_history(self, symbol: str, base_price: float) -> List[Dict[str, Any]]:
        """
        Generate realistic price history for an asset.
        
        Args:
            symbol: Asset symbol
            base_price: Base price to start from
            
        Returns:
            List of price candles
        """
        # Parameters for price generation
        volatility = 0.02  # Daily volatility
        if symbol in ["BTC", "ETH"]:
            volatility = 0.015  # Lower volatility for major assets
        elif symbol in ["SHIB", "DOGE"]:
            volatility = 0.04  # Higher volatility for meme coins
        
        # Generate daily returns with slight upward bias
        daily_returns = np.random.normal(0.0002, volatility, 90)  # 90 days of history
        
        # Calculate price path
        prices = [base_price]
        for ret in daily_returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate candles for different timeframes
        candles = []
        
        # Start from 90 days ago
        start_time = int((datetime.now() - timedelta(days=90)).timestamp() * 1000)
        
        # Generate 1-day candles
        for i in range(len(prices) - 1):
            day_volatility = volatility / 2
            open_price = prices[i]
            close_price = prices[i + 1]
            high_price = max(open_price, close_price) * (1 + random.uniform(0, day_volatility))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, day_volatility))
            volume = base_price * random.uniform(1000, 10000) / 1000  # Volume proportional to price
            
            candle = {
                "timestamp": start_time + i * 86400000,  # 1 day in milliseconds
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume
            }
            candles.append(candle)
        
        return candles
    
    def _generate_order_book(self, symbol: str) -> Dict[str, Any]:
        """
        Generate realistic order book for an asset.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Order book with bids and asks
        """
        # Get current price
        current_price = self.price_history[symbol][-1]["close"]
        
        # Parameters for order book generation
        spread_percent = 0.001  # 0.1% spread
        if symbol in ["BTC", "ETH"]:
            spread_percent = 0.0005  # Tighter spread for major assets
        elif symbol in ["SHIB", "DOGE"]:
            spread_percent = 0.002  # Wider spread for meme coins
        
        depth = 20  # Number of levels
        
        # Calculate bid and ask prices
        mid_price = current_price
        spread = mid_price * spread_percent
        best_bid = mid_price - spread / 2
        best_ask = mid_price + spread / 2
        
        # Generate bids (buy orders)
        bids = []
        for i in range(depth):
            price = best_bid * (1 - i * 0.001)
            size = random.uniform(0.1, 10.0) * current_price / 1000  # Size proportional to price
            bids.append({"price": price, "quantity": size})
        
        # Generate asks (sell orders)
        asks = []
        for i in range(depth):
            price = best_ask * (1 + i * 0.001)
            size = random.uniform(0.1, 10.0) * current_price / 1000  # Size proportional to price
            asks.append({"price": price, "quantity": size})
        
        return {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "timestamp": int(time.time() * 1000)
        }
    
    def _generate_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Generate realistic funding rate for an asset.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Funding rate data
        """
        # Parameters for funding rate generation
        base_rate = 0.0001  # 0.01% per 8 hours
        if symbol in ["BTC", "ETH"]:
            base_rate = 0.00005  # Lower funding rate for major assets
        elif symbol in ["SHIB", "DOGE"]:
            base_rate = 0.0002  # Higher funding rate for meme coins
        
        # Add some randomness
        funding_rate = base_rate * random.uniform(0.5, 1.5)
        
        # Randomly flip sign (positive or negative)
        if random.random() < 0.5:
            funding_rate = -funding_rate
        
        # Calculate next funding time (every 8 hours)
        current_time = int(time.time())
        hours_since_midnight = datetime.fromtimestamp(current_time).hour
        next_funding_hour = ((hours_since_midnight // 8) + 1) * 8
        if next_funding_hour >= 24:
            next_funding_hour = 0
        
        next_funding_time = datetime.fromtimestamp(current_time).replace(
            hour=next_funding_hour, minute=0, second=0, microsecond=0
        )
        
        if next_funding_time < datetime.fromtimestamp(current_time):
            next_funding_time += timedelta(days=1)
        
        return {
            "symbol": symbol,
            "funding_rate": funding_rate,
            "next_funding_time": int(next_funding_time.timestamp() * 1000),
            "timestamp": int(time.time() * 1000)
        }
    
    def update_data(self):
        """
        Update mock data with realistic price movements.
        """
        current_time = int(time.time() * 1000)
        
        # Update price history
        for symbol in self.price_history.keys():
            last_candle = self.price_history[symbol][-1]
            last_price = last_candle["close"]
            
            # Parameters for price update
            volatility = 0.005  # 5-minute volatility
            if symbol in ["BTC", "ETH"]:
                volatility = 0.003  # Lower volatility for major assets
            elif symbol in ["SHIB", "DOGE"]:
                volatility = 0.01  # Higher volatility for meme coins
            
            # Generate new price with random walk
            price_change = last_price * random.normalvariate(0, volatility)
            new_price = last_price + price_change
            
            # Ensure price doesn't go negative
            new_price = max(new_price, last_price * 0.1)
            
            # Add some mean reversion to base price
            base_price = self.base_prices[symbol]
            mean_reversion = 0.001 * (base_price - new_price) / base_price
            new_price = new_price * (1 + mean_reversion)
            
            # Create new candle
            new_candle = {
                "timestamp": current_time,
                "open": last_price,
                "high": max(last_price, new_price) * (1 + random.uniform(0, volatility / 2)),
                "low": min(last_price, new_price) * (1 - random.uniform(0, volatility / 2)),
                "close": new_price,
                "volume": base_price * random.uniform(10, 100) / 1000  # Volume proportional to price
            }
            
            # Add new candle
            self.price_history[symbol].append(new_candle)
            
            # Limit history to 10000 candles
            if len(self.price_history[symbol]) > 10000:
                self.price_history[symbol] = self.price_history[symbol][-10000:]
        
        # Update order books
        for symbol in self.order_books.keys():
            self.order_books[symbol] = self._generate_order_book(symbol)
        
        # Update funding rates (less frequently)
        if random.random() < 0.01:  # 1% chance of updating funding rates
            for symbol in self.funding_rates.keys():
                self.funding_rates[symbol] = self._generate_funding_rate(symbol)
        
        # Save updated data
        self._save_data()
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get mock market data for an asset.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Market data
        """
        # Update data to ensure it's fresh
        self.update_data()
        
        # Check if symbol exists
        if symbol not in self.price_history:
            # Return error for unknown symbols
            return {"error": f"Symbol {symbol} not found"}
        
        # Get latest price
        latest_candle = self.price_history[symbol][-1]
        current_price = latest_candle["close"]
        
        # Get funding rate
        funding_rate = self.funding_rates[symbol]["funding_rate"]
        
        # Calculate 24h volume
        volume_24h = sum(candle["volume"] for candle in self.price_history[symbol][-288:])  # 288 5-minute candles in 24h
        
        # Calculate open interest (roughly 10x 24h volume)
        open_interest = volume_24h * 10 * random.uniform(0.8, 1.2)
        
        return {
            "symbol": symbol,
            "price": current_price,
            "index_price": current_price * (1 + random.uniform(-0.0005, 0.0005)),  # Slight variation from mark price
            "mark_price": current_price,
            "open_interest": open_interest,
            "funding_rate": funding_rate,
            "volume_24h": volume_24h,
            "timestamp": int(time.time() * 1000)
        }
    
    def get_order_book(self, symbol: str, depth: int = 10) -> Dict[str, Any]:
        """
        Get mock order book for an asset.
        
        Args:
            symbol: Asset symbol
            depth: Order book depth
            
        Returns:
            Order book
        """
        # Update data to ensure it's fresh
        self.update_data()
        
        # Check if symbol exists
        if symbol not in self.order_books:
            # Return error for unknown symbols
            return {"error": f"Symbol {symbol} not found"}
        
        # Get order book
        order_book = self.order_books[symbol]
        
        # Limit depth
        limited_order_book = {
            "symbol": symbol,
            "bids": order_book["bids"][:depth],
            "asks": order_book["asks"][:depth],
            "timestamp": int(time.time() * 1000)
        }
        
        return limited_order_book
    
    def get_historical_data(self, symbol: str, timeframe: str = "1m", limit: int = 100) -> Dict[str, Any]:
        """
        Get mock historical data for an asset.
        
        Args:
            symbol: Asset symbol
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles
            
        Returns:
            Historical data
        """
        # Update data to ensure it's fresh
        self.update_data()
        
        # Check if symbol exists
        if symbol not in self.price_history:
            # Return error for unknown symbols
            return {"error": f"Symbol {symbol} not found"}
        
        # Get raw candles
        raw_candles = self.price_history[symbol]
        
        # Determine aggregation factor based on timeframe
        aggregation_factor = 1
        if timeframe == "5m":
            aggregation_factor = 5
        elif timeframe == "15m":
            aggregation_factor = 15
        elif timeframe == "1h":
            aggregation_factor = 60
        elif timeframe == "4h":
            aggregation_factor = 240
        elif timeframe == "1d":
            aggregation_factor = 1440
        
        # Aggregate candles if needed
        if aggregation_factor > 1:
            aggregated_candles = []
            for i in range(0, len(raw_candles), aggregation_factor):
                chunk = raw_candles[i:i+aggregation_factor]
                if not chunk:
                    continue
                
                aggregated_candle = {
                    "timestamp": chunk[0]["timestamp"],
                    "open": chunk[0]["open"],
                    "high": max(c["high"] for c in chunk),
                    "low": min(c["low"] for c in chunk),
                    "close": chunk[-1]["close"],
                    "volume": sum(c["volume"] for c in chunk)
                }
                aggregated_candles.append(aggregated_candle)
            
            candles = aggregated_candles
        else:
            candles = raw_candles
        
        # Limit number of candles
        limited_candles = candles[-limit:]
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": limited_candles
        }
    
    def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Get mock funding rate for an asset.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Funding rate
        """
        # Check if symbol exists
        if symbol not in self.funding_rates:
            # Return error for unknown symbols
            return {"error": f"Symbol {symbol} not found"}
        
        return self.funding_rates[symbol]
    
    def get_all_markets(self) -> Dict[str, Any]:
        """
        Get all available markets.
        
        Returns:
            All markets
        """
        # Update data to ensure it's fresh
        self.update_data()
        
        markets = []
        for symbol in self.price_history.keys():
            latest_candle = self.price_history[symbol][-1]
            current_price = latest_candle["close"]
            
            funding_rate = self.funding_rates[symbol]["funding_rate"]
            
            # Calculate 24h volume
            volume_24h = sum(candle["volume"] for candle in self.price_history[symbol][-288:])  # 288 5-minute candles in 24h
            
            # Calculate open interest (roughly 10x 24h volume)
            open_interest = volume_24h * 10 * random.uniform(0.8, 1.2)
            
            markets.append({
                "symbol": symbol,
                "price": current_price,
                "index_price": current_price * (1 + random.uniform(-0.0005, 0.0005)),
                "mark_price": current_price,
                "open_interest": open_interest,
                "funding_rate": funding_rate,
                "volume_24h": volume_24h
            })
        
        return {
            "markets": markets,
            "timestamp": int(time.time() * 1000)
        }
