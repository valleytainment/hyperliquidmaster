#!/usr/bin/env python3
"""
Enhanced Mock Data Provider for Hyperliquid Trading Bot

This module provides realistic mock market data for development and testing
when the Hyperliquid API is rate limited or unavailable. It generates synthetic
price movements, order book dynamics, and other market data based on historical
patterns and statistical models.
"""

import os
import time
import json
import random
import logging
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union

# Configure logging
logger = logging.getLogger("MockDataProvider")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(console_handler)

class EnhancedMockDataProvider:
    """
    Enhanced Mock Data Provider for Hyperliquid Trading Bot.
    
    This class provides realistic mock market data for development and testing
    when the Hyperliquid API is rate limited or unavailable.
    """
    
    def __init__(self, data_dir: str = "mock_data", cache_dir: str = "cache"):
        """
        Initialize the mock data provider.
        
        Args:
            data_dir: Directory for storing mock data
            cache_dir: Directory for caching real API responses
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        
        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "market_data"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "order_book"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "candles"), exist_ok=True)
        
        # Initialize price models
        self.price_models = {}
        self.last_prices = {}
        self.volatility = {}
        self.trend_direction = {}
        self.trend_strength = {}
        
        # Default parameters for common cryptocurrencies
        self.default_params = {
            "BTC": {
                "base_price": 60000.0,
                "volatility": 0.02,
                "trend_strength": 0.3,
                "mean_reversion": 0.05,
                "volume_base": 1000.0,
                "volume_volatility": 0.5,
                "book_depth": 20,
                "spread_percent": 0.001
            },
            "ETH": {
                "base_price": 3000.0,
                "volatility": 0.025,
                "trend_strength": 0.25,
                "mean_reversion": 0.06,
                "volume_base": 5000.0,
                "volume_volatility": 0.6,
                "book_depth": 15,
                "spread_percent": 0.002
            },
            "SOL": {
                "base_price": 100.0,
                "volatility": 0.035,
                "trend_strength": 0.4,
                "mean_reversion": 0.08,
                "volume_base": 20000.0,
                "volume_volatility": 0.7,
                "book_depth": 10,
                "spread_percent": 0.003
            },
            "XRP": {
                "base_price": 0.5,
                "volatility": 0.03,
                "trend_strength": 0.35,
                "mean_reversion": 0.07,
                "volume_base": 50000.0,
                "volume_volatility": 0.65,
                "book_depth": 12,
                "spread_percent": 0.004
            }
        }
        
        # Initialize models for default symbols
        for symbol in self.default_params:
            self._init_price_model(symbol)
            
        logger.info("Enhanced Mock Data Provider initialized")
        
    def _init_price_model(self, symbol: str) -> None:
        """
        Initialize price model for a symbol.
        
        Args:
            symbol: Symbol to initialize
        """
        # Get parameters for symbol
        params = self.default_params.get(symbol, self.default_params["BTC"])
        
        # Initialize price model
        self.price_models[symbol] = {
            "base_price": params["base_price"],
            "volatility": params["volatility"],
            "trend_strength": params["trend_strength"],
            "mean_reversion": params["mean_reversion"],
            "volume_base": params["volume_base"],
            "volume_volatility": params["volume_volatility"],
            "book_depth": params["book_depth"],
            "spread_percent": params["spread_percent"]
        }
        
        # Initialize last price
        self.last_prices[symbol] = params["base_price"]
        
        # Initialize volatility
        self.volatility[symbol] = params["volatility"]
        
        # Initialize trend
        self.trend_direction[symbol] = random.choice([-1, 1])
        self.trend_strength[symbol] = params["trend_strength"]
        
        logger.debug(f"Initialized price model for {symbol}")
        
    def _generate_next_price(self, symbol: str) -> float:
        """
        Generate next price for a symbol.
        
        Args:
            symbol: Symbol to generate price for
            
        Returns:
            Next price
        """
        # Get parameters for symbol
        if symbol not in self.price_models:
            self._init_price_model(symbol)
            
        params = self.price_models[symbol]
        last_price = self.last_prices[symbol]
        
        # Random walk with drift and mean reversion
        random_component = np.random.normal(0, params["volatility"])
        trend_component = self.trend_direction[symbol] * self.trend_strength[symbol] * params["volatility"]
        mean_reversion = params["mean_reversion"] * (params["base_price"] - last_price) / params["base_price"]
        
        # Calculate price change
        price_change = last_price * (random_component + trend_component + mean_reversion)
        
        # Calculate new price
        new_price = max(0.00001, last_price + price_change)
        
        # Update last price
        self.last_prices[symbol] = new_price
        
        # Randomly change trend direction with small probability
        if random.random() < 0.05:
            self.trend_direction[symbol] *= -1
            
        # Randomly change trend strength with small probability
        if random.random() < 0.05:
            self.trend_strength[symbol] = max(0.1, min(0.5, self.trend_strength[symbol] + np.random.normal(0, 0.05)))
            
        # Randomly change volatility with small probability
        if random.random() < 0.05:
            self.volatility[symbol] = max(0.005, min(0.05, self.volatility[symbol] + np.random.normal(0, 0.005)))
            
        return new_price
        
    def _generate_volume(self, symbol: str) -> float:
        """
        Generate volume for a symbol.
        
        Args:
            symbol: Symbol to generate volume for
            
        Returns:
            Volume
        """
        # Get parameters for symbol
        if symbol not in self.price_models:
            self._init_price_model(symbol)
            
        params = self.price_models[symbol]
        
        # Generate volume with log-normal distribution
        volume = np.random.lognormal(np.log(params["volume_base"]), params["volume_volatility"])
        
        return volume
        
    def _generate_order_book(self, symbol: str, price: float) -> Dict:
        """
        Generate order book for a symbol.
        
        Args:
            symbol: Symbol to generate order book for
            price: Current price
            
        Returns:
            Order book dictionary
        """
        # Get parameters for symbol
        if symbol not in self.price_models:
            self._init_price_model(symbol)
            
        params = self.price_models[symbol]
        
        # Calculate bid and ask prices
        spread = price * params["spread_percent"]
        bid_price = price - spread / 2
        ask_price = price + spread / 2
        
        # Generate bids
        bids = []
        for i in range(params["book_depth"]):
            bid_price_level = bid_price * (1 - i * 0.001 * (1 + np.random.normal(0, 0.2)))
            bid_size = np.random.lognormal(np.log(params["volume_base"] / 100), 0.5)
            bids.append([bid_price_level, bid_size])
            
        # Generate asks
        asks = []
        for i in range(params["book_depth"]):
            ask_price_level = ask_price * (1 + i * 0.001 * (1 + np.random.normal(0, 0.2)))
            ask_size = np.random.lognormal(np.log(params["volume_base"] / 100), 0.5)
            asks.append([ask_price_level, ask_size])
            
        # Create order book
        order_book = {
            "bids": bids,
            "asks": asks,
            "timestamp": int(time.time() * 1000)
        }
        
        return order_book
        
    def _generate_candle(self, symbol: str, timeframe: str, start_time: int) -> Dict:
        """
        Generate candle for a symbol.
        
        Args:
            symbol: Symbol to generate candle for
            timeframe: Timeframe (e.g., "1m", "5m", "1h")
            start_time: Start time in milliseconds
            
        Returns:
            Candle dictionary
        """
        # Get parameters for symbol
        if symbol not in self.price_models:
            self._init_price_model(symbol)
            
        # Generate price
        open_price = self._generate_next_price(symbol)
        high_price = open_price * (1 + np.random.uniform(0, self.volatility[symbol] * 2))
        low_price = open_price * (1 - np.random.uniform(0, self.volatility[symbol] * 2))
        close_price = self._generate_next_price(symbol)
        
        # Ensure high >= open, close and low <= open, close
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Generate volume
        volume = self._generate_volume(symbol)
        
        # Create candle
        candle = {
            "timestamp": start_time,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume
        }
        
        return candle
        
    def _get_timeframe_milliseconds(self, timeframe: str) -> int:
        """
        Get timeframe in milliseconds.
        
        Args:
            timeframe: Timeframe (e.g., "1m", "5m", "1h")
            
        Returns:
            Timeframe in milliseconds
        """
        # Parse timeframe
        if timeframe.endswith("m"):
            return int(timeframe[:-1]) * 60 * 1000
        elif timeframe.endswith("h"):
            return int(timeframe[:-1]) * 60 * 60 * 1000
        elif timeframe.endswith("d"):
            return int(timeframe[:-1]) * 24 * 60 * 60 * 1000
        else:
            return 60 * 1000  # Default to 1m
            
    def get_market_data(self, symbol: str) -> Dict:
        """
        Get market data for a symbol.
        
        Args:
            symbol: Symbol to get market data for
            
        Returns:
            Market data dictionary
        """
        # Check if symbol is supported
        if symbol not in self.price_models:
            self._init_price_model(symbol)
            
        # Generate price
        price = self._generate_next_price(symbol)
        
        # Generate volume
        volume_24h = self._generate_volume(symbol) * 24
        
        # Generate change
        change_24h = np.random.normal(0, self.volatility[symbol] * 5)
        
        # Create market data
        market_data = {
            "symbol": symbol,
            "last_price": price,
            "volume_24h": volume_24h,
            "change_24h": change_24h,
            "timestamp": int(time.time() * 1000)
        }
        
        # Save to cache
        self._save_to_cache("market_data", symbol, market_data)
        
        return market_data
        
    def get_order_book(self, symbol: str) -> Dict:
        """
        Get order book for a symbol.
        
        Args:
            symbol: Symbol to get order book for
            
        Returns:
            Order book dictionary
        """
        # Check if symbol is supported
        if symbol not in self.price_models:
            self._init_price_model(symbol)
            
        # Get current price
        price = self.last_prices[symbol]
        
        # Generate order book
        order_book = self._generate_order_book(symbol, price)
        
        # Save to cache
        self._save_to_cache("order_book", symbol, order_book)
        
        return order_book
        
    def get_candles(self, symbol: str, timeframe: str, limit: int = 100) -> List[Dict]:
        """
        Get candles for a symbol.
        
        Args:
            symbol: Symbol to get candles for
            timeframe: Timeframe (e.g., "1m", "5m", "1h")
            limit: Number of candles to return
            
        Returns:
            List of candle dictionaries
        """
        # Check if symbol is supported
        if symbol not in self.price_models:
            self._init_price_model(symbol)
            
        # Get timeframe in milliseconds
        timeframe_ms = self._get_timeframe_milliseconds(timeframe)
        
        # Generate candles
        candles = []
        current_time = int(time.time() * 1000)
        
        for i in range(limit):
            start_time = current_time - (limit - i) * timeframe_ms
            candle = self._generate_candle(symbol, timeframe, start_time)
            candles.append(candle)
            
        # Save to cache
        self._save_to_cache("candles", f"{symbol}_{timeframe}", candles)
        
        return candles
    
    def get_klines(self, symbol: str, timeframe: str, limit: int = 100) -> List[Dict]:
        """
        Get klines (candlestick data) for a symbol.
        This is an alias for get_candles to maintain compatibility with test suite,
        but ensures timestamps are in seconds instead of milliseconds.
        
        Args:
            symbol: Symbol to get klines for
            timeframe: Timeframe (e.g., "1m", "5m", "1h")
            limit: Number of klines to return
            
        Returns:
            List of kline dictionaries with timestamps in seconds
        """
        logger.info(f"Getting klines for {symbol} on {timeframe} timeframe (limit: {limit})")
        
        # Get candles with millisecond timestamps
        candles = self.get_candles(symbol, timeframe, limit)
        
        # Convert timestamps from milliseconds to seconds for test compatibility
        for candle in candles:
            candle["timestamp"] = int(candle["timestamp"] / 1000)
        
        return candles
        
    def get_historical_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical data for a symbol.
        
        Args:
            symbol: Symbol to get historical data for
            timeframe: Timeframe (e.g., "1m", "5m", "1h")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with historical data
        """
        # Check if symbol is supported
        if symbol not in self.price_models:
            self._init_price_model(symbol)
            
        # Parse dates
        start_timestamp = int(datetime.datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_timestamp = int(datetime.datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        
        # Get timeframe in milliseconds
        timeframe_ms = self._get_timeframe_milliseconds(timeframe)
        
        # Calculate number of candles
        num_candles = (end_timestamp - start_timestamp) // timeframe_ms + 1
        
        # Generate candles
        candles = []
        for i in range(num_candles):
            start_time = start_timestamp + i * timeframe_ms
            candle = self._generate_candle(symbol, timeframe, start_time)
            candles.append(candle)
            
        # Create DataFrame
        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("datetime", inplace=True)
        
        # Save to cache
        self._save_to_cache("historical_data", f"{symbol}_{timeframe}_{start_date}_{end_date}", df.to_dict("records"))
        
        return df
        
    def get_positions(self, symbol: str = None) -> List[Dict]:
        """
        Get positions for a symbol.
        
        Args:
            symbol: Symbol to get positions for (if None, get all positions)
            
        Returns:
            List of position dictionaries
        """
        # Generate mock positions
        positions = []
        
        # If symbol is specified, generate position for that symbol
        if symbol:
            if symbol not in self.price_models:
                self._init_price_model(symbol)
                
            # Generate random position
            if random.random() < 0.5:
                # Long position
                size = np.random.lognormal(np.log(1), 0.5)
                entry_price = self.last_prices[symbol] * (1 - np.random.uniform(0, 0.05))
                liquidation_price = entry_price * 0.8
                
                positions.append({
                    "symbol": symbol,
                    "size": size,
                    "entry_price": entry_price,
                    "mark_price": self.last_prices[symbol],
                    "liquidation_price": liquidation_price,
                    "unrealized_pnl": (self.last_prices[symbol] - entry_price) * size,
                    "realized_pnl": 0,
                    "side": "long",
                    "timestamp": int(time.time() * 1000)
                })
            else:
                # Short position
                size = -np.random.lognormal(np.log(1), 0.5)
                entry_price = self.last_prices[symbol] * (1 + np.random.uniform(0, 0.05))
                liquidation_price = entry_price * 1.2
                
                positions.append({
                    "symbol": symbol,
                    "size": size,
                    "entry_price": entry_price,
                    "mark_price": self.last_prices[symbol],
                    "liquidation_price": liquidation_price,
                    "unrealized_pnl": (entry_price - self.last_prices[symbol]) * abs(size),
                    "realized_pnl": 0,
                    "side": "short",
                    "timestamp": int(time.time() * 1000)
                })
        else:
            # Generate positions for all symbols
            for symbol in self.price_models:
                # Generate random position with 50% probability
                if random.random() < 0.5:
                    # Long position
                    size = np.random.lognormal(np.log(1), 0.5)
                    entry_price = self.last_prices[symbol] * (1 - np.random.uniform(0, 0.05))
                    liquidation_price = entry_price * 0.8
                    
                    positions.append({
                        "symbol": symbol,
                        "size": size,
                        "entry_price": entry_price,
                        "mark_price": self.last_prices[symbol],
                        "liquidation_price": liquidation_price,
                        "unrealized_pnl": (self.last_prices[symbol] - entry_price) * size,
                        "realized_pnl": 0,
                        "side": "long",
                        "timestamp": int(time.time() * 1000)
                    })
                elif random.random() < 0.5:
                    # Short position
                    size = -np.random.lognormal(np.log(1), 0.5)
                    entry_price = self.last_prices[symbol] * (1 + np.random.uniform(0, 0.05))
                    liquidation_price = entry_price * 1.2
                    
                    positions.append({
                        "symbol": symbol,
                        "size": size,
                        "entry_price": entry_price,
                        "mark_price": self.last_prices[symbol],
                        "liquidation_price": liquidation_price,
                        "unrealized_pnl": (entry_price - self.last_prices[symbol]) * abs(size),
                        "realized_pnl": 0,
                        "side": "short",
                        "timestamp": int(time.time() * 1000)
                    })
                    
        return positions
        
    def get_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """
        Get trades for a symbol.
        
        Args:
            symbol: Symbol to get trades for
            limit: Number of trades to return
            
        Returns:
            List of trade dictionaries
        """
        # Check if symbol is supported
        if symbol not in self.price_models:
            self._init_price_model(symbol)
            
        # Generate trades
        trades = []
        current_time = int(time.time() * 1000)
        
        for i in range(limit):
            # Generate random trade
            price = self.last_prices[symbol] * (1 + np.random.normal(0, 0.001))
            size = np.random.lognormal(np.log(0.1), 0.5)
            side = random.choice(["buy", "sell"])
            
            trades.append({
                "id": current_time - i * 1000 + random.randint(0, 999),
                "price": price,
                "size": size,
                "side": side,
                "timestamp": current_time - i * 1000
            })
            
        return trades
        
    def get_funding_rate(self, symbol: str) -> Dict:
        """
        Get funding rate for a symbol.
        
        Args:
            symbol: Symbol to get funding rate for
            
        Returns:
            Funding rate dictionary
        """
        # Check if symbol is supported
        if symbol not in self.price_models:
            self._init_price_model(symbol)
            
        # Generate funding rate
        funding_rate = np.random.normal(0, 0.0001)
        
        # Create funding rate dictionary
        funding_rate_dict = {
            "symbol": symbol,
            "funding_rate": funding_rate,
            "next_funding_time": int(time.time() * 1000) + 8 * 60 * 60 * 1000,
            "timestamp": int(time.time() * 1000)
        }
        
        return funding_rate_dict
        
    def get_ticker(self, symbol: str) -> Dict:
        """
        Get ticker for a symbol.
        
        Args:
            symbol: Symbol to get ticker for
            
        Returns:
            Ticker dictionary
        """
        # Check if symbol is supported
        if symbol not in self.price_models:
            self._init_price_model(symbol)
            
        # Generate price
        price = self._generate_next_price(symbol)
        
        # Generate volume
        volume_24h = self._generate_volume(symbol) * 24
        
        # Generate change
        change_24h = np.random.normal(0, self.volatility[symbol] * 5)
        
        # Create ticker
        ticker = {
            "symbol": symbol,
            "price": price,
            "volume": volume_24h,
            "change": change_24h,
            "timestamp": int(time.time() * 1000)
        }
        
        return ticker
        
    def get_mock_data(self, endpoint: str, params: Dict) -> Any:
        """
        Get mock data for an endpoint.
        
        Args:
            endpoint: API endpoint
            params: Parameters for the endpoint
            
        Returns:
            Mock data
        """
        # Get symbol from params
        symbol = params.get("symbol", "BTC")
        
        # Get mock data based on endpoint
        if endpoint == "klines":
            return self.get_klines(
                symbol,
                params.get("interval", "1h"),
                params.get("limit", 100)
            )
        elif endpoint == "ticker":
            return self.get_ticker(symbol)
        elif endpoint == "orderbook":
            return self.get_order_book(symbol)
        elif endpoint == "trades":
            return self.get_trades(
                symbol,
                params.get("limit", 100)
            )
        elif endpoint == "positions":
            return self.get_positions(symbol)
        elif endpoint == "funding_rate":
            return self.get_funding_rate(symbol)
        else:
            # Default to empty response
            return {}
            
    def get_mock_klines(self, symbol: str, interval: str, limit: int = 100) -> List[Dict]:
        """
        Get mock klines for a symbol.
        
        Args:
            symbol: Symbol to get klines for
            interval: Interval (e.g., "1m", "5m", "1h")
            limit: Number of klines to return
            
        Returns:
            List of kline dictionaries
        """
        return self.get_klines(symbol, interval, limit)
        
    def get_mock_ticker(self, symbol: str) -> Dict:
        """
        Get mock ticker for a symbol.
        
        Args:
            symbol: Symbol to get ticker for
            
        Returns:
            Ticker dictionary
        """
        return self.get_ticker(symbol)
        
    def get_mock_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """
        Get mock order book for a symbol.
        
        Args:
            symbol: Symbol to get order book for
            limit: Number of levels to return
            
        Returns:
            Order book dictionary
        """
        return self.get_order_book(symbol)
        
    def get_mock_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """
        Get mock trades for a symbol.
        
        Args:
            symbol: Symbol to get trades for
            limit: Number of trades to return
            
        Returns:
            List of trade dictionaries
        """
        return self.get_trades(symbol, limit)
        
    def _save_to_cache(self, data_type: str, key: str, data: Any) -> None:
        """
        Save data to cache.
        
        Args:
            data_type: Type of data
            key: Cache key
            data: Data to cache
        """
        try:
            # Create cache directory if it doesn't exist
            cache_dir = os.path.join(self.cache_dir, data_type)
            os.makedirs(cache_dir, exist_ok=True)
            
            # Save data to cache
            cache_file = os.path.join(cache_dir, f"{key}.json")
            
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
                
            logger.debug(f"Saved {data_type} data for {key} to cache")
        except Exception as e:
            logger.warning(f"Error saving {data_type} data for {key} to cache: {str(e)}")
            
    def _load_from_cache(self, data_type: str, key: str) -> Any:
        """
        Load data from cache.
        
        Args:
            data_type: Type of data
            key: Cache key
            
        Returns:
            Cached data
        """
        try:
            # Check if cache file exists
            cache_file = os.path.join(self.cache_dir, data_type, f"{key}.json")
            
            if os.path.exists(cache_file):
                # Load data from cache
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    
                logger.debug(f"Loaded {data_type} data for {key} from cache")
                return data
            else:
                logger.debug(f"No cached {data_type} data for {key}")
                return None
        except Exception as e:
            logger.warning(f"Error loading {data_type} data for {key} from cache: {str(e)}")
            return None
            
    def cache_real_response(self, endpoint: str, params: Dict, data: Any) -> None:
        """
        Cache real API response.
        
        Args:
            endpoint: API endpoint
            params: Parameters for the endpoint
            data: Response data
        """
        # Create cache key
        key = f"{endpoint}_{json.dumps(params, sort_keys=True)}"
        
        # Save to cache
        self._save_to_cache("real_responses", key, data)
        
    def get_cached_response(self, endpoint: str, params: Dict) -> Any:
        """
        Get cached API response.
        
        Args:
            endpoint: API endpoint
            params: Parameters for the endpoint
            
        Returns:
            Cached response data
        """
        # Create cache key
        key = f"{endpoint}_{json.dumps(params, sort_keys=True)}"
        
        # Load from cache
        return self._load_from_cache("real_responses", key)
