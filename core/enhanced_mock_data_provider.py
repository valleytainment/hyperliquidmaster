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
            # 50% chance of having a position
            if random.random() < 0.5:
                price = self.last_prices.get(symbol, self.default_params[symbol]["base_price"])
                size = np.random.lognormal(np.log(1.0), 0.5) * (1 if random.random() < 0.5 else -1)
                entry_price = price * (1 + np.random.normal(0, 0.01))
                unrealized_pnl = size * (price - entry_price)
                
                positions.append({
                    "symbol": symbol,
                    "size": size,
                    "entry_price": entry_price,
                    "unrealized_pnl": unrealized_pnl,
                    "liquidation_price": entry_price * (0.8 if size > 0 else 1.2)
                })
        else:
            # Generate positions for all symbols
            for sym in self.price_models:
                # 30% chance of having a position
                if random.random() < 0.3:
                    price = self.last_prices.get(sym, self.default_params[sym]["base_price"])
                    size = np.random.lognormal(np.log(1.0), 0.5) * (1 if random.random() < 0.5 else -1)
                    entry_price = price * (1 + np.random.normal(0, 0.01))
                    unrealized_pnl = size * (price - entry_price)
                    
                    positions.append({
                        "symbol": sym,
                        "size": size,
                        "entry_price": entry_price,
                        "unrealized_pnl": unrealized_pnl,
                        "liquidation_price": entry_price * (0.8 if size > 0 else 1.2)
                    })
                    
        return positions
        
    def execute_order(self, order: Dict) -> Dict:
        """
        Execute an order.
        
        Args:
            order: Order dictionary
            
        Returns:
            Order result dictionary
        """
        # Extract order details
        symbol = order.get("symbol", "BTC")
        order_type = order.get("type", "limit")
        side = order.get("side", "buy")
        size = order.get("size", 1.0)
        price = order.get("price", self.last_prices.get(symbol, self.default_params[symbol]["base_price"]))
        
        # Generate order ID
        order_id = f"mock_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        
        # Create order result
        order_result = {
            "order_id": order_id,
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "size": size,
            "price": price,
            "status": "filled",
            "filled_size": size,
            "filled_price": price * (1 + np.random.normal(0, 0.001)),
            "timestamp": int(time.time() * 1000)
        }
        
        return order_result
        
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
                json.dump(data, f, default=str)
        except Exception as e:
            logger.warning(f"Error saving to cache: {str(e)}")
            
    def _load_from_cache(self, data_type: str, key: str) -> Optional[Any]:
        """
        Load data from cache.
        
        Args:
            data_type: Type of data
            key: Cache key
            
        Returns:
            Cached data or None if not found
        """
        try:
            # Check if cache file exists
            cache_file = os.path.join(self.cache_dir, data_type, f"{key}.json")
            if not os.path.exists(cache_file):
                return None
                
            # Load data from cache
            with open(cache_file, "r") as f:
                data = json.load(f)
                
            return data
        except Exception as e:
            logger.warning(f"Error loading from cache: {str(e)}")
            return None
            
    def cache_real_response(self, endpoint: str, params: Dict, response: Any) -> None:
        """
        Cache real API response.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            response: API response
        """
        try:
            # Create cache key
            key = f"{endpoint}_{json.dumps(params, sort_keys=True)}"
            
            # Save to cache
            self._save_to_cache("api_responses", key, response)
        except Exception as e:
            logger.warning(f"Error caching real response: {str(e)}")
            
    def get_cached_response(self, endpoint: str, params: Dict) -> Optional[Any]:
        """
        Get cached API response.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            Cached response or None if not found
        """
        try:
            # Create cache key
            key = f"{endpoint}_{json.dumps(params, sort_keys=True)}"
            
            # Load from cache
            return self._load_from_cache("api_responses", key)
        except Exception as e:
            logger.warning(f"Error getting cached response: {str(e)}")
            return None
            
    def generate_synthetic_data(self, symbol: str, timeframe: str, days: int = 30) -> pd.DataFrame:
        """
        Generate synthetic data for a symbol.
        
        Args:
            symbol: Symbol to generate data for
            timeframe: Timeframe (e.g., "1m", "5m", "1h")
            days: Number of days of data to generate
            
        Returns:
            DataFrame with synthetic data
        """
        # Check if symbol is supported
        if symbol not in self.price_models:
            self._init_price_model(symbol)
            
        # Get timeframe in milliseconds
        timeframe_ms = self._get_timeframe_milliseconds(timeframe)
        
        # Calculate number of candles
        num_candles = days * 24 * 60 * 60 * 1000 // timeframe_ms
        
        # Generate candles
        candles = []
        current_time = int(time.time() * 1000)
        start_time = current_time - num_candles * timeframe_ms
        
        for i in range(num_candles):
            candle_time = start_time + i * timeframe_ms
            candle = self._generate_candle(symbol, timeframe, candle_time)
            candles.append(candle)
            
        # Create DataFrame
        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("datetime", inplace=True)
        
        # Add technical indicators
        df = self._add_technical_indicators(df)
        
        # Save to file
        os.makedirs(os.path.join(self.data_dir, "synthetic"), exist_ok=True)
        df.to_csv(os.path.join(self.data_dir, "synthetic", f"{symbol}_{timeframe}_{days}d.csv"))
        
        return df
        
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        # Copy DataFrame
        df_copy = df.copy()
        
        # Calculate SMA
        df_copy["sma_5"] = df_copy["close"].rolling(window=5).mean()
        df_copy["sma_10"] = df_copy["close"].rolling(window=10).mean()
        df_copy["sma_20"] = df_copy["close"].rolling(window=20).mean()
        df_copy["sma_50"] = df_copy["close"].rolling(window=50).mean()
        df_copy["sma_100"] = df_copy["close"].rolling(window=100).mean()
        df_copy["sma_200"] = df_copy["close"].rolling(window=200).mean()
        
        # Calculate EMA
        df_copy["ema_5"] = df_copy["close"].ewm(span=5, adjust=False).mean()
        df_copy["ema_10"] = df_copy["close"].ewm(span=10, adjust=False).mean()
        df_copy["ema_20"] = df_copy["close"].ewm(span=20, adjust=False).mean()
        df_copy["ema_50"] = df_copy["close"].ewm(span=50, adjust=False).mean()
        df_copy["ema_100"] = df_copy["close"].ewm(span=100, adjust=False).mean()
        df_copy["ema_200"] = df_copy["close"].ewm(span=200, adjust=False).mean()
        
        # Calculate VWMA
        df_copy["vwma_5"] = (df_copy["close"] * df_copy["volume"]).rolling(window=5).sum() / df_copy["volume"].rolling(window=5).sum()
        df_copy["vwma_10"] = (df_copy["close"] * df_copy["volume"]).rolling(window=10).sum() / df_copy["volume"].rolling(window=10).sum()
        df_copy["vwma_20"] = (df_copy["close"] * df_copy["volume"]).rolling(window=20).sum() / df_copy["volume"].rolling(window=20).sum()
        
        # Calculate RSI
        delta = df_copy["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df_copy["rsi_14"] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        df_copy["macd"] = df_copy["close"].ewm(span=12, adjust=False).mean() - df_copy["close"].ewm(span=26, adjust=False).mean()
        df_copy["macd_signal"] = df_copy["macd"].ewm(span=9, adjust=False).mean()
        df_copy["macd_histogram"] = df_copy["macd"] - df_copy["macd_signal"]
        
        # Calculate Bollinger Bands
        df_copy["bb_middle"] = df_copy["close"].rolling(window=20).mean()
        df_copy["bb_std"] = df_copy["close"].rolling(window=20).std()
        df_copy["bb_upper"] = df_copy["bb_middle"] + 2 * df_copy["bb_std"]
        df_copy["bb_lower"] = df_copy["bb_middle"] - 2 * df_copy["bb_std"]
        
        # Calculate ATR
        high_low = df_copy["high"] - df_copy["low"]
        high_close = (df_copy["high"] - df_copy["close"].shift()).abs()
        low_close = (df_copy["low"] - df_copy["close"].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df_copy["atr_14"] = true_range.rolling(window=14).mean()
        
        # Calculate Stochastic Oscillator
        low_14 = df_copy["low"].rolling(window=14).min()
        high_14 = df_copy["high"].rolling(window=14).max()
        df_copy["stoch_k"] = 100 * (df_copy["close"] - low_14) / (high_14 - low_14)
        df_copy["stoch_d"] = df_copy["stoch_k"].rolling(window=3).mean()
        
        # Calculate OBV
        obv = (df_copy["close"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * df_copy["volume"]).cumsum()
        df_copy["obv"] = obv
        
        # Calculate ADX
        plus_dm = df_copy["high"].diff()
        minus_dm = df_copy["low"].diff(-1).abs()
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
        tr = pd.concat([
            (df_copy["high"] - df_copy["low"]).abs(),
            (df_copy["high"] - df_copy["close"].shift()).abs(),
            (df_copy["low"] - df_copy["close"].shift()).abs()
        ], axis=1).max(axis=1)
        atr_14 = tr.rolling(window=14).mean()
        plus_di_14 = 100 * (plus_dm.rolling(window=14).mean() / atr_14)
        minus_di_14 = 100 * (minus_dm.rolling(window=14).mean() / atr_14)
        dx = 100 * ((plus_di_14 - minus_di_14).abs() / (plus_di_14 + minus_di_14).abs())
        df_copy["adx_14"] = dx.rolling(window=14).mean()
        
        return df_copy
        
    def generate_all_synthetic_data(self, symbols: List[str] = None, timeframes: List[str] = None, days: int = 30) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Generate synthetic data for all symbols and timeframes.
        
        Args:
            symbols: List of symbols to generate data for (if None, use all supported symbols)
            timeframes: List of timeframes to generate data for (if None, use default timeframes)
            days: Number of days of data to generate
            
        Returns:
            Dictionary of synthetic data by symbol and timeframe
        """
        # Use default symbols if not specified
        if symbols is None:
            symbols = list(self.default_params.keys())
            
        # Use default timeframes if not specified
        if timeframes is None:
            timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
            
        # Generate synthetic data
        synthetic_data = {}
        
        for symbol in symbols:
            synthetic_data[symbol] = {}
            
            for timeframe in timeframes:
                logger.info(f"Generating synthetic data for {symbol} {timeframe}")
                synthetic_data[symbol][timeframe] = self.generate_synthetic_data(symbol, timeframe, days)
                
        return synthetic_data
        
    def save_synthetic_data(self, synthetic_data: Dict[str, Dict[str, pd.DataFrame]], output_dir: str = None) -> None:
        """
        Save synthetic data to files.
        
        Args:
            synthetic_data: Dictionary of synthetic data by symbol and timeframe
            output_dir: Output directory (if None, use default data directory)
        """
        # Use default data directory if not specified
        if output_dir is None:
            output_dir = os.path.join(self.data_dir, "synthetic")
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save synthetic data
        for symbol, timeframe_data in synthetic_data.items():
            for timeframe, df in timeframe_data.items():
                output_file = os.path.join(output_dir, f"{symbol}_{timeframe}.csv")
                df.to_csv(output_file)
                logger.info(f"Saved synthetic data to {output_file}")
                
    def load_synthetic_data(self, symbol: str, timeframe: str, input_dir: str = None) -> Optional[pd.DataFrame]:
        """
        Load synthetic data from file.
        
        Args:
            symbol: Symbol to load data for
            timeframe: Timeframe to load data for
            input_dir: Input directory (if None, use default data directory)
            
        Returns:
            DataFrame with synthetic data or None if not found
        """
        # Use default data directory if not specified
        if input_dir is None:
            input_dir = os.path.join(self.data_dir, "synthetic")
            
        # Check if file exists
        input_file = os.path.join(input_dir, f"{symbol}_{timeframe}.csv")
        if not os.path.exists(input_file):
            logger.warning(f"Synthetic data file not found: {input_file}")
            return None
            
        # Load synthetic data
        df = pd.read_csv(input_file, index_col=0, parse_dates=True)
        logger.info(f"Loaded synthetic data from {input_file}")
        
        return df
        
    def get_synthetic_candles(self, symbol: str, timeframe: str, limit: int = 100) -> List[Dict]:
        """
        Get synthetic candles for a symbol.
        
        Args:
            symbol: Symbol to get candles for
            timeframe: Timeframe (e.g., "1m", "5m", "1h")
            limit: Number of candles to return
            
        Returns:
            List of candle dictionaries
        """
        # Try to load synthetic data
        df = self.load_synthetic_data(symbol, timeframe)
        
        # If synthetic data not found, generate it
        if df is None:
            df = self.generate_synthetic_data(symbol, timeframe)
            
        # Get last 'limit' candles
        df = df.iloc[-limit:]
        
        # Convert to list of dictionaries
        candles = []
        for _, row in df.iterrows():
            candle = {
                "timestamp": int(row.name.timestamp() * 1000),
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"]
            }
            candles.append(candle)
            
        return candles
        
    def get_synthetic_market_data(self, symbol: str) -> Dict:
        """
        Get synthetic market data for a symbol.
        
        Args:
            symbol: Symbol to get market data for
            
        Returns:
            Market data dictionary
        """
        # Try to load synthetic data
        df = self.load_synthetic_data(symbol, "1m")
        
        # If synthetic data not found, generate it
        if df is None:
            df = self.generate_synthetic_data(symbol, "1m")
            
        # Get last row
        last_row = df.iloc[-1]
        
        # Create market data
        market_data = {
            "symbol": symbol,
            "last_price": last_row["close"],
            "volume_24h": df.iloc[-1440:]["volume"].sum(),  # Last 24 hours (1440 minutes)
            "change_24h": (last_row["close"] / df.iloc[-1440]["close"] - 1) * 100,  # Last 24 hours
            "timestamp": int(last_row.name.timestamp() * 1000)
        }
        
        return market_data
        
    def get_synthetic_order_book(self, symbol: str) -> Dict:
        """
        Get synthetic order book for a symbol.
        
        Args:
            symbol: Symbol to get order book for
            
        Returns:
            Order book dictionary
        """
        # Try to load synthetic data
        df = self.load_synthetic_data(symbol, "1m")
        
        # If synthetic data not found, generate it
        if df is None:
            df = self.generate_synthetic_data(symbol, "1m")
            
        # Get last price
        last_price = df.iloc[-1]["close"]
        
        # Generate order book
        order_book = self._generate_order_book(symbol, last_price)
        
        return order_book
        
    def get_synthetic_historical_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get synthetic historical data for a symbol.
        
        Args:
            symbol: Symbol to get historical data for
            timeframe: Timeframe (e.g., "1m", "5m", "1h")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with historical data
        """
        # Try to load synthetic data
        df = self.load_synthetic_data(symbol, timeframe)
        
        # If synthetic data not found, generate it
        if df is None:
            # Calculate days needed
            start_timestamp = datetime.datetime.strptime(start_date, "%Y-%m-%d").timestamp()
            end_timestamp = datetime.datetime.strptime(end_date, "%Y-%m-%d").timestamp()
            days_needed = int((end_timestamp - start_timestamp) / (24 * 60 * 60)) + 1
            
            # Generate synthetic data
            df = self.generate_synthetic_data(symbol, timeframe, days=days_needed)
            
        # Filter by date range
        df = df.loc[start_date:end_date]
        
        return df
        
def main():
    """
    Main function for testing.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Create mock data provider
    mock_provider = EnhancedMockDataProvider()
    
    # Generate synthetic data for all symbols and timeframes
    symbols = ["BTC", "ETH", "SOL", "XRP"]
    timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
    
    synthetic_data = mock_provider.generate_all_synthetic_data(symbols, timeframes, days=30)
    mock_provider.save_synthetic_data(synthetic_data)
    
    # Test market data
    for symbol in symbols:
        market_data = mock_provider.get_synthetic_market_data(symbol)
        print(f"Market data for {symbol}: {market_data}")
        
    # Test order book
    for symbol in symbols:
        order_book = mock_provider.get_synthetic_order_book(symbol)
        print(f"Order book for {symbol}: {len(order_book['bids'])} bids, {len(order_book['asks'])} asks")
        
    # Test candles
    for symbol in symbols:
        for timeframe in timeframes:
            candles = mock_provider.get_synthetic_candles(symbol, timeframe, limit=10)
            print(f"Candles for {symbol} {timeframe}: {len(candles)} candles")
            
    # Test historical data
    for symbol in symbols:
        for timeframe in timeframes:
            df = mock_provider.get_synthetic_historical_data(symbol, timeframe, "2025-04-24", "2025-05-24")
            print(f"Historical data for {symbol} {timeframe}: {len(df)} rows")
            
    print("All tests completed successfully")
    
if __name__ == "__main__":
    main()
