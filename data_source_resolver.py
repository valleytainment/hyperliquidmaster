"""
Enhanced Data Source Resolver Module

This module provides robust data collection from multiple sources with fallback mechanisms
to ensure high-quality data for backtesting and live trading. It prioritizes real market
data from Hyperliquid and falls back to alternative sources when necessary.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import requests
import time
import asyncio
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_source_resolver.log")
    ]
)
logger = logging.getLogger(__name__)

class DataSourceResolver:
    """
    Data Source Resolver for cryptocurrency market data.
    """
    
    def __init__(self, base_dir: str = "real_market_data"):
        """
        Initialize the data source resolver.
        
        Args:
            base_dir: Base directory for data storage
        """
        self.base_dir = base_dir
        
        # Create directories
        os.makedirs(f"{self.base_dir}/raw", exist_ok=True)
        os.makedirs(f"{self.base_dir}/processed", exist_ok=True)
        
        # API endpoints
        self.hyperliquid_rest_url = "https://api.hyperliquid.xyz/info"
        self.hyperliquid_ws_url = "wss://api.hyperliquid.xyz/ws"
        self.binance_url = "https://api.binance.com/api/v3"
        self.coingecko_url = "https://api.coingecko.com/api/v3"
        
        # Symbol mappings
        self.symbol_mappings = {
            "BTC-USD-PERP": {
                "hyperliquid": "BTC",
                "binance": "BTCUSDT",
                "coingecko": "bitcoin"
            },
            "ETH-USD-PERP": {
                "hyperliquid": "ETH",
                "binance": "ETHUSDT",
                "coingecko": "ethereum"
            },
            "SOL-USD-PERP": {
                "hyperliquid": "SOL",
                "binance": "SOLUSDT",
                "coingecko": "solana"
            },
            # Additional mappings omitted for brevity
        }
        
        # Interval mappings
        self.interval_mappings = {
            "1m": {
                "hyperliquid": "1m",
                "binance": "1m",
                "coingecko": "minute"
            },
            "5m": {
                "hyperliquid": "5m",
                "binance": "5m",
                "coingecko": "minute"
            },
            "15m": {
                "hyperliquid": "15m",
                "binance": "15m",
                "coingecko": "minute"
            },
            "30m": {
                "hyperliquid": "30m",
                "binance": "30m",
                "coingecko": "minute"
            },
            "1h": {
                "hyperliquid": "1h",
                "binance": "1h",
                "coingecko": "hourly"
            },
            "4h": {
                "hyperliquid": "4h",
                "binance": "4h",
                "coingecko": "hourly"
            },
            "1d": {
                "hyperliquid": "1d",
                "binance": "1d",
                "coingecko": "daily"
            },
            "1w": {
                "hyperliquid": "1w",
                "binance": "1w",
                "coingecko": "weekly"
            }
        }
        
        logger.info("Data Source Resolver initialized")
        
    async def fetch_data(self, symbol: str, interval: str, days: int = 30) -> Tuple[pd.DataFrame, Dict]:
        """
        Fetch data for a symbol and interval from multiple sources with fallback.
        
        Args:
            symbol: Symbol to fetch data for
            interval: Interval to fetch data for
            days: Number of days to fetch
            
        Returns:
            Tuple of (DataFrame with data, Dictionary with data quality metrics)
        """
        logger.info(f"Fetching data for {symbol}, interval {interval}, days {days}")
        
        # Check if processed data exists and is recent
        processed_path = f"{self.base_dir}/processed/{symbol}_{interval}_processed.csv"
        if os.path.exists(processed_path):
            df = pd.read_csv(processed_path)
            
            # Check if data is recent enough
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                latest_timestamp = df["timestamp"].max()
                
                if latest_timestamp >= datetime.now() - timedelta(hours=1):
                    logger.info(f"Using existing processed data for {symbol}, interval {interval}")
                    
                    # Calculate data quality metrics
                    data_quality = self.calculate_data_quality(df, symbol)
                    
                    return df, data_quality
        
        # Initialize alternative sources list
        alternative_sources = []
        
        # Try to fetch from Hyperliquid
        try:
            logger.info(f"Attempting to fetch data from Hyperliquid for {symbol}, interval {interval}")
            df_hyperliquid = await self.fetch_from_hyperliquid(symbol, interval, days)
            
            if df_hyperliquid is not None and not df_hyperliquid.empty:
                logger.info(f"Successfully fetched data from Hyperliquid for {symbol}, interval {interval}")
                
                # Save raw data
                df_hyperliquid.to_csv(f"{self.base_dir}/raw/{symbol}_{interval}_hyperliquid_klines.csv", index=False)
                
                # Calculate technical indicators
                df_processed = self.calculate_technical_indicators(df_hyperliquid)
                
                # Save processed data
                df_processed.to_csv(processed_path, index=False)
                
                # Calculate data quality metrics
                data_quality = {
                    "source": "hyperliquid",
                    "is_synthetic": False,
                    "synthetic_ratio": 0.0,
                    "alternative_sources": alternative_sources,
                    "missing_values": df_processed.isnull().sum().sum() / (df_processed.shape[0] * df_processed.shape[1]),
                    "data_points": df_processed.shape[0],
                    "start_date": df_processed["timestamp"].min().strftime("%Y-%m-%d"),
                    "end_date": df_processed["timestamp"].max().strftime("%Y-%m-%d")
                }
                
                return df_processed, data_quality
        except Exception as e:
            error_msg = f"Failed to fetch data from Hyperliquid for {symbol}, interval {interval}: {str(e)}"
            logger.warning(error_msg)
            # Log detailed error information
            logger.error(f"Hyperliquid API error details: {type(e).__name__}, {str(e)}")
        
        # Try to fetch from Binance
        try:
            logger.info(f"Attempting to fetch data from Binance for {symbol}, interval {interval}")
            df_binance = await self.fetch_from_binance(symbol, interval, days)
            
            if df_binance is not None and not df_binance.empty:
                logger.info(f"Successfully fetched data from Binance for {symbol}, interval {interval}")
                alternative_sources.append("binance")
                
                # Save raw data
                df_binance.to_csv(f"{self.base_dir}/raw/{symbol}_{interval}_binance_klines.csv", index=False)
                
                # Calculate technical indicators
                df_processed = self.calculate_technical_indicators(df_binance)
                
                # Save processed data
                df_processed.to_csv(processed_path, index=False)
                
                # Calculate data quality metrics
                data_quality = {
                    "source": "binance",
                    "is_synthetic": False,
                    "synthetic_ratio": 0.0,
                    "alternative_sources": alternative_sources,
                    "missing_values": df_processed.isnull().sum().sum() / (df_processed.shape[0] * df_processed.shape[1]),
                    "data_points": df_processed.shape[0],
                    "start_date": df_processed["timestamp"].min().strftime("%Y-%m-%d"),
                    "end_date": df_processed["timestamp"].max().strftime("%Y-%m-%d")
                }
                
                return df_processed, data_quality
        except Exception as e:
            error_msg = f"Failed to fetch data from Binance for {symbol}, interval {interval}: {str(e)}"
            logger.warning(error_msg)
            # Log detailed error information
            logger.error(f"Binance API error details: {type(e).__name__}, {str(e)}")
        
        # Try to fetch from CoinGecko
        try:
            logger.info(f"Attempting to fetch data from CoinGecko for {symbol}, interval {interval}")
            df_coingecko = await self.fetch_from_coingecko(symbol, interval, days)
            
            if df_coingecko is not None and not df_coingecko.empty:
                logger.info(f"Successfully fetched data from CoinGecko for {symbol}, interval {interval}")
                alternative_sources.append("coingecko")
                
                # Save raw data
                df_coingecko.to_csv(f"{self.base_dir}/raw/{symbol}_{interval}_coingecko_klines.csv", index=False)
                
                # Calculate technical indicators
                df_processed = self.calculate_technical_indicators(df_coingecko)
                
                # Save processed data
                df_processed.to_csv(processed_path, index=False)
                
                # Calculate data quality metrics
                data_quality = {
                    "source": "coingecko",
                    "is_synthetic": False,
                    "synthetic_ratio": 0.0,
                    "alternative_sources": alternative_sources,
                    "missing_values": df_processed.isnull().sum().sum() / (df_processed.shape[0] * df_processed.shape[1]),
                    "data_points": df_processed.shape[0],
                    "start_date": df_processed["timestamp"].min().strftime("%Y-%m-%d"),
                    "end_date": df_processed["timestamp"].max().strftime("%Y-%m-%d")
                }
                
                return df_processed, data_quality
        except Exception as e:
            error_msg = f"Failed to fetch data from CoinGecko for {symbol}, interval {interval}: {str(e)}"
            logger.warning(error_msg)
            # Log detailed error information
            logger.error(f"CoinGecko API error details: {type(e).__name__}, {str(e)}")
        
        # Generate synthetic data as last resort
        logger.warning(f"All data sources failed. Generating synthetic data for {symbol}, interval {interval}")
        
        df_synthetic = self.generate_synthetic_data(symbol, interval, days)
        
        # Save raw data
        df_synthetic.to_csv(f"{self.base_dir}/raw/{symbol}_{interval}_synthetic_klines.csv", index=False)
        
        # Calculate technical indicators
        df_processed = self.calculate_technical_indicators(df_synthetic)
        
        # Save processed data
        df_processed.to_csv(processed_path, index=False)
        
        # Calculate data quality metrics
        data_quality = {
            "source": "synthetic",
            "is_synthetic": True,
            "synthetic_ratio": 1.0,
            "alternative_sources": alternative_sources,
            "missing_values": 0.0,
            "data_points": df_processed.shape[0],
            "start_date": df_processed["timestamp"].min().strftime("%Y-%m-%d"),
            "end_date": df_processed["timestamp"].max().strftime("%Y-%m-%d")
        }
        
        return df_processed, data_quality
        
    async def fetch_from_hyperliquid(self, symbol: str, interval: str, days: int = 30) -> pd.DataFrame:
        """
        Fetch data from Hyperliquid.
        
        Args:
            symbol: Symbol to fetch data for
            interval: Interval to fetch data for
            days: Number of days to fetch
            
        Returns:
            DataFrame with data
        """
        # Get mapped symbol
        if symbol not in self.symbol_mappings:
            logger.warning(f"Symbol {symbol} not found in mappings")
            return None
            
        mapped_symbol = self.symbol_mappings[symbol]["hyperliquid"]
        
        # Get mapped interval
        if interval not in self.interval_mappings:
            logger.warning(f"Interval {interval} not found in mappings")
            return None
            
        mapped_interval = self.interval_mappings[interval]["hyperliquid"]
        
        # Calculate start and end timestamps
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        # Construct request payload
        payload = {
            "type": "klines",
            "coin": mapped_symbol,
            "interval": mapped_interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000
        }
        
        # Log the request details
        logger.info(f"Hyperliquid API request: URL={self.hyperliquid_rest_url}, Payload={json.dumps(payload)}")
        
        # Make request
        try:
            response = requests.post(self.hyperliquid_rest_url, json=payload)
            
            # Log response status and headers
            logger.info(f"Hyperliquid API response status: {response.status_code}")
            logger.info(f"Hyperliquid API response headers: {response.headers}")
            
            # Check if response is successful
            response.raise_for_status()
            
            # Try to parse response as JSON
            try:
                data = response.json()
                logger.info(f"Hyperliquid API response data structure: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            except json.JSONDecodeError:
                logger.error(f"Hyperliquid API response is not valid JSON: {response.text[:200]}...")
                return None
            
            # Check if data is valid
            if not data or "data" not in data or not data["data"]:
                logger.warning(f"No data returned from Hyperliquid for {symbol}, interval {interval}")
                logger.info(f"Hyperliquid API response content: {json.dumps(data)[:500]}...")
                return None
                
            # Parse data
            klines = []
            for kline in data["data"]:
                timestamp = datetime.fromtimestamp(kline[0] / 1000)
                open_price = float(kline[1])
                high_price = float(kline[2])
                low_price = float(kline[3])
                close_price = float(kline[4])
                volume = float(kline[5])
                
                klines.append([timestamp, open_price, high_price, low_price, close_price, volume])
                
            # Create DataFrame
            df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume"])
            
            return df
        except requests.exceptions.RequestException as e:
            logger.error(f"Hyperliquid API request error: {type(e).__name__}, {str(e)}")
            
            # If we have a response, log its content
            if 'response' in locals():
                logger.error(f"Hyperliquid API error response: {response.text[:500]}...")
                
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching data from Hyperliquid: {type(e).__name__}, {str(e)}")
            return None
            
    async def fetch_from_binance(self, symbol: str, interval: str, days: int = 30) -> pd.DataFrame:
        """
        Fetch data from Binance.
        
        Args:
            symbol: Symbol to fetch data for
            interval: Interval to fetch data for
            days: Number of days to fetch
            
        Returns:
            DataFrame with data
        """
        # Get mapped symbol
        if symbol not in self.symbol_mappings:
            logger.warning(f"Symbol {symbol} not found in mappings")
            return None
            
        mapped_symbol = self.symbol_mappings[symbol]["binance"]
        
        # Get mapped interval
        if interval not in self.interval_mappings:
            logger.warning(f"Interval {interval} not found in mappings")
            return None
            
        mapped_interval = self.interval_mappings[interval]["binance"]
        
        # Calculate start and end timestamps
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        # Construct URL
        url = f"{self.binance_url}/klines?symbol={mapped_symbol}&interval={mapped_interval}&startTime={start_time}&endTime={end_time}&limit=1000"
        
        # Log the request details
        logger.info(f"Binance API request: URL={url}")
        
        # Make request
        try:
            response = requests.get(url)
            
            # Log response status and headers
            logger.info(f"Binance API response status: {response.status_code}")
            logger.info(f"Binance API response headers: {response.headers}")
            
            # Check if response is successful
            response.raise_for_status()
            
            # Try to parse response as JSON
            try:
                data = response.json()
                logger.info(f"Binance API response data type: {type(data)}")
                logger.info(f"Binance API response data length: {len(data) if isinstance(data, list) else 'Not a list'}")
            except json.JSONDecodeError:
                logger.error(f"Binance API response is not valid JSON: {response.text[:200]}...")
                return None
            
            # Check if data is valid
            if not data:
                logger.warning(f"No data returned from Binance for {symbol}, interval {interval}")
                return None
                
            # Parse data
            klines = []
            for kline in data:
                timestamp = datetime.fromtimestamp(kline[0] / 1000)
                open_price = float(kline[1])
                high_price = float(kline[2])
                low_price = float(kline[3])
                close_price = float(kline[4])
                volume = float(kline[5])
                
                klines.append([timestamp, open_price, high_price, low_price, close_price, volume])
                
            # Create DataFrame
            df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume"])
            
            return df
        except requests.exceptions.RequestException as e:
            logger.error(f"Binance API request error: {type(e).__name__}, {str(e)}")
            
            # If we have a response, log its content
            if 'response' in locals():
                logger.error(f"Binance API error response: {response.text[:500]}...")
                
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching data from Binance: {type(e).__name__}, {str(e)}")
            return None
            
    async def fetch_from_coingecko(self, symbol: str, interval: str, days: int = 30) -> pd.DataFrame:
        """
        Fetch data from CoinGecko.
        
        Args:
            symbol: Symbol to fetch data for
            interval: Interval to fetch data for
            days: Number of days to fetch
            
        Returns:
            DataFrame with data
        """
        # Get mapped symbol
        if symbol not in self.symbol_mappings:
            logger.warning(f"Symbol {symbol} not found in mappings")
            return None
            
        mapped_symbol = self.symbol_mappings[symbol]["coingecko"]
        
        # CoinGecko has limited interval options
        if interval in ["1m", "5m", "15m", "30m"]:
            # For small intervals, use hourly and then resample
            days = min(days, 7)  # CoinGecko limits hourly data to 7 days
            url = f"{self.coingecko_url}/coins/{mapped_symbol}/market_chart?vs_currency=usd&days={days}&interval=hourly"
        elif interval in ["1h", "2h", "4h", "6h", "8h", "12h"]:
            # For medium intervals, use hourly and then resample
            days = min(days, 90)  # CoinGecko limits hourly data to 90 days
            url = f"{self.coingecko_url}/coins/{mapped_symbol}/market_chart?vs_currency=usd&days={days}&interval=hourly"
        else:
            # For large intervals, use daily
            url = f"{self.coingecko_url}/coins/{mapped_symbol}/market_chart?vs_currency=usd&days={days}&interval=daily"
        
        # Log the request details
        logger.info(f"CoinGecko API request: URL={url}")
        
        # Make request
        try:
            response = requests.get(url)
            
            # Log response status and headers
            logger.info(f"CoinGecko API response status: {response.status_code}")
            logger.info(f"CoinGecko API response headers: {response.headers}")
            
            # Check if response is successful
            response.raise_for_status()
            
            # Try to parse response as JSON
            try:
                data = response.json()
                logger.info(f"CoinGecko API response data structure: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            except json.JSONDecodeError:
                logger.error(f"CoinGecko API response is not valid JSON: {response.text[:200]}...")
                return None
            
            # Check if data is valid
            if not data or "prices" not in data or not data["prices"]:
                logger.warning(f"No data returned from CoinGecko for {symbol}, interval {interval}")
                return None
                
            # Parse data
            prices = data["prices"]
            volumes = data["total_volumes"]
            
            # Create DataFrame
            df = pd.DataFrame(prices, columns=["timestamp", "close"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            
            # Add volume
            volume_df = pd.DataFrame(volumes, columns=["timestamp", "volume"])
            volume_df["timestamp"] = pd.to_datetime(volume_df["timestamp"], unit="ms")
            df = pd.merge(df, volume_df, on="timestamp", how="left")
            
            # CoinGecko doesn't provide OHLC data directly, so we need to estimate
            # We'll use the close price as a base and add some random variation
            np.random.seed(42)  # For reproducibility
            
            # Generate OHLC
            df["open"] = df["close"].shift(1).fillna(df["close"])
            df["high"] = df.apply(lambda row: max(row["open"], row["close"]) * (1 + np.random.uniform(0, 0.01)), axis=1)
            df["low"] = df.apply(lambda row: min(row["open"], row["close"]) * (1 - np.random.uniform(0, 0.01)), axis=1)
            
            # Reorder columns
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            
            # Resample if needed
            if interval in ["1m", "5m", "15m", "30m", "2h", "4h", "6h", "8h", "12h"]:
                # Convert interval to pandas frequency
                freq_map = {
                    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
                    "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h"
                }
                freq = freq_map[interval]
                
                # Resample
                df = df.set_index("timestamp")
                df = df.resample(freq).agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum"
                }).dropna().reset_index()
            
            return df
        except requests.exceptions.RequestException as e:
            logger.error(f"CoinGecko API request error: {type(e).__name__}, {str(e)}")
            
            # If we have a response, log its content
            if 'response' in locals():
                logger.error(f"CoinGecko API error response: {response.text[:500]}...")
                
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching data from CoinGecko: {type(e).__name__}, {str(e)}")
            return None
            
    def generate_synthetic_data(self, symbol: str, interval: str, days: int = 30) -> pd.DataFrame:
        """
        Generate synthetic data for a symbol and interval.
        
        Args:
            symbol: Symbol to generate data for
            interval: Interval to generate data for
            days: Number of days to generate
            
        Returns:
            DataFrame with synthetic data
        """
        logger.info(f"Generating synthetic data for {symbol}, interval {interval}")
        
        # Determine start and end dates
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=days)
        
        # Determine number of periods
        hours_per_period = self.interval_to_hours(interval)
        periods = int((end_dt - start_dt).total_seconds() / (3600 * hours_per_period))
        
        # Generate timestamps
        timestamps = [end_dt - timedelta(hours=i * hours_per_period) for i in range(periods)]
        timestamps.reverse()
        
        # Base price depends on symbol
        if "BTC" in symbol:
            base_price = 60000
            volatility = 0.02
        elif "ETH" in symbol:
            base_price = 3500
            volatility = 0.025
        elif "SOL" in symbol:
            base_price = 150
            volatility = 0.035
        else:
            base_price = 100
            volatility = 0.05
            
        # Generate price data with realistic patterns
        np.random.seed(42)  # For reproducibility
        
        # Generate returns with slight upward bias and autocorrelation
        returns = np.random.normal(0.0001, volatility, periods)
        
        # Add autocorrelation
        for i in range(1, periods):
            returns[i] = 0.7 * returns[i] + 0.3 * returns[i-1]
            
        # Generate prices
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
            
        # Generate OHLCV data
        data = []
        for i, timestamp in enumerate(timestamps):
            price = prices[i]
            
            # Generate realistic OHLC
            high_low_range = price * volatility * np.random.uniform(0.5, 1.5)
            high = price + high_low_range / 2
            low = price - high_low_range / 2
            
            # Randomly determine if open > close or close > open
            if np.random.random() > 0.5:
                open_price = price - high_low_range * np.random.uniform(0, 0.4)
                close = price + high_low_range * np.random.uniform(0, 0.4)
            else:
                open_price = price + high_low_range * np.random.uniform(0, 0.4)
                close = price - high_low_range * np.random.uniform(0, 0.4)
                
            # Ensure high >= max(open, close) and low <= min(open, close)
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Generate volume
            volume = base_price * 10 * np.random.uniform(0.5, 1.5)
            
            data.append([timestamp, open_price, high, low, close, volume])
            
        # Create DataFrame
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        return df
        
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Convert columns to numeric if they aren't already
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                
        # Calculate SMA
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()
        df["sma_200"] = df["close"].rolling(window=200).mean()
        
        # Calculate EMA
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
        
        # Calculate MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # Calculate RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        df["bb_std"] = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
        df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]
        
        # Calculate ATR
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df["atr"] = true_range.rolling(window=14).mean()
        
        # Calculate VWAP
        df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()
        
        # Calculate Stochastic Oscillator
        low_14 = df["low"].rolling(window=14).min()
        high_14 = df["high"].rolling(window=14).max()
        
        df["stoch_k"] = 100 * ((df["close"] - low_14) / (high_14 - low_14))
        df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()
        
        # Calculate OBV (On-Balance Volume)
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        
        return df
        
    def calculate_data_quality(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Calculate data quality metrics for a DataFrame.
        
        Args:
            df: DataFrame with data
            symbol: Symbol for the data
            
        Returns:
            Dictionary with data quality metrics
        """
        # Check if data is synthetic
        synthetic_path = f"{self.base_dir}/raw/{symbol}_synthetic_klines.csv"
        is_synthetic = os.path.exists(synthetic_path)
        
        # Check if alternative data was used
        alternative_sources = []
        for source in ["binance", "coingecko"]:
            alt_path = f"{self.base_dir}/raw/{symbol}_{source}_klines.csv"
            if os.path.exists(alt_path):
                alternative_sources.append(source)
                
        # Determine source
        if is_synthetic:
            source = "synthetic"
            synthetic_ratio = 1.0
        elif alternative_sources:
            source = alternative_sources[0]
            synthetic_ratio = 0.0
        else:
            source = "hyperliquid"
            synthetic_ratio = 0.0
            
        # Calculate metrics
        data_quality = {
            "source": source,
            "is_synthetic": is_synthetic,
            "alternative_sources": alternative_sources,
            "synthetic_ratio": synthetic_ratio,
            "missing_values": df.isnull().sum().sum() / (df.shape[0] * df.shape[1]),
            "data_points": df.shape[0],
            "start_date": df["timestamp"].min().strftime("%Y-%m-%d"),
            "end_date": df["timestamp"].max().strftime("%Y-%m-%d")
        }
        
        return data_quality
        
    def interval_to_hours(self, interval: str) -> int:
        """
        Convert interval string to hours.
        
        Args:
            interval: Interval string (e.g., "1h", "4h", "1d")
            
        Returns:
            Number of hours
        """
        if interval.endswith("m"):
            return int(interval[:-1]) / 60
        elif interval.endswith("h"):
            return int(interval[:-1])
        elif interval.endswith("d"):
            return int(interval[:-1]) * 24
        elif interval.endswith("w"):
            return int(interval[:-1]) * 24 * 7
        elif interval.endswith("M"):
            return int(interval[:-1]) * 24 * 30
        else:
            return 1  # Default to 1 hour

async def main():
    """
    Main function to test the data source resolver.
    """
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Data Source Resolver")
    parser.add_argument("--symbols", type=str, default="BTC-USD-PERP,ETH-USD-PERP,SOL-USD-PERP", help="Comma-separated list of symbols")
    parser.add_argument("--intervals", type=str, default="1h,4h", help="Comma-separated list of intervals")
    parser.add_argument("--days", type=int, default=30, help="Number of days to fetch")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Create resolver
    resolver = DataSourceResolver()
    
    # Get symbols and intervals
    symbols = args.symbols.split(",")
    intervals = args.intervals.split(",")
    
    # Fetch data for each symbol and interval
    for symbol in symbols:
        for interval in intervals:
            logger.info(f"Fetching data for {symbol}, interval {interval}")
            df, data_quality = await resolver.fetch_data(symbol, interval, args.days)
            
            logger.info(f"Data quality for {symbol}, interval {interval}: {data_quality}")
            logger.info(f"Data shape: {df.shape}")
            
            # Print first few rows
            logger.info(f"First few rows:\n{df.head()}")
            
            # Print last few rows
            logger.info(f"Last few rows:\n{df.tail()}")
            
            # Print summary statistics
            logger.info(f"Summary statistics:\n{df.describe()}")
            
            # Print correlation matrix
            logger.info(f"Correlation matrix:\n{df.corr()}")
            
            # Print missing values
            logger.info(f"Missing values:\n{df.isnull().sum()}")
            
            # Print data types
            logger.info(f"Data types:\n{df.dtypes}")
            
            # Print memory usage
            logger.info(f"Memory usage:\n{df.memory_usage(deep=True)}")
            
            # Print data quality
            logger.info(f"Data quality:\n{data_quality}")
            
            # Print data source
            logger.info(f"Data source: {data_quality['source']}")
            
            # Print synthetic ratio
            logger.info(f"Synthetic ratio: {data_quality['synthetic_ratio']}")
            
            # Print missing values
            logger.info(f"Missing values: {data_quality['missing_values']}")
            
            # Print data points
            logger.info(f"Data points: {data_quality['data_points']}")
            
            # Print start date
            logger.info(f"Start date: {data_quality['start_date']}")
            
            # Print end date
            logger.info(f"End date: {data_quality['end_date']}")
            
            # Print alternative sources
            logger.info(f"Alternative sources: {data_quality['alternative_sources']}")
            
            # Print is synthetic
            logger.info(f"Is synthetic: {data_quality['is_synthetic']}")
            
            # Print separator
            logger.info("-" * 80)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
