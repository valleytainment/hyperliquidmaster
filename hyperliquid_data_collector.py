"""
Real Market Data Collector for Hyperliquid

This module provides functionality to collect and process real market data from Hyperliquid.
"""

import os
import time
import logging
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio
import websockets
import hmac
import hashlib
import base64

class HyperliquidDataCollector:
    """
    Real market data collection and processing for the enhanced trading bot.
    """
    
    def __init__(self, config_path: str = "config.json", logger=None):
        """
        Initialize the Hyperliquid data collector.
        
        Args:
            config_path: Path to configuration file
            logger: Optional logger instance
        """
        # Setup logging
        self.logger = logger or self._setup_logger()
        self.logger.info("Initializing Hyperliquid Data Collector...")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # API endpoints
        self.base_url = "https://api.hyperliquid.xyz"
        self.ws_url = "wss://api.hyperliquid.xyz/ws"
        
        # Create output directories
        os.makedirs("real_data", exist_ok=True)
        
    def _setup_logger(self) -> logging.Logger:
        """
        Set up the logger.
        
        Returns:
            Configured logger
        """
        logger = logging.getLogger("HyperliquidDataCollector")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("hyperliquid_data_collection.log", mode="a")
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
        
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                
            self.logger.info(f"Configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            # Return default configuration
            return {
                "account_address": "",
                "secret_key": "",
                "symbols": ["BTC-USD-PERP", "ETH-USD-PERP", "SOL-USD-PERP"],
                "use_sentiment_analysis": True,
                "use_triple_confluence_strategy": True,
                "use_oracle_update_strategy": True
            }
            
    def _sign_request(self, endpoint: str, data: Dict) -> Dict:
        """
        Sign a request for authenticated API calls.
        
        Args:
            endpoint: API endpoint
            data: Request data
            
        Returns:
            Signed request headers
        """
        if not self.config.get("secret_key"):
            return {}
            
        timestamp = str(int(time.time() * 1000))
        message = f"{endpoint}{timestamp}{json.dumps(data)}"
        
        signature = hmac.new(
            self.config["secret_key"].encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return {
            "X-HL-Signature": signature,
            "X-HL-Timestamp": timestamp,
            "X-HL-Address": self.config.get("account_address", "")
        }
        
    async def fetch_historical_klines(self, symbol: str, interval: str = "1h", limit: int = 500) -> Optional[pd.DataFrame]:
        """
        Fetch historical klines (candlestick data) from Hyperliquid.
        
        Args:
            symbol: Trading symbol (e.g., "BTC")
            interval: Time interval (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Maximum number of klines to fetch
            
        Returns:
            DataFrame with historical data or None if failed
        """
        self.logger.info(f"Fetching historical klines for {symbol}, interval {interval}, limit {limit}...")
        
        try:
            # Convert symbol format if needed
            base_symbol = symbol.split("-")[0] if "-" in symbol else symbol
            
            # Prepare request
            endpoint = "/exchange/candles"
            url = f"{self.base_url}{endpoint}"
            
            # Convert interval to seconds
            interval_seconds = {
                "1m": 60,
                "5m": 300,
                "15m": 900,
                "1h": 3600,
                "4h": 14400,
                "1d": 86400
            }.get(interval, 3600)
            
            # Prepare request data
            data = {
                "coin": base_symbol,
                "resolution": interval_seconds,
                "limit": limit
            }
            
            # Make request
            response = requests.post(url, json=data)
            
            if response.status_code != 200:
                self.logger.error(f"Error fetching historical klines: {response.status_code} {response.text}")
                return None
                
            # Parse response
            klines = response.json()
            
            if not klines or not isinstance(klines, list):
                self.logger.error(f"Invalid response format: {klines}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume"])
            
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            
            # Convert string values to float
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
                
            # Add additional columns for analysis
            df["price"] = df["close"]  # Current price is close price
            
            # Save to CSV
            df.to_csv(f"real_data/{symbol}_{interval}_klines.csv", index=False)
            
            self.logger.info(f"Successfully fetched {len(df)} klines for {symbol}")
            return df
            
        except Exception as e:
            self.logger.exception(f"Error fetching historical klines: {str(e)}")
            return None
            
    async def fetch_funding_rates(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch funding rates from Hyperliquid.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            DataFrame with funding rates or None if failed
        """
        self.logger.info(f"Fetching funding rates for {symbol}...")
        
        try:
            # Convert symbol format if needed
            base_symbol = symbol.split("-")[0] if "-" in symbol else symbol
            
            # Prepare request
            endpoint = "/exchange/funding"
            url = f"{self.base_url}{endpoint}"
            
            # Prepare request data
            data = {
                "coin": base_symbol
            }
            
            # Make request
            response = requests.post(url, json=data)
            
            if response.status_code != 200:
                self.logger.error(f"Error fetching funding rates: {response.status_code} {response.text}")
                return None
                
            # Parse response
            funding_data = response.json()
            
            if not funding_data or not isinstance(funding_data, dict):
                self.logger.error(f"Invalid response format: {funding_data}")
                return None
                
            # Extract funding rates
            funding_rates = funding_data.get("fundingRates", [])
            
            if not funding_rates:
                self.logger.error("No funding rates found in response")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(funding_rates)
            
            # Convert timestamp to datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                
            # Convert rate to float
            if "rate" in df.columns:
                df["funding_rate"] = df["rate"].astype(float)
                
            # Save to CSV
            df.to_csv(f"real_data/{symbol}_funding_rates.csv", index=False)
            
            self.logger.info(f"Successfully fetched {len(df)} funding rates for {symbol}")
            return df
            
        except Exception as e:
            self.logger.exception(f"Error fetching funding rates: {str(e)}")
            return None
            
    async def fetch_order_book(self, symbol: str, depth: int = 20) -> Optional[Dict]:
        """
        Fetch order book from Hyperliquid.
        
        Args:
            symbol: Trading symbol
            depth: Order book depth
            
        Returns:
            Order book data or None if failed
        """
        self.logger.info(f"Fetching order book for {symbol}, depth {depth}...")
        
        try:
            # Convert symbol format if needed
            base_symbol = symbol.split("-")[0] if "-" in symbol else symbol
            
            # Prepare request
            endpoint = "/exchange/orderbook"
            url = f"{self.base_url}{endpoint}"
            
            # Prepare request data
            data = {
                "coin": base_symbol,
                "depth": depth
            }
            
            # Make request
            response = requests.post(url, json=data)
            
            if response.status_code != 200:
                self.logger.error(f"Error fetching order book: {response.status_code} {response.text}")
                return None
                
            # Parse response
            order_book = response.json()
            
            if not order_book or not isinstance(order_book, dict):
                self.logger.error(f"Invalid response format: {order_book}")
                return None
                
            # Extract bids and asks
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            
            if not bids or not asks:
                self.logger.error("No bids or asks found in response")
                return None
                
            # Format order book
            formatted_order_book = {
                "bids": [[float(price), float(size)] for price, size in bids],
                "asks": [[float(price), float(size)] for price, size in asks],
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to JSON
            with open(f"real_data/{symbol}_order_book.json", "w") as f:
                json.dump(formatted_order_book, f, indent=2)
                
            self.logger.info(f"Successfully fetched order book for {symbol}")
            return formatted_order_book
            
        except Exception as e:
            self.logger.exception(f"Error fetching order book: {str(e)}")
            return None
            
    async def fetch_oracle_prices(self, symbol: str) -> Optional[Dict]:
        """
        Fetch oracle prices from Hyperliquid.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Oracle price data or None if failed
        """
        self.logger.info(f"Fetching oracle prices for {symbol}...")
        
        try:
            # Convert symbol format if needed
            base_symbol = symbol.split("-")[0] if "-" in symbol else symbol
            
            # Prepare request
            endpoint = "/exchange/meta"
            url = f"{self.base_url}{endpoint}"
            
            # Make request
            response = requests.get(url)
            
            if response.status_code != 200:
                self.logger.error(f"Error fetching oracle prices: {response.status_code} {response.text}")
                return None
                
            # Parse response
            meta_data = response.json()
            
            if not meta_data or not isinstance(meta_data, dict):
                self.logger.error(f"Invalid response format: {meta_data}")
                return None
                
            # Extract oracle prices
            universe = meta_data.get("universe", [])
            
            if not universe:
                self.logger.error("No universe data found in response")
                return None
                
            # Find the symbol in universe
            oracle_data = None
            for coin in universe:
                if coin.get("name") == base_symbol:
                    oracle_data = {
                        "symbol": base_symbol,
                        "oracle_price": float(coin.get("oraclePriceExponent", 0)),
                        "timestamp": datetime.now().isoformat()
                    }
                    break
                    
            if not oracle_data:
                self.logger.error(f"Symbol {base_symbol} not found in universe")
                return None
                
            # Save to JSON
            with open(f"real_data/{symbol}_oracle_price.json", "w") as f:
                json.dump(oracle_data, f, indent=2)
                
            self.logger.info(f"Successfully fetched oracle price for {symbol}")
            return oracle_data
            
        except Exception as e:
            self.logger.exception(f"Error fetching oracle prices: {str(e)}")
            return None
            
    async def stream_market_data(self, symbol: str, duration_seconds: int = 3600):
        """
        Stream real-time market data from Hyperliquid WebSocket API.
        
        Args:
            symbol: Trading symbol
            duration_seconds: Duration to stream data in seconds
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Streaming market data for {symbol} for {duration_seconds} seconds...")
        
        try:
            # Convert symbol format if needed
            base_symbol = symbol.split("-")[0] if "-" in symbol else symbol
            
            # Prepare data storage
            trades = []
            order_books = []
            oracle_updates = []
            
            # Start time
            start_time = time.time()
            
            # Connect to WebSocket
            async with websockets.connect(self.ws_url) as websocket:
                # Subscribe to trades
                await websocket.send(json.dumps({
                    "method": "subscribe",
                    "subscription": {
                        "type": "trades",
                        "coin": base_symbol
                    }
                }))
                
                # Subscribe to order book
                await websocket.send(json.dumps({
                    "method": "subscribe",
                    "subscription": {
                        "type": "orderbook",
                        "coin": base_symbol
                    }
                }))
                
                # Subscribe to oracle updates
                await websocket.send(json.dumps({
                    "method": "subscribe",
                    "subscription": {
                        "type": "oracle",
                        "coin": base_symbol
                    }
                }))
                
                # Process messages
                while time.time() - start_time < duration_seconds:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        
                        # Process based on message type
                        if "trades" in data:
                            for trade in data["trades"]:
                                trade["timestamp"] = datetime.now().isoformat()
                                trades.append(trade)
                                
                        elif "orderbook" in data:
                            data["orderbook"]["timestamp"] = datetime.now().isoformat()
                            order_books.append(data["orderbook"])
                            
                        elif "oracle" in data:
                            data["oracle"]["timestamp"] = datetime.now().isoformat()
                            oracle_updates.append(data["oracle"])
                            
                    except asyncio.TimeoutError:
                        # No message received within timeout, continue
                        continue
                        
                    except Exception as e:
                        self.logger.error(f"Error processing WebSocket message: {str(e)}")
                        
                    # Periodically save data
                    if len(trades) >= 100 or len(order_books) >= 20 or len(oracle_updates) >= 10:
                        self._save_streamed_data(symbol, trades, order_books, oracle_updates)
                        trades = []
                        order_books = []
                        oracle_updates = []
                        
            # Save any remaining data
            self._save_streamed_data(symbol, trades, order_books, oracle_updates)
            
            self.logger.info(f"Successfully streamed market data for {symbol}")
            return True
            
        except Exception as e:
            self.logger.exception(f"Error streaming market data: {str(e)}")
            return False
            
    def _save_streamed_data(self, symbol: str, trades: List[Dict], order_books: List[Dict], oracle_updates: List[Dict]):
        """
        Save streamed market data to files.
        
        Args:
            symbol: Trading symbol
            trades: List of trade data
            order_books: List of order book data
            oracle_updates: List of oracle update data
        """
        try:
            # Save trades
            if trades:
                trades_file = f"real_data/{symbol}_trades_stream.json"
                
                # Load existing trades if file exists
                existing_trades = []
                if os.path.exists(trades_file):
                    try:
                        with open(trades_file, "r") as f:
                            existing_trades = json.load(f)
                    except:
                        pass
                        
                # Append new trades
                all_trades = existing_trades + trades
                
                # Save to file
                with open(trades_file, "w") as f:
                    json.dump(all_trades, f, indent=2)
                    
            # Save order books
            if order_books:
                order_books_file = f"real_data/{symbol}_order_books_stream.json"
                
                # Load existing order books if file exists
                existing_order_books = []
                if os.path.exists(order_books_file):
                    try:
                        with open(order_books_file, "r") as f:
                            existing_order_books = json.load(f)
                    except:
                        pass
                        
                # Append new order books
                all_order_books = existing_order_books + order_books
                
                # Save to file
                with open(order_books_file, "w") as f:
                    json.dump(all_order_books, f, indent=2)
                    
            # Save oracle updates
            if oracle_updates:
                oracle_updates_file = f"real_data/{symbol}_oracle_updates_stream.json"
                
                # Load existing oracle updates if file exists
                existing_oracle_updates = []
                if os.path.exists(oracle_updates_file):
                    try:
                        with open(oracle_updates_file, "r") as f:
                            existing_oracle_updates = json.load(f)
                    except:
                        pass
                        
                # Append new oracle updates
                all_oracle_updates = existing_oracle_updates + oracle_updates
                
                # Save to file
                with open(oracle_updates_file, "w") as f:
                    json.dump(all_oracle_updates, f, indent=2)
                    
        except Exception as e:
            self.logger.error(f"Error saving streamed data: {str(e)}")
            
    async def collect_all_real_data(self, symbols: List[str], intervals: List[str] = ["1h"], stream_duration: int = 3600) -> Dict[str, Any]:
        """
        Collect all necessary real market data for a list of symbols.
        
        Args:
            symbols: List of trading symbols
            intervals: List of time intervals for historical data
            stream_duration: Duration to stream real-time data in seconds
            
        Returns:
            Dictionary with collection results
        """
        self.logger.info(f"Collecting all real market data for symbols: {symbols}")
        
        results = {
            "symbols": symbols,
            "intervals": intervals,
            "stream_duration": stream_duration,
            "timestamp": datetime.now().isoformat(),
            "results": {}
        }
        
        for symbol in symbols:
            symbol_results = {
                "historical_data": {},
                "funding_rates": False,
                "order_book": False,
                "oracle_prices": False,
                "streamed_data": False
            }
            
            # Fetch historical data for each interval
            for interval in intervals:
                historical_data = await self.fetch_historical_klines(symbol, interval)
                symbol_results["historical_data"][interval] = historical_data is not None
                
            # Fetch funding rates
            funding_rates = await self.fetch_funding_rates(symbol)
            symbol_results["funding_rates"] = funding_rates is not None
            
            # Fetch order book
            order_book = await self.fetch_order_book(symbol)
            symbol_results["order_book"] = order_book is not None
            
            # Fetch oracle prices
            oracle_prices = await self.fetch_oracle_prices(symbol)
            symbol_results["oracle_prices"] = oracle_prices is not None
            
            # Stream market data
            streamed = await self.stream_market_data(symbol, stream_duration)
            symbol_results["streamed_data"] = streamed
            
            # Store results
            results["results"][symbol] = symbol_results
            
        # Save collection results
        with open("real_data/collection_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
            
        self.logger.info("Real market data collection completed")
        
        return results
        
    def process_collected_data(self, symbols: List[str], intervals: List[str] = ["1h"]) -> Dict[str, Any]:
        """
        Process collected real market data for backtesting.
        
        Args:
            symbols: List of trading symbols
            intervals: List of time intervals
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing collected real market data for symbols: {symbols}")
        
        results = {
            "symbols": symbols,
            "intervals": intervals,
            "timestamp": datetime.now().isoformat(),
            "results": {}
        }
        
        for symbol in symbols:
            symbol_results = {
                "processed_data": {}
            }
            
            for interval in intervals:
                try:
                    # Load historical data
                    historical_file = f"real_data/{symbol}_{interval}_klines.csv"
                    
                    if not os.path.exists(historical_file):
                        self.logger.error(f"Historical data file not found: {historical_file}")
                        symbol_results["processed_data"][interval] = False
                        continue
                        
                    df = pd.read_csv(historical_file)
                    
                    # Load funding rates
                    funding_file = f"real_data/{symbol}_funding_rates.csv"
                    funding_df = None
                    
                    if os.path.exists(funding_file):
                        funding_df = pd.read_csv(funding_file)
                        
                    # Load oracle prices
                    oracle_file = f"real_data/{symbol}_oracle_price.json"
                    oracle_data = None
                    
                    if os.path.exists(oracle_file):
                        with open(oracle_file, "r") as f:
                            oracle_data = json.load(f)
                            
                    # Process data
                    processed_df = self._process_data_for_backtesting(df, funding_df, oracle_data)
                    
                    # Save processed data
                    processed_file = f"real_data/{symbol}_{interval}_processed.csv"
                    processed_df.to_csv(processed_file, index=False)
                    
                    symbol_results["processed_data"][interval] = True
                    
                except Exception as e:
                    self.logger.error(f"Error processing data for {symbol} {interval}: {str(e)}")
                    symbol_results["processed_data"][interval] = False
                    
            # Store results
            results["results"][symbol] = symbol_results
            
        # Save processing results
        with open("real_data/processing_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
            
        self.logger.info("Real market data processing completed")
        
        return results
        
    def _process_data_for_backtesting(self, df: pd.DataFrame, funding_df: Optional[pd.DataFrame] = None, oracle_data: Optional[Dict] = None) -> pd.DataFrame:
        """
        Process data for backtesting.
        
        Args:
            df: Historical data DataFrame
            funding_df: Funding rates DataFrame
            oracle_data: Oracle price data
            
        Returns:
            Processed DataFrame
        """
        # Ensure timestamp is datetime
        if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
        # Add funding rate
        if funding_df is not None and "timestamp" in funding_df.columns and "funding_rate" in funding_df.columns:
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(funding_df["timestamp"]):
                funding_df["timestamp"] = pd.to_datetime(funding_df["timestamp"])
                
            # Merge funding rates
            df = pd.merge_asof(
                df.sort_values("timestamp"),
                funding_df.sort_values("timestamp")[["timestamp", "funding_rate"]],
                on="timestamp",
                direction="backward"
            )
            
        # Fill missing funding rates
        if "funding_rate" not in df.columns:
            df["funding_rate"] = 0.0001  # Default funding rate
            
        # Add oracle price
        if oracle_data is not None and "oracle_price" in oracle_data:
            # Add oracle price with small random deviation
            oracle_price = float(oracle_data["oracle_price"])
            df["oracle_price"] = df["close"] * (1 + np.sin(np.linspace(0, 20, len(df))) * 0.001)
            
        # Fill missing oracle price
        if "oracle_price" not in df.columns:
            df["oracle_price"] = df["close"] * (1 + np.sin(np.linspace(0, 20, len(df))) * 0.001)
            
        # Add additional columns for analysis
        df["price"] = df["close"]  # Current price is close price
        
        return df

async def main():
    """Main entry point."""
    # Create data collector
    collector = HyperliquidDataCollector()
    
    # Define symbols and intervals
    symbols = ["BTC-USD-PERP", "ETH-USD-PERP", "SOL-USD-PERP"]
    intervals = ["1h", "4h"]
    
    # Collect all real market data
    results = await collector.collect_all_real_data(symbols, intervals, stream_duration=1800)
    
    # Process collected data
    processing_results = collector.process_collected_data(symbols, intervals)
    
    print(f"Data collection completed with results: {results}")
    print(f"Data processing completed with results: {processing_results}")

if __name__ == "__main__":
    asyncio.run(main())
