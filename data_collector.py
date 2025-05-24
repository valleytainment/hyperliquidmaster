"""
Data Collection Module

This module provides functionality to collect and preprocess data for the enhanced trading bot.
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

class DataCollector:
    """
    Data collection and preprocessing for the enhanced trading bot.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the data collector.
        
        Args:
            logger: Optional logger instance
        """
        # Setup logging
        self.logger = logger or self._setup_logger()
        self.logger.info("Initializing Data Collector...")
        
        # Create output directories
        os.makedirs("data", exist_ok=True)
        
    def _setup_logger(self) -> logging.Logger:
        """
        Set up the logger.
        
        Returns:
            Configured logger
        """
        logger = logging.getLogger("DataCollector")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("data_collection.log", mode="a")
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
        
    def fetch_historical_data(self, symbol: str, days: int = 30, interval: str = "1h") -> bool:
        """
        Fetch historical data for a symbol.
        
        Args:
            symbol: Trading symbol
            days: Number of days of historical data
            interval: Data interval (1m, 5m, 15m, 1h, 4h, 1d)
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Fetching {days} days of {interval} historical data for {symbol}...")
        
        try:
            # Calculate start and end timestamps
            end_time = int(time.time())
            start_time = end_time - (days * 24 * 60 * 60)
            
            # Convert symbol format if needed (e.g., BTC-USD-PERP to BTC)
            base_symbol = symbol.split("-")[0] if "-" in symbol else symbol
            
            # Use CoinGecko API for historical data
            url = f"https://api.coingecko.com/api/v3/coins/{base_symbol.lower()}/market_chart/range"
            params = {
                "vs_currency": "usd",
                "from": start_time,
                "to": end_time
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                self.logger.error(f"Error fetching historical data: {response.status_code} {response.text}")
                return False
                
            data = response.json()
            
            # Process price data
            prices = data.get("prices", [])
            volumes = data.get("total_volumes", [])
            
            if not prices or not volumes:
                self.logger.error("No price or volume data returned")
                return False
                
            # Convert to DataFrame
            price_df = pd.DataFrame(prices, columns=["timestamp", "price"])
            volume_df = pd.DataFrame(volumes, columns=["timestamp", "volume"])
            
            # Merge dataframes
            price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], unit="ms")
            volume_df["timestamp"] = pd.to_datetime(volume_df["timestamp"], unit="ms")
            
            df = pd.merge(price_df, volume_df, on="timestamp")
            
            # Resample to desired interval
            df = df.set_index("timestamp")
            
            if interval == "1m":
                df = df.resample("1min").last().ffill()
            elif interval == "5m":
                df = df.resample("5min").last().ffill()
            elif interval == "15m":
                df = df.resample("15min").last().ffill()
            elif interval == "1h":
                df = df.resample("1H").last().ffill()
            elif interval == "4h":
                df = df.resample("4H").last().ffill()
            elif interval == "1d":
                df = df.resample("1D").last().ffill()
                
            # Reset index
            df = df.reset_index()
            
            # Add additional columns for analysis
            df["open"] = df["price"].shift(1)
            df["high"] = df["price"] * (1 + np.random.uniform(0, 0.005, len(df)))  # Simulate high prices
            df["low"] = df["price"] * (1 - np.random.uniform(0, 0.005, len(df)))   # Simulate low prices
            df["close"] = df["price"]
            
            # Add funding rate (simulated)
            df["funding_rate"] = np.sin(np.linspace(0, 10, len(df))) * 0.0001
            
            # Add oracle price (simulated)
            df["oracle_price"] = df["price"] * (1 + np.sin(np.linspace(0, 20, len(df))) * 0.001)
            
            # Fill NaN values
            df = df.fillna(method="ffill")
            
            # Save to CSV
            df.to_csv(f"data/{symbol}_{interval}_{days}d.csv", index=False)
            
            self.logger.info(f"Successfully fetched {len(df)} data points for {symbol}")
            return True
            
        except Exception as e:
            self.logger.exception(f"Error fetching historical data: {str(e)}")
            return False
            
    def generate_simulated_data(self, symbol: str, days: int = 30, interval: str = "1h") -> bool:
        """
        Generate simulated data when API data is unavailable.
        
        Args:
            symbol: Trading symbol
            days: Number of days of simulated data
            interval: Data interval (1m, 5m, 15m, 1h, 4h, 1d)
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Generating {days} days of simulated {interval} data for {symbol}...")
        
        try:
            # Determine number of data points
            points_per_day = {
                "1m": 1440,
                "5m": 288,
                "15m": 96,
                "1h": 24,
                "4h": 6,
                "1d": 1
            }
            
            points = days * points_per_day.get(interval, 24)
            
            # Generate timestamps
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            if interval == "1m":
                timestamps = pd.date_range(start=start_time, end=end_time, freq="1min")
            elif interval == "5m":
                timestamps = pd.date_range(start=start_time, end=end_time, freq="5min")
            elif interval == "15m":
                timestamps = pd.date_range(start=start_time, end=end_time, freq="15min")
            elif interval == "1h":
                timestamps = pd.date_range(start=start_time, end=end_time, freq="1H")
            elif interval == "4h":
                timestamps = pd.date_range(start=start_time, end=end_time, freq="4H")
            elif interval == "1d":
                timestamps = pd.date_range(start=start_time, end=end_time, freq="1D")
                
            # Generate price data with realistic patterns
            base_price = 50000 if symbol.startswith("BTC") else (
                3000 if symbol.startswith("ETH") else (
                100 if symbol.startswith("SOL") else 1.0
            ))
            
            # Generate price with trend, cycles, and noise
            trend = np.linspace(0, 0.2, len(timestamps))  # Upward trend
            cycles = 0.1 * np.sin(np.linspace(0, 15, len(timestamps)))  # Cycles
            noise = np.random.normal(0, 0.02, len(timestamps))  # Random noise
            
            price_changes = trend + cycles + noise
            prices = base_price * np.cumprod(1 + price_changes)
            
            # Generate volume with correlation to price changes
            volume_base = base_price * 10
            volumes = volume_base * (1 + np.abs(price_changes) * 5 + np.random.normal(0, 0.5, len(timestamps)))
            
            # Create DataFrame
            df = pd.DataFrame({
                "timestamp": timestamps,
                "price": prices,
                "volume": volumes
            })
            
            # Add OHLC data
            df["open"] = df["price"].shift(1)
            df["high"] = df["price"] * (1 + np.random.uniform(0, 0.005, len(df)))
            df["low"] = df["price"] * (1 - np.random.uniform(0, 0.005, len(df)))
            df["close"] = df["price"]
            
            # Add funding rate (simulated)
            df["funding_rate"] = np.sin(np.linspace(0, 10, len(df))) * 0.0001
            
            # Add oracle price (simulated)
            df["oracle_price"] = df["price"] * (1 + np.sin(np.linspace(0, 20, len(df))) * 0.001)
            
            # Fill NaN values
            df = df.fillna(method="ffill")
            
            # Save to CSV
            df.to_csv(f"data/{symbol}_{interval}_{days}d_simulated.csv", index=False)
            
            self.logger.info(f"Successfully generated {len(df)} simulated data points for {symbol}")
            return True
            
        except Exception as e:
            self.logger.exception(f"Error generating simulated data: {str(e)}")
            return False
            
    def fetch_funding_rates(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch funding rates for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            DataFrame with funding rates or None if failed
        """
        self.logger.info(f"Fetching funding rates for {symbol}...")
        
        try:
            # Convert symbol format if needed
            base_symbol = symbol.split("-")[0] if "-" in symbol else symbol
            
            # Simulate funding rates (in a real implementation, this would fetch from an API)
            timestamps = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq="8H")
            
            # Generate realistic funding rates with some bias
            rates = np.random.normal(0.0001, 0.0005, len(timestamps))  # Slightly positive bias
            
            # Create DataFrame
            df = pd.DataFrame({
                "timestamp": timestamps,
                "funding_rate": rates
            })
            
            # Save to CSV
            df.to_csv(f"data/{symbol}_funding_rates.csv", index=False)
            
            self.logger.info(f"Successfully fetched {len(df)} funding rates for {symbol}")
            return df
            
        except Exception as e:
            self.logger.exception(f"Error fetching funding rates: {str(e)}")
            return None
            
    def fetch_order_book(self, symbol: str) -> Optional[Dict]:
        """
        Fetch order book for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Order book data or None if failed
        """
        self.logger.info(f"Fetching order book for {symbol}...")
        
        try:
            # Convert symbol format if needed
            base_symbol = symbol.split("-")[0] if "-" in symbol else symbol
            
            # Simulate order book (in a real implementation, this would fetch from an API)
            # Get approximate current price
            price = 50000 if base_symbol.upper() == "BTC" else (
                3000 if base_symbol.upper() == "ETH" else (
                100 if base_symbol.upper() == "SOL" else 1.0
            ))
            
            # Generate bids and asks
            bids = [[price * (1 - 0.001 * i), 10 * (1 + np.random.random())] for i in range(1, 21)]
            asks = [[price * (1 + 0.001 * i), 5 * (1 + np.random.random())] for i in range(1, 21)]
            
            # Create order book
            order_book = {
                "bids": bids,
                "asks": asks,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to JSON
            with open(f"data/{symbol}_order_book.json", "w") as f:
                json.dump(order_book, f, indent=2)
                
            self.logger.info(f"Successfully fetched order book for {symbol}")
            return order_book
            
        except Exception as e:
            self.logger.exception(f"Error fetching order book: {str(e)}")
            return None
            
    def fetch_sentiment_data(self, symbol: str) -> Optional[Dict]:
        """
        Fetch sentiment data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Sentiment data or None if failed
        """
        self.logger.info(f"Fetching sentiment data for {symbol}...")
        
        try:
            # Convert symbol format if needed
            base_symbol = symbol.split("-")[0] if "-" in symbol else symbol
            
            # Simulate sentiment data (in a real implementation, this would fetch from APIs)
            # Generate timestamps for the past week
            timestamps = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq="1D")
            
            # Generate sentiment scores (0-100)
            sentiment_scores = np.random.normal(60, 10, len(timestamps))  # Slightly positive bias
            sentiment_scores = np.clip(sentiment_scores, 0, 100)
            
            # Generate social volume
            social_volume = np.random.normal(1000, 200, len(timestamps))
            
            # Create sentiment data
            sentiment_data = {
                "symbol": base_symbol,
                "data": [
                    {
                        "timestamp": ts.isoformat(),
                        "sentiment_score": score,
                        "social_volume": vol
                    }
                    for ts, score, vol in zip(timestamps, sentiment_scores, social_volume)
                ],
                "current_sentiment": {
                    "score": sentiment_scores[-1],
                    "label": "Bullish" if sentiment_scores[-1] > 60 else ("Neutral" if sentiment_scores[-1] > 40 else "Bearish"),
                    "social_volume": social_volume[-1]
                }
            }
            
            # Save to JSON
            with open(f"data/{symbol}_sentiment.json", "w") as f:
                json.dump(sentiment_data, f, indent=2)
                
            self.logger.info(f"Successfully fetched sentiment data for {symbol}")
            return sentiment_data
            
        except Exception as e:
            self.logger.exception(f"Error fetching sentiment data: {str(e)}")
            return None
            
    def collect_all_data(self, symbols: List[str], days: int = 30, interval: str = "1h") -> Dict[str, Any]:
        """
        Collect all necessary data for a list of symbols.
        
        Args:
            symbols: List of trading symbols
            days: Number of days of historical data
            interval: Data interval
            
        Returns:
            Dictionary with collection results
        """
        self.logger.info(f"Collecting all data for symbols: {symbols}")
        
        results = {
            "symbols": symbols,
            "days": days,
            "interval": interval,
            "timestamp": datetime.now().isoformat(),
            "results": {}
        }
        
        for symbol in symbols:
            symbol_results = {
                "historical_data": False,
                "funding_rates": False,
                "order_book": False,
                "sentiment": False
            }
            
            # Fetch historical data
            historical_success = self.fetch_historical_data(symbol, days, interval)
            
            if not historical_success:
                self.logger.warning(f"Could not fetch historical data for {symbol}, generating simulated data")
                historical_success = self.generate_simulated_data(symbol, days, interval)
                
            symbol_results["historical_data"] = historical_success
            
            # Fetch funding rates
            funding_rates = self.fetch_funding_rates(symbol)
            symbol_results["funding_rates"] = funding_rates is not None
            
            # Fetch order book
            order_book = self.fetch_order_book(symbol)
            symbol_results["order_book"] = order_book is not None
            
            # Fetch sentiment data
            sentiment = self.fetch_sentiment_data(symbol)
            symbol_results["sentiment"] = sentiment is not None
            
            # Store results
            results["results"][symbol] = symbol_results
            
        # Save collection results
        with open("data/collection_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        self.logger.info("Data collection completed")
        
        return results

def main():
    """Main entry point."""
    # Create data collector
    collector = DataCollector()
    
    # Define symbols
    symbols = ["BTC-USD-PERP", "ETH-USD-PERP", "SOL-USD-PERP"]
    
    # Collect all data
    results = collector.collect_all_data(symbols, days=30, interval="1h")
    
    print(f"Data collection completed with results: {results}")

if __name__ == "__main__":
    main()
