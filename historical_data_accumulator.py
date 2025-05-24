"""
Historical Data Accumulator

This module provides functionality to accumulate historical market data
for use in technical analysis and trading strategies.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class HistoricalDataAccumulator:
    """
    Accumulates and manages historical market data for multiple symbols.
    """
    
    def __init__(self, data_dir: str = "historical_data", max_data_points: int = 500):
        """
        Initialize the historical data accumulator.
        
        Args:
            data_dir: Directory to store historical data
            max_data_points: Maximum number of data points to keep per symbol
        """
        self.data_dir = data_dir
        self.max_data_points = max_data_points
        self.data = {}  # In-memory cache of historical data
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing data if available
        self._load_data()
        
    def _load_data(self):
        """
        Load historical data from disk.
        """
        try:
            for filename in os.listdir(self.data_dir):
                if filename.endswith(".json"):
                    symbol = filename.split(".")[0]
                    file_path = os.path.join(self.data_dir, filename)
                    
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        
                    self.data[symbol] = data
                    logger.info(f"Loaded {len(data)} historical data points for {symbol}")
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            
    def _save_data(self, symbol: str):
        """
        Save historical data to disk.
        
        Args:
            symbol: Symbol to save data for
        """
        try:
            if symbol not in self.data:
                return
                
            file_path = os.path.join(self.data_dir, f"{symbol}.json")
            
            with open(file_path, "w") as f:
                json.dump(self.data[symbol], f)
                
            logger.info(f"Saved {len(self.data[symbol])} historical data points for {symbol}")
        except Exception as e:
            logger.error(f"Error saving historical data for {symbol}: {str(e)}")
            
    def add_data_point(self, symbol: str, timestamp=None, price=None, volume=None, funding_rate=None, market_data: Dict = None):
        """
        Add a new data point for a symbol.
        
        Args:
            symbol: Symbol to add data for
            timestamp: Optional timestamp (will use current time if None)
            price: Optional price (will use market_data if None)
            volume: Optional volume (will use 0 if None)
            funding_rate: Optional funding rate (will use market_data if None)
            market_data: Optional market data dictionary
        """
        try:
            if symbol not in self.data:
                self.data[symbol] = []
                
            # Use market_data if provided, otherwise use individual parameters
            if market_data is not None:
                # Create a new data point from market_data
                data_point = {
                    "timestamp": market_data.get("timestamp", datetime.now().timestamp()),
                    "open": market_data.get("last_price", 0),
                    "high": market_data.get("last_price", 0),
                    "low": market_data.get("last_price", 0),
                    "close": market_data.get("last_price", 0),
                    "volume": market_data.get("volume", 0),
                    "funding_rate": market_data.get("funding_rate", 0)
                }
            else:
                # Create a new data point from individual parameters
                current_timestamp = timestamp if timestamp is not None else datetime.now().timestamp()
                current_price = price if price is not None else 0
                current_volume = volume if volume is not None else 0
                current_funding_rate = funding_rate if funding_rate is not None else 0
                
                data_point = {
                    "timestamp": current_timestamp,
                    "open": current_price,
                    "high": current_price,
                    "low": current_price,
                    "close": current_price,
                    "volume": current_volume,
                    "funding_rate": current_funding_rate
                }
            
            # Add the data point
            self.data[symbol].append(data_point)
            
            # Trim to max data points
            if len(self.data[symbol]) > self.max_data_points:
                self.data[symbol] = self.data[symbol][-self.max_data_points:]
                
            # Save to disk
            self._save_data(symbol)
            
            logger.info(f"Added data point for {symbol}: price={data_point['close']}")
        except Exception as e:
            logger.error(f"Error adding data point for {symbol}: {str(e)}")
            
    def get_dataframe(self, symbol: str, min_data_points: int = 1) -> Optional[pd.DataFrame]:
        """
        Get historical data as a pandas DataFrame.
        
        Args:
            symbol: Symbol to get data for
            min_data_points: Minimum number of data points required
            
        Returns:
            DataFrame with historical data or None if not enough data
        """
        try:
            if symbol not in self.data or len(self.data[symbol]) < min_data_points:
                logger.warning(f"Not enough data for {symbol}: {len(self.data.get(symbol, []))} < {min_data_points}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(self.data[symbol])
            
            # Convert timestamp to datetime
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
            
            # Set datetime as index
            df.set_index("datetime", inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error getting DataFrame for {symbol}: {str(e)}")
            return None
            
    def get_synthetic_dataframe(self, symbol: str, periods: int = 100) -> pd.DataFrame:
        """
        Get a synthetic DataFrame when not enough real data is available.
        This creates a DataFrame with the available real data plus synthetic
        historical data based on the latest price with realistic market patterns.
        
        Args:
            symbol: Symbol to get data for
            periods: Number of periods to generate
            
        Returns:
            DataFrame with real and synthetic historical data
        """
        try:
            # Get real data if available
            real_df = self.get_dataframe(symbol, min_data_points=1)
            
            if real_df is not None and len(real_df) > 0:
                # Use the latest price as the base for synthetic data
                latest_price = real_df["close"].iloc[-1]
                real_data_count = len(real_df)
            else:
                # No real data available, use a default price
                latest_price = 100.0
                real_data_count = 0
                
            # Generate synthetic data
            synthetic_count = max(0, periods - real_data_count)
            
            if synthetic_count > 0:
                logger.info(f"Generating {synthetic_count} synthetic data points for {symbol}")
                
                # Create timestamps for synthetic data
                now = datetime.now()
                timestamps = [(now - timedelta(minutes=i)).timestamp() for i in range(synthetic_count, 0, -1)]
                
                # Create more realistic price data with patterns that technical indicators can detect
                np.random.seed(42)  # For reproducibility
                
                # Generate a trend component (random walk with drift)
                trend = np.random.normal(0.0002, 0.001, synthetic_count).cumsum()
                
                # Generate a cyclical component (sine wave)
                cycle_length = synthetic_count // 4  # Complete 4 cycles
                cycle = 0.01 * np.sin(np.linspace(0, 4 * np.pi, synthetic_count))
                
                # Generate a volatility component (GARCH-like)
                volatility = np.random.normal(0, 0.002, synthetic_count)
                for i in range(1, synthetic_count):
                    volatility[i] = 0.9 * volatility[i-1] + 0.1 * volatility[i]
                
                # Combine components
                variations = trend + cycle + volatility
                closes = latest_price * (1 + variations)
                
                # Ensure prices are positive and realistic
                closes = np.maximum(closes, latest_price * 0.9)
                
                # Create synthetic data points with realistic OHLC relationships
                synthetic_data = []
                
                for i in range(synthetic_count):
                    # Create realistic OHLC values
                    close = closes[i]
                    
                    # Previous close becomes the open (or use slight variation for first point)
                    if i > 0:
                        open_price = closes[i-1]
                    else:
                        open_price = close * (1 - 0.001 * np.random.random())
                    
                    # High and low based on volatility
                    price_range = abs(close - open_price) + (close * 0.003 * (1 + abs(volatility[i])))
                    if close > open_price:
                        high = close + (price_range * 0.3)
                        low = open_price - (price_range * 0.2)
                    else:
                        high = open_price + (price_range * 0.2)
                        low = close - (price_range * 0.3)
                    
                    # Ensure high is always highest and low is always lowest
                    high = max(high, open_price, close)
                    low = min(low, open_price, close)
                    
                    # Volume correlates with volatility
                    volume = 1000 * (1 + 5 * abs(volatility[i]))
                    
                    # Funding rate correlates with recent trend
                    if i > 5:
                        recent_trend = (closes[i] - closes[i-5]) / closes[i-5]
                        funding_rate = 0.0001 * (1 + 10 * recent_trend)
                    else:
                        funding_rate = 0.0001
                    
                    synthetic_data.append({
                        "timestamp": timestamps[i],
                        "open": open_price,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                        "funding_rate": funding_rate
                    })
                    
                # Convert to DataFrame
                synthetic_df = pd.DataFrame(synthetic_data)
                synthetic_df["datetime"] = pd.to_datetime(synthetic_df["timestamp"], unit="s")
                synthetic_df.set_index("datetime", inplace=True)
                
                # Combine with real data if available
                if real_df is not None and len(real_df) > 0:
                    combined_df = pd.concat([synthetic_df, real_df])
                    combined_df = combined_df.sort_index()
                    return combined_df
                else:
                    return synthetic_df
            else:
                # Enough real data available
                return real_df
        except Exception as e:
            logger.error(f"Error getting synthetic DataFrame for {symbol}: {str(e)}")
            
            # Return a minimal synthetic DataFrame as fallback
            dates = pd.date_range(end=datetime.now(), periods=periods)
            df = pd.DataFrame({
                "timestamp": [d.timestamp() for d in dates],
                "open": [100.0] * periods,
                "high": [101.0] * periods,
                "low": [99.0] * periods,
                "close": [100.0] * periods,
                "volume": [1000] * periods,
                "funding_rate": [0.0001] * periods
            })
            df["datetime"] = dates
            df.set_index("datetime", inplace=True)
            return df
