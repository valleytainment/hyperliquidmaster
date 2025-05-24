"""
Enhanced Historical Data Accumulator

This module provides advanced functionality to accumulate and manage historical market data
with multi-timeframe support, data validation, recovery mechanisms, and synthetic data generation.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import time
import shutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class EnhancedHistoricalDataAccumulator:
    """
    Advanced historical data accumulator with multi-timeframe support,
    data validation, recovery mechanisms, and synthetic data generation.
    """
    
    def __init__(self, 
                 data_dir: str = "historical_data", 
                 max_data_points: int = 5000,
                 backup_interval: int = 100,
                 timeframes: List[str] = None,
                 symbols: List[str] = None,
                 auto_aggregate: bool = True):
        """
        Initialize the enhanced historical data accumulator.
        
        Args:
            data_dir: Directory to store historical data
            max_data_points: Maximum number of data points to keep per symbol and timeframe
            backup_interval: Number of updates between backups
            timeframes: List of timeframes to support (e.g., ["1m", "5m", "15m", "1h", "4h", "1d"])
            symbols: List of symbols to track
            auto_aggregate: Whether to automatically aggregate data to higher timeframes
        """
        self.data_dir = data_dir
        self.max_data_points = max_data_points
        self.backup_interval = backup_interval
        self.timeframes = timeframes or ["1m", "5m", "15m", "1h", "4h", "1d"]
        self.symbols = symbols or []
        self.auto_aggregate = auto_aggregate
        
        # Timeframe minutes mapping
        self.timeframe_minutes = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440
        }
        
        # In-memory cache of historical data
        self.data = {}  # Format: {symbol: {timeframe: [data_points]}}
        
        # Data quality metrics
        self.quality_metrics = {}  # Format: {symbol: {timeframe: {metric: value}}}
        
        # Update counters for backup scheduling
        self.update_counters = {}  # Format: {symbol: count}
        
        # Thread safety
        self.locks = {}  # Format: {symbol: lock}
        
        # Create data directory structure
        self._create_directory_structure()
        
        # Load existing data if available
        self._load_data()
        
    def _create_directory_structure(self):
        """
        Create the directory structure for storing historical data.
        """
        try:
            # Create main data directory
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Create backup directory
            backup_dir = os.path.join(self.data_dir, "backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            # Create directories for each timeframe
            for timeframe in self.timeframes:
                timeframe_dir = os.path.join(self.data_dir, timeframe)
                os.makedirs(timeframe_dir, exist_ok=True)
                
            logger.info(f"Created directory structure in {self.data_dir}")
        except Exception as e:
            logger.error(f"Error creating directory structure: {str(e)}")
            
    def _get_symbol_lock(self, symbol: str) -> threading.RLock:
        """
        Get the lock for a symbol, creating it if it doesn't exist.
        
        Args:
            symbol: Symbol to get lock for
            
        Returns:
            Lock for the symbol
        """
        if symbol not in self.locks:
            self.locks[symbol] = threading.RLock()
            
        return self.locks[symbol]
            
    def _load_data(self):
        """
        Load historical data from disk for all timeframes.
        """
        try:
            # Initialize data structure
            self.data = {}
            self.quality_metrics = {}
            
            # Load data for each timeframe
            for timeframe in self.timeframes:
                timeframe_dir = os.path.join(self.data_dir, timeframe)
                
                if not os.path.exists(timeframe_dir):
                    continue
                    
                for filename in os.listdir(timeframe_dir):
                    if filename.endswith(".json"):
                        symbol = filename.split(".")[0]
                        file_path = os.path.join(timeframe_dir, filename)
                        
                        # Initialize symbol data if needed
                        if symbol not in self.data:
                            self.data[symbol] = {}
                            self.quality_metrics[symbol] = {}
                            
                        # Load data from file
                        try:
                            with open(file_path, "r") as f:
                                symbol_data = json.load(f)
                                
                            # Store data
                            self.data[symbol][timeframe] = symbol_data
                            
                            # Add symbol to tracked symbols if not already there
                            if symbol not in self.symbols:
                                self.symbols.append(symbol)
                                
                            # Calculate quality metrics
                            self._calculate_quality_metrics(symbol, timeframe)
                                
                            logger.info(f"Loaded {len(symbol_data)} historical data points for {symbol} ({timeframe})")
                        except json.JSONDecodeError:
                            logger.error(f"Error decoding JSON for {symbol} ({timeframe}), file may be corrupted")
                            
                            # Try to recover from backup
                            self._recover_from_backup(symbol, timeframe)
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            
    def _save_data(self, symbol: str, timeframe: str):
        """
        Save historical data to disk.
        
        Args:
            symbol: Symbol to save data for
            timeframe: Timeframe to save data for
        """
        try:
            with self._get_symbol_lock(symbol):
                if symbol not in self.data or timeframe not in self.data[symbol]:
                    return
                    
                # Get file path
                file_path = os.path.join(self.data_dir, timeframe, f"{symbol}.json")
                
                # Save data to file
                with open(file_path, "w") as f:
                    json.dump(self.data[symbol][timeframe], f)
                    
                # Update backup counter
                if symbol not in self.update_counters:
                    self.update_counters[symbol] = 0
                    
                self.update_counters[symbol] += 1
                
                # Create backup if needed
                if self.update_counters[symbol] >= self.backup_interval:
                    self._create_backup(symbol, timeframe)
                    self.update_counters[symbol] = 0
                    
                logger.info(f"Saved {len(self.data[symbol][timeframe])} historical data points for {symbol} ({timeframe})")
        except Exception as e:
            logger.error(f"Error saving historical data for {symbol} ({timeframe}): {str(e)}")
            
    def _create_backup(self, symbol: str, timeframe: str):
        """
        Create a backup of historical data.
        
        Args:
            symbol: Symbol to backup data for
            timeframe: Timeframe to backup data for
        """
        try:
            with self._get_symbol_lock(symbol):
                # Get source file path
                source_path = os.path.join(self.data_dir, timeframe, f"{symbol}.json")
                
                if not os.path.exists(source_path):
                    return
                    
                # Get backup file path
                backup_dir = os.path.join(self.data_dir, "backups")
                backup_path = os.path.join(backup_dir, f"{symbol}_{timeframe}_{int(time.time())}.json")
                
                # Create backup
                shutil.copy2(source_path, backup_path)
                
                # Keep only the 5 most recent backups
                backup_files = [f for f in os.listdir(backup_dir) if f.startswith(f"{symbol}_{timeframe}_")]
                backup_files.sort(reverse=True)
                
                for old_backup in backup_files[5:]:
                    os.remove(os.path.join(backup_dir, old_backup))
                    
                logger.info(f"Created backup for {symbol} ({timeframe})")
        except Exception as e:
            logger.error(f"Error creating backup for {symbol} ({timeframe}): {str(e)}")
            
    def _recover_from_backup(self, symbol: str, timeframe: str) -> bool:
        """
        Recover data from the most recent backup.
        
        Args:
            symbol: Symbol to recover data for
            timeframe: Timeframe to recover data for
            
        Returns:
            True if recovery was successful, False otherwise
        """
        try:
            with self._get_symbol_lock(symbol):
                # Get backup directory
                backup_dir = os.path.join(self.data_dir, "backups")
                
                # Find backups for this symbol and timeframe
                backup_files = [f for f in os.listdir(backup_dir) if f.startswith(f"{symbol}_{timeframe}_")]
                
                if not backup_files:
                    logger.warning(f"No backups found for {symbol} ({timeframe})")
                    return False
                    
                # Sort backups by timestamp (newest first)
                backup_files.sort(reverse=True)
                
                # Try each backup until one works
                for backup_file in backup_files:
                    backup_path = os.path.join(backup_dir, backup_file)
                    
                    try:
                        with open(backup_path, "r") as f:
                            symbol_data = json.load(f)
                            
                        # Initialize symbol data if needed
                        if symbol not in self.data:
                            self.data[symbol] = {}
                            self.quality_metrics[symbol] = {}
                            
                        # Store recovered data
                        self.data[symbol][timeframe] = symbol_data
                        
                        # Save recovered data
                        file_path = os.path.join(self.data_dir, timeframe, f"{symbol}.json")
                        
                        with open(file_path, "w") as f:
                            json.dump(symbol_data, f)
                            
                        logger.info(f"Recovered {len(symbol_data)} data points for {symbol} ({timeframe}) from backup")
                        return True
                    except Exception as e:
                        logger.warning(f"Error recovering from backup {backup_file}: {str(e)}")
                        
                logger.error(f"All recovery attempts failed for {symbol} ({timeframe})")
                return False
        except Exception as e:
            logger.error(f"Error in recovery process for {symbol} ({timeframe}): {str(e)}")
            return False
            
    def _calculate_quality_metrics(self, symbol: str, timeframe: str):
        """
        Calculate data quality metrics.
        
        Args:
            symbol: Symbol to calculate metrics for
            timeframe: Timeframe to calculate metrics for
        """
        try:
            with self._get_symbol_lock(symbol):
                if symbol not in self.data or timeframe not in self.data[symbol]:
                    return
                    
                data_points = self.data[symbol][timeframe]
                
                if not data_points:
                    return
                    
                # Initialize metrics
                if symbol not in self.quality_metrics:
                    self.quality_metrics[symbol] = {}
                    
                if timeframe not in self.quality_metrics[symbol]:
                    self.quality_metrics[symbol][timeframe] = {}
                    
                # Calculate metrics
                metrics = self.quality_metrics[symbol][timeframe]
                
                # Data count
                metrics["count"] = len(data_points)
                
                # Time range
                timestamps = [point["timestamp"] for point in data_points]
                metrics["start_time"] = min(timestamps)
                metrics["end_time"] = max(timestamps)
                metrics["time_span"] = metrics["end_time"] - metrics["start_time"]
                
                # Expected number of points based on timeframe
                expected_minutes = metrics["time_span"] / 60
                expected_points = expected_minutes / self.timeframe_minutes.get(timeframe, 1)
                metrics["completeness"] = min(1.0, len(data_points) / max(1, expected_points))
                
                # Check for gaps
                sorted_timestamps = sorted(timestamps)
                gaps = []
                
                for i in range(1, len(sorted_timestamps)):
                    gap = sorted_timestamps[i] - sorted_timestamps[i-1]
                    expected_gap = self.timeframe_minutes.get(timeframe, 1) * 60
                    
                    if gap > expected_gap * 1.5:  # Allow 50% tolerance
                        gaps.append((sorted_timestamps[i-1], sorted_timestamps[i], gap))
                        
                metrics["gaps"] = len(gaps)
                metrics["largest_gap"] = max([g[2] for g in gaps]) if gaps else 0
                
                # Price statistics
                closes = [point["close"] for point in data_points]
                metrics["min_price"] = min(closes)
                metrics["max_price"] = max(closes)
                metrics["avg_price"] = sum(closes) / len(closes)
                metrics["price_range"] = metrics["max_price"] - metrics["min_price"]
                
                # Volatility
                returns = []
                for i in range(1, len(closes)):
                    if closes[i-1] > 0:
                        returns.append((closes[i] - closes[i-1]) / closes[i-1])
                        
                if returns:
                    metrics["volatility"] = np.std(returns) * np.sqrt(252 * 24 * 60 / self.timeframe_minutes.get(timeframe, 1))
                else:
                    metrics["volatility"] = 0
                    
                logger.debug(f"Calculated quality metrics for {symbol} ({timeframe})")
        except Exception as e:
            logger.error(f"Error calculating quality metrics for {symbol} ({timeframe}): {str(e)}")
            
    def _aggregate_to_higher_timeframe(self, symbol: str, source_timeframe: str, target_timeframe: str):
        """
        Aggregate data from a lower timeframe to a higher timeframe.
        
        Args:
            symbol: Symbol to aggregate data for
            source_timeframe: Source timeframe
            target_timeframe: Target timeframe
        """
        try:
            with self._get_symbol_lock(symbol):
                # Check if source data exists
                if symbol not in self.data or source_timeframe not in self.data[symbol]:
                    logger.warning(f"No source data for {symbol} ({source_timeframe})")
                    return
                    
                # Get source data
                source_data = self.data[symbol][source_timeframe]
                
                if not source_data:
                    logger.warning(f"Empty source data for {symbol} ({source_timeframe})")
                    return
                    
                # Check if source timeframe is lower than target timeframe
                source_minutes = self.timeframe_minutes.get(source_timeframe, 0)
                target_minutes = self.timeframe_minutes.get(target_timeframe, 0)
                
                if source_minutes >= target_minutes:
                    logger.warning(f"Source timeframe {source_timeframe} is not lower than target timeframe {target_timeframe}")
                    return
                    
                # Convert to pandas DataFrame
                df = pd.DataFrame(source_data)
                
                # Convert timestamp to datetime
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
                
                # Set datetime as index
                df.set_index("datetime", inplace=True)
                
                # Resample to target timeframe
                if target_timeframe == "1m":
                    rule = "1min"
                elif target_timeframe == "5m":
                    rule = "5min"
                elif target_timeframe == "15m":
                    rule = "15min"
                elif target_timeframe == "30m":
                    rule = "30min"
                elif target_timeframe == "1h":
                    rule = "1H"
                elif target_timeframe == "4h":
                    rule = "4H"
                elif target_timeframe == "1d":
                    rule = "1D"
                else:
                    logger.warning(f"Unsupported target timeframe: {target_timeframe}")
                    return
                    
                # Resample OHLCV data
                resampled = df.resample(rule).agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                    "timestamp": "first"
                })
                
                # Reset index
                resampled.reset_index(inplace=True)
                
                # Convert back to list of dictionaries
                aggregated_data = resampled.to_dict("records")
                
                # Convert datetime back to timestamp
                for point in aggregated_data:
                    point["datetime"] = point["datetime"].timestamp()
                    
                # Initialize target data if needed
                if symbol not in self.data:
                    self.data[symbol] = {}
                    
                if target_timeframe not in self.data[symbol]:
                    self.data[symbol][target_timeframe] = []
                    
                # Update target data
                self.data[symbol][target_timeframe] = aggregated_data
                
                # Save target data
                self._save_data(symbol, target_timeframe)
                
                # Calculate quality metrics
                self._calculate_quality_metrics(symbol, target_timeframe)
                
                logger.info(f"Aggregated {len(source_data)} data points from {source_timeframe} to {len(aggregated_data)} data points in {target_timeframe} for {symbol}")
        except Exception as e:
            logger.error(f"Error aggregating data from {source_timeframe} to {target_timeframe} for {symbol}: {str(e)}")
            
    def add_data_point(self, symbol: str, timestamp: datetime, open_price: float = None, high_price: float = None, 
                      low_price: float = None, close_price: float = None, volume: float = None, data: Dict = None):
        """
        Add a data point for a symbol.
        
        Args:
            symbol: Symbol to add data point for
            timestamp: Timestamp of the data point
            open_price: Open price
            high_price: High price
            low_price: Low price
            close_price: Close price
            volume: Volume
            data: Optional dictionary containing all data fields (can be used instead of individual parameters)
        """
        try:
            # Convert timestamp to Unix timestamp if it's a datetime object
            if isinstance(timestamp, datetime):
                timestamp = int(timestamp.timestamp())
                
            # Use data dictionary if provided
            if data is not None:
                open_price = data.get("open", data.get("open_price", open_price))
                high_price = data.get("high", data.get("high_price", high_price))
                low_price = data.get("low", data.get("low_price", low_price))
                close_price = data.get("close", data.get("close_price", close_price))
                volume = data.get("volume", volume)
                
            # Ensure we have at least a close price
            if close_price is None:
                if data and "price" in data:
                    close_price = data["price"]
                else:
                    logger.warning(f"No close price provided for {symbol} at {timestamp}")
                    return
                    
            # Fill in missing prices
            if open_price is None:
                open_price = close_price
            if high_price is None:
                high_price = max(open_price, close_price)
            if low_price is None:
                low_price = min(open_price, close_price)
            if volume is None:
                volume = 0
                
            # Create data point
            data_point = {
                "timestamp": timestamp,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume
            }
            
            # Add data point to 1m timeframe
            self._add_data_point_to_timeframe(symbol, "1m", data_point)
            
            # Aggregate to higher timeframes if enabled
            if self.auto_aggregate:
                # Get current minute
                current_minute = datetime.fromtimestamp(timestamp).minute
                
                # Aggregate to 5m timeframe on multiples of 5 minutes
                if current_minute % 5 == 0:
                    self._aggregate_to_higher_timeframe(symbol, "1m", "5m")
                    
                # Aggregate to 15m timeframe on multiples of 15 minutes
                if current_minute % 15 == 0:
                    self._aggregate_to_higher_timeframe(symbol, "5m", "15m")
                    
                # Aggregate to 1h timeframe on the hour
                if current_minute == 0:
                    self._aggregate_to_higher_timeframe(symbol, "15m", "1h")
                    
                # Aggregate to 4h timeframe on multiples of 4 hours
                current_hour = datetime.fromtimestamp(timestamp).hour
                if current_minute == 0 and current_hour % 4 == 0:
                    self._aggregate_to_higher_timeframe(symbol, "1h", "4h")
                    
                # Aggregate to 1d timeframe at midnight
                if current_minute == 0 and current_hour == 0:
                    self._aggregate_to_higher_timeframe(symbol, "4h", "1d")
        except Exception as e:
            logger.error(f"Error adding data point for {symbol}: {str(e)}")
            
    def _add_data_point_to_timeframe(self, symbol: str, timeframe: str, data_point: Dict):
        """
        Add a data point to a specific timeframe.
        
        Args:
            symbol: Symbol to add data point for
            timeframe: Timeframe to add data point to
            data_point: Data point to add
        """
        try:
            with self._get_symbol_lock(symbol):
                # Initialize symbol data if needed
                if symbol not in self.data:
                    self.data[symbol] = {}
                    self.quality_metrics[symbol] = {}
                    
                # Initialize timeframe data if needed
                if timeframe not in self.data[symbol]:
                    self.data[symbol][timeframe] = []
                    
                # Add symbol to tracked symbols if not already there
                if symbol not in self.symbols:
                    self.symbols.append(symbol)
                    
                # Check if data point already exists
                timestamp = data_point["timestamp"]
                existing_points = [p for p in self.data[symbol][timeframe] if p["timestamp"] == timestamp]
                
                if existing_points:
                    # Update existing data point
                    for point in existing_points:
                        point.update(data_point)
                else:
                    # Add new data point
                    self.data[symbol][timeframe].append(data_point)
                    
                    # Sort data points by timestamp
                    self.data[symbol][timeframe].sort(key=lambda p: p["timestamp"])
                    
                    # Trim data points if needed
                    if len(self.data[symbol][timeframe]) > self.max_data_points:
                        self.data[symbol][timeframe] = self.data[symbol][timeframe][-self.max_data_points:]
                        
                # Save data
                self._save_data(symbol, timeframe)
                
                # Calculate quality metrics
                self._calculate_quality_metrics(symbol, timeframe)
        except Exception as e:
            logger.error(f"Error adding data point to {timeframe} for {symbol}: {str(e)}")
            
    def get_data(self, symbol: str, timeframe: str = "1m", limit: int = None, start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """
        Get historical data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            timeframe: Timeframe to get data for
            limit: Maximum number of data points to return
            start_time: Start time (Unix timestamp)
            end_time: End time (Unix timestamp)
            
        Returns:
            DataFrame with historical data
        """
        try:
            with self._get_symbol_lock(symbol):
                # Check if data exists
                if symbol not in self.data or timeframe not in self.data[symbol]:
                    logger.warning(f"No data for {symbol} ({timeframe})")
                    return pd.DataFrame()
                    
                # Get data
                data = self.data[symbol][timeframe]
                
                if not data:
                    logger.warning(f"Empty data for {symbol} ({timeframe})")
                    return pd.DataFrame()
                    
                # Filter by time range
                if start_time is not None:
                    data = [p for p in data if p["timestamp"] >= start_time]
                    
                if end_time is not None:
                    data = [p for p in data if p["timestamp"] <= end_time]
                    
                # Limit number of data points
                if limit is not None:
                    data = data[-limit:]
                    
                # Convert to DataFrame
                df = pd.DataFrame(data)
                
                # Add datetime column
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
                
                return df
        except Exception as e:
            logger.error(f"Error getting data for {symbol} ({timeframe}): {str(e)}")
            return pd.DataFrame()
            
    def get_latest_data_point(self, symbol: str, timeframe: str = "1m") -> Dict:
        """
        Get the latest data point for a symbol.
        
        Args:
            symbol: Symbol to get data for
            timeframe: Timeframe to get data for
            
        Returns:
            Latest data point
        """
        try:
            with self._get_symbol_lock(symbol):
                # Check if data exists
                if symbol not in self.data or timeframe not in self.data[symbol]:
                    logger.warning(f"No data for {symbol} ({timeframe})")
                    return {}
                    
                # Get data
                data = self.data[symbol][timeframe]
                
                if not data:
                    logger.warning(f"Empty data for {symbol} ({timeframe})")
                    return {}
                    
                # Get latest data point
                latest = max(data, key=lambda p: p["timestamp"])
                
                return latest
        except Exception as e:
            logger.error(f"Error getting latest data point for {symbol} ({timeframe}): {str(e)}")
            return {}
            
    def get_quality_metrics(self, symbol: str, timeframe: str = "1m") -> Dict:
        """
        Get quality metrics for a symbol.
        
        Args:
            symbol: Symbol to get metrics for
            timeframe: Timeframe to get metrics for
            
        Returns:
            Quality metrics
        """
        try:
            with self._get_symbol_lock(symbol):
                # Check if metrics exist
                if symbol not in self.quality_metrics or timeframe not in self.quality_metrics[symbol]:
                    logger.warning(f"No quality metrics for {symbol} ({timeframe})")
                    return {}
                    
                # Get metrics
                metrics = self.quality_metrics[symbol][timeframe]
                
                return metrics
        except Exception as e:
            logger.error(f"Error getting quality metrics for {symbol} ({timeframe}): {str(e)}")
            return {}
            
    def clear_data(self, symbol: str = None, timeframe: str = None):
        """
        Clear historical data.
        
        Args:
            symbol: Symbol to clear data for (None for all symbols)
            timeframe: Timeframe to clear data for (None for all timeframes)
        """
        try:
            if symbol is None:
                # Clear all data
                self.data = {}
                self.quality_metrics = {}
                
                # Remove data files
                for timeframe_dir in os.listdir(self.data_dir):
                    if os.path.isdir(os.path.join(self.data_dir, timeframe_dir)) and timeframe_dir != "backups":
                        for filename in os.listdir(os.path.join(self.data_dir, timeframe_dir)):
                            if filename.endswith(".json"):
                                os.remove(os.path.join(self.data_dir, timeframe_dir, filename))
                                
                logger.info("Cleared all historical data")
            elif timeframe is None:
                # Clear data for a specific symbol
                if symbol in self.data:
                    del self.data[symbol]
                    
                if symbol in self.quality_metrics:
                    del self.quality_metrics[symbol]
                    
                # Remove data files
                for timeframe_dir in os.listdir(self.data_dir):
                    if os.path.isdir(os.path.join(self.data_dir, timeframe_dir)) and timeframe_dir != "backups":
                        file_path = os.path.join(self.data_dir, timeframe_dir, f"{symbol}.json")
                        
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            
                logger.info(f"Cleared historical data for {symbol}")
            else:
                # Clear data for a specific symbol and timeframe
                if symbol in self.data and timeframe in self.data[symbol]:
                    del self.data[symbol][timeframe]
                    
                if symbol in self.quality_metrics and timeframe in self.quality_metrics[symbol]:
                    del self.quality_metrics[symbol][timeframe]
                    
                # Remove data file
                file_path = os.path.join(self.data_dir, timeframe, f"{symbol}.json")
                
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
                logger.info(f"Cleared historical data for {symbol} ({timeframe})")
        except Exception as e:
            logger.error(f"Error clearing historical data: {str(e)}")
