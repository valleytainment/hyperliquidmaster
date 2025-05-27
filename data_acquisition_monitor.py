"""
Data Acquisition Monitor Module

This module provides robust monitoring, logging, and user notification for data acquisition
processes. It ensures transparency in data quality and provides fallback mechanisms when
real market data cannot be obtained.
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
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_acquisition_monitor.log")
    ]
)
logger = logging.getLogger(__name__)

class DataAcquisitionMonitor:
    """
    Monitor and manage data acquisition processes, providing fallback mechanisms
    and user notifications when real market data cannot be obtained.
    """
    
    def __init__(self, base_dir: str = "real_market_data", notification_callback: Optional[Callable] = None):
        """
        Initialize the data acquisition monitor.
        
        Args:
            base_dir: Base directory for data storage
            notification_callback: Optional callback function for user notifications
        """
        self.base_dir = base_dir
        self.notification_callback = notification_callback
        
        # Create directories
        os.makedirs(f"{self.base_dir}/logs", exist_ok=True)
        os.makedirs(f"{self.base_dir}/reports", exist_ok=True)
        
        # Initialize metrics
        self.reset_metrics()
        
        logger.info("Data Acquisition Monitor initialized")
        
    def reset_metrics(self):
        """Reset acquisition metrics."""
        self.metrics = {
            "start_time": datetime.now(),
            "end_time": None,
            "total_symbols": 0,
            "total_intervals": 0,
            "successful_acquisitions": 0,
            "failed_acquisitions": 0,
            "synthetic_fallbacks": 0,
            "real_data_ratio": 0.0,
            "symbols_with_real_data": [],
            "symbols_with_synthetic_data": [],
            "api_errors": {
                "hyperliquid": 0,
                "binance": 0,
                "coingecko": 0
            },
            "data_quality": {}
        }
        
    def log_acquisition_attempt(self, symbol: str, interval: str, source: str):
        """
        Log an acquisition attempt.
        
        Args:
            symbol: Symbol being acquired
            interval: Interval being acquired
            source: Source being attempted
        """
        logger.info(f"Attempting to acquire data for {symbol}, interval {interval} from {source}")
        
    def log_acquisition_success(self, symbol: str, interval: str, source: str, data_quality: Dict):
        """
        Log a successful acquisition.
        
        Args:
            symbol: Symbol acquired
            interval: Interval acquired
            source: Source used
            data_quality: Data quality metrics
        """
        logger.info(f"Successfully acquired data for {symbol}, interval {interval} from {source}")
        
        # Update metrics
        self.metrics["successful_acquisitions"] += 1
        
        # Track symbols with real data
        if not data_quality.get("is_synthetic", True):
            if symbol not in self.metrics["symbols_with_real_data"]:
                self.metrics["symbols_with_real_data"].append(symbol)
        else:
            if symbol not in self.metrics["symbols_with_synthetic_data"]:
                self.metrics["symbols_with_synthetic_data"].append(symbol)
                self.metrics["synthetic_fallbacks"] += 1
                
        # Store data quality
        if symbol not in self.metrics["data_quality"]:
            self.metrics["data_quality"][symbol] = {}
        self.metrics["data_quality"][symbol][interval] = data_quality
        
        # Update real data ratio
        total_symbols = len(set(self.metrics["symbols_with_real_data"] + self.metrics["symbols_with_synthetic_data"]))
        if total_symbols > 0:
            self.metrics["real_data_ratio"] = len(self.metrics["symbols_with_real_data"]) / total_symbols
            
    def log_acquisition_failure(self, symbol: str, interval: str, source: str, error: str):
        """
        Log an acquisition failure.
        
        Args:
            symbol: Symbol being acquired
            interval: Interval being acquired
            source: Source being attempted
            error: Error message
        """
        logger.error(f"Failed to acquire data for {symbol}, interval {interval} from {source}: {error}")
        
        # Update metrics
        self.metrics["failed_acquisitions"] += 1
        
        # Track API errors
        if source in self.metrics["api_errors"]:
            self.metrics["api_errors"][source] += 1
            
    def log_synthetic_fallback(self, symbol: str, interval: str):
        """
        Log a fallback to synthetic data.
        
        Args:
            symbol: Symbol being acquired
            interval: Interval being acquired
        """
        logger.warning(f"Falling back to synthetic data for {symbol}, interval {interval}")
        
        # Update metrics
        if symbol not in self.metrics["symbols_with_synthetic_data"]:
            self.metrics["symbols_with_synthetic_data"].append(symbol)
            self.metrics["synthetic_fallbacks"] += 1
            
        # Update real data ratio
        total_symbols = len(set(self.metrics["symbols_with_real_data"] + self.metrics["symbols_with_synthetic_data"]))
        if total_symbols > 0:
            self.metrics["real_data_ratio"] = len(self.metrics["symbols_with_real_data"]) / total_symbols
            
    def notify_user_if_needed(self, force: bool = False):
        """
        Notify user if needed based on acquisition metrics.
        
        Args:
            force: Force notification regardless of metrics
        """
        # Check if notification is needed
        notification_needed = force or (
            self.metrics["real_data_ratio"] < 0.5 and
            self.metrics["synthetic_fallbacks"] > 0
        )
        
        if notification_needed and self.notification_callback:
            message = self.generate_notification_message()
            self.notification_callback(message)
            
    def generate_notification_message(self) -> str:
        """
        Generate a notification message based on acquisition metrics.
        
        Returns:
            Notification message
        """
        message = "Data Acquisition Alert:\n\n"
        
        # Add summary
        message += f"Real Data Ratio: {self.metrics['real_data_ratio']:.2f}\n"
        message += f"Successful Acquisitions: {self.metrics['successful_acquisitions']}\n"
        message += f"Failed Acquisitions: {self.metrics['failed_acquisitions']}\n"
        message += f"Synthetic Fallbacks: {self.metrics['synthetic_fallbacks']}\n\n"
        
        # Add API errors
        message += "API Errors:\n"
        for source, count in self.metrics["api_errors"].items():
            message += f"  {source}: {count}\n"
        message += "\n"
        
        # Add symbols with synthetic data
        if self.metrics["symbols_with_synthetic_data"]:
            message += "Symbols using synthetic data:\n"
            for symbol in self.metrics["symbols_with_synthetic_data"]:
                message += f"  {symbol}\n"
            message += "\n"
            
        # Add recommendation
        if self.metrics["real_data_ratio"] < 0.5:
            message += "Recommendation: Consider checking API access and connectivity issues.\n"
            message += "The trading bot is currently relying heavily on synthetic data, which may affect performance.\n"
            
        return message
        
    def generate_acquisition_report(self) -> Dict:
        """
        Generate a comprehensive acquisition report.
        
        Returns:
            Acquisition report
        """
        # Update end time
        self.metrics["end_time"] = datetime.now()
        
        # Calculate duration
        duration = (self.metrics["end_time"] - self.metrics["start_time"]).total_seconds()
        
        # Create report
        report = {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "metrics": self.metrics,
            "recommendations": self.generate_recommendations()
        }
        
        # Save report
        report_path = f"{self.base_dir}/reports/acquisition_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
            
        return report
        
    def generate_recommendations(self) -> List[str]:
        """
        Generate recommendations based on acquisition metrics.
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check real data ratio
        if self.metrics["real_data_ratio"] < 0.25:
            recommendations.append("CRITICAL: Almost all data is synthetic. Check API access and connectivity immediately.")
        elif self.metrics["real_data_ratio"] < 0.5:
            recommendations.append("WARNING: More than half of the data is synthetic. Consider checking API access and connectivity.")
        elif self.metrics["real_data_ratio"] < 0.75:
            recommendations.append("NOTICE: Some data is synthetic. Monitor API access and connectivity.")
            
        # Check API errors
        for source, count in self.metrics["api_errors"].items():
            if count > 10:
                recommendations.append(f"CRITICAL: High number of {source} API errors ({count}). Check API access and connectivity.")
            elif count > 5:
                recommendations.append(f"WARNING: Multiple {source} API errors ({count}). Monitor API access and connectivity.")
            elif count > 0:
                recommendations.append(f"NOTICE: Some {source} API errors ({count}).")
                
        # Check synthetic fallbacks
        if self.metrics["synthetic_fallbacks"] > 10:
            recommendations.append(f"CRITICAL: High number of synthetic fallbacks ({self.metrics['synthetic_fallbacks']}). Check data sources.")
        elif self.metrics["synthetic_fallbacks"] > 5:
            recommendations.append(f"WARNING: Multiple synthetic fallbacks ({self.metrics['synthetic_fallbacks']}). Monitor data sources.")
        elif self.metrics["synthetic_fallbacks"] > 0:
            recommendations.append(f"NOTICE: Some synthetic fallbacks ({self.metrics['synthetic_fallbacks']}).")
            
        return recommendations
        
    def start_acquisition_session(self, symbols: List[str], intervals: List[str]):
        """
        Start a new acquisition session.
        
        Args:
            symbols: List of symbols to acquire
            intervals: List of intervals to acquire
        """
        # Reset metrics
        self.reset_metrics()
        
        # Update metrics
        self.metrics["total_symbols"] = len(symbols)
        self.metrics["total_intervals"] = len(intervals)
        
        logger.info(f"Starting acquisition session for {len(symbols)} symbols and {len(intervals)} intervals")
        
    def end_acquisition_session(self):
        """End the current acquisition session and generate a report."""
        logger.info("Ending acquisition session")
        
        # Generate report
        report = self.generate_acquisition_report()
        
        # Notify user if needed
        self.notify_user_if_needed()
        
        return report
        
    def monitor_data_quality(self, data_quality: Dict) -> Tuple[bool, List[str]]:
        """
        Monitor data quality and generate warnings if needed.
        
        Args:
            data_quality: Data quality metrics
            
        Returns:
            Tuple of (is_acceptable, warnings)
        """
        warnings = []
        is_acceptable = True
        
        # Check if data is synthetic
        if data_quality.get("is_synthetic", False):
            warnings.append("Data is synthetic")
            is_acceptable = False
            
        # Check synthetic ratio
        if data_quality.get("synthetic_ratio", 0.0) > 0.0:
            warnings.append(f"Synthetic ratio: {data_quality['synthetic_ratio']:.2f}")
            if data_quality["synthetic_ratio"] > 0.5:
                is_acceptable = False
                
        # Check missing values
        if data_quality.get("missing_values", 0.0) > 0.1:
            warnings.append(f"High missing values: {data_quality['missing_values']:.2f}")
            if data_quality["missing_values"] > 0.3:
                is_acceptable = False
                
        # Check data points
        if data_quality.get("data_points", 0) < 100:
            warnings.append(f"Low data points: {data_quality['data_points']}")
            if data_quality["data_points"] < 50:
                is_acceptable = False
                
        return is_acceptable, warnings

def example_notification_callback(message: str):
    """
    Example notification callback function.
    
    Args:
        message: Notification message
    """
    print("\n" + "=" * 80)
    print("DATA ACQUISITION NOTIFICATION")
    print("=" * 80)
    print(message)
    print("=" * 80 + "\n")

async def main():
    """
    Main function to test the data acquisition monitor.
    """
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Data Acquisition Monitor")
    parser.add_argument("--symbols", type=str, default="BTC-USD-PERP,ETH-USD-PERP,SOL-USD-PERP", help="Comma-separated list of symbols")
    parser.add_argument("--intervals", type=str, default="1h,4h", help="Comma-separated list of intervals")
    args = parser.parse_args()
    
    # Create monitor
    monitor = DataAcquisitionMonitor(notification_callback=example_notification_callback)
    
    # Get symbols and intervals
    symbols = args.symbols.split(",")
    intervals = args.intervals.split(",")
    
    # Start acquisition session
    monitor.start_acquisition_session(symbols, intervals)
    
    # Simulate acquisition process
    for symbol in symbols:
        for interval in intervals:
            # Simulate acquisition attempt
            monitor.log_acquisition_attempt(symbol, interval, "hyperliquid")
            
            # Simulate success or failure
            if symbol == "BTC-USD-PERP":
                # Simulate success with real data
                data_quality = {
                    "source": "hyperliquid",
                    "is_synthetic": False,
                    "synthetic_ratio": 0.0,
                    "alternative_sources": [],
                    "missing_values": 0.05,
                    "data_points": 720,
                    "start_date": "2025-04-23",
                    "end_date": "2025-05-23"
                }
                monitor.log_acquisition_success(symbol, interval, "hyperliquid", data_quality)
            elif symbol == "ETH-USD-PERP":
                # Simulate failure with hyperliquid but success with binance
                monitor.log_acquisition_failure(symbol, interval, "hyperliquid", "API error: 404 Not Found")
                
                # Simulate binance attempt
                monitor.log_acquisition_attempt(symbol, interval, "binance")
                
                # Simulate success with binance
                data_quality = {
                    "source": "binance",
                    "is_synthetic": False,
                    "synthetic_ratio": 0.0,
                    "alternative_sources": ["binance"],
                    "missing_values": 0.02,
                    "data_points": 720,
                    "start_date": "2025-04-23",
                    "end_date": "2025-05-23"
                }
                monitor.log_acquisition_success(symbol, interval, "binance", data_quality)
            else:
                # Simulate failure with all sources
                monitor.log_acquisition_failure(symbol, interval, "hyperliquid", "API error: 404 Not Found")
                monitor.log_acquisition_attempt(symbol, interval, "binance")
                monitor.log_acquisition_failure(symbol, interval, "binance", "API error: 429 Too Many Requests")
                monitor.log_acquisition_attempt(symbol, interval, "coingecko")
                monitor.log_acquisition_failure(symbol, interval, "coingecko", "API error: 403 Forbidden")
                
                # Simulate synthetic fallback
                monitor.log_synthetic_fallback(symbol, interval)
                
                # Simulate success with synthetic data
                data_quality = {
                    "source": "synthetic",
                    "is_synthetic": True,
                    "synthetic_ratio": 1.0,
                    "alternative_sources": [],
                    "missing_values": 0.0,
                    "data_points": 720,
                    "start_date": "2025-04-23",
                    "end_date": "2025-05-23"
                }
                monitor.log_acquisition_success(symbol, interval, "synthetic", data_quality)
                
            # Monitor data quality
            is_acceptable, warnings = monitor.monitor_data_quality(data_quality)
            if not is_acceptable:
                logger.warning(f"Data quality issues for {symbol}, interval {interval}: {', '.join(warnings)}")
    
    # End acquisition session
    report = monitor.end_acquisition_session()
    
    # Print report
    print("\nAcquisition Report:")
    print(json.dumps(report, indent=2, default=str))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
