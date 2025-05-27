#!/usr/bin/env python3
"""
Enhanced Market Data Generator

This script generates realistic market data for testing trading strategies
with different market regimes.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("market_data_generator.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MarketDataGenerator:
    """
    Enhanced market data generator for testing trading strategies.
    """
    
    def __init__(self):
        """Initialize the market data generator."""
        self.logger = logger
        self.logger.info("Initializing Market Data Generator")
        
        # Symbols to generate data for
        self.symbols = ["BTC", "ETH", "XRP", "SOL"]
        
        # Market regimes
        self.regimes = ["BULLISH", "BEARISH", "RANGING", "VOLATILE"]
        
        # Output directory
        self.output_dir = "market_data"
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger.info("Market Data Generator initialized")
    
    def generate_all_data(self) -> None:
        """Generate market data for all symbols and regimes."""
        self.logger.info("Generating market data for all symbols and regimes")
        
        for symbol in self.symbols:
            for regime in self.regimes:
                self.generate_data(symbol, regime)
        
        self.logger.info("Market data generation completed")
    
    def generate_data(self, symbol: str, regime: str) -> None:
        """
        Generate market data for a symbol and regime.
        
        Args:
            symbol: Trading symbol
            regime: Market regime
        """
        try:
            self.logger.info(f"Generating {regime} market data for {symbol}")
            
            # Generate price data
            data = self._generate_price_data(symbol, regime)
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            # Add funding rate
            data = self._add_funding_rate(data, regime)
            
            # Add regime
            data["regime"] = regime
            
            # Save to file
            output_file = f"{self.output_dir}/{symbol}_{regime}.json"
            
            with open(output_file, "w") as f:
                json.dump(data, f, indent=4)
            
            self.logger.info(f"Market data saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error generating market data for {symbol} {regime}: {e}")
    
    def _generate_price_data(self, symbol: str, regime: str) -> Dict[str, Any]:
        """
        Generate price data for a symbol and regime.
        
        Args:
            symbol: Trading symbol
            regime: Market regime
            
        Returns:
            Dictionary containing price data
        """
        # Number of data points
        n_points = 200
        
        # Initialize price data
        data = {
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": []
        }
        
        # Set base price based on symbol
        if symbol == "BTC":
            base_price = 50000.0
        elif symbol == "ETH":
            base_price = 3000.0
        elif symbol == "XRP":
            base_price = 0.5
        elif symbol == "SOL":
            base_price = 100.0
        else:
            base_price = 100.0
        
        # Set parameters based on regime
        if regime == "BULLISH":
            trend = 0.001  # Positive trend
            volatility = 0.01  # Moderate volatility
            volume_base = 1000000
            volume_volatility = 0.3
        elif regime == "BEARISH":
            trend = -0.001  # Negative trend
            volatility = 0.015  # Higher volatility
            volume_base = 1200000
            volume_volatility = 0.4
        elif regime == "RANGING":
            trend = 0.0  # No trend
            volatility = 0.005  # Lower volatility
            volume_base = 800000
            volume_volatility = 0.2
        elif regime == "VOLATILE":
            trend = 0.0  # No trend
            volatility = 0.025  # High volatility
            volume_base = 1500000
            volume_volatility = 0.5
        else:
            trend = 0.0
            volatility = 0.01
            volume_base = 1000000
            volume_volatility = 0.3
        
        # Generate timestamps
        start_time = datetime.now() - timedelta(days=n_points)
        
        for i in range(n_points):
            timestamp = start_time + timedelta(days=i)
            data["timestamp"].append(timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        
        # Generate prices
        prices = np.zeros(n_points)
        prices[0] = base_price
        
        # Add more realistic price patterns based on regime
        if regime == "BULLISH":
            # Add upward momentum with occasional pullbacks
            for i in range(1, n_points):
                # Calculate price change
                momentum = 0.002 * np.sin(i / 10)  # Cyclical momentum
                trend_component = (trend + momentum) * (1 + 0.2 * np.random.randn())
                volatility_component = volatility * np.random.randn()
                
                # Apply price change
                price_change = trend_component + volatility_component
                prices[i] = prices[i-1] * (1 + price_change)
                
                # Add occasional pullbacks
                if i % 20 == 0:
                    prices[i] = prices[i] * 0.98
        
        elif regime == "BEARISH":
            # Add downward momentum with occasional bounces
            for i in range(1, n_points):
                # Calculate price change
                momentum = 0.002 * np.sin(i / 10)  # Cyclical momentum
                trend_component = (trend + momentum) * (1 + 0.2 * np.random.randn())
                volatility_component = volatility * np.random.randn()
                
                # Apply price change
                price_change = trend_component + volatility_component
                prices[i] = prices[i-1] * (1 + price_change)
                
                # Add occasional bounces
                if i % 20 == 0:
                    prices[i] = prices[i] * 1.02
        
        elif regime == "RANGING":
            # Add mean-reverting behavior
            for i in range(1, n_points):
                # Calculate mean reversion
                deviation = (prices[i-1] - base_price) / base_price
                mean_reversion = -0.1 * deviation
                
                # Calculate price change
                trend_component = mean_reversion * (1 + 0.2 * np.random.randn())
                volatility_component = volatility * np.random.randn()
                
                # Apply price change
                price_change = trend_component + volatility_component
                prices[i] = prices[i-1] * (1 + price_change)
        
        elif regime == "VOLATILE":
            # Add high volatility with regime shifts
            for i in range(1, n_points):
                # Calculate regime shift
                if i % 40 == 0:
                    # Shift between mini-bull and mini-bear regimes
                    trend = -trend
                
                # Calculate price change
                trend_component = trend * (1 + 0.5 * np.random.randn())
                volatility_component = volatility * np.random.randn()
                
                # Apply price change
                price_change = trend_component + volatility_component
                prices[i] = prices[i-1] * (1 + price_change)
        
        # Generate OHLC data
        for i in range(n_points):
            # Generate intraday volatility
            intraday_volatility = volatility * prices[i] * 0.5
            
            # Generate OHLC
            open_price = prices[i]
            close_price = prices[i]
            high_price = open_price + intraday_volatility * np.random.rand()
            low_price = open_price - intraday_volatility * np.random.rand()
            
            # Ensure high >= open, close and low <= open, close
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Generate volume
            volume = volume_base * (1 + volume_volatility * np.random.randn())
            volume = max(volume, 0)  # Ensure positive volume
            
            # Add to data
            data["open"].append(float(open_price))
            data["high"].append(float(high_price))
            data["low"].append(float(low_price))
            data["close"].append(float(close_price))
            data["volume"].append(float(volume))
        
        return data
    
    def _add_technical_indicators(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add technical indicators to market data.
        
        Args:
            data: Market data dictionary
            
        Returns:
            Market data dictionary with technical indicators
        """
        try:
            # Convert to pandas DataFrame for easier calculation
            df = pd.DataFrame({
                "timestamp": data["timestamp"],
                "open": data["open"],
                "high": data["high"],
                "low": data["low"],
                "close": data["close"],
                "volume": data["volume"]
            })
            
            # Calculate RSI
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            data["rsi"] = rsi.fillna(50).tolist()
            
            # Calculate MACD
            ema12 = df["close"].ewm(span=12, adjust=False).mean()
            ema26 = df["close"].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            
            data["macd"] = macd.fillna(0).tolist()
            data["macd_signal"] = signal.fillna(0).tolist()
            data["macd_histogram"] = histogram.fillna(0).tolist()
            
            # Calculate Bollinger Bands
            sma20 = df["close"].rolling(window=20).mean()
            std20 = df["close"].rolling(window=20).std()
            
            upper_band = sma20 + 2 * std20
            lower_band = sma20 - 2 * std20
            
            data["bb_upper"] = upper_band.fillna(df["close"]).tolist()
            data["bb_middle"] = sma20.fillna(df["close"]).tolist()
            data["bb_lower"] = lower_band.fillna(df["close"]).tolist()
            
            # Calculate ATR
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift()).abs()
            low_close = (df["low"] - df["close"].shift()).abs()
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            
            data["atr"] = atr.fillna(tr).tolist()
            
            # Calculate Stochastic
            low_min = df["low"].rolling(window=14).min()
            high_max = df["high"].rolling(window=14).max()
            
            stoch_k = 100 * ((df["close"] - low_min) / (high_max - low_min))
            stoch_d = stoch_k.rolling(window=3).mean()
            
            data["stoch_k"] = stoch_k.fillna(50).tolist()
            data["stoch_d"] = stoch_d.fillna(50).tolist()
            
            # Calculate OBV
            obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
            
            data["obv"] = obv.tolist()
            
            return data
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
            return data
    
    def _add_funding_rate(self, data: Dict[str, Any], regime: str) -> Dict[str, Any]:
        """
        Add funding rate to market data.
        
        Args:
            data: Market data dictionary
            regime: Market regime
            
        Returns:
            Market data dictionary with funding rate
        """
        try:
            # Set funding rate based on regime
            if regime == "BULLISH":
                # In bullish markets, funding rate is typically positive
                funding_rate = 0.001 + 0.0005 * np.random.randn()
            elif regime == "BEARISH":
                # In bearish markets, funding rate is typically negative
                funding_rate = -0.001 + 0.0005 * np.random.randn()
            elif regime == "RANGING":
                # In ranging markets, funding rate is typically close to zero
                funding_rate = 0.0001 * np.random.randn()
            elif regime == "VOLATILE":
                # In volatile markets, funding rate can swing between positive and negative
                funding_rate = 0.002 * np.random.randn()
            else:
                funding_rate = 0.0
            
            data["funding_rate"] = float(funding_rate)
            
            return data
        except Exception as e:
            self.logger.error(f"Error adding funding rate: {e}")
            return data

if __name__ == "__main__":
    # Generate market data
    generator = MarketDataGenerator()
    generator.generate_all_data()
    
    print("Market data generation completed successfully!")
