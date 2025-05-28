"""
Enhanced Mock Data Provider for Hyperliquid Trading Bot

This module provides mock data for the trading bot when API rate limits are reached
or when running in simulation mode. It generates realistic market data, order book,
and account information for testing and development purposes.

Features:
- Realistic OHLCV data generation with configurable volatility
- Order book simulation with proper depth and spread
- Position tracking with P&L calculation
- Order management with execution simulation
- Trade history generation
- Persistent data storage for consistent behavior across restarts
"""

import os
import json
import time
import random
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedMockDataProvider:
    """
    Enhanced mock data provider for Hyperliquid Trading Bot.
    
    Provides realistic market data, order book, and account information
    for testing and development purposes.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the mock data provider.
        
        Args:
            data_dir: Directory for storing mock data. If None, uses default.
        """
        # Set data directory
        if data_dir is None:
            self.data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "mock_data"
            )
        else:
            self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize data storage
        self.symbols = ["BTC", "ETH", "XRP", "SOL", "DOGE"]
        self.timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        self.ohlcv_data = {}
        self.orderbooks = {}
        self.positions = {}
        self.orders = []
        self.trade_history = []
        self.account_balance = 10000.0
        
        # Load or generate data
        self._load_or_generate_data()
        
        logger.info("Enhanced mock data provider initialized")
    
    def _load_or_generate_data(self) -> None:
        """Load existing data or generate new data if not available."""
        # Check if data file exists
        data_file = os.path.join(self.data_dir, "mock_data.json")
        
        if os.path.exists(data_file):
            try:
                # Load data
                with open(data_file, "r") as f:
                    data = json.load(f)
                
                # Set account data
                self.positions = data.get("positions", {})
                self.orders = data.get("orders", [])
                self.trade_history = data.get("trade_history", [])
                self.account_balance = data.get("account_balance", 10000.0)
                
                logger.info("Mock data loaded from file")
            except Exception as e:
                logger.error(f"Error loading mock data: {str(e)}")
                # Generate new data
                self._generate_initial_data()
        else:
            # Generate new data
            self._generate_initial_data()
    
    def _save_data(self) -> None:
        """Save data to file."""
        try:
            # Prepare data
            data = {
                "positions": self.positions,
                "orders": self.orders,
                "trade_history": self.trade_history,
                "account_balance": self.account_balance
            }
            
            # Save data
            data_file = os.path.join(self.data_dir, "mock_data.json")
            with open(data_file, "w") as f:
                json.dump(data, f, indent=4)
            
            logger.debug("Mock data saved to file")
        except Exception as e:
            logger.error(f"Error saving mock data: {str(e)}")
    
    def _generate_initial_data(self) -> None:
        """Generate initial mock data."""
        logger.info("Generating initial mock data")
        
        # Generate OHLCV data for each symbol and timeframe
        for symbol in self.symbols:
            self.ohlcv_data[symbol] = {}
            for timeframe in self.timeframes:
                self.ohlcv_data[symbol][timeframe] = self._generate_ohlcv(symbol, timeframe)
            
            # Generate order book
            self.orderbooks[symbol] = self._generate_orderbook(symbol)
        
        # Generate account data
        self._generate_account_data()
        
        # Save data
        self._save_data()
    
    def _generate_ohlcv(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Generate OHLCV data for a symbol and timeframe.
        
        Args:
            symbol: Symbol to generate data for
            timeframe: Timeframe to generate data for
            
        Returns:
            DataFrame with OHLCV data
        """
        # Set parameters based on symbol
        if symbol == "BTC":
            base_price = 50000.0
            volatility = 0.02
        elif symbol == "ETH":
            base_price = 3000.0
            volatility = 0.025
        elif symbol == "XRP":
            base_price = 0.5
            volatility = 0.03
        elif symbol == "SOL":
            base_price = 100.0
            volatility = 0.035
        elif symbol == "DOGE":
            base_price = 0.1
            volatility = 0.04
        else:
            base_price = 100.0
            volatility = 0.03
        
        # Set number of periods based on timeframe
        if timeframe == "1m":
            periods = 1440  # 1 day
            minutes_per_candle = 1
        elif timeframe == "5m":
            periods = 1440  # 5 days
            minutes_per_candle = 5
        elif timeframe == "15m":
            periods = 960  # 10 days
            minutes_per_candle = 15
        elif timeframe == "1h":
            periods = 720  # 30 days
            minutes_per_candle = 60
        elif timeframe == "4h":
            periods = 720  # 120 days
            minutes_per_candle = 240
        elif timeframe == "1d":
            periods = 365  # 1 year
            minutes_per_candle = 1440
        else:
            periods = 100
            minutes_per_candle = 60
        
        # Generate timestamps
        end_time = datetime.now()
        timestamps = [end_time - timedelta(minutes=i * minutes_per_candle) for i in range(periods)]
        timestamps.reverse()
        
        # Generate price data using geometric Brownian motion
        np.random.seed(hash(symbol + timeframe) % 2**32)
        returns = np.random.normal(0, volatility, periods)
        price_changes = np.exp(returns)
        prices = [base_price]
        
        for change in price_changes:
            prices.append(prices[-1] * change)
        
        prices = prices[1:]  # Remove initial price
        
        # Generate OHLCV data
        data = []
        
        for i in range(periods):
            # Calculate price range for this candle
            close = prices[i]
            price_range = close * volatility
            
            # Generate open, high, low
            if i == 0:
                open_price = close * (1 + np.random.normal(0, volatility/2))
            else:
                open_price = data[i-1]["close"]
            
            high = max(open_price, close) + abs(np.random.normal(0, price_range/2))
            low = min(open_price, close) - abs(np.random.normal(0, price_range/2))
            
            # Generate volume
            volume = abs(np.random.normal(base_price * 100, base_price * 50))
            
            # Add candle
            data.append({
                "timestamp": timestamps[i],
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        
        return df
    
    def _generate_orderbook(self, symbol: str) -> Dict:
        """
        Generate order book for a symbol.
        
        Args:
            symbol: Symbol to generate order book for
            
        Returns:
            Dictionary with bids and asks
        """
        # Get latest price
        if symbol in self.ohlcv_data and "1m" in self.ohlcv_data[symbol]:
            price = self.ohlcv_data[symbol]["1m"].iloc[-1]["close"]
        else:
            # Set default price based on symbol
            if symbol == "BTC":
                price = 50000.0
            elif symbol == "ETH":
                price = 3000.0
            elif symbol == "XRP":
                price = 0.5
            elif symbol == "SOL":
                price = 100.0
            elif symbol == "DOGE":
                price = 0.1
            else:
                price = 100.0
        
        # Set parameters based on symbol
        if symbol == "BTC":
            spread = 0.0002  # 0.02%
            depth = 50
            quantity_scale = 0.1
        elif symbol == "ETH":
            spread = 0.0003  # 0.03%
            depth = 40
            quantity_scale = 0.5
        elif symbol == "XRP":
            spread = 0.0005  # 0.05%
            depth = 30
            quantity_scale = 1000
        elif symbol == "SOL":
            spread = 0.0004  # 0.04%
            depth = 35
            quantity_scale = 10
        elif symbol == "DOGE":
            spread = 0.0006  # 0.06%
            depth = 25
            quantity_scale = 10000
        else:
            spread = 0.0005  # 0.05%
            depth = 30
            quantity_scale = 10
        
        # Generate bids
        bids = []
        bid_price = price * (1 - spread/2)
        
        for i in range(depth):
            # Calculate price and quantity
            level_price = bid_price * (1 - i * spread/2)
            level_quantity = quantity_scale * (1 + np.random.exponential(2))
            
            # Add level
            bids.append([level_price, level_quantity])
        
        # Generate asks
        asks = []
        ask_price = price * (1 + spread/2)
        
        for i in range(depth):
            # Calculate price and quantity
            level_price = ask_price * (1 + i * spread/2)
            level_quantity = quantity_scale * (1 + np.random.exponential(2))
            
            # Add level
            asks.append([level_price, level_quantity])
        
        # Sort bids and asks
        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])
        
        return {
            "bids": bids,
            "asks": asks
        }
    
    def _generate_account_data(self) -> None:
        """Generate account data."""
        # Generate positions
        self.positions = {}
        
        # 50% chance to have a position for each symbol
        for symbol in self.symbols:
            if random.random() < 0.5:
                # Get latest price
                if symbol in self.ohlcv_data and "1m" in self.ohlcv_data[symbol]:
                    price = self.ohlcv_data[symbol]["1m"].iloc[-1]["close"]
                else:
                    # Set default price based on symbol
                    if symbol == "BTC":
                        price = 50000.0
                    elif symbol == "ETH":
                        price = 3000.0
                    elif symbol == "XRP":
                        price = 0.5
                    elif symbol == "SOL":
                        price = 100.0
                    elif symbol == "DOGE":
                        price = 0.1
                    else:
                        price = 100.0
                
                # Generate position
                side = 1 if random.random() < 0.5 else -1
                size = side * random.uniform(0.1, 1.0)
                entry_price = price * random.uniform(0.9, 1.1)
                
                # Calculate PnL
                unrealized_pnl = size * (price - entry_price)
                unrealized_pnl_pct = (price / entry_price - 1) * 100 * side
                
                # Calculate liquidation price
                if side > 0:
                    liquidation_price = entry_price * 0.5
                else:
                    liquidation_price = entry_price * 1.5
                
                # Add position
                self.positions[symbol] = {
                    "symbol": symbol,
                    "size": size,
                    "entry_price": entry_price,
                    "mark_price": price,
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_pct": unrealized_pnl_pct,
                    "liquidation_price": liquidation_price
                }
        
        # Generate orders
        self.orders = []
        
        # 50% chance to have an order for each symbol
        for symbol in self.symbols:
            if random.random() < 0.5:
                # Get latest price
                if symbol in self.ohlcv_data and "1m" in self.ohlcv_data[symbol]:
                    price = self.ohlcv_data[symbol]["1m"].iloc[-1]["close"]
                else:
                    # Set default price based on symbol
                    if symbol == "BTC":
                        price = 50000.0
                    elif symbol == "ETH":
                        price = 3000.0
                    elif symbol == "XRP":
                        price = 0.5
                    elif symbol == "SOL":
                        price = 100.0
                    elif symbol == "DOGE":
                        price = 0.1
                    else:
                        price = 100.0
                
                # Generate order
                order_id = f"order_{int(time.time())}_{random.randint(1000, 9999)}"
                side = "buy" if random.random() < 0.5 else "sell"
                order_type = random.choice(["limit", "stop", "stop_limit"])
                order_price = price * random.uniform(0.9, 1.1)
                quantity = random.uniform(0.1, 1.0)
                filled = random.uniform(0, quantity) if random.random() < 0.3 else 0
                status = "open"
                order_time = time.time() - random.uniform(0, 86400)
                
                # Add order
                self.orders.append({
                    "id": order_id,
                    "symbol": symbol,
                    "type": order_type,
                    "side": side,
                    "price": order_price,
                    "quantity": quantity,
                    "filled": filled,
                    "status": status,
                    "time": order_time
                })
        
        # Generate trade history
        self.trade_history = []
        
        # Generate 10 random trades
        for i in range(10):
            # Select random symbol
            symbol = random.choice(self.symbols)
            
            # Get latest price
            if symbol in self.ohlcv_data and "1m" in self.ohlcv_data[symbol]:
                price = self.ohlcv_data[symbol]["1m"].iloc[-1]["close"]
            else:
                # Set default price based on symbol
                if symbol == "BTC":
                    price = 50000.0
                elif symbol == "ETH":
                    price = 3000.0
                elif symbol == "XRP":
                    price = 0.5
                elif symbol == "SOL":
                    price = 100.0
                elif symbol == "DOGE":
                    price = 0.1
                else:
                    price = 100.0
            
            # Generate trade
            trade_id = f"trade_{int(time.time())}_{random.randint(1000, 9999)}"
            side = "buy" if random.random() < 0.5 else "sell"
            trade_price = price * random.uniform(0.9, 1.1)
            quantity = random.uniform(0.1, 1.0)
            fee = quantity * trade_price * 0.001
            trade_time = time.time() - random.uniform(0, 604800)  # Within last week
            
            # Add trade
            self.trade_history.append({
                "id": trade_id,
                "symbol": symbol,
                "side": side,
                "price": trade_price,
                "quantity": quantity,
                "fee": fee,
                "time": trade_time
            })
        
        # Sort trade history by time (newest first)
        self.trade_history.sort(key=lambda x: x["time"], reverse=True)
        
        # Set account balance
        self.account_balance = 10000.0
    
    def _update_data(self) -> None:
        """Update mock data with small changes to simulate market movement."""
        # Update OHLCV data
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                if symbol in self.ohlcv_data and timeframe in self.ohlcv_data[symbol]:
                    # Get latest candle
                    latest_candle = self.ohlcv_data[symbol][timeframe].iloc[-1].copy()
                    
                    # Set parameters based on symbol
                    if symbol == "BTC":
                        volatility = 0.02
                    elif symbol == "ETH":
                        volatility = 0.025
                    elif symbol == "XRP":
                        volatility = 0.03
                    elif symbol == "SOL":
                        volatility = 0.035
                    elif symbol == "DOGE":
                        volatility = 0.04
                    else:
                        volatility = 0.03
                    
                    # Generate new price
                    price_change = np.random.normal(0, volatility/10)
                    new_price = latest_candle["close"] * (1 + price_change)
                    
                    # Update candle
                    latest_candle["close"] = new_price
                    latest_candle["high"] = max(latest_candle["high"], new_price)
                    latest_candle["low"] = min(latest_candle["low"], new_price)
                    
                    # Update volume
                    latest_candle["volume"] += abs(np.random.normal(0, latest_candle["volume"]/100))
                    
                    # Update data
                    self.ohlcv_data[symbol][timeframe].iloc[-1] = latest_candle
            
            # Update order book
            if symbol in self.orderbooks:
                self.orderbooks[symbol] = self._generate_orderbook(symbol)
        
        # Update positions
        for symbol, position in self.positions.items():
            if symbol in self.ohlcv_data and "1m" in self.ohlcv_data[symbol]:
                # Get latest price
                price = self.ohlcv_data[symbol]["1m"].iloc[-1]["close"]
                
                # Update position
                position["mark_price"] = price
                position["unrealized_pnl"] = position["size"] * (price - position["entry_price"])
                position["unrealized_pnl_pct"] = (price / position["entry_price"] - 1) * 100
                if position["size"] < 0:
                    position["unrealized_pnl_pct"] *= -1
        
        # Process orders
        for order in self.orders:
            # Skip filled or canceled orders
            if order["status"] != "open":
                continue
            
            # Get symbol and latest price
            symbol = order["symbol"]
            if symbol in self.ohlcv_data and "1m" in self.ohlcv_data[symbol]:
                price = self.ohlcv_data[symbol]["1m"].iloc[-1]["close"]
                
                # Check if order should be filled
                if order["type"] == "limit":
                    if (order["side"] == "buy" and price <= order["price"]) or \
                       (order["side"] == "sell" and price >= order["price"]):
                        # Fill order
                        order["filled"] = order["quantity"]
                        order["status"] = "filled"
                        
                        # Add to trade history
                        self._add_trade(order)
                        
                        # Update position
                        self._update_position(order)
                elif order["type"] == "stop":
                    if (order["side"] == "buy" and price >= order["price"]) or \
                       (order["side"] == "sell" and price <= order["price"]):
                        # Fill order
                        order["filled"] = order["quantity"]
                        order["status"] = "filled"
                        
                        # Add to trade history
                        self._add_trade(order)
                        
                        # Update position
                        self._update_position(order)
                elif order["type"] == "stop_limit":
                    # Not implemented yet
                    pass
    
    def _add_trade(self, order: Dict) -> None:
        """
        Add a trade to the trade history.
        
        Args:
            order: Order that was filled
        """
        # Generate trade
        trade_id = f"trade_{int(time.time())}_{random.randint(1000, 9999)}"
        trade_price = order["price"]
        quantity = order["quantity"]
        fee = quantity * trade_price * 0.001
        trade_time = time.time()
        
        # Add trade
        self.trade_history.append({
            "id": trade_id,
            "symbol": order["symbol"],
            "side": order["side"],
            "price": trade_price,
            "quantity": quantity,
            "fee": fee,
            "time": trade_time
        })
        
        # Sort trade history by time (newest first)
        self.trade_history.sort(key=lambda x: x["time"], reverse=True)
    
    def _update_position(self, order: Dict) -> None:
        """
        Update position based on filled order.
        
        Args:
            order: Order that was filled
        """
        symbol = order["symbol"]
        
        # Check if position exists
        if symbol in self.positions:
            # Get position
            position = self.positions[symbol]
            
            # Update position
            if order["side"] == "buy":
                # Calculate new position
                new_size = position["size"] + order["quantity"]
                
                if new_size == 0:
                    # Position closed
                    del self.positions[symbol]
                else:
                    # Update position
                    if position["size"] < 0 and new_size > 0:
                        # Position flipped from short to long
                        position["entry_price"] = order["price"]
                    else:
                        # Calculate new entry price
                        position["entry_price"] = (position["entry_price"] * position["size"] + order["price"] * order["quantity"]) / new_size
                    
                    position["size"] = new_size
                    
                    # Update PnL
                    position["unrealized_pnl"] = position["size"] * (position["mark_price"] - position["entry_price"])
                    position["unrealized_pnl_pct"] = (position["mark_price"] / position["entry_price"] - 1) * 100
                    if position["size"] < 0:
                        position["unrealized_pnl_pct"] *= -1
                    
                    # Update liquidation price
                    if position["size"] > 0:
                        position["liquidation_price"] = position["entry_price"] * 0.5
                    else:
                        position["liquidation_price"] = position["entry_price"] * 1.5
            else:  # sell
                # Calculate new position
                new_size = position["size"] - order["quantity"]
                
                if new_size == 0:
                    # Position closed
                    del self.positions[symbol]
                else:
                    # Update position
                    if position["size"] > 0 and new_size < 0:
                        # Position flipped from long to short
                        position["entry_price"] = order["price"]
                    else:
                        # Calculate new entry price
                        position["entry_price"] = (position["entry_price"] * position["size"] - order["price"] * order["quantity"]) / new_size
                    
                    position["size"] = new_size
                    
                    # Update PnL
                    position["unrealized_pnl"] = position["size"] * (position["mark_price"] - position["entry_price"])
                    position["unrealized_pnl_pct"] = (position["mark_price"] / position["entry_price"] - 1) * 100
                    if position["size"] < 0:
                        position["unrealized_pnl_pct"] *= -1
                    
                    # Update liquidation price
                    if position["size"] > 0:
                        position["liquidation_price"] = position["entry_price"] * 0.5
                    else:
                        position["liquidation_price"] = position["entry_price"] * 1.5
        else:
            # Create new position
            if order["side"] == "buy":
                size = order["quantity"]
            else:
                size = -order["quantity"]
            
            # Get latest price
            if symbol in self.ohlcv_data and "1m" in self.ohlcv_data[symbol]:
                price = self.ohlcv_data[symbol]["1m"].iloc[-1]["close"]
            else:
                price = order["price"]
            
            # Calculate liquidation price
            if size > 0:
                liquidation_price = order["price"] * 0.5
            else:
                liquidation_price = order["price"] * 1.5
            
            # Add position
            self.positions[symbol] = {
                "symbol": symbol,
                "size": size,
                "entry_price": order["price"],
                "mark_price": price,
                "unrealized_pnl": size * (price - order["price"]),
                "unrealized_pnl_pct": (price / order["price"] - 1) * 100 * (1 if size > 0 else -1),
                "liquidation_price": liquidation_price
            }
        
        # Update account balance
        self.account_balance -= order["quantity"] * order["price"] * 0.001  # Subtract fee
    
    def get_klines(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Get OHLCV data for a symbol and timeframe.
        
        Args:
            symbol: Symbol to get data for
            timeframe: Timeframe to get data for
            
        Returns:
            DataFrame with OHLCV data
        """
        # Update data
        self._update_data()
        
        # Check if data exists
        if symbol in self.ohlcv_data and timeframe in self.ohlcv_data[symbol]:
            return self.ohlcv_data[symbol][timeframe].copy()
        else:
            # Generate data
            self.ohlcv_data[symbol] = self.ohlcv_data.get(symbol, {})
            self.ohlcv_data[symbol][timeframe] = self._generate_ohlcv(symbol, timeframe)
            return self.ohlcv_data[symbol][timeframe].copy()
    
    def get_orderbook(self, symbol: str) -> Dict:
        """
        Get order book for a symbol.
        
        Args:
            symbol: Symbol to get order book for
            
        Returns:
            Dictionary with bids and asks
        """
        # Update data
        self._update_data()
        
        # Check if data exists
        if symbol in self.orderbooks:
            return self.orderbooks[symbol].copy()
        else:
            # Generate data
            self.orderbooks[symbol] = self._generate_orderbook(symbol)
            return self.orderbooks[symbol].copy()
    
    def get_recent_trades(self, symbol: str, limit: int = 20) -> List:
        """
        Get recent trades for a symbol.
        
        Args:
            symbol: Symbol to get trades for
            limit: Maximum number of trades to return
            
        Returns:
            List of trades
        """
        # Update data
        self._update_data()
        
        # Filter trades by symbol
        trades = [trade for trade in self.trade_history if trade["symbol"] == symbol]
        
        # Return limited number of trades
        return trades[:limit]
    
    def get_positions(self) -> Dict:
        """
        Get positions.
        
        Returns:
            Dictionary with positions
        """
        # Update data
        self._update_data()
        
        return self.positions.copy()
    
    def get_open_orders(self) -> List:
        """
        Get open orders.
        
        Returns:
            List of open orders
        """
        # Update data
        self._update_data()
        
        return [order.copy() for order in self.orders if order["status"] == "open"]
    
    def get_trade_history(self, limit: int = 20) -> List:
        """
        Get trade history.
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            List of trades
        """
        # Update data
        self._update_data()
        
        return self.trade_history[:limit]
    
    def get_account_balance(self) -> float:
        """
        Get account balance.
        
        Returns:
            Account balance
        """
        # Update data
        self._update_data()
        
        return self.account_balance
    
    def place_order(self, symbol: str, order_type: str, side: str, quantity: float, 
                   price: Optional[float] = None, stop_price: Optional[float] = None) -> bool:
        """
        Place an order.
        
        Args:
            symbol: Symbol to place order for
            order_type: Order type (market, limit, stop, stop_limit)
            side: Order side (buy, sell)
            quantity: Order quantity
            price: Order price (required for limit and stop_limit orders)
            stop_price: Stop price (required for stop and stop_limit orders)
            
        Returns:
            True if order was placed successfully, False otherwise
        """
        try:
            # Validate inputs
            if symbol not in self.symbols:
                logger.error(f"Invalid symbol: {symbol}")
                return False
            
            if order_type not in ["market", "limit", "stop", "stop_limit"]:
                logger.error(f"Invalid order type: {order_type}")
                return False
            
            if side not in ["buy", "sell"]:
                logger.error(f"Invalid side: {side}")
                return False
            
            if quantity <= 0:
                logger.error(f"Invalid quantity: {quantity}")
                return False
            
            if order_type in ["limit", "stop_limit"] and (price is None or price <= 0):
                logger.error(f"Invalid price for {order_type} order: {price}")
                return False
            
            if order_type in ["stop", "stop_limit"] and (stop_price is None or stop_price <= 0):
                logger.error(f"Invalid stop price for {order_type} order: {stop_price}")
                return False
            
            # Get latest price
            if symbol in self.ohlcv_data and "1m" in self.ohlcv_data[symbol]:
                current_price = self.ohlcv_data[symbol]["1m"].iloc[-1]["close"]
            else:
                # Generate data
                self.ohlcv_data[symbol] = self.ohlcv_data.get(symbol, {})
                self.ohlcv_data[symbol]["1m"] = self._generate_ohlcv(symbol, "1m")
                current_price = self.ohlcv_data[symbol]["1m"].iloc[-1]["close"]
            
            # Set price for market orders
            if order_type == "market":
                price = current_price
            
            # Generate order ID
            order_id = f"order_{int(time.time())}_{random.randint(1000, 9999)}"
            
            # Create order
            order = {
                "id": order_id,
                "symbol": symbol,
                "type": order_type,
                "side": side,
                "price": price,
                "stop_price": stop_price,
                "quantity": quantity,
                "filled": 0,
                "status": "open",
                "time": time.time()
            }
            
            # Add order
            self.orders.append(order)
            
            # Check if order should be filled immediately
            if order_type == "market":
                # Fill order
                order["filled"] = order["quantity"]
                order["status"] = "filled"
                
                # Add to trade history
                self._add_trade(order)
                
                # Update position
                self._update_position(order)
            elif order_type == "limit":
                if (side == "buy" and current_price <= price) or \
                   (side == "sell" and current_price >= price):
                    # Fill order
                    order["filled"] = order["quantity"]
                    order["status"] = "filled"
                    
                    # Add to trade history
                    self._add_trade(order)
                    
                    # Update position
                    self._update_position(order)
            elif order_type == "stop":
                if (side == "buy" and current_price >= stop_price) or \
                   (side == "sell" and current_price <= stop_price):
                    # Fill order
                    order["filled"] = order["quantity"]
                    order["status"] = "filled"
                    
                    # Add to trade history
                    self._add_trade(order)
                    
                    # Update position
                    self._update_position(order)
            elif order_type == "stop_limit":
                if (side == "buy" and current_price >= stop_price) or \
                   (side == "sell" and current_price <= stop_price):
                    # Convert to limit order
                    order["type"] = "limit"
                    
                    # Check if limit price is hit
                    if (side == "buy" and current_price <= price) or \
                       (side == "sell" and current_price >= price):
                        # Fill order
                        order["filled"] = order["quantity"]
                        order["status"] = "filled"
                        
                        # Add to trade history
                        self._add_trade(order)
                        
                        # Update position
                        self._update_position(order)
            
            # Save data
            self._save_data()
            
            return True
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return False
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if order was cancelled successfully, False otherwise
        """
        try:
            # Find order
            for order in self.orders:
                if order["id"] == order_id:
                    # Check if order can be cancelled
                    if order["status"] != "open":
                        logger.error(f"Cannot cancel order {order_id} with status {order['status']}")
                        return False
                    
                    # Cancel order
                    order["status"] = "cancelled"
                    
                    # Save data
                    self._save_data()
                    
                    return True
            
            # Order not found
            logger.error(f"Order not found: {order_id}")
            return False
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            return False
    
    def close_position(self, symbol: str) -> bool:
        """
        Close a position.
        
        Args:
            symbol: Symbol to close position for
            
        Returns:
            True if position was closed successfully, False otherwise
        """
        try:
            # Check if position exists
            if symbol not in self.positions:
                logger.error(f"No position found for {symbol}")
                return False
            
            # Get position
            position = self.positions[symbol]
            
            # Create order to close position
            side = "sell" if position["size"] > 0 else "buy"
            quantity = abs(position["size"])
            
            # Place order
            return self.place_order(symbol, "market", side, quantity)
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            return False
