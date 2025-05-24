"""
Order Book Handler

This module provides robust order book data retrieval and processing
for the Hyperliquid trading bot.
"""

import logging
import time
import requests
import json
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class OrderBookHandler:
    """
    Handles order book data retrieval and processing with fallbacks
    and synthetic data generation when needed.
    """
    
    def __init__(self, base_url: str = "https://api.hyperliquid.xyz"):
        """
        Initialize the Order Book Handler.
        
        Args:
            base_url: Base URL for Hyperliquid API
        """
        self.base_url = base_url
        self.cache = {}
        self.cache_ttl = 5  # 5 seconds
        self.last_update = {}
        
    def get_order_book(self, symbol: str, retries: int = 3) -> Dict:
        """
        Get order book with retries and fallbacks.
        
        Args:
            symbol: Symbol to get order book for
            retries: Number of retries
            
        Returns:
            Order book dictionary
        """
        try:
            # Check cache first
            current_time = time.time()
            if symbol in self.cache and current_time - self.last_update.get(symbol, 0) < self.cache_ttl:
                logger.debug(f"Using cached order book for {symbol}")
                return self.cache[symbol]
                
            # Try to get real order book data
            order_book = self._fetch_order_book(symbol, retries)
            
            # Validate order book data
            if self._is_valid_order_book(order_book):
                # Update cache
                self.cache[symbol] = order_book
                self.last_update[symbol] = current_time
                return order_book
                
            # If invalid, try to get from cache even if expired
            if symbol in self.cache:
                logger.warning(f"Using stale cached order book for {symbol}")
                return self.cache[symbol]
                
            # If no cache, generate synthetic order book
            logger.warning(f"Generating synthetic order book for {symbol}")
            synthetic_order_book = self._generate_synthetic_order_book(symbol)
            
            # Update cache with synthetic data
            self.cache[symbol] = synthetic_order_book
            self.last_update[symbol] = current_time
            
            return synthetic_order_book
        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {str(e)}")
            
            # Try to get from cache even if expired
            if symbol in self.cache:
                logger.warning(f"Using stale cached order book for {symbol} after error")
                return self.cache[symbol]
                
            # If no cache, generate synthetic order book
            logger.warning(f"Generating synthetic order book for {symbol} after error")
            return self._generate_synthetic_order_book(symbol)
            
    def _fetch_order_book(self, symbol: str, retries: int = 3) -> Dict:
        """
        Fetch order book from API with retries.
        
        Args:
            symbol: Symbol to get order book for
            retries: Number of retries
            
        Returns:
            Order book dictionary
        """
        for attempt in range(retries):
            try:
                payload = {"type": "l2Book", "coin": symbol}
                response = requests.post(f"{self.base_url}/info", json=payload)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limit hit, wait and retry
                    wait_time = (2 ** attempt) * 1.0  # Exponential backoff
                    logger.warning(f"Rate limit hit, retrying in {wait_time:.2f} seconds (attempt {attempt+1}/{retries})")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"Failed to get order book: {response.status_code}")
                    time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Error fetching order book: {str(e)}")
                time.sleep(0.5)
                
        return {}
        
    def _is_valid_order_book(self, order_book: Dict) -> bool:
        """
        Check if order book data is valid.
        
        Args:
            order_book: Order book dictionary
            
        Returns:
            True if valid, False otherwise
        """
        if not order_book:
            return False
            
        # Check if bids and asks exist and are non-empty
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        if not bids or not asks:
            return False
            
        # Check if bids and asks have valid format
        if not all(len(bid) >= 2 for bid in bids) or not all(len(ask) >= 2 for ask in asks):
            return False
            
        return True
        
    def _generate_synthetic_order_book(self, symbol: str, price: float = None) -> Dict:
        """
        Generate synthetic order book data.
        
        Args:
            symbol: Symbol to generate order book for
            price: Current price (optional)
            
        Returns:
            Synthetic order book dictionary
        """
        try:
            # Use provided price or estimate from other sources
            if price is None:
                # Try to get price from other sources
                try:
                    payload = {"type": "allMids"}
                    response = requests.post(f"{self.base_url}/info", json=payload)
                    
                    if response.status_code == 200:
                        all_mids = response.json()
                        price = float(all_mids.get(symbol, 0))
                except Exception:
                    # Default prices if all else fails
                    default_prices = {
                        "BTC": 107000.0,
                        "ETH": 2500.0,
                        "SOL": 175.0
                    }
                    price = default_prices.get(symbol, 100.0)
                    
            if price <= 0:
                price = 100.0  # Fallback default
                
            # Generate synthetic bids and asks
            bids = []
            asks = []
            
            # Number of levels
            num_levels = 10
            
            # Spread as percentage of price
            spread_pct = 0.001  # 0.1%
            
            # Level step as percentage of price
            level_step_pct = 0.0005  # 0.05%
            
            # Generate bids (buy orders)
            for i in range(num_levels):
                level_price = price * (1 - spread_pct/2 - i * level_step_pct)
                level_size = 1.0 + (num_levels - i) * 0.5  # Higher size for better prices
                bids.append([str(level_price), str(level_size)])
                
            # Generate asks (sell orders)
            for i in range(num_levels):
                level_price = price * (1 + spread_pct/2 + i * level_step_pct)
                level_size = 1.0 + (num_levels - i) * 0.5  # Higher size for better prices
                asks.append([str(level_price), str(level_size)])
                
            # Create order book dictionary
            order_book = {
                "bids": bids,
                "asks": asks,
                "synthetic": True  # Mark as synthetic
            }
            
            return order_book
        except Exception as e:
            logger.error(f"Error generating synthetic order book: {str(e)}")
            
            # Return minimal valid order book
            return {
                "bids": [["100.0", "1.0"]],
                "asks": [["101.0", "1.0"]],
                "synthetic": True
            }
            
    def analyze_order_book(self, order_book: Dict, price: float = None) -> Dict:
        """
        Analyze order book to extract useful metrics.
        
        Args:
            order_book: Order book dictionary
            price: Current price (optional)
            
        Returns:
            Dictionary of order book metrics
        """
        try:
            if not self._is_valid_order_book(order_book):
                logger.warning("Invalid order book for analysis")
                return {
                    "bid_ask_imbalance": 0.0,
                    "spread_pct": 0.001,
                    "depth_imbalance": 0.0,
                    "is_synthetic": True
                }
                
            # Extract bids and asks
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            # Calculate bid-ask spread
            best_bid = float(bids[0][0]) if bids else 0
            best_ask = float(asks[0][0]) if asks else 0
            
            if best_bid <= 0 or best_ask <= 0:
                spread_pct = 0.001  # Default 0.1%
            else:
                mid_price = (best_bid + best_ask) / 2
                spread_pct = (best_ask - best_bid) / mid_price
                
            # Calculate volume imbalance at top 5 levels
            bid_volume = sum(float(bid[1]) for bid in bids[:5]) if len(bids) >= 5 else sum(float(bid[1]) for bid in bids)
            ask_volume = sum(float(ask[1]) for ask in asks[:5]) if len(asks) >= 5 else sum(float(ask[1]) for ask in asks)
            
            total_volume = bid_volume + ask_volume
            if total_volume > 0:
                bid_ask_imbalance = (bid_volume - ask_volume) / total_volume
            else:
                bid_ask_imbalance = 0.0
                
            # Calculate depth imbalance (weighted by distance from mid)
            if price is None:
                price = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 100.0
                
            weighted_bid_volume = 0
            weighted_ask_volume = 0
            
            for i, bid in enumerate(bids[:10]):
                bid_price = float(bid[0])
                bid_size = float(bid[1])
                distance = (price - bid_price) / price
                weight = 1.0 / (1.0 + distance * 10)  # Higher weight for closer levels
                weighted_bid_volume += bid_size * weight
                
            for i, ask in enumerate(asks[:10]):
                ask_price = float(ask[0])
                ask_size = float(ask[1])
                distance = (ask_price - price) / price
                weight = 1.0 / (1.0 + distance * 10)  # Higher weight for closer levels
                weighted_ask_volume += ask_size * weight
                
            total_weighted_volume = weighted_bid_volume + weighted_ask_volume
            if total_weighted_volume > 0:
                depth_imbalance = (weighted_bid_volume - weighted_ask_volume) / total_weighted_volume
            else:
                depth_imbalance = 0.0
                
            return {
                "bid_ask_imbalance": bid_ask_imbalance,
                "spread_pct": spread_pct,
                "depth_imbalance": depth_imbalance,
                "is_synthetic": order_book.get('synthetic', False)
            }
        except Exception as e:
            logger.error(f"Error analyzing order book: {str(e)}")
            return {
                "bid_ask_imbalance": 0.0,
                "spread_pct": 0.001,
                "depth_imbalance": 0.0,
                "is_synthetic": True
            }

# Create singleton instance
order_book_handler = OrderBookHandler()

def get_order_book(symbol: str) -> Dict:
    """
    Get order book using the singleton handler.
    
    Args:
        symbol: Symbol to get order book for
        
    Returns:
        Order book dictionary
    """
    return order_book_handler.get_order_book(symbol)
    
def analyze_order_book(order_book: Dict, price: float = None) -> Dict:
    """
    Analyze order book using the singleton handler.
    
    Args:
        order_book: Order book dictionary
        price: Current price (optional)
        
    Returns:
        Dictionary of order book metrics
    """
    return order_book_handler.analyze_order_book(order_book, price)
