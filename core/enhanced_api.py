#!/usr/bin/env python3
"""
Enhanced API Module with Real-time Data Feeds and Advanced Features
-----------------------------------------------------------------
Integrates all missing API endpoints and data feeds from master_bot.py including:
• Real-time price feeds with technical indicators
• Enhanced wallet equity monitoring
• Advanced order management
• Live market data streaming
• Position tracking with P&L calculations
"""

import asyncio
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
import numpy as np
import pandas as pd

# Technical analysis imports
try:
    import ta
    from ta.trend import macd, macd_signal, ADXIndicator
    from ta.momentum import rsi, StochasticOscillator
    from ta.volatility import BollingerBands, AverageTrueRange
except ImportError:
    logging.warning("Technical analysis library 'ta' not found. Some features may be limited.")

# Hyperliquid SDK imports
try:
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange
    from hyperliquid.utils import constants
    from hyperliquid.websocket_manager import WebsocketManager
except ImportError as e:
    logging.error(f"Error importing Hyperliquid SDK: {e}")
    raise

from utils.logger import get_logger, TradingLogger
from utils.config_manager import ConfigManager
from utils.security import SecurityManager

logger = get_logger(__name__)


class EnhancedHyperliquidAPI:
    """Enhanced Hyperliquid API with comprehensive real-time features"""
    
    def __init__(self, config_path: str = None, testnet: bool = False):
        """
        Initialize the enhanced API wrapper
        
        Args:
            config_path: Path to configuration file
            testnet: Whether to use testnet (default: False for mainnet)
        """
        self.config = ConfigManager(config_path)
        self.security = SecurityManager()
        self.testnet = testnet
        
        # API URLs
        self.api_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        
        # Initialize API components
        self.info = Info(self.api_url, skip_ws=True)
        self.exchange = None
        self.ws_manager = None
        
        # Trading state
        self.account_address = None
        self.is_authenticated = False
        self.positions = {}
        self.orders = {}
        self.account_value = 0.0
        
        # Real-time data storage
        self.price_data = {}
        self.market_data = {}
        self.historical_data = {}
        self.technical_indicators = {}
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Real-time update threads
        self.price_update_thread = None
        self.market_data_thread = None
        self.running = False
        
        logger.info(f"Enhanced Hyperliquid API initialized ({'testnet' if testnet else 'mainnet'})")
    
    def authenticate(self, private_key: str = None, wallet_address: str = None) -> bool:
        """
        Authenticate with Hyperliquid using private key
        
        Args:
            private_key: Private key for authentication
            wallet_address: Wallet address
            
        Returns:
            bool: True if authentication successful
        """
        try:
            if not private_key:
                private_key = self.security.get_private_key()
            
            if not wallet_address:
                wallet_address = self.config.get('wallet_address')
            
            # Initialize exchange with authentication
            self.exchange = Exchange(
                account=private_key,
                base_url=self.api_url,
                skip_ws=False
            )
            
            self.account_address = wallet_address
            self.is_authenticated = True
            
            # Initialize WebSocket manager for real-time data
            self.ws_manager = WebsocketManager(
                base_url=self.api_url.replace('https', 'wss'),
                skip_ws=False
            )
            
            # Start real-time data feeds
            self.start_real_time_feeds()
            
            logger.info(f"Successfully authenticated with address: {wallet_address}")
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def start_real_time_feeds(self):
        """Start real-time data feed threads"""
        try:
            self.running = True
            
            # Start price update thread
            self.price_update_thread = threading.Thread(
                target=self._price_update_loop, 
                daemon=True
            )
            self.price_update_thread.start()
            
            # Start market data thread
            self.market_data_thread = threading.Thread(
                target=self._market_data_loop, 
                daemon=True
            )
            self.market_data_thread.start()
            
            logger.info("Real-time data feeds started")
            
        except Exception as e:
            logger.error(f"Failed to start real-time feeds: {e}")
    
    def stop_real_time_feeds(self):
        """Stop real-time data feeds"""
        try:
            self.running = False
            
            if self.price_update_thread and self.price_update_thread.is_alive():
                self.price_update_thread.join(timeout=5)
            
            if self.market_data_thread and self.market_data_thread.is_alive():
                self.market_data_thread.join(timeout=5)
            
            logger.info("Real-time data feeds stopped")
            
        except Exception as e:
            logger.error(f"Error stopping real-time feeds: {e}")
    
    def _price_update_loop(self):
        """Continuous price update loop"""
        while self.running:
            try:
                # Update all tracked symbols
                for symbol in self.get_tracked_symbols():
                    price_data = self.fetch_enhanced_price_data(symbol)
                    if price_data:
                        self.price_data[symbol] = price_data
                        
                        # Update historical data
                        self._update_historical_data(symbol, price_data)
                        
                        # Calculate technical indicators
                        self._calculate_technical_indicators(symbol)
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in price update loop: {e}")
                time.sleep(5)
    
    def _market_data_loop(self):
        """Continuous market data update loop"""
        while self.running:
            try:
                # Update account state
                if self.is_authenticated:
                    account_state = self.get_enhanced_account_state()
                    if account_state:
                        self.account_value = account_state.get('account_value', 0)
                        self.positions = {pos['coin']: pos for pos in account_state.get('positions', [])}
                        self.orders = {order['oid']: order for order in account_state.get('orders', [])}
                
                # Update market metadata
                self._update_market_metadata()
                
                time.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                logger.error(f"Error in market data loop: {e}")
                time.sleep(10)
    
    def _rate_limit(self):
        """Implement rate limiting to avoid API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_enhanced_account_state(self) -> Dict[str, Any]:
        """
        Get comprehensive account state with enhanced details
        
        Returns:
            Dict containing detailed account state information
        """
        self._rate_limit()
        
        try:
            user_state = self.info.user_state(self.account_address)
            
            # Extract comprehensive account information
            margin_summary = user_state.get('marginSummary', {})
            cross_summary = user_state.get('crossMarginSummary', {})
            
            account_state = {
                'timestamp': datetime.now(),
                'account_value': float(margin_summary.get('accountValue', 0)),
                'total_margin_used': float(margin_summary.get('totalMarginUsed', 0)),
                'total_ntl_pos': float(margin_summary.get('totalNtlPos', 0)),
                'total_raw_usd': float(margin_summary.get('totalRawUsd', 0)),
                'cross_account_value': float(cross_summary.get('accountValue', 0)),
                'cross_total_margin_used': float(cross_summary.get('totalMarginUsed', 0)),
                'withdrawable': float(user_state.get('withdrawable', 0)),
                'positions': [],
                'orders': [],
                'funding_history': [],
                'trade_history': []
            }
            
            # Process positions with enhanced details
            for position in user_state.get('assetPositions', []):
                pos_data = position.get('position', {})
                if float(pos_data.get('szi', 0)) != 0:  # Only active positions
                    
                    # Get current market price for P&L calculation
                    coin = pos_data.get('coin')
                    current_price = self.get_current_price(f"{coin}-USD-PERP")
                    entry_price = float(pos_data.get('entryPx', 0))
                    size = float(pos_data.get('szi', 0))
                    
                    # Calculate enhanced P&L metrics
                    if size > 0:  # Long position
                        unrealized_pnl = size * (current_price - entry_price) if current_price > 0 else 0
                        pnl_percent = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                    else:  # Short position
                        unrealized_pnl = abs(size) * (entry_price - current_price) if current_price > 0 else 0
                        pnl_percent = ((entry_price - current_price) / entry_price * 100) if entry_price > 0 else 0
                    
                    position_info = {
                        'coin': coin,
                        'symbol': f"{coin}-USD-PERP",
                        'size': size,
                        'side': 1 if size > 0 else 2,
                        'entry_px': entry_price,
                        'current_price': current_price,
                        'position_value': float(pos_data.get('positionValue', 0)),
                        'unrealized_pnl': unrealized_pnl,
                        'pnl_percent': pnl_percent,
                        'leverage': pos_data.get('leverage', {}).get('value', 1),
                        'margin_used': float(pos_data.get('marginUsed', 0)),
                        'liquidation_px': float(pos_data.get('liquidationPx', 0)),
                        'max_leverage': pos_data.get('maxLeverage', 1),
                        'funding': float(pos_data.get('cumFunding', {}).get('allTime', 0))
                    }
                    account_state['positions'].append(position_info)
            
            # Process open orders with enhanced details
            open_orders = self.info.open_orders(self.account_address)
            for order in open_orders:
                order_info = {
                    'coin': order.get('coin'),
                    'symbol': f"{order.get('coin')}-USD-PERP",
                    'side': order.get('side'),
                    'sz': float(order.get('sz', 0)),
                    'limit_px': float(order.get('limitPx', 0)),
                    'oid': order.get('oid'),
                    'timestamp': order.get('timestamp'),
                    'order_type': order.get('orderType'),
                    'reduce_only': order.get('reduceOnly', False),
                    'time_in_force': order.get('tif', 'Gtc'),
                    'cloid': order.get('cloid')
                }
                account_state['orders'].append(order_info)
            
            # Get recent funding history
            try:
                funding_history = self.info.funding_history(self.account_address, startTime=int((datetime.now() - timedelta(days=7)).timestamp() * 1000))
                account_state['funding_history'] = funding_history[:10]  # Last 10 funding payments
            except:
                pass
            
            # Get recent trade history
            try:
                fills = self.info.user_fills(self.account_address)
                account_state['trade_history'] = fills[:20]  # Last 20 trades
            except:
                pass
            
            return account_state
            
        except Exception as e:
            logger.error(f"Failed to get enhanced account state: {e}")
            return {}
    
    def fetch_enhanced_price_data(self, symbol: str) -> Optional[Dict]:
        """
        Fetch comprehensive price data with technical analysis
        
        Args:
            symbol: Trading symbol (e.g., 'BTC-USD-PERP')
            
        Returns:
            Dict containing enhanced price data
        """
        try:
            self._rate_limit()
            
            # Parse symbol
            coin = self._parse_coin_from_symbol(symbol)
            is_perp = symbol.endswith('-PERP')
            
            if is_perp:
                # Get perpetual market data
                meta = self.info.meta()
                universe = meta.get('universe', [])
                
                for market in universe:
                    if market.get('name') == coin:
                        # Get current price data
                        mid_px = float(market.get('midPx', 0))
                        
                        # Get 24h statistics
                        day_ntl_vlm = float(market.get('dayNtlVlm', 0))
                        funding = float(market.get('funding', 0))
                        open_interest = float(market.get('openInterest', 0))
                        
                        # Get orderbook for spread calculation
                        l2_book = self.info.l2_snapshot(coin)
                        
                        bid_price = 0
                        ask_price = 0
                        if l2_book and 'levels' in l2_book:
                            levels = l2_book['levels']
                            if levels and len(levels) >= 2:
                                bids = levels[0]  # Bid levels
                                asks = levels[1]  # Ask levels
                                
                                if bids:
                                    bid_price = float(bids[0]['px'])
                                if asks:
                                    ask_price = float(asks[0]['px'])
                        
                        spread = ask_price - bid_price if ask_price > 0 and bid_price > 0 else 0
                        spread_pct = (spread / mid_px * 100) if mid_px > 0 else 0
                        
                        # Get recent candle data for additional metrics
                        candles = self.get_candle_data(coin, '1m', 100)
                        
                        price_data = {
                            'symbol': symbol,
                            'coin': coin,
                            'timestamp': datetime.now(),
                            'price': mid_px,
                            'bid': bid_price,
                            'ask': ask_price,
                            'spread': spread,
                            'spread_pct': spread_pct,
                            'volume_24h': day_ntl_vlm,
                            'funding_rate': funding,
                            'funding_rate_8h': funding * 3,  # Approximate 8h rate
                            'open_interest': open_interest,
                            'market_type': 'perp'
                        }
                        
                        # Add OHLCV data if available
                        if candles and len(candles) > 0:
                            latest_candle = candles[-1]
                            price_data.update({
                                'open': float(latest_candle.get('o', mid_px)),
                                'high': float(latest_candle.get('h', mid_px)),
                                'low': float(latest_candle.get('l', mid_px)),
                                'close': float(latest_candle.get('c', mid_px)),
                                'volume': float(latest_candle.get('v', 0))
                            })
                            
                            # Calculate price changes
                            if len(candles) > 1:
                                prev_close = float(candles[-2].get('c', mid_px))
                                price_change = mid_px - prev_close
                                price_change_pct = (price_change / prev_close * 100) if prev_close > 0 else 0
                                
                                price_data.update({
                                    'price_change': price_change,
                                    'price_change_pct': price_change_pct
                                })
                        
                        return price_data
            else:
                # Get spot market data
                spot_meta = self.info.spot_meta()
                tokens = spot_meta.get('tokens', [])
                
                for token in tokens:
                    if token.get('name') == coin:
                        mid_px = float(token.get('midPx', 0))
                        
                        price_data = {
                            'symbol': symbol,
                            'coin': coin,
                            'timestamp': datetime.now(),
                            'price': mid_px,
                            'market_type': 'spot'
                        }
                        
                        return price_data
            
            return None
            
        except Exception as e:
            logger.warning(f"Error fetching enhanced price data for {symbol}: {e}")
            return None
    
    def get_candle_data(self, coin: str, interval: str = '1m', limit: int = 100) -> List[Dict]:
        """
        Get historical candle data
        
        Args:
            coin: Coin symbol (e.g., 'BTC')
            interval: Time interval ('1m', '5m', '1h', '1d')
            limit: Number of candles to fetch
            
        Returns:
            List of candle data
        """
        try:
            self._rate_limit()
            
            # Calculate start time based on interval and limit
            interval_minutes = {
                '1m': 1,
                '5m': 5,
                '15m': 15,
                '1h': 60,
                '4h': 240,
                '1d': 1440
            }
            
            minutes = interval_minutes.get(interval, 1)
            start_time = datetime.now() - timedelta(minutes=minutes * limit)
            start_timestamp = int(start_time.timestamp() * 1000)
            
            # Fetch candle data
            candles = self.info.candles_snapshot(
                coin=coin,
                interval=interval,
                startTime=start_timestamp
            )
            
            return candles if candles else []
            
        except Exception as e:
            logger.warning(f"Error fetching candle data for {coin}: {e}")
            return []
    
    def _update_historical_data(self, symbol: str, price_data: Dict):
        """Update historical price data for technical analysis"""
        try:
            if symbol not in self.historical_data:
                self.historical_data[symbol] = []
            
            # Add new data point
            data_point = {
                'timestamp': price_data['timestamp'],
                'price': price_data['price'],
                'volume': price_data.get('volume', 0),
                'high': price_data.get('high', price_data['price']),
                'low': price_data.get('low', price_data['price']),
                'open': price_data.get('open', price_data['price']),
                'close': price_data['price']
            }
            
            self.historical_data[symbol].append(data_point)
            
            # Limit historical data size
            if len(self.historical_data[symbol]) > 1000:
                self.historical_data[symbol] = self.historical_data[symbol][-500:]
                
        except Exception as e:
            logger.error(f"Error updating historical data: {e}")
    
    def _calculate_technical_indicators(self, symbol: str):
        """Calculate technical indicators for symbol"""
        try:
            if symbol not in self.historical_data or len(self.historical_data[symbol]) < 50:
                return
            
            # Convert to DataFrame for technical analysis
            df = pd.DataFrame(self.historical_data[symbol])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Calculate indicators
            indicators = {}
            
            # Moving averages
            indicators['sma_20'] = df['close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else None
            indicators['sma_50'] = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else None
            indicators['ema_12'] = df['close'].ewm(span=12).mean().iloc[-1] if len(df) >= 12 else None
            indicators['ema_26'] = df['close'].ewm(span=26).mean().iloc[-1] if len(df) >= 26 else None
            
            # RSI
            if len(df) >= 14:
                try:
                    indicators['rsi'] = rsi(df['close'], window=14).iloc[-1]
                except:
                    indicators['rsi'] = None
            
            # MACD
            if len(df) >= 26:
                try:
                    macd_line = macd(df['close'], window_slow=26, window_fast=12)
                    macd_sig = macd_signal(df['close'], window_slow=26, window_fast=12, window_sign=9)
                    indicators['macd'] = macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else None
                    indicators['macd_signal'] = macd_sig.iloc[-1] if not pd.isna(macd_sig.iloc[-1]) else None
                    indicators['macd_histogram'] = (macd_line - macd_sig).iloc[-1] if indicators['macd'] and indicators['macd_signal'] else None
                except:
                    indicators['macd'] = None
                    indicators['macd_signal'] = None
                    indicators['macd_histogram'] = None
            
            # Bollinger Bands
            if len(df) >= 20:
                try:
                    bb = BollingerBands(df['close'], window=20, window_dev=2)
                    indicators['bb_upper'] = bb.bollinger_hband().iloc[-1]
                    indicators['bb_middle'] = bb.bollinger_mavg().iloc[-1]
                    indicators['bb_lower'] = bb.bollinger_lband().iloc[-1]
                    indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle'] * 100
                except:
                    indicators['bb_upper'] = None
                    indicators['bb_middle'] = None
                    indicators['bb_lower'] = None
                    indicators['bb_width'] = None
            
            # ATR
            if len(df) >= 14:
                try:
                    atr_indicator = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
                    indicators['atr'] = atr_indicator.average_true_range().iloc[-1]
                except:
                    indicators['atr'] = None
            
            # Volume indicators
            if len(df) >= 20:
                indicators['volume_sma'] = df['volume'].rolling(20).mean().iloc[-1]
                indicators['volume_ratio'] = df['volume'].iloc[-1] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1
            
            # Store indicators
            self.technical_indicators[symbol] = {
                'timestamp': datetime.now(),
                'indicators': indicators
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators for {symbol}: {e}")
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        try:
            if symbol in self.price_data:
                return self.price_data[symbol].get('price', 0)
            
            # Fallback to direct API call
            price_data = self.fetch_enhanced_price_data(symbol)
            return price_data.get('price', 0) if price_data else 0
            
        except Exception as e:
            logger.warning(f"Error getting current price for {symbol}: {e}")
            return 0
    
    def get_real_time_data(self, symbol: str) -> Dict:
        """Get comprehensive real-time data for symbol"""
        try:
            data = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'price_data': self.price_data.get(symbol, {}),
                'technical_indicators': self.technical_indicators.get(symbol, {}),
                'account_value': self.account_value,
                'positions': [pos for pos in self.positions.values() if pos.get('coin') in symbol],
                'orders': list(self.orders.values())
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting real-time data: {e}")
            return {}
    
    def get_tracked_symbols(self) -> List[str]:
        """Get list of symbols to track"""
        # Default symbols to track
        default_symbols = [
            'BTC-USD-PERP', 'ETH-USD-PERP', 'SOL-USD-PERP',
            'AVAX-USD-PERP', 'MATIC-USD-PERP', 'LINK-USD-PERP'
        ]
        
        # Add symbols from current positions
        position_symbols = [f"{pos.get('coin', '')}-USD-PERP" for pos in self.positions.values()]
        
        # Combine and deduplicate
        all_symbols = list(set(default_symbols + position_symbols))
        return [s for s in all_symbols if s.endswith('-PERP')]
    
    def _update_market_metadata(self):
        """Update market metadata and statistics"""
        try:
            # Get market metadata
            meta = self.info.meta()
            if meta:
                self.market_data['meta'] = meta
                self.market_data['last_updated'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating market metadata: {e}")
    
    def _parse_coin_from_symbol(self, symbol: str) -> str:
        """Parse coin from trading symbol"""
        if symbol.endswith('-USD-PERP'):
            return symbol[:-9]
        elif symbol.endswith('-USD-SPOT'):
            return symbol[:-9]
        else:
            return symbol.split('-')[0] if '-' in symbol else symbol
    
    def place_enhanced_market_order(self, coin: str, is_buy: bool, sz: float, 
                                   reduce_only: bool = False, cloid: str = None,
                                   slippage_tolerance: float = 0.01) -> Dict[str, Any]:
        """
        Place market order with enhanced features
        
        Args:
            coin: Trading pair (e.g., 'BTC')
            is_buy: True for buy, False for sell
            sz: Size in USD for buy orders, size in coins for sell orders
            reduce_only: Whether this is a reduce-only order
            cloid: Client order ID
            slippage_tolerance: Maximum acceptable slippage
            
        Returns:
            Dict containing order result with enhanced details
        """
        if not self.is_authenticated:
            raise Exception("Not authenticated. Call authenticate() first.")
        
        self._rate_limit()
        
        try:
            # Get current market data for slippage calculation
            symbol = f"{coin}-USD-PERP"
            price_data = self.price_data.get(symbol, {})
            current_price = price_data.get('price', 0)
            spread = price_data.get('spread', 0)
            
            # Calculate expected slippage
            expected_slippage = spread / current_price if current_price > 0 else 0
            
            if expected_slippage > slippage_tolerance:
                logger.warning(f"High slippage detected: {expected_slippage:.4f} > {slippage_tolerance:.4f}")
            
            # Place order
            order_result = self.exchange.market_open(coin, is_buy, sz, reduce_only, cloid)
            
            # Enhanced result with market context
            enhanced_result = {
                'order_result': order_result,
                'market_context': {
                    'symbol': symbol,
                    'current_price': current_price,
                    'spread': spread,
                    'expected_slippage': expected_slippage,
                    'slippage_tolerance': slippage_tolerance,
                    'timestamp': datetime.now()
                },
                'order_details': {
                    'coin': coin,
                    'is_buy': is_buy,
                    'size': sz,
                    'reduce_only': reduce_only,
                    'cloid': cloid
                }
            }
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Enhanced market order failed: {e}")
            return {'error': str(e), 'success': False}
    
    def get_enhanced_performance_metrics(self) -> Dict[str, Any]:
        """Get enhanced performance metrics"""
        try:
            if not self.is_authenticated:
                return {}
            
            # Get account state
            account_state = self.get_enhanced_account_state()
            
            # Calculate performance metrics
            total_equity = account_state.get('account_value', 0)
            total_unrealized_pnl = sum(pos.get('unrealized_pnl', 0) for pos in account_state.get('positions', []))
            
            # Get trade history for realized P&L
            trade_history = account_state.get('trade_history', [])
            realized_pnl = sum(float(trade.get('closedPnl', 0)) for trade in trade_history)
            
            # Calculate win rate
            profitable_trades = [t for t in trade_history if float(t.get('closedPnl', 0)) > 0]
            win_rate = (len(profitable_trades) / len(trade_history) * 100) if trade_history else 0
            
            # Calculate funding costs
            funding_history = account_state.get('funding_history', [])
            total_funding = sum(float(f.get('delta', 0)) for f in funding_history)
            
            metrics = {
                'timestamp': datetime.now(),
                'total_equity': total_equity,
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_realized_pnl': realized_pnl,
                'total_pnl': total_unrealized_pnl + realized_pnl,
                'total_funding': total_funding,
                'win_rate': win_rate,
                'total_trades': len(trade_history),
                'profitable_trades': len(profitable_trades),
                'losing_trades': len(trade_history) - len(profitable_trades),
                'active_positions': len(account_state.get('positions', [])),
                'open_orders': len(account_state.get('orders', []))
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def shutdown(self):
        """Shutdown the enhanced API"""
        try:
            self.stop_real_time_feeds()
            
            if self.ws_manager:
                self.ws_manager.close()
            
            logger.info("Enhanced Hyperliquid API shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during API shutdown: {e}")


# Backward compatibility wrapper
class EnhancedAPI(EnhancedHyperliquidAPI):
    """Backward compatibility wrapper"""
    pass

