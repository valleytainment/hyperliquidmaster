"""
Enhanced Hyperliquid API Wrapper - BEST BRANCH EDITION
🏆 ULTIMATE VERSION with maximum optimizations and enhancements
Provides comprehensive trading functionality with enhanced features
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal

# Updated imports for Hyperliquid SDK v0.15.0+
try:
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange
    from hyperliquid.utils import constants
    # Fixed: Use WebsocketManager (lowercase 's') instead of WebSocketManager
    from hyperliquid.websocket_manager import WebsocketManager
except ImportError as e:
    print(f"Error importing Hyperliquid SDK: {e}")
    print("Please install the latest version: pip install hyperliquid-python-sdk")
    raise

from utils.logger import get_logger, TradingLogger
from utils.config_manager import ConfigManager
from utils.security import SecurityManager

logger = get_logger(__name__)


class EnhancedHyperliquidAPI:
    """Enhanced Hyperliquid API wrapper with additional trading bot functionality"""
    
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
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
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
            
            # Initialize WebSocket manager
            self.ws_manager = WebsocketManager(
                base_url=self.api_url.replace('https', 'wss'),
                skip_ws=False
            )
            
            logger.info(f"Successfully authenticated with address: {wallet_address}")
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def _rate_limit(self):
        """Implement rate limiting to avoid API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_account_state(self) -> Dict[str, Any]:
        """
        Get comprehensive account state including positions, orders, and balances
        
        Returns:
            Dict containing account state information
        """
        self._rate_limit()
        
        try:
            user_state = self.info.user_state(self.account_address)
            
            # Extract key information
            account_state = {
                'account_value': float(user_state.get('marginSummary', {}).get('accountValue', 0)),
                'total_margin_used': float(user_state.get('marginSummary', {}).get('totalMarginUsed', 0)),
                'total_ntl_pos': float(user_state.get('marginSummary', {}).get('totalNtlPos', 0)),
                'total_raw_usd': float(user_state.get('marginSummary', {}).get('totalRawUsd', 0)),
                'positions': [],
                'orders': []
            }
            
            # Process positions
            for position in user_state.get('assetPositions', []):
                pos_data = position.get('position', {})
                if float(pos_data.get('szi', 0)) != 0:  # Only active positions
                    account_state['positions'].append({
                        'coin': pos_data.get('coin'),
                        'size': float(pos_data.get('szi', 0)),
                        'entry_px': float(pos_data.get('entryPx', 0)),
                        'position_value': float(pos_data.get('positionValue', 0)),
                        'unrealized_pnl': float(pos_data.get('unrealizedPnl', 0)),
                        'leverage': pos_data.get('leverage', {}).get('value', 1)
                    })
            
            # Process open orders
            open_orders = self.info.open_orders(self.account_address)
            for order in open_orders:
                account_state['orders'].append({
                    'coin': order.get('coin'),
                    'side': order.get('side'),
                    'sz': float(order.get('sz', 0)),
                    'limit_px': float(order.get('limitPx', 0)),
                    'oid': order.get('oid'),
                    'timestamp': order.get('timestamp'),
                    'order_type': order.get('orderType')
                })
            
            # Update internal state
            self.account_value = account_state['account_value']
            self.positions = {pos['coin']: pos for pos in account_state['positions']}
            self.orders = {order['oid']: order for order in account_state['orders']}
            
            return account_state
            
        except Exception as e:
            logger.error(f"Failed to get account state: {e}")
            return {}
    
    def place_market_order(self, coin: str, is_buy: bool, sz: float, 
                          reduce_only: bool = False, cloid: str = None) -> Dict[str, Any]:
        """
        Place a market order
        
        Args:
            coin: Trading pair (e.g., 'BTC')
            is_buy: True for buy, False for sell
            sz: Size in USD for buy orders, size in coins for sell orders
            reduce_only: Whether this is a reduce-only order
            cloid: Client order ID
            
        Returns:
            Dict containing order result
        """
        if not self.is_authenticated:
            raise Exception("Not authenticated. Call authenticate() first.")
        
        self._rate_limit()
        
        try:
            result = self.exchange.market_open(
                coin=coin,
                is_buy=is_buy,
                sz=sz,
                reduce_only=reduce_only,
                cloid=cloid
            )
            
            logger.info(f"Market order placed: {coin} {'BUY' if is_buy else 'SELL'} {sz}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def place_limit_order(self, coin: str, is_buy: bool, sz: float, limit_px: float,
                         reduce_only: bool = False, post_only: bool = False, 
                         cloid: str = None) -> Dict[str, Any]:
        """
        Place a limit order
        
        Args:
            coin: Trading pair (e.g., 'BTC')
            is_buy: True for buy, False for sell
            sz: Order size
            limit_px: Limit price
            reduce_only: Whether this is a reduce-only order
            post_only: Whether this is a post-only order
            cloid: Client order ID
            
        Returns:
            Dict containing order result
        """
        if not self.is_authenticated:
            raise Exception("Not authenticated. Call authenticate() first.")
        
        self._rate_limit()
        
        try:
            order_type = {'limit': {'tif': 'Gtc'}}
            if post_only:
                order_type = {'limit': {'tif': 'Alo'}}
            
            result = self.exchange.order(
                coin=coin,
                is_buy=is_buy,
                sz=sz,
                limit_px=limit_px,
                order_type=order_type,
                reduce_only=reduce_only,
                cloid=cloid
            )
            
            logger.info(f"Limit order placed: {coin} {'BUY' if is_buy else 'SELL'} {sz} @ {limit_px}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to place limit order: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def cancel_order(self, coin: str, oid: int) -> Dict[str, Any]:
        """Cancel an order by order ID"""
        if not self.is_authenticated:
            raise Exception("Not authenticated. Call authenticate() first.")
        
        self._rate_limit()
        
        try:
            result = self.exchange.cancel(coin, oid)
            logger.info(f"Order cancelled: {coin} OID {oid}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def cancel_all_orders(self, coin: str = None) -> Dict[str, Any]:
        """Cancel all orders for a specific coin or all coins"""
        if not self.is_authenticated:
            raise Exception("Not authenticated. Call authenticate() first.")
        
        try:
            open_orders = self.info.open_orders(self.account_address)
            results = []
            
            for order in open_orders:
                if coin is None or order.get('coin') == coin:
                    result = self.cancel_order(order.get('coin'), order.get('oid'))
                    results.append(result)
            
            logger.info(f"Cancelled {len(results)} orders" + (f" for {coin}" if coin else ""))
            return {'status': 'ok', 'cancelled_orders': len(results)}
            
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def close_position(self, coin: str, percentage: float = 100.0) -> Dict[str, Any]:
        """
        Close a position (partially or completely)
        
        Args:
            coin: Trading pair
            percentage: Percentage of position to close (default: 100%)
            
        Returns:
            Dict containing result
        """
        if not self.is_authenticated:
            raise Exception("Not authenticated. Call authenticate() first.")
        
        try:
            # Get current position
            account_state = self.get_account_state()
            position = None
            
            for pos in account_state['positions']:
                if pos['coin'] == coin:
                    position = pos
                    break
            
            if not position:
                return {'status': 'error', 'error': f'No position found for {coin}'}
            
            # Calculate size to close
            current_size = position['size']
            close_size = abs(current_size) * (percentage / 100.0)
            
            # Determine order direction (opposite of position)
            is_buy = current_size < 0  # Buy to close short, sell to close long
            
            # Place market order to close
            result = self.place_market_order(
                coin=coin,
                is_buy=is_buy,
                sz=close_size,
                reduce_only=True
            )
            
            logger.info(f"Position closed: {coin} {percentage}% ({close_size})")
            return result
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_market_data(self, coin: str) -> Dict[str, Any]:
        """Get current market data for a coin"""
        self._rate_limit()
        
        try:
            # Get all mids (current prices)
            all_mids = self.info.all_mids()
            
            # Get 24h stats
            meta = self.info.meta()
            
            market_data = {
                'coin': coin,
                'price': 0.0,
                'volume_24h': 0.0,
                'funding_rate': 0.0,
                'open_interest': 0.0
            }
            
            # Find price for the coin
            if coin in all_mids:
                market_data['price'] = float(all_mids[coin])
            
            # Find additional data in meta
            for universe_item in meta.get('universe', []):
                if universe_item.get('name') == coin:
                    market_data.update({
                        'volume_24h': float(universe_item.get('dayNtlVlm', 0)),
                        'funding_rate': float(universe_item.get('funding', 0)),
                        'open_interest': float(universe_item.get('openInterest', 0))
                    })
                    break
            
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to get market data for {coin}: {e}")
            return {}
    
    def get_candles(self, coin: str, interval: str, start_time: int = None, 
                   end_time: int = None) -> List[Dict[str, Any]]:
        """
        Get historical candle data
        
        Args:
            coin: Trading pair
            interval: Time interval ('1m', '5m', '15m', '1h', '4h', '1d')
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            
        Returns:
            List of candle data
        """
        self._rate_limit()
        
        try:
            # Convert interval to API format
            interval_map = {
                '1m': '1m',
                '5m': '5m', 
                '15m': '15m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d'
            }
            
            api_interval = interval_map.get(interval, '1h')
            
            # Get candles
            candles = self.info.candles_snapshot(
                coin=coin,
                interval=api_interval,
                startTime=start_time,
                endTime=end_time
            )
            
            # Format candle data
            formatted_candles = []
            for candle in candles:
                formatted_candles.append({
                    'timestamp': candle['t'],
                    'open': float(candle['o']),
                    'high': float(candle['h']),
                    'low': float(candle['l']),
                    'close': float(candle['c']),
                    'volume': float(candle['v'])
                })
            
            return formatted_candles
            
        except Exception as e:
            logger.error(f"Failed to get candles for {coin}: {e}")
            return []
    
    def set_leverage(self, coin: str, leverage: int, is_cross: bool = True) -> Dict[str, Any]:
        """Set leverage for a trading pair"""
        if not self.is_authenticated:
            raise Exception("Not authenticated. Call authenticate() first.")
        
        self._rate_limit()
        
        try:
            result = self.exchange.update_leverage(leverage, coin, is_cross)
            logger.info(f"Leverage set for {coin}: {leverage}x ({'cross' if is_cross else 'isolated'})")
            return result
            
        except Exception as e:
            logger.error(f"Failed to set leverage: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_funding_history(self, coin: str, start_time: int = None, 
                           end_time: int = None) -> List[Dict[str, Any]]:
        """Get funding rate history for a coin"""
        self._rate_limit()
        
        try:
            funding_history = self.info.funding_history(
                coin=coin,
                startTime=start_time,
                endTime=end_time
            )
            
            return [
                {
                    'timestamp': item['time'],
                    'funding_rate': float(item['fundingRate']),
                    'premium': float(item.get('premium', 0))
                }
                for item in funding_history
            ]
            
        except Exception as e:
            logger.error(f"Failed to get funding history: {e}")
            return []
    
    def get_trade_history(self, coin: str = None, start_time: int = None, 
                         end_time: int = None) -> List[Dict[str, Any]]:
        """Get trade history"""
        if not self.is_authenticated:
            raise Exception("Not authenticated. Call authenticate() first.")
        
        self._rate_limit()
        
        try:
            fills = self.info.user_fills(self.account_address)
            
            # Filter by coin and time if specified
            filtered_fills = []
            for fill in fills:
                if coin and fill.get('coin') != coin:
                    continue
                    
                fill_time = fill.get('time', 0)
                if start_time and fill_time < start_time:
                    continue
                if end_time and fill_time > end_time:
                    continue
                
                filtered_fills.append({
                    'timestamp': fill_time,
                    'coin': fill.get('coin'),
                    'side': fill.get('side'),
                    'size': float(fill.get('sz', 0)),
                    'price': float(fill.get('px', 0)),
                    'fee': float(fill.get('fee', 0)),
                    'oid': fill.get('oid'),
                    'closed_pnl': float(fill.get('closedPnl', 0))
                })
            
            return filtered_fills
            
        except Exception as e:
            logger.error(f"Failed to get trade history: {e}")
            return []
    
    def start_websocket(self, subscriptions: List[str] = None):
        """Start WebSocket connection for real-time data"""
        if not self.ws_manager:
            logger.error("WebSocket manager not initialized")
            return
        
        try:
            if not subscriptions:
                subscriptions = ['allMids', 'notification']
            
            self.ws_manager.start(subscriptions)
            logger.info("WebSocket connection started")
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket: {e}")
    
    def stop_websocket(self):
        """Stop WebSocket connection"""
        if self.ws_manager:
            try:
                self.ws_manager.close()
                logger.info("WebSocket connection stopped")
            except Exception as e:
                logger.error(f"Failed to stop WebSocket: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_websocket()


    # Async methods for GUI integration
    async def authenticate_async(self, private_key: str = None, wallet_address: str = None) -> bool:
        """Async version of authenticate method"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.authenticate, private_key, wallet_address
        )
    
    async def get_available_tokens_async(self) -> List[str]:
        """Get list of available tokens asynchronously"""
        try:
            def get_tokens():
                meta_info = self.info.meta()
                if meta_info and 'universe' in meta_info:
                    return [asset['name'] for asset in meta_info['universe']]
                return ["BTC", "ETH", "SOL", "AVAX", "MATIC"]  # Fallback
            
            return await asyncio.get_event_loop().run_in_executor(None, get_tokens)
        except Exception as e:
            logger.error(f"Failed to get available tokens: {e}")
            return ["BTC", "ETH", "SOL", "AVAX", "MATIC"]  # Fallback
    
    async def get_account_info_async(self) -> Optional[Dict[str, Any]]:
        """Get account information asynchronously"""
        try:
            if not self.is_authenticated:
                return None
            
            def get_account():
                user_state = self.info.user_state(self.account_address)
                if user_state:
                    return {
                        'accountValue': float(user_state.get('marginSummary', {}).get('accountValue', 0)),
                        'totalPnl': float(user_state.get('marginSummary', {}).get('totalPnl', 0)),
                        'marginUsed': float(user_state.get('marginSummary', {}).get('marginUsed', 0)),
                        'withdrawable': float(user_state.get('withdrawable', 0))
                    }
                return None
            
            return await asyncio.get_event_loop().run_in_executor(None, get_account)
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return None
    
    async def get_positions_async(self) -> List[Dict[str, Any]]:
        """Get positions asynchronously"""
        try:
            if not self.is_authenticated:
                return []
            
            def get_positions():
                user_state = self.info.user_state(self.account_address)
                if user_state and 'assetPositions' in user_state:
                    positions = []
                    for pos in user_state['assetPositions']:
                        if float(pos['position']['szi']) != 0:  # Only open positions
                            positions.append({
                                'coin': pos['position']['coin'],
                                'size': float(pos['position']['szi']),
                                'entryPx': float(pos['position']['entryPx']) if pos['position']['entryPx'] else 0,
                                'pnl': float(pos['position']['unrealizedPnl']),
                                'marginUsed': float(pos['position']['marginUsed'])
                            })
                    return positions
                return []
            
            return await asyncio.get_event_loop().run_in_executor(None, get_positions)
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    async def place_order_async(self, symbol: str, side: str, size: float, 
                               order_type: str = "market", price: float = None) -> Dict[str, Any]:
        """Place order asynchronously"""
        try:
            if not self.is_authenticated or not self.exchange:
                return {"success": False, "error": "Not authenticated"}
            
            def place_order():
                try:
                    # Prepare order data
                    order_data = {
                        "coin": symbol,
                        "is_buy": side.lower() == "buy",
                        "sz": size,
                        "limit_px": price if order_type == "limit" and price else None,
                        "order_type": {"limit": "Limit", "market": "Market"}.get(order_type.lower(), "Market"),
                        "reduce_only": False
                    }
                    
                    # Place the order
                    result = self.exchange.order(order_data)
                    
                    if result and result.get('status') == 'ok':
                        return {"success": True, "result": result}
                    else:
                        error_msg = result.get('response', {}).get('data', 'Unknown error') if result else 'No response'
                        return {"success": False, "error": error_msg}
                        
                except Exception as e:
                    return {"success": False, "error": str(e)}
            
            return await asyncio.get_event_loop().run_in_executor(None, place_order)
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return {"success": False, "error": str(e)}
    
    async def cancel_all_orders_async(self) -> Dict[str, Any]:
        """Cancel all orders asynchronously"""
        try:
            if not self.is_authenticated or not self.exchange:
                return {"success": False, "error": "Not authenticated"}
            
            def cancel_orders():
                try:
                    result = self.exchange.cancel_all_orders()
                    return {"success": True, "result": result}
                except Exception as e:
                    return {"success": False, "error": str(e)}
            
            return await asyncio.get_event_loop().run_in_executor(None, cancel_orders)
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_open_orders_async(self) -> List[Dict[str, Any]]:
        """Get open orders asynchronously"""
        try:
            if not self.is_authenticated:
                return []
            
            def get_orders():
                user_state = self.info.user_state(self.account_address)
                if user_state and 'openOrders' in user_state:
                    return user_state['openOrders']
                return []
            
            return await asyncio.get_event_loop().run_in_executor(None, get_orders)
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []
    
    async def get_orderbook_async(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get orderbook for symbol asynchronously"""
        try:
            def get_orderbook():
                return self.info.l2_book(symbol)
            
            return await asyncio.get_event_loop().run_in_executor(None, get_orderbook)
        except Exception as e:
            logger.error(f"Failed to get orderbook for {symbol}: {e}")
            return None
    
    async def get_recent_trades_async(self, symbol: str) -> List[Dict[str, Any]]:
        """Get recent trades for symbol asynchronously"""
        try:
            def get_trades():
                return self.info.recent_trades(symbol)
            
            return await asyncio.get_event_loop().run_in_executor(None, get_trades)
        except Exception as e:
            logger.error(f"Failed to get recent trades for {symbol}: {e}")
            return []
    
    async def close_position_async(self, symbol: str) -> Dict[str, Any]:
        """Close position asynchronously"""
        try:
            if not self.is_authenticated or not self.exchange:
                return {"success": False, "error": "Not authenticated"}
            
            def close_position():
                try:
                    # Get current position
                    positions = self.get_positions()
                    position = next((p for p in positions if p['coin'] == symbol), None)
                    
                    if not position:
                        return {"success": False, "error": "No position found"}
                    
                    # Close position by placing opposite order
                    size = abs(float(position['size']))
                    side = "sell" if float(position['size']) > 0 else "buy"
                    
                    order_data = {
                        "coin": symbol,
                        "is_buy": side == "buy",
                        "sz": size,
                        "limit_px": None,
                        "order_type": "Market",
                        "reduce_only": True
                    }
                    
                    result = self.exchange.order(order_data)
                    
                    if result and result.get('status') == 'ok':
                        return {"success": True, "result": result}
                    else:
                        error_msg = result.get('response', {}).get('data', 'Unknown error') if result else 'No response'
                        return {"success": False, "error": error_msg}
                        
                except Exception as e:
                    return {"success": False, "error": str(e)}
            
            return await asyncio.get_event_loop().run_in_executor(None, close_position)
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return {"success": False, "error": str(e)}
    
    def get_available_tokens(self) -> List[str]:
        """Get list of available tokens (sync version)"""
        try:
            meta_info = self.info.meta()
            if meta_info and 'universe' in meta_info:
                return [asset['name'] for asset in meta_info['universe']]
            return ["BTC", "ETH", "SOL", "AVAX", "MATIC"]  # Fallback
        except Exception as e:
            logger.error(f"Failed to get available tokens: {e}")
            return ["BTC", "ETH", "SOL", "AVAX", "MATIC"]  # Fallback
    
    def stop_websocket(self):
        """Stop WebSocket connections"""
        try:
            if self.ws_manager:
                self.ws_manager.close()
                self.ws_manager = None
            logger.info("WebSocket connections stopped")
        except Exception as e:
            logger.error(f"Failed to stop WebSocket: {e}")
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions (sync version)"""
        try:
            if not self.is_authenticated:
                return []
            
            user_state = self.info.user_state(self.account_address)
            if user_state and 'assetPositions' in user_state:
                positions = []
                for pos in user_state['assetPositions']:
                    if float(pos['position']['szi']) != 0:  # Only open positions
                        positions.append({
                            'coin': pos['position']['coin'],
                            'size': float(pos['position']['szi']),
                            'entryPx': float(pos['position']['entryPx']) if pos['position']['entryPx'] else 0,
                            'pnl': float(pos['position']['unrealizedPnl']),
                            'marginUsed': float(pos['position']['marginUsed'])
                        })
                return positions
            return []
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

