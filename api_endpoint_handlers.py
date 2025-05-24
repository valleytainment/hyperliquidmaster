"""
API Endpoint Handlers for Hyperliquid Exchange with Rate Limit Awareness

This module provides handler functions for Hyperliquid API endpoints
using the official Hyperliquid Python SDK with rate limit awareness.
"""

import logging
import time
import random
import json
from typing import Dict, Any, Optional, Callable, Union

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize Hyperliquid SDK clients with delayed initialization
info_client = None

def get_info_client():
    """
    Get or initialize the Info client with rate limit awareness.
    
    Returns:
        Info client
    """
    global info_client
    
    if info_client is None:
        # Add jitter to prevent multiple instances from hitting API simultaneously
        jitter = random.uniform(0.1, 1.0)
        time.sleep(jitter)
        
        try:
            info_client = Info(base_url=constants.MAINNET_API_URL)
        except Exception as e:
            logger.error(f"Error initializing Info client: {str(e)}")
            # Return a minimal client that will be retried later
            class MinimalClient:
                def __getattr__(self, name):
                    def method(*args, **kwargs):
                        raise Exception("Info client not initialized")
                    return method
            return MinimalClient()
    
    return info_client

def get_handler_for_endpoint(endpoint: str) -> Optional[Callable]:
    """
    Get the handler function for a specific endpoint.
    
    Args:
        endpoint: API endpoint name
        
    Returns:
        Handler function for the endpoint
    """
    handlers = {
        "market_data": handle_market_data,
        "order_book": handle_order_book,
        "historical_data": handle_historical_data,
        "funding_rate": handle_funding_rate,
        "all_markets": handle_all_markets,
        "user_state": handle_user_state,
        "user_fills": handle_user_fills
    }
    
    return handlers.get(endpoint)

def safe_get(data: Any, key: str, default: Any = None) -> Any:
    """
    Safely get a value from a dictionary or object.
    
    Args:
        data: Dictionary, object, or any other data
        key: Key to get
        default: Default value if key is not found or data is not a dictionary
        
    Returns:
        Value for the key or default
    """
    if data is None:
        return default
    
    if isinstance(data, dict):
        return data.get(key, default)
    
    if isinstance(data, str):
        try:
            # Try to parse as JSON
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                return parsed.get(key, default)
        except:
            pass
    
    # For any other type, return default
    return default

def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Float value or default
    """
    if value is None:
        return default
    
    try:
        return float(value)
    except:
        return default

def extract_market_data_from_string(data_str: str, symbol: str) -> Dict[str, Any]:
    """
    Attempt to extract market data from a string response.
    
    Args:
        data_str: String response from API
        symbol: Symbol to extract data for
        
    Returns:
        Market data dictionary or error
    """
    try:
        # Try to parse as JSON
        parsed = json.loads(data_str)
        
        # If parsed is a dictionary, try to extract market data
        if isinstance(parsed, dict):
            # Check if it contains market data fields
            if "midPrice" in parsed or "markPrice" in parsed:
                return {
                    "symbol": symbol,
                    "price": safe_float(safe_get(parsed, "midPrice")),
                    "index_price": safe_float(safe_get(parsed, "indexPrice")),
                    "mark_price": safe_float(safe_get(parsed, "markPrice")),
                    "open_interest": safe_float(safe_get(parsed, "openInterest")),
                    "funding_rate": safe_float(safe_get(parsed, "fundingRate")),
                    "volume_24h": safe_float(safe_get(parsed, "volume24h")),
                    "timestamp": int(time.time() * 1000)
                }
            
            # Check if it's an error response
            if "error" in parsed or "message" in parsed:
                error_msg = safe_get(parsed, "error", safe_get(parsed, "message", "Unknown error"))
                return {"error": f"API error: {error_msg}"}
        
        # If parsed is a list, try to find the symbol
        if isinstance(parsed, list):
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                
                if safe_get(item, "name") == symbol or safe_get(item, "symbol") == symbol:
                    return {
                        "symbol": symbol,
                        "price": safe_float(safe_get(item, "midPrice")),
                        "index_price": safe_float(safe_get(item, "indexPrice")),
                        "mark_price": safe_float(safe_get(item, "markPrice")),
                        "open_interest": safe_float(safe_get(item, "openInterest")),
                        "funding_rate": safe_float(safe_get(item, "fundingRate")),
                        "volume_24h": safe_float(safe_get(item, "volume24h")),
                        "timestamp": int(time.time() * 1000)
                    }
    except:
        pass
    
    # If we couldn't extract market data, return a fallback with default values
    logger.warning(f"Could not parse market data string for {symbol}: {data_str[:100]}...")
    
    # Check if it looks like an error message
    if "error" in data_str.lower() or "not found" in data_str.lower() or "invalid" in data_str.lower():
        return {"error": f"API error: {data_str[:100]}..."}
    
    # Return fallback data with warning
    return {
        "symbol": symbol,
        "price": 0.0,
        "index_price": 0.0,
        "mark_price": 0.0,
        "open_interest": 0.0,
        "funding_rate": 0.0,
        "volume_24h": 0.0,
        "timestamp": int(time.time() * 1000),
        "warning": "Data unavailable, using fallback values"
    }

def handle_market_data(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle market data request.
    
    Args:
        params: Request parameters
        
    Returns:
        Market data
    """
    try:
        symbol = params.get("symbol")
        if not symbol:
            return {"error": "Symbol is required"}
        
        # Get all mids (market data)
        client = get_info_client()
        
        try:
            all_mids_response = client.all_mids()
        except Exception as e:
            logger.error(f"Error calling all_mids: {str(e)}")
            return {"error": f"Failed to retrieve market data: {str(e)}"}
        
        # Handle symbol-keyed dictionary response (new format)
        if isinstance(all_mids_response, dict):
            # Check if the symbol exists directly as a key
            if symbol in all_mids_response:
                market_data = all_mids_response[symbol]
                
                # Handle different types of market data
                if isinstance(market_data, dict):
                    # Extract relevant market data from dictionary
                    return {
                        "symbol": symbol,
                        "price": safe_float(safe_get(market_data, "midPrice")),
                        "index_price": safe_float(safe_get(market_data, "indexPrice")),
                        "mark_price": safe_float(safe_get(market_data, "markPrice")),
                        "open_interest": safe_float(safe_get(market_data, "openInterest")),
                        "funding_rate": safe_float(safe_get(market_data, "fundingRate")),
                        "volume_24h": safe_float(safe_get(market_data, "volume24h")),
                        "timestamp": int(time.time() * 1000)
                    }
                elif isinstance(market_data, str):
                    # Try to extract market data from string response
                    return extract_market_data_from_string(market_data, symbol)
                else:
                    logger.error(f"Unexpected market data type for {symbol}: {type(market_data)}")
                    return {
                        "symbol": symbol,
                        "price": 0.0,
                        "index_price": 0.0,
                        "mark_price": 0.0,
                        "open_interest": 0.0,
                        "funding_rate": 0.0,
                        "volume_24h": 0.0,
                        "timestamp": int(time.time() * 1000),
                        "warning": f"Unexpected data type: {type(market_data)}, using fallback values"
                    }
            else:
                # If symbol not found directly, check if it's in the response with a different case
                for key in all_mids_response.keys():
                    if isinstance(key, str) and key.upper() == symbol.upper():
                        market_data = all_mids_response[key]
                        
                        # Handle different types of market data
                        if isinstance(market_data, dict):
                            # Extract relevant market data from dictionary
                            return {
                                "symbol": key,  # Use the actual key from the response
                                "price": safe_float(safe_get(market_data, "midPrice")),
                                "index_price": safe_float(safe_get(market_data, "indexPrice")),
                                "mark_price": safe_float(safe_get(market_data, "markPrice")),
                                "open_interest": safe_float(safe_get(market_data, "openInterest")),
                                "funding_rate": safe_float(safe_get(market_data, "fundingRate")),
                                "volume_24h": safe_float(safe_get(market_data, "volume24h")),
                                "timestamp": int(time.time() * 1000)
                            }
                        elif isinstance(market_data, str):
                            # Try to extract market data from string response
                            return extract_market_data_from_string(market_data, key)
                        else:
                            logger.error(f"Unexpected market data type for {key}: {type(market_data)}")
                            return {
                                "symbol": key,
                                "price": 0.0,
                                "index_price": 0.0,
                                "mark_price": 0.0,
                                "open_interest": 0.0,
                                "funding_rate": 0.0,
                                "volume_24h": 0.0,
                                "timestamp": int(time.time() * 1000),
                                "warning": f"Unexpected data type: {type(market_data)}, using fallback values"
                            }
                
                # Try to find the symbol in the values if it's not a key
                for key, value in all_mids_response.items():
                    if isinstance(value, dict) and (safe_get(value, "name") == symbol or safe_get(value, "symbol") == symbol):
                        return {
                            "symbol": symbol,
                            "price": safe_float(safe_get(value, "midPrice")),
                            "index_price": safe_float(safe_get(value, "indexPrice")),
                            "mark_price": safe_float(safe_get(value, "markPrice")),
                            "open_interest": safe_float(safe_get(value, "openInterest")),
                            "funding_rate": safe_float(safe_get(value, "fundingRate")),
                            "volume_24h": safe_float(safe_get(value, "volume24h")),
                            "timestamp": int(time.time() * 1000)
                        }
                
                # If we get here, the symbol was not found
                logger.error(f"Symbol {symbol} not found in response keys: {list(all_mids_response.keys())[:10]}...")
                
                # Try to use a default symbol as fallback (BTC or ETH)
                for fallback_symbol in ["BTC", "ETH", "SOL"]:
                    if fallback_symbol in all_mids_response:
                        market_data = all_mids_response[fallback_symbol]
                        if isinstance(market_data, dict):
                            logger.warning(f"Using {fallback_symbol} as fallback for {symbol}")
                            return {
                                "symbol": symbol,
                                "price": safe_float(safe_get(market_data, "midPrice")),
                                "index_price": safe_float(safe_get(market_data, "indexPrice")),
                                "mark_price": safe_float(safe_get(market_data, "markPrice")),
                                "open_interest": safe_float(safe_get(market_data, "openInterest")),
                                "funding_rate": safe_float(safe_get(market_data, "fundingRate")),
                                "volume_24h": safe_float(safe_get(market_data, "volume24h")),
                                "timestamp": int(time.time() * 1000),
                                "warning": f"Symbol {symbol} not found, using {fallback_symbol} as fallback"
                            }
                
                return {"error": f"Symbol {symbol} not found"}
        
        # Handle list response (old format)
        elif isinstance(all_mids_response, list):
            # Find the requested symbol
            for market in all_mids_response:
                if not isinstance(market, dict):
                    continue
                    
                if safe_get(market, "name") == symbol:
                    # Extract relevant market data
                    return {
                        "symbol": symbol,
                        "price": safe_float(safe_get(market, "midPrice")),
                        "index_price": safe_float(safe_get(market, "indexPrice")),
                        "mark_price": safe_float(safe_get(market, "markPrice")),
                        "open_interest": safe_float(safe_get(market, "openInterest")),
                        "funding_rate": safe_float(safe_get(market, "fundingRate")),
                        "volume_24h": safe_float(safe_get(market, "volume24h")),
                        "timestamp": int(time.time() * 1000)
                    }
            
            return {"error": f"Symbol {symbol} not found"}
        
        # If response is a string
        elif isinstance(all_mids_response, str):
            # Try to extract market data from string response
            return extract_market_data_from_string(all_mids_response, symbol)
        
        # If response is any other type
        else:
            logger.error(f"Unexpected response type from all_mids: {type(all_mids_response)}")
            return {
                "symbol": symbol,
                "price": 0.0,
                "index_price": 0.0,
                "mark_price": 0.0,
                "open_interest": 0.0,
                "funding_rate": 0.0,
                "volume_24h": 0.0,
                "timestamp": int(time.time() * 1000),
                "warning": f"Unexpected response type: {type(all_mids_response)}, using fallback values"
            }
    except Exception as e:
        logger.error(f"Error fetching market data for {params.get('symbol')}: {str(e)}")
        return {
            "symbol": params.get('symbol', 'unknown'),
            "price": 0.0,
            "index_price": 0.0,
            "mark_price": 0.0,
            "open_interest": 0.0,
            "funding_rate": 0.0,
            "volume_24h": 0.0,
            "timestamp": int(time.time() * 1000),
            "warning": f"Error: {str(e)}, using fallback values"
        }

def handle_order_book(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle order book request.
    
    Args:
        params: Request parameters
        
    Returns:
        Order book
    """
    try:
        symbol = params.get("symbol")
        depth = params.get("depth", 10)
        
        if not symbol:
            return {"error": "Symbol is required"}
        
        # Get L2 snapshot (order book)
        client = get_info_client()
        
        try:
            # Updated parameter name from 'coin' to 'asset' based on SDK documentation
            l2_snapshot = client.l2_snapshot(asset=symbol)
        except Exception as e:
            logger.error(f"Error calling l2_snapshot: {str(e)}")
            return {"error": f"Failed to retrieve order book: {str(e)}"}
        
        # Check if response is a string
        if isinstance(l2_snapshot, str):
            logger.error(f"Unexpected string response from l2_snapshot: {l2_snapshot}")
            return {"error": f"Unexpected response format: {l2_snapshot}"}
        
        # Check if response is a dictionary
        if not isinstance(l2_snapshot, dict):
            logger.error(f"Unexpected response type from l2_snapshot: {type(l2_snapshot)}")
            return {"error": f"Unexpected response type: {type(l2_snapshot)}"}
        
        # Extract bids and asks
        bids = []
        asks = []
        
        levels = safe_get(l2_snapshot, "levels", [])
        
        if not isinstance(levels, list):
            logger.error(f"Unexpected levels type: {type(levels)}")
            return {"error": f"Unexpected levels type: {type(levels)}"}
        
        for level in levels:
            if not isinstance(level, dict):
                continue
                
            side = safe_get(level, "side")
            price = safe_float(safe_get(level, "px"))
            quantity = safe_float(safe_get(level, "sz"))
            
            if side == "A":
                asks.append({
                    "price": price,
                    "quantity": quantity
                })
            else:
                bids.append({
                    "price": price,
                    "quantity": quantity
                })
        
        # Sort and limit
        bids = sorted(bids, key=lambda x: x["price"], reverse=True)[:depth]
        asks = sorted(asks, key=lambda x: x["price"])[:depth]
        
        return {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "timestamp": int(time.time() * 1000)
        }
    except Exception as e:
        logger.error(f"Error fetching order book for {params.get('symbol')}: {str(e)}")
        return {"error": str(e)}

def handle_historical_data(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle historical data request.
    
    Args:
        params: Request parameters
        
    Returns:
        Historical data
    """
    try:
        symbol = params.get("symbol")
        timeframe = params.get("timeframe", "1m")
        limit = params.get("limit", 100)
        
        if not symbol:
            return {"error": "Symbol is required"}
        
        # Map timeframe to resolution
        resolution_map = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400
        }
        
        resolution = resolution_map.get(timeframe, 60)
        
        # Get candles snapshot
        client = get_info_client()
        
        try:
            # Updated parameter name from 'coin' to 'asset' based on SDK documentation
            candles = client.candles_snapshot(
                asset=symbol,
                resolution=resolution,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Error calling candles_snapshot: {str(e)}")
            return {"error": f"Failed to retrieve historical data: {str(e)}"}
        
        # Check if response is a string
        if isinstance(candles, str):
            logger.error(f"Unexpected string response from candles_snapshot: {candles}")
            return {"error": f"Unexpected response format: {candles}"}
        
        # Check if response is a list
        if not isinstance(candles, list):
            logger.error(f"Unexpected response type from candles_snapshot: {type(candles)}")
            return {"error": f"Unexpected response type: {type(candles)}"}
        
        # Format candles
        formatted_candles = []
        
        for candle in candles:
            if not isinstance(candle, dict):
                continue
                
            formatted_candles.append({
                "timestamp": safe_get(candle, "time", 0),
                "open": safe_float(safe_get(candle, "open")),
                "high": safe_float(safe_get(candle, "high")),
                "low": safe_float(safe_get(candle, "low")),
                "close": safe_float(safe_get(candle, "close")),
                "volume": safe_float(safe_get(candle, "volume"))
            })
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": formatted_candles
        }
    except Exception as e:
        logger.error(f"Error fetching historical data for {params.get('symbol')} ({params.get('timeframe')}): {str(e)}")
        return {"error": str(e)}

def handle_funding_rate(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle funding rate request.
    
    Args:
        params: Request parameters
        
    Returns:
        Funding rate
    """
    try:
        symbol = params.get("symbol")
        
        if not symbol:
            return {"error": "Symbol is required"}
        
        # Get all mids (market data)
        client = get_info_client()
        
        try:
            all_mids_response = client.all_mids()
        except Exception as e:
            logger.error(f"Error calling all_mids: {str(e)}")
            return {"error": f"Failed to retrieve funding rate: {str(e)}"}
        
        # Handle symbol-keyed dictionary response (new format)
        if isinstance(all_mids_response, dict):
            # Check if the symbol exists directly as a key
            if symbol in all_mids_response:
                market_data = all_mids_response[symbol]
                
                # Handle different types of market data
                if isinstance(market_data, dict):
                    # Extract funding rate from dictionary
                    return {
                        "symbol": symbol,
                        "funding_rate": safe_float(safe_get(market_data, "fundingRate")),
                        "next_funding_time": safe_get(market_data, "nextFundingTime", 0),
                        "timestamp": int(time.time() * 1000)
                    }
                elif isinstance(market_data, str):
                    # Try to extract funding rate from string response
                    try:
                        parsed = json.loads(market_data)
                        if isinstance(parsed, dict):
                            return {
                                "symbol": symbol,
                                "funding_rate": safe_float(safe_get(parsed, "fundingRate")),
                                "next_funding_time": safe_get(parsed, "nextFundingTime", 0),
                                "timestamp": int(time.time() * 1000)
                            }
                    except:
                        pass
                    
                    # Return fallback with warning
                    return {
                        "symbol": symbol,
                        "funding_rate": 0.0,
                        "next_funding_time": 0,
                        "timestamp": int(time.time() * 1000),
                        "warning": "Could not parse funding rate data, using fallback values"
                    }
                else:
                    logger.error(f"Unexpected market data type for {symbol}: {type(market_data)}")
                    return {
                        "symbol": symbol,
                        "funding_rate": 0.0,
                        "next_funding_time": 0,
                        "timestamp": int(time.time() * 1000),
                        "warning": f"Unexpected data type: {type(market_data)}, using fallback values"
                    }
            else:
                # If symbol not found directly, check if it's in the response with a different case
                for key in all_mids_response.keys():
                    if isinstance(key, str) and key.upper() == symbol.upper():
                        market_data = all_mids_response[key]
                        
                        # Handle different types of market data
                        if isinstance(market_data, dict):
                            # Extract funding rate from dictionary
                            return {
                                "symbol": key,  # Use the actual key from the response
                                "funding_rate": safe_float(safe_get(market_data, "fundingRate")),
                                "next_funding_time": safe_get(market_data, "nextFundingTime", 0),
                                "timestamp": int(time.time() * 1000)
                            }
                        elif isinstance(market_data, str):
                            # Try to extract funding rate from string response
                            try:
                                parsed = json.loads(market_data)
                                if isinstance(parsed, dict):
                                    return {
                                        "symbol": key,
                                        "funding_rate": safe_float(safe_get(parsed, "fundingRate")),
                                        "next_funding_time": safe_get(parsed, "nextFundingTime", 0),
                                        "timestamp": int(time.time() * 1000)
                                    }
                            except:
                                pass
                            
                            # Return fallback with warning
                            return {
                                "symbol": key,
                                "funding_rate": 0.0,
                                "next_funding_time": 0,
                                "timestamp": int(time.time() * 1000),
                                "warning": "Could not parse funding rate data, using fallback values"
                            }
                        else:
                            logger.error(f"Unexpected market data type for {key}: {type(market_data)}")
                            return {
                                "symbol": key,
                                "funding_rate": 0.0,
                                "next_funding_time": 0,
                                "timestamp": int(time.time() * 1000),
                                "warning": f"Unexpected data type: {type(market_data)}, using fallback values"
                            }
                
                # If we get here, the symbol was not found
                logger.error(f"Symbol {symbol} not found in response keys: {list(all_mids_response.keys())[:10]}...")
                return {"error": f"Symbol {symbol} not found"}
        
        # Handle list response (old format)
        elif isinstance(all_mids_response, list):
            # Find the requested symbol
            for market in all_mids_response:
                if not isinstance(market, dict):
                    continue
                    
                if safe_get(market, "name") == symbol:
                    # Extract funding rate
                    return {
                        "symbol": symbol,
                        "funding_rate": safe_float(safe_get(market, "fundingRate")),
                        "next_funding_time": safe_get(market, "nextFundingTime", 0),
                        "timestamp": int(time.time() * 1000)
                    }
            
            return {"error": f"Symbol {symbol} not found"}
        
        # If response is a string
        elif isinstance(all_mids_response, str):
            logger.error(f"Unexpected string response from all_mids: {all_mids_response}")
            return {"error": f"Unexpected response format: {all_mids_response}"}
        
        # If response is any other type
        else:
            logger.error(f"Unexpected response type from all_mids: {type(all_mids_response)}")
            return {"error": f"Unexpected response type: {type(all_mids_response)}"}
    except Exception as e:
        logger.error(f"Error fetching funding rate for {params.get('symbol')}: {str(e)}")
        return {"error": str(e)}

def handle_all_markets(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle all markets request.
    
    Args:
        params: Request parameters
        
    Returns:
        All markets
    """
    try:
        # Get all mids (market data)
        client = get_info_client()
        
        try:
            all_mids_response = client.all_mids()
        except Exception as e:
            logger.error(f"Error calling all_mids: {str(e)}")
            return {"error": f"Failed to retrieve all markets: {str(e)}"}
        
        # Format markets
        markets = []
        
        # Handle symbol-keyed dictionary response (new format)
        if isinstance(all_mids_response, dict):
            for symbol, market_data in all_mids_response.items():
                # Skip non-string keys or keys starting with '@' (likely internal)
                if not isinstance(symbol, str) or symbol.startswith('@'):
                    continue
                
                # Handle different types of market data
                if isinstance(market_data, dict):
                    markets.append({
                        "symbol": symbol,
                        "price": safe_float(safe_get(market_data, "midPrice")),
                        "index_price": safe_float(safe_get(market_data, "indexPrice")),
                        "mark_price": safe_float(safe_get(market_data, "markPrice")),
                        "open_interest": safe_float(safe_get(market_data, "openInterest")),
                        "funding_rate": safe_float(safe_get(market_data, "fundingRate")),
                        "volume_24h": safe_float(safe_get(market_data, "volume24h"))
                    })
                elif isinstance(market_data, str):
                    # Try to extract market data from string response
                    try:
                        parsed = json.loads(market_data)
                        if isinstance(parsed, dict):
                            markets.append({
                                "symbol": symbol,
                                "price": safe_float(safe_get(parsed, "midPrice")),
                                "index_price": safe_float(safe_get(parsed, "indexPrice")),
                                "mark_price": safe_float(safe_get(parsed, "markPrice")),
                                "open_interest": safe_float(safe_get(parsed, "openInterest")),
                                "funding_rate": safe_float(safe_get(parsed, "fundingRate")),
                                "volume_24h": safe_float(safe_get(parsed, "volume24h"))
                            })
                    except:
                        # Skip this market if we can't parse the data
                        logger.warning(f"Could not parse market data for {symbol}")
        
        # Handle list response (old format)
        elif isinstance(all_mids_response, list):
            for market in all_mids_response:
                if not isinstance(market, dict):
                    continue
                
                markets.append({
                    "symbol": safe_get(market, "name", ""),
                    "price": safe_float(safe_get(market, "midPrice")),
                    "index_price": safe_float(safe_get(market, "indexPrice")),
                    "mark_price": safe_float(safe_get(market, "markPrice")),
                    "open_interest": safe_float(safe_get(market, "openInterest")),
                    "funding_rate": safe_float(safe_get(market, "fundingRate")),
                    "volume_24h": safe_float(safe_get(market, "volume24h"))
                })
        
        # If response is a string
        elif isinstance(all_mids_response, str):
            logger.error(f"Unexpected string response from all_mids: {all_mids_response}")
            return {"error": f"Unexpected response format: {all_mids_response}"}
        
        # If response is any other type
        else:
            logger.error(f"Unexpected response type from all_mids: {type(all_mids_response)}")
            return {"error": f"Unexpected response type: {type(all_mids_response)}"}
        
        return {
            "markets": markets,
            "timestamp": int(time.time() * 1000)
        }
    except Exception as e:
        logger.error(f"Error fetching all markets: {str(e)}")
        return {"error": str(e)}

def handle_user_state(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle user state request.
    
    Args:
        params: Request parameters
        
    Returns:
        User state
    """
    try:
        wallet_address = params.get("wallet_address")
        
        if not wallet_address:
            return {"error": "Wallet address is required"}
        
        # Get user state
        client = get_info_client()
        
        try:
            user_state = client.user_state(wallet_address)
        except Exception as e:
            logger.error(f"Error calling user_state: {str(e)}")
            return {"error": f"Failed to retrieve user state: {str(e)}"}
        
        # Check if response is a string
        if isinstance(user_state, str):
            logger.error(f"Unexpected string response from user_state: {user_state}")
            return {"error": f"Unexpected response format: {user_state}"}
        
        return {
            "wallet_address": wallet_address,
            "user_state": user_state,
            "timestamp": int(time.time() * 1000)
        }
    except Exception as e:
        logger.error(f"Error fetching user state for {params.get('wallet_address')}: {str(e)}")
        return {"error": str(e)}

def handle_user_fills(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle user fills request.
    
    Args:
        params: Request parameters
        
    Returns:
        User fills
    """
    try:
        wallet_address = params.get("wallet_address")
        limit = params.get("limit", 100)
        
        if not wallet_address:
            return {"error": "Wallet address is required"}
        
        # Get user fills
        client = get_info_client()
        
        try:
            user_fills = client.user_fills(wallet_address, limit=limit)
        except Exception as e:
            logger.error(f"Error calling user_fills: {str(e)}")
            return {"error": f"Failed to retrieve user fills: {str(e)}"}
        
        # Check if response is a string
        if isinstance(user_fills, str):
            logger.error(f"Unexpected string response from user_fills: {user_fills}")
            return {"error": f"Unexpected response format: {user_fills}"}
        
        return {
            "wallet_address": wallet_address,
            "fills": user_fills,
            "timestamp": int(time.time() * 1000)
        }
    except Exception as e:
        logger.error(f"Error fetching user fills for {params.get('wallet_address')}: {str(e)}")
        return {"error": str(e)}
