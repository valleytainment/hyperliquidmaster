"""
API Rate Limiter Integration Module

This module provides integration functions for using the API rate limiter
with the endpoint handlers to ensure proper rate limiting for all API calls.
"""

import logging
from typing import Dict, Any, Optional, Callable
import api_rate_limiter
import api_endpoint_handlers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class APIIntegration:
    """
    API integration class for using the rate limiter with endpoint handlers.
    """
    
    def __init__(self, rate_limiter = None):
        """
        Initialize the API integration.
        
        Args:
            rate_limiter: API rate limiter instance (creates new one if None)
        """
        self.rate_limiter = rate_limiter or api_rate_limiter.APIRateLimiter()
    
    def execute_api_call(self, endpoint: str, params: Dict[str, Any] = None, 
                        use_cache: bool = True, use_stale: bool = True,
                        use_batch: bool = False) -> Any:
        """
        Execute an API call with rate limiting.
        
        Args:
            endpoint: API endpoint name
            params: Request parameters
            use_cache: Whether to use caching
            use_stale: Whether to use stale cache as fallback
            use_batch: Whether to batch requests
            
        Returns:
            API response
        """
        if params is None:
            params = {}
        
        # Get the handler function for this endpoint
        handler = api_endpoint_handlers.get_handler_for_endpoint(endpoint)
        
        if handler is None:
            logger.error(f"No handler found for endpoint: {endpoint}")
            return {"error": f"No handler found for endpoint: {endpoint}"}
        
        # Create a function that calls the handler with the params
        def execute_handler():
            return handler(params)
        
        # Execute with rate limiting
        try:
            result = self.rate_limiter.execute_with_rate_limit(
                endpoint=endpoint,
                params=params,
                func=execute_handler,
                use_cache=use_cache,
                use_stale=use_stale,
                use_batch=use_batch
            )
            return result
        except Exception as e:
            logger.error(f"Error executing API call to {endpoint}: {str(e)}")
            return {"error": str(e)}
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get market data for a symbol.
        
        Args:
            symbol: Symbol to get market data for
            
        Returns:
            Market data
        """
        return self.execute_api_call("market_data", {"symbol": symbol})
    
    def get_order_book(self, symbol: str, depth: int = 10) -> Dict[str, Any]:
        """
        Get order book for a symbol.
        
        Args:
            symbol: Symbol to get order book for
            depth: Order book depth
            
        Returns:
            Order book
        """
        return self.execute_api_call("order_book", {"symbol": symbol, "depth": depth})
    
    def get_historical_data(self, symbol: str, timeframe: str = "1m", limit: int = 100) -> Dict[str, Any]:
        """
        Get historical data for a symbol.
        
        Args:
            symbol: Symbol to get historical data for
            timeframe: Timeframe
            limit: Maximum number of data points
            
        Returns:
            Historical data
        """
        return self.execute_api_call("historical_data", {
            "symbol": symbol,
            "timeframe": timeframe,
            "limit": limit
        })
    
    def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Get funding rate for a symbol.
        
        Args:
            symbol: Symbol to get funding rate for
            
        Returns:
            Funding rate
        """
        return self.execute_api_call("funding_rate", {"symbol": symbol})
