"""
API Rate Limiter with Cooldown and Mock Data Integration

This module provides enhanced API rate limiting with cooldown periods
and integration with mock data for development and testing when
API rate limits are encountered.
"""

import os
import json
import time
import random
import logging
import threading
from typing import Dict, Any, Optional, Callable, Union
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class APIRateLimiter:
    """
    Enhanced API rate limiter with cooldown periods and mock data integration.
    """
    
    def __init__(self, 
                 rate_limits: Dict[str, int] = None, 
                 cooldown_file: str = "rate_limit_cooldown.json",
                 use_mock_data: bool = False):
        """
        Initialize the API rate limiter.
        
        Args:
            rate_limits: Dictionary of rate limits per endpoint (requests per minute)
            cooldown_file: File to store cooldown periods
            use_mock_data: Whether to use mock data instead of real API calls
        """
        self.rate_limits = rate_limits or {
            "default": 60,  # Default: 60 requests per minute
            "market_data": 120,
            "order_book": 60,
            "historical_data": 30,
            "user_state": 20
        }
        
        self.cooldown_file = cooldown_file
        self.use_mock_data = use_mock_data
        
        # Initialize request counters and timestamps
        self.request_counts = {}
        self.last_request_time = {}
        self.cooldown_periods = self._load_cooldown_periods()
        
        # Initialize locks for thread safety
        self.locks = {endpoint: threading.Lock() for endpoint in self.rate_limits.keys()}
        self.locks["default"] = threading.Lock()
        
        # Initialize mock data provider if needed
        self.mock_data_provider = None
        if self.use_mock_data:
            try:
                from mock_data_provider import MockDataProvider
                self.mock_data_provider = MockDataProvider()
                logger.info("Mock data provider initialized")
            except ImportError:
                logger.error("Failed to import MockDataProvider, falling back to real API")
                self.use_mock_data = False
        
        logger.info(f"API rate limiter initialized with {'mock' if use_mock_data else 'real'} data mode")
    
    def _load_cooldown_periods(self) -> Dict[str, int]:
        """
        Load cooldown periods from file.
        
        Returns:
            Dictionary of cooldown periods per endpoint
        """
        try:
            if os.path.exists(self.cooldown_file):
                with open(self.cooldown_file, "r") as f:
                    cooldown_data = json.load(f)
                
                # Filter out expired cooldowns
                current_time = int(time.time())
                cooldown_periods = {
                    endpoint: expiry
                    for endpoint, expiry in cooldown_data.items()
                    if expiry > current_time
                }
                
                # Log active cooldowns
                for endpoint, expiry in cooldown_periods.items():
                    expiry_time = datetime.fromtimestamp(expiry).strftime("%Y-%m-%d %H:%M:%S")
                    logger.warning(f"Active cooldown for {endpoint} until {expiry_time}")
                
                return cooldown_periods
        except Exception as e:
            logger.error(f"Error loading cooldown periods: {str(e)}")
        
        return {}
    
    def _save_cooldown_periods(self):
        """
        Save cooldown periods to file.
        """
        try:
            with open(self.cooldown_file, "w") as f:
                json.dump(self.cooldown_periods, f)
        except Exception as e:
            logger.error(f"Error saving cooldown periods: {str(e)}")
    
    def _is_in_cooldown(self, endpoint: str) -> bool:
        """
        Check if an endpoint is in cooldown.
        
        Args:
            endpoint: API endpoint
            
        Returns:
            Whether the endpoint is in cooldown
        """
        current_time = int(time.time())
        
        # Check endpoint-specific cooldown
        if endpoint in self.cooldown_periods and self.cooldown_periods[endpoint] > current_time:
            return True
        
        # Check global cooldown
        if "global" in self.cooldown_periods and self.cooldown_periods["global"] > current_time:
            return True
        
        return False
    
    def _set_cooldown(self, endpoint: str, duration_minutes: int):
        """
        Set a cooldown period for an endpoint.
        
        Args:
            endpoint: API endpoint
            duration_minutes: Cooldown duration in minutes
        """
        current_time = int(time.time())
        expiry = current_time + duration_minutes * 60
        
        self.cooldown_periods[endpoint] = expiry
        
        # Log cooldown
        expiry_time = datetime.fromtimestamp(expiry).strftime("%Y-%m-%d %H:%M:%S")
        logger.warning(f"Setting cooldown for {endpoint} until {expiry_time} ({duration_minutes} minutes)")
        
        # Save cooldown periods
        self._save_cooldown_periods()
    
    def _get_mock_data(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get mock data for an endpoint.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            Mock data
        """
        if not self.mock_data_provider:
            return {"error": "Mock data provider not initialized"}
        
        try:
            if endpoint == "market_data":
                return self.mock_data_provider.get_market_data(params.get("symbol", "BTC"))
            elif endpoint == "order_book":
                return self.mock_data_provider.get_order_book(
                    params.get("symbol", "BTC"),
                    params.get("depth", 10)
                )
            elif endpoint == "historical_data":
                return self.mock_data_provider.get_historical_data(
                    params.get("symbol", "BTC"),
                    params.get("timeframe", "1m"),
                    params.get("limit", 100)
                )
            elif endpoint == "funding_rate":
                return self.mock_data_provider.get_funding_rate(params.get("symbol", "BTC"))
            elif endpoint == "all_markets":
                return self.mock_data_provider.get_all_markets()
            else:
                return {"error": f"Mock data not available for endpoint: {endpoint}"}
        except Exception as e:
            logger.error(f"Error getting mock data for {endpoint}: {str(e)}")
            return {"error": f"Error getting mock data: {str(e)}"}
    
    def execute_with_rate_limit(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute an API call with rate limiting.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            API response
        """
        params = params or {}
        
        # Check if using mock data
        if self.use_mock_data:
            return self._get_mock_data(endpoint, params)
        
        # Check if in cooldown
        if self._is_in_cooldown(endpoint):
            logger.warning(f"Endpoint {endpoint} is in cooldown, using mock data")
            return self._get_mock_data(endpoint, params)
        
        # Get rate limit for endpoint
        rate_limit = self.rate_limits.get(endpoint, self.rate_limits["default"])
        
        # Get lock for endpoint
        lock = self.locks.get(endpoint, self.locks["default"])
        
        with lock:
            current_time = time.time()
            minute_ago = current_time - 60
            
            # Initialize request count if needed
            if endpoint not in self.request_counts:
                self.request_counts[endpoint] = []
            
            # Remove old requests
            self.request_counts[endpoint] = [
                timestamp for timestamp in self.request_counts[endpoint]
                if timestamp > minute_ago
            ]
            
            # Check if rate limit exceeded
            if len(self.request_counts[endpoint]) >= rate_limit:
                logger.warning(f"Rate limit exceeded for {endpoint}, backing off")
                
                # Calculate backoff time
                oldest_request = min(self.request_counts[endpoint])
                backoff_time = 60 - (current_time - oldest_request)
                
                # Add jitter to prevent thundering herd
                backoff_time += random.uniform(0, 1)
                
                logger.info(f"Backing off for {backoff_time:.2f}s")
                time.sleep(backoff_time)
                
                # Recursive call after backoff
                return self.execute_with_rate_limit(endpoint, params)
            
            # Add current request
            self.request_counts[endpoint].append(current_time)
            self.last_request_time[endpoint] = current_time
        
        # Get handler for endpoint
        try:
            from api_endpoint_handlers import get_handler_for_endpoint
            handler = get_handler_for_endpoint(endpoint)
            
            if not handler:
                logger.error(f"No handler found for endpoint: {endpoint}")
                return {"error": f"No handler found for endpoint: {endpoint}"}
            
            # Execute handler
            max_retries = 6
            for attempt in range(1, max_retries + 1):
                try:
                    result = handler(params)
                    
                    # Check for rate limit error in result
                    if isinstance(result, dict) and "error" in result:
                        error_msg = str(result["error"]).lower()
                        if "rate limit" in error_msg or "429" in error_msg:
                            # Set cooldown based on attempt number
                            cooldown_minutes = min(2 ** (attempt - 1) * 15, 240)  # 15min, 30min, 60min, 120min, 240min
                            self._set_cooldown(endpoint, cooldown_minutes)
                            
                            # For repeated rate limits, set global cooldown
                            if attempt >= 3:
                                self._set_cooldown("global", cooldown_minutes * 2)
                            
                            logger.warning(f"Rate limited on attempt {attempt}/{max_retries}, using mock data")
                            return self._get_mock_data(endpoint, params)
                    
                    return result
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # Check for rate limit error
                    if "rate limit" in error_msg or "429" in error_msg:
                        # Set cooldown based on attempt number
                        cooldown_minutes = min(2 ** (attempt - 1) * 15, 240)  # 15min, 30min, 60min, 120min, 240min
                        self._set_cooldown(endpoint, cooldown_minutes)
                        
                        # For repeated rate limits, set global cooldown
                        if attempt >= 3:
                            self._set_cooldown("global", cooldown_minutes * 2)
                        
                        logger.warning(f"Rate limited on attempt {attempt}/{max_retries}, using mock data")
                        return self._get_mock_data(endpoint, params)
                    
                    # For other errors, retry with backoff
                    if attempt < max_retries:
                        backoff = 2 ** (attempt - 1) * (1 + random.uniform(0, 0.5))
                        logger.warning(f"Error executing {endpoint} (attempt {attempt}/{max_retries}): {str(e)}")
                        logger.info(f"Backing off for {backoff:.2f}s before retry")
                        time.sleep(backoff)
                    else:
                        logger.error(f"Failed to execute {endpoint} after {max_retries} attempts: {str(e)}")
                        return {"error": str(e)}
        except Exception as e:
            logger.error(f"Error executing {endpoint}: {str(e)}")
            return {"error": str(e)}
    
    def execute_with_rate_limit_func(self, func: Callable) -> Any:
        """
        Execute a function with rate limiting (legacy method for backward compatibility).
        
        Args:
            func: Function to execute
            
        Returns:
            Function result
        """
        # Default to "default" endpoint for legacy calls
        endpoint = "default"
        
        # Check if in cooldown
        if self._is_in_cooldown(endpoint):
            logger.warning(f"Endpoint {endpoint} is in cooldown, using mock data")
            return {"error": "API in cooldown, please try again later"}
        
        # Get rate limit for endpoint
        rate_limit = self.rate_limits.get(endpoint, self.rate_limits["default"])
        
        # Get lock for endpoint
        lock = self.locks.get(endpoint, self.locks["default"])
        
        with lock:
            current_time = time.time()
            minute_ago = current_time - 60
            
            # Initialize request count if needed
            if endpoint not in self.request_counts:
                self.request_counts[endpoint] = []
            
            # Remove old requests
            self.request_counts[endpoint] = [
                timestamp for timestamp in self.request_counts[endpoint]
                if timestamp > minute_ago
            ]
            
            # Check if rate limit exceeded
            if len(self.request_counts[endpoint]) >= rate_limit:
                logger.warning(f"Rate limit exceeded for {endpoint}, backing off")
                
                # Calculate backoff time
                oldest_request = min(self.request_counts[endpoint])
                backoff_time = 60 - (current_time - oldest_request)
                
                # Add jitter to prevent thundering herd
                backoff_time += random.uniform(0, 1)
                
                logger.info(f"Backing off for {backoff_time:.2f}s")
                time.sleep(backoff_time)
                
                # Recursive call after backoff
                return self.execute_with_rate_limit_func(func)
            
            # Add current request
            self.request_counts[endpoint].append(current_time)
            self.last_request_time[endpoint] = current_time
        
        # Execute function
        max_retries = 6
        for attempt in range(1, max_retries + 1):
            try:
                result = func()
                
                # Check for rate limit error in result
                if isinstance(result, dict) and "error" in result:
                    error_msg = str(result["error"]).lower()
                    if "rate limit" in error_msg or "429" in error_msg:
                        # Set cooldown based on attempt number
                        cooldown_minutes = min(2 ** (attempt - 1) * 15, 240)  # 15min, 30min, 60min, 120min, 240min
                        self._set_cooldown(endpoint, cooldown_minutes)
                        
                        # For repeated rate limits, set global cooldown
                        if attempt >= 3:
                            self._set_cooldown("global", cooldown_minutes * 2)
                        
                        logger.warning(f"Rate limited on attempt {attempt}/{max_retries}, returning error")
                        return {"error": "API rate limited, please try again later"}
                
                return result
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for rate limit error
                if "rate limit" in error_msg or "429" in error_msg:
                    # Set cooldown based on attempt number
                    cooldown_minutes = min(2 ** (attempt - 1) * 15, 240)  # 15min, 30min, 60min, 120min, 240min
                    self._set_cooldown(endpoint, cooldown_minutes)
                    
                    # For repeated rate limits, set global cooldown
                    if attempt >= 3:
                        self._set_cooldown("global", cooldown_minutes * 2)
                    
                    logger.warning(f"Rate limited on attempt {attempt}/{max_retries}, returning error")
                    return {"error": "API rate limited, please try again later"}
                
                # For other errors, retry with backoff
                if attempt < max_retries:
                    backoff = 2 ** (attempt - 1) * (1 + random.uniform(0, 0.5))
                    logger.warning(f"Error executing function (attempt {attempt}/{max_retries}): {str(e)}")
                    logger.info(f"Backing off for {backoff:.2f}s before retry")
                    time.sleep(backoff)
                else:
                    logger.error(f"Failed to execute function after {max_retries} attempts: {str(e)}")
                    return {"error": str(e)}
    
    def force_cooldown(self, endpoint: str = "global", duration_minutes: int = 60):
        """
        Force a cooldown period for an endpoint.
        
        Args:
            endpoint: API endpoint or "global" for all endpoints
            duration_minutes: Cooldown duration in minutes
        """
        self._set_cooldown(endpoint, duration_minutes)
    
    def clear_cooldown(self, endpoint: Optional[str] = None):
        """
        Clear cooldown period for an endpoint or all endpoints.
        
        Args:
            endpoint: API endpoint or None for all endpoints
        """
        if endpoint:
            if endpoint in self.cooldown_periods:
                del self.cooldown_periods[endpoint]
                logger.info(f"Cleared cooldown for {endpoint}")
        else:
            self.cooldown_periods.clear()
            logger.info("Cleared all cooldowns")
        
        self._save_cooldown_periods()
    
    def set_mock_data_mode(self, use_mock_data: bool):
        """
        Set whether to use mock data.
        
        Args:
            use_mock_data: Whether to use mock data
        """
        if use_mock_data and not self.use_mock_data:
            # Initialize mock data provider if needed
            try:
                from mock_data_provider import MockDataProvider
                self.mock_data_provider = MockDataProvider()
                self.use_mock_data = True
                logger.info("Switched to mock data mode")
            except ImportError:
                logger.error("Failed to import MockDataProvider, cannot switch to mock data mode")
        elif not use_mock_data and self.use_mock_data:
            self.use_mock_data = False
            logger.info("Switched to real data mode")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get rate limiter status.
        
        Returns:
            Rate limiter status
        """
        current_time = int(time.time())
        
        # Get active cooldowns
        active_cooldowns = {}
        for endpoint, expiry in self.cooldown_periods.items():
            if expiry > current_time:
                remaining_minutes = (expiry - current_time) // 60
                active_cooldowns[endpoint] = {
                    "expiry": expiry,
                    "expiry_time": datetime.fromtimestamp(expiry).strftime("%Y-%m-%d %H:%M:%S"),
                    "remaining_minutes": remaining_minutes
                }
        
        # Get request counts
        request_counts = {}
        for endpoint, timestamps in self.request_counts.items():
            minute_ago = current_time - 60
            recent_requests = [ts for ts in timestamps if ts > minute_ago]
            request_counts[endpoint] = len(recent_requests)
        
        return {
            "use_mock_data": self.use_mock_data,
            "active_cooldowns": active_cooldowns,
            "request_counts": request_counts,
            "rate_limits": self.rate_limits
        }
