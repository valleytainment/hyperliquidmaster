#!/usr/bin/env python3
"""
API Rate Limiter with Enhanced Mock Data Integration

This module provides rate limiting for API calls with enhanced mock data fallback
when rate limits are encountered. It includes persistent cooldown periods,
circuit breaker patterns, and seamless switching between real and mock data.
"""

import os
import time
import json
import random
import logging
import threading
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

# Import enhanced mock data provider with proper relative import
from core.enhanced_mock_data_provider import EnhancedMockDataProvider

# Configure logging
logger = logging.getLogger("APIRateLimiter")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(console_handler)

class APIRateLimiter:
    """
    API Rate Limiter with Enhanced Mock Data Integration.
    
    This class provides rate limiting for API calls with enhanced mock data fallback
    when rate limits are encountered. It includes persistent cooldown periods,
    circuit breaker patterns, and seamless switching between real and mock data.
    """
    
    def __init__(self, cooldown_period: int = 3600, mock_data_mode: bool = False, 
                 state_file: str = "rate_limiter_state.json", max_retries: int = 6,
                 retry_base_delay: float = 0.5, retry_max_delay: float = 60.0,
                 circuit_breaker_threshold: int = 3, circuit_breaker_timeout: int = 300):
        """
        Initialize the API rate limiter.
        
        Args:
            cooldown_period: Cooldown period in seconds after rate limit is hit (default: 1 hour)
            mock_data_mode: Whether to use mock data mode (default: False)
            state_file: File to store rate limiter state (default: rate_limiter_state.json)
            max_retries: Maximum number of retries for API calls (default: 6)
            retry_base_delay: Base delay for exponential backoff in seconds (default: 0.5)
            retry_max_delay: Maximum delay for exponential backoff in seconds (default: 60.0)
            circuit_breaker_threshold: Number of consecutive failures to open circuit breaker (default: 3)
            circuit_breaker_timeout: Timeout in seconds before circuit breaker resets (default: 300)
        """
        self.cooldown_period = cooldown_period
        self.mock_data_mode = mock_data_mode
        self.state_file = state_file
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        
        # Initialize rate limit state
        self.rate_limit_state = {}
        self.circuit_breakers = {}
        self.endpoint_stats = {}
        self.lock = threading.RLock()
        
        # Initialize mock data provider
        self.mock_data_provider = EnhancedMockDataProvider()
        
        # Load state from file
        self._load_state()
        
        # Log initialization
        if self.mock_data_mode:
            logger.info("API rate limiter initialized with mock data mode")
        else:
            logger.info("API rate limiter initialized with real data mode")
            
    def _load_state(self) -> None:
        """
        Load rate limiter state from file.
        """
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                    
                    # Load rate limit state
                    self.rate_limit_state = state.get("rate_limit_state", {})
                    
                    # Convert string timestamps to float
                    for endpoint, endpoint_state in self.rate_limit_state.items():
                        if "cooldown_until" in endpoint_state:
                            endpoint_state["cooldown_until"] = float(endpoint_state["cooldown_until"])
                            
                    # Load circuit breakers
                    self.circuit_breakers = state.get("circuit_breakers", {})
                    
                    # Convert string timestamps to float
                    for endpoint, breaker_state in self.circuit_breakers.items():
                        if "open_until" in breaker_state:
                            breaker_state["open_until"] = float(breaker_state["open_until"])
                            
                    # Load endpoint stats
                    self.endpoint_stats = state.get("endpoint_stats", {})
                    
                    # Check if mock data mode should be enabled
                    any_cooldown_active = any(
                        endpoint_state.get("cooldown_until", 0) > time.time()
                        for endpoint_state in self.rate_limit_state.values()
                    )
                    
                    any_circuit_open = any(
                        breaker_state.get("open_until", 0) > time.time()
                        for breaker_state in self.circuit_breakers.values()
                    )
                    
                    if any_cooldown_active or any_circuit_open:
                        logger.warning("Rate limits or circuit breakers active, enabling mock data mode")
                        self.mock_data_mode = True
                        
                    logger.info(f"Loaded rate limiter state from {self.state_file}")
        except Exception as e:
            logger.warning(f"Error loading rate limiter state: {str(e)}")
            
    def _save_state(self) -> None:
        """
        Save rate limiter state to file.
        """
        try:
            state = {
                "rate_limit_state": self.rate_limit_state,
                "circuit_breakers": self.circuit_breakers,
                "endpoint_stats": self.endpoint_stats,
                "last_updated": time.time()
            }
            
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2, default=str)
                
            logger.debug(f"Saved rate limiter state to {self.state_file}")
        except Exception as e:
            logger.warning(f"Error saving rate limiter state: {str(e)}")
            
    def _is_rate_limited(self, endpoint: str) -> bool:
        """
        Check if an endpoint is rate limited.
        
        Args:
            endpoint: API endpoint
            
        Returns:
            Whether the endpoint is rate limited
        """
        with self.lock:
            # Check if endpoint is in cooldown
            if endpoint in self.rate_limit_state:
                cooldown_until = self.rate_limit_state[endpoint].get("cooldown_until", 0)
                if cooldown_until > time.time():
                    cooldown_remaining = int(cooldown_until - time.time())
                    logger.debug(f"Endpoint {endpoint} in cooldown for {cooldown_remaining}s")
                    return True
                    
            return False
            
    def _is_circuit_open(self, endpoint: str) -> bool:
        """
        Check if circuit breaker is open for an endpoint.
        
        Args:
            endpoint: API endpoint
            
        Returns:
            Whether the circuit breaker is open
        """
        with self.lock:
            # Check if circuit breaker is open
            if endpoint in self.circuit_breakers:
                open_until = self.circuit_breakers[endpoint].get("open_until", 0)
                if open_until > time.time():
                    open_remaining = int(open_until - time.time())
                    logger.debug(f"Circuit breaker open for {endpoint} for {open_remaining}s")
                    return True
                    
            return False
            
    def _record_success(self, endpoint: str) -> None:
        """
        Record a successful API call.
        
        Args:
            endpoint: API endpoint
        """
        with self.lock:
            # Initialize endpoint stats if not exists
            if endpoint not in self.endpoint_stats:
                self.endpoint_stats[endpoint] = {
                    "total_calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "rate_limited_calls": 0,
                    "mock_data_calls": 0,
                    "consecutive_failures": 0,
                    "last_call_time": 0,
                    "total_response_time": 0
                }
                
            # Update endpoint stats
            self.endpoint_stats[endpoint]["total_calls"] += 1
            self.endpoint_stats[endpoint]["successful_calls"] += 1
            self.endpoint_stats[endpoint]["consecutive_failures"] = 0
            self.endpoint_stats[endpoint]["last_call_time"] = time.time()
            
            # Reset circuit breaker if it was open
            if endpoint in self.circuit_breakers:
                self.circuit_breakers[endpoint]["consecutive_failures"] = 0
                
            # Save state
            self._save_state()
            
    def _record_failure(self, endpoint: str, is_rate_limit: bool = False, response_time: float = 0) -> None:
        """
        Record a failed API call.
        
        Args:
            endpoint: API endpoint
            is_rate_limit: Whether the failure was due to rate limiting
            response_time: Response time in seconds
        """
        with self.lock:
            # Initialize endpoint stats if not exists
            if endpoint not in self.endpoint_stats:
                self.endpoint_stats[endpoint] = {
                    "total_calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "rate_limited_calls": 0,
                    "mock_data_calls": 0,
                    "consecutive_failures": 0,
                    "last_call_time": 0,
                    "total_response_time": 0
                }
                
            # Update endpoint stats
            self.endpoint_stats[endpoint]["total_calls"] += 1
            self.endpoint_stats[endpoint]["failed_calls"] += 1
            self.endpoint_stats[endpoint]["consecutive_failures"] += 1
            self.endpoint_stats[endpoint]["last_call_time"] = time.time()
            self.endpoint_stats[endpoint]["total_response_time"] += response_time
            
            if is_rate_limit:
                self.endpoint_stats[endpoint]["rate_limited_calls"] += 1
                
            # Update circuit breaker
            if endpoint not in self.circuit_breakers:
                self.circuit_breakers[endpoint] = {
                    "consecutive_failures": 0,
                    "open_until": 0
                }
                
            self.circuit_breakers[endpoint]["consecutive_failures"] += 1
            
            # Check if circuit breaker should be opened
            if self.circuit_breakers[endpoint]["consecutive_failures"] >= self.circuit_breaker_threshold:
                self.circuit_breakers[endpoint]["open_until"] = time.time() + self.circuit_breaker_timeout
                logger.warning(f"Circuit breaker opened for {endpoint} for {self.circuit_breaker_timeout}s")
                
            # Save state
            self._save_state()
            
    def _record_mock_data_usage(self, endpoint: str) -> None:
        """
        Record mock data usage.
        
        Args:
            endpoint: API endpoint
        """
        with self.lock:
            # Initialize endpoint stats if not exists
            if endpoint not in self.endpoint_stats:
                self.endpoint_stats[endpoint] = {
                    "total_calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "rate_limited_calls": 0,
                    "mock_data_calls": 0,
                    "consecutive_failures": 0,
                    "last_call_time": 0,
                    "total_response_time": 0
                }
                
            # Update endpoint stats
            self.endpoint_stats[endpoint]["total_calls"] += 1
            self.endpoint_stats[endpoint]["mock_data_calls"] += 1
            self.endpoint_stats[endpoint]["last_call_time"] = time.time()
            
            # Save state
            self._save_state()
            
    def _set_rate_limit_cooldown(self, endpoint: str) -> None:
        """
        Set rate limit cooldown for an endpoint.
        
        Args:
            endpoint: API endpoint
        """
        with self.lock:
            # Initialize rate limit state if not exists
            if endpoint not in self.rate_limit_state:
                self.rate_limit_state[endpoint] = {
                    "rate_limit_count": 0,
                    "cooldown_until": 0
                }
                
            # Update rate limit state
            self.rate_limit_state[endpoint]["rate_limit_count"] += 1
            
            # Calculate cooldown period with exponential backoff
            count = self.rate_limit_state[endpoint]["rate_limit_count"]
            cooldown = min(self.cooldown_period * (2 ** (count - 1)), self.cooldown_period * 24)  # Max 24x cooldown
            
            # Set cooldown until
            self.rate_limit_state[endpoint]["cooldown_until"] = time.time() + cooldown
            
            logger.warning(f"Rate limit cooldown set for {endpoint} for {cooldown}s (count: {count})")
            
            # Enable mock data mode
            self.mock_data_mode = True
            logger.info("Switching to mock data mode due to rate limiting")
            
            # Save state
            self._save_state()
            
    def execute_with_rate_limit(self, func: Callable, endpoint: str, params: Dict = None) -> Any:
        """
        Execute a function with rate limiting.
        
        Args:
            func: Function to execute
            endpoint: API endpoint
            params: Parameters for the function
            
        Returns:
            Result of the function
        """
        # Default params
        if params is None:
            params = {}
            
        # Check if mock data mode is enabled
        if self.mock_data_mode:
            return self._get_mock_data(endpoint, params)
            
        # Check if endpoint is rate limited
        if self._is_rate_limited(endpoint):
            logger.warning(f"Endpoint {endpoint} is rate limited, using mock data")
            return self._get_mock_data(endpoint, params)
            
        # Check if circuit breaker is open
        if self._is_circuit_open(endpoint):
            logger.warning(f"Circuit breaker open for {endpoint}, request blocked")
            return self._get_mock_data(endpoint, params)
            
        # Execute function with retries
        for attempt in range(1, self.max_retries + 1):
            try:
                start_time = time.time()
                result = func()
                response_time = time.time() - start_time
                
                # Record success
                self._record_success(endpoint)
                
                # Cache result for mock data
                if isinstance(result, (dict, list)):
                    self.mock_data_provider.cache_real_response(endpoint, params, result)
                    
                return result
            except Exception as e:
                response_time = time.time() - start_time
                
                # Check if rate limited
                is_rate_limit = False
                if hasattr(e, "args") and len(e.args) >= 3 and e.args[2] == "rate limited":
                    is_rate_limit = True
                    self._set_rate_limit_cooldown(endpoint)
                    
                # Record failure
                self._record_failure(endpoint, is_rate_limit, response_time)
                
                # Calculate retry delay with exponential backoff
                if attempt < self.max_retries:
                    delay = min(self.retry_base_delay * (2 ** (attempt - 1)), self.retry_max_delay)
                    delay = delay * (0.5 + random.random())  # Add jitter
                    logger.warning(f"Retrying {endpoint} in {delay:.2f}s (attempt {attempt}/{self.max_retries})")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to execute {endpoint} after {self.max_retries} attempts")
                    
                    # Use mock data as fallback
                    if attempt == self.max_retries:
                        logger.warning(f"Using mock data for {endpoint} after max retries")
                        return self._get_mock_data(endpoint, params)
                        
        # This should never be reached, but just in case
        return self._get_mock_data(endpoint, params)
        
    def _get_mock_data(self, endpoint: str, params: Dict) -> Any:
        """
        Get mock data for an endpoint.
        
        Args:
            endpoint: API endpoint
            params: Parameters for the endpoint
            
        Returns:
            Mock data
        """
        try:
            # Record mock data usage
            self._record_mock_data_usage(endpoint)
            
            # Get mock data from provider
            if endpoint == "klines":
                return self.mock_data_provider.get_mock_klines(
                    params.get("symbol", "BTC"),
                    params.get("interval", "1h"),
                    params.get("limit", 100)
                )
            elif endpoint == "ticker":
                return self.mock_data_provider.get_mock_ticker(
                    params.get("symbol", "BTC")
                )
            elif endpoint == "orderbook":
                return self.mock_data_provider.get_mock_orderbook(
                    params.get("symbol", "BTC"),
                    params.get("limit", 100)
                )
            elif endpoint == "trades":
                return self.mock_data_provider.get_mock_trades(
                    params.get("symbol", "BTC"),
                    params.get("limit", 100)
                )
            else:
                # Generic mock data
                return self.mock_data_provider.get_mock_data(endpoint, params)
        except Exception as e:
            logger.error(f"Error getting mock data for {endpoint}: {str(e)}")
            
            # Return empty result based on endpoint
            if endpoint == "klines":
                return []
            elif endpoint == "ticker":
                return {}
            elif endpoint == "orderbook":
                return {"bids": [], "asks": []}
            elif endpoint == "trades":
                return []
            else:
                return {}
                
    def record_call(self, endpoint: str) -> None:
        """
        Record an API call for rate limiting purposes.
        
        Args:
            endpoint: API endpoint
        """
        with self.lock:
            # Initialize endpoint stats if not exists
            if endpoint not in self.endpoint_stats:
                self.endpoint_stats[endpoint] = {
                    "total_calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "rate_limited_calls": 0,
                    "mock_data_calls": 0,
                    "consecutive_failures": 0,
                    "last_call_time": 0,
                    "total_response_time": 0,
                    "minute_requests": 0,
                    "hour_requests": 0,
                    "last_minute": 0,
                    "last_hour": 0
                }
                
            # Get current minute and hour
            current_time = time.time()
            current_minute = int(current_time / 60)
            current_hour = int(current_time / 3600)
            
            # Reset counters if minute/hour changed
            if current_minute != self.endpoint_stats[endpoint].get("last_minute", 0):
                self.endpoint_stats[endpoint]["minute_requests"] = 0
                self.endpoint_stats[endpoint]["last_minute"] = current_minute
                
            if current_hour != self.endpoint_stats[endpoint].get("last_hour", 0):
                self.endpoint_stats[endpoint]["hour_requests"] = 0
                self.endpoint_stats[endpoint]["last_hour"] = current_hour
                
            # Increment counters
            self.endpoint_stats[endpoint]["minute_requests"] += 1
            self.endpoint_stats[endpoint]["hour_requests"] += 1
            
            # Check if rate limited
            minute_limit = 60  # 60 requests per minute
            hour_limit = 1000  # 1000 requests per hour
            
            if self.endpoint_stats[endpoint]["minute_requests"] > minute_limit:
                logger.warning(f"Rate limit exceeded for {endpoint}: {self.endpoint_stats[endpoint]['minute_requests']} requests in current minute (limit: {minute_limit})")
                self._set_rate_limit_cooldown(endpoint)
                
            if self.endpoint_stats[endpoint]["hour_requests"] > hour_limit:
                logger.warning(f"Rate limit exceeded for {endpoint}: {self.endpoint_stats[endpoint]['hour_requests']} requests in current hour (limit: {hour_limit})")
                self._set_rate_limit_cooldown(endpoint)
                
    def reset(self) -> None:
        """
        Reset rate limiter state.
        """
        with self.lock:
            # Reset rate limit state
            self.rate_limit_state = {}
            
            # Reset circuit breakers
            self.circuit_breakers = {}
            
            # Reset endpoint stats
            self.endpoint_stats = {}
            
            # Reset mock data mode
            self.mock_data_mode = False
            
            # Save state
            self._save_state()
            
            logger.info("Rate limiter state reset")
            
    def get_status(self) -> Dict:
        """
        Get rate limiter status.
        
        Returns:
            Rate limiter status
        """
        with self.lock:
            # Calculate cooldown remaining
            cooldown_remaining = 0
            for endpoint, endpoint_state in self.rate_limit_state.items():
                cooldown_until = endpoint_state.get("cooldown_until", 0)
                if cooldown_until > time.time():
                    cooldown_remaining = max(cooldown_remaining, int(cooldown_until - time.time()))
                    
            # Format cooldown remaining
            cooldown_remaining_formatted = str(timedelta(seconds=cooldown_remaining))
            
            # Get total requests
            minute_requests = 0
            hour_requests = 0
            
            for endpoint, stats in self.endpoint_stats.items():
                minute_requests += stats.get("minute_requests", 0)
                hour_requests += stats.get("hour_requests", 0)
                
            # Return status
            return {
                "minute_requests": minute_requests,
                "hour_requests": hour_requests,
                "max_requests_per_minute": 60,
                "max_requests_per_hour": 1000,
                "in_cooldown": cooldown_remaining > 0,
                "cooldown_remaining_seconds": cooldown_remaining,
                "cooldown_remaining_formatted": cooldown_remaining_formatted,
                "is_limited": cooldown_remaining > 0  # Added for compatibility with tests
            }
