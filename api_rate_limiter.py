"""
API Rate Limiter

This module provides functionality to manage API rate limits
with throttling, caching, and graceful degradation.
"""

import time
import logging
import threading
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Callable, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class APIRateLimiter:
    """
    Manages API rate limits with throttling, caching, and graceful degradation.
    """
    
    def __init__(self, 
                 requests_per_minute: int = 60,
                 cache_ttl_seconds: int = 5,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """
        Initialize the API rate limiter.
        
        Args:
            requests_per_minute: Maximum number of requests per minute
            cache_ttl_seconds: Time-to-live for cached responses in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Base delay between retries in seconds
        """
        self.requests_per_minute = requests_per_minute
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.request_timestamps = []
        self.cache = {}
        self.lock = threading.RLock()
        
    def _clean_old_timestamps(self):
        """
        Remove timestamps older than 1 minute.
        """
        with self.lock:
            now = time.time()
            self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 60]
            
    def _can_make_request(self) -> bool:
        """
        Check if a request can be made without exceeding the rate limit.
        
        Returns:
            True if request can be made, False otherwise
        """
        with self.lock:
            self._clean_old_timestamps()
            return len(self.request_timestamps) < self.requests_per_minute
            
    def _wait_for_rate_limit(self) -> None:
        """
        Wait until a request can be made without exceeding the rate limit.
        """
        while not self._can_make_request():
            time.sleep(0.1)
            
    def _record_request(self) -> None:
        """
        Record a request timestamp.
        """
        with self.lock:
            self.request_timestamps.append(time.time())
            
    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """
        Generate a cache key for the request.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            Cache key string
        """
        return f"{endpoint}:{json.dumps(params, sort_keys=True)}"
            
    def _get_from_cache(self, endpoint: str, params: Dict) -> Tuple[bool, Any]:
        """
        Get a response from the cache if available and not expired.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            Tuple of (cache_hit, response)
        """
        with self.lock:
            cache_key = self._get_cache_key(endpoint, params)
            
            if cache_key in self.cache:
                timestamp, response = self.cache[cache_key]
                
                if time.time() - timestamp < self.cache_ttl_seconds:
                    return True, response
                    
        return False, None
        
    def _store_in_cache(self, endpoint: str, params: Dict, response: Any) -> None:
        """
        Store a response in the cache.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            response: Response to cache
        """
        with self.lock:
            cache_key = self._get_cache_key(endpoint, params)
            self.cache[cache_key] = (time.time(), response)
            
    def execute_with_rate_limit(self, 
                               func: Callable, 
                               endpoint: str, 
                               params: Dict,
                               use_cache: bool = True) -> Any:
        """
        Execute a function with rate limiting, caching, and retries.
        
        Args:
            func: Function to execute
            endpoint: API endpoint (for caching)
            params: Request parameters (for caching)
            use_cache: Whether to use caching
            
        Returns:
            Function response
        """
        # Check cache first if enabled
        if use_cache:
            cache_hit, cached_response = self._get_from_cache(endpoint, params)
            
            if cache_hit:
                logger.debug(f"Cache hit for {endpoint}")
                return cached_response
                
        # Wait for rate limit if needed
        self._wait_for_rate_limit()
        
        # Record the request
        self._record_request()
        
        # Execute with retries
        for attempt in range(self.max_retries):
            try:
                response = func()
                
                # Cache the successful response if enabled
                if use_cache:
                    self._store_in_cache(endpoint, params, response)
                    
                return response
            except Exception as e:
                if "429" in str(e) and attempt < self.max_retries - 1:
                    # Rate limit hit, apply exponential backoff
                    sleep_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit, retrying in {sleep_time:.2f} seconds (attempt {attempt+1}/{self.max_retries})")
                    time.sleep(sleep_time)
                else:
                    # If it's the last attempt or not a rate limit error, re-raise
                    if attempt == self.max_retries - 1:
                        logger.error(f"All retry attempts failed for {endpoint}")
                    raise
                    
        # This should not be reached due to the re-raise above
        return None

    async def execute(self, func, *args, **kwargs):
        """
        Execute a function with rate limiting, caching, and retries.
        This is a simplified wrapper for async functions that handles both async and sync functions.
        
        Args:
            func: Function to execute (can be async or sync)
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Function response
        """
        # Generate a simple cache key based on function name and args
        endpoint = func.__name__
        params = {"args": str(args), "kwargs": str(kwargs)}
        
        # Check cache first
        cache_hit, cached_response = self._get_from_cache(endpoint, params)
        if cache_hit:
            logger.debug(f"Cache hit for {endpoint}")
            return cached_response
            
        # Wait for rate limit if needed
        self._wait_for_rate_limit()
        
        # Record the request
        self._record_request()
        
        # Execute with retries
        for attempt in range(self.max_retries):
            try:
                # Handle both async and sync functions
                if asyncio.iscoroutinefunction(func):
                    response = await func(*args, **kwargs)
                else:
                    response = func(*args, **kwargs)
                
                # Cache the successful response
                self._store_in_cache(endpoint, params, response)
                
                return response
            except Exception as e:
                if "429" in str(e) and attempt < self.max_retries - 1:
                    # Rate limit hit, apply exponential backoff
                    sleep_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit, retrying in {sleep_time:.2f} seconds (attempt {attempt+1}/{self.max_retries})")
                    await asyncio.sleep(sleep_time)
                else:
                    # If it's the last attempt or not a rate limit error, re-raise
                    if attempt == self.max_retries - 1:
                        logger.error(f"All retry attempts failed for {endpoint}")
                    raise
                    
        # This should not be reached due to the re-raise above
        return None
