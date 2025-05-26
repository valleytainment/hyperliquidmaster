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

# Import enhanced mock data provider
from enhanced_mock_data_provider import EnhancedMockDataProvider

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
                elif str(e).lower().find("rate limit") >= 0 or str(e).lower().find("429") >= 0:
                    is_rate_limit = True
                    self._set_rate_limit_cooldown(endpoint)
                    
                # Record failure
                self._record_failure(endpoint, is_rate_limit, response_time)
                
                # Check if last attempt
                if attempt == self.max_retries:
                    logger.warning(f"Error executing {endpoint} (attempt {attempt}/{self.max_retries}): {str(e)}")
                    
                    # If rate limited, use mock data
                    if is_rate_limit:
                        logger.info(f"Using mock data for {endpoint} due to rate limiting")
                        return self._get_mock_data(endpoint, params)
                        
                    # Otherwise, raise exception
                    raise Exception(f"Failed to execute {endpoint}: {str(e)}")
                    
                # Calculate backoff delay
                backoff_delay = min(self.retry_base_delay * (2 ** (attempt - 1)), self.retry_max_delay)
                backoff_delay *= (1 + random.random() * 0.1)  # Add jitter
                
                logger.info(f"Backing off for {backoff_delay:.2f}s before retry")
                time.sleep(backoff_delay)
                
                logger.warning(f"Error executing {endpoint} (attempt {attempt}/{self.max_retries}): {str(e)}")
                
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
            
            # Check if cached response exists
            cached_response = self.mock_data_provider.get_cached_response(endpoint, params)
            if cached_response is not None:
                logger.debug(f"Using cached response for {endpoint}")
                return cached_response
                
            # Generate mock data based on endpoint
            if endpoint == "market_data":
                symbol = params.get("symbol", "BTC")
                return self.mock_data_provider.get_synthetic_market_data(symbol)
            elif endpoint == "order_book":
                symbol = params.get("symbol", "BTC")
                return self.mock_data_provider.get_synthetic_order_book(symbol)
            elif endpoint == "candles":
                symbol = params.get("symbol", "BTC")
                timeframe = params.get("timeframe", "1m")
                limit = params.get("limit", 100)
                return self.mock_data_provider.get_synthetic_candles(symbol, timeframe, limit)
            elif endpoint == "historical_data":
                symbol = params.get("symbol", "BTC")
                timeframe = params.get("timeframe", "1h")
                start_date = params.get("start_date", (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"))
                end_date = params.get("end_date", datetime.now().strftime("%Y-%m-%d"))
                return self.mock_data_provider.get_synthetic_historical_data(symbol, timeframe, start_date, end_date)
            elif endpoint == "positions":
                symbol = params.get("symbol")
                return self.mock_data_provider.get_positions(symbol)
            elif endpoint == "execute_order":
                return self.mock_data_provider.execute_order(params)
            else:
                logger.warning(f"Unknown endpoint for mock data: {endpoint}")
                return {"error": f"Unknown endpoint: {endpoint}"}
        except Exception as e:
            logger.error(f"Error generating mock data for {endpoint}: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"Error generating mock data: {str(e)}"}
            
    def get_statistics(self) -> Dict:
        """
        Get rate limiter statistics.
        
        Returns:
            Rate limiter statistics
        """
        with self.lock:
            # Calculate overall statistics
            total_calls = sum(stats.get("total_calls", 0) for stats in self.endpoint_stats.values())
            successful_calls = sum(stats.get("successful_calls", 0) for stats in self.endpoint_stats.values())
            failed_calls = sum(stats.get("failed_calls", 0) for stats in self.endpoint_stats.values())
            rate_limited_calls = sum(stats.get("rate_limited_calls", 0) for stats in self.endpoint_stats.values())
            mock_data_calls = sum(stats.get("mock_data_calls", 0) for stats in self.endpoint_stats.values())
            
            # Calculate success rate
            success_rate = successful_calls / total_calls if total_calls > 0 else 0
            
            # Calculate mock data rate
            mock_data_rate = mock_data_calls / total_calls if total_calls > 0 else 0
            
            # Calculate average response time
            total_response_time = sum(stats.get("total_response_time", 0) for stats in self.endpoint_stats.values())
            avg_response_time = total_response_time / total_calls if total_calls > 0 else 0
            
            # Get active cooldowns
            active_cooldowns = {}
            for endpoint, state in self.rate_limit_state.items():
                cooldown_until = state.get("cooldown_until", 0)
                if cooldown_until > time.time():
                    active_cooldowns[endpoint] = {
                        "cooldown_remaining": int(cooldown_until - time.time()),
                        "rate_limit_count": state.get("rate_limit_count", 0)
                    }
                    
            # Get open circuit breakers
            open_circuit_breakers = {}
            for endpoint, state in self.circuit_breakers.items():
                open_until = state.get("open_until", 0)
                if open_until > time.time():
                    open_circuit_breakers[endpoint] = {
                        "open_remaining": int(open_until - time.time()),
                        "consecutive_failures": state.get("consecutive_failures", 0)
                    }
                    
            return {
                "total_calls": total_calls,
                "successful_calls": successful_calls,
                "failed_calls": failed_calls,
                "rate_limited_calls": rate_limited_calls,
                "mock_data_calls": mock_data_calls,
                "success_rate": success_rate,
                "mock_data_rate": mock_data_rate,
                "avg_response_time": avg_response_time,
                "mock_data_mode": self.mock_data_mode,
                "active_cooldowns": active_cooldowns,
                "open_circuit_breakers": open_circuit_breakers,
                "endpoint_stats": self.endpoint_stats
            }
            
    def reset_statistics(self) -> None:
        """
        Reset rate limiter statistics.
        """
        with self.lock:
            self.endpoint_stats = {}
            self._save_state()
            
    def reset_cooldowns(self) -> None:
        """
        Reset all cooldowns.
        """
        with self.lock:
            for endpoint in self.rate_limit_state:
                self.rate_limit_state[endpoint]["cooldown_until"] = 0
                self.rate_limit_state[endpoint]["rate_limit_count"] = 0
                
            self._save_state()
            
            logger.info("All cooldowns reset")
            
    def reset_circuit_breakers(self) -> None:
        """
        Reset all circuit breakers.
        """
        with self.lock:
            for endpoint in self.circuit_breakers:
                self.circuit_breakers[endpoint]["open_until"] = 0
                self.circuit_breakers[endpoint]["consecutive_failures"] = 0
                
            self._save_state()
            
            logger.info("All circuit breakers reset")
            
    def set_mock_data_mode(self, enabled: bool) -> None:
        """
        Set mock data mode.
        
        Args:
            enabled: Whether to enable mock data mode
        """
        with self.lock:
            self.mock_data_mode = enabled
            self._save_state()
            
            if enabled:
                logger.info("Mock data mode enabled")
            else:
                logger.info("Mock data mode disabled")
                
    def is_mock_data_mode(self) -> bool:
        """
        Check if mock data mode is enabled.
        
        Returns:
            Whether mock data mode is enabled
        """
        return self.mock_data_mode
        
    def generate_synthetic_data(self, symbols: List[str] = None, timeframes: List[str] = None, days: int = 30) -> None:
        """
        Generate synthetic data for testing.
        
        Args:
            symbols: List of symbols to generate data for (if None, use default symbols)
            timeframes: List of timeframes to generate data for (if None, use default timeframes)
            days: Number of days of data to generate
        """
        self.mock_data_provider.generate_all_synthetic_data(symbols, timeframes, days)
        logger.info(f"Generated synthetic data for {len(symbols) if symbols else 'default'} symbols and {len(timeframes) if timeframes else 'default'} timeframes")
        
def main():
    """
    Main function for testing.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Create rate limiter
    rate_limiter = APIRateLimiter()
    
    # Generate synthetic data
    rate_limiter.generate_synthetic_data()
    
    # Test rate limiter
    def test_func():
        return {"success": True}
        
    def test_func_rate_limited():
        raise Exception("429", None, "rate limited")
        
    # Test successful call
    result = rate_limiter.execute_with_rate_limit(test_func, "test_endpoint")
    print(f"Result: {result}")
    
    # Test rate limited call
    try:
        result = rate_limiter.execute_with_rate_limit(test_func_rate_limited, "test_endpoint")
        print(f"Result after rate limit: {result}")
    except Exception as e:
        print(f"Error: {str(e)}")
        
    # Get statistics
    stats = rate_limiter.get_statistics()
    print(f"Statistics: {json.dumps(stats, indent=2, default=str)}")
    
if __name__ == "__main__":
    main()
