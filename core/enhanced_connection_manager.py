"""
Enhanced Connection Manager for HyperLiquid Trading Bot

This module provides robust connection management with automatic retry,
exponential backoff, and circuit breaker patterns for API interactions.
"""

import time
import logging
import asyncio
import random
from typing import Dict, Any, Callable, Optional, TypeVar, Generic

T = TypeVar('T')

class EnhancedConnectionManager:
    """
    Enhanced connection manager with circuit breaker pattern and exponential backoff.
    Provides robust connection handling for API interactions.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the connection manager.
        
        Args:
            logger: Optional logger instance
        """
        # Setup logging
        self.logger = logger or logging.getLogger("EnhancedConnectionManager")
        
        # Circuit breaker state
        self.circuit_open = False
        self.failure_count = 0
        self.last_failure_time = 0
        self.reset_timeout = 30  # seconds
        
        # Retry configuration
        self.max_retries = 5
        self.base_delay = 1  # seconds
        self.max_delay = 60  # seconds
        self.jitter_factor = 0.1  # 10% jitter
        
        # Connection state
        self.last_successful_connection = 0
        self.connection_health = 1.0  # 1.0 = perfect health, 0.0 = completely unhealthy
        
        self.logger.info("Enhanced connection manager initialized")
    
    def reset_state(self):
        """Reset the connection manager state."""
        self.circuit_open = False
        self.failure_count = 0
        self.last_failure_time = 0
        self.connection_health = 1.0
        self.logger.info("Connection manager state reset")
    
    def _should_attempt_reconnect(self) -> bool:
        """
        Determine if a reconnection attempt should be made based on circuit breaker state.
        
        Returns:
            True if reconnection should be attempted, False otherwise
        """
        # If circuit is closed, always allow connection attempts
        if not self.circuit_open:
            return True
            
        # If circuit is open, check if enough time has passed to try again
        current_time = time.time()
        time_since_failure = current_time - self.last_failure_time
        
        # Calculate dynamic reset timeout based on failure count
        dynamic_timeout = min(self.reset_timeout * (2 ** min(self.failure_count - 1, 5)), 300)
        
        if time_since_failure > dynamic_timeout:
            self.logger.info(f"Circuit half-open after {time_since_failure:.1f}s, attempting reconnection")
            return True
            
        return False
    
    def _update_circuit_state(self, success: bool):
        """
        Update circuit breaker state based on connection attempt result.
        
        Args:
            success: Whether the connection attempt was successful
        """
        if success:
            # Reset circuit on success
            if self.circuit_open:
                self.logger.info("Connection successful, closing circuit")
            self.circuit_open = False
            self.failure_count = 0
            self.last_successful_connection = time.time()
            
            # Improve connection health
            self.connection_health = min(1.0, self.connection_health + 0.2)
        else:
            # Update failure stats
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # Degrade connection health
            self.connection_health = max(0.0, self.connection_health - 0.25)
            
            # Open circuit if too many failures
            if self.failure_count >= 3 and not self.circuit_open:
                self.logger.warning(f"Opening circuit after {self.failure_count} consecutive failures")
                self.circuit_open = True
    
    def _calculate_backoff_time(self, retry_count: int) -> float:
        """
        Calculate backoff time with exponential backoff and jitter.
        
        Args:
            retry_count: Current retry attempt number
            
        Returns:
            Time to wait in seconds
        """
        # Calculate base delay with exponential backoff
        delay = min(self.max_delay, self.base_delay * (2 ** retry_count))
        
        # Add jitter to prevent thundering herd problem
        jitter = delay * self.jitter_factor * random.random()
        
        return delay + jitter
    
    def ensure_connection(self, connect_func: Callable[[], bool], test_func: Callable[[], bool]) -> bool:
        """
        Ensure connection is established with retry logic.
        
        Args:
            connect_func: Function to establish connection
            test_func: Function to test if connection is valid
            
        Returns:
            True if connection is established, False otherwise
        """
        # Check if circuit is open
        if self.circuit_open and not self._should_attempt_reconnect():
            self.logger.warning("Circuit open, skipping connection attempt")
            return False
            
        # Try to use existing connection
        try:
            if test_func():
                self._update_circuit_state(True)
                return True
        except Exception as e:
            self.logger.debug(f"Connection test failed: {e}")
            # Continue to reconnection attempt
        
        # Attempt reconnection with retry
        for retry in range(self.max_retries):
            try:
                self.logger.info(f"Connection attempt {retry + 1}/{self.max_retries}")
                
                # Attempt to connect
                if connect_func():
                    # Test connection
                    if test_func():
                        self.logger.info(f"Connection established successfully on attempt {retry + 1}")
                        self._update_circuit_state(True)
                        return True
                
                # If we get here, connection failed
                self.logger.warning(f"Connection attempt {retry + 1} failed")
                
                # Calculate backoff time
                backoff_time = self._calculate_backoff_time(retry)
                self.logger.info(f"Waiting {backoff_time:.2f}s before next attempt")
                
                # Wait before next attempt
                time.sleep(backoff_time)
                
            except Exception as e:
                self.logger.error(f"Error during connection attempt {retry + 1}: {e}")
                
                # Calculate backoff time
                backoff_time = self._calculate_backoff_time(retry)
                self.logger.info(f"Waiting {backoff_time:.2f}s before next attempt")
                
                # Wait before next attempt
                time.sleep(backoff_time)
        
        # All attempts failed
        self.logger.error(f"All {self.max_retries} connection attempts failed")
        self._update_circuit_state(False)
        return False
    
    def safe_api_call(self, api_func: Callable[[], T], connect_func: Callable[[], bool] = None, test_func: Callable[[], bool] = None) -> T:
        """
        Safely make an API call with automatic reconnection if needed.
        
        Args:
            api_func: Function to make the API call
            connect_func: Optional function to establish connection if needed
            test_func: Optional function to test if connection is valid
            
        Returns:
            Result of the API call
        """
        # Ensure connection if connect_func and test_func are provided
        if connect_func and test_func:
            if not self.ensure_connection(connect_func, test_func):
                return {"error": "Failed to establish connection"}
        
        # Make API call with retry
        for retry in range(self.max_retries):
            try:
                # Attempt API call
                result = api_func()
                
                # Update connection health on success
                self._update_circuit_state(True)
                
                return result
                
            except Exception as e:
                self.logger.error(f"API call failed (attempt {retry + 1}/{self.max_retries}): {e}")
                
                # Update circuit state
                self._update_circuit_state(False)
                
                # Last attempt, return error
                if retry == self.max_retries - 1:
                    return {"error": f"API call failed after {self.max_retries} attempts: {str(e)}"}
                
                # Calculate backoff time
                backoff_time = self._calculate_backoff_time(retry)
                self.logger.info(f"Waiting {backoff_time:.2f}s before next attempt")
                
                # Wait before next attempt
                time.sleep(backoff_time)
                
                # Try to reconnect if connect_func and test_func are provided
                if connect_func and test_func:
                    if not self.ensure_connection(connect_func, test_func):
                        return {"error": "Failed to re-establish connection"}
        
        # This should never be reached due to the return in the last retry
        return {"error": "Unexpected error in API call retry logic"}
    
    async def async_safe_api_call(self, api_func: Callable[[], T], connect_func: Callable[[], bool] = None, test_func: Callable[[], bool] = None) -> T:
        """
        Safely make an asynchronous API call with automatic reconnection if needed.
        
        Args:
            api_func: Async function to make the API call
            connect_func: Optional async function to establish connection if needed
            test_func: Optional async function to test if connection is valid
            
        Returns:
            Result of the API call
        """
        # Ensure connection if connect_func and test_func are provided
        if connect_func and test_func:
            # Convert to async if needed
            async_connect_func = connect_func if asyncio.iscoroutinefunction(connect_func) else lambda: connect_func()
            async_test_func = test_func if asyncio.iscoroutinefunction(test_func) else lambda: test_func()
            
            # Ensure connection
            connection_result = await self._async_ensure_connection(async_connect_func, async_test_func)
            if not connection_result:
                return {"error": "Failed to establish connection"}
        
        # Make API call with retry
        for retry in range(self.max_retries):
            try:
                # Attempt API call
                result = await api_func()
                
                # Update connection health on success
                self._update_circuit_state(True)
                
                return result
                
            except Exception as e:
                self.logger.error(f"Async API call failed (attempt {retry + 1}/{self.max_retries}): {e}")
                
                # Update circuit state
                self._update_circuit_state(False)
                
                # Last attempt, return error
                if retry == self.max_retries - 1:
                    return {"error": f"Async API call failed after {self.max_retries} attempts: {str(e)}"}
                
                # Calculate backoff time
                backoff_time = self._calculate_backoff_time(retry)
                self.logger.info(f"Waiting {backoff_time:.2f}s before next attempt")
                
                # Wait before next attempt
                await asyncio.sleep(backoff_time)
                
                # Try to reconnect if connect_func and test_func are provided
                if connect_func and test_func:
                    connection_result = await self._async_ensure_connection(async_connect_func, async_test_func)
                    if not connection_result:
                        return {"error": "Failed to re-establish connection"}
        
        # This should never be reached due to the return in the last retry
        return {"error": "Unexpected error in async API call retry logic"}
    
    async def _async_ensure_connection(self, connect_func, test_func) -> bool:
        """
        Ensure connection is established with retry logic (async version).
        
        Args:
            connect_func: Async function to establish connection
            test_func: Async function to test if connection is valid
            
        Returns:
            True if connection is established, False otherwise
        """
        # Check if circuit is open
        if self.circuit_open and not self._should_attempt_reconnect():
            self.logger.warning("Circuit open, skipping connection attempt")
            return False
            
        # Try to use existing connection
        try:
            if await test_func():
                self._update_circuit_state(True)
                return True
        except Exception as e:
            self.logger.debug(f"Async connection test failed: {e}")
            # Continue to reconnection attempt
        
        # Attempt reconnection with retry
        for retry in range(self.max_retries):
            try:
                self.logger.info(f"Async connection attempt {retry + 1}/{self.max_retries}")
                
                # Attempt to connect
                if await connect_func():
                    # Test connection
                    if await test_func():
                        self.logger.info(f"Async connection established successfully on attempt {retry + 1}")
                        self._update_circuit_state(True)
                        return True
                
                # If we get here, connection failed
                self.logger.warning(f"Async connection attempt {retry + 1} failed")
                
                # Calculate backoff time
                backoff_time = self._calculate_backoff_time(retry)
                self.logger.info(f"Waiting {backoff_time:.2f}s before next attempt")
                
                # Wait before next attempt
                await asyncio.sleep(backoff_time)
                
            except Exception as e:
                self.logger.error(f"Error during async connection attempt {retry + 1}: {e}")
                
                # Calculate backoff time
                backoff_time = self._calculate_backoff_time(retry)
                self.logger.info(f"Waiting {backoff_time:.2f}s before next attempt")
                
                # Wait before next attempt
                await asyncio.sleep(backoff_time)
        
        # All attempts failed
        self.logger.error(f"All {self.max_retries} async connection attempts failed")
        self._update_circuit_state(False)
        return False
