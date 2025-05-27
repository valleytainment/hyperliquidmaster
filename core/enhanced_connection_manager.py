"""
Enhanced Connection Manager for Hyperliquid Trading Bot
------------------------------------------------------
Provides robust connection management with persistent reconnection,
state preservation, and comprehensive error handling.
"""

import os
import json
import time
import logging
import threading
import traceback
from typing import Dict, Any, Optional, Callable, List, Tuple

class EnhancedConnectionManager:
    """
    Enhanced connection manager with persistent reconnection logic,
    state preservation, and comprehensive error handling.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the connection manager.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger("EnhancedConnectionManager")
        
        # Connection state
        self.is_connected = False
        self.connection_attempts = 0
        self.last_connection_attempt = 0
        self.last_successful_connection = 0
        self.connection_failures = 0
        self.consecutive_failures = 0
        
        # Connection settings
        self.max_backoff_time = 300  # Maximum backoff time in seconds (5 minutes)
        self.initial_backoff = 1  # Initial backoff time in seconds
        self.backoff_factor = 1.5  # Backoff factor for exponential backoff
        self.jitter_factor = 0.2  # Random jitter factor to avoid thundering herd
        self.health_check_interval = 30  # Health check interval in seconds
        self.last_health_check = 0
        
        # Connection lock for thread safety
        self.connection_lock = threading.Lock()
        
        # Connection state file
        self.state_file = "connection_state.json"
        
        # Load state
        self._load_state()
    
    def _load_state(self) -> None:
        """Load connection state from file."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                # Restore state
                self.is_connected = state.get("is_connected", False)
                self.connection_attempts = state.get("connection_attempts", 0)
                self.last_connection_attempt = state.get("last_connection_attempt", 0)
                self.last_successful_connection = state.get("last_successful_connection", 0)
                self.connection_failures = state.get("connection_failures", 0)
                self.consecutive_failures = state.get("consecutive_failures", 0)
                
                self.logger.info(f"Loaded connection state: attempts={self.connection_attempts}, failures={self.connection_failures}")
        except Exception as e:
            self.logger.error(f"Error loading connection state: {e}")
    
    def _save_state(self) -> None:
        """Save connection state to file."""
        try:
            state = {
                "is_connected": self.is_connected,
                "connection_attempts": self.connection_attempts,
                "last_connection_attempt": self.last_connection_attempt,
                "last_successful_connection": self.last_successful_connection,
                "connection_failures": self.connection_failures,
                "consecutive_failures": self.consecutive_failures,
                "timestamp": time.time()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving connection state: {e}")
    
    def reset_state(self) -> None:
        """Reset connection state."""
        with self.connection_lock:
            self.is_connected = False
            self.connection_attempts = 0
            self.last_connection_attempt = 0
            self.last_successful_connection = 0
            self.connection_failures = 0
            self.consecutive_failures = 0
            self._save_state()
            self.logger.info("Connection state reset")
    
    def connection_successful(self) -> None:
        """Mark connection as successful."""
        with self.connection_lock:
            self.is_connected = True
            self.last_successful_connection = time.time()
            self.consecutive_failures = 0
            self._save_state()
            self.logger.info("Connection marked as successful")
    
    def connection_failed(self) -> None:
        """Mark connection as failed."""
        with self.connection_lock:
            self.is_connected = False
            self.connection_failures += 1
            self.consecutive_failures += 1
            self._save_state()
            self.logger.warning(f"Connection marked as failed (consecutive failures: {self.consecutive_failures})")
    
    def calculate_backoff_time(self) -> float:
        """
        Calculate backoff time with exponential backoff and jitter.
        
        Returns:
            Backoff time in seconds
        """
        # Calculate base backoff time
        backoff = min(
            self.max_backoff_time,
            self.initial_backoff * (self.backoff_factor ** min(self.consecutive_failures, 10))
        )
        
        # Add jitter to avoid thundering herd
        import random
        jitter = random.uniform(-self.jitter_factor, self.jitter_factor) * backoff
        backoff = max(self.initial_backoff, backoff + jitter)
        
        return backoff
    
    def should_reconnect(self) -> bool:
        """
        Check if reconnection should be attempted.
        
        Returns:
            True if reconnection should be attempted, False otherwise
        """
        # Always attempt to reconnect
        return True
    
    def wait_before_reconnect(self) -> None:
        """Wait before reconnection attempt."""
        backoff_time = self.calculate_backoff_time()
        self.logger.info(f"Waiting {backoff_time:.2f}s before reconnection attempt")
        time.sleep(backoff_time)
    
    def ensure_connection(self, connect_func: Callable[[], bool], test_func: Callable[[], bool]) -> bool:
        """
        Ensure connection is established, attempting to reconnect if necessary.
        
        Args:
            connect_func: Function to establish connection
            test_func: Function to test connection
            
        Returns:
            True if connected, False otherwise
        """
        with self.connection_lock:
            # Check if already connected
            if self.is_connected:
                # Check if health check is due
                current_time = time.time()
                if current_time - self.last_health_check > self.health_check_interval:
                    self.last_health_check = current_time
                    if not test_func():
                        self.logger.warning("Health check failed, attempting to reconnect")
                        return self._reconnect(connect_func, test_func)
                return True
            
            # Not connected, attempt to reconnect
            return self._reconnect(connect_func, test_func)
    
    def _reconnect(self, connect_func: Callable[[], bool], test_func: Callable[[], bool]) -> bool:
        """
        Attempt to reconnect with exponential backoff.
        
        Args:
            connect_func: Function to establish connection
            test_func: Function to test connection
            
        Returns:
            True if reconnection is successful, False otherwise
        """
        # Check if reconnection should be attempted
        if not self.should_reconnect():
            self.logger.error("Reconnection not allowed")
            return False
        
        # Wait before reconnection if necessary
        current_time = time.time()
        backoff_time = self.calculate_backoff_time()
        time_since_last_attempt = current_time - self.last_connection_attempt
        
        if time_since_last_attempt < backoff_time:
            wait_time = backoff_time - time_since_last_attempt
            self.logger.info(f"Waiting {wait_time:.2f}s before reconnection attempt")
            time.sleep(wait_time)
        
        # Attempt to reconnect
        self.connection_attempts += 1
        self.last_connection_attempt = time.time()
        self.logger.info(f"Attempting to reconnect (attempt {self.connection_attempts}, consecutive failures: {self.consecutive_failures})")
        
        try:
            # Attempt connection
            if connect_func():
                # Test connection
                if test_func():
                    self.connection_successful()
                    return True
            
            # Connection failed
            self.connection_failed()
            return False
        except Exception as e:
            self.logger.error(f"Error during reconnection: {e}")
            self.logger.error(traceback.format_exc())
            self.connection_failed()
            return False
    
    def safe_api_call(self, api_func: Callable, max_retries: int = 3, retry_delay: float = 1.0,
                     connect_func: Optional[Callable[[], bool]] = None, 
                     test_func: Optional[Callable[[], bool]] = None) -> Any:
        """
        Safely call an API function with retry logic.
        
        Args:
            api_func: Function to call
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            connect_func: Function to establish connection
            test_func: Function to test connection
            
        Returns:
            Result of the API call
        """
        retries = 0
        last_error = None
        
        while retries <= max_retries:
            try:
                # Ensure connection if connect_func and test_func are provided
                if connect_func and test_func:
                    if not self.is_connected and not self.ensure_connection(connect_func, test_func):
                        return {"error": "Not connected to API"}
                
                # Call API function
                result = api_func()
                
                # Check if result is an error
                if isinstance(result, dict) and "error" in result:
                    # Check if it's a connection error
                    error_msg = str(result["error"]).lower()
                    if "connect" in error_msg or "timeout" in error_msg or "network" in error_msg:
                        self.is_connected = False
                        if connect_func and test_func:
                            if not self.ensure_connection(connect_func, test_func):
                                return {"error": f"Connection error: {result['error']}"}
                        retries += 1
                        time.sleep(retry_delay * retries)
                        continue
                
                return result
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                
                # Check if it's a connection error
                if "connect" in error_msg or "timeout" in error_msg or "network" in error_msg:
                    self.is_connected = False
                    if connect_func and test_func:
                        if not self.ensure_connection(connect_func, test_func):
                            return {"error": f"Connection error: {e}"}
                
                retries += 1
                if retries <= max_retries:
                    self.logger.warning(f"API call failed, retrying ({retries}/{max_retries}): {e}")
                    time.sleep(retry_delay * retries)
                else:
                    self.logger.error(f"API call failed after {max_retries} retries: {e}")
                    return {"error": f"API call failed: {e}"}
        
        return {"error": f"API call failed: {last_error}"}
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get connection statistics.
        
        Returns:
            Dict containing connection statistics
        """
        with self.connection_lock:
            current_time = time.time()
            
            stats = {
                "is_connected": self.is_connected,
                "connection_attempts": self.connection_attempts,
                "connection_failures": self.connection_failures,
                "consecutive_failures": self.consecutive_failures,
                "time_since_last_attempt": current_time - self.last_connection_attempt if self.last_connection_attempt > 0 else -1,
                "time_since_last_success": current_time - self.last_successful_connection if self.last_successful_connection > 0 else -1,
                "success_rate": (self.connection_attempts - self.connection_failures) / self.connection_attempts * 100 if self.connection_attempts > 0 else 0
            }
            
            return stats
