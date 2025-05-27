"""
Enhanced Connection Manager for HyperliquidMaster

This module provides robust connection management with automatic reconnection
and connection health monitoring.
"""

import time
import logging
import random
import threading
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta

class ConnectionManager:
    """
    Manages connections to the Hyperliquid exchange with robust error handling
    and automatic reconnection.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the connection manager.
        
        Args:
            logger: Logger instance (optional, will create one if not provided)
        """
        # Initialize logger if not provided
        if logger is None:
            self.logger = logging.getLogger("ConnectionManager")
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger
            
        self.connected = False
        self.last_connected_time = None
        self.connection_attempts = 0
        self.max_attempts = 10  # Unlimited retries, but with increasing backoff
        self.base_retry_delay = 2.0  # Base delay in seconds
        self.max_retry_delay = 60.0  # Maximum delay in seconds
        self.jitter_factor = 0.25  # Random jitter factor (0-1)
        self.health_check_interval = 30.0  # Health check interval in seconds
        self.health_check_timeout = 5.0  # Health check timeout in seconds
        self.connection_lock = threading.RLock()
        self.health_check_thread = None
        self.stop_health_check = threading.Event()
        self.connection_callbacks = []
        self.disconnection_callbacks = []
        
        # Start health check thread
        self._start_health_check()
    
    def connect(self) -> bool:
        """
        Connect to the exchange.
        
        Returns:
            True if connected successfully, False otherwise
        """
        with self.connection_lock:
            if self.connected:
                return True
            
            try:
                self.logger.info("Connecting to exchange...")
                
                # Simulate connection (replace with actual connection code)
                # In a real implementation, this would connect to the exchange API
                time.sleep(0.5)
                
                self.connected = True
                self.last_connected_time = datetime.now()
                self.connection_attempts = 0
                
                self.logger.info("Connected to exchange")
                
                # Call connection callbacks
                for callback in self.connection_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        self.logger.error(f"Error in connection callback: {e}")
                
                return True
            except Exception as e:
                self.connected = False
                self.connection_attempts += 1
                self.logger.error(f"Error connecting to exchange: {e}")
                return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from the exchange.
        
        Returns:
            True if disconnected successfully, False otherwise
        """
        with self.connection_lock:
            if not self.connected:
                return True
            
            try:
                self.logger.info("Disconnecting from exchange...")
                
                # Simulate disconnection (replace with actual disconnection code)
                # In a real implementation, this would disconnect from the exchange API
                time.sleep(0.5)
                
                self.connected = False
                
                self.logger.info("Disconnected from exchange")
                
                # Call disconnection callbacks
                for callback in self.disconnection_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        self.logger.error(f"Error in disconnection callback: {e}")
                
                return True
            except Exception as e:
                self.logger.error(f"Error disconnecting from exchange: {e}")
                return False
    
    def reconnect(self, force: bool = False) -> bool:
        """
        Reconnect to the exchange.
        
        Args:
            force: Force reconnection even if already connected
            
        Returns:
            True if reconnected successfully, False otherwise
        """
        with self.connection_lock:
            if self.connected and not force:
                return True
            
            try:
                self.logger.info("Reconnecting to exchange...")
                
                # Disconnect first if connected
                if self.connected:
                    self.disconnect()
                
                # Calculate retry delay with exponential backoff and jitter
                delay = min(self.base_retry_delay * (2 ** self.connection_attempts), self.max_retry_delay)
                jitter = random.uniform(-self.jitter_factor * delay, self.jitter_factor * delay)
                delay += jitter
                
                self.logger.info(f"Waiting {delay:.2f} seconds before reconnecting (attempt {self.connection_attempts + 1})")
                time.sleep(delay)
                
                # Connect
                return self.connect()
            except Exception as e:
                self.logger.error(f"Error reconnecting to exchange: {e}")
                return False
    
    def reset(self) -> bool:
        """
        Reset connection state and reconnect.
        
        Returns:
            True if reset and reconnected successfully, False otherwise
        """
        with self.connection_lock:
            try:
                self.logger.info("Resetting connection...")
                
                # Disconnect first if connected
                if self.connected:
                    self.disconnect()
                
                # Reset connection state
                self.connected = False
                self.connection_attempts = 0
                
                # Reconnect
                return self.connect()
            except Exception as e:
                self.logger.error(f"Error resetting connection: {e}")
                return False
    
    def is_connected(self) -> bool:
        """
        Check if connected to the exchange.
        
        Returns:
            True if connected, False otherwise
        """
        return self.connected
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get connection status information.
        
        Returns:
            Dictionary containing connection status information
        """
        with self.connection_lock:
            status = {
                "connected": self.connected,
                "last_connected_time": self.last_connected_time,
                "connection_attempts": self.connection_attempts,
                "uptime": None
            }
            
            if self.connected and self.last_connected_time:
                status["uptime"] = (datetime.now() - self.last_connected_time).total_seconds()
            
            return status
    
    def test_connection(self) -> bool:
        """
        Test connection to the exchange.
        
        Returns:
            True if connection test passed, False otherwise
        """
        with self.connection_lock:
            try:
                self.logger.info("Testing connection...")
                
                if not self.connected:
                    self.logger.warning("Not connected, attempting to connect...")
                    if not self.connect():
                        return False
                
                # Simulate connection test (replace with actual test code)
                # In a real implementation, this would test the exchange API connection
                time.sleep(0.5)
                
                self.logger.info("Connection test passed")
                return True
            except Exception as e:
                self.logger.error(f"Connection test failed: {e}")
                return False
    
    def _start_health_check(self) -> None:
        """
        Start the health check thread.
        """
        if hasattr(self, 'health_check_thread') and self.health_check_thread is not None and self.health_check_thread.is_alive():
            return
        
        self.stop_health_check.clear()
        self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_check_thread.start()
        
        self.logger.info("Health check thread started")
    
    def _stop_health_check(self) -> None:
        """
        Stop the health check thread.
        """
        if not hasattr(self, 'health_check_thread') or self.health_check_thread is None or not self.health_check_thread.is_alive():
            return
        
        self.stop_health_check.set()
        self.health_check_thread.join(timeout=5.0)
        
        self.logger.info("Health check thread stopped")
    
    def _health_check_loop(self) -> None:
        """
        Health check loop.
        """
        while not self.stop_health_check.is_set():
            try:
                if self.connected:
                    # Perform health check
                    if not self._perform_health_check():
                        self.logger.warning("Health check failed, reconnecting...")
                        self.reconnect(force=True)
                else:
                    # Try to connect if not connected
                    if self.connection_attempts < self.max_attempts or self.max_attempts <= 0:
                        self.logger.info("Not connected, attempting to connect...")
                        self.reconnect()
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
            
            # Wait for next health check
            self.stop_health_check.wait(self.health_check_interval)
    
    def _perform_health_check(self) -> bool:
        """
        Perform health check.
        
        Returns:
            True if health check passed, False otherwise
        """
        try:
            self.logger.debug("Performing health check...")
            
            # Simulate health check (replace with actual health check code)
            # In a real implementation, this would check the exchange API connection
            time.sleep(0.1)
            
            # Check if last connected time is too old
            if self.last_connected_time and (datetime.now() - self.last_connected_time) > timedelta(minutes=30):
                self.logger.warning("Connection is too old, health check failed")
                return False
            
            self.logger.debug("Health check passed")
            return True
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def add_connection_callback(self, callback: Callable[[], None]) -> None:
        """
        Add a callback to be called when connected.
        
        Args:
            callback: Callback function
        """
        self.connection_callbacks.append(callback)
    
    def add_disconnection_callback(self, callback: Callable[[], None]) -> None:
        """
        Add a callback to be called when disconnected.
        
        Args:
            callback: Callback function
        """
        self.disconnection_callbacks.append(callback)
    
    def remove_connection_callback(self, callback: Callable[[], None]) -> None:
        """
        Remove a connection callback.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self.connection_callbacks:
            self.connection_callbacks.remove(callback)
    
    def remove_disconnection_callback(self, callback: Callable[[], None]) -> None:
        """
        Remove a disconnection callback.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self.disconnection_callbacks:
            self.disconnection_callbacks.remove(callback)
    
    def execute_with_connection_guard(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with connection guard.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if not self.connected:
                    self.logger.warning("Not connected, attempting to connect...")
                    if not self.connect():
                        raise ConnectionError("Failed to connect to exchange")
                
                return func(*args, **kwargs)
            except Exception as e:
                retry_count += 1
                self.logger.error(f"Error executing function: {e}")
                
                if retry_count < max_retries:
                    self.logger.info(f"Retrying ({retry_count}/{max_retries})...")
                    self.reconnect(force=True)
                else:
                    raise
    
    def __del__(self) -> None:
        """
        Clean up resources.
        """
        try:
            if hasattr(self, '_stop_health_check'):
                self._stop_health_check()
        except Exception:
            # Ignore errors during cleanup
            pass
