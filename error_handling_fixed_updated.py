"""
Enhanced Error Handling System

This module provides a comprehensive error handling system for the Hyperliquid trading bot,
with specialized handlers, automatic recovery mechanisms, detailed logging,
error statistics, graceful degradation, and notification capabilities.
"""

import logging
import time
import traceback
import json
import os
import threading
import functools
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable, Type, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Error severity levels
class ErrorSeverity:
    """Error severity levels for categorizing errors."""
    INFO = "INFO"           # Informational, no impact on trading
    WARNING = "WARNING"     # Potential issue, trading can continue
    ERROR = "ERROR"         # Serious issue, may impact current operation
    CRITICAL = "CRITICAL"   # Critical issue, trading should be paused
    FATAL = "FATAL"         # Fatal issue, trading should be stopped

# Error categories
class ErrorCategory:
    """Error categories for grouping related errors."""
    API = "API"                 # API-related errors
    DATA = "DATA"               # Data-related errors
    CALCULATION = "CALCULATION" # Calculation-related errors
    SIGNAL = "SIGNAL"           # Signal generation errors
    ORDER = "ORDER"             # Order execution errors
    SYSTEM = "SYSTEM"           # System-related errors
    NETWORK = "NETWORK"         # Network-related errors
    UNKNOWN = "UNKNOWN"         # Unknown errors

class TradingError(Exception):
    """
    Base class for all trading-related errors.
    """
    
    def __init__(self, 
                 message: str, 
                 severity: str = ErrorSeverity.ERROR,
                 category: str = ErrorCategory.UNKNOWN,
                 recoverable: bool = True,
                 retry_after: float = 1.0,
                 context: Dict = None):
        """
        Initialize a trading error.
        
        Args:
            message: Error message
            severity: Error severity level
            category: Error category
            recoverable: Whether the error is recoverable
            retry_after: Suggested retry delay in seconds
            context: Additional context information
        """
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.context = context or {}
        self.timestamp = datetime.now()
        self.traceback = traceback.format_exc()
        
    def to_dict(self) -> Dict:
        """
        Convert error to dictionary.
        
        Returns:
            Dictionary representation of error
        """
        return {
            "message": self.message,
            "severity": self.severity,
            "category": self.category,
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "traceback": self.traceback
        }
        
    def __str__(self) -> str:
        """
        String representation of error.
        
        Returns:
            String representation
        """
        return f"{self.severity} {self.category} Error: {self.message}"

# Specialized error classes
class APIError(TradingError):
    """Error related to API calls."""
    
    def __init__(self, 
                 message: str, 
                 severity: str = ErrorSeverity.ERROR,
                 recoverable: bool = True,
                 retry_after: float = 1.0,
                 context: Dict = None,
                 status_code: int = None,
                 endpoint: str = None):
        """
        Initialize an API error.
        
        Args:
            message: Error message
            severity: Error severity level
            recoverable: Whether the error is recoverable
            retry_after: Suggested retry delay in seconds
            context: Additional context information
            status_code: HTTP status code
            endpoint: API endpoint
        """
        context = context or {}
        context.update({
            "status_code": status_code,
            "endpoint": endpoint
        })
        
        # Adjust retry_after based on status code
        if status_code == 429:  # Rate limit
            retry_after = 5.0
            severity = ErrorSeverity.WARNING
        elif status_code and 500 <= status_code < 600:  # Server error
            retry_after = 10.0
            
        super().__init__(
            message=message,
            severity=severity,
            category=ErrorCategory.API,
            recoverable=recoverable,
            retry_after=retry_after,
            context=context
        )

class DataError(TradingError):
    """Error related to data handling."""
    
    def __init__(self, 
                 message: str, 
                 severity: str = ErrorSeverity.ERROR,
                 recoverable: bool = True,
                 retry_after: float = 1.0,
                 context: Dict = None,
                 data_source: str = None,
                 data_type: str = None):
        """
        Initialize a data error.
        
        Args:
            message: Error message
            severity: Error severity level
            recoverable: Whether the error is recoverable
            retry_after: Suggested retry delay in seconds
            context: Additional context information
            data_source: Data source
            data_type: Data type
        """
        context = context or {}
        context.update({
            "data_source": data_source,
            "data_type": data_type
        })
        
        super().__init__(
            message=message,
            severity=severity,
            category=ErrorCategory.DATA,
            recoverable=recoverable,
            retry_after=retry_after,
            context=context
        )

class CalculationError(TradingError):
    """Error related to calculations."""
    
    def __init__(self, 
                 message: str, 
                 severity: str = ErrorSeverity.ERROR,
                 recoverable: bool = True,
                 retry_after: float = 1.0,
                 context: Dict = None,
                 calculation_type: str = None):
        """
        Initialize a calculation error.
        
        Args:
            message: Error message
            severity: Error severity level
            recoverable: Whether the error is recoverable
            retry_after: Suggested retry delay in seconds
            context: Additional context information
            calculation_type: Calculation type
        """
        context = context or {}
        context.update({
            "calculation_type": calculation_type
        })
        
        super().__init__(
            message=message,
            severity=severity,
            category=ErrorCategory.CALCULATION,
            recoverable=recoverable,
            retry_after=retry_after,
            context=context
        )

class SignalError(TradingError):
    """Error related to signal generation."""
    
    def __init__(self, 
                 message: str, 
                 severity: str = ErrorSeverity.ERROR,
                 recoverable: bool = True,
                 retry_after: float = 1.0,
                 context: Dict = None,
                 signal_type: str = None,
                 strategy: str = None):
        """
        Initialize a signal error.
        
        Args:
            message: Error message
            severity: Error severity level
            recoverable: Whether the error is recoverable
            retry_after: Suggested retry delay in seconds
            context: Additional context information
            signal_type: Signal type
            strategy: Strategy name
        """
        context = context or {}
        context.update({
            "signal_type": signal_type,
            "strategy": strategy
        })
        
        super().__init__(
            message=message,
            severity=severity,
            category=ErrorCategory.SIGNAL,
            recoverable=recoverable,
            retry_after=retry_after,
            context=context
        )

class OrderError(TradingError):
    """Error related to order execution."""
    
    def __init__(self, 
                 message: str, 
                 severity: str = ErrorSeverity.ERROR,
                 recoverable: bool = True,
                 retry_after: float = 1.0,
                 context: Dict = None,
                 order_id: str = None,
                 order_type: str = None,
                 symbol: str = None):
        """
        Initialize an order error.
        
        Args:
            message: Error message
            severity: Error severity level
            recoverable: Whether the error is recoverable
            retry_after: Suggested retry delay in seconds
            context: Additional context information
            order_id: Order ID
            order_type: Order type
            symbol: Trading symbol
        """
        context = context or {}
        context.update({
            "order_id": order_id,
            "order_type": order_type,
            "symbol": symbol
        })
        
        super().__init__(
            message=message,
            severity=severity,
            category=ErrorCategory.ORDER,
            recoverable=recoverable,
            retry_after=retry_after,
            context=context
        )

class SystemError(TradingError):
    """Error related to system operations."""
    
    def __init__(self, 
                 message: str, 
                 severity: str = ErrorSeverity.ERROR,
                 recoverable: bool = True,
                 retry_after: float = 1.0,
                 context: Dict = None,
                 component: str = None):
        """
        Initialize a system error.
        
        Args:
            message: Error message
            severity: Error severity level
            recoverable: Whether the error is recoverable
            retry_after: Suggested retry delay in seconds
            context: Additional context information
            component: System component
        """
        context = context or {}
        context.update({
            "component": component
        })
        
        super().__init__(
            message=message,
            severity=severity,
            category=ErrorCategory.SYSTEM,
            recoverable=recoverable,
            retry_after=retry_after,
            context=context
        )

class NetworkError(TradingError):
    """Error related to network operations."""
    
    def __init__(self, 
                 message: str, 
                 severity: str = ErrorSeverity.ERROR,
                 recoverable: bool = True,
                 retry_after: float = 1.0,
                 context: Dict = None,
                 host: str = None):
        """
        Initialize a network error.
        
        Args:
            message: Error message
            severity: Error severity level
            recoverable: Whether the error is recoverable
            retry_after: Suggested retry delay in seconds
            context: Additional context information
            host: Network host
        """
        context = context or {}
        context.update({
            "host": host
        })
        
        super().__init__(
            message=message,
            severity=severity,
            category=ErrorCategory.NETWORK,
            recoverable=recoverable,
            retry_after=retry_after,
            context=context
        )

class ErrorHandler:
    """
    Handles trading errors with automatic recovery, logging, and statistics.
    """
    
    def __init__(self, 
                 log_dir: str = "logs",
                 max_retries: int = 3,
                 notification_callback: Callable = None):
        """
        Initialize the error handler.
        
        Args:
            log_dir: Directory to store error logs
            max_retries: Maximum number of retries for recoverable errors
            notification_callback: Callback function for error notifications
        """
        self.log_dir = log_dir
        self.max_retries = max_retries
        self.notification_callback = notification_callback
        
        # Error statistics
        self.error_counts = {}  # Format: {category: count}
        self.error_history = []  # List of recent errors
        self.max_history = 100  # Maximum number of errors to keep in history
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "errors"), exist_ok=True)
        
        # Configure file handler for errors
        self.error_logger = logging.getLogger("error_handler")
        self.error_logger.setLevel(logging.INFO)
        
        error_handler = logging.FileHandler(os.path.join(log_dir, "errors.log"))
        error_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.error_logger.addHandler(error_handler)
        
    def handle_error(self, source: str, error_message: str, 
                    context: Dict = None, severity: str = ErrorSeverity.ERROR) -> Tuple[bool, Any]:
        """
        Handle an error with appropriate logging and recovery.
        
        Args:
            source: Source of the error (function or component name)
            error_message: Error message or exception string
            context: Additional context information
            severity: Error severity level
            
        Returns:
            Tuple of (success, result)
        """
        # Create a TradingError object
        error = TradingError(
            message=f"{source}: {error_message}",
            severity=severity,
            category=ErrorCategory.UNKNOWN,
            recoverable=True,
            context=context or {}
        )
            
        # Log the error
        self._log_error(error)
        
        # Update statistics
        self._update_statistics(error)
        
        # Send notification for severe errors
        if error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
            self._send_notification(error)
            
        # Return failure for non-recoverable errors
        if not error.recoverable:
            return False, error
            
        # Return failure for fatal errors
        if error.severity == ErrorSeverity.FATAL:
            return False, error
            
        # Return success for informational errors
        if error.severity == ErrorSeverity.INFO:
            return True, None
            
        # Return default values for other errors
        return False, error
    
    def log_error(self, error_type: str, message: str, context: Dict = None) -> None:
        """
        Public method to log an error without handling it.
        
        Args:
            error_type: Type of error (API, DATA, etc.)
            message: Error message
            context: Additional context information
        """
        # Map error type to category
        category = getattr(ErrorCategory, error_type, ErrorCategory.UNKNOWN) if hasattr(ErrorCategory, error_type) else ErrorCategory.UNKNOWN
        
        # Create error object
        error = TradingError(
            message=message,
            severity=ErrorSeverity.ERROR,
            category=category,
            recoverable=True,
            context=context or {}
        )
        
        # Log the error
        self._log_error(error)
        
        # Update statistics
        self._update_statistics(error)
    
    def _log_error(self, error: TradingError) -> None:
        """
        Log an error to console and file.
        
        Args:
            error: Error to log
        """
        # Log to console
        log_method = getattr(logger, error.severity.lower(), logger.error)
        log_method(str(error))
        
        # Log to error log
        self.error_logger.error(str(error))
        
        # Log detailed error to file
        try:
            timestamp = error.timestamp.strftime("%Y%m%d_%H%M%S")
            error_file = os.path.join(self.log_dir, "errors", f"{timestamp}_{error.category}.json")
            
            with open(error_file, "w") as f:
                json.dump(error.to_dict(), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to log error to file: {str(e)}")
    
    def _update_statistics(self, error: TradingError) -> None:
        """
        Update error statistics.
        
        Args:
            error: Error to update statistics for
        """
        with self.lock:
            # Update error counts
            if error.category not in self.error_counts:
                self.error_counts[error.category] = 0
            self.error_counts[error.category] += 1
            
            # Update error history
            self.error_history.append(error)
            
            # Trim history if needed
            if len(self.error_history) > self.max_history:
                self.error_history = self.error_history[-self.max_history:]
    
    def _send_notification(self, error: TradingError) -> None:
        """
        Send notification for severe errors.
        
        Args:
            error: Error to send notification for
        """
        if self.notification_callback:
            try:
                self.notification_callback(error)
            except Exception as e:
                logger.error(f"Failed to send error notification: {str(e)}")
    
    def get_error_statistics(self) -> Dict:
        """
        Get error statistics.
        
        Returns:
            Dictionary with error statistics
        """
        with self.lock:
            return {
                "counts": self.error_counts.copy(),
                "total": sum(self.error_counts.values()),
                "recent": [e.to_dict() for e in self.error_history[-10:]]
            }
    
    def reset_statistics(self) -> None:
        """
        Reset error statistics.
        """
        with self.lock:
            self.error_counts = {}
            self.error_history = []
    
    def retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """
        Retry a function with exponential backoff.
        
        Args:
            func: Function to retry
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        max_retries = kwargs.pop("max_retries", self.max_retries)
        base_delay = kwargs.pop("base_delay", 1.0)
        max_delay = kwargs.pop("max_delay", 60.0)
        
        retries = 0
        last_exception = None
        
        while retries <= max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                retries += 1
                
                if retries > max_retries:
                    break
                
                # Calculate delay with jitter
                delay = min(base_delay * (2 ** (retries - 1)) + random.uniform(0, 1), max_delay)
                
                logger.warning(f"Error executing {func.__name__} (attempt {retries}/{max_retries}): {str(e)}")
                logger.info(f"Backing off for {delay:.2f}s before retry")
                
                time.sleep(delay)
        
        # If we get here, all retries failed
        logger.error(f"Failed to execute {func.__name__} after {max_retries} retries: {str(last_exception)}")
        raise last_exception
    
    def with_error_handling(self, func: Callable) -> Callable:
        """
        Decorator to add error handling to a function.
        
        Args:
            func: Function to decorate
            
        Returns:
            Decorated function
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get function name
                func_name = func.__name__
                
                # Log error
                self.handle_error(func_name, str(e))
                
                # Re-raise exception
                raise
        
        return wrapper
    
    def with_retry(self, func: Callable) -> Callable:
        """
        Decorator to add retry with backoff to a function.
        
        Args:
            func: Function to decorate
            
        Returns:
            Decorated function
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.retry_with_backoff(func, *args, **kwargs)
        
        return wrapper
    
    def with_circuit_breaker(self, func: Callable, failure_threshold: int = 3, 
                            reset_timeout: int = 60) -> Callable:
        """
        Decorator to add circuit breaker to a function.
        
        Args:
            func: Function to decorate
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Timeout in seconds before resetting circuit
            
        Returns:
            Decorated function
        """
        # Circuit state
        state = {
            "failures": 0,
            "open": False,
            "last_failure": None
        }
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if circuit is open
            if state["open"]:
                # Check if reset timeout has passed
                if state["last_failure"] and (datetime.now() - state["last_failure"]).total_seconds() > reset_timeout:
                    # Reset circuit
                    state["failures"] = 0
                    state["open"] = False
                    logger.info(f"Circuit breaker reset for {func.__name__}")
                else:
                    # Circuit is still open
                    logger.warning(f"Circuit breaker open for {func.__name__}, request blocked")
                    raise SystemError(f"Service temporarily unavailable (circuit breaker open)")
            
            try:
                # Call function
                result = func(*args, **kwargs)
                
                # Reset failures on success
                state["failures"] = 0
                
                return result
            except Exception as e:
                # Increment failures
                state["failures"] += 1
                state["last_failure"] = datetime.now()
                
                # Check if circuit should be opened
                if state["failures"] >= failure_threshold:
                    state["open"] = True
                    logger.warning(f"Circuit breaker opened for {func.__name__} after {state['failures']} failures")
                
                # Re-raise exception
                raise
        
        return wrapper
