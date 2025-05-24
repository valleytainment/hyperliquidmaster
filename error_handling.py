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
        
    def handle_error(self, error: Union[TradingError, Exception], 
                    context: Dict = None) -> Tuple[bool, Any]:
        """
        Handle an error with appropriate logging and recovery.
        
        Args:
            error: Error to handle
            context: Additional context information
            
        Returns:
            Tuple of (success, result)
        """
        # Convert standard exception to TradingError if needed
        if not isinstance(error, TradingError):
            error = TradingError(
                message=str(error),
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.UNKNOWN,
                recoverable=True,
                context=context
            )
            
        # Update context
        if context:
            error.context.update(context)
            
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
        
    def retry_on_error(self, func: Callable, 
                      args: Tuple = None, 
                      kwargs: Dict = None,
                      max_retries: int = None,
                      retry_exceptions: List[Type[Exception]] = None,
                      context: Dict = None) -> Any:
        """
        Retry a function on error with exponential backoff.
        
        Args:
            func: Function to retry
            args: Function arguments
            kwargs: Function keyword arguments
            max_retries: Maximum number of retries (overrides instance default)
            retry_exceptions: Exceptions to retry on (None for all)
            context: Additional context information
            
        Returns:
            Function result
        """
        args = args or ()
        kwargs = kwargs or {}
        max_retries = max_retries if max_retries is not None else self.max_retries
        context = context or {}
        
        # Add function information to context
        context.update({
            "function": func.__name__,
            "args": str(args),
            "kwargs": str(kwargs)
        })
        
        # Try the function with retries
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Check if we should retry this exception
                if retry_exceptions and not any(isinstance(e, ex) for ex in retry_exceptions):
                    raise
                    
                # Convert to TradingError if needed
                if not isinstance(e, TradingError):
                    error = TradingError(
                        message=str(e),
                        severity=ErrorSeverity.ERROR,
                        category=ErrorCategory.UNKNOWN,
                        recoverable=True,
                        context=context
                    )
                else:
                    error = e
                    
                # Update context with attempt information
                error.context.update({
                    "attempt": attempt + 1,
                    "max_retries": max_retries
                })
                
                # Log the error
                self._log_error(error)
                
                # Update statistics
                self._update_statistics(error)
                
                # Check if we should retry
                if attempt < max_retries and error.recoverable:
                    # Calculate retry delay with exponential backoff and jitter
                    base_delay = error.retry_after * (2 ** attempt)
                    jitter = random.uniform(0, 0.5 * base_delay)
                    delay = base_delay + jitter
                    
                    logger.warning(f"Retry {attempt+1}/{max_retries} after {delay:.2f}s: {error.message}")
                    time.sleep(delay)
                else:
                    # Send notification for severe errors
                    if error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
                        self._send_notification(error)
                        
                    # Re-raise the error
                    raise
                    
    def _log_error(self, error: TradingError) -> None:
        """
        Log an error with appropriate severity.
        
        Args:
            error: Error to log
        """
        # Log to console
        if error.severity == ErrorSeverity.INFO:
            logger.info(str(error))
        elif error.severity == ErrorSeverity.WARNING:
            logger.warning(str(error))
        elif error.severity == ErrorSeverity.ERROR:
            logger.error(str(error))
        elif error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
            logger.critical(str(error))
            
        # Log to error log
        self.error_logger.error(f"{error.severity} {error.category}: {error.message}")
        
        # Log detailed error to file
        try:
            timestamp = error.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{error.category}_{error.severity}.json"
            filepath = os.path.join(self.log_dir, "errors", filename)
            
            with open(filepath, "w") as f:
                json.dump(error.to_dict(), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error logging error to file: {str(e)}")
            
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
            self.error_history.append({
                "timestamp": error.timestamp,
                "category": error.category,
                "severity": error.severity,
                "message": error.message
            })
            
            # Trim history if needed
            if len(self.error_history) > self.max_history:
                self.error_history = self.error_history[-self.max_history:]
                
    def _send_notification(self, error: TradingError) -> None:
        """
        Send notification for an error.
        
        Args:
            error: Error to send notification for
        """
        if self.notification_callback:
            try:
                self.notification_callback(error)
            except Exception as e:
                logger.error(f"Error sending notification: {str(e)}")
                
    def get_statistics(self) -> Dict:
        """
        Get error statistics.
        
        Returns:
            Dictionary with error statistics
        """
        with self.lock:
            return {
                "counts": self.error_counts.copy(),
                "history": self.error_history.copy()
            }
            
    def get_recent_errors(self, 
                         count: int = 10, 
                         severity: str = None, 
                         category: str = None) -> List[Dict]:
        """
        Get recent errors with optional filtering.
        
        Args:
            count: Maximum number of errors to return
            severity: Filter by severity
            category: Filter by category
            
        Returns:
            List of recent errors
        """
        with self.lock:
            # Filter errors
            filtered = self.error_history.copy()
            
            if severity:
                filtered = [e for e in filtered if e["severity"] == severity]
                
            if category:
                filtered = [e for e in filtered if e["category"] == category]
                
            # Return most recent errors
            return filtered[-count:]
            
    def clear_statistics(self) -> None:
        """
        Clear error statistics.
        """
        with self.lock:
            self.error_counts = {}
            self.error_history = []

# Decorator for error handling
def handle_errors(error_handler: ErrorHandler = None, 
                 max_retries: int = None,
                 retry_exceptions: List[Type[Exception]] = None,
                 context: Dict = None):
    """
    Decorator for handling errors in functions.
    
    Args:
        error_handler: Error handler to use
        max_retries: Maximum number of retries
        retry_exceptions: Exceptions to retry on
        context: Additional context information
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create error handler if not provided
            nonlocal error_handler
            if error_handler is None:
                error_handler = ErrorHandler()
                
            # Create context if not provided
            nonlocal context
            context = context or {}
            
            # Add function information to context
            context.update({
                "function": func.__name__
            })
            
            # Retry the function
            return error_handler.retry_on_error(
                func=func,
                args=args,
                kwargs=kwargs,
                max_retries=max_retries,
                retry_exceptions=retry_exceptions,
                context=context
            )
        return wrapper
    return decorator

# Context manager for error handling
class ErrorContext:
    """
    Context manager for handling errors in a block of code.
    """
    
    def __init__(self, 
                error_handler: ErrorHandler = None,
                context: Dict = None,
                reraise: bool = True):
        """
        Initialize the error context.
        
        Args:
            error_handler: Error handler to use
            context: Additional context information
            reraise: Whether to re-raise errors
        """
        self.error_handler = error_handler or ErrorHandler()
        self.context = context or {}
        self.reraise = reraise
        self.error = None
        
    def __enter__(self):
        """
        Enter the context.
        
        Returns:
            Self
        """
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
            
        Returns:
            Whether to suppress the exception
        """
        if exc_val:
            # Handle the error
            success, result = self.error_handler.handle_error(exc_val, self.context)
            self.error = result
            
            # Return True to suppress the exception if not reraising
            return not self.reraise
            
        return True

# Fallback mechanism for graceful degradation
class Fallback:
    """
    Provides fallback mechanisms for graceful degradation.
    """
    
    @staticmethod
    def with_fallback(func: Callable, 
                     fallback_func: Callable,
                     args: Tuple = None, 
                     kwargs: Dict = None,
                     fallback_args: Tuple = None,
                     fallback_kwargs: Dict = None,
                     error_handler: ErrorHandler = None) -> Any:
        """
        Execute a function with fallback on error.
        
        Args:
            func: Primary function to execute
            fallback_func: Fallback function to execute on error
            args: Primary function arguments
            kwargs: Primary function keyword arguments
            fallback_args: Fallback function arguments
            fallback_kwargs: Fallback function keyword arguments
            error_handler: Error handler to use
            
        Returns:
            Function result
        """
        args = args or ()
        kwargs = kwargs or {}
        fallback_args = fallback_args or ()
        fallback_kwargs = fallback_kwargs or {}
        error_handler = error_handler or ErrorHandler()
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log the error
            error_handler.handle_error(e, {
                "function": func.__name__,
                "fallback": fallback_func.__name__
            })
            
            # Execute fallback
            return fallback_func(*fallback_args, **fallback_kwargs)
            
    @staticmethod
    def with_default(func: Callable, 
                    default_value: Any,
                    args: Tuple = None, 
                    kwargs: Dict = None,
                    error_handler: ErrorHandler = None) -> Any:
        """
        Execute a function with default value on error.
        
        Args:
            func: Function to execute
            default_value: Default value to return on error
            args: Function arguments
            kwargs: Function keyword arguments
            error_handler: Error handler to use
            
        Returns:
            Function result or default value
        """
        args = args or ()
        kwargs = kwargs or {}
        error_handler = error_handler or ErrorHandler()
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log the error
            error_handler.handle_error(e, {
                "function": func.__name__,
                "default_value": str(default_value)
            })
            
            # Return default value
            return default_value
            
    @staticmethod
    def with_cache(func: Callable, 
                  cache: Dict,
                  cache_key: Any,
                  args: Tuple = None, 
                  kwargs: Dict = None,
                  error_handler: ErrorHandler = None) -> Any:
        """
        Execute a function with cached value on error.
        
        Args:
            func: Function to execute
            cache: Cache dictionary
            cache_key: Cache key
            args: Function arguments
            kwargs: Function keyword arguments
            error_handler: Error handler to use
            
        Returns:
            Function result or cached value
        """
        args = args or ()
        kwargs = kwargs or {}
        error_handler = error_handler or ErrorHandler()
        
        try:
            result = func(*args, **kwargs)
            
            # Update cache
            cache[cache_key] = result
            
            return result
        except Exception as e:
            # Log the error
            error_handler.handle_error(e, {
                "function": func.__name__,
                "cache_key": str(cache_key)
            })
            
            # Return cached value if available
            if cache_key in cache:
                return cache[cache_key]
                
            # Re-raise if no cached value
            raise

# Global error handler instance
global_error_handler = ErrorHandler()

# Convenience function for handling errors
def handle_error(error: Union[TradingError, Exception], 
                context: Dict = None) -> Tuple[bool, Any]:
    """
    Handle an error with the global error handler.
    
    Args:
        error: Error to handle
        context: Additional context information
        
    Returns:
        Tuple of (success, result)
    """
    return global_error_handler.handle_error(error, context)

# Convenience function for retrying on error
def retry_on_error(func: Callable, 
                  args: Tuple = None, 
                  kwargs: Dict = None,
                  max_retries: int = None,
                  retry_exceptions: List[Type[Exception]] = None,
                  context: Dict = None) -> Any:
    """
    Retry a function on error with the global error handler.
    
    Args:
        func: Function to retry
        args: Function arguments
        kwargs: Function keyword arguments
        max_retries: Maximum number of retries
        retry_exceptions: Exceptions to retry on
        context: Additional context information
        
    Returns:
        Function result
    """
    return global_error_handler.retry_on_error(
        func=func,
        args=args,
        kwargs=kwargs,
        max_retries=max_retries,
        retry_exceptions=retry_exceptions,
        context=context
    )

# Convenience decorator for handling errors
def with_error_handling(max_retries: int = None,
                       retry_exceptions: List[Type[Exception]] = None,
                       context: Dict = None):
    """
    Decorator for handling errors with the global error handler.
    
    Args:
        max_retries: Maximum number of retries
        retry_exceptions: Exceptions to retry on
        context: Additional context information
        
    Returns:
        Decorated function
    """
    return handle_errors(
        error_handler=global_error_handler,
        max_retries=max_retries,
        retry_exceptions=retry_exceptions,
        context=context
    )

# Convenience function for executing with fallback
def with_fallback(func: Callable, 
                 fallback_func: Callable,
                 args: Tuple = None, 
                 kwargs: Dict = None,
                 fallback_args: Tuple = None,
                 fallback_kwargs: Dict = None) -> Any:
    """
    Execute a function with fallback using the global error handler.
    
    Args:
        func: Primary function to execute
        fallback_func: Fallback function to execute on error
        args: Primary function arguments
        kwargs: Primary function keyword arguments
        fallback_args: Fallback function arguments
        fallback_kwargs: Fallback function keyword arguments
        
    Returns:
        Function result
    """
    return Fallback.with_fallback(
        func=func,
        fallback_func=fallback_func,
        args=args,
        kwargs=kwargs,
        fallback_args=fallback_args,
        fallback_kwargs=fallback_kwargs,
        error_handler=global_error_handler
    )

# Convenience function for executing with default value
def with_default(func: Callable, 
                default_value: Any,
                args: Tuple = None, 
                kwargs: Dict = None) -> Any:
    """
    Execute a function with default value using the global error handler.
    
    Args:
        func: Function to execute
        default_value: Default value to return on error
        args: Function arguments
        kwargs: Function keyword arguments
        
    Returns:
        Function result or default value
    """
    return Fallback.with_default(
        func=func,
        default_value=default_value,
        args=args,
        kwargs=kwargs,
        error_handler=global_error_handler
    )

# Convenience function for executing with cache
def with_cache(func: Callable, 
              cache: Dict,
              cache_key: Any,
              args: Tuple = None, 
              kwargs: Dict = None) -> Any:
    """
    Execute a function with cached value using the global error handler.
    
    Args:
        func: Function to execute
        cache: Cache dictionary
        cache_key: Cache key
        args: Function arguments
        kwargs: Function keyword arguments
        
    Returns:
        Function result or cached value
    """
    return Fallback.with_cache(
        func=func,
        cache=cache,
        cache_key=cache_key,
        args=args,
        kwargs=kwargs,
        error_handler=global_error_handler
    )
