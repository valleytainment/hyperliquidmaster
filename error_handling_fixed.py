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
                    context: Dict = None, message: str = None) -> Tuple[bool, Any]:
        """
        Handle an error with appropriate logging and recovery.
        
        Args:
            error: Error to handle
            context: Additional context information
            message: Optional additional message to include in logs
            
        Returns:
            Tuple of (success, result)
        """
        # Convert standard exception to TradingError if needed
        if not isinstance(error, TradingError):
            error_message = message if message else str(error)
            error = TradingError(
                message=error_message,
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
            context=context
        )
        
        # Log the error
        self._log_error(error)
        
        # Update statistics
        self._update_statistics(error)
        
    def _log_error(self, error: TradingError) -> None:
        """
        Log an error to file and console.
        
        Args:
            error: Error to log
        """
        with self.lock:
            # Log to console
            log_level = logging.INFO
            if error.severity == ErrorSeverity.WARNING:
                log_level = logging.WARNING
            elif error.severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
                log_level = logging.ERROR
                
            logger.log(log_level, f"{error}")
            
            # Log to error log
            self.error_logger.error(f"{error}")
            
            # Log detailed error to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_file = os.path.join(self.log_dir, "errors", f"{error.category}_{timestamp}.json")
            
            with open(error_file, "w") as f:
                json.dump(error.to_dict(), f, indent=2, default=str)
                
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
        Send notification for severe errors.
        
        Args:
            error: Error to send notification for
        """
        if self.notification_callback:
            try:
                self.notification_callback(error)
            except Exception as e:
                logger.error(f"Error sending notification: {str(e)}")
                
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
            Function result or raises last error
        """
        args = args or ()
        kwargs = kwargs or {}
        max_retries = max_retries if max_retries is not None else self.max_retries
        
        retries = 0
        last_error = None
        
        while retries <= max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Check if we should retry this exception
                if retry_exceptions and not any(isinstance(e, exc) for exc in retry_exceptions):
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
                    
                # Don't retry non-recoverable errors
                if not error.recoverable:
                    self.handle_error(error, context)
                    raise
                    
                # Log the error
                self.handle_error(error, context)
                
                # Check if we've reached max retries
                if retries >= max_retries:
                    last_error = error
                    break
                    
                # Calculate backoff time
                backoff = error.retry_after * (2 ** retries) + random.uniform(0, 1)
                
                logger.info(f"Retrying in {backoff:.2f} seconds (attempt {retries + 1}/{max_retries})")
                time.sleep(backoff)
                
                retries += 1
                
        # If we get here, we've exhausted all retries
        if last_error:
            raise last_error
        else:
            raise RuntimeError("Exhausted all retries with unknown error")
            
    def retry_decorator(self, max_retries: int = None,
                       retry_exceptions: List[Type[Exception]] = None):
        """
        Decorator to retry a function on error.
        
        Args:
            max_retries: Maximum number of retries (overrides instance default)
            retry_exceptions: Exceptions to retry on (None for all)
            
        Returns:
            Decorated function
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self.retry_on_error(
                    func=func,
                    args=args,
                    kwargs=kwargs,
                    max_retries=max_retries,
                    retry_exceptions=retry_exceptions,
                    context={"function": func.__name__}
                )
            return wrapper
        return decorator
        
    def get_error_statistics(self) -> Dict:
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
            
    def clear_error_statistics(self) -> None:
        """
        Clear error statistics.
        """
        with self.lock:
            self.error_counts = {}
            self.error_history = []
            
    def get_error_rate(self, category: str = None, 
                      time_window: timedelta = None) -> float:
        """
        Get error rate for a category within a time window.
        
        Args:
            category: Error category (None for all)
            time_window: Time window (None for all time)
            
        Returns:
            Error rate (errors per hour)
        """
        with self.lock:
            if not self.error_history:
                return 0.0
                
            # Filter by category if specified
            filtered_history = self.error_history
            if category:
                filtered_history = [e for e in filtered_history if e["category"] == category]
                
            if not filtered_history:
                return 0.0
                
            # Filter by time window if specified
            if time_window:
                cutoff = datetime.now() - time_window
                filtered_history = [e for e in filtered_history if e["timestamp"] >= cutoff]
                
            if not filtered_history:
                return 0.0
                
            # Calculate time span
            start_time = min(e["timestamp"] for e in filtered_history)
            end_time = max(e["timestamp"] for e in filtered_history)
            
            time_span = (end_time - start_time).total_seconds() / 3600  # Convert to hours
            
            if time_span <= 0:
                return 0.0
                
            # Calculate error rate
            return len(filtered_history) / time_span
