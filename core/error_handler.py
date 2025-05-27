"""
Error handler module for the HyperliquidMaster application.
Provides robust error handling and logging for all operations.
"""

import logging
import traceback
from typing import Dict, Any, Optional

class ErrorHandler:
    """
    Handles errors and exceptions in the application.
    Provides consistent error handling, logging, and recovery strategies.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the error handler.
        
        Args:
            logger: Logger instance for error logging
        """
        self.logger = logger
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle an error or exception.
        
        Args:
            error: The exception to handle
            context: Additional context information about the error
            
        Returns:
            Dict containing error handling results and recommendations
        """
        if context is None:
            context = {}
        
        # Log the error with context
        self.logger.error(f"{type(error).__name__}: {str(error)} in context: {context}")
        
        # Determine error type and severity
        error_type = type(error).__name__
        error_message = str(error)
        severity = "LOW"
        action = "log_and_notify"
        retry_recommended = False
        retry_delay = 0
        
        # Handle specific error types
        if isinstance(error, ConnectionError) or "Connection" in error_type:
            severity = "MEDIUM"
            action = "retry_with_backoff"
            retry_recommended = True
            retry_delay = 5
            self.logger.warning(f"Connection error occurred: {error_message}")
        
        elif isinstance(error, TimeoutError) or "Timeout" in error_type:
            severity = "MEDIUM"
            action = "retry_with_backoff"
            retry_recommended = True
            retry_delay = 3
            self.logger.warning(f"Timeout error occurred: {error_message}")
        
        elif isinstance(error, ValueError) or isinstance(error, TypeError):
            severity = "LOW"
            action = "log_and_notify"
            self.logger.error(f"Unhandled error occurred: {error_message}")
        
        elif isinstance(error, KeyError) or isinstance(error, IndexError):
            severity = "LOW"
            action = "log_and_notify"
            self.logger.error(f"Data access error occurred: {error_message}")
        
        elif "API" in error_type or "Http" in error_type:
            severity = "HIGH"
            action = "retry_with_backoff"
            retry_recommended = True
            retry_delay = 10
            self.logger.error(f"API error occurred: {error_message}")
        
        else:
            # Generic error handling
            self.logger.error(f"Unhandled error occurred: {error_message}")
        
        # Return error handling results
        return {
            "error_type": error_type,
            "error_message": error_message,
            "severity": severity,
            "handled": True,
            "action": action,
            "retry_recommended": retry_recommended,
            "retry_delay": retry_delay
        }
    
    def log_exception(self, message: str, exc_info: bool = True) -> None:
        """
        Log an exception with traceback.
        
        Args:
            message: Message to log
            exc_info: Whether to include exception info
        """
        self.logger.error(message, exc_info=exc_info)
    
    def format_exception(self, e: Exception) -> str:
        """
        Format an exception for display.
        
        Args:
            e: The exception to format
            
        Returns:
            Formatted exception string
        """
        return f"{type(e).__name__}: {str(e)}"
    
    def get_traceback(self, e: Exception) -> str:
        """
        Get the traceback for an exception.
        
        Args:
            e: The exception to get traceback for
            
        Returns:
            Traceback string
        """
        return ''.join(traceback.format_exception(type(e), e, e.__traceback__))
