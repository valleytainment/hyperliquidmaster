"""
Enhanced Error Handling System

This module provides a robust error handling system for the Hyperliquid trading bot.
It includes logging, context-aware error handling, and recovery mechanisms.

Classes:
    ErrorHandler: Manages error handling, logging, and recovery
"""

import os
import logging
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/error_handler.log")
    ]
)
logger = logging.getLogger(__name__)

class ErrorHandler:
    """
    Enhanced error handler with context-aware error handling and recovery.
    
    This class provides methods for handling errors, logging them with context,
    and implementing appropriate recovery mechanisms based on error type.
    
    Attributes:
        error_counts (dict): Counter for each type of error
        max_retries (dict): Maximum retry attempts for each operation
        recovery_strategies (dict): Recovery strategies for different error types
    """
    
    def __init__(self):
        """
        Initialize the error handler.
        """
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        os.makedirs("logs/errors", exist_ok=True)
        
        # Initialize error tracking
        self.error_counts = {}
        
        # Define maximum retry attempts for different operations
        self.max_retries = {
            "api_call": 3,
            "data_processing": 2,
            "signal_generation": 2,
            "order_execution": 1
        }
        
        # Define recovery strategies for different error types
        self.recovery_strategies = {
            "ConnectionError": "retry",
            "TimeoutError": "retry",
            "RateLimitError": "cooldown",
            "DataError": "use_fallback",
            "ValidationError": "skip",
            "AuthenticationError": "alert",
            "InternalError": "alert"
        }
        
        logger.info("Error handler initialized")
    
    def handle_error(self, function_name, error_message, error_type=None, context=None, operation=None):
        """
        Handle an error with context and implement recovery if possible.
        
        Args:
            function_name (str): Name of the function where the error occurred
            error_message (str): Error message
            error_type (str, optional): Type of error
            context (dict, optional): Additional context for the error
            operation (str, optional): Type of operation being performed
        
        Returns:
            dict: Result of error handling with recovery strategy
        """
        # Ensure context is a dictionary
        if context is None:
            context = {}
        
        # Determine error type if not provided
        if error_type is None:
            if "timeout" in error_message.lower():
                error_type = "TimeoutError"
            elif "connection" in error_message.lower():
                error_type = "ConnectionError"
            elif "rate limit" in error_message.lower():
                error_type = "RateLimitError"
            elif "data" in error_message.lower():
                error_type = "DataError"
            elif "validation" in error_message.lower():
                error_type = "ValidationError"
            elif "authentication" in error_message.lower() or "auth" in error_message.lower():
                error_type = "AuthenticationError"
            else:
                error_type = "InternalError"
        
        # Determine operation type if not provided
        if operation is None:
            if "api" in function_name.lower() or "request" in function_name.lower():
                operation = "api_call"
            elif "data" in function_name.lower() or "process" in function_name.lower():
                operation = "data_processing"
            elif "signal" in function_name.lower() or "strategy" in function_name.lower():
                operation = "signal_generation"
            elif "order" in function_name.lower() or "trade" in function_name.lower():
                operation = "order_execution"
            else:
                operation = "general"
        
        # Update error count
        error_key = f"{error_type}:{function_name}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log the error with context
        log_message = f"Error in {function_name}: {error_message}"
        if context:
            log_message += f" | Context: {context}"
        
        logger.error(log_message)
        
        # Log detailed error to file
        self._log_error_to_file(function_name, error_message, error_type, context, operation)
        
        # Determine recovery strategy
        recovery_strategy = self.recovery_strategies.get(error_type, "alert")
        retry_count = self.error_counts.get(error_key, 1)
        max_retries = self.max_retries.get(operation, 1)
        
        # Implement recovery strategy
        result = {
            "error_type": error_type,
            "function_name": function_name,
            "recovery_strategy": recovery_strategy,
            "retry_count": retry_count,
            "max_retries": max_retries,
            "should_retry": False
        }
        
        if recovery_strategy == "retry" and retry_count <= max_retries:
            logger.info(f"Retrying {function_name} (attempt {retry_count}/{max_retries})")
            result["should_retry"] = True
        elif recovery_strategy == "cooldown":
            cooldown_minutes = min(5 * retry_count, 60)  # Exponential backoff, max 60 minutes
            logger.info(f"Implementing cooldown for {cooldown_minutes} minutes")
            result["cooldown_minutes"] = cooldown_minutes
        elif recovery_strategy == "use_fallback":
            logger.info("Using fallback data source")
            result["use_fallback"] = True
        elif recovery_strategy == "skip":
            logger.info("Skipping operation")
            result["skip"] = True
        elif recovery_strategy == "alert":
            logger.warning("Critical error requires attention")
            result["alert"] = True
        
        return result
    
    def _log_error_to_file(self, function_name, error_message, error_type, context, operation):
        """
        Log detailed error information to a file.
        
        Args:
            function_name (str): Name of the function where the error occurred
            error_message (str): Error message
            error_type (str): Type of error
            context (dict): Additional context for the error
            operation (str): Type of operation being performed
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/errors/{timestamp}_{error_type}_{function_name}.log"
        
        with open(filename, "w") as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Function: {function_name}\n")
            f.write(f"Error Type: {error_type}\n")
            f.write(f"Operation: {operation}\n")
            f.write(f"Error Message: {error_message}\n")
            f.write(f"Context: {context}\n")
            f.write(f"Error Count: {self.error_counts.get(f'{error_type}:{function_name}', 1)}\n")
            f.write(f"Stack Trace:\n{traceback.format_exc()}\n")
    
    def reset_error_counts(self):
        """
        Reset all error counts.
        """
        self.error_counts = {}
        logger.info("Reset all error counts")
    
    def get_error_summary(self):
        """
        Get a summary of all errors.
        
        Returns:
            dict: Summary of all errors with counts
        """
        return {
            "error_counts": self.error_counts,
            "total_errors": sum(self.error_counts.values()),
            "unique_errors": len(self.error_counts)
        }
    
    def log_warning(self, message, context=None):
        """
        Log a warning message with optional context.
        
        Args:
            message (str): Warning message
            context (dict, optional): Additional context for the warning
        """
        if context:
            logger.warning(f"{message} | Context: {context}")
        else:
            logger.warning(message)
    
    def log_info(self, message, context=None):
        """
        Log an info message with optional context.
        
        Args:
            message (str): Info message
            context (dict, optional): Additional context for the info
        """
        if context:
            logger.info(f"{message} | Context: {context}")
        else:
            logger.info(message)
