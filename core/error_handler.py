"""
Error Handler Module

This module provides centralized error handling, logging, and recovery strategies
for the enhanced trading bot.
"""

import logging
import traceback
import time
from typing import Dict, Any, Optional, Callable

class ErrorHandler:
    """
    Centralized error handler for the trading bot.
    Provides error logging, categorization, and recovery strategies.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the error handler.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts = {}
        self.error_timestamps = {}
        self.recovery_strategies = self._setup_recovery_strategies()
        
    def _setup_recovery_strategies(self) -> Dict[str, Callable]:
        """
        Set up recovery strategies for different error types.
        
        Returns:
            Dictionary mapping error categories to recovery functions
        """
        return {
            "exchange_init": self._handle_exchange_init_error,
            "connection_verification": self._handle_connection_error,
            "fetch_market_data": self._handle_data_error,
            "fetch_order_book": self._handle_data_error,
            "get_user_positions": self._handle_data_error,
            "place_market_order": self._handle_order_error,
            "close_position": self._handle_order_error,
            "get_order_status": self._handle_data_error,
            "retry_failed": self._handle_retry_error,
            "sentiment_analysis": self._handle_sentiment_error,
            "default": self._handle_default_error
        }
        
    def handle_error(self, category: str, message: str, exception: Optional[Exception] = None) -> Dict[str, Any]:
        """
        Handle an error.
        
        Args:
            category: Error category
            message: Error message
            exception: Exception object
            
        Returns:
            Dictionary with error handling results
        """
        # Log the error
        if exception:
            self.logger.error(f"[{category}] {message}: {str(exception)}")
            self.logger.debug(f"[{category}] Traceback: {traceback.format_exc()}")
        else:
            self.logger.error(f"[{category}] {message}")
            
        # Update error counts
        self.error_counts[category] = self.error_counts.get(category, 0) + 1
        self.error_timestamps[category] = time.time()
        
        # Apply recovery strategy
        recovery_func = self.recovery_strategies.get(category, self.recovery_strategies["default"])
        recovery_result = recovery_func(category, message, exception)
        
        return {
            "category": category,
            "message": message,
            "exception": str(exception) if exception else None,
            "count": self.error_counts[category],
            "recovery_action": recovery_result.get("action"),
            "recovery_success": recovery_result.get("success", False)
        }
        
    def _handle_exchange_init_error(self, category: str, message: str, exception: Optional[Exception] = None) -> Dict[str, Any]:
        """
        Handle exchange initialization errors.
        
        Args:
            category: Error category
            message: Error message
            exception: Exception object
            
        Returns:
            Recovery result
        """
        count = self.error_counts.get(category, 0)
        
        if count <= 3:
            return {
                "action": "retry_init",
                "success": False,
                "retry_delay": 5 * count
            }
        else:
            return {
                "action": "abort",
                "success": False,
                "message": "Exchange initialization failed after multiple attempts"
            }
            
    def _handle_connection_error(self, category: str, message: str, exception: Optional[Exception] = None) -> Dict[str, Any]:
        """
        Handle connection errors.
        
        Args:
            category: Error category
            message: Error message
            exception: Exception object
            
        Returns:
            Recovery result
        """
        count = self.error_counts.get(category, 0)
        
        if count <= 5:
            return {
                "action": "retry_connection",
                "success": False,
                "retry_delay": 3 * count
            }
        else:
            return {
                "action": "reconnect",
                "success": False,
                "message": "Connection verification failed after multiple attempts"
            }
            
    def _handle_data_error(self, category: str, message: str, exception: Optional[Exception] = None) -> Dict[str, Any]:
        """
        Handle data retrieval errors.
        
        Args:
            category: Error category
            message: Error message
            exception: Exception object
            
        Returns:
            Recovery result
        """
        count = self.error_counts.get(category, 0)
        
        if "rate limit" in message.lower() or (exception and "rate limit" in str(exception).lower()):
            return {
                "action": "rate_limit_backoff",
                "success": False,
                "retry_delay": 10
            }
        elif count <= 10:
            return {
                "action": "retry",
                "success": False,
                "retry_delay": 2 * count
            }
        else:
            # Reset count after 10 errors to allow future retries
            self.error_counts[category] = 0
            return {
                "action": "use_cached_data",
                "success": False,
                "message": "Data retrieval failed after multiple attempts"
            }
            
    def _handle_order_error(self, category: str, message: str, exception: Optional[Exception] = None) -> Dict[str, Any]:
        """
        Handle order execution errors.
        
        Args:
            category: Error category
            message: Error message
            exception: Exception object
            
        Returns:
            Recovery result
        """
        count = self.error_counts.get(category, 0)
        
        if "insufficient balance" in message.lower() or (exception and "insufficient balance" in str(exception).lower()):
            return {
                "action": "reduce_order_size",
                "success": False,
                "reduction_factor": 0.5
            }
        elif "invalid size" in message.lower() or (exception and "invalid size" in str(exception).lower()):
            return {
                "action": "adjust_order_size",
                "success": False,
                "message": "Order size invalid"
            }
        elif count <= 3:
            return {
                "action": "retry_order",
                "success": False,
                "retry_delay": 2
            }
        else:
            return {
                "action": "abort_order",
                "success": False,
                "message": "Order execution failed after multiple attempts"
            }
            
    def _handle_retry_error(self, category: str, message: str, exception: Optional[Exception] = None) -> Dict[str, Any]:
        """
        Handle retry failures.
        
        Args:
            category: Error category
            message: Error message
            exception: Exception object
            
        Returns:
            Recovery result
        """
        return {
            "action": "fallback",
            "success": False,
            "message": "All retries failed"
        }
        
    def _handle_sentiment_error(self, category: str, message: str, exception: Optional[Exception] = None) -> Dict[str, Any]:
        """
        Handle sentiment analysis errors.
        
        Args:
            category: Error category
            message: Error message
            exception: Exception object
            
        Returns:
            Recovery result
        """
        return {
            "action": "skip_sentiment",
            "success": True,
            "message": "Skipping sentiment analysis due to error"
        }
        
    def _handle_default_error(self, category: str, message: str, exception: Optional[Exception] = None) -> Dict[str, Any]:
        """
        Handle uncategorized errors.
        
        Args:
            category: Error category
            message: Error message
            exception: Exception object
            
        Returns:
            Recovery result
        """
        return {
            "action": "log_only",
            "success": False,
            "message": "Uncategorized error logged"
        }
        
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics.
        
        Returns:
            Dictionary with error statistics
        """
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_counts": self.error_counts.copy(),
            "last_errors": {category: time.time() - timestamp for category, timestamp in self.error_timestamps.items()}
        }
        
    def reset_error_counts(self, category: Optional[str] = None):
        """
        Reset error counts.
        
        Args:
            category: Optional category to reset (None for all)
        """
        if category:
            if category in self.error_counts:
                self.error_counts[category] = 0
                self.error_timestamps[category] = 0
        else:
            self.error_counts = {}
            self.error_timestamps = {}
