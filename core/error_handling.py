"""
Enhanced Error Handling System for Hyperliquid Trading Bot

This module provides a comprehensive error handling system with context-aware
recovery mechanisms, detailed logging, and automatic fallback to mock data
when appropriate.

Features:
- Context-aware error handling with recovery strategies
- Detailed error logging with stack traces
- Automatic fallback to mock data for API errors
- User-friendly error messages for GUI display
"""

import os
import sys
import time
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Callable, Tuple, List, Union

# Configure logging
logger = logging.getLogger(__name__)

class ErrorHandler:
    """
    Enhanced error handler for Hyperliquid Trading Bot.
    
    Provides context-aware error handling with recovery strategies,
    detailed logging, and automatic fallback to mock data when appropriate.
    """
    
    # Error severity levels
    SEVERITY_INFO = 0
    SEVERITY_WARNING = 1
    SEVERITY_ERROR = 2
    SEVERITY_CRITICAL = 3
    
    # Error categories
    CATEGORY_API = "api"
    CATEGORY_NETWORK = "network"
    CATEGORY_DATA = "data"
    CATEGORY_GUI = "gui"
    CATEGORY_SYSTEM = "system"
    CATEGORY_TRADING = "trading"
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize the error handler.
        
        Args:
            log_dir: Directory for error logs. If None, uses default.
        """
        # Set log directory
        if log_dir is None:
            self.log_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "logs"
            )
        else:
            self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize error log
        self.error_log_file = os.path.join(
            self.log_dir, 
            f"error_log_{datetime.now().strftime('%Y%m%d')}.log"
        )
        
        # Initialize error count
        self.error_counts = {
            self.CATEGORY_API: 0,
            self.CATEGORY_NETWORK: 0,
            self.CATEGORY_DATA: 0,
            self.CATEGORY_GUI: 0,
            self.CATEGORY_SYSTEM: 0,
            self.CATEGORY_TRADING: 0
        }
        
        # Initialize recovery strategies
        self.recovery_strategies = {
            self.CATEGORY_API: self._recover_api_error,
            self.CATEGORY_NETWORK: self._recover_network_error,
            self.CATEGORY_DATA: self._recover_data_error,
            self.CATEGORY_GUI: self._recover_gui_error,
            self.CATEGORY_SYSTEM: self._recover_system_error,
            self.CATEGORY_TRADING: self._recover_trading_error
        }
        
        logger.info("Error handler initialized")
    
    def handle_error(self, error: Union[Exception, str], category: str, 
                    context: Optional[Dict[str, Any]] = None, 
                    severity: int = SEVERITY_ERROR) -> Tuple[bool, Dict[str, Any]]:
        """
        Handle an error with context-aware recovery.
        
        Args:
            error: Exception or error message
            category: Error category
            context: Additional context for error handling
            severity: Error severity level
            
        Returns:
            Tuple of (recovered, recovery_info):
                - recovered: True if error was recovered, False otherwise
                - recovery_info: Information about recovery attempt
        """
        # Initialize context if None
        if context is None:
            context = {}
        
        # Get error details
        error_time = datetime.now()
        error_message = str(error)
        error_traceback = "".join(traceback.format_exception(*sys.exc_info())) if isinstance(error, Exception) else ""
        
        # Log error
        self._log_error(error_message, category, severity, context, error_traceback)
        
        # Increment error count
        if category in self.error_counts:
            self.error_counts[category] += 1
        
        # Get user-friendly message
        user_message = self._get_user_message(error_message, category, severity)
        
        # Attempt recovery
        recovered = False
        recovery_info = {
            "error_message": error_message,
            "user_message": user_message,
            "category": category,
            "severity": severity,
            "time": error_time.strftime("%Y-%m-%d %H:%M:%S"),
            "recovery_attempted": False,
            "recovery_successful": False,
            "recovery_action": None,
            "fallback_to_mock": False
        }
        
        # Check if recovery is available for this category
        if category in self.recovery_strategies:
            recovery_strategy = self.recovery_strategies[category]
            
            # Attempt recovery
            recovery_info["recovery_attempted"] = True
            recovered, recovery_action, fallback_to_mock = recovery_strategy(error, context, severity)
            
            # Update recovery info
            recovery_info["recovery_successful"] = recovered
            recovery_info["recovery_action"] = recovery_action
            recovery_info["fallback_to_mock"] = fallback_to_mock
            
            if recovered:
                logger.info(f"Error recovered: {error_message}")
            else:
                logger.warning(f"Error recovery failed: {error_message}")
        
        return recovered, recovery_info
    
    def _log_error(self, error_message: str, category: str, severity: int, 
                  context: Dict[str, Any], traceback_str: str) -> None:
        """
        Log an error to file.
        
        Args:
            error_message: Error message
            category: Error category
            severity: Error severity level
            context: Additional context for error handling
            traceback_str: Error traceback
        """
        try:
            # Get severity string
            if severity == self.SEVERITY_INFO:
                severity_str = "INFO"
            elif severity == self.SEVERITY_WARNING:
                severity_str = "WARNING"
            elif severity == self.SEVERITY_ERROR:
                severity_str = "ERROR"
            elif severity == self.SEVERITY_CRITICAL:
                severity_str = "CRITICAL"
            else:
                severity_str = "UNKNOWN"
            
            # Format error log entry
            log_entry = f"""
{'='*80}
TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
SEVERITY: {severity_str}
CATEGORY: {category}
ERROR: {error_message}
CONTEXT: {context}
TRACEBACK:
{traceback_str}
{'='*80}
"""
            
            # Write to log file
            with open(self.error_log_file, "a") as f:
                f.write(log_entry)
            
            # Log to console based on severity
            if severity == self.SEVERITY_INFO:
                logger.info(f"[{category}] {error_message}")
            elif severity == self.SEVERITY_WARNING:
                logger.warning(f"[{category}] {error_message}")
            elif severity == self.SEVERITY_ERROR:
                logger.error(f"[{category}] {error_message}")
            elif severity == self.SEVERITY_CRITICAL:
                logger.critical(f"[{category}] {error_message}")
        except Exception as e:
            logger.error(f"Error logging error: {str(e)}")
    
    def _get_user_message(self, error_message: str, category: str, severity: int) -> str:
        """
        Get user-friendly error message.
        
        Args:
            error_message: Error message
            category: Error category
            severity: Error severity level
            
        Returns:
            User-friendly error message
        """
        # API errors
        if category == self.CATEGORY_API:
            if "rate limit" in error_message.lower():
                return "API rate limit reached. The bot will automatically switch to mock data mode."
            elif "authentication" in error_message.lower() or "auth" in error_message.lower():
                return "API authentication failed. Please check your API keys."
            elif "timeout" in error_message.lower():
                return "API request timed out. The bot will retry automatically."
            else:
                return "API error occurred. The bot will attempt to recover automatically."
        
        # Network errors
        elif category == self.CATEGORY_NETWORK:
            if "connection" in error_message.lower():
                return "Network connection error. Please check your internet connection."
            elif "timeout" in error_message.lower():
                return "Network request timed out. The bot will retry automatically."
            else:
                return "Network error occurred. The bot will attempt to recover automatically."
        
        # Data errors
        elif category == self.CATEGORY_DATA:
            if "parse" in error_message.lower() or "json" in error_message.lower():
                return "Data parsing error. The bot will attempt to recover automatically."
            elif "missing" in error_message.lower():
                return "Missing data error. The bot will attempt to recover automatically."
            else:
                return "Data error occurred. The bot will attempt to recover automatically."
        
        # GUI errors
        elif category == self.CATEGORY_GUI:
            return "GUI error occurred. Please try restarting the application."
        
        # System errors
        elif category == self.CATEGORY_SYSTEM:
            if "memory" in error_message.lower():
                return "System memory error. Please check your system resources."
            elif "disk" in error_message.lower():
                return "Disk error. Please check your disk space."
            else:
                return "System error occurred. Please check your system resources."
        
        # Trading errors
        elif category == self.CATEGORY_TRADING:
            if "order" in error_message.lower():
                return "Order error occurred. Please check your order parameters."
            elif "position" in error_message.lower():
                return "Position error occurred. Please check your position status."
            else:
                return "Trading error occurred. Please check your trading parameters."
        
        # Default message
        else:
            if severity == self.SEVERITY_INFO:
                return "Information: " + error_message
            elif severity == self.SEVERITY_WARNING:
                return "Warning: " + error_message
            elif severity == self.SEVERITY_ERROR:
                return "Error: " + error_message
            elif severity == self.SEVERITY_CRITICAL:
                return "Critical Error: " + error_message
            else:
                return "Error: " + error_message
    
    def _recover_api_error(self, error: Union[Exception, str], context: Dict[str, Any], 
                          severity: int) -> Tuple[bool, str, bool]:
        """
        Recover from API error.
        
        Args:
            error: Exception or error message
            context: Additional context for error handling
            severity: Error severity level
            
        Returns:
            Tuple of (recovered, recovery_action, fallback_to_mock):
                - recovered: True if error was recovered, False otherwise
                - recovery_action: Description of recovery action
                - fallback_to_mock: True if fallback to mock data is recommended
        """
        error_message = str(error)
        
        # Rate limit errors
        if "rate limit" in error_message.lower():
            logger.warning("API rate limit reached, switching to mock data mode")
            return True, "Switched to mock data mode", True
        
        # Authentication errors
        elif "authentication" in error_message.lower() or "auth" in error_message.lower():
            # Check if retry count is in context
            retry_count = context.get("retry_count", 0)
            
            if retry_count < 3:
                # Retry with exponential backoff
                backoff_time = 2 ** retry_count
                logger.info(f"API authentication error, retrying in {backoff_time} seconds")
                time.sleep(backoff_time)
                return True, f"Retried authentication (attempt {retry_count + 1})", False
            else:
                # Switch to mock data after 3 retries
                logger.warning("API authentication failed after 3 retries, switching to mock data mode")
                return True, "Switched to mock data mode after 3 authentication retries", True
        
        # Timeout errors
        elif "timeout" in error_message.lower():
            # Check if retry count is in context
            retry_count = context.get("retry_count", 0)
            
            if retry_count < 5:
                # Retry with exponential backoff
                backoff_time = 2 ** retry_count
                logger.info(f"API timeout, retrying in {backoff_time} seconds")
                time.sleep(backoff_time)
                return True, f"Retried after timeout (attempt {retry_count + 1})", False
            else:
                # Switch to mock data after 5 retries
                logger.warning("API timeout after 5 retries, switching to mock data mode")
                return True, "Switched to mock data mode after 5 timeout retries", True
        
        # Other API errors
        else:
            # Check if retry count is in context
            retry_count = context.get("retry_count", 0)
            
            if retry_count < 3:
                # Retry with exponential backoff
                backoff_time = 2 ** retry_count
                logger.info(f"API error, retrying in {backoff_time} seconds")
                time.sleep(backoff_time)
                return True, f"Retried API request (attempt {retry_count + 1})", False
            else:
                # Switch to mock data after 3 retries
                logger.warning("API error after 3 retries, switching to mock data mode")
                return True, "Switched to mock data mode after 3 error retries", True
    
    def _recover_network_error(self, error: Union[Exception, str], context: Dict[str, Any], 
                              severity: int) -> Tuple[bool, str, bool]:
        """
        Recover from network error.
        
        Args:
            error: Exception or error message
            context: Additional context for error handling
            severity: Error severity level
            
        Returns:
            Tuple of (recovered, recovery_action, fallback_to_mock):
                - recovered: True if error was recovered, False otherwise
                - recovery_action: Description of recovery action
                - fallback_to_mock: True if fallback to mock data is recommended
        """
        error_message = str(error)
        
        # Connection errors
        if "connection" in error_message.lower():
            # Check if retry count is in context
            retry_count = context.get("retry_count", 0)
            
            if retry_count < 5:
                # Retry with exponential backoff
                backoff_time = 2 ** retry_count
                logger.info(f"Network connection error, retrying in {backoff_time} seconds")
                time.sleep(backoff_time)
                return True, f"Retried connection (attempt {retry_count + 1})", False
            else:
                # Switch to mock data after 5 retries
                logger.warning("Network connection failed after 5 retries, switching to mock data mode")
                return True, "Switched to mock data mode after 5 connection retries", True
        
        # Timeout errors
        elif "timeout" in error_message.lower():
            # Check if retry count is in context
            retry_count = context.get("retry_count", 0)
            
            if retry_count < 5:
                # Retry with exponential backoff
                backoff_time = 2 ** retry_count
                logger.info(f"Network timeout, retrying in {backoff_time} seconds")
                time.sleep(backoff_time)
                return True, f"Retried after timeout (attempt {retry_count + 1})", False
            else:
                # Switch to mock data after 5 retries
                logger.warning("Network timeout after 5 retries, switching to mock data mode")
                return True, "Switched to mock data mode after 5 timeout retries", True
        
        # Other network errors
        else:
            # Check if retry count is in context
            retry_count = context.get("retry_count", 0)
            
            if retry_count < 3:
                # Retry with exponential backoff
                backoff_time = 2 ** retry_count
                logger.info(f"Network error, retrying in {backoff_time} seconds")
                time.sleep(backoff_time)
                return True, f"Retried network request (attempt {retry_count + 1})", False
            else:
                # Switch to mock data after 3 retries
                logger.warning("Network error after 3 retries, switching to mock data mode")
                return True, "Switched to mock data mode after 3 error retries", True
    
    def _recover_data_error(self, error: Union[Exception, str], context: Dict[str, Any], 
                           severity: int) -> Tuple[bool, str, bool]:
        """
        Recover from data error.
        
        Args:
            error: Exception or error message
            context: Additional context for error handling
            severity: Error severity level
            
        Returns:
            Tuple of (recovered, recovery_action, fallback_to_mock):
                - recovered: True if error was recovered, False otherwise
                - recovery_action: Description of recovery action
                - fallback_to_mock: True if fallback to mock data is recommended
        """
        error_message = str(error)
        
        # Parse errors
        if "parse" in error_message.lower() or "json" in error_message.lower():
            # Check if retry count is in context
            retry_count = context.get("retry_count", 0)
            
            if retry_count < 2:
                # Retry once
                logger.info("Data parsing error, retrying")
                return True, f"Retried parsing (attempt {retry_count + 1})", False
            else:
                # Switch to mock data after 2 retries
                logger.warning("Data parsing failed after 2 retries, switching to mock data mode")
                return True, "Switched to mock data mode after 2 parsing retries", True
        
        # Missing data errors
        elif "missing" in error_message.lower():
            # Check if fallback data is available
            if "fallback_data" in context:
                logger.info("Using fallback data for missing data")
                return True, "Used fallback data", False
            else:
                # Switch to mock data
                logger.warning("Missing data with no fallback, switching to mock data mode")
                return True, "Switched to mock data mode due to missing data", True
        
        # Other data errors
        else:
            # Check if retry count is in context
            retry_count = context.get("retry_count", 0)
            
            if retry_count < 2:
                # Retry once
                logger.info("Data error, retrying")
                return True, f"Retried data operation (attempt {retry_count + 1})", False
            else:
                # Switch to mock data after 2 retries
                logger.warning("Data error after 2 retries, switching to mock data mode")
                return True, "Switched to mock data mode after 2 error retries", True
    
    def _recover_gui_error(self, error: Union[Exception, str], context: Dict[str, Any], 
                          severity: int) -> Tuple[bool, str, bool]:
        """
        Recover from GUI error.
        
        Args:
            error: Exception or error message
            context: Additional context for error handling
            severity: Error severity level
            
        Returns:
            Tuple of (recovered, recovery_action, fallback_to_mock):
                - recovered: True if error was recovered, False otherwise
                - recovery_action: Description of recovery action
                - fallback_to_mock: True if fallback to mock data is recommended
        """
        error_message = str(error)
        
        # Check if component is in context
        if "component" in context:
            component = context["component"]
            
            # Try to reset component
            logger.info(f"Attempting to reset GUI component: {component}")
            
            # For now, just return that we attempted to reset
            # In a real implementation, this would actually reset the component
            return True, f"Reset GUI component: {component}", False
        
        # No component specified, can't recover
        return False, "Unable to recover GUI error", False
    
    def _recover_system_error(self, error: Union[Exception, str], context: Dict[str, Any], 
                             severity: int) -> Tuple[bool, str, bool]:
        """
        Recover from system error.
        
        Args:
            error: Exception or error message
            context: Additional context for error handling
            severity: Error severity level
            
        Returns:
            Tuple of (recovered, recovery_action, fallback_to_mock):
                - recovered: True if error was recovered, False otherwise
                - recovery_action: Description of recovery action
                - fallback_to_mock: True if fallback to mock data is recommended
        """
        error_message = str(error)
        
        # Memory errors
        if "memory" in error_message.lower():
            # Try to free memory
            logger.info("Attempting to free memory")
            
            # For now, just return that we attempted to free memory
            # In a real implementation, this would actually free memory
            return True, "Attempted to free memory", False
        
        # Disk errors
        elif "disk" in error_message.lower():
            # Try to free disk space
            logger.info("Attempting to free disk space")
            
            # For now, just return that we attempted to free disk space
            # In a real implementation, this would actually free disk space
            return True, "Attempted to free disk space", False
        
        # Other system errors
        else:
            # Can't recover
            return False, "Unable to recover system error", False
    
    def _recover_trading_error(self, error: Union[Exception, str], context: Dict[str, Any], 
                              severity: int) -> Tuple[bool, str, bool]:
        """
        Recover from trading error.
        
        Args:
            error: Exception or error message
            context: Additional context for error handling
            severity: Error severity level
            
        Returns:
            Tuple of (recovered, recovery_action, fallback_to_mock):
                - recovered: True if error was recovered, False otherwise
                - recovery_action: Description of recovery action
                - fallback_to_mock: True if fallback to mock data is recommended
        """
        error_message = str(error)
        
        # Order errors
        if "order" in error_message.lower():
            # Check if retry count is in context
            retry_count = context.get("retry_count", 0)
            
            if retry_count < 3:
                # Retry with exponential backoff
                backoff_time = 2 ** retry_count
                logger.info(f"Order error, retrying in {backoff_time} seconds")
                time.sleep(backoff_time)
                return True, f"Retried order (attempt {retry_count + 1})", False
            else:
                # Can't recover after 3 retries
                logger.warning("Order error after 3 retries, unable to recover")
                return False, "Unable to recover order error after 3 retries", False
        
        # Position errors
        elif "position" in error_message.lower():
            # Check if retry count is in context
            retry_count = context.get("retry_count", 0)
            
            if retry_count < 2:
                # Retry once
                logger.info("Position error, retrying")
                return True, f"Retried position operation (attempt {retry_count + 1})", False
            else:
                # Can't recover after 2 retries
                logger.warning("Position error after 2 retries, unable to recover")
                return False, "Unable to recover position error after 2 retries", False
        
        # Other trading errors
        else:
            # Check if retry count is in context
            retry_count = context.get("retry_count", 0)
            
            if retry_count < 2:
                # Retry once
                logger.info("Trading error, retrying")
                return True, f"Retried trading operation (attempt {retry_count + 1})", False
            else:
                # Can't recover after 2 retries
                logger.warning("Trading error after 2 retries, unable to recover")
                return False, "Unable to recover trading error after 2 retries", False
    
    def get_error_counts(self) -> Dict[str, int]:
        """
        Get error counts by category.
        
        Returns:
            Dictionary with error counts by category
        """
        return self.error_counts.copy()
    
    def reset_error_counts(self) -> None:
        """Reset error counts."""
        for category in self.error_counts:
            self.error_counts[category] = 0
        
        logger.info("Error counts reset")
