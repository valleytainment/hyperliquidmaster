"""
Error Handler module for HyperliquidMaster.

This module provides comprehensive error handling and logging
for the HyperliquidMaster trading bot.
"""

import logging
import traceback
import time
import json
import os
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime

class ErrorHandler:
    """
    Handles errors and exceptions in the HyperliquidMaster trading bot.
    
    This class provides functionality for logging, reporting, and recovering
    from errors and exceptions.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the error handler.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.error_log_file = "error_log.json"
        self.error_count = 0
        self.error_history = []
        self.max_error_history = 100
        self.recovery_handlers = {}
        
        # Load error history if available
        self._load_error_history()
        
        self.logger.info("Error handler initialized")
    
    def _load_error_history(self) -> None:
        """Load error history from file."""
        try:
            if os.path.exists(self.error_log_file):
                with open(self.error_log_file, 'r') as f:
                    self.error_history = json.load(f)
                    
                    # Limit history size
                    if len(self.error_history) > self.max_error_history:
                        self.error_history = self.error_history[-self.max_error_history:]
                    
                    self.error_count = len(self.error_history)
                    
                    self.logger.info(f"Loaded {self.error_count} errors from history")
        except Exception as e:
            self.logger.error(f"Error loading error history: {e}")
            self.error_history = []
            self.error_count = 0
    
    def _save_error_history(self) -> None:
        """Save error history to file."""
        try:
            with open(self.error_log_file, 'w') as f:
                json.dump(self.error_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving error history: {e}")
    
    def register_recovery_handler(self, error_type: str, handler: Callable) -> None:
        """
        Register a recovery handler for a specific error type.
        
        Args:
            error_type: Type of error to handle
            handler: Function to call for recovery
        """
        self.recovery_handlers[error_type] = handler
        self.logger.info(f"Registered recovery handler for {error_type}")
    
    def handle_error(self, error: Exception, context: str = "", data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle an error.
        
        Args:
            error: The exception to handle
            context: Context in which the error occurred
            data: Additional data related to the error
            
        Returns:
            Dict containing error information
        """
        try:
            # Increment error count
            self.error_count += 1
            
            # Get error details
            error_type = type(error).__name__
            error_message = str(error)
            error_traceback = traceback.format_exc()
            
            # Create error record
            error_record = {
                "id": self.error_count,
                "timestamp": datetime.now().isoformat(),
                "type": error_type,
                "message": error_message,
                "traceback": error_traceback,
                "context": context,
                "data": data or {}
            }
            
            # Log error
            self.logger.error(f"Error in {context}: {error_type}: {error_message}")
            if error_traceback:
                self.logger.debug(f"Traceback: {error_traceback}")
            
            # Add to history
            self.error_history.append(error_record)
            
            # Limit history size
            if len(self.error_history) > self.max_error_history:
                self.error_history.pop(0)
            
            # Save error history
            self._save_error_history()
            
            # Attempt recovery if handler exists
            recovery_result = None
            if error_type in self.recovery_handlers:
                try:
                    self.logger.info(f"Attempting recovery for {error_type}")
                    recovery_handler = self.recovery_handlers[error_type]
                    recovery_result = recovery_handler(error, context, data)
                    
                    if recovery_result:
                        self.logger.info(f"Recovery successful for {error_type}")
                        error_record["recovery"] = {
                            "success": True,
                            "result": recovery_result
                        }
                    else:
                        self.logger.warning(f"Recovery failed for {error_type}")
                        error_record["recovery"] = {
                            "success": False
                        }
                except Exception as recovery_error:
                    self.logger.error(f"Error in recovery handler: {recovery_error}")
                    error_record["recovery"] = {
                        "success": False,
                        "error": str(recovery_error)
                    }
            
            return error_record
        except Exception as e:
            self.logger.error(f"Error in error handler: {e}")
            return {
                "id": -1,
                "timestamp": datetime.now().isoformat(),
                "type": "ErrorHandlerError",
                "message": f"Error in error handler: {e}",
                "context": "error_handler"
            }
    
    def get_error_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get error history.
        
        Args:
            limit: Maximum number of errors to return
            
        Returns:
            List of error records
        """
        return self.error_history[-limit:] if limit > 0 else self.error_history
    
    def clear_error_history(self) -> bool:
        """
        Clear error history.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.error_history = []
            self.error_count = 0
            self._save_error_history()
            self.logger.info("Error history cleared")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing error history: {e}")
            return False
    
    def format_error_for_display(self, error_record: Dict[str, Any]) -> str:
        """
        Format an error record for display.
        
        Args:
            error_record: Error record to format
            
        Returns:
            Formatted error string
        """
        try:
            timestamp = datetime.fromisoformat(error_record["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            
            formatted = f"Error #{error_record['id']} ({timestamp}):\n"
            formatted += f"Type: {error_record['type']}\n"
            formatted += f"Context: {error_record['context']}\n"
            formatted += f"Message: {error_record['message']}\n"
            
            if "recovery" in error_record:
                recovery = error_record["recovery"]
                if recovery.get("success", False):
                    formatted += "Recovery: Successful\n"
                else:
                    formatted += "Recovery: Failed\n"
                    if "error" in recovery:
                        formatted += f"Recovery Error: {recovery['error']}\n"
            
            return formatted
        except Exception as e:
            self.logger.error(f"Error formatting error record: {e}")
            return f"Error formatting error record: {e}"
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of errors.
        
        Returns:
            Dict containing error summary
        """
        try:
            # Count errors by type
            error_types = {}
            for error in self.error_history:
                error_type = error["type"]
                if error_type in error_types:
                    error_types[error_type] += 1
                else:
                    error_types[error_type] = 1
            
            # Count errors by context
            error_contexts = {}
            for error in self.error_history:
                context = error["context"]
                if context in error_contexts:
                    error_contexts[context] += 1
                else:
                    error_contexts[context] = 1
            
            # Get recent errors
            recent_errors = self.get_error_history(5)
            
            return {
                "total_errors": self.error_count,
                "error_types": error_types,
                "error_contexts": error_contexts,
                "recent_errors": recent_errors
            }
        except Exception as e:
            self.logger.error(f"Error getting error summary: {e}")
            return {
                "error": f"Error getting error summary: {e}"
            }
