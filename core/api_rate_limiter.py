"""
API Rate Limiter with Enhanced Cooldown Persistence

This module provides a rate limiter for the Hyperliquid API with enhanced
cooldown persistence across bot restarts. It automatically switches to
mock data mode when rate limits are reached.

Features:
- Persistent cooldown state across bot restarts
- Automatic mock data mode activation during API rate limits
- Configurable rate limits and cooldown periods
- Detailed logging of rate limit events
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any

# Configure logging
logger = logging.getLogger(__name__)

class APIRateLimiter:
    """
    API Rate Limiter with enhanced cooldown persistence.
    
    Manages API request rates and enforces cooldown periods when
    rate limits are reached. Persists cooldown state across bot
    restarts to ensure continuous protection.
    """
    
    def __init__(self, 
                 max_requests_per_minute: int = 60,
                 max_requests_per_hour: int = 1000,
                 cooldown_minutes: int = 60,
                 state_file: Optional[str] = None):
        """
        Initialize the API rate limiter.
        
        Args:
            max_requests_per_minute: Maximum requests allowed per minute
            max_requests_per_hour: Maximum requests allowed per hour
            cooldown_minutes: Cooldown period in minutes when rate limit is reached
            state_file: File to store rate limiter state. If None, uses default.
        """
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_hour = max_requests_per_hour
        self.cooldown_minutes = cooldown_minutes
        
        # Set state file
        if state_file is None:
            self.state_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config",
                "rate_limiter_state.json"
            )
        else:
            self.state_file = state_file
        
        # Initialize state
        self.minute_requests = []
        self.hour_requests = []
        self.in_cooldown = False
        self.cooldown_until = 0
        
        # Load state
        self._load_state()
        
        logger.info(f"API Rate Limiter initialized with {max_requests_per_minute} req/min, "
                   f"{max_requests_per_hour} req/hour, {cooldown_minutes} min cooldown")
    
    def _load_state(self) -> None:
        """Load rate limiter state from file."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                
                self.minute_requests = state.get("minute_requests", [])
                self.hour_requests = state.get("hour_requests", [])
                self.in_cooldown = state.get("in_cooldown", False)
                self.cooldown_until = state.get("cooldown_until", 0)
                
                # Filter out old requests
                current_time = time.time()
                self.minute_requests = [t for t in self.minute_requests 
                                       if current_time - t < 60]
                self.hour_requests = [t for t in self.hour_requests 
                                     if current_time - t < 3600]
                
                # Check if still in cooldown
                if self.in_cooldown and current_time > self.cooldown_until:
                    logger.info("Cooldown period has expired")
                    self.in_cooldown = False
                    self.cooldown_until = 0
                elif self.in_cooldown:
                    cooldown_remaining = timedelta(seconds=self.cooldown_until - current_time)
                    logger.warning(f"Still in cooldown period. Remaining time: {cooldown_remaining}")
                
                logger.debug("Rate limiter state loaded from file")
        except Exception as e:
            logger.error(f"Error loading rate limiter state: {str(e)}")
            # Initialize with empty state
            self.minute_requests = []
            self.hour_requests = []
            self.in_cooldown = False
            self.cooldown_until = 0
    
    def _save_state(self) -> None:
        """Save rate limiter state to file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            
            # Prepare state
            state = {
                "minute_requests": self.minute_requests,
                "hour_requests": self.hour_requests,
                "in_cooldown": self.in_cooldown,
                "cooldown_until": self.cooldown_until
            }
            
            # Save state
            with open(self.state_file, "w") as f:
                json.dump(state, f)
            
            logger.debug("Rate limiter state saved to file")
        except Exception as e:
            logger.error(f"Error saving rate limiter state: {str(e)}")
    
    def check_rate_limit(self) -> Tuple[bool, str]:
        """
        Check if the API rate limit has been reached.
        
        Returns:
            Tuple of (allowed, reason):
                - allowed: True if request is allowed, False if rate limited
                - reason: Reason for rate limiting, empty if allowed
        """
        current_time = time.time()
        
        # Check if in cooldown
        if self.in_cooldown:
            if current_time > self.cooldown_until:
                # Cooldown period has expired
                logger.info("Cooldown period has expired")
                self.in_cooldown = False
                self.cooldown_until = 0
                self._save_state()
            else:
                # Still in cooldown
                cooldown_remaining = timedelta(seconds=self.cooldown_until - current_time)
                return False, f"In cooldown period. Remaining time: {cooldown_remaining}"
        
        # Filter out old requests
        self.minute_requests = [t for t in self.minute_requests 
                               if current_time - t < 60]
        self.hour_requests = [t for t in self.hour_requests 
                             if current_time - t < 3600]
        
        # Check minute limit
        if len(self.minute_requests) >= self.max_requests_per_minute:
            logger.warning(f"Minute rate limit reached: {len(self.minute_requests)} requests in the last minute")
            return False, "Minute rate limit reached"
        
        # Check hour limit
        if len(self.hour_requests) >= self.max_requests_per_hour:
            logger.warning(f"Hour rate limit reached: {len(self.hour_requests)} requests in the last hour")
            
            # Enter cooldown
            self.in_cooldown = True
            self.cooldown_until = current_time + (self.cooldown_minutes * 60)
            self._save_state()
            
            cooldown_until_time = datetime.fromtimestamp(self.cooldown_until).strftime('%Y-%m-%d %H:%M:%S')
            logger.warning(f"Entering cooldown period until {cooldown_until_time}")
            
            return False, f"Hour rate limit reached. Entering cooldown until {cooldown_until_time}"
        
        # Request is allowed
        return True, ""
    
    def add_request(self) -> None:
        """Record an API request."""
        current_time = time.time()
        
        # Add request timestamps
        self.minute_requests.append(current_time)
        self.hour_requests.append(current_time)
        
        # Save state
        self._save_state()
        
        logger.debug(f"Request recorded. Minute: {len(self.minute_requests)}/{self.max_requests_per_minute}, "
                    f"Hour: {len(self.hour_requests)}/{self.max_requests_per_hour}")
    
    def record_call(self, endpoint: str) -> None:
        """
        Record an API call for a specific endpoint.
        This is an alias for add_request to maintain compatibility with test suite.
        
        Args:
            endpoint: API endpoint being called
        """
        logger.debug(f"Recording API call to endpoint: {endpoint}")
        self.add_request()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the rate limiter.
        
        Returns:
            Dictionary with rate limiter status
        """
        current_time = time.time()
        
        # Filter out old requests
        self.minute_requests = [t for t in self.minute_requests 
                               if current_time - t < 60]
        self.hour_requests = [t for t in self.hour_requests 
                             if current_time - t < 3600]
        
        # Check if still in cooldown
        if self.in_cooldown and current_time > self.cooldown_until:
            self.in_cooldown = False
            self.cooldown_until = 0
            self._save_state()
        
        # Calculate cooldown remaining
        cooldown_remaining = max(0, self.cooldown_until - current_time) if self.in_cooldown else 0
        
        return {
            "minute_requests": len(self.minute_requests),
            "hour_requests": len(self.hour_requests),
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_requests_per_hour": self.max_requests_per_hour,
            "in_cooldown": self.in_cooldown,
            "cooldown_remaining_seconds": cooldown_remaining,
            "cooldown_remaining_formatted": str(timedelta(seconds=cooldown_remaining)) if cooldown_remaining > 0 else "0:00:00",
            "is_limited": self.in_cooldown,  # Added for compatibility with tests
            "cooldown_remaining": cooldown_remaining  # Added for compatibility with tests
        }
    
    def reset(self) -> None:
        """Reset the rate limiter state."""
        self.minute_requests = []
        self.hour_requests = []
        self.in_cooldown = False
        self.cooldown_until = 0
        self._save_state()
        
        logger.info("Rate limiter state reset")
