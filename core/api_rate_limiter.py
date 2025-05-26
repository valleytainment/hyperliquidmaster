"""
API Rate Limiter with Enhanced Cooldown Management

This module provides a robust API rate limiter with persistent cooldown state
that survives across bot restarts. It automatically activates mock data mode
during API rate limit periods.

Classes:
    APIRateLimiter: Manages API call rates and enforces cooldown periods
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/api_rate_limiter.log")
    ]
)
logger = logging.getLogger(__name__)

class APIRateLimiter:
    """
    Enhanced API rate limiter with persistent cooldown state.
    
    This class manages API call rates and enforces cooldown periods when
    rate limits are exceeded. The cooldown state persists across bot restarts
    by saving to a state file.
    
    Attributes:
        limits (dict): Maximum calls allowed per endpoint per minute
        calls (dict): Current call counts per endpoint
        cooldowns (dict): Active cooldown periods per endpoint
        state_file (str): Path to the state file for persistence
        mock_mode_active (bool): Whether mock data mode is active
    """
    
    def __init__(self, state_file="rate_limiter_state.json"):
        """
        Initialize the API rate limiter.
        
        Args:
            state_file (str): Path to the state file for persistence
        """
        self.limits = {
            "market_data": 60,  # 60 calls per minute
            "klines": 30,       # 30 calls per minute
            "order_book": 20,   # 20 calls per minute
            "trades": 10,       # 10 calls per minute
            "positions": 5      # 5 calls per minute
        }
        
        self.calls = {endpoint: 0 for endpoint in self.limits}
        self.cooldowns = {}
        self.state_file = state_file
        self.mock_mode_active = False
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Load state from file if it exists
        self._load_state()
        
        # Check if any cooldowns are active
        self._check_cooldowns()
        
        logger.info("API rate limiter initialized with mock data mode")
    
    def _load_state(self):
        """
        Load rate limiter state from file.
        """
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                
                # Convert string timestamps to datetime objects
                cooldowns = {}
                for endpoint, end_time_str in state.get("cooldowns", {}).items():
                    cooldowns[endpoint] = datetime.fromisoformat(end_time_str)
                
                self.cooldowns = cooldowns
                self.mock_mode_active = state.get("mock_mode_active", False)
                
                logger.info(f"Loaded rate limiter state from {self.state_file}")
            except Exception as e:
                logger.error(f"Error loading rate limiter state: {e}")
    
    def _save_state(self):
        """
        Save rate limiter state to file.
        """
        try:
            # Convert datetime objects to ISO format strings
            cooldowns = {}
            for endpoint, end_time in self.cooldowns.items():
                cooldowns[endpoint] = end_time.isoformat()
            
            state = {
                "cooldowns": cooldowns,
                "mock_mode_active": self.mock_mode_active
            }
            
            with open(self.state_file, "w") as f:
                json.dump(state, f)
            
            logger.info(f"Saved rate limiter state to {self.state_file}")
        except Exception as e:
            logger.error(f"Error saving rate limiter state: {e}")
    
    def _check_cooldowns(self):
        """
        Check if any cooldowns are active and update mock mode.
        """
        now = datetime.now()
        active_cooldowns = False
        
        # Check each cooldown
        for endpoint, end_time in list(self.cooldowns.items()):
            if now < end_time:
                # Cooldown is still active
                active_cooldowns = True
                remaining = (end_time - now).total_seconds()
                logger.info(f"Cooldown for {endpoint} is active for {remaining:.1f} more seconds")
            else:
                # Cooldown has expired
                logger.info(f"Cleared cooldown for {endpoint}")
                del self.cooldowns[endpoint]
        
        # Update mock mode based on cooldowns
        if active_cooldowns and not self.mock_mode_active:
            logger.info("Activating mock data mode due to active cooldowns")
            self.mock_mode_active = True
            self._save_state()
        elif not active_cooldowns and self.mock_mode_active:
            logger.info("Deactivating mock data mode as all cooldowns have expired")
            self.mock_mode_active = False
            self._save_state()
    
    def record_call(self, endpoint):
        """
        Record an API call to the specified endpoint.
        
        Args:
            endpoint (str): The API endpoint being called
        """
        if endpoint not in self.limits:
            logger.warning(f"Unknown endpoint: {endpoint}")
            return
        
        # Check if endpoint is in cooldown
        if endpoint in self.cooldowns:
            now = datetime.now()
            if now < self.cooldowns[endpoint]:
                # Endpoint is in cooldown
                remaining = (self.cooldowns[endpoint] - now).total_seconds()
                logger.warning(f"Endpoint {endpoint} is in cooldown for {remaining:.1f} more seconds")
                return
            else:
                # Cooldown has expired
                logger.info(f"Cleared cooldown for {endpoint}")
                del self.cooldowns[endpoint]
                self._save_state()
        
        # Record the call
        self.calls[endpoint] += 1
        
        # Check if rate limit is exceeded
        if self.calls[endpoint] > self.limits[endpoint]:
            # Set cooldown for 1 minute
            cooldown_minutes = 1
            end_time = datetime.now() + timedelta(minutes=cooldown_minutes)
            self.cooldowns[endpoint] = end_time
            
            logger.warning(f"Setting cooldown for {endpoint} until {end_time} ({cooldown_minutes} minutes)")
            
            # Activate mock data mode
            self.mock_mode_active = True
            
            # Save state
            self._save_state()
    
    def set_cooldown(self, endpoint, minutes):
        """
        Manually set a cooldown for the specified endpoint.
        
        Args:
            endpoint (str): The API endpoint to set cooldown for
            minutes (int): Cooldown duration in minutes
        """
        if endpoint not in self.limits:
            logger.warning(f"Unknown endpoint: {endpoint}")
            return
        
        end_time = datetime.now() + timedelta(minutes=minutes)
        self.cooldowns[endpoint] = end_time
        
        logger.warning(f"Setting cooldown for {endpoint} until {end_time} ({minutes} minutes)")
        
        # Activate mock data mode
        self.mock_mode_active = True
        
        # Save state
        self._save_state()
    
    def clear_cooldown(self, endpoint):
        """
        Clear the cooldown for the specified endpoint.
        
        Args:
            endpoint (str): The API endpoint to clear cooldown for
        """
        if endpoint in self.cooldowns:
            del self.cooldowns[endpoint]
            logger.info(f"Cleared cooldown for {endpoint}")
            
            # Check if any cooldowns are still active
            if not self.cooldowns:
                self.mock_mode_active = False
                logger.info("Deactivating mock data mode as all cooldowns have expired")
            
            # Save state
            self._save_state()
    
    def reset(self):
        """
        Reset all call counts and cooldowns.
        """
        self.calls = {endpoint: 0 for endpoint in self.limits}
        self.cooldowns = {}
        self.mock_mode_active = False
        
        logger.info("Reset all call counts and cooldowns")
        
        # Save state
        self._save_state()
    
    def is_rate_limited(self, endpoint):
        """
        Check if the specified endpoint is rate limited.
        
        Args:
            endpoint (str): The API endpoint to check
        
        Returns:
            bool: True if the endpoint is rate limited, False otherwise
        """
        if endpoint not in self.limits:
            logger.warning(f"Unknown endpoint: {endpoint}")
            return False
        
        # Check if endpoint is in cooldown
        if endpoint in self.cooldowns:
            now = datetime.now()
            if now < self.cooldowns[endpoint]:
                # Endpoint is in cooldown
                return True
        
        return False
    
    def get_status(self):
        """
        Get the current status of the rate limiter.
        
        Returns:
            dict: Current status including call counts, cooldowns, and mock mode
        """
        now = datetime.now()
        
        # Calculate remaining cooldown time for each endpoint
        cooldown_remaining = {}
        for endpoint, end_time in self.cooldowns.items():
            if now < end_time:
                cooldown_remaining[endpoint] = (end_time - now).total_seconds()
            else:
                cooldown_remaining[endpoint] = 0
        
        return {
            "calls": self.calls,
            "cooldowns": {endpoint: end_time.isoformat() for endpoint, end_time in self.cooldowns.items()},
            "cooldown_remaining": cooldown_remaining,
            "is_limited": bool(self.cooldowns),
            "mock_mode_active": self.mock_mode_active
        }
    
    def wait_if_needed(self, endpoint):
        """
        Wait if the endpoint is rate limited.
        
        Args:
            endpoint (str): The API endpoint to check
        
        Returns:
            bool: True if waited, False otherwise
        """
        if self.is_rate_limited(endpoint):
            # Calculate wait time
            now = datetime.now()
            end_time = self.cooldowns[endpoint]
            wait_time = (end_time - now).total_seconds()
            
            if wait_time > 0:
                logger.info(f"Waiting {wait_time:.1f} seconds for {endpoint} cooldown to expire")
                time.sleep(wait_time)
                return True
        
        return False
    
    def is_mock_mode_active(self):
        """
        Check if mock data mode is active.
        
        Returns:
            bool: True if mock data mode is active, False otherwise
        """
        # Refresh cooldown status
        self._check_cooldowns()
        
        return self.mock_mode_active
