"""
API rate limiter for the Hyperliquid trading bot.
This is a mock implementation for testing purposes.
"""

import time

class APIRateLimiter:
    def __init__(self):
        self.last_request_time = {}
        self.min_interval = 0.5  # 500ms between requests
        
    def limit_request(self, endpoint):
        """Limit request rate for a specific endpoint."""
        current_time = time.time()
        
        if endpoint in self.last_request_time:
            elapsed = current_time - self.last_request_time[endpoint]
            
            if elapsed < self.min_interval:
                # Need to wait
                wait_time = self.min_interval - elapsed
                time.sleep(wait_time)
        
        # Update last request time
        self.last_request_time[endpoint] = time.time()
        
        return True
