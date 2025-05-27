"""
Error handler module for the Hyperliquid trading bot.
This is a mock implementation for testing purposes.
"""

class ErrorHandler:
    def __init__(self, logger):
        self.logger = logger
        
    def handle_error(self, error, context=None):
        """Handle an error by logging it and taking appropriate action."""
        if context:
            self.logger.error(f"Error in {context}: {error}")
        else:
            self.logger.error(f"Error: {error}")
        
        return {"handled": True, "action": "logged"}
