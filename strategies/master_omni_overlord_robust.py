"""
Master strategy implementation for the Hyperliquid trading bot.
This is a mock implementation for testing purposes.
"""

class MasterOmniOverlordRobustStrategy:
    def __init__(self, logger):
        self.logger = logger
        
    def generate_signal(self, data):
        """Generate a trading signal based on the provided data."""
        return {
            "signal": "NEUTRAL",
            "confidence": 0.5,
            "reason": "Mock strategy implementation for testing"
        }
