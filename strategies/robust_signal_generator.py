"""
Robust signal generator for the Hyperliquid trading bot.
This is a mock implementation for testing purposes.
"""

class RobustSignalGenerator:
    def __init__(self):
        pass
        
    def generate_signal(self, data):
        """Generate a trading signal based on the provided data."""
        return {
            "signal": "NEUTRAL",
            "confidence": 0.5,
            "reason": "Mock signal generator implementation for testing"
        }
