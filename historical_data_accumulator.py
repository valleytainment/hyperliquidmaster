"""
Historical data accumulator for the Hyperliquid trading bot.
This is a mock implementation for testing purposes.
"""

class HistoricalDataAccumulator:
    def __init__(self):
        self.data = {}
        
    def add_data_point(self, symbol, timestamp, price, volume):
        """Add a data point to the historical data."""
        if symbol not in self.data:
            self.data[symbol] = []
            
        self.data[symbol].append({
            "timestamp": timestamp,
            "price": price,
            "volume": volume
        })
        
    def get_data(self, symbol, limit=None):
        """Get historical data for a symbol."""
        if symbol not in self.data:
            return []
            
        if limit:
            return self.data[symbol][-limit:]
        else:
            return self.data[symbol]
