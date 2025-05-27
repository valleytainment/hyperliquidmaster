"""
Order book handler for the Hyperliquid trading bot.
This is a mock implementation for testing purposes.
"""

class OrderBookHandler:
    def __init__(self, logger):
        self.logger = logger
        self.order_books = {}
        
    def update_order_book(self, symbol, bids, asks):
        """Update the order book for a symbol."""
        self.order_books[symbol] = {
            "bids": bids,
            "asks": asks,
            "timestamp": "2023-01-01T00:00:00Z"
        }
        
    def get_order_book(self, symbol):
        """Get the order book for a symbol."""
        if symbol in self.order_books:
            return self.order_books[symbol]
        else:
            return {
                "bids": [],
                "asks": [],
                "timestamp": "2023-01-01T00:00:00Z"
            }
            
    def get_best_bid(self, symbol):
        """Get the best bid for a symbol."""
        order_book = self.get_order_book(symbol)
        
        if order_book["bids"]:
            return order_book["bids"][0]
        else:
            return None
            
    def get_best_ask(self, symbol):
        """Get the best ask for a symbol."""
        order_book = self.get_order_book(symbol)
        
        if order_book["asks"]:
            return order_book["asks"][0]
        else:
            return None
