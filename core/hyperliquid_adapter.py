"""
HyperliquidAdapter module for interfacing with the Hyperliquid exchange API.
This is a mock implementation for testing purposes.
"""

class HyperliquidAdapter:
    def __init__(self, config_path):
        self.config_path = config_path
        
    def get_account_info(self):
        return {
            "equity": 10000.0,
            "available_balance": 9000.0,
            "positions": []
        }
        
    def place_order(self, symbol, side, size, price=None, order_type="MARKET"):
        return {
            "order_id": "mock-order-123",
            "status": "FILLED",
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": price,
            "type": order_type
        }
        
    def cancel_order(self, order_id):
        return {"success": True}
        
    def get_positions(self):
        return []
        
    def close_position(self, symbol, size_percentage=100.0):
        return {"success": True}
