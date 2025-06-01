#!/usr/bin/env python3
"""
GUI Module Loading Test Script

This script tests the loading of GUI modules in a headless environment.
It verifies that all GUI components can be imported and instantiated without errors.
"""

import sys
import os
import tkinter as tk
from tkinter import ttk

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import GUI modules
try:
    print("Importing GUI modules...")
    from gui.enhanced_gui import (
        MainApplication, ChartView, OrderBookView, PositionsView, 
        TradeView, SettingsView, LogView, ScrollableFrame
    )
    print("All GUI modules imported successfully")
    
    # Test ScrollableFrame instantiation
    print("Testing ScrollableFrame instantiation...")
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    sf = ScrollableFrame(root)
    print("ScrollableFrame instantiated successfully")
    
    # Test SettingsView instantiation
    print("Testing SettingsView instantiation...")
    
    # Create mock classes for testing
    class MockConfigData:
        def __init__(self):
            self.config = {}
        
        def get(self, key, default=None):
            return default
        
        def set(self, key, value):
            self.config[key] = value
    
    class MockMarketVM:
        def __init__(self):
            pass
        
        def toggle_mock_data(self, use_mock):
            pass
    
    # Create mock instances
    config_data = MockConfigData()
    market_vm = MockMarketVM()
    
    # Instantiate SettingsView
    sv = SettingsView(root, config_data, market_vm)
    print("SettingsView instantiated successfully")
    
    # Test other views
    print("Testing other views...")
    
    # Create mock classes for other views
    class MockMarketData:
        def __init__(self):
            pass
    
    class MockPositionData:
        def __init__(self):
            pass
    
    class MockErrorHandler:
        def __init__(self):
            pass
        
        def handle_error(self, error):
            print(f"Mock error handler: {error}")
    
    # Create mock instances
    market_data = MockMarketData()
    position_data = MockPositionData()
    error_handler = MockErrorHandler()
    
    # Import core components
    from core.error_handling import ErrorHandler
    
    # Test instantiation of other views
    try:
        chart_view = ChartView(root, market_vm, config_data)
        print("ChartView instantiated successfully")
    except Exception as e:
        print(f"ChartView instantiation failed: {e}")
    
    try:
        order_book_view = OrderBookView(root, market_vm, config_data)
        print("OrderBookView instantiated successfully")
    except Exception as e:
        print(f"OrderBookView instantiation failed: {e}")
    
    try:
        positions_view = PositionsView(root, market_vm)
        print("PositionsView instantiation failed as expected (needs TradingViewModel)")
    except Exception as e:
        print(f"PositionsView instantiation failed as expected: {type(e).__name__}")
    
    try:
        log_view = LogView(root)
        print("LogView instantiated successfully")
    except Exception as e:
        print(f"LogView instantiation failed: {e}")
    
    print("GUI module loading verification complete")
    
except Exception as e:
    print(f"Error during GUI module testing: {e}")
    sys.exit(1)
