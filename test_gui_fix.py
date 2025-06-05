"""
Test script for the fixed GUI implementation
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gui.enhanced_gui_fixed import TradingDashboard
from utils.logger import setup_logging

def test_gui_initialization():
    """Test GUI initialization with the fixed implementation"""
    try:
        # Setup logging
        setup_logging("DEBUG")
        
        # Create GUI instance
        print("Creating TradingDashboard instance...")
        gui = TradingDashboard()
        
        # Test toggle_private_key_visibility method
        print("Testing toggle_private_key_visibility method...")
        gui.toggle_private_key_visibility()
        
        print("GUI initialization test passed!")
        return True
    except Exception as e:
        print(f"GUI initialization test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_gui_initialization()
    sys.exit(0 if success else 1)

