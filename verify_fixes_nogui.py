"""
Verification script for the fixes implemented in the Hyperliquid Trading Bot
This version does not depend on GUI components
"""

import sys
import os
import logging
from pathlib import Path
import unittest

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logger import setup_logging, get_logger

# Setup logging
setup_logging("DEBUG")
logger = get_logger(__name__)

class FixVerificationTests(unittest.TestCase):
    """Test cases to verify the fixes implemented"""
    
    def test_file_exists(self):
        """Test that the fixed files exist"""
        gui_fixed_path = os.path.join(project_root, 'gui', 'enhanced_gui_fixed.py')
        main_fixed_path = os.path.join(project_root, 'main_fixed.py')
        
        self.assertTrue(os.path.exists(gui_fixed_path), "enhanced_gui_fixed.py file exists")
        self.assertTrue(os.path.exists(main_fixed_path), "main_fixed.py file exists")
        logger.info("✅ Fixed files exist")
    
    def test_toggle_private_key_visibility_in_file(self):
        """Test that the toggle_private_key_visibility method is in the file"""
        gui_fixed_path = os.path.join(project_root, 'gui', 'enhanced_gui_fixed.py')
        
        with open(gui_fixed_path, 'r') as f:
            content = f.read()
            
        self.assertIn('def toggle_private_key_visibility', content)
        self.assertIn('private_key', content)
        self.assertIn('show_key_btn', content)
        logger.info("✅ toggle_private_key_visibility method found in file")
    
    def test_defensive_checks(self):
        """Test that defensive checks are added"""
        gui_fixed_path = os.path.join(project_root, 'gui', 'enhanced_gui_fixed.py')
        
        with open(gui_fixed_path, 'r') as f:
            content = f.read()
            
        self.assertIn("if 'private_key' not in self.widgets:", content)
        self.assertIn("if 'show_key_btn' not in self.widgets:", content)
        logger.info("✅ Defensive checks found in file")
    
    def test_error_handling(self):
        """Test that error handling is added"""
        gui_fixed_path = os.path.join(project_root, 'gui', 'enhanced_gui_fixed.py')
        
        with open(gui_fixed_path, 'r') as f:
            content = f.read()
            
        self.assertIn("except Exception as e:", content)
        self.assertIn("logger.error", content)
        logger.info("✅ Error handling found in file")
    
    def test_main_fixed_imports(self):
        """Test that main_fixed.py imports the fixed GUI"""
        main_fixed_path = os.path.join(project_root, 'main_fixed.py')
        
        with open(main_fixed_path, 'r') as f:
            content = f.read()
            
        self.assertIn("from gui.enhanced_gui_fixed import TradingDashboard", content)
        logger.info("✅ main_fixed.py imports the fixed GUI")

def run_tests():
    """Run the verification tests"""
    logger.info("Starting fix verification tests...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    logger.info("Fix verification tests completed")

if __name__ == "__main__":
    run_tests()

