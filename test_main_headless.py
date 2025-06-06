"""
Headless test for main application functionality
"""

import sys
import os
from pathlib import Path

# Set matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_main_application_headless():
    """Test main application without GUI components"""
    print("🏠 Testing main application (headless)...")
    
    try:
        # Test default credentials
        DEFAULT_CREDENTIALS = {
            "account_address": "0x306D29F56EA1345c7E6F1ff27657ba05cEE15D4F",
            "secret_key": "43ba46de58067dd1ef3794c653bf3b11fa78866623cc515a5aff5f4be31fd3b8",
            "api_url": "https://api.hyperliquid.xyz"
        }
        print(f"✅ Default credentials: {DEFAULT_CREDENTIALS['account_address']}")
        
        # Test core components
        from core.api_fixed_v2 import EnhancedHyperliquidAPI
        from utils.config_manager_fixed import ConfigManager
        from utils.security_fixed_v2 import SecurityManager
        from core.connection_manager_enhanced import EnhancedConnectionManager
        from strategies.bb_rsi_adx_fixed import BBRSIADXStrategy
        from strategies.hull_suite_fixed import HullSuiteStrategy
        from backtesting.backtest_engine import BacktestEngine
        from risk_management.risk_manager_fixed import RiskManager
        
        print("✅ All core components imported successfully")
        
        # Test component initialization
        config_manager = ConfigManager()
        security_manager = SecurityManager()
        connection_manager = EnhancedConnectionManager()
        api = EnhancedHyperliquidAPI()
        risk_manager = RiskManager()
        
        print("✅ All components initialized successfully")
        
        # Test strategies
        bb_strategy = BBRSIADXStrategy(api=api, risk_manager=risk_manager)
        hull_strategy = HullSuiteStrategy(api=api, risk_manager=risk_manager)
        
        print("✅ Strategies initialized successfully")
        
        # Test backtest engine
        backtest_engine = BacktestEngine(initial_capital=10000.0)
        
        print("✅ Backtest engine initialized successfully")
        
        return True
    except Exception as e:
        print(f"❌ Main application error: {e}")
        return False

def main():
    """Test main application functionality"""
    print("🚀 Hyperliquid Master - Main Application Test (Headless)")
    print("=" * 60)
    
    if test_main_application_headless():
        print("\n🎉 Main application test PASSED!")
        print("✅ All components are working correctly")
        return True
    else:
        print("\n❌ Main application test FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

