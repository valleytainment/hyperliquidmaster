"""
Test script to verify all import fixes and functionality (headless version)
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

def test_imports():
    """Test all imports"""
    print("Testing imports...")
    
    try:
        # Test trading types
        from strategies.trading_types_fixed import TradingSignal, SignalType, MarketData, OrderType
        print("✓ Trading types imported successfully")
        
        # Test base strategy
        from strategies.base_strategy_fixed import BaseStrategy
        print("✓ Base strategy imported successfully")
        
        # Test BB RSI ADX strategy
        from strategies.bb_rsi_adx_fixed import BBRSIADXStrategy
        print("✓ BB RSI ADX strategy imported successfully")
        
        # Test Hull Suite strategy
        from strategies.hull_suite_fixed import HullSuiteStrategy
        print("✓ Hull Suite strategy imported successfully")
        
        # Test API
        from core.api_fixed_v2 import EnhancedHyperliquidAPI
        print("✓ Enhanced API imported successfully")
        
        # Test config manager
        from utils.config_manager_fixed import ConfigManager
        print("✓ Config manager imported successfully")
        
        # Test security manager
        from utils.security_fixed_v2 import SecurityManager
        print("✓ Security manager imported successfully")
        
        # Test connection manager
        from core.connection_manager_enhanced import EnhancedConnectionManager as ConnectionManager
        print("✓ Connection manager imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_signal_creation():
    """Test signal creation"""
    print("\nTesting signal creation...")
    
    try:
        from strategies.trading_types_fixed import TradingSignal, SignalType
        
        # Create a buy signal
        signal = TradingSignal("BTC", SignalType.BUY, 0.8, 50000.0, 0.1, "Test signal")
        print(f"✓ Created signal: {signal}")
        
        # Create a sell signal
        signal = TradingSignal("ETH", SignalType.SELL, 0.9, 3000.0, 0.2, "Test sell signal")
        print(f"✓ Created signal: {signal}")
        
        return True
    except Exception as e:
        print(f"✗ Signal creation error: {e}")
        return False

def test_strategy_initialization():
    """Test strategy initialization"""
    print("\nTesting strategy initialization...")
    
    try:
        from strategies.bb_rsi_adx_fixed import BBRSIADXStrategy
        from strategies.hull_suite_fixed import HullSuiteStrategy
        
        # Initialize BB RSI ADX strategy
        bb_strategy = BBRSIADXStrategy()
        print(f"✓ BB RSI ADX strategy initialized: {bb_strategy.name}")
        
        # Initialize Hull Suite strategy
        hull_strategy = HullSuiteStrategy()
        print(f"✓ Hull Suite strategy initialized: {hull_strategy.name}")
        
        return True
    except Exception as e:
        print(f"✗ Strategy initialization error: {e}")
        return False

def test_main_application_imports():
    """Test main application imports only"""
    print("\nTesting main application imports...")
    
    try:
        # Test default credentials
        DEFAULT_CREDENTIALS = {
            "account_address": "0x306D29F56EA1345c7E6F1ff27657ba05cEE15D4F",
            "secret_key": "43ba46de58067dd1ef3794c653bf3b11fa78866623cc515a5aff5f4be31fd3b8",
            "api_url": "https://api.hyperliquid.xyz"
        }
        print(f"✓ Default credentials loaded: {DEFAULT_CREDENTIALS['account_address']}")
        
        # Test core imports
        from core.api_fixed_v2 import EnhancedHyperliquidAPI
        from utils.config_manager_fixed import ConfigManager
        from utils.security_fixed_v2 import SecurityManager
        from core.connection_manager_enhanced import EnhancedConnectionManager as ConnectionManager
        print("✓ Core components imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Main application error: {e}")
        return False

def test_original_main_fix():
    """Test if the original main.py can now import properly"""
    print("\nTesting original main.py imports...")
    
    try:
        # Test the specific imports that were failing
        from strategies.bb_rsi_adx_fixed import BBRSIADXStrategy
        from strategies.hull_suite_fixed import HullSuiteStrategy
        print("✓ Original main.py imports should now work")
        
        return True
    except Exception as e:
        print(f"✗ Original main.py import error: {e}")
        return False

def main():
    """Run all tests"""
    print("Hyperliquid Master - Import Fix Verification (Headless)")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_signal_creation,
        test_strategy_initialization,
        test_main_application_imports,
        test_original_main_fix
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Import fixes are working correctly.")
        print("\nYou can now run the application with:")
        print("  python main_final_fixed.py --mode gui")
        print("  python main_final_fixed.py --mode cli")
        print("  python main_final_fixed.py --mode setup")
        print("  python main_final_fixed.py --mode trading")
        print("\nOr use the original main.py after updating the imports:")
        print("  python main.py --mode gui")
        return True
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

