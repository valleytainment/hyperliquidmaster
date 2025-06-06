"""
Test script to verify all import fixes and functionality
"""

import sys
import os
from pathlib import Path

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
        
        # Test GUI
        from gui.enhanced_gui_auto_connect_v2 import TradingDashboard
        print("✓ Trading dashboard imported successfully")
        
        # Test config manager
        from utils.config_manager_fixed import ConfigManager
        print("✓ Config manager imported successfully")
        
        # Test security manager
        from utils.security_fixed_v2 import SecurityManager
        print("✓ Security manager imported successfully")
        
        # Test connection manager
        from core.connection_manager_enhanced import ConnectionManager
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

def test_main_application():
    """Test main application initialization"""
    print("\nTesting main application...")
    
    try:
        # Import the main application
        import main_final_fixed
        print("✓ Main application imported successfully")
        
        # Test default credentials
        credentials = main_final_fixed.DEFAULT_CREDENTIALS
        print(f"✓ Default credentials loaded: {credentials['account_address']}")
        
        return True
    except Exception as e:
        print(f"✗ Main application error: {e}")
        return False

def main():
    """Run all tests"""
    print("Hyperliquid Master - Import Fix Verification")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_signal_creation,
        test_strategy_initialization,
        test_main_application
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Import fixes are working correctly.")
        return True
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

