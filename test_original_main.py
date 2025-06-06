"""
Test the original main.py to ensure it works without import errors
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

def test_original_main_imports():
    """Test that the original main.py imports work"""
    print("🧪 Testing original main.py imports...")
    
    try:
        # Test the imports from main.py
        from core.api import EnhancedHyperliquidAPI
        print("✅ Core API imported successfully")
        
        from strategies.bb_rsi_adx_fixed import BBRSIADXStrategy
        from strategies.hull_suite_fixed import HullSuiteStrategy
        print("✅ Strategies imported successfully")
        
        from backtesting.backtest_engine import BacktestEngine
        print("✅ Backtest engine imported successfully")
        
        from risk_management.risk_manager_fixed import RiskManager, RiskLimits
        print("✅ Risk manager and RiskLimits imported successfully")
        
        from utils.logger import get_logger, setup_logging, TradingLogger
        from utils.config_manager import ConfigManager, TradingConfig
        from utils.security import SecurityManager
        print("✅ Utilities imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_risk_limits_functionality():
    """Test RiskLimits functionality"""
    print("\n🎯 Testing RiskLimits functionality...")
    
    try:
        from risk_management.risk_manager_fixed import RiskManager, RiskLimits
        
        # Create RiskLimits instance
        risk_limits = RiskLimits(
            max_portfolio_risk=0.02,
            max_daily_loss=0.05,
            max_drawdown=0.10,
            max_leverage=3.0,
            max_position_size=0.10
        )
        
        print(f"✅ RiskLimits created: max_daily_loss={risk_limits.max_daily_loss}")
        
        # Create RiskManager with RiskLimits
        risk_manager = RiskManager(risk_limits)
        print("✅ RiskManager created with RiskLimits")
        
        # Test risk manager functionality
        status = risk_manager.get_risk_status()
        print(f"✅ Risk status: {status}")
        
        is_allowed = risk_manager.is_trading_allowed()
        print(f"✅ Trading allowed: {is_allowed}")
        
        return True
    except Exception as e:
        print(f"❌ RiskLimits functionality error: {e}")
        return False

def test_main_initialization():
    """Test main application initialization"""
    print("\n🚀 Testing main application initialization...")
    
    try:
        # Test that we can create the main components
        from utils.config_manager import ConfigManager
        from utils.security import SecurityManager
        from risk_management.risk_manager_fixed import RiskManager, RiskLimits
        
        # Initialize components
        config_manager = ConfigManager()
        security_manager = SecurityManager()
        
        # Create risk limits and manager
        risk_limits = RiskLimits()
        risk_manager = RiskManager(risk_limits)
        
        print("✅ All main components initialized successfully")
        
        return True
    except Exception as e:
        print(f"❌ Main initialization error: {e}")
        return False

def main():
    """Test original main.py functionality"""
    print("🚀 Testing Original Main.py - Import Fix Verification")
    print("=" * 60)
    
    tests = [
        ("Original Main Imports", test_original_main_imports),
        ("RiskLimits Functionality", test_risk_limits_functionality),
        ("Main Initialization", test_main_initialization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} - PASSED")
        else:
            print(f"❌ {test_name} - FAILED")
    
    print(f"\n{'='*60}")
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Original main.py should now work!")
        print("\n🚀 You can now run:")
        print("  python main.py --mode gui")
        print("  python main.py --mode setup")
        print("  python main.py --mode trading")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

