"""
Complete fix for all import errors in Hyperliquid Master
This script updates all files to use the correct imports
"""

import os
import re
from pathlib import Path

def fix_file_imports(file_path):
    """Fix imports in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix base_strategy imports
        content = re.sub(
            r'from strategies\.base_strategy import BaseStrategy, TradingSignal, SignalType, MarketData',
            'from strategies.base_strategy_fixed import BaseStrategy\nfrom strategies.trading_types_fixed import TradingSignal, SignalType, MarketData',
            content
        )
        
        content = re.sub(
            r'from strategies\.base_strategy import BaseStrategy',
            'from strategies.base_strategy_fixed import BaseStrategy',
            content
        )
        
        # Fix strategy imports
        content = re.sub(
            r'from strategies\.bb_rsi_adx import BBRSIADXStrategy',
            'from strategies.bb_rsi_adx_fixed import BBRSIADXStrategy',
            content
        )
        
        content = re.sub(
            r'from strategies\.hull_suite import HullSuiteStrategy',
            'from strategies.hull_suite_fixed import HullSuiteStrategy',
            content
        )
        
        # Fix class name references
        content = re.sub(r'\bBB_RSI_ADX\b', 'BBRSIADXStrategy', content)
        content = re.sub(r'\bHullSuite\b', 'HullSuiteStrategy', content)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed imports in: {file_path}")
            return True
        
        return False
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    """Fix all import errors"""
    print("Fixing all import errors in Hyperliquid Master...")
    
    # Get project root
    project_root = Path(__file__).parent
    
    # Files to fix
    files_to_fix = [
        "main.py",
        "backtesting/backtest_engine.py",
        "strategies/strategy_manager.py",
        "test_comprehensive.py",
        "main_fixed_v2.py",
        "main_fixed_v3.py",
        "main_fixed_v4.py",
        "test_all_enhancements.py",
        "test_all_enhancements_fixed.py",
        "test_all_enhancements_fixed_v2.py"
    ]
    
    fixed_count = 0
    
    for file_path in files_to_fix:
        full_path = project_root / file_path
        if full_path.exists():
            if fix_file_imports(full_path):
                fixed_count += 1
        else:
            print(f"File not found: {file_path}")
    
    print(f"\nFixed imports in {fixed_count} files")
    print("All import errors should now be resolved!")

if __name__ == "__main__":
    main()

