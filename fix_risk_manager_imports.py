"""
Update all risk manager imports to use the fixed version
"""

import os
import re
from pathlib import Path

def fix_risk_manager_imports(file_path):
    """Fix risk manager imports in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix risk manager imports
        content = re.sub(
            r'from risk_management\.risk_manager import RiskManager, RiskLimits',
            'from risk_management.risk_manager_fixed import RiskManager, RiskLimits',
            content
        )
        
        content = re.sub(
            r'from risk_management\.risk_manager import RiskManager',
            'from risk_management.risk_manager_fixed import RiskManager',
            content
        )
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed risk manager imports in: {file_path}")
            return True
        
        return False
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    """Fix all risk manager imports"""
    print("Fixing all risk manager imports...")
    
    # Get project root
    project_root = Path(__file__).parent
    
    # Files to fix
    files_to_fix = [
        "strategies/base_strategy.py",
        "strategies/strategy_manager.py", 
        "strategies/base_strategy_fixed.py",
        "test_comprehensive.py",
        "main_fixed.py",
        "main_fixed_v2.py",
        "main_fixed_v3.py",
        "main_fixed_v4.py",
        "main_final_fixed.py",
        "main_completely_fixed.py",
        "test_comprehensive_final.py",
        "test_main_headless.py",
        "main_headless.py"
    ]
    
    fixed_count = 0
    
    for file_path in files_to_fix:
        full_path = project_root / file_path
        if full_path.exists():
            if fix_risk_manager_imports(full_path):
                fixed_count += 1
        else:
            print(f"File not found: {file_path}")
    
    print(f"\nFixed risk manager imports in {fixed_count} files")
    print("All risk manager import errors should now be resolved!")

if __name__ == "__main__":
    main()

