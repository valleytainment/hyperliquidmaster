#!/usr/bin/env python3
"""
Test script for validating the real trading functionality.
This script tests the core components of the trading system.
"""

import os
import sys
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TradingTest")

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import core components
from core.hyperliquid_adapter import HyperliquidAdapter
from core.api_key_manager import ApiKeyManager
from core.error_handler import ErrorHandler
from core.trading_integration import TradingIntegration

def test_api_key_manager():
    """Test the API key manager functionality."""
    logger.info("Testing API Key Manager...")
    
    # Create a test config file
    test_config_path = "test_config.json"
    
    # Initialize API key manager
    api_key_manager = ApiKeyManager(test_config_path)
    
    # Test setting API keys
    test_account = "0xtest123456789abcdef"
    test_secret = "testsecretkey123456789"
    
    success = api_key_manager.set_api_keys(test_account, test_secret)
    logger.info(f"Set API keys: {success}")
    
    # Test getting API keys
    account, secret = api_key_manager.get_api_keys()
    logger.info(f"Retrieved account: {account}")
    logger.info(f"Keys match: {account == test_account and secret == test_secret}")
    
    # Test has_valid_keys
    has_keys = api_key_manager.has_valid_keys()
    logger.info(f"Has valid keys: {has_keys}")
    
    # Test clearing keys
    success = api_key_manager.clear_keys()
    logger.info(f"Cleared keys: {success}")
    
    # Verify keys are cleared
    account, secret = api_key_manager.get_api_keys()
    logger.info(f"Keys cleared: {account == '' and secret == ''}")
    
    # Clean up
    if os.path.exists(test_config_path):
        os.remove(test_config_path)
    
    logger.info("API Key Manager tests completed")

def test_error_handler():
    """Test the error handler functionality."""
    logger.info("Testing Error Handler...")
    
    # Initialize error handler
    error_handler = ErrorHandler(logger)
    
    # Test handling different types of errors
    test_errors = [
        ("ConnectionError", ConnectionError("Failed to connect to API")),
        ("TimeoutError", TimeoutError("Request timed out")),
        ("ValueError", ValueError("Invalid parameter")),
        ("Exception", Exception("Generic error"))
    ]
    
    for error_name, error in test_errors:
        logger.info(f"Testing error handling for {error_name}")
        result = error_handler.handle_error(error, {"operation": "test", "context": error_name})
        logger.info(f"Result: {result}")
    
    logger.info("Error Handler tests completed")

def test_hyperliquid_adapter_mock():
    """Test the Hyperliquid adapter with mock data (no real API calls)."""
    logger.info("Testing Hyperliquid Adapter (mock)...")
    
    # Create a test config file
    test_config_path = "test_config.json"
    with open(test_config_path, 'w') as f:
        json.dump({
            "account_address": "0xtest123456789abcdef",
            "secret_key": "testsecretkey123456789",
            "symbols": ["BTC", "ETH", "SOL"],
            "theme": "dark"
        }, f)
    
    # Initialize adapter
    adapter = HyperliquidAdapter(test_config_path)
    
    # Test basic functionality (these won't make real API calls due to mock credentials)
    logger.info("Testing get_account_info")
    account_info = adapter.get_account_info()
    logger.info(f"Account info result: {'error' in account_info}")
    
    logger.info("Testing place_order")
    order_result = adapter.place_order("BTC", True, 0.1, 50000, "LIMIT")
    logger.info(f"Order result: {'error' in order_result}")
    
    # Clean up
    if os.path.exists(test_config_path):
        os.remove(test_config_path)
    
    logger.info("Hyperliquid Adapter mock tests completed")

def test_trading_integration():
    """Test the trading integration layer."""
    logger.info("Testing Trading Integration...")
    
    # Create a test config file
    test_config_path = "test_config.json"
    
    # Initialize trading integration
    trading = TradingIntegration(test_config_path, logger)
    
    # Test API key setting
    logger.info("Testing set_api_keys")
    result = trading.set_api_keys("0xtest123456789abcdef", "testsecretkey123456789")
    logger.info(f"Set API keys result: {result}")
    
    # Test connection status
    logger.info(f"Connection status: {trading.is_connected}")
    
    # Test market data retrieval (will fail with mock keys but should handle gracefully)
    logger.info("Testing get_market_data")
    market_data = trading.get_market_data("BTC")
    logger.info(f"Market data result: {market_data['success']}")
    
    # Clean up
    if os.path.exists(test_config_path):
        os.remove(test_config_path)
    
    logger.info("Trading Integration tests completed")

def run_all_tests():
    """Run all tests."""
    logger.info("Starting all tests...")
    
    test_api_key_manager()
    test_error_handler()
    test_hyperliquid_adapter_mock()
    test_trading_integration()
    
    logger.info("All tests completed")

if __name__ == "__main__":
    logger.info(f"Starting trading system tests at {datetime.now().isoformat()}")
    run_all_tests()
    logger.info(f"Tests finished at {datetime.now().isoformat()}")
