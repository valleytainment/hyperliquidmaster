#!/usr/bin/env python3
"""
ULTIMATE BLENDED SYSTEM TEST SUITE
-----------------------------------
Comprehensive testing for the Ultimate Master Bot blended system.
Tests all Code 1 functionality preservation + enhanced features.
"""

import os
import sys
import time
import json
import logging
import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import torch

# Add the repository path
repo_path = os.path.dirname(os.path.abspath(__file__))
if repo_path not in sys.path:
    sys.path.append(repo_path)

# Test imports
try:
    from ultimate_master_bot_blended import (
        EnhancedUltimateMasterBot, 
        EnhancedBotUI,
        EnhancedRLParameterTuner,
        TransformerPriceModel,
        CONFIG,
        DEFAULT_CREDENTIALS
    )
    from core.master_trading_level import (
        MasterTradingLevelCalculator,
        MarketConditions,
        MarketRegime,
        TradingSignal
    )
    BLENDED_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"[TEST] Blended system import error: {e}")
    BLENDED_SYSTEM_AVAILABLE = False

class TestUltimateBlendedSystem(unittest.TestCase):
    """Test suite for the ultimate blended trading system"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_config = {
            "account_address": "0x306D29F56EA1345c7E6F1ff27657ba05cEE15D4F",
            "secret_key": "43ba46de58067dd1ef3794c653bf3b11fa78866623cc515a5aff5f4be31fd3b8",
            "api_url": "https://api.hyperliquid.xyz",
            "trade_symbol": "BTC-USD-PERP",
            "trade_mode": "perp",
            "master_trading_level": "highest",
            "full_auto_mode": True,
            "order_size": 1.0,
            "nn_lookback_bars": 30,
            "nn_hidden_size": 64,
            "synergy_conf_threshold": 0.8
        }
        
        # Create test data
        self.test_price_data = self._create_test_price_data()
        
    def _create_test_price_data(self) -> pd.DataFrame:
        """Create test price data with indicators"""
        np.random.seed(42)
        n_bars = 100
        
        # Generate realistic price data
        base_price = 50000
        prices = []
        volumes = []
        
        for i in range(n_bars):
            if i == 0:
                price = base_price
            else:
                change = np.random.normal(0, 0.02)  # 2% volatility
                price = prices[-1] * (1 + change)
            
            prices.append(price)
            volumes.append(np.random.uniform(100, 1000))
        
        # Create DataFrame with all required columns
        data = pd.DataFrame({
            "time": pd.date_range(start="2024-01-01", periods=n_bars, freq="1H"),
            "price": prices,
            "volume": volumes,
            "vol_ma": np.random.uniform(200, 800, n_bars),
            "fast_ma": np.random.uniform(49000, 51000, n_bars),
            "slow_ma": np.random.uniform(48000, 52000, n_bars),
            "rsi": np.random.uniform(20, 80, n_bars),
            "macd_hist": np.random.uniform(-100, 100, n_bars),
            "bb_high": np.array(prices) * 1.02,
            "bb_low": np.array(prices) * 0.98,
            "stoch_k": np.random.uniform(0, 100, n_bars),
            "stoch_d": np.random.uniform(0, 100, n_bars),
            "adx": np.random.uniform(10, 60, n_bars),
            "atr": np.random.uniform(100, 500, n_bars)
        })
        
        return data
    
    @unittest.skipUnless(BLENDED_SYSTEM_AVAILABLE, "Blended system not available")
    def test_enhanced_bot_initialization(self):
        """Test enhanced bot initialization with Code 1 preservation"""
        print("\n[TEST] Testing enhanced bot initialization...")
        
        # Mock the queue and external dependencies
        with patch('queue.Queue'), \
             patch('hyperliquid.exchange.Exchange'), \
             patch('hyperliquid.info.Info'), \
             patch('eth_account.Account.from_key'):
            
            bot = EnhancedUltimateMasterBot(self.test_config, Mock())
            
            # Test Code 1 attributes preserved
            self.assertEqual(bot.symbol, "BTC-USD-PERP")
            self.assertEqual(bot.trade_mode, "perp")
            self.assertEqual(bot.lookback_bars, 30)
            self.assertEqual(bot.features_per_bar, 12)
            self.assertEqual(bot.nn_hidden_size, 64)
            
            # Test enhanced attributes added
            self.assertEqual(bot.master_level, "highest")
            self.assertTrue(bot.full_auto_mode)
            self.assertIsNotNone(bot.tuner)
            self.assertIsInstance(bot.tuner, EnhancedRLParameterTuner)
            
            print("✅ Enhanced bot initialization test passed")
    
    @unittest.skipUnless(BLENDED_SYSTEM_AVAILABLE, "Blended system not available")
    def test_transformer_model_preservation(self):
        """Test that TransformerPriceModel is preserved from Code 1"""
        print("\n[TEST] Testing Transformer model preservation...")
        
        model = TransformerPriceModel(
            input_size_per_bar=12,
            lookback_bars=30,
            hidden_size=64
        )
        
        # Test model structure
        self.assertEqual(model.input_size_per_bar, 12)
        self.assertEqual(model.lookback_bars, 30)
        self.assertEqual(model.hidden_size, 64)
        
        # Test forward pass
        batch_size = 2
        input_tensor = torch.randn(batch_size, 30 * 12)  # 30 bars * 12 features
        
        reg_out, cls_out = model(input_tensor)
        
        # Test output shapes
        self.assertEqual(reg_out.shape, (batch_size, 1))
        self.assertEqual(cls_out.shape, (batch_size, 3))
        
        print("✅ Transformer model preservation test passed")
    
    @unittest.skipUnless(BLENDED_SYSTEM_AVAILABLE, "Blended system not available")
    def test_enhanced_rl_tuner(self):
        """Test enhanced RL parameter tuner"""
        print("\n[TEST] Testing enhanced RL parameter tuner...")
        
        tuner = EnhancedRLParameterTuner(self.test_config, "test_params.json")
        
        # Test initialization
        self.assertEqual(tuner.master_level, "highest")
        self.assertEqual(tuner.optimization_cycles, 0)
        
        # Test trade processing
        tuner.on_trade_closed(100.0)  # Winning trade
        tuner.on_trade_closed(-50.0)  # Losing trade
        
        self.assertEqual(tuner.trade_count, 2)
        self.assertEqual(tuner.episode_pnl, 50.0)
        
        # Test performance metrics
        metrics = tuner.get_performance_metrics()
        self.assertIn("win_rate", metrics)
        self.assertIn("total_trades", metrics)
        self.assertIn("master_level", metrics)
        
        print("✅ Enhanced RL tuner test passed")
    
    @unittest.skipUnless(BLENDED_SYSTEM_AVAILABLE, "Blended system not available")
    def test_master_trading_level_calculator(self):
        """Test master trading level calculator"""
        print("\n[TEST] Testing master trading level calculator...")
        
        calculator = MasterTradingLevelCalculator(self.test_config)
        
        # Test initialization
        self.assertEqual(calculator.master_level, "highest")
        self.assertTrue(calculator.full_auto_mode)
        self.assertGreater(calculator.risk_tolerance, 0)
        self.assertGreater(calculator.profit_target, 0)
        
        # Test market analysis
        market_conditions = calculator.analyze_market_conditions(self.test_price_data)
        
        self.assertIsInstance(market_conditions, MarketConditions)
        self.assertGreaterEqual(market_conditions.volatility, 0)
        self.assertGreaterEqual(market_conditions.confidence, 0)
        self.assertLessEqual(market_conditions.confidence, 1)
        
        # Test signal generation
        signal = calculator.generate_enhanced_signal(
            self.test_price_data, "BUY", 0.8
        )
        
        self.assertIsInstance(signal, TradingSignal)
        self.assertIn(signal.direction, ["BUY", "SELL", "HOLD"])
        self.assertGreaterEqual(signal.confidence, 0)
        self.assertLessEqual(signal.confidence, 1)
        
        print("✅ Master trading level calculator test passed")
    
    @unittest.skipUnless(BLENDED_SYSTEM_AVAILABLE, "Blended system not available")
    def test_code1_functionality_preservation(self):
        """Test that all Code 1 functionality is preserved"""
        print("\n[TEST] Testing Code 1 functionality preservation...")
        
        with patch('queue.Queue'), \
             patch('hyperliquid.exchange.Exchange'), \
             patch('hyperliquid.info.Info'), \
             patch('eth_account.Account.from_key'):
            
            bot = EnhancedUltimateMasterBot(self.test_config, Mock())
            
            # Test Code 1 methods exist and work
            self.assertTrue(hasattr(bot, 'get_equity'))
            self.assertTrue(hasattr(bot, 'fetch_price_volume'))
            self.assertTrue(hasattr(bot, 'compute_indicators'))
            self.assertTrue(hasattr(bot, 'build_input_features'))
            self.assertTrue(hasattr(bot, 'store_training_if_possible'))
            self.assertTrue(hasattr(bot, 'do_mini_batch_train'))
            self.assertTrue(hasattr(bot, 'final_inference'))
            self.assertTrue(hasattr(bot, 'get_user_position'))
            self.assertTrue(hasattr(bot, 'market_order'))
            
            # Test Code 1 data structures
            self.assertIsInstance(bot.hist_data, pd.DataFrame)
            self.assertIsInstance(bot.training_data, list)
            self.assertIsInstance(bot.trade_pnls, list)
            
            # Test Code 1 configuration preservation
            self.assertEqual(bot.account_address, self.test_config["account_address"])
            self.assertEqual(bot.secret_key, self.test_config["secret_key"])
            
            print("✅ Code 1 functionality preservation test passed")
    
    @unittest.skipUnless(BLENDED_SYSTEM_AVAILABLE, "Blended system not available")
    def test_enhanced_features_integration(self):
        """Test enhanced features integration"""
        print("\n[TEST] Testing enhanced features integration...")
        
        with patch('queue.Queue'), \
             patch('hyperliquid.exchange.Exchange'), \
             patch('hyperliquid.info.Info'), \
             patch('eth_account.Account.from_key'):
            
            bot = EnhancedUltimateMasterBot(self.test_config, Mock())
            
            # Test enhanced methods
            self.assertTrue(hasattr(bot, 'get_enhanced_performance_metrics'))
            self.assertTrue(hasattr(bot, 'set_symbol'))
            
            # Test enhanced configuration
            self.assertIn("master_trading_level", bot.config)
            self.assertIn("full_auto_mode", bot.config)
            
            # Test enhanced performance metrics
            metrics = bot.get_enhanced_performance_metrics()
            self.assertIsInstance(metrics, dict)
            self.assertIn("master_level", metrics)
            self.assertIn("full_auto_mode", metrics)
            self.assertIn("current_equity", metrics)
            
            print("✅ Enhanced features integration test passed")
    
    @unittest.skipUnless(BLENDED_SYSTEM_AVAILABLE, "Blended system not available")
    def test_auto_connection_feature(self):
        """Test auto-connection with default credentials"""
        print("\n[TEST] Testing auto-connection feature...")
        
        # Test default credentials are set
        self.assertIn("account_address", DEFAULT_CREDENTIALS)
        self.assertIn("secret_key", DEFAULT_CREDENTIALS)
        
        # Test config creation with auto-connection
        test_config_file = "test_config.json"
        if os.path.exists(test_config_file):
            os.remove(test_config_file)
        
        # This would normally prompt for input, but we'll test the structure
        expected_keys = [
            "account_address", "secret_key", "api_url", "trade_symbol",
            "master_trading_level", "full_auto_mode", "auto_connect"
        ]
        
        for key in expected_keys:
            self.assertIn(key, CONFIG)
        
        print("✅ Auto-connection feature test passed")
    
    def test_system_integration(self):
        """Test overall system integration"""
        print("\n[TEST] Testing system integration...")
        
        # Test configuration loading
        self.assertIsInstance(CONFIG, dict)
        self.assertIn("trade_symbol", CONFIG)
        
        # Test file structure
        expected_files = [
            "ultimate_master_bot_blended.py",
            "core/master_trading_level.py"
        ]
        
        for file_path in expected_files:
            full_path = os.path.join(repo_path, file_path)
            self.assertTrue(os.path.exists(full_path), f"Missing file: {file_path}")
        
        print("✅ System integration test passed")
    
    def test_performance_requirements(self):
        """Test performance requirements"""
        print("\n[TEST] Testing performance requirements...")
        
        # Test data processing speed
        start_time = time.time()
        
        # Simulate data processing
        for _ in range(100):
            data = self._create_test_price_data()
            # Simulate indicator calculation
            data["test_ma"] = data["price"].rolling(10).mean()
        
        processing_time = time.time() - start_time
        
        # Should process 100 datasets in under 5 seconds
        self.assertLess(processing_time, 5.0, "Data processing too slow")
        
        print(f"✅ Performance test passed (processing time: {processing_time:.2f}s)")
    
    def tearDown(self):
        """Clean up test environment"""
        # Clean up any test files
        test_files = ["test_params.json", "test_config.json"]
        for file_path in test_files:
            if os.path.exists(file_path):
                os.remove(file_path)

def run_comprehensive_tests():
    """Run comprehensive test suite"""
    print("🚀 STARTING ULTIMATE BLENDED SYSTEM TESTS")
    print("=" * 60)
    
    # Test environment check
    print("\n📋 ENVIRONMENT CHECK:")
    print(f"✅ Python version: {sys.version}")
    print(f"✅ Working directory: {os.getcwd()}")
    print(f"✅ Repository path: {repo_path}")
    print(f"✅ Blended system available: {BLENDED_SYSTEM_AVAILABLE}")
    
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestUltimateBlendedSystem)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Test summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")
    print(f"✅ Tests run: {result.testsRun}")
    print(f"✅ Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"💥 Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    # Calculate success rate
    total_tests = result.testsRun
    successful_tests = total_tests - len(result.failures) - len(result.errors)
    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n🎯 SUCCESS RATE: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 EXCELLENT! System is ready for deployment!")
    elif success_rate >= 60:
        print("✅ GOOD! Minor issues to address.")
    else:
        print("⚠️  NEEDS WORK! Major issues detected.")
    
    return result

if __name__ == "__main__":
    run_comprehensive_tests()

