#!/usr/bin/env python3
"""
ULTIMATE COMPREHENSIVE MAIN APPLICATION
=====================================
The complete main application that integrates ALL features from provided files:
• Ultimate comprehensive GUI with all toggles, buttons, tabs, and features
• Ultimate comprehensive trading engine with auto-connection
• Real-time wallet equity display and live token price feeds
• 24/7 automation with multiple strategies and modes
• Complete verification and testing system
• Always connected using default wallet credentials
"""

import sys
import os
import asyncio
import threading
import time
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import all our comprehensive modules
from gui.ultimate_comprehensive_gui import UltimateComprehensiveGUI
from core.ultimate_comprehensive_trading_engine import UltimateComprehensiveTradingEngine
from verification.ultimate_comprehensive_verification import UltimateComprehensiveVerificationSystem
from utils.logger import get_logger, TradingLogger
from utils.config_manager import ConfigManager

logger = get_logger(__name__)
trading_logger = TradingLogger(__name__)


class UltimateComprehensiveApplication:
    """Ultimate comprehensive application with all features"""
    
    def __init__(self):
        """Initialize the ultimate application"""
        self.trading_engine = None
        self.gui = None
        self.verification_system = None
        self.config_manager = ConfigManager()
        
        # Application state
        self.is_running = False
        self.auto_connected = False
        
        # Default credentials for auto-connection
        self.default_credentials = {
            "account_address": "0x306D29F56EA1345c7E6F1ff27657ba05cEE15D4F",
            "private_key": "43ba46de58067dd1ef3794c653bf3b11fa78866623cc515a5aff5f4be31fd3b8"
        }
        
        logger.info("🚀 Ultimate Comprehensive Application initialized")
    
    def initialize_all_components(self) -> bool:
        """Initialize all application components"""
        try:
            logger.info("🔧 Initializing all application components...")
            
            # Initialize trading engine
            logger.info("🔧 Initializing trading engine...")
            self.trading_engine = UltimateComprehensiveTradingEngine()
            
            if not self.trading_engine.initialize():
                logger.error("❌ Failed to initialize trading engine")
                return False
            
            logger.info("✅ Trading engine initialized")
            
            # Initialize verification system
            logger.info("🔧 Initializing verification system...")
            self.verification_system = UltimateComprehensiveVerificationSystem()
            logger.info("✅ Verification system initialized")
            
            # Auto-connect trading engine
            logger.info("🔗 Auto-connecting trading engine...")
            if self.trading_engine.connect():
                self.auto_connected = True
                logger.info("✅ Trading engine auto-connected successfully")
                
                # Get initial account info
                real_time_data = self.trading_engine.get_real_time_data()
                if real_time_data:
                    logger.info(f"💰 Initial Equity: ${real_time_data['equity']:.2f}")
                    logger.info(f"📊 Available Balance: ${real_time_data['available_balance']:.2f}")
            else:
                logger.warning("⚠️ Auto-connection failed, but continuing...")
            
            logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            return False
    
    def run_gui_mode(self):
        """Run the application in GUI mode"""
        try:
            logger.info("🖥️ Starting GUI mode...")
            
            # Initialize GUI with trading engine
            self.gui = UltimateComprehensiveGUI(self.trading_engine)
            
            # Set auto-connection status
            if self.auto_connected:
                self.gui.set_connection_status(True)
                self.gui.update_account_info()
            
            # Start GUI
            self.gui.run()
            
        except Exception as e:
            logger.error(f"❌ GUI mode failed: {e}")
    
    def run_headless_mode(self):
        """Run the application in headless mode"""
        try:
            logger.info("🤖 Starting headless mode...")
            
            if not self.auto_connected:
                logger.error("❌ Cannot run headless mode without connection")
                return
            
            # Start automation
            if self.trading_engine.start_automation("perpetual", "enhanced_neural"):
                logger.info("🚀 Automation started successfully")
                
                # Keep running and monitor
                try:
                    while True:
                        # Get real-time data
                        data = self.trading_engine.get_real_time_data()
                        if data:
                            logger.info(f"💰 Equity: ${data['equity']:.2f} | P&L: ${data['total_pnl']:.2f} | Automation: {'ON' if data['automation_running'] else 'OFF'}")
                        
                        time.sleep(60)  # Update every minute
                        
                except KeyboardInterrupt:
                    logger.info("🛑 Stopping automation...")
                    self.trading_engine.stop_automation()
            else:
                logger.error("❌ Failed to start automation")
            
        except Exception as e:
            logger.error(f"❌ Headless mode failed: {e}")
    
    def run_verification_mode(self):
        """Run comprehensive verification"""
        try:
            logger.info("🔍 Starting verification mode...")
            
            # Run comprehensive verification
            results = self.verification_system.run_comprehensive_verification()
            
            # Print detailed results
            print("\\n" + "=" * 100)
            print("🔍 ULTIMATE COMPREHENSIVE VERIFICATION RESULTS")
            print("=" * 100)
            
            # Overall summary
            print(f"📊 Total Tests: {results['total_tests']}")
            print(f"✅ Passed: {results['passed_tests']}")
            print(f"❌ Failed: {results['failed_tests']}")
            print(f"⏱️ Duration: {results['duration']:.2f} seconds")
            print(f"🎯 Success Rate: {(results['passed_tests']/results['total_tests']*100):.1f}%")
            print(f"🏆 Overall Result: {'✅ PASSED' if results['overall_success'] else '❌ FAILED'}")
            
            # System health
            if results["system_health"]:
                health = results["system_health"]
                print(f"💊 System Health: {health.overall_status} ({health.performance_score:.1f}%)")
                print(f"🔗 Connection: {health.connection_status}")
                print(f"🔧 API: {health.api_status}")
                print(f"⚙️ Trading Engine: {health.trading_engine_status}")
                print(f"🖥️ GUI: {health.gui_status}")
                print(f"🤖 Automation: {health.automation_status}")
            
            print("=" * 100)
            
            # Detailed test results
            if results["test_results"]:
                print("\\n📋 DETAILED TEST RESULTS:")
                print("-" * 100)
                
                for result in results["test_results"]:
                    status = "✅ PASS" if result.success else "❌ FAIL"
                    print(f"{status} | {result.test_name:<30} | {result.message}")
                    if not result.success and result.details:
                        print(f"     Details: {result.details}")
                
                print("-" * 100)
            
            # Recommendations
            if results["recommendations"]:
                print("\\n💡 RECOMMENDATIONS:")
                for i, rec in enumerate(results["recommendations"], 1):
                    print(f"{i}. {rec}")
            
            print("\\n" + "=" * 100)
            
            return results["overall_success"]
            
        except Exception as e:
            logger.error(f"❌ Verification mode failed: {e}")
            return False
    
    def run_trading_mode(self):
        """Run in trading mode with real-time monitoring"""
        try:
            logger.info("💹 Starting trading mode...")
            
            if not self.auto_connected:
                logger.error("❌ Cannot run trading mode without connection")
                return
            
            print("\\n" + "=" * 80)
            print("💹 ULTIMATE HYPERLIQUID MASTER - LIVE TRADING MODE")
            print("=" * 80)
            print(f"🔗 Connected to: Hyperliquid Mainnet")
            print(f"👤 Account: {self.default_credentials['account_address']}")
            print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            
            # Get initial data
            data = self.trading_engine.get_real_time_data()
            if data:
                print(f"💰 Initial Equity: ${data['equity']:.2f}")
                print(f"📊 Available Balance: ${data['available_balance']:.2f}")
                print(f"📈 Total P&L: ${data['total_pnl']:.2f}")
                print(f"🔄 Margin Used: ${data['margin_used']:.2f}")
            
            print("\\n🚀 Starting 24/7 automation...")
            
            # Start automation
            if self.trading_engine.start_automation("perpetual", "enhanced_neural"):
                print("✅ Automation started successfully!")
                print("\\n📊 LIVE MONITORING (Press Ctrl+C to stop):")
                print("-" * 80)
                
                try:
                    update_count = 0
                    while True:
                        # Get real-time data
                        data = self.trading_engine.get_real_time_data()
                        if data:
                            update_count += 1
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            
                            # Display real-time info
                            print(f"[{timestamp}] Equity: ${data['equity']:.2f} | "
                                  f"P&L: ${data['total_pnl']:.2f} | "
                                  f"Daily: ${data['daily_pnl']:.2f} | "
                                  f"Auto: {'🟢 ON' if data['automation_running'] else '🔴 OFF'} | "
                                  f"BTC: ${data.get('price', 0):.2f}")
                            
                            # Show positions if any
                            if data.get('positions'):
                                print(f"     📊 Positions: {len(data['positions'])} open")
                            
                            # Show performance metrics every 10 updates
                            if update_count % 10 == 0:
                                metrics = self.trading_engine.get_performance_metrics()
                                if metrics:
                                    print(f"     📈 Performance: {metrics['total_trades']} trades | "
                                          f"Win Rate: {metrics['win_rate']:.1f}% | "
                                          f"Max DD: {metrics['max_drawdown']:.2f}%")
                        
                        time.sleep(5)  # Update every 5 seconds
                        
                except KeyboardInterrupt:
                    print("\\n\\n🛑 Stopping automation...")
                    self.trading_engine.stop_automation()
                    print("✅ Automation stopped")
                    
                    # Final summary
                    final_data = self.trading_engine.get_real_time_data()
                    if final_data:
                        print("\\n📊 FINAL SUMMARY:")
                        print(f"💰 Final Equity: ${final_data['equity']:.2f}")
                        print(f"📈 Total P&L: ${final_data['total_pnl']:.2f}")
                        print(f"📊 Daily P&L: ${final_data['daily_pnl']:.2f}")
                        
                        final_metrics = self.trading_engine.get_performance_metrics()
                        if final_metrics:
                            print(f"🎯 Total Trades: {final_metrics['total_trades']}")
                            print(f"🏆 Win Rate: {final_metrics['win_rate']:.1f}%")
            else:
                print("❌ Failed to start automation")
            
        except Exception as e:
            logger.error(f"❌ Trading mode failed: {e}")
    
    def run_demo_mode(self):
        """Run in demo mode"""
        try:
            logger.info("🎮 Starting demo mode...")
            
            print("\\n" + "=" * 80)
            print("🎮 ULTIMATE HYPERLIQUID MASTER - DEMO MODE")
            print("=" * 80)
            print("This is a demonstration of all features without live trading.")
            print("=" * 80)
            
            # Show available features
            features = [
                "✅ Real-time wallet equity display",
                "✅ Live token price feeds (BTC, ETH, SOL, AVAX)",
                "✅ Advanced neural network strategies",
                "✅ 24/7 automation with multiple modes",
                "✅ Risk management and circuit breakers",
                "✅ Performance analytics and reporting",
                "✅ Professional GUI with all controls",
                "✅ Comprehensive verification system",
                "✅ Auto-connection with default wallet",
                "✅ Manual trading controls",
                "✅ Strategy backtesting",
                "✅ Real-time charts and monitoring"
            ]
            
            print("\\n🚀 AVAILABLE FEATURES:")
            for feature in features:
                print(f"  {feature}")
            
            print("\\n🎯 USAGE MODES:")
            print("  • GUI Mode: python main_ultimate_comprehensive.py --mode gui")
            print("  • Trading Mode: python main_ultimate_comprehensive.py --mode trading")
            print("  • Headless Mode: python main_ultimate_comprehensive.py --mode headless")
            print("  • Verification Mode: python main_ultimate_comprehensive.py --mode verify")
            
            if self.auto_connected:
                data = self.trading_engine.get_real_time_data()
                if data:
                    print("\\n💰 LIVE ACCOUNT DATA:")
                    print(f"  • Equity: ${data['equity']:.2f}")
                    print(f"  • Available Balance: ${data['available_balance']:.2f}")
                    print(f"  • Total P&L: ${data['total_pnl']:.2f}")
                    print(f"  • BTC Price: ${data.get('price', 0):.2f}")
            
            print("\\n" + "=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Demo mode failed: {e}")
    
    def shutdown(self):
        """Shutdown the application"""
        try:
            logger.info("🔄 Shutting down application...")
            
            # Shutdown trading engine
            if self.trading_engine:
                self.trading_engine.shutdown()
            
            # Close GUI if running
            if self.gui:
                self.gui.destroy()
            
            logger.info("✅ Application shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")


def main():
    """Main entry point"""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Ultimate Comprehensive Hyperliquid Master")
        parser.add_argument("--mode", choices=["gui", "headless", "trading", "verify", "demo"], 
                          default="demo", help="Application mode")
        parser.add_argument("--strategy", default="enhanced_neural", 
                          help="Trading strategy to use")
        parser.add_argument("--trading-mode", choices=["spot", "perpetual"], 
                          default="perpetual", help="Trading mode")
        
        args = parser.parse_args()
        
        print("🚀 Starting Ultimate Comprehensive Hyperliquid Master...")
        print(f"📋 Mode: {args.mode.upper()}")
        print(f"🎯 Strategy: {args.strategy}")
        print(f"💹 Trading Mode: {args.trading_mode}")
        
        # Initialize application
        app = UltimateComprehensiveApplication()
        
        # Initialize all components
        if not app.initialize_all_components():
            print("❌ Failed to initialize application components")
            return False
        
        # Run in specified mode
        try:
            if args.mode == "gui":
                app.run_gui_mode()
            elif args.mode == "headless":
                app.run_headless_mode()
            elif args.mode == "trading":
                app.run_trading_mode()
            elif args.mode == "verify":
                return app.run_verification_mode()
            elif args.mode == "demo":
                app.run_demo_mode()
            else:
                print(f"❌ Unknown mode: {args.mode}")
                return False
                
        except KeyboardInterrupt:
            print("\\n🛑 Application interrupted by user")
        except Exception as e:
            logger.error(f"❌ Application error: {e}")
            return False
        finally:
            app.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ Main application error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

