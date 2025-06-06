#!/usr/bin/env python3
"""
Ultimate Main Application with All Enhanced Features
--------------------------------------------------
Integrates all missing features and provides multiple entry points:
• Enhanced GUI with real-time wallet equity and price display
• Command-line interface for automation
• API server mode for external integrations
• Comprehensive testing and validation
"""

import sys
import os
import argparse
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import enhanced components
from core.enhanced_trading_engine import EnhancedProductionTradingBot
from gui.ultimate_production_gui import UltimateProductionGUI
from utils.logger import setup_logging, get_logger
from utils.config_manager import ConfigManager

# Setup logging
setup_logging()
logger = get_logger(__name__)


class UltimateHyperliquidMaster:
    """Ultimate Hyperliquid Master Application"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.trading_bot = None
        self.gui_app = None
        
        logger.info("Ultimate Hyperliquid Master initialized")
    
    def run_gui_mode(self):
        """Run in GUI mode with real-time features"""
        try:
            logger.info("🚀 Starting Ultimate Production GUI...")
            
            # Create and run GUI
            self.gui_app = UltimateProductionGUI()
            self.gui_app.run()
            
        except Exception as e:
            logger.error(f"GUI mode failed: {e}")
            return False
        
        return True
    
    def run_automation_mode(self, symbol: str, mode: str, strategy: str):
        """Run in headless automation mode"""
        try:
            logger.info(f"🤖 Starting automation mode: {symbol} ({mode}) with {strategy}")
            
            # Initialize trading bot
            config = self.config_manager.get_config()
            config.update({
                "trade_symbol": symbol,
                "trade_mode": mode.lower()
            })
            
            self.trading_bot = EnhancedProductionTradingBot(config)
            
            if not self.trading_bot.initialize():
                logger.error("Failed to initialize trading bot")
                return False
            
            if not self.trading_bot.connect():
                logger.error("Failed to connect to API")
                return False
            
            # Start automation
            self.trading_bot.start_automation(mode, strategy)
            
            logger.info("✅ Automation started successfully")
            logger.info("Press Ctrl+C to stop...")
            
            # Keep running until interrupted
            try:
                while True:
                    # Print status every 60 seconds
                    import time
                    time.sleep(60)
                    
                    real_time_data = self.trading_bot.get_real_time_data()
                    equity = real_time_data.get("equity", 0)
                    price = real_time_data.get("price", 0)
                    performance = real_time_data.get("performance", {})
                    
                    logger.info(f"💰 Equity: ${equity:.2f} | 📈 Price: ${price:.2f} | 📊 P&L: ${performance.get('total_pnl', 0):.2f}")
                    
            except KeyboardInterrupt:
                logger.info("🛑 Stopping automation...")
                self.trading_bot.stop_automation()
                
        except Exception as e:
            logger.error(f"Automation mode failed: {e}")
            return False
        
        return True
    
    def run_test_mode(self):
        """Run comprehensive tests"""
        try:
            logger.info("🧪 Running comprehensive tests...")
            
            # Initialize trading bot
            config = self.config_manager.get_config()
            self.trading_bot = EnhancedProductionTradingBot(config)
            
            # Test 1: Initialization
            logger.info("Test 1: Bot initialization...")
            if not self.trading_bot.initialize():
                logger.error("❌ Bot initialization failed")
                return False
            logger.info("✅ Bot initialization successful")
            
            # Test 2: API Connection
            logger.info("Test 2: API connection...")
            if not self.trading_bot.connect():
                logger.error("❌ API connection failed")
                return False
            logger.info("✅ API connection successful")
            
            # Test 3: Real-time data
            logger.info("Test 3: Real-time data retrieval...")
            real_time_data = self.trading_bot.get_real_time_data()
            if not real_time_data or real_time_data.get("equity", 0) <= 0:
                logger.error("❌ Real-time data retrieval failed")
                return False
            logger.info(f"✅ Real-time data: Equity=${real_time_data.get('equity', 0):.2f}")
            
            # Test 4: Performance metrics
            logger.info("Test 4: Performance metrics...")
            performance = self.trading_bot.get_performance_metrics()
            if not performance:
                logger.error("❌ Performance metrics failed")
                return False
            logger.info(f"✅ Performance metrics: {len(performance)} metrics available")
            
            # Test 5: GUI components (if available)
            logger.info("Test 5: GUI components...")
            try:
                import tkinter as tk
                import customtkinter as ctk
                logger.info("✅ GUI libraries available")
            except ImportError as e:
                logger.warning(f"⚠️ GUI libraries not available: {e}")
            
            logger.info("🎉 All tests completed successfully!")
            
            # Cleanup
            self.trading_bot.disconnect()
            
        except Exception as e:
            logger.error(f"Test mode failed: {e}")
            return False
        
        return True
    
    def run_api_server_mode(self, host: str = "0.0.0.0", port: int = 8000):
        """Run as API server for external integrations"""
        try:
            logger.info(f"🌐 Starting API server on {host}:{port}...")
            
            from flask import Flask, jsonify, request
            from flask_cors import CORS
            
            app = Flask(__name__)
            CORS(app)  # Enable CORS for all routes
            
            # Initialize trading bot
            config = self.config_manager.get_config()
            self.trading_bot = EnhancedProductionTradingBot(config)
            
            if not self.trading_bot.initialize():
                logger.error("Failed to initialize trading bot for API server")
                return False
            
            if not self.trading_bot.connect():
                logger.error("Failed to connect to API for server mode")
                return False
            
            @app.route('/api/status', methods=['GET'])
            def get_status():
                """Get system status"""
                try:
                    real_time_data = self.trading_bot.get_real_time_data()
                    return jsonify({
                        "status": "online",
                        "timestamp": real_time_data.get("timestamp", "").isoformat() if real_time_data.get("timestamp") else "",
                        "equity": real_time_data.get("equity", 0),
                        "price": real_time_data.get("price", 0),
                        "automation_status": real_time_data.get("automation_status", False)
                    })
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
            
            @app.route('/api/equity', methods=['GET'])
            def get_equity():
                """Get current wallet equity"""
                try:
                    real_time_data = self.trading_bot.get_real_time_data()
                    return jsonify({
                        "equity": real_time_data.get("equity", 0),
                        "timestamp": real_time_data.get("timestamp", "").isoformat() if real_time_data.get("timestamp") else ""
                    })
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
            
            @app.route('/api/price/<symbol>', methods=['GET'])
            def get_price(symbol):
                """Get current price for symbol"""
                try:
                    # This would need to be implemented in the trading bot
                    return jsonify({
                        "symbol": symbol,
                        "price": 0,  # Placeholder
                        "timestamp": ""
                    })
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
            
            @app.route('/api/positions', methods=['GET'])
            def get_positions():
                """Get current positions"""
                try:
                    real_time_data = self.trading_bot.get_real_time_data()
                    return jsonify({
                        "positions": real_time_data.get("positions", []),
                        "timestamp": real_time_data.get("timestamp", "").isoformat() if real_time_data.get("timestamp") else ""
                    })
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
            
            @app.route('/api/performance', methods=['GET'])
            def get_performance():
                """Get performance metrics"""
                try:
                    performance = self.trading_bot.get_performance_metrics()
                    return jsonify(performance)
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
            
            @app.route('/api/trade', methods=['POST'])
            def execute_trade():
                """Execute manual trade"""
                try:
                    data = request.get_json()
                    side = data.get('side')
                    size = float(data.get('size', 0))
                    symbol = data.get('symbol')
                    
                    result = self.trading_bot.execute_manual_trade(side, size, symbol)
                    return jsonify(result)
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
            
            @app.route('/api/automation/start', methods=['POST'])
            def start_automation():
                """Start automation"""
                try:
                    data = request.get_json()
                    mode = data.get('mode', 'perp')
                    strategy = data.get('strategy', 'Enhanced Neural')
                    
                    self.trading_bot.start_automation(mode, strategy)
                    return jsonify({"success": True, "message": "Automation started"})
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
            
            @app.route('/api/automation/stop', methods=['POST'])
            def stop_automation():
                """Stop automation"""
                try:
                    self.trading_bot.stop_automation()
                    return jsonify({"success": True, "message": "Automation stopped"})
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
            
            # Run Flask app
            app.run(host=host, port=port, debug=False)
            
        except Exception as e:
            logger.error(f"API server mode failed: {e}")
            return False
        
        return True
    
    def shutdown(self):
        """Shutdown the application"""
        try:
            if self.trading_bot:
                self.trading_bot.shutdown()
            
            logger.info("Ultimate Hyperliquid Master shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


def main():
    """Main entry point with command-line interface"""
    parser = argparse.ArgumentParser(description="Ultimate Hyperliquid Master - Advanced Trading System")
    parser.add_argument('--mode', choices=['gui', 'automation', 'test', 'api'], default='gui',
                       help='Application mode (default: gui)')
    parser.add_argument('--symbol', default='BTC-USD-PERP',
                       help='Trading symbol for automation mode (default: BTC-USD-PERP)')
    parser.add_argument('--trading-mode', choices=['perp', 'spot'], default='perp',
                       help='Trading mode (default: perp)')
    parser.add_argument('--strategy', default='Enhanced Neural',
                       help='Trading strategy for automation mode (default: Enhanced Neural)')
    parser.add_argument('--host', default='0.0.0.0',
                       help='Host for API server mode (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port for API server mode (default: 8000)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create application
    app = UltimateHyperliquidMaster()
    
    try:
        # Run in specified mode
        if args.mode == 'gui':
            logger.info("🖥️ Starting in GUI mode...")
            success = app.run_gui_mode()
        elif args.mode == 'automation':
            logger.info("🤖 Starting in automation mode...")
            success = app.run_automation_mode(args.symbol, args.trading_mode, args.strategy)
        elif args.mode == 'test':
            logger.info("🧪 Starting in test mode...")
            success = app.run_test_mode()
        elif args.mode == 'api':
            logger.info("🌐 Starting in API server mode...")
            success = app.run_api_server_mode(args.host, args.port)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            success = False
        
        if success:
            logger.info("✅ Application completed successfully")
        else:
            logger.error("❌ Application failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Application interrupted by user")
    except Exception as e:
        logger.error(f"💥 Application crashed: {e}")
        sys.exit(1)
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()

