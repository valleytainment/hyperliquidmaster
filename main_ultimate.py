#!/usr/bin/env python3
"""
Main Application - Ultimate Hyperliquid Master
----------------------------------------------
Complete trading system with neural networks, RL optimization,
live trading capabilities, and professional GUI interface.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler('hyperliquid_master.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='Hyperliquid Master Trading Bot')
    parser.add_argument('--mode', choices=['gui', 'cli', 'setup', 'trading', 'backtest'], 
                       default='gui', help='Application mode')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--symbol', type=str, default='BTC-USD-PERP', help='Trading symbol')
    parser.add_argument('--strategy', type=str, default='enhanced_neural', help='Trading strategy')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'gui':
            run_gui_mode(args)
        elif args.mode == 'cli':
            run_cli_mode(args)
        elif args.mode == 'setup':
            run_setup_mode(args)
        elif args.mode == 'trading':
            run_trading_mode(args)
        elif args.mode == 'backtest':
            run_backtest_mode(args)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

def run_gui_mode(args):
    """Run GUI mode with ultimate production interface"""
    try:
        logger.info("Starting GUI mode...")
        
        # Import GUI components
        from gui.ultimate_production_gui import UltimateProductionGUI
        from core.ultimate_trading_engine import ProductionTradingBot
        
        # Create trading bot
        config = load_config(args.config)
        trading_bot = ProductionTradingBot(config)
        
        # Initialize bot
        if not trading_bot.initialize():
            logger.error("Failed to initialize trading bot")
            return
        
        # Create and run GUI
        gui = UltimateProductionGUI(trading_bot)
        trading_bot.gui = gui
        
        logger.info("Launching Ultimate Production GUI...")
        gui.run()
        
    except ImportError as e:
        logger.error(f"GUI dependencies missing: {e}")
        logger.info("Falling back to CLI mode...")
        run_cli_mode(args)
    except Exception as e:
        logger.error(f"GUI mode failed: {e}")
        raise

def run_cli_mode(args):
    """Run CLI mode for headless operation"""
    try:
        logger.info("Starting CLI mode...")
        
        from core.ultimate_trading_engine import ProductionTradingBot
        
        # Create trading bot
        config = load_config(args.config)
        trading_bot = ProductionTradingBot(config)
        
        # Initialize bot
        if not trading_bot.initialize():
            logger.error("Failed to initialize trading bot")
            return
        
        # Connect to API
        if not trading_bot.connect():
            logger.error("Failed to connect to API")
            return
        
        logger.info("CLI mode ready. Bot initialized and connected.")
        logger.info("Available commands:")
        logger.info("  start <mode> <strategy> - Start automated trading")
        logger.info("  stop - Stop automated trading")
        logger.info("  status - Show current status")
        logger.info("  positions - Show current positions")
        logger.info("  metrics - Show performance metrics")
        logger.info("  quit - Exit application")
        
        # CLI command loop
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    break
                elif command == 'status':
                    show_status(trading_bot)
                elif command == 'positions':
                    show_positions(trading_bot)
                elif command == 'metrics':
                    show_metrics(trading_bot)
                elif command.startswith('start'):
                    parts = command.split()
                    mode = parts[1] if len(parts) > 1 else 'perp'
                    strategy = parts[2] if len(parts) > 2 else 'enhanced_neural'
                    trading_bot.start_automation(mode, strategy)
                    logger.info(f"Started automation: {mode} mode, {strategy} strategy")
                elif command == 'stop':
                    trading_bot.stop_automation()
                    logger.info("Stopped automation")
                else:
                    logger.info("Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Command error: {e}")
        
        # Cleanup
        trading_bot.shutdown()
        logger.info("CLI mode ended")
        
    except Exception as e:
        logger.error(f"CLI mode failed: {e}")
        raise

def run_setup_mode(args):
    """Run setup mode for initial configuration"""
    try:
        logger.info("Starting setup mode...")
        
        from utils.config_manager import ConfigManager
        from utils.security import SecurityManager
        
        # Initialize managers
        config_manager = ConfigManager()
        security_manager = SecurityManager()
        
        print("\n🚀 HYPERLIQUID MASTER - Initial Setup")
        print("=" * 50)
        
        # Get user credentials
        print("\n1. API Configuration")
        address = input("Enter your wallet address (0x...): ").strip()
        private_key = input("Enter your private key (0x...): ").strip()
        
        if not address or not private_key:
            logger.error("Address and private key are required")
            return
        
        # Trading preferences
        print("\n2. Trading Preferences")
        symbol = input("Default trading symbol [BTC-USD-PERP]: ").strip() or "BTC-USD-PERP"
        mode = input("Default trading mode (spot/perp) [perp]: ").strip() or "perp"
        capital = input("Starting capital [$100]: ").strip() or "100"
        
        try:
            capital = float(capital)
        except ValueError:
            capital = 100.0
        
        # Risk management
        print("\n3. Risk Management")
        stop_loss = input("Stop loss percentage [2.0]: ").strip() or "2.0"
        take_profit = input("Take profit percentage [4.0]: ").strip() or "4.0"
        position_size = input("Default position size [$20]: ").strip() or "20"
        
        try:
            stop_loss = float(stop_loss) / 100
            take_profit = float(take_profit) / 100
            position_size = float(position_size)
        except ValueError:
            stop_loss = 0.02
            take_profit = 0.04
            position_size = 20.0
        
        # Create configuration
        config = {
            "account_address": address,
            "secret_key": private_key,
            "api_url": "https://api.hyperliquid.xyz",
            "trade_symbol": symbol,
            "trade_mode": mode,
            "starting_capital": capital,
            "stop_loss_pct": stop_loss,
            "take_profit_pct": take_profit,
            "manual_entry_size": position_size,
            "use_manual_entry_size": True,
            "max_positions": 3,
            "circuit_breaker_threshold": 0.1,
            "poll_interval_seconds": 2,
            "min_trade_interval": 60
        }
        
        # Save configuration
        config_manager.save_config(config)
        
        print("\n✅ Setup completed successfully!")
        print(f"Configuration saved to: {config_manager.config_file}")
        print("\nYou can now run the bot with:")
        print("  python main.py --mode gui")
        print("  python main.py --mode cli")
        
    except Exception as e:
        logger.error(f"Setup mode failed: {e}")
        raise

def run_trading_mode(args):
    """Run pure trading mode without GUI"""
    try:
        logger.info("Starting trading mode...")
        
        from core.ultimate_trading_engine import ProductionTradingBot
        
        # Create trading bot
        config = load_config(args.config)
        trading_bot = ProductionTradingBot(config)
        
        # Initialize and connect
        if not trading_bot.initialize():
            logger.error("Failed to initialize trading bot")
            return
        
        if not trading_bot.connect():
            logger.error("Failed to connect to API")
            return
        
        # Start automated trading
        mode = config.get("trade_mode", "perp")
        strategy = args.strategy
        
        logger.info(f"Starting automated trading: {mode} mode, {strategy} strategy")
        trading_bot.start_automation(mode, strategy)
        
        # Keep running until interrupted
        try:
            while True:
                import time
                time.sleep(10)
                
                # Show periodic status
                metrics = trading_bot.get_performance_metrics()
                logger.info(f"Status - Trades: {metrics.get('total_trades', 0)}, "
                          f"Win Rate: {metrics.get('win_rate', 0):.1f}%, "
                          f"P&L: ${metrics.get('total_pnl', 0):.2f}")
                
        except KeyboardInterrupt:
            logger.info("Trading interrupted by user")
        
        # Cleanup
        trading_bot.stop_automation()
        trading_bot.shutdown()
        logger.info("Trading mode ended")
        
    except Exception as e:
        logger.error(f"Trading mode failed: {e}")
        raise

def run_backtest_mode(args):
    """Run backtest mode"""
    try:
        logger.info("Starting backtest mode...")
        
        from backtesting.backtest_engine import BacktestEngine
        from strategies.enhanced_neural_strategy import EnhancedNeuralStrategy
        
        # Create backtest engine
        config = load_config(args.config)
        backtest_engine = BacktestEngine(config.get("starting_capital", 10000))
        
        # Create strategy
        strategy = EnhancedNeuralStrategy(None, None, config)
        
        # Run backtest
        symbol = args.symbol
        start_date = "2024-01-01"
        end_date = "2024-12-31"
        
        logger.info(f"Running backtest for {symbol} from {start_date} to {end_date}")
        results = backtest_engine.run_backtest(strategy, symbol, start_date, end_date)
        
        # Display results
        print("\n📊 BACKTEST RESULTS")
        print("=" * 50)
        print(f"Symbol: {symbol}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Total Return: {results.get('total_return', 0):.2f}%")
        print(f"Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"Total Trades: {results.get('total_trades', 0)}")
        print(f"Win Rate: {results.get('win_rate', 0):.2f}%")
        
    except Exception as e:
        logger.error(f"Backtest mode failed: {e}")
        raise

def load_config(config_path: str = None) -> dict:
    """Load configuration from file or use defaults"""
    try:
        if config_path and os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Try to load from default locations
        from utils.config_manager import ConfigManager
        config_manager = ConfigManager()
        return config_manager.load_config()
        
    except Exception as e:
        logger.warning(f"Failed to load config: {e}. Using defaults.")
        return {
            "account_address": "0x306D29F56EA1345c7E6F1ff27657ba05cEE15D4F",
            "secret_key": "43ba46de58067dd1ef3794c653bf3b11fa78866623cc515a5aff5f4be31fd3b8",
            "api_url": "https://api.hyperliquid.xyz",
            "trade_symbol": "BTC-USD-PERP",
            "trade_mode": "perp",
            "starting_capital": 100.0,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "manual_entry_size": 20.0,
            "use_manual_entry_size": True,
            "max_positions": 3,
            "circuit_breaker_threshold": 0.1,
            "poll_interval_seconds": 2,
            "min_trade_interval": 60
        }

def show_status(trading_bot):
    """Show current bot status"""
    try:
        equity = trading_bot.api.get_equity()
        positions = trading_bot.api.get_user_positions()
        
        print(f"\n📊 Current Status")
        print(f"Equity: ${equity:.2f}")
        print(f"Open Positions: {len(positions)}")
        print(f"Automation: {'Running' if trading_bot.engine.automation_running else 'Stopped'}")
        
    except Exception as e:
        logger.error(f"Error showing status: {e}")

def show_positions(trading_bot):
    """Show current positions"""
    try:
        positions = trading_bot.api.get_user_positions()
        
        if not positions:
            print("\n📈 No open positions")
            return
        
        print(f"\n📈 Open Positions ({len(positions)})")
        print("-" * 60)
        for pos in positions:
            side = "LONG" if pos["side"] == 1 else "SHORT"
            print(f"{pos['symbol']}: {side} {pos['size']:.4f} @ ${pos['entryPrice']:.4f}")
        
    except Exception as e:
        logger.error(f"Error showing positions: {e}")

def show_metrics(trading_bot):
    """Show performance metrics"""
    try:
        metrics = trading_bot.get_performance_metrics()
        
        print(f"\n📊 Performance Metrics")
        print("-" * 30)
        print(f"Total Trades: {metrics.get('total_trades', 0)}")
        print(f"Win Rate: {metrics.get('win_rate', 0):.2f}%")
        print(f"Total P&L: ${metrics.get('total_pnl', 0):.2f}")
        print(f"Total Return: {metrics.get('total_return', 0):.2f}%")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        print(f"Current Equity: ${metrics.get('current_equity', 0):.2f}")
        
    except Exception as e:
        logger.error(f"Error showing metrics: {e}")

if __name__ == "__main__":
    main()

