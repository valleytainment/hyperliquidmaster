"""
Main Application Entry Point for Hyperliquid Trading Bot
🚀 ULTIMATE VERSION with maximum optimizations and enhancements
"""

import os
import sys
import time
import argparse
import asyncio
import threading
import getpass
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import our modules
from core.api_fixed import EnhancedHyperliquidAPI  # Use the fixed API implementation
from gui.enhanced_gui_fixed_v2 import TradingDashboard  # Use the fixed GUI implementation
from utils.logger import setup_logging, get_logger
from utils.config_manager import ConfigManager
from utils.security_fixed import SecurityManager
from risk_management.risk_manager import RiskManager
from strategies.base_strategy import Strategy
from strategies.bb_rsi_adx import BBRSIADXStrategy
from strategies.hull_suite import HullSuiteStrategy
from backtesting.backtest_engine import BacktestEngine

# Setup logging
setup_logging()
logger = get_logger(__name__)


class HyperliquidTradingBot:
    """Main trading bot application"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the trading bot
        
        Args:
            config_path: Path to configuration file
        """
        # Set default config path if not provided
        if not config_path:
            config_path = os.path.join("config", "config.yaml")
        
        # Initialize components
        self.config_manager = ConfigManager(config_path)
        self.security_manager = SecurityManager()
        
        # Get configuration
        self.config = self.config_manager.get_config()
        self.testnet = self.config.get('trading', {}).get('testnet', False)
        
        # Initialize API client
        self.api_client = EnhancedHyperliquidAPI(config_path, self.testnet)
        
        # Initialize risk management
        self.risk_manager = RiskManager(self.config_manager)
        
        # Initialize strategies
        self.strategies = {}
        self.init_strategies()
        
        # Initialize backtest engine
        self.backtest_engine = BacktestEngine(
            initial_capital=10000.0,
            strategies=self.strategies,
            risk_manager=self.risk_manager
        )
        
        # GUI
        self.gui = None
        
        logger.info("Hyperliquid Trading Bot initialized")
    
    def init_strategies(self):
        """Initialize trading strategies"""
        # Get active strategies from config
        active_strategies = self.config_manager.trading.active_strategies
        
        # Initialize BB RSI ADX strategy
        if "bb_rsi_adx" in active_strategies:
            bb_rsi_config = self.config_manager.get_strategy_config("bb_rsi_adx")
            if bb_rsi_config:
                self.strategies["BB_RSI_ADX"] = BBRSIADXStrategy(
                    api_client=self.api_client,
                    risk_manager=self.risk_manager,
                    config=bb_rsi_config
                )
                logger.info("BB RSI ADX strategy initialized")
        
        # Initialize Hull Suite strategy
        if "hull_suite" in active_strategies:
            hull_config = self.config_manager.get_strategy_config("hull_suite")
            if hull_config:
                self.strategies["Hull_Suite"] = HullSuiteStrategy(
                    api_client=self.api_client,
                    risk_manager=self.risk_manager,
                    config=hull_config
                )
                logger.info("Hull Suite strategy initialized")
        
        logger.info(f"Initialized {len(self.strategies)} strategies")
    
    def start_gui(self):
        """Start the GUI interface"""
        try:
            # Initialize the GUI with the fixed implementation
            self.gui = TradingDashboard()
            
            # Pass components to GUI
            self.gui.api = self.api_client
            self.gui.config_manager = self.config_manager
            self.gui.security_manager = self.security_manager
            self.gui.risk_manager = self.risk_manager
            self.gui.strategies = self.strategies
            self.gui.backtest_engine = self.backtest_engine
            
            logger.info("Starting GUI interface")
            
            # Handle window closing
            self.gui.root.protocol("WM_DELETE_WINDOW", self.gui.on_closing)
            
            # Start the GUI main loop
            self.gui.root.mainloop()
            
        except Exception as e:
            logger.error(f"GUI error: {e}")
    
    def start_cli(self):
        """Start the command-line interface"""
        try:
            logger.info("Starting CLI interface")
            
            print("\n🚀 Hyperliquid Trading Bot CLI")
            print("----------------------------")
            
            # Check if we have saved credentials
            private_key = self.security_manager.get_private_key(method="auto")
            wallet_address = self.config_manager.get('trading.wallet_address')
            
            # If no credentials, prompt for them
            if not private_key or not wallet_address:
                print("\nNo saved credentials found. Please enter your credentials:")
                
                # Prompt for wallet address if not configured
                if not wallet_address:
                    wallet_address = input("Enter your wallet address (0x...): ").strip()
                    if wallet_address:
                        # Save to config
                        self.config_manager.set('trading.wallet_address', wallet_address)
                        self.config_manager.save_config()
                        print("✅ Wallet address saved to config")
                    else:
                        print("❌ Wallet address is required")
                        return
                else:
                    print(f"Using saved wallet address: {wallet_address}")
                
                # Prompt for private key if not found
                if not private_key:
                    print("\nEnter your private key (input will be hidden):")
                    print("Tip: You can paste your key even though no characters will be shown")
                    private_key = getpass.getpass("Private key: ")
                    
                    if private_key:
                        # Save private key securely
                        success = self.security_manager.store_private_key(private_key)
                        if success:
                            print("✅ Private key saved securely")
                        else:
                            print("❌ Failed to save private key")
                            return
                    else:
                        print("❌ Private key is required")
                        return
                else:
                    print("Using saved private key (hidden)")
            
            print("\nAuthenticating with Hyperliquid...")
            
            # Authenticate with API
            success = self.api_client.authenticate(private_key, wallet_address)
            if not success:
                print("❌ Authentication failed. Please check your credentials.")
                
                # Ask if user wants to try again with new credentials
                retry = input("\nDo you want to try again with new credentials? (y/n): ").strip().lower()
                if retry == 'y':
                    # Clear saved credentials
                    self.security_manager.clear_private_key()
                    
                    # Prompt for new credentials
                    wallet_address = input("Enter your wallet address (0x...): ").strip()
                    if wallet_address:
                        # Save to config
                        self.config_manager.set('trading.wallet_address', wallet_address)
                        self.config_manager.save_config()
                    else:
                        print("❌ Wallet address is required")
                        return
                    
                    print("\nEnter your private key (input will be hidden):")
                    print("Tip: You can paste your key even though no characters will be shown")
                    private_key = getpass.getpass("Private key: ")
                    
                    if not private_key:
                        print("❌ Private key is required")
                        return
                    
                    # Try authentication again
                    success = self.api_client.authenticate(private_key, wallet_address)
                    if success:
                        # Save private key securely
                        self.security_manager.store_private_key(private_key)
                        print("✅ Authentication successful")
                    else:
                        print("❌ Authentication failed again. Exiting.")
                        return
                else:
                    return
            else:
                print("✅ Authentication successful")
            
            # Get account info
            account = self.api_client.get_account_state()
            print(f"\nAccount Value: ${account.get('account_value', 0):.2f}")
            print(f"Margin Used: ${account.get('total_margin_used', 0):.2f}")
            
            # Simple CLI loop
            print("\nType 'help' for available commands, 'exit' to quit")
            
            while True:
                cmd = input("\n> ").strip().lower()
                
                if cmd == "exit":
                    break
                elif cmd == "help":
                    print("\nAvailable commands:")
                    print("  account  - Show account information")
                    print("  positions - Show open positions")
                    print("  orders   - Show open orders")
                    print("  market   - Show market data")
                    print("  key      - View or update private key")
                    print("  wallet   - View or update wallet address")
                    print("  exit     - Exit the application")
                elif cmd == "account":
                    account = self.api_client.get_account_state()
                    print(f"\nAccount Value: ${account.get('account_value', 0):.2f}")
                    print(f"Margin Used: ${account.get('total_margin_used', 0):.2f}")
                elif cmd == "positions":
                    account = self.api_client.get_account_state()
                    positions = account.get('positions', [])
                    if positions:
                        print("\nOpen Positions:")
                        for pos in positions:
                            print(f"  {pos['coin']}: {pos['size']} @ ${pos['entry_px']:.2f} (PnL: ${pos['unrealized_pnl']:.2f})")
                    else:
                        print("\nNo open positions")
                elif cmd == "orders":
                    account = self.api_client.get_account_state()
                    orders = account.get('orders', [])
                    if orders:
                        print("\nOpen Orders:")
                        for order in orders:
                            print(f"  {order['coin']} {order['side']} {order['sz']} @ ${order['limit_px']:.2f}")
                    else:
                        print("\nNo open orders")
                elif cmd == "market":
                    coin = input("Enter coin symbol (e.g., BTC): ").strip().upper()
                    market_data = self.api_client.get_market_data(coin)
                    print(f"\n{coin} Price: ${market_data.get('price', 0):.2f}")
                    print(f"24h Volume: ${market_data.get('volume_24h', 0):.2f}")
                    print(f"Funding Rate: {market_data.get('funding_rate', 0):.6f}")
                elif cmd == "key":
                    print("\nPrivate Key Management:")
                    print("1. View private key (masked)")
                    print("2. Update private key")
                    print("3. Clear saved private key")
                    print("4. Back to main menu")
                    
                    key_cmd = input("Select option (1-4): ").strip()
                    
                    if key_cmd == "1":
                        # Show masked private key
                        key = self.security_manager.get_private_key()
                        if key:
                            # Show first 6 and last 4 characters
                            masked_key = key[:6] + "..." + key[-4:]
                            print(f"\nPrivate Key (masked): {masked_key}")
                        else:
                            print("\nNo private key saved")
                    elif key_cmd == "2":
                        # Update private key
                        print("\nEnter new private key (input will be hidden):")
                        print("Tip: You can paste your key even though no characters will be shown")
                        new_key = getpass.getpass("New private key: ")
                        
                        if new_key:
                            # Save private key securely
                            success = self.security_manager.store_private_key(new_key)
                            if success:
                                print("✅ New private key saved securely")
                            else:
                                print("❌ Failed to save private key")
                        else:
                            print("❌ No private key entered")
                    elif key_cmd == "3":
                        # Clear private key
                        confirm = input("\nAre you sure you want to clear your saved private key? (y/n): ").strip().lower()
                        if confirm == 'y':
                            success = self.security_manager.clear_private_key()
                            if success:
                                print("✅ Private key cleared")
                            else:
                                print("❌ Failed to clear private key")
                        else:
                            print("Operation cancelled")
                    elif key_cmd == "4":
                        # Back to main menu
                        pass
                    else:
                        print("Invalid option")
                elif cmd == "wallet":
                    print("\nWallet Address Management:")
                    print("1. View wallet address")
                    print("2. Update wallet address")
                    print("3. Back to main menu")
                    
                    wallet_cmd = input("Select option (1-3): ").strip()
                    
                    if wallet_cmd == "1":
                        # Show wallet address
                        address = self.config_manager.get('trading.wallet_address')
                        if address:
                            print(f"\nWallet Address: {address}")
                        else:
                            print("\nNo wallet address saved")
                    elif wallet_cmd == "2":
                        # Update wallet address
                        new_address = input("\nEnter new wallet address (0x...): ").strip()
                        
                        if new_address:
                            # Save to config
                            self.config_manager.set('trading.wallet_address', new_address)
                            self.config_manager.save_config()
                            print("✅ New wallet address saved")
                        else:
                            print("❌ No wallet address entered")
                    elif wallet_cmd == "3":
                        # Back to main menu
                        pass
                    else:
                        print("Invalid option")
                else:
                    print("Unknown command. Type 'help' for available commands.")
            
            print("\nExiting CLI...")
            
        except KeyboardInterrupt:
            print("\nExiting CLI...")
        except Exception as e:
            logger.error(f"CLI error: {e}")
    
    def start_bot(self):
        """Start the trading bot in automated mode"""
        try:
            logger.info("Starting automated trading bot")
            
            # Authenticate
            private_key = self.security_manager.get_private_key()
            wallet_address = self.config_manager.trading.wallet_address
            
            if not private_key or not wallet_address:
                logger.error("Private key or wallet address not configured")
                return
            
            # Authenticate with API
            success = self.api_client.authenticate(private_key, wallet_address)
            if not success:
                logger.error("Authentication failed")
                return
            
            # Start strategies
            for name, strategy in self.strategies.items():
                strategy.start()
            
            # Keep running until interrupted
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
            finally:
                # Stop strategies
                for name, strategy in self.strategies.items():
                    strategy.stop()
            
        except Exception as e:
            logger.error(f"Bot error: {e}")
    
    def run_backtest(self, start_date: str, end_date: str, strategies: List[str] = None):
        """
        Run backtest for specified strategies
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            strategies: List of strategy names to backtest
        """
        try:
            logger.info(f"Running backtest from {start_date} to {end_date}")
            
            # Select strategies to backtest
            backtest_strategies = {}
            if strategies:
                for name in strategies:
                    if name in self.strategies:
                        backtest_strategies[name] = self.strategies[name]
            else:
                backtest_strategies = self.strategies
            
            if not backtest_strategies:
                logger.error("No strategies selected for backtest")
                return
            
            # Run backtest
            results = self.backtest_engine.run_backtest(
                start_date=start_date,
                end_date=end_date,
                strategies=backtest_strategies
            )
            
            # Print results
            print("\nBacktest Results:")
            print(f"Initial Capital: ${results['initial_capital']:.2f}")
            print(f"Final Capital: ${results['final_capital']:.2f}")
            print(f"Total Return: {results['total_return']:.2f}%")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
            print(f"Win Rate: {results['win_rate']:.2f}%")
            print(f"Total Trades: {results['total_trades']}")
            
            # Save results
            self.backtest_engine.save_results("backtest_results.json")
            logger.info("Backtest completed and results saved")
            
        except Exception as e:
            logger.error(f"Backtest error: {e}")


def main():
    """Main entry point"""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Hyperliquid Trading Bot")
        parser.add_argument("--config", type=str, help="Path to configuration file")
        parser.add_argument("--mode", type=str, choices=["gui", "cli", "bot", "backtest"], default="gui", help="Operation mode")
        parser.add_argument("--start-date", type=str, help="Backtest start date (YYYY-MM-DD)")
        parser.add_argument("--end-date", type=str, help="Backtest end date (YYYY-MM-DD)")
        parser.add_argument("--strategies", type=str, help="Comma-separated list of strategies for backtest")
        
        args = parser.parse_args()
        
        # Initialize the trading bot
        bot = HyperliquidTradingBot(args.config)
        
        # Run in specified mode
        if args.mode == "gui":
            bot.start_gui()
        elif args.mode == "cli":
            bot.start_cli()
        elif args.mode == "bot":
            bot.start_bot()
        elif args.mode == "backtest":
            if not args.start_date or not args.end_date:
                logger.error("Backtest requires start-date and end-date parameters")
                return
            
            strategies = args.strategies.split(",") if args.strategies else None
            bot.run_backtest(args.start_date, args.end_date, strategies)
        
        logger.info("Application completed successfully")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

