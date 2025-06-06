"""
Main Application Entry Point for Hyperliquid Trading Bot - FIXED VERSION
Integrates all components and provides unified interface with auto-connection
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
import signal
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.api_fixed_v2 import EnhancedHyperliquidAPI
from gui.enhanced_gui_auto_connect_v2 import TradingDashboard
from strategies.bb_rsi_adx_fixed import BBRSIADXStrategy
from strategies.hull_suite_fixed import HullSuiteStrategy
from backtesting.backtest_engine import BacktestEngine
from risk_management.risk_manager import RiskManager, RiskLimits
from utils.logger import get_logger, setup_logging, TradingLogger
from utils.config_manager_fixed import ConfigManager, TradingConfig
from utils.security_fixed_v2 import SecurityManager
from core.connection_manager_enhanced import EnhancedConnectionManager as ConnectionManager

logger = get_logger(__name__)

# Default connection credentials
DEFAULT_CREDENTIALS = {
    "account_address": "0x306D29F56EA1345c7E6F1ff27657ba05cEE15D4F",
    "secret_key": "43ba46de58067dd1ef3794c653bf3b11fa78866623cc515a5aff5f4be31fd3b8",
    "api_url": "https://api.hyperliquid.xyz"
}


class HyperliquidTradingBot:
    """
    Main Hyperliquid Trading Bot Application
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Initialize the trading bot
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config_manager = None
        self.security_manager = None
        self.connection_manager = None
        self.api = None
        self.risk_manager = None
        self.strategies = {}
        self.backtest_engine = None
        self.gui = None
        self.running = False
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Hyperliquid Trading Bot initialized")
    
    def _initialize_components(self):
        """
        Initialize all components
        """
        try:
            # Initialize configuration manager
            self.config_manager = ConfigManager(self.config_path)
            
            # Initialize security manager
            self.security_manager = SecurityManager()
            
            # Initialize connection manager with default credentials
            self.connection_manager = ConnectionManager(
                default_credentials=DEFAULT_CREDENTIALS
            )
            
            # Initialize API with auto-connection
            self.api = EnhancedHyperliquidAPI(
                connection_manager=self.connection_manager
            )
            
            # Initialize risk manager
            self.risk_manager = RiskManager()
            
            # Initialize strategies
            self._initialize_strategies()
            
            # Initialize backtest engine
            self.backtest_engine = BacktestEngine(
                initial_capital=10000.0,
                api=self.api
            )
            
            logger.info("All components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _initialize_strategies(self):
        """
        Initialize trading strategies
        """
        try:
            # Initialize BB RSI ADX strategy
            bb_rsi_adx = BBRSIADXStrategy(
                api=self.api,
                risk_manager=self.risk_manager,
                max_positions=3
            )
            self.strategies["BB_RSI_ADX"] = bb_rsi_adx
            logger.info("BB RSI ADX strategy initialized")
            
            # Initialize Hull Suite strategy
            hull_suite = HullSuiteStrategy(
                api=self.api,
                risk_manager=self.risk_manager,
                max_positions=3
            )
            self.strategies["Hull_Suite"] = hull_suite
            logger.info("Hull Suite strategy initialized")
            
            logger.info(f"Initialized {len(self.strategies)} strategies")
        except Exception as e:
            logger.error(f"Failed to initialize strategies: {e}")
            raise
    
    def run_gui(self):
        """
        Run the GUI application
        """
        try:
            # Initialize GUI
            self.gui = TradingDashboard(
                api=self.api,
                config_manager=self.config_manager,
                security_manager=self.security_manager,
                connection_manager=self.connection_manager,
                strategies=self.strategies,
                risk_manager=self.risk_manager,
                backtest_engine=self.backtest_engine
            )
            
            # Run GUI
            self.gui.run()
        except Exception as e:
            logger.error(f"GUI error: {e}")
            raise
    
    def run_cli(self):
        """
        Run the CLI application
        """
        try:
            logger.info("Starting CLI mode")
            
            # Auto-connect
            if self.connection_manager.connect():
                logger.info("Auto-connected successfully")
            else:
                logger.warning("Auto-connection failed")
            
            # CLI loop
            while True:
                try:
                    command = input("\nEnter command (help, status, connect, default, generate, quit): ").strip().lower()
                    
                    if command == "help":
                        self._show_help()
                    elif command == "status":
                        self._show_status()
                    elif command == "connect":
                        self._connect_cli()
                    elif command == "default":
                        self._use_default_credentials()
                    elif command == "generate":
                        self._generate_wallet_cli()
                    elif command == "quit":
                        break
                    else:
                        print("Unknown command. Type 'help' for available commands.")
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"CLI error: {e}")
                    print(f"Error: {e}")
            
            logger.info("CLI mode ended")
        except Exception as e:
            logger.error(f"CLI error: {e}")
            raise
    
    def _show_help(self):
        """
        Show CLI help
        """
        print("\nAvailable commands:")
        print("  help     - Show this help message")
        print("  status   - Show connection status")
        print("  connect  - Connect with custom credentials")
        print("  default  - Use default credentials")
        print("  generate - Generate new wallet")
        print("  quit     - Exit the application")
    
    def _show_status(self):
        """
        Show connection status
        """
        try:
            if self.connection_manager.is_connected():
                print("Status: Connected")
                print(f"Address: {self.connection_manager.get_current_address()}")
                
                # Get account info
                account_state = self.api.get_account_state()
                if account_state and "marginSummary" in account_state:
                    account_value = account_state["marginSummary"].get("accountValue", "0")
                    print(f"Account Value: ${account_value}")
            else:
                print("Status: Disconnected")
        except Exception as e:
            logger.error(f"Failed to show status: {e}")
            print(f"Error getting status: {e}")
    
    def _connect_cli(self):
        """
        Connect with custom credentials in CLI
        """
        try:
            print("\nEnter your credentials:")
            address = input("Wallet Address: ").strip()
            private_key = input("Private Key (hidden): ").strip()
            
            if address and private_key:
                if self.connection_manager.connect_with_credentials(address, private_key):
                    print("Connected successfully!")
                    
                    # Save credentials
                    save = input("Save credentials? (y/n): ").strip().lower()
                    if save == 'y':
                        self.security_manager.store_private_key(private_key)
                        self.config_manager.update_config({"wallet_address": address})
                        print("Credentials saved securely.")
                else:
                    print("Connection failed. Please check your credentials.")
            else:
                print("Invalid credentials provided.")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            print(f"Connection error: {e}")
    
    def _use_default_credentials(self):
        """
        Use default credentials
        """
        try:
            if self.connection_manager.connect():
                print("Connected with default credentials!")
                self._show_status()
            else:
                print("Failed to connect with default credentials.")
        except Exception as e:
            logger.error(f"Failed to use default credentials: {e}")
            print(f"Error: {e}")
    
    def _generate_wallet_cli(self):
        """
        Generate new wallet in CLI
        """
        try:
            from eth_account import Account
            
            # Generate new wallet
            account = Account.create()
            address = account.address
            private_key = account.key.hex()
            
            print(f"\nNew wallet generated:")
            print(f"Address: {address}")
            print(f"Private Key: {private_key}")
            
            # Ask if user wants to use this wallet
            use_wallet = input("\nUse this wallet? (y/n): ").strip().lower()
            if use_wallet == 'y':
                if self.connection_manager.connect_with_credentials(address, private_key):
                    print("Connected with new wallet!")
                    
                    # Save credentials
                    save = input("Save credentials? (y/n): ").strip().lower()
                    if save == 'y':
                        self.security_manager.store_private_key(private_key)
                        self.config_manager.update_config({"wallet_address": address})
                        print("Credentials saved securely.")
                else:
                    print("Connection failed with new wallet.")
        except Exception as e:
            logger.error(f"Failed to generate wallet: {e}")
            print(f"Error generating wallet: {e}")
    
    def run_trading(self):
        """
        Run trading mode
        """
        try:
            logger.info("Starting trading mode")
            
            # Auto-connect
            if not self.connection_manager.connect():
                logger.error("Failed to connect. Cannot start trading.")
                return
            
            self.running = True
            
            # Trading loop
            while self.running:
                try:
                    # Execute strategies
                    for name, strategy in self.strategies.items():
                        try:
                            strategy.execute()
                        except Exception as e:
                            logger.error(f"Strategy {name} execution failed: {e}")
                    
                    # Sleep for a while
                    import time
                    time.sleep(60)  # Execute every minute
                except KeyboardInterrupt:
                    logger.info("Trading interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Trading loop error: {e}")
                    import time
                    time.sleep(10)  # Wait before retrying
            
            logger.info("Trading mode ended")
        except Exception as e:
            logger.error(f"Trading error: {e}")
            raise
    
    def run_setup(self):
        """
        Run setup mode
        """
        try:
            logger.info("Starting setup mode")
            
            print("Hyperliquid Trading Bot Setup")
            print("=" * 40)
            
            # Check if credentials exist
            try:
                private_key = self.security_manager.get_private_key()
                config = self.config_manager.get_config()
                address = config.get("wallet_address")
                
                if private_key and address:
                    print(f"Existing credentials found:")
                    print(f"Address: {address}")
                    
                    use_existing = input("Use existing credentials? (y/n): ").strip().lower()
                    if use_existing == 'y':
                        if self.connection_manager.connect_with_credentials(address, private_key):
                            print("Connected successfully with existing credentials!")
                            return
                        else:
                            print("Failed to connect with existing credentials.")
            except Exception as e:
                logger.warning(f"No existing credentials found: {e}")
            
            # Setup new credentials
            print("\nSetup new credentials:")
            print("1. Enter existing wallet credentials")
            print("2. Generate new wallet")
            print("3. Use default credentials")
            
            choice = input("Choose option (1/2/3): ").strip()
            
            if choice == "1":
                self._connect_cli()
            elif choice == "2":
                self._generate_wallet_cli()
            elif choice == "3":
                self._use_default_credentials()
            else:
                print("Invalid choice.")
            
            logger.info("Setup completed")
        except Exception as e:
            logger.error(f"Setup error: {e}")
            raise


def signal_handler(signum, frame):
    """
    Handle shutdown signals
    """
    logger.info("Shutdown signal received")
    sys.exit(0)


def main():
    """
    Main entry point
    """
    try:
        # Setup signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Parse arguments
        parser = argparse.ArgumentParser(description="Hyperliquid Trading Bot")
        parser.add_argument("--mode", choices=["gui", "cli", "trading", "setup"], default="gui", help="Application mode")
        parser.add_argument("--config", default="config/config.yaml", help="Configuration file path")
        parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Log level")
        
        args = parser.parse_args()
        
        # Setup logging
        setup_logging(level=args.log_level)
        
        # Initialize bot
        bot = HyperliquidTradingBot(config_path=args.config)
        
        # Run in specified mode
        if args.mode == "gui":
            bot.run_gui()
        elif args.mode == "cli":
            bot.run_cli()
        elif args.mode == "trading":
            bot.run_trading()
        elif args.mode == "setup":
            bot.run_setup()
        
        logger.info("Application completed successfully")
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

