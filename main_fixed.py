"""
Main Application Entry Point for Hyperliquid Trading Bot - BEST BRANCH EDITION
🏆 ULTIMATE VERSION with maximum optimizations and enhancements
Integrates all components and provides unified interface
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

from core.api import EnhancedHyperliquidAPI
from gui.enhanced_gui_fixed import TradingDashboard  # Use the fixed GUI implementation
from strategies.bb_rsi_adx import BBRSIADXStrategy
from strategies.hull_suite import HullSuiteStrategy
from backtesting.backtest_engine import BacktestEngine
from risk_management.risk_manager_fixed import RiskManager, RiskLimits
from utils.logger import get_logger, setup_logging, TradingLogger
from utils.config_manager import ConfigManager, TradingConfig
from utils.security import SecurityManager

logger = get_logger(__name__)
trading_logger = TradingLogger(__name__)


class HyperliquidTradingBot:
    """Main trading bot application"""
    
    def __init__(self, config_path: str = None, gui_mode: bool = True):
        """
        Initialize the trading bot
        
        Args:
            config_path: Path to configuration file
            gui_mode: Whether to run with GUI
        """
        self.config_path = config_path
        self.gui_mode = gui_mode
        self.running = False
        
        # Initialize components
        self.config_manager = ConfigManager(config_path)
        self.security_manager = SecurityManager()
        self.api_client = None
        self.risk_manager = None
        self.strategies = {}
        self.backtest_engine = None
        
        # GUI component
        self.gui = None
        
        # Trading state
        self.active_strategies = []
        self.portfolio_state = {}
        
        logger.info("Hyperliquid Trading Bot initialized")
    
    def initialize_components(self):
        """Initialize all bot components"""
        try:
            # Initialize API client
            testnet = self.config_manager.trading.testnet
            self.api_client = EnhancedHyperliquidAPI(
                config_path=self.config_path,
                testnet=testnet
            )
            
            # Initialize risk manager
            risk_limits = RiskLimits(
                max_portfolio_risk=self.config_manager.trading.max_daily_loss / 100,
                max_daily_loss=self.config_manager.trading.max_daily_loss / 100,
                max_drawdown=self.config_manager.trading.max_drawdown / 100,
                max_leverage=self.config_manager.trading.max_leverage,
                max_position_size=self.config_manager.trading.default_order_size / 1000  # Convert to percentage
            )
            self.risk_manager = RiskManager(risk_limits, self.api_client)
            
            # Initialize strategies
            self._initialize_strategies()
            
            # Initialize backtest engine
            self.backtest_engine = BacktestEngine(
                initial_capital=10000.0,
                commission_rate=0.0005,
                slippage_rate=0.0001
            )
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False
    
    def _initialize_strategies(self):
        """Initialize trading strategies"""
        try:
            # BB RSI ADX Strategy
            bb_rsi_config = self.config_manager.get_strategy_config('bb_rsi_adx')
            if bb_rsi_config and bb_rsi_config.enabled:
                self.strategies['bb_rsi_adx'] = BBRSIADXStrategy(
                    config=bb_rsi_config,
                    api_client=self.api_client
                )
                logger.info("BB RSI ADX strategy initialized")
            
            # Hull Suite Strategy
            hull_config = self.config_manager.get_strategy_config('hull_suite')
            if hull_config and hull_config.enabled:
                self.strategies['hull_suite'] = HullSuiteStrategy(
                    config=hull_config,
                    api_client=self.api_client
                )
                logger.info("Hull Suite strategy initialized")
            
            logger.info(f"Initialized {len(self.strategies)} strategies")
            
        except Exception as e:
            logger.error(f"Failed to initialize strategies: {e}")
    
    def authenticate(self) -> bool:
        """Authenticate with Hyperliquid API"""
        try:
            # Get private key
            private_key = self.security_manager.get_private_key()
            if not private_key:
                logger.error("Private key not found. Please run setup first.")
                return False
            
            # Get wallet address
            wallet_address = self.config_manager.trading.wallet_address
            if not wallet_address:
                logger.error("Wallet address not configured")
                return False
            
            # Authenticate
            if self.api_client.authenticate(private_key, wallet_address):
                logger.info("Successfully authenticated with Hyperliquid API")
                return True
            else:
                logger.error("Authentication failed")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
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
    
    def run(self, mode: str = 'gui'):
        """
        Run the trading bot
        
        Args:
            mode: Run mode ('gui', 'trading', 'backtest', 'setup')
        """
        try:
            # Setup signal handlers
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
            # Initialize components
            if not self.initialize_components():
                logger.error("Failed to initialize components")
                return False
            
            if mode == 'gui':
                # Run GUI mode
                self.start_gui()
                
            elif mode == 'trading':
                # Run automated trading mode
                if not self.validate_configuration():
                    return False
                
                if not self.authenticate():
                    return False
                
                # Start trading
                asyncio.run(self.start_trading())
                
            elif mode == 'setup':
                # Run setup mode
                print("=== Hyperliquid Trading Bot Setup ===")
                
                # Setup private key
                if self.setup_private_key():
                    print("✓ Private key setup completed")
                else:
                    print("✗ Private key setup failed")
                    return False
                
                # Configure wallet address
                wallet_address = input("Enter your wallet address: ").strip()
                if wallet_address:
                    self.config_manager.set('trading.wallet_address', wallet_address)
                    self.config_manager.save_config()
                    print("✓ Wallet address configured")
                
                # Configure testnet
                use_testnet = input("Use testnet? (y/N): ").strip().lower() == 'y'
                self.config_manager.set('trading.testnet', use_testnet)
                self.config_manager.save_config()
                print(f"✓ Network configured ({'testnet' if use_testnet else 'mainnet'})")
                
                print("\nSetup completed successfully!")
                print("You can now run the bot with: python main.py --mode gui")
                
            elif mode == 'backtest':
                # Run backtest mode
                print("=== Backtest Mode ===")
                strategy = input("Enter strategy name (bb_rsi_adx/hull_suite): ").strip()
                start_date = input("Enter start date (YYYY-MM-DD): ").strip()
                end_date = input("Enter end date (YYYY-MM-DD): ").strip()
                
                results = self.run_backtest(strategy, start_date, end_date)
                
                if 'error' not in results:
                    print(f"\nBacktest Results for {strategy}:")
                    print(f"Total Return: {results['total_return']:.1f}%")
                    print(f"Max Drawdown: {results['max_drawdown']:.1f}%")
                    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
                    print(f"Win Rate: {results['win_rate']:.1f}%")
                    print(f"Total Trades: {results['total_trades']}")
                else:
                    print(f"Backtest failed: {results['error']}")
            
            else:
                logger.error(f"Unknown mode: {mode}")
                return False
            
            return True
            
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
            return True
        except Exception as e:
            logger.error(f"Application error: {e}")
            return False
    
    # Other methods omitted for brevity...
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Hyperliquid Trading Bot')
    parser.add_argument('--mode', choices=['gui', 'trading', 'backtest', 'setup'], 
                       default='gui', help='Run mode')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', type=str, help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Create and run bot
    bot = HyperliquidTradingBot(
        config_path=args.config,
        gui_mode=(args.mode == 'gui')
    )
    
    success = bot.run(args.mode)
    
    if success:
        logger.info("Application completed successfully")
        sys.exit(0)
    else:
        logger.error("Application failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

