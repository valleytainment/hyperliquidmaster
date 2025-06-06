"""
Auto-Connected Main Application for Hyperliquid Trading Bot
Always connected and ready to trade at startup
"""

import os
import sys
import argparse
import logging
import getpass
import time
from typing import Dict, Any, Optional

from utils.logger import setup_logger, get_logger
from utils.config_manager_fixed import ConfigManager
from utils.security_fixed_v2 import SecurityManager
from core.connection_manager_enhanced import EnhancedConnectionManager
from core.api_fixed_v2 import EnhancedHyperliquidAPI

# Setup logger
setup_logger()
logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Hyperliquid Trading Bot")
    parser.add_argument("--mode", choices=["gui", "cli"], default="gui", help="Application mode")
    parser.add_argument("--testnet", action="store_true", help="Use testnet")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    return parser.parse_args()


def start_gui(config_path=None, testnet=False):
    """Start the GUI application"""
    try:
        logger.info("Starting GUI application")
        
        # Import here to avoid circular imports
        from gui.enhanced_gui_auto_connect_v2 import AutoConnectTradingDashboardV2
        
        # Initialize connection manager
        connection_manager = EnhancedConnectionManager(config_path)
        
        # Update testnet setting
        if testnet:
            connection_manager.update_network(True)
        
        # Start GUI
        app = AutoConnectTradingDashboardV2(connection_manager)
        app.run()
    except Exception as e:
        logger.error(f"GUI error: {e}")
        sys.exit(1)


def start_cli(config_path=None, testnet=False):
    """Start the CLI application"""
    try:
        logger.info("Starting CLI application")
        
        # Initialize components
        config_manager = ConfigManager(config_path)
        security_manager = SecurityManager()
        connection_manager = EnhancedConnectionManager(config_path)
        
        # Update testnet setting
        if testnet:
            connection_manager.update_network(True)
        
        # Show connection status
        status = connection_manager.get_connection_status()
        logger.info(f"Connected to Hyperliquid: {status['address']} ({status['network']})")
        
        # Show account state
        account_state = connection_manager.get_account_state()
        if account_state and "marginSummary" in account_state:
            margin_summary = account_state["marginSummary"]
            if "accountValue" in margin_summary:
                account_value = float(margin_summary["accountValue"])
                logger.info(f"Account Value: ${account_value:.2f}")
        
        # Main CLI loop
        run_cli_loop(connection_manager)
        
    except KeyboardInterrupt:
        logger.info("CLI application interrupted")
    except Exception as e:
        logger.error(f"CLI error: {e}")
        sys.exit(1)


def run_cli_loop(connection_manager):
    """Run the main CLI loop"""
    try:
        print("\n=== Hyperliquid Trading Bot CLI ===")
        print("Type 'help' for available commands")
        
        while True:
            # Check connection health
            connection_manager.check_connection_health()
            
            command = input("\n> ").strip().lower()
            
            if command == "exit" or command == "quit":
                print("Exiting...")
                break
            elif command == "help":
                print_cli_help()
            elif command == "status":
                show_connection_status(connection_manager)
            elif command == "account":
                show_account_info(connection_manager)
            elif command == "reconnect":
                reconnect(connection_manager)
            elif command == "generate":
                generate_new_wallet(connection_manager)
            elif command == "default":
                use_default_credentials(connection_manager)
            elif command == "testnet":
                toggle_testnet(connection_manager)
            elif command == "auto":
                toggle_auto_reconnect(connection_manager)
            elif command == "health":
                check_connection_health(connection_manager)
            elif command == "markets":
                show_markets(connection_manager)
            elif command == "positions":
                show_positions(connection_manager)
            elif command == "orders":
                show_orders(connection_manager)
            elif command == "balance":
                show_balance(connection_manager)
            else:
                print(f"Unknown command: {command}")
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"CLI loop error: {e}")


def print_cli_help():
    """Print CLI help"""
    print("\nAvailable commands:")
    print("  help      - Show this help message")
    print("  status    - Show connection status")
    print("  account   - Show account information")
    print("  reconnect - Reconnect to Hyperliquid")
    print("  generate  - Generate a new wallet")
    print("  default   - Use default credentials")
    print("  testnet   - Toggle testnet/mainnet")
    print("  auto      - Toggle auto-reconnect")
    print("  health    - Check connection health")
    print("  markets   - Show available markets")
    print("  positions - Show open positions")
    print("  orders    - Show open orders")
    print("  balance   - Show account balance")
    print("  exit      - Exit the application")


def show_connection_status(connection_manager):
    """Show connection status"""
    status = connection_manager.get_connection_status()
    print(f"\nConnection Status:")
    print(f"  Connected: {'Yes' if status['connected'] else 'No'}")
    print(f"  Address: {status['address'] or 'None'}")
    print(f"  Network: {status['network']}")
    print(f"  Auto-reconnect: {'Enabled' if status['auto_reconnect'] else 'Disabled'}")
    print(f"  Using default credentials: {'Yes' if status['using_default'] else 'No'}")


def show_account_info(connection_manager):
    """Show account information"""
    if not connection_manager.is_connected:
        print("\nNot connected to Hyperliquid")
        return
    
    account_state = connection_manager.get_account_state()
    if not account_state:
        print("\nNo account information available")
        return
    
    print("\nAccount Information:")
    
    if "marginSummary" in account_state:
        margin_summary = account_state["marginSummary"]
        
        if "accountValue" in margin_summary:
            account_value = float(margin_summary["accountValue"])
            print(f"  Account Value: ${account_value:.2f}")
        
        if "totalMarginUsed" in margin_summary:
            margin_used = float(margin_summary["totalMarginUsed"])
            print(f"  Margin Used: ${margin_used:.2f}")
        
        if "totalNtlPos" in margin_summary:
            total_position = float(margin_summary["totalNtlPos"])
            print(f"  Total Position: ${total_position:.2f}")
    
    if "assetPositions" in account_state:
        positions = account_state["assetPositions"]
        if positions:
            print("\n  Open Positions:")
            for pos in positions:
                if "coin" in pos and "position" in pos:
                    coin = pos["coin"]
                    position = float(pos["position"])
                    print(f"    {coin}: {position}")
        else:
            print("\n  No open positions")


def reconnect(connection_manager):
    """Reconnect to Hyperliquid"""
    print("\nReconnecting to Hyperliquid...")
    connection_manager.disconnect()
    
    if connection_manager.ensure_connection():
        print("Reconnected successfully")
        show_connection_status(connection_manager)
    else:
        print("Failed to reconnect")


def generate_new_wallet(connection_manager):
    """Generate a new wallet"""
    print("\nGenerating new wallet...")
    success, address, private_key = connection_manager.connect_with_new_wallet()
    
    if success:
        print(f"\nNew wallet generated and connected!")
        print(f"Address: {address}")
        print(f"Private Key: {private_key}")
        print("\nIMPORTANT: Save your private key securely!")
    else:
        print("\nFailed to generate and connect with new wallet.")
        print("Using default credentials instead.")


def use_default_credentials(connection_manager):
    """Use default credentials"""
    print("\nUsing default credentials...")
    
    if connection_manager._connect_with_default_credentials():
        print("Connected with default credentials")
        show_connection_status(connection_manager)
    else:
        print("Failed to connect with default credentials")


def toggle_testnet(connection_manager):
    """Toggle testnet/mainnet"""
    current_testnet = connection_manager.testnet
    new_testnet = not current_testnet
    
    print(f"\nSwitching to {'testnet' if new_testnet else 'mainnet'}...")
    
    if connection_manager.update_network(new_testnet):
        print(f"Switched to {'testnet' if new_testnet else 'mainnet'}")
        show_connection_status(connection_manager)
    else:
        print("Failed to switch network")


def toggle_auto_reconnect(connection_manager):
    """Toggle auto-reconnect"""
    current_auto_reconnect = connection_manager.auto_reconnect
    new_auto_reconnect = not current_auto_reconnect
    
    print(f"\n{'Enabling' if new_auto_reconnect else 'Disabling'} auto-reconnect...")
    
    if connection_manager.set_auto_reconnect(new_auto_reconnect):
        print(f"Auto-reconnect {'enabled' if new_auto_reconnect else 'disabled'}")
    else:
        print("Failed to update auto-reconnect setting")


def check_connection_health(connection_manager):
    """Check connection health"""
    print("\nChecking connection health...")
    
    if connection_manager.check_connection_health():
        print("Connection is healthy")
    else:
        print("Connection is not healthy")
    
    show_connection_status(connection_manager)


def show_markets(connection_manager):
    """Show available markets"""
    if not connection_manager.is_connected:
        print("\nNot connected to Hyperliquid")
        return
    
    try:
        markets = connection_manager.api.get_markets()
        
        if not markets:
            print("\nNo markets available")
            return
        
        print("\nAvailable Markets:")
        for market in markets:
            if "name" in market:
                name = market["name"]
                print(f"  {name}")
    except Exception as e:
        print(f"\nError getting markets: {e}")


def show_positions(connection_manager):
    """Show open positions"""
    if not connection_manager.is_connected:
        print("\nNot connected to Hyperliquid")
        return
    
    account_state = connection_manager.get_account_state()
    if not account_state:
        print("\nNo account information available")
        return
    
    if "assetPositions" in account_state:
        positions = account_state["assetPositions"]
        if positions:
            print("\nOpen Positions:")
            for pos in positions:
                if "coin" in pos and "position" in pos:
                    coin = pos["coin"]
                    position = float(pos["position"])
                    print(f"  {coin}: {position}")
        else:
            print("\nNo open positions")
    else:
        print("\nNo position information available")


def show_orders(connection_manager):
    """Show open orders"""
    if not connection_manager.is_connected:
        print("\nNot connected to Hyperliquid")
        return
    
    try:
        orders = connection_manager.api.get_open_orders(connection_manager.current_address)
        
        if not orders:
            print("\nNo open orders")
            return
        
        print("\nOpen Orders:")
        for order in orders:
            if "coin" in order and "side" in order and "size" in order and "price" in order:
                coin = order["coin"]
                side = order["side"]
                size = float(order["size"])
                price = float(order["price"])
                print(f"  {coin}: {side} {size} @ {price}")
    except Exception as e:
        print(f"\nError getting orders: {e}")


def show_balance(connection_manager):
    """Show account balance"""
    if not connection_manager.is_connected:
        print("\nNot connected to Hyperliquid")
        return
    
    account_state = connection_manager.get_account_state()
    if not account_state:
        print("\nNo account information available")
        return
    
    if "marginSummary" in account_state:
        margin_summary = account_state["marginSummary"]
        
        if "accountValue" in margin_summary:
            account_value = float(margin_summary["accountValue"])
            print(f"\nAccount Balance: ${account_value:.2f}")
        else:
            print("\nNo balance information available")
    else:
        print("\nNo balance information available")


def main():
    """Main entry point"""
    try:
        # Parse arguments
        args = parse_args()
        
        logger.info("Hyperliquid Trading Bot initialized")
        
        # Start application in selected mode
        if args.mode == "gui":
            start_gui(args.config, args.testnet)
        else:
            start_cli(args.config, args.testnet)
        
        logger.info("Application completed successfully")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

