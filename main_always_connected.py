"""
Main Application Entry Point for Hyperliquid Trading Bot
With always-connected functionality
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
from core.connection_manager import ConnectionManager
from core.api_fixed_v2 import EnhancedHyperliquidAPI
from gui.enhanced_gui_auto_connect import AutoConnectTradingDashboard

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
        app = AutoConnectTradingDashboard()
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
        connection_manager = ConnectionManager(config_path)
        
        # Update testnet setting
        if testnet:
            config_manager.set('trading.testnet', True)
            config_manager.save_config()
            connection_manager.update_network(True)
        
        # Ensure connection
        if not connection_manager.ensure_connection():
            # If connection failed, ask for credentials or generate new wallet
            handle_cli_authentication(connection_manager)
        
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


def handle_cli_authentication(connection_manager):
    """Handle CLI authentication"""
    try:
        print("\n=== Hyperliquid Authentication ===")
        print("No valid credentials found or connection failed.")
        
        # Ask if user wants to generate a new wallet
        generate_new = input("Do you want to generate a new wallet? (y/n): ").lower() == 'y'
        
        if generate_new:
            # Generate new wallet
            success, address, private_key = connection_manager.connect_with_new_wallet()
            
            if success:
                print(f"\nNew wallet generated and connected!")
                print(f"Address: {address}")
                print(f"Private Key: {private_key}")
                print("\nIMPORTANT: Save your private key securely!")
            else:
                print("\nFailed to generate and connect with new wallet.")
                sys.exit(1)
        else:
            # Ask for existing credentials
            print("\nPlease enter your existing credentials:")
            wallet_address = input("Wallet Address (0x...): ").strip()
            private_key = getpass.getpass("Private Key: ").strip()
            
            # Validate inputs
            if not wallet_address or not private_key:
                print("Invalid credentials. Exiting.")
                sys.exit(1)
            
            # Connect with provided credentials
            if not connection_manager._connect_with_credentials(private_key, wallet_address):
                print("Failed to connect with provided credentials. Exiting.")
                sys.exit(1)
            
            # Save credentials
            save_creds = input("Save these credentials for future use? (y/n): ").lower() == 'y'
            if save_creds:
                connection_manager.config_manager.set('trading.wallet_address', wallet_address)
                connection_manager.security_manager.store_private_key(private_key)
                connection_manager.config_manager.save_config()
                print("Credentials saved.")
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        sys.exit(1)


def run_cli_loop(connection_manager):
    """Run the main CLI loop"""
    try:
        print("\n=== Hyperliquid Trading Bot CLI ===")
        print("Type 'help' for available commands")
        
        while True:
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
    print("  exit      - Exit the application")


def show_connection_status(connection_manager):
    """Show connection status"""
    status = connection_manager.get_connection_status()
    print(f"\nConnection Status:")
    print(f"  Connected: {'Yes' if status['connected'] else 'No'}")
    print(f"  Address: {status['address'] or 'None'}")
    print(f"  Network: {status['network']}")


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
        handle_cli_authentication(connection_manager)


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

