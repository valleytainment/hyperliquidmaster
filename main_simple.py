"""
Simple Main Application for Hyperliquid Trading Bot
Focused on fixing private key input and connection testing issues
"""

import os
import sys
import argparse
import getpass
import logging

from utils.logger import setup_logging, get_logger
from utils.config_manager import ConfigManager
from utils.security_fixed_v2 import SecurityManager
from core.api_simple import SimpleHyperliquidAPI
from eth_account import Account

# Setup logging
setup_logging()
logger = get_logger(__name__)


class SimpleHyperliquidBot:
    """Simple Hyperliquid Trading Bot focused on fixing private key input and connection testing"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the bot
        
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
        self.api = SimpleHyperliquidAPI(config_path, self.testnet)
        
        logger.info("Simple Hyperliquid Bot initialized")
    
    def start_gui(self):
        """Start the GUI interface"""
        try:
            # Import GUI module here to avoid circular imports
            from gui.enhanced_gui_simple import SimpleTradingDashboard
            
            # Initialize the GUI
            app = SimpleTradingDashboard()
            
            # Pass components to GUI
            app.api = self.api
            app.config_manager = self.config_manager
            app.security_manager = self.security_manager
            
            logger.info("Starting GUI interface")
            
            # Start the GUI
            app.run()
            
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
                
                # Ask if user wants to generate a new wallet
                generate_wallet = input("Do you want to generate a new wallet? (y/n): ").strip().lower()
                
                if generate_wallet == 'y':
                    # Generate new wallet
                    print("\nGenerating new wallet...")
                    acct = Account.create()
                    new_address = acct.address
                    new_privkey_hex = acct.key.hex()
                    
                    print(f"\n✅ New wallet generated!")
                    print(f"Address: {new_address}")
                    print(f"Private Key: {new_privkey_hex}")
                    
                    # Confirm save
                    save_wallet = input("\nDo you want to save this wallet? (y/n): ").strip().lower()
                    
                    if save_wallet == 'y':
                        # Save to config
                        self.config_manager.set('trading.wallet_address', new_address)
                        self.config_manager.save_config()
                        
                        # Save private key securely
                        success = self.security_manager.store_private_key(new_privkey_hex)
                        if success:
                            print("✅ Wallet saved successfully!")
                            wallet_address = new_address
                            private_key = new_privkey_hex
                        else:
                            print("❌ Failed to save private key")
                            return
                    else:
                        # Use but don't save
                        print("Using generated wallet without saving")
                        wallet_address = new_address
                        private_key = new_privkey_hex
                else:
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
            
            # Test connection
            success = self.api.test_connection(private_key, wallet_address)
            if not success:
                print("❌ Authentication failed. Please check your credentials.")
                
                # Ask if user wants to try again with new credentials
                retry = input("\nDo you want to try again with new credentials? (y/n): ").strip().lower()
                if retry == 'y':
                    # Clear saved credentials
                    self.security_manager.clear_private_key()
                    
                    # Ask if user wants to generate a new wallet
                    generate_wallet = input("Do you want to generate a new wallet? (y/n): ").strip().lower()
                    
                    if generate_wallet == 'y':
                        # Generate new wallet
                        print("\nGenerating new wallet...")
                        acct = Account.create()
                        new_address = acct.address
                        new_privkey_hex = acct.key.hex()
                        
                        print(f"\n✅ New wallet generated!")
                        print(f"Address: {new_address}")
                        print(f"Private Key: {new_privkey_hex}")
                        
                        # Use the new wallet
                        wallet_address = new_address
                        private_key = new_privkey_hex
                        
                        # Save to config
                        self.config_manager.set('trading.wallet_address', wallet_address)
                        self.config_manager.save_config()
                        
                        # Save private key securely
                        self.security_manager.store_private_key(private_key)
                    else:
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
                    success = self.api.test_connection(private_key, wallet_address)
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
            account = self.api.get_account_state(wallet_address)
            if account:
                print("\nAccount Information:")
                if "marginSummary" in account:
                    margin = account["marginSummary"]
                    print(f"Account Value: ${float(margin.get('accountValue', 0)):.2f}")
                    print(f"Total Margin Used: ${float(margin.get('totalMarginUsed', 0)):.2f}")
                else:
                    print("No margin information available")
            
            # Simple CLI loop
            print("\nType 'help' for available commands, 'exit' to quit")
            
            while True:
                cmd = input("\n> ").strip().lower()
                
                if cmd == "exit":
                    break
                elif cmd == "help":
                    print("\nAvailable commands:")
                    print("  account  - Show account information")
                    print("  key      - View or update private key")
                    print("  wallet   - View or update wallet address")
                    print("  generate - Generate a new wallet")
                    print("  test     - Test connection")
                    print("  exit     - Exit the application")
                elif cmd == "account":
                    account = self.api.get_account_state(wallet_address)
                    if account and "marginSummary" in account:
                        margin = account["marginSummary"]
                        print(f"\nAccount Value: ${float(margin.get('accountValue', 0)):.2f}")
                        print(f"Total Margin Used: ${float(margin.get('totalMarginUsed', 0)):.2f}")
                    else:
                        print("\nNo account information available")
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
                                private_key = new_key
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
                                private_key = None
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
                            wallet_address = new_address
                        else:
                            print("❌ No wallet address entered")
                    elif wallet_cmd == "3":
                        # Back to main menu
                        pass
                    else:
                        print("Invalid option")
                elif cmd == "generate":
                    # Generate new wallet
                    print("\nGenerating new wallet...")
                    acct = Account.create()
                    new_address = acct.address
                    new_privkey_hex = acct.key.hex()
                    
                    print(f"\n✅ New wallet generated!")
                    print(f"Address: {new_address}")
                    print(f"Private Key: {new_privkey_hex}")
                    
                    # Ask if user wants to save the new wallet
                    save_wallet = input("\nDo you want to save this wallet? (y/n): ").strip().lower()
                    
                    if save_wallet == 'y':
                        # Save to config
                        self.config_manager.set('trading.wallet_address', new_address)
                        self.config_manager.save_config()
                        
                        # Save private key securely
                        success = self.security_manager.store_private_key(new_privkey_hex)
                        if success:
                            print("✅ Wallet saved successfully!")
                            wallet_address = new_address
                            private_key = new_privkey_hex
                        else:
                            print("❌ Failed to save wallet")
                elif cmd == "test":
                    # Test connection
                    print("\nTesting connection...")
                    success = self.api.test_connection(private_key, wallet_address)
                    if success:
                        print("✅ Connection test successful")
                    else:
                        print("❌ Connection test failed")
                else:
                    print("Unknown command. Type 'help' for available commands.")
            
            print("\nExiting CLI...")
            
        except KeyboardInterrupt:
            print("\nExiting CLI...")
        except Exception as e:
            logger.error(f"CLI error: {e}")


def main():
    """Main entry point"""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Simple Hyperliquid Trading Bot")
        parser.add_argument("--config", type=str, help="Path to configuration file")
        parser.add_argument("--mode", type=str, choices=["gui", "cli"], default="gui", help="Operation mode")
        
        args = parser.parse_args()
        
        # Initialize the bot
        bot = SimpleHyperliquidBot(args.config)
        
        # Run in specified mode
        if args.mode == "gui":
            bot.start_gui()
        elif args.mode == "cli":
            bot.start_cli()
        
        logger.info("Application completed successfully")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

