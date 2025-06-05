"""
Robust Main Application for Hyperliquid Trading Bot
Focused on fixing private key input and connection testing issues
"""

import os
import sys
import argparse
import getpass
import logging
import tkinter as tk
from tkinter import ttk, messagebox

from utils.logger import setup_logging, get_logger
from utils.config_manager import ConfigManager
from utils.security_fixed_v2 import SecurityManager
from core.api_robust import RobustHyperliquidAPI
from eth_account import Account

# Setup logging
setup_logging()
logger = get_logger(__name__)


class RobustHyperliquidBot:
    """Robust Hyperliquid Trading Bot focused on fixing private key input and connection testing"""
    
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
        self.api = RobustHyperliquidAPI(config_path, self.testnet)
        
        logger.info("Robust Hyperliquid Bot initialized")
    
    def start_gui(self):
        """Start the GUI interface"""
        try:
            # Create a simple GUI
            root = tk.Tk()
            root.title("Hyperliquid Trading Bot")
            root.geometry("600x500")
            
            # Variables
            private_key_var = tk.StringVar()
            wallet_address_var = tk.StringVar()
            testnet_var = tk.BooleanVar(value=self.testnet)
            
            # Load existing wallet address if available
            try:
                existing_address = self.config_manager.get('trading.wallet_address', '')
                if existing_address:
                    wallet_address_var.set(existing_address)
            except Exception as e:
                logger.error(f"Error loading wallet address: {e}")
            
            # Main frame
            main_frame = ttk.Frame(root, padding=20)
            main_frame.pack(fill="both", expand=True)
            
            # Title
            title_label = ttk.Label(main_frame, text="Hyperliquid Trading Bot", font=("Arial", 16))
            title_label.pack(pady=(0, 20))
            
            # Wallet Address Section
            wallet_frame = ttk.LabelFrame(main_frame, text="Wallet Address", padding=10)
            wallet_frame.pack(fill="x", pady=10)
            
            wallet_label = ttk.Label(wallet_frame, text="Enter your wallet address (0x...):")
            wallet_label.pack(anchor="w")
            
            wallet_entry = ttk.Entry(wallet_frame, textvariable=wallet_address_var, width=50)
            wallet_entry.pack(fill="x", pady=5)
            
            # Private Key Section
            key_frame = ttk.LabelFrame(main_frame, text="Private Key", padding=10)
            key_frame.pack(fill="x", pady=10)
            
            key_label = ttk.Label(key_frame, text="Enter your private key:")
            key_label.pack(anchor="w")
            
            key_input_frame = ttk.Frame(key_frame)
            key_input_frame.pack(fill="x", pady=5)
            
            private_key_entry = ttk.Entry(
                key_input_frame, 
                textvariable=private_key_var, 
                show="*", 
                width=45
            )
            private_key_entry.pack(side="left", fill="x", expand=True)
            
            # Show/Hide toggle
            show_key_var = tk.BooleanVar(value=False)
            
            def toggle_key_visibility():
                if show_key_var.get():
                    private_key_entry.config(show="")
                else:
                    private_key_entry.config(show="*")
            
            show_key_check = ttk.Checkbutton(
                key_input_frame,
                text="Show",
                variable=show_key_var,
                command=toggle_key_visibility
            )
            show_key_check.pack(side="right", padx=(5, 0))
            
            # Testnet Option
            testnet_frame = ttk.Frame(main_frame)
            testnet_frame.pack(fill="x", pady=5)
            
            testnet_check = ttk.Checkbutton(
                testnet_frame,
                text="Use Testnet",
                variable=testnet_var
            )
            testnet_check.pack(anchor="w")
            
            # Generate Wallet Button
            generate_frame = ttk.Frame(main_frame)
            generate_frame.pack(fill="x", pady=10)
            
            def generate_wallet():
                try:
                    # Create a new random account
                    acct = Account.create()
                    new_address = acct.address
                    new_privkey = acct.key.hex()
                    
                    # Update the entry fields
                    wallet_address_var.set(new_address)
                    private_key_var.set(new_privkey)
                    
                    logger.info(f"Generated new wallet: {new_address}")
                    messagebox.showinfo(
                        "Wallet Generated",
                        f"New wallet generated successfully!\n\nAddress: {new_address}\n\nPlease save your credentials."
                    )
                except Exception as e:
                    logger.error(f"Failed to generate wallet: {e}")
                    messagebox.showerror("Error", f"Failed to generate wallet: {e}")
            
            generate_btn = ttk.Button(
                generate_frame,
                text="Generate New Wallet",
                command=generate_wallet
            )
            generate_btn.pack(anchor="w")
            
            # Action Buttons
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill="x", pady=20)
            
            def test_connection():
                try:
                    # Get values
                    private_key = private_key_var.get().strip()
                    wallet_address = wallet_address_var.get().strip()
                    use_testnet = testnet_var.get()
                    
                    # Validate inputs
                    if not private_key:
                        messagebox.showerror("Error", "Please enter a private key")
                        return
                    
                    if not wallet_address:
                        messagebox.showerror("Error", "Please enter a wallet address")
                        return
                    
                    # Update testnet setting if changed
                    if use_testnet != self.testnet:
                        self.testnet = use_testnet
                        self.api = RobustHyperliquidAPI(None, self.testnet)
                    
                    # Test connection
                    logger.info("Testing connection...")
                    success = self.api.test_connection(private_key, wallet_address)
                    
                    if success:
                        logger.info("Connection test successful")
                        messagebox.showinfo("Success", "Connection test successful!")
                    else:
                        logger.error("Connection test failed")
                        messagebox.showerror("Error", "Connection test failed. Please check your credentials.")
                except Exception as e:
                    logger.error(f"Error testing connection: {e}")
                    messagebox.showerror("Error", f"Error testing connection: {e}")
            
            test_btn = ttk.Button(
                button_frame,
                text="Test Connection",
                command=test_connection
            )
            test_btn.pack(side="left", padx=(0, 10))
            
            def save_credentials():
                try:
                    # Get values
                    private_key = private_key_var.get().strip()
                    wallet_address = wallet_address_var.get().strip()
                    use_testnet = testnet_var.get()
                    
                    # Validate inputs
                    if not private_key:
                        messagebox.showerror("Error", "Please enter a private key")
                        return
                    
                    if not wallet_address:
                        messagebox.showerror("Error", "Please enter a wallet address")
                        return
                    
                    # Save wallet address to config
                    self.config_manager.set('trading.wallet_address', wallet_address)
                    self.config_manager.set('trading.testnet', use_testnet)
                    self.config_manager.save_config()
                    
                    # Save private key securely
                    success = self.security_manager.store_private_key(private_key)
                    
                    if success:
                        logger.info("Credentials saved successfully")
                        messagebox.showinfo("Success", "Credentials saved successfully")
                        
                        # Clear the private key field for security
                        private_key_var.set("")
                    else:
                        logger.error("Failed to save private key")
                        messagebox.showerror("Error", "Failed to save private key")
                except Exception as e:
                    logger.error(f"Error saving credentials: {e}")
                    messagebox.showerror("Error", f"Error saving credentials: {e}")
            
            save_btn = ttk.Button(
                button_frame,
                text="Save Credentials",
                command=save_credentials
            )
            save_btn.pack(side="left")
            
            # Status Frame
            status_frame = ttk.LabelFrame(main_frame, text="Status", padding=10)
            status_frame.pack(fill="both", expand=True, pady=10)
            
            status_text = tk.Text(status_frame, height=8, wrap="word")
            status_text.pack(fill="both", expand=True)
            
            # Redirect logger to status text
            class TextHandler(logging.Handler):
                def __init__(self, text_widget):
                    super().__init__()
                    self.text_widget = text_widget
                
                def emit(self, record):
                    msg = self.format(record) + "\n"
                    self.text_widget.configure(state="normal")
                    self.text_widget.insert("end", msg)
                    self.text_widget.see("end")
                    self.text_widget.configure(state="disabled")
            
            handler = TextHandler(status_text)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")
            handler.setFormatter(formatter)
            
            logger.addHandler(handler)
            
            logger.info("GUI initialized")
            
            # Start the GUI
            root.mainloop()
            
        except Exception as e:
            logger.error(f"GUI error: {e}")
    
    def start_cli(self):
        """Start the command-line interface"""
        try:
            logger.info("Starting CLI interface")
            
            print("\n🚀 Hyperliquid Trading Bot CLI")
            print("----------------------------")
            
            # Check if we have saved credentials
            private_key = self.security_manager.get_private_key()
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
                    new_privkey = acct.key.hex()
                    
                    print(f"\n✅ New wallet generated!")
                    print(f"Address: {new_address}")
                    print(f"Private Key: {new_privkey}")
                    
                    # Confirm save
                    save_wallet = input("\nDo you want to save this wallet? (y/n): ").strip().lower()
                    
                    if save_wallet == 'y':
                        # Save to config
                        self.config_manager.set('trading.wallet_address', new_address)
                        self.config_manager.save_config()
                        
                        # Save private key securely
                        success = self.security_manager.store_private_key(new_privkey)
                        if success:
                            print("✅ Wallet saved successfully!")
                            wallet_address = new_address
                            private_key = new_privkey
                        else:
                            print("❌ Failed to save private key")
                            return
                    else:
                        # Use but don't save
                        print("Using generated wallet without saving")
                        wallet_address = new_address
                        private_key = new_privkey
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
                        new_privkey = acct.key.hex()
                        
                        print(f"\n✅ New wallet generated!")
                        print(f"Address: {new_address}")
                        print(f"Private Key: {new_privkey}")
                        
                        # Use the new wallet
                        wallet_address = new_address
                        private_key = new_privkey
                        
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
                    new_privkey = acct.key.hex()
                    
                    print(f"\n✅ New wallet generated!")
                    print(f"Address: {new_address}")
                    print(f"Private Key: {new_privkey}")
                    
                    # Ask if user wants to save the new wallet
                    save_wallet = input("\nDo you want to save this wallet? (y/n): ").strip().lower()
                    
                    if save_wallet == 'y':
                        # Save to config
                        self.config_manager.set('trading.wallet_address', new_address)
                        self.config_manager.save_config()
                        
                        # Save private key securely
                        success = self.security_manager.store_private_key(new_privkey)
                        if success:
                            print("✅ Wallet saved successfully!")
                            wallet_address = new_address
                            private_key = new_privkey
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
        parser = argparse.ArgumentParser(description="Robust Hyperliquid Trading Bot")
        parser.add_argument("--config", type=str, help="Path to configuration file")
        parser.add_argument("--mode", type=str, choices=["gui", "cli"], default="gui", help="Operation mode")
        
        args = parser.parse_args()
        
        # Initialize the bot
        bot = RobustHyperliquidBot(args.config)
        
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

