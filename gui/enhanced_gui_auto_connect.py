"""
Enhanced GUI for Hyperliquid Trading Bot
With wallet generation and auto-connection functionality
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import asyncio
import logging
import os
import sys
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.config_manager_fixed import ConfigManager
from utils.security_fixed_v2 import SecurityManager
from core.api_fixed_v2 import EnhancedHyperliquidAPI
from eth_account import Account

logger = get_logger(__name__)


class AutoConnectTradingDashboard:
    """Enhanced trading dashboard with wallet generation and auto-connection"""
    
    def __init__(self):
        """Initialize the trading dashboard"""
        self.root = tk.Tk()
        self.root.title("Hyperliquid Trading Bot")
        self.root.geometry("700x600")
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.security_manager = SecurityManager()
        self.api = None
        
        # Default credentials
        self.default_address = "0x306D29F56EA1345c7E6F1ff27657ba05cEE15D4F"
        self.default_private_key = "0x43ba46de58067dd1ef3794c653bf3b11fa78866623cc515a5aff5f4be31fd3b8"
        
        # Variables
        self.private_key_var = tk.StringVar()
        self.wallet_address_var = tk.StringVar()
        self.show_key_var = tk.BooleanVar(value=False)
        self.testnet_var = tk.BooleanVar(value=False)
        self.auto_connect_var = tk.BooleanVar(value=True)
        self.connection_status_var = tk.StringVar(value="Not Connected")
        
        # Load existing wallet address if available
        try:
            existing_address = self.config_manager.get('trading.wallet_address', '')
            if existing_address:
                self.wallet_address_var.set(existing_address)
        except Exception as e:
            logger.error(f"Error loading wallet address: {e}")
        
        # Setup GUI
        self.setup_gui()
        
        # Initialize API
        self.api = EnhancedHyperliquidAPI(testnet=self.testnet_var.get())
        
        # Auto-connect with default credentials if no credentials are found
        self.auto_connect_on_startup()
        
        logger.info("Auto-Connect Trading Dashboard initialized")
    
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Main frame
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)
        
        # Title
        title_label = tk.Label(main_frame, text="Hyperliquid Trading Bot", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Connection Status
        status_frame = tk.Frame(main_frame)
        status_frame.pack(fill="x", pady=5)
        
        status_label = tk.Label(status_frame, text="Status:", font=("Arial", 10, "bold"))
        status_label.pack(side="left")
        
        self.connection_status_label = tk.Label(status_frame, textvariable=self.connection_status_var, fg="red")
        self.connection_status_label.pack(side="left", padx=(5, 0))
        
        # Notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill="both", expand=True, pady=10)
        
        # Settings Tab
        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text="Settings")
        
        # Trading Tab
        trading_frame = ttk.Frame(notebook)
        notebook.add(trading_frame, text="Trading")
        
        # Setup tabs
        self.setup_settings_tab(settings_frame)
        self.setup_trading_tab(trading_frame)
        
        # Status Text
        status_text_frame = tk.LabelFrame(main_frame, text="Log", padx=10, pady=10)
        status_text_frame.pack(fill="both", expand=True, pady=10)
        
        self.status_text = tk.Text(status_text_frame, height=8, wrap="word")
        self.status_text.pack(fill="both", expand=True)
        
        # Scrollbar for status text
        scrollbar = tk.Scrollbar(self.status_text)
        scrollbar.pack(side="right", fill="y")
        self.status_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.status_text.yview)
        
        # Redirect logger to status text
        self.setup_text_handler()
    
    def setup_settings_tab(self, parent):
        """Setup the settings tab"""
        # Wallet Address Section
        wallet_frame = tk.LabelFrame(parent, text="Wallet Address", padx=10, pady=10)
        wallet_frame.pack(fill="x", pady=10)
        
        wallet_label = tk.Label(wallet_frame, text="Enter your wallet address (0x...):")
        wallet_label.pack(anchor="w")
        
        self.wallet_entry = tk.Entry(wallet_frame, textvariable=self.wallet_address_var, width=50)
        self.wallet_entry.pack(fill="x", pady=5)
        
        # Private Key Section
        key_frame = tk.LabelFrame(parent, text="Private Key", padx=10, pady=10)
        key_frame.pack(fill="x", pady=10)
        
        key_label = tk.Label(key_frame, text="Enter your private key:")
        key_label.pack(anchor="w")
        
        key_input_frame = tk.Frame(key_frame)
        key_input_frame.pack(fill="x", pady=5)
        
        self.private_key_entry = tk.Entry(
            key_input_frame, 
            textvariable=self.private_key_var, 
            show="*", 
            width=45
        )
        self.private_key_entry.pack(side="left", fill="x", expand=True)
        
        show_key_btn = tk.Button(
            key_input_frame,
            text="Show",
            command=self.toggle_private_key_visibility,
            width=8
        )
        show_key_btn.pack(side="right", padx=(5, 0))
        
        # Options Frame
        options_frame = tk.LabelFrame(parent, text="Options", padx=10, pady=10)
        options_frame.pack(fill="x", pady=10)
        
        # Testnet Option
        testnet_check = tk.Checkbutton(
            options_frame,
            text="Use Testnet",
            variable=self.testnet_var,
            command=self.update_api_network
        )
        testnet_check.pack(anchor="w")
        
        # Auto-connect Option
        auto_connect_check = tk.Checkbutton(
            options_frame,
            text="Auto-connect on wallet generation",
            variable=self.auto_connect_var
        )
        auto_connect_check.pack(anchor="w")
        
        # Buttons Frame
        buttons_frame = tk.Frame(parent)
        buttons_frame.pack(fill="x", pady=10)
        
        # Generate Wallet Button
        generate_btn = tk.Button(
            buttons_frame,
            text="Generate New Wallet",
            command=self.generate_wallet,
            width=20
        )
        generate_btn.pack(side="left", padx=(0, 10))
        
        # Test Connection Button
        test_btn = tk.Button(
            buttons_frame,
            text="Test Connection",
            command=self.test_connection,
            width=15
        )
        test_btn.pack(side="left", padx=(0, 10))
        
        # Save Credentials Button
        save_btn = tk.Button(
            buttons_frame,
            text="Save Credentials",
            command=self.save_credentials,
            width=15
        )
        save_btn.pack(side="left")
        
        # Use Default Button
        default_btn = tk.Button(
            buttons_frame,
            text="Use Default",
            command=self.use_default_credentials,
            width=15
        )
        default_btn.pack(side="right")
    
    def setup_trading_tab(self, parent):
        """Setup the trading tab"""
        # Account Info Frame
        account_frame = tk.LabelFrame(parent, text="Account Information", padx=10, pady=10)
        account_frame.pack(fill="x", pady=10)
        
        # Account Address
        address_frame = tk.Frame(account_frame)
        address_frame.pack(fill="x", pady=5)
        
        address_label = tk.Label(address_frame, text="Wallet Address:", width=15, anchor="w")
        address_label.pack(side="left")
        
        self.account_address_label = tk.Label(address_frame, text="Not connected", fg="red")
        self.account_address_label.pack(side="left", fill="x", expand=True)
        
        # Account Balance
        balance_frame = tk.Frame(account_frame)
        balance_frame.pack(fill="x", pady=5)
        
        balance_label = tk.Label(balance_frame, text="Account Value:", width=15, anchor="w")
        balance_label.pack(side="left")
        
        self.account_balance_label = tk.Label(balance_frame, text="$0.00")
        self.account_balance_label.pack(side="left")
        
        # Margin Used
        margin_frame = tk.Frame(account_frame)
        margin_frame.pack(fill="x", pady=5)
        
        margin_label = tk.Label(margin_frame, text="Margin Used:", width=15, anchor="w")
        margin_label.pack(side="left")
        
        self.margin_used_label = tk.Label(margin_frame, text="$0.00")
        self.margin_used_label.pack(side="left")
        
        # Refresh Button
        refresh_btn = tk.Button(
            account_frame,
            text="Refresh Account Info",
            command=self.refresh_account_info,
            width=20
        )
        refresh_btn.pack(anchor="w", pady=10)
        
        # Trading Controls Frame
        trading_controls_frame = tk.LabelFrame(parent, text="Trading Controls", padx=10, pady=10)
        trading_controls_frame.pack(fill="x", pady=10)
        
        # Start/Stop Trading Button
        self.trading_button_text = tk.StringVar(value="Start Trading")
        trading_btn = tk.Button(
            trading_controls_frame,
            textvariable=self.trading_button_text,
            command=self.toggle_trading,
            width=15
        )
        trading_btn.pack(side="left", padx=(0, 10))
        
        # Trading Status
        self.trading_status_var = tk.StringVar(value="Inactive")
        trading_status_label = tk.Label(trading_controls_frame, textvariable=self.trading_status_var, fg="red")
        trading_status_label.pack(side="left")
    
    def setup_text_handler(self):
        """Setup text handler for logging"""
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
        
        handler = TextHandler(self.status_text)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
    
    def toggle_private_key_visibility(self):
        """Toggle private key visibility"""
        try:
            if self.private_key_entry.cget("show") == "*":
                self.private_key_entry.config(show="")
            else:
                self.private_key_entry.config(show="*")
        except Exception as e:
            logger.error(f"Error toggling private key visibility: {e}")
    
    def generate_wallet(self):
        """Generate a new wallet with auto-connection"""
        try:
            # Create a new random account
            acct = Account.create()
            new_address = acct.address
            new_privkey = acct.key.hex()
            
            # Update the entry fields
            self.wallet_address_var.set(new_address)
            self.private_key_var.set(new_privkey)
            
            logger.info(f"Generated new wallet: {new_address}")
            
            # Auto-connect if enabled
            if self.auto_connect_var.get():
                # Save credentials first
                self.save_credentials_silent()
                
                # Connect with new wallet
                self.connect_with_credentials(new_privkey, new_address)
                
                messagebox.showinfo(
                    "Wallet Generated",
                    f"New wallet generated and auto-connected!\n\nAddress: {new_address}\n\nCredentials have been saved."
                )
            else:
                messagebox.showinfo(
                    "Wallet Generated",
                    f"New wallet generated successfully!\n\nAddress: {new_address}\n\nPlease save your credentials."
                )
        except Exception as e:
            logger.error(f"Failed to generate wallet: {e}")
            messagebox.showerror("Error", f"Failed to generate wallet: {e}")
    
    def save_credentials(self):
        """Save credentials with user feedback"""
        try:
            # Get values
            private_key = self.private_key_var.get().strip()
            wallet_address = self.wallet_address_var.get().strip()
            
            # Validate inputs
            if not private_key:
                messagebox.showerror("Error", "Please enter a private key")
                return
            
            if not wallet_address:
                messagebox.showerror("Error", "Please enter a wallet address")
                return
            
            # Save credentials
            if self.api.save_credentials(private_key, wallet_address):
                logger.info("Credentials saved successfully")
                messagebox.showinfo("Success", "Credentials saved successfully")
                
                # Update API network setting
                self.config_manager.set('trading.testnet', self.testnet_var.get())
                self.config_manager.save_config()
            else:
                logger.error("Failed to save credentials")
                messagebox.showerror("Error", "Failed to save credentials")
        except Exception as e:
            logger.error(f"Error saving credentials: {e}")
            messagebox.showerror("Error", f"Error saving credentials: {e}")
    
    def save_credentials_silent(self):
        """Save credentials without user feedback"""
        try:
            # Get values
            private_key = self.private_key_var.get().strip()
            wallet_address = self.wallet_address_var.get().strip()
            
            # Validate inputs
            if not private_key or not wallet_address:
                return False
            
            # Save credentials
            if self.api.save_credentials(private_key, wallet_address):
                logger.info("Credentials saved silently")
                
                # Update API network setting
                self.config_manager.set('trading.testnet', self.testnet_var.get())
                self.config_manager.save_config()
                
                return True
            else:
                logger.error("Failed to save credentials silently")
                return False
        except Exception as e:
            logger.error(f"Error saving credentials silently: {e}")
            return False
    
    def test_connection(self):
        """Test connection to Hyperliquid API"""
        # Disable button during test
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Button):
                widget.config(state="disabled")
        
        # Start test in a separate thread
        threading.Thread(target=self._test_connection_thread, daemon=True).start()
    
    def _test_connection_thread(self):
        """Run connection test in a separate thread"""
        try:
            # Get values
            private_key = self.private_key_var.get().strip()
            wallet_address = self.wallet_address_var.get().strip()
            
            # Validate inputs
            if not private_key:
                self.root.after(0, lambda: messagebox.showerror("Error", "Please enter a private key"))
                self.root.after(0, self._enable_buttons)
                return
            
            if not wallet_address:
                self.root.after(0, lambda: messagebox.showerror("Error", "Please enter a wallet address"))
                self.root.after(0, self._enable_buttons)
                return
            
            # Test connection
            logger.info("Testing connection...")
            success = self.api.test_connection(private_key, wallet_address)
            
            if success:
                logger.info("Connection test successful")
                self.root.after(0, lambda: self.update_connection_status(True, wallet_address))
                self.root.after(0, lambda: messagebox.showinfo("Success", "Connection test successful"))
                
                # Refresh account info
                self.root.after(0, self.refresh_account_info)
            else:
                logger.error("Connection test failed")
                self.root.after(0, lambda: self.update_connection_status(False))
                self.root.after(0, lambda: messagebox.showerror("Error", "Connection test failed. Please check your credentials."))
        except Exception as e:
            logger.error(f"Error testing connection: {e}")
            self.root.after(0, lambda: self.update_connection_status(False))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error testing connection: {e}"))
        finally:
            # Re-enable buttons
            self.root.after(0, self._enable_buttons)
    
    def connect_with_credentials(self, private_key, wallet_address):
        """Connect with the provided credentials"""
        try:
            # Initialize API if needed
            if not self.api:
                self.api = EnhancedHyperliquidAPI(testnet=self.testnet_var.get())
            
            # Initialize exchange
            success = self.api.initialize_exchange(private_key, wallet_address)
            
            if success:
                logger.info(f"Connected with address: {wallet_address}")
                self.update_connection_status(True, wallet_address)
                
                # Refresh account info
                self.refresh_account_info()
                
                return True
            else:
                logger.error("Failed to connect with credentials")
                self.update_connection_status(False)
                return False
        except Exception as e:
            logger.error(f"Error connecting with credentials: {e}")
            self.update_connection_status(False)
            return False
    
    def update_connection_status(self, connected, address=None):
        """Update connection status display"""
        if connected:
            self.connection_status_var.set("Connected")
            self.connection_status_label.config(fg="green")
            
            if address:
                self.account_address_label.config(text=address, fg="green")
        else:
            self.connection_status_var.set("Not Connected")
            self.connection_status_label.config(fg="red")
            self.account_address_label.config(text="Not connected", fg="red")
    
    def refresh_account_info(self):
        """Refresh account information"""
        try:
            # Get wallet address
            wallet_address = self.wallet_address_var.get().strip()
            if not wallet_address:
                wallet_address = self.config_manager.get('trading.wallet_address')
                if not wallet_address:
                    logger.warning("No wallet address available")
                    return
            
            # Get account state
            if self.api:
                account_state = self.api.get_account_state(wallet_address)
                
                if account_state and "marginSummary" in account_state:
                    margin_summary = account_state["marginSummary"]
                    
                    # Update account value
                    if "accountValue" in margin_summary:
                        account_value = float(margin_summary["accountValue"])
                        self.account_balance_label.config(text=f"${account_value:.2f}")
                    
                    # Update margin used
                    if "totalMarginUsed" in margin_summary:
                        margin_used = float(margin_summary["totalMarginUsed"])
                        self.margin_used_label.config(text=f"${margin_used:.2f}")
                    
                    logger.info("Account information refreshed")
                else:
                    logger.warning("No account information available")
        except Exception as e:
            logger.error(f"Error refreshing account information: {e}")
    
    def _enable_buttons(self):
        """Re-enable all buttons"""
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Button):
                widget.config(state="normal")
    
    def toggle_trading(self):
        """Toggle trading status"""
        if self.trading_status_var.get() == "Active":
            self.trading_status_var.set("Inactive")
            self.trading_button_text.set("Start Trading")
            self.trading_status_var.set("Inactive")
            logger.info("Trading stopped")
        else:
            # Check connection first
            if self.connection_status_var.get() != "Connected":
                messagebox.showerror("Error", "Please connect to Hyperliquid first")
                return
            
            self.trading_status_var.set("Active")
            self.trading_button_text.set("Stop Trading")
            logger.info("Trading started")
    
    def update_api_network(self):
        """Update API network setting"""
        try:
            if self.api:
                self.api.testnet = self.testnet_var.get()
                logger.info(f"API network updated: {'testnet' if self.testnet_var.get() else 'mainnet'}")
                
                # Update config
                self.config_manager.set('trading.testnet', self.testnet_var.get())
                self.config_manager.save_config()
        except Exception as e:
            logger.error(f"Error updating API network: {e}")
    
    def use_default_credentials(self):
        """Use default credentials"""
        try:
            # Set default credentials
            self.wallet_address_var.set(self.default_address)
            self.private_key_var.set(self.default_private_key)
            
            # Save credentials
            self.save_credentials_silent()
            
            # Connect with default credentials
            success = self.connect_with_credentials(self.default_private_key, self.default_address)
            
            if success:
                messagebox.showinfo("Success", "Connected with default credentials")
            else:
                messagebox.showerror("Error", "Failed to connect with default credentials")
        except Exception as e:
            logger.error(f"Error using default credentials: {e}")
            messagebox.showerror("Error", f"Error using default credentials: {e}")
    
    def auto_connect_on_startup(self):
        """Auto-connect on startup"""
        try:
            # Try to get saved credentials
            private_key = self.security_manager.get_private_key()
            wallet_address = self.config_manager.get('trading.wallet_address')
            
            if private_key and wallet_address:
                # Connect with saved credentials
                success = self.connect_with_credentials(private_key, wallet_address)
                
                if success:
                    logger.info("Auto-connected with saved credentials")
                else:
                    logger.warning("Failed to auto-connect with saved credentials")
            else:
                # Use default credentials
                logger.info("No saved credentials found, using default credentials")
                self.use_default_credentials()
        except Exception as e:
            logger.error(f"Error auto-connecting on startup: {e}")
    
    def run(self):
        """Run the application"""
        self.root.mainloop()


def main():
    """Main entry point"""
    app = AutoConnectTradingDashboard()
    app.run()


if __name__ == "__main__":
    main()

