"""
Simplified Enhanced GUI for Hyperliquid Trading Bot
Focused on fixing private key input and connection testing issues
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import asyncio
import logging
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.config_manager import ConfigManager
from utils.security_fixed_v2 import SecurityManager
from core.api_fixed import EnhancedHyperliquidAPI
from eth_account import Account

logger = get_logger(__name__)


class SimpleTradingDashboard:
    """Simplified trading dashboard focused on fixing private key input and connection testing"""
    
    def __init__(self):
        """Initialize the trading dashboard"""
        self.root = tk.Tk()
        self.root.title("Hyperliquid Trading Bot")
        self.root.geometry("600x500")
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.security_manager = SecurityManager()
        self.api = None
        
        # Variables
        self.private_key_var = tk.StringVar()
        self.wallet_address_var = tk.StringVar()
        self.show_key_var = tk.BooleanVar(value=False)
        self.testnet_var = tk.BooleanVar(value=False)
        
        # Load existing wallet address if available
        try:
            existing_address = self.config_manager.get('trading.wallet_address', '')
            if existing_address:
                self.wallet_address_var.set(existing_address)
        except Exception as e:
            logger.error(f"Error loading wallet address: {e}")
        
        # Setup GUI
        self.setup_gui()
        
        logger.info("Simplified Trading Dashboard initialized")
    
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Main frame
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)
        
        # Title
        title_label = tk.Label(main_frame, text="Hyperliquid Trading Bot", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Wallet Address Section
        wallet_frame = tk.LabelFrame(main_frame, text="Wallet Address", padx=10, pady=10)
        wallet_frame.pack(fill="x", pady=10)
        
        wallet_label = tk.Label(wallet_frame, text="Enter your wallet address (0x...):")
        wallet_label.pack(anchor="w")
        
        self.wallet_entry = tk.Entry(wallet_frame, textvariable=self.wallet_address_var, width=50)
        self.wallet_entry.pack(fill="x", pady=5)
        
        # Private Key Section
        key_frame = tk.LabelFrame(main_frame, text="Private Key", padx=10, pady=10)
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
        
        # Testnet Option
        testnet_frame = tk.Frame(main_frame)
        testnet_frame.pack(fill="x", pady=5)
        
        testnet_check = tk.Checkbutton(
            testnet_frame,
            text="Use Testnet",
            variable=self.testnet_var
        )
        testnet_check.pack(anchor="w")
        
        # Generate Wallet Button
        generate_frame = tk.Frame(main_frame)
        generate_frame.pack(fill="x", pady=10)
        
        generate_btn = tk.Button(
            generate_frame,
            text="Generate New Wallet",
            command=self.generate_wallet,
            width=20
        )
        generate_btn.pack(anchor="w")
        
        # Action Buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill="x", pady=20)
        
        test_btn = tk.Button(
            button_frame,
            text="Test Connection",
            command=self.test_connection,
            width=15
        )
        test_btn.pack(side="left", padx=(0, 10))
        
        save_btn = tk.Button(
            button_frame,
            text="Save Credentials",
            command=self.save_credentials,
            width=15
        )
        save_btn.pack(side="left")
        
        # Status Frame
        status_frame = tk.LabelFrame(main_frame, text="Status", padx=10, pady=10)
        status_frame.pack(fill="both", expand=True, pady=10)
        
        self.status_text = tk.Text(status_frame, height=8, wrap="word")
        self.status_text.pack(fill="both", expand=True)
        
        # Redirect logger to status text
        self.setup_text_handler()
    
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
        if self.private_key_entry.cget("show") == "*":
            self.private_key_entry.config(show="")
        else:
            self.private_key_entry.config(show="*")
    
    def generate_wallet(self):
        """Generate a new wallet"""
        try:
            # Create a new random account
            acct = Account.create()
            new_address = acct.address
            new_privkey = acct.key.hex()
            
            # Update the entry fields
            self.wallet_address_var.set(new_address)
            self.private_key_var.set(new_privkey)
            
            logger.info(f"Generated new wallet: {new_address}")
            messagebox.showinfo(
                "Wallet Generated",
                f"New wallet generated successfully!\n\nAddress: {new_address}\n\nPlease save your credentials."
            )
        except Exception as e:
            logger.error(f"Failed to generate wallet: {e}")
            messagebox.showerror("Error", f"Failed to generate wallet: {e}")
    
    def save_credentials(self):
        """Save credentials"""
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
            
            # Save wallet address to config
            self.config_manager.set('trading.wallet_address', wallet_address)
            self.config_manager.set('trading.testnet', self.testnet_var.get())
            self.config_manager.save_config()
            
            # Save private key securely
            success = self.security_manager.store_private_key(private_key)
            
            if success:
                logger.info("Credentials saved successfully")
                messagebox.showinfo("Success", "Credentials saved successfully")
            else:
                logger.error("Failed to save private key")
                messagebox.showerror("Error", "Failed to save private key")
        except Exception as e:
            logger.error(f"Error saving credentials: {e}")
            messagebox.showerror("Error", f"Error saving credentials: {e}")
    
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
            use_testnet = self.testnet_var.get()
            
            # Validate inputs
            if not private_key:
                self.root.after(0, lambda: messagebox.showerror("Error", "Please enter a private key"))
                return
            
            if not wallet_address:
                self.root.after(0, lambda: messagebox.showerror("Error", "Please enter a wallet address"))
                return
            
            # Initialize API if needed
            if not self.api:
                self.api = EnhancedHyperliquidAPI(testnet=use_testnet)
            
            # Test connection
            logger.info("Testing connection...")
            success = self.api.test_connection(private_key, wallet_address)
            
            if success:
                logger.info("Connection test successful")
                self.root.after(0, lambda: messagebox.showinfo("Success", "Connection test successful"))
            else:
                logger.error("Connection test failed")
                self.root.after(0, lambda: messagebox.showerror("Error", "Connection test failed. Please check your credentials."))
        except Exception as e:
            logger.error(f"Error testing connection: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error testing connection: {e}"))
        finally:
            # Re-enable buttons
            self.root.after(0, self._enable_buttons)
    
    def _enable_buttons(self):
        """Re-enable all buttons"""
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Button):
                widget.config(state="normal")
    
    def run(self):
        """Run the application"""
        self.root.mainloop()


def main():
    """Main entry point"""
    app = SimpleTradingDashboard()
    app.run()


if __name__ == "__main__":
    main()

