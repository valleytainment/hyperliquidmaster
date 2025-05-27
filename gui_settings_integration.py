"""
Enhanced GUI settings integration for the HyperliquidMaster trading bot.
Ensures flawless API key handling and settings persistence in the GUI.
"""

import os
import json
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any, Optional, Callable

from core.settings_manager import SettingsManager
from core.trading_integration import TradingIntegration

class GUISettingsIntegration:
    """
    GUI settings integration for the HyperliquidMaster trading bot.
    Handles the integration between the GUI and the settings manager.
    """
    
    def __init__(self, 
                 root: tk.Tk, 
                 settings_manager: SettingsManager, 
                 trading_integration: TradingIntegration,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the GUI settings integration.
        
        Args:
            root: The root Tk instance
            settings_manager: The settings manager instance
            trading_integration: The trading integration instance
            logger: Logger instance
        """
        self.root = root
        self.settings_manager = settings_manager
        self.trading_integration = trading_integration
        self.logger = logger or logging.getLogger("GUISettingsIntegration")
        
        # Initialize variables
        self.account_address = tk.StringVar()
        self.secret_key = tk.StringVar()
        self.show_secret_key = tk.BooleanVar(value=False)
        
        # Load initial values
        self._load_initial_values()
    
    def _load_initial_values(self) -> None:
        """Load initial values from settings."""
        settings = self.settings_manager.settings
        self.account_address.set(settings.get("account_address", ""))
        self.secret_key.set(settings.get("secret_key", ""))
    
    def create_settings_tab(self, parent: ttk.Frame, style_manager) -> ttk.Frame:
        """
        Create the settings tab.
        
        Args:
            parent: The parent frame
            style_manager: The style manager instance
            
        Returns:
            The settings frame
        """
        # Create scrollable frame
        container, settings_frame = style_manager.create_scrollable_frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create API key management section
        api_frame = ttk.Frame(settings_frame)
        api_frame.pack(fill=tk.X, pady=(0, 20))
        
        api_title = ttk.Label(api_frame, text="API Key Management", style="Header.TLabel")
        api_title.pack(anchor=tk.W, pady=(0, 10))
        
        # Create account address input
        addr_frame = ttk.Frame(api_frame)
        addr_frame.pack(fill=tk.X, pady=5)
        
        addr_label = ttk.Label(addr_frame, text="Account Address:")
        addr_label.pack(side=tk.LEFT, padx=(0, 5))
        
        addr_entry = tk.Entry(addr_frame, textvariable=self.account_address, width=50)
        addr_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        style_manager.style_entry(addr_entry)
        
        # Create secret key input
        key_frame = ttk.Frame(api_frame)
        key_frame.pack(fill=tk.X, pady=5)
        
        key_label = ttk.Label(key_frame, text="Secret Key:")
        key_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.key_entry = tk.Entry(key_frame, textvariable=self.secret_key, width=50, show="*")
        self.key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        style_manager.style_entry(self.key_entry)
        
        # Create show/hide secret key checkbox
        show_key_check = ttk.Checkbutton(
            key_frame, 
            text="Show", 
            variable=self.show_secret_key, 
            command=self._toggle_show_secret_key
        )
        show_key_check.pack(side=tk.LEFT, padx=(5, 0))
        
        # Create API key buttons
        api_buttons_frame = ttk.Frame(api_frame)
        api_buttons_frame.pack(fill=tk.X, pady=5)
        
        save_api_button = tk.Button(
            api_buttons_frame, 
            text="Save API Keys", 
            command=self._save_api_keys
        )
        save_api_button.pack(side=tk.LEFT, padx=(0, 5))
        style_manager.style_button(save_api_button, "success")
        
        test_api_button = tk.Button(
            api_buttons_frame, 
            text="Test Connection", 
            command=self._test_connection
        )
        test_api_button.pack(side=tk.LEFT)
        style_manager.style_button(test_api_button)
        
        # Create load from file button
        load_api_button = tk.Button(
            api_buttons_frame, 
            text="Load from File", 
            command=self._load_api_keys_from_file
        )
        load_api_button.pack(side=tk.LEFT, padx=(5, 0))
        style_manager.style_button(load_api_button)
        
        # Create status indicator
        self.api_status_label = ttk.Label(api_frame, text="")
        self.api_status_label.pack(fill=tk.X, pady=5)
        
        return settings_frame
    
    def _toggle_show_secret_key(self) -> None:
        """Toggle showing/hiding the secret key."""
        if self.show_secret_key.get():
            self.key_entry.config(show="")
        else:
            self.key_entry.config(show="*")
    
    def _save_api_keys(self) -> None:
        """Save API keys to settings."""
        account_address = self.account_address.get().strip()
        secret_key = self.secret_key.get().strip()
        
        # Check if keys are empty
        if not account_address or not secret_key:
            self.api_status_label.config(
                text="Account address and secret key cannot be empty",
                foreground="red"
            )
            messagebox.showerror("Error", "Account address and secret key cannot be empty")
            return
        
        # Validate and save API keys
        success, message = self.settings_manager.save_api_keys(account_address, secret_key)
        
        if success:
            # Update trading integration with new keys
            result = self.trading_integration.set_api_keys(account_address, secret_key)
            
            if result.get("success", False):
                self.api_status_label.config(
                    text="API keys saved and applied successfully",
                    foreground="green"
                )
                messagebox.showinfo("Success", "API keys saved and applied successfully")
                
                # Force connection test to update connection status immediately
                self.logger.info("Testing connection with new API keys")
                connection_result = self.trading_integration.test_connection()
                if connection_result.get("success", False):
                    self.logger.info("Connection established with new API keys")
                else:
                    self.logger.warning(f"Connection test with new keys failed: {connection_result.get('message', 'Unknown error')}")
            else:
                self.api_status_label.config(
                    text=f"API keys saved but failed to apply: {result.get('message', 'Unknown error')}",
                    foreground="orange"
                )
                messagebox.showwarning(
                    "Warning", 
                    f"API keys saved but failed to apply: {result.get('message', 'Unknown error')}"
                )
        else:
            self.api_status_label.config(
                text=f"Failed to save API keys: {message}",
                foreground="red"
            )
            messagebox.showerror("Error", f"Failed to save API keys: {message}")
    
    def _test_connection(self) -> None:
        """Test connection to the exchange."""
        account_address = self.account_address.get().strip()
        secret_key = self.secret_key.get().strip()
        
        # Check if keys are empty
        if not account_address or not secret_key:
            self.api_status_label.config(
                text="Account address and secret key cannot be empty",
                foreground="red"
            )
            messagebox.showerror("Error", "Account address and secret key cannot be empty")
            return
            
        # First save the current keys
        success, message = self.settings_manager.save_api_keys(account_address, secret_key)
        if not success:
            self.api_status_label.config(
                text=f"Failed to save API keys: {message}",
                foreground="red"
            )
            messagebox.showerror("Error", f"Failed to save API keys: {message}")
            return
            
        # Update trading integration with new keys
        key_result = self.trading_integration.set_api_keys(account_address, secret_key)
        if not key_result.get("success", False):
            self.api_status_label.config(
                text=f"Failed to apply API keys: {key_result.get('message', 'Unknown error')}",
                foreground="red"
            )
            messagebox.showerror("Error", f"Failed to apply API keys: {key_result.get('message', 'Unknown error')}")
            return
        
        # Test connection
        self.api_status_label.config(
            text="Testing connection...",
            foreground="blue"
        )
        self.root.update()  # Force UI update to show testing status
        
        result = self.trading_integration.test_connection()
        
        if result.get("success", False):
            self.api_status_label.config(
                text="Connection test successful",
                foreground="green"
            )
            messagebox.showinfo("Success", "Connection test successful")
        else:
            self.api_status_label.config(
                text=f"Connection test failed: {result.get('message', 'Unknown error')}",
                foreground="red"
            )
            messagebox.showerror(
                "Error", 
                f"Connection test failed: {result.get('message', 'Unknown error')}"
            )
    
    def _load_api_keys_from_file(self) -> None:
        """Load API keys from a file."""
        try:
            from tkinter import filedialog
            
            # Ask for file path
            file_path = filedialog.askopenfilename(
                title="Select API Keys File",
                filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
            )
            
            if not file_path:
                return
            
            # Load keys from file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract keys - support multiple formats
            account_address = ""
            secret_key = ""
            
            # Try direct fields
            if "account_address" in data and "secret_key" in data:
                account_address = data.get("account_address", "")
                secret_key = data.get("secret_key", "")
            # Try nested in settings
            elif "settings" in data and isinstance(data["settings"], dict):
                account_address = data["settings"].get("account_address", "")
                secret_key = data["settings"].get("secret_key", "")
            # Try first key in keys dict if using advanced API format
            elif "keys" in data and isinstance(data["keys"], dict) and data["keys"]:
                first_key = next(iter(data["keys"].values()))
                if isinstance(first_key, dict):
                    account_address = first_key.get("account_address", "")
                    secret_key = first_key.get("secret_key", "")
            
            if not account_address or not secret_key:
                messagebox.showerror(
                    "Error", 
                    "Invalid API keys file format. File must contain account_address and secret_key fields."
                )
                return
            
            # Set keys in UI
            self.account_address.set(account_address)
            self.secret_key.set(secret_key)
            
            # Save and apply keys
            self.logger.info(f"Loaded API keys from file: {file_path}")
            self.api_status_label.config(
                text="API keys loaded from file. Click Save to apply.",
                foreground="blue"
            )
            
            # Auto-save and test connection
            self._save_api_keys()
            
        except Exception as e:
            self.logger.error(f"Error loading API keys from file: {e}")
            messagebox.showerror("Error", f"Error loading API keys from file: {e}")
    
    def update_from_settings(self) -> None:
        """Update UI from settings."""
        self._load_initial_values()
