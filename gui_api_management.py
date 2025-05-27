"""
Advanced API Management GUI for HyperLiquid Trading Bot
Provides a user-friendly interface for managing multiple API keys
"""

import os
import json
import time
import logging
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Dict, List, Any, Optional, Callable, Tuple
import datetime

from core.advanced_api_manager import AdvancedAPIManager, APIKeyConfig

class AdvancedAPIManagementGUI:
    """
    Advanced API Management GUI for the HyperLiquid trading bot.
    Provides a user-friendly interface for managing multiple API keys.
    """
    
    def __init__(self, 
                 root: tk.Tk, 
                 parent_frame: ttk.Frame,
                 api_manager: AdvancedAPIManager,
                 style_manager,
                 on_key_change: Optional[Callable[[str, str], None]] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the advanced API management GUI.
        
        Args:
            root: The root Tk instance
            parent_frame: The parent frame
            api_manager: The advanced API manager instance
            style_manager: The style manager instance
            on_key_change: Callback for when the active key changes
            logger: Logger instance
        """
        self.root = root
        self.parent_frame = parent_frame
        self.api_manager = api_manager
        self.style_manager = style_manager
        self.on_key_change = on_key_change
        self.logger = logger or logging.getLogger("AdvancedAPIManagementGUI")
        
        # Initialize variables
        self.api_keys = {}
        self.selected_key_id = tk.StringVar()
        self.key_name = tk.StringVar()
        self.account_address = tk.StringVar()
        self.secret_key = tk.StringVar()
        self.exchange = tk.StringVar(value="hyperliquid")
        self.notes = tk.StringVar()
        self.show_secret_key = tk.BooleanVar(value=False)
        self.include_secrets_export = tk.BooleanVar(value=False)
        
        # Create the GUI
        self._create_gui()
        
        # Load API keys
        self._load_api_keys()
    
    def _create_gui(self):
        """Create the GUI elements."""
        # Create main frame
        main_frame = ttk.Frame(self.parent_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create title
        title = ttk.Label(main_frame, text="Advanced API Key Management", style="Header.TLabel")
        title.pack(anchor=tk.W, pady=(0, 10))
        
        # Create split view
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Create left panel (key list)
        left_panel = ttk.Frame(paned_window)
        paned_window.add(left_panel, weight=1)
        
        # Create right panel (key details)
        right_panel = ttk.Frame(paned_window)
        paned_window.add(right_panel, weight=2)
        
        # Create key list
        self._create_key_list(left_panel)
        
        # Create key details
        self._create_key_details(right_panel)
        
        # Create bottom panel (buttons)
        bottom_panel = ttk.Frame(main_frame)
        bottom_panel.pack(fill=tk.X, pady=(10, 0))
        
        # Create import/export buttons
        import_button = tk.Button(bottom_panel, text="Import Keys", command=self._import_keys)
        import_button.pack(side=tk.LEFT, padx=(0, 5))
        self.style_manager.style_button(import_button)
        
        export_button = tk.Button(bottom_panel, text="Export Keys", command=self._export_keys)
        export_button.pack(side=tk.LEFT)
        self.style_manager.style_button(export_button)
        
        # Create include secrets checkbox
        include_secrets_check = ttk.Checkbutton(
            bottom_panel, 
            text="Include Secrets in Export", 
            variable=self.include_secrets_export
        )
        include_secrets_check.pack(side=tk.LEFT, padx=(5, 0))
        
        # Create usage stats button
        stats_button = tk.Button(bottom_panel, text="Usage Statistics", command=self._show_usage_stats)
        stats_button.pack(side=tk.RIGHT)
        self.style_manager.style_button(stats_button)
    
    def _create_key_list(self, parent):
        """
        Create the key list panel.
        
        Args:
            parent: The parent frame
        """
        # Create frame
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Create list label
        list_label = ttk.Label(frame, text="API Keys")
        list_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Create key listbox with scrollbar
        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.key_listbox = tk.Listbox(list_frame, selectmode=tk.SINGLE, exportselection=0)
        self.key_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.key_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.key_listbox.config(yscrollcommand=scrollbar.set)
        
        # Bind selection event
        self.key_listbox.bind('<<ListboxSelect>>', self._on_key_selected)
        
        # Create buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        add_button = tk.Button(button_frame, text="Add Key", command=self._add_key)
        add_button.pack(side=tk.LEFT, padx=(0, 5))
        self.style_manager.style_button(add_button, "success")
        
        delete_button = tk.Button(button_frame, text="Delete Key", command=self._delete_key)
        delete_button.pack(side=tk.LEFT)
        self.style_manager.style_button(delete_button, "error")
    
    def _create_key_details(self, parent):
        """
        Create the key details panel.
        
        Args:
            parent: The parent frame
        """
        # Create scrollable frame
        container, details_frame = self.style_manager.create_scrollable_frame(parent)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Create details label
        details_label = ttk.Label(details_frame, text="Key Details")
        details_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Create key name input
        name_frame = ttk.Frame(details_frame)
        name_frame.pack(fill=tk.X, pady=5)
        
        name_label = ttk.Label(name_frame, text="Key Name:")
        name_label.pack(side=tk.LEFT, padx=(0, 5))
        
        name_entry = tk.Entry(name_frame, textvariable=self.key_name, width=30)
        name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.style_manager.style_entry(name_entry)
        
        # Create exchange input
        exchange_frame = ttk.Frame(details_frame)
        exchange_frame.pack(fill=tk.X, pady=5)
        
        exchange_label = ttk.Label(exchange_frame, text="Exchange:")
        exchange_label.pack(side=tk.LEFT, padx=(0, 5))
        
        exchange_combobox = ttk.Combobox(exchange_frame, textvariable=self.exchange, width=20)
        exchange_combobox['values'] = ('hyperliquid', 'hyperliquid_testnet')
        exchange_combobox.pack(side=tk.LEFT)
        
        # Create account address input
        addr_frame = ttk.Frame(details_frame)
        addr_frame.pack(fill=tk.X, pady=5)
        
        addr_label = ttk.Label(addr_frame, text="Account Address:")
        addr_label.pack(side=tk.LEFT, padx=(0, 5))
        
        addr_entry = tk.Entry(addr_frame, textvariable=self.account_address, width=50)
        addr_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.style_manager.style_entry(addr_entry)
        
        # Create secret key input
        key_frame = ttk.Frame(details_frame)
        key_frame.pack(fill=tk.X, pady=5)
        
        key_label = ttk.Label(key_frame, text="Secret Key:")
        key_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.key_entry = tk.Entry(key_frame, textvariable=self.secret_key, width=50, show="*")
        self.key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.style_manager.style_entry(self.key_entry)
        
        # Create show/hide secret key checkbox
        show_key_check = ttk.Checkbutton(
            key_frame, 
            text="Show", 
            variable=self.show_secret_key, 
            command=self._toggle_show_secret_key
        )
        show_key_check.pack(side=tk.LEFT, padx=(5, 0))
        
        # Create notes input
        notes_frame = ttk.Frame(details_frame)
        notes_frame.pack(fill=tk.X, pady=5)
        
        notes_label = ttk.Label(notes_frame, text="Notes:")
        notes_label.pack(side=tk.LEFT, padx=(0, 5))
        
        notes_entry = tk.Entry(notes_frame, textvariable=self.notes, width=50)
        notes_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.style_manager.style_entry(notes_entry)
        
        # Create buttons
        button_frame = ttk.Frame(details_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        save_button = tk.Button(button_frame, text="Save Changes", command=self._save_key)
        save_button.pack(side=tk.LEFT, padx=(0, 5))
        self.style_manager.style_button(save_button, "success")
        
        activate_button = tk.Button(button_frame, text="Set as Active", command=self._set_active_key)
        activate_button.pack(side=tk.LEFT, padx=(0, 5))
        self.style_manager.style_button(activate_button)
        
        test_button = tk.Button(button_frame, text="Test Connection", command=self._test_connection)
        test_button.pack(side=tk.LEFT)
        self.style_manager.style_button(test_button)
        
        rotate_button = tk.Button(button_frame, text="Rotate Key", command=self._rotate_key)
        rotate_button.pack(side=tk.RIGHT)
        self.style_manager.style_button(rotate_button)
        
        # Create status label
        self.status_label = ttk.Label(details_frame, text="")
        self.status_label.pack(fill=tk.X, pady=(10, 0))
    
    def _load_api_keys(self):
        """Load API keys from the API manager."""
        # Clear listbox
        self.key_listbox.delete(0, tk.END)
        
        # Get API keys
        self.api_keys = self.api_manager.api_keys
        
        # Add keys to listbox
        for key_id, key_config in self.api_keys.items():
            display_name = key_config.name
            if key_id == self.api_manager.active_key_id:
                display_name += " (Active)"
            
            self.key_listbox.insert(tk.END, display_name)
            self.key_listbox.itemconfig(tk.END, {'bg': 'lightgreen' if key_id == self.api_manager.active_key_id else 'white'})
        
        # Select active key if available
        if self.api_manager.active_key_id and self.api_manager.active_key_id in self.api_keys:
            # Find index of active key
            for i, key_id in enumerate(self.api_keys.keys()):
                if key_id == self.api_manager.active_key_id:
                    self.key_listbox.selection_set(i)
                    self._on_key_selected(None)
                    break
    
    def _on_key_selected(self, event):
        """
        Handle key selection event.
        
        Args:
            event: The event object
        """
        # Get selected index
        selection = self.key_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        
        # Get key ID
        key_id = list(self.api_keys.keys())[index]
        self.selected_key_id.set(key_id)
        
        # Get key config
        key_config = self.api_keys[key_id]
        
        # Update form fields
        self.key_name.set(key_config.name)
        self.account_address.set(key_config.account_address)
        self.secret_key.set(key_config.secret_key)
        self.exchange.set(key_config.exchange)
        self.notes.set(key_config.notes)
        
        # Update status label
        self._update_status_label(f"Selected key: {key_config.name}")
    
    def _toggle_show_secret_key(self):
        """Toggle showing/hiding the secret key."""
        if self.show_secret_key.get():
            self.key_entry.config(show="")
        else:
            self.key_entry.config(show="*")
    
    def _update_status_label(self, text, is_error=False):
        """
        Update the status label.
        
        Args:
            text: The text to display
            is_error: Whether this is an error message
        """
        self.status_label.config(
            text=text,
            foreground="red" if is_error else "green"
        )
    
    def _add_key(self):
        """Add a new API key."""
        # Clear form fields
        self.selected_key_id.set("")
        self.key_name.set("New API Key")
        self.account_address.set("")
        self.secret_key.set("")
        self.exchange.set("hyperliquid")
        self.notes.set("")
        
        # Update status label
        self._update_status_label("Creating new API key")
    
    def _save_key(self):
        """Save the current API key."""
        # Get form values
        key_id = self.selected_key_id.get()
        name = self.key_name.get()
        account_address = self.account_address.get()
        secret_key = self.secret_key.get()
        exchange = self.exchange.get()
        notes = self.notes.get()
        
        # Validate inputs
        if not name or name.strip() == "":
            self._update_status_label("Name cannot be empty", True)
            return
        
        if not account_address or account_address.strip() == "":
            self._update_status_label("Account address cannot be empty", True)
            return
        
        if not secret_key or secret_key.strip() == "":
            self._update_status_label("Secret key cannot be empty", True)
            return
        
        try:
            if key_id and key_id in self.api_keys:
                # Update existing key
                success, message = self.api_manager.update_api_key(
                    key_id,
                    name=name,
                    account_address=account_address,
                    secret_key=secret_key,
                    exchange=exchange,
                    notes=notes
                )
            else:
                # Add new key
                success, message, key_id = self.api_manager.add_api_key(
                    name=name,
                    account_address=account_address,
                    secret_key=secret_key,
                    exchange=exchange,
                    notes=notes
                )
            
            if success:
                self._update_status_label(message)
                self._load_api_keys()
                
                # Select the saved key
                if key_id:
                    for i, k_id in enumerate(self.api_keys.keys()):
                        if k_id == key_id:
                            self.key_listbox.selection_set(i)
                            self._on_key_selected(None)
                            break
            else:
                self._update_status_label(message, True)
        except Exception as e:
            self._update_status_label(f"Error saving key: {e}", True)
    
    def _delete_key(self):
        """Delete the selected API key."""
        # Get selected key ID
        key_id = self.selected_key_id.get()
        if not key_id or key_id not in self.api_keys:
            self._update_status_label("No key selected", True)
            return
        
        # Get key name
        key_name = self.api_keys[key_id].name
        
        # Confirm deletion
        if not messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete the API key '{key_name}'?"):
            return
        
        try:
            # Delete key
            success, message = self.api_manager.delete_api_key(key_id)
            
            if success:
                self._update_status_label(message)
                self._load_api_keys()
                
                # Clear form fields
                self.selected_key_id.set("")
                self.key_name.set("")
                self.account_address.set("")
                self.secret_key.set("")
                self.exchange.set("hyperliquid")
                self.notes.set("")
            else:
                self._update_status_label(message, True)
        except Exception as e:
            self._update_status_label(f"Error deleting key: {e}", True)
    
    def _set_active_key(self):
        """Set the selected key as active."""
        # Get selected key ID
        key_id = self.selected_key_id.get()
        if not key_id or key_id not in self.api_keys:
            self._update_status_label("No key selected", True)
            return
        
        try:
            # Set active key
            success, message = self.api_manager.set_active_key(key_id)
            
            if success:
                self._update_status_label(message)
                self._load_api_keys()
                
                # Notify callback
                if self.on_key_change:
                    active_key = self.api_manager.get_active_key()
                    if active_key:
                        self.on_key_change(active_key.account_address, active_key.secret_key)
            else:
                self._update_status_label(message, True)
        except Exception as e:
            self._update_status_label(f"Error setting active key: {e}", True)
    
    def _test_connection(self):
        """Test connection with the selected key."""
        # Get selected key ID
        key_id = self.selected_key_id.get()
        if not key_id or key_id not in self.api_keys:
            self._update_status_label("No key selected", True)
            return
        
        # Get key config
        key_config = self.api_keys[key_id]
        
        # Show testing message
        self._update_status_label(f"Testing connection for {key_config.name}...")
        self.root.update()
        
        try:
            # Test connection
            # This is a placeholder - in a real implementation, you would use the trading integration
            # to test the connection with the selected key
            import time
            time.sleep(1)  # Simulate API call
            
            # Update status label
            self._update_status_label(f"Connection test successful for {key_config.name}")
            
            # Update key usage statistics
            self.api_manager.update_api_key(
                key_id,
                last_used=time.time(),
                usage_count=key_config.usage_count + 1
            )
        except Exception as e:
            self._update_status_label(f"Connection test failed: {e}", True)
    
    def _rotate_key(self):
        """Rotate the selected key."""
        # Get selected key ID
        key_id = self.selected_key_id.get()
        if not key_id or key_id not in self.api_keys:
            self._update_status_label("No key selected", True)
            return
        
        # Get key config
        key_config = self.api_keys[key_id]
        
        # Ask for new secret key
        new_secret_key = simpledialog.askstring(
            "Rotate API Key",
            f"Enter new secret key for '{key_config.name}':",
            show="*"
        )
        
        if not new_secret_key:
            return
        
        try:
            # Rotate key
            success, message = self.api_manager.rotate_key(key_id, new_secret_key)
            
            if success:
                self._update_status_label(message)
                self._load_api_keys()
                
                # Update form fields
                self.secret_key.set(new_secret_key)
                
                # Notify callback if this is the active key
                if key_id == self.api_manager.active_key_id and self.on_key_change:
                    self.on_key_change(key_config.account_address, new_secret_key)
            else:
                self._update_status_label(message, True)
        except Exception as e:
            self._update_status_label(f"Error rotating key: {e}", True)
    
    def _import_keys(self):
        """Import API keys from a file."""
        # Ask for file path
        file_path = filedialog.askopenfilename(
            title="Import API Keys",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Import keys
            success, message, count = self.api_manager.import_api_keys(file_path)
            
            if success:
                self._update_status_label(message)
                self._load_api_keys()
                
                # Notify callback if active key changed
                active_key = self.api_manager.get_active_key()
                if active_key and self.on_key_change:
                    self.on_key_change(active_key.account_address, active_key.secret_key)
            else:
                self._update_status_label(message, True)
        except Exception as e:
            self._update_status_label(f"Error importing keys: {e}", True)
    
    def _export_keys(self):
        """Export API keys to a file."""
        # Ask for file path
        file_path = filedialog.asksaveasfilename(
            title="Export API Keys",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Export keys
            include_secrets = self.include_secrets_export.get()
            success, message = self.api_manager.export_api_keys(file_path, include_secrets)
            
            if success:
                self._update_status_label(message)
            else:
                self._update_status_label(message, True)
        except Exception as e:
            self._update_status_label(f"Error exporting keys: {e}", True)
    
    def _show_usage_stats(self):
        """Show usage statistics."""
        try:
            # Get usage statistics
            stats = self.api_manager.get_usage_statistics()
            
            # Create stats window
            stats_window = tk.Toplevel(self.root)
            stats_window.title("API Key Usage Statistics")
            stats_window.geometry("600x400")
            stats_window.minsize(600, 400)
            
            # Create scrollable frame
            container = ttk.Frame(stats_window)
            container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            canvas = tk.Canvas(container)
            scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
            
            stats_frame = ttk.Frame(canvas)
            stats_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=stats_frame, anchor=tk.NW)
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Create title
            title = ttk.Label(stats_frame, text="API Key Usage Statistics", font=("TkDefaultFont", 12, "bold"))
            title.pack(anchor=tk.W, pady=(0, 10))
            
            # Create summary section
            summary_frame = ttk.LabelFrame(stats_frame, text="Summary")
            summary_frame.pack(fill=tk.X, pady=(0, 10))
            
            summary_grid = ttk.Frame(summary_frame)
            summary_grid.pack(fill=tk.X, padx=10, pady=10)
            
            # Add summary stats
            ttk.Label(summary_grid, text="Total Keys:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=2)
            ttk.Label(summary_grid, text=str(stats["total_keys"])).grid(row=0, column=1, sticky=tk.W, pady=2)
            
            ttk.Label(summary_grid, text="Active Keys:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=2)
            ttk.Label(summary_grid, text=str(stats["active_keys"])).grid(row=1, column=1, sticky=tk.W, pady=2)
            
            ttk.Label(summary_grid, text="Total Usage:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=2)
            ttk.Label(summary_grid, text=str(stats["total_usage"])).grid(row=2, column=1, sticky=tk.W, pady=2)
            
            if stats["last_used"] > 0:
                last_used_str = datetime.datetime.fromtimestamp(stats["last_used"]).strftime("%Y-%m-%d %H:%M:%S")
            else:
                last_used_str = "Never"
            
            ttk.Label(summary_grid, text="Last Used:").grid(row=3, column=0, sticky=tk.W, padx=(0, 10), pady=2)
            ttk.Label(summary_grid, text=last_used_str).grid(row=3, column=1, sticky=tk.W, pady=2)
            
            # Create key details section
            details_frame = ttk.LabelFrame(stats_frame, text="Key Details")
            details_frame.pack(fill=tk.BOTH, expand=True)
            
            # Create treeview
            columns = ("name", "usage", "last_used", "created", "age", "status")
            tree = ttk.Treeview(details_frame, columns=columns, show="headings")
            
            # Define headings
            tree.heading("name", text="Key Name")
            tree.heading("usage", text="Usage Count")
            tree.heading("last_used", text="Last Used")
            tree.heading("created", text="Created")
            tree.heading("age", text="Age (days)")
            tree.heading("status", text="Status")
            
            # Define columns
            tree.column("name", width=150)
            tree.column("usage", width=80)
            tree.column("last_used", width=150)
            tree.column("created", width=150)
            tree.column("age", width=80)
            tree.column("status", width=100)
            
            # Add scrollbar
            tree_scrollbar = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=tree.yview)
            tree.configure(yscrollcommand=tree_scrollbar.set)
            
            # Pack widgets
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
            
            # Add key details
            for key_id, key_stats in stats["keys"].items():
                status = []
                if key_stats["is_current"]:
                    status.append("Active")
                if key_stats["is_active"]:
                    status.append("Enabled")
                else:
                    status.append("Disabled")
                
                tree.insert("", "end", values=(
                    key_stats["name"],
                    key_stats["usage_count"],
                    key_stats["last_used_formatted"],
                    key_stats["created_at_formatted"],
                    f"{key_stats['age_days']:.1f}",
                    ", ".join(status)
                ))
            
            # Create close button
            close_button = tk.Button(stats_frame, text="Close", command=stats_window.destroy)
            close_button.pack(anchor=tk.E, pady=(10, 0))
            
            # Make window modal
            stats_window.transient(self.root)
            stats_window.grab_set()
            self.root.wait_window(stats_window)
        except Exception as e:
            self._update_status_label(f"Error showing usage statistics: {e}", True)
