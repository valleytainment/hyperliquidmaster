"""
Integration of Advanced API Management into the main GUI for HyperLiquid Trading Bot
"""

import os
import logging
import tkinter as tk
from tkinter import ttk, messagebox

from core.advanced_api_manager import AdvancedAPIManager
from gui_api_management import AdvancedAPIManagementGUI

def integrate_advanced_api_management(gui_main_instance):
    """
    Integrate advanced API management into the main GUI.
    
    Args:
        gui_main_instance: The main GUI instance
    """
    # Get references to necessary components
    root = gui_main_instance.root
    settings_tab = gui_main_instance.settings_tab
    style_manager = gui_main_instance.style_manager
    logger = gui_main_instance.logger
    config_dir = os.path.dirname(gui_main_instance.config_path)
    
    # Initialize advanced API manager
    api_manager = AdvancedAPIManager(config_dir, logger)
    
    # Define callback for key changes
    def on_key_change(account_address, secret_key):
        # Update trading integration with new keys
        result = gui_main_instance.trading.set_api_keys(account_address, secret_key)
        
        if result.get("success", False):
            logger.info("API keys updated successfully")
            gui_main_instance._update_connection_status()
        else:
            logger.error(f"Failed to update API keys: {result.get('message', 'Unknown error')}")
    
    # Create advanced API management section in settings tab
    api_section = ttk.LabelFrame(settings_tab, text="Advanced API Management")
    api_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=(20, 10))
    
    # Create advanced API management GUI
    api_gui = AdvancedAPIManagementGUI(
        root=root,
        parent_frame=api_section,
        api_manager=api_manager,
        style_manager=style_manager,
        on_key_change=on_key_change,
        logger=logger
    )
    
    # Store references for later use
    gui_main_instance.api_manager = api_manager
    gui_main_instance.api_gui = api_gui
    
    # Log integration
    logger.info("Advanced API management integrated into GUI")
    
    return api_gui
