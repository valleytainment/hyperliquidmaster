#!/usr/bin/env python3
"""
GUI Style Module for HyperliquidMaster
--------------------------------------
This module provides styling and theming for the HyperliquidMaster GUI.
It defines color schemes, custom styles for ttk widgets, and helper functions
for applying consistent styling across the application.
"""

import tkinter as tk
from tkinter import ttk
import os
import json

# Color schemes
THEMES = {
    "dark": {
        "bg": "#1E1E2E",  # Dark background
        "fg": "#CDD6F4",  # Light text
        "accent": "#89B4FA",  # Blue accent
        "accent_dark": "#74C7EC",  # Darker blue for hover
        "success": "#A6E3A1",  # Green for positive values
        "warning": "#F9E2AF",  # Yellow for warnings
        "error": "#F38BA8",  # Red for errors
        "chart_bg": "#313244",  # Slightly lighter background for charts
        "button_bg": "#45475A",  # Button background
        "entry_bg": "#313244",  # Entry field background
        "border": "#6C7086",  # Border color
        "highlight": "#F5C2E7",  # Highlight color
        "tab_active": "#313244",  # Active tab background
        "tab_inactive": "#1E1E2E",  # Inactive tab background
    },
    "light": {
        "bg": "#EFF1F5",  # Light background
        "fg": "#4C4F69",  # Dark text
        "accent": "#1E66F5",  # Blue accent
        "accent_dark": "#1E66F5",  # Darker blue for hover
        "success": "#40A02B",  # Green for positive values
        "warning": "#DF8E1D",  # Yellow for warnings
        "error": "#D20F39",  # Red for errors
        "chart_bg": "#DCE0E8",  # Slightly darker background for charts
        "button_bg": "#CCD0DA",  # Button background
        "entry_bg": "#DCE0E8",  # Entry field background
        "border": "#9CA0B0",  # Border color
        "highlight": "#EA76CB",  # Highlight color
        "tab_active": "#DCE0E8",  # Active tab background
        "tab_inactive": "#EFF1F5",  # Inactive tab background
    }
}

# Default theme
DEFAULT_THEME = "dark"

class GUIStyleManager:
    """
    Manages the styling of the GUI components.
    """
    def __init__(self, root, theme_name=DEFAULT_THEME):
        """
        Initialize the style manager.
        
        Args:
            root: The root Tkinter window
            theme_name: The name of the theme to use
        """
        self.root = root
        self.theme_name = theme_name
        self.colors = THEMES[theme_name]
        self.style = ttk.Style()
        
        # Apply the theme
        self.apply_theme()
    
    def apply_theme(self):
        """Apply the selected theme to all widgets."""
        # Configure ttk style
        self.style.configure("TFrame", background=self.colors["bg"])
        self.style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["fg"])
        self.style.configure("TButton", 
                            background=self.colors["button_bg"], 
                            foreground=self.colors["fg"],
                            borderwidth=1,
                            focusthickness=3,
                            focuscolor=self.colors["accent"])
        self.style.map("TButton",
                      background=[("active", self.colors["accent"]), 
                                 ("pressed", self.colors["accent_dark"])],
                      foreground=[("active", "#FFFFFF"), 
                                 ("pressed", "#FFFFFF")])
        
        # Configure Entry style
        self.style.configure("TEntry", 
                            fieldbackground=self.colors["entry_bg"],
                            foreground=self.colors["fg"],
                            bordercolor=self.colors["border"],
                            lightcolor=self.colors["border"],
                            darkcolor=self.colors["border"],
                            insertcolor=self.colors["fg"])
        
        # Configure Combobox style
        self.style.configure("TCombobox", 
                            fieldbackground=self.colors["entry_bg"],
                            background=self.colors["button_bg"],
                            foreground=self.colors["fg"],
                            arrowcolor=self.colors["fg"],
                            bordercolor=self.colors["border"])
        self.style.map("TCombobox",
                      fieldbackground=[("readonly", self.colors["entry_bg"])],
                      selectbackground=[("readonly", self.colors["accent"])])
        
        # Configure Checkbutton style
        self.style.configure("TCheckbutton", 
                            background=self.colors["bg"],
                            foreground=self.colors["fg"])
        self.style.map("TCheckbutton",
                      background=[("active", self.colors["bg"])],
                      foreground=[("active", self.colors["accent"])])
        
        # Configure Notebook style
        self.style.configure("TNotebook", 
                            background=self.colors["bg"],
                            borderwidth=0)
        self.style.configure("TNotebook.Tab", 
                            background=self.colors["tab_inactive"],
                            foreground=self.colors["fg"],
                            padding=[10, 5],
                            borderwidth=0)
        self.style.map("TNotebook.Tab",
                      background=[("selected", self.colors["tab_active"])],
                      foreground=[("selected", self.colors["accent"])])
        
        # Configure Scrollbar style
        self.style.configure("TScrollbar", 
                            background=self.colors["button_bg"],
                            troughcolor=self.colors["bg"],
                            bordercolor=self.colors["border"],
                            arrowcolor=self.colors["fg"])
        
        # Configure Progressbar style
        self.style.configure("TProgressbar", 
                            background=self.colors["accent"],
                            troughcolor=self.colors["bg"],
                            bordercolor=self.colors["border"])
        
        # Configure Separator style
        self.style.configure("TSeparator", 
                            background=self.colors["border"])
        
        # Configure root window
        self.root.configure(background=self.colors["bg"])
        
        # Create custom styles for specific widgets
        self.create_custom_styles()
    
    def create_custom_styles(self):
        """Create custom styles for specific widgets."""
        # Success button style
        self.style.configure("Success.TButton", 
                            background=self.colors["success"],
                            foreground="#FFFFFF")
        self.style.map("Success.TButton",
                      background=[("active", self.colors["success"]), 
                                 ("pressed", self.colors["success"])])
        
        # Warning button style
        self.style.configure("Warning.TButton", 
                            background=self.colors["warning"],
                            foreground="#000000")
        self.style.map("Warning.TButton",
                      background=[("active", self.colors["warning"]), 
                                 ("pressed", self.colors["warning"])])
        
        # Error button style
        self.style.configure("Error.TButton", 
                            background=self.colors["error"],
                            foreground="#FFFFFF")
        self.style.map("Error.TButton",
                      background=[("active", self.colors["error"]), 
                                 ("pressed", self.colors["error"])])
        
        # Header label style
        self.style.configure("Header.TLabel", 
                            font=("TkDefaultFont", 12, "bold"),
                            foreground=self.colors["accent"])
        
        # Subheader label style
        self.style.configure("Subheader.TLabel", 
                            font=("TkDefaultFont", 10, "bold"))
        
        # Value label style (for displaying values)
        self.style.configure("Value.TLabel", 
                            font=("TkDefaultFont", 10),
                            foreground=self.colors["accent"])
        
        # Positive value label style
        self.style.configure("Positive.TLabel", 
                            foreground=self.colors["success"])
        
        # Negative value label style
        self.style.configure("Negative.TLabel", 
                            foreground=self.colors["error"])
    
    def get_matplotlib_style(self):
        """
        Get the matplotlib style parameters based on the current theme.
        
        Returns:
            dict: A dictionary of matplotlib style parameters
        """
        return {
            "figure.facecolor": self.colors["chart_bg"],
            "axes.facecolor": self.colors["chart_bg"],
            "axes.edgecolor": self.colors["border"],
            "axes.labelcolor": self.colors["fg"],
            "axes.grid": True,
            "grid.color": self.colors["border"],
            "grid.alpha": 0.3,
            "xtick.color": self.colors["fg"],
            "ytick.color": self.colors["fg"],
            "text.color": self.colors["fg"],
            "lines.color": self.colors["accent"],
            "patch.edgecolor": self.colors["border"],
            "savefig.facecolor": self.colors["chart_bg"],
            "savefig.edgecolor": self.colors["border"],
        }
    
    def switch_theme(self, theme_name):
        """
        Switch to a different theme.
        
        Args:
            theme_name: The name of the theme to switch to
        """
        if theme_name in THEMES:
            self.theme_name = theme_name
            self.colors = THEMES[theme_name]
            self.apply_theme()
            return True
        return False
    
    def save_theme_preference(self, config_path):
        """
        Save the current theme preference to the config file.
        
        Args:
            config_path: Path to the config file
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            config['theme'] = self.theme_name
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error saving theme preference: {e}")
            return False
    
    @staticmethod
    def load_theme_preference(config_path):
        """
        Load the theme preference from the config file.
        
        Args:
            config_path: Path to the config file
            
        Returns:
            str: The name of the theme to use
        """
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                if 'theme' in config:
                    return config['theme']
        except Exception as e:
            print(f"Error loading theme preference: {e}")
        
        return DEFAULT_THEME

# Helper functions for creating styled widgets
def create_header_label(parent, text):
    """Create a header label with the appropriate style."""
    return ttk.Label(parent, text=text, style="Header.TLabel")

def create_subheader_label(parent, text):
    """Create a subheader label with the appropriate style."""
    return ttk.Label(parent, text=text, style="Subheader.TLabel")

def create_value_label(parent, text):
    """Create a value label with the appropriate style."""
    return ttk.Label(parent, text=text, style="Value.TLabel")

def create_positive_label(parent, text):
    """Create a positive value label with the appropriate style."""
    return ttk.Label(parent, text=text, style="Positive.TLabel")

def create_negative_label(parent, text):
    """Create a negative value label with the appropriate style."""
    return ttk.Label(parent, text=text, style="Negative.TLabel")

def create_success_button(parent, text, command):
    """Create a success button with the appropriate style."""
    return ttk.Button(parent, text=text, command=command, style="Success.TButton")

def create_warning_button(parent, text, command):
    """Create a warning button with the appropriate style."""
    return ttk.Button(parent, text=text, command=command, style="Warning.TButton")

def create_error_button(parent, text, command):
    """Create an error button with the appropriate style."""
    return ttk.Button(parent, text=text, command=command, style="Error.TButton")
