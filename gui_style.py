"""
GUI Style Manager for HyperliquidMaster.

This module provides styling and theming functionality for the GUI.
"""

import tkinter as tk
from tkinter import ttk
import logging
from typing import Dict, Any, Optional

class GUIStyleManager:
    """
    Manages GUI styling and theming.
    
    This class provides functionality for applying different themes and styles
    to the GUI components.
    """
    
    def __init__(self, root: tk.Tk, logger: logging.Logger):
        """
        Initialize the GUI style manager.
        
        Args:
            root: The root Tk instance
            logger: Logger instance
        """
        self.root = root
        self.logger = logger
        self.current_theme = "dark"  # Default theme
        
        # Initialize styles
        self._initialize_styles()
        
        # Apply default theme
        self.apply_theme(self.current_theme)
        
        self.logger.info(f"Applied {self.current_theme} theme")
    
    def _initialize_styles(self) -> None:
        """Initialize ttk styles."""
        self.style = ttk.Style()
        
        # Create custom styles
        self.style.configure("TFrame", padding=5)
        self.style.configure("TButton", padding=5, font=("Helvetica", 10))
        self.style.configure("TLabel", padding=5, font=("Helvetica", 10))
        self.style.configure("TEntry", padding=5, font=("Helvetica", 10))
        self.style.configure("TCheckbutton", padding=5, font=("Helvetica", 10))
        self.style.configure("TRadiobutton", padding=5, font=("Helvetica", 10))
        self.style.configure("TCombobox", padding=5, font=("Helvetica", 10))
        self.style.configure("TSpinbox", padding=5, font=("Helvetica", 10))
        self.style.configure("TNotebook", padding=5)
        self.style.configure("TNotebook.Tab", padding=(10, 5), font=("Helvetica", 10))
        
        # Create custom styles for specific widgets
        self.style.configure("Header.TLabel", font=("Helvetica", 14, "bold"))
        self.style.configure("Subheader.TLabel", font=("Helvetica", 12, "bold"))
        self.style.configure("Status.TLabel", font=("Helvetica", 10, "italic"))
        self.style.configure("Error.TLabel", foreground="red", font=("Helvetica", 10, "bold"))
        self.style.configure("Success.TLabel", foreground="green", font=("Helvetica", 10, "bold"))
        self.style.configure("Warning.TLabel", foreground="orange", font=("Helvetica", 10, "bold"))
        self.style.configure("Primary.TButton", font=("Helvetica", 10, "bold"))
        self.style.configure("Secondary.TButton", font=("Helvetica", 10))
        self.style.configure("Danger.TButton", font=("Helvetica", 10, "bold"))
    
    def apply_theme(self, theme_name: str) -> None:
        """
        Apply a theme to the GUI.
        
        Args:
            theme_name: Name of the theme to apply
        """
        self.current_theme = theme_name
        
        if theme_name == "dark":
            self._apply_dark_theme()
        elif theme_name == "light":
            self._apply_light_theme()
        elif theme_name == "blue":
            self._apply_blue_theme()
        elif theme_name == "green":
            self._apply_green_theme()
        else:
            self.logger.warning(f"Unknown theme: {theme_name}, defaulting to dark")
            self._apply_dark_theme()
    
    def _apply_dark_theme(self) -> None:
        """Apply dark theme."""
        # Configure colors
        bg_color = "#2E2E2E"
        fg_color = "#FFFFFF"
        accent_color = "#3498DB"
        button_bg = "#3E3E3E"
        button_fg = "#FFFFFF"
        entry_bg = "#3E3E3E"
        entry_fg = "#FFFFFF"
        
        # Configure root
        self.root.configure(background=bg_color)
        
        # Configure ttk styles
        self.style.configure(".", background=bg_color, foreground=fg_color)
        self.style.configure("TFrame", background=bg_color)
        self.style.configure("TButton", background=button_bg, foreground=button_fg)
        self.style.configure("TLabel", background=bg_color, foreground=fg_color)
        self.style.configure("TEntry", fieldbackground=entry_bg, foreground=entry_fg)
        self.style.configure("TCheckbutton", background=bg_color, foreground=fg_color)
        self.style.configure("TRadiobutton", background=bg_color, foreground=fg_color)
        self.style.configure("TCombobox", fieldbackground=entry_bg, foreground=entry_fg)
        self.style.configure("TSpinbox", fieldbackground=entry_bg, foreground=entry_fg)
        self.style.configure("TNotebook", background=bg_color)
        self.style.configure("TNotebook.Tab", background=button_bg, foreground=button_fg)
        
        # Configure custom styles
        self.style.configure("Header.TLabel", background=bg_color, foreground=accent_color)
        self.style.configure("Subheader.TLabel", background=bg_color, foreground=accent_color)
        self.style.configure("Status.TLabel", background=bg_color, foreground=fg_color)
        self.style.configure("Error.TLabel", background=bg_color, foreground="#E74C3C")
        self.style.configure("Success.TLabel", background=bg_color, foreground="#2ECC71")
        self.style.configure("Warning.TLabel", background=bg_color, foreground="#F39C12")
        self.style.configure("Primary.TButton", background=accent_color, foreground=button_fg)
        self.style.configure("Secondary.TButton", background=button_bg, foreground=button_fg)
        self.style.configure("Danger.TButton", background="#E74C3C", foreground=button_fg)
        
        # Configure maps for hover effects
        self.style.map("TButton",
                      foreground=[('pressed', button_fg), ('active', button_fg)],
                      background=[('pressed', '!disabled', '#2E2E2E'), ('active', '#4E4E4E')])
        self.style.map("Primary.TButton",
                      foreground=[('pressed', button_fg), ('active', button_fg)],
                      background=[('pressed', '!disabled', '#2980B9'), ('active', '#3498DB')])
        self.style.map("Secondary.TButton",
                      foreground=[('pressed', button_fg), ('active', button_fg)],
                      background=[('pressed', '!disabled', '#2E2E2E'), ('active', '#4E4E4E')])
        self.style.map("Danger.TButton",
                      foreground=[('pressed', button_fg), ('active', button_fg)],
                      background=[('pressed', '!disabled', '#C0392B'), ('active', '#E74C3C')])
    
    def _apply_light_theme(self) -> None:
        """Apply light theme."""
        # Configure colors
        bg_color = "#F5F5F5"
        fg_color = "#333333"
        accent_color = "#3498DB"
        button_bg = "#E0E0E0"
        button_fg = "#333333"
        entry_bg = "#FFFFFF"
        entry_fg = "#333333"
        
        # Configure root
        self.root.configure(background=bg_color)
        
        # Configure ttk styles
        self.style.configure(".", background=bg_color, foreground=fg_color)
        self.style.configure("TFrame", background=bg_color)
        self.style.configure("TButton", background=button_bg, foreground=button_fg)
        self.style.configure("TLabel", background=bg_color, foreground=fg_color)
        self.style.configure("TEntry", fieldbackground=entry_bg, foreground=entry_fg)
        self.style.configure("TCheckbutton", background=bg_color, foreground=fg_color)
        self.style.configure("TRadiobutton", background=bg_color, foreground=fg_color)
        self.style.configure("TCombobox", fieldbackground=entry_bg, foreground=entry_fg)
        self.style.configure("TSpinbox", fieldbackground=entry_bg, foreground=entry_fg)
        self.style.configure("TNotebook", background=bg_color)
        self.style.configure("TNotebook.Tab", background=button_bg, foreground=button_fg)
        
        # Configure custom styles
        self.style.configure("Header.TLabel", background=bg_color, foreground=accent_color)
        self.style.configure("Subheader.TLabel", background=bg_color, foreground=accent_color)
        self.style.configure("Status.TLabel", background=bg_color, foreground=fg_color)
        self.style.configure("Error.TLabel", background=bg_color, foreground="#E74C3C")
        self.style.configure("Success.TLabel", background=bg_color, foreground="#2ECC71")
        self.style.configure("Warning.TLabel", background=bg_color, foreground="#F39C12")
        self.style.configure("Primary.TButton", background=accent_color, foreground="#FFFFFF")
        self.style.configure("Secondary.TButton", background=button_bg, foreground=button_fg)
        self.style.configure("Danger.TButton", background="#E74C3C", foreground="#FFFFFF")
        
        # Configure maps for hover effects
        self.style.map("TButton",
                      foreground=[('pressed', button_fg), ('active', button_fg)],
                      background=[('pressed', '!disabled', '#CCCCCC'), ('active', '#D0D0D0')])
        self.style.map("Primary.TButton",
                      foreground=[('pressed', "#FFFFFF"), ('active', "#FFFFFF")],
                      background=[('pressed', '!disabled', '#2980B9'), ('active', '#3498DB')])
        self.style.map("Secondary.TButton",
                      foreground=[('pressed', button_fg), ('active', button_fg)],
                      background=[('pressed', '!disabled', '#CCCCCC'), ('active', '#D0D0D0')])
        self.style.map("Danger.TButton",
                      foreground=[('pressed', "#FFFFFF"), ('active', "#FFFFFF")],
                      background=[('pressed', '!disabled', '#C0392B'), ('active', '#E74C3C')])
    
    def _apply_blue_theme(self) -> None:
        """Apply blue theme."""
        # Configure colors
        bg_color = "#1A2530"
        fg_color = "#FFFFFF"
        accent_color = "#3498DB"
        button_bg = "#2C3E50"
        button_fg = "#FFFFFF"
        entry_bg = "#2C3E50"
        entry_fg = "#FFFFFF"
        
        # Configure root
        self.root.configure(background=bg_color)
        
        # Configure ttk styles
        self.style.configure(".", background=bg_color, foreground=fg_color)
        self.style.configure("TFrame", background=bg_color)
        self.style.configure("TButton", background=button_bg, foreground=button_fg)
        self.style.configure("TLabel", background=bg_color, foreground=fg_color)
        self.style.configure("TEntry", fieldbackground=entry_bg, foreground=entry_fg)
        self.style.configure("TCheckbutton", background=bg_color, foreground=fg_color)
        self.style.configure("TRadiobutton", background=bg_color, foreground=fg_color)
        self.style.configure("TCombobox", fieldbackground=entry_bg, foreground=entry_fg)
        self.style.configure("TSpinbox", fieldbackground=entry_bg, foreground=entry_fg)
        self.style.configure("TNotebook", background=bg_color)
        self.style.configure("TNotebook.Tab", background=button_bg, foreground=button_fg)
        
        # Configure custom styles
        self.style.configure("Header.TLabel", background=bg_color, foreground=accent_color)
        self.style.configure("Subheader.TLabel", background=bg_color, foreground=accent_color)
        self.style.configure("Status.TLabel", background=bg_color, foreground=fg_color)
        self.style.configure("Error.TLabel", background=bg_color, foreground="#E74C3C")
        self.style.configure("Success.TLabel", background=bg_color, foreground="#2ECC71")
        self.style.configure("Warning.TLabel", background=bg_color, foreground="#F39C12")
        self.style.configure("Primary.TButton", background=accent_color, foreground=button_fg)
        self.style.configure("Secondary.TButton", background=button_bg, foreground=button_fg)
        self.style.configure("Danger.TButton", background="#E74C3C", foreground=button_fg)
        
        # Configure maps for hover effects
        self.style.map("TButton",
                      foreground=[('pressed', button_fg), ('active', button_fg)],
                      background=[('pressed', '!disabled', '#1A2530'), ('active', '#34495E')])
        self.style.map("Primary.TButton",
                      foreground=[('pressed', button_fg), ('active', button_fg)],
                      background=[('pressed', '!disabled', '#2980B9'), ('active', '#3498DB')])
        self.style.map("Secondary.TButton",
                      foreground=[('pressed', button_fg), ('active', button_fg)],
                      background=[('pressed', '!disabled', '#1A2530'), ('active', '#34495E')])
        self.style.map("Danger.TButton",
                      foreground=[('pressed', button_fg), ('active', button_fg)],
                      background=[('pressed', '!disabled', '#C0392B'), ('active', '#E74C3C')])
    
    def _apply_green_theme(self) -> None:
        """Apply green theme."""
        # Configure colors
        bg_color = "#1E2E1E"
        fg_color = "#FFFFFF"
        accent_color = "#2ECC71"
        button_bg = "#2E3E2E"
        button_fg = "#FFFFFF"
        entry_bg = "#2E3E2E"
        entry_fg = "#FFFFFF"
        
        # Configure root
        self.root.configure(background=bg_color)
        
        # Configure ttk styles
        self.style.configure(".", background=bg_color, foreground=fg_color)
        self.style.configure("TFrame", background=bg_color)
        self.style.configure("TButton", background=button_bg, foreground=button_fg)
        self.style.configure("TLabel", background=bg_color, foreground=fg_color)
        self.style.configure("TEntry", fieldbackground=entry_bg, foreground=entry_fg)
        self.style.configure("TCheckbutton", background=bg_color, foreground=fg_color)
        self.style.configure("TRadiobutton", background=bg_color, foreground=fg_color)
        self.style.configure("TCombobox", fieldbackground=entry_bg, foreground=entry_fg)
        self.style.configure("TSpinbox", fieldbackground=entry_bg, foreground=entry_fg)
        self.style.configure("TNotebook", background=bg_color)
        self.style.configure("TNotebook.Tab", background=button_bg, foreground=button_fg)
        
        # Configure custom styles
        self.style.configure("Header.TLabel", background=bg_color, foreground=accent_color)
        self.style.configure("Subheader.TLabel", background=bg_color, foreground=accent_color)
        self.style.configure("Status.TLabel", background=bg_color, foreground=fg_color)
        self.style.configure("Error.TLabel", background=bg_color, foreground="#E74C3C")
        self.style.configure("Success.TLabel", background=bg_color, foreground="#2ECC71")
        self.style.configure("Warning.TLabel", background=bg_color, foreground="#F39C12")
        self.style.configure("Primary.TButton", background=accent_color, foreground=button_fg)
        self.style.configure("Secondary.TButton", background=button_bg, foreground=button_fg)
        self.style.configure("Danger.TButton", background="#E74C3C", foreground=button_fg)
        
        # Configure maps for hover effects
        self.style.map("TButton",
                      foreground=[('pressed', button_fg), ('active', button_fg)],
                      background=[('pressed', '!disabled', '#1E2E1E'), ('active', '#3E4E3E')])
        self.style.map("Primary.TButton",
                      foreground=[('pressed', button_fg), ('active', button_fg)],
                      background=[('pressed', '!disabled', '#27AE60'), ('active', '#2ECC71')])
        self.style.map("Secondary.TButton",
                      foreground=[('pressed', button_fg), ('active', button_fg)],
                      background=[('pressed', '!disabled', '#1E2E1E'), ('active', '#3E4E3E')])
        self.style.map("Danger.TButton",
                      foreground=[('pressed', button_fg), ('active', button_fg)],
                      background=[('pressed', '!disabled', '#C0392B'), ('active', '#E74C3C')])
    
    def get_current_theme(self) -> str:
        """
        Get the current theme name.
        
        Returns:
            Current theme name
        """
        return self.current_theme
    
    def get_available_themes(self) -> list:
        """
        Get list of available themes.
        
        Returns:
            List of available theme names
        """
        return ["dark", "light", "blue", "green"]
    
    def create_custom_widget(self, parent: tk.Widget, widget_type: str, style: str = None, **kwargs) -> tk.Widget:
        """
        Create a custom styled widget.
        
        Args:
            parent: Parent widget
            widget_type: Type of widget to create
            style: Style to apply
            **kwargs: Additional widget parameters
            
        Returns:
            Created widget
        """
        if style:
            kwargs["style"] = style
        
        if widget_type == "label":
            return ttk.Label(parent, **kwargs)
        elif widget_type == "button":
            return ttk.Button(parent, **kwargs)
        elif widget_type == "entry":
            return ttk.Entry(parent, **kwargs)
        elif widget_type == "frame":
            return ttk.Frame(parent, **kwargs)
        elif widget_type == "checkbutton":
            return ttk.Checkbutton(parent, **kwargs)
        elif widget_type == "radiobutton":
            return ttk.Radiobutton(parent, **kwargs)
        elif widget_type == "combobox":
            return ttk.Combobox(parent, **kwargs)
        elif widget_type == "spinbox":
            return ttk.Spinbox(parent, **kwargs)
        elif widget_type == "notebook":
            return ttk.Notebook(parent, **kwargs)
        else:
            self.logger.warning(f"Unknown widget type: {widget_type}")
            return None
