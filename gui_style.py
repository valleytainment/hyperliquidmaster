#!/usr/bin/env python3
"""
GUI Style Manager for HyperliquidMaster
Provides consistent styling and theming across the application
"""

import tkinter as tk
from tkinter import ttk
import json
import os
from typing import Dict, Any, Optional

class GUIStyleManager:
    """
    Manages GUI styling and theming for the application.
    Provides consistent colors, fonts, and widget styles.
    """
    
    def __init__(self, root: tk.Tk, theme_name: str = "dark"):
        """
        Initialize the style manager with the specified theme.
        
        Args:
            root: The root Tkinter window
            theme_name: The name of the theme to use (default: "dark")
        """
        self.root = root
        self.theme_name = theme_name
        
        # Define themes
        self.themes = {
            "dark": {
                # Base colors
                "bg_color": "#121212",
                "fg_color": "#FFFFFF",
                "accent_color": "#00FFAA",  # Neon green
                "secondary_accent": "#FF00FF",  # Neon pink
                "tertiary_accent": "#00FFFF",  # Neon cyan
                
                # Button colors
                "button_bg": "#1E1E1E",
                "button_fg": "#FFFFFF",
                "button_active_bg": "#333333",
                "button_active_fg": "#FFFFFF",
                
                # Success/Error colors
                "success_color": "#00FF66",  # Bright neon green
                "warning_color": "#FFFF00",  # Bright yellow
                "error_color": "#FF3366",    # Bright pink-red
                
                # Chart colors
                "chart_bg": "#1A1A1A",
                "chart_fg": "#FFFFFF",
                "chart_grid": "#333333",
                "chart_line": "#00FFAA",
                "chart_candle_up": "#00FF66",
                "chart_candle_down": "#FF3366",
                
                # Text colors
                "text_primary": "#FFFFFF",
                "text_secondary": "#AAAAAA",
                "text_highlight": "#00FFAA",
                
                # Fonts
                "font_family": "Helvetica",
                "header_size": 14,
                "subheader_size": 12,
                "body_size": 10,
                "small_size": 9,
                
                # Borders and padding
                "border_color": "#333333",
                "border_width": 1,
                "padding": 5,
                
                # Tab colors
                "tab_bg": "#1E1E1E",
                "tab_fg": "#FFFFFF",
                "tab_active_bg": "#333333",
                "tab_active_fg": "#00FFAA",
                
                # Scrollbar colors
                "scrollbar_bg": "#1E1E1E",
                "scrollbar_fg": "#00FFAA",
                
                # Entry field colors
                "entry_bg": "#1E1E1E",
                "entry_fg": "#FFFFFF",
                "entry_highlight": "#00FFAA",
                
                # Table colors
                "table_header_bg": "#333333",
                "table_header_fg": "#FFFFFF",
                "table_row_bg": "#1E1E1E",
                "table_row_alt_bg": "#262626",
                "table_row_fg": "#FFFFFF",
                "table_border": "#444444"
            },
            "light": {
                # Base colors
                "bg_color": "#F5F5F5",
                "fg_color": "#333333",
                "accent_color": "#007BFF",
                "secondary_accent": "#6610F2",
                "tertiary_accent": "#17A2B8",
                
                # Button colors
                "button_bg": "#E0E0E0",
                "button_fg": "#333333",
                "button_active_bg": "#CCCCCC",
                "button_active_fg": "#333333",
                
                # Success/Error colors
                "success_color": "#28A745",
                "warning_color": "#FFC107",
                "error_color": "#DC3545",
                
                # Chart colors
                "chart_bg": "#FFFFFF",
                "chart_fg": "#333333",
                "chart_grid": "#DDDDDD",
                "chart_line": "#007BFF",
                "chart_candle_up": "#28A745",
                "chart_candle_down": "#DC3545",
                
                # Text colors
                "text_primary": "#333333",
                "text_secondary": "#666666",
                "text_highlight": "#007BFF",
                
                # Fonts
                "font_family": "Helvetica",
                "header_size": 14,
                "subheader_size": 12,
                "body_size": 10,
                "small_size": 9,
                
                # Borders and padding
                "border_color": "#DDDDDD",
                "border_width": 1,
                "padding": 5,
                
                # Tab colors
                "tab_bg": "#E0E0E0",
                "tab_fg": "#333333",
                "tab_active_bg": "#FFFFFF",
                "tab_active_fg": "#007BFF",
                
                # Scrollbar colors
                "scrollbar_bg": "#E0E0E0",
                "scrollbar_fg": "#999999",
                
                # Entry field colors
                "entry_bg": "#FFFFFF",
                "entry_fg": "#333333",
                "entry_highlight": "#007BFF",
                
                # Table colors
                "table_header_bg": "#E0E0E0",
                "table_header_fg": "#333333",
                "table_row_bg": "#FFFFFF",
                "table_row_alt_bg": "#F5F5F5",
                "table_row_fg": "#333333",
                "table_border": "#DDDDDD"
            }
        }
        
        # Apply the theme
        self.apply_theme(theme_name)
    
    def apply_theme(self, theme_name: str):
        """
        Apply the specified theme to the application.
        
        Args:
            theme_name: The name of the theme to apply
        """
        if theme_name not in self.themes:
            theme_name = "dark"  # Default to dark theme if not found
        
        self.theme_name = theme_name
        theme = self.themes[theme_name]
        
        # Configure ttk styles
        style = ttk.Style()
        
        # Configure the root theme
        style.theme_use('clam')  # Use clam as base theme for better customization
        
        # Configure colors
        style.configure('.',
                        background=theme["bg_color"],
                        foreground=theme["fg_color"],
                        font=(theme["font_family"], theme["body_size"]))
        
        # Configure TButton
        style.configure('TButton',
                        background=theme["button_bg"],
                        foreground=theme["button_fg"],
                        padding=theme["padding"])
        
        style.map('TButton',
                  background=[('active', theme["button_active_bg"])],
                  foreground=[('active', theme["button_active_fg"])])
        
        # Configure Success.TButton
        style.configure('Success.TButton',
                        background=theme["success_color"],
                        foreground=theme["bg_color"])
        
        style.map('Success.TButton',
                  background=[('active', self._adjust_color(theme["success_color"], -20))],
                  foreground=[('active', theme["bg_color"])])
        
        # Configure Warning.TButton
        style.configure('Warning.TButton',
                        background=theme["warning_color"],
                        foreground=theme["bg_color"])
        
        style.map('Warning.TButton',
                  background=[('active', self._adjust_color(theme["warning_color"], -20))],
                  foreground=[('active', theme["bg_color"])])
        
        # Configure Error.TButton
        style.configure('Error.TButton',
                        background=theme["error_color"],
                        foreground=theme["bg_color"])
        
        style.map('Error.TButton',
                  background=[('active', self._adjust_color(theme["error_color"], -20))],
                  foreground=[('active', theme["bg_color"])])
        
        # Configure TLabel
        style.configure('TLabel',
                        background=theme["bg_color"],
                        foreground=theme["fg_color"])
        
        # Configure Header.TLabel
        style.configure('Header.TLabel',
                        font=(theme["font_family"], theme["header_size"], "bold"),
                        foreground=theme["accent_color"])
        
        # Configure Subheader.TLabel
        style.configure('Subheader.TLabel',
                        font=(theme["font_family"], theme["subheader_size"], "bold"),
                        foreground=theme["secondary_accent"])
        
        # Configure TEntry
        style.configure('TEntry',
                        fieldbackground=theme["entry_bg"],
                        foreground=theme["entry_fg"],
                        insertcolor=theme["entry_fg"])
        
        # Configure TNotebook
        style.configure('TNotebook',
                        background=theme["bg_color"],
                        tabmargins=[2, 5, 2, 0])
        
        style.configure('TNotebook.Tab',
                        background=theme["tab_bg"],
                        foreground=theme["tab_fg"],
                        padding=[10, 2])
        
        style.map('TNotebook.Tab',
                  background=[('selected', theme["tab_active_bg"])],
                  foreground=[('selected', theme["tab_active_fg"])])
        
        # Configure TFrame
        style.configure('TFrame',
                        background=theme["bg_color"])
        
        # Configure TScrollbar
        style.configure('TScrollbar',
                        background=theme["scrollbar_bg"],
                        troughcolor=theme["bg_color"],
                        arrowcolor=theme["scrollbar_fg"])
        
        # Configure Treeview (for tables)
        style.configure('Treeview',
                        background=theme["table_row_bg"],
                        foreground=theme["table_row_fg"],
                        fieldbackground=theme["table_row_bg"])
        
        style.configure('Treeview.Heading',
                        background=theme["table_header_bg"],
                        foreground=theme["table_header_fg"],
                        font=(theme["font_family"], theme["body_size"], "bold"))
        
        style.map('Treeview',
                  background=[('selected', theme["accent_color"])],
                  foreground=[('selected', theme["bg_color"])])
        
        # Configure the root window
        self.root.configure(background=theme["bg_color"])
        
        # Update all existing widgets to use the new theme
        self._update_all_widgets(self.root)
    
    def _update_all_widgets(self, parent):
        """
        Recursively update all widgets to use the current theme.
        
        Args:
            parent: The parent widget to update
        """
        for child in parent.winfo_children():
            # Update the child widget
            self._update_widget(child)
            
            # Recursively update children
            if child.winfo_children():
                self._update_all_widgets(child)
    
    def _update_widget(self, widget):
        """
        Update a single widget to use the current theme.
        
        Args:
            widget: The widget to update
        """
        theme = self.themes[self.theme_name]
        
        # Update based on widget type
        if isinstance(widget, tk.Text):
            widget.configure(
                background=theme["entry_bg"],
                foreground=theme["entry_fg"],
                insertbackground=theme["entry_fg"],
                selectbackground=theme["accent_color"],
                selectforeground=theme["bg_color"]
            )
        elif isinstance(widget, tk.Entry):
            widget.configure(
                background=theme["entry_bg"],
                foreground=theme["entry_fg"],
                insertbackground=theme["entry_fg"],
                selectbackground=theme["accent_color"],
                selectforeground=theme["bg_color"]
            )
        elif isinstance(widget, tk.Canvas):
            widget.configure(
                background=theme["chart_bg"]
            )
        elif isinstance(widget, tk.Scrollbar):
            widget.configure(
                background=theme["scrollbar_bg"],
                troughcolor=theme["bg_color"],
                activebackground=theme["scrollbar_fg"]
            )
        elif isinstance(widget, tk.Listbox):
            widget.configure(
                background=theme["entry_bg"],
                foreground=theme["entry_fg"],
                selectbackground=theme["accent_color"],
                selectforeground=theme["bg_color"]
            )
    
    def switch_theme(self, theme_name: str):
        """
        Switch to a different theme.
        
        Args:
            theme_name: The name of the theme to switch to
        """
        self.apply_theme(theme_name)
    
    def get_color(self, color_name: str) -> str:
        """
        Get a color from the current theme.
        
        Args:
            color_name: The name of the color to get
            
        Returns:
            The color value as a hex string
        """
        theme = self.themes[self.theme_name]
        return theme.get(color_name, "#000000")
    
    def get_font(self, size_name: str = "body_size") -> tuple:
        """
        Get a font from the current theme.
        
        Args:
            size_name: The name of the font size to get
            
        Returns:
            A tuple containing the font family and size
        """
        theme = self.themes[self.theme_name]
        return (theme["font_family"], theme[size_name])
    
    def get_matplotlib_style(self) -> Dict[str, Any]:
        """
        Get a dictionary of matplotlib style parameters for the current theme.
        
        Returns:
            A dictionary of matplotlib style parameters
        """
        theme = self.themes[self.theme_name]
        
        return {
            'figure.facecolor': theme["chart_bg"],
            'figure.edgecolor': theme["chart_bg"],
            'axes.facecolor': theme["chart_bg"],
            'axes.edgecolor': theme["chart_grid"],
            'axes.labelcolor': theme["text_primary"],
            'axes.grid': True,
            'grid.color': theme["chart_grid"],
            'grid.linestyle': '--',
            'grid.alpha': 0.7,
            'xtick.color': theme["text_secondary"],
            'ytick.color': theme["text_secondary"],
            'text.color': theme["text_primary"],
            'lines.color': theme["chart_line"],
            'patch.edgecolor': theme["chart_grid"],
            'savefig.facecolor': theme["chart_bg"],
            'savefig.edgecolor': theme["chart_bg"]
        }
    
    def save_theme_preference(self, config_path: str):
        """
        Save the current theme preference to the config file.
        
        Args:
            config_path: Path to the configuration file
        """
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                config["theme"] = self.theme_name
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving theme preference: {e}")
    
    def _adjust_color(self, hex_color: str, amount: int) -> str:
        """
        Adjust a hex color by the specified amount.
        
        Args:
            hex_color: The hex color to adjust
            amount: The amount to adjust by (positive for lighter, negative for darker)
            
        Returns:
            The adjusted hex color
        """
        # Remove the hash
        hex_color = hex_color.lstrip('#')
        
        # Convert to RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        # Adjust
        r = max(0, min(255, r + amount))
        g = max(0, min(255, g + amount))
        b = max(0, min(255, b + amount))
        
        # Convert back to hex
        return f"#{r:02x}{g:02x}{b:02x}"


def create_header_label(parent, text: str) -> ttk.Label:
    """
    Create a header label with the appropriate style.
    
    Args:
        parent: The parent widget
        text: The text for the label
        
    Returns:
        The created label
    """
    return ttk.Label(parent, text=text, style="Header.TLabel")


def create_subheader_label(parent, text: str) -> ttk.Label:
    """
    Create a subheader label with the appropriate style.
    
    Args:
        parent: The parent widget
        text: The text for the label
        
    Returns:
        The created label
    """
    return ttk.Label(parent, text=text, style="Subheader.TLabel")


def create_scrollable_frame(parent) -> tuple:
    """
    Create a scrollable frame.
    
    Args:
        parent: The parent widget
        
    Returns:
        A tuple containing (canvas, scrollable_frame)
    """
    # Create a canvas
    canvas = tk.Canvas(parent)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Add a scrollbar to the canvas
    scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # Configure the canvas
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    
    # Create a frame inside the canvas
    scrollable_frame = ttk.Frame(canvas)
    
    # Add the frame to the canvas
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    
    return canvas, scrollable_frame


def create_scrollable_text(parent, **kwargs) -> tuple:
    """
    Create a scrollable text widget.
    
    Args:
        parent: The parent widget
        **kwargs: Additional arguments for the Text widget
        
    Returns:
        A tuple containing (frame, text_widget)
    """
    frame = ttk.Frame(parent)
    
    # Create a text widget
    text = tk.Text(frame, **kwargs)
    text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Add a scrollbar
    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # Configure the text widget
    text.configure(yscrollcommand=scrollbar.set)
    
    return frame, text
