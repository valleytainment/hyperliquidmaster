#!/usr/bin/env python3
"""
GUI style manager for the HyperliquidMaster application.
Provides consistent styling across the application with theme support.
"""

import tkinter as tk
from tkinter import ttk
import logging
from typing import Dict, Any, Optional

class GUIStyleManager:
    """
    Manages GUI styling for the HyperliquidMaster application.
    Provides methods for applying consistent styling across the application.
    """
    
    def __init__(self, root: tk.Tk, logger: logging.Logger):
        """
        Initialize the GUI style manager.
        
        Args:
            root: The root Tk instance
            logger: Logger instance for logging
        """
        self.root = root
        self.logger = logger
        self.current_theme = "dark"  # Default theme
        
        # Define themes
        self.themes = {
            "dark": {
                "bg": "#121212",
                "fg": "#FFFFFF",
                "button_bg": "#1E1E1E",
                "button_fg": "#FFFFFF",
                "entry_bg": "#2A2A2A",
                "entry_fg": "#FFFFFF",
                "entry_insertbackground": "#00FF00",  # Neon green cursor
                "highlight_bg": "#333333",
                "highlight_fg": "#FFFFFF",
                "accent1": "#00FF00",  # Neon green
                "accent2": "#FF00FF",  # Neon pink
                "accent3": "#00FFFF",  # Neon cyan
                "success": "#00FF00",  # Neon green
                "warning": "#FFFF00",  # Neon yellow
                "error": "#FF0000",    # Bright red
                "chart_bg": "#121212",
                "chart_fg": "#FFFFFF",
                "chart_grid": "#333333",
                "chart_line1": "#00FF00",  # Neon green
                "chart_line2": "#FF00FF",  # Neon pink
                "chart_line3": "#00FFFF",  # Neon cyan
                "scrollbar_bg": "#1E1E1E",
                "scrollbar_fg": "#333333",
                "tab_active": "#1E1E1E",
                "tab_inactive": "#121212",
                "border": "#333333",
                "header_bg": "#1E1E1E",
                "header_fg": "#FFFFFF",
                "treeview_bg": "#1E1E1E",
                "treeview_fg": "#FFFFFF",
                "treeview_selected_bg": "#333333",
                "treeview_selected_fg": "#FFFFFF",
                "menu_bg": "#1E1E1E",
                "menu_fg": "#FFFFFF",
                "tooltip_bg": "#333333",
                "tooltip_fg": "#FFFFFF"
            },
            "light": {
                "bg": "#F0F0F0",
                "fg": "#000000",
                "button_bg": "#E0E0E0",
                "button_fg": "#000000",
                "entry_bg": "#FFFFFF",
                "entry_fg": "#000000",
                "entry_insertbackground": "#000000",  # Black cursor
                "highlight_bg": "#D0D0D0",
                "highlight_fg": "#000000",
                "accent1": "#007BFF",  # Blue
                "accent2": "#6610F2",  # Purple
                "accent3": "#17A2B8",  # Cyan
                "success": "#28A745",  # Green
                "warning": "#FFC107",  # Yellow
                "error": "#DC3545",    # Red
                "chart_bg": "#FFFFFF",
                "chart_fg": "#000000",
                "chart_grid": "#E0E0E0",
                "chart_line1": "#007BFF",  # Blue
                "chart_line2": "#6610F2",  # Purple
                "chart_line3": "#17A2B8",  # Cyan
                "scrollbar_bg": "#F0F0F0",
                "scrollbar_fg": "#C0C0C0",
                "tab_active": "#FFFFFF",
                "tab_inactive": "#E0E0E0",
                "border": "#C0C0C0",
                "header_bg": "#E0E0E0",
                "header_fg": "#000000",
                "treeview_bg": "#FFFFFF",
                "treeview_fg": "#000000",
                "treeview_selected_bg": "#007BFF",
                "treeview_selected_fg": "#FFFFFF",
                "menu_bg": "#F0F0F0",
                "menu_fg": "#000000",
                "tooltip_bg": "#FFFFCC",
                "tooltip_fg": "#000000"
            }
        }
        
        # Initialize styles
        self.init_styles()
    
    def init_styles(self) -> None:
        """Initialize ttk styles with the current theme."""
        try:
            style = ttk.Style(self.root)
            theme = self.themes[self.current_theme]
            
            # Configure ttk styles
            style.configure("TFrame", background=theme["bg"])
            style.configure("TLabel", background=theme["bg"], foreground=theme["fg"])
            style.configure("TButton", background=theme["button_bg"], foreground=theme["button_fg"])
            style.configure("TEntry", fieldbackground=theme["entry_bg"], foreground=theme["entry_fg"], insertbackground=theme["entry_insertbackground"])
            style.configure("TCheckbutton", background=theme["bg"], foreground=theme["fg"])
            style.configure("TRadiobutton", background=theme["bg"], foreground=theme["fg"])
            style.configure("TNotebook", background=theme["bg"], tabmargins=[2, 5, 2, 0])
            style.configure("TNotebook.Tab", background=theme["tab_inactive"], foreground=theme["fg"], padding=[10, 2])
            style.map("TNotebook.Tab", background=[("selected", theme["tab_active"])], foreground=[("selected", theme["fg"])])
            
            # Configure scrollbar style
            style.configure("TScrollbar", background=theme["scrollbar_bg"], troughcolor=theme["scrollbar_fg"], borderwidth=0, arrowsize=16)
            
            # Configure treeview style
            style.configure("Treeview", 
                background=theme["treeview_bg"], 
                foreground=theme["treeview_fg"], 
                fieldbackground=theme["treeview_bg"])
            style.map("Treeview", 
                background=[("selected", theme["treeview_selected_bg"])], 
                foreground=[("selected", theme["treeview_selected_fg"])])
            
            # Configure success button style
            style.configure("Success.TButton", background=theme["success"], foreground="#FFFFFF")
            
            # Configure warning button style
            style.configure("Warning.TButton", background=theme["warning"], foreground="#000000")
            
            # Configure error button style
            style.configure("Error.TButton", background=theme["error"], foreground="#FFFFFF")
            
            # Configure header style
            style.configure("Header.TLabel", background=theme["header_bg"], foreground=theme["header_fg"], font=("Helvetica", 12, "bold"))
            
            # Configure title style
            style.configure("Title.TLabel", background=theme["bg"], foreground=theme["fg"], font=("Helvetica", 14, "bold"))
            
            # Apply theme to root window
            self.root.configure(background=theme["bg"])
            
            self.logger.info(f"Applied {self.current_theme} theme")
        except Exception as e:
            self.logger.error(f"Error initializing styles: {e}")
    
    def toggle_theme(self) -> None:
        """Toggle between dark and light themes."""
        try:
            self.current_theme = "light" if self.current_theme == "dark" else "dark"
            self.init_styles()
            self.apply_theme_to_all_widgets(self.root)
        except Exception as e:
            self.logger.error(f"Error toggling theme: {e}")
    
    def apply_theme_to_all_widgets(self, parent: tk.Widget) -> None:
        """
        Recursively apply the current theme to all widgets.
        
        Args:
            parent: The parent widget to start from
        """
        theme = self.themes[self.current_theme]
        
        for child in parent.winfo_children():
            try:
                # Apply theme based on widget type
                if isinstance(child, tk.Frame) or isinstance(child, ttk.Frame):
                    child.configure(background=theme["bg"])
                
                elif isinstance(child, tk.Label) or isinstance(child, ttk.Label):
                    child.configure(background=theme["bg"], foreground=theme["fg"])
                
                elif isinstance(child, tk.Button) or isinstance(child, ttk.Button):
                    if isinstance(child, tk.Button):
                        child.configure(background=theme["button_bg"], foreground=theme["button_fg"], 
                                       activebackground=theme["highlight_bg"], activeforeground=theme["highlight_fg"])
                
                elif isinstance(child, tk.Entry) or isinstance(child, ttk.Entry):
                    if isinstance(child, tk.Entry):
                        child.configure(background=theme["entry_bg"], foreground=theme["entry_fg"], 
                                       insertbackground=theme["entry_insertbackground"])
                
                elif isinstance(child, tk.Text):
                    child.configure(background=theme["entry_bg"], foreground=theme["entry_fg"], 
                                   insertbackground=theme["entry_insertbackground"])
                
                elif isinstance(child, tk.Canvas):
                    child.configure(background=theme["chart_bg"])
                
                elif isinstance(child, tk.Scrollbar):
                    child.configure(background=theme["scrollbar_bg"], troughcolor=theme["scrollbar_fg"])
                
                elif isinstance(child, ttk.Notebook):
                    pass  # Already styled through ttk.Style
                
                # Recursively apply to children
                self.apply_theme_to_all_widgets(child)
            except Exception as e:
                self.logger.error(f"Error applying theme to widget {child}: {e}")
    
    def get_theme_colors(self) -> Dict[str, str]:
        """
        Get the current theme colors.
        
        Returns:
            Dict containing the current theme colors
        """
        return self.themes[self.current_theme]
    
    def create_scrollable_frame(self, parent: tk.Widget) -> tuple:
        """
        Create a scrollable frame.
        
        Args:
            parent: The parent widget
            
        Returns:
            Tuple containing (container_frame, scrollable_frame)
        """
        theme = self.themes[self.current_theme]
        
        # Create a container frame
        container = ttk.Frame(parent)
        
        # Create a canvas
        canvas = tk.Canvas(container, bg=theme["bg"], highlightthickness=0)
        
        # Create a scrollbar
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        
        # Create a frame inside the canvas
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # Add the frame to the canvas
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack the widgets
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        return container, scrollable_frame
    
    def create_scrollable_text(self, parent: tk.Widget, **kwargs) -> tk.Text:
        """
        Create a scrollable text widget.
        
        Args:
            parent: The parent widget
            **kwargs: Additional arguments for the Text widget
            
        Returns:
            The Text widget
        """
        theme = self.themes[self.current_theme]
        
        # Create a frame
        frame = ttk.Frame(parent)
        
        # Create a text widget
        text = tk.Text(frame, bg=theme["entry_bg"], fg=theme["entry_fg"], 
                      insertbackground=theme["entry_insertbackground"], **kwargs)
        
        # Create a scrollbar
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        
        # Pack the widgets
        text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        return frame, text
    
    def style_button(self, button: tk.Button, button_type: str = "normal") -> None:
        """
        Apply styling to a button.
        
        Args:
            button: The button to style
            button_type: The type of button (normal, success, warning, error)
        """
        theme = self.themes[self.current_theme]
        
        if button_type == "success":
            button.configure(background=theme["success"], foreground="#FFFFFF", 
                           activebackground=theme["success"], activeforeground="#FFFFFF")
        elif button_type == "warning":
            button.configure(background=theme["warning"], foreground="#000000", 
                           activebackground=theme["warning"], activeforeground="#000000")
        elif button_type == "error":
            button.configure(background=theme["error"], foreground="#FFFFFF", 
                           activebackground=theme["error"], activeforeground="#FFFFFF")
        else:
            button.configure(background=theme["button_bg"], foreground=theme["button_fg"], 
                           activebackground=theme["highlight_bg"], activeforeground=theme["highlight_fg"])
    
    def style_entry(self, entry: tk.Entry) -> None:
        """
        Apply styling to an entry widget.
        
        Args:
            entry: The entry widget to style
        """
        theme = self.themes[self.current_theme]
        
        entry.configure(background=theme["entry_bg"], foreground=theme["entry_fg"], 
                       insertbackground=theme["entry_insertbackground"],
                       highlightbackground=theme["border"], highlightcolor=theme["accent1"])
    
    def style_text(self, text: tk.Text) -> None:
        """
        Apply styling to a text widget.
        
        Args:
            text: The text widget to style
        """
        theme = self.themes[self.current_theme]
        
        text.configure(background=theme["entry_bg"], foreground=theme["entry_fg"], 
                      insertbackground=theme["entry_insertbackground"],
                      highlightbackground=theme["border"], highlightcolor=theme["accent1"])
    
    def get_chart_colors(self) -> Dict[str, str]:
        """
        Get chart colors for the current theme.
        
        Returns:
            Dict containing chart colors
        """
        theme = self.themes[self.current_theme]
        
        return {
            "bg": theme["chart_bg"],
            "fg": theme["chart_fg"],
            "grid": theme["chart_grid"],
            "line1": theme["chart_line1"],
            "line2": theme["chart_line2"],
            "line3": theme["chart_line3"]
        }
