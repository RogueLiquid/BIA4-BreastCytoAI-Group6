"""
Breast Cancer Cell Biopsy Classifier - GUI
Medical Interface with ML Backend Integration

Integration points:
1. Feature Extraction (extract_features method)
2. Prediction (predict_classification method)

Group 6

"""

import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import os
import numpy as np
import cv2
import glob
import time
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ==================== APPLICATION CONFIGURATION ====================

class Config:
    """Application-wide configuration settings"""
    
    # Application metadata
    APP_TITLE = "BreastCytoAI - Group 6"
    VERSION = "2.0"
    DATASET_NAME = "BreakHis Dataset"
    
    # Window dimensions
    WINDOW_WIDTH = 1500
    WINDOW_HEIGHT = 950
    MIN_WIDTH = 1200
    MIN_HEIGHT = 800
    
    # Font definitions (增大至专业易读尺寸)
    FONT_FAMILY = "Segoe UI"
    FONT_TITLE = (FONT_FAMILY, 32, "bold")
    FONT_SUBTITLE = (FONT_FAMILY, 14)          # 增大: 12 -> 14
    FONT_HEADER = (FONT_FAMILY, 20, "bold")    # 增大: 18 -> 20
    FONT_SECTION = (FONT_FAMILY, 18, "bold")   # 增大: 16 -> 18
    FONT_BODY = (FONT_FAMILY, 12)              # 增大: 11 -> 12
    FONT_SMALL = (FONT_FAMILY, 11)             # 增大: 10 -> 11
    FONT_BUTTON = (FONT_FAMILY, 13, "bold")    # 增大: 12 -> 13


class Theme:
    """
    Professional Medical Color Theme
    
    Based on modern healthcare application design principles.
    All colors use HEX format (#RRGGBB) for consistency.
    
    Color Psychology:
    - Blue: Trust, professionalism, calm (medical standard)
    - Green: Health, success, positive outcomes
    - Red: Attention, danger, critical results
    - Gray: Neutral, professional, clean
    """
    
    # Primary brand colors
    PRIMARY = "#1173D5"          # Medical blue (main)
    PRIMARY_DARK = "#004C99"     # Darker blue (hover)
    PRIMARY_LIGHT = "#E6F2FF"    # Light blue (background)
    
    # Secondary colors
    SECONDARY = "#00B4D8"        # Teal (secondary actions)
    SECONDARY_DARK = "#0096B8"   # Dark teal (hover)
    
    # Accent colors
    ACCENT = "#FF6B6B"           # Coral (important actions)
    ACCENT_DARK = "#E64545"      # Dark coral (hover)
    
    # Status colors
    SUCCESS = "#06D6A0"          # Green (benign/success)
    SUCCESS_BG = "#E8FFF8"       # Light green (background)
    WARNING = "#FFB627"          # Amber (warnings)
    DANGER = "#FF6B6B"           # Red (malignant/error)
    DANGER_BG = "#FFE8E8"        # Light red (background)
    
    # Neutral colors
    BACKGROUND = "#F8F9FA"       # Page background
    SURFACE = "#FFFFFF"          # Card surfaces
    BORDER = "#E1E8ED"           # Borders and dividers
    
    # Text colors
    TEXT_PRIMARY = "#1A1A2E"     # Main text
    TEXT_SECONDARY = "#6C757D"   # Secondary text
    TEXT_LIGHT = "#ADB5BD"       # Placeholder text
    TEXT_ON_COLOR = "#FFFFFF"    # Text on colored backgrounds
    
    # Effects
    SHADOW = "gray80"
    HOVER = "#F1F3F5"


# ==================== ML BACKEND INTEGRATION ====================

ML_BACKEND_AVAILABLE = False

try:
    from ml_backend import (
        extract_radiomics_features,
        extract_pixel_features,
        extract_sift_bof_features,
        predict_with_model,
        generate_gradcam,
        generate_occlusion,
        process_image,
        apply_enhancement,
        apply_blur,
        apply_sharpen
    )
    ML_BACKEND_AVAILABLE = True
    print("="*70)
    print("ML Backend Integration Successful")
    print("="*70)
    print("Status: REAL MODE - Using ML backend")
    print("Available functions:")
    print("  • extract_radiomics_features()")
    print("  • extract_pixel_features()")
    print("  • predict_with_model()")
    print("  • generate_gradcam()")
    print("  • generate_occlusion()")
    print("  • process_image()")
    print("="*70)
except ImportError as e:
    ML_BACKEND_AVAILABLE = False
    print("="*70)
    print("WARNING: ML Backend Not Available - Running in DEMO MODE")
    print("="*70)
    print(f"Error: {str(e)}")
    print("\nTo enable real ML predictions:")
    print("  1. Create 'ml_backend.py' in the same folder")
    print("  2. Implement the 3 required functions")
    print("  3. Restart this application")
    print("="*70)


# ==================== CUSTOM MESSAGE DIALOG ====================

def show_message(parent, message_type, title, message, **kwargs):
    """
    Custom message dialog to replace tkinter.messagebox

    Args:
        parent: Parent window
        message_type: "error", "info", "warning", "question"
        title: Dialog title
        message: Message text
        **kwargs: Additional options

    Returns:
        For question type: True/False, for others: None
    """
    # Create dialog window
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)  # Bind to parent window
    dialog.grab_set()         # Block input to parent

    # Configure dialog
    dialog.configure(bg=Theme.SURFACE)
    dialog.resizable(False, False)

    # Icon based on message type
    icon_text = ""
    if message_type == "error":
        icon_text = "❌"
        button_bg = Theme.DANGER
        button_fg = Theme.TEXT_ON_COLOR
    elif message_type == "warning":
        icon_text = "⚠️"
        button_bg = Theme.WARNING
        button_fg = Theme.TEXT_PRIMARY
    elif message_type == "info":
        icon_text = "ℹ️"
        button_bg = Theme.SECONDARY
        button_fg = Theme.TEXT_PRIMARY
    elif message_type == "question":
        icon_text = "❓"
        button_bg = Theme.PRIMARY
        button_fg = Theme.TEXT_ON_COLOR
    else:
        icon_text = "💬"
        button_bg = Theme.SECONDARY
        button_fg = Theme.TEXT_PRIMARY

    # Content frame
    content = tk.Frame(dialog, bg=Theme.SURFACE, padx=20, pady=20)
    content.pack(fill=tk.BOTH, expand=True)

    # Icon and message
    msg_frame = tk.Frame(content, bg=Theme.SURFACE)
    msg_frame.pack(fill=tk.X, pady=(0, 20))

    icon_label = tk.Label(
        msg_frame,
        text=icon_text,
        font=(Config.FONT_FAMILY, 24),
        bg=Theme.SURFACE
    )
    icon_label.pack(side=tk.LEFT, padx=(0, 15))

    message_label = tk.Label(
        msg_frame,
        text=message,
        font=Config.FONT_BODY,
        bg=Theme.SURFACE,
        fg=Theme.TEXT_PRIMARY,
        wraplength=400,
        justify=tk.LEFT,
        anchor=tk.W
    )
    message_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    # Buttons frame
    btn_frame = tk.Frame(content, bg=Theme.SURFACE)
    btn_frame.pack(fill=tk.X)

    result = None

    def set_result(value):
        nonlocal result
        result = value
        dialog.destroy()

    if message_type == "question":
        # Yes/No buttons
        ModernButton(
            btn_frame,
            text="Yes",
            command=lambda: set_result(True),
            bg=button_bg,
            fg=button_fg,
            font=Config.FONT_BUTTON,
            padx=20,
            pady=8
        ).pack(side=tk.LEFT, padx=(0, 10))

        ModernButton(
            btn_frame,
            text="No",
            command=lambda: set_result(False),
            bg=Theme.SURFACE,
            fg=Theme.TEXT_PRIMARY,
            font=Config.FONT_BUTTON,
            padx=20,
            pady=8
        ).pack(side=tk.LEFT)
    else:
        # OK button
        ModernButton(
            btn_frame,
            text="OK",
            command=lambda: set_result(None),
            bg=button_bg,
            fg=button_fg,
            font=Config.FONT_BUTTON,
            padx=30,
            pady=8
        ).pack(side=tk.LEFT)

    # Center dialog
    dialog.update_idletasks()
    x = parent.winfo_x() + (parent.winfo_width() - dialog.winfo_width()) // 2
    y = parent.winfo_y() + (parent.winfo_height() - dialog.winfo_height()) // 2
    dialog.geometry(f"+{x}+{y}")

    # Wait for dialog to close
    parent.wait_window(dialog)

    return result

# ==================== CUSTOM UI COMPONENTS ====================

class ModernButton(tk.Button):
    """Custom button without hover effects to prevent focus loss color changes"""

    def __init__(self, parent, **kwargs):
        # Remove hover_color parameter since we don't use hover effects
        kwargs.pop('hover_color', None)
        default_bg = kwargs.get('bg', Theme.PRIMARY)

        kwargs.setdefault('relief', tk.FLAT)
        kwargs.setdefault('borderwidth', 0)
        kwargs.setdefault('cursor', 'hand2')
        kwargs.setdefault('font', Config.FONT_BODY)
        kwargs.setdefault('padx', 20)
        kwargs.setdefault('pady', 10)

        super().__init__(parent, **kwargs)

        self._normal_bg = default_bg
        self._current_bg = default_bg

    @property
    def default_bg(self):
        """Get the default background color"""
        return self._normal_bg

    @default_bg.setter
    def default_bg(self, color):
        """Set the default background color"""
        self._normal_bg = color
        self._current_bg = color
        self.config(bg=color)

    def set_pressed_state(self, pressed_bg, pressed_fg=None):
        """Set button to pressed state"""
        self._current_bg = pressed_bg
        self.config(
            bg=pressed_bg,
            relief=tk.SUNKEN,  # Make button appear pressed
            borderwidth=2      # Slightly thicker border for pressed effect
        )
        if pressed_fg:
            self.config(fg=pressed_fg)

    def set_normal_state(self, normal_bg=None, normal_fg=None):
        """Set button to normal state"""
        if normal_bg is not None:
            self._normal_bg = normal_bg
        self._current_bg = self._normal_bg
        self.config(
            bg=self._normal_bg,
            relief=tk.FLAT,     # Restore flat relief
            borderwidth=0      # Restore thin border
        )
        if normal_fg:
            self.config(fg=normal_fg)


class Card(tk.Frame):
    """Modern card container"""
    
    def __init__(self, parent, **kwargs):
        kwargs.setdefault('bg', Theme.SURFACE)
        kwargs.setdefault('relief', tk.FLAT)
        kwargs.setdefault('borderwidth', 1)
        super().__init__(parent, **kwargs)


# ==================== MAIN APPLICATION ====================

class BreastCancerClassifierGUI:
    """
    Main Application Class
    
    Professional GUI for breast cancer cell classification using ML.
    Integrates with teammate's ML backend for feature extraction and prediction.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title(Config.APP_TITLE)
        self.root.geometry(f"{Config.WINDOW_WIDTH}x{Config.WINDOW_HEIGHT}")
        self.root.configure(bg=Theme.BACKGROUND)
        self.root.minsize(Config.MIN_WIDTH, Config.MIN_HEIGHT)
        
        # Initialize state
        self.current_image_path = None
        self.original_image = None
        self.current_image = None
        self.cropped_image = None  # Store cropped image separately
        self.loaded_images = []

        self.feature_extraction_method = tk.StringVar(value="radiomics")
        self.model_type = tk.StringVar(value="radiomics")  # radiomics, pixel, fusion
        self.model_path = tk.StringVar(value="")
        self.model_path_map = {}  # Map display names to full paths
        self.current_model_path = ""  # Store the currently selected full path

        self.extracted_features = None
        self.feature_names = None

        self.crop_mode = False
        self.crop_start = None
        self.crop_rect = None

        # GradCAM
        self.gradcam_image = None

        # Processing parameters
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.brightness_var = tk.IntVar(value=0)
        self.kernel_var = tk.IntVar(value=1)
        self.sigma_var = tk.DoubleVar(value=0.0)
        self.strength_var = tk.DoubleVar(value=0.0)
        
        # --- 新增：配置 TTK 样式以美化滚动条 ---
        style = ttk.Style()
        style.theme_use('clam') # 使用 'clam' 主题作为基础，更容易定制
        
        # 配置 Scrollbar 样式
        style.configure("TScrollbar", 
            gripcolor=Theme.TEXT_SECONDARY,    # 滑块抓手颜色 (灰色)
            troughcolor=Theme.BACKGROUND,      # 凹槽背景色 (浅页面背景)
            background=Theme.BORDER,           # 滚动条背景色 (边框色)
            bordercolor=Theme.BORDER,
            arrowcolor=Theme.TEXT_SECONDARY
        )
        style.map("TScrollbar",
            background=[('active', Theme.TEXT_SECONDARY)] # 鼠标悬停时
        )
        # ------------------------------------
        
        # Build UI
        self.create_interface()
    
    def create_interface(self):
        """Build the complete interface"""
        self.create_header()
        self.create_main_content()
        self.create_footer()
    
    # ==================== HEADER ====================
    
    def create_header(self):
        """Create application header"""
        header_shadow = tk.Frame(self.root, bg=Theme.SHADOW, height=2)
        header_shadow.pack(fill=tk.X, side=tk.TOP)
        
        header = tk.Frame(self.root, bg=Theme.PRIMARY, height=120)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        
        content = tk.Frame(header, bg=Theme.PRIMARY)
        content.pack(expand=True, fill=tk.BOTH, padx=40)
        
        # Left: Title
        left = tk.Frame(content, bg=Theme.PRIMARY)
        left.pack(side=tk.LEFT, fill=tk.Y, pady=20)
        
        title_frame = tk.Frame(left, bg=Theme.PRIMARY)
        title_frame.pack(side=tk.LEFT)
        
        tk.Label(
            title_frame,
            text="BreastCytoAI - Group 6",
            font=Config.FONT_TITLE,
            bg=Theme.PRIMARY,
            fg=Theme.TEXT_ON_COLOR
        ).pack(anchor=tk.W)
        
        tk.Label(
            title_frame,
            text=f"{Config.DATASET_NAME} Analysis | Powered by Machine Learning",
            font=Config.FONT_SUBTITLE,
            bg=Theme.PRIMARY,
            fg=Theme.PRIMARY_LIGHT
        ).pack(anchor=tk.W, pady=(5, 0))
        
        # Right: Buttons
        right = tk.Frame(content, bg=Theme.PRIMARY)
        right.pack(side=tk.RIGHT, fill=tk.Y, pady=20)
        
        ModernButton(
            right,
            text="Open Image",
            command=self.open_image,
            bg=Theme.SURFACE,
            fg=Theme.PRIMARY,
            font=Config.FONT_BODY,
            width=10,
            pady=5
        ).pack(side=tk.TOP, pady=(0, 12))

        ModernButton(
            right,
            text="Help",
            command=self.show_help,
            bg=Theme.SURFACE,
            fg=Theme.PRIMARY,
            font=Config.FONT_BODY,
            width=10,
            pady=5
        ).pack(side=tk.TOP)
    
    # ==================== MAIN CONTENT ====================
    
    def create_main_content(self):
        """Create main content area"""
        main = tk.Frame(self.root, bg=Theme.BACKGROUND)
        main.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)
        
        # Left: Image viewer
        left = tk.Frame(main, bg=Theme.BACKGROUND)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        # 替换 Right: Controls 为带滚动条的区域
        right_container = tk.Frame(main, bg=Theme.BACKGROUND)
        right_container.pack(side=tk.RIGHT, fill=tk.Y, padx=(15, 0))
        right_container.configure(width=500)
        right_container.pack_propagate(False)
        
        # 创建 Canvas 和 Scrollbar
        self.control_canvas = tk.Canvas(right_container, bg=Theme.BACKGROUND, highlightthickness=0)
        self.control_scrollbar = ttk.Scrollbar(right_container, orient="vertical", command=self.control_canvas.yview)
        
        # 将 Scrollbar 链接到 Canvas
        self.control_canvas.configure(yscrollcommand=self.control_scrollbar.set)
        
        self.control_scrollbar.pack(side="right", fill="y")
        self.control_canvas.pack(side="left", fill="both", expand=True)
        
        # 创建一个 Frame 放置所有控制卡片 (这个 Frame 才是真正滚动的)
        right = tk.Frame(self.control_canvas, bg=Theme.BACKGROUND)
        self.control_canvas.create_window((0, 0), window=right, anchor="nw", width=500)
        
        # 绑定事件，当内容 Frame 尺寸改变时，更新 Canvas 的滚动区域
        right.bind("<Configure>", lambda e: self.control_canvas.configure(
            scrollregion=self.control_canvas.bbox("all")
        ))
        
        # === 修复鼠标滚轮绑定 ===
        # 捕获 Windows/Linux 的滚轮事件
        self.control_canvas.bind('<MouseWheel>', self._on_mousewheel)
        
        # 捕获 Linux/X11 系统的鼠标按钮 4 和 5 (通常是滚轮上下)
        self.control_canvas.bind('<Button-4>', self._on_mousewheel)
        self.control_canvas.bind('<Button-5>', self._on_mousewheel)
        # =======================
        
        self.create_image_viewer(left)
        self.create_control_panel(right)

    def _on_mousewheel(self, event):
        """处理鼠标滚轮事件，实现 Canvas 滚动"""
        # Windows 和 Linux/macOS 的事件 delta 机制不同
        
        # Windows/Linux (通常是 <MouseWheel>)
        if event.delta: 
            # 滚动幅度通常是 120 的倍数
            self.control_canvas.yview_scroll(int(-1 * (event.delta/120)), "units")
        
        # macOS/Linux (有时是 <Button-4> 和 <Button-5>)
        elif event.num == 4:
            # 向上滚动
            self.control_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            # 向下滚动
            self.control_canvas.yview_scroll(1, "units")

    def create_image_viewer(self, parent):
        """Create image viewer section"""
        shadow = tk.Frame(parent, bg=Theme.SHADOW, height=4)
        shadow.pack(fill=tk.BOTH, expand=True)
        
        card = Card(parent)
        card.place(in_=shadow, relx=0, rely=0, relwidth=1, relheight=1)
        
        # Header
        header = tk.Frame(card, bg=Theme.SURFACE, height=60)
        header.pack(fill=tk.X, padx=20, pady=(20, 0))
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="Image Viewer",
            font=Config.FONT_HEADER,
            bg=Theme.SURFACE,
            fg=Theme.TEXT_PRIMARY
        ).pack(side=tk.LEFT)
        
        # Toolbar
        toolbar = tk.Frame(header, bg=Theme.SURFACE)
        toolbar.pack(side=tk.RIGHT)

        self.crop_button = ModernButton(
            toolbar,
            text="Crop",
            command=self.crop_image,
            bg=Theme.PRIMARY_LIGHT,
            fg=Theme.PRIMARY,
            font=Config.FONT_SMALL,
            padx=15,
            pady=5
        )
        # Store normal and pressed colors for crop button
        self.crop_button_normal_bg = Theme.PRIMARY_LIGHT
        self.crop_button_normal_fg = Theme.PRIMARY
        self.crop_button_pressed_bg = Theme.PRIMARY_LIGHT  # Keep same blue background
        self.crop_button_pressed_fg = Theme.PRIMARY
        # Add pressed state tracking
        self.crop_button._pressed_bg = self.crop_button_pressed_bg
        self.crop_button._pressed_fg = self.crop_button_pressed_fg
        self.crop_button._is_pressed = False
        self.crop_button.pack(side=tk.LEFT, padx=5)

        ModernButton(
            toolbar,
            text="Reset",
            command=self.reset_to_original,
            bg=Theme.WARNING,
            fg=Theme.TEXT_PRIMARY,
            font=Config.FONT_SMALL,
            padx=15,
            pady=5
        ).pack(side=tk.LEFT, padx=5)

        ModernButton(
            toolbar,
            text="GradCAM",
            command=self.show_gradcam,
            bg=Theme.ACCENT,
            fg=Theme.DANGER, # <-- 修改为红色字
            font=Config.FONT_SMALL,
            padx=15,
            pady=5
        ).pack(side=tk.LEFT, padx=5)

        ModernButton(
            toolbar,
            text="Occlusion",
            command=self.show_occlusion,
            bg=Theme.WARNING,
            fg=Theme.SUCCESS,
            font=Config.FONT_SMALL,
            padx=15,
            pady=5
        ).pack(side=tk.LEFT, padx=5)
        
        # Canvas
        canvas_frame = tk.Frame(card, bg=Theme.BACKGROUND)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.canvas = tk.Canvas(
            canvas_frame,
            bg=Theme.BACKGROUND,
            highlightthickness=2,
            highlightbackground=Theme.BORDER,
            relief=tk.FLAT
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        self.placeholder = self.canvas.create_text(
            400, 300,
            text="No image loaded\n\nClick 'Open Image' to begin",
            font=Config.FONT_SECTION,
            fill=Theme.TEXT_LIGHT,
            justify=tk.CENTER
        )
        
        # Info bar
        info = tk.Frame(card, bg=Theme.PRIMARY_LIGHT, height=40)
        info.pack(fill=tk.X, padx=20, pady=(0, 20))
        info.pack_propagate(False)
        
        self.image_info = tk.Label(
            info,
            text="Image: None | Size: 0x0 | Format: N/A",
            font=Config.FONT_BODY,
            bg=Theme.PRIMARY_LIGHT,
            fg=Theme.TEXT_SECONDARY,
            anchor=tk.W
        )
        self.image_info.pack(side=tk.LEFT, padx=15, fill=tk.X, expand=True)
    
    def create_control_panel(self, parent):
        """Create control panel"""
        self.create_model_selection_card(parent)
        self.create_image_processing_card(parent)
        self.create_model_card(parent)
        self.create_results_card(parent)
    
    def create_model_selection_card(self, parent):
        """Model selection card with dropdown"""
        card = Card(parent)
        card.pack(fill=tk.X, pady=(0, 12))

        tk.Label(
            card,
            text="Model Selection",
            font=Config.FONT_SECTION,
            bg=Theme.SURFACE,
            fg=Theme.TEXT_PRIMARY,
            anchor=tk.W
        ).pack(fill=tk.X, padx=20, pady=(15, 10))

        # Model dropdown
        model_frame = tk.Frame(card, bg=Theme.SURFACE)
        model_frame.pack(fill=tk.X, padx=20, pady=(0, 15))

        tk.Label(
            model_frame,
            text="Select Model:",
            font=(Config.FONT_FAMILY, 11, "bold"),
            bg=Theme.SURFACE,
            fg=Theme.TEXT_PRIMARY,
            anchor=tk.W
        ).pack(anchor=tk.W, pady=(0, 5))

        # Create dropdown (Combobox)
        self.model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_path,
            font=Config.FONT_SMALL,
            state="readonly"  # Make it read-only dropdown
        )
        self.model_combo.pack(fill=tk.X, pady=(0, 10))

        # Auto-detect models and populate dropdown
        self.populate_model_dropdown()

        # Bind selection event
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_selected)

        # Refresh button
        ModernButton(
            model_frame,
            text="Refresh Models",
            command=self.populate_model_dropdown,
            bg=Theme.SECONDARY,
            fg=Theme.TEXT_PRIMARY,
            font=Config.FONT_SMALL,
            padx=10,
            pady=5
        ).pack(pady=(5, 0))
    
    def populate_model_dropdown(self):
        """Populate dropdown with available .pth and .pkl model files"""
        # Find all .pth and .pkl files recursively
        all_models = glob.glob("**/*.pth", recursive=True) + glob.glob("**/*.pkl", recursive=True)

        # Filter and sort models
        model_options = []
        for model_path in all_models:
            # Get display name (folder name or file name)
            dirname = os.path.dirname(model_path)
            folder_name = os.path.basename(dirname) if dirname else "root"
            display_name = f"{folder_name} ({os.path.basename(model_path)})"
            model_options.append((display_name, model_path))

        # Sort by display name
        model_options.sort(key=lambda x: x[0])

        # Extract display names and full paths
        display_names = [name for name, path in model_options]
        full_paths = [path for name, path in model_options]

        # Add "All models" option at the beginning
        display_names.insert(0, "All models")
        full_paths.insert(0, "ALL_MODELS")  # Special marker for all models

        # Create mapping dictionary
        self.model_path_map = dict(zip(display_names, full_paths))

        # Update combobox
        self.model_combo['values'] = display_names

        # Set default selection if available
        if display_names:
            if not self.model_path.get() or self.model_path.get() not in display_names:
                self.model_combo.current(0)
                selected_display = display_names[0]
                self.model_path.set(selected_display)
                self.current_model_path = self.model_path_map[selected_display]
            else:
                # If already selected, update current_model_path
                selected_display = self.model_path.get()
                self.current_model_path = self.model_path_map.get(selected_display, "")

        print(f"Found {len(display_names)-1} model files + All models option")
        return len(display_names)

    def on_model_selected(self, event):
        """Handle model selection change"""
        selected_display = self.model_path.get()
        self.current_model_path = self.model_path_map.get(selected_display, "")
        print(f"Selected model: {selected_display} -> {self.current_model_path}")

        # Special handling for "All models" selection
        if self.current_model_path == "ALL_MODELS":
            print("All models mode selected")
        else:
            print(f"Single model mode: {self.current_model_path}")

    def create_feature_card(self, parent):
        """Feature extraction card"""
        card = Card(parent)
        card.pack(fill=tk.X, pady=(0, 12))  # Reduced from 15 to 12
        
        tk.Label(
            card,
            text="Feature Extraction",
            font=Config.FONT_SECTION,
            bg=Theme.SURFACE,
            fg=Theme.TEXT_PRIMARY,
            anchor=tk.W
        ).pack(fill=tk.X, padx=20, pady=(15, 10))  # Reduced padding
        
        options = tk.Frame(card, bg=Theme.SURFACE)
        options.pack(fill=tk.X, padx=20, pady=(0, 10))  # Reduced padding
        
        # Radiomics
        rad_frame = tk.Frame(options, bg=Theme.SURFACE)
        rad_frame.pack(fill=tk.X, pady=5)  # Reduced from 8 to 5
        
        tk.Radiobutton(
            rad_frame,
            text="Radiomics Features",
            variable=self.feature_extraction_method,
            value="radiomics",
            font=(Config.FONT_FAMILY, 12, "bold"),  # Increased from 11 to 12
            bg=Theme.SURFACE,
            fg=Theme.TEXT_PRIMARY,
            selectcolor=Theme.PRIMARY,
            activebackground=Theme.SURFACE,
            relief=tk.FLAT
        ).pack(anchor=tk.W)
        
        tk.Label(
            rad_frame,
            text="Texture, shape, and intensity features\nGLCM, GLRLM, GLSZM statistics",
            font=Config.FONT_SMALL,
            bg=Theme.SURFACE,
            fg=Theme.TEXT_SECONDARY,
            justify=tk.LEFT
        ).pack(anchor=tk.W, padx=25, pady=(2, 0))  # Reduced from 3 to 2
        
        # Pixel
        pix_frame = tk.Frame(options, bg=Theme.SURFACE)
        pix_frame.pack(fill=tk.X, pady=5)  # Reduced from 8 to 5
        
        tk.Radiobutton(
            pix_frame,
            text="Raw Pixel Features",
            variable=self.feature_extraction_method,
            value="pixel",
            font=(Config.FONT_FAMILY, 12, "bold"),  # Increased from 11 to 12
            bg=Theme.SURFACE,
            fg=Theme.TEXT_PRIMARY,
            selectcolor=Theme.PRIMARY,
            activebackground=Theme.SURFACE,
            relief=tk.FLAT
        ).pack(anchor=tk.W)
        
        tk.Label(
            pix_frame,
            text="Raw pixel intensity values\nRGB/Grayscale extraction",
            font=Config.FONT_SMALL,
            bg=Theme.SURFACE,
            fg=Theme.TEXT_SECONDARY,
            justify=tk.LEFT
        ).pack(anchor=tk.W, padx=25, pady=(2, 0))  # Reduced from 3 to 2
        
        ModernButton(
            card,
            text="Extract Features",
            command=self.extract_features,
            bg=Theme.SECONDARY,
            fg=Theme.TEXT_PRIMARY,  # Changed from TEXT_ON_COLOR to TEXT_PRIMARY for visibility
            font=Config.FONT_BUTTON,
            width=12,
            padx=30,
            pady=10  # Reduced from 12 to 10
        ).pack(pady=(0, 15))  # Reduced from 20 to 15
    
    def create_image_processing_card(self, parent):
        """Image processing card with internal scrollbar and larger height"""
        card = Card(parent)
        card.pack(fill=tk.X, pady=(0, 12))

        # Header with fold/unfold button
        header_frame = tk.Frame(card, bg=Theme.SURFACE)
        header_frame.pack(fill=tk.X, padx=20, pady=(15, 10))

        tk.Label(
            header_frame,
            text="Image Processing",
            font=Config.FONT_SECTION,
            bg=Theme.SURFACE,
            fg=Theme.TEXT_PRIMARY,
            anchor=tk.W
        ).pack(side=tk.LEFT)

        # Fold/Unfold button
        self.processing_expanded = False  # Start collapsed
        self.unfold_button = ModernButton(
            header_frame,
            text="▶ Unfold",
            command=self.toggle_processing_panel,
            bg=Theme.SECONDARY,
            fg=Theme.TEXT_PRIMARY,
            font=Config.FONT_BUTTON,
            padx=15,
            pady=8
        )
        self.unfold_button.pack(side=tk.RIGHT)

        # --- 新增滚动容器结构 ---
        # 创建一个容器来包裹 Canvas 和 Scrollbar，用于控制显示/隐藏
        self.processing_scroll_container = tk.Frame(card, bg=Theme.SURFACE)
        
        # 1. 创建滚动条
        proc_scrollbar = ttk.Scrollbar(self.processing_scroll_container, orient="vertical")
        proc_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 2. 创建 Canvas (设置 height=350 来增加高度)
        self.proc_canvas = tk.Canvas(
            self.processing_scroll_container, 
            bg=Theme.SURFACE, 
            highlightthickness=0,
            height=350,  # <--- 这里增加了高度
            yscrollcommand=proc_scrollbar.set
        )
        self.proc_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 3. 链接滚动条
        proc_scrollbar.config(command=self.proc_canvas.yview)

        # 4. 创建内部 Frame (放置原来的滑块内容)
        self.processing_frame = tk.Frame(self.proc_canvas, bg=Theme.SURFACE)
        
        # 5. 将 Frame 放入 Canvas
        self.proc_canvas_window = self.proc_canvas.create_window(
            (0, 0), 
            window=self.processing_frame, 
            anchor="nw", 
            width=self.proc_canvas.winfo_reqwidth()
        )

        # 6. 绑定事件：当内部 Frame 大小改变时，更新滚动区域
        self.processing_frame.bind("<Configure>", lambda e: self.proc_canvas.configure(
            scrollregion=self.proc_canvas.bbox("all")
        ))
        
        # 7. 绑定事件：当 Canvas 大小改变时，调整内部 window 宽度以自适应
        self.proc_canvas.bind("<Configure>", lambda e: self.proc_canvas.itemconfig(
            self.proc_canvas_window, width=e.width
        ))

        # 8. 绑定鼠标滚轮 (在该区域内滚动)
        def _on_proc_mousewheel(event):
            if event.delta:
                self.proc_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif event.num == 4:
                self.proc_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.proc_canvas.yview_scroll(1, "units")

        self.proc_canvas.bind('<MouseWheel>', _on_proc_mousewheel)
        self.processing_frame.bind('<MouseWheel>', _on_proc_mousewheel)

        # ---------------------------
        # 以下是原有的滑块代码，完全保持不变，只是父级容器变成了 self.processing_frame
        # ---------------------------

        # Enhancement section
        enhance_frame = tk.Frame(self.processing_frame, bg=Theme.SURFACE)
        enhance_frame.pack(fill=tk.X, pady=(0, 10), padx=10) # 增加一点 padx 防止贴边

        tk.Label(
            enhance_frame,
            text="Enhancement",
            font=(Config.FONT_FAMILY, 13, "bold"),
            bg=Theme.SURFACE,
            fg=Theme.TEXT_PRIMARY,
            anchor=tk.W
        ).pack(anchor=tk.W, pady=(0, 5))

        # Contrast slider
        contrast_frame = tk.Frame(enhance_frame, bg=Theme.SURFACE)
        contrast_frame.pack(fill=tk.X, pady=2)

        tk.Label(
            contrast_frame,
            text="Contrast:",
            font=Config.FONT_SMALL,
            bg=Theme.SURFACE,
            fg=Theme.TEXT_SECONDARY,
            width=10,
            anchor=tk.W
        ).pack(side=tk.LEFT)

        self.contrast_var = tk.DoubleVar(value=1.0)
        contrast_scale = tk.Scale(
            contrast_frame,
            from_=0.1,
            to=3.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.contrast_var,
            bg=Theme.SURFACE,
            fg=Theme.TEXT_PRIMARY,
            highlightthickness=0,
            length=200
        )
        contrast_scale.pack(side=tk.LEFT, padx=(0, 10))

        self.contrast_entry = tk.Entry(
            contrast_frame,
            textvariable=self.contrast_var,
            font=Config.FONT_SMALL,
            width=6,
            justify=tk.CENTER
        )
        self.contrast_entry.pack(side=tk.LEFT)

        # Brightness slider
        brightness_frame = tk.Frame(enhance_frame, bg=Theme.SURFACE)
        brightness_frame.pack(fill=tk.X, pady=2)

        tk.Label(
            brightness_frame,
            text="Brightness:",
            font=Config.FONT_SMALL,
            bg=Theme.SURFACE,
            fg=Theme.TEXT_SECONDARY,
            width=10,
            anchor=tk.W
        ).pack(side=tk.LEFT)

        self.brightness_var = tk.IntVar(value=0)
        brightness_scale = tk.Scale(
            brightness_frame,
            from_=-100,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.brightness_var,
            bg=Theme.SURFACE,
            fg=Theme.TEXT_PRIMARY,
            highlightthickness=0,
            length=200
        )
        brightness_scale.pack(side=tk.LEFT, padx=(0, 10))

        self.brightness_entry = tk.Entry(
            brightness_frame,
            textvariable=self.brightness_var,
            font=Config.FONT_SMALL,
            width=6,
            justify=tk.CENTER
        )
        self.brightness_entry.pack(side=tk.LEFT)

        # Blur section
        blur_frame = tk.Frame(self.processing_frame, bg=Theme.SURFACE)
        blur_frame.pack(fill=tk.X, pady=(10, 10), padx=10)

        tk.Label(
            blur_frame,
            text="Gaussian Blur",
            font=(Config.FONT_FAMILY, 13, "bold"),
            bg=Theme.SURFACE,
            fg=Theme.TEXT_PRIMARY,
            anchor=tk.W
        ).pack(anchor=tk.W, pady=(0, 5))

        # Kernel size slider
        kernel_frame = tk.Frame(blur_frame, bg=Theme.SURFACE)
        kernel_frame.pack(fill=tk.X, pady=2)

        tk.Label(
            kernel_frame,
            text="Kernel Size:",
            font=Config.FONT_SMALL,
            bg=Theme.SURFACE,
            fg=Theme.TEXT_SECONDARY,
            width=10,
            anchor=tk.W
        ).pack(side=tk.LEFT)

        self.kernel_var = tk.IntVar(value=1)
        kernel_scale = tk.Scale(
            kernel_frame,
            from_=1,
            to=15,
            orient=tk.HORIZONTAL,
            variable=self.kernel_var,
            bg=Theme.SURFACE,
            fg=Theme.TEXT_PRIMARY,
            highlightthickness=0,
            length=200
        )
        kernel_scale.pack(side=tk.LEFT, padx=(0, 10))

        self.kernel_entry = tk.Entry(
            kernel_frame,
            textvariable=self.kernel_var,
            font=Config.FONT_SMALL,
            width=6,
            justify=tk.CENTER
        )
        self.kernel_entry.pack(side=tk.LEFT)

        # Sigma slider
        sigma_frame = tk.Frame(blur_frame, bg=Theme.SURFACE)
        sigma_frame.pack(fill=tk.X, pady=2)

        tk.Label(
            sigma_frame,
            text="Sigma:",
            font=Config.FONT_SMALL,
            bg=Theme.SURFACE,
            fg=Theme.TEXT_SECONDARY,
            width=10,
            anchor=tk.W
        ).pack(side=tk.LEFT)

        self.sigma_var = tk.DoubleVar(value=0.0)
        sigma_scale = tk.Scale(
            sigma_frame,
            from_=0.0,
            to=5.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.sigma_var,
            bg=Theme.SURFACE,
            fg=Theme.TEXT_PRIMARY,
            highlightthickness=0,
            length=200
        )
        sigma_scale.pack(side=tk.LEFT, padx=(0, 10))

        self.sigma_entry = tk.Entry(
            sigma_frame,
            textvariable=self.sigma_var,
            font=Config.FONT_SMALL,
            width=6,
            justify=tk.CENTER
        )
        self.sigma_entry.pack(side=tk.LEFT)

        # Sharpen section
        sharpen_frame = tk.Frame(self.processing_frame, bg=Theme.SURFACE)
        sharpen_frame.pack(fill=tk.X, pady=(10, 10), padx=10)

        tk.Label(
            sharpen_frame,
            text="Sharpening",
            font=(Config.FONT_FAMILY, 13, "bold"),
            bg=Theme.SURFACE,
            fg=Theme.TEXT_PRIMARY,
            anchor=tk.W
        ).pack(anchor=tk.W, pady=(0, 5))

        # Strength slider
        strength_frame = tk.Frame(sharpen_frame, bg=Theme.SURFACE)
        strength_frame.pack(fill=tk.X, pady=2)

        tk.Label(
            strength_frame,
            text="Strength:",
            font=Config.FONT_SMALL,
            bg=Theme.SURFACE,
            fg=Theme.TEXT_SECONDARY,
            width=10,
            anchor=tk.W
        ).pack(side=tk.LEFT)

        self.strength_var = tk.DoubleVar(value=0.0)
        strength_scale = tk.Scale(
            strength_frame,
            from_=0.0,
            to=3.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.strength_var,
            bg=Theme.SURFACE,
            fg=Theme.TEXT_PRIMARY,
            highlightthickness=0,
            length=200
        )
        strength_scale.pack(side=tk.LEFT, padx=(0, 10))

        self.strength_entry = tk.Entry(
            strength_frame,
            textvariable=self.strength_var,
            font=Config.FONT_SMALL,
            width=6,
            justify=tk.CENTER
        )
        self.strength_entry.pack(side=tk.LEFT)

        # Reset button
        button_frame = tk.Frame(self.processing_frame, bg=Theme.SURFACE)
        button_frame.pack(fill=tk.X, pady=(10, 15), padx=10)

        self.reset_processing_button = ModernButton(
            button_frame,
            text="Reset Processing",
            command=self.reset_processing,
            bg=Theme.WARNING,
            fg=Theme.TEXT_PRIMARY,
            font=Config.FONT_SMALL,
            padx=15,
            pady=8
        )
        self.reset_processing_button.pack(side=tk.LEFT)

        # Use variable tracing for real-time updates
        self.contrast_var.trace("w", lambda *args: self.apply_realtime_processing())
        self.brightness_var.trace("w", lambda *args: self.apply_realtime_processing())
        self.kernel_var.trace("w", lambda *args: self.apply_realtime_processing())
        self.sigma_var.trace("w", lambda *args: self.apply_realtime_processing())
        self.strength_var.trace("w", lambda *args: self.apply_realtime_processing())
        
    def create_model_card(self, parent):
        """Prediction card"""
        card = Card(parent)
        card.pack(fill=tk.X, pady=(0, 12))

        tk.Label(
            card,
            text="Prediction",
            font=Config.FONT_SECTION,
            bg=Theme.SURFACE,
            fg=Theme.TEXT_PRIMARY,
            anchor=tk.W
        ).pack(fill=tk.X, padx=20, pady=(15, 10))

        ModernButton(
            card,
            text="Predict Classification",
            command=self.predict_classification,
            bg=Theme.SECONDARY,
            fg=Theme.TEXT_PRIMARY,
            font=Config.FONT_BUTTON,
            width=12,
            padx=30,
            pady=10
        ).pack(pady=(0, 15))
    
    def create_results_card(self, parent):
        """Results display card"""
        card = Card(parent)
        card.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            card,
            text="Prediction Results",
            font=Config.FONT_SECTION,
            bg=Theme.SURFACE,
            fg=Theme.TEXT_PRIMARY,
            anchor=tk.W
        ).pack(fill=tk.X, padx=20, pady=(15, 8))

        results = tk.Frame(card, bg=Theme.PRIMARY_LIGHT)
        results.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))

        self.result_label = tk.Label(
            results,
            text="Awaiting prediction...",
            font=(Config.FONT_FAMILY, 18, "bold"),
            bg=Theme.PRIMARY_LIGHT,
            fg=Theme.TEXT_PRIMARY,
            wraplength=400,
            justify=tk.CENTER
        )
        self.result_label.pack(pady=8)

        self.confidence_label = tk.Label(
            results,
            text="Confidence: N/A",
            font=(Config.FONT_FAMILY, 12, "bold"),
            bg=Theme.PRIMARY_LIGHT,
            fg=Theme.TEXT_SECONDARY
        )
        self.confidence_label.pack(pady=(0, 8))

        # Colorbar frame - remove fixed height to allow adaptive sizing
        colorbar_frame = tk.Frame(results, bg=Theme.SURFACE)
        colorbar_frame.pack(fill=tk.X, padx=12, pady=(0, 8))

        self.colorbar_canvas_frame = tk.Frame(colorbar_frame, bg=Theme.SURFACE)
        self.colorbar_canvas_frame.pack(fill=tk.BOTH, expand=True)

        details_frame = tk.Frame(results, bg=Theme.SURFACE)
        details_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        # Text scrollbar
        text_scrollbar = ttk.Scrollbar(details_frame)
        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.details_text = tk.Text(
            details_frame,
            height=8,  # <--- 修改这里：从 15 减小到 8，减小下方框的高度
            font=("Consolas", 10),
            bg=Theme.SURFACE,
            fg=Theme.TEXT_PRIMARY,
            relief=tk.FLAT,
            borderwidth=6,
            wrap=tk.WORD,
            yscrollcommand=text_scrollbar.set
        )
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        text_scrollbar.config(command=self.details_text.yview)

        self.details_text.insert(1.0, 
            "Results will appear here.\n\n"
            "Includes:\n"
            "• Model details\n"
            "• Confidence visualization"
        )
        self.details_text.config(state=tk.DISABLED)
    
    # ==================== FOOTER ====================

    def create_footer(self):
        """Create status bar"""
        # Add separator line above footer
        separator = tk.Frame(self.root, bg=Theme.BORDER, height=1)
        separator.pack(fill=tk.X, side=tk.BOTTOM)

        footer = tk.Frame(self.root, bg=Theme.TEXT_PRIMARY, height=50)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)
        
        self.status = tk.Label(
            footer,
            text=f"Ready | {Config.DATASET_NAME} Classifier v{Config.VERSION}",
            font=(Config.FONT_FAMILY, 10, "bold"),  # Increased size and made bold
            bg=Theme.TEXT_PRIMARY,
            fg=Theme.TEXT_ON_COLOR,
            anchor=tk.W
        )
        self.status.pack(side=tk.LEFT, padx=30)
        
        status_text = "ML Backend: Active" if ML_BACKEND_AVAILABLE else "Demo Mode"
        status_color = Theme.SUCCESS if ML_BACKEND_AVAILABLE else Theme.WARNING
        
        tk.Label(
            footer,
            text=status_text,
            font=(Config.FONT_FAMILY, 10, "bold"),  # Increased size
            bg=Theme.TEXT_PRIMARY,
            fg=status_color
        ).pack(side=tk.RIGHT, padx=30)
    
    # ==================== FILE OPERATIONS ====================
    
    def open_image(self):
        """Open image file dialog"""
        path = filedialog.askopenfilename(
            title="Select Biopsy Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.load_image(path)
    
    def load_image(self, path):
        """Load and display image"""
        try:
            self.current_image_path = path
            self.original_image = Image.open(path)
            self.current_image = self.original_image.copy()
            self.display_image()
            
            self.image_info.config(
                text=f"File: {os.path.basename(path)} | "
                     f"Size: {self.original_image.size[0]}x{self.original_image.size[1]} | "
                     f"Format: {self.original_image.format}",
                font=(Config.FONT_FAMILY, 12, "bold")  # 增大字体
            )
            self.status.config(text=f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            show_message(self.root, "error", "Error", f"Failed to load image:\n{str(e)}")
    
    def display_image(self):
        """Display current image on canvas"""
        if not self.current_image:
            return
        
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        
        img = self.current_image.copy()
        if w > 1 and h > 1:
            img.thumbnail((w-40, h-40), Image.Resampling.LANCZOS)
        
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(
            w//2 if w>1 else 400,
            h//2 if h>1 else 300,
            image=self.photo
        )
    
    # ==================== CROP FUNCTIONALITY ====================
    
    def crop_image(self):
        """Toggle crop mode"""
        if not self.current_image_path:
            show_message(self.root, "warning", "No Image", "Please load an image first.")
            return

        self.crop_mode = not self.crop_mode
        self.canvas.config(cursor="cross" if self.crop_mode else "")

        # Update button appearance and status message
        if self.crop_mode:
            # Pressed state - darker background
            self.crop_button.set_pressed_state(self.crop_button_pressed_bg, self.crop_button_pressed_fg)
            self.status.config(text="Crop mode: ON - Click and drag on image to select area, release to crop")
        else:
            # Normal state - original appearance
            self.crop_button.set_normal_state(self.crop_button_normal_bg, self.crop_button_normal_fg)
            self.status.config(text="Crop mode: OFF")
    
    def on_canvas_press(self, e):
        """Start crop selection"""
        if self.crop_mode and self.current_image:
            self.crop_start = (e.x, e.y)
            if self.crop_rect:
                self.canvas.delete(self.crop_rect)
    
    def on_canvas_drag(self, e):
        """Update crop selection"""
        if self.crop_mode and self.crop_start:
            if self.crop_rect:
                self.canvas.delete(self.crop_rect)
            self.crop_rect = self.canvas.create_rectangle(
                *self.crop_start, e.x, e.y,
                outline=Theme.ACCENT,
                width=3,
                dash=(5,5)
            )
    
    def on_canvas_release(self, e):
        """Complete crop selection"""
        if not self.crop_mode or not self.crop_start:
            return

        end = (e.x, e.y)
        if abs(end[0]-self.crop_start[0])<50 or abs(end[1]-self.crop_start[1])<50:
            self.status.config(text="Crop cancelled: Selection too small (minimum 50x50 pixels)")
            if self.crop_rect:
                self.canvas.delete(self.crop_rect)
            self.crop_start = None
            return

        # Apply crop automatically without confirmation dialog
        self.apply_crop(self.crop_start, end)

        # Clean up
        if self.crop_rect:
            self.canvas.delete(self.crop_rect)
        self.crop_start = None
        self.crop_mode = False
        self.canvas.config(cursor="")

        # Reset crop button to normal state
        self.crop_button.set_normal_state(self.crop_button_normal_bg, self.crop_button_normal_fg)
    
    def apply_crop(self, start, end):
        """Apply crop to image and pad to square to maintain aspect ratio"""
        try:
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()

            disp = self.current_image.copy()
            disp.thumbnail((w-40, h-40), Image.Resampling.LANCZOS)
            dw, dh = disp.size

            ox = (w-dw)//2
            oy = (h-dh)//2

            x1 = max(0, start[0]-ox)
            y1 = max(0, start[1]-oy)
            x2 = min(dw, end[0]-ox)
            y2 = min(dh, end[1]-oy)

            sx = self.current_image.size[0]/dw
            sy = self.current_image.size[1]/dh

            cx1 = int(x1*sx)
            cy1 = int(y1*sy)
            cx2 = int(x2*sx)
            cy2 = int(y2*sy)

            cropped = self.current_image.crop((cx1,cy1,cx2,cy2))

            # Pad to square to maintain aspect ratio
            cw, ch = cropped.size
            max_side = max(cw, ch)
            square_img = Image.new('RGB', (max_side, max_side), (255, 255, 255))  # White background
            offset_x = (max_side - cw) // 2
            offset_y = (max_side - ch) // 2
            square_img.paste(cropped, (offset_x, offset_y))

            self.current_image = square_img
            self.cropped_image = square_img.copy()  # Store cropped version
            self.display_image()

            self.image_info.config(
                text=f"File: {os.path.basename(self.current_image_path)} (CROPPED) | "
                     f"Size: {self.current_image.size[0]}x{self.current_image.size[1]}",
                font=Config.FONT_BODY
            )
            self.status.config(text="Image cropped and padded to square")
        except Exception as e:
            show_message(self.root, "error", "Error", f"Crop failed:\n{str(e)}")
    
    def reset_image(self):
        """Reset to original image"""
        if not self.original_image:
            show_message(self.root, "warning", "No Image", "No image loaded")
            return

        if show_message(self.root, "question", "Reset", "Reset to original?"):
            self.current_image = self.original_image.copy()
            self.display_image()
            self.image_info.config(
                text=f"File: {os.path.basename(self.current_image_path)} | "
                     f"Size: {self.original_image.size[0]}x{self.original_image.size[1]} | "
                     f"Format: {self.original_image.format}",
                font=(Config.FONT_FAMILY, 12, "bold") # 增大字体
            )
            self.status.config(text="Image reset")
    
    # ==================== ML INTEGRATION ====================
    
    def extract_features(self):
        """Extract features using ML backend"""
        if not self.current_image_path:
            show_message(self.root, "warning", "No Image", "Load an image first")
            return

        method = self.feature_extraction_method.get()
        model_type = self.model_type.get()

        # Determine what features to extract based on model type
        if model_type == "fusion":
            # Fusion needs both radiomics and pixel features
            # But we'll extract radiomics here, pixel is handled in prediction
            method = "radiomics"
        elif model_type == "pixel":
            # For pixel models, we might not need explicit feature extraction
            # (CNN models work directly with images)
            if "MLP" in self.current_model_path.lower():
                method = "pixel"
            else:
                show_message(self.root, "info", "Info",
                    "CNN models work directly with images.\n"
                    "No explicit feature extraction needed.")
                return
        # For radiomics model, use radiomics features

        self.status.config(text=f"Extracting {method} features...")

        # Save cropped if needed
        if self.current_image != self.original_image:
            path = "/tmp/cropped.png"
            self.current_image.save(path)
        else:
            path = self.current_image_path

        try:
            if ML_BACKEND_AVAILABLE:
                print(f"\n⭐ Calling {method} feature extraction")
                if method == "radiomics":
                    features, names = extract_radiomics_features(path)
                else:
                    features, names = extract_pixel_features(path)

                self.extracted_features = features
                self.feature_names = names

                show_message(self.root, "info", "Success",
                    f"Features Extracted Successfully!\n\n"
                    f"Number of features: {len(features)}\n"
                    f"Method: {method.title()}"
                )
                self.status.config(text=f"{len(features)} features extracted successfully")
            else:
                show_message(self.root, "info", "Demo Mode",
                    "Feature extraction simulated\n"
                    "Create ml_backend.py for real extraction"
                )
                self.status.config(text="Demo: Features extracted")
        except Exception as e:
            show_message(self.root, "error", "Error", f"Extraction failed:\n{str(e)}")
            self.status.config(text="Extraction failed")
    
    def get_model_type_from_path(self, model_path):
        """Determine model type from model path"""
        model_name = os.path.basename(model_path).lower()
        folder_name = os.path.basename(os.path.dirname(model_path)).lower()

        # Check for Random Forest models first (highest priority)
        if "randomforest" in folder_name or "randomforest" in model_name:
            if "radiomics" in folder_name or "radiomics" in model_name:
                return "radiomics_ml"
            elif "sift" in folder_name or "sift" in model_name:
                return "sift_ml"
            else:
                return "pixel_ml"  # fallback

        # Check for radiomics models
        elif "radiomics" in folder_name or "radiomics" in model_name:
            return "radiomics"
        # Check for SIFT fusion models (highest priority - check for both SIFT and Fusion)
        elif ("sift" in folder_name or "sift" in model_name) and ("fusion" in folder_name or "fusion" in model_name):
            return "sift_fusion"
        # Check for SIFT fusion models (alternative naming)
        elif "siftfusion" in folder_name or "siftfusion" in model_name:
            return "sift_fusion"
        # Check for SIFT ML models
        elif ("sift" in folder_name or "sift" in model_name) and ("ml" in model_name or "ml" in folder_name):
            return "sift_ml"
        # Check for fusion models
        elif "fusion" in folder_name or "fusion" in model_name:
            return "fusion"
        # Check for MLP models (pixel-based)
        elif "mlp" in model_name or "simplemlp" in model_name:
            return "pixel"
        # Default to pixel (CNN models)
        else:
            return "pixel"

    def auto_extract_features(self, model_type, img_path):
        """Auto-extract features based on model type"""
        try:
            if model_type == "radiomics":
                self.status.config(text="Auto-extracting radiomics features...")
                features, names = extract_radiomics_features(img_path)
            elif model_type == "fusion":
                self.status.config(text="Auto-extracting radiomics features for fusion...")
                features, names = extract_radiomics_features(img_path)
            elif model_type == "sift_ml":
                self.status.config(text="Auto-extracting SIFT features...")
                features, names = extract_sift_bof_features(img_path)
            elif model_type == "sift_fusion":
                self.status.config(text="Auto-extracting SIFT features for fusion...")
                features, names = extract_sift_bof_features(img_path)
            else:  # pixel models
                if "mlp" in self.current_model_path.lower():
                    self.status.config(text="Auto-extracting pixel features...")
                    features, names = extract_pixel_features(img_path)
                else:
                    # CNN models don't need features
                    return None, None

            self.extracted_features = features
            self.feature_names = names
            return features, names

        except Exception as e:
            show_message(self.root, "error", "Feature Extraction Error", f"Failed to extract features:\n{str(e)}")
            return None, None

    def predict_all_models(self):
        """Run predictions on all available models and aggregate results"""
        if not self.current_image_path:
            show_message(self.root, "warning", "No Image", "Load an image first")
            return

        self.status.config(text="Running predictions on all models...")

        import time
        start_time = time.time()

        try:
            # Get all model paths (excluding "All models")
            all_models = []
            for display_name, model_path in self.model_path_map.items():
                if model_path != "ALL_MODELS" and os.path.exists(model_path):
                    all_models.append((display_name, model_path))

            if not all_models:
                show_message(self.root, "error", "No Models", "No valid model files found")
                return

            # Prepare image path
            if self.current_image != self.original_image:
                img_path = "/tmp/cropped.png"
                self.current_image.save(img_path)
            else:
                img_path = self.current_image_path

            # Run predictions on all models
            all_results = []
            failed_models = []

            for display_name, model_path in all_models:
                try:
                    model_type = self.get_model_type_from_path(model_path)

                    # Auto-extract features if needed
                    features = None
                    if model_type in ["radiomics", "fusion", "sift_ml", "sift_fusion"] or ("mlp" in model_path.lower()):
                        features, _ = self.auto_extract_features(model_type, img_path)
                        if features is None and model_type in ["radiomics", "fusion", "sift_ml", "sift_fusion"]:
                            failed_models.append(display_name)
                            continue

                    print(f"🔍 Predicting with {display_name}")
                    pred, conf, details = predict_with_model(
                        model_path,
                        features=features,
                        img_path=img_path if (model_type == "pixel" and "mlp" not in model_path.lower()) or model_type in ["sift_fusion"] else None,
                        model_type=model_type
                    )

                    if pred == "ERROR":
                        failed_models.append(display_name)
                        continue

                    all_results.append({
                        'model_name': display_name,
                        'model_path': model_path,
                        'model_type': model_type,
                        'prediction': pred,
                        'confidence': conf,
                        'details': details
                    })

                except Exception as e:
                    print(f"❌ Failed to predict with {display_name}: {e}")
                    failed_models.append(display_name)
                    continue

            if not all_results:
                show_message(self.root, "error", "All Models Failed", "No models were able to make predictions")
                self.status.config(text="All predictions failed")
                return

            # Aggregate results using majority vote + confidence sorting
            aggregated_result = self.aggregate_model_results(all_results)

            processing_time = time.time() - start_time

            # Display aggregated results
            self.display_all_models_results(aggregated_result, all_results, failed_models, processing_time)

            self.status.config(text=f"All models prediction complete: {aggregated_result['final_prediction']}")

        except Exception as e:
            show_message(self.root, "error", "Error", f"All models prediction failed:\n{str(e)}")
            import traceback
            traceback.print_exc()
            self.status.config(text="All models prediction failed")

    def aggregate_model_results(self, all_results):
        """Aggregate results from all models using majority vote and confidence sorting"""
        # Group results by prediction
        benign_results = [r for r in all_results if r['prediction'] == 'BENIGN']
        malignant_results = [r for r in all_results if r['prediction'] == 'MALIGNANT']

        # Determine majority vote
        benign_count = len(benign_results)
        malignant_count = len(malignant_results)

        if benign_count > malignant_count:
            majority_prediction = 'BENIGN'
            majority_results = benign_results
            minority_results = malignant_results
        elif malignant_count > benign_count:
            majority_prediction = 'MALIGNANT'
            majority_results = malignant_results
            minority_results = benign_results
        else:
            # Tie - use highest confidence
            all_sorted = sorted(all_results, key=lambda x: x['confidence'], reverse=True)
            majority_prediction = all_sorted[0]['prediction']
            majority_results = [r for r in all_results if r['prediction'] == majority_prediction]
            minority_results = [r for r in all_results if r['prediction'] != majority_prediction]

        # Sort majority results by confidence (highest first)
        majority_sorted = sorted(majority_results, key=lambda x: x['confidence'], reverse=True)
        minority_sorted = sorted(minority_results, key=lambda x: x['confidence'], reverse=True)

        # Calculate average confidence for majority prediction
        if majority_results:
            avg_confidence = sum(r['confidence'] for r in majority_results) / len(majority_results)
        else:
            avg_confidence = 0.5

        return {
            'final_prediction': majority_prediction,
            'avg_confidence': avg_confidence,
            'majority_count': len(majority_results),
            'minority_count': len(minority_results),
            'total_models': len(all_results),
            'majority_results': majority_sorted,
            'minority_results': minority_sorted
        }

    def display_all_models_results(self, aggregated_result, all_results, failed_models, processing_time):
        """Display aggregated results from all models"""
        final_pred = aggregated_result['final_prediction']
        avg_conf = aggregated_result['avg_confidence']

        # Set result colors
        if final_pred == "BENIGN":
            color = Theme.SUCCESS
            bg = Theme.SUCCESS_BG
            interpretation = "Low risk - recommend routine monitoring"
        else:
            color = Theme.DANGER
            bg = Theme.DANGER_BG
            interpretation = "High risk - recommend immediate clinical review"

        self.result_label.config(text=f"[{final_pred}] (All Models)", fg=color, bg=bg)
        self.confidence_label.config(text=f"Average Confidence: {avg_conf*100:.1f}%", bg=bg)

        # Display confidence visualization
        self.display_all_models_confidence(aggregated_result, all_results)

        # Generate detailed report
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)

        report = f"ALL MODELS PREDICTION REPORT\n{'='*60}\n\n"
        report += f"Final Classification: {final_pred}\n"
        report += f"Average Confidence: {avg_conf*100:.1f}%\n"
        report += f"Clinical Interpretation: {interpretation}\n\n"

        report += f"Ensemble Statistics:\n"
        report += f"  Total Models: {aggregated_result['total_models']}\n"
        report += f"  Majority Vote: {aggregated_result['majority_count']} vs {aggregated_result['minority_count']}\n"
        report += f"  Consensus: {aggregated_result['majority_count']/aggregated_result['total_models']*100:.1f}%\n"
        report += f"  Processing Time: {processing_time:.2f} seconds\n\n"

        # List all model results
        report += "INDIVIDUAL MODEL RESULTS\n"
        report += "(Sorted by majority prediction first, then by confidence)\n\n"

        # Show majority results first
        if aggregated_result['majority_results']:
            report += f"MAJORITY PREDICTION ({final_pred}):\n"
            for i, result in enumerate(aggregated_result['majority_results'], 1):
                report += f"  {i}. {result['model_name']}\n"
                report += f"     Prediction: {result['prediction']} | Confidence: {result['confidence']*100:.1f}%\n"
                report += f"     Type: {result['model_type'].upper()}\n\n"

        # Show minority results
        if aggregated_result['minority_results']:
            minority_pred = aggregated_result['minority_results'][0]['prediction']
            report += f"MINORITY PREDICTION ({minority_pred}):\n"
            for i, result in enumerate(aggregated_result['minority_results'], 1):
                report += f"  {i}. {result['model_name']}\n"
                report += f"     Prediction: {result['prediction']} | Confidence: {result['confidence']*100:.1f}%\n"
                report += f"     Type: {result['model_type'].upper()}\n\n"

        # Show failed models
        if failed_models:
            report += f"FAILED MODELS ({len(failed_models)}):\n"
            for model in failed_models:
                report += f"  • {model}\n"
            report += "\n"

        # Confidence assessment
        conf_level = "High" if avg_conf > 0.8 else "Medium" if avg_conf > 0.6 else "Low"
        report += f"Ensemble Confidence Assessment: {conf_level}\n"

        if aggregated_result['majority_count'] / aggregated_result['total_models'] < 0.6:
            report += "⚠️  Low consensus - consider additional testing\n"
        elif avg_conf < 0.6:
            report += "⚠️  Low average confidence - clinical correlation recommended\n"
        elif aggregated_result['majority_count'] == aggregated_result['total_models']:
            report += "✓ Unanimous agreement - high reliability\n"
        else:
            report += "✓ Majority consensus achieved\n"

        report += "\n" + "="*60 + "\n"
        report += f"Image: {os.path.basename(self.current_image_path)}\n"
        report += f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"

        self.details_text.insert(1.0, report)
        self.details_text.config(state=tk.DISABLED)

    def display_all_models_confidence(self, aggregated_result, all_results):
        """Display confidence visualization for all models"""
        try:
            # Clear previous widgets
            for widget in self.colorbar_canvas_frame.winfo_children():
                widget.destroy()

            # Create a frame for the visualization
            main_frame = tk.Frame(self.colorbar_canvas_frame, bg=Theme.SURFACE)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Title
            title_label = tk.Label(
                main_frame,
                text=f"All Models Confidence Distribution",
                font=(Config.FONT_FAMILY, 12, "bold"),
                bg=Theme.SURFACE,
                fg=Theme.TEXT_PRIMARY,
                anchor=tk.W
            )
            title_label.pack(fill=tk.X, pady=(0, 10))

            # Create two columns for majority and minority
            columns_frame = tk.Frame(main_frame, bg=Theme.SURFACE)
            columns_frame.pack(fill=tk.BOTH, expand=True)

            # Majority column
            majority_frame = tk.Frame(columns_frame, bg=Theme.SURFACE)
            majority_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

            majority_pred = aggregated_result['final_prediction']
            tk.Label(
                majority_frame,
                text=f"Majority ({majority_pred})",
                font=(Config.FONT_FAMILY, 11, "bold"),
                bg=Theme.SURFACE,
                fg=Theme.SUCCESS if majority_pred == "BENIGN" else Theme.DANGER,
                anchor=tk.W
            ).pack(fill=tk.X, pady=(0, 5))

            # Show majority models
            for result in aggregated_result['majority_results']:
                model_frame = tk.Frame(majority_frame, bg=Theme.SURFACE)
                model_frame.pack(fill=tk.X, pady=1)

                # Model name (truncated)
                model_name = result['model_name']
                if len(model_name) > 20:
                    model_name = model_name[:17] + "..."

                tk.Label(
                    model_frame,
                    text=model_name,
                    font=(Config.FONT_FAMILY, 9),
                    bg=Theme.SURFACE,
                    fg=Theme.TEXT_SECONDARY,
                    anchor=tk.W,
                    width=20
                ).pack(side=tk.LEFT)

                # Confidence bar
                bar_frame = tk.Frame(model_frame, bg=Theme.BORDER, height=12)
                bar_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
                bar_frame.pack_propagate(False)

                bar_width = int(result['confidence'] * 100)
                if bar_width > 0:
                    tk.Frame(
                        bar_frame,
                        bg=Theme.SUCCESS if majority_pred == "BENIGN" else Theme.DANGER,
                        width=bar_width,
                        height=12
                    ).pack(side=tk.LEFT)

                # Remove percentage display to save space in all models mode

            # Minority column
            if aggregated_result['minority_results']:
                minority_frame = tk.Frame(columns_frame, bg=Theme.SURFACE)
                minority_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

                minority_pred = aggregated_result['minority_results'][0]['prediction']
                tk.Label(
                    minority_frame,
                    text=f"Minority ({minority_pred})",
                    font=(Config.FONT_FAMILY, 11, "bold"),
                    bg=Theme.SURFACE,
                    fg=Theme.SUCCESS if minority_pred == "BENIGN" else Theme.DANGER,
                    anchor=tk.W
                ).pack(fill=tk.X, pady=(0, 5))

                # Show minority models
                for result in aggregated_result['minority_results']:
                    model_frame = tk.Frame(minority_frame, bg=Theme.SURFACE)
                    model_frame.pack(fill=tk.X, pady=1)

                    # Model name (truncated)
                    model_name = result['model_name']
                    if len(model_name) > 20:
                        model_name = model_name[:17] + "..."

                    tk.Label(
                        model_frame,
                        text=model_name,
                        font=(Config.FONT_FAMILY, 9),
                        bg=Theme.SURFACE,
                        fg=Theme.TEXT_SECONDARY,
                        anchor=tk.W,
                        width=20
                    ).pack(side=tk.LEFT)

                    # Confidence bar
                    bar_frame = tk.Frame(model_frame, bg=Theme.BORDER, height=12)
                    bar_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
                    bar_frame.pack_propagate(False)

                    bar_width = int(result['confidence'] * 100)
                    if bar_width > 0:
                        tk.Frame(
                            bar_frame,
                            bg=Theme.SUCCESS if minority_pred == "BENIGN" else Theme.DANGER,
                            width=bar_width,
                            height=12
                        ).pack(side=tk.LEFT)

                # Remove percentage display to save space in all models mode

            # Summary stats
            stats_frame = tk.Frame(main_frame, bg=Theme.SURFACE)
            stats_frame.pack(fill=tk.X, pady=(10, 0))

            tk.Label(
                stats_frame,
                text=f"Consensus: {aggregated_result['majority_count']}/{aggregated_result['total_models']} "
                     f"({aggregated_result['majority_count']/aggregated_result['total_models']*100:.1f}%) | "
                     f"Avg Confidence: {aggregated_result['avg_confidence']*100:.1f}%",
                font=(Config.FONT_FAMILY, 9, "bold"),
                bg=Theme.SURFACE,
                fg=Theme.TEXT_PRIMARY,
                anchor=tk.CENTER
            ).pack(fill=tk.X)

        except Exception as e:
            print(f"All models confidence display failed: {e}")
            # Fallback: display simple text
            fallback_label = tk.Label(
                self.colorbar_canvas_frame,
                text=f"All Models - {aggregated_result['final_prediction']} "
                     f"(Consensus: {aggregated_result['majority_count']}/{aggregated_result['total_models']})",
                font=(Config.FONT_FAMILY, 12, "bold"),
                bg=Theme.SURFACE,
                fg=Theme.TEXT_PRIMARY
            )
            fallback_label.pack(pady=20)

    def predict_classification(self):
        """Predict using ML backend with auto feature extraction"""
        if not self.current_image_path:
            show_message(self.root, "warning", "No Image", "Load an image first")
            return

        # Check if "All models" is selected
        if self.current_model_path == "ALL_MODELS":
            self.predict_all_models()
            return

        if not self.current_model_path or not os.path.exists(self.current_model_path):
            show_message(self.root, "error", "No Model", "Please select a valid model file")
            return

        # Auto-determine model type
        model_type = self.get_model_type_from_path(self.current_model_path)
        self.status.config(text=f"Predicting with {model_type} model...")

        import time
        start_time = time.time()

        try:
            if ML_BACKEND_AVAILABLE:
                # Prepare image path
                if self.current_image != self.original_image:
                    img_path = "/tmp/cropped.png"
                    self.current_image.save(img_path)
                else:
                    img_path = self.current_image_path

                # Auto-extract features if needed
                features = None
                feature_count = 0
                if model_type in ["radiomics", "fusion", "sift_ml", "sift_fusion"] or ("mlp" in self.current_model_path.lower()):
                    features, _ = self.auto_extract_features(model_type, img_path)
                    if features is not None:
                        feature_count = len(features)
                    if features is None and model_type in ["radiomics", "fusion", "sift_ml", "sift_fusion"]:
                        return  # Error already shown

                print(f"\n⭐ Calling prediction with {model_type} model")
                pred, conf, details = predict_with_model(
                    self.current_model_path,
                    features=features,
                    img_path=img_path if (model_type == "pixel" and "mlp" not in self.current_model_path.lower()) or model_type in ["sift_fusion"] else None,
                    model_type=model_type
                )

                processing_time = time.time() - start_time

                if pred == "ERROR":
                    show_message(self.root, "error", "Prediction Error", details.get('error', 'Unknown error'))
                    self.status.config(text="Prediction failed")
                else:
                    self.display_results(pred, conf, model_type, details, processing_time, feature_count)
                    self.status.config(text=f"Prediction complete: {pred}")
            else:
                import random
                pred = random.choice(["BENIGN", "MALIGNANT"])
                conf = random.uniform(0.75, 0.99)
                processing_time = time.time() - start_time
                self.display_results(pred, conf, model_type, {'demo': True}, processing_time, 0)
                self.status.config(text=f"Demo: {pred}")
        except Exception as e:
            show_message(self.root, "error", "Error", f"Prediction failed:\n{str(e)}")
            import traceback
            traceback.print_exc()
            self.status.config(text="Prediction failed")
    
    def display_results(self, pred, conf, model, details=None, processing_time=None, feature_count=0):
        """Display prediction results with colorbar visualization"""
        if pred == "BENIGN":
            color = Theme.SUCCESS
            bg = Theme.SUCCESS_BG
            status_text = "[BENIGN]"
            interpretation = "Low risk - recommend routine monitoring"
        else:
            color = Theme.DANGER
            bg = Theme.DANGER_BG
            status_text = "[MALIGNANT]"
            interpretation = "High risk - recommend immediate clinical review"

        self.result_label.config(text=status_text, fg=color, bg=bg)
        self.confidence_label.config(text=f"Confidence: {conf*100:.1f}%", bg=bg)

        # Display colorbar for confidence visualization
        self.display_confidence_colorbar(conf, pred)

        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)

        report = f"PREDICTION REPORT\n{'='*50}\n\n"
        report += f"Classification: {pred}\n"
        report += f"Confidence Level: {conf*100:.2f}%\n"
        report += f"Clinical Interpretation: {interpretation}\n\n"

        report += f"Model Details:\n"
        report += f"  Type: {model.upper()}\n"
        report += f"  Path: {os.path.basename(self.current_model_path)}\n"

        if feature_count > 0:
            report += f"  Features Used: {feature_count}\n"

        if processing_time is not None:
            report += f"  Processing Time: {processing_time:.2f} seconds\n"

        report += "\n"

        if details and 'probabilities' in details:
            p = details['probabilities']
            report += f"Probability Breakdown:\n"
            report += f"  Benign: {p.get('benign',0)*100:.2f}%\n"
            report += f"  Malignant: {p.get('malignant',0)*100:.2f}%\n\n"

        # Confidence interpretation
        conf_level = "High" if conf > 0.8 else "Medium" if conf > 0.6 else "Low"
        report += f"Confidence Assessment: {conf_level}\n"

        if conf < 0.6:
            report += "⚠️  Low confidence - consider additional testing\n"
        elif conf < 0.8:
            report += "ℹ️  Moderate confidence - clinical correlation recommended\n"
        else:
            report += "✓ High confidence - reliable prediction\n"

        report += "\n" + "="*50 + "\n"
        report += f"Image: {os.path.basename(self.current_image_path)}\n"
        report += f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"

        if details and details.get('demo'):
            report += "\n⚠️  Note: Demo mode - simulated results for testing"

        self.details_text.insert(1.0, report)
        self.details_text.config(state=tk.DISABLED)

    def display_confidence_colorbar(self, confidence, prediction):
        """Display a progress bar visualization of confidence level"""
        try:
            # Clear previous widgets
            for widget in self.colorbar_canvas_frame.winfo_children():
                widget.destroy()

            # Create a frame for the progress bar and labels
            progress_frame = tk.Frame(self.colorbar_canvas_frame, bg=Theme.SURFACE)
            progress_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Title label
            title_label = tk.Label(
                progress_frame,
                text=f"Confidence Level - {prediction}",
                font=(Config.FONT_FAMILY, 12, "bold"),
                bg=Theme.SURFACE,
                fg=Theme.SUCCESS if prediction == "BENIGN" else Theme.DANGER,
                anchor=tk.W
            )
            title_label.pack(fill=tk.X, pady=(0, 5))

            # Progress bar style configuration
            style = ttk.Style()
            style.configure("Confidence.Horizontal.TProgressbar",
                           background=Theme.SUCCESS if prediction == "BENIGN" else Theme.DANGER,
                           troughcolor=Theme.BORDER,
                           borderwidth=1,
                           lightcolor=Theme.SURFACE,
                           darkcolor=Theme.SURFACE)

            # Create progress bar
            progress_bar = ttk.Progressbar(
                progress_frame,
                style="Confidence.Horizontal.TProgressbar",
                orient=tk.HORIZONTAL,
                length=400,
                mode='determinate',
                maximum=100,
                value=confidence * 100
            )
            progress_bar.pack(fill=tk.X, pady=(0, 5))

            # Percentage labels
            labels_frame = tk.Frame(progress_frame, bg=Theme.SURFACE)
            labels_frame.pack(fill=tk.X)

            tk.Label(
                labels_frame,
                text="0%",
                font=(Config.FONT_FAMILY, 9),
                bg=Theme.SURFACE,
                fg=Theme.TEXT_SECONDARY,
                anchor=tk.W
            ).pack(side=tk.LEFT)

            tk.Label(
                labels_frame,
                text="50%",
                font=(Config.FONT_FAMILY, 9),
                bg=Theme.SURFACE,
                fg=Theme.TEXT_SECONDARY,
                anchor=tk.CENTER
            ).pack(side=tk.LEFT, expand=True)

            tk.Label(
                labels_frame,
                text="100%",
                font=(Config.FONT_FAMILY, 9),
                bg=Theme.SURFACE,
                fg=Theme.TEXT_SECONDARY,
                anchor=tk.E
            ).pack(side=tk.RIGHT)

            # Current confidence value label
            conf_label = tk.Label(
                progress_frame,
                text=f"Current: {confidence*100:.1f}%",
                font=(Config.FONT_FAMILY, 11, "bold"),
                bg=Theme.SURFACE,
                fg=Theme.TEXT_PRIMARY,
                anchor=tk.CENTER
            )
            conf_label.pack(fill=tk.X, pady=(5, 0))

        except Exception as e:
            print(f"Progress bar creation failed: {e}")
            # Fallback: display text-based confidence indicator
            fallback_label = tk.Label(
                self.colorbar_canvas_frame,
                text=f"Confidence: {confidence*100:.1f}%",
                font=(Config.FONT_FAMILY, 12, "bold"),
                bg=Theme.SURFACE,
                fg=Theme.TEXT_PRIMARY
            )
            fallback_label.pack(pady=20)
    
    def show_gradcam(self):
        """Show GradCAM visualization"""
        if not self.current_image_path:
            show_message(self.root, "warning", "No Image", "Load an image first")
            return

        if not self.current_model_path or not os.path.exists(self.current_model_path):
            show_message(self.root, "error", "No Model", "Please select a valid model file.\nThis model is for one model only, not 'All models'.")
            return

        model_type = self.get_model_type_from_path(self.current_model_path)

        # GradCAM only works with pixel CNN models
        if model_type != "pixel" or "mlp" in self.current_model_path.lower():
            show_message(self.root, "warning", "GradCAM Not Available",
                "GradCAM is only available for CNN models (pixel-only).\n"
                "Please select a CNN model like ResNet50_PT_FT."
                "Make sure you have predictions available.")
            return

        try:
            self.status.config(text="Generating GradCAM...")

            # Prepare image path
            if self.current_image != self.original_image:
                img_path = "/tmp/cropped.png"
                self.current_image.save(img_path)
            else:
                img_path = self.current_image_path

            if ML_BACKEND_AVAILABLE:
                cam, img_rgb, pred_info = generate_gradcam(
                    self.current_model_path,
                    img_path,
                    model_type=model_type
                )

                # Display GradCAM in a new window
                self.display_gradcam_window(img_rgb, cam, pred_info)
                self.status.config(text="GradCAM generated")
            else:
                show_message(self.root, "info", "Demo Mode", "GradCAM not available in demo mode")
        except Exception as e:
            show_message(self.root, "error", "Error", f"GradCAM generation failed:\n{str(e)}")
            import traceback
            traceback.print_exc()
            self.status.config(text="GradCAM failed")

    def show_occlusion(self):
        """Show Occlusion Sensitivity visualization"""
        if not self.current_image_path:
            show_message(self.root, "warning", "No Image", "Load an image first")
            return

        if not self.current_model_path or not os.path.exists(self.current_model_path):
            show_message(self.root, "error", "No Model", "Please select a valid model file.\nThis model is for one model only, not 'All models'.")
            return

        model_type = self.get_model_type_from_path(self.current_model_path)

        # Occlusion works with pixel CNN models
        if model_type != "pixel" or "mlp" in self.current_model_path.lower():
            show_message(self.root, "warning", "Occlusion Not Available",
                "Occlusion sensitivity is available for CNN models (pixel-only).\n"
                "Please select a CNN model like ResNet50_PT_FT."
                "Make sure you have predictions available.")
            return

        try:
            self.status.config(text="Generating Occlusion Map...")

            # Prepare image path
            if self.current_image != self.original_image:
                img_path = "/tmp/cropped.png"
                self.current_image.save(img_path)
            else:
                img_path = self.current_image_path

            if ML_BACKEND_AVAILABLE:
                occlusion_map, img_rgb, pred_info = generate_occlusion(
                    self.current_model_path,
                    img_path,
                    model_type=model_type
                )

                # Display Occlusion in a new window
                self.display_occlusion_window(img_rgb, occlusion_map, pred_info)
                self.status.config(text="Occlusion map generated")
            else:
                show_message(self.root, "info", "Demo Mode", "Occlusion not available in demo mode")
        except Exception as e:
            show_message(self.root, "error", "Error", f"Occlusion generation failed:\n{str(e)}")
            import traceback
            traceback.print_exc()
            self.status.config(text="Occlusion failed")
    
    def display_gradcam_window(self, img_rgb, cam, pred_info):
        """Display GradCAM visualization in a new window"""
        try:
            import matplotlib
            matplotlib.use('TkAgg')  # Use TkAgg backend
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.colors import Normalize
            from matplotlib.cm import ScalarMappable
        except ImportError:
            show_message(self.root, "error", "Error", "matplotlib is required for GradCAM visualization")
            return

        # Create new window
        gradcam_window = tk.Toplevel(self.root)
        gradcam_window.title("GradCAM Visualization")
        # Remove fixed geometry to allow adaptive resizing

        # Create figure with original layout + colorbar below heatmap
        fig = plt.figure(figsize=(10, 7))

        # Create subplots: 2 images side by side, colorbar below the second image
        gs = fig.add_gridspec(2, 2, height_ratios=[5, 0.8], hspace=0.1)

        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img_rgb)
        ax1.set_title("Original Image", fontsize=12, fontweight='bold')
        ax1.axis("off")

        # GradCAM overlay
        ax2 = fig.add_subplot(gs[0, 1])
        cam_resized = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = 0.5 * heatmap + 0.5 * (img_rgb / 255.0)
        overlay = np.clip(overlay, 0, 1)

        im = ax2.imshow(overlay)
        ax2.set_title("GradCAM Heatmap Overlay", fontsize=12, fontweight='bold')
        ax2.axis("off")

        # Horizontal colorbar below the heatmap
        ax_cbar = fig.add_subplot(gs[1, 1])
        # Create a ScalarMappable with the same colormap as our heatmap
        norm = Normalize(vmin=0, vmax=1)
        sm = ScalarMappable(cmap=plt.cm.jet, norm=norm)
        sm.set_array([])  # No data needed, just for colorbar

        cbar = plt.colorbar(sm, cax=ax_cbar, orientation='horizontal', shrink=0.8)
        cbar.set_label('Activation Intensity (Red: High, Blue: Low)', fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)

        # Description text below colorbar
        desc_ax = fig.add_subplot(gs[1, 0])
        desc_ax.axis('off')
        desc_text = (
            "GradCAM Explanation:\n"
            "• Red: High activation areas\n"
            "• Blue: Low activation areas\n"
            "Shows which image regions influenced the prediction most."
        )
        desc_ax.text(0.5, 0.5, desc_text, ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray', alpha=0.7))

        # Add prediction info
        pred = pred_info['prediction']
        conf = pred_info['confidence']
        color = Theme.SUCCESS if pred == "BENIGN" else Theme.DANGER
        fig.suptitle(
            f"GradCAM Analysis - Prediction: {pred} | Confidence: {conf*100:.1f}%",
            fontsize=14,
            weight='bold',
            color=color,
            y=0.95
        )

        plt.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, gradcam_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Close button
        btn_frame = tk.Frame(gradcam_window)
        btn_frame.pack(pady=10)

        ModernButton(
            btn_frame,
            text="Close",
            command=gradcam_window.destroy,
            bg=Theme.PRIMARY,
            fg=Theme.TEXT_ON_COLOR,
            font=Config.FONT_BODY,
            padx=20,
            pady=5
        ).pack()

    def display_occlusion_window(self, img_rgb, occlusion_map, pred_info):
        """Display Occlusion Sensitivity visualization in a new window"""
        try:
            import matplotlib
            matplotlib.use('TkAgg')  # Use TkAgg backend
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.colors import Normalize
            from matplotlib.cm import ScalarMappable
        except ImportError:
            show_message(self.root, "error", "Error", "matplotlib is required for Occlusion visualization")
            return

        # Create new window
        occlusion_window = tk.Toplevel(self.root)
        occlusion_window.title("Occlusion Sensitivity Map")
        # Remove fixed geometry to allow adaptive resizing

        # Create figure with original layout + colorbar below heatmap
        fig = plt.figure(figsize=(10, 7))

        # Create subplots: 2 images side by side, colorbar below the second image
        gs = fig.add_gridspec(2, 2, height_ratios=[5, 0.8], hspace=0.1)

        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img_rgb)
        ax1.set_title("Original Image", fontsize=12, fontweight='bold')
        ax1.axis("off")

        # Occlusion overlay
        ax2 = fig.add_subplot(gs[0, 1])
        occ_resized = cv2.resize(occlusion_map, (img_rgb.shape[1], img_rgb.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * occ_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = 0.5 * heatmap + 0.5 * (img_rgb / 255.0)
        overlay = np.clip(overlay, 0, 1)

        im = ax2.imshow(overlay)
        ax2.set_title("Occlusion Sensitivity Overlay", fontsize=12, fontweight='bold')
        ax2.axis("off")

        # Horizontal colorbar below the heatmap
        ax_cbar = fig.add_subplot(gs[1, 1])
        # Create a ScalarMappable with the same colormap as our heatmap
        norm = Normalize(vmin=0, vmax=1)
        sm = ScalarMappable(cmap=plt.cm.jet, norm=norm)
        sm.set_array([])  # No data needed, just for colorbar

        cbar = plt.colorbar(sm, cax=ax_cbar, orientation='horizontal', shrink=0.8)
        cbar.set_label('Sensitivity Score (Red: High, Blue: Low)', fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)

        # Description text below colorbar
        desc_ax = fig.add_subplot(gs[1, 0])
        desc_ax.axis('off')
        desc_text = (
            "Occlusion Explanation:\n"
            "• Red: High sensitivity areas\n"
            "• Blue: Low sensitivity areas\n"
            "Shows which image regions are critical for prediction."
        )
        desc_ax.text(0.5, 0.5, desc_text, ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray', alpha=0.7))

        # Add prediction info
        pred = pred_info['prediction']
        conf = pred_info['confidence']
        color = Theme.SUCCESS if pred == "BENIGN" else Theme.DANGER
        fig.suptitle(
            f"Occlusion Sensitivity Analysis - Prediction: {pred} | Confidence: {conf*100:.1f}%",
            fontsize=14,
            weight='bold',
            color=color,
            y=0.95
        )

        plt.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, occlusion_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Close button
        btn_frame = tk.Frame(occlusion_window)
        btn_frame.pack(pady=10)

        ModernButton(
            btn_frame,
            text="Close",
            command=occlusion_window.destroy,
            bg=Theme.PRIMARY,
            fg=Theme.TEXT_ON_COLOR,
            font=Config.FONT_BODY,
            padx=20,
            pady=5
        ).pack()

    def apply_image_processing(self, operation):
        """Apply image processing operation"""
        if not self.current_image_path:
            show_message(self.root, "warning", "No Image", "Load an image first")
            return

        if not ML_BACKEND_AVAILABLE:
            show_message(self.root, "warning", "Backend Not Available",
                "Image processing requires the ML backend.\n"
                "Please ensure ml_backend.py is available.")
            return

        self.status.config(text=f"Applying {operation}...")

        try:
            # Prepare parameters based on operation
            params = {}
            if operation == "enhance":
                params['contrast'] = self.contrast_var.get()
                params['brightness'] = self.brightness_var.get()
            elif operation == "blur":
                params['kernel_size'] = self.kernel_var.get()
                params['sigma'] = self.sigma_var.get()
            elif operation == "sharpen":
                params['strength'] = self.strength_var.get()

            # Apply processing using backend
            processed_array = process_image(self.current_image_path, operation, **params)

            if processed_array is None:
                show_message(self.root, "error", "Processing Error", "Failed to process image")
                self.status.config(text="Processing failed")
                return

            # Convert numpy array back to PIL Image
            processed_image = Image.fromarray(processed_array)

            # Update current image
            self.current_image = processed_image
            self.display_image()

            # Update info
            self.image_info.config(
                text=f"File: {os.path.basename(self.current_image_path)} (PROCESSED) | "
                     f"Size: {self.current_image.size[0]}x{self.current_image.size[1]}",
                font=Config.FONT_BODY
            )

            self.status.config(text=f"{operation.title()} applied successfully")

        except Exception as e:
            show_message(self.root, "error", "Processing Error", f"Failed to apply {operation}:\n{str(e)}")
            self.status.config(text="Processing failed")



    def apply_realtime_processing(self):
        """Apply all current processing settings in real-time"""
        if not self.current_image or not ML_BACKEND_AVAILABLE:
            return

        try:
            # Start with cropped image if available, otherwise original image
            base_image = self.cropped_image if self.cropped_image is not None else self.original_image
            current_array = np.array(base_image)

            # Always apply enhancement (contrast and brightness)
            contrast = self.contrast_var.get()
            brightness = self.brightness_var.get()
            current_array = apply_enhancement(current_array, contrast, brightness)

            # Always apply blur
            kernel = self.kernel_var.get()
            sigma = self.sigma_var.get()
            current_array = apply_blur(current_array, kernel, sigma)

            # Always apply sharpen
            strength = self.strength_var.get()
            current_array = apply_sharpen(current_array, strength)

            # Update current image
            self.current_image = Image.fromarray(current_array)
            self.display_image()

            # Update info
            self.image_info.config(
                text=f"File: {os.path.basename(self.current_image_path)} (PROCESSED) | "
                     f"Size: {self.current_image.size[0]}x{self.current_image.size[1]}",
                font=Config.FONT_BODY
            )

        except Exception as e:
            print(f"Real-time processing error: {e}")
            import traceback
            traceback.print_exc()

    def reset_to_original(self):
        """Reset to original image (clear crop and processing)"""
        if not self.original_image:
            show_message(self.root, "warning", "No Image", "No image loaded")
            return

        if show_message(self.root, "question", "Reset to Original", "Reset to original image (remove crop and processing)?"):
            self.current_image = self.original_image.copy()
            self.cropped_image = None  # Clear cropped version
            self.display_image()
            self.image_info.config(
                text=f"File: {os.path.basename(self.current_image_path)} | "
                     f"Size: {self.original_image.size[0]}x{self.original_image.size[1]} | "
                     f"Format: {self.original_image.format}",
                font=(Config.FONT_FAMILY, 12, "bold")
            )
            self.status.config(text="Reset to original image")

    def reset_processing(self):
        """Reset image to cropped version (or original if no crop)"""
        if not self.original_image:
            show_message(self.root, "warning", "No Image", "No image loaded")
            return

        if show_message(self.root, "question", "Reset Processing", "Reset image processing (keep crop if any)?"):
            # Reset to cropped image if available, otherwise to original
            if self.cropped_image is not None:
                self.current_image = self.cropped_image.copy()
                self.image_info.config(
                    text=f"File: {os.path.basename(self.current_image_path)} (CROPPED) | "
                         f"Size: {self.current_image.size[0]}x{self.current_image.size[1]}",
                    font=Config.FONT_BODY
                )
            else:
                self.current_image = self.original_image.copy()
                self.image_info.config(
                    text=f"File: {os.path.basename(self.current_image_path)} | "
                         f"Size: {self.original_image.size[0]}x{self.original_image.size[1]} | "
                         f"Format: {self.original_image.format}",
                    font=(Config.FONT_FAMILY, 12, "bold")
                )
            self.display_image()
            self.status.config(text="Image processing reset")

    def toggle_processing_panel(self):
            """Toggle the visibility of the processing controls"""
            if self.processing_expanded:
                # Collapse
                self.processing_scroll_container.pack_forget()  # 修改这里：隐藏滚动容器
                self.unfold_button.config(text="▶ Unfold")
                self.processing_expanded = False
            else:
                # Expand
                self.processing_scroll_container.pack(fill=tk.X, padx=2, pady=(0, 15)) # 修改这里：显示滚动容器
                self.unfold_button.config(text="▼ Fold")
                self.processing_expanded = True

            # Update global canvas scroll region
            self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))

    def reset_processing_defaults(self):
        """Reset all processing parameters to default values"""
        if show_message(self.root, "question", "Reset Processing Defaults",
                        "Reset all processing sliders to default values?\n"
                        "This will not change the displayed image."):
            # Reset all variables to defaults
            self.contrast_var.set(1.0)
            self.brightness_var.set(0)
            self.kernel_var.set(1)
            self.sigma_var.set(0.0)
            self.strength_var.set(0.0)

            # Entry fields will update automatically due to variable tracing
            self.status.config(text="Processing parameters reset to defaults")

    def show_processed_on_left(self):
        """Apply current processing settings and display on left"""
        if not self.original_image:
            show_message(self.root, "warning", "No Image", "Load an image first")
            return

        if not ML_BACKEND_AVAILABLE:
            show_message(self.root, "warning", "Backend Not Available",
                "Image processing requires the ML backend.\n"
                "Please ensure ml_backend.py is available.")
            return

        try:
            self.status.config(text="Applying processing...")

            # Start with original image
            current_array = np.array(self.original_image)

            # Apply all current settings
            contrast = self.contrast_var.get()
            brightness = self.brightness_var.get()
            current_array = apply_enhancement(current_array, contrast, brightness)

            kernel = self.kernel_var.get()
            sigma = self.sigma_var.get()
            current_array = apply_blur(current_array, kernel, sigma)

            strength = self.strength_var.get()
            current_array = apply_sharpen(current_array, strength)

            # Update current image and display
            self.current_image = Image.fromarray(current_array)
            self.display_image()

            # Update info
            self.image_info.config(
                text=f"File: {os.path.basename(self.current_image_path)} (PROCESSED) | "
                     f"Size: {self.current_image.size[0]}x{self.current_image.size[1]}",
                font=Config.FONT_BODY
            )

            self.status.config(text="Processing applied to displayed image")

        except Exception as e:
            show_message(self.root, "error", "Processing Error", f"Failed to apply processing:\n{str(e)}")
            self.status.config(text="Processing failed")

    def show_help(self):
        """Show help dialog"""
        show_message(self.root, "info", "Help",
            "WORKFLOW:\n"
            "1. Open Image\n"
            "2. Select Model (automatically detected from .pth files)\n"
            "3. [Optional] Crop ROI\n"
            "4. [Optional] Adjust image processing sliders (Contrast, Brightness, Blur, Sharpen)\n"
            "5. Predict Classification (features extracted automatically)\n"
            "6. [Optional] Generate GradCAM or Occlusion Map (CNN models only)\n\n"
            "MODEL SELECTION:\n"
            "• Models are automatically detected from .pth files in project folders\n"
            "• Radiomics models: Extract statistical texture, shape, and intensity features from medical images (GLCM, GLRLM, shape descriptors)\n"
            "• SIFT models: Use Scale-Invariant Feature Transform for robust keypoint detection and local feature description. SIFT features are invariant to scale, rotation, and illumination changes, making them ideal for medical image analysis where images may vary in orientation, magnification, or lighting conditions. Useful for detecting distinctive anatomical landmarks and texture patterns in histological images.\n"
            "• Pixel models: Use CNN architectures (ResNet, DenseNet, EfficientNet, etc.) for end-to-end image analysis and feature learning\n"
            "• Fusion models: Combine multiple complementary approaches (radiomics + pixel, SIFT + pixel, multi-modal fusion) for enhanced diagnostic accuracy\n\n"
            "IMAGE PROCESSING:\n"
            "• Contrast: 0.1-3.0 (default 1.0)\n"
            "• Brightness: -50 to +50 (default 0)\n"
            "• Blur: Kernel size 1-15 (default 1 = no blur)\n"
            "• Sharpen: Strength 0.0-3.0 (default 0.0 = no sharpening)\n"
            "• Real-time preview: Adjust sliders to see changes instantly\n\n"
            "TOOLBAR CONTROLS:\n"
            "• Crop: Click and drag to select region of interest (stays pressed when active)\n"
            "• Reset: Restore original image\n"
            "• GradCAM: Visualize which pixels influenced prediction (CNN only)\n"
            "• Occlusion: Show sensitivity map by hiding image regions\n\n"
            "PREDICTION RESULTS:\n"
            "• Classification: BENIGN or MALIGNANT with confidence score\n"
            "• Confidence visualization: Color-coded progress bar\n"
            "• Detailed report: Model info, processing time, probabilities\n"
            "• Clinical interpretation: Risk assessment and recommendations\n\n"
            "VISUALIZATION:\n"
            "• GradCAM: Heatmap showing influential image regions\n"
            "• Occlusion: Sensitivity analysis by measuring confidence drops\n\n"
            f"Version: {Config.VERSION}"
        )


# ==================== MAIN ====================

def main():
    """Run the application"""
    root = tk.Tk()
    app = BreastCancerClassifierGUI(root)
    
    if ML_BACKEND_AVAILABLE:
        print("\nRunning in REAL MODE with ML backend")
    else:
        print("\nRunning in DEMO MODE without ML backend")
    
    root.mainloop()


if __name__ == "__main__":
    main()
