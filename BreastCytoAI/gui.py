"""
Breast Cancer Cell Biopsy Classifier - GUI
Medical Interface with ML Backend Integration

Integration points:
1. Feature Extraction (extract_features method)
2. Prediction (predict_classification method)

Group 6

"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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
    APP_TITLE = "Breast Cancer Cell Biopsy Classifier - Group 6"
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
        predict_with_model,
        generate_gradcam,
        generate_occlusion,
        process_image
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


# ==================== CUSTOM UI COMPONENTS ====================

class ModernButton(tk.Button):
    """Custom button with hover effects"""
    
    def __init__(self, parent, **kwargs):
        hover_color = kwargs.pop('hover_color', Theme.PRIMARY_DARK)
        default_bg = kwargs.get('bg', Theme.PRIMARY)
        
        kwargs.setdefault('relief', tk.FLAT)
        kwargs.setdefault('borderwidth', 0)
        kwargs.setdefault('cursor', 'hand2')
        kwargs.setdefault('font', Config.FONT_BODY)
        kwargs.setdefault('padx', 20)
        kwargs.setdefault('pady', 10)
        
        super().__init__(parent, **kwargs)
        
        self.default_bg = default_bg
        self.hover_color = hover_color
        
        self.bind('<Enter>', lambda e: self.config(bg=self.hover_color))
        self.bind('<Leave>', lambda e: self.config(bg=self.default_bg))


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
            text="Breast Cancer Cell Classifier",
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
            hover_color=Theme.PRIMARY_LIGHT,
            font=Config.FONT_BODY,
            width=10,
            pady=5
        ).pack(side=tk.TOP, pady=(0, 12))
        
        ModernButton(
            right,
            text="Help",
            command=self.show_help,
            bg=Theme.SURFACE,
            fg=Theme.PRIMARY,  # Changed from TEXT_ON_COLOR to PRIMARY_LIGHT for better visibility
            hover_color=Theme.PRIMARY_DARK,
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

        ModernButton(
            toolbar,
            text="Crop",
            command=self.crop_image,
            bg=Theme.PRIMARY_LIGHT,
            fg=Theme.PRIMARY,
            hover_color=Theme.PRIMARY,
            font=Config.FONT_SMALL,
            padx=15,
            pady=5
        ).pack(side=tk.LEFT, padx=5)

        ModernButton(
            toolbar,
            text="Reset",
            command=self.reset_processing,
            bg=Theme.WARNING,
            fg=Theme.TEXT_PRIMARY,
            hover_color=Theme.DANGER,
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
            hover_color=Theme.ACCENT_DARK,
            font=Config.FONT_SMALL,
            padx=15,
            pady=5
        ).pack(side=tk.LEFT, padx=5)

        ModernButton(
            toolbar,
            text="Occlusion",
            command=self.show_occlusion,
            bg=Theme.WARNING,
            fg=Theme.TEXT_PRIMARY,
            hover_color=Theme.DANGER,
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
            hover_color=Theme.SECONDARY_DARK,
            font=Config.FONT_SMALL,
            padx=10,
            pady=5
        ).pack(pady=(5, 0))
    
    def populate_model_dropdown(self):
        """Populate dropdown with available .pth model files"""
        # Find all .pth files recursively
        all_models = glob.glob("**/*.pth", recursive=True)

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

        print(f"Found {len(display_names)} model files")
        return len(display_names)

    def on_model_selected(self, event):
        """Handle model selection change"""
        selected_display = self.model_path.get()
        self.current_model_path = self.model_path_map.get(selected_display, "")
        print(f"Selected model: {selected_display} -> {self.current_model_path}")

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
            hover_color=Theme.SECONDARY_DARK,
            font=Config.FONT_BUTTON,
            width=12,
            padx=30,
            pady=10  # Reduced from 12 to 10
        ).pack(pady=(0, 15))  # Reduced from 20 to 15
    
    def create_image_processing_card(self, parent):
        """Image processing card with enhancement, blur, and sharpen controls"""
        card = Card(parent)
        card.pack(fill=tk.X, pady=(0, 12))

        tk.Label(
            card,
            text="Image Processing",
            font=Config.FONT_SECTION,
            bg=Theme.SURFACE,
            fg=Theme.TEXT_PRIMARY,
            anchor=tk.W
        ).pack(fill=tk.X, padx=20, pady=(15, 10))

        # Processing options frame
        processing_frame = tk.Frame(card, bg=Theme.SURFACE)
        processing_frame.pack(fill=tk.X, padx=20, pady=(0, 15))

        # Enhancement section
        enhance_frame = tk.Frame(processing_frame, bg=Theme.SURFACE)
        enhance_frame.pack(fill=tk.X, pady=(0, 10))

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
        blur_frame = tk.Frame(processing_frame, bg=Theme.SURFACE)
        blur_frame.pack(fill=tk.X, pady=(10, 10))

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
        sharpen_frame = tk.Frame(processing_frame, bg=Theme.SURFACE)
        sharpen_frame.pack(fill=tk.X, pady=(10, 10))

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
            hover_color=Theme.ACCENT_DARK,
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

        # Colorbar frame
        colorbar_frame = tk.Frame(results, bg=Theme.SURFACE, height=80)
        colorbar_frame.pack(fill=tk.X, padx=12, pady=(0, 8))
        colorbar_frame.pack_propagate(False)

        self.colorbar_canvas_frame = tk.Frame(colorbar_frame, bg=Theme.SURFACE)
        self.colorbar_canvas_frame.pack(fill=tk.BOTH, expand=True)

        details_frame = tk.Frame(results, bg=Theme.SURFACE)
        details_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        # Text scrollbar
        text_scrollbar = ttk.Scrollbar(details_frame)
        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.details_text = tk.Text(
            details_frame,
            height=4,
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
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
    
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
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        self.crop_mode = not self.crop_mode
        self.canvas.config(cursor="cross" if self.crop_mode else "")
        self.status.config(text="Crop mode: " + ("enabled" if self.crop_mode else "disabled"))
        
        if self.crop_mode:
            messagebox.showinfo("Crop Mode",
                "Click and drag to select area.\n"
                "Release to confirm crop.\n"
                "Use Reset to restore original."
            )
    
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
            messagebox.showwarning("Invalid", "Selection too small (minimum 50x50 pixels)")
            if self.crop_rect:
                self.canvas.delete(self.crop_rect)
            self.crop_start = None
            return
        
        if messagebox.askyesno("Apply Crop", "Apply this crop?"):
            self.apply_crop(self.crop_start, end)
        
        if self.crop_rect:
            self.canvas.delete(self.crop_rect)
        self.crop_start = None
        self.crop_mode = False
        self.canvas.config(cursor="")
    
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
            self.display_image()

            self.image_info.config(
                text=f"File: {os.path.basename(self.current_image_path)} (CROPPED) | "
                     f"Size: {self.current_image.size[0]}x{self.current_image.size[1]}",
                font=Config.FONT_BODY
            )
            self.status.config(text="Image cropped and padded to square")
        except Exception as e:
            messagebox.showerror("Error", f"Crop failed:\n{str(e)}")
    
    def reset_image(self):
        """Reset to original image"""
        if not self.original_image:
            messagebox.showwarning("No Image", "No image loaded")
            return
        
        if messagebox.askyesno("Reset", "Reset to original?"):
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
            messagebox.showwarning("No Image", "Load an image first")
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
                messagebox.showinfo("Info",
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
                
                messagebox.showinfo("Success",
                    f"Features Extracted Successfully!\n\n"
                    f"Number of features: {len(features)}\n"
                    f"Method: {method.title()}"
                )
                self.status.config(text=f"{len(features)} features extracted successfully")
            else:
                messagebox.showinfo("Demo Mode",
                    "Feature extraction simulated\n"
                    "Create ml_backend.py for real extraction"
                )
                self.status.config(text="Demo: Features extracted")
        except Exception as e:
            messagebox.showerror("Error", f"Extraction failed:\n{str(e)}")
            self.status.config(text="Extraction failed")
    
    def get_model_type_from_path(self, model_path):
        """Determine model type from model path"""
        model_name = os.path.basename(model_path).lower()
        folder_name = os.path.basename(os.path.dirname(model_path)).lower()

        # Check for radiomics models
        if "radiomics" in folder_name or "radiomics" in model_name:
            return "radiomics"
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
            messagebox.showerror("Feature Extraction Error", f"Failed to extract features:\n{str(e)}")
            return None, None

    def predict_classification(self):
        """Predict using ML backend with auto feature extraction"""
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Load an image first")
            return

        if not self.current_model_path or not os.path.exists(self.current_model_path):
            messagebox.showerror("No Model", "Please select a valid model file")
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
                if model_type in ["radiomics", "fusion"] or ("mlp" in self.current_model_path.lower()):
                    features, _ = self.auto_extract_features(model_type, img_path)
                    if features is not None:
                        feature_count = len(features)
                    if features is None and model_type in ["radiomics", "fusion"]:
                        return  # Error already shown

                print(f"\n⭐ Calling prediction with {model_type} model")
                pred, conf, details = predict_with_model(
                    self.current_model_path,
                    features=features,
                    img_path=img_path if model_type == "pixel" and "mlp" not in self.current_model_path.lower() else None,
                    model_type=model_type
                )

                processing_time = time.time() - start_time

                if pred == "ERROR":
                    messagebox.showerror("Prediction Error", details.get('error', 'Unknown error'))
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
            messagebox.showerror("Error", f"Prediction failed:\n{str(e)}")
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
            messagebox.showwarning("No Image", "Load an image first")
            return

        if not self.current_model_path or not os.path.exists(self.current_model_path):
            messagebox.showerror("No Model", "Please select a valid model file")
            return

        model_type = self.get_model_type_from_path(self.current_model_path)

        # GradCAM only works with pixel CNN models
        if model_type != "pixel" or "mlp" in self.current_model_path.lower():
            messagebox.showwarning("GradCAM Not Available",
                "GradCAM is only available for CNN models (pixel-only).\n"
                "Please select a CNN model like ResNet50_PT_FT.")
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
                messagebox.showinfo("Demo Mode", "GradCAM not available in demo mode")
        except Exception as e:
            messagebox.showerror("Error", f"GradCAM generation failed:\n{str(e)}")
            import traceback
            traceback.print_exc()
            self.status.config(text="GradCAM failed")

    def show_occlusion(self):
        """Show Occlusion Sensitivity visualization"""
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Load an image first")
            return

        if not self.current_model_path or not os.path.exists(self.current_model_path):
            messagebox.showerror("No Model", "Please select a valid model file")
            return

        model_type = self.get_model_type_from_path(self.current_model_path)

        # Occlusion works with pixel CNN models
        if model_type != "pixel" or "mlp" in self.current_model_path.lower():
            messagebox.showwarning("Occlusion Not Available",
                "Occlusion sensitivity is available for CNN models (pixel-only).\n"
                "Please select a CNN model like ResNet50_PT_FT.")
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
                messagebox.showinfo("Demo Mode", "Occlusion not available in demo mode")
        except Exception as e:
            messagebox.showerror("Error", f"Occlusion generation failed:\n{str(e)}")
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
        except ImportError:
            messagebox.showerror("Error", "matplotlib is required for GradCAM visualization")
            return

        # Create new window
        gradcam_window = tk.Toplevel(self.root)
        gradcam_window.title("GradCAM Visualization")
        gradcam_window.geometry("800x600")

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Original image
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original Image", fontsize=12)
        axes[0].axis("off")

        # GradCAM overlay
        cam_resized = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = 0.5 * heatmap + 0.5 * (img_rgb / 255.0)
        overlay = np.clip(overlay, 0, 1)

        axes[1].imshow(overlay)
        axes[1].set_title("GradCAM Heatmap", fontsize=12)
        axes[1].axis("off")

        # Add prediction info
        pred = pred_info['prediction']
        conf = pred_info['confidence']
        color = Theme.SUCCESS if pred == "BENIGN" else Theme.DANGER
        fig.suptitle(
            f"Prediction: {pred} | Confidence: {conf*100:.1f}%",
            fontsize=14,
            weight='bold',
            color=color
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
            hover_color=Theme.PRIMARY_DARK,
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
        except ImportError:
            messagebox.showerror("Error", "matplotlib is required for Occlusion visualization")
            return

        # Create new window
        occlusion_window = tk.Toplevel(self.root)
        occlusion_window.title("Occlusion Sensitivity Map")
        occlusion_window.geometry("800x600")

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Original image
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original Image", fontsize=12)
        axes[0].axis("off")

        # Occlusion overlay
        occ_resized = cv2.resize(occlusion_map, (img_rgb.shape[1], img_rgb.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * occ_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = 0.5 * heatmap + 0.5 * (img_rgb / 255.0)
        overlay = np.clip(overlay, 0, 1)

        axes[1].imshow(overlay)
        axes[1].set_title("Occlusion Sensitivity", fontsize=12)
        axes[1].axis("off")

        # Add prediction info
        pred = pred_info['prediction']
        conf = pred_info['confidence']
        color = Theme.SUCCESS if pred == "BENIGN" else Theme.DANGER
        fig.suptitle(
            f"Prediction: {pred} | Confidence: {conf*100:.1f}%",
            fontsize=14,
            weight='bold',
            color=color
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
            hover_color=Theme.PRIMARY_DARK,
            font=Config.FONT_BODY,
            padx=20,
            pady=5
        ).pack()

    def apply_image_processing(self, operation):
        """Apply image processing operation"""
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Load an image first")
            return

        if not ML_BACKEND_AVAILABLE:
            messagebox.showwarning("Backend Not Available",
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
                messagebox.showerror("Processing Error", "Failed to process image")
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
            messagebox.showerror("Processing Error", f"Failed to apply {operation}:\n{str(e)}")
            self.status.config(text="Processing failed")



    def apply_realtime_processing(self):
        """Apply all current processing settings in real-time"""
        if not self.original_image or not ML_BACKEND_AVAILABLE:
            return

        try:
            # Start with original image
            current_array = np.array(self.original_image)

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

    def reset_processing(self):
        """Reset image to original (before any processing)"""
        if not self.original_image:
            messagebox.showwarning("No Image", "No image loaded")
            return

        if messagebox.askyesno("Reset Processing", "Reset to original image (remove all processing)?"):
            self.current_image = self.original_image.copy()
            self.display_image()
            self.image_info.config(
                text=f"File: {os.path.basename(self.current_image_path)} | "
                     f"Size: {self.original_image.size[0]}x{self.original_image.size[1]} | "
                     f"Format: {self.original_image.format}",
                font=(Config.FONT_FAMILY, 12, "bold")
            )
            self.status.config(text="Image reset to original")

    def show_help(self):
        """Show help dialog"""
        messagebox.showinfo("Help",
            "WORKFLOW:\n"
            "1. Select Model Type (Radiomics/Pixel/Fusion)\n"
            "2. Select Model Path (or use Auto-detect)\n"
            "3. Open Image\n"
            "4. [Optional] Adjust image processing sliders (C=Contrast, B=Brightness, Blur, Sharp=Sharpen)\n"
            "5. [Optional] Apply processing with buttons below\n"
            "6. [Optional] Crop ROI\n"
            "7. Extract features (if needed)\n"
            "8. Predict Classification\n"
            "9. [Optional] Generate GradCAM or Occlusion Map (pixel CNN only)\n\n"
            "MODEL TYPES:\n"
            "• Radiomics Only: Uses 39 radiomics features\n"
            "• Pixel Only: Uses image directly (CNN) or pixel features (MLP)\n"
            "• Fusion: Uses both image and radiomics features\n\n"
            "IMAGE PROCESSING:\n"
            "• Contrast (C): 0.1-3.0 (default 1.0)\n"
            "• Brightness (B): -50 to +50 (default 0)\n"
            "• Blur: 1-11 kernel size (default 1 = no blur)\n"
            "• Sharpen (Sharp): 0.0-2.0 strength (default 0.0 = no sharpening)\n\n"
            "TOOLBAR CONTROLS:\n"
            "• Crop: Click and drag to select region of interest\n"
            "• Reset: Restore original image (no processing)\n"
            "• GradCAM: Show which pixels influenced prediction\n"
            "• Occlusion: Show sensitivity map\n\n"
            "VISUALIZATION:\n"
            "• GradCAM: Shows which pixels influenced the prediction most\n"
            "• Occlusion: Shows importance by measuring confidence drops when parts are hidden\n\n"
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
