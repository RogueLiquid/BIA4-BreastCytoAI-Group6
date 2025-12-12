# --- Make the process DPI-aware on Windows (must be before tkinter import) ---
import platform
if platform.system() == "Windows":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-Monitor v2
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()    # Fallback
        except Exception:
            pass

import sys
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont
from PIL import Image, ImageTk, ImageOps
import sv_ttk  # Sun Valley ttk theme


class BreaKHisApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("BreaKHis Classifier — Tkinter + Sun Valley")
        self.geometry("1100x700")
        self.minsize(900, 550)

        # Apply Sun Valley theme
        sv_ttk.set_theme("dark")

        # -------- Consistent, crisp fonts (KEY FIX) --------
        # Use a safe, non-variable TrueType family and bind ttk to TkDefaultFont
        if sys.platform.startswith("win"):
            ui_family = "Segoe UI"         # NOT "Segoe UI Variable"
            mono_family = "Consolas"
        elif sys.platform == "darwin":
            ui_family = "Helvetica"
            mono_family = "Menlo"
        else:
            ui_family = "DejaVu Sans"
            mono_family = "DejaVu Sans Mono"

        tkfont.nametofont("TkDefaultFont").configure(family=ui_family, size=11)
        tkfont.nametofont("TkTextFont").configure(family=ui_family, size=11)
        tkfont.nametofont("TkFixedFont").configure(family=mono_family, size=11)

        style = ttk.Style(self)
        # Make ALL ttk widgets (including ttk.Label) use TkDefaultFont
        style.configure(".", font=tkfont.nametofont("TkDefaultFont"))
        # Optional heading styles
        style.configure("Heading.TLabel", font=(ui_family, 13, "bold"))
        style.configure("Subheading.TLabel", font=(ui_family, 12))

        # State
        self.current_image = None
        self.tk_image = None
        self.image_path = None
        self.model = None

        # UI
        self._build_menubar()
        self._build_toolbar()
        self._build_layout()

    # ----- UI builders -----
    def _build_menubar(self):
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open image…", command=self.open_image)
        file_menu.add_command(label="Open folder…", command=self.open_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(menubar, tearoff=False)
        view_menu.add_command(label="Toggle dark/light", command=self.toggle_theme)
        menubar.add_cascade(label="View", menu=view_menu)

        model_menu = tk.Menu(menubar, tearoff=False)
        model_menu.add_command(label="Load model…", command=self.load_model)
        menubar.add_cascade(label="Model", menu=model_menu)

        self.config(menu=menubar)

    def _build_toolbar(self):
        bar = ttk.Frame(self, padding=(10, 8))
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(bar, text="Open image", command=self.open_image).pack(side=tk.LEFT)
        ttk.Button(bar, text="Open folder", command=self.open_folder).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Button(bar, text="Load model", command=self.load_model).pack(side=tk.LEFT)
        ttk.Button(bar, text="Predict", command=self.predict).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Button(bar, text="Dark/Light", command=self.toggle_theme).pack(side=tk.LEFT)

        ttk.Label(bar, text="Magnification:").pack(side=tk.LEFT, padx=(16, 4))
        self.mag_var = tk.StringVar(value="Any")
        ttk.Combobox(
            bar, width=10, textvariable=self.mag_var,
            values=["Any", "40X", "100X", "200X", "400X"],
            state="readonly"
        ).pack(side=tk.LEFT)

    def _build_layout(self):
        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, padding=10)
        right = ttk.Frame(main, padding=10, width=320)

        self.canvas = tk.Canvas(left, highlightthickness=0, bd=0, background=self._bg_color())
        self.canvas.pack(fill=tk.BOTH, expand=True)

        ttk.Label(right, text="Prediction", style="Heading.TLabel").pack(anchor="w")
        self.pred_var = tk.StringVar(value="—")
        ttk.Label(right, textvariable=self.pred_var).pack(anchor="w", pady=(0, 8))

        ttk.Label(right, text="Confidence", style="Heading.TLabel").pack(anchor="w")
        self.conf_var = tk.StringVar(value="—")
        ttk.Label(right, textvariable=self.conf_var).pack(anchor="w", pady=(0, 8))

        ttk.Separator(right).pack(fill=tk.X, pady=8)
        ttk.Label(right, text="Image path", style="Subheading.TLabel").pack(anchor="w")
        self.path_txt = tk.Text(right, height=4, wrap="word")
        self.path_txt.pack(fill=tk.BOTH, expand=False)

        ttk.Separator(right).pack(fill=tk.X, pady=8)
        ttk.Label(right, text="Log", style="Subheading.TLabel").pack(anchor="w")
        self.log_txt = tk.Text(right, height=10, wrap="word")
        self.log_txt.pack(fill=tk.BOTH, expand=True)

        main.add(left, weight=3)
        main.add(right, weight=1)

        self.canvas.bind("<Configure>", lambda e: self._refresh_canvas_image())

    # ----- Actions -----
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")],
        )
        if not path:
            return
        self._load_and_show(path)

    def open_folder(self):
        folder = filedialog.askdirectory(title="Open BreaKHis folder")
        if not folder:
            return
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    self._load_and_show(os.path.join(root, f))
                    return
        messagebox.showinfo("No images", "No image files found in that folder.")

    def load_model(self):
        messagebox.showinfo("Model", "Hook this up to your classifier (PyTorch, ONNX, etc.).")
        self._log("Model loaded (placeholder).")

    def predict(self):
        if self.current_image is None:
            messagebox.showwarning("No image", "Open an image first.")
            return
        # TODO: plug in your model inference here
        self.pred_var.set("benign / malignant (demo)")
        self.conf_var.set("0.85 (demo)")
        self._log("Predicted class with demo values. Wire up your model here.")

    def toggle_theme(self):
        mode = sv_ttk.get_theme()
        sv_ttk.set_theme("light" if mode == "dark" else "dark")
        self.canvas.configure(background=self._bg_color())
        self._refresh_canvas_image()

    # ----- Helpers -----
    def _load_and_show(self, path):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Open failed", str(e))
            return
        self.image_path = path
        self.current_image = img
        self._refresh_canvas_image()
        self._set_path(path)
        self._log(f"Opened: {path}")

    # High-quality resize for crisp images
    def _refresh_canvas_image(self):
        if self.current_image is None:
            return
        c_w = max(self.canvas.winfo_width(), 1)
        c_h = max(self.canvas.winfo_height(), 1)
        img = ImageOps.contain(self.current_image, (c_w, c_h), method=Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(c_w // 2, c_h // 2, image=self.tk_image, anchor="center")

    def _set_path(self, text):
        self.path_txt.configure(state="normal")
        self.path_txt.delete("1.0", "end")
        self.path_txt.insert("1.0", text)
        self.path_txt.configure(state="disabled")

    def _log(self, msg):
        self.log_txt.insert("end", f"{msg}\n")
        self.log_txt.see("end")

    def _bg_color(self):
        return "#1c1c1c" if sv_ttk.get_theme() == "dark" else "#fafafa"


if __name__ == "__main__":
    app = BreaKHisApp()
    app.mainloop()
