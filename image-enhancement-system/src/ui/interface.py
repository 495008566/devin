"""
User interface for the Image Enhancement System.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Optional, Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.database.db_manager import DatabaseManager
from src.core.processor import ImageProcessor
from src.core.metrics import get_image_metrics
from src.utils.comparison import create_side_by_side_comparison, plot_histograms, create_before_after_slider
from src.utils.helpers import create_directory_if_not_exists, is_valid_image_file
from src.ui.ui_utils import convert_cv_to_tk, calculate_display_size

class Application:
    """Main application class for the UI."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the application."""
        self.db = db_manager
        self.processor = ImageProcessor()
        self.root = None
        self.original_image = None
        self.processed_image = None
        self.current_image_id = None
        self.current_enhancement_id = None
        self.original_tk_image = None
        self.processed_tk_image = None
        
        # UI components
        self.original_canvas = None
        self.processed_canvas = None
        self.enhancement_frame = None
        self.metrics_frame = None
        self.histogram_frame = None
        self.status_var = None
        
        # Enhancement parameters
        self.enhancement_params = {}
        
    def run(self) -> None:
        """Run the application."""
        self.root = tk.Tk()
        self.root.title("Image Enhancement System")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Setup UI components
        self._setup_ui()
        
        # Start the main loop
        self.root.mainloop()
    
    # The rest of the implementation will be added in subsequent files
    def _setup_ui(self) -> None:
        """Set up the UI components."""
        # Create main frames
        self._create_menu()
        self._create_main_frame()
        self._create_status_bar()
        
        # Set initial status
        self.status_var.set("Ready. Please load an image to begin.")
    
    def _create_menu(self) -> None:
        """Create the application menu."""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self._open_image)
        file_menu.add_command(label="Save Enhanced Image", command=self._save_processed_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Enhancement menu
        enhance_menu = tk.Menu(menubar, tearoff=0)
        enhance_menu.add_command(label="Grayscale", command=lambda: self._apply_enhancement("grayscale"))
        enhance_menu.add_command(label="Contrast Adjustment", command=lambda: self._apply_enhancement("contrast"))
        enhance_menu.add_command(label="Histogram Equalization", command=lambda: self._apply_enhancement("histogram_eq"))
        enhance_menu.add_separator()
        enhance_menu.add_command(label="Mean Filter", command=lambda: self._apply_enhancement("mean_filter"))
        enhance_menu.add_command(label="Gaussian Filter", command=lambda: self._apply_enhancement("gaussian_filter"))
        enhance_menu.add_command(label="Median Filter", command=lambda: self._apply_enhancement("median_filter"))
        enhance_menu.add_separator()
        enhance_menu.add_command(label="Edge Detection (Sobel)", command=lambda: self._apply_enhancement("sobel"))
        enhance_menu.add_command(label="Sharpen", command=lambda: self._apply_enhancement("sharpen"))
        enhance_menu.add_separator()
        enhance_menu.add_command(label="Resize/Magnify", command=lambda: self._apply_enhancement("resize"))
        enhance_menu.add_command(label="Pseudocolor", command=lambda: self._apply_enhancement("pseudocolor"))
        menubar.add_cascade(label="Enhance", menu=enhance_menu)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Show Histograms", command=self._show_histograms)
        view_menu.add_command(label="Show Metrics", command=self._show_metrics)
        view_menu.add_command(label="Create Before/After Comparison", command=self._create_comparison)
        menubar.add_cascade(label="View", menu=view_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def _create_main_frame(self) -> None:
        """Create the main application frame."""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left frame for original image
        left_frame = ttk.LabelFrame(main_frame, text="Original Image")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create right frame for processed image and controls
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Processed image frame
        processed_frame = ttk.LabelFrame(right_frame, text="Enhanced Image")
        processed_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Enhancement controls frame
        self.enhancement_frame = ttk.LabelFrame(right_frame, text="Enhancement Controls")
        self.enhancement_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
        
        # Metrics frame
        self.metrics_frame = ttk.LabelFrame(right_frame, text="Image Metrics")
        self.metrics_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
        
        # Create canvases for images
        self.original_canvas = tk.Canvas(left_frame, bg="lightgray")
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.processed_canvas = tk.Canvas(processed_frame, bg="lightgray")
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Add placeholder text
        self.original_canvas.create_text(
            self.original_canvas.winfo_reqwidth() // 2,
            self.original_canvas.winfo_reqheight() // 2,
            text="No image loaded",
            fill="darkgray",
            font=("Arial", 14)
        )
        
        self.processed_canvas.create_text(
            self.processed_canvas.winfo_reqwidth() // 2,
            self.processed_canvas.winfo_reqheight() // 2,
            text="No enhancement applied",
            fill="darkgray",
            font=("Arial", 14)
        )
        
        # Add initial metrics display
        self._update_metrics_display(None, None)
    
    def _create_status_bar(self) -> None:
        """Create the status bar."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar()
        status_label = ttk.Label(
            status_frame, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        status_label.pack(fill=tk.X)
    
    def _open_image(self) -> None:
        """Open an image file."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        if not is_valid_image_file(file_path):
            messagebox.showerror("Error", "Invalid image file format.")
            return
        
        try:
            # Load the image
            self.original_image = self.processor.load_image(file_path)
            if self.original_image is None:
                messagebox.showerror("Error", "Failed to load image.")
                return
            
            # Reset processed image
            self.processed_image = None
            
            # Get image info
            image_info = self.processor.get_image_info(self.original_image)
            
            # Store in database
            image_data = {
                'filename': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'width': image_info['width'],
                'height': image_info['height'],
                'color_space': 'BGR' if image_info['channels'] == 3 else 'Grayscale',
                'file_format': os.path.splitext(file_path)[1][1:].upper()
            }
            
            self.current_image_id = self.db.add_image(image_data)
            
            # Display the image
            self._display_original_image()
            
            # Update status
            self.status_var.set(f"Loaded image: {os.path.basename(file_path)} ({image_info['width']}x{image_info['height']})")
            
            # Clear processed image
            self.processed_canvas.delete("all")
            self.processed_canvas.create_text(
                self.processed_canvas.winfo_reqwidth() // 2,
                self.processed_canvas.winfo_reqheight() // 2,
                text="Apply an enhancement to see results",
                fill="darkgray",
                font=("Arial", 14)
            )
            
            # Update metrics
            self._update_metrics_display(self.original_image, None)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def _display_original_image(self) -> None:
        """Display the original image on the canvas."""
        if self.original_image is None:
            return
        
        # Clear canvas
        self.original_canvas.delete("all")
        
        # Calculate display size
        canvas_width = self.original_canvas.winfo_width()
        canvas_height = self.original_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not yet properly sized, use default size
            canvas_width = 400
            canvas_height = 300
        
        display_size = calculate_display_size(
            self.original_image.shape, 
            (canvas_width, canvas_height)
        )
        
        # Convert to Tkinter image
        self.original_tk_image = convert_cv_to_tk(self.original_image, display_size)
        
        # Display on canvas
        self.original_canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=self.original_tk_image,
            anchor=tk.CENTER
        )
    
    def _display_processed_image(self) -> None:
        """Display the processed image on the canvas."""
        if self.processed_image is None:
            return
        
        # Clear canvas
        self.processed_canvas.delete("all")
        
        # Calculate display size
        canvas_width = self.processed_canvas.winfo_width()
        canvas_height = self.processed_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not yet properly sized, use default size
            canvas_width = 400
            canvas_height = 300
        
        display_size = calculate_display_size(
            self.processed_image.shape, 
            (canvas_width, canvas_height)
        )
        
        # Convert to Tkinter image
        self.processed_tk_image = convert_cv_to_tk(self.processed_image, display_size)
        
        # Display on canvas
        self.processed_canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=self.processed_tk_image,
            anchor=tk.CENTER
        )
    
    def _apply_enhancement(self, enhancement_type: str) -> None:
        """
        Apply an enhancement to the original image.
        
        Args:
            enhancement_type: Type of enhancement to apply
        """
        if self.original_image is None:
            messagebox.showinfo("Info", "Please load an image first.")
            return
        
        try:
            # Clear enhancement frame
            for widget in self.enhancement_frame.winfo_children():
                widget.destroy()
            
            # Set up enhancement parameters based on type
            if enhancement_type == "grayscale":
                self._setup_grayscale_controls()
            elif enhancement_type == "contrast":
                self._setup_contrast_controls()
            elif enhancement_type == "histogram_eq":
                self._setup_histogram_eq_controls()
            elif enhancement_type == "mean_filter":
                self._setup_mean_filter_controls()
            elif enhancement_type == "gaussian_filter":
                self._setup_gaussian_filter_controls()
            elif enhancement_type == "median_filter":
                self._setup_median_filter_controls()
            elif enhancement_type == "sobel":
                self._setup_sobel_controls()
            elif enhancement_type == "sharpen":
                self._setup_sharpen_controls()
            elif enhancement_type == "resize":
                self._setup_resize_controls()
            elif enhancement_type == "pseudocolor":
                self._setup_pseudocolor_controls()
            else:
                messagebox.showerror("Error", f"Unknown enhancement type: {enhancement_type}")
                return
            
            # Update status
            self.status_var.set(f"Applied {enhancement_type} enhancement")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply enhancement: {str(e)}")
    
    def _setup_grayscale_controls(self) -> None:
        """Set up controls for grayscale conversion."""
        # Apply button
        apply_btn = ttk.Button(
            self.enhancement_frame,
            text="Apply Grayscale",
            command=self._apply_grayscale
        )
        apply_btn.pack(pady=10)
        
        # Apply immediately
        self._apply_grayscale()
    
    def _apply_grayscale(self) -> None:
        """Apply grayscale conversion."""
        if self.original_image is None:
            return
        
        # Process the image
        self.processed_image = self.processor.convert_to_grayscale(self.original_image)
        
        # Display the processed image
        self._display_processed_image()
        
        # Update metrics
        self._update_metrics_display(self.original_image, self.processed_image)
        
        # Store in database
        if self.current_image_id is not None:
            # Create output filename
            output_filename = f"grayscale_{self.current_image_id}.jpg"
            output_path = os.path.join("data", "output", output_filename)
            
            # Save the processed image
            create_directory_if_not_exists(os.path.dirname(output_path))
            self.processor.save_image(self.processed_image, output_path)
            
            # Store enhancement in database
            enhancement_data = {
                'image_id': self.current_image_id,
                'enhancement_type': 'grayscale',
                'parameters': {},
                'output_filename': output_filename
            }
            
            self.current_enhancement_id = self.db.add_enhancement(enhancement_data)
            
            # Store metrics
            if self.current_enhancement_id is not None:
                metrics = get_image_metrics(self.original_image, self.processed_image)
                metrics_data = [
                    {
                        'enhancement_id': self.current_enhancement_id,
                        'metric_name': name,
                        'metric_value': value
                    }
                    for name, value in metrics.items()
                ]
                
                self.db.add_metrics(metrics_data)
    
    def _setup_contrast_controls(self) -> None:
        """Set up controls for contrast adjustment."""
        # Parameters frame
        params_frame = ttk.Frame(self.enhancement_frame)
        params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Alpha (contrast) control
        ttk.Label(params_frame, text="Contrast (alpha):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        alpha_var = tk.DoubleVar(value=1.5)
        alpha_scale = ttk.Scale(
            params_frame,
            from_=0.5,
            to=3.0,
            variable=alpha_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        alpha_scale.grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        alpha_label = ttk.Label(params_frame, text="1.5")
        alpha_label.grid(row=0, column=2, padx=5, pady=5)
        
        # Beta (brightness) control
        ttk.Label(params_frame, text="Brightness (beta):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        beta_var = tk.IntVar(value=0)
        beta_scale = ttk.Scale(
            params_frame,
            from_=-50,
            to=50,
            variable=beta_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        beta_scale.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        beta_label = ttk.Label(params_frame, text="0")
        beta_label.grid(row=1, column=2, padx=5, pady=5)
        
        # Update labels when sliders change
        def update_alpha_label(*args):
            alpha_label.config(text=f"{alpha_var.get():.1f}")
        
        def update_beta_label(*args):
            beta_label.config(text=str(beta_var.get()))
        
        alpha_var.trace_add("write", update_alpha_label)
        beta_var.trace_add("write", update_beta_label)
        
        # Apply button
        apply_btn = ttk.Button(
            self.enhancement_frame,
            text="Apply Contrast Adjustment",
            command=lambda: self._apply_contrast(alpha_var.get(), beta_var.get())
        )
        apply_btn.pack(pady=10)
        
        # Apply with default values
        self._apply_contrast(alpha_var.get(), beta_var.get())
    
    def _apply_contrast(self, alpha: float, beta: int) -> None:
        """
        Apply contrast adjustment.
        
        Args:
            alpha: Contrast control (1.0-3.0)
            beta: Brightness control (-50-50)
        """
        if self.original_image is None:
            return
        
        # Process the image
        self.processed_image = self.processor.adjust_contrast(self.original_image, alpha, beta)
        
        # Display the processed image
        self._display_processed_image()
        
        # Update metrics
        self._update_metrics_display(self.original_image, self.processed_image)
        
        # Store in database
        if self.current_image_id is not None:
            # Create output filename
            output_filename = f"contrast_{self.current_image_id}_a{alpha:.1f}_b{beta}.jpg"
            output_path = os.path.join("data", "output", output_filename)
            
            # Save the processed image
            create_directory_if_not_exists(os.path.dirname(output_path))
            self.processor.save_image(self.processed_image, output_path)
            
            # Store enhancement in database
            enhancement_data = {
                'image_id': self.current_image_id,
                'enhancement_type': 'contrast',
                'parameters': {'alpha': alpha, 'beta': beta},
                'output_filename': output_filename
            }
            
            self.current_enhancement_id = self.db.add_enhancement(enhancement_data)
            
            # Store metrics
            if self.current_enhancement_id is not None:
                metrics = get_image_metrics(self.original_image, self.processed_image)
                metrics_data = [
                    {
                        'enhancement_id': self.current_enhancement_id,
                        'metric_name': name,
                        'metric_value': value
                    }
                    for name, value in metrics.items()
                ]
                
                self.db.add_metrics(metrics_data)
    
    def _setup_histogram_eq_controls(self) -> None:
        """Set up controls for histogram equalization."""
        # Radio buttons for method
        method_var = tk.StringVar(value="standard")
        
        ttk.Radiobutton(
            self.enhancement_frame,
            text="Standard Histogram Equalization",
            variable=method_var,
            value="standard"
        ).pack(anchor=tk.W, padx=10, pady=5)
        
        ttk.Radiobutton(
            self.enhancement_frame,
            text="Adaptive Histogram Equalization (CLAHE)",
            variable=method_var,
            value="adaptive"
        ).pack(anchor=tk.W, padx=10, pady=5)
        
        # CLAHE parameters (only shown when adaptive is selected)
        clahe_frame = ttk.Frame(self.enhancement_frame)
        
        ttk.Label(clahe_frame, text="Clip Limit:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        clip_var = tk.DoubleVar(value=2.0)
        clip_scale = ttk.Scale(
            clahe_frame,
            from_=1.0,
            to=5.0,
            variable=clip_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        clip_scale.grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        clip_label = ttk.Label(clahe_frame, text="2.0")
        clip_label.grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(clahe_frame, text="Tile Size:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        tile_var = tk.IntVar(value=8)
        tile_scale = ttk.Scale(
            clahe_frame,
            from_=2,
            to=16,
            variable=tile_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        tile_scale.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        tile_label = ttk.Label(clahe_frame, text="8x8")
        tile_label.grid(row=1, column=2, padx=5, pady=5)
        
        # Update labels when sliders change
        def update_clip_label(*args):
            clip_label.config(text=f"{clip_var.get():.1f}")
        
        def update_tile_label(*args):
            tile_label.config(text=f"{tile_var.get()}x{tile_var.get()}")
        
        clip_var.trace_add("write", update_clip_label)
        tile_var.trace_add("write", update_tile_label)
        
        # Show/hide CLAHE parameters based on method selection
        def toggle_clahe_params(*args):
            if method_var.get() == "adaptive":
                clahe_frame.pack(fill=tk.X, padx=20, pady=5)
            else:
                clahe_frame.pack_forget()
        
        method_var.trace_add("write", toggle_clahe_params)
        toggle_clahe_params()  # Initial state
        
        # Apply button
        apply_btn = ttk.Button(
            self.enhancement_frame,
            text="Apply Histogram Equalization",
            command=lambda: self._apply_histogram_eq(
                method_var.get(),
                clip_var.get(),
                tile_var.get()
            )
        )
        apply_btn.pack(pady=10)
        
        # Apply with default values
        self._apply_histogram_eq(method_var.get(), clip_var.get(), tile_var.get())
    
    def _apply_histogram_eq(self, method: str, clip_limit: float = 2.0, tile_size: int = 8) -> None:
        """
        Apply histogram equalization.
        
        Args:
            method: Equalization method ('standard' or 'adaptive')
            clip_limit: CLAHE clip limit
            tile_size: CLAHE tile size
        """
        if self.original_image is None:
            return
        
        # Process the image
        if method == "standard":
            self.processed_image = self.processor.histogram_equalization(self.original_image)
            parameters = {'method': 'standard'}
        else:  # adaptive
            self.processed_image = self.processor.adaptive_histogram_equalization(
                self.original_image,
                clip_limit=clip_limit,
                tile_grid_size=(tile_size, tile_size)
            )
            parameters = {
                'method': 'adaptive',
                'clip_limit': clip_limit,
                'tile_size': tile_size
            }
        
        # Display the processed image
        self._display_processed_image()
        
        # Update metrics
        self._update_metrics_display(self.original_image, self.processed_image)
        
        # Store in database
        if self.current_image_id is not None:
            # Create output filename
            if method == "standard":
                output_filename = f"histeq_{self.current_image_id}.jpg"
            else:
                output_filename = f"clahe_{self.current_image_id}_c{clip_limit:.1f}_t{tile_size}.jpg"
            
            output_path = os.path.join("data", "output", output_filename)
            
            # Save the processed image
            create_directory_if_not_exists(os.path.dirname(output_path))
            self.processor.save_image(self.processed_image, output_path)
            
            # Store enhancement in database
            enhancement_data = {
                'image_id': self.current_image_id,
                'enhancement_type': 'histogram_equalization',
                'parameters': parameters,
                'output_filename': output_filename
            }
            
            self.current_enhancement_id = self.db.add_enhancement(enhancement_data)
            
            # Store metrics
            if self.current_enhancement_id is not None:
                metrics = get_image_metrics(self.original_image, self.processed_image)
                metrics_data = [
                    {
                        'enhancement_id': self.current_enhancement_id,
                        'metric_name': name,
                        'metric_value': value
                    }
                    for name, value in metrics.items()
                ]
                
                self.db.add_metrics(metrics_data)
    def _setup_mean_filter_controls(self) -> None:
        """Set up controls for mean filter."""
        # Parameters frame
        params_frame = ttk.Frame(self.enhancement_frame)
        params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Kernel size control
        ttk.Label(params_frame, text="Kernel Size:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        kernel_var = tk.IntVar(value=3)
        kernel_values = [3, 5, 7, 9, 11]
        kernel_combo = ttk.Combobox(
            params_frame,
            textvariable=kernel_var,
            values=kernel_values,
            state="readonly",
            width=5
        )
        kernel_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Apply button
        apply_btn = ttk.Button(
            self.enhancement_frame,
            text="Apply Mean Filter",
            command=lambda: self._apply_mean_filter(kernel_var.get())
        )
        apply_btn.pack(pady=10)
        
        # Apply with default values
        self._apply_mean_filter(kernel_var.get())
    
    def _apply_mean_filter(self, kernel_size: int) -> None:
        """
        Apply mean filter.
        
        Args:
            kernel_size: Size of the filter kernel
        """
        if self.original_image is None:
            return
        
        # Process the image
        self.processed_image = self.processor.apply_mean_filter(self.original_image, kernel_size)
        
        # Display the processed image
        self._display_processed_image()
        
        # Update metrics
        self._update_metrics_display(self.original_image, self.processed_image)
        
        # Store in database
        if self.current_image_id is not None:
            # Create output filename
            output_filename = f"mean_filter_{self.current_image_id}_k{kernel_size}.jpg"
            output_path = os.path.join("data", "output", output_filename)
            
            # Save the processed image
            create_directory_if_not_exists(os.path.dirname(output_path))
            self.processor.save_image(self.processed_image, output_path)
            
            # Store enhancement in database
            enhancement_data = {
                'image_id': self.current_image_id,
                'enhancement_type': 'mean_filter',
                'parameters': {'kernel_size': kernel_size},
                'output_filename': output_filename
            }
            
            self.current_enhancement_id = self.db.add_enhancement(enhancement_data)
            
            # Store metrics
            if self.current_enhancement_id is not None:
                metrics = get_image_metrics(self.original_image, self.processed_image)
                metrics_data = [
                    {
                        'enhancement_id': self.current_enhancement_id,
                        'metric_name': name,
                        'metric_value': value
                    }
                    for name, value in metrics.items()
                ]
                
                self.db.add_metrics(metrics_data)
    
    def _setup_gaussian_filter_controls(self) -> None:
        """Set up controls for Gaussian filter."""
        # Parameters frame
        params_frame = ttk.Frame(self.enhancement_frame)
        params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Kernel size control
        ttk.Label(params_frame, text="Kernel Size:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        kernel_var = tk.IntVar(value=3)
        kernel_values = [3, 5, 7, 9, 11]
        kernel_combo = ttk.Combobox(
            params_frame,
            textvariable=kernel_var,
            values=kernel_values,
            state="readonly",
            width=5
        )
        kernel_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Sigma control
        ttk.Label(params_frame, text="Sigma:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        sigma_var = tk.DoubleVar(value=1.0)
        sigma_scale = ttk.Scale(
            params_frame,
            from_=0.1,
            to=5.0,
            variable=sigma_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        sigma_scale.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        sigma_label = ttk.Label(params_frame, text="1.0")
        sigma_label.grid(row=1, column=2, padx=5, pady=5)
        
        # Update sigma label when slider changes
        def update_sigma_label(*args):
            sigma_label.config(text=f"{sigma_var.get():.1f}")
        
        sigma_var.trace_add("write", update_sigma_label)
        
        # Apply button
        apply_btn = ttk.Button(
            self.enhancement_frame,
            text="Apply Gaussian Filter",
            command=lambda: self._apply_gaussian_filter(kernel_var.get(), sigma_var.get())
        )
        apply_btn.pack(pady=10)
        
        # Apply with default values
        self._apply_gaussian_filter(kernel_var.get(), sigma_var.get())
    
    def _apply_gaussian_filter(self, kernel_size: int, sigma: float) -> None:
        """
        Apply Gaussian filter.
        
        Args:
            kernel_size: Size of the filter kernel
            sigma: Standard deviation of the Gaussian kernel
        """
        if self.original_image is None:
            return
        
        # Process the image
        self.processed_image = self.processor.apply_gaussian_filter(self.original_image, kernel_size, sigma)
        
        # Display the processed image
        self._display_processed_image()
        
        # Update metrics
        self._update_metrics_display(self.original_image, self.processed_image)
        
        # Store in database
        if self.current_image_id is not None:
            # Create output filename
            output_filename = f"gaussian_filter_{self.current_image_id}_k{kernel_size}_s{sigma:.1f}.jpg"
            output_path = os.path.join("data", "output", output_filename)
            
            # Save the processed image
            create_directory_if_not_exists(os.path.dirname(output_path))
            self.processor.save_image(self.processed_image, output_path)
            
            # Store enhancement in database
            enhancement_data = {
                'image_id': self.current_image_id,
                'enhancement_type': 'gaussian_filter',
                'parameters': {'kernel_size': kernel_size, 'sigma': sigma},
                'output_filename': output_filename
            }
            
            self.current_enhancement_id = self.db.add_enhancement(enhancement_data)
            
            # Store metrics
            if self.current_enhancement_id is not None:
                metrics = get_image_metrics(self.original_image, self.processed_image)
                metrics_data = [
                    {
                        'enhancement_id': self.current_enhancement_id,
                        'metric_name': name,
                        'metric_value': value
                    }
                    for name, value in metrics.items()
                ]
                
                self.db.add_metrics(metrics_data)
    
    def _setup_median_filter_controls(self) -> None:
        """Set up controls for median filter."""
        # Parameters frame
        params_frame = ttk.Frame(self.enhancement_frame)
        params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Kernel size control
        ttk.Label(params_frame, text="Kernel Size:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        kernel_var = tk.IntVar(value=3)
        kernel_values = [3, 5, 7, 9, 11]
        kernel_combo = ttk.Combobox(
            params_frame,
            textvariable=kernel_var,
            values=kernel_values,
            state="readonly",
            width=5
        )
        kernel_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Apply button
        apply_btn = ttk.Button(
            self.enhancement_frame,
            text="Apply Median Filter",
            command=lambda: self._apply_median_filter(kernel_var.get())
        )
        apply_btn.pack(pady=10)
        
        # Apply with default values
        self._apply_median_filter(kernel_var.get())
    
    def _apply_median_filter(self, kernel_size: int) -> None:
        """
        Apply median filter.
        
        Args:
            kernel_size: Size of the filter kernel
        """
        if self.original_image is None:
            return
        
        # Process the image
        self.processed_image = self.processor.apply_median_filter(self.original_image, kernel_size)
        
        # Display the processed image
        self._display_processed_image()
        
        # Update metrics
        self._update_metrics_display(self.original_image, self.processed_image)
        
        # Store in database
        if self.current_image_id is not None:
            # Create output filename
            output_filename = f"median_filter_{self.current_image_id}_k{kernel_size}.jpg"
            output_path = os.path.join("data", "output", output_filename)
            
            # Save the processed image
            create_directory_if_not_exists(os.path.dirname(output_path))
            self.processor.save_image(self.processed_image, output_path)
            
            # Store enhancement in database
            enhancement_data = {
                'image_id': self.current_image_id,
                'enhancement_type': 'median_filter',
                'parameters': {'kernel_size': kernel_size},
                'output_filename': output_filename
            }
            
            self.current_enhancement_id = self.db.add_enhancement(enhancement_data)
            
            # Store metrics
            if self.current_enhancement_id is not None:
                metrics = get_image_metrics(self.original_image, self.processed_image)
                metrics_data = [
                    {
                        'enhancement_id': self.current_enhancement_id,
                        'metric_name': name,
                        'metric_value': value
                    }
                    for name, value in metrics.items()
                ]
                
                self.db.add_metrics(metrics_data)
    
    def _setup_sobel_controls(self) -> None:
        """Set up controls for Sobel edge detection."""
        # Parameters frame
        params_frame = ttk.Frame(self.enhancement_frame)
        params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Direction control
        ttk.Label(params_frame, text="Direction:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        direction_var = tk.StringVar(value="both")
        
        ttk.Radiobutton(
            params_frame,
            text="Horizontal (dx=1, dy=0)",
            variable=direction_var,
            value="horizontal"
        ).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Radiobutton(
            params_frame,
            text="Vertical (dx=0, dy=1)",
            variable=direction_var,
            value="vertical"
        ).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Radiobutton(
            params_frame,
            text="Both (dx=1, dy=1)",
            variable=direction_var,
            value="both"
        ).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Kernel size control
        ttk.Label(params_frame, text="Kernel Size:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        kernel_var = tk.IntVar(value=3)
        kernel_values = [3, 5, 7]
        kernel_combo = ttk.Combobox(
            params_frame,
            textvariable=kernel_var,
            values=kernel_values,
            state="readonly",
            width=5
        )
        kernel_combo.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Apply button
        apply_btn = ttk.Button(
            self.enhancement_frame,
            text="Apply Sobel Edge Detection",
            command=lambda: self._apply_sobel(direction_var.get(), kernel_var.get())
        )
        apply_btn.pack(pady=10)
        
        # Apply with default values
        self._apply_sobel(direction_var.get(), kernel_var.get())
    
    def _apply_sobel(self, direction: str, kernel_size: int) -> None:
        """
        Apply Sobel edge detection.
        
        Args:
            direction: Direction of the Sobel operator ('horizontal', 'vertical', or 'both')
            kernel_size: Size of the Sobel kernel
        """
        if self.original_image is None:
            return
        
        # Set dx and dy based on direction
        if direction == "horizontal":
            dx, dy = 1, 0
        elif direction == "vertical":
            dx, dy = 0, 1
        else:  # both
            dx, dy = 1, 1
        
        # Process the image
        self.processed_image = self.processor.detect_edges_sobel(self.original_image, dx, dy, kernel_size)
        
        # Display the processed image
        self._display_processed_image()
        
        # Update metrics
        self._update_metrics_display(self.original_image, self.processed_image)
        
        # Store in database
        if self.current_image_id is not None:
            # Create output filename
            output_filename = f"sobel_{self.current_image_id}_{direction}_k{kernel_size}.jpg"
            output_path = os.path.join("data", "output", output_filename)
            
            # Save the processed image
            create_directory_if_not_exists(os.path.dirname(output_path))
            self.processor.save_image(self.processed_image, output_path)
            
            # Store enhancement in database
            enhancement_data = {
                'image_id': self.current_image_id,
                'enhancement_type': 'sobel',
                'parameters': {'direction': direction, 'kernel_size': kernel_size, 'dx': dx, 'dy': dy},
                'output_filename': output_filename
            }
            
            self.current_enhancement_id = self.db.add_enhancement(enhancement_data)
            
            # Store metrics
            if self.current_enhancement_id is not None:
                metrics = get_image_metrics(self.original_image, self.processed_image)
                metrics_data = [
                    {
                        'enhancement_id': self.current_enhancement_id,
                        'metric_name': name,
                        'metric_value': value
                    }
                    for name, value in metrics.items()
                ]
                
                self.db.add_metrics(metrics_data)
    
    def _setup_sharpen_controls(self) -> None:
        """Set up controls for image sharpening."""
        # Parameters frame
        params_frame = ttk.Frame(self.enhancement_frame)
        params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Method control
        ttk.Label(params_frame, text="Method:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        method_var = tk.StringVar(value="unsharp")
        
        ttk.Radiobutton(
            params_frame,
            text="Unsharp Masking",
            variable=method_var,
            value="unsharp"
        ).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Radiobutton(
            params_frame,
            text="Laplacian",
            variable=method_var,
            value="laplacian"
        ).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Amount control (for unsharp masking)
        ttk.Label(params_frame, text="Amount:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        amount_var = tk.DoubleVar(value=1.0)
        amount_scale = ttk.Scale(
            params_frame,
            from_=0.1,
            to=3.0,
            variable=amount_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        amount_scale.grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        amount_label = ttk.Label(params_frame, text="1.0")
        amount_label.grid(row=2, column=2, padx=5, pady=5)
        
        # Kernel size control (for Laplacian)
        ttk.Label(params_frame, text="Kernel Size:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        kernel_var = tk.IntVar(value=3)
        kernel_values = [3, 5, 7]
        kernel_combo = ttk.Combobox(
            params_frame,
            textvariable=kernel_var,
            values=kernel_values,
            state="readonly",
            width=5
        )
        kernel_combo.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Update amount label when slider changes
        def update_amount_label(*args):
            amount_label.config(text=f"{amount_var.get():.1f}")
        
        amount_var.trace_add("write", update_amount_label)
        
        # Apply button
        apply_btn = ttk.Button(
            self.enhancement_frame,
            text="Apply Sharpening",
            command=lambda: self._apply_sharpen(
                method_var.get(),
                amount_var.get(),
                kernel_var.get()
            )
        )
        apply_btn.pack(pady=10)
        
        # Apply with default values
        self._apply_sharpen(method_var.get(), amount_var.get(), kernel_var.get())
    
    def _apply_sharpen(self, method: str, amount: float, kernel_size: int) -> None:
        """
        Apply image sharpening.
        
        Args:
            method: Sharpening method ('unsharp' or 'laplacian')
            amount: Sharpening amount (for unsharp masking)
            kernel_size: Kernel size (for Laplacian)
        """
        if self.original_image is None:
            return
        
        # Process the image
        if method == "unsharp":
            self.processed_image = self.processor.sharpen_image(self.original_image, amount)
            parameters = {'method': 'unsharp', 'amount': amount}
        else:  # laplacian
            self.processed_image = self.processor.laplacian_sharpening(self.original_image, kernel_size)
            parameters = {'method': 'laplacian', 'kernel_size': kernel_size}
        
        # Display the processed image
        self._display_processed_image()
        
        # Update metrics
        self._update_metrics_display(self.original_image, self.processed_image)
        
        # Store in database
        if self.current_image_id is not None:
            # Create output filename
            if method == "unsharp":
                output_filename = f"sharpen_unsharp_{self.current_image_id}_a{amount:.1f}.jpg"
            else:
                output_filename = f"sharpen_laplacian_{self.current_image_id}_k{kernel_size}.jpg"
            
            output_path = os.path.join("data", "output", output_filename)
            
            # Save the processed image
            create_directory_if_not_exists(os.path.dirname(output_path))
            self.processor.save_image(self.processed_image, output_path)
            
            # Store enhancement in database
            enhancement_data = {
                'image_id': self.current_image_id,
                'enhancement_type': 'sharpen',
                'parameters': parameters,
                'output_filename': output_filename
            }
            
            self.current_enhancement_id = self.db.add_enhancement(enhancement_data)
            
            # Store metrics
            if self.current_enhancement_id is not None:
                metrics = get_image_metrics(self.original_image, self.processed_image)
                metrics_data = [
                    {
                        'enhancement_id': self.current_enhancement_id,
                        'metric_name': name,
                        'metric_value': value
                    }
                    for name, value in metrics.items()
                ]
                
                self.db.add_metrics(metrics_data)
    def _setup_resize_controls(self) -> None:
        """Set up controls for image resizing/magnification."""
        # Parameters frame
        params_frame = ttk.Frame(self.enhancement_frame)
        params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Scale factor control
        ttk.Label(params_frame, text="Scale Factor:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        scale_var = tk.DoubleVar(value=2.0)
        scale_scale = ttk.Scale(
            params_frame,
            from_=0.5,
            to=4.0,
            variable=scale_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        scale_scale.grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        scale_label = ttk.Label(params_frame, text="2.0")
        scale_label.grid(row=0, column=2, padx=5, pady=5)
        
        # Interpolation method control
        ttk.Label(params_frame, text="Interpolation:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        interp_var = tk.StringVar(value="linear")
        
        interp_methods = [
            ("Nearest Neighbor", "nearest"),
            ("Linear", "linear"),
            ("Cubic", "cubic"),
            ("Lanczos", "lanczos")
        ]
        
        for i, (text, value) in enumerate(interp_methods):
            ttk.Radiobutton(
                params_frame,
                text=text,
                variable=interp_var,
                value=value
            ).grid(row=i+1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Update scale label when slider changes
        def update_scale_label(*args):
            scale_label.config(text=f"{scale_var.get():.1f}")
        
        scale_var.trace_add("write", update_scale_label)
        
        # Apply button
        apply_btn = ttk.Button(
            self.enhancement_frame,
            text="Apply Resize/Magnify",
            command=lambda: self._apply_resize(scale_var.get(), interp_var.get())
        )
        apply_btn.pack(pady=10)
        
        # Apply with default values
        self._apply_resize(scale_var.get(), interp_var.get())
    
    def _apply_resize(self, scale_factor: float, interpolation: str) -> None:
        """
        Apply image resizing/magnification.
        
        Args:
            scale_factor: Scale factor for resizing
            interpolation: Interpolation method
        """
        if self.original_image is None:
            return
        
        # Map interpolation method to OpenCV constant
        interp_map = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4
        }
        
        # Process the image
        self.processed_image = self.processor.resize_image(
            self.original_image,
            scale_factor,
            interp_map.get(interpolation, cv2.INTER_LINEAR)
        )
        
        # Display the processed image
        self._display_processed_image()
        
        # Update metrics
        self._update_metrics_display(self.original_image, self.processed_image)
        
        # Store in database
        if self.current_image_id is not None:
            # Create output filename
            output_filename = f"resize_{self.current_image_id}_s{scale_factor:.1f}_{interpolation}.jpg"
            output_path = os.path.join("data", "output", output_filename)
            
            # Save the processed image
            create_directory_if_not_exists(os.path.dirname(output_path))
            self.processor.save_image(self.processed_image, output_path)
            
            # Store enhancement in database
            enhancement_data = {
                'image_id': self.current_image_id,
                'enhancement_type': 'resize',
                'parameters': {'scale_factor': scale_factor, 'interpolation': interpolation},
                'output_filename': output_filename
            }
            
            self.current_enhancement_id = self.db.add_enhancement(enhancement_data)
            
            # Store metrics
            if self.current_enhancement_id is not None:
                metrics = get_image_metrics(self.original_image, self.processed_image)
                metrics_data = [
                    {
                        'enhancement_id': self.current_enhancement_id,
                        'metric_name': name,
                        'metric_value': value
                    }
                    for name, value in metrics.items()
                ]
                
                self.db.add_metrics(metrics_data)
    def _setup_pseudocolor_controls(self) -> None:
        """Set up controls for pseudocolor processing."""
        # Parameters frame
        params_frame = ttk.Frame(self.enhancement_frame)
        params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Method control
        ttk.Label(params_frame, text="Method:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        method_var = tk.StringVar(value="colormap")
        
        ttk.Radiobutton(
            params_frame,
            text="OpenCV Colormap",
            variable=method_var,
            value="colormap"
        ).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Radiobutton(
            params_frame,
            text="Custom False Color",
            variable=method_var,
            value="false_color"
        ).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Colormap selection (for OpenCV colormap)
        ttk.Label(params_frame, text="Colormap:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        colormap_var = tk.StringVar(value="jet")
        
        colormap_options = [
            ("Jet", "jet"),
            ("Hot", "hot"),
            ("Cool", "cool"),
            ("Rainbow", "rainbow"),
            ("Viridis", "viridis"),
            ("Plasma", "plasma"),
            ("Inferno", "inferno")
        ]
        
        colormap_combo = ttk.Combobox(
            params_frame,
            textvariable=colormap_var,
            values=[opt[0] for opt in colormap_options],
            state="readonly",
            width=10
        )
        colormap_combo.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        colormap_combo.current(0)
        
        # Apply button
        apply_btn = ttk.Button(
            self.enhancement_frame,
            text="Apply Pseudocolor",
            command=lambda: self._apply_pseudocolor(method_var.get(), colormap_var.get())
        )
        apply_btn.pack(pady=10)
        
        # Apply with default values
        self._apply_pseudocolor(method_var.get(), colormap_var.get())
    
    def _apply_pseudocolor(self, method: str, colormap_name: str) -> None:
        """
        Apply pseudocolor processing.
        
        Args:
            method: Pseudocolor method ('colormap' or 'false_color')
            colormap_name: Name of the colormap to use
        """
        if self.original_image is None:
            return
        
        # Map colormap name to OpenCV constant
        colormap_map = {
            "jet": cv2.COLORMAP_JET,
            "hot": cv2.COLORMAP_HOT,
            "cool": cv2.COLORMAP_COOL,
            "rainbow": cv2.COLORMAP_RAINBOW,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "plasma": cv2.COLORMAP_PLASMA,
            "inferno": cv2.COLORMAP_INFERNO
        }
        
        # Process the image
        if method == "colormap":
            self.processed_image = self.processor.apply_pseudocolor(
                self.original_image,
                colormap_map.get(colormap_name, cv2.COLORMAP_JET)
            )
            parameters = {'method': 'colormap', 'colormap': colormap_name}
        else:  # false_color
            self.processed_image = self.processor.apply_false_color(self.original_image)
            parameters = {'method': 'false_color'}
        
        # Display the processed image
        self._display_processed_image()
        
        # Update metrics
        self._update_metrics_display(self.original_image, self.processed_image)
        
        # Store in database
        if self.current_image_id is not None:
            # Create output filename
            if method == "colormap":
                output_filename = f"pseudocolor_{self.current_image_id}_{colormap_name}.jpg"
            else:
                output_filename = f"pseudocolor_{self.current_image_id}_false_color.jpg"
            
            output_path = os.path.join("data", "output", output_filename)
            
            # Save the processed image
            create_directory_if_not_exists(os.path.dirname(output_path))
            self.processor.save_image(self.processed_image, output_path)
            
            # Store enhancement in database
            enhancement_data = {
                'image_id': self.current_image_id,
                'enhancement_type': 'pseudocolor',
                'parameters': parameters,
                'output_filename': output_filename
            }
            
            self.current_enhancement_id = self.db.add_enhancement(enhancement_data)
            # Store metrics
            if self.current_enhancement_id is not None:
                metrics = get_image_metrics(self.original_image, self.processed_image)
                metrics_data = [
                    {
                        'enhancement_id': self.current_enhancement_id,
                        'metric_name': name,
                        'metric_value': value
                    }
                    for name, value in metrics.items()
                ]
                
                self.db.add_metrics(metrics_data)
    
    def _update_metrics_display(self, original_image: Optional[np.ndarray], 
                               processed_image: Optional[np.ndarray]) -> None:
        """
        Update the metrics display.
        
        Args:
            original_image: Original image
            processed_image: Processed image
        """
        # Clear metrics frame
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()
        
        if original_image is None:
            # No image loaded
            ttk.Label(
                self.metrics_frame,
                text="No image loaded",
                font=("Arial", 10, "italic")
            ).pack(padx=10, pady=10)
            return
        
        # Get original image info
        original_info = self.processor.get_image_info(original_image)
        
        # Create metrics grid
        metrics_grid = ttk.Frame(self.metrics_frame)
        metrics_grid.pack(fill=tk.X, padx=10, pady=10)
        
        # Original image metrics
        ttk.Label(
            metrics_grid,
            text="Original Image:",
            font=("Arial", 10, "bold")
        ).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(
            metrics_grid,
            text=f"Dimensions: {original_info['width']}x{original_info['height']}"
        ).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(
            metrics_grid,
            text=f"Channels: {original_info['channels']}"
        ).grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        
        # Mean and standard deviation
        if original_info['channels'] == 1:
            ttk.Label(
                metrics_grid,
                text=f"Mean: {original_info['mean']:.2f}"
            ).grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
            
            ttk.Label(
                metrics_grid,
                text=f"Std Dev: {original_info['std']:.2f}"
            ).grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        else:
            ttk.Label(
                metrics_grid,
                text=f"Mean (BGR): ({original_info['mean'][0]:.2f}, {original_info['mean'][1]:.2f}, {original_info['mean'][2]:.2f})"
            ).grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
            
            ttk.Label(
                metrics_grid,
                text=f"Std Dev (BGR): ({original_info['std'][0]:.2f}, {original_info['std'][1]:.2f}, {original_info['std'][2]:.2f})"
            ).grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        
        # If processed image exists, show comparison metrics
        if processed_image is not None:
            # Get processed image info
            processed_info = self.processor.get_image_info(processed_image)
            
            # Processed image metrics
            ttk.Label(
                metrics_grid,
                text="Enhanced Image:",
                font=("Arial", 10, "bold")
            ).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
            
            ttk.Label(
                metrics_grid,
                text=f"Dimensions: {processed_info['width']}x{processed_info['height']}"
            ).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
            
            ttk.Label(
                metrics_grid,
                text=f"Channels: {processed_info['channels']}"
            ).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
            
            # Mean and standard deviation
            if processed_info['channels'] == 1:
                ttk.Label(
                    metrics_grid,
                    text=f"Mean: {processed_info['mean']:.2f}"
                ).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
                
                ttk.Label(
                    metrics_grid,
                    text=f"Std Dev: {processed_info['std']:.2f}"
                ).grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)
            else:
                ttk.Label(
                    metrics_grid,
                    text=f"Mean (BGR): ({processed_info['mean'][0]:.2f}, {processed_info['mean'][1]:.2f}, {processed_info['mean'][2]:.2f})"
                ).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
                
                ttk.Label(
                    metrics_grid,
                    text=f"Std Dev (BGR): ({processed_info['std'][0]:.2f}, {processed_info['std'][1]:.2f}, {processed_info['std'][2]:.2f})"
                ).grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)
            
            # Comparison metrics
            ttk.Label(
                metrics_grid,
                text="Comparison Metrics:",
                font=("Arial", 10, "bold")
            ).grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
            
            # Calculate metrics
            metrics = get_image_metrics(original_image, processed_image)
            
            # Display metrics
            ttk.Label(
                metrics_grid,
                text=f"PSNR: {metrics['psnr']:.2f} dB"
            ).grid(row=6, column=0, sticky=tk.W, padx=5, pady=2)
            
            ttk.Label(
                metrics_grid,
                text=f"SSIM: {metrics['ssim']:.4f}"
            ).grid(row=6, column=1, sticky=tk.W, padx=5, pady=2)
            
            ttk.Label(
                metrics_grid,
                text=f"MSE: {metrics['mse']:.2f}"
            ).grid(row=7, column=0, sticky=tk.W, padx=5, pady=2)
            
            ttk.Label(
                metrics_grid,
                text=f"Histogram Similarity: {metrics['hist_similarity']:.4f}"
            ).grid(row=7, column=1, sticky=tk.W, padx=5, pady=2)
    
    def _show_histograms(self) -> None:
        """Show histograms of original and processed images."""
        if self.original_image is None:
            messagebox.showinfo("Info", "Please load an image first.")
            return
        
        if self.processed_image is None:
            messagebox.showinfo("Info", "Please apply an enhancement first.")
            return
        
        # Create a new window
        histogram_window = tk.Toplevel(self.root)
        histogram_window.title("Image Histograms")
        histogram_window.geometry("800x600")
        
        # Create a figure for the histograms
        fig = plt.Figure(figsize=(8, 6), dpi=100)
        
        # Original image histogram
        ax1 = fig.add_subplot(221)
        ax1.set_title("Original Image")
        if len(self.original_image.shape) == 3:
            ax1.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        else:
            ax1.imshow(self.original_image, cmap='gray')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(222)
        ax2.set_title("Original Histogram")
        if len(self.original_image.shape) == 3:
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([self.original_image], [i], None, [256], [0, 256])
                ax2.plot(hist, color=color)
        else:
            hist = cv2.calcHist([self.original_image], [0], None, [256], [0, 256])
            ax2.plot(hist, color='black')
        ax2.set_xlim([0, 256])
        ax2.grid(alpha=0.3)
        
        # Processed image histogram
        ax3 = fig.add_subplot(223)
        ax3.set_title("Enhanced Image")
        if len(self.processed_image.shape) == 3:
            ax3.imshow(cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))
        else:
            ax3.imshow(self.processed_image, cmap='gray')
        ax3.axis('off')
        
        ax4 = fig.add_subplot(224)
        ax4.set_title("Enhanced Histogram")
        if len(self.processed_image.shape) == 3:
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([self.processed_image], [i], None, [256], [0, 256])
                ax4.plot(hist, color=color)
        else:
            hist = cv2.calcHist([self.processed_image], [0], None, [256], [0, 256])
            ax4.plot(hist, color='black')
        ax4.set_xlim([0, 256])
        ax4.grid(alpha=0.3)
        
        fig.tight_layout()
        
        # Add the figure to the window
        canvas = FigureCanvasTkAgg(fig, master=histogram_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _show_metrics(self) -> None:
        """Show detailed metrics in a separate window."""
        if self.original_image is None:
            messagebox.showinfo("Info", "Please load an image first.")
            return
        
        if self.processed_image is None:
            messagebox.showinfo("Info", "Please apply an enhancement first.")
            return
        
        # Calculate metrics
        metrics = get_image_metrics(self.original_image, self.processed_image)
        
        # Create a new window
        metrics_window = tk.Toplevel(self.root)
        metrics_window.title("Image Quality Metrics")
        metrics_window.geometry("400x400")
        
        # Create a frame for the metrics
        metrics_frame = ttk.Frame(metrics_window, padding=10)
        metrics_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add metrics
        row = 0
        ttk.Label(
            metrics_frame,
            text="Image Quality Metrics",
            font=("Arial", 12, "bold")
        ).grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=10)
        
        row += 1
        ttk.Separator(metrics_frame, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=2, sticky=tk.EW, pady=5
        )
        
        # Add each metric
        for name, value in metrics.items():
            row += 1
            # Format the metric name
            formatted_name = name.replace('_', ' ').title()
            
            ttk.Label(
                metrics_frame,
                text=f"{formatted_name}:",
                font=("Arial", 10, "bold")
            ).grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
            
            ttk.Label(
                metrics_frame,
                text=f"{value:.4f}"
            ).grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
    
    def _create_comparison(self) -> None:
        """Create a before/after comparison."""
        if self.original_image is None:
            messagebox.showinfo("Info", "Please load an image first.")
            return
        
        if self.processed_image is None:
            messagebox.showinfo("Info", "Please apply an enhancement first.")
            return
        
        # Create a new window
        comparison_window = tk.Toplevel(self.root)
        comparison_window.title("Before/After Comparison")
        comparison_window.geometry("800x600")
        
        # Create a notebook for different comparison views
        notebook = ttk.Notebook(comparison_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Side-by-side comparison tab
        side_by_side_frame = ttk.Frame(notebook)
        notebook.add(side_by_side_frame, text="Side-by-Side")
        
        # Create the side-by-side comparison
        comparison_img = create_side_by_side_comparison(
            self.original_image,
            self.processed_image,
            ('Original', 'Enhanced')
        )
        
        # Convert to Tkinter image
        comparison_tk = convert_cv_to_tk(comparison_img)
        
        # Display on canvas
        canvas = tk.Canvas(side_by_side_frame)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        canvas.create_image(
            canvas.winfo_reqwidth() // 2,
            canvas.winfo_reqheight() // 2,
            image=comparison_tk,
            anchor=tk.CENTER
        )
        
        # Keep a reference to prevent garbage collection
        canvas.image = comparison_tk
        
        # Histogram comparison tab
        histogram_frame = ttk.Frame(notebook)
        notebook.add(histogram_frame, text="Histograms")
        
        # Create a figure for the histograms
        fig = plt.Figure(figsize=(8, 6), dpi=100)
        
        # Original image histogram
        ax1 = fig.add_subplot(221)
        ax1.set_title("Original Image")
        if len(self.original_image.shape) == 3:
            ax1.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        else:
            ax1.imshow(self.original_image, cmap='gray')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(222)
        ax2.set_title("Original Histogram")
        if len(self.original_image.shape) == 3:
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([self.original_image], [i], None, [256], [0, 256])
                ax2.plot(hist, color=color)
        else:
            hist = cv2.calcHist([self.original_image], [0], None, [256], [0, 256])
            ax2.plot(hist, color='black')
        ax2.set_xlim([0, 256])
        ax2.grid(alpha=0.3)
        
        # Processed image histogram
        ax3 = fig.add_subplot(223)
        ax3.set_title("Enhanced Image")
        if len(self.processed_image.shape) == 3:
            ax3.imshow(cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))
        else:
            ax3.imshow(self.processed_image, cmap='gray')
        ax3.axis('off')
        
        ax4 = fig.add_subplot(224)
        ax4.set_title("Enhanced Histogram")
        if len(self.processed_image.shape) == 3:
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([self.processed_image], [i], None, [256], [0, 256])
                ax4.plot(hist, color=color)
        else:
            hist = cv2.calcHist([self.processed_image], [0], None, [256], [0, 256])
            ax4.plot(hist, color='black')
        ax4.set_xlim([0, 256])
        ax4.grid(alpha=0.3)
        
        fig.tight_layout()
        
        # Add the figure to the histogram frame
        canvas = FigureCanvasTkAgg(fig, master=histogram_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Interactive slider tab
        slider_frame = ttk.Frame(notebook)
        notebook.add(slider_frame, text="Interactive Slider")
        
        # Create a temporary HTML file for the slider
        slider_path = os.path.join("data", "output", "slider.html")
        create_directory_if_not_exists(os.path.dirname(slider_path))
        
        # Create the slider HTML
        create_before_after_slider(self.original_image, self.processed_image, slider_path)
        
        # Display a message about the slider
        ttk.Label(
            slider_frame,
            text="Interactive slider saved to:",
            font=("Arial", 12)
        ).pack(pady=20)
        
        ttk.Label(
            slider_frame,
            text=os.path.abspath(slider_path),
            font=("Arial", 10, "italic")
        ).pack(pady=5)
        
        # Button to open the slider in a browser
        ttk.Button(
            slider_frame,
            text="Open in Browser",
            command=lambda: os.system(f"xdg-open {slider_path}")
        ).pack(pady=20)
    
    def _save_processed_image(self) -> None:
        """Save the processed image to a file."""
        if self.processed_image is None:
            messagebox.showinfo("Info", "Please apply an enhancement first.")
            return
        
        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            title="Save Enhanced Image",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG files", "*.jpg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # Save the image
            result = self.processor.save_image(self.processed_image, file_path)
            
            if result:
                self.status_var.set(f"Image saved to: {file_path}")
            else:
                messagebox.showerror("Error", "Failed to save image.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {str(e)}")
    
    def _show_about(self) -> None:
        """Show about dialog."""
        messagebox.showinfo(
            "About",
            "Image Enhancement System\n\n"
            "A spatial domain image enhancement system using Python and OpenCV.\n\n"
            "Features:\n"
            "- Grayscale conversion and contrast adjustment\n"
            "- Noise reduction using mean filter\n"
            "- Edge enhancement and sharpening using Sobel operator\n"
            "- Image filtering\n"
            "- Interpolation and magnification\n"
            "- Pseudocolor processing\n"
            "- Before/after comparison view\n"
            "- Image statistics and metrics\n\n"
            "Version 1.0"
        )
