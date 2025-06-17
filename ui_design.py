import io
import tkinter as tk
from tkinter import Toplevel, messagebox, ttk
import cv2
from PIL import Image, ImageTk, ImageSequence
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import time
import lgpio
import os
import webbrowser
from fpdf import FPDF
import base64
import smtplib 
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from ultralytics import YOLO

class ImageCaptureApp:
    def __init__(self, root, emailid):
        self.root = root
        self.emailid = emailid
        
        # Get screen dimensions for responsive layout
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Set window to full screen or reasonable size
        if screen_width > 1400 and screen_height > 900:
            self.root.geometry("1400x900")
        else:
            # Use 90% of screen size if smaller than default
            self.root.geometry(f"{int(screen_width*0.9)}x{int(screen_height*0.9)}")
        
        self.root.title("Oral Image Capture System")
        self.root.configure(bg="#f0f2f5")
        
        # Make the window maximized by default
        self.root.state('zoomed')  # This will maximize the window
        
        # Style configuration
        self.style = ttk.Style()
        self.style.configure('TFrame', background="#f0f2f5")
        self.style.configure('TLabel', background="#f0f2f5", font=('Arial', 11))
        self.style.configure('TButton', font=('Arial', 11), padding=5)
        self.style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        self.style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        self.style.configure('Accent.TButton', foreground='white', background='#3498db', font=('Arial', 11, 'bold'))
        
        # Initialize variables
        self.current_image_index = 0
        self._camera_running = True
        
        # Constants for UI (will be adjusted dynamically)
        self.IMAGE_WIDTH = 350  # Will be adjusted based on window size
        self.IMAGE_HEIGHT = 250
        self.CAMERA_WIDTH = 640
        self.CAMERA_HEIGHT = 480
        
        # Setup database connection
        self.setup_database()
        
        # Setup UI components
        self.setup_ui()
        
        # Initialize camera
        self.setup_camera()
        
        # Center servos initially
        reset_servos()
        
        # Bind window resize event
        self.root.bind('<Configure>', self.on_window_resize)

    def on_window_resize(self, event):
        """Handle window resize events to adjust UI elements"""
        if event.widget == self.root:
            # Calculate new sizes based on window dimensions
            window_width = self.root.winfo_width()
            window_height = self.root.winfo_height()
            
            # Adjust image sizes proportionally
            self.IMAGE_WIDTH = max(200, min(400, int(window_width * 0.15)))
            self.IMAGE_HEIGHT = max(150, min(300, int(window_height * 0.25)))
            
            # Update displayed images if they exist
            if hasattr(self, 'ref_image_label') and hasattr(self.ref_image_label, 'image'):
                self.display_current_view()
            
            # Adjust camera size if needed
            self.CAMERA_WIDTH = max(400, min(800, int(window_width * 0.4)))
            self.CAMERA_HEIGHT = max(300, min(600, int(self.CAMERA_WIDTH * 0.75)))
            
            # Update camera feed if it exists
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_HEIGHT)

    def setup_ui(self):
        """Set up all UI components with responsive layout"""
        # Main container using grid for responsive layout
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure grid weights for responsive layout
        self.main_container.grid_columnconfigure(0, weight=1)  # Left panel (reference images)
        self.main_container.grid_columnconfigure(1, weight=2)  # Middle panel (instructions)
        self.main_container.grid_columnconfigure(2, weight=3)  # Right panel (camera and controls)
        self.main_container.grid_rowconfigure(0, weight=1)     # Main content area
        self.main_container.grid_rowconfigure(1, weight=0)     # Bottom controls (fixed height)
        
        # Left Panel (Reference Images and GIF)
        left_panel = ttk.Frame(self.main_container)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        left_panel.grid_propagate(False)
        
        # Configure left panel grid
        left_panel.grid_rowconfigure(0, weight=1)  # Reference image
        left_panel.grid_rowconfigure(1, weight=1)  # GIF
        left_panel.grid_columnconfigure(0, weight=1)
        
        # Reference image
        ref_frame = ttk.LabelFrame(left_panel, text="Reference Image", style='Header.TLabel')
        ref_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        ref_frame.grid_propagate(False)
        
        self.ref_image_label = ttk.Label(ref_frame)
        self.ref_image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # GIF demonstration
        gif_frame = ttk.LabelFrame(left_panel, text="Demonstration", style='Header.TLabel')
        gif_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        gif_frame.grid_propagate(False)
        
        self.gif_label = ttk.Label(gif_frame)
        self.gif_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Middle Panel (Instructions)
        middle_panel = ttk.Frame(self.main_container)
        middle_panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Instructions frame with scrollbar
        instr_frame = ttk.LabelFrame(middle_panel, text="Instructions", style='Header.TLabel')
        instr_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(instr_frame, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(instr_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack the canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Instructions label
        self.instructions_label = ttk.Label(
            scrollable_frame,
            text="",
            wraplength=400,  # Will adjust with window size
            justify="left",
            font=('Arial', 11)
        )
        self.instructions_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Right Panel (Camera and Controls)
        right_panel = ttk.Frame(self.main_container)
        right_panel.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        
        # Configure right panel grid
        right_panel.grid_rowconfigure(0, weight=3)  # Camera feed
        right_panel.grid_rowconfigure(1, weight=2)  # Controls
        right_panel.grid_columnconfigure(0, weight=1)
        
        # Camera feed
        camera_frame = ttk.LabelFrame(right_panel, text="Camera Feed", style='Header.TLabel')
        camera_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.camera_label = ttk.Label(camera_frame)
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Motor controls frame
        controls_frame = ttk.Frame(right_panel)
        controls_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure controls frame grid
        controls_frame.grid_columnconfigure(0, weight=1)
        controls_frame.grid_rowconfigure(0, weight=1)  # Servo controls
        controls_frame.grid_rowconfigure(1, weight=1)  # Stepper controls
        
        # Servo controls
        servo_frame = ttk.LabelFrame(controls_frame, text="Servo Controls", style='Header.TLabel')
        servo_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Pan (X-axis) controls
        pan_frame = ttk.Frame(servo_frame)
        pan_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        ttk.Label(pan_frame, text="Pan (X-axis)").pack()
        
        btn_frame = ttk.Frame(pan_frame)
        btn_frame.pack()
        
        ttk.Button(btn_frame, text="◄ Left", 
                  command=lambda: move_servo_x(-10), 
                  width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Right ►", 
                  command=lambda: move_servo_x(10), 
                  width=8).pack(side=tk.LEFT, padx=2)
        
        # Tilt (Y-axis) controls
        tilt_frame = ttk.Frame(servo_frame)
        tilt_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        ttk.Label(tilt_frame, text="Tilt (Y-axis)").pack()
        
        btn_frame = ttk.Frame(tilt_frame)
        btn_frame.pack()
        
        ttk.Button(btn_frame, text="▲ Up", 
                  command=lambda: move_servo_y(-10), 
                  width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="▼ Down", 
                  command=lambda: move_servo_y(10), 
                  width=8).pack(side=tk.LEFT, padx=2)
        
        # Reset button
        ttk.Button(servo_frame, text="Reset", 
                  command=reset_servos,
                  style='Accent.TButton').pack(side=tk.RIGHT, padx=10)
        
        # Stepper Motor controls
        stepper_frame = ttk.LabelFrame(controls_frame, text="Stepper Motor Controls", style='Header.TLabel')
        stepper_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create a grid for stepper controls
        control_grid = ttk.Frame(stepper_frame)
        control_grid.pack(pady=5)
        
        # Add buttons in a cross pattern
        ttk.Button(control_grid, text="▲ Up", 
                  command=lambda: stepper_move_y(-100), 
                  width=8).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(control_grid, text="◄ Left", 
                  command=lambda: stepper_move_x(-100), 
                  width=8).grid(row=1, column=0, padx=5, pady=2)
        ttk.Button(control_grid, text="Center", 
                  command=reset_servos,
                  width=8).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(control_grid, text="Right ►", 
                  command=lambda: stepper_move_x(100), 
                  width=8).grid(row=1, column=2, padx=5, pady=2)
        ttk.Button(control_grid, text="▼ Down", 
                  command=lambda: stepper_move_y(100), 
                  width=8).grid(row=2, column=1, padx=5, pady=2)
        
        # Bottom Controls (Navigation and Capture)
        bottom_frame = ttk.Frame(self.main_container)
        bottom_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=10)
        
        # Configure grid for bottom controls
        bottom_frame.grid_columnconfigure(0, weight=1)
        bottom_frame.grid_columnconfigure(1, weight=1)
        bottom_frame.grid_columnconfigure(2, weight=1)
        
        # Previous button
        self.prev_btn = ttk.Button(
            bottom_frame,
            text="◄ Previous",
            command=self.prev_image,
            width=12
        )
        self.prev_btn.grid(row=0, column=0, padx=5, sticky="w")
        
        # Capture button (centered)
        self.capture_btn = ttk.Button(
            bottom_frame,
            text="Capture Image",
            command=self.capture_image,
            style='Accent.TButton',
            width=15
        )
        self.capture_btn.grid(row=0, column=1, padx=5)
        
        # Next/Finish button
        self.next_btn = ttk.Button(
            bottom_frame,
            text="Next ►",
            command=self.next_image,
            width=12
        )
        self.next_btn.grid(row=0, column=2, padx=5, sticky="e")
        
        # Initially disable previous button
        self.prev_btn.config(state=tk.DISABLED)

    # [Rest of the methods remain the same...]

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCaptureApp(root, "example@example.com")  # Replace with actual email
    
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
