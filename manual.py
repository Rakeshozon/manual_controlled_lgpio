import io
import tkinter as tk
from tkinter import Toplevel, messagebox
import cv2
from PIL import Image, ImageTk, ImageSequence
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import threading
import time
import RPi.GPIO as GPIO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from ultralytics import YOLO
import os
import webbrowser
from fpdf import FPDF
import base64

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Servo Setup
SERVO_X_PIN = 17
SERVO_Y_PIN = 27
GPIO.setup(SERVO_X_PIN, GPIO.OUT)
GPIO.setup(SERVO_Y_PIN, GPIO.OUT)
servo_x = GPIO.PWM(SERVO_X_PIN, 50)  # 50Hz frequency
servo_y = GPIO.PWM(SERVO_Y_PIN, 50)
servo_x.start(0)
servo_y.start(0)

# Stepper Motor Setup
STEPPER_X_DIR = 5
STEPPER_X_STEP = 6
STEPPER_Y_DIR = 13
STEPPER_Y_STEP = 19
GPIO.setup([STEPPER_X_DIR, STEPPER_X_STEP, STEPPER_Y_DIR, STEPPER_Y_STEP], GPIO.OUT)

# Servo Control Functions
def angle_to_duty_cycle(angle):
    """Convert angle (-90 to 90) to duty cycle (2.5% to 12.5%)"""
    return 2.5 + (angle + 90) * (10 / 180)

def move_servo_x(angle):
    angle = max(-90, min(90, angle))
    duty_cycle = angle_to_duty_cycle(angle)
    servo_x.ChangeDutyCycle(duty_cycle)
    time.sleep(0.1)
    servo_x.ChangeDutyCycle(0)  # Prevents jitter

def move_servo_y(angle):
    angle = max(-90, min(90, angle))
    duty_cycle = angle_to_duty_cycle(angle)
    servo_y.ChangeDutyCycle(duty_cycle)
    time.sleep(0.1)
    servo_y.ChangeDutyCycle(0)

# Stepper Motor Control
def stepper_move(direction_pin, step_pin, steps, delay=0.002):
    GPIO.output(direction_pin, GPIO.HIGH if steps > 0 else GPIO.LOW)
    for _ in range(abs(steps)):
        GPIO.output(step_pin, GPIO.HIGH)
        time.sleep(delay)
        GPIO.output(step_pin, GPIO.LOW)
        time.sleep(delay)

class ReportGenerator:
    # (Keep your existing ReportGenerator class exactly as is)
    pass

class ImageCaptureApp:
    def __init__(self, root, emailid):
        self.root = root
        self.emailid = emailid
        self.root.title("Manual Oral Scan System")
        self.root.geometry("1300x920+10+10")
        self.root.configure(bg="white")
        
        # Current positions
        self.current_x_angle = 0
        self.current_y_angle = 0
        
        # Constants
        self.IMAGE_WIDTH = 300
        self.IMAGE_HEIGHT = 300
        self.IMAGE_CAPTURE_VIEWER_WIDTH = 400
        self.IMAGE_CAPTURE_VIEWER_HEIGHT = 300
        
        # Predefined positions for each view
        self.PRESET_POSITIONS = [
            {'x': 0, 'y': 70},     # Bottom of tongue
            {'x': 15, 'y': -10},   # Bottom teeth view
            {'x': 20, 'y': -15},   # Bottom teeth with lower lip pulled down
            {'x': 25, 'y': -5},    # Cheek side of bottom teeth 2
            {'x': -25, 'y': -5},   # Cheek side of bottom teeth
            {'x': 30, 'y': 10},    # Cheek side of top teeth
            {'x': -30, 'y': 10},   # Cheek side of top teeth 2
            {'x': 0, 'y': 15},     # Front of tongue
            {'x': -20, 'y': 5},    # Left side of tongue
            {'x': 20, 'y': 5},     # Right side of tongue
            {'x': 0, 'y': 0},      # Smile showing front teeth
            {'x': 10, 'y': -75},   # Top teeth view
            {'x': -10, 'y': -75}   # Top teeth with upper lip pulled up
        ]
        
        # (Keep your existing image_list, gif_list, and instructions_list)
        
        # Database Connection
        try:
            self.connection = mysql.connector.connect(
                host="localhost",
                user="root",
                password="123456",
                database="register"
            )
            self.patient_id = self.get_patient_id(emailid)
            if self.patient_id is None:
                messagebox.showerror("Error", "Patient not found")
                root.destroy()
                return
        except Error as e:
            messagebox.showerror("Database Error", f"Error: {e}")
            root.destroy()
        
        # Setup UI
        self.setup_ui()
        
        # Initialize Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open camera")
            root.destroy()
        
        # Start camera feed
        self.show_camera_feed()
        
        # Center servos initially
        self.reset_servos()

    def setup_ui(self):
        """Set up the user interface"""
        # Left Frame - Instructions and reference images
        self.setup_left_frame()
        
        # Right Frame - Camera feed and controls
        self.setup_right_frame()
        
        # Navigation and capture buttons
        self.setup_control_buttons()
        
        # Motor/Servo controls
        self.setup_motor_controls()

    def setup_left_frame(self):
        """Left side with instructions and reference images"""
        self.left_image_label = tk.Label(self.root, bg="white")
        self.left_image_label.place(x=0, y=0)
        
        self.gif_label = tk.Label(self.root, bg="white")
        self.gif_label.place(x=0, y=self.IMAGE_HEIGHT + 10)
        
        self.instructions_label_frame = tk.Frame(self.root, bg="white", width=650-self.IMAGE_WIDTH, 
                                               height=690, highlightbackground="lightblue", highlightthickness=3)
        self.instructions_label_frame.place(x=self.IMAGE_WIDTH+10, y=0)
        
        self.instructions_heading = tk.Label(self.instructions_label_frame, text="INSTRUCTIONS:", 
                                           font=("times new roman", 18, 'bold'), fg="lightblue", bg='white')
        self.instructions_heading.place(x=0, y=0)
        
        self.instructions_label = tk.Label(self.instructions_label_frame, text="", bg="white",
                                         wraplength=650-self.IMAGE_WIDTH-10, 
                                         font=("times new roman", 12), justify="left")
        self.instructions_label.place(x=0, y=40)
        
        # Display first image
        self.display_left_image()

    def setup_right_frame(self):
        """Right side with camera feed"""
        self.right_frame = tk.Frame(self.root, bg="white", width=650, height=700,
                                  highlightbackground="blue", highlightthickness=3)
        self.right_frame.place(x=650, y=0)
        
        self.camera_label = tk.Label(self.right_frame, bg="black")
        self.camera_label.pack()

    def setup_control_buttons(self):
        """Navigation and capture buttons"""
        self.button_frame = tk.Frame(self.right_frame, bg="white")
        self.button_frame.pack(side=tk.BOTTOM, pady=10)
        
        self.prev_button = tk.Button(self.button_frame, text="Previous", command=self.prev_image,
                                   font=("times new roman", 16, 'bold'), bg="lightgreen", fg="white")
        self.prev_button.pack(side=tk.LEFT, padx=20)
        
        self.next_button = tk.Button(self.button_frame, text="Next", command=self.next_image,
                                   font=("times new roman", 16, 'bold'), bg="lightgreen", fg="white")
        self.next_button.pack(side=tk.LEFT, padx=20)
        
        self.capture_button = tk.Button(self.button_frame, text="Capture", command=self.capture_and_show_popup,
                                      font=("times new roman", 16, 'bold'), bg="lightblue", fg="white")
        self.capture_button.pack(side=tk.LEFT, padx=20)
        
        self.analysis_button = tk.Button(self.button_frame, text="Analyze", command=self.show_analysis,
                                       font=("times new roman", 16, 'bold'), bg="purple", fg="white")
        self.analysis_button.pack(side=tk.LEFT, padx=20)

    def setup_motor_controls(self):
        """Manual motor/servo controls"""
        control_frame = tk.Frame(self.right_frame, bg="white")
        control_frame.pack(side=tk.BOTTOM, pady=10)
        
        # Servo Control
        servo_frame = tk.LabelFrame(control_frame, text="Servo Control", bg="lightgray")
        servo_frame.pack(side=tk.LEFT, padx=10)
        
        # X-axis (Pan) control
        tk.Label(servo_frame, text="Pan (X-axis)", bg="lightgray").grid(row=0, column=0, columnspan=3)
        self.x_slider = tk.Scale(servo_frame, from_=-90, to=90, orient=tk.HORIZONTAL,
                                command=self.update_servo_x, length=200)
        self.x_slider.grid(row=1, column=0, columnspan=3)
        
        # Y-axis (Tilt) control
        tk.Label(servo_frame, text="Tilt (Y-axis)", bg="lightgray").grid(row=2, column=0, columnspan=3)
        self.y_slider = tk.Scale(servo_frame, from_=-90, to=90, orient=tk.HORIZONTAL,
                                command=self.update_servo_y, length=200)
        self.y_slider.grid(row=3, column=0, columnspan=3)
        
        # Stepper Control
        stepper_frame = tk.LabelFrame(control_frame, text="Stepper Control", bg="lightgray")
        stepper_frame.pack(side=tk.LEFT, padx=10)
        
        # X-axis stepper
        tk.Button(stepper_frame, text="Left", command=lambda: stepper_move(STEPPER_X_DIR, STEPPER_X_STEP, -100)).grid(row=1, column=0)
        tk.Button(stepper_frame, text="Right", command=lambda: stepper_move(STEPPER_X_DIR, STEPPER_X_STEP, 100)).grid(row=1, column=2)
        
        # Y-axis stepper
        tk.Button(stepper_frame, text="Up", command=lambda: stepper_move(STEPPER_Y_DIR, STEPPER_Y_STEP, -100)).grid(row=0, column=1)
        tk.Button(stepper_frame, text="Down", command=lambda: stepper_move(STEPPER_Y_DIR, STEPPER_Y_STEP, 100)).grid(row=2, column=1)
        
        # Position Presets
        preset_frame = tk.LabelFrame(control_frame, text="Position Presets", bg="lightgray")
        preset_frame.pack(side=tk.LEFT, padx=10)
        
        for i, pos in enumerate(self.PRESET_POSITIONS[:6]):  # First 6 presets
            btn = tk.Button(preset_frame, text=f"Pos {i+1}", 
                          command=lambda p=pos: self.move_to_position(p))
            btn.grid(row=i//3, column=i%3, padx=2, pady=2)
        
        # Reset Button
        tk.Button(control_frame, text="Reset", command=self.reset_servos, 
                bg="orange", fg="white").pack(side=tk.LEFT, padx=10)

    def move_to_position(self, position):
        """Move to a predefined position"""
        self.x_slider.set(position['x'])
        self.y_slider.set(position['y'])
        move_servo_x(position['x'])
        move_servo_y(position['y'])

    def update_servo_x(self, angle):
        try:
            angle = float(angle)
            self.current_x_angle = angle
            move_servo_x(angle)
        except ValueError:
            pass

    def update_servo_y(self, angle):
        try:
            angle = float(angle)
            self.current_y_angle = angle
            move_servo_y(angle)
        except ValueError:
            pass

    def reset_servos(self):
        """Reset servos to center position"""
        self.x_slider.set(0)
        self.y_slider.set(0)
        move_servo_x(0)
        move_servo_y(0)

    # (Keep all your existing methods for:
    # display_left_image, load_gif, animate_gif, capture_and_show_popup, 
    # show_popup, save_and_next, get_next_identifier, show_camera_feed,
    # show_analysis, send_email_report, cleanup, etc.)
    
    # Only remove the automatic tracking related methods

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'cap'):
            self.cap.release()
        servo_x.stop()
        servo_y.stop()
        GPIO.cleanup()
        if hasattr(self, 'connection'):
            self.connection.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCaptureApp(root, "example@example.com")
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()
