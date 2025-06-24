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
from gpiozero import AngularServo, Device
from gpiozero.pins.pigpio import PiGPIOFactory
from flask import Flask, send_from_directory
import threading
import smtplib 
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

from ultralytics import YOLO
import os
import webbrowser
from datetime import datetime
from fpdf import FPDF

# Initialize PiGPIO
Device.pin_factory = PiGPIOFactory()
class ReportServer:
    def __init__(self, report_dir):
        self.app = Flask(__name__)
        self.report_dir = report_dir
        self.server_thread = None
        self.running = False
        
        @self.app.route('/reports/<path:filename>')
        def serve_report(filename):
            return send_from_directory(self.report_dir, filename)
    
    def start(self, port=5000):
        if not self.running:
            self.running = True
            self.server_thread = threading.Thread(
                target=lambda: self.app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
            )
            self.server_thread.daemon = True
            self.server_thread.start()
    
    def stop(self):
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=1)
            
            
# Custom DRV8825 class (from your working code)
class DRV8825():
    def __init__(self, dir_pin, step_pin, enable_pin, mode_pins):
        self.dir_pin = dir_pin
        self.step_pin = step_pin        
        self.enable_pin = enable_pin
        self.mode_pins = mode_pins
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.dir_pin, GPIO.OUT)
        GPIO.setup(self.step_pin, GPIO.OUT)
        GPIO.setup(self.enable_pin, GPIO.OUT)
        GPIO.setup(self.mode_pins, GPIO.OUT)
        
    def digital_write(self, pin, value):
        GPIO.output(pin, value)
        
    def Stop(self):
        self.digital_write(self.enable_pin, 0)
    
    def SetMicroStep(self, mode, stepformat):
        microstep = {'fullstep': (0, 0, 0),
                     'halfstep': (1, 0, 0),
                     '1/4step': (0, 1, 0),
                     '1/8step': (1, 1, 0),
                     '1/16step': (0, 0, 1),
                     '1/32step': (1, 0, 1)}

        if (mode == 'software'):
            self.digital_write(self.mode_pins, microstep[stepformat])
        
    def TurnStep(self, Dir, steps, stepdelay=0.005):
        if (Dir == 'forward'):
            self.digital_write(self.enable_pin, 1)
            self.digital_write(self.dir_pin, 0)
        elif (Dir == 'backward'):
            self.digital_write(self.enable_pin, 1)
            self.digital_write(self.dir_pin, 1)
        else:
            self.digital_write(self.enable_pin, 0)
            return

        if (steps == 0):
            return
            
        for _ in range(steps):
            self.digital_write(self.step_pin, True)
            time.sleep(stepdelay)
            self.digital_write(self.step_pin, False)
            time.sleep(stepdelay)

# Initialize motor drivers
MotorX = DRV8825(
    dir_pin=13,     # STEPPER_X_DIR
    step_pin=19,    # STEPPER_X_STEP
    enable_pin=12,
    mode_pins=(16, 17, 20)
)

MotorY = DRV8825(
    dir_pin=24,     # STEPPER_Y_DIR
    step_pin=18,    # STEPPER_Y_STEP
    enable_pin=4,
    mode_pins=(21, 22, 27)
)

# Initialize servos
servo_x = AngularServo(5, min_angle=-90, max_angle=90)  # Pan servo
servo_y = AngularServo(6, min_angle=-90, max_angle=90)  # Tilt servo


# Load Haar cascades for face and mouth detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Predefined servo positions for each instruction
SERVO_POSITIONS = [
    {'x': 0, 'y': 70},     # Position 1 - Bottom of tongue
    {'x': 15, 'y': -10},   # Position 2 - Bottom teeth view
    {'x': 20, 'y': -15},   # Position 3 - Bottom teeth with lower lip pulled down
    {'x': 25, 'y': -5},    # Position 4 - Cheek side of bottom teeth 2
    {'x': -25, 'y': -5},   # Position 5 - Cheek side of bottom teeth
    {'x': 30, 'y': 10},    # Position 6 - Cheek side of top teeth
    {'x': -30, 'y': 10},   # Position 7 - Cheek side of top teeth 2
    {'x': 0, 'y': 15},     # Position 8 - Front of tongue
    {'x': -20, 'y': 5},    # Position 9 - Left side of tongue
    {'x': 20, 'y': 5},     # Position 10 - Right side of tongue
    {'x': 0, 'y': 0},     # Position 11 - Smile showing front teeth
    {'x': 10, 'y': -75},    # Position 12 - Top teeth view
    {'x': -10, 'y': -75}    # Position 13 - Top teeth with upper lip pulled up
]
# Required imports
from ultralytics import YOLO
import os
import webbrowser
from datetime import datetime
from fpdf import FPDF
import base64
from PIL import Image
import io

class ReportGenerator:
    def __init__(self, patient_id, connection):
        self.patient_id = patient_id
        self.connection = connection
        self.model = YOLO("/home/pi/Downloads/best.pt")  # Load your trained YOLO model
        self.results = []  # Will store (img_num, result, original_path, detection_path)
        self.report_html = ""
        self.report_dir = "/home/pi/patient_reports"
        self.temp_files = []  # Track all temporary files for cleanup
        os.makedirs(self.report_dir, exist_ok=True)

    def analyze_images(self):
        """Analyze all images for the patient and prepare results"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT image_data, image_number FROM patient_images WHERE patient_id=%s", (self.patient_id,))
            images = cursor.fetchall()

            for img_data, img_num in images:
                # Create temporary file paths
                img_path = f"/tmp/patient_{self.patient_id}_img_{img_num}.jpg"
                original_path = f"/tmp/patient_{self.patient_id}_img_{img_num}_original.jpg"
                detection_path = f"/tmp/patient_{self.patient_id}_img_{img_num}_detection.jpg"

                # Save original image
                with open(original_path, 'wb') as f:
                    f.write(img_data)
                self.temp_files.append(original_path)

                # Save temporary image for detection
                with open(img_path, 'wb') as f:
                    f.write(img_data)
                self.temp_files.append(img_path)

                # Run detection
                result = self.model(img_path, save=True)
                
                # Save detection image with annotations
                im_array = result[0].plot()  # Get annotated image array
                im = Image.fromarray(im_array[..., ::-1])  # Convert to PIL Image
                im.save(detection_path)
                self.temp_files.append(detection_path)

                # Store results (img_num, result, original_path, detection_path)
                self.results.append((img_num, result[0], original_path, detection_path))

        except Exception as e:
            print(f"Error analyzing images: {e}")
            raise
        finally:
            if 'cursor' in locals():
                cursor.close()

    def generate_findings_html(self, result):
        """Generate HTML for the findings section based on detection results"""
        findings = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])

            severity = (
                "high" if class_name in ["cavity", "lesion"]
                else "medium" if class_name == "plaque"
                else "low"
            )

            findings.append(f"""
            <div class="finding-item">
                <span class="severity-{severity}">
                {class_name.title()} detected (confidence: {confidence:.2f})
                </span>
            </div>
            """)
        return "\n".join(findings) if findings else "<p>No significant findings detected.</p>"

    def generate_report(self):
        """Generate HTML report with image comparisons and findings"""
        self.report_html = f"""
        <html>
        <head>
            <title>Oral Health Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                .image-container {{ margin: 20px 0; border: 1px solid #ddd; padding: 10px; }}
                .image-title {{ font-weight: bold; margin-bottom: 10px; }}
                .findings {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
                .finding-item {{ margin: 5px 0; }}
                .severity-high {{ color: #e74c3c; font-weight: bold; }}
                .severity-medium {{ color: #f39c12; }}
                .severity-low {{ color: #27ae60; }}
                .image-comparison {{
                    display: flex;
                    justify-content: space-between;
                    margin: 10px 0;
                }}
                .image-comparison img {{
                    width: 48%;
                    border: 1px solid #ddd;
                }}
            </style>
        </head>
        <body>
            <h1>Oral Health Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Patient ID: {self.patient_id}</p>
        """

        for img_num, result, original_path, detection_path in self.results:
            try:
                # Convert images to base64 for HTML embedding
                with open(original_path, "rb") as f:
                    original_base64 = base64.b64encode(f.read()).decode('utf-8')
                with open(detection_path, "rb") as f:
                    detection_base64 = base64.b64encode(f.read()).decode('utf-8')

                self.report_html += f"""
                <div class="image-container">
                    <div class="image-title">Image #{img_num} Analysis</div>
                    <div class="image-comparison">
                        <div>
                            <h3>Original Image</h3>
                            <img src="data:image/jpeg;base64,{original_base64}" width="400">
                        </div>
                        <div>
                            <h3>Detection Results</h3>
                            <img src="data:image/jpeg;base64,{detection_base64}" width="400">
                        </div>
                    </div>
                    <div class="findings">
                        <h3>Findings:</h3>
                        {self.generate_findings_html(result)}
                    </div>
                </div>
                """
            except Exception as e:
                print(f"Error processing image {img_num}: {e}")
                continue

        self.report_html += """
        </body>
        </html>
        """

        report_path = os.path.join(
            self.report_dir,
            f"report_{self.patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        with open(report_path, 'w') as f:
            f.write(self.report_html)

        return report_path

    def generate_pdf_report(self, html_path):
        """Generate PDF version of the report"""
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            # Add title
            pdf.cell(200, 10, txt="Oral Health Analysis Report", ln=1, align='C')
            pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
            pdf.cell(200, 10, txt=f"Patient ID: {self.patient_id}", ln=1)
            pdf.ln(10)

            for img_num, result, original_path, detection_path in self.results:
                # Verify files exist before adding to PDF
                if not os.path.exists(original_path) or not os.path.exists(detection_path):
                    print(f"Warning: Missing image files for image #{img_num}")
                    continue
                
                # Add image number
                pdf.set_font("Arial", 'B', size=12)
                pdf.cell(200, 10, txt=f"Image #{img_num} Analysis", ln=1)
                pdf.set_font("Arial", size=12)
                
                # Add original image
                pdf.cell(200, 10, txt="Original Image:", ln=1)
                pdf.image(original_path, x=10, w=90)
                
                # Add detection image
                pdf.cell(200, 10, txt="Detection Results:", ln=1)
                pdf.image(detection_path, x=10, w=90)
                
                # Add findings
                pdf.cell(200, 10, txt="Findings:", ln=1)
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])
                    pdf.cell(200, 10, txt=f"- {class_name.title()} (confidence: {confidence:.2f})", ln=1)
                
                pdf.ln(10)

            pdf_path = html_path.replace('.html', '.pdf')
            pdf.output(pdf_path)
            return pdf_path
            
        except Exception as e:
            print(f"Error generating PDF: {e}")
            raise

    def cleanup_temp_files(self):
        """Clean up all temporary files created during report generation"""
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting temporary file {file_path}: {e}")


class ImageCaptureApp:
    def __init__(self, root, emailid):
        self.root = root
        self.emailid = emailid
        self.patient_id = 1
        self.root.title("Oral Image Capture System")
        self.root.geometry("1300x920+10+10")
        self.root.configure(bg="white")
        
        # Initialize variables
        self.camera_image = None
        self.current_image_index = 0
        self.tracking_active = True
        self.auto_capture_active = False
        self.auto_capture_interval = 12 # seconds
        self.last_capture_time = 0
        self.next_capture_time = 0
        self.current_servo_positions = {'x': 0, 'y': 0}
        self._camera_running = True
        
        # Constants for UI
        self.IMAGE_WIDTH = 300
        self.IMAGE_HEIGHT = 300
        self.IMAGE_CAPTURE_VIEWER_WIDTH = 400
        self.IMAGE_CAPTURE_VIEWER_HEIGHT = 300
        
        # Image and instruction data
        self.image_list = [
            r"/home/pi/Downloads/Patient_app/Oral images samples/Oral images samples/Bottom of tongue_29092023_final.png",
            r"/home/pi/Downloads/Patient_app/Oral images samples/Oral images samples/Bottom teeth view_29092023_final.png",
            r"/home/pi/Downloads/Patient_app/Oral images samples/Oral images samples/Bottom teeth with lower lip pulled down_29092023_final.png",
            r"/home/pi/Downloads/Patient_app/Oral images samples/Oral images samples/Cheek side of bottom teeth 2_29092023_final.png",
            r"/home/pi/Downloads/Patient_app/Oral images samples/Oral images samples/Cheek side of bottom teeth_29092023_final.png",
            r"/home/pi/Downloads/Patient_app/Oral images samples/Oral images samples/Cheek side of top teeth _29092023_final.png",
            r"/home/pi/Downloads/Patient_app/Oral images samples/Oral images samples/Cheek side of top teeth 2 _29092023_final.png",
            r"/home/pi/Downloads/Patient_app/Oral images samples/Oral images samples/Front of tongue_29092023_final.png",
            r"/home/pi/Downloads/Patient_app/Oral images samples/Oral images samples/Left side of tongue _29092023_final.png",
            r"/home/pi/Downloads/Patient_app/Oral images samples/Oral images samples/Right side of tongue  _29092023_final.png",
            r"/home/pi/Downloads/Patient_app/Oral images samples/Oral images samples/Smile showing front teeth_29092023_final.png",
            r"/home/pi/Downloads/Patient_app/Oral images samples/Oral images samples/Top teeth view_29092023_final.png",
            r"/home/pi/Downloads/Patient_app/Oral images samples/Oral images samples/Top teeth with upper lip pulled up_29092023_final.png"
        ]
        
        self.gif_list = [
            r"/home/pi/Downloads/Patient_app/Combined/Combined/GIF - 14.gif",
            r"/home/pi/Downloads/Patient_app/Combined/Combined/GIF - 4.gif",
            r"/home/pi/Downloads/Patient_app/Combined/Combined/GIF - 6.gif",
            r"/home/pi/Downloads/Patient_app/Combined/Combined/GIF - 10.gif",
            r"/home/pi/Downloads/Patient_app/Combined/Combined/GIF - 9.gif",
            r"/home/pi/Downloads/Patient_app/Combined/Combined/GIF - 7.gif",
            r"/home/pi/Downloads/Patient_app/Combined/Combined/GIF - 8.gif",
            r"/home/pi/Downloads/Patient_app/Combined/Combined/GIF - 11.gif",
            r"/home/pi/Downloads/Patient_app/Combined/Combined/GIF - 12.gif",
            r"/home/pi/Downloads/Patient_app/Combined/Combined/GIF - 13.gif",
            r"/home/pi/Downloads/Patient_app/Combined/Combined/GIF - 2.gif",
            r"/home/pi/Downloads/Patient_app/Combined/Combined/GIF - 3.gif",
            r"/home/pi/Downloads/Patient_app/Combined/Combined/GIF - 5.gif"
        ]
        
        self.instructions_list = [
            '''1. Natural lighting is preferred, ensure that the light source is in the opposite direction of the mouth
2. Brush your teeth or at least rinse your mouth with water before clicking pictures
3. Do not click the pictures through the mirror (No mirror selfies)
4. Say "aaah!" open mouth wide/big
5. Tilt head backward
6. Touch the tip of your tongue to the top of your mouth (palate)''',
            
            '''1. Say "aaah!" open mouth wide/big
2. Tilt head backward
3. Touch the tip of your tongue to the top of your mouth (palate)''',
            
            '''1. Say "aaah!" open mouth wide/big
2. Use tips of finger and thumb to pull down the lower lip from the center''',
            
            '''1. Say "aaah!" open mouth wide/big
2. Use the tip of the finger of the left hand to pull your left cheek towards the left side
3. Click the picture from the right hand, from a lower left angle''',
            
            '''1. Say "aaah!" open mouth wide/big
2. Use the tip of the finger of the right hand to pull your right cheek towards the right side
3. Click the picture from the right hand, from a lower right angle''',
            
            '''1. Say "aaah!" Open your mouth and use your fingertip to pull the left corner of the mouth towards the lower left side
2. Place your hand on the upper right side to click picture from the opposite side''',
            
            '''1. Say "aaah!" Open your mouth and use your fingertip to pull the right corner of the mouth towards the lower left side
2. Place your hand on the upper left side to click picture from the opposite side''',
            
            '''1. Say "aaah!" Open your mouth
2. Stick your tongue outside''',
            
            '''1. Open wide mouth. Pull out your tongue to the opposite side
2. Pull cheek to the side with the fingertip''',
            
            '''1. Open wide mouth. Pull out your tongue to the opposite side
2. Pull cheek to the side with the fingertip''',
            
            '''1. Natural lighting is preferred, ensure that the light source is in the opposite direction of the mouth
2. Say "aaah!" open mouth wide/big
3. Tilt head backward''',
            
            '''1. Natural lighting is preferred, ensure that the light source is in the opposite direction of the mouth
2. Say "aaah!" open mouth wide/big
3. Tilt head backward''',
            
            '''1. Natural lighting is preferred, ensure that the light source is in the opposite direction of the mouth
2. Say "aaah!" open mouth wide/big
3. Tilt head backward'''
        ]

        # Database Connection
        try:
            self.connection = mysql.connector.connect(
                host="localhost",
                user="root",
                password="123456",
                database="register"
            )
        except Error as e:
            messagebox.showerror("Database Error", f"Error: {e}")
            root.destroy()
             # Get patient ID
        self.patient_id = self.get_patient_id(emailid)
        if self.patient_id is None:
            messagebox.showerror("Error", "Patient not found in database")
            root.destroy()
            return

        # Setup UI
        self.setup_ui()
        
        # Initialize USB Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open USB camera.")
            self.root.destroy()

        # Start camera feed and tracking
        self.show_camera_feed()
        
        # Center servos initially
        self.reset_servos()
    def get_patient_id(self, email):
        """Get patient ID from database using email"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT id FROM patient WHERE email = %s", (email,))
            result = cursor.fetchone()
            return result[0] if result else None
        except Error as e:
            messagebox.showerror("Database Error", f"Error fetching patient ID: {e}")
            return None
        finally:
            if 'cursor' in locals():
                cursor.close()
    def setup_ui(self):
        """Set up all UI components"""
        # Left Frame - Instructions and reference images
        self.setup_left_frame()
        
        # Right Frame - Camera feed and controls
        self.setup_right_frame()
        
        # Navigation and capture buttons
        self.setup_control_buttons()

    def setup_left_frame(self):
        """Set up the left side with instructions and reference images"""
        # Reference image display
        self.left_image_label = tk.Label(self.root, bg="white")
        self.left_image_label.place(x=0, y=0)
        
        # GIF demonstration
        self.gif_label = tk.Label(self.root, bg="white")
        self.gif_label.place(x=0, y=self.IMAGE_HEIGHT + 10)
        
        # Instructions frame
        self.instructions_label_frame = tk.Frame(self.root, bg="white", width=650 - self.IMAGE_WIDTH, 
                                               height=690, highlightbackground="lightblue", highlightthickness=3)
        self.instructions_label_frame.place(x=self.IMAGE_WIDTH + 10, y=0)
        
        self.instructions_heading = tk.Label(self.instructions_label_frame, text="INSTRUCTIONS:", 
                                           font=("times new roman", 18, 'bold'), fg="lightblue", bg='white')
        self.instructions_heading.place(x=0, y=0)
        
        self.instructions_label = tk.Label(self.instructions_label_frame, text="", bg="white",
                                         wraplength=650 - self.IMAGE_WIDTH - 10, 
                                         font=("times new roman", 12), justify="left")
        self.instructions_label.place(x=0, y=40)
        
        # Display first image and instructions
        self.display_left_image()
    # Add this method to your ImageCaptureApp class
    def setup_report_server(self):
        self.report_server = ReportServer("/home/pi/patient_reports")
        self.report_server.start()

    def setup_right_frame(self):
        """Set up the right side with camera feed and status"""
        self.right_frame = tk.Frame(self.root, bg="white", width=650, height=700,
                                  highlightbackground="blue", highlightthickness=3)
        self.right_frame.place(x=650, y=0)
        
        # Camera feed display
        self.camera_label = tk.Label(self.right_frame, bg="black")
        self.camera_label.pack()
        
        # Status and timer display
        self.status_frame = tk.Frame(self.right_frame, bg="white")
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        self.tracking_status = tk.Label(self.status_frame, text="Tracking: ACTIVE", 
                                      font=("times new roman", 14), bg="white", fg="green")
        self.tracking_status.pack(side=tk.LEFT, padx=10)
        
        self.timer_label = tk.Label(self.status_frame, text="Auto Capture: OFF", 
                                  font=("times new roman", 14), bg="white", fg="black")
        self.timer_label.pack(side=tk.RIGHT, padx=10)

    def setup_control_buttons(self):
        """Set up the control buttons at the bottom"""
        self.button_frame = tk.Frame(self.right_frame, bg="white")
        self.button_frame.pack(side=tk.BOTTOM, pady=10)
        
        # Navigation buttons
        self.prev_button = tk.Button(self.button_frame, text="Previous", command=self.prev_image,
                                   font=("times new roman", 16, 'bold'), 
                                   bg="lightgreen", fg="white", bd=0)
        self.prev_button.pack(side=tk.LEFT, padx=20)
        
        self.next_button = tk.Button(self.button_frame, text="Next", command=self.next_image,
                                   font=("times new roman", 16, 'bold'), 
                                   bg="lightgreen", fg="white", bd=0)
        self.next_button.pack(side=tk.LEFT, padx=20)
        
        # Capture buttons
        self.capture_button = tk.Button(self.button_frame, text="Capture", command=self.capture_and_show_popup,
                                      font=("times new roman", 16, 'bold'), 
                                      bg="lightblue", fg="white", bd=0)
        self.capture_button.pack(side=tk.LEFT, padx=20)
        
        self.auto_capture_button = tk.Button(self.button_frame, text="Auto Capture OFF", 
                                           command=self.toggle_auto_capture,
                                           font=("times new roman", 16, 'bold'), 
                                           bg="orange", fg="white", bd=0)
        self.auto_capture_button.pack(side=tk.LEFT, padx=20)
        
        # Analysis button (hidden until all images are captured)
        self.analysis_button = tk.Button(self.button_frame, text="Analyze Results", 
                                       command=self.show_analysis,
                                       font=("times new roman", 16, 'bold'), 
                                       bg="purple", fg="white", bd=0)
        self.analysis_button.pack_forget()

    def prev_image(self):
        """Navigate to the previous oral image view"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_left_image()
            self.reset_servos()

    def next_image(self):
        """Navigate to the next oral image view"""
        if self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.display_left_image()
            self.reset_servos()
        else:
            self.show_analysis()

    def display_left_image(self):
        """Display the current reference image, GIF, and instructions"""
        if 0 <= self.current_image_index < len(self.instructions_list):
            try:
                # Load static image
                image_path = self.image_list[self.current_image_index]
                image = Image.open(image_path)
                image = image.resize((self.IMAGE_WIDTH, self.IMAGE_HEIGHT), resample=Image.LANCZOS)
                tk_image = ImageTk.PhotoImage(image)
                self.current_tk_image = tk_image

                # Load GIF
                gif_path = self.gif_list[self.current_image_index]
                tk_gif_frames = self.load_gif(gif_path)

                # Update or create image labels
                if hasattr(self, 'left_image_label'):
                    self.left_image_label.destroy()
                if hasattr(self, 'gif_label'):
                    self.gif_label.destroy()

                self.left_image_label = tk.Label(self.root, image=self.current_tk_image, bg="white")
                self.left_image_label.image = self.current_tk_image
                self.left_image_label.place(x=0, y=0)

                self.gif_label = tk.Label(self.root, bg="white")
                self.gif_label.place(x=0, y=self.IMAGE_HEIGHT + 10)

                # Update instructions
                self.instructions_text = self.instructions_list[self.current_image_index]
                if hasattr(self, 'instructions_label'):
                    self.instructions_label.config(text=self.instructions_text)

                # Start GIF animation
                if tk_gif_frames:
                    self.animate_gif(tk_gif_frames, 0)

                return self.instructions_text
            except Exception as e:
                print(f"Error displaying image: {e}")
                return "Error loading instructions"
        else:
            print("Invalid index or end of images reached")
            return "No instructions available"

    def load_gif(self, gif_path):
        """Load GIF frames from file"""
        try:
            gif_image = Image.open(gif_path)
            frames = [self.resize_frame(frame) for frame in ImageSequence.Iterator(gif_image)]
            return frames
        except Exception as e:
            print(f"Error loading GIF: {e}")
            return []

    def resize_frame(self, frame):
        """Resize a frame to standard dimensions"""
        desired_width = self.IMAGE_WIDTH
        desired_height = self.IMAGE_HEIGHT
        resized_frame = frame.resize((desired_width, desired_height), resample=Image.LANCZOS)
        return ImageTk.PhotoImage(resized_frame)

    def animate_gif(self, frames, count=0):
        """Animate GIF frames"""
        if hasattr(self, 'gif_label') and frames:
            frame = frames[count % len(frames)]
            self.gif_label.config(image=frame)
            count = (count + 1) % len(frames)
            if hasattr(self, '_camera_running') and self._camera_running:
                self.root.after(1, lambda: self.animate_gif(frames, count))

    def reset_servos(self):
        """Reset servos to center position"""
        servo_x.angle = 0
        servo_y.angle = 0
        self.current_servo_positions = {'x': 0, 'y': 0}

    def move_to_position(self, index=None):
        """Move servos to predefined position for current or specified index"""
        if index is None:
            index = self.current_image_index
        
        if 0 <= index < len(SERVO_POSITIONS):
            target_pos = SERVO_POSITIONS[index]
            servo_x.angle = target_pos['x']
            servo_y.angle = target_pos['y']
            self.current_servo_positions = target_pos.copy()
            return True
        return False

    def update_timer_display(self):
        """Update the auto-capture timer display"""
        try:
            if self.auto_capture_active:
                remaining = max(0, self.next_capture_time - time.time())
                self.timer_label.config(text=f"Next auto capture in: {int(remaining)}s", fg="green")
                self.auto_capture_button.config(text="Auto Capture ON", bg="green")
            else:
                self.timer_label.config(text="Auto Capture: OFF", fg="black")
                self.auto_capture_button.config(text="Auto Capture OFF", bg="orange")
        except Exception as e:
            print(f"[TIMER UI ERROR]: {e}")

        # Recursively update every 1s if camera is running or auto capture is on
        try:
            if self.auto_capture_active or (hasattr(self, '_camera_running') and self._camera_running):
                self.root.after(1000, self.update_timer_display)
        except Exception as e:
            print(f"[AFTER LOOP ERROR]: {e}")


    def toggle_auto_capture(self):
        """Toggle auto-capture mode"""
        self.auto_capture_active = not self.auto_capture_active

        if self.auto_capture_active:
            self.last_capture_time = time.time()
            self.next_capture_time = self.last_capture_time + self.auto_capture_interval
            self.tracking_active = False
            if hasattr(self, "tracking_status"):
                self.tracking_status.config(text="Tracking: INACTIVE", fg="red")
            try:
                messagebox.showinfo("Auto Capture", f"Auto capture enabled - will capture every {self.auto_capture_interval} seconds")
            except Exception as e:
                print(f"[INFO BOX ERROR]: {e}")
        else:
            self.tracking_active = True
            if hasattr(self, "tracking_status"):
                self.tracking_status.config(text="Tracking: ACTIVE", fg="green")

        try:
            self.update_timer_display()
        except Exception as e:
            print(f"[TIMER DISPLAY ERROR]: {e}")


    def capture_and_show_popup(self):
        """Capture image and show preview popup"""
        try:
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture image from USB camera.")
                return

            # Convert and resize the captured image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            pil_image_resized = pil_image.resize((self.IMAGE_CAPTURE_VIEWER_WIDTH, self.IMAGE_CAPTURE_VIEWER_HEIGHT), 
                                               resample=Image.LANCZOS)
            
            # Store the image for display and saving
            self.captured_image = ImageTk.PhotoImage(pil_image_resized)
            self.captured_image_pil = pil_image_resized
            self.original_captured_image = pil_image  # Store full resolution image

            self.show_popup()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture image: {str(e)}")

    def show_popup(self):
        """Show image preview popup with save options"""
        popup = tk.Toplevel(self.root)
        popup.title("Captured Image")
        popup.configure(bg="white")
        popup.geometry("800x600")

        # Left image (reference)
        left_image_path = self.image_list[self.current_image_index]
        try:
            left_image = Image.open(left_image_path)
            left_image = left_image.resize((self.IMAGE_WIDTH, self.IMAGE_HEIGHT), resample=Image.LANCZOS)
            tk_left_image = ImageTk.PhotoImage(left_image)
            
            left_frame = tk.Frame(popup, bg="white")
            left_frame.pack(side=tk.LEFT, padx=10, pady=10)
            
            tk.Label(left_frame, text="Reference Image", font=("Helvetica", 12)).pack()
            popup_left_label = tk.Label(left_frame, image=tk_left_image, bg="white")
            popup_left_label.image = tk_left_image
            popup_left_label.pack()
        except Exception as e:
            print(f"Error loading reference image: {e}")

        # Captured image
        right_frame = tk.Frame(popup, bg="white")
        right_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        tk.Label(right_frame, text="Your Image", font=("Helvetica", 12)).pack()
        popup_captured_label = tk.Label(right_frame, image=self.captured_image, bg="white")
        popup_captured_label.image = self.captured_image
        popup_captured_label.pack()

        # Buttons
        button_frame = tk.Frame(popup, bg="white")
        button_frame.pack(side=tk.BOTTOM, pady=20)
        
        tk.Button(button_frame, 
                 text="Retake", 
                 command=lambda: [popup.destroy(), self.capture_and_show_popup()],
                 font=("Helvetica", 12), 
                 bg="#e74c3c", 
                 fg="white").pack(side=tk.LEFT, padx=20)
        
        tk.Button(button_frame, 
                 text="Save", 
                 command=lambda: [self.save_current_image(popup), popup.destroy()],
                 font=("Helvetica", 12), 
                 bg="#2ecc71", 
                 fg="white").pack(side=tk.LEFT, padx=15)
        
        tk.Button(button_frame, 
                 text="Save & Next", 
                 command=lambda: self.save_and_next(popup, self.emailid),
                 font=("Helvetica", 12), 
                 bg="#3498db", 
                 fg="white").pack(side=tk.LEFT, padx=10)

        popup.attributes('-topmost', True)
        popup.grab_set()

    def auto_capture_image(self):
        """Automatically capture image at predefined position"""
        try:
            if self.current_image_index + 1 >= len(self.image_list):
                # No more images to capture
                self.auto_capture_active = False
                self._camera_running = False
                if hasattr(self, 'cap') and self.cap:
                    self.cap.release()
                self.update_timer_display()
                self.show_analysis()
                return

            # Try to move to position first
            if not self.move_to_position():
                messagebox.showerror("Error", "Invalid position for auto-capture")
                return

            time.sleep(1)  # Can consider using after() instead for UI responsiveness

            # Capture image
            if not hasattr(self, 'cap') or self.cap is None:
                messagebox.showerror("Error", "Camera not initialized.")
                return

            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture image from USB camera.")
                return

            # Convert and process image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            self.captured_image_pil = pil_image.resize(
                (self.IMAGE_CAPTURE_VIEWER_WIDTH, self.IMAGE_CAPTURE_VIEWER_HEIGHT),
                resample=Image.LANCZOS
            )
            self.original_captured_image = pil_image

            self.flash_screen()
            self.save_current_image(auto_save=True)

            # Advance to next image
            self.current_image_index += 1
            self.display_left_image()
            self.last_capture_time = time.time()
            self.next_capture_time = self.last_capture_time + self.auto_capture_interval

            # If this was the last image, finalize
            if self.current_image_index >= len(self.image_list):
                self.auto_capture_active = False
                self._camera_running = False
                if hasattr(self, 'cap') and self.cap:
                    self.cap.release()
                self.update_timer_display()
                self.show_analysis()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to auto-capture image: {str(e)}")

    def save_current_image(self, popup=None, auto_save=False):
        """Save current image to database"""
        if not hasattr(self, 'captured_image_pil'):
            if not auto_save:
                messagebox.showerror("Error", "No image to save. Please capture an image first.")
            return
            
        try:
            # Convert image to bytes
            image_bytes_io = io.BytesIO()
            self.original_captured_image.convert("RGB").save(image_bytes_io, format='JPEG')
            image_bytes = image_bytes_io.getvalue()

            current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
            
            # Get phone number for image identifier
            cursor = self.connection.cursor()
            cursor.execute("SELECT contact FROM patient WHERE id = %s", (self.patient_id,))
            phone_number = cursor.fetchone()[0]
            
            image_identifier = self.get_next_identifier(phone_number, current_datetime, self.patient_id)

            query = """INSERT INTO patient_images 
                    (patient_id, image_number, image_data, image_identifier, created_at) 
                    VALUES (%s, %s, %s, %s, NOW())"""
            cursor.execute(query, (self.patient_id, self.current_image_index + 1, image_bytes, image_identifier))
            self.connection.commit()
            
            if not auto_save:
                messagebox.showinfo("Success", "Image saved successfully.")
            
            if popup:
                popup.destroy()
                
        except Exception as e:
            if not auto_save:
                messagebox.showerror("Database Error", f"Failed to save image: {str(e)}")
        finally:
            if 'cursor' in locals():
                cursor.close()

    def save_and_next(self, popup, emailid):
        #"""Save current image and move to next position"""
        try:
            # Convert image to bytes
            image_bytes_io = io.BytesIO()
            self.original_captured_image.convert("RGB").save(image_bytes_io, format='JPEG')
            image_bytes = image_bytes_io.getvalue()

            image_number = self.current_image_index + 1
            current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
            
            # Get phone number
            cursor = self.connection.cursor()
            cursor.execute("SELECT contact FROM patient WHERE id = %s", (self.patient_id,))
            phone_number = cursor.fetchone()[0]
            
            image_identifier = self.get_next_identifier(phone_number, current_datetime, self.patient_id)

            query = """INSERT INTO patient_images 
                    (patient_id, image_number, image_data, image_identifier, created_at) 
                    VALUES (%s, %s, %s, %s, NOW())"""
            cursor.execute(query, (self.patient_id, image_number, image_bytes, image_identifier))
            self.connection.commit()
            messagebox.showinfo("Success", "Image saved successfully.")
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to save image: {str(e)}")
        finally:
            if 'cursor' in locals():
                cursor.close()
            popup.destroy()

            # Move to next image or finish
            # Move to next image or finish
            if self.current_image_index + 1 < len(self.image_list):
                self.current_image_index += 1
                self.display_left_image()
                self.move_to_position()
            else:
                # All images captured - stop camera
                self._camera_running = False
                if hasattr(self, 'cap'):
                    self.cap.release()
                self.show_analysis()
                    

    def get_next_identifier(self, phone_number, current_datetime, patient_id):
        """Generate unique identifier for saved images"""
        cursor = self.connection.cursor()
        current_date = current_datetime[:8]
        formatted_date = f"{current_date[:4]}-{current_date[4:6]}-{current_date[6:8]}"

        query = """
        SELECT MAX(image_identifier) FROM patient_images 
        WHERE patient_id = %s AND DATE(created_at) = %s
        """
        cursor.execute(query, (patient_id, formatted_date))
        result = cursor.fetchone()[0]

        if result:
            try:
                prefix, last_char = result.rsplit('_', 1)
                next_char = chr(ord(last_char) + 1) if last_char != 'Z' else 'A'
                new_identifier = f"{prefix}_{next_char}"
            except:
                new_identifier = f"{phone_number}_{formatted_date.replace('-', '')}A"
        else:
            new_identifier = f"{phone_number}_{formatted_date.replace('-', '')}A"

        cursor.close()
        return new_identifier

    def show_analysis(self):
        """Show analysis completion screen"""
        # Stop camera
        self._camera_running = False
        if hasattr(self, 'cap'):
            self.cap.release()

        # Hide all other UI elements
        self.left_image_label.place_forget()
        self.gif_label.place_forget()
        self.instructions_label_frame.place_forget()
        self.right_frame.place_forget()
        
        # Create analysis frame
        self.analysis_frame = tk.Frame(self.root, bg="white", width=1300, height=920)
        self.analysis_frame.place(x=0, y=0)
        
        # Add analysis message
        tk.Label(self.analysis_frame,  
                text="Analyzing Images...", 
                font=("times new roman", 36, 'bold'), bg="white", fg="blue").pack(pady=100)
        
        # Add progress bar (simulated)
        self.progress = tk.ttk.Progressbar(self.analysis_frame, orient="horizontal", 
                                         length=920, mode="determinate")
        self.progress.pack()
        self.progress["value"] = 0
        self._progress_active = True  # Add this flag
        self.update_progress()
        
        # Hide navigation buttons
        self.prev_button.pack_forget()
        self.next_button.pack_forget()
        self.capture_button.pack_forget()
        self.auto_capture_button.pack_forget()
        
        # Show thank you message after delay
        self.root.after(5000, self.show_thank_you)

    def update_progress(self):
        """Update progress bar animation"""
        if self.progress["value"] < 100:
            self.progress["value"] += 5
            self.root.after(200, self.update_progress)

    def show_thank_you(self):
        """Show thank you message with report viewer"""
        self._progress_active = False

        # First generate the report
        report_gen = ReportGenerator(self.patient_id, self.connection)
        report_gen.analyze_images()
        report_path = report_gen.generate_report()
        pdf_path = report_gen.generate_pdf_report(report_path)
        
        # Get just the filename from the full path
        report_filename = os.path.basename(report_path)
        
        # Clear the current UI
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Create thank you frame with report viewer
        thank_you_frame = tk.Frame(self.root, bg="white")
        thank_you_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add thank you message
        tk.Label(thank_you_frame, 
                 text="Examination Complete!", 
                 font=("times new roman", 24, 'bold'), 
                 bg="white", fg="green").pack(pady=20)
        
        # Add report viewing frame
        report_viewer_frame = tk.Frame(thank_you_frame, bg="white")
        report_viewer_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Add web browser frame
        self.browser_frame = tk.Frame(report_viewer_frame, bg="white")
        self.browser_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = tk.Scrollbar(self.browser_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add HTML viewer
        self.report_text = tk.Text(
            self.browser_frame, 
            wrap=tk.WORD, 
            yscrollcommand=scrollbar.set,
            font=("Arial", 10),
            padx=10,
            pady=10
        )
        self.report_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.report_text.yview)
        
        # Load the HTML content into the text widget
        try:
            with open(report_path, 'r') as f:
                html_content = f.read()
                # Basic HTML stripping for display (you might want to improve this)
                text_content = html_content.replace('<', ' <').replace('>', '> ')  # Simple space addition to separate tags
                self.report_text.insert(tk.END, text_content)
                self.report_text.config(state=tk.DISABLED)  # Make it read-only
        except Exception as e:
            print(f"Error loading report: {e}")
            self.report_text.insert(tk.END, "Could not load report content")
        
        # Add button frame
        button_frame = tk.Frame(thank_you_frame, bg="white")
        button_frame.pack(pady=20)
        
        # Add buttons
        tk.Button(button_frame, 
                  text="View in Browser", 
                  command=lambda: webbrowser.open(f"http://localhost:5000/reports/{report_filename}"),
                  font=("times new roman", 14), 
                  bg="#3498db", 
                  fg="white").pack(side=tk.LEFT, padx=10)
        
        tk.Button(button_frame, 
                  text="Download PDF", 
                  command=lambda: webbrowser.open(f"file://{pdf_path}"),
                  font=("times new roman", 14), 
                  bg="#2ecc71", 
                  fg="white").pack(side=tk.LEFT, padx=10)
        
        tk.Button(button_frame, 
                  text="Email Report", 
                  command=lambda: self.send_email_report(pdf_path),
                  font=("times new roman", 14), 
                  bg="#9b59b6", 
                  fg="white").pack(side=tk.LEFT, padx=10)
        
        tk.Button(button_frame, 
                  text="Exit", 
                  command=self.cleanup_and_exit,
                  font=("times new roman", 14, 'bold'), 
                  bg="#e74c3c", 
                  fg="white").pack(side=tk.LEFT, padx=10)

    # Add this method to your ImageCaptureApp class for email functionality
    def send_email_report(self, pdf_path):
        """Send the PDF report via email"""
        try:
            # Get patient email from database
            cursor = self.connection.cursor()
            cursor.execute("SELECT email FROM patient WHERE id = %s", (self.patient_id,))
            patient_email = cursor.fetchone()[0]
            
            if not patient_email:
                messagebox.showerror("Error", "No email found for this patient")
                return
            
            # Email configuration (you'll need to set these up)
            smtp_server = "smtp.gmail.com"  # Example with Gmail
            smtp_port = 587
            sender_email = "rakeshozon46@gmail.com"
            sender_password = "knovzqbmcryhoznh"  # Consider using app-specific password
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = patient_email
            msg['Subject'] = "Your Oral Health Report"
            
            # Email body
            body = """Please find attached your oral health examination report.
            
    Best regards,
    Your Dental Clinic"""
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach PDF
            with open(pdf_path, "rb") as f:
                attach = MIMEApplication(f.read(), _subtype="pdf")
                attach.add_header('Content-Disposition', 'attachment', filename=os.path.basename(pdf_path))
                msg.attach(attach)
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
            messagebox.showinfo("Success", "Report sent successfully via email")
        except Exception as e:
            messagebox.showerror("Email Error", f"Failed to send email: {str(e)}")
        finally:
            if 'cursor' in locals():
                cursor.close()
            # Add this method to your ImageCaptureApp class for email functionality
    def send_email_report(self, pdf_path):
        """Send the PDF report via email"""
        try:
            # Get patient email from database
            cursor = self.connection.cursor()
            cursor.execute("SELECT email FROM patient WHERE id = %s", (self.patient_id,))
            patient_email = cursor.fetchone()[0]
            
            if not patient_email:
                messagebox.showerror("Error", "No email found for this patient")
                return
            
            # Email configuration (you'll need to set these up)
            smtp_server = "smtp.gmail.com"  # Example with Gmail
            smtp_port = 587
            sender_email = "your_email@gmail.com"
            sender_password = "your_password"  # Consider using app-specific password
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = patient_email
            msg['Subject'] = "Your Oral Health Report"
            
            # Email body
            body = """Please find attached your oral health examination report.
            
    Best regards,
    Your Dental Clinic"""
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach PDF
            with open(pdf_path, "rb") as f:
                attach = MIMEApplication(f.read(), _subtype="pdf")
                attach.add_header('Content-Disposition', 'attachment', filename=os.path.basename(pdf_path))
                msg.attach(attach)
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
            messagebox.showinfo("Success", "Report sent successfully via email")
        except Exception as e:
            messagebox.showerror("Email Error", f"Failed to send email: {str(e)}")
        finally:
            if 'cursor' in locals():
                cursor.close()
    
    def cleanup_and_exit(self):
        """Clean up resources and exit"""
        self.cleanup()
        self.root.destroy()

    def process_frame_for_tracking(self, frame):
        """Process frame for face tracking and motor control"""
        frame_height, frame_width = frame.shape[:2]
        center_x = frame_width // 2
        center_y = frame_height // 2

        if self.tracking_active:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # Draw face bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Get face center
                face_center_x = x + w // 2
                face_center_y = y + h // 2

                # Calculate offsets from center
                offset_x = face_center_x - center_x
                offset_y = face_center_y - center_y

                # Adjust motors based on offset
                self.adjust_motors(offset_x, offset_y)

                # Mouth detection
                roi_gray = gray[y + h//2:y + h, x:x + w]
                mouths = mouth_cascade.detectMultiScale(roi_gray, 1.7, 11)

                for (mx, my, mw, mh) in mouths:
                    mouth_x = x + mx + mw // 2
                    mouth_y = y + h//2 + my + mh // 2
                    
                    # Fine-tune alignment to mouth
                    mouth_offset_x = mouth_x - center_x
                    mouth_offset_y = mouth_y - center_y

                    if abs(mouth_offset_x) > 10:
                        self.adjust_motors(mouth_offset_x, 0, fine_tune=True)
                    if abs(mouth_offset_y) > 10:
                        self.adjust_motors(0, mouth_offset_y, fine_tune=True)
                    break  # Process only one mouth

        # Draw center lines
        cv2.line(frame, (0, center_y), (frame_width, center_y), (0, 255, 255), 1)
        cv2.line(frame, (center_x, 0), (center_x, frame_height), (0, 255, 255), 1)
        
        return frame

    def adjust_motors(self, offset_x, offset_y, fine_tune=False):
        """Adjust both servos and steppers based on offset"""
        # Servo adjustments
        if abs(offset_x) > (10 if fine_tune else 20):
            adjustment = -1 if offset_x > 0 else 1
            if fine_tune:
                adjustment *= 0.5  # Smaller adjustments for fine tuning
            new_angle_x = max(-90, min(90, (servo_x.angle or 0) + adjustment))
            servo_x.angle = new_angle_x
            self.current_servo_positions['x'] = new_angle_x
            
            # Stepper motor adjustment for X axis using DRV8825
            steps = int(abs(offset_x)/5)
            if steps > 0:
                MotorX.SetMicroStep('software', '1/8step')
                MotorX.TurnStep(
                    Dir='forward' if offset_x < 0 else 'backward',
                    steps=steps,
                    stepdelay=0.001 if fine_tune else 0.002
                )
                MotorX.Stop()
        
        if abs(offset_y) > (10 if fine_tune else 20):
            adjustment = -1 if offset_y > 0 else 1
            if fine_tune:
                adjustment *= 0.5  # Smaller adjustments for fine tuning
            new_angle_y = max(-90, min(90, (servo_y.angle or 0) + adjustment))
            servo_y.angle = new_angle_y
            self.current_servo_positions['y'] = new_angle_y
            
            # Stepper motor adjustment for Y axis using DRV8825
            steps = int(abs(offset_y)/5)
            if steps > 0:
                MotorY.SetMicroStep('software', '1/8step')
                MotorY.TurnStep(
                    Dir='forward' if offset_y < 0 else 'backward',
                    steps=steps,
                    stepdelay=0.001 if fine_tune else 0.002
                )
                MotorY.Stop()


    def update_camera_display(self, imgtk):
        """Update camera display in main thread"""
        if hasattr(self, 'camera_label'):
            self.camera_label.imgtk = imgtk  # Keep reference
            self.camera_label.config(image=imgtk)

    def flash_screen(self):
        """Briefly flash the camera display to indicate capture"""
        original_bg = self.camera_label.cget('bg')
        self.camera_label.config(bg='white')
        self.root.update()
        time.sleep(0.1)
        self.camera_label.config(bg=original_bg)

    def show_camera_feed(self):
        #"""Show live camera feed with face tracking"""
        def update_feed():
            while getattr(self, '_camera_running', False):
                try:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to read frame")
                        time.sleep(0.1)
                        continue

                    # Handle auto-capture timing
                    current_time = time.time()
                    if self.auto_capture_active and current_time >= self.next_capture_time:
                        self.last_capture_time = current_time
                        self.next_capture_time = current_time + self.auto_capture_interval
                        self.root.after(0, self.auto_capture_image)

                    # Convert frame to RGB for display
                    try:
                        if frame is not None:
                            # Ensure we have a 3-channel BGR image
                            if len(frame.shape) == 2:  # Grayscale
                                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                            elif frame.shape[2] == 4:  # RGBA
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                            
                            # Convert to RGB for display
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            
                            # Face detection and tracking
                            frame = self.process_frame_for_tracking(frame)
                            
                            # Resize for display
                            frame = cv2.resize(frame, (640, 480))
                            
                            # Convert to PhotoImage
                            img = Image.fromarray(frame)
                            imgtk = ImageTk.PhotoImage(image=img)
                            
                            # Update display in main thread
                            self.root.after(0, lambda: self.update_camera_display(imgtk))
                            
                    except Exception as conv_error:
                        print(f"Frame conversion error: {conv_error}")
                        continue

                    time.sleep(0.03)  # ~30 FPS
                except Exception as e:
                    print(f"Camera feed error: {e}")
                    continue

        # Start the thread
        self._camera_running = True
        if not hasattr(self, 'camera_thread') or not self.camera_thread.is_alive():
            self.camera_thread = threading.Thread(target=update_feed, daemon=True)
            self.camera_thread.start()
        self.update_timer_display()

    # Remove the duplicate send_email_report method (keep only one)
    def send_email_report(self, pdf_path):
        """Send the PDF report via email"""
        try:
            # Get patient email from database
            cursor = self.connection.cursor()
            cursor.execute("SELECT email FROM patient WHERE id = %s", (self.patient_id,))
            patient_email = cursor.fetchone()[0]
            
            if not patient_email:
                messagebox.showerror("Error", "No email found for this patient")
                return
            
            # Email configuration (you'll need to set these up)
            smtp_server = "smtp.gmail.com"  # Example with Gmail
            smtp_port = 587
            sender_email = "rakeshozon46@gmail.com"
            sender_password = "knovzqbmcryhoznh"  # Consider using app-specific password
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = patient_email
            msg['Subject'] = "Your Oral Health Report"
            
            # Email body
            body = """Please find attached your oral health examination report.
            
    Best regards,
    Your Dental Clinic"""
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach PDF
            with open(pdf_path, "rb") as f:
                attach = MIMEApplication(f.read(), _subtype="pdf")
                attach.add_header('Content-Disposition', 'attachment', filename=os.path.basename(pdf_path))
                msg.attach(attach)
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
            messagebox.showinfo("Success", "Report sent successfully via email")
        except Exception as e:
            messagebox.showerror("Email Error", f"Failed to send email: {str(e)}")
        finally:
            if 'cursor' in locals():
                cursor.close()
    def cleanup(self):
        """Clean up resources before closing"""
        
        self._progress_active = False 
        self._camera_running = False
        if hasattr(self, 'cap'):
            self.cap.release()
        self._camera_running = False
        self.camera_image = None  # Clear the reference when closing

        if hasattr(self, 'report_server'):
            self.report_server.stop()
        if hasattr(self, 'camera_thread'):
            self.camera_thread.join(timeout=1)
        
        if hasattr(self, 'cap'):
            self.cap.release()
        
        servo_x.close()
        servo_y.close() 
        MotorX.Stop()
        MotorY.Stop()
        GPIO.cleanup()
        
        if hasattr(self, 'connection') and self.connection:
            self.connection.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCaptureApp(root, "example@example.com")
    self.setup_report_server()
    #self.current_servo_positions = {'x': 30, 'y': 0}
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

#this is my  working code , for automated controlled camera for my oral scan ,
