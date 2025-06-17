import io
import tkinter as tk
from tkinter import Toplevel, messagebox
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

# Initialize GPIO
h = lgpio.gpiochip_open(0)

# Servo Configuration
SERVO_X_PIN = 17  # BCM 17 for pan servo
SERVO_Y_PIN = 27  # BCM 27 for tilt servo

# Claim pins as outputs for servo PWM
lgpio.gpio_claim_output(h, SERVO_X_PIN)
lgpio.gpio_claim_output(h, SERVO_Y_PIN)

class StableServo:
    def __init__(self, pin):
        self.pin = pin
        self.current_angle = 90
    
    def move_to_angle(self, angle):
        """Move to specified angle (0-180) and then detach servo signal"""
        angle = max(0, min(180, angle))
        pulsewidth = int(500 + (angle / 180.0) * 2000)  # 500-2500μs
        lgpio.tx_servo(h, self.pin, pulsewidth)  # Attach and move servo
        time.sleep(0.3)  # Allow time to reach position
        lgpio.tx_servo(h, self.pin, 0)  # Detach servo (stop PWM signal)
        self.current_angle = angle
        
    def close(self):
        """Stop servo signal"""
        lgpio.tx_servo(h, self.pin, 0)

# Initialize servos with stable control
servo_x = StableServo(SERVO_X_PIN)
servo_y = StableServo(SERVO_Y_PIN)

def reset_servos():
    """Reset servos to center position"""
    servo_x.move_to_angle(90)
    servo_y.move_to_angle(90)

def cleanup_servos():
    """Clean up servo resources"""
    servo_x.close()
    servo_y.close()
    lgpio.gpiochip_close(h)

# Stepper motor control
MotorDir = [
    'forward',
    'backward',
]

ControlMode = [
    'hardward',
    'softward',
]

class DRV8825:
    def __init__(self, dir_pin, step_pin, enable_pin, mode_pins):
        self.dir_pin = dir_pin
        self.step_pin = step_pin        
        self.enable_pin = enable_pin
        self.mode_pins = mode_pins if isinstance(mode_pins, (list, tuple)) else [mode_pins]

        self.h = lgpio.gpiochip_open(0)

        # Claim all pins as outputs
        lgpio.gpio_claim_output(self.h, self.dir_pin)
        lgpio.gpio_claim_output(self.h, self.step_pin)
        lgpio.gpio_claim_output(self.h, self.enable_pin)
        for pin in self.mode_pins:
            lgpio.gpio_claim_output(self.h, pin)

    def digital_write(self, pin, value):
        lgpio.gpio_write(self.h, pin, value)

    def Stop(self):
        self.digital_write(self.enable_pin, 0)

    def SetMicroStep(self, mode, stepformat):
        microstep = {
            'fullstep': (0, 0, 0),
            'halfstep': (1, 0, 0),
            '1/4step': (0, 1, 0),
            '1/8step': (1, 1, 0),
            '1/16step': (0, 0, 1),
            '1/32step': (1, 0, 1)
        }

        if mode == ControlMode[1]:  # software
            values = microstep.get(stepformat)
            if values and len(values) == len(self.mode_pins):
                for pin, val in zip(self.mode_pins, values):
                    self.digital_write(pin, val)
            else:
                print("Invalid step format or mode pin count mismatch.")

    def TurnStep(self, Dir, steps, stepdelay=0.005):
        if Dir == MotorDir[0]:  # forward
            self.digital_write(self.enable_pin, 1)
            self.digital_write(self.dir_pin, 0)
        elif Dir == MotorDir[1]:  # backward
            self.digital_write(self.enable_pin, 1)
            self.digital_write(self.dir_pin, 1)
        else:
            self.digital_write(self.enable_pin, 0)
            return

        for _ in range(steps):
            self.digital_write(self.step_pin, 1)
            time.sleep(stepdelay)
            self.digital_write(self.step_pin, 0)
            time.sleep(stepdelay)

    def cleanup(self):
        self.Stop()
        lgpio.gpiochip_close(self.h)

# Initialize motor drivers
try:
    MotorX = DRV8825(dir_pin=13, step_pin=19, enable_pin=25, mode_pins=(16, 5, 20))
    MotorY = DRV8825(dir_pin=24, step_pin=18, enable_pin=23, mode_pins=(21, 22, 6))
except Exception as e:
    print(f"Motor initialization error: {e}")

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
        self.root.title("Oral Image Capture System")
        self.root.geometry("1300x920")
        self.root.configure(bg="white")
        
        # Initialize variables
        self.current_image_index = 0
        self._camera_running = True
        
        # Constants for UI
        self.IMAGE_WIDTH = 300
        self.IMAGE_HEIGHT = 300
        self.CAMERA_WIDTH = 640
        self.CAMERA_HEIGHT = 480
              
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
3. Tilt head backward''']
        # Setup database connection
        self.setup_database()
        
        # Setup UI components
        self.setup_ui()
        
        # Initialize camera
        self.setup_camera()
        
        # Center servos initially
        reset_servos()
    
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
    def setup_database(self):
        """Initialize database connection"""
        try:
            self.connection = mysql.connector.connect(
                host="localhost",
                user="root",
                password="123456",
                database="register"
            )
            # Get patient ID
            self.patient_id = self.get_patient_id(self.emailid)
            if self.patient_id is None:
                messagebox.showerror("Error", "Patient not found in database")
                self.root.destroy()
        except Error as e:
            messagebox.showerror("Database Error", f"Error: {e}")
            self.root.destroy()

    def setup_ui(self):
        """Set up all UI components"""
        # Main frames
        self.setup_left_panel()  # Instructions and reference images
        self.setup_right_panel()  # Camera feed and controls
        self.setup_bottom_controls()  # Navigation buttons
        
        # Display first image and instructions
        self.display_current_view()

    def setup_left_panel(self):
        """Left panel with instructions and reference images"""
        # Main frame
        self.left_frame = tk.Frame(self.root, bg="white", width=400, height=800)
        self.left_frame.pack_propagate(False)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Reference image display
        self.ref_image_frame = tk.Frame(self.left_frame, bg="white")
        self.ref_image_frame.pack(fill=tk.X, pady=5)
        
        self.ref_image_label = tk.Label(self.ref_image_frame, bg="white")
        self.ref_image_label.pack()
        
        # GIF demonstration
        self.gif_frame = tk.Frame(self.left_frame, bg="white")
        self.gif_frame.pack(fill=tk.X, pady=5)
        
        self.gif_label = tk.Label(self.gif_frame, bg="white")
        self.gif_label.pack()
        
        # Instructions frame
        self.instructions_frame = tk.LabelFrame(
            self.left_frame, 
            text="INSTRUCTIONS", 
            font=("Arial", 12, "bold"),
            bg="white",
            padx=10,
            pady=10
        )
        self.instructions_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Scrollable instructions
        self.instructions_canvas = tk.Canvas(
            self.instructions_frame, 
            bg="white", 
            highlightthickness=0
        )
        self.scrollbar = tk.Scrollbar(
            self.instructions_frame, 
            orient="vertical", 
            command=self.instructions_canvas.yview
        )
        self.instructions_scroll_frame = tk.Frame(self.instructions_canvas, bg="white")
        
        self.instructions_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.instructions_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.instructions_canvas.create_window(
            (0, 0), 
            window=self.instructions_scroll_frame, 
            anchor="nw",
            tags="frame"
        )
        
        self.instructions_scroll_frame.bind(
            "<Configure>",
            lambda e: self.instructions_canvas.configure(
                scrollregion=self.instructions_canvas.bbox("all")
            )
        )
        
        self.instructions_label = tk.Label(
            self.instructions_scroll_frame,
            text="",
            bg="white",
            font=("Arial", 11),
            wraplength=350,
            justify="left"
        )
        self.instructions_label.pack(fill=tk.BOTH, expand=True)

    def setup_right_panel(self):
        """Right panel with camera feed and controls"""
        self.right_frame = tk.Frame(self.root, bg="white")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Camera feed display
        self.camera_frame = tk.LabelFrame(
            self.right_frame,
            text="Camera Feed",
            font=("Arial", 12, "bold"),
            bg="white"
        )
        self.camera_frame.pack(fill=tk.BOTH, expand=True)
        
        self.camera_label = tk.Label(self.camera_frame, bg="black")
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Motor controls
        self.setup_motor_controls()

    def setup_motor_controls(self):
        """Setup servo and stepper motor controls"""
        # Servo Control Frame
        servo_frame = tk.LabelFrame(
            self.right_frame,
            text="Servo Controls",
            font=("Arial", 12, "bold"),
            bg="white"
        )
        servo_frame.pack(fill=tk.X, pady=5)
        
        # Pan (X-axis) controls
        pan_frame = tk.Frame(servo_frame, bg="white")
        pan_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        tk.Label(pan_frame, text="Pan (X-axis)", bg="white").pack()
        
        btn_frame = tk.Frame(pan_frame, bg="white")
        btn_frame.pack()
        
        tk.Button(btn_frame, text="◄ Left", 
                 command=lambda: move_servo_x(-10), 
                 width=8).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Right ►", 
                 command=lambda: move_servo_x(10), 
                 width=8).pack(side=tk.LEFT, padx=2)
        
        # Tilt (Y-axis) controls
        tilt_frame = tk.Frame(servo_frame, bg="white")
        tilt_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        tk.Label(tilt_frame, text="Tilt (Y-axis)", bg="white").pack()
        
        btn_frame = tk.Frame(tilt_frame, bg="white")
        btn_frame.pack()
        
        tk.Button(btn_frame, text="▲ Up", 
                 command=lambda: move_servo_y(-10), 
                 width=8).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="▼ Down", 
                 command=lambda: move_servo_y(10), 
                 width=8).pack(side=tk.LEFT, padx=2)
        
        # Reset button
        tk.Button(servo_frame, text="Reset", 
                 command=reset_servos,
                 bg="#3498db", fg="white").pack(side=tk.RIGHT, padx=10)
        
        # Stepper Motor Frame
        stepper_frame = tk.LabelFrame(
            self.right_frame,
            text="Stepper Motor Controls",
            font=("Arial", 12, "bold"),
            bg="white"
        )
        stepper_frame.pack(fill=tk.X, pady=5)
        
        # Movement buttons
        control_frame = tk.Frame(stepper_frame, bg="white")
        control_frame.pack(pady=5)
        
        tk.Button(control_frame, text="◄ Left", 
                 command=lambda: stepper_move_x(-100), 
                 width=8).grid(row=1, column=0, padx=2, pady=2)
        tk.Button(control_frame, text="Right ►", 
                 command=lambda: stepper_move_x(100), 
                 width=8).grid(row=1, column=2, padx=2, pady=2)
        tk.Button(control_frame, text="▲ Up", 
                 command=lambda: stepper_move_y(-100), 
                 width=8).grid(row=0, column=1, padx=2, pady=2)
        tk.Button(control_frame, text="▼ Down", 
                 command=lambda: stepper_move_y(100), 
                 width=8).grid(row=2, column=1, padx=2, pady=2)

    def setup_bottom_controls(self):
        """Setup navigation and capture buttons"""
        self.control_frame = tk.Frame(self.root, bg="white")
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        # Navigation buttons
        self.prev_btn = tk.Button(
            self.control_frame,
            text="◄ Previous",
            command=self.prev_image,
            font=("Arial", 12),
            bg="#3498db",
            fg="white",
            width=12
        )
        self.prev_btn.pack(side=tk.LEFT, padx=10)
        
        self.next_btn = tk.Button(
            self.control_frame,
            text="Next ►",
            command=self.next_image,
            font=("Arial", 12),
            bg="#3498db",
            fg="white",
            width=12
        )
        self.next_btn.pack(side=tk.LEFT, padx=10)
        
        # Capture button
        self.capture_btn = tk.Button(
            self.control_frame,
            text="Capture Image",
            command=self.capture_image,
            font=("Arial", 12, "bold"),
            bg="#2ecc71",
            fg="white",
            width=15
        )
        self.capture_btn.pack(side=tk.LEFT, padx=20)
        
        # Initially disable previous button
        self.prev_btn.config(state=tk.DISABLED)

    def setup_camera(self):
        """Initialize USB camera"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open USB camera.")
            self.root.destroy()
            return
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_HEIGHT)
        
        # Start camera feed
        self.update_camera_feed()

    def update_camera_feed(self):
        """Update the camera feed display"""
        if self._camera_running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = img.resize((self.CAMERA_WIDTH, self.CAMERA_HEIGHT))
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.camera_label.imgtk = imgtk
                self.camera_label.config(image=imgtk)
            
            self.root.after(30, self.update_camera_feed)

    def display_current_view(self):
        """Display current reference image, GIF and instructions"""
        # Load static reference image
        img_path = self.image_list[self.current_image_index]
        img = Image.open(img_path)
        img = img.resize((self.IMAGE_WIDTH, self.IMAGE_HEIGHT), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        
        self.ref_image_label.config(image=img_tk)
        self.ref_image_label.image = img_tk
        
        # Load and animate GIF
        gif_path = self.gif_list[self.current_image_index]
        self.animate_gif(gif_path)
        
        # Update instructions
        instructions = self.instructions_list[self.current_image_index]
        self.instructions_label.config(text=instructions)
        
        # Update button states
        self.update_button_states()

    def animate_gif(self, gif_path):
        """Animate GIF demonstration"""
        try:
            gif = Image.open(gif_path)
            frames = []
            
            for frame in ImageSequence.Iterator(gif):
                frame = frame.resize((self.IMAGE_WIDTH, self.IMAGE_HEIGHT), Image.LANCZOS)
                frames.append(ImageTk.PhotoImage(frame))
            
            self.gif_frames = frames
            self.current_frame = 0
            self.animate_next_frame()
        except Exception as e:
            print(f"Error loading GIF: {e}")

    def animate_next_frame(self):
        """Show next frame of GIF animation"""
        if hasattr(self, 'gif_frames') and self.gif_frames:
            frame = self.gif_frames[self.current_frame % len(self.gif_frames)]
            self.gif_label.config(image=frame)
            self.gif_label.image = frame
            self.current_frame += 1
            self.root.after(100, self.animate_next_frame)

    def update_button_states(self):
        """Update navigation button states based on current position"""
        self.prev_btn.config(state=tk.NORMAL if self.current_image_index > 0 else tk.DISABLED)
        
        if self.current_image_index == len(self.image_list) - 1:
            self.next_btn.config(text="Finish", command=self.finish_capture)
        else:
            self.next_btn.config(text="Next ►", command=self.next_image)

    def prev_image(self):
        """Navigate to previous image view"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_current_view()
            reset_servos()

    def next_image(self):
        """Navigate to next image view"""
        if self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.display_current_view()
            reset_servos()

    def capture_image(self):
        """Capture and preview current camera image"""
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image")
            return
        
        # Convert and store image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.captured_image = Image.fromarray(frame_rgb)
        
        # Show preview popup
        self.show_capture_preview()

    def show_capture_preview(self):
        """Show preview of captured image with save options"""
        preview = Toplevel(self.root)
        preview.title("Image Preview")
        preview.geometry("800x600")
        preview.resizable(False, False)
        
        # Reference image
        ref_frame = tk.Frame(preview, bg="white")
        ref_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(ref_frame, text="Reference Image", font=("Arial", 12, "bold")).pack()
        
        ref_img = Image.open(self.image_list[self.current_image_index])
        ref_img = ref_img.resize((350, 350), Image.LANCZOS)
        ref_img_tk = ImageTk.PhotoImage(ref_img)
        
        ref_label = tk.Label(ref_frame, image=ref_img_tk, bg="white")
        ref_label.image = ref_img_tk
        ref_label.pack(pady=10)
        
        # Captured image
        cap_frame = tk.Frame(preview, bg="white")
        cap_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(cap_frame, text="Your Image", font=("Arial", 12, "bold")).pack()
        
        cap_img = self.captured_image.resize((350, 350), Image.LANCZOS)
        cap_img_tk = ImageTk.PhotoImage(cap_img)
        
        cap_label = tk.Label(cap_frame, image=cap_img_tk, bg="white")
        cap_label.image = cap_img_tk
        cap_label.pack(pady=10)
        
        # Buttons
        btn_frame = tk.Frame(preview, bg="white")
        btn_frame.pack(side=tk.BOTTOM, pady=20)
        
        tk.Button(
            btn_frame,
            text="Retake",
            command=lambda: [preview.destroy(), self.capture_image()],
            font=("Arial", 12),
            bg="#e74c3c",
            fg="white",
            width=10
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            btn_frame,
            text="Save",
            command=lambda: [self.save_image(), preview.destroy()],
            font=("Arial", 12),
            bg="#2ecc71",
            fg="white",
            width=10
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            btn_frame,
            text="Save & Next",
            command=lambda: [self.save_image(), preview.destroy(), self.next_image()],
            font=("Arial", 12),
            bg="#3498db",
            fg="white",
            width=12
        ).pack(side=tk.LEFT, padx=10)

    def save_image(self):
        """Save current captured image to database"""
        try:
            # Convert image to bytes
            img_bytes = io.BytesIO()
            self.captured_image.save(img_bytes, format="JPEG")
            img_data = img_bytes.getvalue()
            
            # Generate unique identifier
            current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
            cursor = self.connection.cursor()
            cursor.execute("SELECT contact FROM patient WHERE id = %s", (self.patient_id,))
            phone = cursor.fetchone()[0]
            img_id = self.generate_image_id(phone, current_datetime)
            
            # Save to database
            query = """
            INSERT INTO patient_images 
            (patient_id, image_number, image_data, image_identifier, created_at)
            VALUES (%s, %s, %s, %s, NOW())
            """
            cursor.execute(query, (
                self.patient_id,
                self.current_image_index + 1,  # 1-based index
                img_data,
                img_id
            ))
            self.connection.commit()
            
            messagebox.showinfo("Success", "Image saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {str(e)}")
        finally:
            if 'cursor' in locals():
                cursor.close()

    def generate_image_id(self, phone, datetime_str):
        """Generate unique image identifier"""
        date_part = datetime_str[:8]  # YYYYMMDD
        cursor = self.connection.cursor()
        
        query = """
        SELECT MAX(image_identifier) FROM patient_images 
        WHERE patient_id = %s AND DATE(created_at) = %s
        """
        cursor.execute(query, (self.patient_id, f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"))
        result = cursor.fetchone()[0]
        
        if result:
            base_id, last_char = result.rsplit('_', 1)
            next_char = chr(ord(last_char) + 1) if last_char != 'Z' else 'A'
            return f"{base_id}_{next_char}"
        else:
            return f"{phone}_{date_part}A"

    def finish_capture(self):
        """Finish image capture and generate report"""
        self._camera_running = False
        self.cap.release()
        
        # Show analysis screen
        self.show_analysis_screen()

    def show_analysis_screen(self):
        """Show analysis progress screen"""
        # Clear current UI
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Analysis frame
        analysis_frame = tk.Frame(self.root, bg="white")
        analysis_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(
            analysis_frame,
            text="Analyzing Images...",
            font=("Arial", 24, "bold"),
            bg="white",
            fg="#3498db"
        ).pack(pady=50)
        
        # Progress bar
        self.progress = tk.ttk.Progressbar(
            analysis_frame,
            orient="horizontal",
            length=600,
            mode="determinate"
        )
        self.progress.pack(pady=20)
        
        # Start analysis after short delay
        self.root.after(1000, self.perform_analysis)

    def perform_analysis(self):
        """Perform image analysis and generate report"""
        try:
            # Initialize report generator
            report_gen = ReportGenerator(self.patient_id, self.connection)
            
            # Simulate progress
            for i in range(0, 101, 10):
                self.progress["value"] = i
                self.root.update()
                time.sleep(0.2)
            
            # Analyze images
            report_gen.analyze_images()
            
            # Generate reports
            html_path = report_gen.generate_report()
            pdf_path = report_gen.generate_pdf_report(html_path)
            
            # Clean up temporary files
            report_gen.cleanup_temp_files()
            
            # Show results
            self.show_results(html_path, pdf_path)
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Failed to analyze images: {str(e)}")
            self.root.destroy()

    def show_results(self, html_path, pdf_path):
        """Show final results with report options"""
        # Clear analysis screen
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Results frame
        results_frame = tk.Frame(self.root, bg="white")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        tk.Label(
            results_frame,
            text="Examination Complete!",
            font=("Arial", 24, "bold"),
            bg="white",
            fg="#2ecc71"
        ).pack(pady=20)
        
        tk.Label(
            results_frame,
            text="Your oral examination has been completed successfully.",
            font=("Arial", 14),
            bg="white"
        ).pack(pady=10)
        
        # Options frame
        options_frame = tk.Frame(results_frame, bg="white")
        options_frame.pack(pady=30)
        
        # View Report button
        tk.Button(
            options_frame,
            text="View Report in Browser",
            command=lambda: webbrowser.open(html_path),
            font=("Arial", 12),
            bg="#3498db",
            fg="white",
            width=25
        ).pack(pady=10)
        
        # Download PDF button
        tk.Button(
            options_frame,
            text="Download PDF Report",
            command=lambda: webbrowser.open(pdf_path),
            font=("Arial", 12),
            bg="#9b59b6",
            fg="white",
            width=25
        ).pack(pady=10)
        
        # Email Report button
        tk.Button(
            options_frame,
            text="Email Report",
            command=lambda: self.email_report(pdf_path),
            font=("Arial", 12),
            bg="#e67e22",
            fg="white",
            width=25
        ).pack(pady=10)
        
        # Exit button
        tk.Button(
            options_frame,
            text="Exit",
            command=self.cleanup_and_exit,
            font=("Arial", 12, "bold"),
            bg="#e74c3c",
            fg="white",
            width=25
        ).pack(pady=20)

    def email_report(self, pdf_path):
        """Email the PDF report to patient"""
        try:
            # Get patient email
            cursor = self.connection.cursor()
            cursor.execute("SELECT email FROM patient WHERE id = %s", (self.patient_id,))
            patient_email = cursor.fetchone()[0]
            
            if not patient_email:
                messagebox.showerror("Error", "No email address found for patient")
                return
            
            # Email configuration (replace with your SMTP details)
            smtp_server = "smtp.gmail.com"
            smtp_port = 587
            sender_email = "your_email@gmail.com"
            sender_password = "your_app_password"  # Use app-specific password
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = patient_email
            msg['Subject'] = "Your Oral Health Report"
            
            # Email body
            body = """Dear Patient,

Please find attached your oral health examination report.

Best regards,
Your Dental Clinic"""
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach PDF
            with open(pdf_path, "rb") as f:
                attach = MIMEApplication(f.read(), _subtype="pdf")
                attach.add_header('Content-Disposition', 'attachment', 
                                filename=os.path.basename(pdf_path))
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

    def cleanup(self):
        """Clean up all resources"""
        self._camera_running = False
        
        # Release camera
        if hasattr(self, 'cap'):
            self.cap.release()
        
        # Clean up servos and motors
        cleanup_servos()
        MotorX.cleanup()
        MotorY.cleanup()
        
        # Close database connection
        if hasattr(self, 'connection') and self.connection:
            self.connection.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCaptureApp(root, "example@example.com")  # Replace with actual email
    
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

