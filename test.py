import io
import tkinter as tk
from tkinter import Toplevel, messagebox, ttk, filedialog
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
from HR8825 import HR8825

# Initialize GPIO
h = lgpio.gpiochip_open(0)

# Servo Configuration
SERVO_X_PIN = 27  # BCM 17 for pan servo
SERVO_Y_PIN = 17  # BCM 27 for tilt servo

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

def move_servo_x(angle_change):
    """Move servo X by specified angle change"""
    new_angle = servo_x.current_angle + angle_change
    servo_x.move_to_angle(new_angle)

def move_servo_y(angle_change):
    """Move servo Y by specified angle change"""
    new_angle = servo_y.current_angle + angle_change
    servo_y.move_to_angle(new_angle)

def reset_servos():
    """Reset servos to center position"""
    servo_x.move_to_angle(90)
    servo_y.move_to_angle(90)

def cleanup_servos():
    """Clean up servo resources"""
    servo_x.close()
    servo_y.close()
    lgpio.gpiochip_close(h)

try:
    # MotorY: Vertical movement (Y-axis)
    MotorY = HR8825(
        dir_pin=13,    # BCM 27 (Physical pin 13)
        step_pin=19,   # BCM 10 (Physical pin 19)
        enable_pin=12, # BCM 18 (Physical pin 12)
        mode_pins=(16, 6, 20)  # M0, M1, M2 (BCM 16,17,20)
    )
    
    # MotorX: Horizontal movement (X-axis)
    MotorX = HR8825(
        dir_pin=24,    # BCM 19 (Physical pin 24)
        step_pin=18,   # BCM 24 (Physical pin 18)
        enable_pin=25,  # BCM 23 (Physical pin 16)
        mode_pins=(21, 22, 5)  # M0, M1, M2 (BCM 21,22,27)
    )
    
    # Set microstepping modes
    MotorX.SetMicroStep('softward', 'fullstep')
    MotorY.SetMicroStep('hardward', 'halfstep')
    
except Exception as e:
    print(f"Motor initialization error: {e}")
    raise

def stepper_move_x(steps):
    """Move X stepper motor by specified steps"""
    try:
        direction = 'forward' if steps > 0 else 'backward'
        MotorX.TurnStep(Dir=direction, steps=abs(steps), stepdelay=0.005)
    except Exception as e:
        print(f"Error moving X stepper: {e}")

def stepper_move_y(steps):
    """Move Y stepper motor by specified steps"""
    try:
        direction = 'forward' if steps > 0 else 'backward'
        MotorY.TurnStep(Dir=direction, steps=abs(steps), stepdelay=0.005)
    except Exception as e:
        print(f"Error moving Y stepper: {e}")

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
        
        # Configure window properties
        self.root.title("Oral Image Capture System")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        self.root.configure(bg="#f5f5f5")
        
        # Make window resizable and movable
        self.root.resizable(True, True)
        
        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure styles
        self.style.configure('.', background="#f5f5f5", font=('Helvetica', 10))
        self.style.configure('TFrame', background="#f5f5f5")
        self.style.configure('TLabel', background="#f5f5f5", font=('Helvetica', 10))
        self.style.configure('TButton', font=('Helvetica', 10), padding=5)
        self.style.configure('Title.TLabel', font=('Helvetica', 14, 'bold'), foreground="#2c3e50")
        self.style.configure('Header.TLabel', font=('Helvetica', 11, 'bold'), foreground="#34495e")
        self.style.configure('Accent.TButton', foreground='white', background='#3498db', font=('Helvetica', 10, 'bold'))
        self.style.map('Accent.TButton', 
                      background=[('active', '#2980b9'), ('disabled', '#bdc3c7')])
        
        # Initialize variables
        self.current_image_index = 0
        self._camera_running = True
        self.captured_images_dir = "/home/pi/patient_images"
        
        # Create directory for captured images if it doesn't exist
        os.makedirs(self.captured_images_dir, exist_ok=True)
        
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
        
        # Bind window events
        self.root.bind("<Configure>", self.on_window_resize)
    
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

    def on_window_resize(self, event):
        """Handle window resize events to adjust UI elements"""
        if event.widget == self.root:
            # Update UI elements based on new window size
            self.update_ui_sizes()
    
    def update_ui_sizes(self):
        """Update UI element sizes based on current window size"""
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()
        
        # Adjust reference image and GIF sizes
        ref_width = max(200, min(400, int(window_width * 0.2)))
        ref_height = max(150, min(300, int(ref_width * 0.75)))
        
        # Update reference image if it exists
        if hasattr(self, 'ref_image_label') and hasattr(self.ref_image_label, 'image'):
            img_path = self.image_list[self.current_image_index]
            img = Image.open(img_path)
            img = img.resize((ref_width, ref_height), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            
            self.ref_image_label.config(image=img_tk)
            self.ref_image_label.image = img_tk
        
        # Update GIF if it exists
        if hasattr(self, 'gif_label') and hasattr(self.gif_label, 'image'):
            gif_path = self.gif_list[self.current_image_index]
            self.animate_gif(gif_path, ref_width, ref_height)
        
        # Adjust camera size
        cam_width = max(400, min(800, int(window_width * 0.4)))
        cam_height = max(300, min(600, int(cam_width * 0.75)))
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

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
        left_panel = ttk.Frame(self.main_container, style='TFrame')
        left_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure left panel grid
        left_panel.grid_rowconfigure(0, weight=1)  # Reference image
        left_panel.grid_rowconfigure(1, weight=1)  # GIF
        left_panel.grid_columnconfigure(0, weight=1)
        
        # Reference image frame
        ref_frame = ttk.LabelFrame(left_panel, text="Reference Image", style='Header.TLabel')
        ref_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.ref_image_label = ttk.Label(ref_frame)
        self.ref_image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # GIF demonstration frame
        gif_frame = ttk.LabelFrame(left_panel, text="Demonstration", style='Header.TLabel')
        gif_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        self.gif_label = ttk.Label(gif_frame)
        self.gif_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Middle Panel (Instructions)
        middle_panel = ttk.Frame(self.main_container, style='TFrame')
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
            wraplength=400,
            justify="left",
            font=('Helvetica', 10)
        )
        self.instructions_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Right Panel (Camera and Controls)
        right_panel = ttk.Frame(self.main_container, style='TFrame')
        right_panel.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        
        # Configure right panel grid
        right_panel.grid_rowconfigure(0, weight=3)  # Camera feed
        right_panel.grid_rowconfigure(1, weight=2)  # Controls
        right_panel.grid_columnconfigure(0, weight=1)
        
        # Camera feed frame
        camera_frame = ttk.LabelFrame(right_panel, text="Camera Feed", style='Header.TLabel')
        camera_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.camera_label = ttk.Label(camera_frame)
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Motor controls frame
        controls_frame = ttk.Frame(right_panel, style='TFrame')
        controls_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure controls frame grid
        controls_frame.grid_columnconfigure(0, weight=1)
        controls_frame.grid_rowconfigure(0, weight=1)  # Servo controls
        controls_frame.grid_rowconfigure(1, weight=1)  # Stepper controls
        
        # Servo controls frame
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
        bottom_frame = ttk.Frame(self.main_container, style='TFrame')
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
        
        # Display the first view
        self.display_current_view()

    def setup_camera(self):
        """Initialize USB camera"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open USB camera.")
            self.root.destroy()
            return
        
        # Set initial camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Start camera feed
        self.update_camera_feed()

    def update_camera_feed(self):
        """Update the camera feed display"""
        if self._camera_running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                
                # Get current label dimensions
                label_width = self.camera_label.winfo_width()
                label_height = self.camera_label.winfo_height()
                
                # Only resize if we have valid dimensions
                if label_width > 1 and label_height > 1:
                    img = img.resize((label_width, label_height), Image.LANCZOS)
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    self.camera_label.imgtk = imgtk
                    self.camera_label.config(image=imgtk)
            
            self.root.after(30, self.update_camera_feed)

    def display_current_view(self):
        """Display current reference image, GIF and instructions"""
        # Load static reference image
        img_path = self.image_list[self.current_image_index]
        img = Image.open(img_path)
        
        # Get reference frame dimensions
        ref_width = self.ref_image_label.winfo_width() - 10
        ref_height = self.ref_image_label.winfo_height() - 10
        
        # Only resize if we have valid dimensions
        if ref_width > 1 and ref_height > 1:
            img = img.resize((ref_width, ref_height), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            
            self.ref_image_label.config(image=img_tk)
            self.ref_image_label.image = img_tk
        
        # Load and animate GIF
        gif_path = self.gif_list[self.current_image_index]
        
        # Get GIF frame dimensions
        gif_width = self.gif_label.winfo_width() - 10
        gif_height = self.gif_label.winfo_height() - 10
        
        # Only animate if we have valid dimensions
        if gif_width > 1 and gif_height > 1:
            self.animate_gif(gif_path, gif_width, gif_height)
        
        # Update instructions
        instructions = self.instructions_list[self.current_image_index]
        self.instructions_label.config(text=instructions)
        
        # Update button states
        self.update_button_states()

    def animate_gif(self, gif_path, width, height):
        """Animate GIF demonstration with specified dimensions"""
        try:
            gif = Image.open(gif_path)
            frames = []
            
            for frame in ImageSequence.Iterator(gif):
                frame = frame.resize((width, height), Image.LANCZOS)
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
        preview.geometry("900x700")
        preview.resizable(False, False)
        preview.configure(bg="#f5f5f5")
        
        # Main frame
        main_frame = ttk.Frame(preview)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        ttk.Label(main_frame, 
                 text="Image Preview", 
                 style='Title.TLabel').pack(pady=10)
        
        # Images frame
        images_frame = ttk.Frame(main_frame)
        images_frame.pack(fill=tk.BOTH, expand=True)
        
        # Reference image
        ref_frame = ttk.LabelFrame(images_frame, text="Reference Image", style='Header.TLabel')
        ref_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ref_img = Image.open(self.image_list[self.current_image_index])
        ref_img = ref_img.resize((400, 400), Image.LANCZOS)
        ref_img_tk = ImageTk.PhotoImage(ref_img)
        
        ref_label = ttk.Label(ref_frame, image=ref_img_tk)
        ref_label.image = ref_img_tk
        ref_label.pack(pady=10)
        
        # Captured image
        cap_frame = ttk.LabelFrame(images_frame, text="Your Image", style='Header.TLabel')
        cap_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        cap_img = self.captured_image.resize((400, 400), Image.LANCZOS)
        cap_img_tk = ImageTk.PhotoImage(cap_img)
        
        cap_label = ttk.Label(cap_frame, image=cap_img_tk)
        cap_label.image = cap_img_tk
        cap_label.pack(pady=10)
        
        # Buttons frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(side=tk.BOTTOM, pady=20)
        
        ttk.Button(
            btn_frame,
            text="Retake",
            command=lambda: [preview.destroy(), self.capture_image()],
            width=12
        ).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(
            btn_frame,
            text="Save",
            command=lambda: [self.save_image(), preview.destroy()],
            style='Accent.TButton',
            width=12
        ).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(
            btn_frame,
            text="Save & Next",
            command=lambda: [self.save_image(), preview.destroy(), self.next_image()],
            style='Accent.TButton',
            width=12
        ).pack(side=tk.LEFT, padx=10)

    def save_image(self):
        """Save current captured image to database and local folder"""
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_num = self.current_image_index + 1
            filename = f"patient_{self.patient_id}_img_{img_num}_{timestamp}.jpg"
            filepath = os.path.join(self.captured_images_dir, filename)
            
            # Save image to file
            self.captured_image.save(filepath, "JPEG")
            
            # Convert image to bytes for database
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
            (patient_id, image_number, image_data, image_identifier, created_at, file_path)
            VALUES (%s, %s, %s, %s, NOW(), %s)
            """
            cursor.execute(query, (
                self.patient_id,
                img_num,  # 1-based index
                img_data,
                img_id,
                filepath
            ))
            self.connection.commit()
            
            messagebox.showinfo("Success", f"Image saved successfully to:\n{filepath}")
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
        analysis_frame = ttk.Frame(self.root)
        analysis_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(
            analysis_frame,
            text="Analyzing Images...",
            style='Title.TLabel'
        ).pack(pady=50)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            analysis_frame,
            orient="horizontal",
            length=600,
            mode="determinate"
        )
        self.progress.pack(pady=20)
        
        # Status label
        self.status_label = ttk.Label(
            analysis_frame,
            text="Processing images and generating report...",
            style='Header.TLabel'
        )
        self.status_label.pack(pady=10)
        
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
                self.status_label.config(text=f"Processing... {i}% complete")
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
        results_frame = ttk.Frame(self.root)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        ttk.Label(
            results_frame,
            text="Examination Complete!",
            style='Title.TLabel',
            foreground="#2ecc71"
        ).pack(pady=20)
        
        ttk.Label(
            results_frame,
            text="Your oral examination has been completed successfully.",
            style='Header.TLabel'
        ).pack(pady=10)
        
        # Options frame
        options_frame = ttk.Frame(results_frame)
        options_frame.pack(pady=30)
        
        # View Report button
        ttk.Button(
            options_frame,
            text="View Report in Browser",
            command=lambda: webbrowser.open(html_path),
            style='Accent.TButton',
            width=25
        ).pack(pady=10)
        
        # Download PDF button
        ttk.Button(
            options_frame,
            text="Download PDF Report",
            command=lambda: webbrowser.open(pdf_path),
            style='Accent.TButton',
            width=25
        ).pack(pady=10)
        
        # Email Report button
        ttk.Button(
            options_frame,
            text="Email Report",
            command=lambda: self.email_report(pdf_path),
            style='Accent.TButton',
            width=25
        ).pack(pady=10)
        
        # Exit button
        ttk.Button(
            options_frame,
            text="Exit",
            command=self.cleanup_and_exit,
            style='Accent.TButton',
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
