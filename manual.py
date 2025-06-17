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
        self.root.geometry("1300x920+10+10")
        self.root.configure(bg="white")
        
        # Initialize variables
        self.camera_image = None
        self.current_image_index = 0
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

        # Start camera feed
        self.show_camera_feed()
        
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

    def setup_right_frame(self):
        """Set up the right side with camera feed and status"""
        self.right_frame = tk.Frame(self.root, bg="white", width=650, height=700,
                                  highlightbackground="blue", highlightthickness=3)
        self.right_frame.place(x=650, y=0)
        
        # Camera feed display
        self.camera_label = tk.Label(self.right_frame, bg="black")
        self.camera_label.pack()
        
        # Status display
        self.status_frame = tk.Frame(self.right_frame, bg="white")
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

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
            reset_servos()

    def next_image(self):
        """Navigate to the next oral image view"""
        if self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.display_left_image()
            reset_servos()
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
                self.root.after(100, lambda: self.animate_gif(frames, count))

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
                 command=lambda: self.save_and_next(popup),
                 font=("Helvetica", 12), 
                 bg="#3498db", 
                 fg="white").pack(side=tk.LEFT, padx=10)

        popup.attributes('-topmost', True)
        popup.grab_set()

    def save_current_image(self, popup=None):
        """Save current image to database"""
        if not hasattr(self, 'captured_image_pil'):
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
            
            messagebox.showinfo("Success", "Image saved successfully.")
            
            if popup:
                popup.destroy()
                
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to save image: {str(e)}")
        finally:
            if 'cursor' in locals():
                cursor.close()

    def save_and_next(self, popup):
        """Save current image and move to next position"""
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
            if self.current_image_index + 1 < len(self.image_list):
                self.current_image_index += 1
                self.display_left_image()
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
        self._progress_active = True
        self.update_progress()
        
        # Hide navigation buttons
        self.prev_button.pack_forget()
        self.next_button.pack_forget()
        self.capture_button.pack_forget()
        
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
                # Basic HTML stripping for display
                text_content = html_content.replace('<', ' <').replace('>', '> ')
                self.report_text.insert(tk.END, text_content)
                self.report_text.config(state=tk.DISABLED)
        except Exception as e:
            print(f"Error loading report: {e}")
            self.report_text.insert(tk.END, "Could not load report content")
        
        # Add button frame
        button_frame = tk.Frame(thank_you_frame, bg="white")
        button_frame.pack(pady=20)
        
        # Add buttons
        tk.Button(button_frame, 
                  text="View Full Report", 
                  command=lambda: webbrowser.open(report_path),
                  font=("times new roman", 14), 
                  bg="#3498db", 
                  fg="white").pack(side=tk.LEFT, padx=10)
        
        tk.Button(button_frame, 
                  text="Download PDF", 
                  command=lambda: webbrowser.open(pdf_path),
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
            
            # Email configuration
            smtp_server = "smtp.gmail.com"
            smtp_port = 587
            sender_email = "rakeshozon46@gmail.com"
            sender_password = "knovzqbmcryhoznh"
            
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

    def show_camera_feed(self):
        """Show live camera feed with thread-safe updates"""
        if not hasattr(self, '_camera_running'):
            self._camera_running = True
            self._update_camera_feed()

    def _update_camera_feed(self):
        """Update camera feed in main thread"""
        if getattr(self, '_camera_running', False):
            try:
                ret, frame = self.cap.read()
                if ret:
                    # Convert color space
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize for display
                    frame_rgb = cv2.resize(frame_rgb, (640, 480))
                    
                    # Convert to PhotoImage
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    # Update display
                    if hasattr(self, 'camera_label'):
                        self.camera_label.imgtk = imgtk
                        self.camera_label.config(image=imgtk)
                    
            except Exception as e:
                print(f"Camera error: {e}")
            
            # Schedule next update
            self.root.after(30, self._update_camera_feed)

    def cleanup_and_exit(self):
        """Clean up resources and exit"""
        self.cleanup()
        self.root.destroy()

    def cleanup(self):
        """Clean up resources before closing"""
        self._camera_running = False
        
        if hasattr(self, 'cap'):
            self.cap.release()
        
        # Clean up servos
        cleanup_servos()
        
        # Clean up motors
        MotorX.cleanup()
        MotorY.cleanup()
        
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
