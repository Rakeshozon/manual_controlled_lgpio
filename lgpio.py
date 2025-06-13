import lgpio
import time

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
        """Initialize with lgpio"""
        self.h = lgpio.gpiochip_open(0)  # Open gpiochip0
        
        # Store pin numbers
        self.dir_pin = dir_pin
        self.step_pin = step_pin
        self.enable_pin = enable_pin
        self.mode_pins = mode_pins if isinstance(mode_pins, (list, tuple)) else [mode_pins]
        
        # Claim all pins as outputs
        lgpio.gpio_claim_output(self.h, self.dir_pin)
        lgpio.gpio_claim_output(self.h, self.step_pin)
        lgpio.gpio_claim_output(self.h, self.enable_pin)
        for pin in self.mode_pins:
            lgpio.gpio_claim_output(self.h, pin)
        
        # Initialize all pins to LOW
        self.digital_write(self.dir_pin, 0)
        self.digital_write(self.step_pin, 0)
        self.digital_write(self.enable_pin, 0)
        for pin in self.mode_pins:
            self.digital_write(pin, 0)
    
    def digital_write(self, pin, value):
        """Write to a GPIO pin using lgpio"""
        lgpio.gpio_write(self.h, pin, value)
    
    def Stop(self):
        """Disable the motor"""
        self.digital_write(self.enable_pin, 0)
    
    def SetMicroStep(self, mode, stepformat):
        """
        Set microstepping mode
        (1) mode: 'hardward' or 'softward'
        (2) stepformat: e.g. 'fullstep', 'halfstep', etc.
        """
        microstep = {
            'fullstep': (0, 0, 0),
            'halfstep': (1, 0, 0),
            '1/4step': (0, 1, 0),
            '1/8step': (1, 1, 0),
            '1/16step': (0, 0, 1),
            '1/32step': (1, 0, 1)
        }

        if mode == ControlMode[1]:  # software control
            values = microstep.get(stepformat)
            if values and len(values) == len(self.mode_pins):
                for pin, val in zip(self.mode_pins, values):
                    self.digital_write(pin, val)
            else:
                print("Invalid step format or mode pin count mismatch.")
    
    def TurnStep(self, Dir, steps, stepdelay=0.005):
        """Turn motor steps with direction"""
        if Dir == MotorDir[0]:  # forward
            self.digital_write(self.enable_pin, 1)
            self.digital_write(self.dir_pin, 0)
        elif Dir == MotorDir[1]:  # backward
            self.digital_write(self.enable_pin, 1)
            self.digital_write(self.dir_pin, 1)
        else:
            print("Direction must be 'forward' or 'backward'")
            self.digital_write(self.enable_pin, 0)
            return

        for _ in range(steps):
            self.digital_write(self.step_pin, 1)
            time.sleep(stepdelay)
            self.digital_write(self.step_pin, 0)
            time.sleep(stepdelay)
    
    def cleanup(self):
        """Clean up GPIO resources"""
        self.Stop()
        lgpio.gpiochip_close(self.h)
