import math
import time
from backend.robot_execution.robot_interface import RobotInterface

class WebotDriver(RobotInterface):
    def __init__(self, robot):
        self.robot = robot
        self.time_step = int(self.robot.getBasicTimeStep())
        
        # Sensors
        self.camera = None
        self.distance_sensor = None
        self.gps = None
        self.compass = None
        
        # Actuators
        self.left_motor = None
        self.right_motor = None
        self.arm_motor = None
        self.gripper_motor = None

    def initialize_devices(self):
        """Connects to the simulator hardware/devices."""
        # 1. Sensors
        # Note: If your world definition doesn't include a camera, 
        # this safely catches the None without crashing.
        self.camera = self.robot.getDevice("camera") or self.robot.getDevice("khepera3_gripper_camera")
        if self.camera:
            self.camera.enable(self.time_step)

        # The C code uses "ds0" as the first distance sensor
        self.distance_sensor = self.robot.getDevice("ds0")
        if self.distance_sensor:
            self.distance_sensor.enable(self.time_step)

        # Tracking sensors (Safely handled if not present in your specific .wbt world file)
        self.gps = self.robot.getDevice("gps")
        if self.gps:
            self.gps.enable(self.time_step)

        self.compass = self.robot.getDevice("compass")
        if self.compass:
            self.compass.enable(self.time_step)

        # 2. Wheel Actuators (Velocity Control)
        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        
        if self.left_motor and self.right_motor:
            self.left_motor.setPosition(float('inf'))
            self.right_motor.setPosition(float('inf'))
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)

        # 3. Gripper Actuators (Position Control)
        # exact strings extracted from the provided C controller code
        self.arm_motor = self.robot.getDevice("horizontal_motor")
        self.gripper_motor = self.robot.getDevice("finger_motor::left")

    def get_camera_width(self) -> int:
        if self.camera:
            return self.camera.getWidth()
        return 640  # Sensible default fallback

    def get_distance_to_front(self) -> float:
        if self.distance_sensor:
            return self.distance_sensor.getValue()
        return 0.0

    def get_position(self) -> tuple[float, float]:
        """Returns the current (x, y) coordinates of the robot from the GPS."""
        if self.gps:
            values = self.gps.getValues()
            return (values[0], values[1])
        return (0.0, 0.0)

    def get_angle(self) -> float:
        """Returns the current orientation/heading angle of the robot in radians using the Compass."""
        if self.compass:
            north = self.compass.getValues()
            return math.atan2(north[1], north[0])
        return 0.0

    def move_forward(self, distance: float):
        """Drives forward at a set velocity and blocks until distance is covered."""
        if not self.left_motor or not self.right_motor:
            return
            
        speed = 12.8  # Using the baseline operational speed from your C code
        self.left_motor.setVelocity(speed)
        self.right_motor.setVelocity(speed)
        
        duration = distance / 0.2 
        start_time = self.robot.getTime()
        
        while self.robot.getTime() - start_time < duration:
            if self.robot.step(self.time_step) == -1:
                break
        self.stop()
        
    def turn(self, angle: float):
        """Rotates in place. Positive angle = Right, Negative = Left."""
        if not self.left_motor or not self.right_motor:
            return
            
        speed = 2.4  # Matches your C source baseline rotation speed
        if angle > 0:
            self.left_motor.setVelocity(speed)
            self.right_motor.setVelocity(-speed)
        else:
            self.left_motor.setVelocity(-speed)
            self.right_motor.setVelocity(speed)
            
        duration = abs(angle) * 0.012  
        start_time = self.robot.getTime()
        
        while self.robot.getTime() - start_time < duration:
            if self.robot.step(self.time_step) == -1:
                break
        self.stop()
        
    def stop(self):
        """Stops wheel velocities completely."""
        if self.left_motor and self.right_motor:
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)

    def raise_arm(self):
        """Uses position control to lift the gripper assembly upwards."""
        if self.arm_motor:
            self.arm_motor.setVelocity(2.0)
            self.arm_motor.setPosition(0.0)  
            self._wait_for_actuator(1.0)      

    def lower_arm(self):
        """Uses position control to lower the arm down near the cylinder."""
        if self.arm_motor:
            self.arm_motor.setVelocity(2.0)
            self.arm_motor.setPosition(-3.0)  # Value pulled directly from line 118 of C source
            self._wait_for_actuator(1.0)
        
    def grab_item(self):
        """Closes the gripper claws inwards."""
        if self.gripper_motor:
            self.gripper_motor.setVelocity(2.0)
            self.gripper_motor.setPosition(0.42)  # Value pulled directly from line 122 of C source
            self._wait_for_actuator(1.0)
        
    def release_item(self):
        """Opens the gripper claws wide."""
        if self.gripper_motor:
            self.gripper_motor.setVelocity(2.0)
            self.gripper_motor.setPosition(0.0) 
            self._wait_for_actuator(1.0)

    def _wait_for_actuator(self, duration: float):
        """Helper to pause execution and let mechanical parts finish moving."""
        start = self.robot.getTime()
        while self.robot.getTime() - start < duration:
            if self.robot.step(self.time_step) == -1:
                break