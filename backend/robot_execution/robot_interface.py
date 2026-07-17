from abc import ABC, abstractmethod

class RobotInterface(ABC):
    
    @abstractmethod
    def initialize_devices(self):
        """Connects to the simulator hardware/devices."""
        pass

    # --- SENSORS (Getting Live Simulation Data) ---
    @abstractmethod
    def get_camera_width(self) -> int:
        """Returns the pixel width of the robot's camera (e.g., 640)."""
        pass

    @abstractmethod
    def get_distance_to_front(self) -> float:
        """Returns the live distance in meters from the front sensor/Lidar."""
        pass

    # --- LOCOMOTION (Moving the Body) ---
    @abstractmethod
    def move_forward(self, distance: float):
        """Drives forward by a specific distance in meters."""
        pass
        
    @abstractmethod
    def turn(self, angle: float):
        """Rotates in place by a specific relative angle (positive right, negative left)."""
        pass
        
    @abstractmethod
    def stop(self):
        """Cuts power to the motors instantly."""
        pass

    # --- MANIPULATION (Arm Control) ---
    @abstractmethod
    def raise_arm(self):
        pass
        
    @abstractmethod
    def lower_arm(self):
        pass
        
    @abstractmethod
    def grab_item(self):
        pass
        
    @abstractmethod
    def release_item(self):
        pass