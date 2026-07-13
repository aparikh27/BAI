from abc import ABC, abstractmethod

class RobotInterface(ABC):
    
    @abstractmethod
    def initialize_devices(self):
        """Connects to the simulator hardware/devices."""
        pass

    @abstractmethod
    def get_position(self) -> tuple[float, float]:
        """Returns the current (x, y) coordinates from the robot's GPS/odometry."""
        pass
        
    @abstractmethod
    def get_angle(self) -> float:
        """Returns the current heading/orientation in degrees."""
        pass

    @abstractmethod
    def move_forward(self, distance: float):
        """Drives forward by a specific distance."""
        pass
        
    @abstractmethod
    def turn(self, angle: float):
        """Rotates in place by a specific relative angle (e.g., +90 or -45 degrees)."""
        pass
        
    @abstractmethod
    def stop(self):
        """Cuts power to the motors instantly."""
        pass


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