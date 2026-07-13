import time
from backend.robot_execution.robot_interface import RobotInterface
from backend.features.memory import World

class RobotExecutor:
    def __init__(self, robot_driver: RobotInterface, world: World):
        """
        Initializes the executor using reactive visual servoing.
        
        :param robot_driver: An implementation of RobotInterface (e.g., WebotsRobotDriver)
        :param world: The active instance of your World memory tracking system
        """
        self.robot = robot_driver 
        self.world = world

    def _align_and_approach(self, target_item: str) -> float:
        """
        Rotates the robot until the object is perfectly centered in the 
        YOLO camera view, then returns the distance sensor measurement.
        """
        camera_width = self.robot.get_camera_width()
        screen_center = camera_width / 2
        
        # Deadzone: how many pixels off-center we tolerate before stopping rotation
        pixel_tolerance = 20  

        print(f"Visual Servoing: Aligning camera with target ID {target_item}...")

        while True:
            obj = self.world.get_object_by_track_id(int(target_item))
            if not obj or not obj.visible:
                print(f"Target {target_item} lost from camera view! Stopping.")
                self.robot.stop()
                return 0.0

            # YOLO bounding box: [x_center, y_center, width, height]
            obj_x_center = obj.box[0]
            error_pixels = obj_x_center - screen_center

            # If it's close enough to the center, break the alignment loop
            if abs(error_pixels) <= pixel_tolerance:
                print("Target centered successfully!")
                self.robot.stop()
                break

            # Reactive adjustment: Turn small increments based on direction
            # If target is to the left (error < 0), turn left (-3 degrees)
            # If target is to the right (error > 0), turn right (+3 degrees)
            turn_step = 3.0 if error_pixels > 0 else -3.0
            self.robot.turn(turn_step)
            
            # Short pause to allow simulator frame buffer / YOLO memory to refresh
            time.sleep(0.05)

        # Now that we are perfectly facing the object, read physical distance
        distance_meters = self.robot.get_distance_to_front()
        return distance_meters

    def _get_object(self, target_item: str) -> bool:
        """Visually aligns with an object, approaches it, and picks it up."""
        # 1. Align using vision and find out how far away it is
        distance = self._align_and_approach(target_item)
        if distance <= 0.0:
            return False

        # 2. Drive up to it
        print(f"Driving forward {distance:.2f} meters to target.")
        self.robot.move_forward(distance)

        # 3. Manipulation sequence
        print(f"Grabbing item {target_item}.")
        self.robot.lower_arm()
        self.robot.grab_item()
        self.robot.raise_arm()

        # 4. Return to original position by reversing out
        print("Returning to origin base...")
        self.robot.turn(180)
        self.robot.move_forward(distance)
        self.robot.turn(180)  # Face original direction again
        return True

    def _put_object(self, target_item: str) -> bool:
        """Visually aligns with a placement zone/box, approaches, and drops the object."""
        distance = self._align_and_approach(target_item)
        if distance <= 0.0:
            return False

        print(f"Driving forward {distance:.2f} meters to placement spot.")
        self.robot.move_forward(distance)

        print("Depositing item.")
        self.robot.lower_arm()
        self.robot.release_item()
        self.robot.raise_arm()

        print("Returning to origin base...")
        self.robot.turn(180)
        self.robot.move_forward(distance)
        self.robot.turn(180)
        return True
    
    def execute_command(self, command: str, target_item: str) -> bool:
        """
        Executes high-level intent commands coming from the Qwen planner.
        """
        print(f"Executor running: '{command}' on target ID: '{target_item}'")
        
        if command == "detect_item":
            return int(target_item) in self.world.memory
            
        elif command == "pick_up":
            self.robot.lower_arm()
            self.robot.grab_item()
            self.robot.raise_arm()    
            return True
            
        elif command == "get_object":
            return self._get_object(target_item)
            
        elif command == "put_object":
            return self._put_object(target_item)
            
        print(f"Unrecognized execution command keyword: '{command}'")
        return False