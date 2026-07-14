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

    def _resolve_object(self, target_item: str):
        """
        Resolve a Qwen target to a world object.

        The planner may output either a numeric track ID or a class name.
        """
        if target_item is None:
            return None

        target_text = str(target_item).strip()
        if not target_text:
            return None
        print(f"[EXECUTOR LOG] Attempting to resolve target text: '{target_text}'")
        

        if target_text.isdigit():
            obj = self.world.get_object_by_track_id(int(target_text))
            if obj and obj.visible:
                print(f"[EXECUTOR LOG] Match found! Resolved '{target_text}' to object label '{obj.class_name}'")
                return obj

        normalized_target = target_text.lower()
        for obj in self.world.get_visible_objects():
            if obj.class_name.lower() == normalized_target:
                print(f"[EXECUTOR LOG] Match found! Resolved '{target_text}' to object label '{obj.class_name}'")
                return obj

        print(f"[EXECUTOR LOG] No match found for target text: '{target_text}'")
        return None

    def _align_and_approach(self, target_item: str) -> float:
        """
        Rotates the robot until the object is perfectly centered in the 
        YOLO camera view, then returns the distance sensor measurement.
        """
        obj = self._resolve_object(target_item)
        if obj is None:
            print(f"Target '{target_item}' not found in the visible world.")
            return 0.0

        camera_width = self.robot.get_camera_width()
        screen_center = camera_width / 2
        
        # Deadzone: how many pixels off-center we tolerate before stopping rotation
        pixel_tolerance = 20  

        print(f"Visual Servoing: Aligning camera with target '{target_item}'...")

        while True:
            obj = self._resolve_object(target_item)
            if not obj:
                print(f"Target {target_item} lost from camera view! Stopping.")
                self.robot.stop()
                return 0.0

            # YOLO bounding box: [x_min, y_min, x_max, y_max]
            obj_x_center = (obj.box[0] + obj.box[2]) / 2
            error_pixels = obj_x_center - screen_center

            # If it's close enough to the center, break the alignment loop
            if abs(error_pixels) <= pixel_tolerance:
                print("Target centered successfully!")
                self.robot.stop()
                break

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

    def _scan_360_for_object(self, target_item: str) -> bool:
        """
        Rotates the robot in a full 360-degree circle in small increments,
        checking if the target object becomes visible to the YOLO camera.
        """
        print(f"🔍 Starting active 360-degree scan for '{target_item}'...")
        
        # Define search steps (36 steps of 10 degrees = 360 degrees)
        step_angle = 10.0
        total_steps = 36
        
        for step in range(total_steps):
            # 1. Check if we can see the object in the current frame
            obj = self._resolve_object(target_item)
            if obj is not None:
                print(f"Target '{target_item}' spotted during scan!")
                self.robot.stop()
                return True
            
            # 2. Rotate slightly to scan the next slice of the room
            # Positive angle rotates in one direction
            self.robot.turn(step_angle)
            
            # 3. Pause briefly to allow Webots physics and YOLO memory to update
            time.sleep(0.1)
            
        print(f"Completed 360-degree scan. '{target_item}' was not found.")
        self.robot.stop()
        return False

    def execute_command(self, command: str, target_item: str) -> bool:
        """
        Executes high-level intent commands coming from the Qwen planner.
        """
        print(f"Executor running: '{command}' on target ID: '{target_item}'")
        
        if command in {"detect_item", "detect_object"}:
            # Call our new active scanning method instead of a passive check
            return self._scan_360_for_object(target_item)
            
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