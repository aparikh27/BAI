#  Using a Class organizes everything beautifully
class RobotExecutor:
    def __init__(self, robot_driver, world_model):
        self.robot = robot_driver     # Saved to "self" memory!
        self.world = world_model       # Saved to "self" memory!
        self.is_holding_item = False   # Track internal state easily

    def move_forward(self, distance):
        # Easily accesses the driver via self
        self.robot.set_wheel_speeds(2.0) 

    def grab_object(self, target_item):
        # Instantly accesses the world model and driver
        coords = self.world.get_coords(target_item)
        self.move_forward(5) 
        self.is_holding_item = True  # Updates its own state