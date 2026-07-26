# backend/rl/webots_bridge.py
import numpy as np

class WebotsBridge:
    def __init__(self, driver, world_memory):
        self.driver = driver
        self.world = world_memory

    def reset_world(self, random_seed=None):
        """Resets robot and goal positions in Webots and clears track memory."""
        # 1. Reset Webots supervisor physics / robot pose
        self.driver.reset_pose(x=0.0, y=0.0, heading=0.0)
        
        # 2. Randomly spawn goal object within a 2m x 2m box
        goal_x = np.random.uniform(0.5, 2.0)
        goal_y = np.random.uniform(-1.0, 1.0)
        self.driver.teleport_object("bottle", x=goal_x, y=goal_y)

        # 3. Clear World Model memory state
        self.world.clear()
        
        # Advance 1 step to populate initial sensors
        self.driver.step_simulation()

    def apply_action(self, action_id: int):
        """Translates discrete RL action into high-level Execution Agent commands."""
        if action_id == 0:    # MOVE_FORWARD
            self.driver.move(linear_v=0.2, angular_v=0.0)
        elif action_id == 1:  # TURN_LEFT
            self.driver.move(linear_v=0.0, angular_v=0.5)
        elif action_id == 2:  # TURN_RIGHT
            self.driver.move(linear_v=0.0, angular_v=-0.5)
        elif action_id == 3:  # STOP
            self.driver.move(linear_v=0.0, angular_v=0.0)

        # Step Webots physics by 100ms
        self.driver.step_simulation(duration_ms=100)

    def get_state(self):
        """Fetches current pose and relative distance/heading to goal."""
        robot_pose = self.driver.get_robot_pose()  # (x, y, heading)
        goal_pose = self.driver.get_object_pose("bottle")  # (x, y)
        
        dx = goal_pose[0] - robot_pose[0]
        dy = goal_pose[1] - robot_pose[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Relative angle to goal in robot frame
        angle_to_goal = np.arctan2(dy, dx) - robot_pose[2]
        
        return {
            "robot_pose": robot_pose,
            "goal_dx": dx,
            "goal_dy": dy,
            "distance": distance,
            "angle_to_goal": angle_to_goal,
            "is_collision": self.driver.check_collision()
        }