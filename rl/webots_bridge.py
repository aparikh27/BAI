# backend/rl/webots_bridge.py
from typing import Dict, List, Optional
import numpy as np


class WebotsBridge:
    """
    Bridge between BAI and the Webots simulator.

    Responsibilities:
    - Reset the simulation
    - Step the simulation
    - Send robot actions / velocity commands
    - Read simulator state into self-contained snapshots
    """

    def __init__(self, driver, world_model):
        self.driver = driver
        self.world_model = world_model

    def reset_world(self, random_seed: Optional[int] = None):
        """Reset the simulation for a new episode."""
        if random_seed is not None:
            np.random.seed(random_seed)

        # Reset robot pose
        self.driver.reset_pose(
            x=0.0,
            y=0.0,
            heading=0.0,
        )

        # Randomize environment
        self.randomize_world()

        # Clear only temporary world state
        self.world_model.clear()

        # Advance one simulation step
        self.step()

    def randomize_world(self):
        """Randomize object locations."""
        bottle_x = np.random.uniform(0.5, 2.0)
        bottle_y = np.random.uniform(-1.0, 1.0)

        self.driver.teleport_object(
            object_name="bottle",
            x=bottle_x,
            y=bottle_y,
        )

    def step(self, duration_ms: int = 100):
        """Advance Webots simulation."""
        self.driver.step_simulation(duration_ms)


    def execute_action(self, action: int):
        """
        Translates abstract RL discrete actions into robot-specific velocities.
        
        0: MOVE_FORWARD
        1: TURN_LEFT
        2: TURN_RIGHT
        3: STOP
        """
        if action == 0:    # MOVE_FORWARD
            self.move(0.2, 0.0)
        elif action == 1:  # TURN_LEFT
            self.move(0.0, 0.5)
        elif action == 2:  # TURN_RIGHT
            self.move(0.0, -0.5)
        elif action == 3:  # STOP
            self.stop()
        else:
            raise ValueError(f"Invalid discrete action index: {action}")

    def move(self, linear_velocity: float, angular_velocity: float):
        """Send velocity command to robot."""
        self.driver.move(
            linear_v=linear_velocity,
            angular_v=angular_velocity,
        )

    def stop(self):
        """Stop robot."""
        self.move(0.0, 0.0)


    def get_robot_pose(self):
        """Returns: (x, y, heading)"""
        return self.driver.get_robot_pose()

    def get_robot_velocity(self):
        """Optional velocity retrieval."""
        if hasattr(self.driver, "get_robot_velocity"):
            return self.driver.get_robot_velocity()
        return None

    def get_object_pose(self, object_name: str):
        """Returns object pose (x, y) if available."""
        return self.driver.get_object_pose(object_name)

    def collision_detected(self) -> bool:
        """Returns True if robot is colliding."""
        return self.driver.check_collision()

    def get_state(self) -> Dict:
        """
        Generic simulator snapshot.
        Contains all ground-truth object and robot data needed for observations.
        """
        # Collect target object ground-truth pose
        bottle_pose = self.get_object_pose("bottle")

        return {
            "robot_pose": self.get_robot_pose(),
            "robot_velocity": self.get_robot_velocity(),
            "objects": {
                "bottle": bottle_pose
            },
            "collision": self.collision_detected(),
        }

    def close(self):
        """Shutdown simulator."""
        if hasattr(self.driver, "shutdown"):
            self.driver.shutdown()