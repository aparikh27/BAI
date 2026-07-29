# backend/rl/webots_bridge.py
from __future__ import annotations

from typing import Any, Dict, Optional

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

        if hasattr(self.driver, "reset_pose"):
            self.driver.reset_pose(x=0.0, y=0.0, heading=0.0)

        self.randomize_world()

        if self.world_model is not None and hasattr(self.world_model, "clear"):
            self.world_model.clear()

        self.step()

    def randomize_world(self):
        """Randomize object locations."""
        bottle_x = np.random.uniform(0.5, 2.0)
        bottle_y = np.random.uniform(-1.0, 1.0)

        if hasattr(self.driver, "teleport_object"):
            self.driver.teleport_object(object_name="bottle", x=bottle_x, y=bottle_y)

    def step(self, duration_ms: int = 100):
        """Advance Webots simulation."""
        if hasattr(self.driver, "step_simulation"):
            self.driver.step_simulation(duration_ms)
        elif hasattr(self.driver, "step"):
            self.driver.step(duration_ms)

    def step_simulation(self, duration_ms: int = 100):
        """Compatibility wrapper for simulator-style interfaces."""
        self.step(duration_ms=duration_ms)

    def execute_action(self, action: int):
        """
        Translates abstract RL discrete actions into robot-specific velocities.

        0: MOVE_FORWARD
        1: TURN_LEFT
        2: TURN_RIGHT
        3: STOP
        """
        if action == 0:
            self.move(0.2, 0.0)
        elif action == 1:
            self.move(0.0, 0.5)
        elif action == 2:
            self.move(0.0, -0.5)
        elif action == 3:
            self.stop()
        else:
            raise ValueError(f"Invalid discrete action index: {action}")

    def move(self, linear_velocity: float, angular_velocity: float):
        """Send velocity command to robot."""
        if hasattr(self.driver, "move"):
            self.driver.move(linear_v=linear_velocity, angular_v=angular_velocity)
        elif hasattr(self.driver, "set_velocity"):
            self.driver.set_velocity(linear_velocity, angular_velocity)
        elif angular_velocity == 0.0 and linear_velocity > 0.0 and hasattr(self.driver, "move_forward"):
            # Legacy drivers expose a blocking distance primitive instead of
            # velocity control.  Keep the fallback small and explicit.
            self.driver.move_forward(linear_velocity * 0.1)
        elif linear_velocity == 0.0 and angular_velocity != 0.0 and hasattr(self.driver, "turn"):
            self.driver.turn(-angular_velocity * 6.0)

    def stop(self):
        """Stop robot."""
        if hasattr(self.driver, "stop"):
            self.driver.stop()
        else:
            self.move(0.0, 0.0)

    def get_robot_pose(self):
        """Returns: (x, y, heading)."""
        if hasattr(self.driver, "get_robot_pose"):
            return self.driver.get_robot_pose()
        if hasattr(self.driver, "get_position") and hasattr(self.driver, "get_angle"):
            x, y = self.driver.get_position()
            heading = self.driver.get_angle()
            return (x, y, heading)
        return (0.0, 0.0, 0.0)

    def get_robot_velocity(self):
        """Optional velocity retrieval."""
        if hasattr(self.driver, "get_robot_velocity"):
            return self.driver.get_robot_velocity()
        return None

    def get_object_pose(self, object_name: str):
        """Returns object pose (x, y) if available."""
        if hasattr(self.driver, "get_object_pose"):
            return self.driver.get_object_pose(object_name)
        return None

    def check_collision(self) -> bool:
        """Returns True if robot is colliding."""
        if hasattr(self.driver, "check_collision"):
            return bool(self.driver.check_collision())
        return False

    def collision_detected(self) -> bool:
        """Backward-compatible alias for collision checks."""
        return self.check_collision()

    def get_state(self) -> Dict[str, Any]:
        """
        Generic simulator snapshot.
        Contains all ground-truth object and robot data needed for observations.
        """
        bottle_pose = self.get_object_pose("bottle")

        return {
            "robot_pose": self.get_robot_pose(),
            "robot_velocity": self.get_robot_velocity(),
            "objects": {"bottle": bottle_pose},
            "collision": self.collision_detected(),
        }

    def close(self):
        """Shutdown simulator."""
        if hasattr(self.driver, "shutdown"):
            self.driver.shutdown()
