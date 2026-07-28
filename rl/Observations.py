# backend/rl/observations.py
from typing import Dict, Any
import numpy as np
from gymnasium import spaces


class ObservationBuilder:
    """
    Translates raw simulator snapshots from WebotsBridge into normalized 1D 
    observation vectors for Gymnasium / RL models.
    """

    def __init__(self, target_name: str = "bottle"):
        self.target_name = target_name

        # Observation space definition: [dx, dy, distance, angle_to_goal, is_collision]
        low = np.array([-10.0, -10.0, 0.0, -np.pi, 0.0], dtype=np.float32)
        high = np.array([10.0, 10.0, 15.0, np.pi, 1.0], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def build_observation(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Parses state dictionary and constructs a 1D float32 numpy observation array.
        Pure function: does not access simulator bridge or external scope.
        """
        # 1. Extract robot pose (x, y, heading)
        robot_x, robot_y, heading = state["robot_pose"]

        # 2. Extract target pose (x, y) from state dictionary
        target_pose = state.get("objects", {}).get(self.target_name)
        if target_pose is not None:
            target_x, target_y = target_pose[0], target_pose[1]
        else:
            target_x, target_y = 0.0, 0.0

        # 3. Calculate relative distance and angle
        dx = target_x - robot_x
        dy = target_y - robot_y
        distance = float(np.hypot(dx, dy))

        # Relative heading angle normalized strictly to [-pi, pi]
        raw_angle = np.arctan2(dy, dx) - heading
        angle = float((raw_angle + np.pi) % (2 * np.pi) - np.pi)

        # 4. Extract collision status
        is_collision = float(state.get("collision", False))

        return np.array([dx, dy, distance, angle, is_collision], dtype=np.float32)