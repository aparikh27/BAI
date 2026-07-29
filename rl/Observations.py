# backend/rl/observations.py
from __future__ import annotations

from typing import Any, Dict

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

    def build_observation(self, state: Dict[str, Any] | None) -> np.ndarray:
        """
        Parses a state dictionary and constructs a 1D float32 numpy observation array.
        Safe defaults are returned if the state is missing or malformed.
        """
        if not isinstance(state, dict):
            state = {}

        robot_pose = state.get("robot_pose")
        if isinstance(robot_pose, (tuple, list)) and len(robot_pose) >= 3:
            robot_x, robot_y, heading = robot_pose[0], robot_pose[1], robot_pose[2]
        else:
            robot_x, robot_y, heading = 0.0, 0.0, 0.0

        objects = state.get("objects")
        if not isinstance(objects, dict):
            objects = {}

        target_pose = objects.get(self.target_name)
        if isinstance(target_pose, (tuple, list)) and len(target_pose) >= 2:
            target_x, target_y = float(target_pose[0]), float(target_pose[1])
        else:
            target_x, target_y = 0.0, 0.0

        try:
            dx = float(target_x - robot_x)
            dy = float(target_y - robot_y)
            distance = float(np.hypot(dx, dy))
            raw_angle = np.arctan2(dy, dx) - float(heading)
            angle = float((raw_angle + np.pi) % (2 * np.pi) - np.pi)
        except (TypeError, ValueError, OverflowError):
            dx, dy, distance, angle = 0.0, 0.0, 0.0, 0.0

        try:
            is_collision = float(bool(state.get("collision", False)))
        except (TypeError, ValueError):
            is_collision = 0.0

        obs = np.array([dx, dy, distance, angle, is_collision], dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)