# backend/rl/rewards.py
from __future__ import annotations

from typing import Tuple


class RewardEngine:
    """
    Calculates shaped step rewards and terminal conditions based on state transitions.
    """

    def __init__(self, goal_threshold: float = 0.15):
        self.goal_threshold = goal_threshold

    def compute_reward_and_termination(
        self,
        current_distance: float,
        is_collision: bool,
        previous_distance: float | None = None,
    ) -> Tuple[float, bool]:
        """
        Computes step reward and checks if episode is terminated (success or collision).

        Returns:
            reward (float): Step reward
            terminated (bool): Success or collision state reached
        """
        terminated = False

        try:
            current_distance_value = float(current_distance)
        except (TypeError, ValueError):
            current_distance_value = float("nan")

        try:
            previous_distance_value = float(previous_distance) if previous_distance is not None else None
        except (TypeError, ValueError):
            previous_distance_value = None

        if not self._is_finite(current_distance_value):
            current_distance_value = 0.0
            reward = -0.05
            if bool(is_collision):
                reward -= 25.0
                terminated = True
            return float(reward), bool(terminated)

        reward = -0.05

        if previous_distance_value is not None and self._is_finite(previous_distance_value):
            distance_delta = previous_distance_value - current_distance_value
            reward += distance_delta * 10.0

        if current_distance_value < self.goal_threshold:
            reward += 100.0
            terminated = True
        elif bool(is_collision):
            reward -= 25.0
            terminated = True

        return float(reward), bool(terminated)

    @staticmethod
    def _is_finite(value: float) -> bool:
        return value == value and value not in (float("inf"), float("-inf"))