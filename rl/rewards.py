# backend/rl/rewards.py
from typing import Tuple, Dict, Any


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
        previous_distance: float = None
    ) -> Tuple[float, bool]:
        """
        Computes step reward and checks if episode is terminated (success or collision).

        Returns:
            reward (float): Step reward
            terminated (bool): Success or collision state reached
        """
        terminated = False

        # 1. Base step penalty (discourages spinning/idling)
        reward = -0.05

        # 2. Distance Shaping (reward moving closer, penalize drifting away)
        if previous_distance is not None:
            distance_delta = previous_distance - current_distance
            reward += distance_delta * 10.0

        # 3. Terminal Success
        if current_distance < self.goal_threshold:
            reward += 100.0
            terminated = True

        # 4. Terminal Collision
        elif is_collision:
            reward -= 25.0
            terminated = True

        return reward, terminated