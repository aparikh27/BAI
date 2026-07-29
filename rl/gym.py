# backend/rl/gym.py
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces

from rl.webots_bridge import WebotsBridge
from rl.Observations import ObservationBuilder
from rl.rewards import RewardEngine


class BAIEnv(gym.Env):
    """
    Gymnasium Environment adapter for BAI.
    Orchestrates the environment step loop by delegating observation building,
    reward calculation, and physics execution to dedicated modules.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self, 
        bridge: WebotsBridge, 
        obs_builder: ObservationBuilder = None,
        reward_engine: RewardEngine = None,
        max_steps: int = 300
    ):
        super().__init__()

        self.bridge = bridge
        self.obs_builder = obs_builder or ObservationBuilder()
        self.reward_engine = reward_engine or RewardEngine()

        self.max_steps = max_steps
        self.step_count = 0
        self.previous_distance = None

        # Spaces defined by submodules / bridge contracts
        self.action_space = spaces.Discrete(4)
        self.observation_space = self.obs_builder.observation_space

    def reset(self, seed=None, options=None):
        """Start a new RL episode."""
        super().reset(seed=seed)

        self.step_count = 0
        self.previous_distance = None

        if self.bridge is not None:
            self.bridge.reset_world(random_seed=seed)

        state = self.bridge.get_state() if self.bridge is not None else {}
        observation = self.obs_builder.build_observation(state)

        self.previous_distance = float(observation[2])

        info = {
            "distance_to_goal": self.previous_distance,
            "is_collision": bool(state.get("collision", False)),
            "step": self.step_count,
        }

        return observation, info

    def step(self, action: int):
        """Execute one step in the environment."""
        self.step_count += 1

        if self.bridge is not None:
            self.bridge.execute_action(action)
            self.bridge.step(duration_ms=100)

        state = self.bridge.get_state() if self.bridge is not None else {}
        observation = self.obs_builder.build_observation(state)

        current_distance = float(observation[2])
        is_collision = bool(observation[4])

        reward, terminated = self.reward_engine.compute_reward_and_termination(
            current_distance=current_distance,
            is_collision=is_collision,
            previous_distance=self.previous_distance,
        )

        truncated = self.step_count >= self.max_steps

        self.previous_distance = current_distance

        info = {
            "distance_to_goal": current_distance,
            "is_collision": is_collision,
            "step": self.step_count,
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        """Webots renders directly in its 3D environment window."""
        pass

    def close(self):
        """Clean up simulator connection."""
        self.bridge.close()