# backend/rl/gym.py
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

        # Reset Webots world
        self.bridge.reset_world(random_seed=seed)

        # Get initial state and build observation
        state = self.bridge.get_state()
        observation = self.obs_builder.build_observation(state)

        # Track target distance index [2] from obs array
        self.previous_distance = float(observation[2])

        info = {
            "distance_to_goal": self.previous_distance,
            "is_collision": bool(state["collision"]),
            "step": self.step_count
        }

        return observation, info

    def step(self, action: int):
        """Execute one step in the environment."""
        self.step_count += 1

        # 1. Dispatch action to bridge and step simulator
        self.bridge.execute_action(action)
        self.bridge.step(duration_ms=100)

        # 2. Get state and build observation
        state = self.bridge.get_state()
        observation = self.obs_builder.build_observation(state)

        current_distance = float(observation[2])
        is_collision = bool(observation[4])

        # 3. Compute reward & termination via RewardEngine
        reward, terminated = self.reward_engine.compute_reward_and_termination(
            current_distance=current_distance,
            is_collision=is_collision,
            previous_distance=self.previous_distance
        )

        # 4. Check step truncation limit
        truncated = self.step_count >= self.max_steps

        # Update distance memory for next delta calculation
        self.previous_distance = current_distance

        info = {
            "distance_to_goal": current_distance,
            "is_collision": is_collision,
            "step": self.step_count
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        """Webots renders directly in its 3D environment window."""
        pass

    def close(self):
        """Clean up simulator connection."""
        self.bridge.close()