import gymnasium as gym

from rl.webots_bridge import WebotsBridge


class BAIEnv(gym.Env):

    def __init__(self, bridge: WebotsBridge, max_steps: int):
        super().__init__()

        self.bridge = bridge

        self.max_steps = max_steps
        self.step_count = 0

        self.previous_distance = None


    def reset(self, seed=None, options=None):
        """
        Start a new RL episode.

        Returns:
            observation: np.array
            info: dict
        """

        # Reset Gym random seed if provided
        super().reset(seed=seed)

        # Reset episode bookkeeping
        self.step_count = 0

        self.previous_distance = None

        # Reset Webots world
        self.bridge.reset_world(
            random_seed=seed
        )

        # Get current simulator state
        state = self.bridge.get_state()

        # Extract robot position
        robot_x, robot_y, heading = state["robot_pose"]

        # For now we assume bottle is the goal
        goal_x, goal_y = self.bridge.get_object_pose("bottle")


        # Calculate goal relative position
        dx = goal_x - robot_x
        dy = goal_y - robot_y

        distance = (dx ** 2 + dy ** 2) ** 0.5


        # Angle from robot to goal
        import numpy as np

        angle = np.arctan2(dy, dx) - heading


        # Save for reward calculation later
        self.previous_distance = distance


        # Build RL observation vector
        observation = np.array(
            [
                dx,
                dy,
                distance,
                angle,
                float(state["collision"])
            ],
            dtype=np.float32
        )


        info = {
            "distance_to_goal": distance
        }


        return observation, info


    def step(self, action):
        pass


    def render(self):
        pass


    def close(self):
        pass