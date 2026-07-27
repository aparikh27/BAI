from typing import Dict, List, Optional


class WebotsBridge:
    """
    Bridge between BAI and the Webots simulator.

    Responsibilities:
    - Reset the simulation
    - Step the simulation
    - Send robot commands
    - Read simulator state
    """

    def __init__(self, driver, world_model):
        self.driver = driver
        self.world_model = world_model

    # ------------------------------------------------------------------
    # Simulation Control
    # ------------------------------------------------------------------

    def reset_world(self, random_seed: Optional[int] = None):
        """
        Reset the simulation for a new episode.
        """

        if random_seed is not None:
            import numpy as np
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
        """
        Randomize object locations.
        """

        import numpy as np

        bottle_x = np.random.uniform(0.5, 2.0)
        bottle_y = np.random.uniform(-1.0, 1.0)

        self.driver.teleport_object(
            object_name="bottle",
            x=bottle_x,
            y=bottle_y,
        )

    def step(self, duration_ms: int = 100):
        """
        Advance Webots simulation.
        """
        self.driver.step_simulation(duration_ms)

    # ------------------------------------------------------------------
    # Robot Commands
    # ------------------------------------------------------------------

    def move(self, linear_velocity: float, angular_velocity: float):
        """
        Send velocity command to robot.
        """
        self.driver.move(
            linear_v=linear_velocity,
            angular_v=angular_velocity,
        )

    def stop(self):
        """
        Stop robot.
        """
        self.move(0.0, 0.0)

    # ------------------------------------------------------------------
    # Robot State
    # ------------------------------------------------------------------

    def get_robot_pose(self):
        """
        Returns:
            (x, y, heading)
        """
        return self.driver.get_robot_pose()

    def get_robot_velocity(self):
        """
        Optional.
        """
        if hasattr(self.driver, "get_robot_velocity"):
            return self.driver.get_robot_velocity()

        return None

    # ------------------------------------------------------------------
    # World State
    # ------------------------------------------------------------------

    def get_object_pose(self, object_name: str):
        """
        Returns object pose.

        Example:
            bottle
            chair
            person
        """
        return self.driver.get_object_pose(object_name)

    def get_detected_objects(self):
        """
        Returns detections from the world model.

        The World Model is the single source of truth for
        perceived objects.
        """
        if hasattr(self.world_model, "get_all_objects"):
            return self.world_model.get_all_objects()

        return []

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------

    def get_lidar(self):
        """
        Returns lidar scan if available.
        """
        if hasattr(self.driver, "get_lidar"):
            return self.driver.get_lidar()

        return None

    def get_camera_image(self):
        """
        Optional camera image.
        RL will probably never use this directly.
        """
        if hasattr(self.driver, "get_camera_image"):
            return self.driver.get_camera_image()

        return None

    # ------------------------------------------------------------------
    # Collision
    # ------------------------------------------------------------------

    def collision_detected(self):
        """
        Returns True if robot is colliding.
        """
        return self.driver.check_collision()

    # ------------------------------------------------------------------
    # Generic State Snapshot
    # ------------------------------------------------------------------

    def get_state(self) -> Dict:
        """
        Generic simulator snapshot.

        This is NOT an RL observation.

        ObservationBuilder will convert this into an
        observation vector later.
        """

        return {
            "robot_pose": self.get_robot_pose(),
            "robot_velocity": self.get_robot_velocity(),
            "objects": self.get_detected_objects(),
            "collision": self.collision_detected(),
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """
        Shutdown simulator.
        """
        if hasattr(self.driver, "shutdown"):
            self.driver.shutdown()