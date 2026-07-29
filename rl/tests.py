import numpy as np
import pytest
from unittest.mock import MagicMock

from rl.Observations import ObservationBuilder
from rl.rewards import RewardEngine
from rl.webots_bridge import WebotsBridge
from rl.gym import BAIEnv


# ----------------------------------------------------------------------
# 1. Tests for ObservationBuilder
# ----------------------------------------------------------------------

def test_observation_builder_shape_and_types():
    builder = ObservationBuilder(target_name="bottle")
    
    mock_state = {
        "robot_pose": (0.0, 0.0, 0.0),  # x, y, heading
        "objects": {"bottle": (3.0, 4.0)},
        "collision": False
    }

    obs = builder.build_observation(mock_state)

    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert obs.shape == (5,)
    
    # Check bounds against observation space
    assert builder.observation_space.contains(obs)


def test_observation_builder_distance_and_angle_calculation():
    builder = ObservationBuilder(target_name="bottle")

    # Target is at (3, 4) relative to robot at (0, 0) facing 0 rad
    # 3-4-5 right triangle -> distance should be 5.0
    mock_state = {
        "robot_pose": (0.0, 0.0, 0.0),
        "objects": {"bottle": (3.0, 4.0)},
        "collision": False
    }

    obs = builder.build_observation(mock_state)
    dx, dy, distance, angle, is_collision = obs

    assert dx == pytest.approx(3.0)
    assert dy == pytest.approx(4.0)
    assert distance == pytest.approx(5.0)
    assert angle == pytest.approx(np.arctan2(4.0, 3.0))
    assert is_collision == 0.0


def test_observation_builder_missing_target_fallback():
    builder = ObservationBuilder(target_name="bottle")

    # Target not in objects map
    mock_state = {
        "robot_pose": (1.0, 1.0, 0.0),
        "objects": {},
        "collision": False
    }

    obs = builder.build_observation(mock_state)
    dx, dy, distance, angle, _ = obs

    # Fallback target defaults to (0,0)
    assert dx == pytest.approx(-1.0)
    assert dy == pytest.approx(-1.0)


def test_observation_builder_handles_missing_or_invalid_state():
    builder = ObservationBuilder(target_name="bottle")

    obs = builder.build_observation(None)
    assert obs.shape == (5,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()

    obs = builder.build_observation({"robot_pose": None, "objects": {}, "collision": False})
    assert obs.shape == (5,)
    assert np.isfinite(obs).all()


# ----------------------------------------------------------------------
# 2. Tests for RewardEngine
# ----------------------------------------------------------------------

def test_reward_engine_moving_closer():
    engine = RewardEngine(goal_threshold=0.15)

    # Robot moved 0.5m closer (prev: 2.0m, curr: 1.5m)
    reward, terminated = engine.compute_reward_and_termination(
        current_distance=1.5,
        is_collision=False,
        previous_distance=2.0
    )

    # Base penalty (-0.05) + Delta reward (0.5 * 10.0 = 5.0) = 4.95
    assert reward == pytest.approx(4.95)
    assert terminated is False


def test_reward_engine_terminal_success():
    engine = RewardEngine(goal_threshold=0.15)

    # Robot reaches within 0.15m of goal
    reward, terminated = engine.compute_reward_and_termination(
        current_distance=0.10,
        is_collision=False,
        previous_distance=0.20
    )

    # Base penalty (-0.05) + Delta (0.1 * 10 = 1.0) + Success (100.0) = 100.95
    assert reward == pytest.approx(100.95)
    assert terminated is True


def test_reward_engine_terminal_collision():
    engine = RewardEngine(goal_threshold=0.15)

    reward, terminated = engine.compute_reward_and_termination(
        current_distance=1.0,
        is_collision=True,
        previous_distance=1.0
    )

    # Base penalty (-0.05) + Collision penalty (-25.0) = -25.05
    assert reward == pytest.approx(-25.05)
    assert terminated is True


def test_reward_engine_handles_non_finite_values():
    engine = RewardEngine(goal_threshold=0.15)

    reward, terminated = engine.compute_reward_and_termination(
        current_distance=float("nan"),
        is_collision=False,
        previous_distance=None
    )

    assert isinstance(reward, float)
    assert terminated is False
    assert np.isfinite(reward)


# ----------------------------------------------------------------------
# 3. Tests for WebotsBridge Action Dispatch
# ----------------------------------------------------------------------

def test_webots_bridge_execute_action():
    mock_driver = MagicMock()
    mock_world_model = MagicMock()
    bridge = WebotsBridge(driver=mock_driver, world_model=mock_world_model)

    # Action 0: MOVE_FORWARD
    bridge.execute_action(0)
    mock_driver.move.assert_called_with(linear_v=0.2, angular_v=0.0)

    # Action 1: TURN_LEFT
    bridge.execute_action(1)
    mock_driver.move.assert_called_with(linear_v=0.0, angular_v=0.5)

    # Invalid action
    with pytest.raises(ValueError):
        bridge.execute_action(99)


def test_webots_bridge_supports_webotdriver_style_interface():
    class WebotDriverStyle:
        def __init__(self):
            self.calls = []

        def get_position(self):
            return (1.5, -0.5)

        def get_angle(self):
            return 0.25

        def check_collision(self):
            return True

        def step_simulation(self, duration_ms):
            self.calls.append(duration_ms)

        def move(self, linear_v, angular_v):
            self.calls.append((linear_v, angular_v))

    driver = WebotDriverStyle()
    bridge = WebotsBridge(driver=driver, world_model=None)

    pose = bridge.get_robot_pose()
    assert pose == pytest.approx((1.5, -0.5, 0.25))

    assert bridge.check_collision() is True
    bridge.step_simulation(120)
    assert driver.calls[-1] == 120


# ----------------------------------------------------------------------
# 4. End-to-End Environment Step Test (BAIEnv with Mocks)
# ----------------------------------------------------------------------

def test_gym_env_reset_and_step():
    mock_driver = MagicMock()
    mock_driver.get_robot_pose.return_value = (0.0, 0.0, 0.0)
    mock_driver.get_object_pose.return_value = (2.0, 0.0)
    mock_driver.check_collision.return_value = False

    mock_world_model = MagicMock()
    bridge = WebotsBridge(driver=mock_driver, world_model=mock_world_model)

    env = BAIEnv(bridge=bridge, max_steps=10)

    # Test Reset
    obs, info = env.reset()
    assert obs.shape == (5,)
    assert info["distance_to_goal"] == pytest.approx(2.0)

    # Test Step
    obs, reward, terminated, truncated, info = env.step(action=0)
    assert obs.shape == (5,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert env.step_count == 1