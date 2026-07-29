# backend/rl/evaluate.py
import time
from pathlib import Path

from stable_baselines3 import PPO

from rl.gym import BAIEnv
from rl.webots_bridge import WebotsBridge
from rl.Observations import ObservationBuilder
from rl.rewards import RewardEngine
from backend.robot_execution.webot import WebotDriver
from backend.features.memory import World


def evaluate():
    workspace_root = Path(__file__).resolve().parents[1]
    model_path = workspace_root / "models" / "ppo_navigation" / "ppo_bottle_navigator_final"

    try:
        from controller import Robot

        robot_instance = Robot()
        driver = WebotDriver(robot_instance)
        driver.initialize_devices()
        world_model = World()
    except Exception as exc:  # pragma: no cover - runtime fallback
        print(f"Webots runtime unavailable, using placeholder objects: {exc}")
        driver = None
        world_model = None

    bridge = WebotsBridge(driver=driver, world_model=world_model)
    env = BAIEnv(
        bridge=bridge,
        obs_builder=ObservationBuilder(target_name="bottle"),
        reward_engine=RewardEngine(goal_threshold=0.15),
    )

    model = PPO.load(model_path)

    obs, info = env.reset()
    done = False
    total_reward = 0.0

    print("Running evaluation episode...")

    while not done:
        # Predict action (deterministic=True disables exploration noise)
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        time.sleep(0.05)  # Throttle step visualization

    print(f"Episode finished! Total Reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    evaluate()