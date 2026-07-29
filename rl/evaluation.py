# backend/rl/evaluate.py
import time
from stable_baselines3 import PPO
from rl.gym import BAIEnv
from rl.webots_bridge import WebotsBridge

def evaluate():
    # Load environment
    bridge = WebotsBridge(driver=driver, world_model=world_model)
    env = BAIEnv(bridge=bridge)

    # Load trained model
    model = PPO.load("models/ppo_navigation/ppo_bottle_navigator_final")

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