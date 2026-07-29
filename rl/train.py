# backend/rl/train.py
import os
import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from rl.gym import BAIEnv
from rl.webots_bridge import WebotsBridge
from rl.Observations import ObservationBuilder
from rl.rewards import RewardEngine

from backend.robot_execution.webot import WebotDriver
from backend.features.memory import World


def train(total_timesteps: int = 100_000, dry_run: bool = False):
    # ------------------------------------------------------------------
    # 1. Setup Logging & Checkpoint Directories
    # ------------------------------------------------------------------
    workspace_root = Path(__file__).resolve().parents[1]
    log_dir = str(workspace_root / "logs" / "ppo_navigation")
    model_dir = str(workspace_root / "models" / "ppo_navigation")
    if not dry_run:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

    print("Initializing Webots Bridge & Environment...")

    # ------------------------------------------------------------------
    # 2. Instantiate Bridge & Environment
    # ------------------------------------------------------------------
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

    obs_builder = ObservationBuilder(target_name="bottle")
    reward_engine = RewardEngine(goal_threshold=0.15)

    raw_env = BAIEnv(
        bridge=bridge,
        obs_builder=obs_builder,
        reward_engine=reward_engine,
        max_steps=300,
    )

    if dry_run:
        try:
            observation, info = raw_env.reset(seed=0)
            _, reward, terminated, truncated, _ = raw_env.step(
                raw_env.action_space.sample()
            )
            print(
                "Dry run complete: "
                f"observation_shape={observation.shape}, reward={reward:.3f}, "
                f"terminated={terminated}, truncated={truncated}, info={info}"
            )
        finally:
            raw_env.close()
        return

    # Wrap with Gym Monitor to record episode reward and length statistics
    env = Monitor(raw_env, filename=os.path.join(log_dir, "monitor.csv"))

    # ------------------------------------------------------------------
    # 3. Setup Training Callbacks
    # ------------------------------------------------------------------
    # Save model weights every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="ppo_bai_model"
    )

    # ------------------------------------------------------------------
    # 4. Initialize PPO Agent
    # ------------------------------------------------------------------
    print("Configuring PPO Model...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,          # Number of steps per update
        batch_size=64,         # Mini-batch size for SGD
        n_epochs=10,           # Optimization epochs per update
        gamma=0.99,            # Discount factor
        gae_lambda=0.95,       # Factor for trade-off of bias vs variance for GAE
        clip_range=0.2,        # PPO Clipping parameter
        ent_coef=0.01,         # Entropy coefficient to encourage exploration
        verbose=1,
        tensorboard_log=log_dir
    )

    # ------------------------------------------------------------------
    # 5. Launch Training
    # ------------------------------------------------------------------
    print(f"Starting training for {total_timesteps} timesteps...")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            tb_log_name="PPO_Webots_Run_1"
        )
        
        # Save final trained model
        final_model_path = os.path.join(model_dir, "ppo_bottle_navigator_final")
        model.save(final_model_path)
        print(f"\nTraining Complete! Model saved to: {final_model_path}.zip")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current weights...")
        model.save(os.path.join(model_dir, "ppo_bottle_navigator_interrupted"))
    
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the PPO navigation agent.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Construct and exercise one environment step without training or writing a model.",
    )
    args = parser.parse_args()
    train(total_timesteps=args.timesteps, dry_run=args.dry_run)
