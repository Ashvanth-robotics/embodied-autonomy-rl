import os
import json

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from src.env import WheelchairNav2D


def main():
    # =========================
    # Experiment configuration
    # =========================
    scenario = "clutter"
    seed = 0
    total_timesteps = 100_000

    # =========================
    # Output directories
    # =========================
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    # =========================
    # Environments
    # =========================
    env = WheelchairNav2D(scenario=scenario, seed=seed)
    eval_env = WheelchairNav2D(scenario=scenario, seed=seed + 42)

    # =========================
    # PPO model
    # =========================
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=seed,
    )

    # =========================
    # Evaluation callback
    # =========================
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="outputs/models",
        log_path="outputs/logs",
        eval_freq=5_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # =========================
    # Train PPO (NO progress bar)
    # =========================
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=False,
    )

    # =========================
    # Save models
    # =========================
    model.save("outputs/models/ppo_final")

    # =========================
    # Save JSON-safe config
    # =========================
    config = {
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "scenario": scenario,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "hyperparameters": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 256,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
        },
    }

    with open("outputs/models/ppo_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("✅ PPO training finished successfully")
    print("📦 Saved files:")
    print(" - outputs/models/best_model.zip")
    print(" - outputs/models/ppo_final.zip")
    print(" - outputs/models/ppo_config.json")


if __name__ == "__main__":
    main()
