import os
import numpy as np
import pandas as pd

from src.env import WheelchairNav2D
from src.classical import classical_policy

def run_episode(env, policy_fn, max_steps=2000):
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    terminated = truncated = False

    # path length + smoothness proxies
    prev_xy = (info.get("x", None), info.get("y", None))
    path_len = 0.0
    v_hist, w_hist = [], []

    while not (terminated or truncated):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

        # collect velocity profiles if available
        v_hist.append(float(info.get("v", 0.0)))
        w_hist.append(float(info.get("w", 0.0)))

        # path length
        x = float(info.get("x", 0.0))
        y = float(info.get("y", 0.0))
        if prev_xy[0] is not None:
            path_len += float(np.hypot(x - prev_xy[0], y - prev_xy[1]))
        prev_xy = (x, y)

        if steps >= max_steps:
            break

    # smoothness: variance of velocity changes
    smooth = 0.0
    if len(v_hist) > 2:
        smooth = float(np.var(np.diff(v_hist)) + np.var(np.diff(w_hist)))

    # success/collision (best-effort from env info)
    reached = int(info.get("reached", info.get("done_goal", False)))
    collided = int(info.get("collided", info.get("collision", False)))

    return {
        "return": total_reward,
        "steps": steps,
        "time": steps * float(getattr(env, "dt", getattr(env, "sc", None).dt if hasattr(env, "sc") else 0.1)),
        "path_length": path_len,
        "smoothness": smooth,
        "reached": reached,
        "collision": collided,
    }

def eval_classical(scenario="clutter", episodes=30, seed=0):
    env = WheelchairNav2D(scenario=scenario, seed=seed)
    rows = []
    for ep in range(episodes):
        m = run_episode(env, classical_policy)
        m.update({"episode": ep, "seed": seed, "scenario": scenario, "method": "classical"})
        rows.append(m)
    return rows

def main():
    os.makedirs("outputs", exist_ok=True)

    scenarios = ["open", "corridor", "clutter"]
    seeds = [0, 1, 2]
    episodes = 30

    all_rows = []
    for sc in scenarios:
        for sd in seeds:
            all_rows += eval_classical(scenario=sc, episodes=episodes, seed=sd)

    df = pd.DataFrame(all_rows)
    df.to_csv("outputs/metrics_classical.csv", index=False)
    print("Saved: outputs/metrics_classical.csv")
    print(df.groupby(["scenario"])["reached"].mean())

if __name__ == "__main__":
    main()
