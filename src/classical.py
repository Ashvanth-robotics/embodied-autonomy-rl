import numpy as np

def classical_policy(obs: np.ndarray) -> np.ndarray:
    """
    Simple classical baseline:
    - Uses goal bearing (sin, cos) in the observation to steer toward the goal.
    - Drives forward with speed that reduces when turning sharply.

    Assumes obs contains:
      [lidar..., goal_dist, goal_sin, goal_cos, v_norm, w_norm]
    This matches the docstring in your env.py.
    """
    # Extract goal bearing (sin, cos) from the tail
    goal_sin = float(obs[-4])
    goal_cos = float(obs[-3])

    # Heading error in radians
    heading_err = np.arctan2(goal_sin, goal_cos)

    # Angular command: steer proportionally toward goal direction
    w_cmd = np.clip(2.0 * heading_err, -1.0, 1.0)

    # Linear command: go forward, but slow down when turning a lot
    v_cmd = 0.8 * (1.0 - 0.5 * abs(w_cmd))
    v_cmd = np.clip(v_cmd, -1.0, 1.0)

    return np.array([v_cmd, w_cmd], dtype=np.float32)
