#THIS IS THE CUSTOM ENVIRONMENT CODE - THIS PROJECT IS NOT EXECUTED IN ROS OR GAZEBO#

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class Scenario:
    name: str
    world_size: float = 10.0   # world is [-size/2, size/2] in x,y
    n_obstacles: int = 10
    obstacle_radius_range: Tuple[float, float] = (0.25, 0.6)
    start_goal_min_dist: float = 6.0
    max_steps: int = 400
    lidar_num_rays: int = 36
    lidar_max_range: float = 6.0
    dt: float = 0.1


DEFAULT_SCENARIOS: Dict[str, Scenario] = {
    "open": Scenario(name="open", n_obstacles=0, max_steps=250),
    "corridor": Scenario(name="corridor", n_obstacles=0, max_steps=350),
    "clutter": Scenario(name="clutter", n_obstacles=14, max_steps=400),
}


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class WheelchairNav2D(gym.Env):
    """
    2D differential-drive wheelchair navigation with simple lidar.
    Observation:
      - lidar distances (N rays) normalized [0,1]
      - goal distance normalized
      - goal bearing (sin, cos)
      - current velocity v, w normalized
    Action:
      - (v_cmd, w_cmd) in [-1,1] scaled to v_max, w_max
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: str = "clutter",
        seed: Optional[int] = None,
        v_max: float = 1.0,        # m/s
        w_max: float = 1.5,        # rad/s
        goal_radius: float = 0.35,
        collision_radius: float = 0.35,
    ):
        super().__init__()
        if scenario not in DEFAULT_SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'. Use one of: {list(DEFAULT_SCENARIOS)}")
        self.sc = DEFAULT_SCENARIOS[scenario]
        self.v_max = v_max
        self.w_max = w_max
        self.goal_radius = goal_radius
        self.collision_radius = collision_radius

        self._rng = np.random.default_rng(seed)

        # obs dim
        n = self.sc.lidar_num_rays
        # lidar + goal_dist + goal_sin + goal_cos + v + w
        obs_dim = n + 1 + 2 + 2

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.reset(seed=seed)

    # ---------- World generation ----------
    def _random_point(self) -> np.ndarray:
        half = self.sc.world_size / 2.0
        return self._rng.uniform(-half, half, size=(2,)).astype(np.float32)

    def _distance_to_obstacles(self, p: np.ndarray) -> float:
        if len(self.obstacles) == 0:
            return float("inf")
        d = float("inf")
        for ox, oy, r in self.obstacles:
            d = min(d, math.hypot(p[0] - ox, p[1] - oy) - r)
        return d

    def _make_obstacles(self):
        self.obstacles: List[Tuple[float, float, float]] = []

        # corridor walls for "corridor"
        if self.sc.name == "corridor":
            # Two long walls along x direction
            half = self.sc.world_size / 2
            y1, y2 = -1.5, 1.5
            # represent walls as many small circles for lidar intersection simplicity
            xs = np.linspace(-half, half, 80)
            for x in xs:
                self.obstacles.append((float(x), float(y1), 0.15))
                self.obstacles.append((float(x), float(y2), 0.15))

        # random clutter
        for _ in range(self.sc.n_obstacles):
            for _try in range(200):
                c = self._random_point()
                r = float(self._rng.uniform(*self.sc.obstacle_radius_range))
                # keep obstacles away from borders a bit
                half = self.sc.world_size / 2
                if abs(c[0]) > half - 0.8 or abs(c[1]) > half - 0.8:
                    continue
                # avoid overlap
                ok = True
                for ox, oy, orad in self.obstacles:
                    if math.hypot(c[0] - ox, c[1] - oy) < (r + orad + 0.3):
                        ok = False
                        break
                if ok:
                    self.obstacles.append((float(c[0]), float(c[1]), r))
                    break

    def _sample_start_goal(self):
        for _ in range(500):
            s = self._random_point()
            g = self._random_point()
            if math.hypot(*(s - g)) < self.sc.start_goal_min_dist:
                continue
            # ensure not inside obstacles
            if self._distance_to_obstacles(s) < (self.collision_radius + 0.2):
                continue
            if self._distance_to_obstacles(g) < (self.goal_radius + 0.2):
                continue
            return s, g
        # fallback
        return np.array([-3.0, -3.0], dtype=np.float32), np.array([3.0, 3.0], dtype=np.float32)

    # ---------- Lidar ----------
    def _ray_circle_intersection(self, p, d, c, r) -> Optional[float]:
        # p: ray origin, d: unit direction, circle center c, radius r
        # solve ||p + t d - c||^2 = r^2, t>=0
        px, py = p
        dx, dy = d
        cx, cy = c
        ox, oy = px - cx, py - cy
        b = 2 * (ox * dx + oy * dy)
        cterm = ox * ox + oy * oy - r * r
        disc = b * b - 4 * cterm
        if disc < 0:
            return None
        sqrt_disc = math.sqrt(disc)
        t1 = (-b - sqrt_disc) / 2
        t2 = (-b + sqrt_disc) / 2
        t = None
        if t1 >= 0 and t2 >= 0:
            t = min(t1, t2)
        elif t1 >= 0:
            t = t1
        elif t2 >= 0:
            t = t2
        return t

    def _lidar(self) -> np.ndarray:
        n = self.sc.lidar_num_rays
        max_r = self.sc.lidar_max_range
        angles = np.linspace(-math.pi, math.pi, n, endpoint=False)

        out = np.full((n,), max_r, dtype=np.float32)
        p = self.pos
        heading = self.yaw

        for i, a in enumerate(angles):
            th = heading + a
            d = np.array([math.cos(th), math.sin(th)], dtype=np.float32)

            best = max_r
            for ox, oy, r in self.obstacles:
                t = self._ray_circle_intersection(p, d, (ox, oy), r + self.collision_radius)
                if t is not None and 0 <= t < best:
                    best = t

            # world boundary as square
            half = self.sc.world_size / 2.0
            # intersect with x = +/-half, y = +/-half
            # compute smallest positive t where ray hits boundary
            if abs(d[0]) > 1e-6:
                tx1 = (half - p[0]) / d[0]
                tx2 = (-half - p[0]) / d[0]
                for tx in (tx1, tx2):
                    if 0 <= tx < best:
                        y_hit = p[1] + tx * d[1]
                        if -half <= y_hit <= half:
                            best = tx
            if abs(d[1]) > 1e-6:
                ty1 = (half - p[1]) / d[1]
                ty2 = (-half - p[1]) / d[1]
                for ty in (ty1, ty2):
                    if 0 <= ty < best:
                        x_hit = p[0] + ty * d[0]
                        if -half <= x_hit <= half:
                            best = ty

            out[i] = float(best)

        return out

    # ---------- Gym API ----------
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._make_obstacles()
        self.pos, self.goal = self._sample_start_goal()
        self.yaw = float(self._rng.uniform(-math.pi, math.pi))
        self.v = 0.0
        self.w = 0.0
        self.step_count = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _get_obs(self) -> np.ndarray:
        lidar = self._lidar()
        lidar_n = np.clip(lidar / self.sc.lidar_max_range, 0.0, 1.0)

        gvec = self.goal - self.pos
        gdist = float(np.linalg.norm(gvec))
        gbearing = wrap_to_pi(math.atan2(gvec[1], gvec[0]) - self.yaw)
        gdist_n = np.clip(gdist / (self.sc.world_size), 0.0, 1.0)

        v_n = np.clip((self.v / self.v_max + 1) / 2, 0.0, 1.0)
        w_n = np.clip((self.w / self.w_max + 1) / 2, 0.0, 1.0)

        obs = np.concatenate([
            lidar_n.astype(np.float32),
            np.array([gdist_n, math.sin(gbearing), math.cos(gbearing), v_n, w_n], dtype=np.float32)
        ])
        return obs.astype(np.float32)

    def _get_info(self) -> Dict:
        gdist = float(np.linalg.norm(self.goal - self.pos))
        min_lidar = float(np.min(self._lidar())) if self.sc.lidar_num_rays > 0 else float("inf")
        return {
            "goal_dist": gdist,
            "min_lidar": min_lidar,
            "pos_x": float(self.pos[0]),
            "pos_y": float(self.pos[1]),
            "yaw": float(self.yaw),
            "v": float(self.v),
            "w": float(self.w),
            "step": int(self.step_count),
            "scenario": self.sc.name,
        }

    def step(self, action):
        self.step_count += 1

        # scale actions
        v_cmd = float(action[0]) * self.v_max
        w_cmd = float(action[1]) * self.w_max

        # first-order dynamics (smooth changes)
        alpha = 0.5
        self.v = (1 - alpha) * self.v + alpha * v_cmd
        self.w = (1 - alpha) * self.w + alpha * w_cmd

        # integrate
        dt = self.sc.dt
        self.yaw = wrap_to_pi(self.yaw + self.w * dt)
        self.pos = self.pos + np.array([math.cos(self.yaw), math.sin(self.yaw)], dtype=np.float32) * (self.v * dt)

        # compute termination
        gdist = float(np.linalg.norm(self.goal - self.pos))
        done_goal = gdist <= self.goal_radius

        # collision if too close to obstacle or outside boundary
        half = self.sc.world_size / 2.0
        outside = abs(self.pos[0]) > half or abs(self.pos[1]) > half
        clearance = self._distance_to_obstacles(self.pos)
        collided = clearance <= 0.0

        terminated = done_goal or collided or outside
        truncated = self.step_count >= self.sc.max_steps

        # reward shaping (MSc-friendly: explainable)
        # - progress reward
        # - small control penalty
        # - big collision penalty
        # - success bonus
        prev_dist = getattr(self, "_prev_dist", gdist)
        progress = prev_dist - gdist
        self._prev_dist = gdist

        r = 2.0 * progress
        r -= 0.01 * (abs(self.v) / self.v_max + abs(self.w) / self.w_max)
        # encourage safe distance
        min_lidar = float(np.min(self._lidar()))
        if min_lidar < 0.6:
            r -= 0.05 * (0.6 - min_lidar)

        if done_goal:
            r += 5.0
        if collided or outside:
            r -= 5.0

        obs = self._get_obs()
        info = self._get_info()
        info.update({"done_goal": done_goal, "collided": collided, "outside": outside})
        return obs, float(r), terminated, truncated, info
