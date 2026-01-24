# Embodied Autonomous Navigation: Classical Planning vs Reinforcement Learning

## Overview
This repository investigates **embodied autonomous navigation** using a custom-built **2D simulation environment implemented entirely in Python**, comparing a **classical navigation approach** with a **modified Proximal Policy Optimization (PPO)** reinforcement learning policy.

The project focuses on understanding how hand-engineered navigation logic compares to learning-based control in terms of **adaptability, stability, and failure modes** for goal-directed navigation in safety-critical settings.

Rather than relying on existing simulators or middleware, the environment and training pipeline were designed from scratch to allow full control over dynamics, observations, and reward structure.

---

## Problem Statement
Autonomous navigation for embodied agents requires robust decision-making under uncertainty, partial observability, and environmental constraints. Classical navigation methods often depend on carefully tuned rules and heuristics, which can struggle to generalise beyond their design assumptions.

This project explores:
- The behaviour of classical navigation logic in structured environments
- The ability of a PPO-based policy to learn navigation directly from state observations
- Trade-offs between interpretability, robustness, and adaptability in both approaches

---

## System Setup
- **Simulation**: Custom-built 2D navigation environment (Python)
- **Agent**: Differential-drive wheelchair-like robot model
- **Observations**: Agent pose, obstacle proximity, goal-relative state
- **Action Space**: Continuous linear and angular velocity commands
- **Objective**: Reach target goals while avoiding collisions

> While evaluated in a simplified 2D setting, the formulation reflects core challenges present in real-world embodied autonomy systems.

---

## Methods Compared

### Classical Navigation Approach
- Rule-based navigation logic
- Explicit obstacle avoidance heuristics
- Deterministic control decisions
- High interpretability but limited adaptability to unseen scenarios

### Reinforcement Learning (Modified PPO)
- End-to-end policy learning for navigation control
- PPO algorithm with custom modifications to improve:
  - training stability
  - convergence behaviour
  - smoothness of motion
- Reward shaping designed to balance:
  - collision avoidance
  - efficient goal reaching
  - stable control actions

---

## Experiments
Both approaches were evaluated under identical environment conditions to ensure fair comparison. Evaluation metrics include:
- Navigation success rate
- Collision frequency
- Path efficiency
- Control smoothness
- Behavioural consistency across scenarios

The analysis includes both quantitative metrics and qualitative inspection of agent trajectories and failure cases.

---

## Results & Observations
Key observations from the experiments:
- Classical navigation demonstrates consistent behaviour in simple, well-defined environments but degrades under increasing complexity
- The PPO-based agent exhibits adaptive behaviour and smoother control, but remains sensitive to reward design and training parameters
- Learning-based navigation reveals distinct failure modes that are difficult to anticipate using classical logic

Detailed results, plots, and rollout visualisations are provided in the `docs/` and `videos/` directories.

---

## Limitations
- Simplified 2D simulation environment
- No physics engine or real-world sensor noise
- No sim-to-real transfer or hardware deployment
- Safety handled implicitly through reward shaping rather than formal constraints

These limitations highlight important challenges in deploying learning-based navigation systems beyond controlled environments.

---

## Future Work
- Extending to richer observation spaces and partial observability
- Introducing stochastic dynamics and sensor noise
- Formal safety constraints in RL training
- Scaling to more complex environments
- Integrating perception learning into the navigation policy

---

