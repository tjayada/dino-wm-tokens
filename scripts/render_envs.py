"""Render a few frames of the four target environments under random actions.

Usage: python scripts/render_envs.py  ->  results/figures/env_frames.png
"""

import os
import sys
from pathlib import Path

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stable_worldmodel  # noqa: F401  registers the swm/* envs

ENVS = {
    "PushT": "swm/PushT-v1",
    "TwoRoom": "swm/TwoRoom-v1",
    "Cube": "swm/OGBCube-v0",
    "Reacher": "swm/ReacherDMControl-v0",
}
N_FRAMES = 5
STEPS_BETWEEN = 10
OUT = Path(__file__).resolve().parents[1] / "results" / "figures" / "env_frames.png"


def rollout(env_id, seed=0):
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    frames = [env.render()]
    for _ in range(N_FRAMES - 1):
        for _ in range(STEPS_BETWEEN):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                env.reset()
        frames.append(env.render())
    print(f"{env_id}: frame {frames[0].shape}, action {env.action_space.shape}")
    env.close()
    return frames


def main():
    fig, axes = plt.subplots(len(ENVS), N_FRAMES, figsize=(2.4 * N_FRAMES, 2.5 * len(ENVS)))
    for row, (name, env_id) in zip(axes, ENVS.items(), strict=True):
        for t, (ax, frame) in enumerate(zip(row, rollout(env_id), strict=True)):
            ax.imshow(frame)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"step {t * STEPS_BETWEEN}", fontsize=9)
        row[0].set_ylabel(name, fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print("saved", OUT)


if __name__ == "__main__":
    main()
