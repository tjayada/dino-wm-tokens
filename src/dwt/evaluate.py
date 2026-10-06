"""Planning evaluation on the pinned queries, with timing, and the results table."""

import csv
import datetime
import time
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from .envs import CEM, PLAN, PROTOCOL, EnvSpec

FIELDS = [
    "run_id",
    "date",
    "git_commit",
    "swm_version",
    "gpu",
    "env",
    "method",
    "encoder",
    "layer",
    "pooling",
    "tokens_per_frame",
    "feat_dim",
    "train_episodes",
    "cem_samples",
    "cem_iters",
    "horizon",
    "precision",
    "compiled",
    "seed",
    "n_episodes",
    "success_rate",
    "time_per_replan_median_s",
    "time_encode_s",
    "time_predictor_s",
    "time_cem_other_s",
    "notes",
]


@dataclass
class ModelSpec:
    name: str
    model: object
    transform: object
    process: dict
    tokens_per_frame: int
    feat_dim: int
    encoder: str
    pooling: str = "none"
    precision: str = "fp32"
    train_episodes: str = ""
    notes: dict = field(default_factory=dict)


def gpu_name(device: str) -> str:
    return torch.cuda.get_device_name(0) if device == "cuda" else device


def run_eval(
    spec: ModelSpec,
    env: EnvSpec,
    dataset,
    queries: dict,
    n_eval: int,
    device: str,
    cem: dict = CEM,
    plan: dict = PLAN,
    video_dir: Path | None = None,
) -> dict:
    """One model on the first `n_eval` queries. Time per replan = solver time / environments planned."""
    import stable_worldmodel as swm

    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    solver = swm.solver.CEMSolver(model=spec.model, device=device, seed=queries["seed"], **cem)
    solve_log = []
    plain_solve = solver.solve

    def timed_solve(info_dict, init_action=None):
        n_envs = len(next(iter(info_dict.values())))
        sync()
        t0 = time.perf_counter()  # noqa: E702
        out = plain_solve(info_dict, init_action=init_action)
        sync()
        solve_log.append((n_envs, time.perf_counter() - t0))  # noqa: E702
        return out

    solver.solve = timed_solve
    world = swm.World(
        env.env_id,
        num_envs=n_eval,
        image_shape=(224, 224),
        max_episode_steps=2 * PROTOCOL["eval_budget"],
        **env.world_kwargs,
    )
    world.set_policy(
        swm.policy.WorldModelPolicy(
            solver=solver,
            config=swm.PlanConfig(**plan),
            process=spec.process,
            transform={"pixels": spec.transform, "goal": spec.transform},
        )
    )
    t0 = time.perf_counter()
    metrics = world.evaluate(
        dataset=dataset,
        episodes_idx=queries["episodes"][:n_eval],
        start_steps=queries["start_steps"][:n_eval],
        goal_offset=queries["goal_offset"],
        eval_budget=PROTOCOL["eval_budget"],
        callables=list(env.callables),
        video=video_dir,
    )
    wall = time.perf_counter() - t0
    world.close()
    per_replan = [s / n for n, s in solve_log]
    return {
        "method": spec.name,
        "n_episodes": n_eval,
        "success_rate": float(metrics["success_rate"]),
        "episode_successes": np.asarray(metrics["episode_successes"]).astype(int).tolist(),
        "replans": int(sum(n for n, _ in solve_log)),
        "time_per_replan_mean_s": float(sum(s for _, s in solve_log) / sum(n for n, _ in solve_log)),
        "time_per_replan_median_s": float(np.median(per_replan)),
        "wall_time_s": wall,
    }


def log_run(
    csv_path: Path,
    spec: ModelSpec,
    env: EnvSpec,
    result: dict,
    device: str,
    cem: dict = CEM,
    plan: dict = PLAN,
    seed: int = PROTOCOL["seed"],
) -> dict:
    """Append one row to the results table."""
    notes = {
        "mean_replan_s": f"{result['time_per_replan_mean_s']:.3f}",
        "wall_s": f"{result['wall_time_s']:.0f}",
        "replans": result["replans"],
        **spec.notes,
    }
    row = dict.fromkeys(FIELDS, "")
    row.update(
        run_id=f"{spec.name}_{env.name}_seed{seed}_{datetime.datetime.now():%Y%m%d-%H%M%S}",
        date=f"{datetime.date.today()}",
        swm_version=version("stable-worldmodel"),
        gpu=gpu_name(device),
        env=env.name,
        method=spec.name,
        encoder=spec.encoder,
        layer="last",
        pooling=spec.pooling,
        tokens_per_frame=spec.tokens_per_frame,
        feat_dim=spec.feat_dim,
        train_episodes=spec.train_episodes,
        cem_samples=cem["num_samples"],
        cem_iters=cem["n_steps"],
        horizon=plan["horizon"],
        precision=spec.precision,
        compiled=False,
        seed=seed,
        n_episodes=result["n_episodes"],
        success_rate=result["success_rate"],
        time_per_replan_median_s=round(result["time_per_replan_median_s"], 3),
        notes="; ".join(f"{k}={v}" for k, v in notes.items()),
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return row
