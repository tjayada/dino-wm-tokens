"""Per-environment settings: env id, dataset, how to set start state and goal, reference checkpoints."""

from dataclasses import dataclass, field

HF_DATASETS = "https://huggingface.co/datasets/quentinll"
HF_DINOWM = "https://huggingface.co/fafraob"


@dataclass(frozen=True)
class EnvSpec:
    name: str
    env_id: str
    dataset: str  # name under $STABLEWM_HOME/datasets, without .h5
    archive: str  # file in the HuggingFace dataset repo
    hf_repo: str
    action_dim: int
    keys_to_cache: tuple[str, ...]
    callables: tuple[dict, ...]  # World.evaluate per-env setup, as in le-wm's eval configs
    lewm_repo: str
    dinowm_repo: str
    world_kwargs: dict = field(default_factory=dict)
    smoke_expert: str | None = None  # expert policy for collecting a tiny local dataset
    tested: bool = True


def _set(method, **args):
    return {
        "method": method,
        "args": {k: ({"value": v} if not isinstance(v, dict) else v) for k, v in args.items()},
    }


ENVS = {
    "tworoom": EnvSpec(
        name="tworoom",
        env_id="swm/TwoRoom-v1",
        dataset="tworoom",
        archive="tworoom.tar.zst",
        hf_repo=f"{HF_DATASETS}/lewm-tworooms",
        action_dim=2,
        keys_to_cache=("action", "proprio"),
        callables=(_set("_set_state", state="proprio"), _set("_set_goal_state", goal_state="goal_proprio")),
        lewm_repo="quentinll/lewm-tworooms",
        dinowm_repo=f"{HF_DINOWM}/point-cloud-tworoom",
        smoke_expert="stable_worldmodel.envs.two_room.ExpertPolicy",
    ),
    "pusht": EnvSpec(
        name="pusht",
        env_id="swm/PushT-v1",
        dataset="pusht_expert_train",
        archive="pusht_expert_train.h5.zst",
        hf_repo=f"{HF_DATASETS}/lewm-pusht",
        action_dim=2,
        keys_to_cache=("action", "proprio", "state"),
        callables=(_set("_set_state", state="state"), _set("_set_goal_state", goal_state="goal_state")),
        lewm_repo="quentinll/lewm-pusht",
        dinowm_repo=f"{HF_DINOWM}/point-cloud-pusht",
    ),
    "reacher": EnvSpec(
        name="reacher",
        env_id="swm/ReacherDMControl-v0",
        dataset="dmc/reacher_random",
        archive="reacher.tar.zst",
        hf_repo=f"{HF_DATASETS}/lewm-reacher",
        action_dim=2,
        keys_to_cache=("action",),
        callables=(
            _set("set_state", qpos="qpos", qvel="qvel"),
            _set("set_target_qpos", target_qpos="goal_qpos"),
        ),
        lewm_repo="quentinll/lewm-reacher",
        dinowm_repo=f"{HF_DINOWM}/point-cloud-reacher",
        world_kwargs={"task": "qpos_match"},
        tested=False,
    ),
    "cube": EnvSpec(
        name="cube",
        env_id="swm/OGBCube-v0",
        dataset="ogbench/cube_single_expert",
        archive="cube_single_expert.tar.zst",
        hf_repo=f"{HF_DATASETS}/lewm-cube",
        action_dim=5,
        keys_to_cache=("action",),
        callables=(
            _set("set_state", qpos="qpos", qvel="qvel"),
            _set(
                "set_target_pos",
                cube_id={"value": 0, "in_dataset": False},
                target_pos="goal_privileged_block_0_pos",
                target_quat="goal_privileged_block_0_quat",
            ),
        ),
        lewm_repo="quentinll/lewm-cube",
        dinowm_repo=f"{HF_DINOWM}/point-cloud-ogb-cube",
        world_kwargs={
            "env_type": "single",
            "ob_type": "states",
            "multiview": False,
            "width": 224,
            "height": 224,
            "visualize_info": False,
            "terminate_at_goal": True,
        },
        tested=False,
    ),
}

# LeWM's protocol, identical for the four environments
PROTOCOL = {"n_eval": 50, "goal_offset": 25, "eval_budget": 50, "seed": 42}
PLAN = {"horizon": 5, "receding_horizon": 5, "action_block": 5}
CEM = {"num_samples": 300, "n_steps": 30, "topk": 30}
