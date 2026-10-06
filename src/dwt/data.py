"""Datasets, pinned start/goal queries and action scaling."""

import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import zstandard
from sklearn import preprocessing
from tqdm.auto import tqdm

from .envs import EnvSpec
from .paths import data_root, persist, restore


def dataset_path(spec: EnvSpec) -> Path:
    return data_root() / "datasets" / f"{spec.dataset}.h5"


def ensure_dataset(spec: EnvSpec) -> Path:
    """Download and unpack the released dataset unless it is already local or on the mirror."""
    path = dataset_path(spec)
    if restore(path, "datasets"):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{spec.hf_repo}/resolve/main/{spec.archive}"
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0)) or None
        with (
            tqdm.wrapattr(resp, "read", total=total, desc=spec.archive) as raw,
            zstandard.ZstdDecompressor().stream_reader(raw) as zr,
        ):
            if spec.archive.endswith(".tar.zst"):
                with tarfile.open(fileobj=zr, mode="r|") as tar:
                    tar.extractall(path.parent)
            else:
                with path.open("wb") as out:
                    shutil.copyfileobj(zr, out)
    if not path.exists():  # archive laid out differently than expected: adopt the single h5 it contained
        found = [p for p in path.parent.rglob("*.h5") if p != path]
        assert len(found) == 1, f"expected {path} after unpacking, found {found}"
        found[0].replace(path)
    persist(path, "datasets")
    return path


def collect_smoke_dataset(spec: EnvSpec, episodes: int = 20) -> Path:
    """A tiny locally collected dataset for runs without a GPU."""
    import stable_worldmodel as swm
    from hydra.utils import get_class

    path = data_root() / "datasets" / f"{spec.dataset}_local.h5"
    if path.exists():
        return path
    assert spec.smoke_expert, f"no expert policy registered for {spec.name}"
    world = swm.World(
        spec.env_id,
        num_envs=4,
        image_shape=(224, 224),
        max_episode_steps=100,
        render_mode="rgb_array",
        **spec.world_kwargs,
    )
    world.set_policy(get_class(spec.smoke_expert)(action_noise=2.0, action_repeat_prob=0.05))
    world.collect(path, episodes=episodes, seed=0, format="hdf5")
    world.close()
    return path


def load_dataset(spec: EnvSpec, smoke: bool = False):
    import stable_worldmodel as swm

    name = f"{spec.dataset}_local" if smoke else spec.dataset
    return swm.data.HDF5Dataset(name, keys_to_cache=list(spec.keys_to_cache))


def draw_queries(dataset, seed: int, n: int, goal_offset: int) -> dict:
    """LeWM's query draw: uniform over dataset rows that leave room for the goal, without replacement."""
    lengths, offsets = np.asarray(dataset.lengths), np.asarray(dataset.offsets)
    row_episode = np.repeat(np.arange(len(lengths)), lengths)
    row_step = np.arange(lengths.sum()) - np.repeat(offsets, lengths)
    valid = np.nonzero(row_step <= np.repeat(lengths, lengths) - goal_offset - 1)[0]
    rng = np.random.default_rng(seed)
    rows = np.sort(valid[rng.choice(len(valid) - 1, size=min(n, len(valid) - 1), replace=False)])
    return {
        "seed": seed,
        "goal_offset": goal_offset,
        "episodes": row_episode[rows].tolist(),
        "start_steps": row_step[rows].tolist(),
    }


def pinned_queries(spec: EnvSpec, dataset, results_dir: Path, seed: int, n: int, goal_offset: int) -> dict:
    """Load the stored queries for this environment and seed, or draw and store them."""
    path = results_dir / "queries" / f"{spec.dataset.replace('/', '_')}_seed{seed}.json"
    queries = draw_queries(dataset, seed, n, goal_offset)
    if path.exists():
        stored = json.loads(path.read_text())
        assert (
            stored["episodes"] == queries["episodes"] and stored["start_steps"] == queries["start_steps"]
        ), f"queries drawn now differ from {path}"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"dataset": spec.dataset, **queries}, indent=1))
    return queries


def action_scaler(dataset) -> preprocessing.StandardScaler:
    data = dataset.get_col_data("action")
    return preprocessing.StandardScaler().fit(data[~np.isnan(data).any(axis=1)])


def scaler_from_stats(mean, std) -> preprocessing.StandardScaler:
    s = preprocessing.StandardScaler()
    s.mean_, s.scale_, s.var_ = np.asarray(mean), np.asarray(std), np.asarray(std) ** 2
    s.n_features_in_ = len(mean)
    return s
