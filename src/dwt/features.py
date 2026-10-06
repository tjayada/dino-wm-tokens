"""Run the frozen DINOv2 once over a dataset and store pooled tokens per variant (float16 HDF5)."""

import queue
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .data import dataset_path
from .envs import EnvSpec
from .models import ENCODER_PX, FEAT_DIM, GRID, HUB_NAME, TOKENS, pool_tokens
from .paths import data_root, persist, restore

VARIANTS = ("cls", "mean", "g2", "g4", "g7")


def feature_tag(spec: EnvSpec, n_episodes: int, smoke: bool = False) -> str:
    return f"{spec.dataset.replace('/', '_')}{'_local' if smoke else ''}_ep{n_episodes}"


def feature_path(tag: str, variant: str) -> Path:
    return data_root() / "features" / f"{tag}_{variant}.h5"


class Encoder:
    def __init__(self, device: str):
        self.device = device
        self.dino = torch.hub.load("facebookresearch/dinov2", HUB_NAME).to(device).eval()
        self.dino.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, frames_uint8: np.ndarray) -> dict[str, np.ndarray]:
        """(B, 224, 224, 3) uint8 -> {variant: (B, tokens, 384) float16}, preprocessing as the reference."""
        x = torch.from_numpy(frames_uint8).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(
            (x - 0.5) / 0.5,
            size=(ENCODER_PX, ENCODER_PX),
            mode="bilinear",
            antialias=True,
            align_corners=False,
        )
        out = self.dino.forward_features(x)
        cls_token, patches = out["x_norm_clstoken"], out["x_norm_patchtokens"]
        assert patches.shape[1] == GRID * GRID
        return {v: pool_tokens(cls_token, patches, v).to(torch.float16).cpu().numpy() for v in VARIANTS}


def cache_features(
    spec: EnvSpec, n_episodes: int | None, device: str, batch: int = 256, smoke: bool = False
) -> dict[str, Path]:
    """Write one file per variant for the first `n_episodes` episodes; resumable; mirrored when done."""
    src_path = dataset_path(spec)
    if smoke:
        src_path = src_path.with_name(src_path.stem + "_local.h5")
    with h5py.File(src_path, "r") as src:
        lengths, offsets = src["ep_len"][:], src["ep_offset"][:]
    n_ep = len(lengths) if n_episodes is None else min(n_episodes, len(lengths))
    n_frames = int(lengths[:n_ep].sum())
    tag = feature_tag(spec, n_ep, smoke)
    files = {v: feature_path(tag, v) for v in VARIANTS}
    if all(restore(p, "features") for p in files.values()) and all(
        _done(p, n_frames) for p in files.values()
    ):
        print(f"features {tag}: already cached")
        return files

    encoder = Encoder(device)
    handles = {}
    for v, path in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        f = h5py.File(path, "a")
        if "feat" not in f:
            k = TOKENS[v]
            f.create_dataset(
                "feat",
                shape=(n_frames, k, FEAT_DIM),
                dtype="float16",
                chunks=(min(256, n_frames), k, FEAT_DIM),
            )
            with h5py.File(src_path, "r") as src:
                f.create_dataset("action", data=src["action"][:n_frames])
                f.create_dataset("ep_len", data=lengths[:n_ep])
                f.create_dataset("ep_offset", data=offsets[:n_ep])
            f.attrs.update(
                dict(
                    variant=v,
                    tokens_per_frame=k,
                    feat_dim=FEAT_DIM,
                    source=src_path.name,
                    n_episodes=n_ep,
                    n_frames=n_frames,
                    encoder=HUB_NAME,
                    layer="last",
                    encoder_px=ENCODER_PX,
                    normalization="(x/255 - 0.5) / 0.5",
                    frames_done=0,
                )
            )
        handles[v] = f
    start = min(int(f.attrs["frames_done"]) for f in handles.values())

    def read_batches(q):
        with h5py.File(src_path, "r") as src:
            for i in range(start, n_frames, batch):
                q.put((i, src["pixels"][i : min(i + batch, n_frames)]))
        q.put(None)

    q = queue.Queue(maxsize=3)
    threading.Thread(target=read_batches, args=(q,), daemon=True).start()
    t0 = time.perf_counter()
    with tqdm(total=n_frames, initial=start, unit="frame", desc=f"features {tag}") as bar:
        while (item := q.get()) is not None:
            i, frames = item
            feats = encoder(frames)
            for v, f in handles.items():
                f["feat"][i : i + len(frames)] = feats[v]
            if (i // batch) % 50 == 0:
                for f in handles.values():
                    f.attrs["frames_done"] = i + len(frames)
                    f.flush()
            bar.update(len(frames))
    for f in handles.values():
        f.attrs["frames_done"] = n_frames
        f.close()
    print(f"features {tag}: {n_frames} frames in {(time.perf_counter() - t0) / 60:.1f} min")
    for path in files.values():
        persist(path, "features")
    return files


def _done(path: Path, n_frames: int) -> bool:
    with h5py.File(path, "r") as f:
        return int(f.attrs.get("frames_done", 0)) == n_frames
