"""Train one predictor per token budget on cached features, with the reference recipe.

Learned-readout variants (`r1`, `r4`) train the readout jointly with the predictor and add LeWM's
SIGReg on the readout tokens; everything else is the DINO-WM recipe.
"""

import csv
import json
import time
from pathlib import Path

import h5py
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from .envs import EnvSpec
from .features import feature_path
from .models import FEAT_DIM, TOKENS, build_model, feature_variant, model_config, trainable_parameters
from .paths import data_root, persist, restore

HISTORY, FRAMESKIP = 3, 5
N_STEPS_CLIP = HISTORY + 1
SPAN = N_STEPS_CLIP * FRAMESKIP
FRAME_OFFSETS = np.arange(N_STEPS_CLIP) * FRAMESKIP
LR, WEIGHT_DECAY, GRAD_CLIP = 5e-4, 0.0, 1.0
SIGREG_WEIGHT = 0.09  # LeWM's value; readout variants also use LeWM's batch size (128), see pipeline


def run_name(tag: str, variant: str) -> str:
    return f"dwt_{tag}_{variant}"


class Features:
    """One variant's features; in RAM when free memory allows, else gathered from HDF5 (slow)."""

    def __init__(self, path: Path, device: str, val_fraction: float = 0.1):
        self.device = device
        self.f = h5py.File(path, "r")
        self.in_ram = path.stat().st_size <= 0.6 * psutil.virtual_memory().available
        self.feat = self.f["feat"][:] if self.in_ram else self.f["feat"]
        self.tokens = self.feat.shape[1]
        ep_len, ep_offset = self.f["ep_len"][:], self.f["ep_offset"][:]
        actions = self.f["action"][:].astype(np.float32)
        valid = ~np.isnan(actions).any(axis=1)
        self.action_mean, self.action_std = actions[valid].mean(0), actions[valid].std(0)
        self.actions = np.nan_to_num((actions - self.action_mean) / self.action_std, nan=0.0).astype(
            np.float32
        )
        self.action_dim = actions.shape[1]
        n_val = max(1, round(len(ep_len) * val_fraction))
        self.episodes = {
            "train": np.arange(len(ep_len) - n_val),
            "val": np.arange(len(ep_len) - n_val, len(ep_len)),
        }
        self.clips = {
            split: np.array([ep_offset[e] + s for e in eps for s in range(ep_len[e] - SPAN + 1)])
            for split, eps in self.episodes.items()
        }

    def batch(self, starts: np.ndarray):
        """starts (B,) -> frames (B, 4, tokens, 384), action blocks (B, 4, 5 * action_dim)."""
        rows = (starts[:, None] + FRAME_OFFSETS[None]).reshape(-1)
        if self.in_ram:
            frames = self.feat[rows]
        else:
            unique_rows, inverse = np.unique(rows, return_inverse=True)
            frames = self.feat[unique_rows][inverse]
        frames = torch.from_numpy(frames.astype(np.float32)).view(
            len(starts), N_STEPS_CLIP, self.tokens, FEAT_DIM
        )
        act_rows = starts[:, None] + np.arange(SPAN)[None]
        actions = torch.from_numpy(self.actions[act_rows]).view(
            len(starts), N_STEPS_CLIP, FRAMESKIP * self.action_dim
        )
        return frames.to(self.device), actions.to(self.device)

    def close(self):
        self.f.close()


def embed(model, frames):
    """Cached tokens (B, T, n, 384) -> the predictor's tokens, through the readout if the model has one."""
    readout = getattr(model.backbone, "readout", None)
    if readout is None:
        return frames
    B, T = frames.shape[:2]
    return readout(frames.flatten(0, 1)).view(B, T, -1, FEAT_DIM)


def loss_fn(model, frames, actions, sigreg=None):
    """Teacher-forced next-token MSE on the visual dims (the reference's z_loss without proprio), plus
    SIGReg on the readout tokens when a readout is trained (LeWM's objective). Returns (loss, parts)."""
    tokens = embed(model, frames)
    act_emb = model.extra_encoders["action"](actions)
    emb = torch.cat([tokens, act_emb[:, :, None].expand(-1, -1, tokens.shape[2], -1)], dim=-1)
    pred = model.predict(emb[:, :HISTORY])
    mse = F.mse_loss(pred[..., :FEAT_DIM], emb[:, 1:, :, :FEAT_DIM])
    loss, parts = mse, {"mse": mse.item()}
    if sigreg is not None:
        B, T, k, D = tokens.shape
        reg = sigreg(
            tokens.permute(1, 2, 0, 3).reshape(T * k, B, D)
        )  # each (frame, token) slot over the batch
        loss = mse + SIGREG_WEIGHT * reg
        parts["sigreg"] = reg.item()
        parts["token_std"] = tokens.flatten(0, 2).std(0).mean().item()  # -> 0 means collapse
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss {parts}")
    return loss, parts


@torch.no_grad()
def validate(model, feats: Features, rng, batch_size: int, sigreg=None, n_batches: int = 20) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        starts = rng.choice(feats.clips["val"], size=min(batch_size, len(feats.clips["val"])), replace=False)
        losses.append(loss_fn(model, *feats.batch(starts), sigreg)[1]["mse"])
    model.train()
    return float(np.mean(losses))


def train_variant(
    spec: EnvSpec,
    tag: str,
    variant: str,
    steps: int,
    device: str,
    batch_size: int = 64,
    milestones: tuple[int, ...] = (10_000, 30_000),
    eval_every: int = 500,
    ckpt_every: int = 1000,
    log_dir: Path = Path("results/training"),
    seed: int = 0,
    probe=None,
) -> dict:
    """`probe(model) -> success %`, if given, plans on a few pinned queries at every milestone."""
    import stable_worldmodel as swm
    from stable_worldmodel.wm.loss import SIGReg

    name = run_name(tag, variant)
    folder = data_root() / "checkpoints" / name
    restore(folder, "checkpoints")
    log_path = log_dir / f"{name}.csv"
    log_dir.mkdir(parents=True, exist_ok=True)
    if not log_path.exists() and (folder / log_path.name).exists():  # the curve travels with the checkpoint
        log_path.write_bytes((folder / log_path.name).read_bytes())

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    feats = Features(feature_path(tag, feature_variant(variant)), device)
    model = build_model(variant, action_dim=feats.action_dim).to(device).train()
    params = trainable_parameters(model)
    sigreg = SIGReg().to(device) if variant.startswith("r") else None
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    step = 0
    if (folder / "train_state.pt").exists():
        state = torch.load(folder / "train_state.pt", map_location=device)
        model.load_state_dict(torch.load(folder / "weights.pt", map_location=device))
        opt.load_state_dict(state["optimizer"])
        step = state["step"]
        rng = np.random.default_rng(seed + step)
        if not all(torch.isfinite(p).all() for p in params):
            raise RuntimeError(f"{folder} holds non-finite weights; delete it and its mirror")
        print(f"{name}: resuming at step {step}")
    meta_path = folder / "meta.json"
    if step >= steps:
        print(f"{name}: already trained for {steps} steps")
        feats.close()
        return json.loads(meta_path.read_text())

    config = OmegaConf.create(model_config(variant, action_dim=feats.action_dim))

    def save(val_loss, train_loss):
        swm.wm.utils.save_pretrained(model, name, config=config)
        torch.save({"optimizer": opt.state_dict(), "step": step}, folder / "train_state.pt")
        meta = {
            "variant": variant,
            "tokens_per_frame": TOKENS[variant],
            "features": tag,
            "train_episodes": int(len(feats.episodes["train"])),
            "val_episodes": int(len(feats.episodes["val"])),
            "steps_done": step,
            "steps_planned": steps,
            "batch": batch_size,
            "lr": LR,
            "action_mean": feats.action_mean.tolist(),
            "action_std": feats.action_std.tolist(),
            "val_mse": val_loss,
            "train_mse": train_loss,
            "seed": seed,
        }
        meta_path.write_text(json.dumps(meta, indent=1))
        (folder / log_path.name).write_bytes(log_path.read_bytes())
        persist(folder, "checkpoints")
        if step in milestones:
            snapshot = folder.with_name(f"{name}_step{step}")
            swm.wm.utils.save_pretrained(model, snapshot.name, config=config)
            (snapshot / "meta.json").write_text(json.dumps(meta, indent=1))
            persist(snapshot, "checkpoints")
        return meta

    with log_path.open("a", newline="") as log:
        writer = csv.writer(log)
        if log_path.stat().st_size == 0:
            writer.writerow(
                ["variant", "step", "train_mse", "val_mse", "elapsed_s", "sigreg", "token_std", "probe"]
            )
        t0, recent, meta, parts = time.perf_counter(), [], None, {}
        val_loss = validate(model, feats, rng, batch_size, sigreg)
        print(
            f"{name}: {TOKENS[variant]} tokens | {sum(p.numel() for p in params) / 1e6:.1f} M trainable | "
            f"val MSE {val_loss:.4f} | features {'in RAM' if feats.in_ram else 'from disk'}"
        )
        while step < steps:
            starts = rng.choice(feats.clips["train"], size=batch_size, replace=False)
            loss, parts = loss_fn(model, *feats.batch(starts), sigreg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            step += 1
            recent.append(parts["mse"])
            if step == 100:
                per_step = (time.perf_counter() - t0) / 100
                print(
                    f"  {per_step * 1000:.0f} ms per step -> about {(steps - step) * per_step / 60:.0f} min"
                )
            if step % eval_every == 0 or step == steps:
                val_loss = validate(model, feats, rng, batch_size, sigreg)
                train_loss, recent = float(np.mean(recent)), []
                probe_success = ""
                if probe is not None and (step in milestones or step == steps):
                    model.eval()
                    probe_success = f"{probe(model):.0f}"
                    model.train()
                line = f"  step {step:6d} | train {train_loss:.4f} | val {val_loss:.4f}"
                if sigreg is not None:
                    line += f" | sigreg {parts['sigreg']:.3f} | token std {parts['token_std']:.3f}"
                if probe_success:
                    line += f" | probe success {probe_success}%"
                print(line)
                writer.writerow(
                    [
                        variant,
                        step,
                        f"{train_loss:.6f}",
                        f"{val_loss:.6f}",
                        f"{time.perf_counter() - t0:.0f}",
                        f"{parts['sigreg']:.5f}" if sigreg is not None else "",
                        f"{parts['token_std']:.5f}" if sigreg is not None else "",
                        probe_success,
                    ]
                )
                log.flush()
            if step % ckpt_every == 0 or step == steps:
                meta = save(val_loss, train_loss)
    feats.close()
    return meta
