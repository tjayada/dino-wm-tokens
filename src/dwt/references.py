"""The two reference models: the released LeWM and a full-grid DINO-WM (third-party retrain)."""

import subprocess
import sys
import urllib.request
from pathlib import Path

import torch
from torchvision.transforms import v2 as transforms

from .envs import EnvSpec
from .paths import data_root

ADAPTER_URL = "https://github.com/fafraob/point-lewm.git"
ADAPTER_COMMIT = "d2ba2f7e3a1cacdddb3b2991dcc7bd463b733a48"
DINOWM_FILES = ("hydra.yaml", "swm_h5_norm_stats.json", "checkpoints/model_latest.pth")


def lewm_transform():
    import stable_pretraining as spt

    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=224),
        ]
    )


def load_lewm(spec: EnvSpec, device: str):
    import stable_worldmodel as swm

    model = swm.wm.utils.load_pretrained(spec.lewm_repo).to(device).eval()
    model.requires_grad_(False)
    return model


def _adapter_dir() -> Path:
    path = data_root() / "third_party" / "point-lewm"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", ADAPTER_URL, str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", "-q", ADAPTER_COMMIT], check=True)
    return path


def _import_adapter():
    """Import the adapter with HuggingFace `datasets` stepped aside: the vendored dino_wm shadows it."""
    import stable_worldmodel  # noqa: F401  must come first

    adapter = _adapter_dir()
    dino_root = str(adapter / "third_party" / "dino_wm")
    is_ds = lambda k: k == "datasets" or k.startswith("datasets.")  # noqa: E731
    hf = {k: sys.modules.pop(k) for k in list(sys.modules) if is_ds(k)}
    sys.path[:0] = [dino_root, str(adapter)]
    from dinowm_swm import planning

    for k in [k for k in sys.modules if is_ds(k)]:
        del sys.modules[k]
    sys.modules.update(hf)
    sys.path.remove(dino_root)
    sys.path.append(dino_root)  # the checkpoint unpickles `models.*`
    return planning


def load_dinowm(spec: EnvSpec, device: str):
    """Full-grid DINO-WM (196 tokens) through the point-lewm adapter; fp16 autocast on CUDA."""
    planning = _import_adapter()
    folder = data_root() / "checkpoints" / f"dinowm_{spec.name}"
    for name in DINOWM_FILES:
        target = folder / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{spec.dinowm_repo}/resolve/main/dino-wm/{name}", target)
    model = planning.load_dinowm_cost_model(folder, device=device)
    model.requires_grad_(False)
    planning._autocast = lambda enabled: torch.autocast("cuda", dtype=torch.float16, enabled=bool(enabled))
    model.bf16 = device == "cuda"
    return model
