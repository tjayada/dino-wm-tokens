"""Pooled-token DINO-WM variants on stable_worldmodel's PreJEPA.

The frozen DINOv2 ViT-S/14 is fed as the full-grid reference feeds it (frames in [-1, 1], resized to
196 px, `forward_features`); its 14 x 14 patch grid is then pooled to the variant's token count.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import v2 as transforms

GRID = 14
FEAT_DIM = 384
ENCODER_PX = GRID * 14
HUB_NAME = "dinov2_vits14"
TOKENS = {"cls": 1, "mean": 1, "g2": 4, "g4": 16, "g7": 49, "full": GRID * GRID, "r1": 1, "r4": 4}
_GRID_SIDE = {"g2": 2, "g4": 4, "g7": 7, "full": GRID}
READOUT_SOURCE = "g7"  # learned-readout variants (r*) read the cached 7 x 7 tokens


def feature_variant(variant: str) -> str:
    """Which cached feature file a variant trains from."""
    return READOUT_SOURCE if variant.startswith("r") else variant


def pixel_transform(img_size=224):
    """uint8 HWC frame -> float CHW in [-1, 1], as the reference's `default_transform`."""
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def pool_tokens(cls_token, patches, variant):
    """cls_token (B, 384), patches (B, 196, 384) -> (B, TOKENS[variant], 384)."""
    if variant == "cls":
        return cls_token[:, None]
    if variant == "mean":
        return patches.mean(1, keepdim=True)
    side = _GRID_SIDE[variant]
    if side == GRID:
        return patches
    grid = patches.transpose(1, 2).reshape(patches.shape[0], FEAT_DIM, GRID, GRID)
    if grid.device.type == "mps":  # MPS cannot pool to non-divisible sizes
        grid = grid.cpu()
    return F.adaptive_avg_pool2d(grid, side).flatten(2).transpose(1, 2).to(patches.device)


class PooledDino(nn.Module):
    """Frozen DINOv2 -> pooled tokens, in the `.last_hidden_state` form PreJEPA expects.

    PreJEPA drops the first token as CLS. The backbone is kept out of the module tree so checkpoints
    hold only the trained parts; it follows the input's device.
    """

    def __init__(self, variant, hub_name=HUB_NAME):
        super().__init__()
        assert variant in TOKENS, f"unknown variant {variant!r}"
        self.variant = variant
        self.pool_variant = variant  # what the frozen grid is pooled to (ReadoutDino overrides `variant`)
        self.hub_name = hub_name
        backbone = torch.hub.load("facebookresearch/dinov2", hub_name).eval()
        backbone.requires_grad_(False)
        self._backbone = [backbone]
        self.config = SimpleNamespace(hidden_size=FEAT_DIM)

    @property
    def backbone(self):
        return self._backbone[0]

    @property
    def num_tokens(self):
        return TOKENS[self.variant]

    @torch.no_grad()
    def forward(self, pixels, **kwargs):
        """pixels (B, 3, H, W) in [-1, 1] -> last_hidden_state (B, 1 + tokens, 384)."""
        backbone = self.backbone
        if next(backbone.parameters()).device != pixels.device:
            backbone.to(pixels.device)
        x = F.interpolate(
            pixels, size=(ENCODER_PX, ENCODER_PX), mode="bilinear", antialias=True, align_corners=False
        )
        out = backbone.forward_features(x)
        tokens = pool_tokens(out["x_norm_clstoken"], out["x_norm_patchtokens"], self.pool_variant)
        return SimpleNamespace(last_hidden_state=torch.cat([out["x_norm_clstoken"][:, None], tokens], dim=1))


class CrossBlock(nn.Module):
    """Queries attend to the input tokens, then to each other, then an MLP (pre-norm)."""

    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm_q, self.norm_kv, self.norm_s, self.norm_m = (nn.LayerNorm(dim) for _ in range(4))
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))

    def forward(self, q, x):
        kv = self.norm_kv(x)
        q = q + self.cross(self.norm_q(q), kv, kv, need_weights=False)[0]
        s = self.norm_s(q)
        q = q + self.self_attn(s, s, s, need_weights=False)[0]
        return q + self.mlp(self.norm_m(q))


class Readout(nn.Module):
    """k learned queries that cross-attend to n_in frozen tokens (Set Transformer / Perceiver pooling)."""

    def __init__(self, k, n_in=TOKENS[READOUT_SOURCE], dim=FEAT_DIM, heads=6, depth=2):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, n_in, dim) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, k, dim) * 0.02)
        self.blocks = nn.ModuleList([CrossBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens):
        """(B, n_in, dim) -> (B, k, dim)."""
        x = tokens + self.pos
        q = self.queries.expand(tokens.shape[0], -1, -1)
        for block in self.blocks:
            q = block(q, x)
        return self.norm(q)


class ReadoutDino(PooledDino):
    """Frozen DINOv2 -> fixed 7 x 7 pooling -> learned readout to k tokens."""

    def __init__(self, variant, hub_name=HUB_NAME, **readout_kwargs):
        super().__init__(READOUT_SOURCE, hub_name)
        self.variant = variant
        self.readout = Readout(TOKENS[variant], **readout_kwargs)

    @property
    def num_tokens(self):
        return TOKENS[self.variant]

    def forward(self, pixels, **kwargs):
        with torch.no_grad():
            source = super().forward(pixels).last_hidden_state
        tokens = self.readout(source[:, 1:].float())
        return SimpleNamespace(last_hidden_state=torch.cat([source[:, :1], tokens], dim=1))


def model_config(
    variant,
    action_dim=2,
    frameskip=5,
    history=3,
    action_emb=10,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dim_head=64,
    dropout=0.1,
):
    """Hydra-instantiable config of a pooled DINO-WM with the reference predictor hyperparameters."""
    return {
        "_target_": "stable_worldmodel.wm.PreJEPA",
        "history_size": history,
        "num_pred": 1,
        "interpolate_pos_encoding": False,
        "encoder": {
            "_target_": "dwt.models.ReadoutDino" if variant.startswith("r") else "dwt.models.PooledDino",
            "variant": variant,
        },
        "predictor": {
            "_target_": "stable_worldmodel.wm.prejepa.module.CausalPredictor",
            "num_patches": TOKENS[variant],
            "num_frames": history,
            "dim": FEAT_DIM + action_emb,
            "depth": depth,
            "heads": heads,
            "mlp_dim": mlp_dim,
            "dim_head": dim_head,
            "dropout": dropout,
            "emb_dropout": 0.0,
        },
        "extra_encoders": {
            "_target_": "torch.nn.ModuleDict",
            "modules": {
                "action": {
                    "_target_": "stable_worldmodel.wm.prejepa.module.Embedder",
                    "in_chans": action_dim * frameskip,
                    "emb_dim": action_emb,
                }
            },
        },
    }


def build_model(variant, **kwargs):
    from hydra.utils import instantiate

    return instantiate(model_config(variant, **kwargs))


def trainable_parameters(model):
    """Everything registered: predictor, action embedder and, if present, the readout (DINOv2 is outside)."""
    return list(model.parameters())
