# Implementation notes

Reference material for working with the code base: what the upstream library does, how the evaluation
protocol is pinned, where data and checkpoints come from, and the measurements behind the results.

Versions: `stable-worldmodel` 0.1.1 (PyPI); `galilai-group/stable-worldmodel` at `21446f1` for the
training and evaluation scripts, which are not part of the package; `lucas-maes/le-wm` at `8edfeb3`;
`fafraob/point-lewm` at `d2ba2f7`, which vendors `gaoyuezhou/dino_wm` at `0a9492f`.
Paths below are relative to the installed `stable_worldmodel` package unless marked `repo/`.

## The DINO-WM model in stable-worldmodel

- Model class `PreJEPA` (`wm/prejepa/prejepa.py`); building blocks in `wm/prejepa/module.py`.
- Encoder: a HuggingFace backbone from `create_backbone` (`module.py:46`); the CLS token is dropped and
  the patch tokens kept (`prejepa.py:81-83`). The package's own training config uses 224 px input with
  patch size 14, giving 256 tokens; the DINO-WM paper and the reference checkpoints used here resize to
  196 px, giving 196 tokens.
- Action (and optionally proprioception) embedded to 10 dims by a `Conv1d` `Embedder` (`module.py:79`),
  tiled over all patches and concatenated on the feature axis (`prejepa.py:49-63`). One token step
  carries a block of `frameskip` actions (`repo/scripts/train/prejepa.py:212-214`).
- Predictor `CausalPredictor(num_patches, num_frames, dim, depth, heads, mlp_dim, ...)`
  (`module.py:105`): learned positional embedding of shape `(1, num_frames * num_patches, dim)`
  (`module.py:129-131`), frame-causal attention mask (`module.py:225-235`). Reference size: depth 6,
  16 heads of 64, MLP 2048, dropout 0.1.
- The token count is a constructor argument, and `predict`, `rollout` and `get_cost` read it from tensor
  shapes. A pooled variant is `PreJEPA` with a pooling encoder and `CausalPredictor(num_patches=k)`;
  see `src/dwt/models.py`.
- Loss: MSE between predicted and target next-frame embeddings with the action dims removed
  (`repo/scripts/train/prejepa.py:142-145`).

## Planner

- `CEMSolver` (`solver/cem.py:17`): 300 samples, 30 iterations, 30 elites by default. All candidates of
  one environment go through `model.get_cost` in a single batch (`cem.py:210`, `prejepa.py:324`);
  environments are solved sequentially (`cem.py:152`).
- No key-value cache: each rollout step reruns the predictor on the last `history_size` frames
  (`prejepa.py:327-348`). Planning starts from one context frame (`PlanConfig.history_len` defaults to
  1, `policy.py:31`), so a horizon-5 rollout makes 5 predictor calls with 1, 2, 3, 3, 3 context frames.
- Per replan: 30 x 5 = 150 batched predictor passes; the encoder runs once for the current frame and
  once for the goal, cached by `(id, step_idx)` (`prejepa.py:239-245`, `387-395`).
- Model interface: `get_cost(info_dict, action_candidates) -> (n_envs, num_samples)` with
  `action_candidates` of shape `(n_envs, num_samples, horizon, action_dim * action_block)`
  (`solver/solver.py:7-36`). `PreJEPA` reads `pixels`, `goal`, `action`, `id`, `step_idx`.
- Checkpoints: a folder with `weights.pt` and a Hydra-instantiable `config.json`, loaded by
  `wm.utils.load_pretrained` from a local folder or a HuggingFace repo id (`wm/utils.py:47`). The
  `_object.ckpt` / `AutoCostModel` path (`policy.py:442-537`) is the older format.

## Evaluation protocol

Follows `le-wm/eval.py` and its configs, identical for the four environments: 50 episodes, goal 25 steps
ahead, 50-step budget, horizon 5 blocks of 5 steps (frameskip), full plan executed before replanning,
CEM 300 / 30 / 30.

- Queries: every dataset row whose step leaves room for the goal offset is a candidate; 50 are drawn
  without replacement with `np.random.default_rng(seed)` (`le-wm/eval.py:108-134`). A query is an
  `(episode, start_step)` pair; the goal is the same episode 25 steps later (`world/world.py:567-598`).
- `World.evaluate(dataset=...)` runs one environment per query, sets state and goal through per-env
  `callables`, and counts success if the episode terminates within the budget (`world.py:492-543`).
- The planner's noise generator uses the same seed, so the seed fixes both the queries and the
  planning noise. The drawn queries are stored in `results/queries/<env>_seed<seed>.json` and passed
  to every model.
- Datasets written by `stable-worldmodel` 0.1.1 have no `ep_idx` column; queries are derived from
  `ep_len` / `ep_offset` instead, which yields the same rows.
- Actions use a `StandardScaler` fitted on the dataset with `nan` rows removed; pooled models use the
  statistics saved at training time.

## Training data

- A training clip is `num_steps * frameskip` consecutive steps (20 for history 3 + 1 prediction);
  clips start at every step of every episode (`data/dataset.py:48-55`). Observations are subsampled at
  the frameskip, actions kept at full rate and reshaped to `(num_steps, frameskip * action_dim)`
  (`data/formats/hdf5.py:121-122`).
- A feature cache therefore holds every frame; the dataset is reduced by episode, not by frame.
- The released datasets store `nan` actions at some steps. They are excluded from the scaling
  statistics and set to zero, as the upstream training script does; the training loop stops on a
  non-finite loss so that a bad checkpoint cannot propagate through a resume.

## Datasets and checkpoints

Datasets (HuggingFace `quentinll/lewm-*`, MIT), compressed: TwoRoom 3.4 GB (12.8 GB unpacked),
PushT 13.1 GB, Reacher 23.8 GB, Cube 46.2 GB. In `stable-worldmodel` 0.1.1 they live in
`$STABLEWM_HOME/datasets/` (the LeWM README predates this).

LeWM checkpoints: `quentinll/lewm-<env>` (MIT), `weights.pt` + `config.json`, loaded by
`load_pretrained` as they are.

DINO-WM checkpoints: the baseline suite linked from the LeWM README (Google Drive) is no longer
accessible (`le-wm` issues #58, #85, #93, #95, #102). The DINO-WM reference used here is the retrain
published for arXiv:2608.29434 (`fafraob/point-cloud-<env>`, folder `dino-wm/`, CC-BY-4.0): no
proprioception, official `dino_wm` code, the four LeWM datasets, 72 hours on two A100s. Reported
success at seed 42: TwoRoom 100, Reacher 78, PushT 76, Cube 78 (LeWM paper: 100 / 79 / 74 / 86).

- Format: original `dino_wm` pickles (`checkpoints/model_latest.pth`, `hydra.yaml`). They plan inside
  `stable-worldmodel` through `dinowm_swm/planning.py::load_dinowm_cost_model` from `point-lewm` (MIT).
- Shape: 196 tokens per frame (frames resized to 196 px), token dim 404 = 384 visual + 10 proprio
  (constant, no-proprio model) + 10 action, history 3, 20.1 M predictor parameters, pixels normalised
  with mean 0.5 / std 0.5, DINOv2 ViT-S/14 from `torch.hub`.
- The adapter hard-codes bf16 autocast. Tesla T4 GPUs have no native bf16; fp16 autocast is used
  instead (2.4x faster than fp32 at equal success).
- Import `stable_worldmodel` before the adapter: the vendored `dino_wm` has a top-level `datasets`
  package that shadows HuggingFace `datasets`.

The pooled models copy the reference's preprocessing exactly (scale to [-1, 1], resize to 196 px,
`forward_features` of `dinov2_vits14`); the standalone pipeline reproduces the reference encoder's
tokens to 5e-4.

## Environment and install

- Python 3.10 to 3.12. On 3.13+ `labmaze` (a `dm_control` dependency used only for maze arenas) has no
  wheel and fails to build; an empty stand-in package satisfies the requirement. `box2d-py` needs `swig`.
- Pins and their reasons are in `pyproject.toml`: `hdf5plugin` (without it `HDF5Dataset` is silently
  absent), `mujoco<3.15` (`dm_control` 1.0.47 crashes on 3.15), `transformers<5` (ViT layer names in
  the released checkpoints), `datasets>=3` with `huggingface-hub<1` (the resolver otherwise picks an
  ancient `datasets`), `scipy==1.14.1` on macOS 27.
- Set `MUJOCO_GL=egl` on Linux only; macOS uses its default backend. `le-wm/eval.py` hard-codes `egl`
  and `cuda`.
- Google Colab: Python 3.13; Tesla T4 on the free tier (dynamic usage quota, sessions end without
  warning), A100/L4 with Colab Pro. Session disks are wiped, so datasets are re-downloaded and
  checkpoints and features are kept on Drive. Feature files held in RAM train an order of magnitude
  faster than random-access reads from HDF5.

## Measurements: TwoRoom

Reference points (notebook 01), 50 episodes, seed 42:

| model | tokens | precision | GPU | success | s per replan (median) |
|---|---|---|---|---|---|
| LeWM | 1 | fp32 | Tesla T4 | 86% | 2.06 |
| LeWM | 1 | fp32 | A100 40GB | 86% | 0.71 |
| DINO-WM full grid | 196 | fp16 | Tesla T4 | 100% (10 ep.) | 252 |
| DINO-WM full grid | 196 | fp16 | A100 40GB | 100% | 45.0 |

DINO-WM in fp32 on the T4: 613 s per replan (2 replans measured). Success rates do not depend on the
GPU; timings do, so a comparison table uses one GPU.

Feature caching (notebook 03): first 2,000 episodes (183,800 frames), 9.5 min on a T4; five variants,
10 GB in total.

Training (notebook 04): reference predictor, AdamW 5e-4, batch 64, 30,000 steps per variant, last 10%
of episodes held out. Step times: `cls` 40 ms (T4), `g4` 190 ms (T4), `g7` 144 ms (A100).

Pooled models (notebook 05), 30,000-step snapshots, fp32, A100:

| variant | tokens | val MSE | success | s per replan (median) |
|---|---|---|---|---|
| `cls` | 1 | 0.0047 | 92% | 0.72 |
| `mean` | 1 | 0.0022 | 96% | 0.70 |
| `g2` | 4 | 0.0030 | 100% | 1.54 |
| `g4` | 16 | 0.0036 | 100% | 4.81 |
| `g7` | 49 | 0.0051 | 100% | 14.9 |

Reading: the curve is flat at 100% from 4 to 196 tokens, a 29x planning-time saving at 4 tokens.
One-token models run at the planner's overhead floor (about 0.7 s per replan on the A100 at this
budget, which LeWM also pays) with success at or above LeWM's; 92 vs 86% is three episodes of 50 and
within single-seed noise. `mean` outperforming `cls` is plausible for a navigation task with one moving
agent and should not be assumed to transfer. Time per replan grows roughly linearly with tokens below
about 16 and faster above, as attention is quadratic in sequence length. The `g7` validation loss was
still decreasing at 30,000 steps.

## Licenses

`stable-worldmodel` MIT; `le-wm` MIT; LeWM datasets and checkpoints MIT; `point-lewm` MIT, with its
retrained DINO-WM checkpoints CC-BY-4.0; `dino_wm` MIT; DINOv2 Apache-2.0.
