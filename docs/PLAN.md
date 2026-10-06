# Study design

**Question.** DINO-WM plans over a frozen DINOv2 patch grid (196 tokens per frame) and is accurate but slow;
LeWM plans over a single learned 192-dim vector and is fast. For a *frozen* foundation encoder, how does
planning success degrade as the number of visual tokens per frame shrinks, and how much planning time does
that save?

**Lever.** Pool the DINOv2 patch grid to k x k tokens (or the CLS token) before the predictor, keep the
encoder frozen, retrain only the predictor. Secondary levers: planner budget and inference engineering.

**Headline output.** Per environment, planning success against time per replan, with the released LeWM
and a full-grid DINO-WM as reference points.

## Background

- DINO-WM (Zhou et al., ICML 2025): frozen DINOv2 encoder, ViT predictor over patch tokens, action
  concatenated to every token, frame-causal mask, MSE on next-frame features, CEM planning inside MPC.
  Its CLS-only ablation (PushT 0.44, Wall 0.58, Reach 0.60 vs. 0.90 / 0.96 / 0.92 with patches) gives
  the two endpoints of the token-budget curve; the middle is open.
- LeWM (Maes et al., 2026): end-to-end ViT-Tiny encoder and predictor, one latent per frame, up to 48x
  faster planning than DINO-WM, attributed to the token count. Its evaluation protocol and datasets are
  used here.
- Planning cost model per replan: `samples x iterations x horizon` predictor passes (45,000 at the
  standard budget), each over up to 3 history frames of `tokens` tokens. The encoder runs a handful of
  times per replan.
- Success rates are sensitive to how start/goal queries are drawn ("Aim Short to Reach Far", 2026), so
  every comparison uses one pinned protocol.

## Hypotheses

- H1: success stays close to the full grid down to a moderate grid, then drops toward the CLS level.
- H2: time per replan falls with token count; a 4 x 4 grid is an order of magnitude faster than the
  full grid.
- H3: environments that need fine spatial detail (PushT block angle, Cube end-effector pose) need more
  tokens than navigation (TwoRoom).
- H4 (optional): intermediate DINOv2 layers keep more geometry and do at least as well at small k.

## Method

Harness: `stable-worldmodel` (datasets, environments, CEM planner, evaluation, model classes), on which
LeWM is built. Pooled models are `PreJEPA` with a pooling encoder wrapper and the reference predictor
at the reduced token count; training, planning and checkpoint loading reuse the library.

| variant | tokens | operation on the 14 x 14 patch grid |
|---|---|---|
| `cls` | 1 | CLS token |
| `mean` | 1 | mean over all patches |
| `g2` | 4 | adaptive average pool to 2 x 2 |
| `g4` | 16 | adaptive average pool to 4 x 4 |
| `g7` | 49 | adaptive average pool to 7 x 7 |
| full | 196 | reference checkpoint, no pooling |

Preprocessing and the predictor copy the full-grid reference (DINOv2 ViT-S/14 at 196 px, scale to
[-1, 1], predictor depth 6, 16 heads, MLP 2048, action block of 5 steps embedded to 10 dims, history 3,
MSE on next-frame visual tokens). Features are cached once per variant; predictors train from the cache.
All variants of an environment train on the same episodes for the same number of steps.

Environments: TwoRoom, PushT, Reacher, Cube (the four LeWM releases).

## Evaluation protocol

As in LeWM's `eval.py`: 50 episodes per environment, start state and goal image from a dataset episode
with the goal 25 steps ahead, 50-step budget, CEM with 300 samples, 30 iterations, 30 elites, horizon 5
action blocks of 5 steps, the whole plan executed before replanning. Queries are drawn once per
environment with a fixed seed, stored, and shared by every model.

Timing: median time per replan on one GPU type per table, warm-up noted; precision recorded per row.
Also recorded: validation prediction error of the predictor (a cheap proxy when success saturates).

Reference points: the released LeWM checkpoint, and DINO-WM without proprioception retrained with the
official code on the same datasets (the original baseline checkpoints are no longer downloadable).

## Experiments

- E0 References: LeWM and full-grid DINO-WM under the pinned protocol.
- E0b Profile one replan: encoder vs. predictor vs. planner overhead.
- E1 Token budget: the five pooled variants per environment.
- E3 Planner budget: CEM samples, iterations and horizon on the best pooled variant and the full grid.
- E4 Engineering: mixed precision, `torch.compile`, batching, reported separately from pooling gains.
- E2 (stretch) Layer axis: DINOv2 layers 6, 9, 12 at the best k.
- E5 (stretch) Feature dimension: PCA 384 -> 128 / 64 at fixed k.

## Backlog

Foreground masking at test time, event-triggered replanning, shared-prefix (KV) caching across
candidates, learned token selection, iCEM, structured predictor pruning, other encoders (DINOv3, SigLIP 2,
V-JEPA 2), cross-embodiment with one frozen encoder, test-time adaptation of the predictor.

## References

- DINO-WM: https://arxiv.org/abs/2411.04983, code https://github.com/gaoyuezhou/dino_wm
- LeWM: https://arxiv.org/abs/2603.19312, code https://github.com/lucas-maes/le-wm
- stable-worldmodel: https://github.com/galilai-group/stable-worldmodel
- Datasets and LeWM checkpoints: https://huggingface.co/collections/quentinll/lewm
- Retrained DINO-WM checkpoints: https://github.com/fafraob/point-lewm (arXiv:2608.29434)
- AdaJEPA (planning-time table): https://arxiv.org/abs/2606.32026
- Aim Short to Reach Far (protocol sensitivity): https://arxiv.org/abs/2609.30036
