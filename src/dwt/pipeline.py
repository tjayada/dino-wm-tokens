"""Command line: `dwt run --env pusht --stages references,features,train,evaluate,plot`.

Environment variables: STABLEWM_HOME (local data root), DWT_PERSIST (mirror that survives sessions),
DWT_RESULTS (results folder, default ./results).
"""

import argparse
import json
import os
import sys
import warnings

STAGES = ("references", "features", "train", "evaluate", "plot")


def main(argv=None):
    p = argparse.ArgumentParser(prog="dwt")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run pipeline stages for one environment")
    r.add_argument("--env", required=True)
    r.add_argument("--stages", default=",".join(STAGES))
    r.add_argument("--variants", default="cls,mean,g2,g4,g7")
    r.add_argument("--episodes", type=int, default=2000, help="episodes to cache and train on (0 = all)")
    r.add_argument("--steps", type=int, default=30_000)
    r.add_argument("--milestones", default="10000,30000")
    r.add_argument("--snapshot", type=int, default=None, help="milestone to evaluate (default: --steps)")
    r.add_argument("--n-eval", type=int, default=50)
    r.add_argument("--n-eval-dinowm", type=int, default=50)
    r.add_argument("--device", default=None)
    r.add_argument("--smoke", action="store_true", help="tiny local run without a GPU")
    args = p.parse_args(argv)
    if args.command == "run":
        run(args)


def run(args):
    if sys.platform == "linux":
        os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("WANDB_MODE", "disabled")
    warnings.filterwarnings("ignore")
    import torch

    from . import data, envs, evaluate, features, references, train
    from .models import TOKENS, feature_variant, pixel_transform
    from .paths import data_root, persist_root, results_root

    spec = envs.ENVS[args.env]
    device = args.device or (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    smoke = args.smoke
    stages = args.stages.split(",")
    variants = args.variants.split(",")
    results = results_root()
    runs_csv = results / ("runs_smoke.csv" if smoke else "runs.csv")
    cem = {"num_samples": 16, "n_steps": 2, "topk": 4} if smoke else envs.CEM
    n_eval = 2 if smoke else args.n_eval
    n_eval_dinowm = 2 if smoke else args.n_eval_dinowm
    steps = 30 if smoke else args.steps
    milestones = (10, 30) if smoke else tuple(int(m) for m in args.milestones.split(","))
    n_episodes = None if (smoke or args.episodes == 0) else args.episodes
    print(
        f"dwt run | env {spec.name} | device {device} | stages {stages} | data {data_root()} | "
        f"mirror {persist_root()} | results {results}{' | SMOKE' if smoke else ''}"
    )
    if not spec.tested:
        print(f"note: {spec.name} has not been run end to end yet")

    if smoke:
        data.collect_smoke_dataset(spec)
    else:
        data.ensure_dataset(spec)
    dataset = data.load_dataset(spec, smoke)
    queries = (
        data.pinned_queries(
            spec,
            dataset,
            results,
            envs.PROTOCOL["seed"],
            envs.PROTOCOL["n_eval"],
            envs.PROTOCOL["goal_offset"],
        )
        if not smoke
        else data.draw_queries(dataset, envs.PROTOCOL["seed"], n_eval, envs.PROTOCOL["goal_offset"])
    )
    scaler = data.action_scaler(dataset)
    videos = data_root() / "videos" / spec.name

    def evaluate_spec(model_spec, n):
        result = evaluate.run_eval(
            model_spec, spec, dataset, queries, n, device, cem=cem, video_dir=videos / model_spec.name
        )
        evaluate.log_run(runs_csv, model_spec, spec, result, device, cem=cem)
        print(
            f"{model_spec.name}: success {result['success_rate']:.0f}% | "
            f"{result['time_per_replan_median_s']:.2f} s per replan (median) | "
            f"wall {result['wall_time_s'] / 60:.1f} min"
        )

    if "references" in stages:
        lewm = references.load_lewm(spec, device)
        evaluate_spec(
            evaluate.ModelSpec(
                "lewm", lewm, references.lewm_transform(), {"action": scaler}, 1, 192, "vit-tiny (end-to-end)"
            ),
            n_eval,
        )
        del lewm
        dinowm = references.load_dinowm(spec, device)
        evaluate_spec(
            evaluate.ModelSpec(
                "dinowm_noprop",
                dinowm,
                dinowm.img_transform,
                {"action": scaler},
                196,
                384,
                "dinov2_vits14 (frozen)",
                precision="fp16" if dinowm.bf16 else "fp32",
            ),
            n_eval_dinowm,
        )
        del dinowm

    tag = None
    if {"features", "train", "evaluate"} & set(stages):
        files = (
            features.cache_features(spec, n_episodes, device, smoke=smoke) if "features" in stages else None
        )
        n_ep = n_episodes if n_episodes is not None else len(dataset.lengths)
        tag = features.feature_tag(spec, n_ep, smoke)
        needed = {feature_variant(v) for v in variants}
        if files is None:
            missing = [v for v in needed if not features.feature_path(tag, v).exists()]
            assert not missing, f"features missing for {missing}: run the features stage"

    if "train" in stages:

        def probe(model, n=2 if smoke else 10):
            """Planning success on the first n pinned queries, reported at milestones."""
            model_spec = evaluate.ModelSpec("probe", model, pixel_transform(), {"action": scaler}, 0, 384, "")
            return evaluate.run_eval(model_spec, spec, dataset, queries, n, device, cem=cem)["success_rate"]

        for v in variants:
            train.train_variant(
                spec,
                tag,
                v,
                steps,
                device,
                batch_size=8 if smoke else (128 if v.startswith("r") else 64),  # readouts: LeWM's batch
                milestones=milestones,
                eval_every=10 if smoke else 500,
                ckpt_every=10 if smoke else 1000,
                log_dir=results / "training",
                probe=probe,
            )

    if "evaluate" in stages:
        import stable_worldmodel as swm

        from .paths import restore

        snapshot = args.snapshot or steps
        for v in variants:
            name = f"{train.run_name(tag, v)}_step{snapshot}"
            folder = data_root() / "checkpoints" / name
            assert restore(folder, "checkpoints"), f"checkpoint {name} not found"
            meta = json.loads((folder / "meta.json").read_text())
            model = swm.wm.utils.load_pretrained(name).to(device).eval()
            model.requires_grad_(False)
            evaluate_spec(
                evaluate.ModelSpec(
                    f"pooled_{v}",
                    model,
                    pixel_transform(),
                    {"action": data.scaler_from_stats(meta["action_mean"], meta["action_std"])},
                    TOKENS[v],
                    384,
                    "dinov2_vits14 (frozen)" + (" + learned readout" if v.startswith("r") else ""),
                    pooling="readout" if v.startswith("r") else v,
                    train_episodes=str(meta["train_episodes"]),
                    notes={"checkpoint": name, "val_mse": f"{meta['val_mse']:.4f}"},
                ),
                n_eval,
            )
            del model

    if "plot" in stages:
        from .plots import pareto

        out = results / "figures" / f"pareto_{spec.name}{'_smoke' if smoke else ''}.png"
        rows = pareto(runs_csv, spec.name, out)
        print(
            rows[
                ["method", "tokens_per_frame", "n_episodes", "success_rate", "time_per_replan_median_s"]
            ].to_string(index=False)
        )
        print("figure:", out)


if __name__ == "__main__":
    main()
