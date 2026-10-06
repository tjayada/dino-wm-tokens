"""Success against time per replan, one panel per environment."""

from pathlib import Path

import pandas as pd

COLORS = {"pooled": "#2a78d6", "dinowm_noprop": "#eb6834", "lewm": "#1baf7a"}
LABELS = {"dinowm_noprop": "DINO-WM full (196)", "lewm": "LeWM (1)"}


def latest_rows(runs_csv: Path, env: str) -> pd.DataFrame:
    """One row per method for `env`: the most recent run, all on one GPU."""
    runs = pd.read_csv(runs_csv)
    rows = runs[runs.env == env].drop_duplicates("method", keep="last")
    gpus = rows.gpu.unique()
    assert len(gpus) == 1, f"rows for {env} were timed on different GPUs: {gpus}"
    return rows


def pareto(runs_csv: Path, env: str, out_png: Path, title: str | None = None):
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    rows = latest_rows(runs_csv, env).sort_values("time_per_replan_median_s")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, row in enumerate(rows.itertuples()):
        family = "pooled" if row.method.startswith("pooled_") else row.method
        label = LABELS.get(family, f"{row.method[7:]} ({row.tokens_per_frame})")
        ax.scatter(
            row.time_per_replan_median_s, row.success_rate, s=60, color=COLORS.get(family, "#888"), zorder=3
        )
        ax.annotate(
            label,
            (row.time_per_replan_median_s, row.success_rate),
            xytext=(6, 6 if i % 2 == 0 else -12),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_xlabel(f"time per replan (s, median, {rows.gpu.iloc[0]})")
    ax.set_ylabel(f"success rate (%), {int(rows.n_episodes.max())} episodes")
    ax.set_ylim(0, 105)
    ax.set_title(
        title
        or f"{env}: planning success vs. time, CEM {int(rows.cem_samples.iloc[0])} x "
        f"{int(rows.cem_iters.iloc[0])}, horizon {int(rows.horizon.iloc[0])}"
    )
    ax.grid(True, which="major", color="#dddddd", linewidth=0.6, zorder=0)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=c, label=lbl)
        for lbl, c in [
            ("frozen DINOv2, pooled tokens (ours)", COLORS["pooled"]),
            ("DINO-WM full grid (reference)", COLORS["dinowm_noprop"]),
            ("LeWM (reference)", COLORS["lewm"]),
        ]
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return rows
