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


def tokens_curve(runs_csv: Path, env: str, out_png: Path):
    """Success against tokens per frame per (pooling family, training set), with the references."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    runs = pd.read_csv(runs_csv)
    rows = runs[runs.env == env]
    refs = rows[rows.method.isin(LABELS)].drop_duplicates("method", keep="last")
    pooled = rows[rows.method.str.startswith("pooled_")].drop_duplicates(
        ["method", "train_episodes"], keep="last"
    )
    pooled = pooled.assign(family=pooled.pooling.where(pooled.pooling == "readout", "fixed"))
    styles = {"fixed": ("#2a78d6", "o"), "readout": ("#1baf7a", "s")}
    most = pooled.train_episodes.max()

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for (family, n_train), grp in sorted(pooled.groupby(["family", "train_episodes"])):
        color, marker = styles[family]
        alpha = 1.0 if n_train == most else 0.45
        name = "fixed pooling" if family == "fixed" else "learned readout"
        line = grp[grp.method != "pooled_mean"].sort_values(
            "tokens_per_frame"
        )  # `mean` shares x=1 with `cls`
        ax.plot(
            line.tokens_per_frame,
            line.success_rate,
            marker=marker,
            color=color,
            alpha=alpha,
            linewidth=1.5,
            markersize=7,
            label=f"{name}, {int(round(n_train / 0.9, -3)):,} episodes",
            zorder=3,
        )
        mean = grp[grp.method == "pooled_mean"]
        if len(mean):
            ax.scatter(
                mean.tokens_per_frame, mean.success_rate, marker="x", color=color, alpha=alpha, zorder=3
            )
            ax.annotate(
                "mean",
                (1, float(mean.success_rate.iloc[0])),
                xytext=(5, -4),
                textcoords="offset points",
                fontsize=8,
                color=color,
                alpha=alpha,
            )
    for ref in refs.itertuples():
        ax.axhline(ref.success_rate, color=COLORS[ref.method], linestyle="--", linewidth=1.2, zorder=2)
        ax.annotate(
            f"{LABELS[ref.method]}, all episodes",
            (196, ref.success_rate),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=8,
            color=COLORS[ref.method],
            ha="right",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 49, 196])
    ax.set_xticklabels(["1", "4", "16", "49", "196"])
    ax.set_xlabel("visual tokens per frame (frozen DINOv2 ViT-S/14)")
    ax.set_ylabel("success rate (%), 50 episodes")
    ax.set_ylim(0, 105)
    ax.set_title(f"{env}: planning success vs. token budget")
    ax.grid(True, which="major", color="#dddddd", linewidth=0.6, zorder=0)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
