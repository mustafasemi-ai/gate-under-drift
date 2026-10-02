"""Render the figure that carries the finding.

Two panels over the same time axis (validation, then batches 4..10).

  Left  -- what the averages show: accuracy falls, confidence follows only part
           of the way. The space between the lines is overconfidence.
  Right -- what the gate does: the error among released items, one dot per
           training run. Some batches break every run, some break three in
           fifty, some break none.

Light and dark variants are rendered separately. Palette: categorical slots
1-2 of the reference palette, checked for colour-vision deficiency and surface
contrast in both modes.

Input: the prediction cache written by `gas_seeds.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gas_study import load_gas  # noqa: E402

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e3e2df",
        "accuracy": "#2a78d6", "confidence": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "grid": "#33322f",
        "accuracy": "#3987e5", "confidence": "#d95926",
    },
}


def style(ax, c: dict, ylabel: str, title: str, labels: list[str]) -> None:
    ax.set_facecolor(c["surface"])
    ax.grid(True, axis="y", color=c["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(c["grid"])
    ax.tick_params(colors=c["ink2"], labelsize=9, length=0)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_xlabel("batch  (time →)", color=c["ink2"], fontsize=10)
    ax.set_ylabel(ylabel, color=c["ink2"], fontsize=10)
    ax.set_title(title, color=c["ink"], fontsize=12, fontweight="bold", loc="left", pad=12)


def render(stats: dict, mode: str, out: Path, threshold: float, promised: float) -> Path:
    c = THEMES[mode]
    labels = stats["labels"]
    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.9), dpi=200)
    fig.patch.set_facecolor(c["surface"])

    # --- left: the averages ------------------------------------------------
    style(ax1, c, "share of items", "What the averages show", labels)
    ax1.set_ylim(0.4, 1.03)
    ax1.fill_between(x, stats["accuracy"], stats["confidence"], color=c["confidence"],
                     alpha=0.10, linewidth=0, zorder=1)
    for key, label, dy in (("confidence", "mean confidence", 12), ("accuracy", "accuracy", -18)):
        ax1.plot(x, stats[key], color=c[key], linewidth=2.0, zorder=3)
        ax1.plot(x, stats[key], "o", color=c[key], markersize=8, markeredgecolor=c["surface"],
                 markeredgewidth=2, zorder=4)
        ax1.annotate(label, (x[-1], stats[key][-1]), textcoords="offset points",
                     xytext=(0, dy), color=c["ink"], fontsize=10, fontweight="bold", ha="right")
    ax1.annotate("overconfidence", (x[-1] - 0.12, (stats["accuracy"][-1] + stats["confidence"][-1]) / 2),
                 color=c["ink2"], fontsize=8.5, ha="right", va="center")

    # --- right: the gate, run by run --------------------------------------
    style(ax2, c, "error among released items",
          f"What the gate does at {threshold}, run by run", labels)
    top = 0.40
    ax2.set_ylim(-0.012, top)
    rng = np.random.default_rng(0)
    for i, errs in enumerate(stats["gate_error"]):
        jitter = rng.uniform(-0.22, 0.22, size=len(errs))
        ax2.plot(i + jitter, np.minimum(errs, top - 0.01), "o", color=c["accuracy"], markersize=5,
                 alpha=0.55, markeredgewidth=0, zorder=3)
        if len(errs):
            ax2.plot([i - 0.3, i + 0.3], [np.median(errs)] * 2, color=c["ink"], linewidth=2.0,
                     solid_capstyle="round", zorder=4)
        ax2.annotate(f"{int((errs > 0.05).sum())}/{len(errs)}", (i, top - 0.012), color=c["ink2"],
                     fontsize=8.5, ha="center", va="top")
    ax2.axhline(promised, color=c["ink2"], linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.7, zorder=2)
    ax2.annotate(f"promised: {promised:.0%}", (-0.55, promised), textcoords="offset points",
                 xytext=(0, 5), color=c["ink2"], fontsize=8.5, ha="left")

    fig.suptitle(
        "Accuracy drifts down in every batch. The gate breaks in some, and not in every run.",
        color=c["ink"], fontsize=13.5, fontweight="bold", x=0.012, ha="left", y=0.99,
    )
    fig.text(
        0.012, 0.015,
        f"UCI Gas Sensor Array Drift, MLP trained on batches 1–3, {stats['n_runs']} training runs.  "
        "Left: mean over runs.\n"
        "Right: one dot per run, bar = median, top row = runs with more than 5% error; "
        "runs releasing fewer than 30 items in a batch are omitted.",
        color=c["ink2"], fontsize=9, ha="left",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.945))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=c["surface"], bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Render the gate figure")
    ap.add_argument("--data-dir", type=Path, default=Path("data/gas"))
    ap.add_argument("--cache", type=Path, default=Path("runs/gas/seeds.npz"))
    ap.add_argument("--train-batches", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--promised", type=float, default=0.01)
    ap.add_argument("--out-dir", type=Path, default=Path("docs"))
    a = ap.parse_args()

    _, y, batch = load_gas(a.data_dir)
    probs = np.load(a.cache)["probs"]
    conf, ok = probs.max(axis=2), probs.argmax(axis=2) == y
    groups = [("validation", batch <= a.train_batches)]
    groups += [(str(b), batch == b) for b in range(a.train_batches + 1, 11)]

    stats = {"labels": [g for g, _ in groups], "n_runs": len(probs),
             "accuracy": [], "confidence": [], "gate_error": []}
    for _, m in groups:
        stats["accuracy"].append(float(ok[:, m].mean()))
        stats["confidence"].append(float(conf[:, m].mean()))
        errs = []
        for i in range(len(probs)):
            rel = m & (conf[i] >= a.threshold)
            if rel.sum() >= 30:
                errs.append(1.0 - ok[i][rel].mean())
        stats["gate_error"].append(np.asarray(errs))

    for mode in ("light", "dark"):
        print(f"  -> {render(stats, mode, a.out_dir / f'gate-{mode}.png', a.threshold, a.promised)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
