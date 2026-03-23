"""
Plot training curves from JSONL logs.

Usage:
    python scripts/plot_training.py --runs outputs/runs/sfp_planner_full outputs/runs/sfp_planner_obj outputs/runs/sfp_planner_proprio
    python scripts/plot_training.py --runs_dir outputs/runs --prefix sfp_planner
    python scripts/plot_training.py --runs outputs/runs/sfp_planner_full  # single run
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_log(run_dir: str):
    path = os.path.join(run_dir, "train_log.jsonl")
    if not os.path.exists(path):
        print(f"[warn] No log found: {path}")
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_series(rows, key):
    xs, ys = [], []
    for r in rows:
        if key in r:
            xs.append(r["epoch"])
            ys.append(r[key])
    return xs, ys


def run_label(run_dir: str):
    return os.path.basename(run_dir.rstrip("/"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--runs_dir", default=None, help="Auto-discover runs under this dir")
    parser.add_argument("--prefix", default="", help="Filter runs by prefix when using --runs_dir")
    parser.add_argument("--out", default=None, help="Output PNG path (auto if not set)")
    args = parser.parse_args()

    runs = list(args.runs)
    if args.runs_dir:
        for name in sorted(os.listdir(args.runs_dir)):
            if name.startswith(args.prefix):
                p = os.path.join(args.runs_dir, name)
                if os.path.isdir(p) and os.path.exists(os.path.join(p, "train_log.jsonl")):
                    runs.append(p)

    if not runs:
        print("No runs found. Use --runs or --runs_dir.")
        sys.exit(1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load all logs
    data = {r: load_log(r) for r in runs}
    labels = [run_label(r) for r in runs]
    colors = plt.cm.tab10.colors

    # Determine which val metrics exist
    val_keys_set = set()
    for rows in data.values():
        for row in rows:
            for k in row:
                if k.startswith("val/"):
                    val_keys_set.add(k)
    val_keys = sorted(val_keys_set)

    train_loss_keys = [
        "train/total_loss",
        "train/action_flow_loss",
        "train/prior_flow_loss",
        "train/semantic_future_loss",
    ]
    # only keep keys that appear in at least one run
    train_loss_keys = [k for k in train_loss_keys if any(k in r for rows in data.values() for r in rows)]

    n_train = len(train_loss_keys)
    n_val = len(val_keys)
    total_plots = n_train + n_val
    if total_plots == 0:
        print("No metrics to plot.")
        sys.exit(0)

    ncols = min(3, total_plots)
    nrows = (total_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten() if total_plots > 1 else [axes]

    all_keys = train_loss_keys + val_keys
    for ax_idx, key in enumerate(all_keys):
        ax = axes[ax_idx]
        for i, (run_dir, rows) in enumerate(data.items()):
            xs, ys = get_series(rows, key)
            if xs:
                ax.plot(xs, ys, label=labels[i], color=colors[i % len(colors)], linewidth=1.5, marker="." if len(xs) < 30 else None)
        ax.set_title(key.replace("train/", "").replace("val/", "val: ").replace("_", " "))
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=8)

    # Hide unused axes
    for ax in axes[total_plots:]:
        ax.set_visible(False)

    fig.suptitle("Training Curves — " + ", ".join(labels), fontsize=10)
    plt.tight_layout()

    if args.out:
        out_path = args.out
    elif len(runs) == 1:
        out_path = os.path.join(runs[0], "training_curves.png")
    else:
        # Save next to the common parent
        parent = os.path.commonpath(runs)
        prefix_str = "_".join(labels)[:60]
        out_path = os.path.join(parent, f"training_curves_{prefix_str}.png")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved: {out_path}")


if __name__ == "__main__":
    main()
