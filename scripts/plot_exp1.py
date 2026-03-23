"""
Exp 1 Scatter Plot: z_shuffle_gap vs action_mse_prior
──────────────────────────────────────────────────────
x축: z_shuffle_gap (z 품질 proxy)
y축: action_mse_prior (행동 예측 오차, 낮을수록 좋음)

각 모델이 점 하나로 표시됨.
논문 Figure 1 후보.

사용법:
  python scripts/plot_exp1.py --results_dir outputs/exp1/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


MODEL_LABELS = {
    "flat_flow":          ("FlatFlow (M1)",        "#999999", "o"),
    "det_latent":         ("DetLatent (M2)",        "#4e79a7", "s"),
    "stoch_vae":          ("StochVAE (M3)",         "#f28e2b", "^"),
    "stoch_flow_prior":   ("StochFlowPrior (M4)",   "#e15759", "D"),
    "vlm_sfp_last":       ("VLM-z last (ours)",     "#76b7b2", "P"),
    "vlm_sfp_pool":       ("VLM-z pool (ours)",     "#59a14f", "P"),
    "vlm_sfp_plan":       ("VLM-z plan (ours)",     "#b07aa1", "P"),
}


def load_metrics(results_dir: str) -> dict:
    data = {}
    for fname in os.listdir(results_dir):
        if not fname.endswith("_metrics.json"):
            continue
        key = fname.replace("_metrics.json", "")
        with open(os.path.join(results_dir, fname)) as f:
            raw = json.load(f)
        # temp_1.0 또는 직접 키로 저장된 경우 모두 처리
        if "temp_1.0" in raw:
            data[key] = raw["temp_1.0"]
        else:
            data[key] = raw
    return data


def plot(results_dir: str, save_path: str = None):
    data = load_metrics(results_dir)
    if not data:
        print(f"[plot_exp1] 결과 파일 없음: {results_dir}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for key, metrics in data.items():
        z_gap = metrics.get("z_shuffle_gap", None)
        mse   = metrics.get("action_mse_prior", None)
        if z_gap is None or mse is None:
            continue

        label, color, marker = MODEL_LABELS.get(
            key, (key, "#aaaaaa", "o")
        )
        ax.scatter(z_gap, mse, c=color, marker=marker,
                   s=120, zorder=5, label=label)
        ax.annotate(label, (z_gap, mse),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8, color=color)

    # 추세선 (선형)
    xs = [v.get("z_shuffle_gap", 0) for v in data.values()
          if "z_shuffle_gap" in v and "action_mse_prior" in v]
    ys = [v.get("action_mse_prior", 0) for v in data.values()
          if "z_shuffle_gap" in v and "action_mse_prior" in v]
    if len(xs) >= 3:
        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(xs) * 0.9, max(xs) * 1.1, 100)
        ax.plot(x_range, p(x_range), "--", color="#cccccc",
                alpha=0.7, linewidth=1, label="trend")
        r = np.corrcoef(xs, ys)[0, 1]
        ax.text(0.05, 0.92, f"r = {r:.3f}", transform=ax.transAxes,
                fontsize=10, color="#555555")

    ax.set_xlabel("z_shuffle_gap  (↑ z 품질 높음)", fontsize=12)
    ax.set_ylabel("action_mse_prior  (↓ 좋음)", fontsize=12)
    ax.set_title("Exp 1: z 품질과 행동 예측 오차의 관계\n"
                 "(Aha 1: z quality → performance)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(results_dir, "scatter_z_quality.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[plot_exp1] 저장: {save_path}")
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--save_path", default=None)
    args = p.parse_args()
    plot(args.results_dir, args.save_path)
