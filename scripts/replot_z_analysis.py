"""
replot_z_analysis.py — 저장된 데이터로 플롯만 재생성 (모델 재로드 불필요)

partial_shuffle: z_analysis_summary.json에서 재생성
t-SNE:          z_vectors.npz에서 재생성

Usage:
    # partial_shuffle.png 재생성
    conda run -n vla python3 scripts/replot_z_analysis.py \
        --json outputs/runs/vlm_sfp_plan_100ep_20260405/z_analysis/z_analysis_summary.json

    # tsne.png 재생성
    conda run -n vla python3 scripts/replot_z_analysis.py \
        --json outputs/runs/vlm_sfp_plan_100ep_20260405/z_analysis/z_analysis_summary.json \
        --npz  outputs/runs/vlm_sfp_plan_100ep_20260405/z_analysis/z_vectors.npz

    # 둘 다 재생성
    conda run -n vla python3 scripts/replot_z_analysis.py \
        --json outputs/runs/vlm_sfp_plan_100ep_20260405/z_analysis/z_analysis_summary.json \
        --npz  outputs/runs/vlm_sfp_plan_100ep_20260405/z_analysis/z_vectors.npz \
        --all
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.dpi":       150,
})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder


# ── partial_shuffle.png ────────────────────────────────────────────────────────

def plot_partial_shuffle(results: dict, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "z Partial Shuffle Analysis  —  M5 VLM SFP Plan\n"
        "z_task = first half dims,  z_motion = second half dims",
        fontsize=13, fontweight="bold", y=1.01,
    )

    COLORS = ["#4CAF50", "#E53935", "#FB8C00", "#1E88E5"]
    CATS   = ["Normal\n(baseline)", "Full\nShuffle", "Task Half\nShuffle", "Motion Half\nShuffle"]

    for ax, (z_key, title) in zip(axes, [("z_post", "Posterior z  (mu_q)"),
                                          ("z_prior", "Prior z")]):
        r = results[z_key]
        mses = [r["mse_normal"], r["mse_full"], r["mse_task"], r["mse_motion"]]
        gaps = [0, r["gap_full"], r["gap_task"], r["gap_motion"]]

        bars = ax.bar(CATS, mses, color=COLORS, alpha=0.85,
                      edgecolor="black", linewidth=0.6, width=0.52)

        # MSE 수치 레이블
        ymax = max(mses)
        for bar, val, gap in zip(bars, mses, gaps):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ymax * 0.015,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold")
            if gap != 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + ymax * 0.075,
                        f"(+{gap:.4f})", ha="center", va="bottom",
                        fontsize=8, color="#555555")

        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel("Action MSE", fontsize=11)
        ax.set_ylim(0, ymax * 1.42)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(r["mse_normal"], color="#4CAF50", linestyle="--",
                   alpha=0.5, linewidth=1.2)

        # 정보 박스
        tr = r["task_ratio"]
        mr = r["motion_ratio"]
        if tr > mr:
            verdict = f">> z_task leads  (InfoNCE likely effective)"
        elif mr > tr:
            verdict = f">> z_motion leads  (consider KD approach)"
        else:
            verdict = f">> task / motion contribution similar"

        info = (
            f"z_dim={r['z_dim']}  (task={r['D_task']}, motion={r['D_motion']})\n"
            f"gap_full   = {r['gap_full']:+.4f}  (100%)\n"
            f"gap_task   = {r['gap_task']:+.4f}  ({tr*100:.1f}%)\n"
            f"gap_motion = {r['gap_motion']:+.4f}  ({mr*100:.1f}%)\n"
            f"{verdict}"
        )
        ax.text(0.97, 0.97, info,
                transform=ax.transAxes, va="top", ha="right",
                fontsize=8.5, family="monospace",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFFDE7",
                          edgecolor="#BDBDBD", alpha=0.95))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


# ── t-SNE 재생성 ──────────────────────────────────────────────────────────────

def plot_tsne(npz_path: str, out_path: str, max_samples: int = 2000):
    data = np.load(npz_path, allow_pickle=True)
    z_post  = data["z_post"]   # (N, z_dim)
    z_prior = data["z_prior"]  # (N, z_dim)
    tasks   = data["tasks"].tolist()

    le = LabelEncoder()
    task_ids = le.fit_transform(tasks)
    n_tasks = len(le.classes_)
    print(f"[t-SNE] {len(tasks)} samples, {n_tasks} tasks")

    N = len(tasks)
    idx = np.arange(N)
    if N > max_samples:
        np.random.seed(42)
        idx = np.random.choice(N, max_samples, replace=False)
        print(f"[t-SNE] {N} -> {max_samples} samples")

    # task별 고정 색상 (ListedColormap으로 확실하게 매핑)
    base = matplotlib.colormaps["tab20"]
    palette = [base(i % 20) for i in range(n_tasks)]
    task_cmap = matplotlib.colors.ListedColormap(palette)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "t-SNE of z  —  M5 VLM SFP Plan\n"
        "Color = task.  Tight clusters per task = z encodes task information.",
        fontsize=13, fontweight="bold",
    )

    for ax, (z_np_full, label) in zip(
        axes,
        [(z_post, "Posterior z (mu_q)"), (z_prior, "Prior z")],
    ):
        z_np = z_np_full[idx]
        t_ids = task_ids[idx].astype(float)

        perp = min(30, len(z_np) // 5)
        print(f"[t-SNE] {label}  (n={len(z_np)}, perplexity={perp})...")
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
        z_2d = tsne.fit_transform(z_np)

        ax.scatter(
            z_2d[:, 0], z_2d[:, 1],
            c=t_ids, cmap=task_cmap,
            vmin=0, vmax=n_tasks - 1,
            alpha=0.65, s=16,
            linewidths=0.3, edgecolors="white",
        )
        ax.set_title(label, fontsize=12, fontweight="bold", pad=10)

        # 테두리만 남기고 tick 제거
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("#AAAAAA")
        ax.set_facecolor("#F8F8F8")

        # 범례 (최대 20개)
        shown = min(n_tasks, 20)
        handles = [
            mpatches.Patch(color=palette[i], label=le.classes_[i][:32])
            for i in range(shown)
        ]
        suffix = f" (+{n_tasks - shown} more)" if n_tasks > shown else ""
        ax.legend(
            handles=handles, fontsize=6.5, loc="lower right",
            ncol=2, framealpha=0.85, edgecolor="#CCCCCC",
            title=f"Tasks ({n_tasks} total){suffix}", title_fontsize=7,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[saved] {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True,
                        help="z_analysis_summary.json 경로")
    parser.add_argument("--npz", default=None,
                        help="z_vectors.npz 경로 (t-SNE 재생성 시 필요)")
    parser.add_argument("--all", action="store_true",
                        help="partial_shuffle + tsne 둘 다 재생성")
    parser.add_argument("--tsne_max", type=int, default=2000,
                        help="t-SNE 최대 샘플 수 (기본: 2000)")
    parser.add_argument("--out_dir", default=None,
                        help="출력 디렉토리 (기본: json과 같은 폴더)")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.json))
    os.makedirs(out_dir, exist_ok=True)

    with open(args.json) as f:
        results = json.load(f)

    plot_partial_shuffle(results, os.path.join(out_dir, "partial_shuffle.png"))

    if args.npz or args.all:
        npz = args.npz or os.path.join(out_dir, "z_vectors.npz")
        if not os.path.exists(npz):
            print(f"[오류] z_vectors.npz 없음: {npz}")
            print("  eval_z_analysis.py를 다시 실행하면 자동 저장됩니다.")
        else:
            plot_tsne(npz, os.path.join(out_dir, "tsne.png"), args.tsne_max)

    print("완료.")


if __name__ == "__main__":
    main()
