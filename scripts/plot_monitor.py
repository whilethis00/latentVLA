"""
학습 진행 모니터링 그래프 생성.
어떤 run이든 train_log.jsonl이 있으면 동작.

사용법:
  conda run -n vla python3 scripts/plot_monitor.py --run_dir outputs/runs/<name>
  conda run -n vla python3 scripts/plot_monitor.py --run_dir outputs/runs/<name> --out monitor.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── M2 (DetLatent, 이론적 상한) ──────────────────────────────────
M2 = {
    "action_mse_prior":     0.4776,
    "action_mse_posterior": 0.0017,
    "z_shuffle_gap":        0.7837,
    "prior_posterior_gap":  0.4759,
}

# ── 스타일 ────────────────────────────────────────────────────────
AX_COLOR   = "#f7f8fc"
GRID_COLOR = "#d0d4e0"
TEXT_COLOR = "#1a1d2e"
ACCENT     = ["#2563eb", "#e53935", "#2e7d32", "#f59e0b", "#7c3aed"]
M2_COLOR   = "#ff6b35"  # M2 목표선


def style_ax(ax, title):
    ax.set_facecolor(AX_COLOR)
    ax.set_title(title, color=TEXT_COLOR, fontsize=9, fontweight="bold", pad=6)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7.5)
    ax.spines[:].set_color(GRID_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.7)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)


def add_m2_line(ax, value, label=None, side="min"):
    """M2 목표선 추가. side='min'이면 값이 낮을수록 좋은 지표(mse), 'max'면 높을수록 좋은 지표."""
    ax.axhline(value, color=M2_COLOR, lw=1.2, linestyle="--", alpha=0.8,
               label=label or f"M2={value:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="outputs/runs/<name>")
    parser.add_argument("--out", default=None, help="출력 파일명 (default: monitor.png)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    log_path = run_dir / "train_log.jsonl"

    if not log_path.exists():
        print(f"[plot_monitor] 로그 없음: {log_path}", file=sys.stderr)
        sys.exit(1)

    records = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    if not records:
        print("[plot_monitor] 로그가 비어있음", file=sys.stderr)
        sys.exit(1)

    # ── 데이터 분리 ───────────────────────────────────────────────
    train_epochs = [r["epoch"] for r in records]
    total        = [r.get("train/total_loss", float("nan")) for r in records]
    action_flow  = [r.get("train/action_flow_loss", float("nan")) for r in records]
    prior_flow   = [r.get("train/prior_flow_loss", float("nan")) for r in records]
    semantic     = [r.get("train/semantic_future_loss", float("nan")) for r in records]
    infonce      = [r.get("train/infonce_loss") for r in records]
    has_infonce  = any(v is not None for v in infonce)

    val_records  = [r for r in records if "val/action_mse_prior" in r]
    if not val_records:
        print("[plot_monitor] val 기록 없음 — train loss만 그립니다.")

    v_ep   = [r["epoch"] for r in val_records]
    v_pri  = [r["val/action_mse_prior"]     for r in val_records]
    v_post = [r["val/action_mse_posterior"] for r in val_records]
    v_b1   = [r.get("val/best_of_1")        for r in val_records]
    v_b5   = [r.get("val/best_of_5")        for r in val_records]
    v_gap  = [r.get("val/z_shuffle_gap", 0) for r in val_records]
    v_pp   = [r.get("val/prior_posterior_gap", 0) for r in val_records]

    run_name = run_dir.name
    n_epochs_done = train_epochs[-1] if train_epochs else 0

    # ── 레이아웃 ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        top=0.87, bottom=0.07,
        left=0.06, right=0.97,
        hspace=0.52, wspace=0.36,
    )

    fig.text(0.50, 0.955, "LatentVLA — Training Monitor", ha="center",
             fontsize=17, fontweight="bold", color=TEXT_COLOR)
    fig.text(0.50, 0.918, run_name, ha="center", fontsize=10, color="#555e7a")
    fig.text(0.50, 0.897, f"Epoch {n_epochs_done}  ·  dashed line: M2 target (theoretical upper bound)",
             ha="center", fontsize=8.5, color=M2_COLOR)

    # ── Plot 1: Train Loss ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(train_epochs, total,      color=ACCENT[0], lw=2,   label="total")
    ax1.plot(train_epochs, action_flow,color=ACCENT[1], lw=1.5, linestyle="--", label="action_flow")
    ax1.plot(train_epochs, prior_flow, color=ACCENT[2], lw=1.5, linestyle="--", label="prior_flow")
    ax1.plot(train_epochs, semantic,   color=ACCENT[3], lw=1.2, linestyle=":",  label="semantic_future")
    if has_infonce:
        infonce_vals = [v if v is not None else float("nan") for v in infonce]
        ax1.plot(train_epochs, infonce_vals, color=ACCENT[4], lw=1.2, linestyle=":", label="infonce")
    ax1.set_xlabel("Epoch", color=TEXT_COLOR, fontsize=8)
    ax1.set_ylabel("Loss", color=TEXT_COLOR, fontsize=8)
    ax1.legend(fontsize=7.5, framealpha=0.8, facecolor="white", edgecolor=GRID_COLOR)
    style_ax(ax1, "Train Loss Curve")

    # ── Plot 2: 최신 Loss breakdown bar ───────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    bar_labels = ["action\nflow", "prior\nflow", "semantic"]
    bar_vals   = [action_flow[-1], prior_flow[-1], semantic[-1]]
    bar_colors = [ACCENT[1], ACCENT[2], ACCENT[3]]
    if has_infonce and infonce[-1] is not None:
        bar_labels.append("infonce")
        bar_vals.append(infonce[-1])
        bar_colors.append(ACCENT[4])
    bars = ax2.bar(bar_labels, bar_vals, color=bar_colors, width=0.5, alpha=0.85)
    for bar, v in zip(bars, bar_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=7, color=TEXT_COLOR)
    ax2.set_ylabel(f"Loss (Ep {n_epochs_done})", color=TEXT_COLOR, fontsize=8)
    style_ax(ax2, "Latest Loss Breakdown")

    if val_records:
        # ── Plot 3: Val MSE + M2 목표선 ───────────────────────────
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.plot(v_ep, v_pri,  color=ACCENT[0], lw=1.8, marker="o", ms=4, label="mse_prior")
        ax3.plot(v_ep, v_post, color=ACCENT[1], lw=1.8, marker="o", ms=4, label="mse_posterior")
        if any(v is not None for v in v_b1):
            ax3.plot(v_ep, v_b1, color=ACCENT[2], lw=1.5, marker="s", ms=4,
                     linestyle="--", label="best_of_1")
        if any(v is not None for v in v_b5):
            ax3.plot(v_ep, v_b5, color=ACCENT[3], lw=1.5, marker="s", ms=4,
                     linestyle="--", label="best_of_5")
        add_m2_line(ax3, M2["action_mse_prior"],     label=f"M2 prior={M2['action_mse_prior']:.4f}")
        add_m2_line(ax3, M2["action_mse_posterior"],  label=f"M2 post={M2['action_mse_posterior']:.4f}")
        ax3.set_xlabel("Epoch", color=TEXT_COLOR, fontsize=8)
        ax3.set_ylabel("MSE", color=TEXT_COLOR, fontsize=8)
        ax3.legend(fontsize=7, framealpha=0.8, facecolor="white", edgecolor=GRID_COLOR, ncol=2)
        style_ax(ax3, "Val MSE  (dashed: M2 target)")

        # ── Plot 4: z Quality + M2 목표선 ─────────────────────────
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(v_ep, v_gap, color=ACCENT[4], lw=1.8, marker="o", ms=5, label="z_shuffle_gap")
        ax4.plot(v_ep, v_pp,  color=ACCENT[0], lw=1.8, marker="^", ms=5, label="prior_posterior_gap")
        add_m2_line(ax4, M2["z_shuffle_gap"],     label=f"M2 shuf={M2['z_shuffle_gap']:.3f}")
        add_m2_line(ax4, M2["prior_posterior_gap"], label=f"M2 pp={M2['prior_posterior_gap']:.3f}")
        ax4.axhline(0, color="#aaa", lw=0.7, linestyle=":")
        ax4.set_xlabel("Epoch", color=TEXT_COLOR, fontsize=8)
        ax4.legend(fontsize=7, framealpha=0.8, facecolor="white", edgecolor=GRID_COLOR)
        style_ax(ax4, "z Quality  (dashed: M2 target)")

    # ── 저장 ──────────────────────────────────────────────────────
    out_name = args.out or "monitor.png"
    out_path = run_dir / out_name
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"saved → {out_path}")


if __name__ == "__main__":
    main()
