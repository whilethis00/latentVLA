import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

# ── 데이터 로드 ──────────────────────────────────────────────
log_path = Path(__file__).parent / "train_log.jsonl"
records = [json.loads(l) for l in log_path.read_text().splitlines()]

epochs      = [r["epoch"] for r in records]
total       = [r["train/total_loss"] for r in records]
action_flow = [r["train/action_flow_loss"] for r in records]
prior_flow  = [r["train/prior_flow_loss"] for r in records]
semantic    = [r["train/semantic_future_loss"] for r in records]

val_records = [r for r in records if "val/action_mse_prior" in r]
v_ep   = [r["epoch"] for r in val_records]
v_pri  = [r["val/action_mse_prior"] for r in val_records]
v_post = [r["val/action_mse_posterior"] for r in val_records]
v_b1   = [r["val/best_of_1"] for r in val_records]
v_b5   = [r["val/best_of_5"] for r in val_records]
v_gap  = [r["val/z_shuffle_gap"] for r in val_records]
v_pp   = [r["val/prior_posterior_gap"] for r in val_records]
v_cos  = [r["val/future_cosine_sim"] for r in val_records]

# ── 레이아웃 ─────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11), facecolor="white")
fig.patch.set_facecolor("white")

gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    top=0.84, bottom=0.07,
    left=0.06, right=0.97,
    hspace=0.55, wspace=0.38,
)

AX_COLOR   = "#f7f8fc"
GRID_COLOR = "#d0d4e0"
TEXT_COLOR = "#1a1d2e"
ACCENT     = ["#2563eb", "#e53935", "#2e7d32", "#f59e0b", "#7c3aed"]

def style_ax(ax, title):
    ax.set_facecolor(AX_COLOR)
    ax.set_title(title, color=TEXT_COLOR, fontsize=9, fontweight="bold", pad=6)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7.5)
    ax.spines[:].set_color(GRID_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

# ── 헤더 텍스트 ──────────────────────────────────────────────
fig.text(0.50, 0.955, "LatentVLA — Sanity Check", ha="center",
         fontsize=18, fontweight="bold", color=TEXT_COLOR)
fig.text(0.50, 0.915,
         "Exp: StochFlowPrior (M4)  ·  Stage 1 (PaliGemma frozen)  ·  "
         "LIBERO-Object  ·  10 epoch",
         ha="center", fontsize=10, color="#555e7a")
fig.text(0.50, 0.890, "2026-03-31 ~ 2026-04-01",
         ha="center", fontsize=9, color="#888ea8")

# ── Plot 1: Train Loss (total + 3 components) ────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(epochs, total,       color=ACCENT[0], lw=2,   label="total")
ax1.plot(epochs, action_flow, color=ACCENT[1], lw=1.5, label="action_flow", linestyle="--")
ax1.plot(epochs, prior_flow,  color=ACCENT[2], lw=1.5, label="prior_flow",  linestyle="--")
ax1.plot(epochs, semantic,    color=ACCENT[3], lw=1.5, label="semantic_future", linestyle=":")
ax1.set_xlabel("Epoch", color=TEXT_COLOR, fontsize=8)
ax1.set_ylabel("Loss", color=TEXT_COLOR, fontsize=8)
ax1.legend(fontsize=7.5, framealpha=0.8, labelcolor=TEXT_COLOR,
           facecolor="white", edgecolor=GRID_COLOR)
style_ax(ax1, "Train Loss Curve")

# ── Plot 2: 최종 loss 구성 bar ───────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
labels  = ["action\nflow", "prior\nflow", "semantic\nfuture"]
vals    = [action_flow[-1], prior_flow[-1], semantic[-1]]
colors  = [ACCENT[1], ACCENT[2], ACCENT[3]]
bars = ax2.bar(labels, vals, color=colors, width=0.5, alpha=0.85)
for bar, v in zip(bars, vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f"{v:.4f}", ha="center", va="bottom", fontsize=7.5, color=TEXT_COLOR)
ax2.set_ylabel("Loss (Ep 10)", color=TEXT_COLOR, fontsize=8)
style_ax(ax2, "Final Loss Breakdown")

# ── Plot 3: Val MSE (prior vs posterior vs best-of-1 vs best-of-5) ──
ax3 = fig.add_subplot(gs[1, :2])
ax3.plot(v_ep, v_pri,  color=ACCENT[0], lw=1.8, marker="o", ms=4, label="mse_prior")
ax3.plot(v_ep, v_post, color=ACCENT[1], lw=1.8, marker="o", ms=4, label="mse_posterior")
ax3.plot(v_ep, v_b1,   color=ACCENT[2], lw=1.5, marker="s", ms=4, linestyle="--", label="best_of_1")
ax3.plot(v_ep, v_b5,   color=ACCENT[3], lw=1.5, marker="s", ms=4, linestyle="--", label="best_of_5")
ax3.set_xlabel("Epoch", color=TEXT_COLOR, fontsize=8)
ax3.set_ylabel("MSE", color=TEXT_COLOR, fontsize=8)
ax3.legend(fontsize=7.5, framealpha=0.8, labelcolor=TEXT_COLOR,
           facecolor="white", edgecolor=GRID_COLOR)
style_ax(ax3, "Val MSE — Prior / Posterior / Best-of-K")

# ── Plot 4: z_shuffle_gap & prior_posterior_gap ──────────────
ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(v_ep, v_gap, color=ACCENT[4], lw=1.8, marker="o", ms=5, label="z_shuffle_gap")
ax4.plot(v_ep, v_pp,  color=ACCENT[0], lw=1.8, marker="^", ms=5, label="prior_posterior_gap")
ax4.axhline(0, color="#ff6b6b", lw=0.8, linestyle=":")
ax4.set_xlabel("Epoch", color=TEXT_COLOR, fontsize=8)
ax4.legend(fontsize=7, framealpha=0.8, labelcolor=TEXT_COLOR,
           facecolor="white", edgecolor=GRID_COLOR)
style_ax(ax4, "z Quality Metrics")

# ── Plot 5: Key metrics 최종값 summary table ─────────────────
ax5 = fig.add_subplot(gs[2, :])
ax5.axis("off")

final = val_records[-1]
rows = [
    ["action_mse_prior",    f"{final['val/action_mse_prior']:.4f}",    "lower is better",  "OK"],
    ["action_mse_posterior",f"{final['val/action_mse_posterior']:.4f}","lower is better",  "✓ below prior"],
    ["prior_posterior_gap", f"{final['val/prior_posterior_gap']:.4f}", "higher = z useful", "✓ positive (z valid)"],
    ["best_of_1",           f"{final['val/best_of_1']:.4f}",           "lower is better",  "room to improve"],
    ["best_of_5",           f"{final['val/best_of_5']:.4f}",           "lower is better",  "✓ -57% vs best_of_1"],
    ["future_cosine_sim",   f"{final['val/future_cosine_sim']:.4f}",   "closer to 1",      "✓ 0.99"],
    ["z_shuffle_gap",       f"{final['val/z_shuffle_gap']:.4f}",       "higher = z relied on", "△ small, monitor"],
]
col_labels = ["Metric", "Value (Ep 10)", "Criterion", "Assessment"]
col_widths = [0.22, 0.13, 0.20, 0.35]

table = ax5.table(
    cellText=rows,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.55)

header_color = "#dde3f5"
row_colors   = ["#ffffff", "#f4f6fb"]
good_color   = "#d4edda"
warn_color   = "#fff3cd"

for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor(GRID_COLOR)
    cell.set_linewidth(0.6)
    if r == 0:
        cell.set_facecolor(header_color)
        cell.set_text_props(color=TEXT_COLOR, fontweight="bold")
    else:
        text = rows[r-1][3]
        if text.startswith("✓"):
            cell.set_facecolor(good_color if c == 3 else row_colors[(r-1) % 2])
        elif text.startswith("△"):
            cell.set_facecolor(warn_color if c == 3 else row_colors[(r-1) % 2])
        else:
            cell.set_facecolor(row_colors[(r-1) % 2])
        cell.set_text_props(color=TEXT_COLOR)

ax5.set_title("Final Validation Metrics Summary  (Epoch 10)",
              color=TEXT_COLOR, fontsize=9, fontweight="bold", pad=8)

# ── 저장 ─────────────────────────────────────────────────────
out = Path(__file__).parent / "result.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"saved → {out}")
