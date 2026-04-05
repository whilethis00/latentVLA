"""
generate_result.py — 학습 완료 후 result.png + result.md 자동 생성

Usage:
    python scripts/generate_result.py --run_dir outputs/runs/<run_name>

학습 끝날 때 trainer.py / trainer_vlm.py에서 자동 호출됨.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ── 스타일 상수 (sanity_check 기준 유지) ─────────────────────────────────────
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


def load_log(run_dir: Path):
    log_path = run_dir / "train_log.jsonl"
    records = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    return records


def get(records, key, default=None):
    vals = []
    for r in records:
        if key in r:
            vals.append(r[key])
        else:
            vals.append(default)
    return vals


def generate_png(run_dir: Path, records: list, model_type: str):
    epochs = [r["epoch"] for r in records]

    # Train metrics
    total       = get(records, "train/total_loss")
    action_flow = get(records, "train/action_flow_loss")
    prior_flow  = get(records, "train/prior_flow_loss")
    semantic    = get(records, "train/semantic_future_loss")

    has_prior    = any(v is not None for v in prior_flow)
    has_semantic = any(v is not None for v in semantic)

    # Val records
    val_records = [r for r in records if "val/action_mse_prior" in r]
    v_ep   = [r["epoch"] for r in val_records]
    v_pri  = [r.get("val/action_mse_prior") for r in val_records]
    v_post = [r.get("val/action_mse_posterior") for r in val_records]
    v_b1   = [r.get("val/best_of_1") for r in val_records]
    v_b5   = [r.get("val/best_of_5") for r in val_records]
    v_gap  = [r.get("val/z_shuffle_gap") for r in val_records]
    v_pp   = [r.get("val/prior_posterior_gap") for r in val_records]
    v_cos  = [r.get("val/future_cosine_sim") for r in val_records]

    has_val_z   = val_records and v_gap[0] is not None
    has_val_pp  = val_records and v_post[0] is not None

    # ── 레이아웃 ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11), facecolor="white")
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        top=0.84, bottom=0.07,
        left=0.06, right=0.97,
        hspace=0.55, wspace=0.38,
    )

    run_name = run_dir.name
    final_epoch = epochs[-1]
    fig.text(0.50, 0.955, f"LatentVLA — {run_name}",
             ha="center", fontsize=17, fontweight="bold", color=TEXT_COLOR)
    fig.text(0.50, 0.915,
             f"Model: {model_type}  ·  {final_epoch} epochs",
             ha="center", fontsize=10, color="#555e7a")
    fig.text(0.50, 0.890,
             datetime.now().strftime("%Y-%m-%d"),
             ha="center", fontsize=9, color="#888ea8")

    # ── Plot 1: Train Loss Curve ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(epochs, total,       color=ACCENT[0], lw=2,   label="total")
    ax1.plot(epochs, action_flow, color=ACCENT[1], lw=1.5, linestyle="--", label="action_flow")
    if has_prior:
        pf = [v for v in prior_flow if v is not None]
        ep_pf = [e for e, v in zip(epochs, prior_flow) if v is not None]
        ax1.plot(ep_pf, pf, color=ACCENT[2], lw=1.5, linestyle="--", label="prior_flow")
    if has_semantic:
        sm = [v for v in semantic if v is not None]
        ep_sm = [e for e, v in zip(epochs, semantic) if v is not None]
        ax1.plot(ep_sm, sm, color=ACCENT[3], lw=1.5, linestyle=":", label="semantic")
    ax1.set_xlabel("Epoch", color=TEXT_COLOR, fontsize=8)
    ax1.set_ylabel("Loss", color=TEXT_COLOR, fontsize=8)
    ax1.legend(fontsize=7.5, framealpha=0.8, labelcolor=TEXT_COLOR,
               facecolor="white", edgecolor=GRID_COLOR)
    style_ax(ax1, "Train Loss Curve")

    # ── Plot 2: Final Loss Breakdown bar ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    labels, vals, colors = [], [], []
    labels.append("action\nflow"); vals.append(action_flow[-1] or 0); colors.append(ACCENT[1])
    if has_prior and prior_flow[-1] is not None:
        labels.append("prior\nflow"); vals.append(prior_flow[-1]); colors.append(ACCENT[2])
    if has_semantic and semantic[-1] is not None:
        labels.append("semantic"); vals.append(semantic[-1]); colors.append(ACCENT[3])
    bars = ax2.bar(labels, vals, color=colors, width=0.5, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=7.5, color=TEXT_COLOR)
    ax2.set_ylabel(f"Loss (Ep {final_epoch})", color=TEXT_COLOR, fontsize=8)
    style_ax(ax2, "Final Loss Breakdown")

    # ── Plot 3: Val MSE ───────────────────────────────────────────────────────
    if val_records:
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.plot(v_ep, v_pri, color=ACCENT[0], lw=1.8, marker="o", ms=4, label="mse_prior")
        if has_val_pp:
            ax3.plot(v_ep, v_post, color=ACCENT[1], lw=1.8, marker="o", ms=4, label="mse_posterior")
        if v_b1[0] is not None:
            ax3.plot(v_ep, v_b1, color=ACCENT[2], lw=1.5, marker="s", ms=4,
                     linestyle="--", label="best_of_1")
        if v_b5[0] is not None:
            ax3.plot(v_ep, v_b5, color=ACCENT[3], lw=1.5, marker="s", ms=4,
                     linestyle="--", label="best_of_5")
        ax3.set_xlabel("Epoch", color=TEXT_COLOR, fontsize=8)
        ax3.set_ylabel("MSE", color=TEXT_COLOR, fontsize=8)
        ax3.legend(fontsize=7.5, framealpha=0.8, labelcolor=TEXT_COLOR,
                   facecolor="white", edgecolor=GRID_COLOR)
        style_ax(ax3, "Val MSE — Prior / Posterior / Best-of-K")

    # ── Plot 4: z Quality Metrics ─────────────────────────────────────────────
    if has_val_z:
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(v_ep, v_gap, color=ACCENT[4], lw=1.8, marker="o", ms=5, label="z_shuffle_gap")
        if v_pp[0] is not None:
            ax4.plot(v_ep, v_pp, color=ACCENT[0], lw=1.8, marker="^", ms=5,
                     label="prior_posterior_gap")
        ax4.axhline(0, color="#ff6b6b", lw=0.8, linestyle=":")
        ax4.set_xlabel("Epoch", color=TEXT_COLOR, fontsize=8)
        ax4.legend(fontsize=7, framealpha=0.8, labelcolor=TEXT_COLOR,
                   facecolor="white", edgecolor=GRID_COLOR)
        style_ax(ax4, "z Quality Metrics")

    # ── Plot 5: Final Val Summary Table ───────────────────────────────────────
    if val_records:
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis("off")
        final = val_records[-1]

        def fmt(key):
            v = final.get(key)
            return f"{v:.4f}" if v is not None else "—"

        def assess(key, v):
            if v is None: return "—"
            if key == "val/action_mse_prior":    return "lower is better"
            if key == "val/action_mse_posterior": return "lower is better"
            if key == "val/prior_posterior_gap":  return "✓ positive" if v > 0 else "△ negative"
            if key == "val/best_of_1":            return "lower is better"
            if key == "val/best_of_5":            return "lower is better"
            if key == "val/future_cosine_sim":    return "✓ close to 1" if v > 0.99 else "monitor"
            if key == "val/z_shuffle_gap":        return "✓ positive" if v > 0.01 else "△ small"
            return ""

        row_keys = [
            ("action_mse_prior",    "val/action_mse_prior"),
            ("action_mse_posterior","val/action_mse_posterior"),
            ("prior_posterior_gap", "val/prior_posterior_gap"),
            ("best_of_1",           "val/best_of_1"),
            ("best_of_5",           "val/best_of_5"),
            ("future_cosine_sim",   "val/future_cosine_sim"),
            ("z_shuffle_gap",       "val/z_shuffle_gap"),
        ]
        rows = []
        for label, key in row_keys:
            v = final.get(key)
            rows.append([label, fmt(key), assess(key, v)])

        table = ax5.table(
            cellText=rows,
            colLabels=["Metric", f"Value (Ep {final['epoch']})", "Assessment"],
            loc="center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.55)

        good_color   = "#d4edda"
        warn_color   = "#fff3cd"
        row_colors   = ["#ffffff", "#f4f6fb"]
        header_color = "#dde3f5"
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor(GRID_COLOR)
            cell.set_linewidth(0.6)
            if r == 0:
                cell.set_facecolor(header_color)
                cell.set_text_props(color=TEXT_COLOR, fontweight="bold")
            else:
                text = rows[r - 1][2]
                if text.startswith("✓"):
                    cell.set_facecolor(good_color if c == 2 else row_colors[(r - 1) % 2])
                elif text.startswith("△"):
                    cell.set_facecolor(warn_color if c == 2 else row_colors[(r - 1) % 2])
                else:
                    cell.set_facecolor(row_colors[(r - 1) % 2])
                cell.set_text_props(color=TEXT_COLOR)

        ax5.set_title(f"Final Validation Metrics Summary  (Epoch {final['epoch']})",
                      color=TEXT_COLOR, fontsize=9, fontweight="bold", pad=8)

    out = run_dir / "result.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[generate_result] result.png → {out}")
    return out


def generate_md(run_dir: Path, records: list, model_type: str):
    epochs = [r["epoch"] for r in records]
    val_records = [r for r in records if "val/action_mse_prior" in r]
    final_epoch = epochs[-1]
    today = datetime.now().strftime("%Y-%m-%d")

    # ── 손실 곡선 표 ─────────────────────────────────────────────────────────
    sample_epochs = sorted(set(
        [1] + list(range(10, final_epoch + 1, 10))
    ))
    ep_map = {r["epoch"]: r for r in records}

    has_prior    = any("train/prior_flow_loss"     in r for r in records)
    has_semantic = any("train/semantic_future_loss" in r for r in records)

    loss_header = "| Epoch | total_loss | action_flow |"
    loss_sep    = "|-------|-----------|-------------|"
    if has_prior:
        loss_header += " prior_flow |"
        loss_sep    += "------------|"
    if has_semantic:
        loss_header += " semantic |"
        loss_sep    += "----------|"

    loss_rows = ""
    for ep in sample_epochs:
        r = ep_map.get(ep)
        if r is None:
            continue
        row = f"| {ep:<5} | {r.get('train/total_loss', 0):.4f}    | {r.get('train/action_flow_loss', 0):.4f}      |"
        if has_prior:
            row += f" {r.get('train/prior_flow_loss', 0):.4f}      |"
        if has_semantic:
            row += f" {r.get('train/semantic_future_loss', 0):.6f} |"
        loss_rows += row + "\n"

    # ── Val 지표 표 ──────────────────────────────────────────────────────────
    has_val_full = val_records and val_records[0].get("val/action_mse_posterior") is not None
    has_val_z    = val_records and val_records[0].get("val/z_shuffle_gap") is not None

    val_header = "| Epoch | mse_prior |"
    val_sep    = "|-------|-----------|"
    if has_val_full:
        val_header += " mse_posterior | pp_gap | best_of_1 | best_of_5 | cosine_sim | z_shuffle_gap |"
        val_sep    += "---------------|--------|-----------|-----------|------------|---------------|"

    val_rows = ""
    for vr in val_records:
        ep = vr["epoch"]
        row = f"| {ep:<5} | {vr.get('val/action_mse_prior', 0):.4f}    |"
        if has_val_full:
            row += (
                f" {vr.get('val/action_mse_posterior', 0):.4f}         |"
                f" {vr.get('val/prior_posterior_gap', 0):.4f}  |"
                f" {vr.get('val/best_of_1', 0):.4f}    |"
                f" {vr.get('val/best_of_5', 0):.4f}    |"
                f" {vr.get('val/future_cosine_sim', 0):.4f}      |"
                f" {vr.get('val/z_shuffle_gap', 0):.4f}        |"
            )
        val_rows += row + "\n"

    final_v = val_records[-1] if val_records else {}

    def fv(key):
        v = final_v.get(key)
        return f"{v:.4f}" if v is not None else "—"

    md = f"""# {run_dir.name} 실험 결과

- **날짜**: {today}
- **모델**: {model_type}
- **에포크**: {final_epoch}
- **현황**: {final_epoch} epoch 완료

---

## 학습 손실 곡선 (주요 epoch)

{loss_header}
{loss_sep}
{loss_rows.strip()}

---

## 검증 지표 (Val)

{val_header}
{val_sep}
{val_rows.strip() if val_rows else "_(없음)_"}

---

## 최종 Val 지표

| 지표 | 값 |
|------|---|
| action_mse_prior | {fv('val/action_mse_prior')} |
| action_mse_posterior | {fv('val/action_mse_posterior')} |
| prior_posterior_gap | {fv('val/prior_posterior_gap')} |
| best_of_1 | {fv('val/best_of_1')} |
| best_of_5 | {fv('val/best_of_5')} |
| future_cosine_sim | {fv('val/future_cosine_sim')} |
| z_shuffle_gap | {fv('val/z_shuffle_gap')} |

---

## 저장 파일

| 파일 | 내용 |
|------|------|
| `ckpt_10.pt` ~ `ckpt_{final_epoch}.pt` | 10 epoch 단위 체크포인트 |
| `ckpt_final.pt` | 최종 체크포인트 |
| `train_log.jsonl` | Epoch별 전체 loss / val 지표 |
| `result.png` | 학습 곡선 및 val 지표 시각화 |
"""

    out = run_dir / "result.md"
    out.write_text(md)
    print(f"[generate_result] result.md  → {out}")
    return out


def generate(run_dir_str: str, model_type: str = None):
    run_dir = Path(run_dir_str)
    if not (run_dir / "train_log.jsonl").exists():
        print(f"[generate_result] train_log.jsonl not found in {run_dir}")
        return

    records = load_log(run_dir)
    if not records:
        print(f"[generate_result] empty log in {run_dir}")
        return

    # model_type 자동 추론 (checkpoint에서)
    if model_type is None:
        ckpt_path = run_dir / "ckpt_final.pt"
        if ckpt_path.exists():
            try:
                import torch
                ck = torch.load(ckpt_path, map_location="cpu")
                model_type = ck.get("cfg", {}).get("model", {}).get("type", "unknown")
                if model_type == "unknown" and "z_form" in ck:
                    model_type = f"vlm_sfp_{ck['z_form']}"
            except Exception:
                pass
        if model_type is None:
            model_type = run_dir.name

    generate_png(run_dir, records, model_type)
    generate_md(run_dir, records, model_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--model_type", default=None)
    args = parser.parse_args()
    generate(args.run_dir, args.model_type)
