"""
eval_z_analysis.py — M5 VLM z 품질 진단 스크립트

z 벡터를 두 파트로 분리하여 어느 쪽이 작업(task) 정보를 담는지 분석.
재학습 없이 기존 체크포인트만으로 실행 가능.

분석:
  1. Partial Shuffle  : z를 z_task(앞 절반), z_motion(뒤 절반)으로 분리
                        각각 셔플 후 action MSE 변화 측정
                        → "VLM z가 task/motion 중 무엇에 더 집중하는가?" 진단
  2. t-SNE           : z 분포를 2D로 시각화, task별 색상
                        → "z가 task를 구분하는 클러스터를 형성하는가?" 확인

Usage:
    conda run -n vla python3 scripts/eval_z_analysis.py \\
        --checkpoint outputs/runs/vlm_sfp_plan_100ep_20260405/ckpt_final.pt

    # 빠른 smoke 테스트 (배치 20개만)
    conda run -n vla python3 scripts/eval_z_analysis.py \\
        --checkpoint outputs/runs/vlm_sfp_plan_100ep_20260405/ckpt_final.pt \\
        --max_batches 20

출력:
    <ckpt_dir>/z_analysis/partial_shuffle.png
    <ckpt_dir>/z_analysis/tsne.png
    <ckpt_dir>/z_analysis/z_analysis_report.md
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from collections import Counter

import torch
import torch.nn.functional as F
import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

from training.builder import build_vlm_datasets, build_dataloaders_vlm, build_vlm_model
from models.flow_utils import euler_integrate


# ── 모델 로드 ──────────────────────────────────────────────────────────────────

def load_model_and_data(ckpt_path, device, data_path_override=None):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]

    if data_path_override:
        cfg.setdefault("data", {})["dataset_path"] = data_path_override

    if "data" not in cfg:
        raise ValueError(
            "체크포인트 cfg에 'data' 키가 없습니다.\n"
            "  --data_path /path/to/libero_object/  를 지정하세요."
        )

    train_ds, val_ds = build_vlm_datasets(cfg)
    _, val_loader, _ = build_dataloaders_vlm(train_ds, val_ds, cfg)

    model = build_vlm_model(cfg, action_dim=train_ds.action_dim,
                            proprio_dim=train_ds.proprio_dim)

    saved = ckpt["model"]
    if saved.get("system2_lora"):
        model.system2.enable_lora()
    model.load_state_dict_from_save(saved)
    model = model.to(device).eval()

    return model, val_loader, cfg


# ── z 벡터 수집 ────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_z_vectors(model, val_loader, device, max_batches=None):
    """
    val set 전체에서 z 관련 벡터를 수집.

    반환:
        z_post  : posterior mean (mu_q)  — (N, z_dim)
        z_prior : prior flow 샘플        — (N, z_dim)
        f_tilde : VLM context            — (N, context_dim)
        actions : ground truth actions   — (N, H, action_dim)
        tasks   : task 언어 설명 목록    — List[str] (N,)
    """
    z_posts, z_priors, f_tildes, actions_all, tasks_all = [], [], [], [], []

    for i, batch in enumerate(tqdm(val_loader, desc="z 벡터 수집")):
        if max_batches is not None and i >= max_batches:
            break

        actions = batch["actions"].to(device)
        proprio = batch["proprio"].to(device)
        B = actions.shape[0]

        # System2: VLM → f̃
        raw_image = batch.get("raw_image", batch["image"])
        pv, ids, mask = model.system2.prepare_inputs(raw_image, batch["language"], device)
        f_tilde = model.system2(pv, ids, mask, proprio)  # (B, context_dim)

        # Posterior z (mu_q: deterministic, 분석에 안정적)
        future_feat = None
        if model.system1.use_future and "future_image" in batch:
            future_feat = model._siglip.encode_image_only(
                batch["future_image"].to(device)
            )
        parts = [f_tilde, actions.reshape(B, -1)]
        if future_feat is not None:
            parts.append(future_feat)
        q_in = torch.cat(parts, dim=-1)
        mu_q, _ = model.system1.posterior_enc(q_in)
        z_post = mu_q  # (B, z_dim)

        # Prior z (prior flow)
        z_prior = euler_integrate(
            model.system1.prior_flow, f_tilde,
            model.system1.z_dim, model.system1.flow_steps,
        )  # (B, z_dim)

        # task 레이블: 파일명에서 추출 (language가 모두 같을 경우 대비)
        file_tasks = [
            os.path.basename(fp).replace("_demo.hdf5", "").replace("_", " ")
            for fp in batch["file"]
        ]
        lang_tasks = batch["language"]
        # language가 모두 같으면 파일명 기반 레이블 사용
        tasks_batch = file_tasks if len(set(lang_tasks)) <= 1 else lang_tasks

        z_posts.append(z_post.cpu())
        z_priors.append(z_prior.cpu())
        f_tildes.append(f_tilde.cpu())
        actions_all.append(actions.cpu())
        tasks_all.extend(tasks_batch)

    return {
        "z_post":  torch.cat(z_posts,   dim=0),   # (N, z_dim)
        "z_prior": torch.cat(z_priors,  dim=0),   # (N, z_dim)
        "f_tilde": torch.cat(f_tildes,  dim=0),   # (N, context_dim)
        "actions": torch.cat(actions_all, dim=0), # (N, H, action_dim)
        "tasks":   tasks_all,                      # List[str]
    }


# ── action MSE 계산 (custom z 주입) ───────────────────────────────────────────

@torch.no_grad()
def compute_mse_with_z(model, f_tilde_all, z_all, actions_all, device, batch_size=64):
    """
    주어진 z로 action flow를 돌려 action MSE를 계산.
    f_tilde, z, actions는 CPU 텐서로 전달.
    """
    N = actions_all.shape[0]
    total_mse = 0.0
    x_dim = model.system1.x_dim
    steps = model.system1.flow_steps

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        f = f_tilde_all[start:end].to(device)
        z = z_all[start:end].to(device)
        a_gt = actions_all[start:end].to(device)

        action_cond = torch.cat([f, z], dim=-1)
        a_pred = euler_integrate(
            model.system1.action_flow, action_cond, x_dim, steps
        ).reshape(end - start, model.system1.action_horizon, model.system1.action_dim)

        total_mse += F.mse_loss(a_pred, a_gt).item() * (end - start)

    return total_mse / N


# ── Partial Shuffle 분석 ───────────────────────────────────────────────────────

def run_partial_shuffle(model, data, device, n_repeat=3):
    """
    z를 앞/뒤 절반으로 분리해 각각 셔플한 뒤 action MSE 변화 측정.

    n_repeat: 셔플 랜덤성 완화를 위해 여러 번 반복 후 평균

    반환:
        dict keyed by "z_post", "z_prior"
        각각: mse_normal, mse_full, mse_task, mse_motion 및 gap/ratio
    """
    z_dim = data["z_post"].shape[1]
    D_task = z_dim // 2  # 앞 절반 → z_task
    N = data["actions"].shape[0]

    results = {}

    for z_key, label in [("z_post", "Posterior z"), ("z_prior", "Prior z")]:
        z_orig = data[z_key].clone()  # (N, z_dim), CPU

        mse_normal = compute_mse_with_z(
            model, data["f_tilde"], z_orig, data["actions"], device
        )

        mse_full_list, mse_task_list, mse_motion_list = [], [], []

        for _ in range(n_repeat):
            perm = torch.randperm(N)

            # 전체 셔플
            z_full = z_orig[perm]
            mse_full_list.append(compute_mse_with_z(
                model, data["f_tilde"], z_full, data["actions"], device
            ))

            # task 파트만 셔플 (앞 절반)
            z_task_shuf = z_orig.clone()
            z_task_shuf[:, :D_task] = z_orig[perm, :D_task]
            mse_task_list.append(compute_mse_with_z(
                model, data["f_tilde"], z_task_shuf, data["actions"], device
            ))

            # motion 파트만 셔플 (뒤 절반)
            z_motion_shuf = z_orig.clone()
            z_motion_shuf[:, D_task:] = z_orig[perm, D_task:]
            mse_motion_list.append(compute_mse_with_z(
                model, data["f_tilde"], z_motion_shuf, data["actions"], device
            ))

        mse_full   = float(np.mean(mse_full_list))
        mse_task   = float(np.mean(mse_task_list))
        mse_motion = float(np.mean(mse_motion_list))

        gap_full   = mse_full   - mse_normal
        gap_task   = mse_task   - mse_normal
        gap_motion = mse_motion - mse_normal
        denom = gap_full + 1e-8

        results[z_key] = {
            "label":       label,
            "mse_normal":  mse_normal,
            "mse_full":    mse_full,
            "mse_task":    mse_task,
            "mse_motion":  mse_motion,
            "gap_full":    gap_full,
            "gap_task":    gap_task,
            "gap_motion":  gap_motion,
            "task_ratio":  gap_task   / denom,
            "motion_ratio":gap_motion / denom,
            "D_task":      D_task,
            "D_motion":    z_dim - D_task,
            "z_dim":       z_dim,
        }

    return results


# ── Partial Shuffle 플롯 ───────────────────────────────────────────────────────

def plot_partial_shuffle(results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "z Partial Shuffle Analysis — M5 VLM SFP Plan\n"
        "(z_task = 앞 절반 dims, z_motion = 뒤 절반 dims)",
        fontsize=13, fontweight="bold",
    )

    colors = {
        "Normal":       "#4CAF50",
        "Full Shuffle": "#F44336",
        "Task Shuffle": "#FF9800",
        "Motion Shuffle": "#2196F3",
    }

    for ax, z_key in zip(axes, ["z_post", "z_prior"]):
        r = results[z_key]
        cats = ["Normal\n(baseline)", "Full\nShuffle", "Task Half\nShuffle", "Motion Half\nShuffle"]
        mses = [r["mse_normal"], r["mse_full"], r["mse_task"], r["mse_motion"]]
        cols = list(colors.values())

        bars = ax.bar(cats, mses, color=cols, alpha=0.82, edgecolor="black", linewidth=0.6, width=0.55)
        ax.set_title(r["label"], fontsize=12, fontweight="bold")
        ax.set_ylabel("Action MSE")
        ax.set_ylim(0, max(mses) * 1.38)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, val in zip(bars, mses):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(mses) * 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9,
            )

        # gap 요약 박스
        info_lines = [
            f"z_dim={r['z_dim']}  (task={r['D_task']}, motion={r['D_motion']})",
            f"gap_full   = {r['gap_full']:+.4f}",
            f"gap_task   = {r['gap_task']:+.4f}  ({r['task_ratio']*100:.1f}% of full)",
            f"gap_motion = {r['gap_motion']:+.4f}  ({r['motion_ratio']*100:.1f}% of full)",
        ]
        verdict = ""
        if r["task_ratio"] > 0.5:
            verdict = ">> z_task leads  (InfoNCE likely effective)"
        elif r["motion_ratio"] > 0.5:
            verdict = ">> z_motion leads  (consider KD approach)"
        else:
            verdict = ">> task / motion contribution similar"
        info_lines.append(verdict)

        ax.text(
            0.97, 0.97, "\n".join(info_lines),
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.9),
        )

        ax.axhline(r["mse_normal"], color="green", linestyle="--", alpha=0.45, linewidth=1.1)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "partial_shuffle.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {out_path}")


# ── t-SNE 분석 ─────────────────────────────────────────────────────────────────

def run_tsne(data, out_dir, max_samples=2000):
    """
    z_post, z_prior의 t-SNE 시각화.
    task(language)별 색상으로 클러스터링 확인.
    """
    tasks = data["tasks"]
    le = LabelEncoder()
    task_ids = le.fit_transform(tasks)
    n_tasks = len(le.classes_)

    # 샘플 수 제한 (t-SNE 속도)
    N = len(tasks)
    idx = np.arange(N)
    if N > max_samples:
        np.random.seed(42)
        idx = np.random.choice(N, max_samples, replace=False)
        print(f"[t-SNE] {N}개 → {max_samples}개 랜덤 샘플링")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "t-SNE of z  —  M5 VLM SFP Plan\n"
        "Color = task.  Tight clusters per task = z encodes task information.",
        fontsize=13, fontweight="bold",
    )

    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(min(n_tasks, 20))

    for ax, (z_key, label) in zip(axes, [("z_post", "Posterior z (mu_q)"), ("z_prior", "Prior z")]):
        if isinstance(data[z_key], torch.Tensor):
            z_np = data[z_key][idx].numpy()
        else:
            z_np = data[z_key][idx]
        t_ids = task_ids[idx]

        perp = min(30, len(z_np) // 5)
        print(f"[t-SNE] {label}  (n={len(z_np)}, perplexity={perp})...")
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
        z_2d = tsne.fit_transform(z_np)

        ax.scatter(
            z_2d[:, 0], z_2d[:, 1],
            c=t_ids % 20, cmap=cmap,
            alpha=0.55, s=12, linewidths=0,
        )
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.axis("off")

        shown = min(n_tasks, 20)
        handles = [
            mpatches.Patch(color=cmap(i % 20), label=le.classes_[i][:30])
            for i in range(shown)
        ]
        ax.legend(handles=handles, fontsize=6, loc="lower right",
                  ncol=2, framealpha=0.75,
                  title=f"Tasks ({n_tasks} total)", title_fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "tsne.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {out_path}")


# ── 텍스트 리포트 ──────────────────────────────────────────────────────────────

def save_report(shuffle_results, data, out_dir, ckpt_path):
    tasks = data["tasks"]
    task_counts = Counter(tasks)
    n_tasks = len(task_counts)
    N = len(tasks)
    z_dim = data["z_post"].shape[1]
    D_task = z_dim // 2

    lines = [
        "# z Analysis Report — M5 VLM SFP Plan",
        "",
        f"- **체크포인트**: `{ckpt_path}`",
        f"- **총 샘플 수**: {N}",
        f"- **작업(task) 수**: {n_tasks}",
        f"- **z_dim**: {z_dim}  (z_task: 앞 {D_task}dim, z_motion: 뒤 {z_dim - D_task}dim)",
        "",
        "---",
        "",
        "## 1. Partial Shuffle 결과",
        "",
    ]

    for z_key in ["z_post", "z_prior"]:
        r = shuffle_results[z_key]
        lines += [
            f"### {r['label']}",
            "",
            "| 조건 | Action MSE | gap vs Normal |",
            "|------|:----------:|:-------------:|",
            f"| Normal (기준) | {r['mse_normal']:.6f} | — |",
            f"| Full Shuffle | {r['mse_full']:.6f} | {r['gap_full']:+.6f} |",
            f"| z_task 절반 셔플 | {r['mse_task']:.6f} | {r['gap_task']:+.6f}  ({r['task_ratio']*100:.1f}%) |",
            f"| z_motion 절반 셔플 | {r['mse_motion']:.6f} | {r['gap_motion']:+.6f}  ({r['motion_ratio']*100:.1f}%) |",
            "",
        ]

        if r["task_ratio"] > 0.5:
            verdict = (
                f"**→ z_task 파트가 전체 gap의 {r['task_ratio']*100:.1f}%를 담당**: "
                "VLM z에 task 정보가 존재함. InfoNCE loss 적용 시 효과적일 가능성 높음."
            )
        elif r["motion_ratio"] > 0.5:
            verdict = (
                f"**→ z_motion 파트가 전체 gap의 {r['motion_ratio']*100:.1f}%를 담당**: "
                "VLM z가 task 구분보다 현재 motion/state 정보에 집중. "
                "prior flow 구조 재검토 또는 action-conditioned KD 방식 고려 필요."
            )
        else:
            verdict = (
                "**→ task/motion 기여가 비슷**: "
                "z의 절반 단위 분리가 task vs motion에 잘 대응하지 않을 가능성. "
                "disentanglement loss (classification loss on z_task) 추가 후 재분석 권장."
            )
        lines += [verdict, ""]

    lines += [
        "---",
        "",
        "## 2. t-SNE 분석",
        "",
        "→ `tsne.png` 참조.",
        "",
        "판단 기준:",
        "- task별로 **뭉치는 클러스터** 가 보이면: z가 task 정보를 잘 담음",
        "- **고르게 섞여 있으면**: z가 task를 구분하지 못함 → z_shuffle_gap 낮은 원인",
        "",
        "---",
        "",
        "## 3. 가설 검증 요약",
        "",
        "| 가설 | 내용 | 확인 방법 |",
        "|------|------|----------|",
        "| 가설 A | z_task gap이 높음 → InfoNCE가 효과적일 것 | gap_task > gap_motion |",
        "| 가설 B | z_motion gap이 높음 → prior 구조 재검토 필요 | gap_motion > gap_task |",
        "",
        "→ **위 결과를 보고 아이디어 1(InfoNCE) vs 아이디어 2(KD) 중 처방 선택**",
        "",
        "---",
        "",
        "## 4. 다음 스텝",
        "",
        "- **가설 A 지지 시**: `scripts/train_vlm.py`에 z-InfoNCE loss 추가 실험",
        "- **가설 B 지지 시**: action-conditioned plan token (KD 방향 A) 실험",
        "- **공통**: t-SNE 클러스터링 결과를 논문 Figure에 추가 (z 품질 시각적 증거)",
    ]

    out_path = os.path.join(out_dir, "z_analysis_report.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[저장] {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="M5 VLM z 품질 진단 — partial shuffle + t-SNE"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="LatentVLA 체크포인트 경로")
    parser.add_argument("--data_path", default=None,
                        help="데이터셋 경로 override")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="수집할 최대 배치 수 (기본: val set 전체)")
    parser.add_argument("--tsne_max", type=int, default=2000,
                        help="t-SNE 최대 샘플 수 (기본: 2000)")
    parser.add_argument("--n_repeat", type=int, default=3,
                        help="셔플 반복 횟수 (기본: 3)")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (기본: 자동)")
    parser.add_argument("--out_dir", default=None,
                        help="결과 저장 디렉토리 (기본: <ckpt_dir>/z_analysis/)")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[eval_z_analysis] device={device}")

    # 출력 디렉토리
    from datetime import date
    today = date.today().strftime("%Y%m%d")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out_dir or os.path.join(
        project_root, "outputs", "analyses", f"z_analysis_m5_{today}"
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[eval_z_analysis] 결과 저장 위치: {out_dir}")

    # 모델 + 데이터 로드
    print("\n[1/4] 모델 로드...")
    model, val_loader, cfg = load_model_and_data(
        args.checkpoint, device, args.data_path
    )
    print(f"  z_form = {cfg['system2']['z_form']}")
    print(f"  z_dim  = {model.system1.z_dim}")

    # z 벡터 수집
    print("\n[2/4] z 벡터 수집 (재학습 없음, eval only)...")
    data = collect_z_vectors(model, val_loader, device, args.max_batches)
    N = len(data["tasks"])
    n_tasks = len(set(data["tasks"]))
    print(f"  수집 완료: {N}개 샘플, {n_tasks}개 task")

    # Partial Shuffle 분석
    print(f"\n[3/4] Partial Shuffle 분석 (n_repeat={args.n_repeat})...")
    shuffle_results = run_partial_shuffle(model, data, device, n_repeat=args.n_repeat)

    for z_key in ["z_post", "z_prior"]:
        r = shuffle_results[z_key]
        print(f"\n  [{r['label']}]")
        print(f"    mse_normal  = {r['mse_normal']:.6f}")
        print(f"    gap_full    = {r['gap_full']:+.6f}")
        print(f"    gap_task    = {r['gap_task']:+.6f}  ({r['task_ratio']*100:.1f}% of full)")
        print(f"    gap_motion  = {r['gap_motion']:+.6f}  ({r['motion_ratio']*100:.1f}% of full)")
        if r["task_ratio"] > 0.5:
            print(f"    → 가설 A 지지: z_task 파트가 주도 (InfoNCE 유효할 것)")
        elif r["motion_ratio"] > 0.5:
            print(f"    → 가설 B 지지: z_motion 파트가 주도 (KD 방향 고려)")
        else:
            print(f"    → task/motion 기여 비슷 (추가 분석 필요)")

    plot_partial_shuffle(shuffle_results, out_dir)

    # z 벡터 저장 (t-SNE 재생성 시 모델 재로드 불필요)
    npz_path = os.path.join(out_dir, "z_vectors.npz")
    np.savez_compressed(
        npz_path,
        z_post=data["z_post"].numpy(),
        z_prior=data["z_prior"].numpy(),
        tasks=np.array(data["tasks"]),
    )
    print(f"[저장] {npz_path}")

    # t-SNE
    print(f"\n[4/4] t-SNE 시각화 (max_samples={args.tsne_max})...")
    run_tsne(data, out_dir, max_samples=args.tsne_max)

    # 리포트
    save_report(shuffle_results, data, out_dir, args.checkpoint)

    # JSON 요약
    summary = {}
    for z_key in ["z_post", "z_prior"]:
        r = shuffle_results[z_key]
        summary[z_key] = {k: round(v, 6) if isinstance(v, float) else v
                          for k, v in r.items()}
    json_path = os.path.join(out_dir, "z_analysis_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[저장] {json_path}")

    print("\n완료.")
    print(f"  partial_shuffle.png  → {out_dir}/partial_shuffle.png")
    print(f"  tsne.png             → {out_dir}/tsne.png")
    print(f"  z_analysis_report.md → {out_dir}/z_analysis_report.md")
    print(f"  z_vectors.npz        → {out_dir}/z_vectors.npz")


if __name__ == "__main__":
    main()
