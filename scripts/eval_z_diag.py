"""
eval_z_diag.py — z-space 판결 실험 (M8 이후 진단용)

실험 1. z-space spread      : mu_q 분산 / exp(logvar_q) / 샘플 spread (전체 + task별)
실험 2. z-drop test         : null-z / shuffle-z → action MSE 민감도
실험 3. future-probe        : same-context / different-future 쌍에서 mu_q 벌어짐 확인

판정 매트릭스:
  케이스 A: z_spread 낮음 + z_drop 민감도 낮음  → collapse + non-usage 동시 발생
  케이스 B: z_spread 높음 + z_drop 민감도 낮음  → binding failure
  케이스 C: z_spread 낮음 + z_drop 민감도 높음  → posterior weak (decoder는 쓰려 함)
  케이스 D: z_spread 높음 + z_drop 민감도 높음  → z 살아 있음, prior/metric 재점검

Usage:
    conda run -n vla python3 scripts/eval_z_diag.py \\
        --checkpoint outputs/runs/vlm_sfp_infonce_s1only_20260418/ckpt_90.pt

    # 빠른 테스트 (val 배치 30개만)
    conda run -n vla python3 scripts/eval_z_diag.py \\
        --checkpoint outputs/runs/vlm_sfp_infonce_s1only_20260418/ckpt_90.pt \\
        --max_batches 30

출력:
    outputs/analyses/z_diag_<run_name>_<date>/
        diag_report.md
        z_spread.png
        z_drop.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm

from training.builder import build_vlm_datasets, build_dataloaders_vlm, build_vlm_model
from models.flow_utils import euler_integrate


# ── 모델 로드 ──────────────────────────────────────────────────────────────────

def load_model_and_data(ckpt_path, device, data_path_override=None):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    if data_path_override:
        cfg.setdefault("data", {})["dataset_path"] = data_path_override
    if "data" not in cfg:
        raise ValueError("cfg에 'data' 키 없음. --data_path 지정 필요.")
    train_ds, val_ds = build_vlm_datasets(cfg)
    _, val_loader, _ = build_dataloaders_vlm(train_ds, val_ds, cfg)
    model = build_vlm_model(cfg, action_dim=train_ds.action_dim, proprio_dim=train_ds.proprio_dim)
    saved = ckpt["model"]
    if saved.get("system2_lora"):
        model.system2.enable_lora()
    model.load_state_dict_from_save(saved)
    model = model.to(device).eval()
    return model, val_loader, cfg


# ── 데이터 수집 ────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_data(model, val_loader, device, max_batches=None):
    """val set에서 mu_q, logvar_q, z_star, f_tilde, actions, task 레이블 수집."""
    mus, logvars, zstars, ftildes, actions_all, tasks_all = [], [], [], [], [], []

    for i, batch in enumerate(tqdm(val_loader, desc="데이터 수집")):
        if max_batches is not None and i >= max_batches:
            break

        actions = batch["actions"].to(device)
        proprio  = batch["proprio"].to(device)
        B = actions.shape[0]

        raw_image = batch.get("raw_image", batch["image"])
        pv, ids, mask = model.system2.prepare_inputs(raw_image, batch["language"], device)
        f_tilde = model.system2(pv, ids, mask, proprio)

        future_feat = None
        if model.system1.use_future and "future_image" in batch:
            future_feat = model._siglip.encode_image_only(batch["future_image"].to(device))

        parts = [f_tilde, actions.reshape(B, -1)]
        if future_feat is not None:
            parts.append(future_feat)
        q_in = torch.cat(parts, dim=-1)
        mu_q, logvar_q = model.system1.posterior_enc(q_in)
        z_star = mu_q + (0.5 * logvar_q).exp() * torch.randn_like(mu_q)

        lang_tasks = batch["language"]
        file_tasks = [
            os.path.basename(fp).replace("_demo.hdf5", "").replace("_", " ")
            for fp in batch["file"]
        ]
        tasks_batch = file_tasks if len(set(lang_tasks)) <= 1 else lang_tasks

        mus.append(mu_q.cpu())
        logvars.append(logvar_q.cpu())
        zstars.append(z_star.cpu())
        ftildes.append(f_tilde.cpu())
        actions_all.append(actions.cpu())
        tasks_all.extend(tasks_batch)

    return {
        "mu_q":    torch.cat(mus,       dim=0),
        "logvar_q":torch.cat(logvars,   dim=0),
        "z_star":  torch.cat(zstars,    dim=0),
        "f_tilde": torch.cat(ftildes,   dim=0),
        "actions": torch.cat(actions_all, dim=0),
        "tasks":   tasks_all,
    }


# ── action MSE (임의 z 주입) ───────────────────────────────────────────────────

@torch.no_grad()
def mse_with_z(model, f_tilde_cpu, z_cpu, actions_cpu, device, batch_size=64):
    N = actions_cpu.shape[0]
    total, count = 0.0, 0
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        f = f_tilde_cpu[s:e].to(device)
        z = z_cpu[s:e].to(device)
        a_gt = actions_cpu[s:e].to(device)
        cond = torch.cat([f, z], dim=-1)
        a_pred = euler_integrate(
            model.system1.action_flow, cond,
            model.system1.x_dim, model.system1.flow_steps
        ).reshape(e - s, model.system1.action_horizon, model.system1.action_dim)
        total += F.mse_loss(a_pred, a_gt).item() * (e - s)
        count += (e - s)
    return total / count


# ── 실험 1: z-space spread ─────────────────────────────────────────────────────

def run_z_spread(data):
    mu_q     = data["mu_q"]      # (N, z_dim)
    logvar_q = data["logvar_q"]  # (N, z_dim)
    z_star   = data["z_star"]    # (N, z_dim)
    tasks    = data["tasks"]

    # 전체 통계
    global_stats = {
        "z_mu_norm_mean":  mu_q.norm(dim=-1).mean().item(),
        "z_mu_norm_std":   mu_q.norm(dim=-1).std().item(),
        "z_mu_var_mean":   mu_q.var(dim=0).mean().item(),      # 핵심: batch 방향 분산
        "z_var_mean":      logvar_q.exp().mean().item(),
        "z_var_std":       logvar_q.exp().std().item(),
        "z_sample_var":    z_star.var(dim=0).mean().item(),
    }

    # task별 mu_q centroid 및 within/between 분산
    task_groups = defaultdict(list)
    for idx, t in enumerate(tasks):
        task_groups[t].append(mu_q[idx])

    task_centroids = {}
    within_vars = []
    for t, vecs in task_groups.items():
        stack = torch.stack(vecs, dim=0)       # (n_t, z_dim)
        centroid = stack.mean(dim=0)
        task_centroids[t] = centroid
        within_vars.append(stack.var(dim=0).mean().item())

    centroids = torch.stack(list(task_centroids.values()), dim=0)  # (T, z_dim)
    between_var = centroids.var(dim=0).mean().item() if len(task_centroids) > 1 else 0.0

    task_stats = {
        "n_tasks":        len(task_groups),
        "within_var_mean": float(np.mean(within_vars)),
        "between_var_mean": between_var,
        "between_within_ratio": between_var / (np.mean(within_vars) + 1e-8),
    }

    return global_stats, task_stats, task_groups, mu_q


# ── 실험 2: z-drop test ────────────────────────────────────────────────────────

def run_z_drop(model, data, device, n_repeat=3):
    f   = data["f_tilde"]
    z   = data["z_star"]
    a   = data["actions"]
    N   = z.shape[0]
    z_dim = z.shape[1]

    mse_orig = mse_with_z(model, f, z, a, device)

    # null-z: 전체 평균으로 치환 (zero mean, no signal)
    z_null = torch.zeros_like(z)
    mse_null = mse_with_z(model, f, z_null, a, device)

    # batch shuffle (여러 번 평균)
    mse_shuffled_list = []
    for _ in range(n_repeat):
        perm = torch.randperm(N)
        mse_shuffled_list.append(mse_with_z(model, f, z[perm], a, device))
    mse_shuffle = float(np.mean(mse_shuffled_list))

    return {
        "mse_posterior":       mse_orig,
        "mse_null_z":          mse_null,
        "mse_shuffle_z":       mse_shuffle,
        "delta_null":          mse_null    - mse_orig,
        "delta_shuffle":       mse_shuffle - mse_orig,
        "null_ratio":          mse_null    / (mse_orig + 1e-8),
        "shuffle_ratio":       mse_shuffle / (mse_orig + 1e-8),
    }


# ── 실험 3: future-probe ───────────────────────────────────────────────────────

def run_future_probe(data, n_pairs=500):
    """
    같은 task의 다른 샘플 쌍에서 |mu_q^i - mu_q^j| 측정.
    random pair와 비교해 posterior가 미래 정보를 담는지 확인.
    """
    mu_q  = data["mu_q"]
    tasks = data["tasks"]

    task_groups = defaultdict(list)
    for idx, t in enumerate(tasks):
        task_groups[t].append(idx)

    # same-task 쌍
    same_dists = []
    rng = np.random.default_rng(42)
    for t, idxs in task_groups.items():
        if len(idxs) < 2:
            continue
        idxs = np.array(idxs)
        for _ in range(min(n_pairs // max(len(task_groups), 1), 200)):
            i, j = rng.choice(len(idxs), size=2, replace=False)
            d = (mu_q[idxs[i]] - mu_q[idxs[j]]).norm().item()
            same_dists.append(d)

    # random 쌍
    N = mu_q.shape[0]
    rand_dists = []
    for _ in range(min(n_pairs, 1000)):
        i, j = rng.choice(N, size=2, replace=False)
        d = (mu_q[i] - mu_q[j]).norm().item()
        rand_dists.append(d)

    return {
        "same_task_dist_mean":  float(np.mean(same_dists))  if same_dists  else None,
        "same_task_dist_std":   float(np.std(same_dists))   if same_dists  else None,
        "random_dist_mean":     float(np.mean(rand_dists))  if rand_dists  else None,
        "random_dist_std":      float(np.std(rand_dists))   if rand_dists  else None,
        "ratio_same_to_random": (float(np.mean(same_dists)) / float(np.mean(rand_dists)))
                                if same_dists and rand_dists else None,
    }


# ── 판정 ───────────────────────────────────────────────────────────────────────

def verdict(global_stats, drop_res):
    """
    z_mu_var_mean 기준: 0.1 미만이면 spread 낮음
    delta_null 기준: 0.02 미만이면 민감도 낮음 (= decoder가 z 거의 안 씀)
    """
    spread_low   = global_stats["z_mu_var_mean"] < 0.1
    usage_low    = drop_res["delta_null"] < 0.02

    if spread_low and usage_low:
        case = "A"
        msg  = "collapse + non-usage 동시 발생 → posterior/encoder 와 binding 모두 점검"
    elif not spread_low and usage_low:
        case = "B"
        msg  = "binding failure 주범 → FiLM / CFG / conditioning 강화 우선"
    elif spread_low and not usage_low:
        case = "C"
        msg  = "posterior 약함, decoder는 z 쓰려 함 → posterior/encoder 강화 우선"
    else:
        case = "D"
        msg  = "z 살아 있음 → prior 학습 문제 또는 metric 재점검"

    return case, msg


# ── 시각화 ─────────────────────────────────────────────────────────────────────

def plot_z_spread(global_stats, task_stats, task_groups, mu_q, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Exp 1: z-space Spread Analysis", fontsize=13)

    # (a) mu_q norm 분포
    norms = mu_q.norm(dim=-1).numpy()
    axes[0].hist(norms, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].set_title("‖mu_q‖ distribution")
    axes[0].set_xlabel("norm")
    axes[0].set_ylabel("count")

    # (b) per-dim mu_q variance
    dim_var = mu_q.var(dim=0).numpy()
    axes[1].bar(range(len(dim_var)), np.sort(dim_var)[::-1], color="tomato", width=1.0)
    axes[1].set_title(f"mu_q var per dim (mean={global_stats['z_mu_var_mean']:.4f})")
    axes[1].set_xlabel("dim (sorted desc)")
    axes[1].set_ylabel("variance")

    # (c) within vs between task variance
    labels  = ["within-task\n(should be high)", "between-task\n(should be high)"]
    values  = [task_stats["within_var_mean"], task_stats["between_var_mean"]]
    colors  = ["#4C8BF5", "#F5774C"]
    axes[2].bar(labels, values, color=colors)
    axes[2].set_title(f"Task spread  (B/W ratio={task_stats['between_within_ratio']:.2f})")
    axes[2].set_ylabel("mean variance")
    for ax, v in zip([axes[2]], values):
        pass

    plt.tight_layout()
    path = os.path.join(out_dir, "z_spread.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def plot_z_drop(drop_res, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Exp 2: z-drop Test", fontsize=13)

    # (a) MSE 비교 bar
    labels = ["posterior z\n(original)", "null z\n(zeros)", "shuffle z\n(batch perm)"]
    values = [drop_res["mse_posterior"], drop_res["mse_null_z"], drop_res["mse_shuffle_z"]]
    colors = ["#4C8BF5", "#F5774C", "#F5C54C"]
    bars = axes[0].bar(labels, values, color=colors)
    axes[0].set_title("Action MSE by z condition")
    axes[0].set_ylabel("MSE")
    for bar, v in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.002, f"{v:.4f}",
                     ha="center", va="bottom", fontsize=9)

    # (b) delta (= MSE 상승량)
    deltas = [drop_res["delta_null"], drop_res["delta_shuffle"]]
    dlabels = ["Δ null", "Δ shuffle"]
    dcolors = ["#F5774C", "#F5C54C"]
    dbars = axes[1].bar(dlabels, deltas, color=dcolors)
    axes[1].axhline(0.02, color="red", linestyle="--", linewidth=1, label="threshold 0.02")
    axes[1].set_title("Δ MSE (condition − original)")
    axes[1].set_ylabel("ΔMSE")
    axes[1].legend(fontsize=8)
    for bar, v in zip(dbars, deltas):
        axes[1].text(bar.get_x() + bar.get_width() / 2, max(v, 0) + 0.001, f"{v:.4f}",
                     ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "z_drop.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


# ── 리포트 ─────────────────────────────────────────────────────────────────────

def write_report(ckpt_path, global_stats, task_stats, drop_res, probe_res,
                 case, verdict_msg, out_dir):
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    run_name = os.path.basename(os.path.dirname(ckpt_path))
    ckpt_name = os.path.basename(ckpt_path)

    lines = [
        f"# z-space 판결 실험 리포트",
        f"",
        f"- **실행일**: {date_str}",
        f"- **체크포인트**: `{run_name}/{ckpt_name}`",
        f"",
        f"---",
        f"",
        f"## 판정: 케이스 {case}",
        f"",
        f"> {verdict_msg}",
        f"",
        f"| 판정 기준 | 값 | 임계값 | 결론 |",
        f"|-----------|-----|--------|------|",
        f"| z_mu_var_mean (spread) | {global_stats['z_mu_var_mean']:.5f} | < 0.1 = 낮음 | {'**낮음**' if global_stats['z_mu_var_mean'] < 0.1 else '높음'} |",
        f"| delta_null (decoder 사용) | {drop_res['delta_null']:.5f} | < 0.02 = 안씀 | {'**안 씀**' if drop_res['delta_null'] < 0.02 else '쓰고 있음'} |",
        f"",
        f"---",
        f"",
        f"## 실험 1: z-space Spread",
        f"",
        f"### 전체 통계",
        f"",
        f"| 지표 | 값 |",
        f"|------|-----|",
    ]
    for k, v in global_stats.items():
        lines.append(f"| {k} | {v:.6f} |")

    lines += [
        f"",
        f"### Task별 spread",
        f"",
        f"| 지표 | 값 |",
        f"|------|-----|",
        f"| 태스크 수 | {task_stats['n_tasks']} |",
        f"| within-task mu_q var | {task_stats['within_var_mean']:.6f} |",
        f"| between-task mu_q var | {task_stats['between_var_mean']:.6f} |",
        f"| between/within ratio | {task_stats['between_within_ratio']:.4f} |",
        f"",
        f"> **해석 기준**: between/within ratio >> 1 이면 z가 task를 구분함.",
        f"  ratio ≈ 1 이면 구분 못 함. < 1 이면 within 분산이 더 크다(노이즈 우세).",
        f"",
        f"---",
        f"",
        f"## 실험 2: z-drop Test",
        f"",
        f"| 조건 | action MSE | Δ(vs original) | ratio |",
        f"|------|-----------|----------------|-------|",
        f"| posterior z (원본) | {drop_res['mse_posterior']:.5f} | — | 1.000 |",
        f"| null z (zeros)     | {drop_res['mse_null_z']:.5f} | {drop_res['delta_null']:+.5f} | {drop_res['null_ratio']:.3f} |",
        f"| shuffle z (batch perm) | {drop_res['mse_shuffle_z']:.5f} | {drop_res['delta_shuffle']:+.5f} | {drop_res['shuffle_ratio']:.3f} |",
        f"",
        f"> **해석 기준**: Δ < 0.02 → decoder가 z를 거의 쓰지 않음 (non-usage).",
        f"  Δ > 0.05 → z가 실질적으로 action에 기여하고 있음.",
        f"",
        f"---",
        f"",
        f"## 실험 3: Future Probe",
        f"",
    ]

    if probe_res["same_task_dist_mean"] is not None:
        lines += [
            f"| 지표 | 값 |",
            f"|------|-----|",
            f"| same-task ‖Δmu_q‖ mean | {probe_res['same_task_dist_mean']:.4f} |",
            f"| same-task ‖Δmu_q‖ std  | {probe_res['same_task_dist_std']:.4f} |",
            f"| random ‖Δmu_q‖ mean    | {probe_res['random_dist_mean']:.4f} |",
            f"| random ‖Δmu_q‖ std     | {probe_res['random_dist_std']:.4f} |",
            f"| same / random ratio    | {probe_res['ratio_same_to_random']:.4f} |",
            f"",
            f"> **해석 기준**: ratio ≈ 1 → same-task 샘플도 서로 퍼져 있음 (미래 정보 인코딩 중).",
            f"  ratio << 1 → 같은 task면 mu_q가 거의 같음 (미래 정보 무시 또는 posterior collapse).",
        ]
    else:
        lines.append("_(태스크별 샘플 부족으로 실험 3 스킵)_")

    lines += [
        f"",
        f"---",
        f"",
        f"## 권장 다음 스텝",
        f"",
    ]

    if case == "A":
        lines += [
            f"- posterior encoder 입력/출력 크기, logvar 범위 점검",
            f"- KL-like regularizer 또는 InfoNCE temperature 강화 검토",
            f"- binding failure 해소를 위해 z-conditioning 방식 변경 (FiLM 등)",
        ]
    elif case == "B":
        lines += [
            f"- z conditioning 방식 강화 (FiLM, cross-attention, CFG)",
            f"- action_flow에 z가 실제로 개입하는지 attention weight 확인",
        ]
    elif case == "C":
        lines += [
            f"- posterior encoder 표현력 강화 (deeper, wider, or contrastive)",
            f"- future_feat 신호 품질 점검 (SigLIP feature가 충분히 다양한지)",
            f"- semantic_weight 올리거나 InfoNCE temperature 낮추기",
        ]
    else:
        lines += [
            f"- z_shuffle_gap 메트릭 재점검 (테스트 방식 버그 가능성)",
            f"- prior flow 학습 안정성 확인 (prior_flow_loss 역증가 원인)",
            f"- 더 긴 학습 (100ep 완주) 후 재판정",
        ]

    report_path = os.path.join(out_dir, "diag_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return report_path


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--max_batches", type=int, default=None,
                        help="None=전체 val. 빠른 테스트 시 30 정도 지정")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default=None,
                        help="기본값: outputs/analyses/z_diag_<run>_<date>/")
    args = parser.parse_args()

    run_name = os.path.basename(os.path.dirname(args.checkpoint))
    date_str = datetime.now().strftime("%Y%m%d")
    out_dir  = args.out_dir or os.path.join(
        "outputs", "analyses", f"z_diag_{run_name}_{date_str}"
    )
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"[eval_z_diag] checkpoint : {args.checkpoint}")
    print(f"[eval_z_diag] device     : {device}")
    print(f"[eval_z_diag] out_dir    : {out_dir}")
    if args.max_batches:
        print(f"[eval_z_diag] max_batches: {args.max_batches} (빠른 테스트 모드)")

    # 로드
    model, val_loader, cfg = load_model_and_data(args.checkpoint, device, args.data_path)

    # 데이터 수집
    data = collect_data(model, val_loader, device, args.max_batches)
    N = data["mu_q"].shape[0]
    print(f"[eval_z_diag] 수집 완료: N={N}")

    # 실험 1
    print("[eval_z_diag] 실험 1: z-space spread ...")
    global_stats, task_stats, task_groups, mu_q = run_z_spread(data)
    plot_z_spread(global_stats, task_stats, task_groups, mu_q, out_dir)

    # 실험 2
    print("[eval_z_diag] 실험 2: z-drop test ...")
    drop_res = run_z_drop(model, data, device)

    # 실험 3
    print("[eval_z_diag] 실험 3: future probe ...")
    probe_res = run_future_probe(data)

    # 판정
    case, verdict_msg = verdict(global_stats, drop_res)
    plot_z_drop(drop_res, out_dir)

    # 리포트
    report_path = write_report(
        args.checkpoint, global_stats, task_stats,
        drop_res, probe_res, case, verdict_msg, out_dir
    )

    # 콘솔 요약
    print()
    print("=" * 60)
    print(f"  판정: 케이스 {case}")
    print(f"  {verdict_msg}")
    print("=" * 60)
    print(f"  z_mu_var_mean : {global_stats['z_mu_var_mean']:.5f}")
    print(f"  z_sample_var  : {global_stats['z_sample_var']:.5f}")
    print(f"  delta_null    : {drop_res['delta_null']:+.5f}")
    print(f"  delta_shuffle : {drop_res['delta_shuffle']:+.5f}")
    if probe_res["ratio_same_to_random"] is not None:
        print(f"  probe ratio   : {probe_res['ratio_same_to_random']:.4f}")
    print("=" * 60)
    print(f"  리포트: {report_path}")
    print(f"  그래프: {out_dir}/z_spread.png, z_drop.png")

    # JSON 저장
    summary = {
        "checkpoint": args.checkpoint,
        "N": N,
        "case": case,
        "verdict": verdict_msg,
        "global_stats": global_stats,
        "task_stats": task_stats,
        "drop_test": drop_res,
        "future_probe": probe_res,
    }
    with open(os.path.join(out_dir, "diag_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
