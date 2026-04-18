# VLA 프로젝트 규칙

- 실험 결과는 `experiments/` 폴더에 저장, `experiments.md`에 기록
- 새 config는 기존 yaml 복사 후 수정
- screen 세션으로 장시간 학습 실행
- 학습은 항상 DDP 기본 — `torchrun` 대신 `python -m torch.distributed.run` 사용 (Singularity 환경에서 torchrun이 base conda를 잡는 문제 회피)
- 실행 커맨드는 항상 한 줄로 제공
- GPU walltime 72시간 제한 → 끊기면 재실행만 하면 latest.pt에서 자동 resume
- best 체크포인트: `best_ep{epoch:03d}.pt` (최신 1개만 유지)
- 학습 그래프: `save_interval`마다 `curve_ep{epoch}.png` 자동 저장

## 실험 결과 정리 규칙

각 runs 디렉토리(`outputs/runs/<run_name>/`)에는 반드시 아래 두 파일을 생성한다:

### monitor.png (자동 생성)
- 학습 완료 시 `VLMTrainer._generate_monitor()`가 자동으로 `scripts/plot_monitor.py`를 실행해 생성
- M2(DetLatent) 값을 목표 기준선으로 표시: mse_prior=0.4776, mse_posterior=0.0017, z_shuffle_gap=0.7837, prior_posterior_gap=0.4759
- 중간 확인: `conda run -n vla python3 scripts/plot_monitor.py --run_dir outputs/runs/<name>`
- `.gitignore`에서 추적 허용 → `git add outputs/runs/<name>/monitor.png`로 push 가능

### result.png
- `plot_result.py`를 작성하고 `conda run -n vla python3 plot_result.py`로 생성
- 포함할 패널:
  1. Train Loss Curve (total + 각 component)
  2. Latest Loss Breakdown (bar chart, 마지막 epoch 기준)
  3. Val MSE — Prior / Posterior / Best-of-K
  4. z Quality Metrics (z_shuffle_gap, prior_posterior_gap)
  5. Final Validation Metrics Summary (table, 최신 val epoch 기준)
- 스타일: `sanity_check_20260331/plot_result.py` 레이아웃/색상 기준 유지

### result.md
- 포함할 섹션:
  1. 실험 메타 (날짜, 목적, 설정, 현황)
  2. 무엇을 검증하나
  3. 학습 손실 곡선 (주요 epoch 표)
  4. 검증 지표 (val 전체 표)
  5. 결과 해석 및 인사이트 (긍정적 / 주의)
  6. 다음 스텝
  7. 저장 파일 목록

## 분석 결과 정리 규칙

학습 run이 아닌 분석(eval, ablation, visualization 등)은 `outputs/analyses/<name>_<YYYYMMDD>/`에 저장한다.

```
outputs/
├── runs/          ← 학습 run (torchrun으로 실행한 것)
└── analyses/      ← 분석 (기존 체크포인트 기반 eval, 시각화 등)
    └── <name>_<YYYYMMDD>/
        ├── <analysis>.png
        ├── <analysis>_report.md
        └── <analysis>_summary.json
```

- 분석 스크립트의 기본 `out_dir`은 `outputs/analyses/<name>_<YYYYMMDD>/`로 설정
- 분석명은 `z_analysis_m5`, `tsne_compare`, `shuffle_ablation` 등 내용을 반영
