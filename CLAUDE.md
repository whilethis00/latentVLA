# VLA 프로젝트 규칙

- 실험 결과는 `experiments/` 폴더에 저장, `experiments.md`에 기록
- 새 config는 기존 yaml 복사 후 수정
- screen 세션으로 장시간 학습 실행
- 학습은 항상 DDP (torchrun) 기본
- GPU walltime 72시간 제한 → 끊기면 재실행만 하면 latest.pt에서 자동 resume
- best 체크포인트: `best_ep{epoch:03d}.pt` (최신 1개만 유지)
- 학습 그래프: `save_interval`마다 `curve_ep{epoch}.png` 자동 저장
