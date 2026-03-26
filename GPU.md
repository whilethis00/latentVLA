# SNU ECE GPU 클러스터 — 실전 운영 노트

> 작성일: 2026-03-26 | 현재 로그인 노드: ECE-util1

---

## 0. 전체 워크플로우 구조

```
로그인 노드 (ECE-util1)
 └─ screen  ← Claude Code 세션 유지 (screen은 Claude가 제어 못함)
     └─ qsub → GPU 노드 SSH 접속 권한 획득
         └─ tmux  ← Claude가 제어 가능한 단위
             └─ singularity 컨테이너 (실제 실험 환경)
                 └─ CUDA_VISIBLE_DEVICES=N python train.py
                    (할당 GPU 외 노드 내 다른 GPU도 접근 가능)
```

**핵심 원리:**
- PBS 잡을 받는 이유 = 그 **GPU 노드에 SSH 접속 권한**을 얻기 위함
- 노드에 들어오면 해당 노드의 **모든 GPU 접근 가능** (PBS는 CUDA_VISIBLE_DEVICES만 설정, 격리 없음)
- `screen` → Claude Code 프로세스 유지 / `tmux` → Claude가 명령 전달하는 작업 단위
- 여러 introai 계정으로 각각 잡을 받으면 GPU를 더 많이 확보 가능

**Claude 제어 방법:**
```bash
# tmux 세션 새로 만들기
tmux new-session -s work

# Claude가 tmux에 명령 보내기
tmux send-keys -t work "CUDA_VISIBLE_DEVICES=0,1 python train.py" Enter

# singularity 안으로 들어가기
tmux send-keys -t work "singularity exec --nv /path/to/container.sif bash" Enter
```

---

## 1. 나의 현재 할당 현황 (introai4)

| 항목 | 값 |
|------|---|
| 잡 ID | 83455 |
| 노드 | **ece-agpu8** |
| GPU | **1개** |
| CPU | 6코어 |
| RAM | 192GB |
| 큐 | coss_agpu |
| walltime | 72시간 (경과: ~68시간) |

> ⚠️ walltime이 거의 다 됐음 — 잡 재제출 준비 필요

접속 방법:
```bash
# 로그인 노드에서 GPU 노드로 직접 ssh (잡 실행 중일 때)
ssh ece-agpu8
```

---

## 2. 현재 내 노드 확인 & GPU 세팅

### 현재 노드 이름
```bash
hostname
```

### PBS 할당 GPU 확인
```bash
/opt/pbs/bin/qstat -f $(qstat | grep $USER | awk '{print $1}' | head -1) | grep -E "exec_vnode|ngpus"
```

### 빠른 버전: 내 잡 한 줄 확인
```bash
/opt/pbs/bin/qstat -ans | grep $(whoami)
```

### GPU 노드에서 CUDA 확인
```bash
# GPU 노드에 들어간 후
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
```

> 로그인 노드(ECE-util1)에서는 `nvidia-smi` 불가 — GPU 노드에 들어가야 함

---

## 3. introai 계열 계정 GPU 점유 현황 (2026-03-26 기준)

| 계정 | 잡 ID | 노드 | GPU수 | RAM | 큐 | 가동시간 |
|------|-------|------|-------|-----|-----|---------|
| **introai4** | 83455 | ece-agpu8 | 1 | 192gb | coss_agpu | ~68h |
| introai1 | 83467 | ece-agpu9 | 4 | 768gb | test_agpu | ~71h |
| introai1 | 85179 | ece-agpu11 | 2 | 384gb | coss_agpu | ~38h |
| introai5 | 85216 | ece-agpu6 | 1 | 192gb | coss_agpu | ~25h |
| introai11 | 85140 | ece-agpu9 | 1 | 192gb | coss_agpu | ~48h |
| introai14 | 85300 | ece-a6gpu1 | 1 | 128gb | coss_a6gpu | ~4h |
| introai15 | 85223 | ece-agpu15 | 2 | 384gb | coss_agpu | ~24h |
| introai15 | 85286 | ece-a6gpu4 | 1 | 128gb | coss_a6gpu | ~7h |
| introai20 | 85199 | ece-a6gpu4 | 1 | 128gb | coss_a6gpu | ~27h |
| introai27 | 83450 | ece-agpu6 | 1 | 192gb | coss_agpu | ~2h |
| introai27 | 83995 | ece-a6gpu2 | 1 | 128gb | coss_a6gpu | ~50h |
| introai27 | 85219 | ece-agpu17 | 1(MIG) | 62gb | mig_agpu | ~6h |
| introai28 | 85283 | ece-a6gpu2 | 1 | 128gb | coss_a6gpu | ~7h |
| introai28 | 85287 | ece-agpu13 | 2 | 384gb | coss_agpu | ~6h |
| introai29 | 85285 | ece-a6gpu2 | 1 | 128gb | coss_a6gpu | ~7h |
| introai29 | 85292 | ece-agpu2 | 1 | 192gb | coss_agpu | ~5h |

---

## 4. 잡 제출 명령어

### 인터랙티브 잡 (표준 — coss_agpu 큐)
```bash
/opt/pbs/bin/qsub -I -q coss_agpu \
    -l select=1:ncpus=6:mem=192g:ngpus=1:Qlist=agpu:container_engine=singularity \
    -l walltime=72:00:00
```

### GPU 2개 요청 (introai4 계정 최대)
```bash
/opt/pbs/bin/qsub -I -q coss_agpu \
    -l select=1:ncpus=12:mem=384g:ngpus=2:Qlist=agpu:container_engine=singularity \
    -l walltime=72:00:00
```

### A6000 노드 사용 (coss_a6gpu 큐)
```bash
/opt/pbs/bin/qsub -I -q coss_a6gpu \
    -l select=1:ncpus=6:mem=128g:ngpus=1:Qlist=a6gpu:container_engine=singularity \
    -l walltime=72:00:00
```

---

## 5. 유용한 모니터링 명령어

```bash
# 내 잡 상태
/opt/pbs/bin/qstat -ans | grep $(whoami)

# 내 잡 상세 정보 (GPU 개수, 노드명)
/opt/pbs/bin/qstat -f <JOB_ID> | grep -E "Job_Owner|Resource_List|exec_vnode"

# 클러스터 전체 running 잡
/opt/pbs/bin/qstat -ans | grep " R "

# 특정 노드 상황
/opt/pbs/bin/qstat -ans | grep "ece-agpu8"

# GPU 노드에서: 현재 GPU 사용 프로세스
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

# GPU 노드에서: 유휴 GPU 찾기
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

---

## 6. PBS 잡 환경변수 파일에서 할당 GPU 확인

```bash
# GPU 노드에서 (잡 실행 중)
ls /var/spool/pbs/aux/
cat /var/spool/pbs/aux/<JOB_ID>.ECE-util1.env
# → CUDA_VISIBLE_DEVICES=GPU-xxxx... 확인 가능
```

---

## 7. 계정별 GPU 제한 & 전략

| 항목 | 값 |
|------|---|
| introai4 계정 최대 GPU/잡 | **2개** |
| 큐 | coss_agpu (agpu 노드), coss_a6gpu (A6000 노드) |
| walltime 최대 | 72시간 |
| 주의 | 할당 안 된 GPU 사용 시 `nvidia-smi`로 PID+사용자 다 보임 |

### 여러 계정으로 GPU 확보 전략

각 introai 계정은 독립적으로 잡 제출 가능 → 계정마다 GPU 2개씩 확보 가능:

```
introai4  → GPU 2개
introai12 → GPU 2개
introai13 → GPU 2개
introai15 → GPU 2개
─────────────────
합계        GPU 8개
```

**방법:** SSH로 각 계정에 접속 후 qsub 제출, 각 노드에서 tmux로 작업:

```bash
ssh introai12@147.46.121.38
/opt/pbs/bin/qsub -I -q coss_agpu -l select=1:ncpus=12:mem=384g:ngpus=2:Qlist=agpu:container_engine=singularity -l walltime=72:00:00
```

### 노드 내 추가 GPU 접근

PBS 잡으로 노드 접속 권한을 얻으면, 그 노드의 다른 GPU도 접근 가능:

```bash
# 내 할당: GPU 0 / 실제로 GPU 1,2,3도 비어있으면
CUDA_VISIBLE_DEVICES=1,2,3 python train.py
```

유휴 GPU 확인 (GPU 노드에서):
```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
# memory.used가 0 MiB = 사용 중인 프로세스 없음
```

---

## 8. SSH 키 설정 현황 (2026-03-26 완료)

### 등록된 키

| 키 | 위치 | 용도 |
|----|------|------|
| `~/.ssh/id_rsa` (RSA 4096) | introai4 서버 | 서버↔서버 계정 간 이동 |
| `id_ed25519` (맥북) | `hyeonseok@MacBookAir` | 맥북→서버 직접 접속 |

### SSH 키가 등록된 계정 (총 35개)

```
introai4~10  (PW: 1qaz2wsx!!)
introai11~33 (각자 PW)
kkheon, rintern07~10, rintern14
```

> 서버: `147.46.121.38` (ECE-util1), `147.46.121.39` (ECE-util2)

### 비밀번호 없이 접속 확인됨

```bash
ssh introai12@147.46.121.38   # → SSH OK as introai12 on ECE-util1
```

### ~/.ssh/config 설정

```
Host introai4~33, kkheon, rintern*
    HostName 147.46.121.38
    User %h
    IdentityFile ~/.ssh/id_rsa
    StrictHostKeyChecking no
    ServerAliveInterval 60
```

---

## 9. Claude Code 운영 방법

```bash
# 1. screen으로 Claude Code 세션 유지
screen -S claude
claude

# 2. 나갈 때 (Claude는 계속 살아있음)
Ctrl+A, D

# 3. 다시 붙기
screen -r claude
```

**핵심:** screen은 Claude Code 유지용, tmux는 Claude가 GPU 노드 작업 제어용

```bash
# Claude가 다른 계정 GPU 노드에서 작업 실행하는 흐름
ssh introai12@147.46.121.38          # 다른 계정 접속
qsub ...                              # 잡 제출 → GPU 노드 배정
ssh <gpu-node>                        # GPU 노드 진입
tmux new-session -s work             # tmux 세션 생성
# → Claude가 tmux send-keys로 명령 전달
```

---

## 10. vla conda 환경 현황

| 항목 | 버전/상태 |
|------|----------|
| Python | 3.10.18 |
| PyTorch | 2.6.0 |
| torchvision | 0.21.0 |
| CUDA (torch) | 12.4 (로그인 노드에서는 인식 안 됨) |
| robomimic | 0.3.0 |
| timm | 1.0.25 |
| open_clip_torch | 3.3.0 |
| einops | 0.8.2 |
| wandb | 0.15.12 |
| **transformers** | ❌ 미설치 |
| **diffusers** | ❌ 미설치 |
| **mujoco/robosuite** | ❌ 미설치 (robomimic만 있음) |

### vla 환경 활성화
```bash
conda activate vla

# GPU 노드에서 CUDA 인식 확인
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
