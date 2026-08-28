#!/bin/bash
# Release note: paths are parameterized. Override with REPO_ROOT=/path ./scripts/<script>.sh
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# N-SIAC v2 (Full RLVR / GRPO / Qwen2.5-1.5B dense Full FT) — docker 실행
# 사용: ./run_nsiac_v2.sh [smoke|full] [nsiac|fixed_grpo|sft_star] [TAG]
#   TAG(선택): 재실험 구분용 접미사. 예) ./run_nsiac_v2.sh full nsiac v2b
# HF_TOKEN 은 셸 환경변수로 주입(하드코딩 금지): export HF_TOKEN=hf_xxx
# 실행 시 plan/실험기록.md 에 [RUN] 항목을 자동 기록한다.
set -e
MODE="${1:-smoke}"; BL="${2:-nsiac}"; TAG="${3:-}"
IMG="nvcr.io/nvidia/pytorch:26.03-py3"
SUF="${TAG:+_$TAG}"
RUN="v2_${MODE}_${BL}${SUF}"
NAME="nsiac-v2-${MODE}-${BL}${TAG:+-$TAG}"
REC="plan/실험기록.md"
PIP="pip install --quiet -U 'transformers>=4.51' 'trl>=0.15' 'datasets>=2.19' accelerate 'numpy<2' pandas sentencepiece"

# ---- 보호 가드 (2026-08-04 추가; run_nsiac_v2i.sh 와 동일 취지) --------
# 다른 세션이 같은 RUN_NAME 으로 호출하면 아래 stop/rm 이 실행 중이던 런을 죽인다.
# 이 런처는 **발표된 1.5B(v2h) 재현용**이라 특히 보호가 중요하다.
if [ -n "$(docker ps -q -f "name=^${NAME}$")" ]; then
  echo "❌ [$NAME] 이미 실행 중 — 기동을 거부합니다 (기존 런 보호)."
  exit 1
fi
docker rm "$NAME" 2>/dev/null || true
# -------------------------------------------------------------------------

if [ "$MODE" = "smoke" ]; then
  CMD="python main_nsiac_v2.py --run_name ${RUN} --baseline ${BL} \
       --order GSM8K MedicalMC --shift_interval 2 --max_cycles 4 \
       --n_train 16 --num_gen 4 --grad_accum 2 --grpo_steps 5 \
       --eval_subset 20 --eval_nsamples 8 --ks 1 4 8"
else
  CMD="python main_nsiac_v2.py --run_name ${RUN} --baseline ${BL} \
       --order GSM8K Code MedicalMC --shift_interval 10 --max_cycles 30 \
       --n_train 64 --num_gen 8 --grad_accum 4 --grpo_steps 20 \
       --eval_subset 50 --eval_nsamples 16 --ks 1 4 16"
fi

# --- 자동 기록: [RUN] 항목 (실험 수정/실행 때마다 항상 남김) ---
{
  echo ""
  echo "### [RUN] $(date '+%Y-%m-%d %H:%M') — ${NAME} (mode=${MODE}, baseline=${BL}, tag=${TAG:-none})"
  echo '```'
  echo "$CMD" | tr -s ' '
  echo '```'
} >> "$REC" 2>/dev/null || true

mkdir -p "$HOME/.cache/huggingface"
docker run -d --gpus all --shm-size=16g \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$REPO_ROOT":/workspace -w /workspace \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --name "$NAME" "$IMG" \
  bash -c "$PIP && $CMD"

echo "✅ [$NAME] 시작. (plan/실험기록.md 에 [RUN] 기록됨)"
echo "   로그:   docker logs -f $NAME"
echo "   결과:   logs/${RUN}.csv   |   Q&A: logs/${RUN}_samples.jsonl"
