#!/bin/bash
# Release note: paths are parameterized. Override with REPO_ROOT=/path ./scripts/<script>.sh
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# N-SIAC v2i — scale ladder (1.5B/3B/7B) + 4-domain (FinQA) runs.
# run_nsiac_v2.sh (the v2h reproduction launcher) is intentionally left untouched.
#
# 사용: ./run_nsiac_v2i.sh <RUN_NAME> <PRESET> <BASELINE> <MODEL> [EXTRA ARGS...]
#   PRESET: smoke2 | smoke4 | full3 | full4 | env3 | env4
#   MODEL : Qwen/Qwen2.5-1.5B | Qwen/Qwen2.5-3B | Qwen/Qwen2.5-7B
# 예)
#   ./run_nsiac_v2i.sh v2i_finqa_smoke smoke4 nsiac Qwen/Qwen2.5-1.5B
#   ./run_nsiac_v2i.sh v2i_4dom_nsiac  full4  nsiac Qwen/Qwen2.5-1.5B
#   ./run_nsiac_v2i.sh v2i_3b_fixedmid full3  fixed_grpo Qwen/Qwen2.5-3B --fixed_knobs "$FIXED_MID"
# HF_TOKEN 은 셸 환경변수로 주입(하드코딩 금지). 실행 시 plan/실험기록.md 에 [RUN] 자동 기록.
set -e
RUN="${1:?run_name required}"; PRESET="${2:?preset required}"
BL="${3:-nsiac}"; MODEL="${4:-Qwen/Qwen2.5-1.5B}"; shift 4 2>/dev/null || shift $#
EXTRA="$*"
IMG="nvcr.io/nvidia/pytorch:26.03-py3"
NAME="nsiac-${RUN//_/-}"
REC="plan/실험기록.md"
# bitsandbytes: optional 8-bit optimizer fallback for 7B (graceful no-op if the
# aarch64 wheel is unavailable — main_nsiac_v2.py falls back to adamw_torch).
# Versions PINNED for the ladder. The published 1.5B (v2h) runs resolved to
# transformers 5.14.1 + trl 1.9.0; today's floating spec resolves to trl 1.9.2, whose
# grpo_trainer.py is byte-identical to 1.9.0 (verified by full-file diff), so the pin is
# set to the verified pair. Without a pin, a release landing mid-ladder would silently
# change the trainer between scales and confound the comparison.
PIP="pip install --quiet -U 'transformers==5.14.1' 'trl==1.9.2' 'datasets>=2.19' accelerate 'numpy<2' pandas sentencepiece"
case "$MODEL" in *7B*) PIP="$PIP; pip install --quiet bitsandbytes || true";; esac

# The v2h fixed mid-point knob string (unchanged across scales on purpose: whether the
# 1.5B operating-point structure TRANSFERS is the question, so knobs are not retuned).
FIXED_MID="temperature=0.9,learning_rate=1.5e-5,weight_decay=0.02,rehearsal_ratio=0.2,difficulty=0.25,ref_reset=0,entropy_coef=0.01,kl_beta=0.04"
export FIXED_MID

D3="GSM8K Code MedicalMC"
D4="GSM8K Code MedicalMC FinQA"
case "$PRESET" in
  smoke2) ARGS="--order GSM8K MedicalMC --shift_interval 2 --max_cycles 4 \
                --n_train 16 --num_gen 4 --grad_accum 2 --grpo_steps 5 \
                --eval_subset 20 --eval_nsamples 8 --ks 1 4 8";;
  smoke4) ARGS="--order GSM8K FinQA --shift_interval 2 --max_cycles 4 \
                --n_train 16 --num_gen 4 --grad_accum 2 --grpo_steps 5 \
                --eval_subset 20 --eval_nsamples 8 --ks 1 4 8";;
  full3)  ARGS="--order $D3 --shift_interval 10 --max_cycles 30 \
                --n_train 64 --num_gen 8 --grad_accum 4 --grpo_steps 20 \
                --eval_subset 50 --eval_nsamples 16 --ks 1 4 16 --seed 0";;
  full4)  ARGS="--order $D4 --shift_interval 10 --max_cycles 40 \
                --n_train 64 --num_gen 8 --grad_accum 4 --grpo_steps 20 \
                --eval_subset 50 --eval_nsamples 16 --ks 1 4 16 --seed 0";;
  env3)   ARGS="--eval_only --order $D3 --shift_interval 1 --max_cycles 6 \
                --n_train 64 --num_gen 8 --grad_accum 4 --grpo_steps 20 \
                --eval_subset 50 --eval_nsamples 16 --ks 1 4 16 --seed 0";;
  env4)   ARGS="--eval_only --order $D4 --shift_interval 1 --max_cycles 8 \
                --n_train 64 --num_gen 8 --grad_accum 4 --grpo_steps 20 \
                --eval_subset 50 --eval_nsamples 16 --ks 1 4 16 --seed 0";;
  *) echo "unknown preset: $PRESET"; exit 1;;
esac

CMD="python main_nsiac_v2.py --run_name ${RUN} --baseline ${BL} --model_name ${MODEL} ${ARGS} ${EXTRA}"

# ---- 보호 가드 (2026-08-04 추가) ----------------------------------------
# 사고: 같은 프로젝트에서 다른 세션이 동일 RUN_NAME 으로 이 런처를 호출하면, 아래
# docker stop/rm 이 **실행 중이던 런을 죽인다**. 실제로 v2j_noreh 가 c27 에서
# exit 137 로 사망했다(2026-08-04 06:18). 실행 중인 컨테이너는 절대 건드리지 않는다.
if [ -n "$(docker ps -q -f "name=^${NAME}$")" ]; then
  echo "❌ [$NAME] 이미 실행 중 — 기동을 거부합니다 (기존 런 보호)."
  echo "   진행 상황: docker logs -f $NAME   |   중단하려면 수동으로: docker stop $NAME"
  {
    echo ""
    echo "### ⚠️ [RUN-거부] $(date '+%Y-%m-%d %H:%M') — ${NAME} 는 이미 실행 중이어서 중복 기동을 거부함"
  } >> "$REC" 2>/dev/null || true
  exit 1
fi
# 결과 파일이 남아 있으면 append 되어 두 시도가 한 파일에 섞인다(분석 시 치명적).
# 실행 중이 아닌 것이 위에서 확인됐으므로 여기서 회전시킨다.
for ext in csv _meta.jsonl _samples.jsonl _mem.csv; do
  f="logs/${RUN}${ext}"; [ "$ext" = "csv" ] && f="logs/${RUN}.csv"
  if [ -f "$f" ]; then
    n=1; while [ -e "${f%.*}_part${n}.${f##*.}" ]; do n=$((n+1)); done
    mv "$f" "${f%.*}_part${n}.${f##*.}"
    echo "↩︎  기존 결과 회전: $f → ${f%.*}_part${n}.${f##*.}"
  fi
done
docker rm "$NAME" 2>/dev/null || true
# -------------------------------------------------------------------------

{
  echo ""
  echo "### [RUN] $(date '+%Y-%m-%d %H:%M') — ${NAME} (preset=${PRESET}, baseline=${BL}, model=${MODEL})"
  echo '```'
  echo "$CMD" | tr -s ' '
  echo '```'
} >> "$REC" 2>/dev/null || true

mkdir -p "$HOME/.cache/huggingface"
docker run -d --gpus all --shm-size=16g \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$REPO_ROOT":/workspace -w /workspace \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --name "$NAME" "$IMG" \
  bash -c "$PIP && $CMD"

echo "✅ [$NAME] 시작. (plan/실험기록.md 에 [RUN] 기록됨)"
echo "   로그:   docker logs -f $NAME"
echo "   결과:   logs/${RUN}.csv   |   Q&A: logs/${RUN}_samples.jsonl"
