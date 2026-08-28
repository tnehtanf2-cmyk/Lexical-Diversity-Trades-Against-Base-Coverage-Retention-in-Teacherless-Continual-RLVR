#!/bin/bash
# Release note: paths are parameterized. Override with REPO_ROOT=/path ./scripts/<script>.sh
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# v2m — 엔트로피 계수 도즈-리스폰스 보강 (사전등록: wiki/방법론-사전등록-도즈리스폰스.md).
#
# 기존 스윕(sweep2)의 ent_lo / fixed_mid / ent_hi 는 entropy_coef 만 다르고(0 / 0.01 / 0.04)
# BASE·kl_beta·프로토콜이 동일하다 → 이미 3점 용량곡선. 여기서 0.02 와 0.08 두 수준을
# 추가해 5점(0, 0.01, 0.02, 0.04, 0.08) × 3시드로 만든다.
#
# 목적: 이질적 6조건 상관(로버스트니스 실패) 대신 단일 레버의 단조 용량-반응으로
#       "다양성을 올리는 레버가 base coverage 를 깎는가"를 직접 검정.
# 신규 6런 × ~2.3h ≈ 14h.
cd "$REPO_ROOT" || exit 1
L=logs/v2m_dose.log
REC=plan/실험기록.md
IMG="nvcr.io/nvidia/pytorch:26.03-py3"
PIP="pip install --quiet -U 'transformers==5.14.1' 'trl==1.9.2' 'datasets>=2.19' accelerate 'numpy<2' pandas sentencepiece"

# sweep2 와 완전히 동일한 프로토콜·BASE (한 글자도 바꾸지 않는다 — 기존 3점과 합쳐야 하므로)
COMMON="--order GSM8K Code MedicalMC --shift_interval 5 --max_cycles 15 \
  --n_train 20 --num_gen 6 --grad_accum 3 --grpo_steps 12 \
  --eval_subset 40 --eval_nsamples 16 --ks 1 4 16 --harvest_source train"
BASE="temperature=0.9,learning_rate=1.5e-5,weight_decay=0.02,rehearsal_ratio=0.2,difficulty=0.25,ref_reset=0"

echo "[$(date +'%F %T')] v2m dose-response start (entropy_coef 0.02, 0.08)" >> "$L"
echo "" >> "$REC"
echo "### [RUN] $(date '+%Y-%m-%d %H:%M') — v2m 엔트로피 도즈-리스폰스 보강 (0.02·0.08 × 3시드)" >> "$REC"

for lvl in "002|0.02" "008|0.08"; do
  IFS='|' read -r tag ec <<< "$lvl"
  for seed in 0 1 2; do
    RUN="sweep2_ent_${tag}_s${seed}"
    NAME="nsiac-${RUN//_/-}"
    if [ -f "logs/${RUN}.csv" ]; then
      echo "[$(date +'%F %T')] SKIP  $RUN (csv exists)" >> "$L"; continue
    fi
    docker rm "$NAME" 2>/dev/null || true
    CMD="python main_nsiac_v2.py --run_name ${RUN} --baseline fixed_grpo --seed ${seed} ${COMMON} \
      --fixed_knobs ${BASE},entropy_coef=${ec},kl_beta=0.04"
    echo "[$(date +'%F %T')] START $RUN (entropy_coef=$ec)" >> "$L"
    echo "-  [v2m] $CMD" | tr -s ' ' >> "$REC" 2>/dev/null || true
    docker run --rm --gpus all --shm-size=16g \
      -e HF_TOKEN="${HF_TOKEN:-}" \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      -v "$REPO_ROOT":/workspace -w /workspace \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      --name "$NAME" "$IMG" \
      bash -c "$PIP && $CMD" > "logs/${RUN}.log" 2>&1 \
      && echo "[$(date +'%F %T')] DONE  $RUN" >> "$L" \
      || echo "[$(date +'%F %T')] FAIL  $RUN (exit $?)" >> "$L"
  done
done
echo "[$(date +'%F %T')] v2m dose-response complete" >> "$L"
