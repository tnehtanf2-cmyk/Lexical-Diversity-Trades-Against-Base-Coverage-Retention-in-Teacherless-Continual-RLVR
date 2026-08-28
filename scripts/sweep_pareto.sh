#!/bin/bash
# Release note: paths are parameterized. Override with REPO_ROOT=/path ./scripts/<script>.sh
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# 헤드라인 실험: 2축 vs 1축 Pareto (RQ0). 정합 프로토콜·paired seed, 단일 GB10 직렬.
# 각 조건을 동일한 경량 연속 프로토콜로 실행하고 컨테이너 종료까지 대기(직렬).
# 사용: [SEEDS="0 1"] ./sweep_pareto.sh
set -e
IMG="nvcr.io/nvidia/pytorch:26.03-py3"
PIP="pip install --quiet -U 'transformers>=4.51' 'trl>=0.15' 'datasets>=2.19' accelerate 'numpy<2' pandas sentencepiece"
REC="plan/실험기록.md"
SEEDS="${SEEDS:-0 1 2}"
# 공통 경량 프로토콜 (모든 조건 동일)
COMMON="--order GSM8K Code MedicalMC --shift_interval 5 --max_cycles 15 \
  --n_train 20 --num_gen 6 --grad_accum 3 --grpo_steps 12 \
  --eval_subset 40 --eval_nsamples 16 --ks 1 4 16"
# 고정 노브 베이스(스윕 안 하는 축은 중립 고정). 조건별로 entropy_coef/kl_beta만 덮어씀.
BASE="temperature=0.9,learning_rate=1.5e-5,weight_decay=0.02,rehearsal_ratio=0.2,difficulty=0.25,ref_reset=0"

# base 포락선(제3축 기준선): 동결 베이스 eval-only, 도메인당 반복 draw(≥4)로 per-prompt
# 해결확률·CI 추정 가능(모든 스윕 조건과 동일 eval 프로토콜·subset → per-prompt 정렬).
# 가드는 done-센티널(부분/크래시 파일 존재만으로 스킵되는 버그 방지, PAT#3).
if [ ! -f "logs/sweep_base_env.done" ]; then
  echo "▶ [base envelope] 동결 베이스 pass@k 포락선 측정 (6 eval-only cycles)"
  rm -f logs/sweep_base_env.csv logs/sweep_base_env_meta.jsonl logs/sweep_base_env_samples.jsonl
  docker run --rm --gpus all --shm-size=16g \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v "$REPO_ROOT":/workspace -w /workspace \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    --name nsiac-sweep-base-env "$IMG" \
    bash -c "$PIP && python main_nsiac_v2.py --run_name sweep_base_env --baseline fixed_grpo \
      --eval_only --seed 0 $COMMON --max_cycles 6 --shift_interval 1" > logs/sweep_base_env.log 2>&1 \
    && touch logs/sweep_base_env.done
  echo "  ✓ base 포락선 완료"
fi

# 조건: "label|baseline|fixed_knobs(빈칸=학습컨트롤러)"
CONDS=(
  "nsiac|nsiac|"                                        # 학습 2축(다축)
  "ent_lo|fixed_grpo|entropy_coef=0.0,kl_beta=0.04"     # 1축(entropy) 저
  "ent_hi|fixed_grpo|entropy_coef=0.04,kl_beta=0.04"    # 1축(entropy) 고
  "kl_lo|fixed_grpo|entropy_coef=0.01,kl_beta=0.02"     # 1축(KL) 저
  "kl_hi|fixed_grpo|entropy_coef=0.01,kl_beta=0.10"     # 1축(KL) 고
  "fixed_mid|fixed_grpo|entropy_coef=0.01,kl_beta=0.04" # 고정 참조점
)

echo "### [SWEEP] $(date '+%Y-%m-%d %H:%M') — Pareto 2축vs1축 (seeds=$SEEDS)" >> "$REC" 2>/dev/null || true
for seed in $SEEDS; do
  for c in "${CONDS[@]}"; do
    IFS='|' read -r label bl fk <<< "$c"
    RUN="sweep_${label}_s${seed}"
    NAME="nsiac-sweep-${label}-s${seed}"
    if [ -f "logs/${RUN}.csv" ]; then          # 재실행 중복 append 방지 (PAT#9)
      echo "⏭  [$NAME] logs/${RUN}.csv 존재 → 스킵"
      continue
    fi
    FKARG=""
    [ -n "$fk" ] && FKARG="--fixed_knobs ${BASE},${fk}"
    docker rm "$NAME" 2>/dev/null || true
    CMD="python main_nsiac_v2.py --run_name ${RUN} --baseline ${bl} --seed ${seed} ${COMMON} ${FKARG}"
    echo "▶ [$NAME] seed=$seed  knobs=${fk:-learned}"
    echo "-  [$NAME] $CMD" | tr -s ' ' >> "$REC" 2>/dev/null || true
    docker run --rm --gpus all --shm-size=16g \
      -e HF_TOKEN="${HF_TOKEN:-}" \
      -v "$REPO_ROOT":/workspace -w /workspace \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      --name "$NAME" "$IMG" \
      bash -c "$PIP && $CMD" > "logs/${RUN}.log" 2>&1
    echo "  ✓ [$NAME] 완료 → logs/${RUN}.csv"
  done
done
echo "✅ 스윕 완료. Pareto 분석: python paper/make_results.py (Pareto 플롯 포함)"
