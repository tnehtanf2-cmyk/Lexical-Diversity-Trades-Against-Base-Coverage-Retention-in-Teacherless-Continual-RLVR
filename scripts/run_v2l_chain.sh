#!/bin/bash
# Release note: paths are parameterized. Override with REPO_ROOT=/path ./scripts/<script>.sh
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# v2l — 구 파이프라인(스윕·ZPD절제·3B·7B)의 **청정 재실행** 체인.
# 사용자 결정(2026-08-07): "재실행하고 결과를 보고 [버그 공개를] 넣을지 판단".
#
# 원 런과의 차이는 정확히 세 가지:
#   1) --harvest_source train  (eval-rollout 수확 오염 제거; v2j/v2k와 동일)
#   2) 수리된 MBPP 검증기      (nsiac2/domains.py code_prefix — 코드에 이미 반영)
#   3) 버전 핀                 (transformers==5.14.1, trl==1.9.2 — run_nsiac_v2i.sh와 동일;
#                               구 sweep_pareto.sh는 미핀 상태였음)
# 프로토콜·시드·노브 문자열은 원 런과 동일하게 유지한다(비교 가능성).
# 베이스 포락선(sweep_base_env, v2i_3b/7b_base_env)은 eval-only라 두 버그와 무관 → 재사용.
#
# 순서: 스윕 18런(~50h) → no_zpd(~8h) → 3B 3런(~40h) → 7B 1런(~30h). 총 ~5.5일.
# 3B 재시드(s1)는 청정 3B 결과 확인 후 필요 시 별도 실행.
cd "$REPO_ROOT" || exit 1
L=logs/v2l_chain.log
REC=plan/실험기록.md
IMG="nvcr.io/nvidia/pytorch:26.03-py3"
PIP="pip install --quiet -U 'transformers==5.14.1' 'trl==1.9.2' 'datasets>=2.19' accelerate 'numpy<2' pandas sentencepiece"
FIXED_MID="temperature=0.9,learning_rate=1.5e-5,weight_decay=0.02,rehearsal_ratio=0.2,difficulty=0.25,ref_reset=0,entropy_coef=0.01,kl_beta=0.04"

echo "[$(date +'%F %T')] v2l chain start (clean rerun: train harvest + repaired verifier + pinned deps)" >> "$L"
echo "" >> "$REC"
echo "### [RUN] $(date '+%Y-%m-%d %H:%M') — v2l 청정 재실행 체인 착수 (스윕 18 + no_zpd + 3B 3 + 7B 1)" >> "$REC"

# ---------- Stage 1: 스윕 18런 (sweep_pareto.sh 프로토콜 동일, 이름 sweep2_*) ----------
COMMON="--order GSM8K Code MedicalMC --shift_interval 5 --max_cycles 15 \
  --n_train 20 --num_gen 6 --grad_accum 3 --grpo_steps 12 \
  --eval_subset 40 --eval_nsamples 16 --ks 1 4 16 --harvest_source train"
BASE="temperature=0.9,learning_rate=1.5e-5,weight_decay=0.02,rehearsal_ratio=0.2,difficulty=0.25,ref_reset=0"
CONDS=(
  "nsiac|nsiac|"
  "ent_lo|fixed_grpo|entropy_coef=0.0,kl_beta=0.04"
  "ent_hi|fixed_grpo|entropy_coef=0.04,kl_beta=0.04"
  "kl_lo|fixed_grpo|entropy_coef=0.01,kl_beta=0.02"
  "kl_hi|fixed_grpo|entropy_coef=0.01,kl_beta=0.10"
  "fixed_mid|fixed_grpo|entropy_coef=0.01,kl_beta=0.04"
)
for seed in 0 1 2; do
  for c in "${CONDS[@]}"; do
    IFS='|' read -r label bl fk <<< "$c"
    RUN="sweep2_${label}_s${seed}"
    NAME="nsiac-${RUN//_/-}"
    if [ -f "logs/${RUN}.csv" ]; then
      echo "[$(date +'%F %T')] SKIP  $RUN (csv exists)" >> "$L"; continue
    fi
    FKARG=""
    [ -n "$fk" ] && FKARG="--fixed_knobs ${BASE},${fk}"
    docker rm "$NAME" 2>/dev/null || true
    CMD="python main_nsiac_v2.py --run_name ${RUN} --baseline ${bl} --seed ${seed} ${COMMON} ${FKARG}"
    echo "[$(date +'%F %T')] START $RUN" >> "$L"
    echo "-  [v2l] $CMD" | tr -s ' ' >> "$REC" 2>/dev/null || true
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
echo "[$(date +'%F %T')] stage1 sweep complete" >> "$L"

# ---------- Stage 2–4: 절제·사다리 (run_nsiac_v2i.sh 재사용, 가드·회전·기록 상속) ----------
run () {   # $1=run_name $2=preset $3=baseline $4=model $5..=extra
  local name="$1"; local preset="$2"; local bl="$3"; local model="$4"; shift 4
  local cname="nsiac-${name//_/-}"
  if [ -f "logs/${name}.csv" ]; then
    echo "[$(date +'%F %T')] SKIP  $name (csv exists)" >> "$L"; return 0
  fi
  echo "[$(date +'%F %T')] START $name" >> "$L"
  ./run_nsiac_v2i.sh "$name" "$preset" "$bl" "$model" --harvest_source train "$@" >> "$L" 2>&1
  for _ in $(seq 1 60); do docker ps -q -f "name=^${cname}$" | grep -q . && break; sleep 10; done
  docker wait "$cname" >> "$L" 2>&1
  echo "[$(date +'%F %T')] DONE  $name" >> "$L"
  return 0
}

run v2l_nozpd    full3 nsiac      Qwen/Qwen2.5-1.5B --no_zpd
run v2l_3b_nsiac    full3 nsiac      Qwen/Qwen2.5-3B
run v2l_3b_fixedmid full3 fixed_grpo Qwen/Qwen2.5-3B --fixed_knobs "$FIXED_MID"
run v2l_3b_sftstar  full3 sft_star   Qwen/Qwen2.5-3B
run v2l_7b_nsiac    full3 nsiac      Qwen/Qwen2.5-7B
echo "[$(date +'%F %T')] v2l chain complete" >> "$L"
