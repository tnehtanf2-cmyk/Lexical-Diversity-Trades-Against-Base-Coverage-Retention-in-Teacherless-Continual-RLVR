#!/bin/bash
# Release note: paths are parameterized. Override with REPO_ROOT=/path ./scripts/<script>.sh
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# v2o — 수확 소스 절제 (사전등록: wiki/방법론-사전등록-수확절제.md, C안).
# 수리된 검증기 + eval 수확 = v2k(train 수확)와 "수확 소스만" 다른 단일변수 대조군.
# 4런 × ~8h ≈ 33h. v2n 체인 완주 후 착수할 것 (GPU 직렬).
cd "$REPO_ROOT" || exit 1
L=logs/v2o_harvest.log
FIXED_MID="temperature=0.9,learning_rate=1.5e-5,weight_decay=0.02,rehearsal_ratio=0.2,difficulty=0.25,ref_reset=0,entropy_coef=0.01,kl_beta=0.04"

echo "[$(date +'%F %T')] v2o harvest-ablation start (eval harvest + repaired verifier)" >> "$L"

run () {   # $1=run_name $2=baseline $3..=extra
  local name="$1" bl="$2"; shift 2
  local cname="nsiac-${name//_/-}"
  if [ -f "logs/${name}.csv" ]; then
    echo "[$(date +'%F %T')] SKIP  $name (csv exists)" >> "$L"; return 0
  fi
  echo "[$(date +'%F %T')] START $name" >> "$L"
  ./run_nsiac_v2i.sh "$name" full3 "$bl" Qwen/Qwen2.5-1.5B --harvest_source eval "$@" >> "$L" 2>&1
  for _ in $(seq 1 60); do docker ps -q -f "name=^${cname}$" | grep -q . && break; sleep 10; done
  docker wait "$cname" >> "$L" 2>&1
  echo "[$(date +'%F %T')] DONE  $name" >> "$L"
  return 0
}

run v2o_nsiac    nsiac
run v2o_fixedmid fixed_grpo --fixed_knobs "$FIXED_MID"
run v2o_sftstar  sft_star
run v2o_noreh    nsiac      --no_rehearsal
echo "[$(date +'%F %T')] v2o harvest-ablation complete" >> "$L"
