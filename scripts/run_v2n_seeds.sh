#!/bin/bash
# Release note: paths are parameterized. Override with REPO_ROOT=/path ./scripts/<script>.sh
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# v2n — Multi-Seed 확장 (사전등록: wiki/방법론-사전등록-멀티시드.md).
# Table 2(1.5B 4조건)·Table 8(3B 2조건·7B nsiac)에 시드 1,2 추가 → 기존 시드 0과 합쳐 3시드.
# 목적: "오차 범위 겹침 → 비지배" 실측 방어. 전부 청정 파이프라인. 14런 ≈ 7일.
# full3 프리셋의 --seed 0 뒤에 --seed N을 덧붙이면 argparse가 마지막 값을 취한다(v2j_s1에서 검증된 패턴).
cd "$REPO_ROOT" || exit 1
L=logs/v2n_seeds.log
FIXED_MID="temperature=0.9,learning_rate=1.5e-5,weight_decay=0.02,rehearsal_ratio=0.2,difficulty=0.25,ref_reset=0,entropy_coef=0.01,kl_beta=0.04"

echo "[$(date +'%F %T')] v2n multi-seed chain start (14 runs)" >> "$L"

run () {   # $1=run_name $2=baseline $3=model $4=seed $5..=extra
  local name="$1" bl="$2" model="$3" seed="$4"; shift 4
  local cname="nsiac-${name//_/-}"
  if [ -f "logs/${name}.csv" ]; then
    echo "[$(date +'%F %T')] SKIP  $name (csv exists)" >> "$L"; return 0
  fi
  echo "[$(date +'%F %T')] START $name" >> "$L"
  ./run_nsiac_v2i.sh "$name" full3 "$bl" "$model" --harvest_source train --seed "$seed" "$@" >> "$L" 2>&1
  for _ in $(seq 1 60); do docker ps -q -f "name=^${cname}$" | grep -q . && break; sleep 10; done
  docker wait "$cname" >> "$L" 2>&1
  echo "[$(date +'%F %T')] DONE  $name" >> "$L"
  return 0
}

# Stage 1: 1.5B 4조건 × 시드 1,2 (핵심 — Table 2)
for s in 1 2; do
  run "v2n_nsiac_s${s}"    nsiac      Qwen/Qwen2.5-1.5B "$s"
  run "v2n_fixedmid_s${s}" fixed_grpo Qwen/Qwen2.5-1.5B "$s" --fixed_knobs "$FIXED_MID"
  run "v2n_sftstar_s${s}"  sft_star   Qwen/Qwen2.5-1.5B "$s"
  run "v2n_noreh_s${s}"    nsiac      Qwen/Qwen2.5-1.5B "$s" --no_rehearsal
done
echo "[$(date +'%F %T')] stage1 (1.5B x8) complete" >> "$L"

# Stage 2: 3B 2조건 × 시드 1,2
for s in 1 2; do
  run "v2n_3b_nsiac_s${s}"    nsiac      Qwen/Qwen2.5-3B "$s"
  run "v2n_3b_fixedmid_s${s}" fixed_grpo Qwen/Qwen2.5-3B "$s" --fixed_knobs "$FIXED_MID"
done
echo "[$(date +'%F %T')] stage2 (3B x4) complete" >> "$L"

# Stage 3: 7B nsiac × 시드 1,2 (앵커링 주장 재현)
for s in 1 2; do
  run "v2n_7b_nsiac_s${s}" nsiac Qwen/Qwen2.5-7B "$s"
done
echo "[$(date +'%F %T')] v2n multi-seed chain complete" >> "$L"
