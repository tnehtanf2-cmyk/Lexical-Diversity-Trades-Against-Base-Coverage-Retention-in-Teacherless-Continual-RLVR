#!/bin/bash
# Release note: paths are parameterized. Override with REPO_ROOT=/path ./scripts/<script>.sh
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# v2j 시드 보강 — 본표의 최대 약점(조건당 단일 시드)을 겨냥.
# v2j 와 동일 설정(full3, 1.5B, --harvest_source train)에 --seed 1 만 추가.
# EXTRA 는 CMD 끝에 붙으므로 full3 프리셋의 --seed 0 을 argparse 가 덮어쓴다.
# 순차 실행: nsiac → fixedmid (각 ~8h). 이 둘의 순위가 시드 1에서도 유지되는지가 질문.
cd "$REPO_ROOT" || exit 1
L=logs/v2j_seed1_chain.log
FIXED_MID="temperature=0.9,learning_rate=1.5e-5,weight_decay=0.02,rehearsal_ratio=0.2,difficulty=0.25,ref_reset=0,entropy_coef=0.01,kl_beta=0.04"

echo "[$(date +'%F %T')] v2j seed1 chain start" >> "$L"

run () {   # $1=run_name  $2=baseline  $3..=extra args
  local name="$1" bl="$2"; shift 2
  local cname="nsiac-${name//_/-}"
  echo "[$(date +'%F %T')] START $name" >> "$L"
  ./run_nsiac_v2i.sh "$name" full3 "$bl" Qwen/Qwen2.5-1.5B \
      --harvest_source train --seed 1 "$@" >> "$L" 2>&1
  # 컨테이너가 뜰 때까지 기다린 뒤 종료 대기 (고정 sleep 은 경합으로 동시 기동을 부른다)
  for _ in $(seq 1 60); do docker ps -q -f "name=^${cname}$" | grep -q . && break; sleep 10; done
  docker wait "$cname" >> "$L" 2>&1
  echo "[$(date +'%F %T')] DONE  $name" >> "$L"
  return 0
}

run v2j_nsiac_s1    nsiac
run v2j_fixedmid_s1 fixed_grpo --fixed_knobs "$FIXED_MID"
echo "[$(date +'%F %T')] v2j seed1 chain complete" >> "$L"
