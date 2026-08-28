#!/bin/bash
# Release note: paths are parameterized. Override with REPO_ROOT=/path ./scripts/<script>.sh
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# v2k — 코드 검증기 수정 후 재실험.
# v2j 와 **동일 설정**(full3, 1.5B, --harvest_source train)이고 달라진 것은 오직
# nsiac2/{domains,reward_arbiter}.py 의 code_prefix 수정뿐이다. 따라서 v2j→v2k 차이는
# "MBPP 학습 보상이 살아난 효과"로 해석된다.
#
# 배경(2026-08-05 PAT): code_correct 가 자연어 MBPP 프롬프트를 파이썬 파일 앞에 붙여
# SyntaxError 를 냈고, 전 런에서 rewards/code_fn/mean = 0 이었다. 그 결과 Code 블록은
# advantage=0 이라 학습이 없었고, TrainHarvest 는 r>0 만 저장하므로 v2j 의
# episodic['Code'] 는 끝까지 비어 있었다. 스모크(v2k_smoke_code)에서 0.125 로 회복 확인.
#
# 순차 실행: nsiac → fixedmid → noreh → sftstar (각 ~8h, 총 ~33h)
cd "$REPO_ROOT" || exit 1
L=logs/v2k_chain.log
FIXED_MID="temperature=0.9,learning_rate=1.5e-5,weight_decay=0.02,rehearsal_ratio=0.2,difficulty=0.25,ref_reset=0,entropy_coef=0.01,kl_beta=0.04"

echo "[$(date +'%F %T')] v2k chain start (code verifier fixed)" >> "$L"

run () {   # $1=run_name  $2=baseline  $3..=extra args
  local name="$1" bl="$2"; shift 2
  local cname="nsiac-${name//_/-}"
  echo "[$(date +'%F %T')] START $name" >> "$L"
  ./run_nsiac_v2i.sh "$name" full3 "$bl" Qwen/Qwen2.5-1.5B --harvest_source train "$@" >> "$L" 2>&1
  # 컨테이너가 생길 때까지 기다린 뒤 종료 대기 — 고정 sleep 은 경합으로 동시 기동을 부른다
  for _ in $(seq 1 60); do docker ps -q -f "name=^${cname}$" | grep -q . && break; sleep 10; done
  docker wait "$cname" >> "$L" 2>&1
  echo "[$(date +'%F %T')] DONE  $name" >> "$L"
  return 0
}

run v2k_nsiac    nsiac
run v2k_fixedmid fixed_grpo --fixed_knobs "$FIXED_MID"
run v2k_noreh    nsiac      --no_rehearsal
run v2k_sftstar  sft_star
echo "[$(date +'%F %T')] v2k chain complete" >> "$L"
