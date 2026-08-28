#!/bin/bash
# Release note: paths are parameterized. Override with REPO_ROOT=/path ./scripts/<script>.sh
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# v2j — 오염제거(train-split harvest) 재현 체인.
# v2h와 동일 설정(full3, Qwen2.5-1.5B) + --harvest_source train 만 추가.
# 목적: 리허설이 '측정된 eval 롤아웃'을 재생하며 생긴 오염(2026-08-03 축자재현 실측:
#       도메인 이탈 직후 29%→67%, 무리허설 대조군은 점프 없음)을 제거한 조건에서
#       헤드라인 결론(습득 순서·리허설 효과·붕괴 대조)이 유지되는지 판정.
# 순차 실행: nsiac → fixedmid → noreh → sftstar (각 ~8h, 총 ~33h)
cd "$REPO_ROOT" || exit 1
L=logs/v2j_chain.log

# run_nsiac_v2i.sh 내부 정의와 동일해야 함 (v2h 고정 중점 노브)
FIXED_MID="temperature=0.9,learning_rate=1.5e-5,weight_decay=0.02,rehearsal_ratio=0.2,difficulty=0.25,ref_reset=0,entropy_coef=0.01,kl_beta=0.04"

echo "[$(date +'%F %T')] v2j chain start" >> "$L"

# run_nsiac_v2i.sh 는 `docker run -d`(detached)라 즉시 반환한다.
# 순차 실행을 강제하려면 컨테이너가 끝날 때까지 기다려야 한다 — 기다리지 않으면
# 4개 런이 한 GPU에 동시에 올라가 메모리 경합/OOM이 난다(2026-08-03 실제 발생).
run () {   # $1=run_name  $2=baseline  $3..=extra args
  local name="$1" bl="$2"; shift 2
  local cname="nsiac-${name//_/-}"
  echo "[$(date +'%F %T')] START $name" >> "$L"
  ./run_nsiac_v2i.sh "$name" full3 "$bl" Qwen/Qwen2.5-1.5B --harvest_source train "$@" >> "$L" 2>&1
  # 컨테이너가 '생길 때까지' 먼저 기다린다. 고정 sleep 20 으로는 런처가 늦게 뜨는 경우
  # docker wait 가 "no such container" 로 즉시 실패해 다음 런을 동시 기동시킨다
  # (2026-08-03 v2j_fixedmid 에서 실제 발생: START 3분 뒤 DONE → noreh 와 15시간 동시 실행 → OOM kill 137).
  for _ in $(seq 1 60); do docker ps -q -f "name=^${cname}$" | grep -q . && break; sleep 10; done
  docker wait "$cname" >> "$L" 2>&1     # 컨테이너 종료까지 블록
  echo "[$(date +'%F %T')] DONE  $name" >> "$L"
  return 0    # 한 런이 실패해도 체인은 계속 — 실패는 로그로 판정
}

# v2j_nsiac 은 이미 실행 중이므로 여기서 완료만 기다린다.
echo "[$(date +'%F %T')] WAIT v2j_nsiac (already running)" >> "$L"
docker wait nsiac-v2j-nsiac >> "$L" 2>&1
echo "[$(date +'%F %T')] DONE  v2j_nsiac" >> "$L"
run v2j_fixedmid fixed_grpo --fixed_knobs "$FIXED_MID"
run v2j_noreh    nsiac      --no_rehearsal
run v2j_sftstar  sft_star
echo "[$(date +'%F %T')] v2j chain complete" >> "$L"
