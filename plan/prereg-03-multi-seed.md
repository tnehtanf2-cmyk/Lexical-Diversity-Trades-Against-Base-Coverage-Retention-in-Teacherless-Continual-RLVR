# Preregistration — multi-seed replication

- Registered: 2026-08-14, before the fourteen added-seed runs were executed
- Adjudicated: 2026-08-20, after all fourteen completed
- Corrected: 2026-08-21 (H3 metric), 2026-08-25 (H4 metric) — both corrections kept below
- Manuscript: Sec. 4.4 (registration announced), RQ3 (Sec. 5.5), RQ5 (Sec. 5.7), Table 5

---

## English

### Purpose

To remove the single-seed limitation of the 30-cycle main table and the scale table.
**Not to establish which method wins**, but to defend the paper's central conclusion —
that the error ranges overlap and no strategy Pareto-dominates — with measurements
rather than an estimated noise floor of 0.05.

### Design (14 runs, about 7 days)

- **1.5B**: nsiac / fixedmid / sftstar / noreh x seeds 1, 2 (8 runs), combined with the
  existing seed 0 for three seeds.
- **3B**: nsiac / fixedmid x seeds 1, 2 (4 runs), combined with the existing seed 0.
  sftstar excluded: the collapse contrast is a large margin and does not need seeds.
  Recorded as a limitation.
- **7B**: nsiac x seeds 1, 2 (2 runs), to check whether the anchoring claim (10/10
  negative cycles) replicates across seeds. fixedmid and sftstar excluded on cost;
  the limitation is kept.
- All on the current pipeline, identical protocol, run names `v2n_*`.

### Predictions (fixed before the runs)

- **H1 (central)**: At 1.5B the per-seed `avg_final` **ranges of nsiac and fixedmid
  overlap**, completing the no-dominance defence. Same at 3B.
- **H2**: The direction (controller ahead on average) survives in the seed means, but
  within the overlapping range.
- **H3**: The 7B code-block negative BWT (10/10 cycles at seed 0) replicates in the two
  new seeds (>= 8/10 negative cycles per seed).
- **H4**: The STaR/SFT collapse (lowest distinct-3 and lowest average accuracy)
  replicates in every seed, the margin being large.

### Decision rule (fixed in advance)

| Outcome | Treatment in the paper |
|---|---|
| H1 holds (overlap) | Replace the "sub-noise" estimate with measured overlap; no-dominance complete. Print per-seed values and state the limits of n = 3 |
| H1 fails (one side separates 3/3) | **That is the result** — promote the directional claim to a resolved difference, whichever way it goes, and rewrite the relevant section |
| H3 fails | Keep the anchoring claim demoted to a single-seed observation |

Reporting rule: no Gaussian-style +/- notation. **Print per-seed values**, same
convention as the sweep table. The n = 3 standard deviation is for reference only.

### Outcome (2026-08-20, all fourteen runs plus the seven seed-0 runs)

| Hypothesis | Result | Verdict |
|---|---|---|
| **H1 overlap** | 1.5B nsiac [0.4329, 0.4804] vs fixedmid [0.4200, 0.4492] overlap; 3B [0.4629, 0.5187] vs [0.4400, 0.5125] overlap | **Holds** — replace the sub-noise estimate with measured overlap |
| **H2 direction** | Seed means favour nsiac by +0.0216 at 1.5B and +0.0144 at 3B, inside the overlap | **Holds** |
| **H3 7B anchoring** | Negative cycles: seed 0 **10/10**, seed 1 **1/10** (code-block BWT +0.0409), seed 2 **1/10** (+0.0480) | **Fails** — by the registered rule, demoted to a single-seed observation |
| **H4 STaR collapse** | See the 2026-08-25 correction below | **Partial** — the distinct-3 conjunct holds in all three seeds, the lowest-average-accuracy conjunct fails in one |

Per-seed `avg_final` (seeds 0/1/2): 1.5B nsiac 0.4433 / 0.4329 / 0.4804; fixedmid
0.4225 / 0.4200 / 0.4492; sftstar 0.3517 / 0.4183 / 0.4100; noreh 0.4166 / 0.3962 /
0.4250. 3B nsiac 0.5025 / 0.5187 / 0.4629; fixedmid 0.4883 / 0.4400 / 0.5125.

### Correction 1 (2026-08-21) — H3 computed on the wrong quantity

The first tally computed H3 as post-shift Code drift (c21–30 vs c20). The quantity the
manuscript's claim rests on is the **code-block BWT column** (GSM8K anchored at c10),
as defined in `make_results_v2k.py`. Recomputed: seed 0 10/10 negative, mean -0.0517
(matching the paper's -0.052); seed 1 1/10, +0.0409; seed 2 1/10, +0.0480. **The
verdict is unchanged — H3 still fails** — only the numbers are corrected. The two new
7B seeds show the same positive block BWT as 1.5B and 3B; the 7B negativity is a
seed-0 observation.

### Correction 2 (2026-08-25) — H4 distinct-3 conjunct wrongly recorded as failing

The verdict table recorded the H4 distinct-3 conjunct as holding at seed 0 only. That
was an error. Recomputation on every slice (steady-state mean, final cycle, whole-run
mean) gives sftstar distinct-3 = 0.3134 / 0.3569 / 0.2545 for seeds 0/1/2, against a
lowest GRPO value of 0.6826 / 0.7394 / 0.7250 in the respective seeds — **lowest by a
large margin in all three seeds, so the conjunct holds**. The "0.3933" in the original
table matches no reproducible slice (steady-state seed mean 0.3083, final-cycle mean
0.1798, whole-run mean 0.4186). H4's failing conjunct is the lowest-average-accuracy
one only (seed 1). The manuscript's RQ3 wording (Sec. 5.5) — diversity collapse held in all three
seeds, lowest-average-accuracy failed in one — is the accurate statement.

Lesson recorded with the correction: **a verdict entry must name the metric it used**
(aggregation window and statistic). Not naming it caused this error.

---

## 원문 (Korean original)

### 목적
Table 2(30사이클 메인)·Table 8(스케일)의 단일시드 한계 해소. **우열을 가리기 위함이 아니라** "오차 범위가 겹쳐 Pareto-dominance가 없다"는 핵심 결론을 실측으로 방어하기 위함. 현재는 추정 노이즈 하한(0.05)에 의존.

### 설계 (14런, ~7일)
- **1.5B**: nsiac / fixedmid / sftstar / noreh × 시드 1,2 (8런) — 기존 시드 0과 합쳐 3시드.
- **3B**: nsiac / fixedmid × 시드 1,2 (4런) — 기존 시드 0과 합쳐 3시드. sftstar는 제외(붕괴 대조는 대마진이라 시드 불요 — 한계로 명시).
- **7B**: nsiac × 시드 1,2 (2런) — 앵커링 주장(10/10 사이클 음수)의 시드 간 재현 확인. fixedmid/sftstar는 비용상 제외(한계 유지).
- 전부 현행 파이프라인, 프로토콜 동일. 런명 v2n_*.

### 예측 (고정)
- **H1 (핵심)**: 1.5B에서 nsiac↔fixedmid의 avg_final **시드별 값 범위가 겹친다**(비지배 방어 완성). 3B도 동일.
- **H2**: 방향성(컨트롤러 avg 우위)은 시드 평균에서 유지되나 겹침 범위 내.
- **H3**: 7B 코드블록 BWT 음수(10/10 일관)가 신규 2시드에서도 재현된다(각 시드 ≥8/10 음수).
- **H4**: STaR 붕괴(d3 최하·avg 최하)는 전 시드 재현(대마진이므로).

### 판정 규칙
| 결과 | 논문 처리 |
|---|---|
| H1 성립(겹침) | "sub-noise" 추정 서술을 **실측 겹침**으로 교체 — 비지배 완결. 시드별 값 병기(n=3 ±sd의 한계 명시) |
| H1 불성립(한쪽이 3/3 분리) | **그것이 결과** — 방향 주장을 해상된 차이로 승격(어느 쪽이든), 관련 절 재작성 |
| H3 불성립 | 앵커링 주장을 "단일 시드 관측"으로 강등 유지 |
- 표기 원칙: 가우시안 ± 흉내 금지, **시드별 값 병기**(스윕 표와 동일 규약). n=3 sd는 참고로만.

### 판정 (2026-08-20, 14런 일괄 집계 + 시드0 7런)
| 가설 | 결과 | 판정 |
|---|---|---|
| **H1 겹침** | 1.5B nsiac [0.4329,0.4804] vs fixedmid [0.4200,0.4492] 겹침 ✓ · 3B [0.4629,0.5187] vs [0.4400,0.5125] 겹침 ✓ | **성립** |
| **H2 방향** | 시드평균 nsiac 우위 1.5B +0.0216 · 3B +0.0144 (겹침 범위 내) | **성립** |
| **H3 7B 앵커링** | 음수 사이클 시드0 **10/10** / 시드1 **1/10**(+0.0409) / 시드2 **1/10**(+0.0480) | **불성립** → 규칙대로 "단일 시드 관측"으로 강등 |
| **H4 STaR 붕괴** | 아래 08-25 정정 참조 | **부분** |

**avg_final 시드별 값** (시드0/1/2): 1.5B nsiac 0.4433/0.4329/0.4804 · fixedmid 0.4225/0.4200/0.4492 · sftstar 0.3517/0.4183/0.4100 · noreh 0.4166/0.3962/0.4250 | 3B nsiac 0.5025/0.5187/0.4629 · fixedmid 0.4883/0.4400/0.5125

### ⚠️ 판정 수치 교정 (2026-08-21)
최초 판정에서 H3를 "이탈 후 Code 드리프트(c21–30 vs c20)"로 계산했으나 논문 주장의 정본 양은 **코드블록 bwt 열**(make_results 정의: GSM8K를 c10에 앵커). 정본 재계산: 시드0 10/10 음수·평균 −0.0517(논문 −0.052와 일치), **시드1 1/10·+0.0409, 시드2 1/10·+0.0480** — 불성립 결론 동일, 수치만 교정.

### ⚠️ 판정 정정 (2026-08-25)
위 판정표의 **H4 d3(distinct-3) 최하 "시드0만 ✓ — 부분 불성립" 기재는 오류**로 판명.
로그 재계산(정상상태 평균, 최종사이클, 전구간 평균 모두): sftstar d3 = 0.3134/0.3569/0.2545(시드0/1/2)
vs 각 시드의 GRPO 최저 0.6826/0.7394/0.7250 → **3시드 전부 대마진 최하 = H4 d3 연언지 성립**.
판정표의 "0.3933"은 어떤 재현 가능한 슬라이스와도 일치하지 않음(정상상태 시드평균 0.3083,
최종사이클 평균 0.1798, 전구간 평균 0.4186). H4의 실패 연언지는 avg-최하(시드1 ✗)뿐이며,
논문 본문(RQ3)의 서술이 정확함.
교훈: 판정 기록에 **사용 지표(집계 창·통계량)를 명기**할 것 — 지표 미명기가 이 오기의 원인.
