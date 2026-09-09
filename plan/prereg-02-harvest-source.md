# Preregistration — harvest-source ablation

- Registered: 2026-08-13, before the four eval-harvest runs were executed
- Adjudicated: 2026-08-21, after all four runs completed
- Manuscript: RQ2 (Sec. 5.4), Table 6

The manuscript describes this registration as "confirmatory rather than blind,"
because an earlier revision of the pipeline used the evaluation harvest by default
and its behaviour informed the magnitude expectation recorded below. That
qualification is stated in Sec. 5.4 and is repeated here.

---

## English

### Why new runs were needed

The existing eval-harvest runs came from an earlier pipeline revision that differed
from the current one in more than the harvest source, so they were not a
single-variable comparison. A strict ablation required the current pipeline with the
harvest source as the only difference.

### Design

`v2o_{nsiac, fixedmid, sftstar, noreh}`: full three-domain protocol, 1.5B, seed 0,
`--harvest_source eval`, current pipeline unchanged in every other respect. The
control arm is the four `v2k` runs (train-split harvest, same seed, same protocol).
The only difference between the paired columns is where the rehearsal buffer is
filled from. Four runs, about 33 hours.

### Predictions (fixed before the runs)

- **H1**: In the rehearsal-enabled conditions (nsiac, fixedmid, sftstar), post-shift
  Code drift (cycle 21 to cycle 30) **reappears as positive** (it was <= +0.024 under
  the train harvest). Mechanism: the eval harvest places verifier-correct completions
  of evaluation items into the training stream at the moment their domain becomes
  past — a harvest-side channel.
- **H2**: The no-rehearsal condition shows no drift, because without a rehearsal pass
  the buffer has no consumer and the harvest source is inert.
- **H3**: Backward transfer in the rehearsal-enabled conditions rises relative to the
  train-harvest control (the re-exposure component).
- Magnitude expectation: the earlier pipeline revision produced +0.186 / +0.146, so a
  similar order of magnitude is expected, but **the exact size is an open question** —
  that is what this experiment measures.

### Decision rule (fixed in advance)

| Outcome | Treatment in the paper |
|---|---|
| H1 and H2 hold | Present the contamination channel as a harvest-source ablation table (two columns: eval vs train), single-variable |
| H1 fails (drift does not reappear) | **Serious**: the earlier drift had some cause other than the harvest source. Re-examine the channel claim itself and report the result as it stands |

### Outcome (2026-08-21)

Canonical definitions used: `post_shift_drift` = acc_Code(c30) - acc_Code(c21);
BWT = mean over the steady-state window (cycles 21–30). Both taken from
`paper/make_results_v2k.py`.

| Condition | Drift, train -> eval | BWT, train -> eval | Reading |
|---|---|---|---|
| nsiac | +0.024 -> **+0.214** | +0.037 -> **+0.181** | H1 strong reappearance |
| fixedmid | +0.009 -> **+0.254** | +0.051 -> **+0.164** | H1 strong reappearance |
| sftstar | +0.000 -> +0.008 | -0.009 -> **+0.139** | No drift channel; expresses through BWT instead (H3) — reported as a failed conjunct |
| noreh | +0.042 -> +0.004 | -0.012 -> +0.006 | H2 holds (no channel without a rehearsal pass) |

**Verdict: H1 holds for the two GRPO rehearsal conditions, with an effect five to six
times the run-to-run resolution (about 0.04); H1 fails for sftstar; H2 holds; H3 holds
for all three rehearsal-enabled conditions.** With the current pipeline, changing only
the harvest source manufactures +0.19 to +0.25 of drift. The sftstar arm is flat in
drift because its Code accuracy is flat after the shift, and the contamination appears
in BWT instead; both columns are printed in the paper's table so the failure is visible.

---

## 원문 (Korean original)

### 왜 새 런이 필요한가
기존 eval-수확 런(v2h)은 파이프라인 구버전이라 **단일 변수 비교가 아니다**. "수확 소스만 다른" 엄밀한 절제에는 현행 파이프라인 + eval 수확 조건이 필요하다.

### 설계 (v2o, 4런 ≈ 33h)
`v2o_{nsiac,fixedmid,sftstar,noreh}` : full3 · 1.5B · 시드 0 · **`--harvest_source eval`** · 현행 코드 그대로. 대조군 = v2k 4런(train 수확, 동일 시드·프로토콜). 유일한 차이 = 수확 소스.

### 예측 (고정)
- **H1**: 리허설 활성 조건(nsiac·fixedmid·sftstar)의 **도메인 이탈 후 Code 드리프트(c21→c30)가 양(+)으로 재출현**(v2k에서는 ≤+0.024). 기전: eval 수확은 평가 문항 정답을 도메인이 '과거'가 되는 순간 학습 스트림에 넣는다 — 수확 측 채널.
- **H2**: 무리허설(noreh)은 드리프트 없음(재생 채널 부재; 수확 소스는 붕괴 대조군).
- **H3**: 리허설 조건의 BWT가 v2k 대비 상승(재노출 성분).
- 크기 예상: 구버전에서 +0.186/+0.146이었으므로 비슷한 자릿수 예상하나 **정확한 크기는 열린 질문** — 그것이 이 실험의 측정 대상.

### 판정 규칙
| 결과 | 논문 처리 |
|---|---|
| H1∧H2 성립 | 오염 채널을 "수확 소스 절제" 표(2열: eval vs train)로 제시 — 단일 변수 |
| H1 불성립(드리프트 재출현 안 함) | **중대**: 드리프트가 수확 아닌 다른 원인이었다는 뜻 — 채널 주장 자체 재검토, 결과 그대로 보고 |

### 판정 (2026-08-21, 정본 정의: make_results_v2k.py post_shift_drift = acc_Code(c30)−acc_Code(c21), BWT = steady(c21–30) 평균)
| 조건 | 드리프트 v2k(train)→v2o(eval) | BWT v2k→v2o | 해석 |
|---|---|---|---|
| nsiac | +0.024 → **+0.214** | +0.037 → **+0.181** | H1 강한 재출현 ✓ |
| fixedmid | +0.009 → **+0.254** | +0.051 → **+0.164** | H1 강한 재출현 ✓ |
| sftstar | +0.000 → +0.008 | −0.009 → **+0.139** | 드리프트 채널로는 무발현, **BWT 채널로 발현**(H3 ✓) — 정직 보고 |
| noreh | +0.042 → +0.004 | −0.012 → +0.006 | H2 ✓ (재생 채널 부재 시 무발현) |

**종합: H1 성립(GRPO+리허설 2조건, 효과 크기가 노이즈(~0.04)의 5–6배), H2 성립, H3 성립(리허설 3조건 전부 BWT 상승).** eval 수확만으로 드리프트 +0.19~0.25 제조. sftstar는 Code 정확도가 이탈 후 평탄해 드리프트 지표에 안 잡히고 BWT로만 발현 — 표에 그대로 병기.
