# Preregistration — entropy-coefficient dose–response

- Registered: 2026-08-12, before the two added dose levels were run
- Adjudicated: 2026-08-12, after all fifteen runs completed
- Manuscript: Sec. 4.4 (design and thresholds), RQ0a (Sec. 5.1), Table 2, Fig. 1

---

## English

### Why a dose–response replaced a correlation

In the clean sweep the within-seed diversity–coverage correlation (rho = -0.638,
p = 0.011) failed three robustness checks:

1. the diversity lever was not separated (between-condition SD 0.0337 vs
   within-condition seed SD 0.0305, ratio 1.11);
2. one setting carried the signal (leave-one-out rho = -0.400, one seed flipping sign);
3. the support sets did not match — distinct-3 is measured on the domain currently
   being trained, coverage retention is averaged over all seen domains. Matching them
   moved rho from -0.638 to -0.173, sign-inconsistent.

A correlation over six heterogeneous settings cannot answer a causal question, so it
was replaced by a monotone dose–response on a single lever.

### Design

`entropy_coef` in {0, 0.01, 0.02, 0.04, 0.08} x 3 seeds. Every other knob and the
protocol held fixed (BASE + kl_beta = 0.04, 15-cycle light protocol).

- Nine reused runs: ent_lo (0.0) = sweep2_ent_lo, fixed_mid (0.01) = sweep2_fixed_mid,
  ent_hi (0.04) = sweep2_ent_hi. The launchers were checked to confirm these three
  differ only in `entropy_coef`.
- Six new runs: `sweep2_ent_002_s{0,1,2}` (0.02) and `sweep2_ent_008_s{0,1,2}` (0.08).
  About 14 hours.

### Predictions (fixed before the new runs)

- **H1**: distinct-3 increases monotonically with `entropy_coef`. Criterion: Spearman
  **rho >= +0.8** over the condition means of the five dose levels.
- **H2**: coverage retention (all seen domains) decreases monotonically.
  Criterion: **rho <= -0.8**.
- **H3 (decisive)**: coverage retention computed on the **currently trained domain
  only** also decreases monotonically. Criterion: **rho <= -0.8**.

### Decision rule (fixed in advance, to foreclose post-hoc discretion)

| Outcome | Conclusion | Treatment in the paper |
|---|---|---|
| H1 and H3 hold | The diversity lever really does spend coverage; not a support-set artifact | Promote to headline, stated as a dose–response rather than a correlation |
| H1 holds, H3 fails | The tension is an aggregation-scope artifact | No headline. Report as a measurement-validity result: differing support sets manufacture a correlation |
| H1 fails | The entropy coefficient does not control diversity in this regime | Withdraw the sweep's "diversity lever" premise and add the knob to the inert list |
| H1 and H2 hold, H3 borderline (-0.8 < rho <= -0.5) | Partial support | Exploratory report, both computations shown side by side |

Additional commitment: report **both** the condition-mean Spearman (5 points) **and**
the mean within-seed Spearman. Neither may be selected over the other. Publish every
per-seed curve in the figure.

### Falsifiability, as registered

If distinct-3 stays in the 0.73–0.76 band regardless of `entropy_coef`, H1 fails.
That is the natural extension of the already-observed "five conditions packed into a
0.020 range" and **may well be the most likely outcome**. It would be reported as-is.

### Outcome (2026-08-12) — all three hypotheses held

| ent_coef | distinct-3 | gen_entropy | covret (all) | covret (current) | covret (past) |
|---|---|---|---|---|---|
| 0.00 | 0.736 | 2.44 | 0.943 | 0.965 | 0.931 |
| 0.01 | 0.755 | 2.30 | 0.934 | 0.938 | 0.932 |
| 0.02 | 0.778 | 2.43 | 0.936 | 0.962 | 0.923 |
| 0.04 | **0.831** | 2.33 | 0.908 | 0.928 | 0.898 |
| 0.08 | 0.811 | 2.35 | **0.862** | 0.888 | 0.850 |

H1 rho = +0.900 (pass) / H2 rho = -0.900 (pass) / H3 rho = -0.900, within-seed -0.807
(pass). Under the registered rule this promotes to the headline.

### Caveats recorded with the verdict, to be carried into the paper

1. Within-seed consistency of H1 is weak (per-seed rho 0.7 / 1.0 / **-0.1**).
2. **Pure-loss region above 0.04** — diversity falls after its peak (0.831 to 0.811)
   while coverage keeps falling (0.908 to 0.862).
3. **The two diversity metrics dissociate** — gen_entropy is unrelated to dose
   (rho = -0.30). Note that the entropy hypothesis was **not** among the three
   registered; this is an unregistered observation made on the same runs.

---

## 원문 (Korean original, as written in the research log)

### 왜 하는가
청정 스윕에서 얻은 div–cov 반상관(시드내 ρ −0.638, p=0.011)이 로버스트니스 3종을 통과하지 못했다:
① 다양성 레버가 분리되지 않음(조건간 SD 0.0337 ≈ 조건내 시드 SD 0.0305, 비 1.11)
② ent_hi 한 점 의존(leave-one-out −0.400, 한 시드 부호 반전)
③ **지지집합 불일치** — distinct-3는 현재 도메인, covret은 전 도메인 평균. 맞추면 −0.638 → **−0.173, 부호 비일관**
⇒ 이질적 6조건에 대한 상관 대신, **단일 레버의 단조 용량-반응**으로 인과를 직공한다.

### 설계
`entropy_coef ∈ {0, 0.01, 0.02, 0.04, 0.08}` × 시드 3. 나머지 노브·프로토콜 전부 고정(BASE + kl_beta=0.04, 15사이클 경량).
- **기존 재사용 9런**: ent_lo(0.0)=sweep2_ent_lo, fixed_mid(0.01)=sweep2_fixed_mid, ent_hi(0.04)=sweep2_ent_hi — 셋은 entropy_coef만 다름을 런처에서 확인.
- **신규 6런**: `sweep2_ent_002_s{0,1,2}`(0.02), `sweep2_ent_008_s{0,1,2}`(0.08). ~14h.

### 예측 (고정)
- **H1**: distinct-3가 entropy_coef에 단조 증가. 판정: 5개 용량수준의 조건평균 Spearman **ρ ≥ +0.8**.
- **H2**: covret(전 도메인)이 단조 감소. 판정: **ρ ≤ −0.8**.
- **H3 (결정적)**: covret을 **현재 도메인만으로** 산출해도 단조 감소. 판정: **ρ ≤ −0.8**.

### 판정 규칙 (사후 해석 재량 차단)
| 결과 | 결론 | 논문 처리 |
|---|---|---|
| H1 ∧ H3 성립 | 다양성 레버가 실제로 커버리지를 깎는다(지지집합 교란 아님) | **헤드라인 승격** — 상관이 아니라 용량-반응으로 서술 |
| H1 성립, H3 실패 | 긴장은 **집계 범위 아티팩트** | 헤드라인 불가. "두 지표의 지지집합이 다르면 상관이 만들어진다"는 **측정 타당도 결과**로 보고 |
| H1 실패 | 이 레짐에서 엔트로피 계수는 다양성을 제어하지 못함 | 스윕의 "다양성 레버" 전제 자체를 철회, 노브 무효 목록에 추가 |
| H1 ∧ H2 성립, H3 경계(−0.8<ρ≤−0.5) | 부분 지지 | 탐색적 보고 + 두 산출 방식 병기 |

**추가 고정**: 단조성 판정에 사용할 통계는 **조건평균 Spearman**(5점)과 **시드내 Spearman 평균**을 모두 보고한다. 어느 하나만 취사선택하지 않는다. 시드별 곡선도 그림으로 전부 공개한다.

### 반증 가능성 (이 예측이 틀렸다면)
distinct-3가 entropy_coef와 무관하게 0.73–0.76 밴드에 머문다면 H1 실패 — 이는 우리가 이미 관측한 "5개 조건이 폭 0.020에 밀집"의 자연스러운 연장이며, **가장 가능성 높은 결과일 수 있다**. 그 경우에도 결과는 결과로 보고한다.

### 결과 (2026-08-12 완주) — 3가설 전부 성립
**H1 ρ=+0.900 ✅ / H2 ρ=−0.900 ✅ / H3(결정적, 지지집합 정합) ρ=−0.900·시드내 −0.807 ✅** → 사전등록 규칙상 **헤드라인 승격**.

**반드시 병기할 카비앗**: ①H1의 시드내 일관성 약함(시드별 0.7/1.0/**−0.1**) ②**0.04 이후 순손실 구간** — 다양성은 정점 후 하락(0.831→0.811)하는데 커버리지는 계속 하락(0.908→0.862) ③**두 다양성 지표 해리** — gen_entropy는 도즈와 무관(ρ=−0.30).
