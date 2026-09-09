# Preregistration — capacity-overfitting signature at 7B

- Registered: 2026-07-31, **after the 3B run completed and before the 7B run started**
- Adjudicated: 2026-08-03, after the 7B run completed
- Manuscript: Discussion, Limitations (vii)

Note on scope. This registration was written once the 3B result was in hand, so it
covers the **7B run only**. It is not a registration made before the scale ladder
began. Limitations (vii) states it in those terms.

---

## English

### Where the prediction came from

At 3B the model underperformed 1.5B. Two readings were possible and they predict
opposite curves: **underfitting** (too little learned, so accuracy rises monotonically
to the end of the block) versus **overfitting** (a small sample memorised quickly, so
accuracy peaks and then declines). The 3B data matched the overfitting reading: the
rise was comparable to 1.5B (+0.306 vs +0.331), the peak was higher (0.679 vs 0.610),
and the post-peak decline was three times larger (-0.089 vs -0.029), with the peak at
cycle 7 rather than at the end of the block.

### Prediction for 7B (fixed before the run)

With params/token 4,651 — 2.5 times the 3B figure:

1. the GSM8K block peak is **higher than or comparable to** the 3B peak (0.679);
2. the **post-peak decline exceeds -0.089**;
3. the peak is reached at an **earlier cycle** within the block.

Conjunct 2 is the primary indicator. **If it is refuted, the overfitting explanation
is withdrawn** and the knob-calibration hypothesis returns.

### Outcome (2026-08-03)

- Conjunct 1, peak height: 7B 0.755 > 3B 0.679 — **holds**.
- Conjunct 3, peak timing: cycle 4 < cycle 7 — **holds**.
- Conjunct 2, decline magnitude: **fails**.

**Correction (2026-09-10).** The verdict stands, but the quantity first used to reach it was not
the registered one, and it was read off runs that a later pipeline revision superseded. The
registered quantity is the decline from the within-block peak to the end of the first block
(the 3B threshold of -0.089 is peak@c7 to c10). On that quantity the 7B run declines by only
**-0.019** -- the block-internal dip to -0.116 at cycle 7 recovered to 0.736 by cycle 10 -- so
the conjunct fails clearly. Across three seeds the peak-to-block-end means are -0.066 at 3B and
-0.065 at 7B, so it fails there too. The figures first recorded here, -0.0225 at 7B versus
-0.0249 at 3B, are peak-to-cycle-30 values from the superseded `v2i_*` runs; the first block is
bit-identical between revisions, so the conclusion is unaffected.

So "scale accelerates overfitting" is supported for **earlier peaking** and not
supported for **decline magnitude**, and the paper states it that way.

A separate observation from the same runs: 3B being below 1.5B holds (0.438 < 0.500)
but 7B is highest (0.604), so the pattern is not monotone — a U shape with 3B alone
depressed. The data/parameter ratio does not explain this on its own, and 3B seed
noise (measured at 0.128) remains the more likely account.

**Correction (2026-09-10).** This paragraph describes the superseded `v2i_*` runs and no longer
matches the manuscript. In the runs reported there, average final accuracy is monotone in scale
(0.443 < 0.503 < 0.533 at seed 0; 0.452 < 0.495 < 0.528 in controller seed means), so the U shape
and the 3B low point belong to the earlier revision. The premise that generated this
registration — 3B underperforming 1.5B — therefore does not hold in the reported runs. The
registration is kept as a record of a prediction fixed in advance about peak shape, not as a
statement about the scale ordering.

---

## 원문 (Korean original)

**배경(2026-07-31)**: 사용자 가설 "파라미터 수에 비해 학습 데이터가 너무 적다"를 두 버전으로 나눠 검증. **과소적합**(덜 배움 ⇒ 블록말까지 단조 상승)과 **과대적합**(적은 표본 암기 ⇒ 정점 후 하락)은 예측이 정반대. **판정: 과대적합 버전이 맞다.** 3B는 (i) 상승폭이 1.5B와 비슷하고(+0.306 vs +0.331) (ii) **정점이 더 높으며**(0.679 vs 0.610) (iii) **정점 후 하락이 3배**(−0.089 vs −0.029). 과소적합이면 블록말까지 오르고 있어야 하는데 cycle 7에 정점 후 하락.

**7B에 대한 사전 등록 예측**: params/token 4,651(3B의 2.5배) ⇒ ①GSM8K 블록 정점이 3B(0.679)보다 **높거나 비슷** ②**정점 후 하락은 −0.089보다 큼** ③블록 내 정점 도달이 **더 이른 사이클**. 셋 중 ②가 핵심 지표. 반증되면 과대적합 설명을 철회하고 노브 캘리브레이션 가설로 회귀.

**사전등록 예측 최종 판정 (2026-08-03, 과적합 서명)**:
- ① 정점 높이(7B 0.755 > 3B 0.679) **적중**, ③ 정점 시점(c4 < c7) **적중**.
- ② 정점 후 하락폭 > 3B: **미지지(최종)**. GSM8K 블록 내 정점→c30이 7B **−0.0225** vs 3B **−0.0249**로 사실상 동일. 블록 내 일시 딥(−0.116)은 컸으나 회복했다. ⇒ "스케일이 과적합을 가속한다"는 주장은 **정점 시점 조기화에는 지지, 하락 크기에는 미지지**로 한정 서술.
- 3B가 1.5B보다 낮은 것은 유지되나(0.438 < 0.500), 7B가 최고(0.604)라 **단조 감소가 아니다** — 3B만 저조한 U자형. 데이터/파라미터 비만으로는 설명 불충분, 3B 시드 잡음(기존 실측 0.128)이 여전히 유력. 스케일 절은 "3B 단일 저점은 시드 잡음과 구분 불가"로 정직 서술.
