# Registration ledger

Every preregistration this study relies on, with its outcome and where the manuscript
reports it. Predictions that failed are listed alongside those that held; the practice
is that a failed registration is disclosed rather than dropped, since selective
reporting is what registration exists to prevent.

| Registration | Predictions | Outcome | Reported in |
|---|---|---|---|
| Dose–response (`prereg-01`) | H1 distinct-3 up, H2 coverage retention down, H3 matched-support coverage down (\|rho\| >= 0.8) | **3/3 held** | Headline: abstract, Sec. 4.4, RQ0a (Sec. 5.1), Table 2 |
| Multi-seed (`prereg-03`) | H1 ranges overlap, H2 direction preserved, H3 7B negative BWT replicates, H4 STaR collapse in every seed | H1 held, H2 held, **H3 failed** (1/10, 1/10), H4 split: distinct-3 conjunct held in all three seeds, lowest-average-accuracy conjunct failed in one | Both failures stated in RQ3 (Sec. 5.5) and RQ5 (Sec. 5.7); Sec. 4.4 announces that failures will be reported |
| Harvest source (`prereg-02`) | H1 drift reappears in all three rehearsal arms, H2 no-rehearsal arm shows none, H3 BWT rises | H1 held for the two GRPO arms and **failed for STaR/SFT**, H2 held, H3 held | RQ2 (Sec. 5.4), with the STaR/SFT failure printed in Table 6 and discussed; RQ2 also records that this registration was confirmatory rather than blind |
| Scale overfitting (`prereg-04`) | Peak height, decline magnitude, earlier peak at 7B | Peak height held, earlier peak held, **decline magnitude failed** | Limitations (vii), which states the scale claim is limited to earlier peaking |

Counting each conjunct separately (three for the dose-response, five for the multi-seed
registration, three for the harvest source, three for the scale prediction), fourteen predictions
were registered and **four did not hold**: the 7B backward-transfer replication, the
lowest-average-accuracy conjunct of the STaR/SFT prediction, the drift conjunct for the STaR/SFT
arm, and the decline-magnitude conjunct at 7B.

## Two verdict corrections

Both are kept in the record files rather than folded away.

1. **2026-08-21** — the multi-seed H3 tally had been computed on post-shift drift
   instead of the code-block BWT column the manuscript's claim uses. Recomputed on the
   canonical quantity; the verdict (H3 fails) was unchanged, the numbers corrected.
2. **2026-08-25** — the multi-seed H4 distinct-3 conjunct had been recorded as holding
   at one seed when recomputation shows it holds at all three. The figure in the
   original table matched no reproducible slice. The manuscript's RQ3 wording was
   already the accurate one.

The lesson recorded at the time: a verdict entry must name the aggregation window and
statistic it used. Not naming them caused the second error.

Section and table numbers refer to the manuscript version deposited with this release.
