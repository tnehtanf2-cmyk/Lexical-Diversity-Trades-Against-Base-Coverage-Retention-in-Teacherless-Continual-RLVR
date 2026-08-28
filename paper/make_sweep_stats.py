#!/usr/bin/env python3
"""Correlation statistics for the trilemma sweep, emitted as JSON so the prose never
hand-carries them.

Two estimators are reported for each of the three axis pairs, because they answer
different questions and the paper previously mixed them:

  condition-mean : Spearman over the six condition means (one point per setting).
                   Significance by EXACT permutation over 6! = 720 relabelings.
  seed-wise      : Spearman computed within each seed, then averaged. Significance by
                   EXACT block permutation: the null distribution of the seed-averaged
                   statistic is the 3-fold convolution of each seed's 720-permutation
                   distribution (720^3 relabelings, counted exactly via pair-sum
                   binary search).

The seed-wise estimator is the honest one: averaging over seeds before correlating
inflates the condition-mean magnitudes.

v2 (2026-08-06, PAT finding): inputs are now recomputed at FULL PRECISION directly
from the sweep logs. The previous version read make_results_v2h.py's summary JSON,
which rounds to 4 decimals; that rounding created a spurious covret tie between
kl_hi_s1 (0.928108...) and fixed_mid_s1 (0.928060...), which produced a spurious
per-seed rho of exactly 0.00 for div-cov and a false "sign inconsistency" claim.
The earlier Monte-Carlo block permutation (hand-rolled LCG) is replaced by the exact
convolution; the LCG's low-bit bias had shifted p by ~4 sigma on tie-free pairs.

Input : logs/sweep_{cond}_s{seed}.csv + *_meta.jsonl, logs/sweep_base_env_meta.jsonl
Output: paper/results_sweep_stats.json
"""
import bisect, json, math, os, statistics as st
from itertools import permutations

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG = os.path.join(ROOT, "logs")

# covret helpers reused verbatim from make_results_v2h.py (same trick as
# make_results_v2k.py) so the definition cannot drift. Do NOT import/run the v2h
# script itself: executing it would overwrite results_main.tex with v2h numbers.
_src = open(os.path.join(HERE, "make_results_v2h.py")).read()
_head = _src.split("# ---------------- main continual table ----------------")[0]
_v2h = {"__file__": os.path.join(HERE, "make_results_v2h.py")}
exec(compile(_head, "make_results_v2h.py", "exec"), _v2h)
base_sets, covret_mean = _v2h["base_sets"], _v2h["covret_mean"]

SWEEP_CONDS = ["nsiac", "ent_lo", "ent_hi", "kl_lo", "kl_hi", "fixed_mid"]
SEEDS = [0, 1, 2]
STEADY_SWEEP = range(11, 16)
PAIRS = [("ret", "div"), ("ret", "cov"), ("div", "cov")]
LABEL = {"ret": "BWT", "div": "distinct-3", "cov": "base-coverage"}


def spearman(x, y):
    n = len(x)
    if n < 3:
        return float("nan")

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:                      # average ranks within ties
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def exact_perm_p(x, y, rho):
    """Two-sided exact permutation p over all relabelings of y (n!, n=6 here)."""
    n = len(x)
    hits = tot = 0
    for perm in permutations(range(n)):
        r = spearman(x, [y[i] for i in perm])
        tot += 1
        if abs(r) >= abs(rho) - 1e-12:
            hits += 1
    return hits / tot


def exact_block_perm_p(per_seed_x, per_seed_y, stat):
    """Exact two-sided p for the seed-averaged Spearman under independent
    within-seed permutation of condition labels. The null of the SUM of the three
    per-seed statistics is the convolution of three 720-value distributions;
    720^3 combinations are counted exactly with a sorted pair-sum + binary search."""
    dists = []
    for xs, ys in zip(per_seed_x, per_seed_y):
        dists.append([spearman(xs, [ys[i] for i in perm])
                      for perm in permutations(range(len(xs)))])
    d1, d2, d3 = dists
    pair = sorted(a + b for a in d1 for b in d2)          # 720^2 sums
    t = 3.0 * abs(stat) - 1e-9                             # threshold on the sum scale
    hits = 0
    for v in d3:
        # upper tail: pair + v >= t   -> pair >= t - v
        hits += len(pair) - bisect.bisect_left(pair, t - v)
        # lower tail: pair + v <= -t  -> pair <= -t - v
        hits += bisect.bisect_right(pair, -t - v)
    tot = len(d1) * len(d2) * len(d3)
    return hits / tot


# ---------------- full-precision sweep metrics from the logs ----------------
base_sweep = base_sets(f"{LOG}/sweep_base_env_meta.jsonl")
sweep = {}
for c in SWEEP_CONDS:
    per = {"ret": [], "div": [], "cov": []}
    for s in SEEDS:
        df = pd.read_csv(f"{LOG}/sweep_{c}_s{s}.csv")
        S = df[df.cycle.isin(STEADY_SWEEP)]
        per["ret"].append(float(S.bwt.mean()))
        per["div"].append(float(S.distinct_3.mean()))
        per["cov"].append(covret_mean(f"{LOG}/sweep_{c}_s{s}_meta.jsonl", base_sweep,
                                      set(STEADY_SWEEP)))
    sweep[c] = per

conds = SWEEP_CONDS
out = {"n_conditions": len(conds), "n_seeds": len(SEEDS), "conditions": conds,
       "precision": "full (recomputed from logs; no intermediate rounding)",
       "pairs": {}}

for a, b in PAIRS:
    key = f"{a}-{b}"
    mean_a = [st.mean(sweep[c][a]) for c in conds]
    mean_b = [st.mean(sweep[c][b]) for c in conds]
    rho_mean = spearman(mean_a, mean_b)

    per_x = [[sweep[c][a][s] for c in conds] for s in range(len(SEEDS))]
    per_y = [[sweep[c][b][s] for c in conds] for s in range(len(SEEDS))]
    per_seed = [spearman(per_x[s], per_y[s]) for s in range(len(SEEDS))]
    rho_seed = st.mean(per_seed)

    signs = {(1 if r > 0 else -1 if r < 0 else 0) for r in per_seed}
    out["pairs"][key] = {
        "label": f"{LABEL[a]} vs {LABEL[b]}",
        "condition_mean_rho": rho_mean,
        "condition_mean_exact_p": exact_perm_p(mean_a, mean_b, rho_mean),
        "per_seed_rho": per_seed,
        "seed_mean_rho": rho_seed,
        "block_perm_p": exact_block_perm_p(per_x, per_y, rho_seed),
        "block_perm_kind": "exact (720^3 convolution)",
        "sign_consistent": len(signs) == 1 and 0 not in signs,
    }

# critical |rho| at n=6 for a two-sided exact test at alpha=0.05
grid = sorted({abs(spearman(list(range(6)), list(p))) for p in permutations(range(6))})
crit = next((r for r in grid if exact_perm_p(list(range(6)),
            [i for i in range(6)], r) <= 0.05), None)
out["n6_two_sided_alpha05_critical_rho"] = crit

with open(f"{HERE}/results_sweep_stats.json", "w") as f:
    json.dump(out, f, indent=1)

print("OK: results_sweep_stats.json (full precision, exact block permutation)")
for k, v in out["pairs"].items():
    print(f"  {k:<9} cond-mean rho {v['condition_mean_rho']:+.3f} (exact p {v['condition_mean_exact_p']:.3f}) | "
          f"seed-wise {[round(r,3) for r in v['per_seed_rho']]} mean {v['seed_mean_rho']:+.3f} "
          f"(exact block p {v['block_perm_p']:.3f}) | sign-consistent {v['sign_consistent']}")
