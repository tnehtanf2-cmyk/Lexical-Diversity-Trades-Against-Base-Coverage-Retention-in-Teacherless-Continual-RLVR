#!/usr/bin/env python3
"""Figure 2 of the manuscript: clean-sweep settings in the (BWT, base-coverage) plane,
marker size proportional to diversity.

Reads the sweep aggregates that make_results_v2l.py writes to results_summary_v2l.json
(condition means over three paired seeds, steady state = cycles 11-15 of the sweep2_* logs),
so the figure is a pure function of that deposited summary. Run make_results_v2l.py first
on a clean checkout to regenerate the summary from logs/.
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)
S = json.load(open(os.path.join(HERE, "results_summary_v2l.json")))["sweep"]

CONDS = ["nsiac", "ent_lo", "ent_hi", "kl_lo", "kl_hi", "fixed_mid"]
LABEL = {"nsiac": "learned", "ent_lo": "ent-low", "ent_hi": "ent-high",
         "kl_lo": "KL-low", "kl_hi": "KL-high", "fixed_mid": "fixed-mid"}

plt.figure(figsize=(4.2, 3.2))
for c in CONDS:
    ret, div, cov = S[c]["ret"][0], S[c]["div"][0], S[c]["cov"][0]
    size = max(30.0, 60.0 + 900.0 * (div - 0.7))
    plt.scatter(ret, cov, s=size, marker="*" if c == "nsiac" else "o", alpha=.85,
                label=LABEL[c], edgecolors="k", linewidths=.5, zorder=3)
plt.xlabel("BWT (steady state)")
plt.ylabel("Base-coverage retention")
plt.grid(alpha=.3)
plt.legend(fontsize=6, loc="lower left")
plt.tight_layout()
out = os.path.join(FIG, "pareto_sweep.pdf")
plt.savefig(out); plt.close()
print("wrote", out, "from", {c: [round(S[c][k][0], 3) for k in ("ret", "div", "cov")] for c in CONDS})
