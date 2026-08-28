#!/usr/bin/env python3
"""Entropy-coefficient dose-response table + figure (pre-registered; see the record in
wiki/방법론-사전등록-도즈리스폰스.md).

Why a dose-response instead of a correlation
--------------------------------------------
A Spearman correlation over six heterogeneous sweep settings failed three robustness
checks: the diversity lever barely separated conditions (between-condition SD ~= the
within-condition seed SD), one condition carried most of the signal, and the two
statistics were computed on different support sets (distinct-3 on the current domain,
coverage retention averaged over all seen domains). Varying a single lever monotonically
answers the causal question directly and lets us match support sets.

Design: entropy_coef in {0, 0.01, 0.02, 0.04, 0.08} x 3 seeds, every other knob and the
protocol held fixed. Three of the five levels already existed in the clean sweep
(ent_lo/fixed_mid/ent_hi differ only in this coefficient); two were added.

Coverage retention is reported three ways because the support set matters:
  all    - averaged over every domain seen so far (the paper's axis-3 definition)
  current- the domain being trained in the steady-state window (matches distinct-3)
  past   - domains already left behind
Outputs: paper/results_dose.tex, paper/figs/dose_response.pdf, paper/results_dose.json
"""
import json, math, os
import statistics as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG = os.path.join(ROOT, "logs")
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)

_src = open(os.path.join(HERE, "make_results_v2h.py")).read()
_head = _src.split("# ---------------- main continual table ----------------")[0]
_v2h = {"__file__": os.path.join(HERE, "make_results_v2h.py")}
exec(compile(_head, "make_results_v2h.py", "exec"), _v2h)
base_sets, covret_mean = _v2h["base_sets"], _v2h["covret_mean"]

BASE = base_sets(f"{LOG}/sweep_base_env_meta.jsonl")
STEADY = set(range(11, 16))
SEEDS = (0, 1, 2)
DOSE = [(0.0, "sweep2_ent_lo"), (0.01, "sweep2_fixed_mid"), (0.02, "sweep2_ent_002"),
        (0.04, "sweep2_ent_hi"), (0.08, "sweep2_ent_008")]


def spearman(x, y):
    n = len(x)

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
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


def cov_scope(stem, scope):
    """Coverage retention restricted to a support set. `scope` in {all,current,past}."""
    df = pd.read_csv(f"{LOG}/{stem}.csv")
    cur = {int(r["cycle"]): r["domain"] for _, r in df.iterrows()}
    vals = []
    for line in open(f"{LOG}/{stem}_meta.jsonl"):
        j = json.loads(line)
        if j["cycle"] not in STEADY:
            continue
        c, per = cur.get(j["cycle"]), []
        for d, v in j.get("solved", {}).items():
            if not BASE.get(d):
                continue
            if scope == "current" and d != c:
                continue
            if scope == "past" and d == c:
                continue
            pol = {i for i, x in enumerate(v) if x > 0}
            per.append(len(pol & BASE[d]) / len(BASE[d]))
        if per:
            vals.append(st.mean(per))
    return st.mean(vals) if vals else float("nan")


D, levels = {}, [ec for ec, _ in DOSE]
for ec, stem in DOSE:
    rec = {k: [] for k in ("div", "ent", "all", "current", "past")}
    for s in SEEDS:
        run = f"{stem}_s{s}"
        blk = pd.read_csv(f"{LOG}/{run}.csv")
        blk = blk[blk.cycle.isin(STEADY)]
        rec["div"].append(float(blk.distinct_3.mean()))
        rec["ent"].append(float(blk.gen_entropy.mean()))
        rec["all"].append(covret_mean(f"{LOG}/{run}_meta.jsonl", BASE, STEADY))
        rec["current"].append(cov_scope(run, "current"))
        rec["past"].append(cov_scope(run, "past"))
    D[ec] = rec

stats = {}
for key in ("div", "ent", "all", "current", "past"):
    cm = spearman(levels, [st.mean(D[ec][key]) for ec in levels])
    per = [spearman(levels, [D[ec][key][s] for ec in levels]) for s in range(len(SEEDS))]
    stats[key] = dict(condition_mean_rho=cm, per_seed_rho=per, seed_mean_rho=st.mean(per))

summary = {"levels": levels, "seeds": list(SEEDS),
           "values": {str(ec): {k: v for k, v in D[ec].items()} for ec in levels},
           "spearman": stats,
           "preregistered": {"H1": "div rho >= +0.8", "H2": "cov(all) rho <= -0.8",
                             "H3": "cov(current) rho <= -0.8"},
           "verdict": {"H1": stats["div"]["condition_mean_rho"] >= 0.8,
                       "H2": stats["all"]["condition_mean_rho"] <= -0.8,
                       "H3": stats["current"]["condition_mean_rho"] <= -0.8}}

with open(f"{HERE}/results_dose.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\caption{Entropy-coefficient dose--response (light protocol,"
            " 15 cycles, 3 paired seeds, mean$\\pm$sd over seeds; steady state = cycles 11--15)."
            " Every other knob and the protocol are held fixed, so the coefficient is the only"
            " thing that varies. Coverage retention is reported on three support sets because"
            " distinct-3 is measured on the domain currently being trained: \\emph{all} seen"
            " domains (the axis-3 definition), the \\emph{current} domain alone (matched support),"
            " and the \\emph{past} domains alone. $\\rho$ = Spearman over the five dose levels of"
            " the condition means; the hypotheses and thresholds were registered before these"
            " runs.}\n"
            "\\label{tab:dose}\n\\centering\n\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
            "$\\beta_{\\mathrm{ent}}$ & distinct-3 & gen.\\ entropy & Cov.\\ ret.\\ (all) &"
            " (current) & (past) \\\\\n\\midrule\n")
    for ec in levels:
        d = D[ec]
        f.write(f"{ec:.2f} & {st.mean(d['div']):.3f}$\\pm${st.pstdev(d['div']):.3f} & "
                f"{st.mean(d['ent']):.2f} & {st.mean(d['all']):.3f} & "
                f"{st.mean(d['current']):.3f} & {st.mean(d['past']):.3f} \\\\\n")
    f.write("\\midrule\n$\\rho$ (dose) & "
            f"{stats['div']['condition_mean_rho']:+.2f} & {stats['ent']['condition_mean_rho']:+.2f} & "
            f"{stats['all']['condition_mean_rho']:+.2f} & {stats['current']['condition_mean_rho']:+.2f} & "
            f"{stats['past']['condition_mean_rho']:+.2f} \\\\\n"
            "\\bottomrule\n\\end{tabular}\n\\end{table}\n")

fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7))
x = range(len(levels))
lab = [f"{e:g}" for e in levels]
for s in SEEDS:
    ax[0].plot(x, [D[ec]["div"][s] for ec in levels], color="0.7", lw=.8)
    ax[1].plot(x, [D[ec]["current"][s] for ec in levels], color="0.7", lw=.8)
ax[0].plot(x, [st.mean(D[ec]["div"]) for ec in levels], "o-", color="C0", lw=1.6)
ax[1].plot(x, [st.mean(D[ec]["current"]) for ec in levels], "o-", color="C3", lw=1.6,
           label="current domain")
ax[1].plot(x, [st.mean(D[ec]["all"]) for ec in levels], "s--", color="C1", lw=1.2,
           label="all domains")
ax[0].set_ylabel("distinct-3"); ax[1].set_ylabel("base-coverage retention")
for a in ax:
    a.set_xticks(list(x)); a.set_xticklabels(lab)
    a.set_xlabel(r"entropy coefficient $\beta_{\mathrm{ent}}$"); a.grid(alpha=.3)
ax[1].legend(fontsize=6)
plt.tight_layout(); plt.savefig(f"{FIG}/dose_response.pdf"); plt.close()

with open(f"{HERE}/results_dose.json", "w") as f:
    json.dump(summary, f, indent=1)

print("OK: results_dose.tex, figs/dose_response.pdf, results_dose.json")
for k in ("div", "ent", "all", "current", "past"):
    s = stats[k]
    print(f"  {k:<8} cond-mean rho {s['condition_mean_rho']:+.3f} | "
          f"per-seed {[round(v,2) for v in s['per_seed_rho']]} mean {s['seed_mean_rho']:+.3f}")
print(f"  verdict: {summary['verdict']}")
