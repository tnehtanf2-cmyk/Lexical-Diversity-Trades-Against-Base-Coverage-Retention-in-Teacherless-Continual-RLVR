#!/usr/bin/env python3
"""IEEE LaTeX tables + figures from the v2h experiment set (2026-07-29).

Inputs: logs/v2_full_{nsiac,fixedmid,sft_star}_v2h*.csv + *_meta.jsonl,
        logs/sweep_{cond}_s{seed}.csv + *_meta.jsonl, logs/{full,sweep}_base_env_meta.jsonl.
Outputs: paper/results_main.tex, paper/results_sweep.tex, paper/figs/*.pdf,
         paper/results_summary.json (numbers quoted in prose come from here).
"""
import os, json, itertools
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

FULL = {"nsiac": "v2_full_nsiac_v2h", "fixedmid": "v2_full_fixedmid_v2h",
        "sft_star": "v2_full_sft_star_v2h"}
FULL_LABEL = {"nsiac": "MAPC (learned multi-axis)", "fixedmid": "Fixed mid-point GRPO",
              "sft_star": "STaR/SFT self-distillation"}
SWEEP_CONDS = ["nsiac", "ent_lo", "ent_hi", "kl_lo", "kl_hi", "fixed_mid"]
SWEEP_LABEL = {"nsiac": "learned (multi-axis)", "ent_lo": "entropy low", "ent_hi": "entropy high",
               "kl_lo": "KL low", "kl_hi": "KL high", "fixed_mid": "fixed mid-point"}
SEEDS = [0, 1, 2]
STEADY_FULL = range(21, 31)
STEADY_SWEEP = range(11, 16)
DOMS = ["GSM8K", "Code", "MedicalMC"]
BLOCK_END = {"GSM8K": 10, "Code": 20, "MedicalMC": 30}   # full runs, shift_interval=10


def base_sets(path):
    base = {}
    with open(path) as f:
        for line in f:
            j = json.loads(line)
            for d, v in j["solved"].items():
                base.setdefault(d, set()).update(i for i, x in enumerate(v) if x > 0)
    return base


def covret(meta_path, base, cycles):
    out = []
    with open(meta_path) as f:
        for line in f:
            j = json.loads(line)
            if j["cycle"] in cycles:
                vals = []
                for d, v in j.get("solved", {}).items():
                    if base.get(d):
                        pol = {i for i, x in enumerate(v) if x > 0}
                        vals.append(len(pol & base[d]) / len(base[d]))
                if vals:
                    out.append((j["cycle"], st.mean(vals)))
    return out


def covret_mean(meta_path, base, cycles):
    xs = [v for _, v in covret(meta_path, base, cycles)]
    return st.mean(xs) if xs else float("nan")


# ---------------- main continual table ----------------
base_full = base_sets(f"{LOG}/full_base_env_meta.jsonl")
summary = {"base_full_solved": {d: len(s) for d, s in base_full.items()}}
rows_main = []
for key, run in FULL.items():
    df = pd.read_csv(f"{LOG}/{run}.csv")
    S = df[df.cycle.isin(STEADY_FULL)]
    last = df.iloc[-1]
    finals = {d: float(last[f"acc_{d}"]) for d in DOMS}
    # relative retention R_final/R_ii for domains left before the end
    rr = {}
    for d in ["GSM8K", "Code"]:
        rii = float(df[df.cycle == BLOCK_END[d]][f"acc_{d}"].iloc[0])
        rr[d] = finals[d] / rii if rii > 0 else float("nan")
    cm = covret_mean(f"{LOG}/{run}_meta.jsonl", base_full, STEADY_FULL)
    rows_main.append(dict(key=key, label=FULL_LABEL[key],
                          avg_final=st.mean(finals.values()), **{f"final_{d}": finals[d] for d in DOMS},
                          bwt=S.bwt.mean(), dist3=S.distinct_3.mean(),
                          entropy=S.gen_entropy.mean(), covret=cm,
                          rr_gsm=rr["GSM8K"], rr_code=rr["Code"]))
summary["main"] = rows_main

with open(f"{HERE}/results_main.tex", "w") as f:
    f.write("\\begin{table*}[t]\n\\caption{Continual-story runs (30 cycles, GSM8K$\\to$Code$\\to$MedicalMC,"
            " full protocol). Steady state = cycles 21--30. Cov.\\ ret.\\ = retained fraction of the frozen"
            " base's solved set. $R_f/R_{ii}$ = final over block-end accuracy (relative retention).}\n"
            "\\label{tab:main}\n\\centering\n"
            "\\begin{tabular}{@{}lccccccccc@{}}\n\\toprule\n"
            "Method & Avg.\\ final & GSM8K & Code & MedMC & BWT & $R_f/R_{ii}$ G/C & distinct-3 &"
            " entropy & Cov.\\ ret. \\\\\n\\midrule\n")
    for r in rows_main:
        hi = "\\textbf" if r["key"] == "nsiac" else ""
        f.write(f"{r['label']} & {hi}{{{r['avg_final']:.3f}}} & {r['final_GSM8K']:.3f} & "
                f"{r['final_Code']:.3f} & {r['final_MedicalMC']:.3f} & {r['bwt']:.3f} & "
                f"{r['rr_gsm']:.2f}/{r['rr_code']:.2f} & {r['dist3']:.3f} & {r['entropy']:.2f} & "
                f"{r['covret']:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

# ---------------- sweep table + Pareto ----------------
base_sweep = base_sets(f"{LOG}/sweep_base_env_meta.jsonl")
sweep = {}
for c in SWEEP_CONDS:
    per = {"ret": [], "div": [], "cov": []}
    for s in SEEDS:
        df = pd.read_csv(f"{LOG}/sweep_{c}_s{s}.csv")
        S = df[df.cycle.isin(STEADY_SWEEP)]
        per["ret"].append(S.bwt.mean())
        per["div"].append(S.distinct_3.mean())
        per["cov"].append(covret_mean(f"{LOG}/sweep_{c}_s{s}_meta.jsonl", base_sweep, STEADY_SWEEP))
    sweep[c] = per
summary["sweep"] = {c: {k: [round(x, 4) for x in v] for k, v in per.items()} for c, per in sweep.items()}

with open(f"{HERE}/results_sweep.tex", "w") as f:
    # NOTE (2026-08-07): the checked-in results_sweep.tex caption additionally carries a
    # pre-repair provenance sentence; keep it if this script is ever re-run.
    f.write("\\begin{table}[t]\n\\caption{Single-lever sweep vs.\\ learned control (light protocol,"
            " 15 cycles, 3 paired seeds, mean$\\pm$sd; steady state = cycles 11--15).}\n"
            "\\label{tab:sweep}\n\\centering\n\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
            "Setting & BWT & distinct-3 & Cov.\\ ret. \\\\\n\\midrule\n")
    for c in SWEEP_CONDS:
        p = sweep[c]
        cells = [f"{st.mean(p[k]):.3f}$\\pm${st.stdev(p[k]):.3f}" for k in ("ret", "div", "cov")]
        f.write(f"{SWEEP_LABEL[c]} & " + " & ".join(cells) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# spearman between axis means over conditions
def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i]); rk = [0] * len(xs)
    for r, i in enumerate(order):
        rk[i] = r
    return rk
means = {c: {k: st.mean(v) for k, v in per.items()} for c, per in sweep.items()}
summary["spearman"] = {}
for a, b in itertools.combinations(("ret", "div", "cov"), 2):
    ra, rb = rank([means[c][a] for c in SWEEP_CONDS]), rank([means[c][b] for c in SWEEP_CONDS])
    n = len(SWEEP_CONDS)
    summary["spearman"][f"{a}_{b}"] = round(1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1)), 3)

# Fig: Pareto scatter (ret x cov, size=div)
plt.figure(figsize=(4.2, 3.2))
for c in SWEEP_CONDS:
    x, y = st.mean(sweep[c]["ret"]), st.mean(sweep[c]["cov"])
    sz = 60 + 900 * (st.mean(sweep[c]["div"]) - 0.7)
    m = "*" if c == "nsiac" else "o"
    plt.scatter(x, y, s=max(sz, 30), marker=m, alpha=.85,
                label=SWEEP_LABEL[c], edgecolors="k", linewidths=.5, zorder=3)
plt.xlabel("Retention (BWT, steady)"); plt.ylabel("Base-coverage retention")
plt.grid(alpha=.3); plt.legend(fontsize=6, loc="lower left")
plt.tight_layout(); plt.savefig(f"{FIG}/pareto_sweep.pdf"); plt.close()

# Fig: coverage-retention trajectories (full runs)
plt.figure(figsize=(4.2, 3.0))
for key, run in FULL.items():
    tr = covret(f"{LOG}/{run}_meta.jsonl", base_full, range(1, 31))
    plt.plot([c for c, _ in tr], [v for _, v in tr], label=FULL_LABEL[key])
plt.xlabel("Cycle"); plt.ylabel("Base-coverage retention"); plt.ylim(0.6, 1.02)
plt.grid(alpha=.3); plt.legend(fontsize=7); plt.tight_layout()
plt.savefig(f"{FIG}/covret_traj.pdf"); plt.close()

# Fig: diversity trajectories (collapse contrast)
plt.figure(figsize=(4.2, 3.0))
for key, run in FULL.items():
    df = pd.read_csv(f"{LOG}/{run}.csv")
    plt.plot(df.cycle, df.distinct_3, label=FULL_LABEL[key])
plt.xlabel("Cycle"); plt.ylabel("distinct-3 (current domain)")
plt.grid(alpha=.3); plt.legend(fontsize=7); plt.tight_layout()
plt.savefig(f"{FIG}/diversity_traj.pdf"); plt.close()

# Fig: budget evolution (nsiac)
df = pd.read_csv(f"{LOG}/{FULL['nsiac']}.csv")
plt.figure(figsize=(4.2, 3.0))
for ax_, lbl in [("w_explore", "explore"), ("w_stable", "stable"),
                 ("w_prec", "precision"), ("w_diff", "difficulty")]:
    plt.plot(df.cycle, df[ax_], label=lbl)
for b in (10, 20):
    plt.axvline(b, color="gray", ls="--", lw=.7)
plt.xlabel("Cycle"); plt.ylabel("Budget share $w$"); plt.grid(alpha=.3)
plt.legend(fontsize=7); plt.tight_layout()
plt.savefig(f"{FIG}/budget_traj.pdf"); plt.close()

with open(f"{HERE}/results_summary.json", "w") as f:
    json.dump(summary, f, indent=1, default=float)
print("OK: results_main.tex, results_sweep.tex, figs/{pareto_sweep,covret_traj,diversity_traj,budget_traj}.pdf, results_summary.json")
print(json.dumps(summary["spearman"], indent=1))
