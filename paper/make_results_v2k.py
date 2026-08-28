#!/usr/bin/env python3
"""Main table for the 30-cycle continual-story runs (v2k set).

Set: v2k_{nsiac,fixedmid,sftstar,noreh} -- full3, 1.5B, seed 0, train-split rehearsal
harvest, per-item execution-prefix code verifier (Code training reward nonzero in 40/40
logged steps of every GRPO run). The eval-harvest arm of the harvest-source ablation
(v2o) and the multi-seed replication (v2n) are handled by make_results_v2o.py.

Outputs (only when all four runs have 30 cycles; otherwise stdout preview only)
  paper/results_main.tex          -- main table
  paper/results_summary_v2k.json  -- every number quoted in prose

Usage: python3 paper/make_results_v2k.py
"""
import os, json, csv
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG = os.path.join(ROOT, "logs")

# covret must come from the per-cycle `solved` vectors in *_meta.jsonl against the
# frozen-base envelope -- NOT the CSV's coverage_solved column (different quantity;
# this exact confusion produced two wrong conclusions on 2026-08-03). Reuse the v2h
# helpers verbatim so the definition cannot drift between tables.
_src = open(os.path.join(HERE, "make_results_v2h.py")).read()
_head = _src.split("# ---------------- main continual table ----------------")[0]
_v2h = {"__file__": os.path.join(HERE, "make_results_v2h.py")}
exec(compile(_head, "make_results_v2h.py", "exec"), _v2h)
base_sets, covret_mean = _v2h["base_sets"], _v2h["covret_mean"]
BASE = base_sets(os.path.join(LOG, "full_base_env_meta.jsonl"))

DOMS = ["GSM8K", "Code", "MedicalMC"]
STEADY = range(21, 31)
BLOCK_END = {"GSM8K": 10, "Code": 20, "MedicalMC": 30}

# (key, csv stem, needs restart-split, label)
V2K = [("nsiac",    "v2k_nsiac",    False, "MAPC (learned multi-axis)"),
       ("fixedmid", "v2k_fixedmid", False, "Fixed mid-point GRPO"),
       ("sft_star", "v2k_sftstar",  False, "STaR/SFT self-distillation"),
       ("no_reh",   "v2k_noreh",    False, "MAPC, rehearsal ablated")]
V2J = {"nsiac": ("v2j_nsiac", False), "fixedmid": ("v2j_fixedmid", False),
       "sft_star": ("v2j_sftstar", False), "no_reh": ("v2j_noreh", True)}
V2H = {"nsiac": ("v2_full_nsiac_v2h", False), "fixedmid": ("v2_full_fixedmid_v2h", False),
       "sft_star": ("v2_full_sft_star_v2h", False), "no_reh": ("v2_full_nsiac_no_rehearsal", False)}


def load(stem, split_restart=False):
    """Read a run CSV. `split_restart` keeps only the rows after the cycle counter
    resets (two attempts appended into one file; v2j_noreh only). The v2k_sftstar
    interruption of 2026-08-06 does NOT need this: the launcher guard rotated the
    partial attempt to v2k_sftstar_part1.* before the rerun, so the live file holds a
    single clean attempt."""
    rows = [r for r in csv.DictReader(open(f"{LOG}/{stem}.csv")) if r["cycle"] != "cycle"]
    if split_restart:
        cyc = [int(r["cycle"]) for r in rows]
        cut = next((i for i in range(1, len(cyc)) if cyc[i] < cyc[i - 1]), None)
        if cut is not None:
            rows = rows[cut:]
    return rows


def _split_meta(path):
    """Restart-split copy of a meta.jsonl (mirrors load(..., split_restart=True))."""
    lines = [l for l in open(path) if l.strip()]
    cyc = [json.loads(l)["cycle"] for l in lines]
    cut = next((i for i in range(1, len(cyc)) if cyc[i] < cyc[i - 1]), None)
    if cut is None:
        return path
    out = path.replace(".jsonl", "_restartsplit.jsonl")
    with open(out, "w") as f:
        f.writelines(lines[cut:])
    return out


def num(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return None


def by_cycle(rows):
    return {int(r["cycle"]): r for r in rows}


def summarize(stem, split_restart=False):
    """Steady-state summary for one run. `covret` uses the frozen-base envelope helper."""
    rows = load(stem, split_restart)
    at = by_cycle(rows)
    steady = [r for r in rows if int(r["cycle"]) in STEADY]
    last = rows[-1]
    finals = {d: num(last, f"acc_{d}") for d in DOMS}
    rr = {}
    for d in DOMS[:2]:
        ii = num(at.get(BLOCK_END[d], {}), f"acc_{d}")
        rr[d] = (finals[d] / ii) if (ii and finals[d] is not None) else float("nan")
    mean = lambda k: st.mean([v for v in (num(r, k) for r in steady) if v is not None])
    mp = f"{LOG}/{stem}_meta.jsonl"
    if not os.path.exists(mp) and stem.endswith("_part1"):
        mp = f"{LOG}/{stem[:-6]}_meta_part1.jsonl"
    if not os.path.exists(mp):
        cov = float("nan")
    else:
        cov = covret_mean(_split_meta(mp) if split_restart else mp, BASE, set(STEADY))
    return dict(avg_final=st.mean([v for v in finals.values() if v is not None]),
                **{f"final_{d}": finals[d] for d in DOMS},
                bwt=mean("bwt"), dist3=mean("distinct_3"), entropy=mean("gen_entropy"),
                covret=cov, rr_gsm=rr["GSM8K"], rr_code=rr["Code"],
                n_cycles=len(rows))


def post_shift_drift(stem, split_restart=False):
    """Code accuracy at the first vs last cycle of the MedicalMC block."""
    at = by_cycle(load(stem, split_restart))
    a, b = num(at.get(21, {}), "acc_Code"), num(at.get(30, {}), "acc_Code")
    return a, b, (b - a if (a is not None and b is not None) else None)


def complete(stem):
    try:
        rows = load(stem)
        return len(rows) >= 30 and max(int(r["cycle"]) for r in rows) == 30
    except FileNotFoundError:
        return False


ready = {k: complete(stem) for k, stem, _, _ in V2K}
PREVIEW = not all(ready.values())

summary = {"main": {}}

# ---------------- main table (v2k) ----------------
rows_main = [(k, lbl, summarize(stem, sr)) for k, stem, sr, lbl in V2K if ready[k]]
summary["main"] = {k: v for k, _, v in rows_main}

if not PREVIEW:
    best = max(rows_main, key=lambda t: t[2]["avg_final"])[0]
    with open(f"{HERE}/results_main.tex", "w") as f:
        f.write("\\begin{table*}[t]\n\\caption{Continual-story runs (30 cycles, GSM8K$\\to$Code$\\to$MedicalMC,"
                " full protocol, train-split rehearsal harvest)."
                " Steady state = cycles 21--30."
                " Cov.\\ ret.\\ = retained fraction of the frozen base's solved set."
                " $R_f/R_{ii}$ = final over block-end accuracy (relative retention)."
                " Seed 0 per condition; Table~\\ref{tab:seeds} lists the three-seed replication"
                " and Table~\\ref{tab:harvest} the harvest-source ablation.}\n"
                "\\label{tab:main}\n\\centering\\footnotesize\n"
                "\\begin{tabular}{@{}lccccccccc@{}}\n\\toprule\n"
                "Method & Avg.\\ final & GSM8K & Code & MedMC & BWT & $R_f/R_{ii}$ G/C & distinct-3 &"
                " entropy & Cov.\\ ret. \\\\\n\\midrule\n")
        for key, label, r in rows_main:
            hi = "\\textbf" if key == best else ""
            f.write(f"{label} & {hi}{{{r['avg_final']:.3f}}} & {r['final_GSM8K']:.3f} & "
                    f"{r['final_Code']:.3f} & {r['final_MedicalMC']:.3f} & {r['bwt']:.3f} & "
                    f"{r['rr_gsm']:.2f}/{r['rr_code']:.2f} & {r['dist3']:.3f} & {r['entropy']:.2f} & "
                    f"{r['covret']:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

# ---------------- figures (regenerated from the v2k set) ----------------
if not PREVIEW:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG = os.path.join(HERE, "figs")
    os.makedirs(FIG, exist_ok=True)

    plt.figure(figsize=(4.2, 3.0))
    for _, stem, sr, lbl in V2K:
        mp = f"{LOG}/{stem}_meta.jsonl"
        if not os.path.exists(mp):
            continue
        pts = _v2h["covret"](mp, BASE, set(range(1, 31)))
        plt.plot([c for c, _ in pts], [v for _, v in pts], label=lbl, lw=1.2)
    for b in (10, 20):
        plt.axvline(b, color="gray", ls="--", lw=.7)
    plt.xlabel("Cycle"); plt.ylabel("Base-coverage retention")
    plt.grid(alpha=.3); plt.legend(fontsize=6); plt.tight_layout()
    plt.savefig(f"{FIG}/covret_traj.pdf"); plt.close()

    plt.figure(figsize=(4.2, 3.0))
    for _, stem, sr, lbl in V2K:
        rows = load(stem, sr)
        plt.plot([int(r["cycle"]) for r in rows], [num(r, "distinct_3") for r in rows], label=lbl, lw=1.2)
    for b in (10, 20):
        plt.axvline(b, color="gray", ls="--", lw=.7)
    plt.xlabel("Cycle"); plt.ylabel("distinct-3 (current domain)")
    plt.grid(alpha=.3); plt.legend(fontsize=6); plt.tight_layout()
    plt.savefig(f"{FIG}/diversity_traj.pdf"); plt.close()

    rows = load("v2k_nsiac")
    plt.figure(figsize=(4.2, 3.0))
    for col, lbl in [("w_explore", "explore"), ("w_stable", "stable"),
                     ("w_prec", "precision"), ("w_diff", "difficulty")]:
        plt.plot([int(r["cycle"]) for r in rows], [num(r, col) for r in rows], label=lbl, lw=1.2)
    for b in (10, 20):
        plt.axvline(b, color="gray", ls="--", lw=.7)
    plt.xlabel("Cycle"); plt.ylabel("Budget share $w$"); plt.grid(alpha=.3)
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(f"{FIG}/budget_traj.pdf"); plt.close()

    with open(f"{HERE}/results_summary_v2k.json", "w") as f:
        json.dump(summary, f, indent=1)

# ---------------- report ----------------
if PREVIEW:
    missing = [k for k, ok in ready.items() if not ok]
    print(f"PREVIEW ONLY — incomplete runs: {missing}. No .tex/.json written.")
else:
    print("OK: results_main.tex (v2k), results_summary_v2k.json")
for key, label, r in rows_main:
    print(f"  {key:<9} avg {r['avg_final']:.3f}  bwt {r['bwt']:+.3f}  dist3 {r['dist3']:.3f}  "
          f"covret {r['covret']:.3f}  ({r['n_cycles']} cycles)")
