#!/usr/bin/env python3
"""Sweep, scale, and ablation tables (v2k main set + v2l runs).

Set (all: train-split harvest + pinned deps)
  main 30-cycle 1.5B : v2k_{nsiac,fixedmid,sftstar,noreh}
  sweep 15-cycle     : sweep2_{cond}_s{0,1,2}
  ZPD ablation       : v2l_nozpd            (seed-paired with v2k_nsiac)
  scale              : v2l_3b_{nsiac,fixedmid,sftstar}, v2l_7b_nsiac
Outputs
  paper/results_sweep.tex          paper/results_scale.tex
  paper/results_ablation.tex
  paper/results_summary_v2l.json   (every number quoted in prose)
Usage: python3 paper/make_results_v2l.py
"""
import json, os
import statistics as st
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG = os.path.join(ROOT, "logs")

# reuse the covret helpers verbatim (definition must not drift between tables)
_src = open(os.path.join(HERE, "make_results_v2h.py")).read()
_head = _src.split("# ---------------- main continual table ----------------")[0]
_v2h = {"__file__": os.path.join(HERE, "make_results_v2h.py")}
exec(compile(_head, "make_results_v2h.py", "exec"), _v2h)
base_sets, covret_mean = _v2h["base_sets"], _v2h["covret_mean"]

DOMS = ["GSM8K", "Code", "MedicalMC"]
ST30, ST15 = set(range(21, 31)), set(range(11, 16))
LBL = {"nsiac": "MAPC", "fixedmid": "Fixed mid-point", "sftstar": "STaR/SFT"}

ENV = {"1.5B": "full_base_env", "3B": "v2i_3b_base_env", "7B": "v2i_7b_base_env",
       "sweep": "sweep_base_env"}
_env_cache = {}


def env(key):
    if key not in _env_cache:
        _env_cache[key] = base_sets(f"{LOG}/{ENV[key]}_meta.jsonl")
    return _env_cache[key]


def run(stem, envkey, steady=ST30, doms=DOMS):
    """One run's summary. Returns None if the log is absent (never silently zero)."""
    path = f"{LOG}/{stem}.csv"
    if not os.path.exists(path):
        return None
    rows = [r for r in pd.read_csv(path).to_dict("records") if str(r["cycle"]) != "cycle"]
    at = {int(r["cycle"]): r for r in rows}
    last, S = rows[-1], [r for r in rows if int(r["cycle"]) in steady]
    fin = {d: float(last[f"acc_{d}"]) for d in doms}
    cb = [float(at[c]["bwt"]) for c in range(11, 21) if c in at]   # code-block window
    return dict(
        avg=st.mean(fin.values()), **{f"final_{d}": fin[d] for d in doms},
        bwt=st.mean(float(r["bwt"]) for r in S),
        d3=st.mean(float(r["distinct_3"]) for r in S),
        entropy=st.mean(float(r["gen_entropy"]) for r in S),
        covret=covret_mean(f"{LOG}/{stem}_meta.jsonl", env(envkey), steady),
        # post-shift Code drift: the re-exposure signature
        drift=(float(at[max(steady)]["acc_Code"]) - float(at[min(steady)]["acc_Code"]))
        if min(steady) in at and max(steady) in at else float("nan"),
        cb_bwt=(st.mean(cb) if cb else float("nan")),
        cb_neg=sum(1 for v in cb if v < 0), n=len(rows))


S = {}   # summary bucket


# ---------------- sweep (clean) ----------------
CONDS = ["nsiac", "ent_lo", "ent_hi", "kl_lo", "kl_hi", "fixed_mid"]
SWEEP_LBL = {"nsiac": "learned (multi-axis)", "ent_lo": "entropy low", "ent_hi": "entropy high",
             "kl_lo": "KL low", "kl_hi": "KL high", "fixed_mid": "fixed mid-point"}
sweep = {}
for c in CONDS:
    per = {"ret": [], "div": [], "cov": []}
    for s in (0, 1, 2):
        df = pd.read_csv(f"{LOG}/sweep2_{c}_s{s}.csv")
        blk = df[df.cycle.isin(ST15)]
        per["ret"].append(float(blk.bwt.mean()))
        per["div"].append(float(blk.distinct_3.mean()))
        per["cov"].append(covret_mean(f"{LOG}/sweep2_{c}_s{s}_meta.jsonl", env("sweep"), ST15))
    sweep[c] = {k: (st.mean(v), st.pstdev(v)) for k, v in per.items()}
S["sweep"] = {c: {k: list(v) for k, v in d.items()} for c, d in sweep.items()}

with open(f"{HERE}/results_sweep.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\caption{Single-lever sweep vs.\\ learned control"
            " (light protocol, 15 cycles, 3 paired seeds, mean$\\pm$sd; steady state ="
            " cycles 11--15).}\n"
            "\\label{tab:sweep}\n\\centering\n\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
            "Setting & BWT & distinct-3 & Cov.\\ ret. \\\\\n\\midrule\n")
    for c in CONDS:
        d = sweep[c]
        f.write(f"{SWEEP_LBL[c]} & {d['ret'][0]:.3f}$\\pm${d['ret'][1]:.3f} & "
                f"{d['div'][0]:.3f}$\\pm${d['div'][1]:.3f} & "
                f"{d['cov'][0]:.3f}$\\pm${d['cov'][1]:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# 3-axis seed-consistent dominance check (RQ0b)
per_seed = {c: {a: [] for a in ("ret", "div", "cov")} for c in CONDS}
for c in CONDS:
    for s in (0, 1, 2):
        df = pd.read_csv(f"{LOG}/sweep2_{c}_s{s}.csv"); blk = df[df.cycle.isin(ST15)]
        per_seed[c]["ret"].append(float(blk.bwt.mean()))
        per_seed[c]["div"].append(float(blk.distinct_3.mean()))
        per_seed[c]["cov"].append(covret_mean(f"{LOG}/sweep2_{c}_s{s}_meta.jsonl", env("sweep"), ST15))
S["dominance_pairs"] = [[a, b] for a in CONDS for b in CONDS if a != b and
                        all(all(per_seed[a][ax][i] > per_seed[b][ax][i] for i in range(3))
                            for ax in ("ret", "div", "cov"))]
S["sweep_extremes"] = {ax: max(CONDS, key=lambda c: sweep[c][ax][0]) for ax in ("ret", "div", "cov")}

# ---------------- scale ladder (clean) ----------------
SCALE = [("1.5B", [("nsiac", "v2k_nsiac"), ("fixedmid", "v2k_fixedmid"), ("sftstar", "v2k_sftstar")]),
         ("3B", [("nsiac", "v2l_3b_nsiac"), ("fixedmid", "v2l_3b_fixedmid"), ("sftstar", "v2l_3b_sftstar")]),
         ("7B", [("nsiac", "v2l_7b_nsiac")])]
S["scale"] = {}
with open(f"{HERE}/results_scale.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\caption{Scale replication (30 cycles, GSM8K$\\to$Code$\\to$MedicalMC,"
            " identical protocol and knob settings at every scale; steady state = cycles 21--30;"
            " seed 0 per cell; Table~\\ref{tab:seeds} lists the multi-seed replication). Cov.\\ ret.\\ is"
            " measured against the \\emph{same-scale} frozen-base envelope. Drift = post-shift Code"
            " movement (the quantity Table~\\ref{tab:harvest} uses as its re-exposure signature; all runs"
            " here use the train-split harvest).}\n"
            "\\label{tab:scale}\n\\centering\n\\begin{tabular}{@{}llcccccc@{}}\n\\toprule\n"
            "Scale & Method & Avg.\\ final & BWT & Drift & distinct-3 & Cov.\\ ret. \\\\\n\\midrule\n")
    for i, (sc, runs) in enumerate(SCALE):
        if i:
            f.write("\\midrule\n")
        for j, (key, stem) in enumerate(runs):
            r = run(stem, sc)
            if r is None:
                print(f"  ! missing {stem}"); continue
            S["scale"][f"{sc}_{key}"] = r
            head = sc if j == 0 else ""
            f.write(f"{head} & {LBL[key]} & {r['avg']:.3f} & {r['bwt']:.3f} & {r['drift']:+.3f} & "
                    f"{r['d3']:.3f} & {r['covret']:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---------------- ablations (clean, seed-paired with v2k_nsiac) ----------------
ABL = [("MAPC (full)", "v2k_nsiac"), ("\\;-- ZPD off", "v2l_nozpd"),
       ("\\;-- rehearsal off", "v2k_noreh")]
S["ablation"] = {}
with open(f"{HERE}/results_ablation.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\caption{RQ4 component ablations (full protocol, 30 cycles,"
            " seed-paired with the MAPC run; single seed)."
            " Steady state = cycles 21--30. The single-seed noise floor on per-domain pass@1 is"
            " $0.05$, so differences below that magnitude are not resolved.}\n"
            "\\label{tab:ablation}\n\\centering\n\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
            "Condition & BWT & distinct-3 & Cov.\\ ret. & Avg.\\ final \\\\\n\\midrule\n")
    for lbl, stem in ABL:
        r = run(stem, "1.5B")
        S["ablation"][stem] = r
        f.write(f"{lbl} & {r['bwt']:.3f} & {r['d3']:.3f} & {r['covret']:.3f} & {r['avg']:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


with open(f"{HERE}/results_summary_v2l.json", "w") as f:
    json.dump(S, f, indent=1)

print("OK: results_sweep.tex, results_scale.tex, results_ablation.tex, results_summary_v2l.json")
print(f"  3-axis seed-consistent dominance pairs: {S['dominance_pairs'] or 'NONE'}")
print(f"  sweep axis winners: {S['sweep_extremes']}")
for k in ("1.5B_nsiac", "3B_nsiac", "7B_nsiac"):
    if k in S["scale"]:
        r = S["scale"][k]
        print(f"  {k:<12} avg {r['avg']:.3f}  covret {r['covret']:.3f}  "
              f"code-block BWT {r['cb_bwt']:+.3f} ({r['cb_neg']}/10 negative)")
