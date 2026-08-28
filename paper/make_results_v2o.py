#!/usr/bin/env python3
"""Harvest-source ablation table (v2o vs v2k) + multi-seed table (v2n).

v2o = same pipeline, model (1.5B), seed (0) and repaired verifier as v2k; the ONLY
difference is `--harvest_source eval` (rehearsal buffer filled from measured evaluation
rollouts) vs v2k's `train`. Single-variable contrast for the contamination channel.
Preregistration + verdict: wiki/방법론-사전등록-수확절제.md (2026-08-21).

v2n = seeds 1,2 for 1.5B x {nsiac,fixedmid,sftstar,noreh}, 3B x {nsiac,fixedmid},
7B x nsiac; seed 0 is v2k (1.5B) / v2l (3B, 7B).
Preregistration + verdict: wiki/방법론-사전등록-멀티시드.md (2026-08-20).

Outputs
  paper/results_harvest.tex       -- tab:harvest  (drift + BWT, train vs eval)
  paper/results_seeds.tex         -- tab:seeds    (avg_final + BWT, seeds 0/1/2)
  paper/results_summary_v2o.json  -- every number quoted in prose
Usage: python3 paper/make_results_v2o.py
"""
import os, json, csv
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(os.path.dirname(HERE), "logs")
STEADY = set(range(21, 31))


def load(stem):
    return [r for r in csv.DictReader(open(f"{LOG}/{stem}.csv")) if r["cycle"] != "cycle"]


ENV = {"1.5B": "full_base_env", "3B": "v2i_3b_base_env", "7B": "v2i_7b_base_env"}
_env_cache = {}


def _base(scale):
    if scale not in _env_cache:
        base = {}
        for line in open(f"{LOG}/{ENV[scale]}_meta.jsonl"):
            j = json.loads(line)
            for d, v in j.get("solved", {}).items():
                base[d] = base.get(d, set()) | {i for i, x in enumerate(v) if x > 0}
        _env_cache[scale] = base
    return _env_cache[scale]


def covret(stem, scale):
    """Steady-state mean retained fraction of the frozen base's solved set
    (same definition as make_results_v2h.covret, per-scale envelope)."""
    base = _base(scale)
    xs = []
    for line in open(f"{LOG}/{stem}_meta.jsonl"):
        j = json.loads(line)
        if j["cycle"] in STEADY:
            vals = [len({i for i, x in enumerate(v) if x > 0} & base[d]) / len(base[d])
                    for d, v in j.get("solved", {}).items() if base.get(d)]
            if vals:
                xs.append(st.mean(vals))
    return st.mean(xs)


def at(rows, c):
    return next(r for r in rows if int(r["cycle"]) == c)


def drift(stem):
    """Code accuracy, first vs last cycle of the MedicalMC block (= make_results_v2k
    post_shift_drift: acc_Code(c30) - acc_Code(c21))."""
    rows = load(stem)
    return float(at(rows, 30)["acc_Code"]) - float(at(rows, 21)["acc_Code"])


def bwt(stem):
    rows = load(stem)
    return st.mean(float(r["bwt"]) for r in rows if int(r["cycle"]) in STEADY)


def avg_final(stem):
    r = load(stem)[-1]
    return st.mean(float(r["acc_" + d]) for d in ["GSM8K", "Code", "MedicalMC"])


def code_block_bwt(stem):
    """Code-block BWT: the csv bwt column over cycles 11-20 (GSM8K anchored at its
    block end, cycle 10). Returns (#negative cycles of 10, mean)."""
    rows = load(stem)
    ds = [float(at(rows, c)["bwt"]) for c in range(11, 21)]
    return sum(d < 0 for d in ds), st.mean(ds)

COND = [("nsiac", "MAPC (learned multi-axis)"),
        ("fixedmid", "Fixed mid-point GRPO"),
        ("sftstar", "STaR/SFT self-distillation"),
        ("noreh", "MAPC, rehearsal ablated")]

summary = {"harvest": {}, "seeds": {}, "sevenB_codeneg": {}}

# ---------------- harvest-source ablation (tab:harvest) ----------------
rows = []
for k, lbl in COND:
    tr_d, ev_d = drift(f"v2k_{k}"), drift(f"v2o_{k}")
    tr_b, ev_b = bwt(f"v2k_{k}"), bwt(f"v2o_{k}")
    summary["harvest"][k] = dict(train_drift=tr_d, eval_drift=ev_d, train_bwt=tr_b, eval_bwt=ev_b)
    rows.append(f"{lbl} & {tr_d:+.3f} & {ev_d:+.3f} & {tr_b:+.3f} & {ev_b:+.3f} \\\\")
open(os.path.join(HERE, "results_harvest.tex"), "w").write(r"""\begin{table}[t]
\caption{Harvest-source ablation (1.5B, seed 0; the only difference
between the paired columns is whether the rehearsal buffer is harvested from
inner-loop \emph{train}-split rollouts or from \emph{measured evaluation} rollouts).
Post-shift drift is Code accuracy at cycle 30 minus cycle 21; BWT is averaged over
cycles 21--30.}
\label{tab:harvest}
\centering
\begin{tabular}{@{}lcccc@{}}
\toprule
 & \multicolumn{2}{c}{Post-shift Code drift} & \multicolumn{2}{c}{BWT} \\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
Condition & train & eval & train & eval \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- multi-seed table (tab:seeds) ----------------
SEEDS = {("1.5B", k): [f"v2k_{k}", f"v2n_{k}_s1", f"v2n_{k}_s2"] for k, _ in COND}
SEEDS[("3B", "nsiac")] = ["v2l_3b_nsiac", "v2n_3b_nsiac_s1", "v2n_3b_nsiac_s2"]
SEEDS[("3B", "fixedmid")] = ["v2l_3b_fixedmid", "v2n_3b_fixedmid_s1", "v2n_3b_fixedmid_s2"]
SEEDS[("7B", "nsiac")] = ["v2l_7b_nsiac", "v2n_7b_nsiac_s1", "v2n_7b_nsiac_s2"]
LBL = dict(COND)
rows = []
for (scale, k), stems in SEEDS.items():
    av = [avg_final(s) for s in stems]
    bw = [bwt(s) for s in stems]
    cv = [covret(s, scale) for s in stems]
    summary["seeds"][f"{scale}_{k}"] = dict(avg_final=av, bwt=bw, covret=cv)
    rows.append(f"{scale} & {LBL[k]} & " + " / ".join(f"{v:.3f}" for v in av)
                + " & " + " / ".join(f"{v:+.3f}" for v in bw)
                + " & " + " / ".join(f"{v:.3f}" for v in cv) + r" \\")
open(os.path.join(HERE, "results_seeds.tex"), "w").write(r"""\begin{table}[t]
\caption{Multi-seed replication (seeds 0/1/2; seed-wise values listed, no Gaussian
$\pm$). Seed 0 is the main-table run set at 1.5B and the scale-ladder run set at 3B/7B.
avg is the three-domain mean accuracy at cycle 30; BWT and Cov.\ ret.\ (retained fraction of
the frozen base's solved set, per-scale envelope) are averaged over cycles 21--30.}
\label{tab:seeds}
\centering\small
\begin{tabular}{@{}llccc@{}}
\toprule
Scale & Condition & avg (s0/s1/s2) & BWT (s0/s1/s2) & Cov.\ ret.\ (s0/s1/s2) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- 7B code-block BWT per seed (prose) ----------------
for s in ["v2l_7b_nsiac", "v2n_7b_nsiac_s1", "v2n_7b_nsiac_s2"]:
    n, m = code_block_bwt(s)
    summary["sevenB_codeneg"][s] = dict(neg_cycles=n, mean_block_bwt=m)

json.dump(summary, open(os.path.join(HERE, "results_summary_v2o.json"), "w"), indent=1)
print(json.dumps(summary["sevenB_codeneg"], indent=1))
print("seeds keys:", list(summary["seeds"]))
