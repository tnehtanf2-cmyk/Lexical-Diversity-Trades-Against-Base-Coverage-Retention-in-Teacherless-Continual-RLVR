#!/usr/bin/env python3
"""MCQA answer-crystallization analysis (RQ5 mechanism paragraph).

Data unit (what the sample logs actually contain): for each evaluation cycle the
evaluator stores the FIRST 8 eval prompts x the first 4 of the 16 completions
(evaluator.py: collect_samples=8, comps[:4]). All statistics below are therefore
computed on 4 logged samples per prompt over the 8 logged prompts per cycle,
steady cycles 21-30 (80 prompt-cycle units per run), and the paper states this.

Choice extraction = the pipeline's own extractor (nsiac2/reward_arbiter.py
extract_choice: last standalone A-E letter).

Outputs: paper/results_crystallization.json + stdout.
Usage: python3 paper/make_crystallization.py
"""
import json, math, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from nsiac2.reward_arbiter import extract_choice

RUNS = {"1.5B": "v2k_nsiac", "3B": "v2l_3b_nsiac", "7B": "v2l_7b_nsiac"}
# Multi-seed extension (RQ5 seed-dependence paragraph): all nine controller cells.
SEED_RUNS = {"1.5B": ["v2k_nsiac", "v2n_nsiac_s1", "v2n_nsiac_s2"],
             "3B":   ["v2l_3b_nsiac", "v2n_3b_nsiac_s1", "v2n_3b_nsiac_s2"],
             "7B":   ["v2l_7b_nsiac", "v2n_7b_nsiac_s1", "v2n_7b_nsiac_s2"]}
ENV = {"1.5B": "full_base_env", "3B": "v2i_3b_base_env", "7B": "v2i_7b_base_env"}
STEADY = set(range(21, 31))

out = {}
for scale, stem in RUNS.items():
    ents, det, det_wrong, n_units = [], 0, 0, 0
    for line in open(f"{ROOT}/logs/{stem}_samples.jsonl"):
        j = json.loads(line)
        if j.get("eval_domain") != "MedicalMC" or j.get("cycle") not in STEADY:
            continue
        comps = j.get("completions", [])
        if not comps:
            continue
        choices = [extract_choice(c) for c in comps]
        n = len(choices)
        n_units += 1
        hist = {}
        for c in choices:
            hist[c] = hist.get(c, 0) + 1
        ent = -sum((k / n) * math.log2(k / n) for k in hist.values())
        ents.append(ent)
        if len(hist) == 1:
            det += 1
            if not any(j.get("correct", [])):
                det_wrong += 1
    out[scale] = dict(run=stem, units=n_units, mean_entropy_bits=sum(ents) / len(ents),
                      deterministic_frac=det / n_units, deterministic_wrong=det_wrong)


def _crys(stem):
    ents, det, n_units = [], 0, 0
    for line in open(f"{ROOT}/logs/{stem}_samples.jsonl"):
        j = json.loads(line)
        if j.get("eval_domain") != "MedicalMC" or j.get("cycle") not in STEADY:
            continue
        comps = j.get("completions", [])
        if not comps:
            continue
        choices = [extract_choice(c) for c in comps]
        n = len(choices); n_units += 1
        hist = {}
        for c in choices:
            hist[c] = hist.get(c, 0) + 1
        ents.append(-sum((k / n) * math.log2(k / n) for k in hist.values()))
        if len(hist) == 1:
            det += 1
    return dict(entropy_bits=sum(ents) / len(ents), det_frac=det / n_units, n_units=n_units)


def _mcqa_covret(stem, scale):
    base = {}
    for line in open(f"{ROOT}/logs/{ENV[scale]}_meta.jsonl"):
        j = json.loads(line)
        for d, v in j.get("solved", {}).items():
            base[d] = base.get(d, set()) | {i for i, x in enumerate(v) if x > 0}
    xs = []
    for line in open(f"{ROOT}/logs/{stem}_meta.jsonl"):
        j = json.loads(line)
        if j["cycle"] in STEADY and "MedicalMC" in j.get("solved", {}):
            v = j["solved"]["MedicalMC"]
            xs.append(len({i for i, x in enumerate(v) if x > 0} & base["MedicalMC"]) / len(base["MedicalMC"]))
    return sum(xs) / len(xs)


def _spearman_tiecorrected(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den


seed_cells = []
for scale, stems in SEED_RUNS.items():
    for stem in stems:
        c = _crys(stem)
        c.update(scale=scale, run=stem, mcqa_covret=_mcqa_covret(stem, scale))
        seed_cells.append(c)
out["seed_cells"] = seed_cells
_x = [c["det_frac"] for c in seed_cells]
_y = [c["mcqa_covret"] for c in seed_cells]
out["coupling_spearman_tiecorrected"] = _spearman_tiecorrected(_x, _y)


def _exact_perm_p(x, y, rho_obs):
    """Two-sided exact permutation p over all 9! orderings of y."""
    import itertools
    n_ge = total = 0
    for perm in itertools.permutations(y):
        r = _spearman_tiecorrected(x, list(perm))
        total += 1
        if abs(r) >= abs(rho_obs) - 1e-12:
            n_ge += 1
    return n_ge / total


out["coupling_exact_perm_p"] = _exact_perm_p(_x, _y, out["coupling_spearman_tiecorrected"])

json.dump(out, open(f"{HERE}/results_crystallization.json", "w"), indent=1)
for s, v in out.items():
    if not isinstance(v, dict) or "mean_entropy_bits" not in v:
        continue  # seed_cells / coupling entries have their own printout below
    print(f"{s}: H={v['mean_entropy_bits']:.3f} bit  det={v['deterministic_frac']:.1%} "
          f"({v['units']} units)  det-wrong={v['deterministic_wrong']}")
for c in out["seed_cells"]:
    print(f"{c['scale']} {c['run']}: det={c['det_frac']:.1%} H={c['entropy_bits']:.2f} "
          f"MCQA-covret={c['mcqa_covret']:.3f}")
print(f"coupling (tie-corrected Spearman): {out['coupling_spearman_tiecorrected']:.3f}")
