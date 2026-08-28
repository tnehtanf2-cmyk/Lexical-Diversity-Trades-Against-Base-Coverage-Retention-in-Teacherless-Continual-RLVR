"""Evaluator (실험설계 §6): pass@k (unbiased) + coverage@k (both definitions),
diversity/collapse (distinct-n, self-overlap, answer-cluster entropy, repetition),
and plasticity (dormant fraction, effective rank, param-norm growth, weight drift).
Also returns per-sample Q&A for jsonl logging.
"""
from __future__ import annotations
import math
import itertools
import random
import numpy as np
import torch
from collections import Counter

from . import reward_arbiter as RA


# ---------------- pass@k (Chen et al., 2021 unbiased estimator) ----------------
def pass_at_k(n: int, c: int, k: int) -> float:
    if k > n:
        raise ValueError(f"pass@k needs k<=n (got k={k}, n={n})")
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod([(n - c - i) / (n - i) for i in range(k)]))


def _correct_vec(prompt, completions, domain_type, row):
    if domain_type == "numeric":
        return [RA.numeric_correct(x, row.get("answer")) for x in completions]
    if domain_type == "mc":
        return [RA.mc_correct(x, row.get("answer")) for x in completions]
    if domain_type == "code":
        return [RA.code_correct(prompt, x, row.get("test", ""),
                                row.get("entry_point", "")) for x in completions]
    return [0.0] * len(completions)   # non-verifiable: pass@k undefined


# ---------------- diversity / collapse ----------------
def distinct_n(texts, n=3):
    grams, tot = set(), 0
    for t in texts:
        toks = t.split()
        gs = list(zip(*[toks[i:] for i in range(n)]))
        grams.update(gs); tot += max(1, len(gs))
    return len(grams) / max(1, tot)


def self_overlap(texts, n=100):
    s = texts if len(texts) <= n else random.sample(texts, n)
    if len(s) < 2:
        return 0.0
    def jac(a, b):
        A, B = set(a.split()), set(b.split())
        return len(A & B) / max(1, len(A | B))
    return float(np.mean([jac(a, b) for a, b in itertools.combinations(s, 2)]))


def repetition_rate(texts, n=3):
    reps = []
    for t in texts:
        toks = t.split()
        gs = list(zip(*[toks[i:] for i in range(n)]))
        if gs:
            reps.append(1.0 - len(set(gs)) / len(gs))
    return float(np.mean(reps)) if reps else 0.0


def answer_cluster_entropy(keys):
    """Semantic-entropy proxy over one problem's answer clusters."""
    cnt = Counter(keys); tot = sum(cnt.values())
    return float(-sum((v / tot) * math.log(v / tot + 1e-12) for v in cnt.values()))


# ---------------- plasticity ----------------
@torch.no_grad()
def probe_last_hidden(model, tok, texts, max_len=128):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=max_len).to(model.device)
    out = model(**enc, output_hidden_states=True)
    h = out.hidden_states[-1]                    # [B, T, H]
    mask = enc.attention_mask.unsqueeze(-1).float()
    return (h * mask).sum(1) / mask.sum(1).clamp(min=1)   # [B, H] mean over tokens


@torch.no_grad()
def plasticity_metrics(model, theta0: dict | None, probe_hidden: torch.Tensor | None):
    total_sq, drift_sq, n = 0.0, 0.0, 0
    n_moved = 0                      # coordinates that actually changed since theta0
    moved_lo = moved_hi = 0          # split by |theta0| around the bf16 plasticity threshold
    n_lo = n_hi = 0
    for name, p in model.named_parameters():
        pf = p.detach().float()
        total_sq += float(pf.pow(2).sum()); n += p.numel()
        if theta0 is not None and name in theta0:
            # upcast the (possibly bf16) snapshot per-tensor at compare time, so no
            # second full-model fp32 copy is ever materialized (v2i scale fix)
            t0 = theta0[name].float()
            d = pf.cpu() - t0
            drift_sq += float(d.pow(2).sum())
            # Realized plasticity under round-to-nearest storage. In pure bf16 an update
            # smaller than half an ulp(theta) is discarded, and ulp scales with |theta|,
            # so which coordinates move is itself an observable (weight_drift alone cannot
            # distinguish "many tiny moves" from "few one-ulp jumps"). LO/HI split at the
            # single-step threshold |theta| < 256*lr for a unit-normalized Adam step at
            # lr=1.5e-5, i.e. ~3.8e-3 (see 실험기록 v2i precision analysis).
            # chunked so the boolean intermediates stay ~MB even for the 7B embedding
            # (host memory peaked at 120/121 GB in the 7B smoke; no new headroom to spend)
            df, t0f = d.reshape(-1), t0.reshape(-1)
            CH = 1 << 23
            for i in range(0, df.numel(), CH):
                dc, tc = df[i:i + CH], t0f[i:i + CH]
                mv = dc != 0
                lo = tc.abs() < 3.8e-3
                n_moved += int(torch.count_nonzero(mv))
                n_lo += int(torch.count_nonzero(lo))
                moved_lo += int(torch.count_nonzero(mv & lo))
            n_hi = n - n_lo
            moved_hi = n_moved - moved_lo
    out = {"param_norm": math.sqrt(total_sq),
           "weight_drift": (math.sqrt(drift_sq) / math.sqrt(max(1, n))) if theta0 else 0.0,
           "changed_frac": (n_moved / max(1, n)) if theta0 else 0.0,
           "changed_frac_small": (moved_lo / n_lo) if theta0 and n_lo else 0.0,
           "changed_frac_large": (moved_hi / n_hi) if theta0 and n_hi else 0.0,
           "dormant_frac": float("nan"), "effective_rank": float("nan")}
    if probe_hidden is not None and probe_hidden.ndim == 2 and probe_hidden.shape[0] > 1:
        H = probe_hidden.float()
        out["dormant_frac"] = float((H.std(dim=0) < 1e-3).float().mean())
        try:
            sv = torch.linalg.svdvals(H - H.mean(0, keepdim=True))
            p = sv / (sv.sum() + 1e-8)
            out["effective_rank"] = float(torch.exp(-(p * (p + 1e-12).log()).sum()))
        except Exception:
            pass
    return out


def snapshot_params(model, dtype=None):
    """θ0 reference for weight_drift. `dtype=None` keeps the model's own dtype.

    Scale note (v2i): for a bf16 model, storing the snapshot in bf16 and upcasting at
    compare time is bit-identical to storing an fp32 upcast of the same bf16 values, so
    the metric is unchanged — it only halves host memory (7B: 30.5GB -> 15.2GB), which
    is what makes 7B fit in the 121GB unified memory. Pass torch.float32 to reproduce the
    v2h call exactly (required if the model itself is fp32, i.e. --dtype fp32)."""
    return {n: (p.detach().float() if dtype is None and p.dtype == torch.float32
                else p.detach() if dtype is None else p.detach().to(dtype)).cpu().clone()
            for n, p in model.named_parameters()}


# ---------------- main domain evaluation ----------------
@torch.no_grad()
def harvest_holdout(model, tok, eval_ds, domain_type, start, n, n_samples=8,
                    max_new_tokens=256, temperature=1.0, cap=60):
    """Collect verifier-correct traces from eval-split rows [start, start+n) — rows that are
    NEVER scored, since evaluate_domain always measures the deterministic prefix [0, subset).

    Why this exists: by default the rehearsal buffer is filled from the *measured* rollouts
    (zero extra generation cost), which means past-domain retention is measured on items the
    rehearsal pass trained on. That confounds transfer with re-exposure. Harvesting from a
    disjoint holdout slice removes the confound at the price of one extra generation pass, and
    lets the size of the confound be quantified by differencing the two configurations.
    Deduplication matches the default path (120-char completion prefix, <=2 per prompt)."""
    if n <= 0 or start >= len(eval_ds):
        return []
    rows = eval_ds.select(range(start, min(start + n, len(eval_ds))))
    model.eval()
    out_traces = []
    for row in rows:
        if len(out_traces) >= cap:
            break
        prompt = row["prompt"]
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        gen = model.generate(**enc, do_sample=True, temperature=max(temperature, 0.3),
                             top_p=0.95, num_return_sequences=n_samples,
                             max_new_tokens=max_new_tokens,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        comps = [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True) for o in gen]
        corr = _correct_vec(prompt, comps, domain_type, row)
        keys, picks = set(), []
        for i in range(len(comps)):
            if corr[i] > 0:
                k_ = comps[i][:120]
                if k_ not in keys:
                    keys.add(k_); picks.append(prompt + comps[i])
            if len(picks) >= 2:
                break
        out_traces += picks
    return out_traces[:cap]


@torch.no_grad()
def evaluate_domain(model, tok, eval_ds, domain_type, n_samples=64, ks=(1, 4, 16, 64),
                    subset=100, max_new_tokens=256, temperature=1.0, collect_samples=8):
    assert max(ks) <= n_samples, f"max(ks)={max(ks)} > n_samples={n_samples}"
    model.eval()
    rows = eval_ds.select(range(min(subset, len(eval_ds))))
    passk = {k: [] for k in ks}
    cover_solved, cover_solutions, entos, all_texts, samples = [], [], [], [], []
    correct_traces = []   # reuse eval rollouts as a zero-extra-cost rehearsal source (any condition)
    for j, row in enumerate(rows):
        prompt = row["prompt"]
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        out = model.generate(**enc, do_sample=True, temperature=max(temperature, 0.3),
                             top_p=0.95, num_return_sequences=n_samples,
                             max_new_tokens=max_new_tokens,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        comps = [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True) for o in out]
        all_texts.extend(comps)
        corr = _correct_vec(prompt, comps, domain_type, row)
        c = int(sum(x > 0 for x in corr))
        for k in ks:
            passk[k].append(pass_at_k(n_samples, c, k))
        if len(correct_traces) < 60:
            # solution-deduplicated harvest (PAT v2h): dedupe by 120-char completion prefix —
            # the same signature as the distinct-solution metric — keeping up to 2 DISTINCT
            # verified solutions per prompt. (Answer-mode keys degenerate on numeric/mc:
            # every correct completion shares the gold answer -> one key.) This preserves
            # per-prompt solution-string coverage in the per-domain rehearsal buffer.
            keys, picks = set(), []
            for i in range(len(comps)):
                if corr[i] > 0:
                    k_ = comps[i][:120]
                    if k_ not in keys:
                        keys.add(k_); picks.append(prompt + comps[i])
                if len(picks) >= 2:
                    break
            correct_traces = (correct_traces + picks)[:60]   # hard cap (was off-by-one)
        cover_solved.append(1.0 if c > 0 else 0.0)
        cover_solutions.append(len({comps[i][:120] for i in range(len(comps)) if corr[i] > 0}))
        entos.append(answer_cluster_entropy([RA._answer_key(x) for x in comps]))
        if j < collect_samples:
            samples.append({"prompt": prompt, "gold": str(row.get("answer", row.get("test", "")))[:200],
                            "completions": comps[:4], "correct": corr[:4]})
    m = {
        "pass@1": float(np.mean(passk[min(ks)])),
        **{f"pass@{k}": float(np.mean(v)) for k, v in passk.items()},
        "coverage_solved": float(np.mean(cover_solved)) if cover_solved else 0.0,  # (ii) solve-rate
        "coverage_solutions": float(np.mean(cover_solutions)) if cover_solutions else 0.0,  # (i) distinct
        "distinct_3": distinct_n(all_texts, 3),
        "self_overlap": self_overlap(all_texts),
        "repetition": repetition_rate(all_texts),
        "gen_entropy": float(np.mean(entos)) if entos else 0.0,
    }
    m["correct_traces"] = correct_traces
    # per-prompt solve vector (deterministic row order) -> enables base-coverage-retention
    # (trilemma axis 3): compare which prompts the frozen base solves vs the policy.
    m["solved_vec"] = [int(x) for x in cover_solved]
    return m, samples
