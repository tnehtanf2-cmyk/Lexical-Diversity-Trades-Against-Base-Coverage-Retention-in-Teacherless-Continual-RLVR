"""N-SIAC v2 main loop — teacherless continual multi-domain self-improvement.
Per cycle: meta-controller allocates a Dirichlet plasticity budget -> RLVR knobs;
inner loop = GRPO (TRL) on self-generated rollouts scored by the arbitrated reward
source; then evaluate (pass@k / coverage@k / BWT / diversity / plasticity) and
update the meta policy. Base = Qwen2.5-1.5B (dense), Full FT.

Outputs per run:
  logs/<run>.csv            — cycle-level results (pass@k, BWT, diversity, budget, plasticity)
  logs/<run>_samples.jsonl  — model Q&A: prompt / gold / sampled completions / correctness
  logs/<run>_meta.jsonl     — per-cycle knobs + metrics
"""
import os, json, argparse, gc, random
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nsiac2.meta_controller import MetaController, budget_to_knobs, AXES
from nsiac2.state_tracker import StateTracker, STATE_DIM
from nsiac2.reward_arbiter import make_reward_fn
from nsiac2.harvest import TrainHarvest
from nsiac2 import domains as DM
from nsiac2 import evaluator as EV


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])  # fp32=rigor run
    # scale-ladder (v2i) memory knobs. Defaults reproduce v2h numerics exactly.
    p.add_argument("--drift_dtype", default="model", choices=["model", "fp32"])
    p.add_argument("--optim", default="adamw_torch")   # e.g. adamw_bnb_8bit (7B OOM rung 2)
    # Pareto-sweep mode: fix knobs to a grid point, bypassing the learned controller.
    # e.g. "entropy_coef=0.02,kl_beta=0.05,rehearsal_ratio=0.2,temperature=0.9"
    p.add_argument("--fixed_knobs", default="")
    p.add_argument("--run_name", default="nsiac_v2")
    p.add_argument("--order", nargs="+", default=["GSM8K", "Code", "MedicalMC"])
    p.add_argument("--shift_interval", type=int, default=10)
    p.add_argument("--max_cycles", type=int, default=30)
    p.add_argument("--n_train", type=int, default=64)
    p.add_argument("--num_gen", type=int, default=8)          # GRPO group size
    p.add_argument("--grad_accum", type=int, default=4)       # aggregate prompt-groups/step
    p.add_argument("--grpo_steps", type=int, default=20)
    p.add_argument("--eval_subset", type=int, default=100)
    p.add_argument("--eval_nsamples", type=int, default=16)
    p.add_argument("--ks", nargs="+", type=int, default=[1, 4, 16])
    p.add_argument("--baseline", default="nsiac", choices=["nsiac", "fixed_grpo", "sft_star"])
    p.add_argument("--seed", type=int, default=0)   # paired-seed comparisons for the Pareto sweep
    p.add_argument("--eval_only", action="store_true")  # frozen-base pass@k envelope measurement
    p.add_argument("--no_zpd", action="store_true")     # RQ4 ablation: disable ZPD seed selection
    p.add_argument("--no_rehearsal", action="store_true")  # RQ4 ablation: disable rehearsal SFT
    # Legacy contamination control (superseded by --harvest_source): harvest rehearsal traces
    # from eval-split rows AFTER the measured prefix. See evaluator.harvest_holdout.
    p.add_argument("--harvest_holdout", type=int, default=0)
    # Harvest source for rehearsal traces (identical dedup rules; zero extra generation).
    # 'train' (DEFAULT) = the PUBLISHED main-table protocol: harvest from TRAIN-split
    #   inner-loop generations; rehearsed and measured sets are disjoint by construction.
    # 'eval'            = the RQ2 ablation arm ONLY: harvest from measured eval rollouts.
    #   This re-exposes evaluation items once their domain becomes past (the contamination
    #   channel the paper quantifies) -- do NOT use it for headline results.
    p.add_argument("--harvest_source", default="train", choices=["eval", "train"])
    return p.parse_args()


class RunningNorm:
    """Dimension-equalization: divide each reward term by its running std (1/√var)."""
    def __init__(self, keys, ema=0.1):
        self.mean = {k: 0.0 for k in keys}; self.var = {k: 1.0 for k in keys}; self.ema = ema
    def norm(self, k, x):
        d = x - self.mean[k]
        self.mean[k] = (1 - self.ema) * self.mean[k] + self.ema * x
        self.var[k] = (1 - self.ema) * self.var[k] + self.ema * d * d
        return x / (self.var[k] ** 0.5 + 1e-6)


def sft_self_distill(model, tok, train_ds, dtype, knobs, num_gen, steps, max_new=256,
                     harvester=None):
    """STaR/SFT self-distillation baseline: generate, keep verifier-correct (or
    self-consistent-majority) traces, and SFT on them."""
    reward_fn = make_reward_fn(dtype)
    if harvester is not None:
        reward_fn = harvester.wrap(reward_fn)
    kept = []
    model.eval()
    for row in train_ds:
        enc = tok(row["prompt"], return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, do_sample=True, temperature=max(knobs["temperature"], 0.3),
                                 top_p=0.95, num_return_sequences=num_gen, max_new_tokens=max_new,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        comps = [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True) for o in out]
        cols = {k: [row[k]] * len(comps) for k in row.keys() if k != "prompt"}
        r = reward_fn([row["prompt"]] * len(comps), comps, **cols)
        for c, rr in zip(comps, r):
            if rr > 0:
                kept.append(row["prompt"] + c)
    if not kept:
        return model
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=knobs["learning_rate"],
                            weight_decay=knobs["weight_decay"])
    pad_side = tok.padding_side; tok.padding_side = "right"   # left-pad corrupts LM targets
    for step in range(steps):
        batch = kept[(step * 4) % len(kept):][:4] or kept[:4]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(model.device)
        labels = enc.input_ids.clone()
        labels[enc.attention_mask == 0] = -100               # mask pads out of the loss
        out = model(**enc, labels=labels)
        opt.zero_grad(); out.loss.backward(); opt.step()
    tok.padding_side = pad_side
    return model


def _mini_sft(model, tok, texts, lr, wd, steps=6):
    """Small SFT pass on (prompt+correct-completion) strings — used for rehearsal."""
    if not texts:
        return model
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    pad_side = tok.padding_side; tok.padding_side = "right"   # left-pad corrupts LM targets
    for s in range(steps):
        batch = texts[(s * 4) % len(texts):][:4] or texts[:4]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(model.device)
        labels = enc.input_ids.clone()
        labels[enc.attention_mask == 0] = -100          # mask pads
        out = model(**enc, labels=labels)
        opt.zero_grad(); out.loss.backward(); opt.step()
    tok.padding_side = pad_side
    return model


def select_by_difficulty(model, tok, pool, n, difficulty, dtype, temperature=1.0,
                         ndiff=2, cand_mult=1.25, shuffle_seed=0):
    """Difficulty (DA) axis: pick seeds whose estimated pass-rate is near a ZPD target
    set by the `difficulty` knob (sweet-spot learnability; Sachan ZPD / Rho-loss).
    The exploration axis also enters the FILTER here: candidate rollouts are sampled at
    the same `temperature` used for GRPO generation, so the ZPD estimate reflects the
    current exploration regime (not a fixed temp). `ndiff` samples per candidate set the
    pass-rate resolution (ndiff=3 -> {0,1/3,2/3,1}); higher = finer but costlier.
    Also returns harvested correct (prompt+completion) traces for the rehearsal buffer.
    Non-verifiable domains cannot be difficulty-scored -> random fallback."""
    n = min(n, len(pool))
    if dtype == "selfconsist" or len(pool) <= n:
        return pool.shuffle(seed=shuffle_seed).select(range(n)), []
    # The candidate window must vary per cycle: a constant seed would re-train on the
    # same ~cand_mult*n item window every cycle (narrow exposure).
    cand = pool.shuffle(seed=shuffle_seed).select(range(min(int(cand_mult * n), len(pool))))
    target = float(np.clip(0.7 - 0.5 * difficulty, 0.1, 0.8))   # difficulty high -> harder seeds
    scored, harvest = [], []
    model.eval()
    with torch.no_grad():
        for i, row in enumerate(cand):
            enc = tok(row["prompt"], return_tensors="pt", truncation=True, max_length=512).to(model.device)
            out = model.generate(**enc, do_sample=True, temperature=temperature, top_p=0.95,
                                 num_return_sequences=ndiff, max_new_tokens=200,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            comps = [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True) for o in out]
            corr = EV._correct_vec(row["prompt"], comps, dtype, row)
            scored.append((abs(sum(corr) / len(corr) - target), i))
            harvest += [row["prompt"] + comps[j] for j in range(len(comps)) if corr[j] > 0]
    scored.sort()
    keep = sorted(i for _, i in scored[:n])
    return cand.select(keep), harvest


def main():
    a = parse()
    assert a.eval_nsamples >= max(a.ks), f"eval_nsamples({a.eval_nsamples}) < max(ks)={max(a.ks)}"
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    os.makedirs("logs", exist_ok=True); os.makedirs(f"output/{a.run_name}", exist_ok=True)
    log_file = f"logs/{a.run_name}.csv"; samp_file = f"logs/{a.run_name}_samples.jsonl"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    sched = DM.DomainScheduler(a.order, a.shift_interval)
    meta = MetaController(STATE_DIM, device="cpu")
    tracker = StateTracker()
    rnorm = RunningNorm(["d_passk", "bwt", "anticollapse"])

    tok = AutoTokenizer.from_pretrained(a.model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # bf16 (evidence-based, see 실험기록 v2d 관찰): fp32 master was ~4x slower on GB10
    # (fp32 autoregressive generation dominates) making 3x30-cycle runs impractical, while
    # buying little — bf16 weight_drift still ACCUMULATES over cycles (full run 1.6e-4, not
    # a no-op) and the reframed contribution rests on behavioral metrics (retention/diversity/
    # pass@k), not weight_drift magnitude. fp32 available via --dtype fp32 for a final rigor run.
    _dt = torch.float32 if getattr(a, "dtype", "bf16") == "fp32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(a.model_name, torch_dtype=_dt).to(device)
    # cpu snapshot for weight_drift; "model" dtype keeps it bf16 at scale (lossless for
    # bf16 weights, halves host memory — see evaluator.snapshot_params docstring)
    theta0 = EV.snapshot_params(model, torch.float32 if a.drift_dtype == "fp32" else None)

    train_pools = {d: DM.DOMAINS[d]["train"]() for d in set(a.order)}
    eval_sets = {d: DM.DOMAINS[d]["eval"]() for d in set(a.order)}
    if DM.USED_MOCK:
        print(f"[WARN] mock fallback used for: {sorted(DM.USED_MOCK)} — results for these are invalid.")

    perf, first_acc, seen = {}, {}, []
    episodic = {d: [] for d in a.order}   # rehearsal buffer: correct (prompt+completion) per domain
    w_prev = {ax: 0.25 for ax in AXES}; prev_metrics = {}

    for cycle in range(1, a.max_cycles + 1):
        d = sched.domain(cycle); dtype = DM.DOMAINS[d]["type"]; verifiable = DM.DOMAINS[d]["verifiable"]
        if d not in seen:
            seen.append(d)
        print(f"\n=== Cycle {cycle}/{a.max_cycles} | {d} ({dtype}) | baseline={a.baseline} ===")

        s_t = tracker.build(prev_metrics, w_prev, verifiable)
        if a.baseline == "nsiac" and cycle > 2:
            act = meta.act(s_t)
        else:
            act = {ax: 0.25 for ax in AXES} | {"_state": s_t}
        knobs = budget_to_knobs(act)
        if a.fixed_knobs:                      # Pareto-sweep: override with a fixed grid point
            for kv in a.fixed_knobs.split(","):
                k, v = kv.split("="); k = k.strip()
                knobs[k] = bool(float(v)) if k == "ref_reset" else float(v)
        print("  budget:", {k: round(act[k], 3) for k in AXES}, "| knobs:", knobs,
              "| fixed" if a.fixed_knobs else "")

        past = [x for x in seen if x != d]
        # difficulty (DA) axis: ZPD seed selection (filter) is an N-SIAC component -> only the
        # nsiac condition runs the (costly) pre-pass; baselines use random seeds, which is both
        # the honest ablation (no adaptive difficulty axis) and ~2x cheaper. exploration axis
        # sets the candidate sampling temperature so it also shapes the filter, not only gen.
        # Data order depends on (--seed, cycle) in BOTH branches, so
        # paired-seed runs genuinely share the candidate stream across conditions.
        dseed = a.seed * 1000 + cycle
        if a.baseline == "nsiac" and not a.eval_only and not a.no_zpd:
            train_ds, _ = select_by_difficulty(              # harvest handled uniformly post-eval
                model, tok, train_pools[d], a.n_train, knobs["difficulty"], dtype,
                temperature=knobs["temperature"], shuffle_seed=dseed)
        else:
            train_ds = train_pools[d].shuffle(seed=dseed).select(
                range(min(a.n_train, len(train_pools[d]))))

        # ---- inner loop ----
        # v2j train-split harvest: capture verifier-correct traces from the inner loop's own
        # generations (reward-fn hook; same dedup as the eval-path harvest).
        th = TrainHarvest() if a.harvest_source == "train" else None
        if a.eval_only:
            pass                                             # frozen-base envelope: no updates
        elif a.baseline == "sft_star":
            model = sft_self_distill(model, tok, train_ds, dtype, knobs, a.num_gen, a.grpo_steps,
                                     harvester=th)
        else:
            import dataclasses
            from trl import GRPOTrainer, GRPOConfig
            if a.optim != "adamw_torch":       # 8-bit states need bitsandbytes present
                try:
                    import bitsandbytes  # noqa: F401
                except Exception:
                    print(f"  [optim] {a.optim} unavailable (no bitsandbytes) -> adamw_torch")
                    a.optim = "adamw_torch"
            want = dict(
                optim=a.optim,
                output_dir=f"output/{a.run_name}/cycle_{cycle}",
                learning_rate=knobs["learning_rate"], weight_decay=knobs["weight_decay"],
                per_device_train_batch_size=a.num_gen, gradient_accumulation_steps=a.grad_accum,
                num_generations=a.num_gen, temperature=knobs["temperature"], beta=knobs["kl_beta"],
                entropy_coef=knobs["entropy_coef"],                    # exploration axis
                sync_ref_model=knobs["ref_reset"], ref_model_sync_steps=max(1, a.grpo_steps // 2),
                ref_model_mixup_alpha=0.8,       # EMA weight of current policy on ref soft-sync
                max_prompt_length=512, max_completion_length=256, max_steps=a.grpo_steps,
                bf16=True, gradient_checkpointing=True, logging_steps=5,
                save_strategy="no", report_to="none")
            valid = {f.name for f in dataclasses.fields(GRPOConfig)}   # version-robust
            dropped = [k for k in want if k not in valid]
            if dropped:
                print(f"  [trl] GRPOConfig ignores unsupported args: {dropped}")
                # A swept lever silently becoming a no-op (e.g. after a trl version
                # bump) would corrupt the Pareto comparison -> hard-fail instead.
                if a.fixed_knobs:
                    knob2cfg = {"entropy_coef": "entropy_coef", "kl_beta": "beta",
                                "temperature": "temperature", "learning_rate": "learning_rate",
                                "weight_decay": "weight_decay", "ref_reset": "sync_ref_model"}
                    fixed_keys = {kv.split("=")[0].strip() for kv in a.fixed_knobs.split(",")}
                    bad = {k for k in fixed_keys if knob2cfg.get(k) in dropped}
                    if bad:
                        raise RuntimeError(f"--fixed_knobs lever(s) unsupported by this trl: {sorted(bad)}")
            cfg = GRPOConfig(**{k: v for k, v in want.items() if k in valid})
            _rf = make_reward_fn(dtype)
            if th is not None:
                _rf = th.wrap(_rf)
            trainer = GRPOTrainer(model=model, reward_funcs=[_rf],
                                  args=cfg, train_dataset=train_ds)
            trainer.train(); model = trainer.model
            del trainer; gc.collect(); torch.cuda.empty_cache()

        # ---- rehearsal (stability axis): SFT-interleave past-domain correct traces ----
        if past and knobs["rehearsal_ratio"] > 0 and not a.eval_only and not a.no_rehearsal:
            reh = [t for pdom in past for t in episodic.get(pdom, [])]
            n_reh = int(knobs["rehearsal_ratio"] * a.n_train)
            if reh and n_reh > 0:
                random.shuffle(reh)
                model = _mini_sft(model, tok, reh[:n_reh],
                                  knobs["learning_rate"], knobs["weight_decay"])
                gc.collect(); torch.cuda.empty_cache()

        # ---- evaluate all seen domains ----
        dom_acc, cur, samples_all, dom_solved = {}, {}, [], {}
        for ed in seen:
            m, samples = EV.evaluate_domain(model, tok, eval_sets[ed], DM.DOMAINS[ed]["type"],
                                            n_samples=a.eval_nsamples, ks=tuple(a.ks),
                                            subset=a.eval_subset)
            dom_solved[ed] = m.pop("solved_vec", [])   # per-prompt -> meta.jsonl (axis-3 coverage)
            dom_acc[ed] = m["pass@1"]
            for s in samples:
                s.update({"cycle": cycle, "eval_domain": ed})
            samples_all += samples
            if ed == d:
                cur = m
        perf[cycle] = dom_acc
        # Standard BWT (Lopez-Paz) anchors R_{i,i} at the END of task i's
        # training block. Updating the reference on every in-block cycle leaves it at the
        # block's final value once the domain is left (setdefault froze the FIRST cycle,
        # underestimating R_{i,i} and inflating BWT).
        first_acc[d] = dom_acc.get(d, 0.0)
        # universal rehearsal harvest (identical mechanism across ALL conditions -> only seed
        # selection differs): reuse this cycle's eval rollouts that the verifier marked correct.
        # Caution (harvest-source ablation, RQ2): those are the SAME rows that later score
        # past-domain retention, so
        # post-block gains mix transfer with re-exposure. --harvest_holdout N harvests instead
        # from eval rows [eval_subset, eval_subset+N), which are never scored, making the
        # rehearsed and measured sets disjoint at the cost of one extra generation pass.
        measured_traces = cur.pop("correct_traces", [])
        if th is not None:
            new_traces = th.traces          # train-split inner-loop harvest (v2j)
        elif a.harvest_holdout > 0 and not a.eval_only:
            new_traces = EV.harvest_holdout(model, tok, eval_sets[d], dtype,
                                            start=a.eval_subset, n=a.harvest_holdout,
                                            n_samples=max(4, a.eval_nsamples // 2))
        else:
            new_traces = measured_traces
        episodic[d] = (episodic[d] + new_traces)[-300:]
        past = [x for x in seen if x != d]
        bwt = float(np.mean([dom_acc.get(x, 0.0) - first_acc.get(x, 0.0) for x in past])) if past else 0.0
        cur["bwt"] = bwt
        cur["passk_spread"] = cur.get(f"pass@{max(a.ks)}", 0.0) - cur.get("pass@1", 0.0)
        cur["acc"] = cur.get("pass@1", 0.0)

        # ---- plasticity (with probe hidden) ----
        probe = EV.probe_last_hidden(model, tok, [r["prompt"] for r in train_ds.select(range(min(16, len(train_ds))))])
        plas = EV.plasticity_metrics(model, theta0, probe)

        # ---- outer reward (dimension-equalized) ----
        d_passk = rnorm.norm("d_passk", cur.get("pass@1", 0.0) - prev_metrics.get("pass@1", 0.0))
        bwt_n = rnorm.norm("bwt", bwt)
        anti = rnorm.norm("anticollapse", cur.get("distinct_3", 0.0) - 0.5 * cur.get("self_overlap", 0.0))
        # v2b fix: up-weight retention (BWT) so the controller does not trade away
        # past-domain performance for short-term capability gain (v2a forgot more
        # than the fixed baseline). A hard penalty is added when any past domain drops.
        forget_pen = sum(min(0.0, dom_acc.get(x, 0.0) - first_acc.get(x, 0.0)) for x in past)
        r_t = float(d_passk + 2.0 * bwt_n + 0.5 * anti + 0.5 * forget_pen)
        if a.baseline == "nsiac" and cycle > 2:
            meta.store(d, act, r_t); meta.update(d)

        # ---- logging ----
        with open(samp_file, "a") as f:
            for s in samples_all:
                f.write(json.dumps(s, ensure_ascii=False, default=float) + "\n")
        row = {"cycle": cycle, "domain": d, "baseline": a.baseline, "reward": round(r_t, 5),
               "bwt": round(bwt, 5),
               **{f"pass@{k}": round(cur.get(f"pass@{k}", 0.0), 4) for k in a.ks},
               "coverage_solved": round(cur.get("coverage_solved", 0.0), 4),
               "coverage_solutions": round(cur.get("coverage_solutions", 0.0), 4),
               "distinct_3": round(cur.get("distinct_3", 0.0), 4),
               "self_overlap": round(cur.get("self_overlap", 0.0), 4),
               "repetition": round(cur.get("repetition", 0.0), 4),
               "gen_entropy": round(cur.get("gen_entropy", 0.0), 4),
               "weight_drift": round(plas["weight_drift"], 6), "param_norm": round(plas["param_norm"], 3),
               "changed_frac": round(plas["changed_frac"], 6),
               "changed_frac_small": round(plas["changed_frac_small"], 6),
               "changed_frac_large": round(plas["changed_frac_large"], 6),
               "dormant_frac": plas["dormant_frac"], "effective_rank": plas["effective_rank"],
               "used_mock": d in DM.USED_MOCK,
               **{f"w_{ax}": round(act[ax], 4) for ax in AXES},
               **{f"acc_{x}": round(dom_acc.get(x, 0.0), 4) for x in a.order}}
        pd.DataFrame([row]).to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)
        with open(f"logs/{a.run_name}_meta.jsonl", "a") as f:
            f.write(json.dumps({"cycle": cycle, "knobs": knobs, "metrics": cur,
                                "solved": dom_solved,
                                "used_mock": sorted(DM.USED_MOCK)}, default=float) + "\n")

        prev_metrics = cur; w_prev = {ax: act[ax] for ax in AXES}
        gc.collect(); torch.cuda.empty_cache()

    # --- 자동 기록: [RESULT] 항목 (실험 완료 때마다 plan/실험기록.md 에 항상 남김) ---
    try:
        import datetime as _dt
        rdf = pd.read_csv(log_file)
        with open("plan/실험기록.md", "a", encoding="utf-8") as f:
            f.write(f"\n#### [RESULT] {a.run_name} — {len(rdf)} cycles "
                    f"({_dt.datetime.now():%Y-%m-%d %H:%M}) baseline={a.baseline}\n")
            f.write(f"- final pass@1={rdf['pass@1'].iloc[-1]:.3f}, "
                    f"BWT={rdf['bwt'].iloc[-1]:.4f}, "
                    f"weight_drift={rdf['weight_drift'].iloc[-1]:.2e}\n")
            f.write(f"- late budget: w_prec={rdf['w_prec'].tail(5).mean():.2f}, "
                    f"w_stable={rdf['w_stable'].tail(5).mean():.2f}, "
                    f"w_explore={rdf['w_explore'].tail(5).mean():.2f}\n")
            if bool(rdf.get("used_mock", pd.Series([False])).any()):
                f.write("- WARNING: mock-fallback domains present (some results invalid)\n")
    except Exception as _e:
        print("[record] final log skipped:", _e)
    print(f"\nDone. CSV: {log_file} | Q&A: {samp_file}")


if __name__ == "__main__":
    main()
