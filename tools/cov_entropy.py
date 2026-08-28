#!/usr/bin/env python3
"""Post-hoc entropy-covariance diagnostic (Cui et al., 2025 structure, coarse-grained).

Theory ([93] Cui+ 2025, Thm 1/2, bandit view): for a softmax policy over responses,
    H(k+1) - H(k)  ~=  -eta * Cov_{y~pi}( log pi(y|x), pi(y|x) * A(y,x) )      (PG)
                    ~=  -eta * Cov_{y~pi}( log pi(y|x), A(y,x) )               (NPG)

Our logs store no sequence log-probs and the per-cycle checkpoints are empty, so
log pi(y|x) is unavailable. We therefore evaluate the SAME functional on the
answer-cluster partition that already defines the logged `gen_entropy`
(evaluator.answer_cluster_entropy over reward_arbiter._answer_key):

    pi_hat(c) = count_c / G      over clusters c of the G sampled completions
    A(c)      = (r_c - mean_r) / std_r      (GRPO group-normalised; 0 if std=0)
    Cov_pg    = Cov_{c~pi_hat}( log pi_hat(c), pi_hat(c) * A(c) )
    Cov_npg   = Cov_{c~pi_hat}( log pi_hat(c), A(c) )

H on this same partition is exactly the logged gen_entropy, so DeltaH and Cov live
in one space. This is a structural analogy test, not a verification of Thm 1:
the coarse-graining does not preserve the logit-change proposition, the rollouts
are eval-side while the update is train-side, and G is small.

Usage:
  python3 tools/cov_entropy.py --runs v2_full_nsiac_v2h v2_full_fixedmid_v2h ... \
      [--out paper/cov_entropy_report.md] [--csv-out logs/cov_entropy.csv]
"""
import argparse, json, math, os, re, sys
from collections import Counter, defaultdict
from itertools import permutations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nsiac2.reward_arbiter import _answer_key  # same clustering as gen_entropy

LOGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


def cluster_stats(completions, correct):
    """Return (pi_hat, adv, entropy) arrays over answer clusters of one prompt."""
    keys = [_answer_key(c) for c in completions]
    G = len(keys)
    if G == 0:
        return None
    r = [float(x) for x in correct]
    mu = sum(r) / G
    var = sum((x - mu) ** 2 for x in r) / G
    sd = math.sqrt(var)
    # group-normalised advantage per sample, then averaged within a cluster
    adv_s = [((x - mu) / sd) if sd > 1e-12 else 0.0 for x in r]
    cnt = Counter(keys)
    idx = defaultdict(list)
    for i, k in enumerate(keys):
        idx[k].append(i)
    p, a = [], []
    for k, c in cnt.items():
        p.append(c / G)
        a.append(sum(adv_s[i] for i in idx[k]) / c)
    H = -sum(pi * math.log(pi + 1e-12) for pi in p)
    return p, a, H


def cov(xs, ys, w):
    """Weighted covariance under distribution w (sums to 1)."""
    mx = sum(wi * x for wi, x in zip(w, xs))
    my = sum(wi * y for wi, y in zip(w, ys))
    return sum(wi * (x - mx) * (y - my) for wi, x, y in zip(w, xs, ys))


def per_cycle(run):
    """Aggregate covariance per cycle from <run>_samples.jsonl."""
    path = os.path.join(LOGS, f"{run}_samples.jsonl")
    if not os.path.exists(path):
        return None
    acc = defaultdict(lambda: {"cov_pg": [], "cov_npg": [], "H": [], "G": [], "dom": None})
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            comps, corr = d.get("completions") or [], d.get("correct") or []
            if not comps or len(comps) != len(corr):
                continue
            st = cluster_stats(comps, corr)
            if st is None:
                continue
            p, a, H = st
            acc[d["cycle"]]["cov_pg"].append(cov([math.log(pi + 1e-12) for pi in p],
                                                 [pi * ai for pi, ai in zip(p, a)], p))
            acc[d["cycle"]]["cov_npg"].append(cov([math.log(pi + 1e-12) for pi in p], a, p))
            acc[d["cycle"]]["H"].append(H)
            acc[d["cycle"]]["G"].append(len(comps))
            acc[d["cycle"]]["dom"] = d.get("eval_domain")
    out = []
    for c in sorted(acc):
        v = acc[c]
        n = len(v["H"])
        out.append({"cycle": c, "domain": v["dom"], "n_prompts": n,
                    "G": sum(v["G"]) / n,
                    "cov_pg": sum(v["cov_pg"]) / n,
                    "cov_npg": sum(v["cov_npg"]) / n,
                    "H": sum(v["H"]) / n})
    for i in range(len(out) - 1):
        out[i]["dH"] = out[i + 1]["H"] - out[i]["H"]
        out[i]["same_domain"] = out[i + 1]["domain"] == out[i]["domain"]
    if out:
        out[-1]["dH"] = None
        out[-1]["same_domain"] = None
    return out


def spearman(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
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


def perm_p(x, y, rho, sided="less", iters=20000, seed=0):
    """Permutation p-value; exact enumeration when n<=8, else deterministic LCG shuffle."""
    n = len(x)
    if n < 3 or math.isnan(rho):
        return float("nan")
    cnt = tot = 0
    if n <= 8:
        for perm in permutations(range(n)):
            r = spearman(x, [y[i] for i in perm])
            tot += 1
            if (r <= rho) if sided == "less" else (r >= rho):
                cnt += 1
    else:
        state = seed * 2654435761 + 1
        for _ in range(iters):
            idx = list(range(n))
            for i in range(n - 1, 0, -1):
                state = (1103515245 * state + 12345) % (1 << 31)
                j = state % (i + 1)
                idx[i], idx[j] = idx[j], idx[i]
            r = spearman(x, [y[i] for i in idx])
            tot += 1
            if (r <= rho) if sided == "less" else (r >= rho):
                cnt += 1
    return (cnt + 1) / (tot + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv-out", default=None)
    ap.add_argument("--within-domain", action="store_true",
                    help="restrict the primary test to cycle pairs inside one domain block")
    a = ap.parse_args()

    series, lines = {}, []
    for run in a.runs:
        s = per_cycle(run)
        if s is None:
            lines.append(f"- `{run}`: samples.jsonl 없음 — 건너뜀")
            continue
        series[run] = s

    lines.append("# Entropy-covariance post-hoc diagnostic (coarse-grained, answer-cluster partition)\n")
    lines.append("이론 [93] Cui+2025 Thm1/2의 함수형을 `gen_entropy`와 **동일한 답안-클러스터 분할** 위에서 계산.")
    lines.append("시퀀스 로그확률·사이클 체크포인트 부재로 정확한 log pi(y|x)는 사용 불가 — **구조적 유비 검정**이며 Thm1 검증이 아님.\n")

    lines.append("## 런별 요약\n")
    lines.append("| run | 사이클 | 평균 G | mean Cov_pg | mean Cov_npg | Cov_pg>0 비율 | mean H |")
    lines.append("|---|---|---|---|---|---|---|")
    for run, s in series.items():
        n = len(s)
        mpg = sum(r["cov_pg"] for r in s) / n
        mnpg = sum(r["cov_npg"] for r in s) / n
        pos = sum(1 for r in s if r["cov_pg"] > 0) / n
        mh = sum(r["H"] for r in s) / n
        lines.append(f"| {run} | {n} | {s[0]['G']:.0f} | {mpg:+.5f} | {mnpg:+.5f} | {pos:.0%} | {mh:.4f} |")

    lines.append("\n## 주 예측: Cov_k 와 dH_k = H_{k+1}-H_k 의 음의 상관\n")
    lines.append("| run | pairs | rho(Cov_pg, dH) | p(단측,less) | rho(Cov_npg, dH) | p |")
    lines.append("|---|---|---|---|---|---|")
    pooled = []
    for run, s in series.items():
        rows = [r for r in s if r.get("dH") is not None and (not a.within_domain or r.get("same_domain"))]
        if len(rows) < 3:
            lines.append(f"| {run} | {len(rows)} | — | — | — | — |")
            continue
        xp = [r["cov_pg"] for r in rows]; xn = [r["cov_npg"] for r in rows]
        y = [r["dH"] for r in rows]
        rp, rn = spearman(xp, y), spearman(xn, y)
        pp, pn = perm_p(xp, y, rp), perm_p(xn, y, rn)
        pooled.append((run, rp, rn))
        lines.append(f"| {run} | {len(rows)} | {rp:+.3f} | {pp:.3f} | {rn:+.3f} | {pn:.3f} |")
    if pooled:
        sp = sum(1 for _, rp, _ in pooled if rp < 0)
        sn = sum(1 for _, _, rn in pooled if rn < 0)
        lines.append(f"\n부호 일관성: Cov_pg **{sp}/{len(pooled)}** 런에서 음, Cov_npg **{sn}/{len(pooled)}** 런에서 음 "
                     f"(사전등록 예측 = 음).")

    if a.csv_out:
        with open(a.csv_out, "w") as f:
            f.write("run,cycle,domain,n_prompts,G,cov_pg,cov_npg,H,dH,same_domain\n")
            for run, s in series.items():
                for r in s:
                    dh = "" if r.get("dH") is None else f"{r['dH']:.6f}"
                    f.write(f"{run},{r['cycle']},{r['domain']},{r['n_prompts']},{r['G']:.1f},"
                            f"{r['cov_pg']:.6f},{r['cov_npg']:.6f},{r['H']:.6f},{dh},"
                            f"{r.get('same_domain')}\n")
        lines.append(f"\n- 사이클별 원자료: `{a.csv_out}`")

    txt = "\n".join(lines)
    if a.out:
        open(a.out, "w").write(txt + "\n")
        print(f"wrote {a.out}")
    print(txt)


if __name__ == "__main__":
    main()
