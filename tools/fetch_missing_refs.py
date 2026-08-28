#!/usr/bin/env python3
"""인용했으나 로컬 원문이 없는 문헌을 arXiv 에서 받아온다.

citation_check.py 의 [층2] ⑤(미연결) 중 사람이 "원문 없음"으로 판정한 목록을
받아 arXiv API 로 검색·검증·저장한다. **제목이 실제로 일치하는지 확인**하고
저장하며, 일치하지 않으면 받지 않고 수동 처리 목록으로 넘긴다.

실행: python3 tools/fetch_missing_refs.py [--dry-run] [--start-n 173]
"""
import os, re, sys, json, time, argparse, unicodedata
import urllib.request, urllib.parse
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATOM = "{http://www.w3.org/2005/Atom}"

# bibkey -> (파일명 어간, 검색용 제목)
WANTED = [
    ("huang2023selfimprove",       "Huang_2022_LLMs_Can_Self_Improve",              "Large Language Models Can Self-Improve"),
    ("liang2023helm",              "Liang_2023_HELM_Holistic_Evaluation",           "Holistic Evaluation of Language Models"),
    ("henderson2018matters",       "Henderson_2018_Deep_RL_That_Matters",           "Deep Reinforcement Learning that Matters"),
    ("engstrom2020implementation", "Engstrom_2020_Implementation_Matters_PPO_TRPO", "Implementation Matters in Deep Policy Gradients"),
    ("agarwal2021precipice",       "Agarwal_2021_Deep_RL_at_the_Edge_of_Precipice", "Deep Reinforcement Learning at the Edge of the Statistical Precipice"),
    ("mirzadeh2022wide",           "Mirzadeh_2022_Wide_Neural_Networks_Forget_Less", "Wide Neural Networks Forget Less Catastrophically"),
    ("ramasesh2022scale",          "Ramasesh_2022_Effect_of_Scale_on_Forgetting",   "Effect of scale on catastrophic forgetting in neural networks"),
    ("kirk2024rlhf",               "Kirk_2024_Effects_of_RLHF_on_Generalisation_Diversity", "Understanding the Effects of RLHF on LLM Generalisation and Diversity"),
    ("liu2022distinct",            "Liu_2022_Rethinking_Refining_Distinct_Metric",  "Rethinking and Refining the Distinct Metric"),
    ("cha2025realitycheck",        "Cha_2025_Hyperparameters_in_Continual_Learning", "Hyperparameters in Continual Learning: A Reality Check"),
    ("luo2025forgetting",          "Luo_2025_Empirical_Study_Catastrophic_Forgetting_LLM", "An Empirical Study of Catastrophic Forgetting in Large Language Models"),
    ("hochlehnert2025sober",       "Hochlehnert_2025_Sober_Look_at_Progress",       "A Sober Look at Progress in Language Model Reasoning"),
    ("obe2025",                    "Song_2025_Outcome_based_Exploration",           "Outcome-based Exploration for LLM Reasoning"),
    ("darling2025",                "Li_2025_Jointly_Reinforcing_Diversity_Quality", "Jointly Reinforcing Diversity and Quality in Language Model Generations"),
    ("pathnottaken2025",           "Path_Not_Taken_RLVR_Off_the_Principals",        "The Path Not Taken: RLVR Provably Learns Off the Principals"),
    ("wu2026contamination",        "Wu_Reasoning_or_Memorization_Unreliable_Results", "Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination"),
    ("surfaceapproach2026",        "Measuring_Strategy_or_Phrasing_Surface_Diversity", "Are We Measuring Strategy or Phrasing"),
    ("hiddencosts2026",            "Hidden_Costs_Measurement_Gaps_RL",              "The Hidden Costs and Measurement Gaps of Reinforcement Learning"),
    ("harmon2026mapping",          "Mapping_Post_Training_Forgetting_at_Scale",     "Mapping Post-Training Forgetting in Language Models at Scale"),
    ("rethinktransfer2026",        "Rethinking_Transfer_Continual_Learning_Replay",  "Rethinking Transfer in Continual Learning"),
    ("rdbcl2026",                  "Reasoning_Portability_Continual_Learning_MLLM", "Reasoning Portability: Guiding Continual Learning"),
    ("cpo2026",                    "RL_Forgets_Continual_Policy_Optimization",      "RL Forgets! Towards Continual Policy Optimization"),
    ("entsched2026",               "Entropy_Scheduling_in_RL",                      "Entropy Scheduling in Reinforcement Learning"),
    ("llmzero2026",                "LLMZero_Adaptive_Training_Strategies",          "LLMZero: Discovering Adaptive Training Strategies"),
    ("aer2025",                    "Revisiting_Entropy_Regularization_Adaptive_Coefficient", "Revisiting Entropy Regularization"),
]


def toks(s):
    return set(re.findall(r"[a-z]{4,}", unicodedata.normalize("NFKD", s).lower()))


def search(title, rows=6):
    q = urllib.parse.quote(f'ti:"{title}"')
    url = f"https://export.arxiv.org/api/query?search_query={q}&max_results={rows}"
    try:
        raw = urllib.request.urlopen(url, timeout=45).read()
    except Exception:
        # 제목 완전일치 실패 시 전체 필드 검색으로 완화
        q = urllib.parse.quote(f'all:"{title}"')
        url = f"https://export.arxiv.org/api/query?search_query={q}&max_results={rows}"
        raw = urllib.request.urlopen(url, timeout=45).read()
    out = []
    for e in ET.fromstring(raw).findall(ATOM + "entry"):
        t = re.sub(r"\s+", " ", (e.findtext(ATOM + "title") or "")).strip()
        pdf = None
        for l in e.findall(ATOM + "link"):
            if l.attrib.get("title") == "pdf":
                pdf = l.attrib.get("href")
        out.append((t, pdf))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--start-n", type=int, default=173)
    ap.add_argument("--out", default=os.path.join(ROOT, "참고논문"))
    ap.add_argument("--min-overlap", type=float, default=0.75)
    a = ap.parse_args()

    n = a.start_n
    ok, fail = [], []
    for key, stem, title in WANTED:
        want = toks(title)
        best = (0.0, None, None)
        try:
            for got_title, pdf in search(title):
                ov = len(want & toks(got_title)) / max(len(want), 1)
                if ov > best[0]:
                    best = (ov, got_title, pdf)
        except Exception as ex:
            fail.append((key, title, f"검색 실패: {ex}"))
            time.sleep(3)
            continue

        ov, got, pdf = best
        if ov < a.min_overlap or not pdf:
            fail.append((key, title, f"제목 불일치({ov:.2f}) 최근접: {got}"))
            time.sleep(3)
            continue

        fname = f"[{n}]_{stem}.pdf"
        dest = os.path.join(a.out, fname)
        print(f"  {ov:.2f}  {key:26s} → {fname}")
        if not a.dry_run:
            try:
                req = urllib.request.Request(pdf, headers={"User-Agent": "citation-check/1.0"})
                data = urllib.request.urlopen(req, timeout=90).read()
                if not data.startswith(b"%PDF"):
                    fail.append((key, title, "PDF 아님(HTML 응답)"))
                    time.sleep(3)
                    continue
                open(dest, "wb").write(data)
            except Exception as ex:
                fail.append((key, title, f"다운로드 실패: {ex}"))
                time.sleep(3)
                continue
        ok.append((key, n, fname, got))
        n += 1
        time.sleep(3)

    print(f"\n=== 성공 {len(ok)}건 ===")
    for k, num, f, t in ok:
        print(f"  [{num}] {k:26s} {t[:66]}")
    print(f"\n=== 실패 {len(fail)}건 (수동 확보 필요) ===")
    for k, t, why in fail:
        print(f"  {k:26s} {t[:50]}\n      {why[:100]}")
    json.dump({"ok": ok, "fail": [[k, t, w] for k, t, w in fail]},
              open(os.path.join(ROOT, ".cache", "fetch_result.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
