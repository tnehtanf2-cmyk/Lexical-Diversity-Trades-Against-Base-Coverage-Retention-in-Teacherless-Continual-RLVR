#!/usr/bin/env python3
r"""인용 무결성 3층 대조 검사기.

  층1  본문 \cite 키  ↔  refs.bib 항목        — 완전 자동. 여기서 걸리면 곧 LaTeX 오류다
  층2  refs.bib 항목  ↔  참고논문/ 원문 PDF   — `% [N]` 명시 링크가 정본
  층3  참고논문/INDEX.md 등재 여부            — 색인 최신성

왜 `% [N]` 명시 링크인가:
  제목 퍼지 매칭만으로는 판정이 안 된다. 실측(2026-08-28)에서 원문이 **있는**
  dphrl2025 가 0.64, 원문이 **없는** obe2025 이 0.73 을 받아 참·거짓 구간이
  역전됐다. RL 논문 1쪽은 "reinforcement/learning/reward" 같은 공통어로 덮여
  있어 IDF 가중을 넣어도 분리되지 않는다. 따라서 매칭은 **후보 제안**까지만
  하고, 확정은 refs.bib 의 `% [N] ...` 주석(사람이 한 번 확인한 사실)에 맡긴다.
  주석은 BibTeX 가 무시하므로 조판에 영향이 없다.

실행:
  python3 tools/citation_check.py              # 요약
  python3 tools/citation_check.py --suggest    # 미연결 항목의 PDF 후보까지
종료코드: 층1 위반이 있으면 1.
"""
import os, re, sys, glob, json, math, argparse, unicodedata, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_bib(path):
    """중괄호 균형으로 항목을 자르고, 선행 `% [N]` 주석을 함께 잡는다.

    refs.bib 는 `...year={2002}}` 처럼 한 줄로 끝나는 항목이 섞여 있어
    `\n}` 로 끊는 순진한 정규식은 절반을 놓친다(실측 33/83 누락).
    """
    src = open(path, encoding="utf-8").read()
    entries = {}
    for m in re.finditer(r"(?:^|\n)(?:%\s*\[(\d+)\][^\n]*\n)?@(\w+)\{([^,\s]+),", src):
        n, etype, key = m.group(1), m.group(2), m.group(3)
        j = src.index("{", m.start(2))
        depth, k = 0, j
        while k < len(src):
            if src[k] == "{":
                depth += 1
            elif src[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        body = src[j + 1:k]

        def field(name):
            mm = re.search(name + r"\s*=\s*\{", body, re.I)
            if not mm:
                return ""
            s = mm.end()
            d, t = 1, s
            while t < len(body) and d:
                if body[t] == "{":
                    d += 1
                elif body[t] == "}":
                    d -= 1
                t += 1
            return re.sub(r"\s+", " ", body[s:t - 1]).strip()

        entries[key] = {
            "n": int(n) if n else None,
            "type": etype,
            "title": field("title").replace("{", "").replace("}", ""),
            "author": field("author"),
            "year": field("year"),
            "note": field("note"),
            "journal": field("journal") or field("booktitle"),
        }
    return entries


def cited_keys(tex_dir):
    keys, where = set(), {}
    for f in sorted(glob.glob(os.path.join(tex_dir, "*.tex"))):
        txt = re.sub(r"(?m)^\s*%.*$", "", open(f, encoding="utf-8").read())
        for m in re.finditer(r"\\cite[a-zA-Z]*(?:\[[^\]]*\]){0,2}\{([^}]*)\}", txt):
            for k in (x.strip() for x in m.group(1).split(",")):
                if k:
                    keys.add(k)
                    where.setdefault(k, set()).add(os.path.basename(f))
    return keys, where


def norm_tokens(s):
    return set(re.findall(r"[a-z]{4,}", unicodedata.normalize("NFKD", s).lower()))


def load_fulltext(pdf_dir, cache_path):
    """참고논문 PDF 1쪽 텍스트 캐시(제목은 1쪽에 거의 원문대로 찍힌다)."""
    cache = {}
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path, encoding="utf-8"))
    on_disk = {f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")}
    for f in list(cache):                       # 삭제·개명된 PDF 정리
        if f not in on_disk:
            del cache[f]
    todo = sorted(on_disk - set(cache))         # 새로 들어온 것만 추출(증분)
    if todo:
        print(f"  (PDF 1쪽 텍스트 캐시 갱신: 신규 {len(todo)}편)")
    for f in todo:
        try:
            out = subprocess.run(["pdftotext", "-f", "1", "-l", "1", "-q",
                                  os.path.join(pdf_dir, f), "-"],
                                 capture_output=True, timeout=30).stdout
            cache[f] = re.sub(r"\s+", " ", out.decode("utf-8", "ignore"))[:3000]
        except Exception:
            cache[f] = ""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
    return cache


class Suggester:
    """IDF 가중 제목-토큰 커버리지로 후보 PDF를 제안한다. 확정용이 아니다."""

    def __init__(self, fulltext):
        self.docs = {f: norm_tokens(f + " " + t) for f, t in fulltext.items()}
        n, df = len(self.docs), {}
        for d in self.docs.values():
            for w in d:
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}

    def top(self, title, k=2):
        t = norm_tokens(title)
        if len(t) < 3:
            return []
        w = lambda x: self.idf.get(x, 1.0)
        tot = sum(w(x) for x in t) or 1
        r = sorted(((sum(w(x) for x in (t & d)) / tot, f) for f, d in self.docs.items()),
                   reverse=True)
        return [(round(s, 2), f, len(t)) for s, f in r[:k]]


def title_coverage(title, page_text):
    """제목 토큰이 원문 1쪽에 몇 % 나타나는가.

    PDF 는 제목을 자간 넣어 렌더링하는 일이 잦아(`LLMZ ERO`, `RL F ORGETS`,
    `E FFECT OF`) 토큰 대조만 하면 멀쩡한 인용이 불일치로 잡힌다. 공백을 모두
    지운 문자열에 대한 부분문자열 검사를 함께 본다.
    """
    t = norm_tokens(title)
    if not t:
        return None, set()
    toks = norm_tokens(page_text)
    flat = re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", page_text).lower())
    miss = {w for w in t if w not in toks and w not in flat}
    return 1 - len(miss) / len(t), miss



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex-dir", default=os.path.join(ROOT, "paper"))
    ap.add_argument("--bib", default=os.path.join(ROOT, "paper", "refs.bib"))
    ap.add_argument("--pdf-dir", default=os.path.join(ROOT, "참고논문"))
    ap.add_argument("--index", default=os.path.join(ROOT, "참고논문", "INDEX.md"))
    ap.add_argument("--cache", default=os.path.join(ROOT, ".cache", "pdf_firstpage.json"))
    ap.add_argument("--suggest", action="store_true", help="미연결 항목의 PDF 후보 출력")
    a = ap.parse_args()

    entries = parse_bib(a.bib)
    cited, where = cited_keys(a.tex_dir)
    fulltext = load_fulltext(a.pdf_dir, a.cache)
    index_keys = set()
    if os.path.exists(a.index):
        for line in open(a.index, encoding="utf-8"):
            index_keys.update(m.group(1) for m in re.finditer(r"`([A-Za-z][\w:-]+)`", line))

    undefined = sorted(cited - set(entries))
    dead = sorted(set(entries) - cited)
    print(f"본문 인용 {len(cited)}키 · refs.bib {len(entries)}항목 · "
          f"참고논문 PDF {len(fulltext)}편 · INDEX 등재키 {len(index_keys)}")

    print("\n[층1] 본문 ↔ refs.bib   (자동 판정)")
    print(f"  ① 인용됐으나 bib 미정의 : {len(undefined)}건" +
          "".join(f"\n      - {k}  ({', '.join(sorted(where[k]))})" for k in undefined))
    print(f"  ② bib에 있으나 미인용   : {len(dead)}건" +
          "".join(f"\n      - {k}" for k in dead))

    live = sorted(cited & set(entries))
    linked = [k for k in live if entries[k]["n"] is not None]
    unlinked = [k for k in live if entries[k]["n"] is None]
    pdfnums = {}
    for f in os.listdir(a.pdf_dir):
        # 같은 [N] 로 .md 노트가 함께 있으므로 반드시 PDF 만 잡는다
        m = re.match(r"\[(\d+)\]", f)
        if m and f.lower().endswith(".pdf"):
            pdfnums[int(m.group(1))] = f
    broken = [k for k in linked if entries[k]["n"] not in pdfnums]

    print("\n[층2] 인용 키 ↔ 보유 원문   (`% [N]` 명시 링크가 정본)")
    print(f"  ③ [N] 연결 + PDF 실재  : {len(linked) - len(broken)}건")
    print(f"  ④ [N] 연결됐으나 PDF 없음: {len(broken)}건" +
          "".join(f"\n      - {k} → [{entries[k]['n']}]" for k in broken))
    print(f"  ⑤ 미연결(원문 확인 안 됨): {len(unlinked)}건")

    if a.suggest and unlinked:
        sg = Suggester(fulltext)
        print("\n      미연결 항목별 후보 (점수는 참고용 — 반드시 눈으로 확인할 것)")
        rows = []
        for k in unlinked:
            t = sg.top(entries[k]["title"])
            s1, f1, ntok = t[0] if t else (0.0, "—", 0)
            rows.append((-s1, k, s1, f1, ntok))
        for _, k, s1, f1, ntok in sorted(rows):
            flag = "강" if s1 >= 0.99 and ntok >= 6 else ("약" if s1 >= 0.7 else " ")
            print(f"      {flag} {s1:>4}  {k:26s} {entries[k]['title'][:46]:46s} → {f1[:46]}")

    print("\n[층4] bib 제목 ↔ 원문 1쪽   (오탈자·구제목 인용 검출)")
    noText, low, okn = [], [], 0
    for k in sorted(live):
        n = entries[k]["n"]
        f = pdfnums.get(n) if n is not None else None
        if not f:
            continue
        page = fulltext.get(f, "")
        if len(page.strip()) < 200:
            noText.append((k, n, f))
            continue
        cov, miss = title_coverage(entries[k]["title"], page)
        if cov is None:
            continue
        if cov >= 0.75:
            okn += 1
        else:
            low.append((cov, k, n, entries[k]["title"], sorted(miss)))
    print(f"  ⑦ 제목 일치            : {okn}건")
    print(f"  ⑧ 본문 텍스트 없음(스캔본): {len(noText)}건" +
          "".join(f"\n      - {k} → [{n}]  자동 대조 불가, 눈으로 확인" for k, n, f in noText))
    print(f"  ⑨ 제목 불일치 의심      : {len(low)}건")
    for cov, k, n, t, miss in sorted(low):
        print(f"      - {cov:.2f} {k} → [{n}]  {t[:56]}")
        print(f"        1쪽에 없는 단어: {', '.join(miss[:8])}")

    notin = sorted(set(live) - index_keys)
    print(f"\n[층3] INDEX.md 색인 최신성")
    print(f"  ⑥ 인용 중이나 INDEX 미등재 : {len(notin)}건")
    if notin:
        print("      " + ", ".join(notin))

    return 1 if undefined else 0


if __name__ == "__main__":
    sys.exit(main())
