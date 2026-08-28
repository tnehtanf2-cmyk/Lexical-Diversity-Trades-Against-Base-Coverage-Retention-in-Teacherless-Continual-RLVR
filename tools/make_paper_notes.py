#!/usr/bin/env python3
"""참고논문/*.pdf 중 노트가 없는 것에 논문노트/[N]_*.md 스텁을 만든다.

스텁이지 요약이 아니다. 서지·인용위치·위키 연결을 기계가 확정해 두고, 정독 후
사람이 요약을 아래에 덧붙이는 방식(기존 노트들의 관행 그대로).

채우는 정보:
  - 제목/서지 : 인용 문헌은 refs.bib(정본), 미인용은 위키 표의 제목 셀 → 파일명 순
  - 인용 여부 : refs.bib 의 `% [N]` 링크 + 본문 \\cite 등장 여부
  - 인용 위치 : paper/*.tex 중 실제로 인용한 파일
  - 위키 클러스터 : 참고논문/INDEX.md 의 '위키 노트' 열
  - 한 줄 설명 : wiki/*.md 에서 그 번호를 다룬 줄을 그대로 인용
  - 중복 : 같은 논문을 두 번호로 소장한 경우 정본 번호로 넘긴다(내용 중복 방지)

실행: python3 tools/make_paper_notes.py [--dry-run] [--force]
"""
import os, re, sys, json, glob, argparse, importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("cc", os.path.join(ROOT, "tools", "citation_check.py"))
cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)

PDF_DIR = os.path.join(ROOT, "참고논문")
NOTE_DIR = os.path.join(ROOT, "논문노트")
WIKI = os.path.join(ROOT, "wiki")
INDEX = os.path.join(PDF_DIR, "INDEX.md")

# 같은 논문을 두 번호로 소장한 쌍. 값이 정본(인용된 쪽, 없으면 먼저 들어온 쪽).
# md5 동일 6쌍 + 판본 차이 1쌍 — tools 로 재검출하려면 첫 페이지 Jaccard 를 쓴다.
DUPLICATE_OF = {24: 11, 143: 3, 50: 40, 22: 26, 121: 141, 129: 146, 112: 173, 198: 38}
DUP_NOTE = {
    24: "바이트 동일", 143: "바이트 동일", 50: "바이트 동일", 22: "바이트 동일",
    121: "바이트 동일", 129: "바이트 동일", 112: "같은 논문의 다른 arXiv 판본",
    198: "구 프리프린트판 — 판본 혼동 방지용 보존",
}


def numbered(d, ext):
    out = {}
    for f in os.listdir(d):
        m = re.match(r"\[(\d+)\][_ ]?(.*)\." + ext + r"$", f)
        if m:
            out[int(m.group(1))] = f
    return out


def wiki_lines():
    """번호별로 위키에서 그 번호를 설명한 줄 하나."""
    out, best = {}, {}
    for f in sorted(glob.glob(os.path.join(WIKI, "*.md"))):
        base = os.path.basename(f)
        if base.startswith("참고문헌-라이브러리"):
            continue                      # 지도는 목록일 뿐 설명이 아니다
        for line in open(f, encoding="utf-8"):
            l = line.strip()
            if not l.startswith(("-", "|", "*")) or len(l) < 40:
                continue
            ns = re.findall(r"\[(\d{1,3})\]", l)
            if len(set(ns)) != 1:
                continue                  # 여러 번호를 한 줄에 묶은 것은 특정 못 한다
            n = int(ns[0])
            if n not in best or len(l) > len(best[n][1]):
                best[n] = (base, l)
    for n, (base, l) in best.items():
        out[n] = (base, re.sub(r"\s+", " ", l))
    return out


def index_cluster():
    out = {}
    if not os.path.exists(INDEX):
        return out
    for line in open(INDEX, encoding="utf-8"):
        c = [x.strip() for x in line.split("|")]
        if len(c) >= 7 and c[1].isdigit() and c[6]:
            out[int(c[1])] = c[6]
    return out


def pretty(stem):
    return re.sub(r"[_]+", " ", stem).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="이미 있는 노트도 덮어쓴다")
    a = ap.parse_args()

    entries = cc.parse_bib(os.path.join(ROOT, "paper", "refs.bib"))
    cited, where = cc.cited_keys(os.path.join(ROOT, "paper"))
    n2key = {v["n"]: k for k, v in entries.items() if v["n"] is not None and k in cited}

    pdfs = numbered(PDF_DIR, "pdf")
    notes = {**numbered(PDF_DIR, "md"), **numbered(NOTE_DIR, "md")}
    wl, cluster = wiki_lines(), index_cluster()

    todo = sorted(n for n in pdfs if a.force or n not in notes)
    made = 0
    for n in todo:
        fname = pdfs[n]
        stem = re.match(r"\[\d+\][_ ]?(.*)\.pdf$", fname).group(1)
        key = n2key.get(n)
        e = entries.get(key, {}) if key else {}

        if key:
            title = e.get("title", "").replace("{", "").replace("}", "")
            first = e.get("author", "").split(" and ")[0].split(",")[0].strip()
            head = f"{first}+ {e.get('year','')} — {title}" if first else title
            venue = e.get("journal", "") or "(서지 미기재)"
            cls = f"논문 인용 — `{key}`"
            loc = ", ".join(sorted(where.get(key, []))) or "—"
        else:
            head = pretty(stem)
            venue = "(미인용 — refs.bib 없음)"
            cls = "미인용 보유"
            loc = "—"

        lines = [f"# [{n}] {head}", "",
                 f"- **분류**: {cls}",
                 f"- **서지**: {venue}",
                 f"- **PDF**: `참고논문/{fname}`"]
        wiki_ref = cluster.get(n)
        lines.append(f"- **위키**: [[{wiki_ref}]] · [[참고문헌-라이브러리]]" if wiki_ref
                     else "- **위키**: [[참고문헌-라이브러리]] (클러스터 미배정)")
        if key:
            lines.append(f"- **인용 위치**: {loc}")
        if n in DUPLICATE_OF:
            c = DUPLICATE_OF[n]
            lines.append(f"- **중복**: [{c}] 과 같은 논문({DUP_NOTE[n]}). "
                         f"**정본은 [{c}]** — 요약·인용은 그쪽에 모을 것")
        if n in wl:
            src, line = wl[n]
            lines += ["", f"**위키의 한 줄** (`wiki/{src}`):", f"> {line}"]
        lines += ["", "> 자동 생성 스텁(2026-08-28, `tools/make_paper_notes.py`). 서지·연결 확정용이며 "
                      "요약은 정독 후 이 아래에 덧붙일 것. **인용·수식 도입 전 원문 대조는 의무** "
                      "([[참고문헌-라이브러리]] 사용 규칙)."]

        path = os.path.join(NOTE_DIR, f"[{n}]_{stem}.md")
        if not a.dry_run:
            open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        made += 1

    print(f"PDF {len(pdfs)}편 · 기존 노트 {len(notes)}편 · 신규 스텁 {made}편"
          + (" (dry-run)" if a.dry_run else ""))
    nocluster = [n for n in todo if n not in cluster]
    nowiki = [n for n in todo if n not in wl]
    print(f"  클러스터 미배정 {len(nocluster)}편 · 위키 한 줄 없음 {len(nowiki)}편")
    if nowiki:
        print("   한 줄 없음:", " ".join(f"[{n}]" for n in nowiki))
    return 0


if __name__ == "__main__":
    sys.exit(main())
