#!/usr/bin/env python3
"""참고논문/INDEX.md 재생성.

각 [N] 에 대해 PDF·노트 보유, refs.bib 인용 키, 위키 노트 배정을 한 표로 만든다.
bib key 는 refs.bib 의 `% [N]` 링크(citation_check.py 가 검증하는 그 링크)에서
역으로 끌어온다 — 두 파일이 어긋날 수 없게 하려는 것이다.
위키 노트 열은 기존 INDEX.md 의 배정을 그대로 물려받고, 새 항목은 빈칸으로 둔다.
"""
import os, re, sys, importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("cc", os.path.join(ROOT, "tools", "citation_check.py"))
cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)

PDF_DIR = os.path.join(ROOT, "참고논문")
NOTE_DIRS = [os.path.join(ROOT, "논문노트"), PDF_DIR]
INDEX = os.path.join(PDF_DIR, "INDEX.md")

entries = cc.parse_bib(os.path.join(ROOT, "paper", "refs.bib"))
cited, _ = cc.cited_keys(os.path.join(ROOT, "paper"))
n2key = {v["n"]: k for k, v in entries.items() if v["n"] is not None}

def collect(d, ext):
    out = {}
    for f in os.listdir(d):
        m = re.match(r"\[(\d+)\]_(.+)\." + ext + r"$", f)
        if m:
            out[int(m.group(1))] = m.group(2)
    return out

pdfs = collect(PDF_DIR, "pdf")
notes = {}
for d in NOTE_DIRS:
    notes.update(collect(d, "md"))

prev = {}
if os.path.exists(INDEX):
    for line in open(INDEX, encoding="utf-8"):
        c = [x.strip() for x in line.split("|")]
        if len(c) >= 7 and c[1].isdigit():
            prev[int(c[1])] = c[6]

rows = []
for n in sorted(set(pdfs) | set(notes)):
    name = pdfs.get(n) or notes.get(n, "")
    key = n2key.get(n)
    keycol = f"`{key}`" if key and key in cited else ("(미인용)" if not key else f"`{key}`(미인용)")
    rows.append(f"| {n} | {name[:52]} | {'✓' if n in pdfs else '—'} | "
                f"{'✓' if n in notes else '—'} | {keycol} | {prev.get(n, '')} |")

linked = sum(1 for n in n2key if n in pdfs)
out = [
    "# 참고논문 색인",
    "",
    f"자동 생성: `python3 tools/rebuild_ref_index.py` · 최종 {os.popen('date +%Y-%m-%d').read().strip()}",
    "",
    "위키 진입점: `wiki/참고문헌-라이브러리.md` (옵시디언에서 열면 분야별 노트로 연결)",
    "",
    f"총 {len(rows)}항목 · PDF {len(pdfs)}편 · 노트 {len(notes)}편 · "
    f"논문 인용 {len(cited)}키 중 {linked}키가 원문 연결됨.",
    "",
    "bib key 열은 `paper/refs.bib` 의 `% [N]` 주석에서 역으로 끌어온다. "
    "무결성 검사는 `python3 tools/citation_check.py`.",
    "",
    "| [N] | 파일명 | PDF | 노트 | bib key | 위키 노트 |",
    "|---|---|---|---|---|---|",
] + rows
open(INDEX, "w", encoding="utf-8").write("\n".join(out) + "\n")
print(f"INDEX.md 재생성: {len(rows)}행 (PDF {len(pdfs)} · 노트 {len(notes)} · 원문연결 {linked}/{len(cited)})")
