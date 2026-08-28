#!/usr/bin/env python3
"""wiki_coverage — 참고논문/ 의 PDF 중 wiki/ 에 아직 정리되지 않은 것을 찾는다.

왜 필요한가. 참고논문 폴더는 다운로드로 계속 자라는데 wiki/ 반영은 수작업이라
누락이 생긴다(2026-08-07 트리아지 25편이 범위 표기로만 언급되고 개별 정리가
빠진 것이 실제 사례). 이 스크립트가 그 격차를 기계적으로 드러낸다.

커버리지 판정 규칙:
  - wiki/*.md 안의 개별 언급 `[N]`
  - 범위 언급 `[N]-[M]` / `[N]–[M]`(en-dash) / `[N]—[M]` → N..M 전체를 커버로 간주
  - 한 번이라도 언급되면 커버. "얼마나 잘 정리됐는가"는 판정하지 않는다(사람 몫).

사용:
  python3 tools/wiki_coverage.py            # 사람이 읽는 보고
  python3 tools/wiki_coverage.py --quiet    # 미커버가 있을 때만 한 줄 출력(훅용)
종료코드는 항상 0 — 훅에서 워크플로를 막지 않기 위해서다.
"""
import argparse, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPERS = os.path.join(ROOT, "참고논문")
WIKI = os.path.join(ROOT, "wiki")
MOC = os.path.join(WIKI, "참고문헌-라이브러리.md")

NUM_IN_NAME = re.compile(r"^\[(\d{1,3})\]")
SINGLE = re.compile(r"\[(\d{1,3})\]")
RANGE = re.compile(r"\[(\d{1,3})\]\s*[-–—]\s*\[?(\d{1,3})\]?")



def map_note_consistency():
    """지도(참고문헌-라이브러리.md) 표가 가리키는 번호가 해당 클러스터 노트에 실제로 있는가.

    wiki_coverage 본체는 "어딘가에 언급됐는가"만 본다. 그래서 지도 표에 `[196]`
    이라고 써 놓고 정작 그 노트에는 안 써 넣은 상태가 통과해 버린다(실제로
    2026-08-28 에 그렇게 어긋나 있었다). 이 함수가 그 한 칸을 더 좁힌다.
    """
    if not os.path.exists(MOC):
        return []
    def expand(s):
        out = set()
        for a, b in re.findall(r"\[?(\d{1,3})\]?\s*[-–—]\s*\[?(\d{1,3})\]?", s):
            out |= set(range(int(a), int(b) + 1))
        for m in re.finditer(r"(?<![\d–—-])(\d{1,3})(?![\d–—-])", s.replace("[", "").replace("]", "")):
            out.add(int(m.group(1)))
        return out

    bad = []
    for line in open(MOC, encoding="utf-8"):
        c = [x.strip() for x in line.split("|")]
        if len(c) < 6:
            continue
        m = re.match(r"\[\[(.+?)\]\]", c[2])
        if not m or not re.search(r"\d", c[3]):
            continue
        note = os.path.join(WIKI, m.group(1) + ".md")
        if not os.path.exists(note):
            bad.append((m.group(1), "노트 파일 없음"))
            continue
        miss = sorted(expand(c[3]) - expand(open(note, encoding="utf-8").read()))
        if miss:
            bad.append((m.group(1), f"지도엔 있으나 노트에 없는 번호 {miss}"))
    return bad


def papers_on_disk():
    out = {}
    if not os.path.isdir(PAPERS):
        return out
    for fn in os.listdir(PAPERS):
        if not fn.lower().endswith(".pdf"):
            continue
        m = NUM_IN_NAME.match(fn)
        if m:
            out[int(m.group(1))] = fn
    return out


def numbers_in_moc():
    """라이브러리 지도(참고문헌-라이브러리.md)가 색인하는 번호. 개별 노트에만 있고
    지도에서 빠지면 '찾을 수 없는 문헌'이 되므로 별도 층으로 검사한다."""
    if not os.path.exists(MOC):
        return set()
    text = open(MOC, encoding="utf-8", errors="ignore").read()
    hits = set()
    for a, b in RANGE.findall(text):
        lo, hi = sorted((int(a), int(b)))
        if hi - lo <= 60:
            hits.update(range(lo, hi + 1))
    # 지도는 "87–111" 처럼 대괄호 없는 범위도 쓴다
    for a, b in re.findall(r"(?<!\d)(\d{2,3})\s*[-–—]\s*(\d{2,3})(?!\d)", text):
        lo, hi = sorted((int(a), int(b)))
        if hi - lo <= 60:
            hits.update(range(lo, hi + 1))
    # "[N] ≤ 86" / "86 이하" 처럼 상한만 선언하는 층 표기
    for b in re.findall(r"[≤<]=?\s*(\d{1,3})", text) + re.findall(r"(\d{1,3})\s*이하", text):
        hits.update(range(1, int(b) + 1))
    hits.update(int(n) for n in SINGLE.findall(text))
    return hits


def numbers_in_wiki():
    covered, where = set(), {}
    if not os.path.isdir(WIKI):
        return covered, where
    for fn in sorted(os.listdir(WIKI)):
        if not fn.endswith(".md"):
            continue
        text = open(os.path.join(WIKI, fn), encoding="utf-8", errors="ignore").read()
        hits = set()
        for a, b in RANGE.findall(text):          # ranges first: [112]–[122]
            lo, hi = sorted((int(a), int(b)))
            if hi - lo <= 60:                     # sanity: not a year span or typo
                hits.update(range(lo, hi + 1))
        hits.update(int(n) for n in SINGLE.findall(text))
        for n in hits:
            where.setdefault(n, []).append(fn)
        covered |= hits
    return covered, where


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true",
                    help="미커버가 있을 때만 한 줄 출력(훅용)")
    a = ap.parse_args()

    disk = papers_on_disk()
    covered, _ = numbers_in_wiki()
    indexed = numbers_in_moc()
    missing = sorted(n for n in disk if n not in covered)          # 어디에도 언급 없음
    unindexed = sorted(n for n in disk if n in covered and n not in indexed)

    if a.quiet:
        parts = []
        if missing:
            nums = ", ".join(f"[{n}]" for n in missing[:10])
            parts.append(f"미반영 {len(missing)}편({nums}{' 외' if len(missing) > 10 else ''})")
        if unindexed:
            parts.append(f"라이브러리 지도 미색인 {len(unindexed)}편")
        if map_note_consistency():
            parts.append(f"지도↔노트 불일치 {len(map_note_consistency())}건")
        if parts:
            print(f"📚 참고논문 wiki 정리 필요 — {' / '.join(parts)}. "
                  f"`python3 tools/wiki_coverage.py`로 목록 확인.")
        return 0

    drift = map_note_consistency()
    print(f"참고논문 PDF {len(disk)}편 · wiki 언급 {len(covered & set(disk))}편 · "
          f"미반영 {len(missing)}편 · 지도 미색인 {len(unindexed)}편 · 지도↔노트 불일치 {len(drift)}건")
    if missing:
        print("\n미반영(어느 위키 노트에도 없음):")
        for n in missing:
            print(f"  [{n}] {disk[n]}")
    if unindexed:
        print(f"\n지도 미색인(노트엔 있으나 {os.path.basename(MOC)}에 없음):")
        for n in unindexed:
            print(f"  [{n}] {disk[n]}")
    if drift:
        print("\n지도↔노트 불일치:")
        for note, why in drift:
            print(f"  {note}: {why}")
    if not missing and not unindexed and not drift:
        print("전부 wiki에 반영·색인됨.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
