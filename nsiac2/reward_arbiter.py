"""Reward-source arbitration (실험설계 §4). Per domain, pick the reward source:
  - numeric   : rule verifier (math GSM8K/MATH, finance FinQA)  -> exact numeric match
  - mc        : rule verifier (medical MedQA/PubMedQA labels)   -> label match
  - code      : rule verifier (MBPP/HumanEval) -> unit-test execution (guarded)
  - selfconsist: NON-verifiable (legal/medical long-form) -> semantic self-consistency
No external strong LLM judge is used as an answer source (teacherless).
TRL reward funcs: fn(prompts, completions, **cols) -> list[float]; TRL repeats each
dataset column per generation, so cols[...] aligns 1:1 with completions.

SECURITY: the code verifier runs model-generated code in a subprocess with a
timeout. Run ONLY inside the disposable docker container, never on the host.
"""
from __future__ import annotations
import re
import math
import subprocess
import sys
import tempfile
import os
from collections import Counter

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_number(text: str):
    text = str(text).split("####")[-1]
    m = list(_NUM.finditer(text.replace(",", "")))
    if not m:
        return None
    try:
        return float(m[-1].group())
    except ValueError:
        return None


def extract_choice(text: str):
    tail = str(text).split("####")[-1]           # target the final-answer region
    ms = re.findall(r"\b([A-E])\b", tail.upper())
    if ms:
        return ms[-1]                              # last standalone letter
    ms = re.findall(r"\b(yes|no|maybe)\b", tail.lower())
    if ms:
        return ms[-1]
    # fall back to whole text if the tail had nothing
    ms = re.findall(r"\b([A-E])\b", str(text).upper())
    return ms[-1] if ms else None


# ----------------- verifiers -----------------
def numeric_correct(completion: str, gold) -> float:
    p, g = extract_number(completion), extract_number(str(gold))
    if p is None or g is None:
        return 0.0
    return 1.0 if math.isclose(p, g, rel_tol=1e-3, abs_tol=1e-4) else 0.0


def mc_correct(completion: str, gold) -> float:
    p, g = extract_choice(completion), extract_choice(str(gold))
    return 1.0 if (p is not None and g is not None and p == g) else 0.0


def code_correct(prompt: str, completion: str, test: str,
                 entry_point: str = "", timeout=8) -> float:
    """Execute (prompt + completion + test) in a subprocess. `prompt` supplies the
    function signature (HumanEval) or is empty (MBPP, where the completion defines
    the function). Guarded; docker-only."""
    prog = (prompt or "") + completion + "\n" + (test or "") + "\n"
    if entry_point and "check(" in prog and f"check({entry_point})" not in prog:
        prog += f"check({entry_point})\n"
    if "def " not in prog:
        return 0.0
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(prog); path = f.name
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout)
        return 1.0 if r.returncode == 0 else 0.0
    except Exception:
        return 0.0
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


# ----------------- reward-function factory (TRL) -----------------
def make_reward_fn(domain_type: str, gold_col: str = "answer"):
    def numeric_fn(prompts, completions, **cols):
        gold = cols.get(gold_col, [None] * len(completions))
        return [numeric_correct(c, g) for c, g in zip(completions, gold)]

    def mc_fn(prompts, completions, **cols):
        gold = cols.get(gold_col, [None] * len(completions))
        return [mc_correct(c, g) for c, g in zip(completions, gold)]

    def code_fn(prompts, completions, **cols):
        tests = cols.get("test", [""] * len(completions))
        eps = cols.get("entry_point", [""] * len(completions))
        # Execute code_prefix + completion, NOT prompt + completion: MBPP prompts are
        # natural language and prepending them made every MBPP rollout a SyntaxError
        # (reward identically 0 in every run before 2026-08-05). Fall back to the prompt
        # only if the column is absent, which reproduces the old behaviour for datasets
        # whose prompt is itself code.
        pre = cols.get("code_prefix")
        if pre is None:
            pre = prompts
        return [code_correct(p, c, t, e)
                for p, c, t, e in zip(pre, completions, tests, eps)]

    def selfconsist_fn(prompts, completions, **cols):
        groups: dict[str, list[int]] = {}
        for i, p in enumerate(prompts):
            groups.setdefault(p, []).append(i)
        rewards = [0.0] * len(completions)
        for idxs in groups.values():
            keys = [_answer_key(completions[i]) for i in idxs]
            top, n = Counter(keys).most_common(1)[0]
            frac = n / len(idxs)
            for i, k in zip(idxs, keys):
                rewards[i] = frac if k == top else 0.0
        return rewards

    return {"numeric": numeric_fn, "mc": mc_fn, "code": code_fn,
            "selfconsist": selfconsist_fn}[domain_type]


def _answer_key(text: str, n=12) -> str:
    a = extract_number(text)
    if a is None:
        a = extract_choice(text)
    if a is not None:
        return str(a).lower()
    w = re.findall(r"[a-zA-Z가-힣]+", str(text).lower())   # free-form semantic key
    return " ".join(w[:n])
