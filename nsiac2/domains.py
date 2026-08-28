"""Domain scheduler + dataset loaders with verifiability types (실험설계 §4/§5).
Each domain: type in {numeric, mc, code, selfconsist}, verifiable flag, a train
prompt pool (TRL-ready: 'prompt' + gold columns) and a held-out eval set.
If a dataset cannot load offline it falls back to a small mock AND records the
domain in USED_MOCK so the run can flag silent degradation (integrity).
"""
from __future__ import annotations
from datasets import load_dataset, Dataset

INSTR = "Solve the problem. Put the final answer after '####'.\n\n"
USED_MOCK: set[str] = set()          # domains that silently fell back to mock


def _mock(gold_col="answer", n=64):
    return Dataset.from_dict({"prompt": ["What is 2+3?\n#### "] * n,
                              gold_col: ["5"] * n})


def _safe(fn, name, gold_col="answer"):
    try:
        return fn()
    except Exception as e:
        print(f"[domains] {name} load failed ({e}); USING MOCK.")
        USED_MOCK.add(name)
        return _mock(gold_col)


# ---------------- loaders ----------------
def load_gsm8k(split):
    ds = load_dataset("gsm8k", "main", split=split)
    return ds.map(lambda r: {"prompt": INSTR + r["question"],
                             "answer": r["answer"].split("####")[-1].strip()},
                  remove_columns=ds.column_names)


def load_mbpp(split):
    ds = load_dataset("mbpp", split=split)

    def fmt(r):
        tests = r["test_list"]
        # expose the required signature via the first assert (reveals function name)
        prompt = (f"{r['text']}\nYour function must pass this test:\n{tests[0]}\n"
                  f"Write only the function definition.\n")
        # code_prefix is what gets PREPENDED to the completion before execution.
        # MBPP's prompt is natural language, so prepending it is a syntax error and the
        # verifier scored every MBPP rollout 0 (bug found 2026-08-05); the completion
        # itself defines the function, so the prefix must be empty here.
        return {"prompt": prompt, "test": "\n".join(tests), "entry_point": "",
                "code_prefix": ""}
    return ds.map(fmt, remove_columns=ds.column_names)


def load_humaneval():
    ds = load_dataset("openai_humaneval", split="test")   # EVAL ONLY
    # HumanEval's prompt IS python (signature + docstring) and the completion is the
    # body, so here the prefix is the prompt itself.
    return ds.map(lambda r: {"prompt": r["prompt"], "test": r["test"],
                             "entry_point": r["entry_point"],
                             "code_prefix": r["prompt"]},
                  remove_columns=ds.column_names)


def _fin_table_md(table) -> str:
    """Linearize FinQA's 2-D table as markdown (same shape as the legacy
    evaluator.format_finqa_table helper, copied to keep nsiac2 free of the
    legacy evaluator's heavyweight imports)."""
    if not table or not isinstance(table, list) or not table[0]:
        return ""
    try:
        head = [str(c) for c in table[0]]
        md = "| " + " | ".join(head) + " |\n| " + " | ".join(["---"] * len(head)) + " |\n"
        for row in table[1:]:
            md += "| " + " | ".join(str(c) for c in row) + " |\n"
        return md
    except Exception:
        return str(table)


def load_finqa(split):
    """FinQA (numeric finance QA over a report table + surrounding text).
    Rides the existing `numeric` verifier route, so no reward/eval change is needed.

    Gold canonicalization: ~57% of golds are percent strings ("53%", "-3.2%") and
    RA.extract_number already drops the '%' (53.0), so the gold is stored as the bare
    number and the prompt instructs the percent convention (answer 12.5, not 0.125).
    ~2% of rows are yes/no (non-numeric) and are dropped, keeping the domain honestly
    type='numeric'. Char caps keep the prompt inside GRPO's 512-token window with the
    question first, so right-truncation never removes the question.

    Evidence: we use FinQA's annotated supporting facts (`gold_inds`, the benchmark's
    gold-evidence/oracle-retrieval setting) rather than an arbitrary prefix of the
    report text, so the task isolates numerical reasoning from retrieval. Measured on
    200 test rows, all program operands are present in the prompt for 94% of items
    under gold evidence vs. 66% under a pre_text prefix, at a shorter prompt; rows
    with empty gold_inds fall back to the pre_text prefix."""
    from . import reward_arbiter as RA
    try:
        ds = load_dataset("ibm-research/finqa", split=split)
    except Exception:
        from datasets import load_from_disk
        ds = load_from_disk("n_siac/data/FinQA")[split]

    def fmt(r):
        gold = (r.get("answer") or "").strip() or (r.get("final_result") or "").strip()
        num = RA.extract_number(gold)
        ev = " ".join(r.get("gold_inds") or []) or " ".join(r.get("pre_text") or [])
        prompt = (INSTR + "The answer is a number; for percentages give the percent "
                  "value (e.g. '#### 12.5' for 12.5%).\n"
                  f"Question: {r['question']}\n"
                  f"{_fin_table_md(r.get('table'))[:900]}\n{ev[:600]}\n")
        return {"prompt": prompt, "answer": "" if num is None else str(num)}

    ds = ds.map(fmt, remove_columns=ds.column_names)
    return ds.filter(lambda r: r["answer"] != "")


def load_medmcqa(split):
    ds = load_dataset("medmcqa", split=split)
    L = "ABCD"

    def fmt(r):
        opts = "\n".join(f"{L[i]}. {r['op'+c]}" for i, c in enumerate("abcd"))
        return {"prompt": f"{r['question']}\n{opts}\nAnswer with a single letter.\n#### ",
                "answer": L[r["cop"]]}
    return ds.map(fmt, remove_columns=ds.column_names)


# ---------------- registry ----------------
# NOTE: LegalOpen / MedicalOpen loaders are TODO (they also need a rubric reward and
# a non-zero eval path for `selfconsist`, see paper Limitations). Until implemented they
# fall back to mock and appear in USED_MOCK — do NOT report per-domain results for
# them. Default runs should use IMPLEMENTED domains (see main --order).
DOMAINS = {
    "GSM8K":     dict(type="numeric", verifiable=True,
                      train=lambda: _safe(lambda: load_gsm8k("train"), "GSM8K"),
                      eval=lambda: _safe(lambda: load_gsm8k("test"), "GSM8K")),
    "Code":      dict(type="code", verifiable=True,
                      train=lambda: _safe(lambda: load_mbpp("train"), "Code", "test"),
                      eval=lambda: _safe(load_humaneval, "Code", "test")),
    "MedicalMC": dict(type="mc", verifiable=True,
                      train=lambda: _safe(lambda: load_medmcqa("train"), "MedicalMC"),
                      eval=lambda: _safe(lambda: load_medmcqa("validation"), "MedicalMC")),
    "FinQA":     dict(type="numeric", verifiable=True,
                      train=lambda: _safe(lambda: load_finqa("train"), "FinQA"),
                      eval=lambda: _safe(lambda: load_finqa("test"), "FinQA")),
    "LegalOpen": dict(type="selfconsist", verifiable=False,          # TODO LegalBench open
                      train=lambda: _safe(lambda: (_ for _ in ()).throw(NotImplementedError()), "LegalOpen"),
                      eval=lambda: _safe(lambda: (_ for _ in ()).throw(NotImplementedError()), "LegalOpen")),
    "MedicalOpen": dict(type="selfconsist", verifiable=False,        # TODO long-form medical
                        train=lambda: _safe(lambda: (_ for _ in ()).throw(NotImplementedError()), "MedicalOpen"),
                        eval=lambda: _safe(lambda: (_ for _ in ()).throw(NotImplementedError()), "MedicalOpen")),
}

IMPLEMENTED = ["GSM8K", "Code", "MedicalMC", "FinQA"]     # real loaders available now


class DomainScheduler:
    def __init__(self, order, shift_interval=10):
        self.order = order
        self.shift = shift_interval

    def domain(self, cycle: int) -> str:
        return self.order[((cycle - 1) // self.shift) % len(self.order)]

    def is_shift(self, cycle: int) -> bool:
        return cycle > 1 and (cycle - 1) % self.shift == 0
