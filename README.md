# Lexical Diversity Trades Against Base-Coverage Retention in Teacherless Continual RLVR

Code, run logs, and analysis scripts for the manuscript:

> Deahun Kang and Taeyeon Oh. *Lexical Diversity Trades Against Base-Coverage
> Retention in Teacherless Continual RLVR: A Preregistered Dose–Response, a
> Harvest-Source Ablation, and Operating-Point Selection.* Submitted to
> **Neurocomputing**, 2026.

This repository is the deposit referenced in the manuscript's *Data availability*
statement. Archived at Zenodo: [10.5281/zenodo.22672684](https://doi.org/10.5281/zenodo.22672684)
(all versions; the manuscript cites the v1.0.0 snapshot, 10.5281/zenodo.22672685). Every table in the paper is regenerated from the logs in `logs/` by the
scripts in `paper/` — see [Reproducing the tables](#reproducing-the-tables).

## What is here

| Path | Contents |
|---|---|
| `nsiac2/` | The experiment package: meta-controller, state tracker, reward arbiter, harvest, domains, evaluator |
| `main_nsiac_v2.py` | Entry point for a continual multi-domain RLVR run |
| `scripts/` | The exact run scripts used for each experiment block (v2i–v2o, sweeps) |
| `paper/` | Analysis scripts that emit the manuscript's LaTeX tables and summary JSON |
| `logs/` | Per-cycle run logs: 104 CSV, 203 JSONL, 57 plain logs |
| `tools/` | Auxiliary measurement utilities (coverage/entropy, bf16 decay check) |
| `plan/` | The preregistration records the manuscript cites: hypotheses, decision thresholds, and verdicts including the six predictions that failed |

### Log file conventions

For a run named `<run>`:

- `<run>.csv` — per-cycle metrics (accuracy, retention, diversity, knob values)
- `<run>_samples.jsonl` — **per-prompt solve vectors**. One record per evaluated
  prompt per cycle, with fields `prompt`, `gold`, `completions`, `correct`,
  `cycle`, `eval_domain`. `correct` is the per-sample solve vector used for
  pass@k and coverage-retention.
- `<run>_meta.jsonl` — per-cycle metadata and configuration snapshot
- `<run>.log` — stdout trace

Run-name prefixes map to the manuscript's experiment blocks: `sweep_*` and
`sweep2_*` (knob sweeps and the five-level entropy dose–response), `v2i_*`/`v2j_*`
(harvest-source work), `v2k_*` (main table), `v2l_*` (clean re-run, scale ladder),
`v2n_*` (multi-seed), `v2o_*` (harvest-source ablation), `full_base_*` (frozen-base
reference environment).

One edit was made to the logged completions before release: six occurrences of five
email addresses that the model emitted inside generated text were replaced with
`redacted@example.com`. The addresses appear in no prompt or gold answer, and no
metric is computed from the `completions` field, so every table and figure in the
manuscript regenerates unchanged.

## Reproducing the tables

Requires Python 3.12 with `pandas`, `numpy`, `scipy` (see `requirements.txt`).
No GPU and no model download are needed — the analysis reads the deposited logs only.

```bash
python3 paper/make_results_v2k.py        # -> paper/results_main.tex
python3 paper/make_results_v2l.py        # -> paper/results_sweep.tex, results_scale.tex, results_ablation.tex
python3 paper/make_results_v2o.py        # -> paper/results_harvest.tex, results_seeds.tex
python3 paper/make_results_dose.py       # -> paper/results_dose.tex, paper/figs/dose_response.pdf
python3 paper/make_crystallization.py    # -> paper/results_crystallization.json
```

Each script rewrites its `.tex` file in place. The versions committed here are the
ones typeset in the manuscript; re-running the scripts on a clean checkout
reproduces them byte-for-byte (verified 2026-08-28, 7/7 tables identical).

## Re-running the experiments

The training runs need an NVIDIA GPU and are executed inside the NGC PyTorch
container. Paths are parameterized — the scripts resolve the repository root
automatically, or you can override it:

```bash
export HF_TOKEN=...            # your own Hugging Face token
REPO_ROOT=$(pwd) ./scripts/run_v2m_dose.sh
```

Each block takes hours to days on a single GPU (e.g. the v2m dose–response is six
runs at roughly 2.3 h each). The scripts skip any run whose `logs/<run>.csv`
already exists, so they are safe to resume. They also append a one-line record to
`plan/실험기록.md`; that file is lab bookkeeping and is not needed for reproduction.

Models are Qwen2.5 (1.5B / 3B / 7B) and datasets are the public GSM8K, MBPP,
HumanEval, and MedMCQA — neither is redistributed here; both are fetched from the
Hugging Face Hub at run time.

## Scope of this deposit

The deposit covers the experiment blocks reported in the manuscript. Earlier
exploratory experiments that the manuscript does not cite, model checkpoints, and
the manuscript sources themselves are not included.

## License

- **Code** (`nsiac2/`, `main_nsiac_v2.py`, `scripts/`, `paper/`, `tools/`) — MIT, see [`LICENSE`](LICENSE)
- **Logs and derived data** (`logs/`, `paper/results_*.json`, `paper/results_*.tex`) — CC BY 4.0, see [`LICENSE-data`](LICENSE-data)
- **Preregistration records** (`plan/`) — CC BY 4.0, as above

The repository was initialised in August 2026 with an Apache-2.0 file before any content
existed. That file is superseded by the two licences above, which are the ones the
deposit is released under and the ones `CITATION.cff` records.

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the manuscript when using this code or data.
