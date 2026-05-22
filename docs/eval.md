# Benchmarking pipeline (lmms-eval) — design notes

Placeholder. The previous implementation is archived on branch
`archive/lmms-eval-pre-cleanup`. When re-implementing, document:

- Supported benchmarks (POPE, GQA, TextVQA, MME, MMBench, …).
- Per-method adapter interface (was `LlavaPruningBase` — see archive).
- TFLOPs / latency / memory measurement (was `vtp_eval/utils/`).
- Result aggregation (CSV + scatter plots — was
  `vtp_eval/utils/result_io.py`).

## Cross-references

- `vtp_eval/eval/` — source code (empty for now).
- Archive: `git checkout archive/lmms-eval-pre-cleanup` for the full
  prior state.
