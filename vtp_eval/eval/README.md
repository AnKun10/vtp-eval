# lmms-eval benchmarking pipeline

Empty placeholder. The previous implementation lives on branch
`archive/lmms-eval-pre-cleanup` (tag `v0.2-pre-cleanup`).

## When re-implementing, expect

- `run_lmms.py` — orchestrator: `python -m vtp_eval.eval` runs N methods
  on POPE / GQA / TextVQA / MME / MMBench.
- `adapters/` — one file per method (`llava_baseline.py`,
  `llava_fastv.py`, …).
- `utils/` — `tflops.py`, `timing.py`, `result_io.py`.

## Recovering the old implementation

```bash
# Single file
git checkout archive/lmms-eval-pre-cleanup -- vtp_eval/adapters/llava_fastv.py

# Whole tree
git checkout archive/lmms-eval-pre-cleanup
```

## Reference materials

- lmms-eval upstream: https://github.com/EvolvingLMMs-Lab/lmms-eval
- Pre-cleanup adapters: `git show archive/lmms-eval-pre-cleanup:vtp_eval/adapters/`
