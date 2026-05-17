# vtp-eval

Visual Token Pruning evaluation harness for LLaVA-1.5 7B on POPE, via [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

Compares **5 default configs** (FastV is documented but blocked on Colab Python 3.12 — see [`docs/COLAB_FIXES.md`](docs/COLAB_FIXES.md#18) #18):

| Run | Method | Canonical config | Colab status |
|---|---|---|---|
| `baseline` | none | full 576 visual tokens | ✅ |
| `sparsevlm_192` | SparseVLMs | retain=192 multi-layer schedule | ✅ (batch=1) |
| `visionzip_64` | VisionZip | dominant=54 + contextual=10 | ✅ |
| `divprune_0.098` | DivPrune | subset ratio 9.8% | ✅ |
| `sparsevila_0.5_0.5` | SparseVILA | encoder prune 0.5, decode retrieve 0.5 | ✅ |
| `fastv_K2_R128` | FastV | prune after layer 2, keep 128 | ❌ blocked (Py 3.12) |

## Setup

This package is meant to run on **Colab Pro (L4 GPU, 24 GB)**. Each pruning method patches `transformers`/`llava` in incompatible ways, so the notebook restarts the kernel between methods.

### Run on Colab — main 5-method eval
1. Open `notebooks/pope_eval.ipynb` in Colab Pro.
2. Set runtime → L4 GPU.
3. Run cells sequentially. Restart runtime when prompted (markdown cells flag the spot).

### Run on Colab — FastV (separate notebook, Python 3.10)
FastV is blocked under Colab's default Python 3.12 (see [`docs/COLAB_FIXES.md` #18](docs/COLAB_FIXES.md)).
Use `notebooks/fastv_eval.ipynb`, which switches the kernel to Python 3.10 via `condacolab` before installing FastV. Experimental — expect a one-time auto-restart after the conda install cell.

### Run locally (single method, A100 / RTX 4090)
```bash
pip install -e ".[test]"
bash install/baseline.sh           # only one method per env
bash scripts/run_one.sh baseline
```
For all 6 methods locally you need separate conda envs (see `install/*.sh`).

## Output

After all runs, `notebooks/pope_eval.ipynb` aggregates into `results/summary.csv` with columns:
- `pope_acc`, `pope_f1`, `pope_yes_ratio`
- `latency_ms_mean`, `latency_ms_p95`, `peak_mem_mb`
- `keep_ratio_pct`, `tflops_prefill`, `tflops_reduction_pct`

Plus `results/scatter.png` — accuracy-vs-compute and accuracy-vs-latency scatter plots.

## Layout
```
vtp_eval/
  adapters/   # one lmms-eval adapter per method
  utils/      # tflops.py, timing.py, result_io.py
install/      # bash script per method (deps + clone + pip install -e)
configs/      # methods.yaml (canonical per-method ratios)
scripts/      # run_one.sh, run_all.sh
notebooks/    # pope_eval.ipynb (+ generator)
tests/        # pytest: TFLOPs math, result aggregation, smoke
docs/         # design spec + this plan
```

## Adding a new method
1. Add adapter at `vtp_eval/adapters/llava_<name>.py` inheriting `LlavaPruningBase`.
2. Import it in `vtp_eval/adapters/__init__.py`.
3. Add entry to `tests/test_adapters_load.py` ADAPTERS list.
4. Add a `shape_<name>()` to `vtp_eval/utils/tflops.py` + register in `_SHAPE_FNS`.
5. Add a branch in `compute_keep_ratio()` and `_tflops_kwargs_from_meta()` in `vtp_eval/utils/result_io.py`.
6. Add an `install/<name>.sh`.
7. Add an entry to `configs/methods.yaml`.
8. Add 1 run block in `notebooks/_build_pope_eval.py` and regenerate the notebook.

## Adding a new benchmark
Pass `--tasks <name>` to lmms-eval (already supported by `run_one.sh`'s third arg). No code changes required — POPE was chosen as the first benchmark; TextVQA / GQA / MME / MMBench all work the same way.

## License & references
- FastV: https://github.com/pkunlp-icler/FastV
- SparseVLMs: https://github.com/Gumpest/SparseVLMs
- VisionZip: https://github.com/dvlab-research/VisionZip
- DivPrune: https://github.com/vbdi/divprune
- SparseVILA: https://github.com/AnKun10/sparsevila-implementation
- lmms-eval: https://github.com/EvolvingLMMs-Lab/lmms-eval
