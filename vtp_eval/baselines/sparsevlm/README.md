# SparseVLM — Native-Repo Fallback

## Why this is not an in-harness adapter

SparseVLM (ICML 2025) is text-guided and performs **progressive multi-layer pruning with
token recycling** across layers 2, 6, and 15 of LLaVA-1.5. The pruning logic is deeply
woven into the model's attention post-processing hooks (`score.py` / `llava_llama.py`),
making a clean in-harness shim impractical without forking large swaths of the model
code.  Decision: run SparseVLM in **its own official repo** at a tuned `RETAIN_TOKN`,
then merge its `results.json` into our `results/` tree.  Rows from this path are flagged
`harness=sparsevlm_native` in `summary.csv` so the thesis table can mark them.

## The knob: `RETAIN_TOKN`

`RETAIN_TOKN` (env var, integer) is the **initial** number of tokens retained after the
first pruning step.  The natively supported values are **64, 96, 128, 192**.

SparseVLM prunes progressively at three layers (2 → 6 → 15).  The per-layer target
counts in `score.py` (v1.0 / V2.0 modes) are:

| RETAIN_TOKN | layer 2 | layer 6 | layer 15 |
|-------------|---------|---------|----------|
| 192         | 300     | 200     | 110      |
| 128         | 303     | 110     | 36       |
| 96          | 238     | 48      | 26       |
| 64          | 66      | 30      | 17       |

Because pruning is progressive and text-adaptive, the **effective average visual-token
count** seen across all 32 layers is **lower** than `RETAIN_TOKN`.  It **must be measured
on the box** (via our stage_timer's `avg_tokens` field or SparseVLM's own per-sample
logging) before computing TFLOPs efficiency.  Do NOT use `RETAIN_TOKN` directly as
`avg_tokens`.

**To target avg-32 (below the native minimum of 64):** widen the thresholds in
`SparseVLMs/llava/model/language_model/score.py` — add a new entry to `sparse_token_dict`
with values below those of the `64` row.  The file and dict structure make this
straightforward.

## Running an evaluation

Use the helper script from the vtp-eval root:

```bash
# Run on one task at a given RETAIN_TOKN, store output under results/
bash scripts/eval/run_sparsevlm_native.sh <retain_tokn> <task> <run_name>

# Example: MME at retain-128
bash scripts/eval/run_sparsevlm_native.sh 128 mme sparsevlm_retain128
```

The script:
1. `cd`s into the SparseVLM repo (`$SPARSE_DIR`, defaults to `../SparseVLMs` relative to
   this repo).
2. Sets `RETAIN_TOKN` and runs `scripts/v1_5/eval/<task>.sh`.
3. Copies the newest `*.json` produced into `results/<run_name>/<task>/results.json`.
4. Logs stdout to `results/<run_name>/<task>/run.log`.

After the run, populate `results/<run_name>/<task>/timing.json` with the measured
`avg_tokens` and set `"harness": "sparsevlm_native"` in `pruning_meta`:

```json
{
  "encoder_ms": 0, "prefill_ms": 0, "decode_ms": 0,
  "total_latency_ms": 0, "peak_mem_mb": 0,
  "pruning_meta": {
    "method": "sparsevlm_retain128",
    "avg_tokens": <MEASURED_ON_BOX>,
    "harness": "sparsevlm_native"
  }
}
```

Then run `report.py aggregate` as normal; those rows will carry `harness=sparsevlm_native`.

## Efficiency

TFLOPs are computed by the shared `estimate_tflops(avg_tokens)` formula in `report.py`.
Use the **measured** `avg_tokens` (not `RETAIN_TOKN`) as the input.
