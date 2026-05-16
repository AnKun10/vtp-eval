# Design: vtp-eval — Visual Token Pruning Evaluation on POPE via lmms-eval

**Date**: 2026-05-16
**Author**: ankun
**Status**: Draft (pending user review)

## 1. Mục tiêu & Scope

### 1.1 Mục tiêu
Triển khai framework đánh giá thống nhất cho 5 phương pháp visual token pruning trên LLaVA-1.5 7B, dùng lmms-eval làm runner. Phase đầu test trên benchmark POPE; thiết kế dễ mở rộng sang TextVQA/GQA sau.

### 1.2 In-scope
- 5 pruning method: FastV, SparseVLMs, VisionZip, DivPrune, SparseVILA.
- 1 baseline (no pruning) để đối chứng.
- 1 base model: `liuhaotian/llava-v1.5-7b`.
- 1 benchmark: POPE (lmms-eval task `pope`).
- 1 ratio canonical per method (theo paper).
- Metric: POPE accuracy/precision/recall/F1, latency, peak VRAM, theoretical prefill TFLOPs.
- Runtime: Colab Pro với GPU L4 (24GB).
- Output: `summary.csv` + 2 scatter plot (acc-vs-tflops, acc-vs-latency).

### 1.3 Out-of-scope (cho phase này)
- Các benchmark khác (TextVQA, GQA, MME, …).
- Sweep nhiều ratio per method.
- LLaVA-1.6/NeXT, Qwen-VL, các base model khác.
- Multi-turn / video benchmarks.
- AutoAWQ thật cho SparseVILA (dùng NF4 bnb thay thế).
- Auto restart runtime giữa methods (chấp nhận manual restart).

### 1.4 Success criteria
- 6 runs (baseline + 5 methods) hoàn tất trên L4 trong 1 session Colab Pro.
- `summary.csv` đủ 9 cột, không NaN.
- Baseline POPE F1 ∈ [0.85, 0.89].
- Mọi method có TFLOPs prefill và peak VRAM thấp hơn baseline.
- 2 plot accuracy-vs-compression hiển thị được trong notebook.

## 2. Kiến trúc

### 2.1 Directory layout
```
E:\Workspaces\My Projects\DATN\
├── lmms-eval/                  # upstream, KHÔNG sửa
├── FastV/  SparseVLMs/  VisionZip/  divprune/  [man] SparseVILA/   # KHÔNG sửa
└── vtp-eval/                   # workspace mới
    ├── vtp_eval/
    │   ├── __init__.py
    │   ├── adapters/
    │   │   ├── __init__.py
    │   │   ├── _base.py
    │   │   ├── llava_baseline.py
    │   │   ├── llava_fastv.py
    │   │   ├── llava_sparsevlm.py
    │   │   ├── llava_visionzip.py
    │   │   ├── llava_divprune.py
    │   │   └── llava_sparsevila.py
    │   └── utils/
    │       ├── timing.py
    │       ├── tflops.py
    │       └── result_io.py
    ├── install/
    │   ├── _common.sh
    │   ├── _patch_fastv.py
    │   ├── baseline.sh
    │   ├── fastv.sh
    │   ├── sparsevlm.sh
    │   ├── visionzip.sh
    │   ├── divprune.sh
    │   └── sparsevila.sh
    ├── configs/
    │   └── methods.yaml
    ├── notebooks/
    │   └── pope_eval.ipynb
    ├── scripts/
    │   ├── run_one.sh
    │   └── run_all.sh
    ├── results/                # gitignored, .gitkeep only
    ├── tests/
    │   ├── test_adapters_load.py
    │   ├── test_tflops.py
    │   ├── test_result_io.py
    │   └── test_pipeline_smoke.py
    ├── .gitignore
    ├── pyproject.toml
    └── README.md
```

### 2.2 Nguyên tắc
- Adapter là wrapper mỏng (~30-80 dòng), kế thừa adapter `llava` có sẵn của lmms-eval (trừ SparseVILA — custom hoàn toàn).
- Không fork lmms-eval. Dùng `--include_path` flag để register adapter external.
- Mỗi method = 1 install script độc lập + 1 conda env riêng (xung đột `transformers`/`llava` package).
- Notebook driver gọi install script + lmms-eval CLI, **manual restart runtime** giữa các method.

### 2.3 Data flow
```
notebook cell → bash install/<method>.sh
              → python -m lmms_eval --include_path .../adapters \
                                    --model llava_<method> --tasks pope ...
              → results/<method>/{results.json, timing.json, *.jsonl, run.log}
              → result_io.aggregate() → results/summary.csv
              → notebook plot 2 scatter
```

## 3. Adapter design

### 3.1 Base class `_base.py`
```python
from lmms_eval.models.simple.llava import Llava
from vtp_eval.utils.timing import TimingHook

class LlavaPruningBase(Llava):
    def __init__(self, pretrained, log_timing=True, **kwargs):
        super().__init__(pretrained=pretrained, **kwargs)
        self.timing_hook = TimingHook() if log_timing else None
        self.pruning_meta = {}

    def generate_until(self, requests):
        with (self.timing_hook.measure() if self.timing_hook else nullcontext()):
            return super().generate_until(requests)
```

### 3.2 Adapters dễ (kế thừa Llava + set config/env)

**`llava_baseline.py`** — đối chứng:
```python
@register_model("llava_baseline")
class LlavaBaseline(LlavaPruningBase):
    def __init__(self, pretrained, **kw):
        super().__init__(pretrained=pretrained, **kw)
        self.pruning_meta = {"method": "baseline", "keep_ratio": 1.0}
```

**`llava_fastv.py`** — set `model.config.use_fast_v`:
```python
@register_model("llava_fastv")
class LlavaFastV(LlavaPruningBase):
    def __init__(self, pretrained, fast_v_agg_layer=2,
                 fast_v_attention_rank=128, **kw):
        super().__init__(pretrained=pretrained, **kw)
        self._model.config.use_fast_v = True
        self._model.config.fast_v_agg_layer = int(fast_v_agg_layer)
        self._model.config.fast_v_attention_rank = int(fast_v_attention_rank)
        self._model.config.fast_v_sys_length = None
        self._model.config.fast_v_image_token_length = 576
        self._model.model.reset_fastv()
        self.pruning_meta = {"method": "fastv",
                             "K": fast_v_agg_layer, "R": fast_v_attention_rank}
```

**`llava_sparsevlm.py`** — env var trước super:
```python
@register_model("llava_sparsevlm")
class LlavaSparseVLM(LlavaPruningBase):
    def __init__(self, pretrained, retain_token=192, **kw):
        import os
        os.environ["RETAIN_TOKN"] = str(retain_token)
        super().__init__(pretrained=pretrained, **kw)
        self.pruning_meta = {"method": "sparsevlm", "retain": retain_token}
```

**`llava_visionzip.py`**:
```python
@register_model("llava_visionzip")
class LlavaVisionZip(LlavaPruningBase):
    def __init__(self, pretrained, dominant=54, contextual=10, **kw):
        super().__init__(pretrained=pretrained, **kw)
        from visionzip import visionzip
        self._model = visionzip(self._model, dominant=int(dominant),
                                contextual=int(contextual))
        self.pruning_meta = {"method": "visionzip",
                             "dominant": dominant, "contextual": contextual}
```

**`llava_divprune.py`**:
```python
@register_model("llava_divprune")
class LlavaDivPrune(LlavaPruningBase):
    def __init__(self, pretrained, subset_ratio=0.098, layer_index=0, **kw):
        import os
        os.environ.update({"BASELINE": "OURS",
                           "LAYER_INDEX": str(layer_index),
                           "SUBSET_RATIO": str(subset_ratio)})
        super().__init__(pretrained=pretrained, **kw)
        self.pruning_meta = {"method": "divprune", "ratio": subset_ratio}
```

### 3.3 Adapter khó — `llava_sparsevila.py`

SparseVILA dùng `sparse_generate_packed()` thay vì `model.generate()` → bypass `Llava.__init__`, override `generate_until`:

```python
@register_model("llava_sparsevila")
class LlavaSparseVILA(LlavaPruningBase):
    def __init__(self, pretrained="liuhaotian/llava-v1.5-7b",
                 encoder_prune_ratio=0.5, decode_retrieval_ratio=0.5,
                 use_flash_kernel=True, quantize_llm="none", **kw):
        # Bypass Llava.__init__ (load model qua sparsevila API riêng)
        super(Llava, self).__init__()
        from sparsevila import load_sparse_llava, SparseVILAConfig
        cfg = SparseVILAConfig(
            encoder_prune_ratio=encoder_prune_ratio,
            decode_retrieval_ratio=decode_retrieval_ratio,
            use_flash_kernel=use_flash_kernel,
            quantize_llm=quantize_llm,
        )
        self._model, self._image_processor = load_sparse_llava(
            pretrained, config=cfg, dtype=torch.float16, device="cuda")
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=False)
        self._config = cfg
        # set attr lmms-eval cần: self.batch_size_per_gpu, self._device, ...
        self.timing_hook = TimingHook()
        self.pruning_meta = {"method": "sparsevila",
                             "enc_ratio": encoder_prune_ratio,
                             "dec_ratio": decode_retrieval_ratio}

    def generate_until(self, requests):
        from sparsevila import sparse_generate_packed
        results = []
        with self.timing_hook.measure():
            for r in requests:
                ctx, gen_kw, doc_to_visual, doc_id, task, split = r.args
                input_ids, image_tensor, attn_mask = self._prepare_inputs(
                    ctx, doc_to_visual(...))
                out = sparse_generate_packed(
                    self._model, self._tokenizer, input_ids, image_tensor, attn_mask,
                    decode_retrieval_ratio=self._config.decode_retrieval_ratio,
                    max_new_tokens=gen_kw.get("max_new_tokens", 16))
                results.append(self._tokenizer.batch_decode(
                    out, skip_special_tokens=True)[0])
        return results
```

`_prepare_inputs` lấy từ `[man] SparseVILA/notebooks/colab_demo.ipynb`.

### 3.4 Cảnh báo & cách handle
| Vấn đề | Giải pháp |
|---|---|
| 4/5 method patch `modeling_llama.py` chéo nhau | 1 conda env / method + `os.kill(os.getpid(), 9)` restart runtime giữa methods |
| FastV/SparseVLMs cần attention map | Adapter set `attn_implementation="eager"` |
| SparseVILA pin `transformers==4.37.2` | Tất cả method dùng 4.37.2 (VisionZip lên 4.40.0, lifetime ngắn trong cell riêng) |
| `llava` package không trên PyPI | Install scripts `pip install -e` từ submodule LLaVA của từng repo |
| Hardcode 576 visual tokens, 35 sys tokens | Chấp nhận. Scope LLaVA-1.5 7B duy nhất. |

## 4. Install scripts

### 4.1 `install/_common.sh` (chạy 1 lần đầu session)
```bash
set -euo pipefail
cd /content
if [ ! -d lmms-eval ]; then
  git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
fi
pip install -q accelerate==0.34.2 datasets==2.21.0 sentencepiece protobuf==3.20.3
```

### 4.2 `install/fastv.sh`
```bash
set -e
source install/_common.sh
pip install -q transformers==4.37.2 tokenizers==0.15.1
git clone --depth 1 https://github.com/pkunlp-icler/FastV.git /content/FastV || true
python install/_patch_fastv.py
pip install -e /content/lmms-eval
pip install -e /content/FastV/src/LLaVA
```

### 4.3 `install/sparsevlm.sh`
```bash
set -e
source install/_common.sh
pip install -q transformers==4.37.2 tokenizers==0.15.1
git clone --depth 1 https://github.com/Gumpest/SparseVLMs.git /content/SparseVLMs || true
pip install -e /content/SparseVLMs
pip install -e /content/lmms-eval
```

### 4.4 `install/visionzip.sh`
```bash
set -e
source install/_common.sh
pip install -q transformers==4.40.0
git clone --depth 1 https://github.com/dvlab-research/VisionZip.git /content/VisionZip || true
pip install -e /content/VisionZip
pip install -e /content/VisionZip/LLaVA || pip install llava-torch
pip install -e /content/lmms-eval
```

### 4.5 `install/divprune.sh`
```bash
set -e
source install/_common.sh
pip install -q transformers==4.37.2 tokenizers==0.15.1
git clone --depth 1 https://github.com/vbdi/divprune.git /content/divprune || true
pip install -e /content/divprune/LLaVA
pip install -e /content/lmms-eval
```

### 4.6 `install/sparsevila.sh`
```bash
set -e
source install/_common.sh
pip install -q transformers==4.37.2 tokenizers==0.15.1
git clone --depth 1 https://github.com/AnKun10/sparsevila-implementation.git /content/SparseVILA || true
cd /content/SparseVILA
git submodule update --init --recursive
pip install -e ./flash-colreduce
pip install -e .
pip install -e /content/lmms-eval
```

### 4.7 `install/_patch_fastv.py`
```python
import shutil, transformers
from pathlib import Path

tf_dir = Path(transformers.__file__).parent
fastv_src = Path("/content/FastV/src/FastV/inference/transformers_replace")
target = tf_dir / "models" / "llama" / "modeling_llama.py"
target.with_suffix(".py.orig").write_text(target.read_text())  # backup
shutil.copy(fastv_src / "models/llama/modeling_llama.py", target)
print("FastV patch applied to", target)
```

### 4.8 `install/baseline.sh`
```bash
set -e
source install/_common.sh
pip install -q transformers==4.37.2 tokenizers==0.15.1
git clone --depth 1 https://github.com/haotian-liu/LLaVA.git /content/LLaVA || true
pip install -e /content/LLaVA
pip install -e /content/lmms-eval
```

## 5. Notebook orchestration

### 5.1 Cấu trúc `notebooks/pope_eval.ipynb` (~13 cells)

| Cell | Loại | Nội dung |
|---|---|---|
| 1 | md | Tiêu đề, mô tả 6 runs, hướng dẫn restart runtime |
| 2 | code | `!git clone https://github.com/<GITHUB_USER>/vtp-eval.git /content/vtp-eval` + `%cd`. `<GITHUB_USER>` thay bằng username thật khi user đã push repo. Fallback: nếu repo private, dùng `git clone https://<TOKEN>@github.com/...`. |
| 3 | code | `!nvidia-smi`, set `VTP_BATCH_SIZE=2, VTP_DTYPE=float16` (L4 hardcoded) |
| 4 | code | `huggingface_hub.login()` để tải POPE dataset + LLaVA weights |
| 5 | md | # Run 1/6 — Baseline |
| 6 | code | `!bash install/baseline.sh && bash scripts/run_one.sh baseline` |
| 7 | md | # Run 2/6 — FastV. **RESTART RUNTIME trước khi chạy cell này** |
| 8 | code | `import os; os.kill(os.getpid(), 9)` (force restart); after restart: `%cd /content/vtp-eval && !bash install/fastv.sh && bash scripts/run_one.sh fastv_K2_R128` |
| 9-14 | md+code | Tương tự cho sparsevlm, visionzip, divprune, sparsevila |
| 15 | md | # Aggregate & visualize |
| 16 | code | `python -m vtp_eval.utils.result_io aggregate results/ --output results/summary.csv` + 2 scatter plot |

### 5.2 `scripts/run_one.sh`
```bash
#!/usr/bin/env bash
set -e
RUN_NAME=$1
CONFIG=${2:-configs/methods.yaml}

read MODEL MODEL_ARGS < <(python -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG'))
run = next(r for r in cfg['runs'] if r['name']=='$RUN_NAME')
args = ','.join(f'{k}={v}' for k,v in run['model_args'].items())
print(run['model'], 'pretrained=' + cfg['model_base'] + (',' + args if args else ''))
")

OUT_DIR="results/$RUN_NAME"
mkdir -p $OUT_DIR

python -m lmms_eval \
  --include_path /content/vtp-eval/vtp_eval/adapters \
  --model $MODEL \
  --model_args "$MODEL_ARGS,attn_implementation=eager" \
  --tasks pope \
  --batch_size ${VTP_BATCH_SIZE:-2} \
  --log_samples --log_samples_suffix $RUN_NAME \
  --output_path $OUT_DIR \
  2>&1 | tee $OUT_DIR/run.log

python -m vtp_eval.utils.timing parse-log $OUT_DIR/run.log \
  --pruning-meta-from $OUT_DIR/results.json \
  --output $OUT_DIR/timing.json
```

### 5.3 Failure handling
| Tình huống | Hành xử |
|---|---|
| OOM (exit 137) | Log "OOM, skip", tiếp run kế |
| Adapter ImportError | Log rõ → user biết restart runtime |
| POPE download fail | Retry 3 lần với backoff |
| Disconnect giữa chừng | Idempotent: skip nếu `results/<name>/results.json` đã tồn tại |

## 6. Run config & output

### 6.1 `configs/methods.yaml`
```yaml
model_base: liuhaotian/llava-v1.5-7b
common_args:
  attn_implementation: eager
  dtype: float16
runs:
  - { name: baseline,           model: llava_baseline,   model_args: {} }
  - { name: fastv_K2_R128,      model: llava_fastv,      model_args: { fast_v_agg_layer: 2, fast_v_attention_rank: 128 } }
  - { name: sparsevlm_192,      model: llava_sparsevlm,  model_args: { retain_token: 192 } }
  - { name: visionzip_64,       model: llava_visionzip,  model_args: { dominant: 54, contextual: 10 } }
  - { name: divprune_0.098,     model: llava_divprune,   model_args: { subset_ratio: 0.098 } }
  - { name: sparsevila_0.5_0.5, model: llava_sparsevila, model_args: { encoder_prune_ratio: 0.5, decode_retrieval_ratio: 0.5 } }
```

### 6.2 Output structure
```
results/
├── baseline/
│   ├── results.json              # lmms-eval sinh: pope_accuracy, _f1, _yes_ratio
│   ├── llava_baseline_pope_*.jsonl  # per-sample log
│   ├── run.log                   # stdout/stderr
│   └── timing.json               # custom: latency, peak_mem, pruning_meta
├── fastv_K2_R128/
├── sparsevlm_192/
├── visionzip_64/
├── divprune_0.098/
├── sparsevila_0.5_0.5/
└── summary.csv                   # final aggregate
```

### 6.3 `timing.json` schema
```json
{
  "method": "fastv",
  "n_samples": 9000,
  "total_wall_s": 1842.3,
  "latency_per_sample_ms": {"mean": 204.7, "p50": 198.1, "p95": 287.4},
  "peak_gpu_mem_mb": 13420,
  "pruning_meta": {"K": 2, "R": 128}
}
```

### 6.4 `summary.csv` columns
```
method, pope_acc, pope_f1, pope_yes_ratio,
latency_ms_mean, latency_ms_p95, peak_mem_mb,
keep_ratio_pct, tflops_prefill, tflops_reduction_pct
```

### 6.5 TFLOPs computation

`vtp_eval/utils/tflops.py` tính theoretical prefill FLOPs cho LLaVA-1.5 7B:
- Per layer: `attn_proj (4·s·d²) + attn_mm (2·s²·d) + ffn (3·s·d·m)` với `s = sys+visual+text_q+text_a`, `d=4096`, `m=11008`.
- Cộng CLIP encoder FLOPs (24 layer, d=1024).
- Per-method visual shape function (`shape_fastv`, `shape_sparsevlm`, …) trả về `List[int]` length 32 — số visual token tại mỗi LLM layer.
- SparseVLM schedule hardcode từ `SparseVLMs/llava/model/language_model/score.py` `sparse_token_dict`. Giá trị cụ thể trong `shape_sparsevlm()` (vd `{192: [192,128,64], ...}`) là **best-guess** dựa trên design 3-layer-prune (layer 2/6/15); **bắt buộc verify** lại bằng cách đọc thực tế file `score.py` trong implementation phase.

Approximation; bỏ qua RMSNorm/RoPE/SwiGLU activation. Đủ để so sánh tương đối.

### 6.6 Visualization
```python
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("results/summary.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df.plot.scatter(x="tflops_prefill", y="pope_f1", c="method", ax=axes[0])
df.plot.scatter(x="latency_ms_mean", y="pope_f1", c="method", ax=axes[1])
```

## 7. Testing

### 7.1 Offline tests (CPU, không cần GPU)

**`tests/test_adapters_load.py`** — registration & signature:
```python
@pytest.mark.parametrize("name", ["llava_baseline", "llava_fastv", ...])
def test_adapter_registered(name):
    from lmms_eval import models
    cls = models.get_model(name)
    assert "pretrained" in inspect.signature(cls.__init__).parameters
```

**`tests/test_tflops.py`** — sanity số học:
```python
def test_baseline_known_value():
    r = compute_method_tflops("baseline")
    assert 7.0 < r["tflops_total"] < 10.0

def test_pruning_reduces_flops(): ...
def test_visionzip_more_aggressive_than_fastv(): ...
```

**`tests/test_result_io.py`** — fixture results dir, assert CSV columns.

### 7.2 Smoke test (GPU, 5 samples)

**`tests/test_pipeline_smoke.py`** (marker `@pytest.mark.slow`, skip nếu no CUDA):
```python
subprocess.run([
    "python", "-m", "lmms_eval",
    "--include_path", "vtp_eval/adapters",
    "--model", "llava_baseline",
    "--model_args", "pretrained=liuhaotian/llava-v1.5-7b",
    "--tasks", "pope", "--limit", "5",
    "--output_path", "/tmp/smoke",
], check=True)
```

### 7.3 In-notebook smoke (cell 6.5, optional)
```python
!python -m lmms_eval --include_path /content/vtp-eval/vtp_eval/adapters \
  --model llava_baseline --model_args pretrained=liuhaotian/llava-v1.5-7b \
  --tasks pope --limit 5 --output_path results/baseline_smoke
```

### 7.4 Manual verification checklist
- [ ] Baseline `pope_f1` ∈ [0.85, 0.89]
- [ ] Mọi method `peak_mem_mb < baseline.peak_mem_mb`
- [ ] Mọi method `tflops_prefill < baseline.tflops_prefill`
- [ ] `pope_yes_ratio` của các method ∈ [0.45, 0.55]
- [ ] Latency: `visionzip` & `divprune` nhanh nhất (pre-LLM prune); `fastv` chậm hơn

## 8. Prerequisites (trước khi chạy)

1. **GitHub**: Push `vtp-eval/` lên public repo (hoặc private + PAT). Notebook cell 2 sẽ clone từ đó.
2. **HuggingFace account**: Để tải POPE dataset + LLaVA-1.5 weights. `huggingface_hub.login()` trong cell 4.
3. **Colab Pro**: GPU L4 24GB. Free tier T4 không khuyến nghị (thiếu headroom).
4. **SparseVILA repo public**: Đã có tại https://github.com/AnKun10/sparsevila-implementation.
5. **Disk quota Colab**: ~30GB cho 5 method × LLaVA weights cache (clone shallow + symlink).

## 9. Open questions / Phase tiếp theo

- Mở rộng sang TextVQA, GQA, MME: chỉ cần thêm task name vào `--tasks`, không cần sửa adapter.
- Sweep nhiều ratio per method: thêm runs vào `configs/methods.yaml`, notebook iterate.
- LLaVA-1.6 / Qwen-VL support: cần adapter mới + handle dynamic visual token count.
- Multi-turn evaluation: SparseVILA có `SparseCache.snapshot()` nhưng chưa wire.
