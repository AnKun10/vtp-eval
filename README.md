# vtp-eval

Visual Token Pruning research workbench for VLMs (currently targeting
LLaVA-1.5-7B). Organized so each top-level concern lives in one folder.

> ⚠️ The lmms-eval benchmarking pipeline was temporarily removed in
> commit `e7198c7` and will be re-implemented in
> `vtp_eval/eval/`. The pre-cleanup state is preserved on branch
> `archive/lmms-eval-pre-cleanup` (tag `v0.2-pre-cleanup`).

## Where things live

| Concern | Folder | Status |
|---------|--------|--------|
| Interpretability insight visualization (currently: Figure 3 of LearnPruner ICLR 2026) | [`vtp_eval/insight/text_visual_attention/`](vtp_eval/insight/text_visual_attention/) | ✅ done |
| Proposed pruning method (thesis contribution) | [`vtp_eval/proposed_method/`](vtp_eval/proposed_method/) | ⏳ empty placeholder |
| Benchmarking pruning methods via lmms-eval | [`vtp_eval/eval/`](vtp_eval/eval/) | ⏳ empty placeholder |
| Vast.ai deployment template + onstart | [`scripts/vast/`](scripts/vast/) + [`docs/vast.md`](docs/vast.md) | ✅ done |

## Top-level layout

```
vtp-eval/
├── vtp_eval/                       # Python package
│   ├── insight/                    # interpretability analyses
│   │   └── text_visual_attention/  #   one analysis per subpackage
│   ├── proposed_method/            # thesis method (empty)
│   └── eval/                       # lmms-eval (empty)
├── configs/                        # YAML configs
├── docs/                           # per-concern write-ups
├── install/                        # per-concern install scripts
├── notebooks/                      # interactive Jupyter front-ends
├── scripts/                        # CLI runners
│   ├── text_visual_attention/      #   wrappers for the insight CLI
│   └── vast/                       #   Vast on-start script
└── tests/                          # (empty for now)
```

## Quick start

| I want to … | Read |
|-------------|------|
| Reproduce Figure 3 of LearnPruner on Vast.ai | [`docs/vast.md`](docs/vast.md) (operator runbook) |
| Understand the Figure 3 experiment | [`docs/text_visual_attention.md`](docs/text_visual_attention.md) (write-up) |
| Recover the old lmms-eval code | `git checkout archive/lmms-eval-pre-cleanup` |
| Plan my thesis pruning method | [`docs/proposed_method.md`](docs/proposed_method.md) (design notes) |
| Plan the lmms-eval rewrite | [`docs/eval.md`](docs/eval.md) (design notes) |

## License & references

- LearnPruner (ICLR 2026): the insight reproduced in `vtp_eval/insight/text_visual_attention/`.
- LLaVA-1.5-7B: [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf).
- lmms-eval (when re-implemented): https://github.com/EvolvingLMMs-Lab/lmms-eval.
- Prior pruning methods (in the archive branch): FastV, VisionZip, SparseVLMs, DivPrune, SparseVILA.
