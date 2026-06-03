from vtp_eval.insight.prune_viz import datasets


def test_resolve_specs_applies_default_n(tmp_path):
    cfg = {"samples_per_dataset": 4,
           "datasets": {
               "pope": {"hf": "lmms-lab/POPE", "split": "test"},
               "gqa":  {"hf": "lmms-lab/GQA", "split": "testdev", "n": 2,
                        "image_key": "image", "q_key": "question"}}}
    specs = datasets.resolve_specs(cfg)
    assert specs["pope"].n == 4          # falls back to samples_per_dataset
    assert specs["gqa"].n == 2           # explicit override
    assert specs["gqa"].image_key == "image"
    assert specs["pope"].image_key == "image"     # default key
    assert specs["pope"].q_key == "question"      # default key
    assert specs["pope"].config is None           # optional HF config name


def test_resolve_specs_reads_optional_config():
    cfg = {"datasets": {"scienceqa": {"hf": "lmms-lab/ScienceQA",
                                      "config": "ScienceQA-IMG", "split": "test"}}}
    assert datasets.resolve_specs(cfg)["scienceqa"].config == "ScienceQA-IMG"


def test_resolve_specs_requires_hf_path():
    import pytest
    with pytest.raises(KeyError):
        datasets.resolve_specs({"datasets": {"x": {"split": "test"}}})
