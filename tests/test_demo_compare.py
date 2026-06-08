# tests/test_demo_compare.py
from vtp_eval.demo.compare import format_comparison


def test_format_comparison_hit_shows_deterministic_vision_saving():
    md = format_comparison(
        proposed_answer="red and blue", proposed_latency=2.0, cache_hit=True,
        r2_tokens=128, vision_encode_ms=37.0,
        vanilla_answer="red and blue too", vanilla_latency=4.0, vanilla_tokens=576)
    assert "red and blue" in md and "red and blue too" in md
    assert "2.00s" in md and "4.00s" in md
    assert "128 tok" in md and "576 tok" in md
    assert "CACHE HIT" in md
    assert "37ms" in md and "saved" in md            # deterministic vision-encode saving
    assert "2.0×" in md                              # speedup 4.0/2.0


def test_format_comparison_miss_shows_building_cache_not_saving():
    md = format_comparison("a", 1.0, False, 128, 37.0, "b", 3.0, 576)
    assert "cache miss" in md
    assert "building" in md.lower()                  # no saving claim on a miss
    assert "saved" not in md
    assert "3.0×" in md                              # 3.0/1.0


def test_format_comparison_zero_latency_no_crash():
    md = format_comparison("a", 0.0, True, 128, 37.0, "b", 1.0, 576)
    assert "0.0×" in md                              # guarded division
