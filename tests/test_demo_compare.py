# tests/test_demo_compare.py
from vtp_eval.demo.compare import format_comparison


def test_format_comparison_contains_answers_latencies_and_speedup():
    md = format_comparison(
        proposed_answer="red and blue", proposed_latency=2.0, cache_hit=True,
        r2_tokens=128, no_cache_latency=3.1,
        vanilla_answer="red and blue too", vanilla_latency=4.0, vanilla_tokens=576)
    assert "red and blue" in md and "red and blue too" in md
    assert "2.00s" in md and "3.10s" in md and "4.00s" in md   # all three latencies
    assert "128 tok" in md and "576 tok" in md
    assert "CACHE HIT" in md
    assert "2.0" in md                                          # speedup 4.0/2.0 = 2.0x


def test_format_comparison_cache_miss_label_and_speedup_value():
    md = format_comparison("a", 1.0, False, 128, 1.5, "b", 3.0, 576)
    assert "cache miss" in md
    assert "3.0×" in md                                         # 3.0/1.0


def test_format_comparison_zero_latency_no_crash():
    md = format_comparison("a", 0.0, True, 128, 0.0, "b", 1.0, 576)
    assert "0.0×" in md                                         # guarded division
