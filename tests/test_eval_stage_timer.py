# tests/test_eval_stage_timer.py
from vtp_eval.eval import stage_timer as st


def _rec(enc, pre, dec, steps, total, bs=1, mem=100.0):
    return {"encoder_ms": enc, "prefill_ms": pre, "decode_ms": dec,
            "decode_steps": steps, "total_ms": total, "batch_size": bs,
            "peak_mem_mb": mem}


def test_summarize_drops_warmup_and_averages_per_sample():
    recs = [_rec(99, 99, 99, 9, 999),          # warm-up (dropped)
            _rec(10, 20, 30, 3, 60),
            _rec(12, 22, 34, 4, 68)]
    s = st.summarize_stage_records(recs, drop_warmup=True)
    assert s["encoder_ms"] == 11.0          # mean(10,12)
    assert s["prefill_ms"] == 21.0          # mean(20,22)
    assert s["decode_ms"] == 32.0           # mean(30,34)
    assert s["total_latency_ms"] == 64.0    # mean(60,68)
    assert s["decode_tokens"] == 3.5        # mean(3,4)
    assert s["n_samples"] == 2


def test_summarize_divides_by_batch_size():
    recs = [_rec(10, 20, 30, 3, 60), _rec(20, 40, 60, 6, 120, bs=2)]
    s = st.summarize_stage_records(recs, drop_warmup=False)
    # record0: 10/1=10 ; record1: 20/2=10 -> mean 10
    assert s["encoder_ms"] == 10.0
    assert s["n_samples"] == 3              # 1 + 2
    assert s["decode_tokens"] == 3.0        # per-sample: 3/1=3 ; 6/2=3 -> mean 3


def test_summarize_empty_is_zeros():
    s = st.summarize_stage_records([], drop_warmup=True)
    assert s["n_samples"] == 0 and s["encoder_ms"] == 0.0


def test_summarize_single_record_not_dropped():
    s = st.summarize_stage_records([_rec(10, 20, 30, 3, 60)], drop_warmup=True)
    assert s["n_samples"] == 1 and s["prefill_ms"] == 20.0


def test_llm_stage_none_is_prefill():
    assert st.llm_stage(None) == "prefill"


def test_llm_stage_empty_legacy_tuple_is_prefill():
    assert st.llm_stage(()) == "prefill"


def test_llm_stage_nonempty_legacy_tuple_is_decode():
    # legacy cache entry: (key, value) with key.shape == [B, H, S, D]
    class _T:
        def __init__(self, s):
            self.shape = (1, 32, s, 128)
    pkv = (( _T(56), _T(56) ),)          # one layer, cached length 56
    assert st.llm_stage(pkv) == "decode"


def test_llm_stage_cache_object_uses_seq_length():
    class _Cache:
        def __init__(self, n):
            self._n = n
        def get_seq_length(self):
            return self._n
    assert st.llm_stage(_Cache(0)) == "prefill"
    assert st.llm_stage(_Cache(83)) == "decode"


def test_stage_timer_cpu_bookkeeping():
    # On CPU (no CUDA), region timing uses perf_counter; verify the bucket
    # bookkeeping (decode_steps count, all stages present, non-negative).
    t = st.StageTimer(force_cpu=True)
    t.start_batch(batch_size=1)
    with t.region("encoder"):
        pass
    with t.region("prefill"):
        pass
    for _ in range(3):
        with t.region("decode"):
            pass
    t.end_batch()
    assert len(t.records) == 1
    r = t.records[0]
    assert r["decode_steps"] == 3
    assert r["batch_size"] == 1
    assert r["encoder_ms"] >= 0 and r["prefill_ms"] >= 0 and r["decode_ms"] >= 0
