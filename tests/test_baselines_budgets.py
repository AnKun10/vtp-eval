# tests/test_baselines_budgets.py
import pytest
from vtp_eval.baselines import budgets


def test_divprune_ratio_maps_budget_to_fraction_of_576():
    assert budgets.divprune_ratio(64) == pytest.approx(64 / 576, abs=1e-6)
    assert budgets.divprune_ratio(32) == pytest.approx(32 / 576, abs=1e-6)
    assert budgets.divprune_ratio(128) == pytest.approx(128 / 576, abs=1e-6)


def test_visionzip_knobs_sum_to_budget():
    # dominant + contextual must equal the avg-token budget (constant per layer).
    for avg in (32, 64, 128):
        dom, ctx = budgets.visionzip_knobs(avg)
        assert dom + ctx == avg
        assert ctx >= 1 and dom >= 1


def test_fastv_knobs_solve_R_at_K2():
    # avg = (K*576 + (32-K)*R)/32 ; K=2 ; L=32
    assert budgets.fastv_knobs(64) == (2, 30)
    assert budgets.fastv_knobs(128) == (2, 98)
    # the realized average lands within 1 token of the request (R rounded to int)
    for avg in (64, 128):
        k, r = budgets.fastv_knobs(avg)
        realized = (k * 576 + (32 - k) * r) / 32
        assert abs(realized - avg) < 1


def test_fastv_knobs_rejects_out_of_range_k():
    with pytest.raises(ValueError, match="k must be"):
        budgets.fastv_knobs(128, k=32)


def test_fastv_knobs_raise_below_floor():
    # K=2 floor is 2*576/32 = 36 tokens; avg-32 is unreachable.
    with pytest.raises(ValueError, match="floor"):
        budgets.fastv_knobs(32)
