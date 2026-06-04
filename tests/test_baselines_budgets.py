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


