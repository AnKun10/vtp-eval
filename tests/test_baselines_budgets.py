# tests/test_baselines_budgets.py
from vtp_eval.baselines import budgets


def test_visionzip_knobs_sum_to_budget():
    # dominant + contextual must equal the retain-token budget.
    for retain in (32, 64, 128):
        dom, ctx = budgets.visionzip_knobs(retain)
        assert dom + ctx == retain
        assert ctx >= 1 and dom >= 1
