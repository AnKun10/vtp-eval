# tests/test_tva_tokens.py
from vtp_eval.demo.tva import to_merged_index


def test_to_merged_index_before_at_and_after_placeholder():
    # vstart = pre-merge index of the single image placeholder.
    assert to_merged_index(1, vstart=5, n_vis=576) == 1        # before: unchanged
    assert to_merged_index(5, vstart=5, n_vis=576) == 5        # the placeholder itself
    assert to_merged_index(6, vstart=5, n_vis=576) == 6 + 575  # after: + (n_vis-1)
    assert to_merged_index(10, vstart=5, n_vis=4) == 10 + 3    # n_vis-1 = 3
