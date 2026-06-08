# tests/test_demo_benchmarks.py
from PIL import Image
from vtp_eval.demo.benchmarks import group_rows


def _img(color):
    return Image.new("RGB", (8, 8), color)


def test_group_rows_groups_same_image_collects_all_questions():
    a1, a2 = _img((10, 20, 30)), _img((10, 20, 30))   # identical content -> same hash
    b = _img((200, 0, 0))
    rows = [{"image": a1, "question": "q1"},
            {"image": b,  "question": "q3"},
            {"image": a2, "question": "q2"}]
    groups = group_rows(rows, "image", "question", n_images=5, max_scan=10)
    assert len(groups) == 2                       # A (first seen) then B
    assert groups[0]["questions"] == ["q1", "q2"]  # both A questions, in order
    assert groups[1]["questions"] == ["q3"]
    assert groups[0]["image"].size == (8, 8)


def test_group_rows_n_images_caps_output_first_seen():
    rows = [{"image": _img((i, i, i)), "question": f"q{i}"} for i in range(5)]
    groups = group_rows(rows, "image", "question", n_images=2, max_scan=10)
    assert len(groups) == 2
    assert groups[0]["questions"] == ["q0"]
    assert groups[1]["questions"] == ["q1"]


def test_group_rows_max_scan_caps_rows_scanned():
    rows = [{"image": _img((i, i, i)), "question": f"q{i}"} for i in range(5)]
    groups = group_rows(rows, "image", "question", n_images=10, max_scan=2)
    assert len(groups) == 2                        # only first 2 rows were scanned


def test_group_rows_skips_empty_questions():
    a1, a2 = _img((5, 5, 5)), _img((5, 5, 5))
    rows = [{"image": a1, "question": ""}, {"image": a2, "question": "q2"}]
    groups = group_rows(rows, "image", "question", n_images=5, max_scan=10)
    assert len(groups) == 1
    assert groups[0]["questions"] == ["q2"]        # empty string not appended
