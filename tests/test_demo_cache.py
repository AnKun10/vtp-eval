# tests/test_demo_cache.py
import torch
from vtp_eval.demo.cache import image_hash


def test_image_hash_stable_for_same_content():
    t = torch.arange(12, dtype=torch.float16).reshape(1, 3, 2, 2)
    assert image_hash(t) == image_hash(t.clone())


def test_image_hash_differs_for_different_content():
    a = torch.zeros(1, 3, 2, 2, dtype=torch.float16)
    b = a.clone(); b[0, 0, 0, 0] = 1.0
    assert image_hash(a) != image_hash(b)


def test_image_hash_dtype_and_device_independent():
    a = torch.ones(1, 3, 2, 2, dtype=torch.float16)
    b = torch.ones(1, 3, 2, 2, dtype=torch.float32)
    assert image_hash(a) == image_hash(b)   # hashed as float32 bytes
