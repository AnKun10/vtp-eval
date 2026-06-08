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


from vtp_eval.demo.cache import RetainTokenCache


def test_cache_put_get_roundtrip():
    c = RetainTokenCache(maxsize=2)
    feats = torch.randn(1, 384, 8)
    c.put("k1", feats)
    got = c.get("k1")
    assert got is not None and torch.equal(got["feats"], feats)
    assert got["r1_idx"] is None


def test_cache_attach_r1():
    c = RetainTokenCache(maxsize=2)
    c.put("k1", torch.randn(1, 384, 8))
    c.attach_r1("k1", torch.arange(384))
    assert torch.equal(c.get("k1")["r1_idx"], torch.arange(384))


def test_cache_lru_evicts_oldest():
    c = RetainTokenCache(maxsize=2)
    c.put("k1", torch.zeros(1)); c.put("k2", torch.zeros(1))
    c.get("k1")                      # touch k1 so k2 is now oldest
    c.put("k3", torch.zeros(1))      # evicts k2
    assert c.get("k1") is not None
    assert c.get("k2") is None
    assert c.get("k3") is not None


def test_cache_miss_returns_none():
    assert RetainTokenCache().get("nope") is None
