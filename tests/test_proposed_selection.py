# tests/test_proposed_selection.py
import torch
import torch.nn.functional as F
from vtp_eval.proposed_method import selection


def test_cls_topk_dominant_picks_highest():
    # [B=1, P=5] attention sums; top-3 should be indices {4,3,2}
    s = torch.tensor([[0.1, 0.2, 0.5, 0.9, 1.0]])
    idx = selection.cls_topk_dominant(s, dominant_k=3)
    assert set(idx[0].tolist()) == {2, 3, 4}


def test_diversity_excludes_seed_indices():
    torch.manual_seed(0)
    feats = F.normalize(torch.randn(1, 10, 8), dim=-1)
    seed = torch.tensor([[0, 1]])
    out = selection.farthest_point_diversity(feats, seed, m=3)
    assert out.shape == (1, 3)
    assert set(out[0].tolist()).isdisjoint({0, 1})            # never re-pick seeds
    assert len(set(out[0].tolist())) == 3                      # all distinct


def test_diversity_prefers_dissimilar_token():
    # 3 tokens: two near-identical (0,1), one orthogonal (2). Seed={0}.
    # The most-dissimilar pick must be token 2.
    feats = torch.tensor([[[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]])
    feats = F.normalize(feats, dim=-1)
    seed = torch.tensor([[0]])
    out = selection.farthest_point_diversity(feats, seed, m=1)
    assert out[0, 0].item() == 2


def test_select_stage1_shapes_and_sorted():
    B, H, P, D = 2, 4, 20, 16
    attn = torch.rand(B, H, 1 + P, 1 + P)
    hidden = torch.randn(B, 1 + P, D)
    keep = selection.select_stage1(attn, hidden, dominant_k=5, diversity_m=3)
    assert keep.shape == (B, 8)                       # R1 = 5 + 3
    assert (keep[:, 1:] >= keep[:, :-1]).all()        # sorted ascending
    assert keep.max().item() < P and keep.min().item() >= 0
    for b in range(B):
        assert len(set(keep[b].tolist())) == 8        # no duplicates


def test_select_stage1_diversity_zero_returns_dominant_only():
    B, H, P, D = 1, 2, 12, 8
    attn = torch.rand(B, H, 1 + P, 1 + P)
    hidden = torch.randn(B, 1 + P, D)
    keep = selection.select_stage1(attn, hidden, dominant_k=4, diversity_m=0)
    assert keep.shape == (1, 4)


def test_text_to_vision_scores_averages_instruction_rows():
    # seq layout: [sys=2][vision=3][instr=2], B=1, H=2
    B, H, S = 1, 2, 7
    attn = torch.zeros(B, H, S, S)
    # instruction tokens are rows 5,6; vision cols are 2,3,4.
    # Make vision col 3 receive the most instruction attention.
    attn[:, :, 5, 2] = 0.1; attn[:, :, 5, 3] = 0.9; attn[:, :, 5, 4] = 0.2
    attn[:, :, 6, 2] = 0.2; attn[:, :, 6, 3] = 0.8; attn[:, :, 6, 4] = 0.1
    scores = selection.text_to_vision_scores(attn, vision_slice=(2, 5), instr_slice=(5, 7))
    assert scores.shape == (1, 3)
    assert scores.argmax(dim=1).item() == 1            # local index 1 == col 3


def test_build_keep_index_keeps_all_nonvision_plus_topr2():
    # seq=7, vision block [2,5) of length 3, keep r2=1 best vision token.
    scores = torch.tensor([[0.1, 0.9, 0.2]])           # best local idx 1 -> abs 3
    vision_start = torch.tensor([2])
    keep = selection.build_keep_index(scores, r2=1, vision_start=vision_start, seq_len=7)
    # non-vision = {0,1,5,6}, plus vision {3} -> sorted
    assert keep[0].tolist() == [0, 1, 3, 5, 6]


def test_slice_past_key_values_legacy_tuple():
    # legacy cache: tuple per layer of (key, value) each [B, n_heads, S, head_dim]
    B, nh, S, hd = 1, 2, 7, 4
    layer = (torch.randn(B, nh, S, hd), torch.randn(B, nh, S, hd))
    past = (layer, layer)
    keep = torch.tensor([[0, 1, 3, 5, 6]])
    sliced = selection.slice_past_key_values(past, keep)
    assert sliced[0][0].shape == (B, nh, 5, hd)
    assert torch.equal(sliced[0][0][0, :, 2, :], layer[0][0, :, 3, :])  # index 3 -> pos 2


class _FakeOutputs:
    def __init__(self, attentions, hidden_states):
        self.attentions = attentions
        self.hidden_states = hidden_states


def test_stage1_forward_returns_R1_selected_patches():
    from vtp_eval.proposed_method import stage1_vision
    from vtp_eval.proposed_method.config import ProposedConfig

    B, H, P, D = 1, 2, 20, 16
    # need >=2 hidden layers so [-2] is valid
    attn = torch.rand(B, H, 1 + P, 1 + P)
    hidden = torch.randn(B, 1 + P, D)
    outs = _FakeOutputs(attentions=[attn, attn], hidden_states=[hidden, hidden])

    cfg = ProposedConfig(dominant_k=5, diversity_m=3, pruned_layer=12, llm_keep_r2=4)
    feats = stage1_vision.select_features(outs, cfg)
    assert feats.shape == (B, 8, D)                    # R1 = 8 patch features

    # the returned features must be exactly rows of the penultimate patch hidden states
    keep = selection.select_stage1(attn, hidden, 5, 3)
    expected = hidden[:, 1:, :].gather(1, keep[:, :, None].expand(B, 8, D))
    assert torch.allclose(feats, expected)


def test_make_forward_batched_and_list_branches():
    from vtp_eval.proposed_method import stage1_vision
    from vtp_eval.proposed_method.config import ProposedConfig

    H, P, D = 2, 20, 16

    class _StubTower:
        device = "cpu"
        dtype = torch.float32

        def vision_tower(self, images, output_hidden_states, output_attentions):
            n = images.shape[0]
            attn = torch.rand(n, H, 1 + P, 1 + P)
            hidden = torch.randn(n, 1 + P, D)
            return _FakeOutputs(attentions=[attn, attn], hidden_states=[hidden, hidden])

    cfg = ProposedConfig(dominant_k=5, diversity_m=3, pruned_layer=12, llm_keep_r2=4)
    fwd = stage1_vision.make_forward(cfg)
    stub = _StubTower()

    # batched-tensor branch -> [B, R1, D]
    out = fwd(stub, torch.randn(2, 3, 4, 4))
    assert out.shape == (2, 8, D)

    # list branch -> list of per-image [1, R1, D]
    out_list = fwd(stub, [torch.randn(3, 4, 4), torch.randn(3, 4, 4)])
    assert isinstance(out_list, list) and len(out_list) == 2
    assert out_list[0].shape == (1, 8, D)
