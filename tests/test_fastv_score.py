# tests/test_fastv_score.py
import torch
from vtp_eval.baselines.fastv.score import last_token_to_vision_scores


def test_uses_only_last_row_over_vision_slice():
    # attn: [B=1, H=1, S=4, S=4]; vision tokens at columns 1..3.
    attn = torch.zeros(1, 1, 4, 4)
    attn[0, 0, 3, 1] = 0.2   # last token -> vision[0]
    attn[0, 0, 3, 2] = 0.9   # last token -> vision[1]  (highest)
    attn[0, 0, 3, 3] = 0.1   # last token -> vision[2]
    attn[0, 0, 0, 2] = 5.0   # earlier row must be IGNORED
    scores = last_token_to_vision_scores(attn, (1, 4), (0, 0))
    assert scores.shape == (1, 3)
    assert torch.argmax(scores[0]).item() == 1
