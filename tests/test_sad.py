"""Tests for offline SAD scoring and teacher-forcing selection."""

import torch

from castmind.sad import (
    SADScore,
    TokenDifficulty,
    TrajectoryCandidate,
    compute_sad,
    score_teacher_response,
    select_trajectory,
)


def _token(field, rank=1, log_probability=-1.0):
    return TokenDifficulty(field, 1, log_probability, rank)


def _sad(value):
    return SADScore(value, {}, ())


def test_sad_is_field_balanced_and_clips_ranks():
    score = compute_sad([
        _token("analysis", rank=200, log_probability=-2),
        _token("decision", rank=2, log_probability=-4),
        _token("forecast_method", rank=4, log_probability=-6),
    ], r_max=100)
    assert score.field_means["analysis"] == (100, 2)
    assert score.value == (100 + 2 + 4) / (2 + 4 + 6)


def test_teacher_forcing_scores_only_visible_reasoning_fields():
    class CharTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
            assert enable_thinking
            text = "PROMPT|" + messages[0]["content"] + "|ASSISTANT:"
            if not add_generation_prompt:
                text += messages[-1]["content"]
            return [ord(char) for char in text] if tokenize else text

        def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
            assert not add_special_tokens and return_offsets_mapping
            return {
                "input_ids": [ord(char) for char in text],
                "offset_mapping": [(index, index + 1) for index in range(len(text))],
            }

    class PredictNext(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids):
            batch, length = input_ids.shape
            logits = torch.zeros(batch, length, 128, device=input_ids.device)
            for position in range(length - 1):
                for batch_index in range(batch):
                    logits[batch_index, position, int(input_ids[batch_index, position + 1])] = 4
            return type("Output", (), {"logits": logits})()

    response = (
        "<think>\nanalysis: evidence\ndecision: revise\n"
        "forecast_method: adjust level\n</think>\n<answer>\n```\n"
        "index | value\n0 | 1\n```\n</answer>"
    )
    model = PredictNext()
    score = score_teacher_response(model, CharTokenizer(), [{"role": "user", "content": "forecast"}], response)
    assert score.value > 0
    assert {token.field for token in score.tokens} == {"analysis", "decision", "forecast_method"}
    assert all(token.rank == 1 for token in score.tokens)
    assert model.training is True


def test_selection_uses_sad_only_for_close_same_strategy_candidates():
    candidates = [
        TrajectoryCandidate("accurate", "revise", 1.0, _sad(0.9)),
        TrajectoryCandidate("learnable", "revise", 1.02, _sad(0.4)),
    ]
    assert select_trajectory(candidates).candidate_id == "learnable"
    assert select_trajectory([
        candidates[0], TrajectoryCandidate("different", "retain", 1.01, _sad(0.1)),
    ]).candidate_id == "accurate"
    assert select_trajectory([
        candidates[0], TrajectoryCandidate("too_far", "revise", 1.04, _sad(0.1)),
    ]).candidate_id == "accurate"
