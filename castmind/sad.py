"""Offline student-aware difficulty (SAD) and teacher-forcing utilities.

This module is intentionally separate from the SFT/GRPO launchers.  It scores
already-generated teacher trajectories under a frozen student and can select a
trajectory for offline SFT-data preparation.  It never generates teachers and
never changes the training data loader.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


FIELDS = ("analysis", "decision", "forecast_method")
DEFAULT_R_MAX = 100
DEFAULT_DELTA = 0.03


@dataclass(frozen=True)
class TokenDifficulty:
    """Student-forced statistics for one visible reasoning token."""

    field: str
    token_id: int
    log_probability: float
    rank: int

    @property
    def negative_log_probability(self) -> float:
        return -self.log_probability


@dataclass(frozen=True)
class SADScore:
    """SAD and per-field token statistics for one teacher trajectory."""

    value: float
    field_means: Mapping[str, tuple[float, float]]
    tokens: tuple[TokenDifficulty, ...]


@dataclass(frozen=True)
class TrajectoryCandidate:
    """A teacher candidate after forecast validation and SAD scoring."""

    candidate_id: str
    strategy: str
    mae: float
    sad: SADScore


def compute_sad(
    tokens: Sequence[TokenDifficulty], *, r_max: int = DEFAULT_R_MAX,
) -> SADScore:
    """Compute the paper's field-balanced SAD from forced token statistics.

    For each field, ranks are clipped at ``r_max`` and both rank and negative
    log-probability are averaged over that field.  The final score is the sum of
    field mean ranks divided by the sum of field mean negative log-probabilities.
    Forecast tokens are excluded by the caller; ``score_teacher_response`` does
    this automatically.
    """
    if isinstance(r_max, bool) or not isinstance(r_max, int) or r_max <= 0:
        raise ValueError("r_max must be a positive integer")
    if not tokens:
        raise ValueError("SAD requires at least one visible reasoning token")
    grouped: dict[str, list[TokenDifficulty]] = {field: [] for field in FIELDS}
    for token in tokens:
        if token.field not in grouped:
            raise ValueError(f"unknown SAD field: {token.field}")
        if token.rank < 1 or not math.isfinite(token.log_probability):
            raise ValueError("token rank and log probability must be finite and valid")
        grouped[token.field].append(token)

    field_means: dict[str, tuple[float, float]] = {}
    rank_total = 0.0
    nll_total = 0.0
    for field in FIELDS:
        values = grouped[field]
        if not values:
            raise ValueError(f"SAD field {field!r} has no visible tokens")
        mean_rank = sum(min(token.rank, r_max) for token in values) / len(values)
        mean_nll = sum(-token.log_probability for token in values) / len(values)
        if not math.isfinite(mean_rank) or not math.isfinite(mean_nll) or mean_nll <= 0:
            raise ValueError("SAD means must be finite and have positive NLL denominator")
        field_means[field] = (mean_rank, mean_nll)
        rank_total += mean_rank
        nll_total += mean_nll
    if not math.isfinite(rank_total / nll_total):
        raise ValueError("SAD is non-finite")
    return SADScore(rank_total / nll_total, field_means, tuple(tokens))


def score_teacher_response(
    model: Any,
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    response: str,
    *,
    r_max: int = DEFAULT_R_MAX,
) -> SADScore:
    """Teacher-force ``response`` through a frozen student and compute SAD.

    ``messages`` must contain the prompt conversation without the assistant
    answer.  The tokenizer's chat template is used twice: once for the exact
    generation prefix and once with the teacher response appended.  The model
    then receives the complete sequence in one forward pass; each response
    token's logits are evaluated against the preceding gold context token.  No
    sampling or model-generated token is used.

    A fast tokenizer with ``return_offsets_mapping`` is required so visible
    reasoning tokens can be assigned to the three paper fields.  The response
    must follow CastMind's canonical ``<think>`` contract.
    """
    if not isinstance(response, str) or not response.strip():
        raise ValueError("teacher response must be a nonempty string")
    if not messages or any(not isinstance(message, Mapping) for message in messages):
        raise ValueError("messages must be a nonempty prompt conversation")
    if messages[-1].get("role") == "assistant":
        raise ValueError("messages must omit the assistant teacher response")

    prompt_ids = _ids(tokenizer.apply_chat_template(
        list(messages), tokenize=True, add_generation_prompt=True, enable_thinking=True,
    ))
    full_ids = _ids(tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": response}],
        tokenize=True, add_generation_prompt=False, enable_thinking=True,
    ))
    if not prompt_ids or full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("teacher response tokens do not share the tokenizer prompt prefix")
    response_ids = full_ids[len(prompt_ids):]
    encoded = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    local_ids = _ids(encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded.input_ids)
    offsets = encoded.get("offset_mapping") if isinstance(encoded, Mapping) else encoded.offset_mapping
    offsets = _plain_offsets(offsets)
    if local_ids != response_ids or len(offsets) != len(response_ids):
        raise ValueError("response tokenization differs from the chat-template suffix")
    spans = _reasoning_spans(response)
    token_fields = []
    for start, end in offsets:
        field = next((name for name, (lo, hi) in spans.items() if start < hi and end > lo), None)
        token_fields.append(field)

    torch = _torch()
    input_ids = torch.tensor([full_ids], dtype=torch.long)
    device = _model_device(model)
    input_ids = input_ids.to(device)
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.inference_mode():
            output = model(input_ids=input_ids)
    finally:
        model.train(was_training)
    logits = output.logits if hasattr(output, "logits") else output[0]
    if logits.ndim != 3 or logits.shape[1] != len(full_ids):
        raise ValueError("student model returned logits with an incompatible shape")

    forced = []
    for local_index, (token_id, field) in enumerate(zip(response_ids, token_fields, strict=True)):
        if field is None:
            continue
        full_index = len(prompt_ids) + local_index
        if full_index == 0:
            raise ValueError("first forced token has no preceding context")
        distribution = logits[0, full_index - 1]
        token_log_probability = float(torch.log_softmax(distribution.float(), dim=-1)[token_id].item())
        rank = int((distribution > distribution[token_id]).sum().item()) + 1
        forced.append(TokenDifficulty(field, int(token_id), token_log_probability, rank))
    return compute_sad(forced, r_max=r_max)


def select_trajectory(
    candidates: Sequence[TrajectoryCandidate], *, delta: float = DEFAULT_DELTA,
) -> TrajectoryCandidate:
    """Select a teacher trajectory using MAE first and SAD only as a gated tie-break.

    The best-MAE candidate is always retained unless the second-best candidate is
    within ``delta`` relative MAE, has the same strategy, and has lower SAD.
    """
    if not candidates:
        raise ValueError("at least one valid trajectory candidate is required")
    if not math.isfinite(delta) or delta < 0:
        raise ValueError("delta must be finite and nonnegative")
    for candidate in candidates:
        if not math.isfinite(candidate.mae) or candidate.mae < 0:
            raise ValueError("candidate MAE must be finite and nonnegative")
        if not math.isfinite(candidate.sad.value):
            raise ValueError("candidate SAD must be finite")
    ranked = sorted(candidates, key=lambda item: (item.mae, item.candidate_id))
    best = ranked[0]
    if len(ranked) == 1:
        return best
    second = ranked[1]
    close = second.mae <= (1.0 + delta) * best.mae
    if close and second.strategy == best.strategy and second.sad.value < best.sad.value:
        return second
    return best


def _reasoning_spans(response: str) -> dict[str, tuple[int, int]]:
    """Return character spans for visible analysis, decision and method text."""
    think_start = response.find("<think>")
    think_end = response.find("</think>", think_start + len("<think>"))
    if think_start < 0 or think_end < 0:
        raise ValueError("teacher response must contain a complete <think> block")
    body_start = think_start + len("<think>")
    body = response[body_start:think_end]
    match = re.fullmatch(
        r"\s*analysis:\s*(.*?)\s*decision:\s*(.*?)\s*forecast_method:\s*(.*?)\s*",
        body,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("teacher response must contain the three canonical reasoning fields")
    spans = {}
    for field, group in zip(FIELDS, match.groups(), strict=True):
        start, end = match.span(FIELDS.index(field) + 1)
        # Match offsets are relative to the body; trim whitespace from visible text.
        text = body[start:end]
        left = len(text) - len(text.lstrip())
        right = len(text.rstrip())
        spans[field] = (body_start + start + left, body_start + start + right)
        if spans[field][0] >= spans[field][1]:
            raise ValueError(f"teacher field {field!r} is empty")
    return spans


def _ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Sequence) and value and isinstance(value[0], Sequence):
        if len(value) != 1:
            raise ValueError("tokenizer returned multiple sequences")
        value = value[0]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("tokenizer did not return token IDs")
    return [int(item) for item in value]


def _plain_offsets(value: Any) -> list[tuple[int, int]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence):
        raise ValueError("fast tokenizer offsets are required")
    result = []
    for pair in value:
        if hasattr(pair, "tolist"):
            pair = pair.tolist()
        if not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError("invalid tokenizer offset mapping")
        result.append((int(pair[0]), int(pair[1])))
    return result


def _model_device(model: Any):
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return _torch().device("cpu")


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised only without train extras
        raise RuntimeError("teacher forcing requires the training dependency torch") from error
    return torch
