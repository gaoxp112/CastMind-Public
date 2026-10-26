"""The five training rewards and fixed parameters reported in the paper.

The callback consumes prepared, private training metadata. It does not construct
baselines, annotate intervals, or generate teacher demonstrations. All forecast
errors use the future target only inside this offline reward computation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from numbers import Integral, Real
from typing import Any

import numpy as np

from castmind.contract import parse_response


FORMAT_PENALTY = 1.0
SCALE_RHO = 0.05
SCALE_EPS = 1e-8
NMAE_WEIGHT = 0.9
METRIC_CAP = 4.0
UPLIFT_WEIGHT = 0.10
UPLIFT_RELATIVE_THRESHOLD = 0.05
UPLIFT_RELATIVE_RANGE = 0.20
UPLIFT_ABSOLUTE_THRESHOLD = 0.02
UPLIFT_ABSOLUTE_RANGE = 0.08
UPLIFT_DENOMINATOR_FLOOR = 0.10
INTERVAL_WEIGHT = 0.20
INTERVAL_CAP = 1.0
OUTSIDE_WEIGHT = 0.10
OUTSIDE_CAP = 1.0
REASONING_WEIGHT = 0.02
REASONING_THRESHOLD = 0.70
ENCODER_NAME = "BAAI/bge-m3"
BASELINE_NAMES = frozenset({"last_value", "recent_mean", "seasonal_copy", "ridge"})
COMPONENT_NAMES = ("format", "accuracy", "uplift", "interval", "reasoning")
DIAGNOSTIC_NAMES = (
    "nmae", "nmse", "raw_loss", "prior_raw_loss", "baseline_loss",
    "target_scale", "teacher_similarity", "qualified_teachers",
)


@dataclass(frozen=True)
class _Metadata:
    prior: np.ndarray
    baselines: tuple[np.ndarray, ...]
    intervals: tuple[tuple[int, int], ...]
    teacher_thinks: tuple[str, ...]


def compute_score(
    data_source: Any,
    solution_str: str,
    ground_truth: str | list[float],
    extra_info: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """VeRL custom reward callback; return a scalar score and diagnostics.

    ``ground_truth`` is a numeric list or its JSON encoding. ``extra_info`` must
    contain the exact displayed ``prior`` and a ``baselines`` mapping with the
    four named, precomputed full-horizon forecasts. Optional ``intervals`` are
    zero-based half-open pairs. Optional ``teachers`` contain ``response`` and
    ``forecast``; the response uses the same output contract as the student.

    Invalid model output receives -1 and no other reward. Invalid training
    metadata raises ValueError, including when the model output is invalid.
    """
    del data_source
    target = _numeric_vector(_decode_target(ground_truth), "ground_truth")
    horizon = len(target)
    metadata = _validate_metadata(extra_info, target)
    try:
        response = parse_response(_response_prefix(extra_info) + solution_str, horizon, extra_info["prior"])
    except ValueError:
        return _invalid_result()

    prediction = np.asarray(response.values, dtype=np.longdouble)
    scale = max(
        np.std(target, ddof=0),
        np.longdouble(SCALE_RHO) * np.mean(np.abs(target)),
        np.longdouble(SCALE_EPS),
    )
    nmae, nmse, raw_loss, clipped_loss = _loss(prediction, target, scale)
    _, _, prior_loss, _ = _loss(metadata.prior, target, scale)
    baseline_loss = min(_loss(baseline, target, scale)[3] for baseline in metadata.baselines)
    improvement = baseline_loss - clipped_loss
    relative_gain = improvement / max(baseline_loss, UPLIFT_DENOMINATOR_FLOOR)
    uplift = UPLIFT_WEIGHT * min(
        _clip((relative_gain - UPLIFT_RELATIVE_THRESHOLD) / UPLIFT_RELATIVE_RANGE, 0, 1),
        _clip((improvement - UPLIFT_ABSOLUTE_THRESHOLD) / UPLIFT_ABSOLUTE_RANGE, 0, 1),
    )
    interval = _interval_reward(prediction, target, metadata, scale)

    similarity = 0.0
    reasoning = 0.0
    if metadata.teacher_thinks and raw_loss <= prior_loss:
        similarity = _maximum_cosine_similarity(response.think, metadata.teacher_thinks)
        reasoning = REASONING_WEIGHT * _clip(
            (similarity - REASONING_THRESHOLD) / (1.0 - REASONING_THRESHOLD), 0, 1,
        )

    result = {
        "format": 0.0,
        "accuracy": -float(clipped_loss),
        "uplift": float(uplift),
        "interval": float(interval),
        "reasoning": float(reasoning),
        "nmae": _diagnostic_float(nmae),
        "nmse": _diagnostic_float(nmse),
        "raw_loss": _diagnostic_float(raw_loss),
        "prior_raw_loss": _diagnostic_float(prior_loss),
        "baseline_loss": float(baseline_loss),
        "target_scale": _diagnostic_float(scale),
        "teacher_similarity": float(similarity),
        "qualified_teachers": float(len(metadata.teacher_thinks)),
    }
    return {"score": sum(result[name] for name in COMPONENT_NAMES), **result}


def validate_metadata(ground_truth, extra_info) -> bool:
    """Validate prepared reward inputs without scoring or loading BGE-M3.

    Return whether any valid teacher strictly improves the prior in MAE.
    """
    target = _numeric_vector(_decode_target(ground_truth), "ground_truth")
    return bool(_validate_metadata(extra_info, target).teacher_thinks)


def _response_prefix(extra_info) -> str:
    prefix = extra_info.get("response_prefix", "")
    if prefix not in ("", "<think>\n"):
        raise ValueError("response_prefix must be empty or the exact prompt-owned <think> opening")
    return prefix


def _decode_target(ground_truth: Any) -> Any:
    if isinstance(ground_truth, str):
        try:
            return json.loads(ground_truth)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError("ground_truth must be a numeric list or its JSON encoding") from error
    return ground_truth


def _numeric_vector(value: Any, name: str, horizon: int | None = None) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty numeric list")
    if any(isinstance(item, (bool, np.bool_)) or not isinstance(item, Real) for item in value):
        raise ValueError(f"{name} must contain finite numbers")
    try:
        vector = np.asarray(value, dtype=np.longdouble)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain finite numbers") from error
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite numbers")
    if horizon is not None and len(vector) != horizon:
        raise ValueError(f"{name} must cover the complete forecast horizon")
    return vector


def _validate_metadata(extra_info: Any, target: np.ndarray) -> _Metadata:
    if not isinstance(extra_info, Mapping):
        raise ValueError("extra_info must contain prior and baselines")
    _response_prefix(extra_info)
    horizon = len(target)
    prior = _numeric_vector(extra_info.get("prior"), "prior", horizon)
    baselines = extra_info.get("baselines")
    if not isinstance(baselines, Mapping) or set(baselines) != BASELINE_NAMES:
        raise ValueError("baselines must contain last_value, recent_mean, seasonal_copy, and ridge")
    baseline_vectors = tuple(
        _numeric_vector(baselines[name], f"baselines.{name}", horizon)
        for name in sorted(BASELINE_NAMES)
    )
    intervals = _validate_intervals(extra_info.get("intervals", []), horizon)
    teachers = extra_info.get("teachers", [])
    if not isinstance(teachers, (list, tuple)):
        raise ValueError("teachers must be a list of response/forecast mappings")
    prior_mae = np.mean(np.abs(prior - target))
    teacher_thinks = []
    for index, teacher in enumerate(teachers):
        if not isinstance(teacher, Mapping) or not isinstance(teacher.get("response"), str):
            raise ValueError(f"teachers[{index}] must contain a response string and forecast")
        forecast = _numeric_vector(teacher.get("forecast"), f"teachers[{index}].forecast", horizon)
        try:
            parsed = parse_response(teacher["response"], horizon, extra_info["prior"])
        except ValueError:
            # An invalid teacher CoT is not eligible for a reasoning reward.
            continue
        if not np.array_equal(np.asarray(parsed.values, dtype=np.longdouble), forecast):
            raise ValueError(f"teachers[{index}].forecast differs from its response")
        if np.mean(np.abs(forecast - target)) < prior_mae:
            teacher_thinks.append(parsed.think)
    return _Metadata(prior, baseline_vectors, intervals, tuple(teacher_thinks))


def _validate_intervals(value: Any, horizon: int) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("intervals must be a list of [start, end) pairs")
    intervals = []
    for interval in value:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError("intervals must contain [start, end) pairs")
        start, end = interval
        if any(isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral) for item in interval):
            raise ValueError("interval boundaries must be integers")
        if not 0 <= start < end <= horizon:
            raise ValueError("intervals must satisfy 0 <= start < end <= horizon")
        intervals.append((int(start), int(end)))
    return tuple(intervals)


def _loss(prediction: np.ndarray, target: np.ndarray, scale: np.longdouble):
    normalized_error = (prediction - target) / scale
    nmae = np.mean(np.abs(normalized_error))
    nmse = np.mean(np.square(normalized_error))
    raw = NMAE_WEIGHT * nmae + (1.0 - NMAE_WEIGHT) * nmse
    clipped = NMAE_WEIGHT * _clip(nmae, 0, METRIC_CAP) + (1.0 - NMAE_WEIGHT) * _clip(nmse, 0, METRIC_CAP)
    return nmae, nmse, raw, clipped


def _interval_reward(
    prediction: np.ndarray, target: np.ndarray, metadata: _Metadata, scale: np.longdouble,
) -> float:
    if not metadata.intervals:
        return 0.0
    total_points = sum(end - start for start, end in metadata.intervals)
    outside = np.ones(len(target), dtype=bool)
    local_gain = np.longdouble(0)
    for start, end in metadata.intervals:
        outside[start:end] = False
        prior_loss = _loss(metadata.prior[start:end], target[start:end], scale)[2]
        prediction_loss = _loss(prediction[start:end], target[start:end], scale)[2]
        local_gain += (end - start) / total_points * _clip(
            prior_loss - prediction_loss, -INTERVAL_CAP, INTERVAL_CAP,
        )
    outside_degradation = 0.0
    if outside.any():
        prior_loss = _loss(metadata.prior[outside], target[outside], scale)[2]
        prediction_loss = _loss(prediction[outside], target[outside], scale)[2]
        outside_degradation = _clip(prediction_loss - prior_loss, 0, OUTSIDE_CAP)
    return float(INTERVAL_WEIGHT * local_gain - OUTSIDE_WEIGHT * outside_degradation)


@lru_cache(maxsize=1)
def _get_encoder():
    """Load the frozen encoder from the local model cache only when needed."""
    import os
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(
        os.environ.get("CASTMIND_ENCODER_PATH", ENCODER_NAME),
        device="cpu", local_files_only=True,
    )
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def _maximum_cosine_similarity(student_think: str, teacher_thinks: Sequence[str]) -> float:
    encoder = _get_encoder()
    vectors = np.asarray(encoder.encode(
        [student_think, *teacher_thinks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        device="cpu",
    ), dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] != len(teacher_thinks) + 1:
        raise RuntimeError("BGE-M3 returned an invalid embedding shape")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.all(np.isfinite(vectors)) or not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise RuntimeError("BGE-M3 returned invalid embeddings")
    vectors = vectors / norms[:, None]
    return float(_clip(np.max(vectors[1:] @ vectors[0]), -1, 1))


def _clip(value, lower, upper):
    return min(upper, max(lower, value))


def _diagnostic_float(value) -> float:
    # Extended precision keeps valid, extreme finite predictions from making
    # intermediate squared errors overflow. Keep VeRL diagnostics finite too.
    limit = np.finfo(float).max
    return float(_clip(value, -limit, limit))


def _invalid_result() -> dict[str, float]:
    # Native VeRL batches diagnostics by the first response's keys. Keep the
    # schema identical for malformed and valid responses; diagnostic zeros are
    # sentinels and have meaning only when format == 0.
    return {
        "score": -FORMAT_PENALTY,
        "format": -FORMAT_PENALTY,
        **{name: 0.0 for name in COMPONENT_NAMES if name != "format"},
        **{name: 0.0 for name in DIAGNOSTIC_NAMES},
    }
