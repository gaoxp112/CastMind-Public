"""Numerical and contract regression tests for the paper's training rewards."""

import copy
import json
import math
import sys
import types

import numpy as np
import pytest

from castmind import rewards


def response(values, decision="revise", analysis="Future calendar evidence supports this forecast.", method="Correct the supported level change."):
    if decision == "retain":
        method = "none"
    return (
        f"<think>\nanalysis: {analysis}\ndecision: {decision}\nforecast_method: {method}\n</think>\n"
        "<answer>\n```\nindex | value\n"
        + "\n".join(f"{index} | {value}" for index, value in enumerate(values))
        + "\n```\n</answer>"
    )


def metadata(prior, baseline=None, intervals=None, teachers=None):
    baseline = prior if baseline is None else baseline
    return {
        "prior": list(prior),
        "baselines": {name: list(baseline) for name in rewards.BASELINE_NAMES},
        "intervals": [] if intervals is None else intervals,
        "teachers": [] if teachers is None else teachers,
    }


def score(values, target, info, **kwargs):
    return rewards.compute_score("private_training", response(values, **kwargs), target, info)


def test_accuracy_clips_components_before_weighting():
    result = score([0, 6], [0, 2], metadata([0, 6]))
    assert result["nmae"] == pytest.approx(2)
    assert result["nmse"] == pytest.approx(8)
    assert result["accuracy"] == pytest.approx(-2.2)
    assert result["uplift"] == 0
    assert result["score"] == pytest.approx(-2.2)


def test_scale_floor_and_json_ground_truth():
    result = score([105, 105], json.dumps([100, 100]), metadata([105, 105]))
    assert result["target_scale"] == pytest.approx(5)
    assert result["accuracy"] == pytest.approx(-1)
    zero = score([0, 0], [0, 0], metadata([0, 0]), decision="retain")
    assert zero["target_scale"] == pytest.approx(1e-8)
    assert zero["score"] == 0


def test_negative_finite_values_are_valid():
    result = score([-2, -1], [-2, -1], metadata([-2, -1]), decision="retain")
    assert result["format"] == 0
    assert result["accuracy"] == 0


def test_uplift_requires_both_absolute_and_relative_improvement():
    # A large relative improvement is insufficient below the absolute margin.
    absolute_gate = score([0.01, 2.01], [0, 2], metadata([0.02, 2.02]))
    assert absolute_gate["uplift"] == 0
    # A 0.09 absolute gain is insufficient below the 5% relative margin.
    relative_gate = score([3.9, 5.9], [0, 2], metadata([4, 6]))
    assert relative_gate["uplift"] == 0
    positive = score([0.8, 2.8], [0, 2], metadata([1, 3]))
    assert positive["uplift"] == pytest.approx(0.083)


def test_uplift_uses_best_of_all_four_frozen_baselines():
    info = metadata([2, 4])
    info["baselines"]["ridge"] = [0, 2]
    result = score([1, 3], [0, 2], info)
    assert result["baseline_loss"] == 0
    assert result["uplift"] == 0


def test_interval_reward_uses_shared_scale_and_outside_penalty():
    info = metadata([1, 3, 0, 2], baseline=[0, 2, 1, 3], intervals=[[0, 2]])
    result = score([0, 2, 1, 3], [0, 2, 0, 2], info)
    # Local improvement +1 gives +0.2; outside degradation +1 costs 0.1.
    assert result["interval"] == pytest.approx(0.1)
    info["intervals"] = []
    assert score([0, 2, 1, 3], [0, 2, 0, 2], info)["interval"] == 0


def test_interval_gains_are_signed_capped_and_weighted_by_length():
    target = [0, 2, 0, 2]
    info = metadata([10, 2, 0, 2], intervals=[[0, 1], [1, 4]])
    result = score([0, 3, 1, 3], target, info)
    # First interval improves by >1, second degrades by 1, weights 1/4, 3/4.
    assert result["interval"] == pytest.approx(-0.1)


def test_overlapping_annotations_preserve_the_paper_interval_weights():
    target = [0, 2, 0, 2]
    info = metadata([1, 3, 1, 3], intervals=[[0, 3], [2, 4]])
    result = score(target, target, info)
    assert result["interval"] == pytest.approx(0.2)


def test_teacher_reward_embeds_complete_valid_thinks_and_uses_max_similarity(monkeypatch):
    target = [0, 2]
    teacher = response(target, analysis="Teacher evidence analysis.", method="Teacher independent forecast.")
    info = metadata([1, 3], teachers=[{"response": teacher, "forecast": target}])
    seen = []

    def similarity(student, teachers):
        seen.append((student, teachers))
        return 0.85

    monkeypatch.setattr(rewards, "_maximum_cosine_similarity", similarity)
    result = score([0.5, 2.5], target, info)
    assert result["reasoning"] == pytest.approx(0.01)
    assert result["qualified_teachers"] == 1
    assert "analysis:" in seen[0][0] and "decision:" in seen[0][0] and "forecast_method:" in seen[0][0]
    assert "Teacher evidence analysis." in seen[0][1][0]
    assert result["score"] == pytest.approx(sum(result[name] for name in rewards.COMPONENT_NAMES))


def test_teacher_requires_strict_mae_improvement_and_valid_cot(monkeypatch):
    prior = [1, 3]
    info = metadata(prior, teachers=[
        {"response": response(prior, decision="retain"), "forecast": prior},
        {"response": "invalid teacher CoT", "forecast": [0, 2]},
    ])
    monkeypatch.setattr(rewards, "_get_encoder", lambda: pytest.fail("encoder must remain lazy"))
    result = score([0, 2], [0, 2], info)
    assert result["qualified_teachers"] == 0
    assert result["reasoning"] == 0


def test_student_quality_gate_prevents_teacher_embedding(monkeypatch):
    target = [0, 2]
    info = metadata([1, 3], teachers=[{"response": response(target), "forecast": target}])
    monkeypatch.setattr(rewards, "_get_encoder", lambda: pytest.fail("worse student cannot request embeddings"))
    assert score([2, 4], target, info)["reasoning"] == 0


@pytest.mark.parametrize("output", [
    "bad output",
    response([0, 2]).replace("analysis:", "reason:"),
    response([0, 2]).replace("0 | 0\n1 | 2", "1 | 0\n2 | 2"),
    response([0, 2]) + " trailing prose",
    response([0, 2], decision="retain"),
])
def test_invalid_output_gets_only_format_penalty(output, monkeypatch):
    target = [0, 2]
    info = metadata([1, 3], teachers=[{"response": response(target), "forecast": target}])
    monkeypatch.setattr(rewards, "_get_encoder", lambda: pytest.fail("invalid output cannot request embeddings"))
    result = rewards.compute_score(None, output, target, info)
    assert result["score"] == -1
    assert result["format"] == -1
    assert all(result[name] == 0 for name in rewards.COMPONENT_NAMES if name != "format")


def test_mixed_valid_invalid_rollouts_can_batch_native_verl_diagnostics():
    info = metadata([1, 3])
    valid = score([0, 2], [0, 2], info)
    invalid = rewards.compute_score(None, "incomplete rollout", [0, 2], info)
    # AgentLoopWorker._postprocess indexes every result using the first
    # result's keys, then concatenates independently processed worker batches.
    batches = []
    for results in ([valid, invalid], [invalid, valid]):
        batches.append({key: np.array([result[key] for result in results]) for key in results[0]})
    assert batches[0].keys() == batches[1].keys()
    for key in batches[0]:
        assert np.all(np.isfinite(np.concatenate([batch[key] for batch in batches])))
    assert batches[0]["score"].tolist() == [valid["score"], -1]


@pytest.mark.parametrize("mutate", [
    lambda item: item.pop("prior"),
    lambda item: item["prior"].append(0),
    lambda item: item["prior"].__setitem__(0, float("nan")),
    lambda item: item["baselines"].pop("ridge"),
    lambda item: item["baselines"].__setitem__("ridge", [0]),
    lambda item: item.__setitem__("intervals", [[0, 3]]),
    lambda item: item.__setitem__("intervals", [[True, 2]]),
    lambda item: item.__setitem__("teachers", [{"response": "x", "forecast": [float("nan"), 2]}]),
])
def test_metadata_errors_raise_even_for_invalid_student_output(mutate):
    info = copy.deepcopy(metadata([1, 3]))
    mutate(info)
    with pytest.raises(ValueError):
        rewards.compute_score(None, "bad output", [0, 2], info)


@pytest.mark.parametrize("target", ["not JSON", "{}", [], [True, 2], [float("inf"), 2]])
def test_invalid_ground_truth_raises(target):
    with pytest.raises(ValueError):
        rewards.compute_score(None, "bad output", target, metadata([1, 3]))


def test_teacher_forecast_must_match_response():
    info = metadata([1, 3], teachers=[{"response": response([0, 2]), "forecast": [0, 1]}])
    with pytest.raises(ValueError, match="differs"):
        score([0, 2], [0, 2], info)


def test_encoder_is_local_frozen_cached_and_uses_cpu(monkeypatch):
    calls = []

    class Parameter:
        def requires_grad_(self, value):
            calls.append(("gradient", value))

    class Encoder:
        def __init__(self, name, **kwargs):
            calls.append(("load", name, kwargs))

        def eval(self):
            calls.append(("eval",))

        def parameters(self):
            return [Parameter()]

        def encode(self, texts, **kwargs):
            calls.append(("encode", texts, kwargs))
            return np.asarray([[1, 0], [0, 1], [0.8, 0.6]])

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=Encoder))
    rewards._get_encoder.cache_clear()
    try:
        assert rewards._maximum_cosine_similarity("student", ["teacher one", "teacher two"]) == pytest.approx(0.8)
        assert rewards._get_encoder() is rewards._get_encoder()
        assert calls[0] == ("load", "BAAI/bge-m3", {"device": "cpu", "local_files_only": True})
        assert ("eval",) in calls and ("gradient", False) in calls
        encoding = next(call for call in calls if call[0] == "encode")
        assert encoding[1] == ["student", "teacher one", "teacher two"]
        assert encoding[2]["device"] == "cpu"
        assert sum(call[0] == "load" for call in calls) == 1
    finally:
        rewards._get_encoder.cache_clear()


def test_extreme_finite_predictions_keep_rewards_finite():
    result = score([1e308, -1e308], [0, 2], metadata([0, 2]))
    assert result["accuracy"] == -4
    assert all(math.isfinite(value) for value in result.values())


def test_retaining_fractional_displayed_prior_does_not_expand_float_precision():
    result = score([0.1, 0.2], [0.1, 0.2], metadata([0.1, 0.2]), decision="retain")
    assert result["format"] == 0


def test_prompt_owned_think_prefix_is_explicit_and_not_a_general_repair():
    info = metadata([1, 3])
    output = response([0, 2])
    assert rewards.compute_score(None, output.removeprefix("<think>\n"), [0, 2], info)["format"] == -1
    info["response_prefix"] = "<think>\n"
    assert rewards.compute_score(None, output.removeprefix("<think>\n"), [0, 2], info)["format"] == 0
    assert rewards.compute_score(None, output, [0, 2], info)["format"] == -1
    info["response_prefix"] = "<think>"
    with pytest.raises(ValueError):
        rewards.validate_metadata([0, 2], info)


def test_preflight_checks_metadata_without_loading_the_encoder(monkeypatch):
    monkeypatch.setattr(rewards, "_get_encoder", lambda: pytest.fail("preflight must not load weights"))
    info = metadata([1, 3], teachers=[{"response": response([0, 2]), "forecast": [0, 2]}])
    assert rewards.validate_metadata([0, 2], info)
    assert not rewards.validate_metadata([0, 2], metadata([1, 3]))


def test_overlapping_intervals_use_original_lengths_even_when_local_clipping_differs():
    # Shared scale=1. Separate gains are 1.75 (clipped to 1) and .5.
    target = [0, 2, 0, 2]
    info = metadata([3, 3, 0, 2], intervals=[[0, 2], [1, 3]])
    result = score(target, target, info)
    assert result["interval"] == pytest.approx(.15)
