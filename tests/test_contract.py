"""Boundary checks for the new reasoning order and complete forecast protocol."""

import pytest

from castmind.contract import parse_response
from castmind.prompt import render_prompt, validate_prompt


def output(think, rows="0 | -0.125\n1 | 2e-6"):
    return f"<think>\n{think}\n</think>\n<answer>\n```\nindex | value\n{rows}\n```\n</answer>"


THINK = "analysis: Calendar evidence supports the prior.\ndecision: retain\nforecast_method: none"


def test_precision_negative_values_and_interval_notation():
    result = parse_response(output(THINK), 2, [-.125, .000002])
    assert result.values == (-.125, .000002)
    changed = THINK.replace("Calendar evidence", "Evidence in [0, 2)")
    assert parse_response(output(changed), 2, [-.125, .000002]).decision == "retain"
    with pytest.raises(ValueError):
        parse_response(output(THINK, "0 | -0.12\n1 | 0.000002"), 2, [-.125, .000002])


@pytest.mark.parametrize("bad", [
    output(THINK.replace("analysis:", "reason:")),
    output("decision: retain\nanalysis: Evidence.\nforecast_method: none"),
    output(THINK.replace("none", "copy prior")),
    output(THINK, "0 | nan\n1 | 0.1"),
    output(THINK, "0 | 1e999\n1 | 0.1"),
    output(THINK, "1 | -0.125\n2 | 2e-6"),
    output(THINK, "0 | -0.125"),
    output(THINK) + " extra",
    output(THINK.replace("Calendar evidence supports the prior.", "[analyze the context]")),
])
def test_reject_bad_order_incomplete_or_invalid_forecasts(bad):
    with pytest.raises(ValueError):
        parse_response(bad, 2, [-.125, .000002])


def test_canonical_prompt_binds_displayed_prior_without_exposing_reward_inputs():
    prompt = render_prompt(
        task_definition="Forecast two hours.", context_information="Available calendar inputs.",
        observed_rows="index | time | target\n0 | 2026-01-01 00:00 | 1",
        future_context_information="Holiday.",
        forecast_rows="index | time | prior_forecast\n0 | 2026-01-01 01:00 | -0.125\n1 | 2026-01-01 02:00 | 0.000002",
    )
    validate_prompt(prompt, [-.125, .000002])
    with pytest.raises(ValueError, match="differs from the displayed"):
        validate_prompt(prompt, [-.12, .000002])
    with pytest.raises(ValueError):
        validate_prompt(prompt.replace("analysis:", "reason:"))
