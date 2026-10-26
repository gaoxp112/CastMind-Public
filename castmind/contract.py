"""Strict response contract shared by SFT validation and the GRPO reward."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import math
import re
from typing import Sequence


_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
_BLOCKS = re.compile(r"\s*<think>\s*(.*?)\s*</think>\s*<answer>\s*```\s*\n(.*?)\n```\s*</answer>\s*\Z", re.DOTALL)
_FIELDS = re.compile(
    r"analysis:\s*(.+?)\n[ \t]*decision:[ \t]*(retain|revise|reconstruct)[ \t]*\n"
    r"[ \t]*forecast_method:\s*(.+)\Z", re.DOTALL,
)


@dataclass(frozen=True)
class ForecastResponse:
    analysis: str
    decision: str
    forecast_method: str
    values: tuple[float, ...]
    think: str


def parse_response(text: str, horizon: int, prior: Sequence[float]) -> ForecastResponse:
    """Require the three ordered fields and the entire zero-indexed forecast.

    Retain compares decimal values without rounding or a numerical tolerance.
    Negative forecasts are allowed. This validates structure and consistency;
    the quality of the written evidence is assessed by the reasoning reward.
    """
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(text, str):
        raise ValueError("response must be text")
    blocks = _BLOCKS.fullmatch(text)
    if blocks is None:
        raise ValueError("expected exactly a think block and a fenced answer block")
    think, table = blocks.groups()
    if re.search(r"</?(?:think|answer)>", think + table):
        raise ValueError("nested or duplicate output blocks")
    fields = _FIELDS.fullmatch(think)
    if fields is None:
        raise ValueError("expected analysis, decision, forecast_method in that order")
    analysis, decision, method = (field.strip() for field in fields.groups())
    if re.search(r"(?:^|\n)\s*(?:analysis|decision|forecast_method|reason):", analysis + "\n" + method):
        raise ValueError("duplicate or obsolete reasoning field")
    if not analysis or not method or re.search(r"\[(?:analyze|none for|retain,|finite numeric value)", analysis + method) or re.search(r"(?:^|\n)\s*(?:\.\.\.|…)\s*(?:$|\n)", analysis + "\n" + method):
        raise ValueError("reasoning fields must contain completed text")
    if decision == "retain" and method != "none":
        raise ValueError("retain requires forecast_method: none")
    if decision != "retain" and method == "none":
        raise ValueError("revise and reconstruct require a forecasting method")

    rows = table.strip().splitlines()
    if not rows or rows[0].strip() != "index | value" or len(rows) != horizon + 1:
        raise ValueError("forecast must have its header and exactly H rows")
    values = []
    decimals = []
    for index, row in enumerate(rows[1:]):
        cells = [cell.strip() for cell in row.split("|")]
        if len(cells) != 2 or cells[0] != str(index) or _NUMBER.fullmatch(cells[1]) is None:
            raise ValueError("forecast requires consecutive zero-based indices and numeric values")
        value = float(cells[1])
        if not math.isfinite(value):
            raise ValueError("forecast values must be finite")
        values.append(value)
        decimals.append(Decimal(cells[1]))
    try:
        prior_decimals = [Decimal(str(value)) for value in prior]
    except (TypeError, InvalidOperation) as error:
        raise ValueError("prior must contain H finite numeric values") from error
    if len(prior_decimals) != horizon or not all(value.is_finite() for value in prior_decimals):
        raise ValueError("prior must contain H finite numeric values")
    if decision == "retain" and decimals != prior_decimals:
        raise ValueError("retain must copy the displayed prior values exactly")
    return ForecastResponse(analysis, decision, method, tuple(values), think)
