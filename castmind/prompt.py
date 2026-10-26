"""The paper's forecasting instruction; callers supply already serialized inputs."""

from decimal import Decimal, InvalidOperation
from pathlib import Path
import re


PROMPT_TEMPLATE = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
_FIELDS = ("task_definition", "context_information", "observed_rows", "future_context_information", "forecast_rows")


def render_prompt(**fields: str) -> str:
    """Insert prepared history/context/prior text; never accepts future targets."""
    if set(fields) != set(_FIELDS) or any(not isinstance(value, str) for value in fields.values()):
        raise ValueError(f"supply text for exactly these fields: {', '.join(_FIELDS)}")
    if any(not fields[key].strip() for key in ("task_definition", "observed_rows", "forecast_rows")):
        raise ValueError("task, history, and forecast rows must be nonempty")
    return PROMPT_TEMPLATE.format(**fields)


def validate_prompt(text: str, prior=None) -> None:
    """Reject prepared prompts that change the shared instruction or field order."""
    if not isinstance(text, str):
        raise ValueError("prompt must be text")
    parts = re.split(r"\{(" + "|".join(_FIELDS) + r")\}", PROMPT_TEMPLATE.strip())
    pattern = ""
    for index, part in enumerate(parts):
        pattern += (r"(?P<" + part + r">.*?)") if index % 2 else re.escape(part)
    match = re.fullmatch(pattern, text.strip(), re.DOTALL)
    if match is None:
        raise ValueError("prepared prompt differs from castmind/prompt.txt")
    for key in ("task_definition", "observed_rows", "forecast_rows"):
        if not match.group(key).strip() or match.group(key).strip() == "{" + key + "}":
            raise ValueError(f"prompt field {key} is empty or still a placeholder")

    if prior is not None:
        # Bind private reward metadata to the exact point forecast shown to the model.
        rows = [line.strip() for line in match.group("forecast_rows").strip().splitlines()]
        header = [cell.strip() for cell in rows[0].split("|")]
        if len(header) < 2 or header[0] != "index" or header.count("prior_forecast") != 1:
            raise ValueError("forecast rows require an index column and one prior_forecast column")
        column = header.index("prior_forecast")
        if len(rows) != len(prior) + 1:
            raise ValueError("displayed forecast rows must cover the complete prior horizon")
        for index, line in enumerate(rows[1:]):
            cells = [cell.strip() for cell in line.split("|")]
            if len(cells) != len(header) or cells[0] != str(index):
                raise ValueError("displayed forecast rows must use consecutive zero-based indices")
            try:
                shown = Decimal(cells[column])
                expected = Decimal(str(prior[index]))
            except InvalidOperation as error:
                raise ValueError("displayed prior must contain finite numeric values") from error
            if not shown.is_finite() or shown != expected:
                raise ValueError("extra_info.prior differs from the displayed prior_forecast")
