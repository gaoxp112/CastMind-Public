"""Small portable artifact helpers; no training framework imports."""

import json
import math
from collections.abc import Mapping
from pathlib import Path


def plain(value):
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def read_rows(filename):
    import pandas as pd

    return [plain(row) for row in pd.read_parquet(filename).to_dict(orient="records")]


def write_rows(filename, rows):
    import pandas as pd

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(filename, value):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
