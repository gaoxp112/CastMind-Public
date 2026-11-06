"""Leakage boundaries and the raw-CSV → prepared data → evaluation workflow."""

from argparse import Namespace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from castmind.dataio import read_rows, write_rows
from castmind.evaluate import evaluate
from castmind.infer import forecast_rows
from castmind.prepare_price import read_raw, split_windows, window_row, chronos_prior, distill
from castmind.sad import SADScore


RAW = Path(__file__).parents[1] / "examples/price/raw.csv"


def response(prior):
    table = "\n".join(f"{i} | {value}" for i, value in enumerate(prior))
    return ("<think>\nanalysis: The available load forecasts do not establish a reliable correction.\n"
            "decision: retain\nforecast_method: none\n</think>\n<answer>\n```\n"
            "index | value\n" + table + "\n```\n</answer>")


def make_rows():
    frame = read_raw(RAW)
    return [(split, window_row(history, future, [float(history.iloc[-1].zonal_price)] * len(future)))
            for split, history, future in split_windows(frame, limits=(4, 1, 1))]


def test_raw_windows_have_disjoint_segments_and_no_target_leakage():
    frame = read_raw(RAW)
    seen = {}
    for split, history, future in split_windows(frame, limits=(4, 1, 1)):
        seen.setdefault(split, set()).update(history.timestamp)
        seen[split].update(future.timestamp)
        prior = [1.25] * len(future)
        before = window_row(history, future, prior)
        changed = future.copy(); changed["zonal_price"] += 999999
        after = window_row(history, changed, prior)
        assert before["prompt"] == after["prompt"]
        assert before["extra_info"] == after["extra_info"]
        assert before["reward_model"] != after["reward_model"]
    assert not seen["train"] & (seen["validation"] | seen["test"])
    assert not seen["validation"] & seen["test"]


def test_chronos_receives_only_available_future_covariates():
    import pandas as pd
    _, history, future = next(split_windows(read_raw(RAW), limits=(1, 1, 1)))

    class Pipeline:
        def predict_df(self, observed, *, future_df, **kwargs):
            assert "zonal_price" not in future_df
            assert observed.timestamp.max() < future_df.timestamp.min()
            output = future_df[["series_id", "timestamp"]].copy()
            output["predictions"] = 3.14159
            return output.iloc[::-1]  # Alignment is by timestamp, not incidental order.

    assert chronos_prior(Pipeline(), history, future) == [3.14] * 96


def test_distillation_roundtrip_and_evaluation(tmp_path, monkeypatch):
    import castmind.teacher_api as api
    import castmind.infer as inference
    import castmind.sad as sad

    rows = make_rows()
    for name in ("train", "validation", "test"):
        write_rows(tmp_path / f"{name}_windows.parquet", [row for split, row in rows if split == name])
    teachers = [{"name": str(i), "model": str(i)} for i in range(4)]
    monkeypatch.setattr(api, "load_config", lambda path: {"teachers": teachers, "interval_teacher": "0"})
    monkeypatch.setattr(inference, "load_model", lambda *a: (object(), object()))
    monkeypatch.setattr(sad, "score_teacher_response", lambda *a: SADScore(1.0, {}, ()))
    calls = []

    def complete(teacher, messages):
        calls.append((teacher["name"], messages))
        text = messages[0]["content"]
        if "Annotate forecast intervals" in text:
            return '{"intervals": []}'
        forecast = text.split("prior_forecast is a cached")[0].rsplit("index | time | prior_forecast", 1)[1]
        prior = [float(line.split("|")[2]) for line in forecast.splitlines()[1:] if line.strip()]
        return response(prior)

    monkeypatch.setattr(api, "complete", complete)
    args = Namespace(output_dir=str(tmp_path), teacher_config="unused", student_model="unused", device="cpu")
    distill(args)
    assert len(calls) == 20  # Four teachers and one annotation for each of four samples.
    assert len(read_rows(tmp_path / "sft.parquet")) == 4
    assert all(len(r["extra_info"]["teachers"]) == 4 for r in read_rows(tmp_path / "rl.parquet"))
    distill(args)
    assert len(calls) == 20  # Resume consumes caches and does not repeat paid requests.
    test = read_rows(tmp_path / "test_windows.parquet")
    prior = test[0]["extra_info"]["prior"]
    predictions = list(forecast_rows(test, lambda messages: response(prior)))
    summary, records = evaluate(test, predictions)
    assert summary["overall"]["num_valid_samples"] == 1
    target = np.asarray(json.loads(test[0]["reward_model"]["ground_truth"]))
    assert records[0]["nmae"] == pytest.approx(np.abs(target-prior).mean()/target.std())
    predictions[0]["prompt_sha256"] = "wrong"
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluate(test, predictions)


def test_metrics_use_all_gt_and_equal_window_weights():
    from castmind.metrics import aggregate_sample_metrics
    records = [
        {"ground_truth": [0., 2.], "prediction": [1., 3.], "evaluation_dataset": "Price", "metric_group": "one", "attr": "price"},
        {"ground_truth": [4.], "prediction": [7.], "evaluation_dataset": "Price", "metric_group": "one", "attr": "price"},
        {"ground_truth": [6.], "prediction": [], "evaluation_dataset": "Price", "metric_group": "one", "attr": "price"},
    ]
    summary = aggregate_sample_metrics(records)
    assert summary["metric_groups"][0]["num_gt_points"] == 4
    assert summary["overall"]["nmae"] == pytest.approx(2/np.std([0, 2, 4, 6]))
    assert summary["overall"]["num_invalid_samples"] == 1
    records[1]["prediction"] = [1e20]
    summary = aggregate_sample_metrics(records)
    assert summary["overall"]["num_exploded"] == 1
    assert summary["overall"]["num_scored_samples"] == 1


def test_single_domain_profile_does_not_weaken_paper_profile():
    from castmind.train import compose_config, launch_command
    for stage in ("sft", "grpo"):
        c = compose_config(stage, "model", "train.parquet", "out", "val.parquet", profile="price-example", gpus=1, max_steps=1)
        assert c.trainer.n_gpus_per_node == 1
        assert c.trainer.total_training_steps == 1
        assert c.data.train_batch_size == 4
    assert "--nproc_per_node=1" in launch_command("sft", "file.yaml", 1)
    with pytest.raises(ValueError, match="preserves"):
        compose_config("sft", "model", "train", "out", gpus=1)
