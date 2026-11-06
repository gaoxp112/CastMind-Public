"""Convert raw hourly Price CSV into forecast windows and SFT/GRPO Parquet files."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from castmind.contract import parse_response
from castmind.dataio import read_rows, write_json, write_rows
from castmind.prompt import render_prompt, validate_prompt


TARGET = "zonal_price"
EXOG = ["forecasted_total_load", "forecasted_zonal_load"]
COLUMNS = ["series_id", "timestamp", TARGET, *EXOG]


def read_raw(filename):
    frame = pd.read_csv(filename)
    if set(frame.columns) != set(COLUMNS):
        raise ValueError(f"Price CSV must have exactly these columns: {COLUMNS}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    if frame["series_id"].isna().any() or frame["series_id"].nunique() != 1:
        raise ValueError("this example requires one nonempty Price series_id")
    if not np.isfinite(frame[[TARGET, *EXOG]].to_numpy(dtype=float)).all():
        raise ValueError("raw target and load forecasts must be finite")
    if not frame["timestamp"].diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Price timestamps must be sorted, unique, contiguous hourly observations")
    return frame


def split_windows(frame, history=96, horizon=96, stride=24, limits=(32, 8, 8)):
    if min(history, horizon, stride, *limits) < 1:
        raise ValueError("lengths and window limits must be positive")
    boundaries = [0, int(len(frame) * .6), int(len(frame) * .8), len(frame)]
    for index, split in enumerate(("train", "validation", "test")):
        start, end = boundaries[index:index + 2]
        count = 0
        for offset in range(start + history, end - horizon + 1, stride):
            if count >= limits[index]:
                break
            yield split, frame.iloc[offset-history:offset].copy(), frame.iloc[offset:offset+horizon].copy()
            count += 1
        if count == 0:
            raise ValueError(f"{split} segment needs at least history+horizon={history+horizon} rows")


def _features(frame):
    hour = frame["timestamp"].dt.hour.to_numpy()
    day = frame["timestamp"].dt.dayofweek.to_numpy()
    return np.column_stack((np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24), day/6,
                            frame[EXOG].to_numpy(dtype=float)))


def history_baselines(history, future):
    """Example settings: mean/season=24h; ridge alpha=1, fit on history only."""
    y = history[TARGET].to_numpy(dtype=float)
    horizon = len(future)
    if len(y) < 24:
        raise ValueError("Price baselines need at least 24 observed hourly prices")
    x, xf = _features(history), _features(future)
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.maximum(scale, 1e-8)
    x, xf = (x-center)/scale, (xf-center)/scale
    x, xf = np.column_stack((np.ones(len(x)), x)), np.column_stack((np.ones(len(xf)), xf))
    penalty = np.eye(x.shape[1]); penalty[0, 0] = 0
    weights = np.linalg.solve(x.T@x + penalty, x.T@y)
    return {
        "last_value": [float(y[-1])] * horizon,
        "recent_mean": [float(y[-24:].mean())] * horizon,
        "seasonal_copy": [float(y[-24:][i % 24]) for i in range(horizon)],
        "ridge": (xf@weights).tolist(),
    }


def chronos_prior(pipeline, history, future):
    # Never pass the future target column to the forecasting model.
    result = pipeline.predict_df(
        history[COLUMNS], future_df=future[["series_id", "timestamp", *EXOG]],
        id_column="series_id", timestamp_column="timestamp", target=TARGET,
        prediction_length=len(future), quantile_levels=[0.5],
    )
    if "target_name" in result:
        result = result[result["target_name"] == TARGET]
    aligned = future[["series_id", "timestamp"]].merge(
        result[["series_id", "timestamp", "predictions"]],
        on=["series_id", "timestamp"], how="left", validate="one_to_one",
    )
    prior = aligned["predictions"].to_numpy(dtype=float)
    if len(prior) != len(future) or not np.isfinite(prior).all():
        raise ValueError("Chronos-2 did not return a complete finite prior")
    return np.round(prior, 2).tolist()


def window_row(history, future, prior):
    observed = ["index | time | target | forecasted_total_load | forecasted_zonal_load | hour | weekday"]
    forecast = ["index | time | prior_forecast | forecasted_total_load | forecasted_zonal_load | hour | weekday"]
    for index, (_, row) in enumerate(history.iterrows()):
        ts = row["timestamp"]
        observed.append(f"{index} | {ts:%Y-%m-%d %H:%M} | {row[TARGET]} | {row[EXOG[0]]} | {row[EXOG[1]]} | {ts.hour} | {ts.dayofweek}")
    for index, (_, row) in enumerate(future.iterrows()):
        ts = row["timestamp"]
        forecast.append(f"{index} | {ts:%Y-%m-%d %H:%M} | {prior[index]:.2f} | {row[EXOG[0]]} | {row[EXOG[1]]} | {ts.hour} | {ts.dayofweek}")
    prompt = render_prompt(
        task_definition=f"Forecast the next {len(future)} hourly zonal electricity prices for {history.iloc[0]['series_id']}.",
        context_information="GEFCom2014 Price. The two exogenous columns are load forecasts available at prediction time. Weekday: Monday=0, Sunday=6.",
        observed_rows="\n".join(observed),
        future_context_information="Future load forecasts and calendar features are supplied below; future prices are not supplied.",
        forecast_rows="\n".join(forecast),
    )
    validate_prompt(prompt, prior)
    series = str(history.iloc[0]["series_id"])
    return {
        "sample_id": f"{series}_{future.iloc[0]['timestamp']:%Y%m%dT%H%M}",
        "data_source": "Price", "series_id": series, "target_name": TARGET,
        "history_start": str(history.iloc[0]["timestamp"]), "forecast_start": str(future.iloc[0]["timestamp"]),
        "forecast_end": str(future.iloc[-1]["timestamp"]),
        "prompt": [{"role": "user", "content": prompt}],
        "reward_model": {"style": "rule", "ground_truth": json.dumps(future[TARGET].astype(float).tolist())},
        "extra_info": {"prior": prior, "baselines": history_baselines(history, future), "intervals": [], "teachers": []},
    }


def convert(args):
    from chronos import Chronos2Pipeline

    frame = read_raw(args.csv)
    pipeline = Chronos2Pipeline.from_pretrained(args.prior_model, device_map=args.device)
    output = Path(args.output_dir)
    grouped = {name: [] for name in ("train", "validation", "test")}
    for split, history, future in split_windows(frame, args.history, args.horizon, args.stride,
                                               (args.train_windows, args.validation_windows, args.test_windows)):
        grouped[split].append(window_row(history, future, chronos_prior(pipeline, history, future)))
    for split, rows in grouped.items():
        write_rows(output / f"{split}_windows.parquet", rows)
    write_json(output / "conversion.json", {
        "raw_csv_sha256": hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(),
        "raw_rows": len(frame), "split": "chronological 60/20/20; history and future contained in each split",
        "prior_model": args.prior_model, "history": args.history, "horizon": args.horizon, "stride": args.stride,
        "windows": {key: len(value) for key, value in grouped.items()},
        "settings": "single-domain runnable example; not the paper's eight-domain training mixture",
    })


def _cached_completion(teacher, messages, cache_dir, kind):
    from castmind.teacher_api import complete

    identity = {"teacher": teacher, "messages": messages, "kind": kind}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / f"{digest}.json"
    if path.exists():
        return json.loads(path.read_text())["response"]
    text = complete(teacher, messages)
    write_json(path, {"teacher": teacher["name"], "model": teacher["model"], "response": text})
    return text


def distill(args):
    from concurrent.futures import ThreadPoolExecutor
    from castmind.infer import load_model
    from castmind.sad import TrajectoryCandidate, score_teacher_response, select_trajectory
    from castmind.teacher_api import load_config
    from castmind.rewards import validate_metadata

    config = load_config(args.teacher_config)
    model, tokenizer = load_model(args.student_model, args.device)
    output = Path(args.output_dir)
    cache = output / "teacher_cache"
    annotator = next(t for t in config["teachers"] if t["name"] == config["interval_teacher"])
    sft, rl, reports = [], [], []
    for row in read_rows(output / "train_windows.parquet"):
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(lambda teacher: _cached_completion(teacher, row["prompt"], cache, "trajectory"), config["teachers"]))
        target = np.asarray(json.loads(row["reward_model"]["ground_truth"]), dtype=float)
        candidates, teachers, failures = [], [], []
        for teacher, response in zip(config["teachers"], responses, strict=True):
            try:
                parsed = parse_response(response, len(target), row["extra_info"]["prior"])
            except ValueError as error:
                failures.append({"teacher": teacher["name"], "error": str(error)})
                continue
            score = score_teacher_response(model, tokenizer, row["prompt"], response)
            candidates.append(TrajectoryCandidate(teacher["name"], parsed.decision,
                              float(np.abs(np.asarray(parsed.values)-target).mean()), score))
            teachers.append({"response": response, "forecast": list(parsed.values)})
        if not candidates:
            write_json(output / "failed_sample.json", {"sample_id": row["sample_id"], "failures": failures})
            raise ValueError(f"all four teachers returned invalid trajectories for {row['sample_id']}; inspect teacher_cache and failed_sample.json")
        chosen = select_trajectory(candidates)
        selected = responses[[t["name"] for t in config["teachers"]].index(chosen.candidate_id)]
        # The interval annotator sees exactly the forecast-time input, never y.
        annotation_prompt = row["prompt"][0]["content"].split("3. Decision and final forecast.")[0]
        annotation_prompt += '\nAnnotate forecast intervals with exogenous evidence supporting prior correction. Return only JSON {"intervals": [[start, end], ...]}, zero-based half-open intervals within the forecast horizon. Return an empty list when unsupported. Do not predict prices.'
        annotation = _cached_completion(annotator, [{"role": "user", "content": annotation_prompt}], cache, "intervals")
        text = annotation.strip()
        if text.startswith("```json") and text.endswith("```"):
            text = text[7:-3].strip()
        row["extra_info"]["intervals"] = json.loads(text)["intervals"]
        row["extra_info"]["teachers"] = teachers
        validate_metadata(row["reward_model"]["ground_truth"], row["extra_info"])
        rl.append(row)
        sft.append({"data_source": "Price", "sample_id": row["sample_id"],
                    "messages": [*row["prompt"], {"role": "assistant", "content": selected}],
                    "extra_info": {"prior": row["extra_info"]["prior"]}})
        reports.append({"sample_id": row["sample_id"], "selected_teacher": chosen.candidate_id,
                        "candidates": [{"teacher": c.candidate_id, "mae": c.mae, "sad": c.sad.value} for c in candidates],
                        "invalid_teachers": failures})
        write_json(output / "selection.json", reports)
    write_rows(output / "sft.parquet", sft)
    write_rows(output / "rl.parquet", rl)
    # Validation is reward input only; no held-out teacher generation or SFT.
    write_rows(output / "validation.parquet", read_rows(output / "validation_windows.parquet"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["convert", "distill"])
    p.add_argument("--csv", default="examples/price/raw.csv")
    p.add_argument("--output-dir", default="outputs/price/data")
    p.add_argument("--prior-model", default="amazon/chronos-2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--history", type=int, default=96)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--stride", type=int, default=24)
    p.add_argument("--train-windows", type=int, default=32)
    p.add_argument("--validation-windows", type=int, default=8)
    p.add_argument("--test-windows", type=int, default=8)
    p.add_argument("--teacher-config")
    p.add_argument("--student-model")
    args = p.parse_args()
    if args.stage == "distill" and (not args.teacher_config or not args.student_model):
        p.error("distill requires --teacher-config and --student-model")
    (convert if args.stage == "convert" else distill)(args)


if __name__ == "__main__":
    main()
