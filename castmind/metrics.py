"""Portable sample-mean forecasting metrics used by the release evaluation CLI.

The task-scale functions are shared with the experiment implementation.
Training rewards deliberately use a different, per-window scale.
"""
from __future__ import annotations
from collections import defaultdict
import math
from typing import Iterable, Mapping

EXPLOSION_K = 10.0
FORECAST_METRIC_CONTRACT = "task_future_gt_scale_sample_mean_exploded_excluded_v3"
FORECAST_SCALE_FLOOR = 1e-8

def _empty_error_fields() -> dict[str, object]:
    return {
        "num_points": 0,
        "valid": False,
        "exploded": False,
        "squared_error_sum": 0.0,
        "absolute_error_sum": 0.0,
        "normalized_squared_error_sum": 0.0,
        "normalized_absolute_error_sum": 0.0,
        "mse": math.inf,
        "mae": math.inf,
        "nmse": math.inf,
        "nmae": math.inf,
        "metric_error": None,
    }


def compute_forecast_errors(
    ground_truth: Iterable[float | None],
    prediction: Iterable[float | None],
    *,
    explosion_k: float = EXPLOSION_K,
) -> dict[str, object]:
    """Per-sample error fields for the eval metric.

    Returns raw errors plus legacy per-series normalized fields used by older callers.
    Current evaluation must pass complete records through ``aggregate_sample_metrics``,
    which recomputes nMAE/nMSE with scales derived from all future GT in each task group.
    Non-finite predictions are invalid. A finite off-scale prediction may be flagged as
    ``exploded``; the current aggregation contract retains it for diagnostics but
    excludes it from headline sample means.
    """
    gt = [math.nan if value is None else float(value) for value in ground_truth]
    pred = [math.nan if value is None else float(value) for value in prediction]
    if not gt or len(pred) != len(gt):
        result = _empty_error_fields()
        result["metric_error"] = "wrong_horizon" if gt else "missing_ground_truth"
        return result

    valid_indices = [index for index in range(len(gt)) if math.isfinite(gt[index])]
    if not valid_indices:
        return _empty_error_fields()
    finite_gt = [gt[index] for index in valid_indices]
    pred_finite = all(math.isfinite(pred[index]) for index in valid_indices)
    gt_center = sum(finite_gt) / len(finite_gt) if finite_gt else 0.0
    # Z-score / standard-NMSE denominator: the ground truth's own standard deviation.
    # nmse = mean(((gt - pred) / std(gt))**2) == MSE / Var(gt), i.e. exactly the MSE of
    # gt and pred after z-scoring with the ground-truth mean/std (the mean cancels in the
    # difference, so only the std matters). Floored at 1e-8 to avoid divide-by-zero.
    gt_variance = (
        sum((value - gt_center) ** 2 for value in finite_gt) / len(finite_gt)
        if finite_gt
        else 0.0
    )
    gt_scale = math.sqrt(gt_variance)
    gt_scale = gt_scale if gt_scale > 1e-8 else 1e-8

    if pred_finite:
        pred_max_dev = max((abs(pred[index] - gt_center) for index in valid_indices), default=0.0)
        exploded = pred_max_dev > explosion_k * gt_scale
        try:
            squared_error_sum = float(sum((gt[index] - pred[index]) ** 2 for index in valid_indices))
        except OverflowError:
            squared_error_sum = math.inf
        absolute_error_sum = float(sum(abs(gt[index] - pred[index]) for index in valid_indices))
        try:
            normalized_squared_error_sum = float(
                sum(((gt[index] - pred[index]) / gt_scale) ** 2 for index in valid_indices)
            )
        except OverflowError:
            normalized_squared_error_sum = math.inf
        normalized_absolute_error_sum = float(
            sum(abs(gt[index] - pred[index]) / gt_scale for index in valid_indices)
        )
    else:
        result = _empty_error_fields()
        result["metric_error"] = "non_finite_prediction"
        return result

    num_points = len(valid_indices)
    return {
        "num_points": num_points,
        "valid": True,
        "exploded": bool(exploded),
        "squared_error_sum": squared_error_sum,
        "absolute_error_sum": absolute_error_sum,
        "normalized_squared_error_sum": normalized_squared_error_sum,
        "normalized_absolute_error_sum": normalized_absolute_error_sum,
        "mse": squared_error_sum / num_points,
        "mae": absolute_error_sum / num_points,
        "nmse": normalized_squared_error_sum / num_points,
        "nmae": normalized_absolute_error_sum / num_points,
        "metric_error": None,
    }


def aggregate_sample_metrics(records: Iterable[dict]) -> dict[str, object]:
    normalized_records = list(records)
    if any("mase" in record for record in normalized_records) or any(
        not record.get("ground_truth") for record in normalized_records
    ):
        raise ValueError("evaluation requires complete future ground truth; legacy metrics are unsupported")

    metric_groups = _apply_task_scale_metrics(normalized_records)
    dataset_groups: dict[str, list[dict]] = defaultdict(list)
    attr_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for record in normalized_records:
        dataset = str(record["evaluation_dataset"])
        dataset_groups[dataset].append(record)
        attr_groups[(dataset, str(record["attr"]))].append(record)

    dataset_summaries = [
        {"dataset": dataset, **_summarize_sample_mean_group(group)}
        for dataset, group in sorted(dataset_groups.items())
    ]
    overall = _summarize_contract_counts(normalized_records, len(dataset_summaries))
    if len(dataset_summaries) == 1:
        overall.update({key: value for key, value in dataset_summaries[0].items() if key != "dataset"})

    return {
        "metric_contract": FORECAST_METRIC_CONTRACT,
        "overall": overall,
        "datasets": dataset_summaries,
        "metric_groups": metric_groups,
        "attributes": [
            {"dataset": dataset, "attr": attr, **_summarize_sample_mean_group(group)}
            for (dataset, attr), group in sorted(attr_groups.items())
        ],
    }


def _record_metric_identity(record: Mapping[str, object]) -> tuple[str, str]:
    explicit_dataset = str(record.get("evaluation_dataset") or "").strip()
    explicit_group = str(record.get("metric_group") or "").strip()
    if explicit_dataset and explicit_group:
        return explicit_dataset, explicit_group

    dataset = str(record.get("dataset") or "unknown")
    source_sample_id = str(record.get("source_sample_id") or "").lower()
    source_dataset = str(record.get("source_dataset") or "").strip()
    if source_sample_id.startswith("cik_") or source_dataset.lower() == "context-is-key":
        return "CIK", explicit_group or dataset
    if dataset.lower() == "temporalbench":
        tier = str(record.get("track") or "").upper()
        evaluation_dataset = f"TemporalBench {tier}" if tier else "TemporalBench"
        selected_target = str(record.get("selected_target") or record.get("attr") or "").strip()
        metric_group = "::".join(
            value for value in (tier, source_dataset, selected_target) if value
        )
        return (
            explicit_dataset or evaluation_dataset,
            explicit_group or metric_group or evaluation_dataset,
        )
    if dataset.lower() == "timesx currency":
        currency_pair = str(record.get("selected_target") or record.get("attr") or "target").strip()
        return explicit_dataset or dataset, explicit_group or f"TimesX Currency::{currency_pair}"
    if dataset.lower() in {"aemo electricity", "from_news_to_forecast_aemo"}:
        state = str(record.get("state") or "").strip().upper()
        if not state:
            series_id = str(record.get("series_id") or "").strip()
            state = series_id.rsplit("_", 1)[-1].upper() if "_" in series_id else ""
        if not state:
            parts = source_sample_id.split("_")
            if len(parts) >= 3 and parts[:2] == ["from", "news"]:
                state = parts[2].upper()
        return (
            explicit_dataset or "AEMO Electricity",
            explicit_group or f"AEMO Electricity::{state or 'unknown'}",
        )
    return explicit_dataset or dataset, explicit_group or dataset


def _apply_task_scale_metrics(records: list[dict]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        evaluation_dataset, metric_group = _record_metric_identity(record)
        record["evaluation_dataset"] = evaluation_dataset
        record["metric_group"] = metric_group
        grouped[(evaluation_dataset, metric_group)].append(record)

    group_summaries: list[dict[str, object]] = []
    for (evaluation_dataset, metric_group), group_records in sorted(grouped.items()):
        gt_values: list[float] = []
        for record in group_records:
            ground_truth = record.get("ground_truth") or []
            if not ground_truth:
                raise ValueError(f"{metric_group}: missing future ground truth")
            values = [float(value) for value in ground_truth]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{metric_group}: future ground truth must be finite")
            gt_values.extend(values)

        gt_mean = sum(gt_values) / len(gt_values)
        raw_variance = sum((value - gt_mean) ** 2 for value in gt_values) / len(gt_values)
        normalization_std = max(math.sqrt(raw_variance), FORECAST_SCALE_FLOOR)
        normalization_variance = normalization_std * normalization_std

        for record in group_records:
            errors = compute_forecast_errors(
                record.get("ground_truth") or [],
                record.get("prediction") or [],
            )
            contract_invalid = (
                record.get("answer_contract_valid") is False
                or record.get("datav5_answer_contract_valid") is False
            )
            if contract_invalid:
                errors = _empty_error_fields()
                errors["metric_error"] = "answer_contract"
            record.update(errors)
            record["normalization_gt_mean"] = gt_mean
            record["normalization_std"] = normalization_std
            record["normalization_variance"] = normalization_variance
            if bool(record.get("valid")):
                prediction = [float(value) for value in record.get("prediction") or []]
                record["exploded"] = max(
                    (abs(value - gt_mean) for value in prediction), default=0.0
                ) > EXPLOSION_K * normalization_std
                record["nmae"] = float(record["mae"]) / normalization_std
                record["nmse"] = float(record["mse"]) / normalization_variance
                record["normalized_absolute_error_sum"] = (
                    float(record["absolute_error_sum"]) / normalization_std
                )
                record["normalized_squared_error_sum"] = (
                    float(record["squared_error_sum"]) / normalization_variance
                )

        summary = _summarize_sample_mean_group(group_records)
        group_summaries.append(
            {
                "evaluation_dataset": evaluation_dataset,
                "metric_group": metric_group,
                "num_gt_points": len(gt_values),
                "gt_mean": gt_mean,
                "gt_std": normalization_std,
                "gt_variance": normalization_variance,
                **summary,
            }
        )
    return group_summaries


def _summarize_contract_counts(records: list[dict], num_datasets: int) -> dict[str, object]:
    valid_records = [record for record in records if bool(record.get("valid"))]
    scored_records = [record for record in valid_records if not bool(record.get("exploded"))]
    return {
        "num_datasets": num_datasets,
        "num_samples": len(records),
        "num_valid_samples": len(valid_records),
        "num_scored_samples": len(scored_records),
        "num_invalid_samples": len(records) - len(valid_records),
        "num_exploded": sum(bool(record.get("exploded")) for record in valid_records),
        "num_excluded_exploded": len(valid_records) - len(scored_records),
        "num_points": sum(int(record.get("num_points", 0)) for record in scored_records),
        "metrics_aggregated_across_datasets": False,
    }


def _summarize_sample_mean_group(records: list[dict]) -> dict[str, object]:
    valid_records = [record for record in records if bool(record.get("valid"))]
    scored_records = [record for record in valid_records if not bool(record.get("exploded"))]
    count = len(scored_records)
    if count:
        mean = lambda key: float(sum(float(record[key]) for record in scored_records) / count)
        mse, mae, nmse, nmae = (mean(key) for key in ("mse", "mae", "nmse", "nmae"))
    else:
        mse = mae = nmse = nmae = math.inf
    invalid_reasons: dict[str, int] = defaultdict(int)
    for record in records:
        if not bool(record.get("valid")):
            invalid_reasons[str(record.get("metric_error") or "invalid_prediction")] += 1
    return {
        "aggregation": "sample_mean",
        "num_samples": len(records),
        "num_valid_samples": len(valid_records),
        "num_scored_samples": count,
        "num_invalid_samples": len(records) - len(valid_records),
        "invalid_by_reason": dict(sorted(invalid_reasons.items())),
        "num_exploded": sum(bool(record.get("exploded")) for record in valid_records),
        "num_excluded_exploded": len(valid_records) - count,
        "num_points": int(sum(int(record.get("num_points", 0)) for record in scored_records)),
        "squared_error_sum": float(sum(float(record.get("squared_error_sum", 0.0)) for record in scored_records)),
        "absolute_error_sum": float(sum(float(record.get("absolute_error_sum", 0.0)) for record in scored_records)),
        "mse": mse,
        "mae": mae,
        "nmse": nmse,
        "nmae": nmae,
    }
