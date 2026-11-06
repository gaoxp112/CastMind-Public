"""Join predictions to held-out windows and evaluate with pooled future-GT scales."""

import argparse
import hashlib
import json
from pathlib import Path

from castmind.contract import parse_response
from castmind.dataio import read_rows, write_json
from castmind.metrics import aggregate_sample_metrics


def evaluate(rows, predictions):
    by_id = {}
    for prediction in predictions:
        key = prediction["sample_id"]
        if key in by_id:
            raise ValueError(f"duplicate prediction: {key}")
        by_id[key] = prediction
    expected = {row["sample_id"] for row in rows}
    if len(expected) != len(rows):
        raise ValueError("duplicate evaluation sample IDs")
    if set(by_id) - expected:
        raise ValueError("predictions contain sample IDs outside this evaluation set")
    records = []
    for row in rows:
        key = row["sample_id"]
        output = by_id.get(key, {})
        expected_hash = hashlib.sha256(json.dumps(row["prompt"], sort_keys=True).encode()).hexdigest()
        if output and output.get("prompt_sha256") != expected_hash:
            raise ValueError(f"prediction prompt hash mismatch for {key}")
        target = row["reward_model"]["ground_truth"]
        target = json.loads(target) if isinstance(target, str) else target
        record = {
            "sample_id": key, "evaluation_dataset": row["data_source"],
            "metric_group": f"Datav7::{row['data_source']}::{row['series_id']}",
            "series_id": row["series_id"], "attr": row["target_name"],
            "ground_truth": target, "prediction": [], "answer_contract_valid": False,
        }
        try:
            if not output:
                raise ValueError("missing_prediction")
            parsed = parse_response(output.get("response", ""), len(target), row["extra_info"]["prior"])
            record.update(prediction=list(parsed.values), answer_contract_valid=True)
        except ValueError as error:
            record["parse_error"] = str(error)
        records.append(record)
    if not records:
        raise ValueError("evaluation set is empty")
    summary = aggregate_sample_metrics(records)
    return summary, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    predictions = [json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line.strip()]
    summary, records = evaluate(read_rows(args.data), predictions)
    output = Path(args.output_dir)
    write_json(output / "summary.json", summary)
    write_json(output / "samples.json", records)
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
