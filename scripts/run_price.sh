#!/usr/bin/env bash
# Run from any directory after installing the repository and setting API keys.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python_bin=${PYTHON_BIN:-python}
model=${MODEL:-Qwen/Qwen3.5-0.8B}
prior_model=${PRIOR_MODEL:-amazon/chronos-2}
teacher_config=${TEACHER_CONFIG:-examples/price/teachers.local.json}
output=${OUTPUT_DIR:-outputs/price}
device=${DEVICE:-cuda}
gpus=${GPUS:-1}

# Validate API configuration before downloading models or computing priors.
"$python_bin" -c 'import sys; from castmind.teacher_api import load_config; load_config(sys.argv[1])' "$teacher_config"
"$python_bin" -m castmind.prepare_price convert --csv examples/price/raw.csv \
  --output-dir "$output/data" --prior-model "$prior_model" --device "$device" \
  --train-windows "${TRAIN_WINDOWS:-32}" --validation-windows "${VALIDATION_WINDOWS:-8}" --test-windows "${TEST_WINDOWS:-8}"
"$python_bin" -m castmind.prepare_price distill --output-dir "$output/data" \
  --student-model "$model" --teacher-config "$teacher_config" --device "$device"

extra_args=()
if [[ -n "${MAX_STEPS:-}" ]]; then extra_args+=(--max-steps "$MAX_STEPS"); fi
"$python_bin" -m castmind.train sft --profile price-example --gpus "$gpus" \
  --model "$model" --train-file "$output/data/sft.parquet" --output-dir "$output/sft" "${extra_args[@]}"
sft_model=$("$python_bin" -m castmind.export_checkpoint --stage sft --directory "$output/sft")
"$python_bin" -m castmind.train grpo --profile price-example --gpus "$gpus" \
  --model "$sft_model" --train-file "$output/data/rl.parquet" \
  --val-file "$output/data/validation.parquet" --output-dir "$output/grpo" "${extra_args[@]}"
rl_model=$("$python_bin" -m castmind.export_checkpoint --stage grpo --directory "$output/grpo")
"$python_bin" -m castmind.infer --model "$rl_model" --device "$device" \
  --data "$output/data/test_windows.parquet" --output "$output/predictions.jsonl"
"$python_bin" -m castmind.evaluate --data "$output/data/test_windows.parquet" \
  --predictions "$output/predictions.jsonl" --output-dir "$output/evaluation"
