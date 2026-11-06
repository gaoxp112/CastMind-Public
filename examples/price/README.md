# Price example

This example starts with raw observations, prepares trajectories using the paper's four teacher APIs, runs SFT followed by GRPO, exports the final checkpoint, and evaluates held-out forecasts. The default is a small **single-domain demonstration**, using Qwen3.5-0.8B, one CUDA GPU, and 32/8/8 train/validation/test windows. It is not the paper's eight-domain, 2,000-trajectory experiment or a reproduction of its table values. Formal training remains the default `paper` profile.

See [validation scope](VALIDATION.md) for the checks performed and remaining verification limits.

## Raw data

`raw.csv` contains 4,320 consecutive hourly rows from the beginning of the GEFCom2014 Price zone-1 series (2011-01-01 through 2011-06-29). Columns are `series_id`, `timestamp`, `zonal_price`, `forecasted_total_load`, and `forecasted_zonal_load`. The original numerical observations are retained; duplicate competition-task exports were previously consolidated into a unique hourly series. No prompts, trajectories, cached predictions, or Parquet training files are included.

Source: GEFCom2014, described in [Hong et al., *Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond*](https://doi.org/10.1016/j.ijforecast.2016.02.001). Load columns are provided forecasts, so their future values are available at prediction time.

The converter splits the CSV chronologically 60%/20%/20% **before** making windows. Every history and forecast stays wholly inside its split. Defaults match the paper's Price lengths: 96 history steps, 96 forecast steps, stride 24. Earlier windows in each split are used up to the configured example limits. Teacher generation and selection run only on training windows; test targets never enter prompts or checkpoint selection.

## One-command run

Install from the repository root into a CUDA-capable environment. The FlashAttention source build requires a matching CUDA toolkit (`nvcc`); the validated stack uses CUDA 12.8:

```bash
python -m pip install torch==2.10.0 vllm==0.18.1 packaging ninja
MAX_JOBS=4 python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install -e '.[train,example]'
hf download BAAI/bge-m3 --local-dir models/bge-m3
export CASTMIND_ENCODER_PATH="$PWD/models/bge-m3"
cp examples/price/teachers.example.json examples/price/teachers.local.json
```

Edit the local JSON with your accessible model IDs and **complete request URLs** for DeepSeek-V4-Flash, GPT-6.1-Sol, GLM-5.3-Flash and Claude Opus 5.5. Provider aliases differ, so the example deliberately leaves these values for the user to configure. OpenAI-compatible Chat Completions gateways and Anthropic Messages are supported; use `provider: "openai"` for a Claude-compatible Chat Completions gateway. Put keys in the four environment variables named by `api_key_env`, never in the JSON. The local JSON is ignored by Git.

Then run:

```bash
bash scripts/run_price.sh
```

The preparation step makes four forecast requests and one DeepSeek interval annotation per training window (160 requests with the defaults). API use is billable. Responses are cached under the output directory and reused on reruns. API errors and all-invalid teacher sets stop the run; no synthetic or ground-truth forecast is substituted. Models are downloaded on first use unless local paths are supplied. Use an allocated CUDA GPU; CPU is supported for preparation/inference but the distributed training example requires CUDA. GPU memory requirements depend on model size, sequence length and software versions; larger students may need multiple GPUs.

Environment overrides:

```bash
MODEL=/path/to/Qwen3.5-9B GPUS=4 bash scripts/run_price.sh
# Or a bounded training smoke run, with the same preparation and evaluation:
MAX_STEPS=1 TRAIN_WINDOWS=4 VALIDATION_WINDOWS=1 TEST_WINDOWS=1 bash scripts/run_price.sh
```

`PRIOR_MODEL`, `TEACHER_CONFIG`, `OUTPUT_DIR`, `DEVICE`, and `PYTHON_BIN` are also supported. Use a new output directory for a fresh training run. `MAX_STEPS` changes the training budget only in the example profile. The small example batch size is 4; the paper profile retains its original hyperparameters and input checks. Multi-domain quota balancing is outside this single-domain example.

## Individual stages

```bash
python -m castmind.prepare_price convert --csv examples/price/raw.csv --output-dir outputs/price/data
python -m castmind.prepare_price distill --output-dir outputs/price/data \
  --student-model Qwen/Qwen3.5-0.8B --teacher-config examples/price/teachers.local.json
python -m castmind.infer --model /path/to/exported/hf_model \
  --data outputs/price/data/test_windows.parquet --output outputs/price/predictions.jsonl
python -m castmind.evaluate --data outputs/price/data/test_windows.parquet \
  --predictions outputs/price/predictions.jsonl --output-dir outputs/price/evaluation
```

`convert` loads [Chronos-2](https://github.com/amazon-science/chronos-forecasting), provides the two load-forecast covariates, and rounds its point prior to the observed two-decimal price precision. Its outputs are `train_windows.parquet`, `validation_windows.parquet`, `test_windows.parquet`, and `conversion.json`. `distill` uses frozen-student teacher forcing and the paper's MAE/SAD selection rule to write `sft.parquet`, `rl.parquet`, `validation.parquet`, and auditable `selection.json`.

Each window contains a stable `sample_id`, `series_id`, `target_name`, `data_source`, one-user-message `prompt`, `reward_model.ground_truth` (JSON list), and `extra_info`. The latter contains the displayed `prior`, four full-horizon `baselines`, half-open `intervals`, and valid `teachers` with `response`/`forecast`. SFT rows instead contain user/assistant `messages` plus `extra_info.prior`. Future ground truth lives only in offline metadata. Example baselines use last value, a 24-hour recent mean, a repeated 24-hour seasonal copy, and ridge regression (alpha=1, historical-only standardization and fitting) on hourly sine/cosine, weekday and the two load-forecast columns. These are explicit example construction settings, not claims about unspecified paper baseline settings.

Teacher interfaces follow [Chat Completions](https://developers.openai.com/api/reference/resources/chat) or [Anthropic Messages](https://platform.claude.com/docs/en/api/messages). Optional `generation` parameters are merged into the request, including gateway-specific thinking settings. For endpoints requiring `max_completion_tokens`, specify that field in `generation`; it replaces `max_tokens`. Raise the token budget if reasoning consumes tokens before the full forecast is emitted. The visible response must satisfy the repository's `analysis → decision → forecast_method → answer` contract. Future targets are never sent to either the four teachers or the interval annotator.

Evaluation re-parses the full response rather than trusting a reported numeric prediction. It joins by exact sample ID and checks prompt hashes. Missing, malformed, wrong-horizon and non-finite forecasts remain in the evaluation report. Future ground truth from **every** test window determines the pooled population standard deviation/variance per dataset–series group. Per-window nMAE/nMSE receive equal weight; finite predictions exceeding ten pooled standard deviations from the pooled mean are retained and flagged as exploded, but excluded from headline means. `summary.json` reports validity counts and normalization groups; `samples.json` preserves per-sample scales, errors and parse failures. These test metrics are distinct from the per-window training reward scale.
