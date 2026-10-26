# CastMind

Core training code for CastMind, an evidence-grounded model for context-aware time series forecasting. The model analyzes the observed history, available context, and a cached Chronos-2 forecast prior, chooses **retain**, **revise**, or **reconstruct**, and returns a complete forecast.

This initial code release contains supervised fine-tuning (SFT), group relative policy optimization (GRPO), the shared prompt and response contract, and the five reward terms used in the paper. Training consumes externally prepared examples. Data, model weights, data construction, prior generation, teacher trajectory generation, and trajectory selection pipelines are excluded.

## Installation

Use Python 3.10 or newer and a CUDA environment suitable for four 80GB H100 GPUs. The training adapter targets public upstream [VeRL](https://github.com/verl-project/verl/tree/061053139d598cfc172fac10db610d91a61be3e7), pinned at commit `061053139d598cfc172fac10db610d91a61be3e7`, with vLLM for rollouts:

```bash
python -m pip install -e '.[train]'
```

The optional training dependencies are public package requirements, not a validated GPU lockfile. End-to-end training with a clean upstream VeRL installation has not yet been verified on GPUs. Select CUDA builds of PyTorch, vLLM, and the attention kernels for the target machine; do not use VeRL's older `vllm` extra, whose version cap predates the required Qwen3.5 stack. The release's CPU checks cover response parsing, reward arithmetic, prepared-data validation, and training configuration. No model weights or training data are included in this repository.

The reasoning preference reward uses the frozen [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) encoder on CPU. Cache it before training, or point to an existing local copy:

```bash
hf download BAAI/bge-m3 --local-dir models/bge-m3
export CASTMIND_ENCODER_PATH=models/bge-m3
```

The reward loads this encoder from local files when an eligible teacher comparison is needed. A missing encoder raises an error; the reasoning reward is not silently disabled.

## Training

The main experiments use **Qwen3.5-9B**; auxiliary experiments use **Qwen3.5-4B**. Both stages use four GPUs. Supply the base model and private prepared Parquet files:

```bash
python -m castmind.train sft \
  --model Qwen/Qwen3.5-9B \
  --train-file data/sft.parquet \
  --output-dir outputs/sft
```

GRPO starts from the final SFT checkpoint exported in Hugging Face format, including its tokenizer. Supply its actual directory as `SFT_HF_MODEL`:

```bash
python -m castmind.train grpo \
  --model "$SFT_HF_MODEL" \
  --train-file data/rl.parquet \
  --val-file data/validation.parquet \
  --output-dir outputs/grpo
```

A separate, nonempty validation file is required by upstream VeRL's dataset initialization. It must be disjoint from training; teacher metadata is optional for validation. The provided configuration disables periodic validation. Add `--dry-run` to either command to inspect the resolved configuration without reading data or loading a tokenizer. Use `--check-only` to validate prepared data and tokenizer boundaries without starting training.

The shared instruction is [castmind/prompt.txt](castmind/prompt.txt). Prepared prompts must use it exactly, filling the task, context, observed rows, future context, and forecast rows. The supplied prior must match the values displayed in those forecast rows. Future target values belong only in reward metadata and must never appear in the model input.

The response contains exactly two blocks and no surrounding text. Its three reasoning fields occur in this order:

````text
<think>
analysis: [evidence analysis of the supplied context and forecast prior]
decision: [retain, revise, or reconstruct]
forecast_method: [none for retain; supported prior corrections for revise; independent construction for reconstruct]
</think>
<answer>
```
index | value
0 | [finite numeric value]
...
```
</answer>
````

The brackets and ellipsis above describe the schema: actual responses must replace them with completed text and every forecast row. The answer must contain exactly `H` finite values with consecutive indices `0` through `H-1`. Finite negative predictions are allowed, and decimal places are not fixed. `retain` requires `forecast_method: none` and exact numerical agreement with the displayed prior without rounding or a tolerance. `revise` uses the prior as its starting forecast; `reconstruct` independently constructs a forecast from history and supplied covariates. The parser rejects the obsolete `reason` field and a decision placed before analysis.

Qwen thinking mode is explicit in both stages. The dataset adapter handles a `<think>` prefix supplied by the tokenizer, masks prompt tokens during SFT, and rejects overlong examples instead of truncating or filtering them.

## Prepared input schema

The release reads `.parquet` files only. Each stage's training file contains **2,000** prepared examples across the same eight datasets: Load, Price, Solar, Wind, BAQ, MetroPT-3, HS, and BS. Every row includes a `data_source` dataset label. Keep the same labels and fixed per-dataset quotas across SFT and GRPO.

The paper derives quotas from an external equally sampled pilot. If `T_i` is dataset `i`'s best-validation step under `0.9 nMAE + 0.1 nMSE`, allocate `p_i = sqrt(T_i) / sum_j sqrt(T_j)` under budget `B = 2000`, resolve integer quotas before training, and keep them fixed for both stages. Training shuffles the prepared rows with seed 42. Pilot training and quota construction are outside this release.

### SFT rows

| Column | Required content |
| --- | --- |
| `data_source` | Dataset label used to report the domain counts. |
| `messages` | Exactly two mappings: `{"role": "user", "content": PROMPT}` followed by `{"role": "assistant", "content": RESPONSE}`. Each mapping contains only `role` and `content`. |
| `extra_info` | Mapping containing `prior`: the complete list of `H` finite numbers displayed in the prompt. |

`PROMPT` is the filled canonical instruction; `RESPONSE` is a complete, valid response in the schema above. SFT computes loss only on the assistant continuation. It requires prepared trajectories, without generating or selecting them.

### GRPO training and validation rows

| Column | Required content |
| --- | --- |
| `data_source` | Dataset label, matching the SFT labels for training. |
| `prompt` | Exactly one `{"role": "user", "content": PROMPT}` mapping in a list. |
| `reward_model` | Mapping with `style: "rule"` and `ground_truth`: a JSON-encoded list of the `H` finite future target values. |
| `extra_info.prior` | List of the `H` finite displayed prior values. |
| `extra_info.baselines` | Mapping with exactly `last_value`, `recent_mean`, `seasonal_copy`, and `ridge`; each value is a complete finite forecast of length `H`. |
| `extra_info.intervals` | Optional list of zero-based half-open intervals `[start, end)`, with `0 <= start < end <= H`. Use an empty list when no interval is annotated. |
| `extra_info.teachers` | Prepared teacher entries, each containing a complete `response` string in the same response schema and its matching length-`H` numerical `forecast` list. Teacher entries may be absent for validation. |

Baseline forecasts must be constructed without the evaluated future targets. Interval annotations must use only information available in the prompt; their generation is external. Scoring preserves the supplied intervals and weights them by their individual lengths, including when intervals overlap; the outside penalty uses the complement of their union. Teacher entries whose responses are invalid are ineligible; the stored forecast must match the response. Only teachers whose forecast MAE is strictly lower than the prior's MAE can receive a similarity comparison. Include the prepared teacher references for the full training objective; no teacher-generation service is called during training.

Keep observed/future windows indexed independently from zero. Time uses `YYYY-MM-DD HH:MM`. For Price, Solar, and Wind, the paper uses the supplied future exogenous forecasts. For the other datasets, it excludes future observed exogenous values and uses historical observations plus known future calendar features. The history, horizon, and stride below are sampling steps:

| Dataset | History | Horizon | Stride |
| --- | ---: | ---: | ---: |
| Load, Price, Solar, Wind | 96 | 96 | 24 |
| BAQ | 72 | 24 | 24 |
| MetroPT-3 | 120 | 30 | 30 |
| HS | 30 | 7 | 7 |
| BS | 168 | 24 | 24 |

## Paper settings

Configuration files ship in [castmind/configs](castmind/configs).

| Setting | SFT | GRPO |
| --- | --- | --- |
| Epochs | 3 | 2 |
| Learning rate | `2e-5` | `1e-6` |
| Global input batch | 32 | 16 |
| Prepared training examples | 2,000 | 2,000 |
| GPUs | 4 × H100 80GB | 4 × H100 80GB |
| Seed | 42 | 42 |
| Rollouts per input | — | 8 |
| Sampling temperature | — | 1.0 |
| Maximum completion length | 2,048 tokens (runtime limit) | 2,048 tokens |
| Policy clipping | — | 0.2 |
| KL coefficient | — | 0.001 |

Preflight checks the 2,000-row training budget and eight domain labels and reports their counts. The caller must ensure that SFT and GRPO use the same fixed per-dataset quotas and that validation is disjoint; the stages do not compare files automatically. Native VeRL drops the incomplete SFT batch, producing 62 updates per epoch (186 in total) with these settings. GRPO uses 125 updates per epoch (250 in total).

Device-local batch sizes, FSDP options, prompt length limits, and vLLM memory limits are runtime choices recorded in the configurations. The default prompt allowance is 15,000 tokens and total sequence allowance is 17,048 tokens. These limits do not authorize changing the prompt or truncating the complete forecast.

### Reward terms

[castmind/rewards.py](castmind/rewards.py) implements the paper's five terms. Invalid responses receive `-1`; all other terms are gated off. Valid responses receive **zero format reward**, plus:

| Term | Definition and fixed parameters |
| --- | --- |
| Forecast accuracy | Negative clipped loss. Scale `s = max(population_std(y), 0.05 mean(abs(y)), 1e-8)`; loss weights are `0.9 nMAE + 0.1 nMSE`. Clip each normalized error component at `4` before weighting. |
| Simple baseline outperformance | Use the minimum clipped loss among the four prepared baselines. For absolute gain `delta` and relative gain `g = delta / max(reference_loss, 0.10)`, reward `0.10 * min(clip((g - 0.05) / 0.20, 0, 1), clip((delta - 0.02) / 0.08, 0, 1))`. |
| Interval-level prior revision | Interval-length-weighted improvement over the prior, using the **unclipped** loss and clipping each improvement to `[-1, 1]`, weighted by `0.20`. Subtract `0.10` times the positive loss increase outside the intervals, clipped at `1`. This term is zero when intervals are absent. |
| Teacher-guided reasoning preference | Maximum frozen BGE-M3 cosine similarity `sim` to eligible teacher reasoning, giving `0.02 * clip((sim - 0.70) / (1 - 0.70), 0, 1)`. Requires valid reasoning and student **unclipped** loss no worse than the prior. |

The reasoning encoder compares the complete ordered `analysis`, `decision`, and `forecast_method` content. The numerical answer is excluded from its text embedding. Missing eligible references give no reasoning bonus; model-loading or encoding failures propagate as errors.

The training reward's per-example scale differs from the paper's reported test metrics. Reported nMAE and nMSE pool future ground truth across all test windows within each dataset–series group, use its population standard deviation/variance, and average normalized window errors equally. This core release does not include the evaluation pipeline or experimental results.

## Checks

```bash
python -m pip install -e '.[train,dev]'
python -m pytest -q
```

An optional real-tokenizer check uses a locally cached Qwen3.5 tokenizer:

```bash
CASTMIND_TEST_TOKENIZER=models/Qwen3.5-9B \
  python -m pytest -q
```

These checks do not launch a GPU training job or establish end-to-end GPU compatibility.
