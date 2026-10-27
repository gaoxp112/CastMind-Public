# CastMind

CastMind is an evidence-grounded framework for context-aware time series forecasting. It analyzes observed history and available future covariates, then chooses to retain, revise, or reconstruct a forecast prior to produce the final forecast. Training consists of supervised fine-tuning (SFT) followed by GRPO reinforcement learning.

## Environment setup

Use Python 3.10 or newer with CUDA-enabled PyTorch. The default training configuration targets four H100 80GB GPUs.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train]'
```

Training uses VeRL and vLLM. Cache the required text encoder before running GRPO:

```bash
hf download BAAI/bge-m3 --local-dir models/bge-m3
export CASTMIND_ENCODER_PATH="$PWD/models/bge-m3"
```

## Running CastMind

Provide your own model and prepared Parquet training files compatible with [the dataset adapters](castmind/datasets.py). Data and model weights are not included. The main model is Qwen3.5-9B; Qwen3.5-4B uses the same commands with a different model path.

**1. Supervised fine-tuning**

```bash
python -m castmind.train sft \
  --model Qwen/Qwen3.5-9B \
  --train-file data/sft.parquet \
  --output-dir outputs/sft
```

**2. GRPO reinforcement learning**

Set `SFT_HF_MODEL` to the final SFT checkpoint exported in Hugging Face format, including its tokenizer. Supply a separate, nonempty validation file.

```bash
export SFT_HF_MODEL=/path/to/final/sft/hf_model

python -m castmind.train grpo \
  --model "$SFT_HF_MODEL" \
  --train-file data/rl.parquet \
  --val-file data/validation.parquet \
  --output-dir outputs/grpo
```

Add `--check-only` to validate inputs before training, or `--dry-run` to inspect the configuration. Stage settings are in [castmind/configs](castmind/configs), and the shared forecasting prompt is in [castmind/prompt.txt](castmind/prompt.txt).
