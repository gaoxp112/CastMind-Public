# Validation scope

Checks performed on 2026-11-06:

- 71 tests passed with the pinned VeRL commit and stable Transformers 5.3.0. Coverage includes the actual Qwen3.5 tokenizer, SAD token boundaries, four-teacher preparation and caching with simulated HTTP responses, future-target isolation, reward loading, and evaluation validity/normalization.
- Raw Price conversion ran with real Chronos-2 weights. Frozen Qwen3.5-0.8B teacher forcing produced a finite SAD score.
- A single H100 GPU slice with 40 GB memory ran one SFT optimizer step, exported its Hugging Face checkpoint, ran one GRPO step, merged its actor checkpoint, then executed standalone inference and evaluation. This GPU check used PyTorch 2.10.0+cu128, Transformers 5.3.0.dev0, vLLM 0.18.1, FlashAttention 2.8.3 and Flash Linear Attention 0.5.2. GPU execution with stable Transformers 5.3.0 has not been independently repeated.
- Wheel/source-distribution builds and installed inference, preparation and evaluation CLI imports passed.

The GPU smoke check used test-only retain-prior trajectories, not API-generated training data. SFT had a nonzero gradient; all GRPO rollouts were invalid, so their equal rewards produced zero advantages and a zero policy gradient. Standalone generation was deliberately limited to 32 new tokens and was correctly counted as invalid by evaluation. These checks establish execution of the interfaces and checkpoint path, not forecast quality, useful GRPO learning, or reproduction of the paper's results. Complete valid forecasts and metric values are covered separately by deterministic tests.

No paid teacher calls were made during validation, and the BGE-M3 embedding path was not exercised with real encoder weights. Users must supply working teacher endpoints, model IDs, API keys and encoder weights before running the complete example. A fresh environment installation of the entire CUDA dependency stack has not been tested.
