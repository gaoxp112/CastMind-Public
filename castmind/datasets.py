"""Read externally prepared Parquet examples for VeRL; never construct datasets."""

from collections.abc import Mapping, Sequence
import math
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from verl.utils.dataset.rl_dataset import RLHFDataset

from castmind.contract import parse_response
from castmind.prompt import validate_prompt


def _plain(value):
    """Normalize Parquet array containers without changing their contents."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _messages(value, roles):
    value = _plain(value)
    if not isinstance(value, list) or len(value) != len(roles):
        raise ValueError(f"messages must have exactly these roles: {roles}")
    for message, role in zip(value, roles, strict=True):
        if not isinstance(message, dict) or message.get("role") != role:
            raise ValueError(f"messages must have exactly these roles: {roles}")
        if set(message) != {"role", "content"} or not isinstance(message["content"], str):
            raise ValueError("prepared messages require only role and text content")
    validate_prompt(value[0]["content"])
    return value


def _prior(extra_info):
    extra_info = _plain(extra_info)
    if not isinstance(extra_info, dict):
        raise ValueError("extra_info must contain the displayed prior forecast")
    prior = extra_info.get("prior")
    if not isinstance(prior, list) or not prior:
        raise ValueError("extra_info.prior must be a nonempty numeric list")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in prior):
        raise ValueError("extra_info.prior must contain finite numeric values")
    return prior


def _token_ids(value):
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("chat template did not return token IDs")
    if value and isinstance(value[0], Sequence):
        if len(value) != 1:
            raise ValueError("chat template returned more than one sequence")
        value = value[0]
    return [int(item) for item in value]


def prompt_response_prefix(tokenizer, messages):
    """Identify a thinking prefix supplied by this tokenizer's generation prompt.

    Only the exact opening used by Qwen thinking mode is supported. The flag is
    derived from the tokenizer, rather than trusted from prepared metadata.
    """
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    if not isinstance(rendered, str):
        raise ValueError("chat template must return text with tokenize=False")
    opening = rendered.rfind("<think>")
    closing = rendered.rfind("</think>")
    if opening > closing:
        suffix = rendered[opening:]
        if suffix != "<think>\n":
            raise ValueError("unsupported prompt-owned thinking prefix")
        return suffix
    return ""


def tokenize_sft_messages(tokenizer, messages, max_length, max_response_length=2048):
    """Tokenize once as a complete conversation and mask the exact prompt prefix."""
    prompt_ids = _token_ids(tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=True,
    ))
    input_ids = _token_ids(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, enable_thinking=True,
    ))
    if not prompt_ids or input_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("generation prompt tokens must exactly prefix the full SFT conversation")
    response_length = len(input_ids) - len(prompt_ids)
    if response_length <= 0:
        raise ValueError("SFT response has no trainable tokens")
    if len(input_ids) > max_length:
        raise ValueError(f"SFT sequence has {len(input_ids)} tokens, exceeding {max_length}; truncation is disabled")
    if response_length > max_response_length:
        raise ValueError(f"SFT response has {response_length} tokens, exceeding {max_response_length}; truncation is disabled")
    return {
        "input_ids": input_ids,
        "position_ids": list(range(len(input_ids))),
        "loss_mask": [0] * len(prompt_ids) + [1] * response_length,
    }


class PreparedSFTDataset(Dataset):
    """VeRL SFT custom dataset with response-only loss and no token truncation.

    Every input row contains ``messages`` (one user and one assistant message)
    and ``extra_info.prior``. Data preparation and trajectory collection remain
    external to this release. A multimodal processor is unnecessary for these
    text-only rows; the language tokenizer supplies the chat template.
    """

    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        if config.get("truncation", "error") != "error":
            raise ValueError("CastMind requires truncation=error")
        if config.get("pad_mode", "no_padding") != "no_padding":
            raise ValueError("PreparedSFTDataset requires pad_mode=no_padding")
        if isinstance(parquet_files, (str, Path)):
            parquet_files = [parquet_files]
        if not parquet_files:
            raise ValueError("at least one prepared Parquet file is required")
        self.rows = []
        for filename in parquet_files:
            path = Path(filename).expanduser()
            if path.suffix != ".parquet":
                raise ValueError("only externally prepared .parquet files are accepted")
            self.rows.extend(_plain(row) for row in pd.read_parquet(path).to_dict(orient="records"))
        if max_samples > 0:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise ValueError("SFT input is empty")
        self.tokenizer = tokenizer
        self.max_length = int(config.get("max_length", 17048))
        self.max_response_length = int(config.get("max_response_length", 2048))
        self.messages_key = config.get("messages_key", "messages")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        try:
            messages = _messages(row[self.messages_key], ("user", "assistant"))
            prior = _prior(row.get("extra_info"))
            validate_prompt(messages[0]["content"], prior)
            parse_response(messages[-1]["content"], len(prior), prior)
            tokens = tokenize_sft_messages(
                self.tokenizer, messages, self.max_length, self.max_response_length,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid prepared SFT row {index}: {error}") from error
        return {key: torch.tensor(value, dtype=torch.long) for key, value in tokens.items()}


class PreparedRLHFDataset(RLHFDataset):
    """VeRL's RL dataset with canonical prompts and explicit thinking metadata."""

    def __init__(self, data_files, tokenizer, config, processor=None, max_samples=-1):
        if config.get("truncation", "error") != "error":
            raise ValueError("CastMind requires truncation=error")
        if config.get("filter_overlong_prompts", False):
            raise ValueError("overlong prompts must raise an error rather than be filtered")
        if config.get("apply_chat_template_kwargs", {}).get("enable_thinking") is not True:
            raise ValueError("RL chat template requires enable_thinking=True")
        files = [data_files] if isinstance(data_files, (str, Path)) else data_files
        if not files or any(Path(filename).suffix != ".parquet" for filename in files):
            raise ValueError("only externally prepared .parquet files are accepted")
        super().__init__(data_files, tokenizer, config, processor, max_samples)

    def __getitem__(self, index):
        row = super().__getitem__(index)
        try:
            messages = _messages(row["raw_prompt"], ("user",))
            prior = _prior(row.get("extra_info"))
            validate_prompt(messages[0]["content"], prior)
            prompt_ids = _token_ids(self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=True,
            ))
            if len(prompt_ids) > self.max_prompt_length:
                raise ValueError(f"RL prompt has {len(prompt_ids)} tokens, exceeding {self.max_prompt_length}; truncation is disabled")
            prefix = prompt_response_prefix(self.tokenizer, messages)
            metadata = _plain(row["extra_info"])
            if "response_prefix" in metadata and metadata["response_prefix"] != prefix:
                raise ValueError("extra_info.response_prefix disagrees with the tokenizer")
            metadata["response_prefix"] = prefix
            row["extra_info"] = metadata
            row["raw_prompt"] = messages
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid prepared RL row {index}: {error}") from error
        return row
