"""Prepared data boundaries, response masking, and Qwen thinking compatibility."""

import os
from pathlib import Path
import tempfile
import unittest

from omegaconf import OmegaConf
import pandas as pd

from castmind.datasets import (
    PreparedRLHFDataset,
    PreparedSFTDataset,
    prompt_response_prefix,
    tokenize_sft_messages,
)
from castmind.prompt import render_prompt


def prompt():
    return render_prompt(
        task_definition="Forecast the next two hourly demand values.",
        context_information="Observed demand and calendar covariates.",
        observed_rows="index | time | target\n0 | 2025-01-01 00:00 | 1.0",
        future_context_information="The next two hours are a holiday.",
        forecast_rows="index | time | prior_forecast\n0 | 2025-01-01 01:00 | 1.0\n1 | 2025-01-01 02:00 | 2.0",
    )


RESPONSE = (
    "<think>\nanalysis: The holiday context supports retaining the prior forecast.\n"
    "decision: retain\nforecast_method: none\n</think>\n<answer>\n```\n"
    "index | value\n0 | 1.0\n1 | 2.0\n```\n</answer>"
)


class ThinkingTokenizer:
    """Minimal tokenizer with Qwen's prompt-owned opening thinking block."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        if enable_thinking is not True:
            raise AssertionError("thinking mode must be explicit")
        text = "".join(
            f"<|im_start|>{item['role']}\n{item['content']}<|im_end|>\n"
            for item in messages
        )
        if add_generation_prompt:
            text += "<|im_start|>assistant\n<think>\n"
        return [ord(character) for character in text] if tokenize else text


class PreparedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.tokenizer = ThinkingTokenizer()
        self.messages = [{"role": "user", "content": prompt()}, {"role": "assistant", "content": RESPONSE}]

    def parquet(self, rows):
        filename = Path(self.directory.name) / "prepared.parquet"
        pd.DataFrame(rows).to_parquet(filename, index=False)
        return str(filename)

    def test_sft_masks_prompt_without_duplicating_thinking_block(self):
        filename = self.parquet([{"messages": self.messages, "extra_info": {"prior": [1.0, 2.0]}}])
        dataset = PreparedSFTDataset(filename, self.tokenizer, {"max_length": 20000})
        row = dataset[0]
        prompt_length = len(self.tokenizer.apply_chat_template(
            self.messages[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=True,
        ))
        self.assertFalse(row["loss_mask"][:prompt_length].any())
        self.assertTrue(row["loss_mask"][prompt_length:].all())
        continuation = "".join(chr(token) for token in row["input_ids"][prompt_length:].tolist())
        self.assertTrue(continuation.startswith("analysis:"))
        self.assertEqual(row["input_ids"].numel(), row["loss_mask"].numel())

    def test_sft_rejects_truncation_and_overlong_responses(self):
        filename = self.parquet([{"messages": self.messages, "extra_info": {"prior": [1.0, 2.0]}}])
        with self.assertRaisesRegex(ValueError, "truncation=error"):
            PreparedSFTDataset(filename, self.tokenizer, {"truncation": "right"})
        dataset = PreparedSFTDataset(filename, self.tokenizer, {"max_length": 10})
        with self.assertRaisesRegex(ValueError, "truncation is disabled"):
            dataset[0]
        dataset = PreparedSFTDataset(filename, self.tokenizer, {"max_length": 20000, "max_response_length": 4})
        with self.assertRaisesRegex(ValueError, "response has"):
            dataset[0]

    def test_sft_rejects_changed_instruction_and_obsolete_response(self):
        for messages in (
            [{"role": "user", "content": "Forecast the future."}, self.messages[1]],
            [self.messages[0], {"role": "assistant", "content": RESPONSE.replace("analysis:", "reason:")}],
        ):
            filename = self.parquet([{"messages": messages, "extra_info": {"prior": [1.0, 2.0]}}])
            with self.assertRaises(ValueError):
                PreparedSFTDataset(filename, self.tokenizer, {"max_length": 20000})[0]

    def test_sft_requires_exact_chat_template_prefix(self):
        class IncompatibleTokenizer(ThinkingTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                result = super().apply_chat_template(messages, **kwargs)
                return result + [999] if kwargs["tokenize"] and kwargs["add_generation_prompt"] else result

        with self.assertRaisesRegex(ValueError, "exactly prefix"):
            tokenize_sft_messages(IncompatibleTokenizer(), self.messages, 20000)

    def rl_dataset(self, metadata=None, max_length=20000):
        filename = self.parquet([{
            "prompt": self.messages[:-1],
            "data_source": "test-domain",
            "reward_model": {"ground_truth": "[1.0, 2.0]", "style": "rule"},
            "extra_info": {"prior": [1.0, 2.0], **(metadata or {})},
        }])
        config = OmegaConf.create({
            "max_prompt_length": max_length,
            "truncation": "error",
            "filter_overlong_prompts": False,
            "apply_chat_template_kwargs": {"enable_thinking": True},
        })
        return PreparedRLHFDataset(filename, self.tokenizer, config)

    def test_rl_derives_thinking_prefix_from_tokenizer(self):
        row = self.rl_dataset()[0]
        self.assertEqual(row["extra_info"]["response_prefix"], "<think>\n")
        self.assertEqual(row["raw_prompt"], self.messages[:-1])
        self.assertEqual(row["reward_model"]["ground_truth"], "[1.0, 2.0]")

    def test_rl_rejects_conflicting_prefix_and_overlong_prompt(self):
        with self.assertRaisesRegex(ValueError, "disagrees with the tokenizer"):
            self.rl_dataset({"response_prefix": ""})[0]
        with self.assertRaisesRegex(ValueError, "truncation is disabled"):
            self.rl_dataset(max_length=4)[0]

    def test_unknown_unclosed_thinking_prefix_is_rejected(self):
        class UnsupportedTokenizer(ThinkingTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return super().apply_chat_template(messages, **kwargs) + "analysis:"

        with self.assertRaisesRegex(ValueError, "unsupported"):
            prompt_response_prefix(UnsupportedTokenizer(), self.messages[:-1])

    @unittest.skipUnless(os.environ.get("CASTMIND_TEST_TOKENIZER"), "set CASTMIND_TEST_TOKENIZER for a real tokenizer")
    def test_real_qwen_thinking_template(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(os.environ["CASTMIND_TEST_TOKENIZER"], local_files_only=True)
        row = tokenize_sft_messages(tokenizer, self.messages, 20000)
        prefix = prompt_response_prefix(tokenizer, self.messages[:-1])
        self.assertEqual(prefix, "<think>\n")
        continuation = tokenizer.decode([token for token, mask in zip(row["input_ids"], row["loss_mask"], strict=True) if mask])
        self.assertTrue(continuation.startswith("analysis:"))
        self.assertIn("forecast_method: none", continuation)


if __name__ == "__main__":
    unittest.main()
