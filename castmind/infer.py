"""Generate complete forecasts from prepared evaluation Parquet windows."""

import argparse
import hashlib
import json
from pathlib import Path

from castmind.contract import parse_response
from castmind.dataio import read_rows, write_json
from castmind.prompt import validate_prompt


def load_model(model_path, device="cuda"):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    auto_class = AutoModelForImageTextToText if config.model_type == "qwen3_5" else AutoModelForCausalLM
    model = auto_class.from_pretrained(
        model_path, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        trust_remote_code=False,
    ).to(device)
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, messages, *, max_new_tokens=2048, max_length=17048):
    import torch
    from castmind.sad import _ids

    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    ids = _ids(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=True,
    ))
    if len(ids) + max_new_tokens > max_length:
        raise ValueError("prompt plus completion exceeds max_length; truncation is disabled")
    device = next(model.parameters()).device
    inputs = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        result = model.generate(
            input_ids=inputs, attention_mask=torch.ones_like(inputs),
            do_sample=False, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    # Preserve literal <think>/<answer> tokens while removing only turn terminators.
    continuation = result[0, len(ids):].tolist()
    ending_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    while continuation and continuation[-1] in ending_ids:
        continuation.pop()
    text = tokenizer.decode(continuation, skip_special_tokens=False).strip()
    prefix = "<think>\n" if rendered.endswith("<think>\n") else ""
    return prefix + text


def forecast_rows(rows, generator):
    seen = set()
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        messages = row["prompt"]
        prior = row["extra_info"]["prior"]
        validate_prompt(messages[0]["content"], prior)
        response = generator(messages)
        result = {
            "sample_id": sample_id,
            "prompt_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
            "response": response, "prediction": [], "answer_contract_valid": False,
        }
        try:
            parsed = parse_response(response, len(prior), prior)
            result.update(prediction=list(parsed.values), decision=parsed.decision, answer_contract_valid=True)
        except ValueError as error:
            result["parse_error"] = str(error)
        yield result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    args = p.parse_args()
    model, tokenizer = load_model(args.model, args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w") as stream:
        for result in forecast_rows(read_rows(args.data), lambda m: generate_response(
            model, tokenizer, m, max_new_tokens=args.max_new_tokens,
        )):
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
            count += 1
    write_json(str(output) + ".meta.json", {"model": args.model, "num_samples": count, "decoding": "greedy"})


if __name__ == "__main__":
    main()
