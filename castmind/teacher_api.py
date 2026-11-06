"""Configurable teacher APIs. Credentials are read only from environment variables."""

import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def load_config(filename):
    with open(filename) as stream:
        config = json.load(stream)
    teachers = config.get("teachers", [])
    if len(teachers) != 4 or len({t["name"] for t in teachers}) != 4:
        raise ValueError("configure exactly four uniquely named paper teachers")
    for teacher in teachers:
        for key in ("name", "model", "url", "api_key_env"):
            if not teacher.get(key) or "YOUR_" in teacher[key]:
                raise ValueError(f"configure teacher field {key}")
        if not teacher["url"].startswith("https://"):
            raise ValueError("teacher API URL must use HTTPS")
        if teacher.get("provider", "openai") not in {"openai", "anthropic"}:
            raise ValueError("provider must be openai or anthropic")
        if not os.environ.get(teacher["api_key_env"]):
            raise ValueError(f"set environment variable {teacher['api_key_env']}")
    if config.get("interval_teacher") not in {t["name"] for t in teachers}:
        raise ValueError("interval_teacher must name one of the four teachers")
    return config


def complete(teacher, messages, *, max_tokens=2048):
    provider = teacher.get("provider", "openai")
    body = {"model": teacher["model"], "messages": messages, "max_tokens": max_tokens, "temperature": 1.0}
    overrides = dict(teacher.get("generation", {}))
    # A gateway may require max_completion_tokens rather than max_tokens.
    if "max_completion_tokens" in overrides:
        body.pop("max_tokens")
    body.update(overrides)
    if body.get("model") != teacher["model"] or body.get("messages") != messages:
        raise ValueError("generation overrides must not change model or messages")
    headers = {"Content-Type": "application/json"}
    key = os.environ[teacher["api_key_env"]]
    if provider == "anthropic":
        headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
    else:
        headers["Authorization"] = "Bearer " + key
    request = Request(teacher["url"], data=json.dumps(body).encode(), headers=headers)
    for attempt in range(3):
        try:
            with urlopen(request, timeout=teacher.get("timeout", 300)) as response:
                result = json.load(response)
            break
        except HTTPError as error:
            if (error.code != 429 and error.code < 500) or attempt == 2:
                raise RuntimeError(f"teacher {teacher['name']} returned HTTP {error.code}") from None
            time.sleep(2 ** attempt)
        except URLError:
            if attempt == 2:
                raise RuntimeError(f"teacher {teacher['name']} connection failed") from None
            time.sleep(2 ** attempt)
    if provider == "anthropic":
        text = "".join(part.get("text", "") for part in result.get("content", []) if part.get("type") == "text")
    else:
        message = result["choices"][0]["message"]
        text = message.get("content") or ""
        # Some thinking APIs split the visible reasoning from final text.
        if "<think>" not in text and message.get("reasoning_content"):
            text = "<think>\n" + message["reasoning_content"].strip() + "\n</think>\n" + text
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"teacher {teacher['name']} returned no text")
    return text.strip()
