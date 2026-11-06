"""API payload/credential boundaries without network calls or paid requests."""

import io
import json

import pytest

from castmind.teacher_api import complete, load_config


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_provider_messages_and_credentials(monkeypatch, provider):
    import castmind.teacher_api as api
    monkeypatch.setenv("TEST_TEACHER_KEY", "test-only-value")
    messages = [{"role": "user", "content": "forecast-time inputs only"}]
    teacher = {"name": "test", "model": "configured-model", "provider": provider,
               "url": "https://example.invalid/api", "api_key_env": "TEST_TEACHER_KEY",
               "generation": {"max_completion_tokens": 4096} if provider == "openai" else {}}

    def urlopen(request, timeout):
        payload = json.loads(request.data)
        assert payload["messages"] == messages
        assert payload["model"] == "configured-model"
        assert "test-only-value" not in request.data.decode()
        if provider == "openai":
            assert payload["max_completion_tokens"] == 4096 and "max_tokens" not in payload
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": "response"}}]}).encode())
        assert payload["max_tokens"] == 2048
        return io.BytesIO(json.dumps({"content": [{"type": "text", "text": "response"}]}).encode())

    monkeypatch.setattr(api, "urlopen", urlopen)
    assert complete(teacher, messages) == "response"


def test_configuration_requires_four_teachers_and_env_keys(tmp_path, monkeypatch):
    teachers = [{"name": str(i), "model": str(i), "url": "https://example.invalid/api", "api_key_env": "TEST_TEACHER_KEY"} for i in range(4)]
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"teachers": teachers, "interval_teacher": "0"}))
    monkeypatch.delenv("TEST_TEACHER_KEY", raising=False)
    with pytest.raises(ValueError, match="environment variable"):
        load_config(path)
    monkeypatch.setenv("TEST_TEACHER_KEY", "test-only-value")
    assert len(load_config(path)["teachers"]) == 4
    path.write_text(json.dumps({"teachers": teachers[:1], "interval_teacher": "0"}))
    with pytest.raises(ValueError, match="four"):
        load_config(path)
