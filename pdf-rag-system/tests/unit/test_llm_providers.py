from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import httpx2
import openai
import pytest

from app.core.exceptions import ConfigurationError, LLMError, LLMRefusalError
from app.core.retry import RetryPolicy
from app.llm.claude_provider import ClaudeProvider
from app.llm.factory import create_llm_provider
from app.llm.openai_provider import OpenAIProvider
from app.llm.prompts import NOT_FOUND_ANSWER, SYSTEM_PROMPT, build_user_prompt

NO_WAIT = RetryPolicy(max_attempts=3, base_delay=0, max_delay=0, jitter=False)


def _http(status: int, url: str = "https://api.anthropic.com/v1/messages") -> httpx2.Response:
    return httpx2.Response(status, request=httpx2.Request("POST", url))


def claude_message(*blocks, stop_reason="end_turn", model="claude-opus-5-5", input_tokens=1200, output_tokens=250):
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        stop_details=None,
        model=model,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def openai_completion(content="Answer [a.pdf, p. 1]", finish_reason="stop", refusal=None):
    return SimpleNamespace(
        model="gpt-4.1-mini",
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, refusal=refusal), finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=900, completion_tokens=120),
    )


# --------------------------------------------------------------------------
# Provider selection
# --------------------------------------------------------------------------


def test_factory_selects_claude(make_settings) -> None:
    provider = create_llm_provider(make_settings(llm_provider="claude"))
    assert isinstance(provider, ClaudeProvider) and provider.name == "claude" and provider.model == "claude-opus-5-5"


def test_factory_selects_openai(make_settings) -> None:
    provider = create_llm_provider(make_settings(llm_provider="openai"))
    assert isinstance(provider, OpenAIProvider) and provider.name == "openai" and provider.model == "gpt-4.1-mini"


def test_factory_accepts_anthropic_alias(make_settings) -> None:
    assert isinstance(create_llm_provider(make_settings(llm_provider="Anthropic")), ClaudeProvider)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"llm_provider": "claude", "anthropic_api_key": None}, "ANTHROPIC_API_KEY"),
        ({"llm_provider": "claude", "anthropic_model": None}, "ANTHROPIC_MODEL"),
        ({"llm_provider": "openai", "openai_api_key": None}, "OPENAI_API_KEY"),
        ({"llm_provider": "openai", "openai_model": None}, "OPENAI_MODEL"),
    ],
)
def test_factory_requires_key_and_model(make_settings, overrides, message) -> None:
    with pytest.raises(ConfigurationError, match=message):
        create_llm_provider(make_settings(**overrides))


def test_embedding_settings_are_independent_of_llm_provider(make_settings) -> None:
    settings = make_settings(llm_provider="claude")
    assert settings.embedding_provider == "openai" and settings.embedding_model == "text-embedding-3-small"


# --------------------------------------------------------------------------
# Claude
# --------------------------------------------------------------------------


def make_claude(client, **kw) -> ClaudeProvider:
    return ClaudeProvider(api_key=None, model="claude-opus-5-5", max_output_tokens=16000, timeout=30, retry_policy=NO_WAIT, client=client, **kw)


def test_claude_request_and_response_mapping() -> None:
    client = MagicMock()
    client.beta.messages.create.return_value = claude_message(
        SimpleNamespace(type="thinking", thinking=""), text_block("The system has two servers "), text_block("[a.pdf, p. 2].")
    )
    provider = make_claude(client, effort="medium")

    response = provider.generate(system_prompt="SYS", user_prompt="USER")

    assert response.text == "The system has two servers [a.pdf, p. 2]."
    assert (response.provider, response.model, response.input_tokens, response.output_tokens) == ("claude", "claude-opus-5-5", 1200, 250)
    kwargs = client.beta.messages.create.call_args.kwargs
    assert kwargs["system"] == "SYS"
    assert kwargs["messages"] == [{"role": "user", "content": "USER"}]
    assert kwargs["max_tokens"] == 16000
    assert kwargs["output_config"] == {"effort": "medium"}
    assert kwargs["fallbacks"] == "default" and kwargs["betas"] == ["server-side-fallback-2026-07-01"]
    assert "temperature" not in kwargs


def test_claude_without_fallback_uses_standard_endpoint() -> None:
    client = MagicMock()
    client.messages.create.return_value = claude_message(text_block("ok"))
    make_claude(client, refusal_fallback=False).generate(system_prompt="s", user_prompt="u")
    kwargs = client.messages.create.call_args.kwargs
    assert "betas" not in kwargs and "fallbacks" not in kwargs
    client.beta.messages.create.assert_not_called()


def test_claude_refusal_raises() -> None:
    client = MagicMock()
    client.beta.messages.create.return_value = claude_message(stop_reason="refusal")
    with pytest.raises(LLMRefusalError):
        make_claude(client).generate(system_prompt="s", user_prompt="u")


def test_claude_rate_limit_is_retried() -> None:
    client = MagicMock()
    client.beta.messages.create.side_effect = [
        anthropic.RateLimitError("rate limited", response=_http(429), body=None),
        anthropic.InternalServerError("overloaded", response=_http(529), body=None),
        claude_message(text_block("recovered")),
    ]
    assert make_claude(client).generate(system_prompt="s", user_prompt="u").text == "recovered"
    assert client.beta.messages.create.call_count == 3


def test_claude_auth_error_is_not_retried() -> None:
    client = MagicMock()
    client.beta.messages.create.side_effect = anthropic.AuthenticationError("bad key", response=_http(401), body=None)
    with pytest.raises(LLMError) as info:
        make_claude(client).generate(system_prompt="s", user_prompt="u")
    assert not info.value.retryable
    assert client.beta.messages.create.call_count == 1


def test_claude_connection_error_is_retryable() -> None:
    client = MagicMock()
    client.beta.messages.create.side_effect = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com"))
    with pytest.raises(LLMError) as info:
        make_claude(client).generate(system_prompt="s", user_prompt="u")
    assert info.value.retryable and client.beta.messages.create.call_count == NO_WAIT.max_attempts


# --------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------


def make_openai(client, **kw) -> OpenAIProvider:
    return OpenAIProvider(api_key=None, model="gpt-4.1-mini", max_output_tokens=2048, timeout=30, retry_policy=NO_WAIT, client=client, **kw)


def test_openai_request_and_response_mapping() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = openai_completion()
    response = make_openai(client).generate(system_prompt="SYS", user_prompt="USER")

    assert (response.text, response.provider, response.model) == ("Answer [a.pdf, p. 1]", "openai", "gpt-4.1-mini")
    assert (response.input_tokens, response.output_tokens) == (900, 120)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["messages"][0] == {"role": "system", "content": "SYS"}
    assert kwargs["messages"][1] == {"role": "user", "content": "USER"}
    assert kwargs["max_completion_tokens"] == 2048
    assert "temperature" not in kwargs


def test_openai_temperature_sent_only_when_configured() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = openai_completion()
    make_openai(client, temperature=0.0).generate(system_prompt="s", user_prompt="u")
    assert client.chat.completions.create.call_args.kwargs["temperature"] == 0.0


def test_openai_refusal_raises() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = openai_completion(content=None, refusal="I can't help with that")
    with pytest.raises(LLMRefusalError):
        make_openai(client).generate(system_prompt="s", user_prompt="u")


def test_openai_rate_limit_retried() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        openai.RateLimitError("rl", response=_http(429, "https://api.openai.com"), body=None),
        openai_completion("ok"),
    ]
    assert make_openai(client).generate(system_prompt="s", user_prompt="u").text == "ok"


def test_providers_return_identical_response_shape() -> None:
    claude_client, openai_client = MagicMock(), MagicMock()
    claude_client.beta.messages.create.return_value = claude_message(text_block("x"))
    openai_client.chat.completions.create.return_value = openai_completion("x")
    a = make_claude(claude_client).generate(system_prompt="s", user_prompt="u")
    b = make_openai(openai_client).generate(system_prompt="s", user_prompt="u")
    assert type(a) is type(b)
    assert set(a.__slots__) == set(b.__slots__)


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------


def test_system_prompt_enforces_grounding_rules() -> None:
    lowered = SYSTEM_PROMPT.lower()
    for phrase in ("only on the excerpts", "outside knowledge", "do not invent", "contradict", "cite", "terminology", "concise"):
        assert phrase in lowered
    assert NOT_FOUND_ANSWER in SYSTEM_PROMPT


def test_user_prompt_wraps_context_and_question() -> None:
    prompt = build_user_prompt("What is X?", '<source id="1">X is Y.</source>')
    assert prompt.index("<context>") < prompt.index("<question>")
    assert "What is X?" in prompt
