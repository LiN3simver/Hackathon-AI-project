"""Which text model phrases the assistant's summary, and how to reach it.

The assistant's verification never changes with the model: every backend only
turns the same messages into streamed text. The default option is the model the
assistant already uses (GUARDIAN_HISTORY_RAG_MODEL). More options come from
GUARDIAN_HISTORY_LLM_MODELS, a JSON list, so an operator can add a paid API or a
GPU-served open model without code changes, for example:

    [{"id": "gpt", "label": "GPT 付費模型", "provider": "openai_compatible", "tier": "paid",
      "model": "gpt-4.1-mini", "base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY"},
     {"id": "qwen-72b", "label": "Qwen 72B（GPU）", "provider": "openai_compatible", "tier": "gpu",
      "model": "Qwen/Qwen2.5-72B-Instruct", "base_url": "http://gpu-server:8000/v1"}]

`openai_compatible` covers OpenAI and any server speaking its chat API (vLLM, TGI,
LM Studio, most cloud gateways). The answers are built from text records only, so
vision-language models are refused here.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Literal, Protocol

import httpx

from app.services import history_llm

logger = logging.getLogger("api")

Provider = Literal["ollama", "openai_compatible"]
Tier = Literal["local", "paid", "gpu"]
DEFAULT_ID = "default"
_PROVIDERS = {"ollama", "openai_compatible"}
_TIERS = {"local", "paid", "gpu"}
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,59}$")
# Names of vision-language families; the assistant only ever reads text.
_VISION = re.compile(r"(?:^|[^a-z])(?:vl|vision|llava|bakllava|moondream|minicpm-v|pixtral|qvq|cogvlm|internvl|"
                     r"paligemma|idefics|florence)(?:[^a-z]|$)", re.IGNORECASE)


@dataclass(frozen=True)
class ModelOption:
    id: str
    label: str
    provider: Provider
    model: str
    tier: Tier = "local"
    description: str = ""
    base_url: str | None = None
    # Name of the environment variable holding the key; the key itself never sits in settings.
    api_key_env: str | None = None


class ChatBackend(Protocol):
    """One chat endpoint: request body in, (text, done, timing) per streamed line out."""

    option: ModelOption
    url: str

    def headers(self) -> dict[str, str]: ...

    def body(self, messages: list[dict], *, intent: str, stream: bool = True) -> dict: ...

    def parse(self, line: str) -> tuple[str, bool, dict]: ...


@dataclass
class OllamaBackend:
    option: ModelOption
    settings: object
    url: str = field(init=False)

    def __post_init__(self) -> None:
        base = self.option.base_url or self.settings.ollama_base_url
        self.url = f"{base.rstrip('/')}/api/chat"

    def headers(self) -> dict[str, str]:
        return {}

    def body(self, messages: list[dict], *, intent: str, stream: bool = True) -> dict:
        body = history_llm.chat_body(self.settings, messages, intent=intent, stream=stream)
        body["model"] = self.option.model
        return body

    def parse(self, line: str) -> tuple[str, bool, dict]:
        return history_llm.parse_chunk(line)


@dataclass
class OpenAICompatibleBackend:
    """One per request: it remembers the finish reason until the stream's final line."""

    option: ModelOption
    settings: object
    url: str = field(init=False)
    _timing: dict = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.url = f"{(self.option.base_url or '').rstrip('/')}/chat/completions"

    def headers(self) -> dict[str, str]:
        key = os.environ.get(self.option.api_key_env or "", "").strip()
        return {"Authorization": f"Bearer {key}"} if key else {}

    def body(self, messages: list[dict], *, intent: str, stream: bool = True) -> dict:
        # Other tokenizers spend more tokens per Chinese character; the prompt and verifier still cap the answer.
        body = {"model": self.option.model, "messages": messages, "stream": stream, "temperature": 0.2,
                "top_p": 0.9, "max_tokens": history_llm.NUM_PREDICT.get(intent, 160) * 2}
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    def parse(self, line: str) -> tuple[str, bool, dict]:
        """Server-sent `data:` lines → (text, done, timing), shaped like Ollama's so callers don't care."""
        line = line.strip()
        if not line.startswith("data:"):
            return "", False, {}
        payload = line[5:].strip()
        if payload == "[DONE]":
            return "", True, self._timing
        data = json.loads(payload)
        if data.get("error"):
            raise ValueError(f"LLM error: {data['error']}")
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("delta") or choice.get("message") or {}).get("content") or ""
        reason = choice.get("finish_reason")
        usage = data.get("usage") or {}
        if not reason and not usage:
            return text, False, {}
        if reason:
            self._timing["done_reason"] = reason
        if usage:
            self._timing.update(prompt_eval_count=usage.get("prompt_tokens"), eval_count=usage.get("completion_tokens"))
        # finish_reason and usage may arrive in separate chunks; usage (sent last) or [DONE] ends the stream.
        return text, bool(usage), self._timing


def is_vision_model(name: str) -> bool:
    return bool(_VISION.search(name))


def _default(settings) -> ModelOption:
    model = settings.history_rag_model or settings.ollama_vision_model
    return ModelOption(id=DEFAULT_ID, label=model, provider="ollama", model=model, tier="local",
                       description="目前使用的本機模型")


def _configured(settings) -> list[ModelOption]:
    options: list[ModelOption] = []
    for raw in settings.history_llm_models or []:
        try:
            option = ModelOption(
                id=str(raw["id"]).strip(), label=str(raw.get("label") or raw["model"]).strip(),
                provider=raw.get("provider", "ollama"), model=str(raw["model"]).strip(),
                tier=raw.get("tier", "local"), description=str(raw.get("description") or "").strip(),
                base_url=raw.get("base_url") or None, api_key_env=raw.get("api_key_env") or None)
        except (KeyError, TypeError, AttributeError):
            logger.warning("history_llm_model_skipped reason=invalid entry=%r", raw)
            continue
        problem = None
        if not _ID.match(option.id) or option.id == DEFAULT_ID:
            problem = "bad_id"
        elif option.provider not in _PROVIDERS or option.tier not in _TIERS:
            problem = "bad_provider_or_tier"
        elif option.provider == "openai_compatible" and not option.base_url:
            problem = "missing_base_url"
        elif is_vision_model(option.model) or option.model == settings.ollama_vision_model:
            problem = "vision_model"
        elif any(existing.id == option.id for existing in options):
            problem = "duplicate_id"
        if problem:
            logger.warning("history_llm_model_skipped id=%s reason=%s", option.id, problem)
            continue
        options.append(option)
    return options


def list_options(settings) -> list[ModelOption]:
    return [_default(settings), *_configured(settings)]


def resolve(settings, model_id: str | None) -> tuple[ModelOption, bool]:
    """The chosen option, or the default with False when the id is unknown (e.g. removed since the page loaded)."""
    options = list_options(settings)
    if not model_id:
        return options[0], True
    found = next((option for option in options if option.id == model_id), None)
    return (found, True) if found else (options[0], False)


def backend_for(option: ModelOption, settings) -> ChatBackend:
    if option.provider == "openai_compatible":
        return OpenAICompatibleBackend(option, settings)
    return OllamaBackend(option, settings)


def available(options: list[ModelOption], settings) -> dict[str, bool]:
    """Cheap readiness for the picker: Ollama lists its models; an API option needs its key set."""
    state: dict[str, bool] = {}
    tags: dict[str, set[str] | None] = {}
    for option in options:
        if option.provider == "openai_compatible":
            state[option.id] = not option.api_key_env or bool(os.environ.get(option.api_key_env, "").strip())
            continue
        base = (option.base_url or settings.ollama_base_url).rstrip("/")
        if base not in tags:
            try:
                with httpx.Client(transport=history_llm.TRANSPORT, timeout=httpx.Timeout(0.8)) as client:
                    response = client.get(f"{base}/api/tags")
                response.raise_for_status()
                tags[base] = {item.get("name", "") for item in response.json().get("models", [])}
            except (httpx.HTTPError, ValueError, TypeError):
                tags[base] = None
        names = tags[base]
        state[option.id] = names is not None and (option.model in names or f"{option.model}:latest" in names)
    return state
