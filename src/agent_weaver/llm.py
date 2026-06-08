from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ReasoningResponse:
    content: str
    thinking: str = ""


@dataclass(frozen=True)
class ModelProfile:
    name: str
    provider: str
    model: str
    base_url: str
    reasoning: bool = True
    thinking_token_budget: int | None = 512


SMALL_REASONING_MODELS: dict[str, ModelProfile] = {
    "ollama-qwen3-4b": ModelProfile(
        name="ollama-qwen3-4b",
        provider="ollama",
        model="qwen3:4b",
        base_url="http://localhost:11434",
    ),
    "ollama-qwen3-1.7b": ModelProfile(
        name="ollama-qwen3-1.7b",
        provider="ollama",
        model="qwen3:1.7b",
        base_url="http://localhost:11434",
    ),
    "vllm-qwen3-4b": ModelProfile(
        name="vllm-qwen3-4b",
        provider="vllm",
        model="Qwen/Qwen3-4B",
        base_url="http://localhost:8000/v1",
    ),
    "vllm-qwen3-0.6b": ModelProfile(
        name="vllm-qwen3-0.6b",
        provider="vllm",
        model="Qwen/Qwen3-0.6B",
        base_url="http://localhost:8000/v1",
        thinking_token_budget=256,
    ),
}


class ReasoningClient(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> ReasoningResponse:
        ...


@dataclass(frozen=True)
class OllamaReasoningClient:
    profile: ModelProfile = SMALL_REASONING_MODELS["ollama-qwen3-4b"]
    timeout_seconds: int = 60

    def chat(self, messages: list[dict[str, str]]) -> ReasoningResponse:
        payload = {
            "model": self.profile.model,
            "messages": messages,
            "think": self.profile.reasoning,
            "stream": False,
        }
        data = _post_json(f"{self.profile.base_url.rstrip('/')}/api/chat", payload, self.timeout_seconds)
        message = data.get("message", {})
        return ReasoningResponse(content=str(message.get("content", "")).strip(), thinking=str(message.get("thinking", "")).strip())


@dataclass(frozen=True)
class VLLMReasoningClient:
    profile: ModelProfile = SMALL_REASONING_MODELS["vllm-qwen3-4b"]
    timeout_seconds: int = 60

    def chat(self, messages: list[dict[str, str]]) -> ReasoningResponse:
        payload: dict[str, object] = {
            "model": self.profile.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 700,
        }
        if self.profile.reasoning:
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        if self.profile.thinking_token_budget is not None:
            payload["thinking_token_budget"] = self.profile.thinking_token_budget

        data = _post_json(f"{self.profile.base_url.rstrip('/')}/chat/completions", payload, self.timeout_seconds)
        message = data.get("choices", [{}])[0].get("message", {})
        return ReasoningResponse(content=str(message.get("content", "")).strip(), thinking=str(message.get("reasoning", "")).strip())


def client_from_profile(profile_name: str, base_url: str | None = None) -> ReasoningClient:
    profile = SMALL_REASONING_MODELS[profile_name]
    if base_url:
        profile = ModelProfile(
            name=profile.name,
            provider=profile.provider,
            model=profile.model,
            base_url=base_url,
            reasoning=profile.reasoning,
            thinking_token_budget=profile.thinking_token_budget,
        )
    if profile.provider == "ollama":
        return OllamaReasoningClient(profile)
    if profile.provider == "vllm":
        return VLLMReasoningClient(profile)
    raise ValueError(f"Unsupported provider: {profile.provider}")


def _post_json(url: str, payload: dict[str, object], timeout_seconds: int) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))
