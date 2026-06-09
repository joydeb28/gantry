from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass


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
    thinking_token_budget: int = 2048


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
        base_url="http://localhost:8000",
    ),
    "vllm-qwen3-0.6b": ModelProfile(
        name="vllm-qwen3-0.6b",
        provider="vllm",
        model="Qwen/Qwen3-0.6B",
        base_url="http://localhost:8000",
    ),
}


class ReasoningClient:
    def chat(self, messages: list[dict[str, str]]) -> ReasoningResponse:
        raise NotImplementedError


class OllamaReasoningClient(ReasoningClient):
    def __init__(self, profile: ModelProfile) -> None:
        self.profile = profile

    def chat(self, messages: list[dict[str, str]]) -> ReasoningResponse:
        payload = {
            "model": self.profile.model,
            "messages": messages,
            "think": self.profile.reasoning,
            "stream": False,
        }
        data = _post_json(f"{self.profile.base_url}/api/chat", payload)
        msg = data.get("message", {})
        return ReasoningResponse(
            content=msg.get("content", ""),
            thinking=msg.get("thinking", ""),
        )


class VLLMReasoningClient(ReasoningClient):
    def __init__(self, profile: ModelProfile) -> None:
        self.profile = profile

    def chat(self, messages: list[dict[str, str]]) -> ReasoningResponse:
        payload = {
            "model": self.profile.model,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": True},
            "thinking_token_budget": self.profile.thinking_token_budget,
        }
        data = _post_json(f"{self.profile.base_url}/v1/chat/completions", payload)
        choice = data.get("choices", [{}])[0].get("message", {})
        return ReasoningResponse(
            content=choice.get("content", ""),
            thinking=choice.get("reasoning", ""),
        )


def client_from_profile(profile_name: str, base_url: str | None = None) -> ReasoningClient:
    profile = SMALL_REASONING_MODELS[profile_name]
    if base_url:
        from dataclasses import replace
        profile = replace(profile, base_url=base_url)
    if profile.provider == "ollama":
        return OllamaReasoningClient(profile)
    if profile.provider == "vllm":
        return VLLMReasoningClient(profile)
    raise ValueError(f"Unknown provider: {profile.provider}")


def _post_json(url: str, payload: dict, timeout_seconds: int = 120) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        return json.loads(resp.read().decode("utf-8"))
