"""
utils/providers.py — Unified LLM provider abstraction for MemoryLens.

Supports five backends:
  - Groq          (GROQ_API_KEY)
  - OpenAI        (OPENAI_API_KEY)
  - Anthropic     (ANTHROPIC_API_KEY)
  - OpenRouter    (OPENROUTER_API_KEY)   — access 200+ models via one key
  - Ollama        (local, no key)        — any locally running model

Priority for auto-detection:
  Groq → OpenAI → Anthropic → OpenRouter → Ollama → None (content-only mode)

Usage:
    from memorylens.utils.providers import get_provider, list_available

    provider = get_provider()          # auto-detect
    provider = get_provider("openai")  # force a specific one

    if provider:
        answer = provider.chat([{"role": "user", "content": "Hello"}])
        print(provider.name, answer)
    else:
        print("No provider available — running content-only mode")
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    """Common interface for all LLM backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name, e.g. 'groq/llama-3.1-8b-instant'."""

    @property
    @abstractmethod
    def provider_type(self) -> str:
        """Short key: 'groq' | 'openai' | 'anthropic' | 'openrouter' | 'ollama'."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict],
        max_tokens: int = 256,
        temperature: float = 0.1,
    ) -> str:
        """
        Send a chat request.
        Returns the assistant's reply as a plain string.
        Returns '[PROVIDER_ERROR: ...]' on failure — never raises.
        """

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True if the required credentials/service are present."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"


# ─────────────────────────────────────────────────────────────────────────────
# Groq
# ─────────────────────────────────────────────────────────────────────────────

class GroqProvider(LLMProvider):
    DEFAULT_MODEL = "llama-3.1-8b-instant"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self._client = None

    @property
    def name(self) -> str:
        return f"groq/{self.model}"

    @property
    def provider_type(self) -> str:
        return "groq"

    @classmethod
    def is_available(cls) -> bool:
        return bool(os.getenv("GROQ_API_KEY"))

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        return self._client

    def chat(self, messages: List[Dict], max_tokens: int = 256, temperature: float = 0.1) -> str:
        for attempt in range(3):
            try:
                resp = self._get_client().chat.completions.create(
                    model=self.model,
                    messages=_clean_messages(messages),
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    return f"[PROVIDER_ERROR: groq — {e}]"
        return "[PROVIDER_ERROR: groq — max retries]"


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self._client = None

    @property
    def name(self) -> str:
        return f"openai/{self.model}"

    @property
    def provider_type(self) -> str:
        return "openai"

    @classmethod
    def is_available(cls) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    def chat(self, messages: List[Dict], max_tokens: int = 256, temperature: float = 0.1) -> str:
        for attempt in range(3):
            try:
                resp = self._get_client().chat.completions.create(
                    model=self.model,
                    messages=_clean_messages(messages),
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    return f"[PROVIDER_ERROR: openai — {e}]"
        return "[PROVIDER_ERROR: openai — max retries]"


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic
# ─────────────────────────────────────────────────────────────────────────────

class AnthropicProvider(LLMProvider):
    DEFAULT_MODEL = "claude-haiku-4-5"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self._client = None

    @property
    def name(self) -> str:
        return f"anthropic/{self.model}"

    @property
    def provider_type(self) -> str:
        return "anthropic"

    @classmethod
    def is_available(cls) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY"))

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return self._client

    def chat(self, messages: List[Dict], max_tokens: int = 256, temperature: float = 0.1) -> str:
        for attempt in range(3):
            try:
                # Anthropic separates system messages
                system_parts = [m["content"] for m in messages if m["role"] == "system"]
                user_messages = [m for m in messages if m["role"] != "system"]
                system_str = " ".join(system_parts) if system_parts else None

                kwargs: Dict = dict(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=_clean_messages(user_messages),
                )
                if system_str:
                    kwargs["system"] = system_str

                resp = self._get_client().messages.create(**kwargs)
                return resp.content[0].text.strip()
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    return f"[PROVIDER_ERROR: anthropic — {e}]"
        return "[PROVIDER_ERROR: anthropic — max retries]"


# ─────────────────────────────────────────────────────────────────────────────
# OpenRouter (200+ models via one endpoint)
# ─────────────────────────────────────────────────────────────────────────────

class OpenRouterProvider(LLMProvider):
    DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self._client = None

    @property
    def name(self) -> str:
        return f"openrouter/{self.model}"

    @property
    def provider_type(self) -> str:
        return "openrouter"

    @classmethod
    def is_available(cls) -> bool:
        return bool(os.getenv("OPENROUTER_API_KEY"))

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url=self.BASE_URL,
            )
        return self._client

    def chat(self, messages: List[Dict], max_tokens: int = 256, temperature: float = 0.1) -> str:
        for attempt in range(3):
            try:
                resp = self._get_client().chat.completions.create(
                    model=self.model,
                    messages=_clean_messages(messages),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_headers={
                        "HTTP-Referer": "https://github.com/Neal006/memorylens",
                        "X-Title": "MemoryLens Benchmark",
                    },
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    return f"[PROVIDER_ERROR: openrouter — {e}]"
        return "[PROVIDER_ERROR: openrouter — max retries]"


# ─────────────────────────────────────────────────────────────────────────────
# Ollama (local models — no API key)
# ─────────────────────────────────────────────────────────────────────────────

class OllamaProvider(LLMProvider):
    DEFAULT_MODEL = "llama3.2"
    DEFAULT_HOST  = "http://localhost:11434"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
    ) -> None:
        self.model = model
        self.host  = os.getenv("OLLAMA_HOST", host)

    @property
    def name(self) -> str:
        return f"ollama/{self.model}"

    @property
    def provider_type(self) -> str:
        return "ollama"

    @classmethod
    def is_available(cls) -> bool:
        """Check if the Ollama server is reachable."""
        import urllib.request
        host = os.getenv("OLLAMA_HOST", cls.DEFAULT_HOST)
        try:
            urllib.request.urlopen(f"{host}/api/tags", timeout=2)
            return True
        except Exception:
            return False

    def chat(self, messages: List[Dict], max_tokens: int = 256, temperature: float = 0.1) -> str:
        import json
        import urllib.request

        payload = json.dumps({
            "model": self.model,
            "messages": _clean_messages(messages),
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }).encode()

        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    f"{self.host}/api/chat",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read())
                    return body["message"]["content"].strip()
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    return f"[PROVIDER_ERROR: ollama — {e}]"
        return "[PROVIDER_ERROR: ollama — max retries]"


# ─────────────────────────────────────────────────────────────────────────────
# Registry + auto-detection
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, type] = {
    "groq":        GroqProvider,
    "openai":      OpenAIProvider,
    "anthropic":   AnthropicProvider,
    "openrouter":  OpenRouterProvider,
    "ollama":      OllamaProvider,
}

_PRIORITY = ["groq", "openai", "anthropic", "openrouter", "ollama"]


def get_provider(name: Optional[str] = None) -> Optional[LLMProvider]:
    """
    Return a ready-to-use LLMProvider instance.

    Parameters
    ----------
    name : str | None
        Force a specific provider ('groq', 'openai', 'anthropic',
        'openrouter', 'ollama').  Pass None to auto-detect.

    Returns
    -------
    LLMProvider | None
        None if no provider is available — caller should fall back to
        content-only evaluation.
    """
    if name:
        name = name.lower()
        cls = _REGISTRY.get(name)
        if cls is None:
            raise ValueError(f"Unknown provider '{name}'. Choose from: {list(_REGISTRY)}")
        if not cls.is_available():
            raise RuntimeError(
                f"Provider '{name}' is not available. "
                f"Check your environment variables / service status."
            )
        return cls()

    # Auto-detect
    for key in _PRIORITY:
        cls = _REGISTRY[key]
        if cls.is_available():
            return cls()

    return None


def list_available() -> List[str]:
    """Return names of all currently available providers."""
    return [k for k, cls in _REGISTRY.items() if cls.is_available()]


def provider_from_env() -> Optional[LLMProvider]:
    """
    Convenience: read MEMORYLENS_PROVIDER env var, fall back to auto-detect.
    MEMORYLENS_PROVIDER=openai  → forces OpenAI
    MEMORYLENS_PROVIDER=        → auto-detect
    """
    forced = os.getenv("MEMORYLENS_PROVIDER", "").strip().lower()
    return get_provider(forced if forced else None)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_messages(messages: List[Dict]) -> List[Dict]:
    """
    Return only {role, content} pairs with valid roles.
    Merges consecutive messages from the same role to satisfy strict APIs.
    """
    valid_roles = {"system", "user", "assistant"}
    cleaned: List[Dict] = []

    for m in messages:
        role    = m.get("role", "user")
        content = m.get("content", "").strip()
        if not content or role not in valid_roles:
            continue
        # Merge consecutive same-role messages
        if cleaned and cleaned[-1]["role"] == role:
            cleaned[-1]["content"] += "\n" + content
        else:
            cleaned.append({"role": role, "content": content})

    return cleaned
