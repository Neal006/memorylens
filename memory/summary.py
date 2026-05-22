"""
SummaryMemory — rolling LLM-generated compression memory backend.

Strategy:
  Keep the last `window_size` messages verbatim.
  Every time the buffer exceeds `window_size`, compress the overflow
  into a running summary using either:
    - LLM (Groq)  when GROQ_API_KEY is set  → high fidelity
    - Extractive  otherwise                  → zero-cost fallback

This is conceptually how long-horizon chat assistants work:
recent context stays sharp, old context becomes a compressed narrative.
"""

import os
import re
from typing import List, Dict

from .base import BaseMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FACT_PATTERNS = re.compile(
    r"(my \w[\w\s]+ is |i am |i'm |changed to |updated to |now is |"
    r"name|city|age|occupation|company|hobby|language|food|score|subject)",
    re.IGNORECASE,
)

_COMPRESS_SYSTEM = (
    "You are a memory compressor for a conversational AI. "
    "Given a batch of conversation messages, extract and preserve EVERY personal fact, "
    "preference, update, and important detail. "
    "Merge these with the existing summary if one is provided. "
    "Output a single, compact paragraph of key facts — no filler, no opinions. "
    "Always prefer the NEWER value when a fact has been updated."
)


def _extractive_compress(messages: List[Dict], existing_summary: str = "") -> str:
    """
    Zero-cost fallback: keep only lines that look like personal facts.
    Merges with any existing summary.
    """
    kept: List[str] = []

    # Re-include existing summary lines
    if existing_summary:
        kept.append(existing_summary)

    for msg in messages:
        content = msg.get("content", "")
        if _FACT_PATTERNS.search(content):
            kept.append(content.strip())

    merged = " | ".join(kept)
    return merged[:800] if merged else ""


def _llm_compress(messages: List[Dict], existing_summary: str, model: str) -> str:
    """LLM-powered compression via Groq."""
    from utils.llm import chat

    batch_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )
    user_content = ""
    if existing_summary:
        user_content += f"Existing summary:\n{existing_summary}\n\n"
    user_content += f"New messages to absorb:\n{batch_text}"

    result = chat(
        [
            {"role": "system", "content": _COMPRESS_SYSTEM},
            {"role": "user",   "content": user_content},
        ],
        model=model,
        temperature=0.0,
        max_tokens=200,
    )
    # Fallback if LLM call failed
    if result.startswith("[LLM_ERROR"):
        return _extractive_compress(messages, existing_summary)
    return result.strip()


# ---------------------------------------------------------------------------
# SummaryMemory
# ---------------------------------------------------------------------------

class SummaryMemory(BaseMemory):
    """
    Rolling-summary memory backend.

    Parameters
    ----------
    window_size : int
        Number of most-recent messages kept verbatim.
    use_llm : bool | None
        True  → always use Groq for compression.
        False → always use extractive fallback.
        None  → auto-detect from GROQ_API_KEY env var.
    model : str
        Groq model name used for compression calls.
    """

    name = "summary"

    def __init__(
        self,
        window_size: int = 20,
        use_llm: bool | None = None,
        model: str = "llama-3.1-8b-instant",
    ) -> None:
        self.window_size = window_size
        self.model = model
        self._use_llm: bool = (
            bool(os.getenv("GROQ_API_KEY")) if use_llm is None else use_llm
        )

        self.recent: List[Dict] = []
        self.summary: str = ""

    # ------------------------------------------------------------------
    # BaseMemory interface
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str, turn: int) -> None:
        self.recent.append({"role": role, "content": content, "turn": turn})
        # Compress whenever the verbatim buffer grows past the window
        if len(self.recent) > self.window_size:
            self._compress()

    def get_context(self, query: str, current_turn: int) -> List[Dict]:
        context: List[Dict] = []
        if self.summary:
            context.append({
                "role": "system",
                "content": f"[Conversation summary] {self.summary}",
            })
        for msg in self.recent:
            context.append({"role": msg["role"], "content": msg["content"]})
        return context

    def reset(self) -> None:
        self.recent = []
        self.summary = ""

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compress(self) -> None:
        """Move the overflow (everything before the window) into the summary."""
        overflow = self.recent[: len(self.recent) - self.window_size]
        self.recent = self.recent[-self.window_size :]

        if self._use_llm:
            self.summary = _llm_compress(overflow, self.summary, self.model)
        else:
            self.summary = _extractive_compress(overflow, self.summary)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return "llm" if self._use_llm else "extractive"

    def __repr__(self) -> str:
        return (
            f"SummaryMemory(window={self.window_size}, "
            f"mode={self.mode}, "
            f"recent={len(self.recent)}, "
            f"summary_len={len(self.summary)})"
        )
