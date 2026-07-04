import re
from typing import List, Dict, Optional, Callable, Tuple
import numpy as np
from .base import BaseMemory
from .decay import get_decay_fn
from memorylens.utils.embeddings import embed


def _extractive_summary(messages: List[Dict], max_chars: int = 400) -> str:
    """
    Lightweight extractive summary: keep sentences that contain key=value patterns.
    No LLM call needed — fast and cost-free.

    Update messages ("changed to") are placed first so they survive truncation
    and take precedence over initial injection lines for the same fact.
    """
    update_lines = []
    injection_lines = []
    for m in messages:
        content = m.get("content", "")
        if "changed to" in content.lower():
            update_lines.append(f"{m['role']}: {content}")
        elif any(kw in content.lower() for kw in ["my ", "is ", "are ", "name", "city", "age"]):
            injection_lines.append(f"{m['role']}: {content}")

    lines = update_lines + injection_lines
    summary = " | ".join(lines)
    return summary[:max_chars]


def _parse_update(content: str) -> Optional[Tuple[str, str]]:
    """
    Parse "Actually, my <key_name> has changed to <new_value>."
    Returns (key_name, new_value) or None.
    """
    m = re.search(
        r"my\s+(.+?)\s+has\s+changed\s+to\s+(.+?)[\.\!]?\s*$",
        content,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None


def _patch_cold_with_update(cold: List[str], key_name: str, new_value: str) -> List[str]:
    """
    Replace any value associated with *key_name* in cold summary strings so that
    stale original values are overwritten by the current value.

    Targets the pattern produced by Fact.injection_text():
      "my <key> is <old_value>" → "my <key> is <new_value>"
    and also direct "changed to <old_new>" occurrences from earlier updates.
    """
    # Match "my <key_name> is <anything up to a pipe/period/end>"
    pattern = re.compile(
        r"(my\s+" + re.escape(key_name) + r"\s+(?:is|are|was|has been)\s+)([^|.\n]+)",
        re.IGNORECASE,
    )
    result = []
    for entry in cold:
        patched = pattern.sub(lambda m: m.group(1) + new_value, entry)
        result.append(patched)
    return result


class CascadingTemporalMemory(BaseMemory):
    """
    Three-tier cascading memory with pluggable temporal decay.

    Hot  — last `hot_size` messages, verbatim, full fidelity
    Warm — older messages, full text but semantically filtered on retrieval
             with age-based decay weighting (default: Ebbinghaus forgetting curve)
    Cold — ancient context, compressed to extractive summaries

    Decay options: 'ebbinghaus' (default) | 'exponential' | 'linear' | 'default'
    Reference: Ebbinghaus, H. (1885). Über das Gedächtnis.

    Fix (issue #2): when a fact-update message cascades from the warm tier into
    cold, existing cold summaries are patched so the stale original value is
    replaced by the new one. This eliminates the 100% temporal-drift regression
    observed at T=100 where the old value was frozen inside compressed cold text.
    """

    name = "cascading"

    def __init__(
        self,
        hot_size:  int = 12,
        warm_size: int = 30,
        cold_max:  int = 4,
        decay:     str = "ebbinghaus",
    ):
        self.hot_size  = hot_size
        self.warm_size = warm_size
        self.cold_max  = cold_max
        self.decay_fn: Callable[[int, int], float] = get_decay_fn(decay)
        self.decay_name = decay

        self.hot:       List[Dict]        = []
        self.warm:      List[Dict]        = []
        self.warm_embs: List[np.ndarray]  = []
        self.cold:      List[str]         = []
        self.turn_count = 0

        # fact_key → new_value for every update seen so far
        self._fact_updates: Dict[str, str] = {}

    def add_message(self, role: str, content: str, turn: int) -> None:
        msg = {"role": role, "content": content, "turn": turn}

        parsed = _parse_update(content)
        if parsed:
            key_name, new_val = parsed
            self._fact_updates[key_name] = new_val
            # Immediately patch warm messages that are already compressible
            # (they haven't hit cold yet; cold is patched in _cascade_warm)
            self.cold = _patch_cold_with_update(self.cold, key_name, new_val)

        self.hot.append(msg)
        self.turn_count += 1

        if len(self.hot) > self.hot_size:
            self._cascade_hot()

    def _cascade_hot(self) -> None:
        overflow = self.hot[: len(self.hot) - self.hot_size]
        self.hot = self.hot[-self.hot_size :]

        for msg in overflow:
            self.warm.append(msg)
            self.warm_embs.append(embed([msg["content"]])[0])

        if len(self.warm) > self.warm_size:
            self._cascade_warm()

    def _cascade_warm(self) -> None:
        overflow = self.warm[: len(self.warm) - self.warm_size]
        self.warm = self.warm[-self.warm_size :]
        self.warm_embs = self.warm_embs[-self.warm_size :]

        summary = _extractive_summary(overflow)
        if summary:
            self.cold.append(summary)

        # Patch all cold entries with every known fact update so no stale
        # values survive compression into the cold tier.
        for key_name, new_val in self._fact_updates.items():
            self.cold = _patch_cold_with_update(self.cold, key_name, new_val)

        if len(self.cold) > self.cold_max:
            # Merge oldest-first: early facts stay at the head and survive
            # truncation. Stale values are already rewritten in place by
            # _patch_cold_with_update, so newer text needs no priority here.
            merged = self.cold[0] + " | " + self.cold[1]
            self.cold = [merged[:600]] + self.cold[2:]

    def get_context(self, query: str, current_turn: int) -> List[Dict]:
        context: List[Dict] = []

        # Cold tier: inject ALL summaries as a system-level context block
        if self.cold:
            combined = " | ".join(self.cold)
            context.append({"role": "system", "content": f"[Historical context] {combined}"})

        # Warm tier: semantic retrieval with pluggable temporal decay
        if self.warm:
            q_emb  = embed([query])[0]
            corpus = np.stack(self.warm_embs)
            raw_sims = (corpus @ q_emb).tolist()

            scored = []
            for i, sim in enumerate(raw_sims):
                age   = current_turn - self.warm[i].get("turn", 0)
                decay = self.decay_fn(age, max(1, current_turn))
                scored.append((i, sim * decay))

            scored.sort(key=lambda x: x[1], reverse=True)
            top_warm = sorted(idx for idx, _ in scored[:3])
            for i in top_warm:
                context.append({"role": self.warm[i]["role"], "content": self.warm[i]["content"]})

        # Hot tier: always fully included
        for msg in self.hot:
            context.append({"role": msg["role"], "content": msg["content"]})

        return context

    def reset(self) -> None:
        self.hot         = []
        self.warm        = []
        self.warm_embs   = []
        self.cold        = []
        self.turn_count  = 0
        self._fact_updates = {}
