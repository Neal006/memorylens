"""Scenario — a domain-specific benchmark definition (facts + filler turns)."""

from dataclasses import dataclass
from typing import List

from memorylens.simulator.facts import Fact


@dataclass(frozen=True)
class Scenario:
    """
    A benchmark scenario bundles everything domain-specific:

    persona_pool : one fact set per persona; persona 0 is the single-seed default.
                   All personas must share the same fact keys so results are
                   comparable across seeds.
    filler_turns : domain-appropriate distractor questions fired between fact
                   injections.
    """

    name: str
    description: str
    persona_pool: List[List[Fact]]
    filler_turns: List[str]

    @property
    def facts(self) -> List[Fact]:
        return self.persona_pool[0]

    def validate(self) -> None:
        if not self.persona_pool:
            raise ValueError(f"scenario '{self.name}': persona_pool is empty")
        if not self.filler_turns:
            raise ValueError(f"scenario '{self.name}': filler_turns is empty")
        keys = {f.key for f in self.facts}
        for i, persona in enumerate(self.persona_pool):
            persona_keys = {f.key for f in persona}
            if persona_keys != keys:
                raise ValueError(
                    f"scenario '{self.name}': persona {i} fact keys {persona_keys} "
                    f"differ from persona 0 keys {keys}"
                )
