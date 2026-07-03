"""Scenario registry — all built-in benchmark scenarios."""

from typing import Dict, List

from memorylens.simulator.scenarios.base import Scenario
from memorylens.simulator.conversation import FILLER_TURNS
from memorylens.simulator.personas import PERSONA_POOL
from memorylens.simulator.scenarios.edtech import EDTECH
from memorylens.simulator.scenarios.customer_support import CUSTOMER_SUPPORT
from memorylens.simulator.scenarios.medical import MEDICAL

DEFAULT = Scenario(
    name="default",
    description="General tech Q&A conversation: personal profile facts with "
                "mid-conversation updates (city, age).",
    persona_pool=PERSONA_POOL,
    filler_turns=FILLER_TURNS,
)

SCENARIOS: Dict[str, Scenario] = {
    s.name: s for s in (DEFAULT, EDTECH, CUSTOMER_SUPPORT, MEDICAL)
}


def get_scenario(name: str) -> Scenario:
    scenario = SCENARIOS.get(name)
    if scenario is None:
        raise ValueError(f"Unknown scenario '{name}'. Choose from: {list(SCENARIOS)}")
    return scenario


def list_scenarios() -> List[str]:
    return list(SCENARIOS)
