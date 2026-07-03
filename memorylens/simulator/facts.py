from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Fact:
    key: str
    value: str
    injected_at: int
    updated_at: Optional[int] = None
    updated_value: Optional[str] = None

    def current_value(self, at_turn: int) -> str:
        if self.updated_at and at_turn >= self.updated_at and self.updated_value:
            return self.updated_value
        return self.value

    def injection_text(self) -> str:
        return f"My {self.key.replace('_', ' ')} is {self.value}."

    def update_text(self) -> str:
        return f"Actually, my {self.key.replace('_', ' ')} has changed to {self.updated_value}."

    def query_text(self) -> str:
        return f"What is my {self.key.replace('_', ' ')}?"


BENCHMARK_FACTS: List[Fact] = [
    Fact("name",                "Arjun Sharma",         injected_at=0),
    Fact("city",                "Bangalore",            injected_at=1,  updated_at=40, updated_value="Mumbai"),
    Fact("occupation",          "software engineer",    injected_at=2),
    Fact("age",                 "27",                   injected_at=3,  updated_at=60, updated_value="28"),
    Fact("company",             "TechStartup",          injected_at=4),
    Fact("programming language","Python",               injected_at=5),
    Fact("favorite food",       "biryani",              injected_at=7),
    Fact("hobby",               "playing cricket",      injected_at=9),
]
