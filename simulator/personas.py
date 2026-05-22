"""
simulator/personas.py — Diverse persona pool for multi-seed benchmarking.

Each persona produces a distinct but structurally equivalent fact set so
results can be averaged across demographics rather than tied to one user.
"""

from typing import List
from .facts import Fact


PERSONA_POOL: List[List[Fact]] = [
    # ── Persona 0 — Arjun Sharma (original baseline) ────────────────────────
    [
        Fact("name",                "Arjun Sharma",      injected_at=0),
        Fact("city",                "Bangalore",         injected_at=1,  updated_at=40, updated_value="Mumbai"),
        Fact("occupation",          "software engineer", injected_at=2),
        Fact("age",                 "27",                injected_at=3,  updated_at=60, updated_value="28"),
        Fact("company",             "TechStartup",       injected_at=4),
        Fact("programming language","Python",            injected_at=5),
        Fact("favorite food",       "biryani",           injected_at=7),
        Fact("hobby",               "playing cricket",   injected_at=9),
    ],
    # ── Persona 1 — Sofia Reyes ─────────────────────────────────────────────
    [
        Fact("name",                "Sofia Reyes",       injected_at=0),
        Fact("city",                "Mexico City",       injected_at=1,  updated_at=40, updated_value="Guadalajara"),
        Fact("occupation",          "product manager",   injected_at=2),
        Fact("age",                 "31",                injected_at=3,  updated_at=60, updated_value="32"),
        Fact("company",             "FinovaTech",        injected_at=4),
        Fact("programming language","JavaScript",        injected_at=5),
        Fact("favorite food",       "tacos",             injected_at=7),
        Fact("hobby",               "painting",          injected_at=9),
    ],
    # ── Persona 2 — Wei Zhang ───────────────────────────────────────────────
    [
        Fact("name",                "Wei Zhang",         injected_at=0),
        Fact("city",                "Shanghai",          injected_at=1,  updated_at=40, updated_value="Beijing"),
        Fact("occupation",          "data scientist",    injected_at=2),
        Fact("age",                 "29",                injected_at=3,  updated_at=60, updated_value="30"),
        Fact("company",             "CloudMind AI",      injected_at=4),
        Fact("programming language","R",                 injected_at=5),
        Fact("favorite food",       "dumplings",         injected_at=7),
        Fact("hobby",               "chess",             injected_at=9),
    ],
    # ── Persona 3 — Amara Osei ──────────────────────────────────────────────
    [
        Fact("name",                "Amara Osei",        injected_at=0),
        Fact("city",                "Accra",             injected_at=1,  updated_at=40, updated_value="Kumasi"),
        Fact("occupation",          "UX designer",       injected_at=2),
        Fact("age",                 "25",                injected_at=3,  updated_at=60, updated_value="26"),
        Fact("company",             "DesignHub",         injected_at=4),
        Fact("programming language","TypeScript",        injected_at=5),
        Fact("favorite food",       "jollof rice",       injected_at=7),
        Fact("hobby",               "photography",       injected_at=9),
    ],
    # ── Persona 4 — Lars Eriksson ────────────────────────────────────────────
    [
        Fact("name",                "Lars Eriksson",     injected_at=0),
        Fact("city",                "Stockholm",         injected_at=1,  updated_at=40, updated_value="Gothenburg"),
        Fact("occupation",          "ML engineer",       injected_at=2),
        Fact("age",                 "34",                injected_at=3,  updated_at=60, updated_value="35"),
        Fact("company",             "NordAI Labs",       injected_at=4),
        Fact("programming language","Go",                injected_at=5),
        Fact("favorite food",       "meatballs",         injected_at=7),
        Fact("hobby",               "cross-country skiing", injected_at=9),
    ],
]
