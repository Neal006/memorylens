"""
simulator/scenarios/edtech.py — EdTech (student-tutor) benchmark scenario.

Models a student interacting with an AI tutor over many turns.  Facts cover
hierarchical personal and academic attributes; updates simulate real learning
progressions (grade improvement, subject change, learning-style refinement).

The filler turns are domain-specific tutoring requests (concept explanations,
problem-solving help, study strategy questions) rather than generic tech Q&A,
making this a harder benchmark for memory systems that rely on keyword overlap.

Closes #13 / #4.
"""

from typing import List
from simulator.facts import Fact


# ── Fact sets ─────────────────────────────────────────────────────────────────

EDTECH_FACTS: List[Fact] = [
    # Identity
    Fact("name",            "Priya Nair",       injected_at=0),
    Fact("grade",           "10th grade",       injected_at=1,  updated_at=45, updated_value="11th grade"),
    Fact("school",          "Sunrise Academy",  injected_at=2),
    # Academic profile
    Fact("favourite subject","mathematics",      injected_at=4,  updated_at=55, updated_value="physics"),
    Fact("weakest subject",  "history",          injected_at=6),
    Fact("current GPA",      "3.2",              injected_at=8,  updated_at=70, updated_value="3.6"),
    # Learning preferences
    Fact("learning style",   "visual learner",  injected_at=10, updated_at=60, updated_value="hands-on learner"),
    Fact("study hours per day","2 hours",        injected_at=12),
]


# ── Persona pool for multi-seed runs ─────────────────────────────────────────

EDTECH_PERSONA_POOL: List[List[Fact]] = [
    # Persona 0 — Priya Nair (baseline)
    EDTECH_FACTS,
    # Persona 1 — Carlos Mendez
    [
        Fact("name",             "Carlos Mendez",   injected_at=0),
        Fact("grade",            "9th grade",       injected_at=1,  updated_at=45, updated_value="10th grade"),
        Fact("school",           "Westbrook High",  injected_at=2),
        Fact("favourite subject","biology",          injected_at=4,  updated_at=55, updated_value="chemistry"),
        Fact("weakest subject",  "algebra",          injected_at=6),
        Fact("current GPA",      "2.9",              injected_at=8,  updated_at=70, updated_value="3.3"),
        Fact("learning style",   "auditory learner", injected_at=10, updated_at=60, updated_value="reading/writing learner"),
        Fact("study hours per day","1.5 hours",      injected_at=12),
    ],
    # Persona 2 — Aisha Kamara
    [
        Fact("name",             "Aisha Kamara",    injected_at=0),
        Fact("grade",            "11th grade",      injected_at=1,  updated_at=45, updated_value="12th grade"),
        Fact("school",           "Greenfield IB",   injected_at=2),
        Fact("favourite subject","literature",       injected_at=4,  updated_at=55, updated_value="psychology"),
        Fact("weakest subject",  "calculus",         injected_at=6),
        Fact("current GPA",      "3.5",              injected_at=8,  updated_at=70, updated_value="3.8"),
        Fact("learning style",   "reading/writing learner", injected_at=10, updated_at=60, updated_value="visual learner"),
        Fact("study hours per day","3 hours",        injected_at=12),
    ],
    # Persona 3 — Haruto Tanaka
    [
        Fact("name",             "Haruto Tanaka",   injected_at=0),
        Fact("grade",            "8th grade",       injected_at=1,  updated_at=45, updated_value="9th grade"),
        Fact("school",           "Sakura Middle",   injected_at=2),
        Fact("favourite subject","computer science", injected_at=4,  updated_at=55, updated_value="mathematics"),
        Fact("weakest subject",  "essay writing",   injected_at=6),
        Fact("current GPA",      "3.0",             injected_at=8,  updated_at=70, updated_value="3.4"),
        Fact("learning style",   "hands-on learner", injected_at=10, updated_at=60, updated_value="visual learner"),
        Fact("study hours per day","2.5 hours",      injected_at=12),
    ],
    # Persona 4 — Amelia Brooks
    [
        Fact("name",             "Amelia Brooks",   injected_at=0),
        Fact("grade",            "12th grade",      injected_at=1,  updated_at=45, updated_value="1st year university"),
        Fact("school",           "Oakdale High",    injected_at=2),
        Fact("favourite subject","economics",        injected_at=4,  updated_at=55, updated_value="statistics"),
        Fact("weakest subject",  "organic chemistry", injected_at=6),
        Fact("current GPA",      "3.7",             injected_at=8,  updated_at=70, updated_value="3.9"),
        Fact("learning style",   "auditory learner", injected_at=10, updated_at=60, updated_value="hands-on learner"),
        Fact("study hours per day","4 hours",        injected_at=12),
    ],
]


# ── Domain-specific filler turns ─────────────────────────────────────────────

EDTECH_FILLER_TURNS: List[str] = [
    "Can you explain the Pythagorean theorem with a real-world example?",
    "I'm struggling to understand photosynthesis. Can you break it down simply?",
    "What is the difference between mitosis and meiosis?",
    "How do I solve simultaneous equations using substitution?",
    "Can you explain Newton's three laws of motion?",
    "What are the key themes in Romeo and Juliet?",
    "How do I write a strong thesis statement for an essay?",
    "What is the difference between speed and velocity?",
    "Can you explain the water cycle to me?",
    "How do I factorise a quadratic expression?",
    "What caused the First World War?",
    "Can you help me understand the concept of supply and demand?",
    "What is the difference between an atom and a molecule?",
    "How do I find the area of a circle?",
    "Can you explain what DNA replication is?",
    "What is the significance of the French Revolution?",
    "How do I convert fractions to decimals?",
    "What are the main parts of a cell and their functions?",
    "Can you explain the concept of gravity?",
    "How do I improve my reading comprehension skills?",
]
