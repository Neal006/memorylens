"""
Medical patient-consultation benchmark scenario.

Models a patient in ongoing consultations with an AI health assistant.
All personas are synthetic.  Facts cover patient profile attributes; updates
simulate a real care progression (medication change, symptom evolution,
weight change).
"""

from typing import List

from memorylens.simulator.facts import Fact
from memorylens.simulator.scenarios.base import Scenario


MEDICAL_PERSONA_POOL: List[List[Fact]] = [
    # Persona 0 — Elena Vasquez (baseline)
    [
        Fact("name",               "Elena Vasquez",   injected_at=0),
        Fact("age",                "42",              injected_at=1),
        Fact("known allergy",      "penicillin",      injected_at=2),
        Fact("current medication", "lisinopril",      injected_at=4,  updated_at=45, updated_value="losartan"),
        Fact("main symptom",       "persistent headaches", injected_at=6, updated_at=55, updated_value="occasional dizziness"),
        Fact("blood type",         "O positive",      injected_at=8),
        Fact("chronic condition",  "hypertension",    injected_at=10),
        Fact("weight",             "70 kilograms",    injected_at=12, updated_at=70, updated_value="66 kilograms"),
    ],
    # Persona 1 — Samuel Adeyemi
    [
        Fact("name",               "Samuel Adeyemi",  injected_at=0),
        Fact("age",                "35",              injected_at=1),
        Fact("known allergy",      "sulfa drugs",     injected_at=2),
        Fact("current medication", "metformin",       injected_at=4,  updated_at=45, updated_value="insulin glargine"),
        Fact("main symptom",       "fatigue",         injected_at=6,  updated_at=55, updated_value="improved energy levels"),
        Fact("blood type",         "A negative",      injected_at=8),
        Fact("chronic condition",  "type 2 diabetes", injected_at=10),
        Fact("weight",             "88 kilograms",    injected_at=12, updated_at=70, updated_value="83 kilograms"),
    ],
    # Persona 2 — Mei Lin
    [
        Fact("name",               "Mei Lin",         injected_at=0),
        Fact("age",                "58",              injected_at=1),
        Fact("known allergy",      "latex",           injected_at=2),
        Fact("current medication", "atorvastatin",    injected_at=4,  updated_at=45, updated_value="rosuvastatin"),
        Fact("main symptom",       "joint stiffness", injected_at=6,  updated_at=55, updated_value="reduced morning stiffness"),
        Fact("blood type",         "B positive",      injected_at=8),
        Fact("chronic condition",  "osteoarthritis",  injected_at=10),
        Fact("weight",             "61 kilograms",    injected_at=12, updated_at=70, updated_value="63 kilograms"),
    ],
]


MEDICAL_FILLER_TURNS: List[str] = [
    "What is a normal resting heart rate?",
    "How much water should I drink per day?",
    "What are the early signs of the flu?",
    "Is it better to stretch before or after exercise?",
    "How many hours of sleep do adults need?",
    "What foods are high in iron?",
    "How does caffeine affect blood pressure?",
    "What is the difference between a virus and a bacterial infection?",
    "How often should I get a general health check-up?",
    "What are common causes of lower back pain?",
    "Is intermittent fasting safe for most people?",
    "What vitamins support immune function?",
    "How can I improve my posture while working at a desk?",
    "What is considered a healthy body mass index range?",
    "How do vaccines work?",
    "What are the benefits of regular walking?",
    "How can I reduce screen-related eye strain?",
    "What is the recommended daily amount of fibre?",
    "How does stress affect the immune system?",
    "What are good sources of omega-3 fatty acids?",
]


MEDICAL = Scenario(
    name="medical",
    description="Patient-consultation conversation (synthetic data): patient "
                "profile facts with care-progression updates (medication, symptom, weight).",
    persona_pool=MEDICAL_PERSONA_POOL,
    filler_turns=MEDICAL_FILLER_TURNS,
)
