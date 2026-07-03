"""
Customer-support benchmark scenario.

Models a customer working through support tickets with an AI agent over many
turns.  Facts cover account and product attributes; updates simulate real
support lifecycles (plan upgrades, issue re-categorisation, renewal changes).
"""

from typing import List

from memorylens.simulator.facts import Fact
from memorylens.simulator.scenarios.base import Scenario


SUPPORT_PERSONA_POOL: List[List[Fact]] = [
    # Persona 0 — Dana Whitfield (baseline)
    [
        Fact("name",              "Dana Whitfield",  injected_at=0),
        Fact("account tier",      "basic plan",      injected_at=1,  updated_at=45, updated_value="premium plan"),
        Fact("product",           "CloudVault backup", injected_at=2),
        Fact("order number",      "ORD-88213",       injected_at=4),
        Fact("reported issue",    "login failure",   injected_at=6,  updated_at=55, updated_value="billing discrepancy"),
        Fact("operating system",  "Windows 11",      injected_at=8),
        Fact("renewal date",      "March 15",        injected_at=10, updated_at=70, updated_value="September 15"),
        Fact("preferred contact channel", "email",   injected_at=12),
    ],
    # Persona 1 — Ravi Patel
    [
        Fact("name",              "Ravi Patel",      injected_at=0),
        Fact("account tier",      "trial plan",      injected_at=1,  updated_at=45, updated_value="business plan"),
        Fact("product",           "SyncDrive storage", injected_at=2),
        Fact("order number",      "ORD-55901",       injected_at=4),
        Fact("reported issue",    "sync conflict",   injected_at=6,  updated_at=55, updated_value="quota exceeded error"),
        Fact("operating system",  "macOS Sonoma",    injected_at=8),
        Fact("renewal date",      "June 1",          injected_at=10, updated_at=70, updated_value="December 1"),
        Fact("preferred contact channel", "phone",   injected_at=12),
    ],
    # Persona 2 — Ingrid Olsen
    [
        Fact("name",              "Ingrid Olsen",    injected_at=0),
        Fact("account tier",      "family plan",     injected_at=1,  updated_at=45, updated_value="enterprise plan"),
        Fact("product",           "MailGuard filter", injected_at=2),
        Fact("order number",      "ORD-30447",       injected_at=4),
        Fact("reported issue",    "spam leakage",    injected_at=6,  updated_at=55, updated_value="false positive blocking"),
        Fact("operating system",  "Ubuntu 24.04",    injected_at=8),
        Fact("renewal date",      "January 20",      injected_at=10, updated_at=70, updated_value="July 20"),
        Fact("preferred contact channel", "live chat", injected_at=12),
    ],
]


SUPPORT_FILLER_TURNS: List[str] = [
    "How do I reset my password?",
    "Where can I download my invoices?",
    "Is there a mobile app for this service?",
    "What is your refund policy?",
    "How do I enable two-factor authentication?",
    "Can I share my account with a family member?",
    "What happens to my data if I cancel?",
    "How do I export my data?",
    "Why is the app asking me to re-login every day?",
    "Do you offer discounts for annual billing?",
    "How do I change the language of the interface?",
    "What browsers do you officially support?",
    "How long are backups retained?",
    "Can I schedule automatic reports?",
    "Is my data encrypted at rest?",
    "How do I add another user to my workspace?",
    "What is the difference between archive and delete?",
    "How do I contact a human agent?",
    "Are there API rate limits on my plan?",
    "How do I update my payment method?",
]


CUSTOMER_SUPPORT = Scenario(
    name="support",
    description="Customer-support ticket conversation: account facts with "
                "lifecycle updates (plan upgrade, issue re-categorisation, renewal change).",
    persona_pool=SUPPORT_PERSONA_POOL,
    filler_turns=SUPPORT_FILLER_TURNS,
)
