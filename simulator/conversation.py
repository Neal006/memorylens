from typing import List, Dict, Optional
from .facts import Fact, BENCHMARK_FACTS

FILLER_TURNS = [
    "Can you explain the difference between REST and GraphQL APIs?",
    "What are some best practices for writing clean Python code?",
    "I need help understanding Docker volumes.",
    "Tell me about the CAP theorem in distributed systems.",
    "How do I optimize a slow SQL query?",
    "What's the difference between supervised and unsupervised learning?",
    "Can you explain closures in JavaScript?",
    "What are microservices and when should I use them?",
    "How does React's virtual DOM work?",
    "What is the time complexity of quicksort?",
    "Can you recommend resources for learning Kubernetes?",
    "What is eventual consistency in databases?",
    "How does garbage collection work in Python?",
    "What's the difference between authentication and authorization?",
    "Can you explain the SOLID principles?",
    "What are WebSockets used for?",
    "How do I handle race conditions in concurrent code?",
    "What's the purpose of a message queue like Kafka?",
    "Can you explain database indexing?",
    "What is the difference between a stack and a heap?",
]


def generate_conversation(
    facts: List[Fact],
    total_turns: int,
    filler_turns: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Generate a list of conversation events across `total_turns` turns.
    Each event: {turn, role, content, is_fact, fact_key, is_update}

    Parameters
    ----------
    filler_turns : optional list of domain-specific filler questions.
                   Defaults to the generic tech-QA FILLER_TURNS list.
                   Pass a domain-specific list (e.g. EDTECH_FILLER_TURNS) to
                   run a scenario benchmark in a different conversational domain.
    """
    _fillers = filler_turns if filler_turns else FILLER_TURNS

    injection_map: Dict[int, Fact] = {f.injected_at: f for f in facts}
    update_map: Dict[int, Fact] = {
        f.updated_at: f for f in facts if f.updated_at is not None
    }

    events = []
    filler_idx = 0

    for turn in range(total_turns):
        if turn in injection_map:
            fact = injection_map[turn]
            events.append({
                "turn": turn, "role": "user",
                "content": fact.injection_text(),
                "is_fact": True, "fact_key": fact.key, "is_update": False,
            })
        elif turn in update_map:
            fact = update_map[turn]
            events.append({
                "turn": turn, "role": "user",
                "content": fact.update_text(),
                "is_fact": True, "fact_key": fact.key, "is_update": True,
            })
        else:
            msg = _fillers[filler_idx % len(_fillers)]
            filler_idx += 1
            events.append({
                "turn": turn, "role": "user",
                "content": msg,
                "is_fact": False, "fact_key": None, "is_update": False,
            })

    return events
