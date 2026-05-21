"""
LLM-as-Judge evaluation — optional module that uses Groq to assess
answer quality beyond simple content matching.

Only called when GROQ_API_KEY is set and the user explicitly enables
judge mode. All primary benchmark metrics remain content-based.
"""

from typing import Dict, List, Optional
from utils.llm import chat
from memory.base import BaseMemory
from simulator.facts import Fact

JUDGE_SYSTEM = """You are a strict evaluator. Given a question, the correct answer, and a model's response,
output ONLY a JSON object with two keys:
  "correct": true or false
  "reason": one short sentence explaining why
Do not output anything else."""


def judge_answer(
    question: str,
    expected: str,
    actual: str,
    model: str = "llama-3.1-8b-instant",
) -> Dict:
    """Ask the LLM to judge whether `actual` correctly answers `question`."""
    prompt = f"Question: {question}\nCorrect answer: {expected}\nModel response: {actual}"
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user",   "content": prompt},
    ]
    raw = chat(messages, model=model, temperature=0.0, max_tokens=80)

    import json, re
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(match.group()) if match else {}
        return {
            "correct": bool(parsed.get("correct", False)),
            "reason":  parsed.get("reason", raw[:120]),
        }
    except Exception:
        return {"correct": False, "reason": raw[:120]}


def llm_recall(
    memory: BaseMemory,
    fact: Fact,
    current_turn: int,
    model: str = "llama-3.1-8b-instant",
) -> Dict:
    """
    Full LLM-answer recall: retrieve context, ask the LLM, then judge the answer.
    Returns recall result enriched with judge verdict.
    """
    context = memory.get_context(fact.query_text(), current_turn)

    # Build message list for the LLM to answer the question
    clean_ctx: List[Dict] = []
    for m in context:
        if m["role"] in ("system", "user", "assistant"):
            clean_ctx.append(m)

    # Ensure conversation ends with a user turn
    if not clean_ctx or clean_ctx[-1]["role"] != "user":
        clean_ctx.append({"role": "user", "content": fact.query_text()})
    else:
        clean_ctx[-1]["content"] += f"\n\n{fact.query_text()}"

    answer = chat(clean_ctx, model=model, max_tokens=60)
    expected = fact.current_value(current_turn)

    verdict = judge_answer(fact.query_text(), expected, answer, model=model)

    return {
        "recalled_llm":   verdict["correct"],
        "answer":         answer,
        "expected":       expected,
        "judge_reason":   verdict["reason"],
        "tokens":         memory.token_count(fact.query_text(), current_turn),
    }
