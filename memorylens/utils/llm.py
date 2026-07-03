import os
import time

_client = None


def get_client():
    global _client
    if _client is None:
        try:
            from groq import Groq
        except ImportError as e:
            raise ImportError(
                'The groq package is required for LLM compression. '
                'Install it with: pip install "memorylens[groq]"'
            ) from e
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set")
        _client = Groq(api_key=api_key)
    return _client


def chat(
    messages: list,
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.1,
    max_tokens: int = 256,
    retries: int = 3,
) -> str:
    for attempt in range(retries):
        try:
            response = get_client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"[LLM_ERROR: {e}]"
    return "[LLM_ERROR: max retries]"


def count_tokens_approx(messages: list) -> int:
    return sum(len(m.get("content", "")) for m in messages) // 4
