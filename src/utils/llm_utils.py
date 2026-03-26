import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_openrouter_api_key() -> str:
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY is not set in the environment or .env file.")
    return api_key


def get_model_name() -> str:
    model_name = os.getenv("MODEL_NAME")
    if not model_name:
        raise ValueError("MODEL_NAME is not set in the environment or .env file.")
    return model_name


def get_url() -> str:
    api = os.getenv("API_URL")
    if not api:
        raise ValueError("API_URL is not set in the environment or .env file.")
    return api


@lru_cache(maxsize=1)
def build_llm(
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> ChatOpenAI:
    """Create a ChatOpenAI client."""
    return ChatOpenAI(
        model=get_model_name(),
        api_key=get_openrouter_api_key(),
        base_url=get_url(),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def invoke_llm(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> dict:
    """Send a prompt to LLM. Returns {'text', 'tokens_in', 'tokens_out'}."""
    llm = build_llm(temperature=temperature, max_tokens=max_tokens)
    response = llm.invoke(prompt)
    text = getattr(response, "content", str(response))

    usage = getattr(response, "usage_metadata", None) or {}
    tokens_in = usage.get("input_tokens", 0) if isinstance(usage, dict) else 0
    tokens_out = usage.get("output_tokens", 0) if isinstance(usage, dict) else 0

    return {"text": text, "tokens_in": tokens_in, "tokens_out": tokens_out}
