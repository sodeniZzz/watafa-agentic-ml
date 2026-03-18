import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_openrouter_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set in the environment or .env file."
        )
    return api_key


def get_model_name() -> str:
    model_name = os.getenv("OPENROUTER_MODEL")
    if not model_name:
        raise ValueError("OPENROUTER_MODEL is not set in the environment or .env file.")
    return model_name


@lru_cache(maxsize=1)
def build_llm(
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> ChatOpenAI:
    """Create a ChatOpenAI client configured for OpenRouter."""
    return ChatOpenAI(
        model=get_model_name(),
        api_key=get_openrouter_api_key(),
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        max_tokens=max_tokens,
    )


def invoke_llm(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> str:
    """Send a single text prompt to the LLM and return plain text."""
    llm = build_llm(temperature=temperature, max_tokens=max_tokens)
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))
