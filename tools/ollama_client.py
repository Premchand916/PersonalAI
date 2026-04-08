import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

DEFAULT_OLLAMA_MODEL = "llama3.2:1b"
DEFAULT_OLLAMA_FALLBACK_MODEL = "llama3.2:1b"
LOW_MEMORY_MARKERS = (
    "requires more system memory",
    "not enough memory",
    "out of memory",
)


class LLMInvocationError(RuntimeError):
    """Raised when an Ollama-backed prompt cannot be completed."""


@lru_cache(maxsize=None)
def get_llm(model: str, temperature: float) -> ChatOllama:
    return ChatOllama(model=model, temperature=temperature)


def get_primary_model() -> str:
    return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL


def get_fallback_model(primary_model: str) -> str:
    fallback = os.getenv(
        "OLLAMA_FALLBACK_MODEL",
        DEFAULT_OLLAMA_FALLBACK_MODEL,
    ).strip()
    return fallback or primary_model


def _is_low_memory_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in LOW_MEMORY_MARKERS)


def _build_error_message(model: str, exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()

    if "requires more system memory" in lowered:
        return (
            f"Ollama model '{model}' needs more system memory than is currently "
            "available. Free up RAM, choose a smaller installed model, or point "
            "OLLAMA_MODEL and OLLAMA_FALLBACK_MODEL to a lighter model."
        )

    if "connection refused" in lowered or "failed to connect" in lowered:
        return (
            f"Could not connect to Ollama while using model '{model}'. "
            "Make sure the Ollama app or `ollama serve` is running."
        )

    if "not found" in lowered or "pull" in lowered:
        return (
            f"Ollama model '{model}' is unavailable locally. "
            f"Pull it first with `ollama pull {model}`."
        )

    return f"Ollama request failed for model '{model}': {message}"


def invoke_prompt(prompt: str, temperature: float = 0.0) -> str:
    primary_model = get_primary_model()
    fallback_model = get_fallback_model(primary_model)

    try:
        response = get_llm(primary_model, temperature).invoke(prompt)
        return response.content.strip()
    except Exception as exc:
        if fallback_model != primary_model and _is_low_memory_error(exc):
            print(
                "[Ollama] Primary model "
                f"'{primary_model}' exceeded local memory, retrying with "
                f"'{fallback_model}'."
            )
            try:
                response = get_llm(fallback_model, temperature).invoke(prompt)
                return response.content.strip()
            except Exception as fallback_exc:
                raise LLMInvocationError(
                    _build_error_message(fallback_model, fallback_exc)
                ) from fallback_exc

        raise LLMInvocationError(_build_error_message(primary_model, exc)) from exc
