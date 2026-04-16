"""
Unified LLM client wrapper for SafeHarness 2.0.

Wraps the llm_api.py functions into a consistent interface:
  call_model(model_name, messages) -> str | None
"""

import sys
from pathlib import Path

_CODE_ROOT = str(Path(__file__).resolve().parents[1])
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

import llm_api  # noqa: E402

# ---------- dispatch table --------------------------------------------------

_DISPATCH: dict[str, callable] = {
    "deepseek_chat":     llm_api.call_deepseek_chat,
    "deepseek_reasoner": llm_api.call_deepseek_reasoner,
}

# Default model used throughout the framework
DEFAULT_MODEL = "deepseek_chat"


def available_models() -> list[str]:
    """Return the list of available model short-names."""
    return list(_DISPATCH.keys())


def call_model(model_name: str, messages: list[dict]) -> str | None:
    """Unified LLM call interface.

    Args:
        model_name: A short name from the dispatch table (e.g. 'deepseek_chat').
        messages: OpenAI-format message list (system/user/assistant roles).

    Returns:
        The model's text response, or None on failure.
    """
    func = _DISPATCH.get(model_name)
    if func is None:
        raise ValueError(
            f"Unknown model '{model_name}', available: {list(_DISPATCH.keys())}"
        )
    return func(messages)


def call_judge(messages: list[dict], judge_model: str | None = None) -> str | None:
    """Call the LLM-as-Judge model for safety evaluation."""
    model = judge_model or DEFAULT_MODEL
    return call_model(model, messages)
