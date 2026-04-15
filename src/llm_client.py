"""
Unified LLM client wrapper for SafeHarness 2.0.

Wraps the llm_api_openai.py functions into a consistent interface:
  call_model(model_name, messages) -> str | None
"""

import sys
from pathlib import Path

_CODE_ROOT = str(Path(__file__).resolve().parents[1])
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

import llm_api_openai  # noqa: E402

# ---------- dispatch table --------------------------------------------------
# "sample"-style functions take {"messages": [...], "tools": [...]} and return
# a response dict with an "answer" array.
# "convenience"-style functions take (messages, tools=) and return a string.

_SAMPLE_STYLE = "sample"
_CONVENIENCE_STYLE = "convenience"

_DISPATCH: dict[str, tuple] = {
    "doubao_seed2":     (llm_api_openai.get_doubao_seed_2p0_pro_thinking_high, _SAMPLE_STYLE),
    "claude_sonnet_46": (llm_api_openai.get_api_anthropic_claude_sonnet_4_6,   _SAMPLE_STYLE),
    "claude_opus_4":    (llm_api_openai.get_api_aws_claude_opus_4,             _SAMPLE_STYLE),
    "gpt5_chat":        (llm_api_openai.call_gpt5_chat,                        _CONVENIENCE_STYLE),
    "gpt5_response":    (llm_api_openai.call_gpt5_response,                    _CONVENIENCE_STYLE),
    "doubao_thinking":  (llm_api_openai.call_doubao_thinking,                  _CONVENIENCE_STYLE),
    "gemini_pro":       (llm_api_openai.call_gemini_pro,                       _CONVENIENCE_STYLE),
    "gpt_5_1":          (llm_api_openai.call_gpt_51,                           _CONVENIENCE_STYLE),
}

# Default model used throughout the framework
DEFAULT_MODEL = "doubao_seed2"


def _extract_text(response: dict) -> str | None:
    """Extract text answer from a sample-style API response dict."""
    if response is None:
        return None
    answer_list = response.get("answer", [])
    for item in answer_list:
        if item.get("type") == "text":
            value = item.get("value", "")
            if value:
                return value
    return None


def available_models() -> list[str]:
    """Return the list of available model short-names."""
    return list(_DISPATCH.keys())


def call_model(model_name: str, messages: list[dict],
               tools: list | None = None) -> str | None:
    """Unified LLM call interface.

    Args:
        model_name: A short name from the dispatch table (e.g. 'doubao_seed2').
        messages: OpenAI-format message list (system/user/assistant roles).
        tools: Optional tool definitions list.

    Returns:
        The model's text response, or None on failure.
    """
    entry = _DISPATCH.get(model_name)
    if entry is None:
        raise ValueError(
            f"Unknown model '{model_name}', available: {list(_DISPATCH.keys())}"
        )
    func, style = entry

    if style == _SAMPLE_STYLE:
        sample = {"messages": messages}
        if tools is not None:
            sample["tools"] = tools
        response = func(sample)
        return _extract_text(response)
    else:
        return func(messages, tools=tools)


def call_judge(messages: list[dict], judge_model: str | None = None) -> str | None:
    """Call the LLM-as-Judge model for safety evaluation."""
    model = judge_model or DEFAULT_MODEL
    return call_model(model, messages)
