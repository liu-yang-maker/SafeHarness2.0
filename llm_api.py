# -*- coding: UTF-8 -*-

import json
import time

from openai import OpenAI


class InsufficientBalanceError(RuntimeError):
    """API 返回 402 / 余额不足时抛出，供上层终止批处理任务。"""


def _is_insufficient_balance(exc: BaseException) -> bool:
    if getattr(exc, "status_code", None) == 402:
        return True
    return "insufficient balance" in str(exc).lower()


BASE_URL = "https://api.deepseek.com"
API_KEY = "sk-xxxxxxxxxxxxxx"

# SDK 对连接超时、5xx 等可重试错误在 HTTP 层做指数退避重试（默认多为 2 次，此处略调高便于长实验）。
HTTP_MAX_RETRIES = 5

# 应用层：接口已成功返回但 content 为空、非 str 时，重新发起请求（指数退避间隔）。
EMPTY_RESPONSE_MAX_ATTEMPTS = 10
_EMPTY_RETRY_BASE_DELAY_SEC = 0.5


def _sleep_before_empty_retry(attempt_index: int) -> None:
    time.sleep(_EMPTY_RETRY_BASE_DELAY_SEC * (2**attempt_index))


def call_llm(messages, model_name, params=None, timeout=3600):
    """OpenAI-compatible API 调用（用于 DeepSeek）"""
    create_kw = params if params else {}
    last_bad: str | None = None
    try:
        client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
            timeout=timeout,
            max_retries=HTTP_MAX_RETRIES,
        )
    except Exception as e:
        if _is_insufficient_balance(e):
            raise InsufficientBalanceError(str(e)) from e
        print(f"Error occurred while calling LLM: {type(e).__name__}: {e!r}")
        return None

    for attempt in range(EMPTY_RESPONSE_MAX_ATTEMPTS):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **create_kw,
            )
            answer = completion.choices[0].message.content
        except Exception as e:
            if _is_insufficient_balance(e):
                raise InsufficientBalanceError(str(e)) from e
            print(f"Error occurred while calling LLM: {type(e).__name__}: {e!r}")
            return None

        if isinstance(answer, str) and answer != "":
            return answer

        last_bad = repr(answer)
        if attempt + 1 < EMPTY_RESPONSE_MAX_ATTEMPTS:
            print(
                f"LLM empty or non-string content ({last_bad}), "
                f"retry {attempt + 1}/{EMPTY_RESPONSE_MAX_ATTEMPTS} (model={model_name!r})"
            )
            _sleep_before_empty_retry(attempt)

    print(
        f"Error occurred while calling LLM: empty response after "
        f"{EMPTY_RESPONSE_MAX_ATTEMPTS} attempts, last content={last_bad}"
    )
    return None


# ── 便捷函数：DeepSeek 模型 ──

def call_deepseek_chat(messages):
    return call_llm(messages, model_name="deepseek-chat", timeout=3600)

def call_deepseek_reasoner(messages):
    return call_llm(messages, model_name="deepseek-reasoner", timeout=3600)


MODEL_FUNC_MAP = {
    "deepseek_chat": call_deepseek_chat,
    "deepseek_reasoner": call_deepseek_reasoner,
}


def call_by_name(model_short_name, messages):
    """通过简短的模型名调用，例如 call_by_name('deepseek_chat', messages)"""
    func = MODEL_FUNC_MAP.get(model_short_name)
    if func is None:
        raise ValueError(
            f"未知模型 '{model_short_name}'，可选: {list(MODEL_FUNC_MAP.keys())}"
        )
    return func(messages)


if __name__ == "__main__":
    test_messages = [
        {"role": "system", "content": "你是一个数学家，当用户询问你是谁，你要回答自己是一个数学家"},
        {"role": "user", "content": "你是谁"},
    ]

    for name, func in MODEL_FUNC_MAP.items():
        print(f"[{name}]", json.dumps(func(test_messages), ensure_ascii=False))
