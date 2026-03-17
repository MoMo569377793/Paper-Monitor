#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
DEFAULT_SYSTEM_PROMPT = "你是一个准确、简洁、重视事实依据的中文助手。"
DEFAULT_USER_AGENT = "paper-monitor-chat/0.1 (+local)"
REASONING_ERROR_HINTS = (
    "reasoning",
    "thinking",
    "unsupported",
    "unknown field",
    "unknown parameter",
    "invalid parameter",
    "extra inputs",
)
STRING_PREVIEW_LIMIT = 400


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用指定 config 中的 API 模型进行对话")
    parser.add_argument("--config", default="config/config-poe.json", help="配置文件路径")
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT, help="系统提示词")
    parser.add_argument("--message", help="单次提问；不传则进入交互式模式")
    parser.add_argument("--max-output-tokens", type=int, help="显式指定输出 token 上限；默认不向后端发送该字段")
    parser.add_argument("--temperature", type=float, help="覆盖配置中的 temperature")
    parser.add_argument(
        "--reasoning-effort",
        help="覆盖配置中的推理强度；默认读取 llm.model_reasoning_effort 或 llm.reasoning_effort",
    )
    parser.add_argument("--verbose", action="store_true", help="打印实际请求的 endpoint 和 payload 预览")
    parser.add_argument("--no-history", action="store_true", help="交互模式下不保留历史")
    return parser


def load_llm_config(config_path: str) -> dict[str, Any]:
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    llm = raw.get("llm", {})
    if not isinstance(llm, dict):
        raise ValueError("配置文件中的 llm 段无效。")
    return llm


def resolve_api_key(env_name_or_key: str) -> str:
    value = str(env_name_or_key or "").strip()
    if not value:
        return ""
    if ENV_NAME_RE.match(value):
        return os.environ.get(value, "")
    return value


def build_headers(api_key: str) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "User-Agent": DEFAULT_USER_AGENT,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout_seconds: int) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(3):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code in RETRYABLE_STATUS_CODES and attempt < 2:
                time.sleep(1 + attempt)
                continue
            raise RuntimeError(f"HTTP {exc.code}: {error_body}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1 + attempt)
                continue
            raise RuntimeError(str(exc)) from exc
    raise RuntimeError(str(last_error or "unknown error"))


def extract_chat_content(data: dict[str, Any]) -> str:
    choices = data.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        return "\n".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content).strip()
    return str(content).strip()


def extract_responses_text(data: dict[str, Any]) -> str:
    output = data.get("output", [])
    texts: list[str] = []
    for item in output:
        content = item.get("content", []) if isinstance(item, dict) else []
        for chunk in content:
            if isinstance(chunk, dict) and chunk.get("type") == "output_text" and chunk.get("text"):
                texts.append(str(chunk["text"]))
    if texts:
        return "\n".join(texts).strip()
    output_text = data.get("output_text")
    return str(output_text).strip() if output_text else ""


def parse_usage(data: dict[str, Any]) -> dict[str, Any]:
    usage = data.get("usage", {})
    if not isinstance(usage, dict):
        return {}
    return {
        "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
        "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
        "total_tokens": usage.get("total_tokens"),
    }


def should_retry_without_reasoning(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return any(hint in message for hint in REASONING_ERROR_HINTS)


def map_reasoning_effort_to_poe_thinking_level(reasoning_effort: str) -> str:
    value = reasoning_effort.strip().lower()
    if not value:
        return ""
    if value in {"high", "xhigh", "max"}:
        return "high"
    return "low"


def map_reasoning_effort_to_poe_output_effort(reasoning_effort: str) -> str:
    value = reasoning_effort.strip().lower()
    if not value:
        return ""
    if value in {"low", "medium", "high", "xhigh", "max"}:
        return "max" if value in {"xhigh", "max"} else value
    return value


def is_claude_model(model: str) -> bool:
    return model.strip().lower().startswith("claude")


def truncate_for_preview(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) <= STRING_PREVIEW_LIMIT:
            return value
        return value[:STRING_PREVIEW_LIMIT] + "...<truncated>"
    if isinstance(value, list):
        return [truncate_for_preview(item) for item in value]
    if isinstance(value, dict):
        return {key: truncate_for_preview(item) for key, item in value.items()}
    return value


def print_verbose_request(url: str, payload: dict[str, Any]) -> None:
    print(f"[verbose] endpoint={url}", file=sys.stderr)
    print(
        "[verbose] payload="
        + json.dumps(truncate_for_preview(payload), ensure_ascii=False, indent=2),
        file=sys.stderr,
    )


def call_model(
    llm: dict[str, Any],
    *,
    system_prompt: str,
    messages: list[dict[str, str]],
    max_output_tokens: int | None,
    temperature: float | None,
    reasoning_effort: str | None,
    verbose: bool,
) -> tuple[str, dict[str, Any]]:
    provider = str(llm.get("provider", "")).strip().lower()
    base_url = str(llm.get("base_url", "")).rstrip("/")
    model = str(llm.get("model", "")).strip()
    timeout_seconds = int(llm.get("timeout_seconds", 60))
    api_key = resolve_api_key(str(llm.get("api_key_env", "")).strip())
    if not base_url or not model:
        raise RuntimeError("配置文件中的 llm.base_url 或 llm.model 为空。")

    headers = build_headers(api_key)
    chosen_temperature = temperature if temperature is not None else float(llm.get("temperature", 0.2))
    chosen_max_tokens = max_output_tokens
    chosen_reasoning_effort = (
        reasoning_effort
        if reasoning_effort is not None
        else str(llm.get("model_reasoning_effort") or llm.get("reasoning_effort") or "").strip()
    )
    chosen_output_effort = str(llm.get("model_output_effort") or llm.get("output_effort") or "").strip()
    chosen_thinking_level = str(llm.get("model_thinking_level") or llm.get("thinking_level") or "").strip()
    is_poe_api = "api.poe.com" in base_url.lower()
    is_gemini_model = model.lower().startswith("gemini")

    if provider in {"openai_responses", "openai", "responses"}:
        transcript = "\n\n".join(
            f"{'用户' if item['role'] == 'user' else '助手'}: {item['content']}"
            for item in messages
        )
        payload = {
            "model": model,
            "instructions": system_prompt,
            "input": transcript,
            "temperature": chosen_temperature,
            "store": bool(llm.get("store", False)),
        }
        if chosen_max_tokens is not None:
            payload["max_output_tokens"] = chosen_max_tokens
        if chosen_reasoning_effort:
            payload["reasoning"] = {"effort": chosen_reasoning_effort}
        if verbose:
            print_verbose_request(f"{base_url}/responses", payload)
        try:
            data = post_json(f"{base_url}/responses", payload, headers, timeout_seconds)
        except RuntimeError as exc:
            if chosen_reasoning_effort and should_retry_without_reasoning(exc):
                payload.pop("reasoning", None)
                if verbose:
                    print("[verbose] retrying without reasoning field", file=sys.stderr)
                    print_verbose_request(f"{base_url}/responses", payload)
                data = post_json(f"{base_url}/responses", payload, headers, timeout_seconds)
            else:
                raise
        return extract_responses_text(data), parse_usage(data)

    payload_messages: list[dict[str, str]] = []
    if system_prompt.strip():
        payload_messages.append({"role": "system", "content": system_prompt.strip()})
    payload_messages.extend(messages)
    payload = {
        "model": model,
        "temperature": chosen_temperature,
        "messages": payload_messages,
    }
    if chosen_max_tokens is not None:
        payload["max_tokens"] = chosen_max_tokens
    if is_poe_api:
        extra_body = llm.get("extra_body", {})
        if not isinstance(extra_body, dict):
            extra_body = {}
        extra_body = dict(extra_body)
        if is_gemini_model:
            thinking_level = chosen_thinking_level or (
                map_reasoning_effort_to_poe_thinking_level(chosen_reasoning_effort)
                if chosen_reasoning_effort
                else ""
            )
            if thinking_level:
                extra_body["thinking_level"] = thinking_level
        elif is_claude_model(model):
            output_effort = (
                chosen_output_effort
                or str(extra_body.get("output_effort", "")).strip()
                or map_reasoning_effort_to_poe_output_effort(chosen_reasoning_effort)
            )
            if output_effort:
                extra_body["output_effort"] = output_effort
            extra_body.pop("reasoning_effort", None)
        elif chosen_reasoning_effort:
            extra_body["reasoning_effort"] = chosen_reasoning_effort
        if extra_body:
            payload["extra_body"] = extra_body
    elif chosen_reasoning_effort:
        payload["reasoning_effort"] = chosen_reasoning_effort
    if verbose:
        print_verbose_request(f"{base_url}/chat/completions", payload)
    try:
        data = post_json(f"{base_url}/chat/completions", payload, headers, timeout_seconds)
    except RuntimeError as exc:
        if chosen_reasoning_effort and should_retry_without_reasoning(exc):
            payload.pop("reasoning_effort", None)
            extra_body = payload.get("extra_body")
            if isinstance(extra_body, dict):
                extra_body.pop("reasoning_effort", None)
                if is_gemini_model:
                    extra_body.pop("thinking_level", None)
                if not extra_body:
                    payload.pop("extra_body", None)
            if verbose:
                print("[verbose] retrying without reasoning field", file=sys.stderr)
                print_verbose_request(f"{base_url}/chat/completions", payload)
            data = post_json(f"{base_url}/chat/completions", payload, headers, timeout_seconds)
        else:
            raise
    return extract_chat_content(data), parse_usage(data)


def print_usage(usage: dict[str, Any]) -> None:
    if not usage:
        return
    print(
        f"[token] in={usage.get('input_tokens', '未知')} "
        f"out={usage.get('output_tokens', '未知')} total={usage.get('total_tokens', '未知')}",
        file=sys.stderr,
    )


def run_single_turn(
    llm: dict[str, Any],
    prompt: str,
    system_prompt: str,
    max_output_tokens: int | None,
    temperature: float | None,
    reasoning_effort: str | None,
    verbose: bool,
) -> int:
    reply, usage = call_model(
        llm,
        system_prompt=system_prompt,
        messages=[{"role": "user", "content": prompt}],
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        verbose=verbose,
    )
    if not reply:
        print("模型未返回内容。", file=sys.stderr)
        return 1
    print(reply)
    print_usage(usage)
    return 0


def run_repl(
    llm: dict[str, Any],
    system_prompt: str,
    max_output_tokens: int | None,
    temperature: float | None,
    reasoning_effort: str | None,
    verbose: bool,
    keep_history: bool,
) -> int:
    label = str(llm.get("label") or llm.get("model") or llm.get("variant_id") or "unknown")
    print(f"已连接模型: {label}")
    print("输入 /exit 或 /quit 退出，输入 /clear 清空历史。")
    history: list[dict[str, str]] = []
    while True:
        try:
            user_text = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            return 0
        if user_text == "/clear":
            history.clear()
            print("历史已清空。")
            continue

        messages = history + [{"role": "user", "content": user_text}] if keep_history else [{"role": "user", "content": user_text}]
        try:
            reply, usage = call_model(
                llm,
                system_prompt=system_prompt,
                messages=messages,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                verbose=verbose,
            )
        except RuntimeError as exc:
            print(f"请求失败: {exc}", file=sys.stderr)
            continue
        if not reply:
            print("模型未返回内容。", file=sys.stderr)
            continue
        print(f"{label}> {reply}")
        print_usage(usage)
        if keep_history:
            history.extend(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": reply},
                ]
            )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    llm = load_llm_config(args.config)
    if not bool(llm.get("enabled", False)):
        print("当前配置里的 llm.enabled = false，无法对话。", file=sys.stderr)
        return 1
    if args.message:
        try:
            return run_single_turn(
                llm,
                args.message,
                args.system,
                args.max_output_tokens,
                args.temperature,
                args.reasoning_effort,
                args.verbose,
            )
        except RuntimeError as exc:
            print(f"请求失败: {exc}", file=sys.stderr)
            return 1
    return run_repl(
        llm,
        args.system,
        args.max_output_tokens,
        args.temperature,
        args.reasoning_effort,
        args.verbose,
        keep_history=not args.no_history,
    )


if __name__ == "__main__":
    raise SystemExit(main())
