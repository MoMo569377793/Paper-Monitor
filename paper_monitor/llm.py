from __future__ import annotations

import json
import logging
import os
import re
import time
import http.client
import urllib.error
import urllib.request
from typing import Any

from paper_monitor.models import (
    LLMConfig,
    LLMResult,
    PaperRecord,
    ReportEntry,
    TopicDigest,
    TopicEvaluation,
)
from paper_monitor.prompts import PromptLibrary
from paper_monitor.utils import shorten, unique_strings


LOGGER = logging.getLogger(__name__)
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
DEFAULT_LLM_USER_AGENT = "paper-monitor/0.1 (+local)"
THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


class LLMClient:
    def __init__(self, config: LLMConfig, prompt_library: PromptLibrary | None = None) -> None:
        self.config = config
        self.provider = config.provider.strip().lower()
        self.prompt_library = prompt_library
        self.api_key = self._resolve_api_key()
        requires_auth = "api.openai.com" in config.base_url or self.provider in {"openai_responses", "openai", "responses"}
        self.enabled = bool(config.enabled and config.base_url and config.model and (self.api_key or not requires_auth))
        if config.enabled and requires_auth and not self.api_key:
            LOGGER.warning(
                "LLM is configured but no API key was found. Tried env vars: %s",
                ", ".join(self._candidate_api_key_envs()),
            )

    def generate_summary(self, paper: PaperRecord, evaluations: list[TopicEvaluation]) -> LLMResult | None:
        if not self.enabled:
            return None

        prefer_compact = self._prefers_compact_mode()
        parsed = None
        usage: dict[str, Any] = {}
        if prefer_compact:
            parsed, usage = self._request_structured_json(
                system_prompt=(
                    self._paper_summary_system_prompt()
                    + " 不要输出思考过程，不要输出<think>标签，不要输出任何解释文本，只输出最终 JSON。"
                ),
                user_prompt=self._build_compact_paper_prompt(paper, evaluations),
                schema_name="paper_summary",
                schema=self._paper_summary_schema(),
                max_output_tokens=max(self.config.max_output_tokens, 1200),
            )
        if not parsed:
            parsed, usage = self._request_structured_json(
                system_prompt=self._paper_summary_system_prompt(),
                user_prompt=self._build_paper_prompt(paper, evaluations),
                schema_name="paper_summary",
                schema=self._paper_summary_schema(),
            )
        if not parsed and not prefer_compact:
            parsed, usage = self._request_structured_json(
                system_prompt=(
                    self._paper_summary_system_prompt()
                    + " 不要输出思考过程，不要输出<think>标签，不要输出任何解释文本，只输出最终 JSON。"
                ),
                user_prompt=self._build_compact_paper_prompt(paper, evaluations),
                schema_name="paper_summary",
                schema=self._paper_summary_schema(),
                max_output_tokens=max(self.config.max_output_tokens, 1200),
            )
        if not parsed:
            return None

        parsed["usage"] = usage
        tags = unique_strings(parsed.get("tags", []))
        summary_text = self._compose_summary(parsed)
        basis = self._normalize_basis(parsed.get("basis"), has_fulltext=bool(paper.fulltext_excerpt))
        return LLMResult(
            summary_text=summary_text,
            summary_basis=basis,
            tags=tags[:10],
            structured=parsed,
        )

    def generate_topic_digest(self, topic_name: str, description: str, entries: list[ReportEntry]) -> TopicDigest | None:
        if not self.enabled or not self.config.enable_topic_digest or not entries:
            return None

        prefer_compact = self._prefers_compact_mode()
        limited_entries = entries[: self.config.topic_digest_entry_limit]
        compact_entries = entries[: min(4, len(entries))]
        system_prompt = self._topic_digest_system_prompt()
        parsed = None
        usage: dict[str, Any] = {}
        if prefer_compact:
            parsed, usage = self._request_structured_json(
                system_prompt=(
                    "你是一个严谨的中文研究情报分析助手。"
                    "不要输出思考过程，不要输出<think>标签，不要输出任何解释文本，只输出最终 JSON。"
                    "内容要短，突出趋势、代表工作和建议关注点。"
                ),
                user_prompt=self._build_compact_topic_digest_prompt(topic_name, description, compact_entries),
                schema_name="topic_digest",
                schema=self._topic_digest_schema(),
                max_output_tokens=max(self.config.max_output_tokens, 1400),
            )
        if not parsed:
            user_prompt = self._build_topic_digest_prompt(topic_name, description, limited_entries)
            parsed, usage = self._request_structured_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema_name="topic_digest",
                schema=self._topic_digest_schema(),
            )
        if not parsed:
            parsed, usage = self._request_structured_json(
                system_prompt=(
                    system_prompt
                    + " 不要输出思考过程，不要输出<think>标签，不要输出任何解释文本，只输出最终 JSON。"
                ),
                user_prompt=user_prompt,
                schema_name="topic_digest",
                schema=self._topic_digest_schema(),
                max_output_tokens=max(self.config.max_output_tokens, 1800),
            )
        if not parsed and not prefer_compact:
            parsed, usage = self._request_structured_json(
                system_prompt=(
                    "你是一个严谨的中文研究情报分析助手。"
                    "不要输出思考过程，不要输出<think>标签，不要输出任何解释文本，只输出最终 JSON。"
                    "内容要短，突出趋势、代表工作和建议关注点。"
                ),
                user_prompt=self._build_compact_topic_digest_prompt(topic_name, description, compact_entries),
                schema_name="topic_digest",
                schema=self._topic_digest_schema(),
                max_output_tokens=max(self.config.max_output_tokens, 1400),
            )
        if not parsed:
            return None
        parsed["usage"] = usage
        return TopicDigest(
            overview=str(parsed.get("overview", "")).strip(),
            highlights=[str(item).strip() for item in parsed.get("highlights", []) if str(item).strip()],
            watchlist=[str(item).strip() for item in parsed.get("watchlist", []) if str(item).strip()],
            tags=unique_strings(parsed.get("tags", []))[:8],
            structured=parsed,
        )

    def _request_structured_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        max_output_tokens: int | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        if self.provider in {"openai_responses", "openai", "responses"}:
            content, usage = self._post_responses(
                system_prompt,
                user_prompt,
                schema_name,
                schema,
                max_output_tokens=max_output_tokens,
            )
        else:
            content, usage = self._post_chat_completions(
                system_prompt,
                user_prompt,
                schema_name,
                schema,
                max_output_tokens=max_output_tokens,
            )
        if not content:
            return None, usage
        parsed = self._parse_response_json(content)
        if isinstance(parsed, dict) and isinstance(parsed.get(schema_name), dict):
            return parsed.get(schema_name), usage
        return parsed, usage

    def _post_chat_completions(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        *,
        max_output_tokens: int | None = None,
    ) -> tuple[str | None, dict[str, Any]]:
        prompt_with_schema = (
            f"{user_prompt}\n\n"
            f"请只输出严格合法的 JSON，对象名为 {schema_name}，遵守以下 JSON Schema:\n"
            f"{json.dumps(schema, ensure_ascii=False)}"
        )
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": max_output_tokens or self.config.max_output_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_with_schema},
            ],
        }
        data = self._post_json(self.config.base_url.rstrip("/") + "/chat/completions", payload)
        if not data:
            return None, {}
        choices = data.get("choices", [])
        if not choices:
            return None, self._parse_usage(data)
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
        return str(content), self._parse_usage(data)

    def _post_responses(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        *,
        max_output_tokens: int | None = None,
    ) -> tuple[str | None, dict[str, Any]]:
        payload = {
            "model": self.config.model,
            "instructions": system_prompt,
            "input": user_prompt,
            "temperature": self.config.temperature,
            "max_output_tokens": max_output_tokens or self.config.max_output_tokens,
            "store": self.config.store,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        }
        data = self._post_json(self.config.base_url.rstrip("/") + "/responses", payload)
        if not data:
            return None, {}
        return self._extract_responses_text(data), self._parse_usage(data)

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": DEFAULT_LLM_USER_AGENT,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        for attempt in range(3):
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                last_error = exc
                error_body = self._read_error_body(exc)
                if exc.code in RETRYABLE_STATUS_CODES and attempt < 2:
                    time.sleep(1 + attempt)
                    continue
                LOGGER.warning(
                    "llm request failed: status=%s, body=%s",
                    exc.code,
                    shorten(error_body or str(exc), 320),
                )
                return None
            except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1 + attempt)
                    continue
                LOGGER.warning("llm request failed: %s", exc)
                return None
        else:
            LOGGER.warning("llm request failed after retries: %s", last_error or "unknown error")
            return None

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            LOGGER.warning("llm returned non-json response")
            return None

    def _extract_responses_text(self, data: dict[str, Any]) -> str | None:
        output = data.get("output", [])
        texts: list[str] = []
        for item in output:
            content = item.get("content", []) if isinstance(item, dict) else []
            for chunk in content:
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("type") == "output_text" and chunk.get("text"):
                    texts.append(str(chunk["text"]))
        if texts:
            return "\n".join(texts)
        if isinstance(data.get("output_text"), str):
            return str(data["output_text"])
        return None

    def _parse_response_json(self, response_text: str) -> dict[str, Any] | None:
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        cleaned = THINK_TAG_RE.sub("", cleaned).strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            for index, char in enumerate(cleaned):
                if char != "{":
                    continue
                try:
                    parsed, _ = decoder.raw_decode(cleaned[index:])
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed
            LOGGER.warning("llm output is not valid json")
            return None
        return parsed if isinstance(parsed, dict) else None

    def _parse_usage(self, data: dict[str, Any]) -> dict[str, Any]:
        usage = data.get("usage", {})
        if not isinstance(usage, dict):
            return {}
        result: dict[str, Any] = {"raw": usage}
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
        total_tokens = usage.get("total_tokens")
        if input_tokens is not None:
            result["input_tokens"] = input_tokens
        if output_tokens is not None:
            result["output_tokens"] = output_tokens
        if total_tokens is not None:
            result["total_tokens"] = total_tokens
        return result

    def _candidate_api_key_envs(self) -> list[str]:
        candidates = [self.config.api_key_env]
        if self.provider in {"openai_responses", "openai", "responses"}:
            candidates.extend(["OPENAI_API_KEY", "LLM_API_KEY"])
        return unique_strings(candidates)

    def _resolve_api_key(self) -> str:
        direct_key = self.config.api_key_env.strip()
        if direct_key and not ENV_NAME_RE.match(direct_key):
            LOGGER.warning(
                "llm.api_key_env 看起来像直接填写了 API key。当前会兼容使用，但更建议改成环境变量名。"
            )
            return direct_key
        for env_name in self._candidate_api_key_envs():
            value = os.environ.get(env_name, "")
            if value:
                return value
        return ""

    def _read_error_body(self, exc: urllib.error.HTTPError) -> str:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            return ""
        return body.strip()

    def _build_paper_prompt(self, paper: PaperRecord, evaluations: list[TopicEvaluation]) -> str:
        relevant_topics = [item for item in evaluations if item.classification != "irrelevant"]
        topics_text = "\n".join(
            f"- {item.topic_name} / score={item.score} / keywords={', '.join(item.matched_keywords[:6])}"
            for item in relevant_topics[:3]
        )
        fulltext = paper.fulltext_excerpt or ""
        if len(fulltext) > self.config.max_input_chars:
            fulltext = fulltext[: self.config.max_input_chars]
        abstract = paper.abstract or ""
        context = {
            "title": paper.title,
            "authors": ", ".join(paper.authors[:8]) or "未知",
            "venue": paper.venue or "未知",
            "published_at": paper.published_at or "未知",
            "topics_text": topics_text or "- 无",
            "abstract": shorten(abstract, min(self.config.max_input_chars, 4000)) or "无",
            "fulltext": fulltext or "无",
        }
        if self.prompt_library:
            return self.prompt_library.paper_summary_user(context)
        return (
            "请仅依据提供的信息进行总结，不要虚构实验结果。\n"
            "summary 需要 120-220 字中文；contributions 和 limitations 各给 2-4 条短句数组；tags 给 3-8 个关键词。\n\n"
            f"标题: {context['title']}\n"
            f"作者: {context['authors']}\n"
            f"发表信息: venue={context['venue']}, published_at={context['published_at']}\n"
            f"相关主题:\n{context['topics_text']}\n\n"
            f"摘要:\n{context['abstract']}\n\n"
            f"全文节选:\n{context['fulltext']}\n"
        )

    def _build_topic_digest_prompt(self, topic_name: str, description: str, entries: list[ReportEntry]) -> str:
        paper_blocks: list[str] = []
        for index, entry in enumerate(entries, start=1):
            paper_blocks.append(
                (
                    f"{index}. 标题: {entry.paper.title}\n"
                    f"相关性: {entry.classification} / {entry.score}\n"
                    f"匹配词: {', '.join(entry.matched_keywords[:8]) or '无'}\n"
                    f"总结: {entry.paper.summary_text or '无'}\n"
                    f"摘要片段: {shorten(entry.paper.fulltext_excerpt or entry.paper.abstract or '无', 360)}\n"
                )
            )
        joined = "\n".join(paper_blocks)
        context = {
            "topic_name": topic_name,
            "description": description,
            "paper_count": str(len(entries)),
            "paper_blocks": joined,
        }
        if self.prompt_library:
            return self.prompt_library.topic_digest_user(context)
        return (
            f"主题: {topic_name}\n"
            f"说明: {description}\n"
            f"论文数量: {len(entries)}\n\n"
            "请总结这个主题在当前时间窗口的趋势。highlights 应是 2-4 条关键观察，"
            "watchlist 应是 2-4 条建议关注的论文或技术线索。\n\n"
            f"{joined}"
        )

    def _build_compact_topic_digest_prompt(self, topic_name: str, description: str, entries: list[ReportEntry]) -> str:
        paper_blocks: list[str] = []
        for index, entry in enumerate(entries, start=1):
            paper_blocks.append(
                (
                    f"{index}. 标题: {entry.paper.title}\n"
                    f"关键词: {', '.join(entry.matched_keywords[:5]) or '无'}\n"
                    f"摘要: {shorten(entry.paper.summary_text or entry.paper.abstract or '无', 160)}\n"
                )
            )
        joined = "\n".join(paper_blocks)
        return (
            f"主题: {topic_name}\n"
            f"说明: {description}\n"
            f"样本论文数: {len(entries)}\n\n"
            "请只根据这些论文，输出一个简洁主题摘要。"
            "overview 控制在 120-220 字，highlights 和 watchlist 各 2-3 条短句，tags 给 3-6 个短关键词。\n\n"
            f"{joined}"
        )

    def _build_compact_paper_prompt(self, paper: PaperRecord, evaluations: list[TopicEvaluation]) -> str:
        relevant_topics = [item for item in evaluations if item.classification != "irrelevant"]
        topics_text = "\n".join(
            f"- {item.topic_name} / score={item.score} / keywords={', '.join(item.matched_keywords[:5])}"
            for item in relevant_topics[:3]
        )
        abstract = shorten(paper.abstract or "", min(self.config.max_input_chars, 1200)) or "无"
        compact_fulltext = shorten(paper.fulltext_excerpt or "", 800) or "无"
        return (
            "请根据标题、摘要和关键信号，输出中文 JSON。"
            "summary 聚焦解决的问题、核心方法和应用领域，不要虚构实验数据。\n\n"
            f"标题: {paper.title}\n"
            f"相关主题:\n{topics_text or '- 无'}\n\n"
            f"摘要:\n{abstract}\n\n"
            f"全文节选:\n{compact_fulltext}\n"
        )

    def _prefers_compact_mode(self) -> bool:
        model_name = self.config.model.strip().lower()
        provider_name = self.provider.strip().lower()
        return "minimax" in model_name or "m2.5" in model_name or "minimax" in provider_name

    def _paper_summary_system_prompt(self) -> str:
        if self.prompt_library:
            return self.prompt_library.paper_summary_system()
        return "你是一个严谨的中文论文分析助手。请根据提供的标题、摘要、全文节选和相关性标签，输出简洁、具体、避免空话的 JSON。"

    def _topic_digest_system_prompt(self) -> str:
        if self.prompt_library:
            return self.prompt_library.topic_digest_system()
        return "你是一个严谨的中文研究情报分析助手。请根据一个主题下的多篇论文信息，生成简洁的中文主题摘要，突出趋势、代表工作和建议关注点。"

    def _compose_summary(self, parsed: dict[str, Any]) -> str:
        parts: list[str] = []
        summary = str(parsed.get("summary", "")).strip()
        if summary:
            parts.append(summary)
        problem = str(parsed.get("problem", "")).strip()
        if problem:
            parts.append(f"问题：{problem}")
        method = str(parsed.get("method", "")).strip()
        if method:
            parts.append(f"方法：{method}")
        contributions = [str(item).strip() for item in parsed.get("contributions", []) if str(item).strip()]
        if contributions:
            parts.append(f"贡献：{'；'.join(contributions[:3])}")
        limitations = [str(item).strip() for item in parsed.get("limitations", []) if str(item).strip()]
        if limitations:
            parts.append(f"局限：{'；'.join(limitations[:2])}")
        return " ".join(parts) if parts else "LLM 未返回可用总结。"

    def _normalize_basis(self, basis: Any, *, has_fulltext: bool) -> str:
        value = str(basis or "").strip().lower()
        allowed = {"llm+fulltext+metadata", "llm+abstract+metadata"}
        if value in allowed:
            return value
        return "llm+fulltext+metadata" if has_fulltext else "llm+abstract+metadata"

    def _paper_summary_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summary": {"type": "string"},
                "problem": {"type": "string"},
                "method": {"type": "string"},
                "contributions": {"type": "array", "items": {"type": "string"}},
                "limitations": {"type": "array", "items": {"type": "string"}},
                "tags": {"type": "array", "items": {"type": "string"}},
                "basis": {"type": "string", "enum": ["llm+fulltext+metadata", "llm+abstract+metadata"]},
            },
            "required": ["summary", "problem", "method", "contributions", "limitations", "tags", "basis"],
        }

    def _topic_digest_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "overview": {"type": "string"},
                "highlights": {"type": "array", "items": {"type": "string"}},
                "watchlist": {"type": "array", "items": {"type": "string"}},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["overview", "highlights", "watchlist", "tags"],
        }
