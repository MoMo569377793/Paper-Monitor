from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
import http.client
import urllib.error
import urllib.request
from pathlib import Path
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
PDF_STRATEGY_RESPONSES = "responses_input_file"
PDF_STRATEGY_CHAT_FILE = "chat_file"
PDF_STRATEGY_CHAT_INPUT_FILE = "chat_input_file"
TASK_PAPER_SUMMARY = "paper_summary"
TASK_PDF_BRIEF = "pdf_brief"
TASK_PAPER_CHUNK = "paper_chunk"
TASK_PAPER_REDUCE = "paper_reduce"
TASK_TOPIC_DIGEST = "topic_digest"
TASK_JSON_REPAIR = "json_repair"


class LLMClient:
    def __init__(self, config: LLMConfig, prompt_library: PromptLibrary | None = None) -> None:
        self.config = config
        self.provider = config.provider.strip().lower()
        self.prompt_library = prompt_library
        self.api_key = self._resolve_api_key()
        self._pdf_input_strategy: str | None = None
        self._pdf_probe_attempted = False
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

        parsed = None
        usage: dict[str, Any] = {}
        fulltext = ""
        pdf_path = self._resolve_local_pdf_path(paper)
        pdf_strategy = self._ensure_pdf_input_strategy(pdf_path)
        direct_pdf_status = ""
        if pdf_path and pdf_strategy:
            parsed, usage = self._generate_summary_from_pdf(paper, evaluations, pdf_path, pdf_strategy)
            if parsed:
                LOGGER.info(
                    "llm summary route: paper_id=%s model=%s source_mode=pdf_direct strategy=%s",
                    paper.id,
                    self.config.label or self.config.model or self.config.variant_id,
                    pdf_strategy,
                )
            else:
                direct_pdf_status = "request_failed"
                LOGGER.warning(
                    "llm direct pdf summary failed for paper_id=%s model=%s via %s, fallback to extracted text",
                    paper.id,
                    self.config.label or self.config.model or self.config.variant_id,
                    pdf_strategy,
                )
        elif pdf_path:
            direct_pdf_status = "unsupported"
            LOGGER.warning(
                "llm direct pdf unavailable for paper_id=%s model=%s, fallback to extracted text",
                paper.id,
                self.config.label or self.config.model or self.config.variant_id,
            )
        elif self.config.pdf_input_mode.strip().lower() == "disable":
            direct_pdf_status = "disabled"
        else:
            direct_pdf_status = "no_local_pdf"
        fulltext = self._load_fulltext_text(paper)
        if fulltext:
            if not parsed:
                parsed, usage = self._generate_summary_from_fulltext(paper, evaluations, fulltext)
                if parsed:
                    LOGGER.info(
                        "llm summary route: paper_id=%s model=%s source_mode=fulltext_txt direct_pdf_status=%s",
                        paper.id,
                        self.config.label or self.config.model or self.config.variant_id,
                        direct_pdf_status or "not_attempted",
                    )
        if not parsed:
            prefer_compact = self._prefers_compact_mode()
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
                    task_name=TASK_PAPER_SUMMARY,
                )
                if parsed:
                    LOGGER.info(
                        "llm summary route: paper_id=%s model=%s source_mode=abstract_metadata",
                        paper.id,
                        self.config.label or self.config.model or self.config.variant_id,
                    )
            if not parsed:
                parsed, usage = self._request_structured_json(
                    system_prompt=self._paper_summary_system_prompt(),
                    user_prompt=self._build_paper_prompt(paper, evaluations),
                    schema_name="paper_summary",
                    schema=self._paper_summary_schema(),
                    task_name=TASK_PAPER_SUMMARY,
                )
                if parsed:
                    LOGGER.info(
                        "llm summary route: paper_id=%s model=%s source_mode=abstract_metadata",
                        paper.id,
                        self.config.label or self.config.model or self.config.variant_id,
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
                    task_name=TASK_PAPER_SUMMARY,
                )
                if parsed:
                    LOGGER.info(
                        "llm summary route: paper_id=%s model=%s source_mode=abstract_metadata",
                        paper.id,
                        self.config.label or self.config.model or self.config.variant_id,
                    )
        if not parsed:
            return None

        source_mode = str(parsed.get("source_mode", "")).strip().lower()
        if not source_mode:
            parsed["source_mode"] = "abstract_metadata"
        if pdf_path:
            parsed.setdefault("pdf_filename", pdf_path.name)
        if pdf_strategy:
            if parsed.get("source_mode") == "pdf_direct":
                parsed.setdefault("pdf_input_strategy", pdf_strategy)
            else:
                parsed.setdefault("direct_pdf_strategy", pdf_strategy)
        if direct_pdf_status and parsed.get("source_mode") != "pdf_direct":
            parsed.setdefault("direct_pdf_status", direct_pdf_status)

        parsed["usage"] = usage
        tags = unique_strings(parsed.get("tags", []))
        summary_text = self._compose_summary(parsed)
        basis = self._normalize_basis(
            parsed.get("basis"),
            source_mode=str(parsed.get("source_mode", "")).strip().lower(),
            has_fulltext=bool(fulltext or paper.fulltext_excerpt),
        )
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
                task_name=TASK_TOPIC_DIGEST,
            )
        if not parsed:
            user_prompt = self._build_topic_digest_prompt(topic_name, description, limited_entries)
            parsed, usage = self._request_structured_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema_name="topic_digest",
                schema=self._topic_digest_schema(),
                task_name=TASK_TOPIC_DIGEST,
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
                task_name=TASK_TOPIC_DIGEST,
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
                task_name=TASK_TOPIC_DIGEST,
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

    def _reasoning_effort_for_task(self, task_name: str) -> str:
        task_value = str(self.config.reasoning_by_task.get(task_name, "")).strip()
        if task_value:
            return task_value
        global_value = str(self.config.model_reasoning_effort).strip()
        if global_value:
            return global_value
        return str(self.config.extra_body.get("reasoning_effort", "")).strip()

    def _thinking_level_for_task(self, task_name: str) -> str:
        task_value = str(self.config.thinking_level_by_task.get(task_name, "")).strip()
        if task_value:
            return task_value
        global_value = str(self.config.model_thinking_level).strip()
        if global_value:
            return global_value
        configured_value = str(self.config.extra_body.get("thinking_level", "")).strip()
        if configured_value:
            return configured_value
        if self._is_poe_api() and self._is_gemini_model():
            reasoning_effort = self._reasoning_effort_for_task(task_name)
            if reasoning_effort:
                return self._map_reasoning_effort_to_poe_thinking_level(reasoning_effort)
        return ""

    def _is_poe_api(self) -> bool:
        return "api.poe.com" in self.config.base_url.lower()

    def _is_gemini_model(self) -> bool:
        return self.config.model.strip().lower().startswith("gemini")

    def _map_reasoning_effort_to_poe_thinking_level(self, reasoning_effort: str) -> str:
        value = reasoning_effort.strip().lower()
        if not value:
            return ""
        if value in {"high", "xhigh", "max"}:
            return "high"
        return "low"

    def _apply_chat_request_options(self, payload: dict[str, Any], *, task_name: str) -> dict[str, Any]:
        extra_body = dict(self.config.extra_body or {})
        reasoning_effort = self._reasoning_effort_for_task(task_name)
        thinking_level = self._thinking_level_for_task(task_name)

        if self._is_poe_api():
            if self._is_gemini_model():
                if thinking_level:
                    extra_body["thinking_level"] = thinking_level
            elif reasoning_effort and "reasoning_effort" not in extra_body:
                extra_body["reasoning_effort"] = reasoning_effort
        elif reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

        if extra_body:
            payload["extra_body"] = extra_body
        return payload

    def _apply_responses_request_options(self, payload: dict[str, Any], *, task_name: str) -> dict[str, Any]:
        reasoning_effort = self._reasoning_effort_for_task(task_name)
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        return payload

    def _request_structured_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        max_output_tokens: int | None = None,
        task_name: str = TASK_PAPER_SUMMARY,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        if self.provider in {"openai_responses", "openai", "responses"}:
            content, usage = self._post_responses(
                system_prompt,
                user_prompt,
                schema_name,
                schema,
                max_output_tokens=max_output_tokens,
                task_name=task_name,
            )
        else:
            content, usage = self._post_chat_completions(
                system_prompt,
                user_prompt,
                schema_name,
                schema,
                max_output_tokens=max_output_tokens,
                task_name=task_name,
            )
        if not content:
            return None, usage
        parsed = self._parse_response_json(content)
        if parsed is None:
            repaired, repair_usage = self._repair_response_json(content, schema_name, schema, task_name=TASK_JSON_REPAIR)
            if repaired is not None:
                return repaired, self._merge_usage_items([usage, repair_usage])
            return None, self._merge_usage_items([usage])
        if isinstance(parsed, dict) and isinstance(parsed.get(schema_name), dict):
            return parsed.get(schema_name), usage
        return parsed, usage

    def _request_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int | None = None,
        task_name: str = TASK_PAPER_CHUNK,
    ) -> tuple[str | None, dict[str, Any]]:
        if self.provider in {"openai_responses", "openai", "responses"}:
            content, usage = self._post_responses_text(
                system_prompt,
                user_prompt,
                max_output_tokens=max_output_tokens,
                task_name=task_name,
            )
        else:
            content, usage = self._post_chat_completions_text(
                system_prompt,
                user_prompt,
                max_output_tokens=max_output_tokens,
                task_name=task_name,
            )
        if not content:
            return None, usage
        return THINK_TAG_RE.sub("", str(content)).strip() or None, usage

    def _request_structured_json_with_pdf(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        pdf_path: Path,
        pdf_strategy: str,
        max_output_tokens: int | None = None,
        task_name: str = TASK_PAPER_SUMMARY,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        data_url = self._build_pdf_data_url(pdf_path)
        if not data_url:
            return None, {}
        if pdf_strategy == PDF_STRATEGY_RESPONSES:
            content, usage = self._post_responses_with_pdf(
                system_prompt,
                user_prompt,
                schema_name,
                schema,
                pdf_filename=pdf_path.name,
                pdf_data_url=data_url,
                max_output_tokens=max_output_tokens,
                task_name=task_name,
            )
        elif pdf_strategy in {PDF_STRATEGY_CHAT_FILE, PDF_STRATEGY_CHAT_INPUT_FILE}:
            content, usage = self._post_chat_completions_with_pdf(
                system_prompt,
                user_prompt,
                schema_name,
                schema,
                pdf_filename=pdf_path.name,
                pdf_data_url=data_url,
                file_item_type="file" if pdf_strategy == PDF_STRATEGY_CHAT_FILE else "input_file",
                max_output_tokens=max_output_tokens,
                task_name=task_name,
            )
        else:
            return None, {}
        if not content:
            return None, usage
        parsed = self._parse_response_json(content)
        if parsed is None:
            repaired, repair_usage = self._repair_response_json(content, schema_name, schema, task_name=TASK_JSON_REPAIR)
            if repaired is not None:
                return repaired, self._merge_usage_items([usage, repair_usage])
            return None, self._merge_usage_items([usage])
        if isinstance(parsed, dict) and isinstance(parsed.get(schema_name), dict):
            return parsed.get(schema_name), usage
        return parsed, usage

    def _request_text_with_pdf(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        pdf_path: Path,
        pdf_strategy: str,
        max_output_tokens: int | None = None,
        task_name: str = TASK_PAPER_SUMMARY,
    ) -> tuple[str | None, dict[str, Any]]:
        data_url = self._build_pdf_data_url(pdf_path)
        if not data_url:
            return None, {}
        if pdf_strategy == PDF_STRATEGY_RESPONSES:
            content, usage = self._post_responses_text_with_pdf(
                system_prompt,
                user_prompt,
                pdf_filename=pdf_path.name,
                pdf_data_url=data_url,
                max_output_tokens=max_output_tokens,
                task_name=task_name,
            )
        elif pdf_strategy in {PDF_STRATEGY_CHAT_FILE, PDF_STRATEGY_CHAT_INPUT_FILE}:
            content, usage = self._post_chat_completions_text_with_pdf(
                system_prompt,
                user_prompt,
                pdf_filename=pdf_path.name,
                pdf_data_url=data_url,
                file_item_type="file" if pdf_strategy == PDF_STRATEGY_CHAT_FILE else "input_file",
                max_output_tokens=max_output_tokens,
                task_name=task_name,
            )
        else:
            return None, {}
        if not content:
            return None, usage
        return THINK_TAG_RE.sub("", str(content)).strip() or None, usage

    def _repair_response_json(
        self,
        content: str,
        schema_name: str,
        schema: dict[str, Any],
        *,
        task_name: str = TASK_JSON_REPAIR,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        repair_system = (
            "你是一个 JSON 修复助手。"
            "你会把给定内容重写为严格符合 schema 的 JSON。"
            "不要输出解释，不要输出 Markdown，不要输出<think>标签，只输出最终 JSON。"
        )
        repair_user = (
            f"请把下面这段输出修复为严格合法的 JSON，对象名为 {schema_name}。\n"
            f"必须遵守以下 JSON Schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
            f"原始输出:\n{shorten(content, 8000)}"
        )
        if self.provider in {"openai_responses", "openai", "responses"}:
            repaired_text, usage = self._post_responses(
                repair_system,
                repair_user,
                schema_name,
                schema,
                max_output_tokens=max(self.config.max_output_tokens, 1000),
                task_name=task_name,
            )
        else:
            repaired_text, usage = self._post_chat_completions(
                repair_system,
                repair_user,
                schema_name,
                schema,
                max_output_tokens=max(self.config.max_output_tokens, 1000),
                task_name=task_name,
            )
        if not repaired_text:
            return None, usage
        parsed = self._parse_response_json(repaired_text)
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
        task_name: str = TASK_PAPER_SUMMARY,
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
        payload = self._apply_chat_request_options(payload, task_name=task_name)
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

    def _post_chat_completions_with_pdf(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        *,
        pdf_filename: str,
        pdf_data_url: str,
        file_item_type: str,
        max_output_tokens: int | None = None,
        task_name: str = TASK_PAPER_SUMMARY,
    ) -> tuple[str | None, dict[str, Any]]:
        prompt_with_schema = (
            f"{user_prompt}\n\n"
            f"请只输出严格合法的 JSON，对象名为 {schema_name}，遵守以下 JSON Schema:\n"
            f"{json.dumps(schema, ensure_ascii=False)}"
        )
        file_item: dict[str, Any]
        if file_item_type == "input_file":
            file_item = {
                "type": "input_file",
                "filename": pdf_filename,
                "file_data": pdf_data_url,
            }
        else:
            file_item = {
                "type": "file",
                "file": {
                    "filename": pdf_filename,
                    "file_data": pdf_data_url,
                },
            }
        content_items = [file_item, {"type": "text", "text": prompt_with_schema}]
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": max_output_tokens or self.config.max_output_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [file_item, {"type": "text", "text": prompt_with_schema}],
                },
            ],
        }
        payload = self._apply_chat_request_options(payload, task_name=task_name)
        data = self._post_json(self.config.base_url.rstrip("/") + "/chat/completions", payload, warn_on_error=False)
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

    def _post_chat_completions_text_with_pdf(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        pdf_filename: str,
        pdf_data_url: str,
        file_item_type: str,
        max_output_tokens: int | None = None,
        task_name: str = TASK_PAPER_SUMMARY,
    ) -> tuple[str | None, dict[str, Any]]:
        file_item: dict[str, Any]
        if file_item_type == "input_file":
            file_item = {
                "type": "input_file",
                "filename": pdf_filename,
                "file_data": pdf_data_url,
            }
        else:
            file_item = {
                "type": "file",
                "file": {
                    "filename": pdf_filename,
                    "file_data": pdf_data_url,
                },
            }
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": max_output_tokens or self.config.max_output_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        file_item,
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
        }
        payload = self._apply_chat_request_options(payload, task_name=task_name)
        data = self._post_json(self.config.base_url.rstrip("/") + "/chat/completions", payload, warn_on_error=False)
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

    def _post_chat_completions_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_output_tokens: int | None = None,
        task_name: str = TASK_PAPER_CHUNK,
    ) -> tuple[str | None, dict[str, Any]]:
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": max_output_tokens or self.config.max_output_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        payload = self._apply_chat_request_options(payload, task_name=task_name)
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
        task_name: str = TASK_PAPER_SUMMARY,
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
        payload = self._apply_responses_request_options(payload, task_name=task_name)
        data = self._post_json(self.config.base_url.rstrip("/") + "/responses", payload)
        if not data:
            return None, {}
        return self._extract_responses_text(data), self._parse_usage(data)

    def _post_responses_with_pdf(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        *,
        pdf_filename: str,
        pdf_data_url: str,
        max_output_tokens: int | None = None,
        task_name: str = TASK_PAPER_SUMMARY,
    ) -> tuple[str | None, dict[str, Any]]:
        payload = {
            "model": self.config.model,
            "instructions": system_prompt,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                        {
                            "type": "input_file",
                            "filename": pdf_filename,
                            "file_data": pdf_data_url,
                        },
                    ],
                }
            ],
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
        payload = self._apply_responses_request_options(payload, task_name=task_name)
        data = self._post_json(self.config.base_url.rstrip("/") + "/responses", payload, warn_on_error=False)
        if not data:
            return None, {}
        return self._extract_responses_text(data), self._parse_usage(data)

    def _post_responses_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_output_tokens: int | None = None,
        task_name: str = TASK_PAPER_CHUNK,
    ) -> tuple[str | None, dict[str, Any]]:
        payload = {
            "model": self.config.model,
            "instructions": system_prompt,
            "input": user_prompt,
            "temperature": self.config.temperature,
            "max_output_tokens": max_output_tokens or self.config.max_output_tokens,
            "store": self.config.store,
        }
        payload = self._apply_responses_request_options(payload, task_name=task_name)
        data = self._post_json(self.config.base_url.rstrip("/") + "/responses", payload)
        if not data:
            return None, {}
        return self._extract_responses_text(data), self._parse_usage(data)

    def _post_responses_text_with_pdf(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        pdf_filename: str,
        pdf_data_url: str,
        max_output_tokens: int | None = None,
        task_name: str = TASK_PAPER_SUMMARY,
    ) -> tuple[str | None, dict[str, Any]]:
        payload = {
            "model": self.config.model,
            "instructions": system_prompt,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": pdf_filename,
                            "file_data": pdf_data_url,
                        },
                        {"type": "input_text", "text": user_prompt},
                    ],
                }
            ],
            "temperature": self.config.temperature,
            "max_output_tokens": max_output_tokens or self.config.max_output_tokens,
            "store": self.config.store,
        }
        payload = self._apply_responses_request_options(payload, task_name=task_name)
        data = self._post_json(self.config.base_url.rstrip("/") + "/responses", payload, warn_on_error=False)
        if not data:
            return None, {}
        return self._extract_responses_text(data), self._parse_usage(data)

    def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        warn_on_error: bool = True,
    ) -> dict[str, Any] | None:
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
                if warn_on_error:
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
                if warn_on_error:
                    LOGGER.warning("llm request failed: %s", exc)
                return None
        else:
            if warn_on_error:
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

    def _generate_summary_from_fulltext(
        self,
        paper: PaperRecord,
        evaluations: list[TopicEvaluation],
        fulltext: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        chunks = self._chunk_fulltext(fulltext)
        if not chunks:
            return None, {}

        usage_items: list[dict[str, Any]] = []
        chunk_notes: list[str] = []
        for chunk_index, chunk_text in enumerate(chunks, start=1):
            note_text, usage = self._request_text(
                system_prompt=self._paper_chunk_system_prompt(),
                user_prompt=self._build_paper_chunk_prompt(
                    paper,
                    evaluations,
                    chunk_text,
                    chunk_index=chunk_index,
                    chunk_total=len(chunks),
                ),
                max_output_tokens=max(600, min(self.config.max_output_tokens, 900)),
                task_name=TASK_PAPER_CHUNK,
            )
            usage_items.append(usage)
            if note_text:
                chunk_notes.append(f"分块 {chunk_index}/{len(chunks)}\n{note_text.strip()}")

        if not chunk_notes:
            return None, self._merge_usage_items(usage_items)

        parsed, usage = self._request_structured_json(
            system_prompt=self._paper_reduce_system_prompt(),
            user_prompt=self._build_paper_reduce_prompt(paper, evaluations, chunk_notes),
            schema_name="paper_summary",
            schema=self._paper_summary_schema(),
            max_output_tokens=max(self.config.max_output_tokens, 1200),
            task_name=TASK_PAPER_REDUCE,
        )
        usage_items.append(usage)
        if not parsed:
            return None, self._merge_usage_items(usage_items)

        parsed["chunk_count"] = len(chunks)
        parsed["source_mode"] = "fulltext_txt"
        return parsed, self._merge_usage_items(usage_items)

    def _generate_summary_from_pdf(
        self,
        paper: PaperRecord,
        evaluations: list[TopicEvaluation],
        pdf_path: Path,
        pdf_strategy: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        if self.provider not in {"openai_responses", "openai", "responses"}:
            return self._generate_summary_from_pdf_via_brief(paper, evaluations, pdf_path, pdf_strategy)
        prefer_compact = self._prefers_compact_mode()
        parsed = None
        usage_items: list[dict[str, Any]] = []
        if prefer_compact:
            parsed, usage = self._request_structured_json_with_pdf(
                system_prompt=(
                    self._paper_summary_system_prompt()
                    + " 你会同时收到论文完整 PDF 文件。请阅读全文，不要输出思考过程，只输出最终 JSON。"
                ),
                user_prompt=self._build_pdf_paper_prompt(paper, evaluations, compact=True),
                schema_name="paper_summary",
                schema=self._paper_summary_schema(),
                pdf_path=pdf_path,
                pdf_strategy=pdf_strategy,
                max_output_tokens=max(self.config.max_output_tokens, 1200),
                task_name=TASK_PAPER_SUMMARY,
            )
            usage_items.append(usage)
        if not parsed:
            parsed, usage = self._request_structured_json_with_pdf(
                system_prompt=(
                    self._paper_summary_system_prompt()
                    + " 你会同时收到论文完整 PDF 文件。请阅读全文，不要只依赖摘要。"
                ),
                user_prompt=self._build_pdf_paper_prompt(paper, evaluations, compact=False),
                schema_name="paper_summary",
                schema=self._paper_summary_schema(),
                pdf_path=pdf_path,
                pdf_strategy=pdf_strategy,
                max_output_tokens=max(self.config.max_output_tokens, 1400),
                task_name=TASK_PAPER_SUMMARY,
            )
            usage_items.append(usage)
        if not parsed and not prefer_compact:
            parsed, usage = self._request_structured_json_with_pdf(
                system_prompt=(
                    self._paper_summary_system_prompt()
                    + " 你会同时收到论文完整 PDF 文件。请阅读全文，不要输出思考过程，只输出最终 JSON。"
                ),
                user_prompt=self._build_pdf_paper_prompt(paper, evaluations, compact=True),
                schema_name="paper_summary",
                schema=self._paper_summary_schema(),
                pdf_path=pdf_path,
                pdf_strategy=pdf_strategy,
                max_output_tokens=max(self.config.max_output_tokens, 1200),
                task_name=TASK_PAPER_SUMMARY,
            )
            usage_items.append(usage)
        if not parsed:
            return None, self._merge_usage_items(usage_items)
        parsed["source_mode"] = "pdf_direct"
        parsed["pdf_input_used"] = True
        parsed["pdf_filename"] = pdf_path.name
        parsed["pdf_input_strategy"] = pdf_strategy
        parsed["direct_pdf_status"] = "used"
        return parsed, self._merge_usage_items(usage_items)

    def _generate_summary_from_pdf_via_brief(
        self,
        paper: PaperRecord,
        evaluations: list[TopicEvaluation],
        pdf_path: Path,
        pdf_strategy: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        usage_items: list[dict[str, Any]] = []
        brief_text, usage = self._request_text_with_pdf(
            system_prompt=(
                "你是一个中文论文总结助手。"
                "你会收到完整论文 PDF。"
                "请阅读全文后按固定标签输出，不要输出思考过程，不要输出 Markdown 代码块。"
            ),
            user_prompt=self._build_pdf_brief_prompt(paper, evaluations),
            pdf_path=pdf_path,
            pdf_strategy=pdf_strategy,
            max_output_tokens=max(self.config.max_output_tokens, 900),
            task_name=TASK_PDF_BRIEF,
        )
        usage_items.append(usage)
        if not brief_text:
            return None, self._merge_usage_items(usage_items)

        parsed = self._parse_pdf_brief_summary(brief_text)
        if parsed is None:
            parsed, usage = self._request_structured_json(
                system_prompt=(
                    "你是一个中文 JSON 整理助手。"
                    "你会收到一份已经基于整篇论文 PDF 阅读得到的中文摘要备忘录。"
                    "请只把它整理为最终 JSON，不要补充未出现的事实。"
                ),
                user_prompt=self._build_pdf_brief_repair_prompt(paper, evaluations, brief_text),
                schema_name="paper_summary",
                schema=self._paper_summary_schema(),
                max_output_tokens=max(self.config.max_output_tokens, 900),
                task_name=TASK_JSON_REPAIR,
            )
            usage_items.append(usage)
            if not parsed:
                return None, self._merge_usage_items(usage_items)

        parsed["source_mode"] = "pdf_direct"
        parsed["pdf_input_used"] = True
        parsed["pdf_filename"] = pdf_path.name
        parsed["pdf_input_strategy"] = pdf_strategy
        parsed["direct_pdf_status"] = "used"
        parsed["basis"] = "llm+pdf+metadata"
        return parsed, self._merge_usage_items(usage_items)

    def _load_fulltext_text(self, paper: PaperRecord) -> str:
        path_text = (paper.fulltext_txt_path or "").strip()
        if not path_text:
            return ""
        try:
            fulltext = Path(path_text).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            LOGGER.warning("failed to read fulltext for paper_id=%s: %s", paper.id, exc)
            return ""
        return fulltext.strip()

    def _resolve_local_pdf_path(self, paper: PaperRecord) -> Path | None:
        path_text = (paper.pdf_local_path or "").strip()
        if not path_text:
            return None
        path = Path(path_text)
        if not path.exists() or not path.is_file():
            return None
        return path

    def _build_pdf_data_url(self, pdf_path: Path) -> str | None:
        try:
            pdf_bytes = pdf_path.read_bytes()
        except OSError as exc:
            LOGGER.warning("failed to read pdf for direct upload %s: %s", pdf_path, exc)
            return None
        if not pdf_bytes:
            return None
        if len(pdf_bytes) > self.config.pdf_inline_max_bytes:
            LOGGER.info(
                "pdf direct input skipped for %s because file is too large (%s bytes > %s bytes)",
                pdf_path.name,
                len(pdf_bytes),
                self.config.pdf_inline_max_bytes,
            )
            return None
        encoded = base64.b64encode(pdf_bytes).decode("ascii")
        return f"data:application/pdf;base64,{encoded}"

    def _ensure_pdf_input_strategy(self, pdf_path: Path | None) -> str | None:
        mode = self.config.pdf_input_mode.strip().lower()
        if mode == "disable":
            return None
        if self._pdf_input_strategy:
            return self._pdf_input_strategy
        if self._pdf_probe_attempted and mode != "force":
            return None
        strategy_candidates = self._candidate_pdf_strategies()
        if not strategy_candidates:
            self._pdf_probe_attempted = True
            return None
        if mode == "force":
            self._pdf_input_strategy = strategy_candidates[0]
            self._pdf_probe_attempted = True
            return self._pdf_input_strategy
        if pdf_path is None:
            return None
        self._pdf_probe_attempted = True
        for strategy in strategy_candidates:
            parsed, _ = self._request_structured_json_with_pdf(
                system_prompt=(
                    "你是一个能力探测助手。"
                    "如果你能读取附带的 PDF 文件，请返回 {\"supported\": true}。"
                    "不要输出其他字段。"
                ),
                user_prompt="请判断你是否能读取随附 PDF 文件内容，并返回 supported=true。",
                schema_name="pdf_probe",
                schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"supported": {"type": "boolean"}},
                    "required": ["supported"],
                },
                pdf_path=pdf_path,
                pdf_strategy=strategy,
                max_output_tokens=120,
            )
            if parsed and bool(parsed.get("supported")):
                self._pdf_input_strategy = strategy
                LOGGER.info(
                    "llm pdf input enabled for %s via %s",
                    self.config.label or self.config.model or self.config.variant_id,
                    strategy,
                )
                return strategy
        LOGGER.info(
            "llm pdf input unavailable for %s, fallback to extracted text",
            self.config.label or self.config.model or self.config.variant_id,
        )
        return None

    def _candidate_pdf_strategies(self) -> list[str]:
        if self.provider in {"openai_responses", "openai", "responses"}:
            return [PDF_STRATEGY_RESPONSES]
        return [PDF_STRATEGY_CHAT_FILE, PDF_STRATEGY_CHAT_INPUT_FILE]

    def _chunk_fulltext(self, fulltext: str) -> list[str]:
        text = fulltext.strip()
        if not text:
            return []
        chunk_chars = max(2000, min(self.config.fulltext_chunk_chars, self.config.max_input_chars))
        overlap_chars = max(0, min(self.config.fulltext_chunk_overlap_chars, chunk_chars // 3))
        max_chunks = max(1, self.config.fulltext_max_chunks)
        if len(text) <= chunk_chars:
            return [text]

        chunks: list[str] = []
        start = 0
        text_length = len(text)
        while start < text_length and len(chunks) < max_chunks:
            end = min(text_length, start + chunk_chars)
            if end < text_length:
                boundary = self._find_chunk_boundary(text, start, end)
                if boundary > start:
                    end = boundary
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_length:
                break
            next_start = max(end - overlap_chars, start + 1)
            if next_start <= start:
                next_start = end
            start = next_start
        return chunks

    def _find_chunk_boundary(self, text: str, start: int, end: int) -> int:
        minimum = start + max((end - start) // 2, 1)
        candidates = [
            text.rfind("\n\n", minimum, end),
            text.rfind("\n", minimum, end),
            text.rfind("。", minimum, end),
            text.rfind("！", minimum, end),
            text.rfind("？", minimum, end),
            text.rfind(".", minimum, end),
        ]
        boundary = max(candidates)
        if boundary == -1:
            return end
        if text[boundary : boundary + 2] == "\n\n":
            return boundary + 2
        return boundary + 1

    def _merge_usage_items(self, usage_items: list[dict[str, Any]]) -> dict[str, Any]:
        merged: dict[str, Any] = {"calls": len(usage_items)}
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        has_input = False
        has_output = False
        has_total = False
        raw_items: list[dict[str, Any]] = []
        for usage in usage_items:
            if not isinstance(usage, dict):
                continue
            raw_items.append(usage)
            if "input_tokens" in usage:
                has_input = True
                input_tokens += int(usage.get("input_tokens") or 0)
            if "output_tokens" in usage:
                has_output = True
                output_tokens += int(usage.get("output_tokens") or 0)
            if "total_tokens" in usage:
                has_total = True
                total_tokens += int(usage.get("total_tokens") or 0)
        if has_input:
            merged["input_tokens"] = input_tokens
        if has_output:
            merged["output_tokens"] = output_tokens
        if has_total:
            merged["total_tokens"] = total_tokens
        if raw_items:
            merged["steps"] = raw_items
        return merged

    def _topics_text(self, evaluations: list[TopicEvaluation], *, limit: int = 3) -> str:
        relevant_topics = [item for item in evaluations if item.classification != "irrelevant"]
        return "\n".join(
            f"- {item.topic_name} / score={item.score} / keywords={', '.join(item.matched_keywords[:6])}"
            for item in relevant_topics[:limit]
        )

    def _build_paper_prompt(self, paper: PaperRecord, evaluations: list[TopicEvaluation]) -> str:
        topics_text = self._topics_text(evaluations)
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

    def _build_pdf_paper_prompt(
        self,
        paper: PaperRecord,
        evaluations: list[TopicEvaluation],
        *,
        compact: bool,
    ) -> str:
        topics_text = self._topics_text(evaluations)
        abstract = shorten(paper.abstract or "", min(self.config.max_input_chars, 2500 if compact else 4500)) or "无"
        context = {
            "title": paper.title,
            "authors": ", ".join(paper.authors[:8]) or "未知",
            "venue": paper.venue or "未知",
            "published_at": paper.published_at or "未知",
            "topics_text": topics_text or "- 无",
            "abstract": abstract,
            "fulltext": (
                "本次请求已直接附带完整 PDF 文件。"
                "请以阅读全文后的内容为准，明确写出问题、方法、应用、结果、贡献和局限。"
            ),
        }
        if self.prompt_library:
            return self.prompt_library.paper_summary_user(context)
        instruction = "请直接阅读随附 PDF 全文并总结，不要只依赖摘要。"
        return (
            f"{instruction}\n\n"
            f"标题: {context['title']}\n"
            f"作者: {context['authors']}\n"
            f"发表信息: venue={context['venue']}, published_at={context['published_at']}\n"
            f"相关主题:\n{context['topics_text']}\n\n"
            f"摘要:\n{context['abstract']}\n\n"
            f"全文说明:\n{context['fulltext']}\n"
        )

    def _build_pdf_brief_prompt(self, paper: PaperRecord, evaluations: list[TopicEvaluation]) -> str:
        _ = paper, evaluations
        return (
            "请阅读全文后，用中文写一个紧凑摘要，必须包含：问题、方法、应用、结果、局限。"
            "不要输出JSON，不要输出思考过程。总字数控制在300字以内。"
        )

    def _build_pdf_brief_repair_prompt(
        self,
        paper: PaperRecord,
        evaluations: list[TopicEvaluation],
        brief_text: str,
    ) -> str:
        topics_text = self._topics_text(evaluations)
        return (
            "下面是一份已经基于整篇论文 PDF 阅读得到的中文摘要备忘录。"
            "请只依据这份备忘录整理为最终 JSON，不要补充不存在的内容。\n\n"
            f"标题: {paper.title}\n"
            f"相关主题:\n{topics_text or '- 无'}\n\n"
            f"PDF 摘要备忘录:\n{brief_text}\n"
        )

    def _build_paper_chunk_prompt(
        self,
        paper: PaperRecord,
        evaluations: list[TopicEvaluation],
        chunk_text: str,
        *,
        chunk_index: int,
        chunk_total: int,
    ) -> str:
        topics_text = self._topics_text(evaluations)
        context = {
            "title": paper.title,
            "authors": ", ".join(paper.authors[:8]) or "未知",
            "venue": paper.venue or "未知",
            "published_at": paper.published_at or "未知",
            "topics_text": topics_text or "- 无",
            "abstract": shorten(paper.abstract or "", 3000) or "无",
            "chunk_index": str(chunk_index),
            "chunk_total": str(chunk_total),
            "chunk_text": chunk_text,
        }
        if self.prompt_library:
            return self.prompt_library.paper_chunk_user(context)
        return (
            f"论文《{paper.title}》正文分块 {chunk_index}/{chunk_total}\n"
            f"相关主题:\n{topics_text or '- 无'}\n\n"
            f"摘要:\n{context['abstract']}\n\n"
            f"正文分块:\n{chunk_text}\n"
        )

    def _build_paper_reduce_prompt(
        self,
        paper: PaperRecord,
        evaluations: list[TopicEvaluation],
        chunk_notes: list[str],
    ) -> str:
        topics_text = self._topics_text(evaluations)
        context = {
            "title": paper.title,
            "authors": ", ".join(paper.authors[:8]) or "未知",
            "venue": paper.venue or "未知",
            "published_at": paper.published_at or "未知",
            "topics_text": topics_text or "- 无",
            "abstract": shorten(paper.abstract or "", 4000) or "无",
            "chunk_notes": self._render_chunk_notes(chunk_notes),
        }
        if self.prompt_library:
            return self.prompt_library.paper_reduce_user(context)
        return (
            f"标题: {context['title']}\n"
            f"相关主题:\n{context['topics_text']}\n\n"
            f"摘要:\n{context['abstract']}\n\n"
            f"分块分析结果:\n{context['chunk_notes']}\n"
        )

    def _render_chunk_notes(self, chunk_notes: list[str]) -> str:
        return "\n\n".join(note.strip() for note in chunk_notes if note.strip())

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
        topics_text = self._topics_text(evaluations)
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
        return "你是一个严谨的中文论文分析助手。请根据提供的标题、摘要、全文信息和相关性标签，输出简洁、具体、避免空话的 JSON。"

    def _paper_chunk_system_prompt(self) -> str:
        if self.prompt_library:
            return self.prompt_library.paper_chunk_system()
        return "你是一个严谨的中文论文分析助手。请只基于当前正文分块提取事实，不要脑补。"

    def _paper_reduce_system_prompt(self) -> str:
        if self.prompt_library:
            return self.prompt_library.paper_reduce_system()
        return "你是一个严谨的中文论文分析助手。请基于整篇论文的分块分析结果生成高质量最终总结。"

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
        application = str(parsed.get("application", "")).strip()
        if application:
            parts.append(f"应用：{application}")
        results = str(parsed.get("results", "")).strip()
        if results:
            parts.append(f"结果：{results}")
        contributions = [str(item).strip() for item in parsed.get("contributions", []) if str(item).strip()]
        if contributions:
            parts.append(f"贡献：{'；'.join(contributions[:3])}")
        limitations = [str(item).strip() for item in parsed.get("limitations", []) if str(item).strip()]
        if limitations:
            parts.append(f"局限：{'；'.join(limitations[:2])}")
        return " ".join(parts) if parts else "LLM 未返回可用总结。"

    def _normalize_basis(self, basis: Any, *, source_mode: str, has_fulltext: bool) -> str:
        value = str(basis or "").strip().lower()
        allowed = {"llm+pdf+metadata", "llm+fulltext+metadata", "llm+abstract+metadata"}
        if value in allowed:
            return value
        if source_mode == "pdf_direct":
            return "llm+pdf+metadata"
        return "llm+fulltext+metadata" if has_fulltext else "llm+abstract+metadata"

    def _parse_pdf_brief_summary(self, brief_text: str) -> dict[str, Any] | None:
        text = THINK_TAG_RE.sub("", brief_text).strip()
        if not text:
            return None
        labels = ["摘要", "问题", "方法", "应用", "结果", "贡献", "局限", "标签"]
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        values: dict[str, str] = {}
        for index, label in enumerate(labels):
            pattern = rf"{label}\s*[：:]\s*"
            match = re.search(pattern, normalized)
            if not match:
                continue
            start = match.end()
            end = len(normalized)
            for next_label in labels[index + 1 :]:
                next_match = re.search(rf"\n?\s*{next_label}\s*[：:]\s*", normalized[start:])
                if next_match:
                    end = start + next_match.start()
                    break
            values[label] = normalized[start:end].strip()
        required = {"问题", "方法", "应用", "结果"}
        if not required.issubset(values):
            return None
        contributions = self._split_labeled_items(values.get("贡献", ""))
        limitations = self._split_labeled_items(values.get("局限", ""))
        tags = self._split_labeled_items(values.get("标签", ""))
        return {
            "summary": values.get("摘要", "").strip(),
            "problem": values.get("问题", ""),
            "method": values.get("方法", ""),
            "application": values.get("应用", ""),
            "results": values.get("结果", ""),
            "contributions": contributions,
            "limitations": limitations,
            "tags": tags,
            "basis": "llm+pdf+metadata",
        }

    def _split_labeled_items(self, text: str) -> list[str]:
        raw = str(text or "").strip()
        if not raw:
            return []
        parts = re.split(r"[；;，,\n]+", raw)
        return [item.strip(" -•\t") for item in parts if item.strip(" -•\t")]

    def _paper_summary_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summary": {"type": "string"},
                "problem": {"type": "string"},
                "method": {"type": "string"},
                "application": {"type": "string"},
                "results": {"type": "string"},
                "contributions": {"type": "array", "items": {"type": "string"}},
                "limitations": {"type": "array", "items": {"type": "string"}},
                "tags": {"type": "array", "items": {"type": "string"}},
                "basis": {
                    "type": "string",
                    "enum": ["llm+pdf+metadata", "llm+fulltext+metadata", "llm+abstract+metadata"],
                },
            },
            "required": [
                "summary",
                "problem",
                "method",
                "application",
                "results",
                "contributions",
                "limitations",
                "tags",
                "basis",
            ],
        }

    def _paper_chunk_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "chunk_summary": {"type": "string"},
                "key_points": {"type": "array", "items": {"type": "string"}},
                "methods": {"type": "array", "items": {"type": "string"}},
                "evidence": {"type": "array", "items": {"type": "string"}},
                "applications": {"type": "array", "items": {"type": "string"}},
                "limitations": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["chunk_summary", "key_points", "methods", "evidence", "applications", "limitations"],
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
