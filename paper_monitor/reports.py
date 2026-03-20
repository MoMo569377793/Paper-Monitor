from __future__ import annotations

from dataclasses import replace
import html
import json
import logging
from collections import defaultdict
from pathlib import Path

from paper_monitor.llm import LLMClient
from paper_monitor.llm_registry import LLMRuntimeVariant
from paper_monitor.models import PaperLLMSummary, PaperRecord, ReportEntry, Settings, TopicEvaluation
from paper_monitor.progress import ProgressBar
from paper_monitor.storage import Database
from paper_monitor.utils import ensure_directory, now_iso, shorten, to_day_bounds


LOGGER = logging.getLogger(__name__)


def _variant_attr(variant: LLMRuntimeVariant | dict, name: str):
    if isinstance(variant, dict):
        if name == "variant_id":
            return variant.get("variant_id", variant.get("slug"))
        if name == "client":
            return variant.get("client", variant.get("llm_client"))
        return variant.get(name)
    return getattr(variant, name)


def _report_label(report_type: str) -> str:
    return {"daily": "日报", "weekly": "周报"}.get(report_type, report_type)


def _top_tags(entries: list[ReportEntry]) -> list[str]:
    counts: dict[str, int] = {}
    for entry in entries:
        for keyword in entry.matched_keywords[:6]:
            counts[keyword] = counts.get(keyword, 0) + 1
    return [item[0] for item in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:5]]


def _topic_trend_sentence(entries: list[ReportEntry]) -> str:
    if not entries:
        return "本时间窗口内没有新的相关论文。"
    top_tags = _top_tags(entries)
    if top_tags:
        return f"本时间窗口新增 {len(entries)} 篇候选论文，主要集中在 {', '.join(top_tags)}。"
    return f"本时间窗口新增 {len(entries)} 篇候选论文，主题分布较分散。"


def _catalog_topic_sentence(entries: list[ReportEntry]) -> str:
    if not entries:
        return "当前数据库中该领域还没有论文。"
    top_tags = _top_tags(entries)
    if top_tags:
        return f"当前数据库中共有 {len(entries)} 篇论文，常见关键词包括 {', '.join(top_tags)}。"
    return f"当前数据库中共有 {len(entries)} 篇论文。"


def _entry_sort_key(entry: ReportEntry) -> tuple[float, str, int]:
    paper = entry.paper
    published = paper.published_at or paper.created_at or ""
    return (float(entry.score), published, int(paper.id))


def _sorted_entries(entries: list[ReportEntry]) -> list[ReportEntry]:
    return sorted(entries, key=_entry_sort_key, reverse=True)


def _fallback_review_window_days(report_type: str, topic_id: str) -> int:
    if topic_id == "matrix_free_fem":
        return 90
    if topic_id == "ai_operator_acceleration":
        return 30
    return 30 if report_type == "daily" else 90


def _append_report_entry_markdown(
    lines: list[str],
    entry: ReportEntry,
    index: int,
    paper_summaries_by_paper: dict[int, list[PaperLLMSummary]],
    variants: list[LLMRuntimeVariant | dict],
) -> None:
    paper = entry.paper
    lines.append(f"### {index}. {paper.title}")
    lines.append("")
    lines.append(f"- 相关性：`{entry.classification}` / 分数 `{entry.score}`")
    lines.append(f"- 来源：`{', '.join(entry.source_names) or paper.source_first}`")
    lines.append(f"- 发布时间：`{paper.published_at or '未知'}`")
    lines.append(f"- Venue / 分类：`{paper.venue or '未知'}` / `{', '.join(paper.categories) or '无'}`")
    lines.append(
        f"- 全文状态：`{paper.fulltext_status}` / PDF `{paper.pdf_status}` / "
        f"页数 `{paper.page_count or '未知'}` / 总结来源 `{paper.summary_basis or '未知'}`"
    )
    lines.append(f"- 匹配词：`{', '.join(entry.matched_keywords) or '无'}`")
    lines.append(f"- 链接：{paper.primary_url or (entry.source_urls[0] if entry.source_urls else '无')}")
    paper_stem = _paper_report_stem(paper)
    lines.append(f"- 单篇报告：`reports/papers/{paper_stem}.md` / `reports/papers/{paper_stem}.html`")
    if paper.pdf_local_path:
        lines.append(f"- 本地 PDF：`{paper.pdf_local_path}`")
    if paper.fulltext_txt_path:
        lines.append(f"- 全文文本：`{paper.fulltext_txt_path}`")
    lines.append(f"- 默认总结：{paper.summary_text}")
    lines.append("#### 多模型摘要")
    lines.extend(_render_summary_lines(variants, paper_summaries_by_paper.get(paper.id, [])))
    lines.append("")
    if paper.fulltext_excerpt:
        lines.append(f"- 全文节选：{shorten(paper.fulltext_excerpt, 320)}")
    elif paper.abstract:
        lines.append(f"- 摘要片段：{shorten(paper.abstract, 320)}")
    lines.append("")


def _render_report_entry_html(
    entry: ReportEntry,
    paper_summaries_by_paper: dict[int, list[PaperLLMSummary]],
    variants: list[LLMRuntimeVariant | dict],
) -> str:
    paper = entry.paper
    return f"""
                <article class="paper-card">
                  <h3>{html.escape(paper.title)}</h3>
                  <p class="meta">相关性 {entry.classification} / 分数 {entry.score} / 来源 {html.escape(', '.join(entry.source_names) or paper.source_first)}</p>
                  <p class="meta">发布时间 {html.escape(paper.published_at or '未知')} / Venue {html.escape(paper.venue or '未知')}</p>
                  <p class="meta">全文状态 {html.escape(paper.fulltext_status)} / PDF {html.escape(paper.pdf_status)} / 页数 {html.escape(str(paper.page_count or '未知'))} / 总结来源 {html.escape(paper.summary_basis or '未知')}</p>
                  <p class="meta">单篇报告 reports/papers/{html.escape(_paper_report_stem(paper))}.html</p>
                  <div class="paper-grid">
                    <div class="paper-panel">
                      <p><strong>匹配词：</strong>{html.escape(', '.join(entry.matched_keywords) or '无')}</p>
                      <p><strong>默认总结：</strong>{html.escape(paper.summary_text)}</p>
                    </div>
                    <div class="paper-panel">
                      <p><strong>{'全文节选' if paper.fulltext_excerpt else '摘要片段'}：</strong>{html.escape(shorten(paper.fulltext_excerpt or paper.abstract or '无摘要', 360))}</p>
                    </div>
                  </div>
                  <div class="llm-summary-section">
                    <p><strong>多模型摘要：</strong></p>
                    {_render_summary_html(variants, paper_summaries_by_paper.get(paper.id, []))}
                  </div>
                  <p><a href="{html.escape(paper.primary_url or (entry.source_urls[0] if entry.source_urls else '#'))}">打开原始链接</a></p>
                </article>
                """


def _collect_fallback_review_entries(
    db: Database,
    settings: Settings,
    report_type: str,
    report_date: str,
    grouped_entries: dict[str, list[ReportEntry]],
) -> tuple[dict[str, list[ReportEntry]], dict[str, int]]:
    fallback_entries_by_topic: dict[str, list[ReportEntry]] = {}
    fallback_days_by_topic: dict[str, int] = {}
    for topic in settings.topics:
        if grouped_entries.get(topic.id):
            continue
        days = _fallback_review_window_days(report_type, topic.id)
        start_at, end_at = to_day_bounds(report_date, settings.timezone, days)
        entries = db.fetch_recent_topic_entries(
            topic.id,
            start_at,
            end_at,
            include_maybe=False,
            limit=settings.report.top_n_per_topic,
        )
        sorted_entries = _sorted_entries(entries)
        if sorted_entries:
            fallback_entries_by_topic[topic.id] = sorted_entries
            fallback_days_by_topic[topic.id] = days
    return fallback_entries_by_topic, fallback_days_by_topic


def _single_runtime_variant(settings: Settings, llm_client: LLMClient) -> LLMRuntimeVariant:
    return LLMRuntimeVariant(
        variant_id=settings.llm.variant_id,
        label=settings.llm.label or settings.llm.model or settings.llm.variant_id,
        provider=settings.llm.provider,
        base_url=settings.llm.base_url,
        model=settings.llm.model,
        config_path=settings.config_path,
        client=llm_client,
    )


def _collect_topic_digests_by_variant(
    settings: Settings,
    grouped_entries: dict[str, list[ReportEntry]],
    paper_summaries_by_paper: dict[int, list[PaperLLMSummary]],
    variants: list[LLMRuntimeVariant | dict],
    use_llm_topic_digest: bool,
    progress_callback=None,
) -> dict[str, dict[str, dict]]:
    digests_by_variant: dict[str, dict[str, dict]] = {}
    if not variants or not use_llm_topic_digest:
        return digests_by_variant
    for variant in variants:
        variant_id = str(_variant_attr(variant, "variant_id") or "")
        client = _variant_attr(variant, "client")
        if not client or not client.enabled or not variant_id:
            continue
        topic_digests: dict[str, dict] = {}
        for topic in settings.topics:
            topic_entries = _sorted_entries(grouped_entries.get(topic.id, []))
            prepared_entries, digest_input_meta = _prepare_digest_entries_for_variant(
                topic_entries,
                paper_summaries_by_paper,
                variant_id=variant_id,
                entry_limit=getattr(client.config, "topic_digest_entry_limit", settings.llm.topic_digest_entry_limit)
                if hasattr(client, "config")
                else settings.llm.topic_digest_entry_limit,
                variant_label=str(_variant_attr(variant, "label") or variant_id),
            )
            if progress_callback:
                progress_callback(f"{_variant_attr(variant, 'label')} -> {topic.display_name}", True)
            try:
                digest = client.generate_topic_digest(topic.display_name, topic.description, prepared_entries)
            except Exception as exc:  # pragma: no cover - defensive for flaky upstream APIs
                LOGGER.warning("topic digest failed for %s / %s: %s", variant_id, topic.id, exc)
                if progress_callback:
                    progress_callback(f"{_variant_attr(variant, 'label')} -> {topic.display_name} 失败", False)
                continue
            if digest is None:
                if progress_callback:
                    progress_callback(f"{_variant_attr(variant, 'label')} -> {topic.display_name} 未生成", False)
                continue
            topic_digests[topic.id] = {
                "overview": digest.overview,
                "highlights": digest.highlights,
                "watchlist": digest.watchlist,
                "tags": digest.tags,
                "usage": digest.structured.get("usage", {}),
                "input_meta": digest_input_meta,
            }
            if progress_callback:
                progress_callback(f"{_variant_attr(variant, 'label')} -> {topic.display_name} 完成", False)
        digests_by_variant[variant_id] = topic_digests
    return digests_by_variant


def _prepare_digest_entries_for_variant(
    entries: list[ReportEntry],
    paper_summaries_by_paper: dict[int, list[PaperLLMSummary]],
    *,
    variant_id: str,
    entry_limit: int,
    variant_label: str,
) -> tuple[list[ReportEntry], dict[str, object]]:
    selected_entries = entries[: max(int(entry_limit or 0), 0)] if entry_limit else list(entries)
    prepared_entries: list[ReportEntry] = []
    input_papers: list[dict[str, object]] = []
    used_variant_count = 0
    fallback_count = 0
    for entry in selected_entries:
        summaries = paper_summaries_by_paper.get(entry.paper.id, [])
        selected_summary = next((item for item in summaries if item.variant_id == variant_id), None)
        if selected_summary is not None:
            used_variant_count += 1
            summary_text = selected_summary.summary_text or entry.paper.summary_text or entry.paper.abstract or "无"
            summary_basis = selected_summary.summary_basis or entry.paper.summary_basis
            source_label = f"{variant_label} 单篇总结"
            scope_label = _summary_scope_label(selected_summary)
            scope_note = _summary_scope_note(selected_summary)
        else:
            fallback_count += 1
            summary_text = entry.paper.summary_text or entry.paper.abstract or "无"
            summary_basis = entry.paper.summary_basis
            source_label = "默认总结"
            scope_label = "默认总结"
            scope_note = "未找到该模型的逐篇总结，已回退到默认总结。"
        prepared_entries.append(
            replace(
                entry,
                paper=replace(
                    entry.paper,
                    summary_text=summary_text,
                    summary_basis=summary_basis,
                ),
            )
        )
        input_papers.append(
            {
                "paper_id": entry.paper.id,
                "title": entry.paper.title,
                "published_at": entry.paper.published_at,
                "summary_source": source_label,
                "summary_scope": scope_label,
                "summary_scope_note": scope_note,
            }
        )
    return prepared_entries, {
        "available_entry_count": len(entries),
        "selected_entry_count": len(selected_entries),
        "selection_mode": "all" if len(selected_entries) == len(entries) else f"top_{len(selected_entries)}",
        "variant_summary_count": used_variant_count,
        "fallback_summary_count": fallback_count,
        "input_papers": input_papers,
    }


def _render_summary_lines(variants: list[LLMRuntimeVariant | dict], summaries: list[PaperLLMSummary]) -> list[str]:
    summary_map = {summary.variant_id: summary for summary in summaries}
    lines: list[str] = []
    for variant in variants:
        variant_id = str(_variant_attr(variant, "variant_id") or "")
        variant_label = str(_variant_attr(variant, "label") or variant_id or "未知模型")
        summary = summary_map.get(variant_id)
        if not summary:
            lines.append(f"- 模型：`{variant_label}`")
            lines.append("- 输入依据：`未生成`")
            lines.append("- 总结：未生成")
            continue
        usage = summary.usage if isinstance(summary.usage, dict) else {}
        structured = summary.structured if isinstance(summary.structured, dict) else {}
        lines.append(f"- 模型：`{variant_label}`")
        lines.append(f"- 输入依据：`{_summary_scope_label(summary)}`")
        if _summary_scope_note(summary):
            lines.append(f"- 依据说明：{_summary_scope_note(summary)}")
        if usage:
            lines.append(
                f"- Token：in `{usage.get('input_tokens', '未知')}` / "
                f"out `{usage.get('output_tokens', '未知')}` / total `{usage.get('total_tokens', '未知')}`"
            )
        lines.append(f"- 总结：{summary.summary_text}")
        lines.extend(_structured_summary_markdown(structured))
    return lines


def _render_summary_html(variants: list[LLMRuntimeVariant | dict], summaries: list[PaperLLMSummary]) -> str:
    summary_map = {summary.variant_id: summary for summary in summaries}
    items: list[str] = []
    for variant in variants:
        variant_id = str(_variant_attr(variant, "variant_id") or "")
        variant_label = str(_variant_attr(variant, "label") or variant_id or "未知模型")
        summary = summary_map.get(variant_id)
        if summary is None:
            items.append(
                f"""
                <article class="llm-summary-card">
                  <h4>{html.escape(variant_label)}</h4>
                  <p class="meta">输入依据 未生成</p>
                  <p><strong>总结：</strong>未生成</p>
                </article>
                """
            )
            continue
        usage = summary.usage if isinstance(summary.usage, dict) else {}
        structured = summary.structured if isinstance(summary.structured, dict) else {}
        usage_html = ""
        if usage:
            usage_html = (
                f"<p class=\"meta\">Token in {html.escape(str(usage.get('input_tokens', '未知')))} / "
                f"out {html.escape(str(usage.get('output_tokens', '未知')))} / "
                f"total {html.escape(str(usage.get('total_tokens', '未知')))}</p>"
            )
        scope_note = _summary_scope_note(summary)
        items.append(
            f"""
            <article class="llm-summary-card">
              <h4>{html.escape(variant_label)}</h4>
              <p class="meta">输入依据 {html.escape(_summary_scope_label(summary))}</p>
              {f'<p class="meta">{html.escape(scope_note)}</p>' if scope_note else ''}
              {usage_html}
              <p><strong>总结：</strong>{html.escape(summary.summary_text)}</p>
              {_structured_summary_html(structured)}
            </article>
            """
        )
    return '<div class="llm-summary-grid">' + "".join(items) + "</div>"


def _digest_input_meta_lines(digest: dict) -> list[str]:
    input_meta = digest.get("input_meta", {}) if isinstance(digest, dict) else {}
    if not isinstance(input_meta, dict) or not input_meta:
        return []
    selected = input_meta.get("selected_entry_count", 0)
    available = input_meta.get("available_entry_count", 0)
    selection_mode = str(input_meta.get("selection_mode", "")).strip() or "unknown"
    variant_summary_count = input_meta.get("variant_summary_count", 0)
    fallback_summary_count = input_meta.get("fallback_summary_count", 0)
    input_papers = input_meta.get("input_papers", []) if isinstance(input_meta.get("input_papers"), list) else []
    preview = " | ".join(
        f"#{item.get('paper_id')} {item.get('title')} [{item.get('summary_source')}/{item.get('summary_scope')}]"
        for item in input_papers[:5]
        if isinstance(item, dict)
    )
    lines = [
        f"- 聚合输入：使用 `{selected}/{available}` 篇论文（`{selection_mode}`）",
        f"- 逐篇总结来源：本模型总结 `{variant_summary_count}` 篇，回退默认总结 `{fallback_summary_count}` 篇",
    ]
    if preview:
        suffix = " | ..." if len(input_papers) > 5 else ""
        lines.append(f"- 输入论文：`{preview}{suffix}`")
    return lines


def _digest_input_meta_html(digest: dict) -> str:
    input_meta = digest.get("input_meta", {}) if isinstance(digest, dict) else {}
    if not isinstance(input_meta, dict) or not input_meta:
        return ""
    selected = input_meta.get("selected_entry_count", 0)
    available = input_meta.get("available_entry_count", 0)
    selection_mode = str(input_meta.get("selection_mode", "")).strip() or "unknown"
    variant_summary_count = input_meta.get("variant_summary_count", 0)
    fallback_summary_count = input_meta.get("fallback_summary_count", 0)
    input_papers = input_meta.get("input_papers", []) if isinstance(input_meta.get("input_papers"), list) else []
    preview = " | ".join(
        f"#{item.get('paper_id')} {item.get('title')} [{item.get('summary_source')}/{item.get('summary_scope')}]"
        for item in input_papers[:5]
        if isinstance(item, dict)
    )
    preview_html = (
        f"<p><strong>输入论文：</strong>{html.escape(preview)}{' | ...' if len(input_papers) > 5 else ''}</p>"
        if preview
        else ""
    )
    return (
        f"<p><strong>聚合输入：</strong>{html.escape(str(selected))}/{html.escape(str(available))} 篇"
        f"（{html.escape(selection_mode)}）</p>"
        f"<p><strong>逐篇总结来源：</strong>本模型总结 {html.escape(str(variant_summary_count))} 篇，"
        f"回退默认总结 {html.escape(str(fallback_summary_count))} 篇</p>"
        f"{preview_html}"
    )


def _summary_scope_label(summary: PaperLLMSummary) -> str:
    structured = summary.structured if isinstance(summary.structured, dict) else {}
    source_mode = str(structured.get("source_mode", "")).strip().lower()
    basis = (summary.summary_basis or "").strip().lower()
    if source_mode == "pdf_direct" or basis == "llm+pdf+metadata":
        return "已直接读取 PDF"
    if source_mode == "fulltext_txt" or basis == "llm+fulltext+metadata":
        return "已读取完整全文"
    return "仅基于摘要/元数据"


def _summary_scope_note(summary: PaperLLMSummary) -> str:
    structured = summary.structured if isinstance(summary.structured, dict) else {}
    source_mode = str(structured.get("source_mode", "")).strip().lower()
    chunk_count = structured.get("chunk_count")
    pdf_filename = str(structured.get("pdf_filename", "")).strip()
    pdf_strategy = str(
        structured.get("pdf_input_strategy") or structured.get("direct_pdf_strategy") or ""
    ).strip()
    direct_pdf_status = str(structured.get("direct_pdf_status", "")).strip().lower()
    if source_mode == "pdf_direct":
        strategy_text = f"直接 PDF 输入（策略 `{pdf_strategy}`）" if pdf_strategy else "直接 PDF 输入"
        if pdf_filename:
            return f"本次总结由模型通过 {strategy_text} 读取 PDF 文件 {pdf_filename} 后生成，没有经过本地文字节选回退。"
        return f"本次总结由模型通过 {strategy_text} 读取 PDF 文件后生成，没有经过本地文字节选回退。"
    if source_mode == "fulltext_txt":
        fallback_text = ""
        if direct_pdf_status == "unsupported":
            fallback_text = (
                f" 直接 PDF 探测未通过{f'（已尝试 `{pdf_strategy}`）' if pdf_strategy else ''}，"
                "因此回退到全文文本模式。"
            )
        elif direct_pdf_status == "request_failed":
            fallback_text = (
                f" 直接 PDF 请求失败{f'（策略 `{pdf_strategy}`）' if pdf_strategy else ''}，"
                "因此回退到全文文本模式。"
            )
        elif direct_pdf_status == "too_large":
            fallback_text = " 本地 PDF 超过直传大小限制，因此回退到全文文本模式。"
        elif direct_pdf_status == "invalid_response":
            fallback_text = " 直接 PDF 返回了无效内容，因此回退到全文文本模式。"
        elif direct_pdf_status == "disabled":
            fallback_text = " 当前配置已禁用直接 PDF 输入，因此回退到全文文本模式。"
        elif direct_pdf_status == "no_local_pdf":
            fallback_text = " 当前没有可用本地 PDF，因此直接使用抽取后的全文文本。"
        if chunk_count:
            return f"本次总结读取了完整 PDF 提取全文，并按 {chunk_count} 个分块进行分析后聚合。{fallback_text}".strip()
        return f"本次总结读取了完整 PDF 提取全文后再进行聚合分析。{fallback_text}".strip()
    if (summary.summary_basis or "").strip().lower() == "llm+fulltext+metadata":
        return "本次总结包含全文信息。"
    if direct_pdf_status == "unsupported":
        return (
            f"本次总结没有读取完整全文，只基于标题、摘要和元数据。"
            f" 直接 PDF 探测未通过{f'（已尝试 `{pdf_strategy}`）' if pdf_strategy else ''}。"
        ).strip()
    if direct_pdf_status == "request_failed":
        return (
            f"本次总结没有读取完整全文，只基于标题、摘要和元数据。"
            f" 直接 PDF 请求失败{f'（策略 `{pdf_strategy}`）' if pdf_strategy else ''}。"
        ).strip()
    if direct_pdf_status == "too_large":
        return "本次总结没有读取完整全文，只基于标题、摘要和元数据。本地 PDF 超过直传大小限制。"
    if direct_pdf_status == "invalid_response":
        return "本次总结没有读取完整全文，只基于标题、摘要和元数据。直接 PDF 返回了无效内容。"
    if direct_pdf_status == "disabled":
        return "本次总结没有读取完整全文，只基于标题、摘要和元数据。当前配置已禁用直接 PDF 输入。"
    if direct_pdf_status == "no_local_pdf":
        return "本次总结没有读取完整全文，只基于标题、摘要和元数据。当前没有可用本地 PDF。"
    return "本次总结没有读取完整全文，只基于标题、摘要和元数据。"


def _structured_summary_markdown(structured: dict) -> list[str]:
    lines: list[str] = []
    field_map = [
        ("problem", "问题"),
        ("method", "方法"),
        ("application", "应用"),
        ("results", "结果"),
    ]
    for field_name, label in field_map:
        value = str(structured.get(field_name, "")).strip()
        if value:
            lines.append(f"- {label}：{value}")
    contributions = [str(item).strip() for item in structured.get("contributions", []) if str(item).strip()]
    if contributions:
        lines.append(f"- 贡献：`{' | '.join(contributions[:4])}`")
    limitations = [str(item).strip() for item in structured.get("limitations", []) if str(item).strip()]
    if limitations:
        lines.append(f"- 局限：`{' | '.join(limitations[:4])}`")
    tags = [str(item).strip() for item in structured.get("tags", []) if str(item).strip()]
    if tags:
        lines.append(f"- 标签：`{', '.join(tags[:8])}`")
    return lines


def _structured_summary_html(structured: dict) -> str:
    parts: list[str] = []
    field_map = [
        ("problem", "问题"),
        ("method", "方法"),
        ("application", "应用"),
        ("results", "结果"),
    ]
    for field_name, label in field_map:
        value = str(structured.get(field_name, "")).strip()
        if value:
            parts.append(f"<p><strong>{html.escape(label)}：</strong>{html.escape(value)}</p>")
    contributions = [str(item).strip() for item in structured.get("contributions", []) if str(item).strip()]
    if contributions:
        parts.append(f"<p><strong>贡献：</strong>{html.escape(' | '.join(contributions[:4]))}</p>")
    limitations = [str(item).strip() for item in structured.get("limitations", []) if str(item).strip()]
    if limitations:
        parts.append(f"<p><strong>局限：</strong>{html.escape(' | '.join(limitations[:4]))}</p>")
    tags = [str(item).strip() for item in structured.get("tags", []) if str(item).strip()]
    if tags:
        parts.append(f"<p><strong>标签：</strong>{html.escape(', '.join(tags[:8]))}</p>")
    return "".join(parts)


def _paper_report_stem(paper: PaperRecord) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in paper.title)
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug.strip("-")[:72] or f"paper-{paper.id}"
    return f"{paper.id:06d}-{slug}"


def _variants_for_paper(
    summaries: list[PaperLLMSummary],
    variants: list[LLMRuntimeVariant] | None,
) -> list[LLMRuntimeVariant | dict]:
    if variants:
        return variants
    return [
        {
            "variant_id": summary.variant_id,
            "label": summary.variant_label or summary.model or summary.variant_id,
        }
        for summary in summaries
    ]


def _merge_variants_with_stored_summaries(
    variants: list[LLMRuntimeVariant | dict],
    paper_summaries_by_paper: dict[int, list[PaperLLMSummary]],
) -> list[LLMRuntimeVariant | dict]:
    merged: list[LLMRuntimeVariant | dict] = list(variants)
    seen = {str(_variant_attr(variant, "variant_id") or "") for variant in merged}
    for paper_id in sorted(paper_summaries_by_paper):
        for summary in paper_summaries_by_paper.get(paper_id, []):
            if summary.variant_id in seen:
                continue
            merged.append(
                {
                    "variant_id": summary.variant_id,
                    "label": summary.variant_label or summary.model or summary.variant_id,
                    "model": summary.model,
                    "base_url": summary.base_url,
                }
            )
            seen.add(summary.variant_id)
    return merged


def _allowed_variant_ids(variants: list[LLMRuntimeVariant | dict]) -> set[str]:
    return {
        str(_variant_attr(variant, "variant_id") or "").strip()
        for variant in variants
        if str(_variant_attr(variant, "variant_id") or "").strip()
    }


def _filter_summaries_by_variants(
    paper_summaries_by_paper: dict[int, list[PaperLLMSummary]],
    variants: list[LLMRuntimeVariant | dict],
) -> dict[int, list[PaperLLMSummary]]:
    allowed_variant_ids = _allowed_variant_ids(variants)
    if not allowed_variant_ids:
        return paper_summaries_by_paper
    filtered: dict[int, list[PaperLLMSummary]] = {}
    for paper_id, summaries in paper_summaries_by_paper.items():
        filtered[paper_id] = [summary for summary in summaries if summary.variant_id in allowed_variant_ids]
    return filtered


def _resolve_display_variants(
    variants: list[LLMRuntimeVariant | dict],
    paper_summaries_by_paper: dict[int, list[PaperLLMSummary]],
) -> tuple[list[LLMRuntimeVariant | dict], dict[int, list[PaperLLMSummary]]]:
    requested_variants = list(variants)
    if requested_variants:
        merged = _merge_variants_with_stored_summaries(requested_variants, paper_summaries_by_paper)
        return merged, _filter_summaries_by_variants(paper_summaries_by_paper, merged)
    return _merge_variants_with_stored_summaries([], paper_summaries_by_paper), paper_summaries_by_paper


def _render_paper_topics(evaluations: list[TopicEvaluation]) -> list[str]:
    if not evaluations:
        return ["- 主题匹配：无"]
    lines = ["- 主题匹配："]
    for evaluation in evaluations:
        lines.append(
            f"  - `{evaluation.topic_name}` / `{evaluation.classification}` / "
            f"score `{evaluation.score}` / 关键词 `{', '.join(evaluation.matched_keywords[:8]) or '无'}`"
        )
    return lines


def _render_paper_topics_html(evaluations: list[TopicEvaluation]) -> str:
    if not evaluations:
        return "<p><strong>主题匹配：</strong>无</p>"
    items = "".join(
        (
            "<li>"
            f"<strong>{html.escape(item.topic_name)}</strong> / "
            f"{html.escape(item.classification)} / score {html.escape(str(item.score))} / "
            f"关键词 {html.escape(', '.join(item.matched_keywords[:8]) or '无')}"
            "</li>"
        )
        for item in evaluations
    )
    return f"<div><strong>主题匹配：</strong><ul>{items}</ul></div>"


def _render_paper_markdown(
    paper: PaperRecord,
    evaluations: list[TopicEvaluation],
    source_names: list[str],
    source_urls: list[str],
    summaries: list[PaperLLMSummary],
    variants: list[LLMRuntimeVariant | dict],
) -> str:
    lines = [
        f"# 单篇论文总结 - {paper.title}",
        "",
        f"- paper_id：`{paper.id}`",
        f"- 发布时间：`{paper.published_at or '未知'}`",
        f"- Venue / 分类：`{paper.venue or '未知'}` / `{', '.join(paper.categories) or '无'}`",
        f"- 作者：`{', '.join(paper.authors[:12]) or '未知'}`",
        f"- 来源：`{', '.join(source_names) or paper.source_first or '未知'}`",
        f"- 原始链接：{paper.primary_url or (source_urls[0] if source_urls else '无')}",
        f"- PDF 状态：`{paper.pdf_status}` / 全文状态：`{paper.fulltext_status}` / 页数：`{paper.page_count or '未知'}`",
        f"- 默认总结来源：`{paper.summary_basis or '未知'}`",
    ]
    if paper.pdf_local_path:
        lines.append(f"- 本地 PDF：`{paper.pdf_local_path}`")
    if paper.fulltext_txt_path:
        lines.append(f"- 全文文本：`{paper.fulltext_txt_path}`")
    lines.append("")
    lines.extend(_render_paper_topics(evaluations))
    lines.append("")
    if paper.abstract:
        lines.append("## 原始摘要")
        lines.append("")
        lines.append(paper.abstract)
        lines.append("")
    lines.append("## 默认总结")
    lines.append("")
    lines.append(paper.summary_text or "无")
    lines.append("")
    lines.append("## 多模型详细总结")
    lines.append("")
    lines.extend(_render_summary_lines(variants, summaries))
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def _render_paper_html(
    paper: PaperRecord,
    evaluations: list[TopicEvaluation],
    source_names: list[str],
    source_urls: list[str],
    summaries: list[PaperLLMSummary],
    variants: list[LLMRuntimeVariant | dict],
) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>单篇论文总结 - {html.escape(paper.title)}</title>
  <style>
    :root {{
      --bg: #f3efe7;
      --panel: #fffdf8;
      --ink: #18222d;
      --muted: #52606d;
      --accent: #0a7d6f;
      --border: #d7d0c3;
    }}
    body {{
      margin: 0;
      font-family: "Noto Sans SC", "Source Han Sans SC", "PingFang SC", sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #f9f5ed 0%, var(--bg) 100%);
      line-height: 1.65;
    }}
    main {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 32px 20px 64px;
    }}
    article {{
      padding: 22px;
      border: 1px solid var(--border);
      border-radius: 22px;
      background: rgba(255, 253, 248, 0.94);
      box-shadow: 0 12px 30px rgba(24, 34, 45, 0.06);
    }}
    h1, h2 {{
      font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
      margin: 0 0 14px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 0.96rem;
    }}
    .panel {{
      margin-top: 14px;
      padding: 14px 16px;
      border: 1px solid var(--border);
      border-radius: 16px;
      background: rgba(243, 239, 231, 0.4);
    }}
    .llm-summary-grid {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      margin-top: 10px;
    }}
    .llm-summary-card {{
      padding: 14px 16px;
      border: 1px solid var(--border);
      border-radius: 16px;
      background: rgba(255, 253, 248, 0.96);
    }}
    a {{
      color: var(--accent);
    }}
  </style>
</head>
<body>
  <main>
    <article>
      <h1>{html.escape(paper.title)}</h1>
      <p class="meta">paper_id {paper.id} / 发布时间 {html.escape(paper.published_at or '未知')} / Venue {html.escape(paper.venue or '未知')}</p>
      <p class="meta">来源 {html.escape(', '.join(source_names) or paper.source_first or '未知')} / PDF {html.escape(paper.pdf_status)} / 全文 {html.escape(paper.fulltext_status)} / 页数 {html.escape(str(paper.page_count or '未知'))}</p>
      <p class="meta">作者 {html.escape(', '.join(paper.authors[:12]) or '未知')}</p>
      <p><a href="{html.escape(paper.primary_url or (source_urls[0] if source_urls else '#'))}">打开原始链接</a></p>
      {_render_paper_topics_html(evaluations)}
      <div class="panel">
        <p><strong>默认总结：</strong>{html.escape(paper.summary_text or '无')}</p>
      </div>
      {f'<div class="panel"><p><strong>原始摘要：</strong>{html.escape(paper.abstract)}</p></div>' if paper.abstract else ''}
      <div class="panel">
        <p><strong>多模型详细总结：</strong></p>
        {_render_summary_html(variants, summaries)}
      </div>
      {f'<div class="panel"><p><strong>本地 PDF：</strong>{html.escape(paper.pdf_local_path)}</p></div>' if paper.pdf_local_path else ''}
      {f'<div class="panel"><p><strong>全文文本：</strong>{html.escape(paper.fulltext_txt_path)}</p></div>' if paper.fulltext_txt_path else ''}
    </article>
  </main>
</body>
</html>
"""


def generate_paper_reports(
    db: Database,
    settings: Settings,
    paper_ids: list[int],
    *,
    llm_variants: list[LLMRuntimeVariant] | None = None,
    progress_bar: ProgressBar | None = None,
) -> dict[int, dict[str, str]]:
    unique_paper_ids = list(dict.fromkeys(paper_ids))
    if not unique_paper_ids:
        return {}

    report_root = settings.report_dir / "papers"
    export_root = settings.export_dir / "papers"
    ensure_directory(report_root)
    ensure_directory(export_root)

    summaries_by_paper = db.fetch_paper_llm_summaries(unique_paper_ids)
    outputs: dict[int, dict[str, str]] = {}
    for paper_id in unique_paper_ids:
        paper = db.get_paper(paper_id)
        evaluations = db.fetch_paper_evaluations(paper_id)
        source_names, source_urls = db.fetch_paper_sources(paper_id)
        summaries = summaries_by_paper.get(paper_id, [])
        requested_variants = _variants_for_paper(summaries, llm_variants)
        if llm_variants:
            allowed_variant_ids = _allowed_variant_ids(list(llm_variants))
            summaries = [summary for summary in summaries if summary.variant_id in allowed_variant_ids]
            variants = list(llm_variants)
        else:
            variants = _merge_variants_with_stored_summaries(requested_variants, {paper_id: summaries})
        stem = _paper_report_stem(paper)
        if progress_bar:
            progress_bar.set_detail(f"单篇导出 {shorten(paper.title, 52)}")
        markdown_text = _render_paper_markdown(paper, evaluations, source_names, source_urls, summaries, variants)
        html_text = _render_paper_html(paper, evaluations, source_names, source_urls, summaries, variants)
        json_text = json.dumps(
            {
                "paper_id": paper.id,
                "title": paper.title,
                "published_at": paper.published_at,
                "authors": paper.authors,
                "venue": paper.venue,
                "categories": paper.categories,
                "primary_url": paper.primary_url,
                "pdf_status": paper.pdf_status,
                "pdf_local_path": paper.pdf_local_path,
                "fulltext_status": paper.fulltext_status,
                "fulltext_txt_path": paper.fulltext_txt_path,
                "summary_text": paper.summary_text,
                "summary_basis": paper.summary_basis,
                "topics": [
                    {
                        "topic_id": item.topic_id,
                        "topic_name": item.topic_name,
                        "score": item.score,
                        "classification": item.classification,
                        "matched_keywords": item.matched_keywords,
                        "reasons": item.reasons,
                    }
                    for item in evaluations
                ],
                "sources": {
                    "names": source_names,
                    "urls": source_urls,
                },
                "llm_summaries": [
                    {
                        "variant_id": summary.variant_id,
                        "variant_label": summary.variant_label,
                        "provider": summary.provider,
                        "base_url": summary.base_url,
                        "model": summary.model,
                        "summary_text": summary.summary_text,
                        "summary_basis": summary.summary_basis,
                        "summary_scope": _summary_scope_label(summary),
                        "summary_scope_note": _summary_scope_note(summary),
                        "source_mode": (
                            summary.structured.get("source_mode", "")
                            if isinstance(summary.structured, dict)
                            else ""
                        ),
                        "pdf_input_strategy": (
                            summary.structured.get("pdf_input_strategy", "")
                            if isinstance(summary.structured, dict)
                            else ""
                        ),
                        "direct_pdf_strategy": (
                            summary.structured.get("direct_pdf_strategy", "")
                            if isinstance(summary.structured, dict)
                            else ""
                        ),
                        "direct_pdf_status": (
                            summary.structured.get("direct_pdf_status", "")
                            if isinstance(summary.structured, dict)
                            else ""
                        ),
                        "structured": summary.structured,
                        "usage": summary.usage,
                    }
                    for summary in summaries
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        path_md = report_root / f"{stem}.md"
        path_html = report_root / f"{stem}.html"
        path_json = export_root / f"{stem}.json"
        path_md.write_text(markdown_text, encoding="utf-8")
        path_html.write_text(html_text, encoding="utf-8")
        path_json.write_text(json_text + "\n", encoding="utf-8")
        outputs[paper_id] = {
            "markdown": str(path_md),
            "html": str(path_html),
            "json": str(path_json),
        }
        if progress_bar:
            progress_bar.advance(detail=f"单篇完成 {shorten(paper.title, 52)}")
    return outputs


def _serialize_variants(variants: list[LLMRuntimeVariant | dict]) -> list[dict[str, str]]:
    return [
        {
            "slug": str(_variant_attr(variant, "variant_id") or ""),
            "label": str(_variant_attr(variant, "label") or ""),
            "model": str(_variant_attr(variant, "model") or ""),
            "base_url": str(_variant_attr(variant, "base_url") or ""),
        }
        for variant in variants
    ]


def _active_variant_count(variants: list[LLMRuntimeVariant | dict]) -> int:
    return sum(1 for variant in variants if _variant_attr(variant, "client") and _variant_attr(variant, "client").enabled)


def _render_markdown(
    report_type: str,
    report_date: str,
    start_at: str,
    end_at: str,
    grouped_entries: dict[str, list[ReportEntry]],
    fallback_entries_by_topic: dict[str, list[ReportEntry]],
    fallback_days_by_topic: dict[str, int],
    paper_summaries_by_paper: dict[int, list[PaperLLMSummary]],
    topic_digests_by_variant: dict[str, dict[str, dict]],
    variants: list[LLMRuntimeVariant | dict],
    settings: Settings,
) -> str:
    total = sum(len(items) for items in grouped_entries.values())
    label = _report_label(report_type)
    lines = [
        f"# 论文监控{label} - {report_date}",
        "",
        f"- 时间窗口：`{start_at}` 到 `{end_at}`",
        f"- 主题数量：`{len(settings.topics)}`",
        f"- 命中文章数：`{total}`",
        f"- LLM 模型数：`{len(variants)}`",
        f"- 报告规则：Top `{settings.report.top_n_per_topic}`，`{'包含 maybe' if settings.report.include_maybe else '仅 relevant'}`",
        "",
    ]

    for topic in settings.topics:
        entries = _sorted_entries(grouped_entries.get(topic.id, []))
        fallback_entries = _sorted_entries(fallback_entries_by_topic.get(topic.id, []))
        fallback_days = fallback_days_by_topic.get(topic.id)
        lines.append(f"## {topic.display_name}")
        lines.append("")
        lines.append(topic.description)
        lines.append("")
        lines.append(f"- 命中数量：`{len(entries)}`")
        lines.append(f"- 趋势摘要：{_topic_trend_sentence(entries)}")
        for variant in variants:
            variant_id = str(_variant_attr(variant, "variant_id") or "")
            variant_label = str(_variant_attr(variant, "label") or variant_id or "未知模型")
            digest = topic_digests_by_variant.get(variant_id, {}).get(topic.id)
            if not digest:
                lines.append(f"- {variant_label} 主题概览：未生成")
                continue
            lines.append(f"- {variant_label} 主题概览：{digest.get('overview', '')}")
            for meta_line in _digest_input_meta_lines(digest):
                lines.append(f"- {variant_label} {meta_line[2:]}" if meta_line.startswith("- ") else f"- {variant_label} {meta_line}")
            highlights = digest.get("highlights", [])
            if highlights:
                lines.append(f"- {variant_label} 关键观察：`{' | '.join(highlights[:3])}`")
            watchlist = digest.get("watchlist", [])
            if watchlist:
                lines.append(f"- {variant_label} 建议关注：`{' | '.join(watchlist[:3])}`")
            usage = digest.get("usage", {})
            if usage:
                lines.append(
                    f"- {variant_label} Token：in `{usage.get('input_tokens', '未知')}` / "
                    f"out `{usage.get('output_tokens', '未知')}` / total `{usage.get('total_tokens', '未知')}`"
                )
        lines.append("")

        if not entries:
            lines.append("本窗口内没有新的严格命中论文。")
            lines.append("")
            if fallback_entries and fallback_days:
                lines.append(f"- 自动补充：近 `{fallback_days}` 天高分回顾（按相关性分数降序）")
                lines.append("")
                for index, entry in enumerate(fallback_entries[: settings.report.top_n_per_topic], start=1):
                    _append_report_entry_markdown(lines, entry, index, paper_summaries_by_paper, variants)
            else:
                lines.append("在补充回顾窗口内也没有可展示的严格命中论文。")
            lines.append("")
            continue

        for index, entry in enumerate(entries[: settings.report.top_n_per_topic], start=1):
            _append_report_entry_markdown(lines, entry, index, paper_summaries_by_paper, variants)

    return "\n".join(lines).strip() + "\n"


def _render_html(
    report_type: str,
    report_date: str,
    start_at: str,
    end_at: str,
    grouped_entries: dict[str, list[ReportEntry]],
    fallback_entries_by_topic: dict[str, list[ReportEntry]],
    fallback_days_by_topic: dict[str, int],
    paper_summaries_by_paper: dict[int, list[PaperLLMSummary]],
    topic_digests_by_variant: dict[str, dict[str, dict]],
    variants: list[LLMRuntimeVariant | dict],
    settings: Settings,
) -> str:
    label = _report_label(report_type)
    sections: list[str] = []
    for topic in settings.topics:
        entries = _sorted_entries(grouped_entries.get(topic.id, []))
        fallback_entries = _sorted_entries(fallback_entries_by_topic.get(topic.id, []))
        fallback_days = fallback_days_by_topic.get(topic.id)
        digest_cards: list[str] = []
        for variant in variants:
            variant_id = str(_variant_attr(variant, "variant_id") or "")
            variant_label = str(_variant_attr(variant, "label") or variant_id or "未知模型")
            digest = topic_digests_by_variant.get(variant_id, {}).get(topic.id, {})
            digest_cards.append(
                f"""
                <article class="topic-digest-card">
                  <p class="meta">模型 {html.escape(variant_label)}</p>
                  <p><strong>主题概览：</strong>{html.escape(digest.get('overview', '未生成'))}</p>
                  {_digest_input_meta_html(digest)}
                  {f"<p><strong>关键观察：</strong>{html.escape(' | '.join(digest.get('highlights', [])[:3]))}</p>" if digest.get('highlights') else ''}
                  {f"<p><strong>建议关注：</strong>{html.escape(' | '.join(digest.get('watchlist', [])[:3]))}</p>" if digest.get('watchlist') else ''}
                </article>
                """
            )
        cards: list[str] = []
        for entry in entries[: settings.report.top_n_per_topic]:
            cards.append(_render_report_entry_html(entry, paper_summaries_by_paper, variants))
        fallback_note = ""
        if not cards:
            if fallback_entries and fallback_days:
                fallback_note = (
                    f'<p class="trend">本窗口内没有新的严格命中论文，已自动补充近 {fallback_days} 天高分回顾。</p>'
                )
                cards.extend(
                    _render_report_entry_html(entry, paper_summaries_by_paper, variants)
                    for entry in fallback_entries[: settings.report.top_n_per_topic]
                )
            else:
                cards.append('<article class="paper-card"><p>本窗口内没有新的严格命中论文，且补充回顾窗口内也没有可展示论文。</p></article>')
        sections.append(
            f"""
            <section class="topic-section">
              <div class="topic-header">
                <h2>{html.escape(topic.display_name)}</h2>
                <p>{html.escape(topic.description)}</p>
                <p class="trend">{html.escape(_topic_trend_sentence(entries))}</p>
                {fallback_note}
              </div>
              <div class="topic-digest-grid">
                {''.join(digest_cards)}
              </div>
              {''.join(cards)}
            </section>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>论文监控{html.escape(label)} - {html.escape(report_date)}</title>
  <style>
    :root {{
      --bg: #f3efe7;
      --panel: #fffdf8;
      --ink: #18222d;
      --muted: #52606d;
      --accent: #0a7d6f;
      --border: #d7d0c3;
    }}
    body {{
      margin: 0;
      font-family: "Noto Sans SC", "Source Han Sans SC", "PingFang SC", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(10, 125, 111, 0.14), transparent 28%),
        linear-gradient(180deg, #f9f5ed 0%, var(--bg) 100%);
      line-height: 1.6;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 64px;
    }}
    header {{
      margin-bottom: 28px;
      padding: 28px;
      border: 1px solid var(--border);
      border-radius: 24px;
      background: rgba(255, 253, 248, 0.86);
      backdrop-filter: blur(8px);
    }}
    h1, h2, h3 {{
      font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
      margin: 0 0 12px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 0.96rem;
    }}
    .topic-section {{
      margin-top: 28px;
    }}
    .topic-header {{
      margin-bottom: 14px;
    }}
    .trend {{
      color: var(--accent);
      font-weight: 600;
    }}
    .topic-digest-grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      margin-bottom: 16px;
    }}
    .topic-digest-card {{
      padding: 16px;
      border: 1px solid var(--border);
      border-radius: 18px;
      background: rgba(255, 253, 248, 0.92);
    }}
    .paper-card {{
      padding: 18px 18px 14px;
      margin-bottom: 16px;
      border: 1px solid var(--border);
      border-radius: 18px;
      background: var(--panel);
      box-shadow: 0 12px 30px rgba(24, 34, 45, 0.06);
    }}
    .paper-grid {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      margin: 12px 0;
    }}
    .paper-panel {{
      padding: 12px 14px;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: rgba(243, 239, 231, 0.45);
    }}
    .llm-summary-section {{
      margin-top: 12px;
    }}
    .llm-summary-grid {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      margin-top: 8px;
    }}
    .llm-summary-card {{
      padding: 14px 16px;
      border: 1px solid var(--border);
      border-radius: 16px;
      background: rgba(255, 253, 248, 0.94);
    }}
    a {{
      color: var(--accent);
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>论文监控{html.escape(label)}</h1>
      <p class="meta">报告日期 {html.escape(report_date)}</p>
      <p class="meta">时间窗口 {html.escape(start_at)} 到 {html.escape(end_at)}</p>
      <p class="meta">Top {settings.report.top_n_per_topic}，{'包含 maybe' if settings.report.include_maybe else '仅 relevant'}</p>
      <p class="meta">LLM 模型数 {len(variants)}</p>
    </header>
    {''.join(sections)}
  </main>
</body>
</html>
"""


def _render_catalog_markdown(
    generated_at: str,
    grouped_entries: dict[str, list[ReportEntry]],
    paper_summaries_by_paper: dict[int, list[PaperLLMSummary]],
    topic_digests_by_variant: dict[str, dict[str, dict]],
    variants: list[LLMRuntimeVariant],
    settings: Settings,
) -> str:
    total = sum(len(items) for items in grouped_entries.values())
    unique_paper_ids = {entry.paper.id for items in grouped_entries.values() for entry in items}
    lines = [
        f"# 论文库总览 - {generated_at[:19]}",
        "",
        f"- 生成时间：`{generated_at}`",
        f"- 主题数量：`{len(settings.topics)}`",
        f"- 领域命中文章数：`{total}`",
        f"- 去重后论文数：`{len(unique_paper_ids)}`",
        f"- LLM 模型数：`{len(variants)}`",
        "",
    ]

    for topic in settings.topics:
        entries = _sorted_entries(grouped_entries.get(topic.id, []))
        lines.append(f"## {topic.display_name}")
        lines.append("")
        lines.append(topic.description)
        lines.append("")
        lines.append(f"- 当前论文数：`{len(entries)}`")
        lines.append(f"- 总览：{_catalog_topic_sentence(entries)}")
        for variant in variants:
            variant_id = str(_variant_attr(variant, "variant_id") or "")
            variant_label = str(_variant_attr(variant, "label") or variant_id or "未知模型")
            digest = topic_digests_by_variant.get(variant_id, {}).get(topic.id)
            if not digest:
                lines.append(f"- {variant_label} 领域概览：未生成")
                continue
            lines.append(f"- {variant_label} 领域概览：{digest.get('overview', '')}")
            for meta_line in _digest_input_meta_lines(digest):
                lines.append(f"- {variant_label} {meta_line[2:]}" if meta_line.startswith("- ") else f"- {variant_label} {meta_line}")
            highlights = digest.get("highlights", [])
            if highlights:
                lines.append(f"- {variant_label} 关键观察：`{' | '.join(highlights[:3])}`")
            watchlist = digest.get("watchlist", [])
            if watchlist:
                lines.append(f"- {variant_label} 建议关注：`{' | '.join(watchlist[:3])}`")
        lines.append("")

        if not entries:
            lines.append("当前没有已入库论文。")
            lines.append("")
            continue

        for index, entry in enumerate(entries, start=1):
            paper = entry.paper
            lines.append(f"### {index}. {paper.title}")
            lines.append("")
            lines.append(f"- paper_id：`{paper.id}`")
            lines.append(f"- 相关性：`{entry.classification}` / 分数 `{entry.score}`")
            lines.append(f"- 发布时间：`{paper.published_at or '未知'}` / 最近入库：`{paper.created_at}`")
            lines.append(f"- 来源：`{', '.join(entry.source_names) or paper.source_first}`")
            lines.append(f"- Venue / 分类：`{paper.venue or '未知'}` / `{', '.join(paper.categories) or '无'}`")
            lines.append(
                f"- 全文状态：`{paper.fulltext_status}` / PDF `{paper.pdf_status}` / "
                f"页数 `{paper.page_count or '未知'}` / 默认总结来源 `{paper.summary_basis or '未知'}`"
            )
            lines.append(f"- 匹配词：`{', '.join(entry.matched_keywords) or '无'}`")
            lines.append(f"- 链接：{paper.primary_url or (entry.source_urls[0] if entry.source_urls else '无')}")
            lines.append(f"- 默认总结：{paper.summary_text or '无'}")
            lines.append("#### 多模型摘要")
            lines.extend(_render_summary_lines(variants, paper_summaries_by_paper.get(paper.id, [])))
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def _render_catalog_html(
    generated_at: str,
    grouped_entries: dict[str, list[ReportEntry]],
    paper_summaries_by_paper: dict[int, list[PaperLLMSummary]],
    topic_digests_by_variant: dict[str, dict[str, dict]],
    variants: list[LLMRuntimeVariant],
    settings: Settings,
) -> str:
    total = sum(len(items) for items in grouped_entries.values())
    unique_paper_ids = {entry.paper.id for items in grouped_entries.values() for entry in items}
    sections: list[str] = []
    for topic in settings.topics:
        entries = _sorted_entries(grouped_entries.get(topic.id, []))
        digest_cards: list[str] = []
        for variant in variants:
            variant_id = str(_variant_attr(variant, "variant_id") or "")
            variant_label = str(_variant_attr(variant, "label") or variant_id or "未知模型")
            digest = topic_digests_by_variant.get(variant_id, {}).get(topic.id, {})
            digest_cards.append(
                f"""
                <article class="topic-digest-card">
                  <p class="meta">模型 {html.escape(variant_label)}</p>
                  <p><strong>领域概览：</strong>{html.escape(digest.get('overview', '未生成'))}</p>
                  {_digest_input_meta_html(digest)}
                  {f"<p><strong>关键观察：</strong>{html.escape(' | '.join(digest.get('highlights', [])[:3]))}</p>" if digest.get('highlights') else ''}
                  {f"<p><strong>建议关注：</strong>{html.escape(' | '.join(digest.get('watchlist', [])[:3]))}</p>" if digest.get('watchlist') else ''}
                </article>
                """
            )
        cards: list[str] = []
        for entry in entries:
            paper = entry.paper
            cards.append(
                f"""
                <article class="paper-card">
                  <h3>{html.escape(paper.title)}</h3>
                  <p class="meta">paper_id {paper.id} / 相关性 {entry.classification} / 分数 {entry.score}</p>
                  <p class="meta">发布时间 {html.escape(paper.published_at or '未知')} / 最近入库 {html.escape(paper.created_at)}</p>
                  <p class="meta">来源 {html.escape(', '.join(entry.source_names) or paper.source_first)} / Venue {html.escape(paper.venue or '未知')}</p>
                  <p class="meta">全文状态 {html.escape(paper.fulltext_status)} / PDF {html.escape(paper.pdf_status)} / 页数 {html.escape(str(paper.page_count or '未知'))}</p>
                  <div class="paper-grid">
                    <div class="paper-panel">
                      <p><strong>匹配词：</strong>{html.escape(', '.join(entry.matched_keywords) or '无')}</p>
                      <p><strong>默认总结：</strong>{html.escape(paper.summary_text or '无')}</p>
                    </div>
                    <div class="paper-panel">
                      <p><strong>链接：</strong><a href="{html.escape(paper.primary_url or (entry.source_urls[0] if entry.source_urls else '#'))}">原始链接</a></p>
                      <p><strong>分类：</strong>{html.escape(', '.join(paper.categories) or '无')}</p>
                    </div>
                  </div>
                  <div class="llm-summary-section">
                    <p><strong>多模型摘要：</strong></p>
                    {_render_summary_html(variants, paper_summaries_by_paper.get(paper.id, []))}
                  </div>
                </article>
                """
            )
        if not cards:
            cards.append('<article class="paper-card"><p>当前没有已入库论文。</p></article>')
        sections.append(
            f"""
            <section class="topic-section">
              <div class="topic-header">
                <h2>{html.escape(topic.display_name)}</h2>
                <p>{html.escape(topic.description)}</p>
                <p class="trend">{html.escape(_catalog_topic_sentence(entries))}</p>
              </div>
              <div class="topic-digest-grid">
                {''.join(digest_cards)}
              </div>
              {''.join(cards)}
            </section>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>论文库总览</title>
  <style>
    :root {{
      --bg: #f3efe7;
      --panel: #fffdf8;
      --ink: #18222d;
      --muted: #52606d;
      --accent: #0a7d6f;
      --border: #d7d0c3;
    }}
    body {{
      margin: 0;
      font-family: "Noto Sans SC", "Source Han Sans SC", "PingFang SC", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(10, 125, 111, 0.14), transparent 28%),
        linear-gradient(180deg, #f9f5ed 0%, var(--bg) 100%);
      line-height: 1.6;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 64px;
    }}
    header {{
      margin-bottom: 28px;
      padding: 28px;
      border: 1px solid var(--border);
      border-radius: 24px;
      background: rgba(255, 253, 248, 0.86);
      backdrop-filter: blur(8px);
    }}
    h1, h2, h3 {{
      font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
      margin: 0 0 12px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 0.96rem;
    }}
    .topic-section {{
      margin-top: 28px;
    }}
    .topic-header {{
      margin-bottom: 14px;
    }}
    .trend {{
      color: var(--accent);
      font-weight: 600;
    }}
    .topic-digest-grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      margin-bottom: 16px;
    }}
    .topic-digest-card {{
      padding: 16px;
      border: 1px solid var(--border);
      border-radius: 18px;
      background: rgba(255, 253, 248, 0.92);
    }}
    .paper-card {{
      padding: 18px 18px 14px;
      margin-bottom: 16px;
      border: 1px solid var(--border);
      border-radius: 18px;
      background: var(--panel);
      box-shadow: 0 12px 30px rgba(24, 34, 45, 0.06);
    }}
    .paper-grid {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      margin: 12px 0;
    }}
    .paper-panel {{
      padding: 12px 14px;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: rgba(243, 239, 231, 0.45);
    }}
    .llm-summary-section {{
      margin-top: 12px;
    }}
    .llm-summary-grid {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      margin-top: 8px;
    }}
    .llm-summary-card {{
      padding: 14px 16px;
      border: 1px solid var(--border);
      border-radius: 16px;
      background: rgba(255, 253, 248, 0.94);
    }}
    a {{
      color: var(--accent);
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>论文库总览</h1>
      <p class="meta">生成时间 {html.escape(generated_at)}</p>
      <p class="meta">领域命中文章数 {total} / 去重后论文数 {len(unique_paper_ids)} / LLM 模型数 {len(variants)}</p>
    </header>
    {''.join(sections)}
  </main>
</body>
</html>
"""


def generate_catalog_report(
    db: Database,
    settings: Settings,
    *,
    topic_ids: list[str] | None = None,
    classifications: list[str] | None = None,
    llm_variants: list[LLMRuntimeVariant] | None = None,
    topic_digest_variants: list[LLMRuntimeVariant] | None = None,
    use_llm_topic_digest: bool = False,
) -> dict[str, str]:
    report_variants = list(llm_variants or [])
    digest_variants = list(topic_digest_variants or report_variants)
    digest_steps = _active_variant_count(digest_variants) * len(settings.topics) if use_llm_topic_digest else 0
    progress_bar = ProgressBar("论文库总览", 6 + digest_steps)
    progress_bar.advance(detail="加载数据库论文")
    entries = db.fetch_catalog_entries(
        include_maybe=settings.report.include_maybe,
        topic_ids=topic_ids,
        classifications=classifications,
    )
    grouped_entries: dict[str, list[ReportEntry]] = defaultdict(list)
    for entry in entries:
        grouped_entries[entry.topic_id].append(entry)

    progress_bar.advance(detail="读取逐篇 LLM 摘要")
    paper_summaries_by_paper = db.fetch_paper_llm_summaries([entry.paper.id for entry in entries])
    display_variants, visible_summaries_by_paper = _resolve_display_variants(report_variants, paper_summaries_by_paper)
    topic_digests_by_variant = _collect_topic_digests_by_variant(
        settings,
        grouped_entries,
        visible_summaries_by_paper,
        digest_variants,
        use_llm_topic_digest,
        progress_callback=lambda detail, advance: (
            progress_bar.advance(detail=detail) if advance else progress_bar.set_detail(detail)
        ),
    )

    progress_bar.advance(detail="导出单篇论文报告")
    paper_report_outputs = generate_paper_reports(
        db,
        settings,
        [entry.paper.id for entry in entries],
        llm_variants=display_variants,
    )

    generated_at = now_iso(settings.timezone)
    report_root = settings.report_dir / "catalog"
    ensure_directory(report_root)
    ensure_directory(settings.export_dir)

    progress_bar.advance(detail="渲染 Markdown")
    markdown_text = _render_catalog_markdown(
        generated_at,
        grouped_entries,
        visible_summaries_by_paper,
        topic_digests_by_variant,
        display_variants,
        settings,
    )
    progress_bar.advance(detail="渲染 HTML")
    html_text = _render_catalog_html(
        generated_at,
        grouped_entries,
        visible_summaries_by_paper,
        topic_digests_by_variant,
        display_variants,
        settings,
    )
    progress_bar.advance(detail="导出 JSON 与写入文件")
    json_text = json.dumps(
        {
            "generated_at": generated_at,
            "variants": _serialize_variants(display_variants),
            "topic_digests": topic_digests_by_variant,
            "topics": {
                topic.id: [
                    {
                        "topic_name": entry.topic_name,
                        "score": entry.score,
                        "classification": entry.classification,
                        "matched_keywords": entry.matched_keywords,
                        "paper": {
                            "id": entry.paper.id,
                            "title": entry.paper.title,
                            "published_at": entry.paper.published_at,
                            "created_at": entry.paper.created_at,
                            "primary_url": entry.paper.primary_url,
                            "summary_text": entry.paper.summary_text,
                            "summary_basis": entry.paper.summary_basis,
                            "paper_report": paper_report_outputs.get(entry.paper.id, {}),
                            "llm_summaries": [
                                {
                                    "variant_id": summary.variant_id,
                                    "variant_label": summary.variant_label,
                                    "model": summary.model,
                                    "summary_text": summary.summary_text,
                                    "summary_basis": summary.summary_basis,
                                    "summary_scope": _summary_scope_label(summary),
                                    "summary_scope_note": _summary_scope_note(summary),
                                    "structured": summary.structured,
                                }
                                for summary in visible_summaries_by_paper.get(entry.paper.id, [])
                            ],
                        },
                    }
                    for entry in _sorted_entries(grouped_entries.get(topic.id, []))
                ]
                for topic in settings.topics
            },
        },
        ensure_ascii=False,
        indent=2,
    )

    path_md = report_root / "current.md"
    path_html = report_root / "current.html"
    path_json = settings.export_dir / "catalog-current.json"
    path_md.write_text(markdown_text, encoding="utf-8")
    path_html.write_text(html_text, encoding="utf-8")
    path_json.write_text(json_text + "\n", encoding="utf-8")
    progress_bar.close("论文库总览已生成")
    return {
        "markdown": str(path_md),
        "html": str(path_html),
        "json": str(path_json),
        "papers_dir": str(settings.report_dir / "papers"),
    }


def _render_comparison_markdown(
    report_type: str,
    report_date: str,
    start_at: str,
    end_at: str,
    grouped_entries: dict[str, list[ReportEntry]],
    variants: list[dict[str, str]],
    digests_by_variant: dict[str, dict[str, dict]],
    settings: Settings,
) -> str:
    label = _report_label(report_type)
    variant_names = " / ".join(item["label"] for item in variants)
    lines = [
        f"# 论文监控{label}模型对比 - {report_date}",
        "",
        f"- 时间窗口：`{start_at}` 到 `{end_at}`",
        f"- 对比配置：`{variant_names}`",
        f"- 主题数量：`{len(settings.topics)}`",
        "",
    ]

    for topic in settings.topics:
        entries = _sorted_entries(grouped_entries.get(topic.id, []))
        lines.append(f"## {topic.display_name}")
        lines.append("")
        lines.append(topic.description)
        lines.append("")
        lines.append(f"- 命中数量：`{len(entries)}`")
        lines.append(f"- 共享趋势：{_topic_trend_sentence(entries)}")
        if entries:
            titles = " | ".join(entry.paper.title for entry in entries[: settings.report.top_n_per_topic])
            lines.append(f"- Top 论文：`{titles}`")
        lines.append("")
        for variant in variants:
            digest = digests_by_variant.get(variant["slug"], {}).get(topic.id, {})
            lines.append(f"### {variant['label']}")
            lines.append("")
            lines.append(f"- 模型：`{variant['model']}`")
            lines.append(f"- 基础地址：`{variant['base_url']}`")
            if digest:
                lines.append(f"- LLM 主题概览：{digest.get('overview', '')}")
                lines.extend(_digest_input_meta_lines(digest))
                highlights = digest.get("highlights", [])
                if highlights:
                    lines.append(f"- LLM 关键观察：`{' | '.join(highlights[:3])}`")
                watchlist = digest.get("watchlist", [])
                if watchlist:
                    lines.append(f"- LLM 建议关注：`{' | '.join(watchlist[:3])}`")
                usage = digest.get("usage", {})
                if usage:
                    lines.append(
                        f"- LLM Token：in `{usage.get('input_tokens', '未知')}` / "
                        f"out `{usage.get('output_tokens', '未知')}` / total `{usage.get('total_tokens', '未知')}`"
                    )
            else:
                lines.append("- LLM 主题概览：未生成")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def _render_comparison_html(
    report_type: str,
    report_date: str,
    start_at: str,
    end_at: str,
    grouped_entries: dict[str, list[ReportEntry]],
    variants: list[dict[str, str]],
    digests_by_variant: dict[str, dict[str, dict]],
    settings: Settings,
) -> str:
    label = _report_label(report_type)
    sections: list[str] = []
    for topic in settings.topics:
        entries = _sorted_entries(grouped_entries.get(topic.id, []))
        variant_blocks: list[str] = []
        for variant in variants:
            digest = digests_by_variant.get(variant["slug"], {}).get(topic.id, {})
            variant_blocks.append(
                f"""
                <article class="compare-card">
                  <h3>{html.escape(variant['label'])}</h3>
                  <p class="meta">模型 {html.escape(variant['model'])}</p>
                  <p class="meta">Base URL {html.escape(variant['base_url'])}</p>
                  <p><strong>LLM 主题概览：</strong>{html.escape(digest.get('overview', '未生成'))}</p>
                  {_digest_input_meta_html(digest)}
                  {f"<p><strong>LLM 关键观察：</strong>{html.escape(' | '.join(digest.get('highlights', [])[:3]))}</p>" if digest.get('highlights') else ''}
                  {f"<p><strong>LLM 建议关注：</strong>{html.escape(' | '.join(digest.get('watchlist', [])[:3]))}</p>" if digest.get('watchlist') else ''}
                </article>
                """
            )
        sections.append(
            f"""
            <section class="topic-section">
              <div class="topic-header">
                <h2>{html.escape(topic.display_name)}</h2>
                <p>{html.escape(topic.description)}</p>
                <p class="trend">{html.escape(_topic_trend_sentence(entries))}</p>
              </div>
              <div class="compare-grid">
                {''.join(variant_blocks)}
              </div>
            </section>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>论文监控{html.escape(label)}模型对比 - {html.escape(report_date)}</title>
  <style>
    :root {{
      --bg: #f3efe7;
      --panel: #fffdf8;
      --ink: #18222d;
      --muted: #52606d;
      --accent: #0a7d6f;
      --border: #d7d0c3;
    }}
    body {{
      margin: 0;
      font-family: "Noto Sans SC", "Source Han Sans SC", "PingFang SC", sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #f9f5ed 0%, var(--bg) 100%);
      line-height: 1.6;
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 64px;
    }}
    header {{
      margin-bottom: 28px;
      padding: 28px;
      border: 1px solid var(--border);
      border-radius: 24px;
      background: rgba(255, 253, 248, 0.92);
    }}
    h1, h2, h3 {{
      font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
      margin: 0 0 12px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 0.96rem;
    }}
    .topic-section {{
      margin-top: 28px;
    }}
    .topic-header {{
      margin-bottom: 14px;
    }}
    .trend {{
      color: var(--accent);
      font-weight: 600;
    }}
    .compare-grid {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    }}
    .compare-card {{
      padding: 18px;
      border: 1px solid var(--border);
      border-radius: 18px;
      background: var(--panel);
      box-shadow: 0 12px 30px rgba(24, 34, 45, 0.06);
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>论文监控{html.escape(label)}模型对比</h1>
      <p class="meta">报告日期 {html.escape(report_date)}</p>
      <p class="meta">时间窗口 {html.escape(start_at)} 到 {html.escape(end_at)}</p>
      <p class="meta">对比配置 {' / '.join(html.escape(item['label']) for item in variants)}</p>
    </header>
    {''.join(sections)}
  </main>
</body>
</html>
"""


def generate_report(
    db: Database,
    settings: Settings,
    report_date: str,
    report_type: str,
    lookback_days: int | None = None,
    llm_client: LLMClient | None = None,
    llm_variants: list[LLMRuntimeVariant] | None = None,
    topic_digest_variants: list[LLMRuntimeVariant] | None = None,
    use_llm_topic_digest: bool = False,
) -> dict[str, str]:
    actual_lookback = lookback_days or settings.report.lookback_days
    start_at, end_at = to_day_bounds(report_date, settings.timezone, actual_lookback)
    report_variants = list(llm_variants or [])
    if llm_client is not None and not report_variants:
        report_variants = [_single_runtime_variant(settings, llm_client)]
    digest_variants = list(topic_digest_variants or report_variants)
    digest_steps = _active_variant_count(digest_variants) * len(settings.topics) if use_llm_topic_digest else 0
    progress_bar = ProgressBar(_report_label(report_type), 6 + digest_steps)
    progress_bar.advance(detail="加载命中论文")
    entries = db.fetch_report_entries(start_at, end_at, settings.report.include_maybe)
    grouped_entries: dict[str, list[ReportEntry]] = defaultdict(list)
    for entry in entries:
        grouped_entries[entry.topic_id].append(entry)
    fallback_entries_by_topic, fallback_days_by_topic = _collect_fallback_review_entries(
        db,
        settings,
        report_type,
        report_date,
        grouped_entries,
    )
    digest_grouped_entries = {
        topic.id: _sorted_entries(grouped_entries.get(topic.id, []) or fallback_entries_by_topic.get(topic.id, []))
        for topic in settings.topics
    }
    all_report_entries: list[ReportEntry] = []
    for items in grouped_entries.values():
        all_report_entries.extend(items)
    for items in fallback_entries_by_topic.values():
        all_report_entries.extend(items)
    all_paper_ids = list({entry.paper.id for entry in all_report_entries})

    progress_bar.advance(detail="读取逐篇 LLM 摘要")
    paper_summaries_by_paper = db.fetch_paper_llm_summaries(all_paper_ids)
    display_variants, visible_summaries_by_paper = _resolve_display_variants(report_variants, paper_summaries_by_paper)
    topic_digests_by_variant = _collect_topic_digests_by_variant(
        settings,
        digest_grouped_entries,
        visible_summaries_by_paper,
        digest_variants,
        use_llm_topic_digest,
        progress_callback=lambda detail, advance: (
            progress_bar.advance(detail=detail) if advance else progress_bar.set_detail(detail)
        ),
    )

    report_root = settings.report_dir / report_type
    ensure_directory(report_root)
    ensure_directory(settings.export_dir)

    progress_bar.advance(detail="导出单篇论文报告")
    paper_report_outputs = generate_paper_reports(
        db,
        settings,
        all_paper_ids,
        llm_variants=display_variants,
    )

    progress_bar.advance(detail="渲染 Markdown")
    markdown_text = _render_markdown(
        report_type,
        report_date,
        start_at,
        end_at,
        grouped_entries,
        fallback_entries_by_topic,
        fallback_days_by_topic,
        visible_summaries_by_paper,
        topic_digests_by_variant,
        display_variants,
        settings,
    )
    progress_bar.advance(detail="渲染 HTML")
    html_text = _render_html(
        report_type,
        report_date,
        start_at,
        end_at,
        grouped_entries,
        fallback_entries_by_topic,
        fallback_days_by_topic,
        visible_summaries_by_paper,
        topic_digests_by_variant,
        display_variants,
        settings,
    )
    progress_bar.advance(detail="导出 JSON 与写入文件")
    json_text = json.dumps(
        {
            "report_type": report_type,
            "report_date": report_date,
            "start_at": start_at,
            "end_at": end_at,
            "variants": _serialize_variants(display_variants),
            "topic_digests": topic_digests_by_variant,
            "topics": {
                topic_id: [
                    {
                        "topic_name": entry.topic_name,
                        "score": entry.score,
                        "classification": entry.classification,
                        "matched_keywords": entry.matched_keywords,
                        "reasons": entry.reasons,
                        "paper": {
                            "title": entry.paper.title,
                            "published_at": entry.paper.published_at,
                            "primary_url": entry.paper.primary_url,
                            "summary_text": entry.paper.summary_text,
                            "summary_basis": entry.paper.summary_basis,
                            "venue": entry.paper.venue,
                            "categories": entry.paper.categories,
                            "pdf_status": entry.paper.pdf_status,
                            "pdf_local_path": entry.paper.pdf_local_path,
                            "fulltext_status": entry.paper.fulltext_status,
                            "fulltext_txt_path": entry.paper.fulltext_txt_path,
                            "page_count": entry.paper.page_count,
                            "llm_summaries": [
                                {
                                    "variant_id": summary.variant_id,
                                    "variant_label": summary.variant_label,
                                    "model": summary.model,
                                    "summary_text": summary.summary_text,
                                    "summary_basis": summary.summary_basis,
                                    "summary_scope": _summary_scope_label(summary),
                                    "summary_scope_note": _summary_scope_note(summary),
                                    "source_mode": (
                                        summary.structured.get("source_mode", "")
                                        if isinstance(summary.structured, dict)
                                        else ""
                                    ),
                                    "pdf_input_strategy": (
                                        summary.structured.get("pdf_input_strategy", "")
                                        if isinstance(summary.structured, dict)
                                        else ""
                                    ),
                                    "direct_pdf_strategy": (
                                        summary.structured.get("direct_pdf_strategy", "")
                                        if isinstance(summary.structured, dict)
                                        else ""
                                    ),
                                    "direct_pdf_status": (
                                        summary.structured.get("direct_pdf_status", "")
                                        if isinstance(summary.structured, dict)
                                        else ""
                                    ),
                                    "chunk_count": (
                                        summary.structured.get("chunk_count")
                                        if isinstance(summary.structured, dict)
                                        else None
                                    ),
                                    "structured": summary.structured,
                                    "usage": summary.usage,
                                }
                                for summary in visible_summaries_by_paper.get(entry.paper.id, [])
                            ],
                            "paper_report": paper_report_outputs.get(entry.paper.id, {}),
                        },
                    }
                    for entry in topic_entries
                ]
                for topic_id, topic_entries in grouped_entries.items()
            },
            "fallback_reviews": {
                topic_id: {
                    "days": fallback_days_by_topic[topic_id],
                    "entries": [
                        {
                            "topic_name": entry.topic_name,
                            "score": entry.score,
                            "classification": entry.classification,
                            "matched_keywords": entry.matched_keywords,
                            "reasons": entry.reasons,
                            "paper": {
                                "title": entry.paper.title,
                                "published_at": entry.paper.published_at,
                                "primary_url": entry.paper.primary_url,
                                "summary_text": entry.paper.summary_text,
                                "summary_basis": entry.paper.summary_basis,
                                "venue": entry.paper.venue,
                                "categories": entry.paper.categories,
                                "pdf_status": entry.paper.pdf_status,
                                "pdf_local_path": entry.paper.pdf_local_path,
                                "fulltext_status": entry.paper.fulltext_status,
                                "fulltext_txt_path": entry.paper.fulltext_txt_path,
                                "page_count": entry.paper.page_count,
                                "llm_summaries": [
                                    {
                                        "variant_id": summary.variant_id,
                                        "variant_label": summary.variant_label,
                                        "model": summary.model,
                                        "summary_text": summary.summary_text,
                                        "summary_basis": summary.summary_basis,
                                        "summary_scope": _summary_scope_label(summary),
                                        "summary_scope_note": _summary_scope_note(summary),
                                        "source_mode": (
                                            summary.structured.get("source_mode", "")
                                            if isinstance(summary.structured, dict)
                                            else ""
                                        ),
                                        "pdf_input_strategy": (
                                            summary.structured.get("pdf_input_strategy", "")
                                            if isinstance(summary.structured, dict)
                                            else ""
                                        ),
                                        "direct_pdf_strategy": (
                                            summary.structured.get("direct_pdf_strategy", "")
                                            if isinstance(summary.structured, dict)
                                            else ""
                                        ),
                                        "direct_pdf_status": (
                                            summary.structured.get("direct_pdf_status", "")
                                            if isinstance(summary.structured, dict)
                                            else ""
                                        ),
                                        "chunk_count": (
                                            summary.structured.get("chunk_count")
                                            if isinstance(summary.structured, dict)
                                            else None
                                        ),
                                        "structured": summary.structured,
                                        "usage": summary.usage,
                                    }
                                    for summary in visible_summaries_by_paper.get(entry.paper.id, [])
                                ],
                                "paper_report": paper_report_outputs.get(entry.paper.id, {}),
                            },
                        }
                        for entry in topic_entries
                    ],
                }
                for topic_id, topic_entries in fallback_entries_by_topic.items()
            },
        },
        ensure_ascii=False,
        indent=2,
    )

    path_md = report_root / f"{report_date}.md"
    path_html = report_root / f"{report_date}.html"
    path_json = settings.export_dir / f"{report_type}-{report_date}.json"
    path_md.write_text(markdown_text, encoding="utf-8")
    path_html.write_text(html_text, encoding="utf-8")
    path_json.write_text(json_text + "\n", encoding="utf-8")

    db.record_report(
        report_type=report_type,
        report_date=report_date,
        path_md=str(path_md),
        path_html=str(path_html),
        path_json=str(path_json),
        meta={
            "match_count": len(entries),
            "lookback_days": actual_lookback,
            "top_n_per_topic": settings.report.top_n_per_topic,
            "topic_digests": {
                variant_id: list(items.keys()) for variant_id, items in topic_digests_by_variant.items()
            },
            "fallback_reviews": fallback_days_by_topic,
            "variants": _serialize_variants(display_variants),
            "paper_reports": paper_report_outputs,
        },
    )
    progress_bar.close(f"{_report_label(report_type)}已生成 {report_date}")

    return {
        "markdown": str(path_md),
        "html": str(path_html),
        "json": str(path_json),
        "papers_dir": str(settings.report_dir / "papers"),
    }


def generate_comparison_report(
    db: Database,
    settings: Settings,
    report_date: str,
    report_type: str,
    variants: list[LLMRuntimeVariant],
    lookback_days: int | None = None,
) -> dict[str, str]:
    actual_lookback = lookback_days or settings.report.lookback_days
    progress_bar = ProgressBar(f"{_report_label(report_type)}对比", 4 + _active_variant_count(variants) * len(settings.topics))
    progress_bar.advance(detail="加载命中论文")
    start_at, end_at = to_day_bounds(report_date, settings.timezone, actual_lookback)
    entries = db.fetch_report_entries(start_at, end_at, settings.report.include_maybe)
    grouped_entries: dict[str, list[ReportEntry]] = defaultdict(list)
    for entry in entries:
        grouped_entries[entry.topic_id].append(entry)
    paper_summaries_by_paper = db.fetch_paper_llm_summaries([entry.paper.id for entry in entries])

    digests_by_variant = _collect_topic_digests_by_variant(
        settings,
        grouped_entries,
        paper_summaries_by_paper,
        variants,
        True,
        progress_callback=lambda detail, advance: (
            progress_bar.advance(detail=detail) if advance else progress_bar.set_detail(detail)
        ),
    )
    serialized_variants = _serialize_variants(variants)

    report_root = settings.report_dir / "compare"
    ensure_directory(report_root)
    ensure_directory(settings.export_dir)

    progress_bar.advance(detail="渲染 Markdown")
    markdown_text = _render_comparison_markdown(
        report_type,
        report_date,
        start_at,
        end_at,
        grouped_entries,
        serialized_variants,
        digests_by_variant,
        settings,
    )
    progress_bar.advance(detail="渲染 HTML")
    html_text = _render_comparison_html(
        report_type,
        report_date,
        start_at,
        end_at,
        grouped_entries,
        serialized_variants,
        digests_by_variant,
        settings,
    )
    progress_bar.advance(detail="导出 JSON 与写入文件")
    json_text = json.dumps(
        {
            "report_type": report_type,
            "report_date": report_date,
            "start_at": start_at,
            "end_at": end_at,
            "variants": serialized_variants,
            "topic_digests": digests_by_variant,
        },
        ensure_ascii=False,
        indent=2,
    )

    path_md = report_root / f"{report_type}-{report_date}.md"
    path_html = report_root / f"{report_type}-{report_date}.html"
    path_json = settings.export_dir / f"compare-{report_type}-{report_date}.json"
    path_md.write_text(markdown_text, encoding="utf-8")
    path_html.write_text(html_text, encoding="utf-8")
    path_json.write_text(json_text + "\n", encoding="utf-8")
    progress_bar.close(f"{_report_label(report_type)}对比已生成 {report_date}")
    return {
        "markdown": str(path_md),
        "html": str(path_html),
        "json": str(path_json),
    }
