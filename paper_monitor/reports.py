from __future__ import annotations

import html
import json
import logging
from collections import defaultdict
from pathlib import Path

from paper_monitor.llm import LLMClient
from paper_monitor.llm_registry import LLMRuntimeVariant
from paper_monitor.models import PaperLLMSummary, ReportEntry, Settings
from paper_monitor.progress import ProgressBar
from paper_monitor.storage import Database
from paper_monitor.utils import ensure_directory, shorten, to_day_bounds


LOGGER = logging.getLogger(__name__)


def _variant_attr(variant: LLMRuntimeVariant | dict, name: str):
    if isinstance(variant, dict):
        if name == "variant_id":
            return variant.get("slug")
        if name == "client":
            return variant.get("llm_client")
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
            topic_entries = grouped_entries.get(topic.id, [])
            if progress_callback:
                progress_callback(f"{_variant_attr(variant, 'label')} -> {topic.display_name}", True)
            try:
                digest = client.generate_topic_digest(topic.display_name, topic.description, topic_entries)
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
            }
            if progress_callback:
                progress_callback(f"{_variant_attr(variant, 'label')} -> {topic.display_name} 完成", False)
        digests_by_variant[variant_id] = topic_digests
    return digests_by_variant


def _render_summary_lines(variants: list[LLMRuntimeVariant], summaries: list[PaperLLMSummary]) -> list[str]:
    summary_map = {summary.variant_id: summary for summary in summaries}
    lines: list[str] = []
    for variant in variants:
        summary = summary_map.get(variant.variant_id)
        if not summary:
            lines.append(f"- {variant.label}：未生成")
            continue
        usage = summary.usage if isinstance(summary.usage, dict) else {}
        token_text = ""
        if usage:
            token_text = (
                f" (Token in {usage.get('input_tokens', '未知')} / "
                f"out {usage.get('output_tokens', '未知')})"
            )
        lines.append(f"- {variant.label}：{summary.summary_text}{token_text}")
    return lines


def _render_summary_html(variants: list[LLMRuntimeVariant], summaries: list[PaperLLMSummary]) -> str:
    summary_map = {summary.variant_id: summary for summary in summaries}
    items: list[str] = []
    for variant in variants:
        summary = summary_map.get(variant.variant_id)
        if summary is None:
            items.append(f"<li><strong>{html.escape(variant.label)}：</strong>未生成</li>")
            continue
        usage = summary.usage if isinstance(summary.usage, dict) else {}
        token_text = ""
        if usage:
            token_text = (
                f"Token in {usage.get('input_tokens', '未知')} / "
                f"out {usage.get('output_tokens', '未知')}"
            )
        items.append(
            f"<li><strong>{html.escape(variant.label)}：</strong>{html.escape(summary.summary_text)}"
            f"{f' <span class=\"meta\">{html.escape(token_text)}</span>' if token_text else ''}</li>"
        )
    return '<ul class="llm-summary-list">' + "".join(items) + "</ul>"


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
    paper_summaries_by_paper: dict[int, list[PaperLLMSummary]],
    topic_digests_by_variant: dict[str, dict[str, dict]],
    variants: list[LLMRuntimeVariant],
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
        entries = grouped_entries.get(topic.id, [])
        lines.append(f"## {topic.display_name}")
        lines.append("")
        lines.append(topic.description)
        lines.append("")
        lines.append(f"- 命中数量：`{len(entries)}`")
        lines.append(f"- 趋势摘要：{_topic_trend_sentence(entries)}")
        for variant in variants:
            digest = topic_digests_by_variant.get(variant.variant_id, {}).get(topic.id)
            if not digest:
                lines.append(f"- {variant.label} 主题概览：未生成")
                continue
            lines.append(f"- {variant.label} 主题概览：{digest.get('overview', '')}")
            highlights = digest.get("highlights", [])
            if highlights:
                lines.append(f"- {variant.label} 关键观察：`{' | '.join(highlights[:3])}`")
            watchlist = digest.get("watchlist", [])
            if watchlist:
                lines.append(f"- {variant.label} 建议关注：`{' | '.join(watchlist[:3])}`")
            usage = digest.get("usage", {})
            if usage:
                lines.append(
                    f"- {variant.label} Token：in `{usage.get('input_tokens', '未知')}` / "
                    f"out `{usage.get('output_tokens', '未知')}` / total `{usage.get('total_tokens', '未知')}`"
                )
        lines.append("")

        if not entries:
            lines.append("本窗口内没有新记录。")
            lines.append("")
            continue

        for index, entry in enumerate(entries[: settings.report.top_n_per_topic], start=1):
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
            if paper.pdf_local_path:
                lines.append(f"- 本地 PDF：`{paper.pdf_local_path}`")
            if paper.fulltext_txt_path:
                lines.append(f"- 全文文本：`{paper.fulltext_txt_path}`")
            lines.append(f"- 默认总结：{paper.summary_text}")
            lines.append("#### 多模型摘要")
            lines.extend(_render_summary_lines(variants, paper_summaries_by_paper.get(paper.id, [])))
            if paper.fulltext_excerpt:
                lines.append(f"- 全文节选：{shorten(paper.fulltext_excerpt, 320)}")
            elif paper.abstract:
                lines.append(f"- 摘要片段：{shorten(paper.abstract, 320)}")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def _render_html(
    report_type: str,
    report_date: str,
    start_at: str,
    end_at: str,
    grouped_entries: dict[str, list[ReportEntry]],
    paper_summaries_by_paper: dict[int, list[PaperLLMSummary]],
    topic_digests_by_variant: dict[str, dict[str, dict]],
    variants: list[LLMRuntimeVariant],
    settings: Settings,
) -> str:
    label = _report_label(report_type)
    sections: list[str] = []
    for topic in settings.topics:
        entries = grouped_entries.get(topic.id, [])
        digest_cards: list[str] = []
        for variant in variants:
            digest = topic_digests_by_variant.get(variant.variant_id, {}).get(topic.id, {})
            digest_cards.append(
                f"""
                <article class="topic-digest-card">
                  <p class="meta">模型 {html.escape(variant.label)}</p>
                  <p><strong>主题概览：</strong>{html.escape(digest.get('overview', '未生成'))}</p>
                  {f"<p><strong>关键观察：</strong>{html.escape(' | '.join(digest.get('highlights', [])[:3]))}</p>" if digest.get('highlights') else ''}
                  {f"<p><strong>建议关注：</strong>{html.escape(' | '.join(digest.get('watchlist', [])[:3]))}</p>" if digest.get('watchlist') else ''}
                </article>
                """
            )
        cards: list[str] = []
        for entry in entries[: settings.report.top_n_per_topic]:
            paper = entry.paper
            cards.append(
                f"""
                <article class="paper-card">
                  <h3>{html.escape(paper.title)}</h3>
                  <p class="meta">相关性 {entry.classification} / 分数 {entry.score} / 来源 {html.escape(', '.join(entry.source_names) or paper.source_first)}</p>
                  <p class="meta">发布时间 {html.escape(paper.published_at or '未知')} / Venue {html.escape(paper.venue or '未知')}</p>
                  <p class="meta">全文状态 {html.escape(paper.fulltext_status)} / PDF {html.escape(paper.pdf_status)} / 页数 {html.escape(str(paper.page_count or '未知'))} / 总结来源 {html.escape(paper.summary_basis or '未知')}</p>
                  <p><strong>匹配词：</strong>{html.escape(', '.join(entry.matched_keywords) or '无')}</p>
                  <p><strong>默认总结：</strong>{html.escape(paper.summary_text)}</p>
                  <div><strong>多模型摘要：</strong>{_render_summary_html(variants, paper_summaries_by_paper.get(paper.id, []))}</div>
                  <p><strong>{'全文节选' if paper.fulltext_excerpt else '摘要片段'}：</strong>{html.escape(shorten(paper.fulltext_excerpt or paper.abstract or '无摘要', 360))}</p>
                  <p><a href="{html.escape(paper.primary_url or (entry.source_urls[0] if entry.source_urls else '#'))}">打开原始链接</a></p>
                </article>
                """
            )
        if not cards:
            cards.append('<article class="paper-card"><p>本窗口内没有新记录。</p></article>')
        sections.append(
            f"""
            <section class="topic-section">
              <div class="topic-header">
                <h2>{html.escape(topic.display_name)}</h2>
                <p>{html.escape(topic.description)}</p>
                <p class="trend">{html.escape(_topic_trend_sentence(entries))}</p>
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
    .llm-summary-list {{
      margin: 8px 0 0;
      padding-left: 18px;
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
        entries = grouped_entries.get(topic.id, [])
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
        entries = grouped_entries.get(topic.id, [])
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
    use_llm_topic_digest: bool = False,
) -> dict[str, str]:
    actual_lookback = lookback_days or settings.report.lookback_days
    start_at, end_at = to_day_bounds(report_date, settings.timezone, actual_lookback)
    report_variants = list(llm_variants or [])
    if llm_client is not None and not report_variants:
        report_variants = [_single_runtime_variant(settings, llm_client)]
    digest_steps = _active_variant_count(report_variants) * len(settings.topics) if use_llm_topic_digest else 0
    progress_bar = ProgressBar(_report_label(report_type), 5 + digest_steps)
    progress_bar.advance(detail="加载命中论文")
    entries = db.fetch_report_entries(start_at, end_at, settings.report.include_maybe)
    grouped_entries: dict[str, list[ReportEntry]] = defaultdict(list)
    for entry in entries:
        grouped_entries[entry.topic_id].append(entry)

    progress_bar.advance(detail="读取逐篇 LLM 摘要")
    paper_summaries_by_paper = db.fetch_paper_llm_summaries([entry.paper.id for entry in entries])
    topic_digests_by_variant = _collect_topic_digests_by_variant(
        settings,
        grouped_entries,
        report_variants,
        use_llm_topic_digest,
        progress_callback=lambda detail, advance: (
            progress_bar.advance(detail=detail) if advance else progress_bar.set_detail(detail)
        ),
    )

    report_root = settings.report_dir / report_type
    ensure_directory(report_root)
    ensure_directory(settings.export_dir)

    progress_bar.advance(detail="渲染 Markdown")
    markdown_text = _render_markdown(
        report_type,
        report_date,
        start_at,
        end_at,
        grouped_entries,
        paper_summaries_by_paper,
        topic_digests_by_variant,
        report_variants,
        settings,
    )
    progress_bar.advance(detail="渲染 HTML")
    html_text = _render_html(
        report_type,
        report_date,
        start_at,
        end_at,
        grouped_entries,
        paper_summaries_by_paper,
        topic_digests_by_variant,
        report_variants,
        settings,
    )
    progress_bar.advance(detail="导出 JSON 与写入文件")
    json_text = json.dumps(
        {
            "report_type": report_type,
            "report_date": report_date,
            "start_at": start_at,
            "end_at": end_at,
            "variants": _serialize_variants(report_variants),
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
                                    "usage": summary.usage,
                                }
                                for summary in paper_summaries_by_paper.get(entry.paper.id, [])
                            ],
                        },
                    }
                    for entry in topic_entries
                ]
                for topic_id, topic_entries in grouped_entries.items()
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
            "variants": _serialize_variants(report_variants),
        },
    )
    progress_bar.close(f"{_report_label(report_type)}已生成 {report_date}")

    return {
        "markdown": str(path_md),
        "html": str(path_html),
        "json": str(path_json),
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

    digests_by_variant = _collect_topic_digests_by_variant(
        settings,
        grouped_entries,
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
