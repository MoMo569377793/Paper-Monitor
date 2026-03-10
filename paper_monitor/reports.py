from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path

from paper_monitor.models import ReportEntry, Settings
from paper_monitor.storage import Database
from paper_monitor.utils import ensure_directory, shorten, to_day_bounds


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


def _render_markdown(
    report_type: str,
    report_date: str,
    start_at: str,
    end_at: str,
    grouped_entries: dict[str, list[ReportEntry]],
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
            lines.append(f"- 匹配词：`{', '.join(entry.matched_keywords) or '无'}`")
            lines.append(f"- 链接：{paper.primary_url or (entry.source_urls[0] if entry.source_urls else '无')}")
            lines.append(f"- 机器总结：{paper.summary_text}")
            if paper.abstract:
                lines.append(f"- 摘要片段：{shorten(paper.abstract, 320)}")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def _render_html(
    report_type: str,
    report_date: str,
    start_at: str,
    end_at: str,
    grouped_entries: dict[str, list[ReportEntry]],
    settings: Settings,
) -> str:
    label = _report_label(report_type)
    sections: list[str] = []
    for topic in settings.topics:
        entries = grouped_entries.get(topic.id, [])
        cards: list[str] = []
        for entry in entries[: settings.report.top_n_per_topic]:
            paper = entry.paper
            cards.append(
                f"""
                <article class="paper-card">
                  <h3>{html.escape(paper.title)}</h3>
                  <p class="meta">相关性 {entry.classification} / 分数 {entry.score} / 来源 {html.escape(', '.join(entry.source_names) or paper.source_first)}</p>
                  <p class="meta">发布时间 {html.escape(paper.published_at or '未知')} / Venue {html.escape(paper.venue or '未知')}</p>
                  <p><strong>匹配词：</strong>{html.escape(', '.join(entry.matched_keywords) or '无')}</p>
                  <p><strong>机器总结：</strong>{html.escape(paper.summary_text)}</p>
                  <p><strong>摘要片段：</strong>{html.escape(shorten(paper.abstract or '无摘要', 360))}</p>
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
      max-width: 1100px;
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
    .paper-card {{
      padding: 18px 18px 14px;
      margin-bottom: 16px;
      border: 1px solid var(--border);
      border-radius: 18px;
      background: var(--panel);
      box-shadow: 0 12px 30px rgba(24, 34, 45, 0.06);
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
) -> dict[str, str]:
    actual_lookback = lookback_days or settings.report.lookback_days
    start_at, end_at = to_day_bounds(report_date, settings.timezone, actual_lookback)
    entries = db.fetch_report_entries(start_at, end_at, settings.report.include_maybe)
    grouped_entries: dict[str, list[ReportEntry]] = defaultdict(list)
    for entry in entries:
        grouped_entries[entry.topic_id].append(entry)

    report_root = settings.report_dir / report_type
    ensure_directory(report_root)
    ensure_directory(settings.export_dir)

    markdown_text = _render_markdown(report_type, report_date, start_at, end_at, grouped_entries, settings)
    html_text = _render_html(report_type, report_date, start_at, end_at, grouped_entries, settings)
    json_text = json.dumps(
        {
            "report_type": report_type,
            "report_date": report_date,
            "start_at": start_at,
            "end_at": end_at,
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
                            "venue": entry.paper.venue,
                            "categories": entry.paper.categories,
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
        },
    )

    return {
        "markdown": str(path_md),
        "html": str(path_html),
        "json": str(path_json),
    }
