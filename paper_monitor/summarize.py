from __future__ import annotations

from paper_monitor.models import PaperRecord, TopicEvaluation
from paper_monitor.utils import shorten, split_sentences, unique_strings


def build_paper_summary(paper: PaperRecord, evaluations: list[TopicEvaluation]) -> tuple[str, str, list[str]]:
    relevant = [item for item in evaluations if item.classification == "relevant"]
    maybe = [item for item in evaluations if item.classification == "maybe"]
    sorted_items = sorted(relevant or maybe, key=lambda item: item.score, reverse=True)
    lead = sorted_items[0] if sorted_items else None

    keywords: list[str] = []
    for item in sorted_items[:2]:
        keywords.extend(item.matched_keywords[:4])
    keywords = unique_strings(keywords)

    sentences = split_sentences(paper.abstract)
    sentence_one = sentences[0] if sentences else "当前记录主要来自标题和元数据，尚未获得可用摘要。"
    sentence_two = sentences[1] if len(sentences) > 1 else ""

    parts: list[str] = []
    if lead:
        parts.append(f"最接近主题「{lead.topic_name}」，评分 {lead.score:.1f}。")
    if keywords:
        parts.append(f"关键信号包括 {', '.join(keywords[:6])}。")
    parts.append(shorten(sentence_one, 220))
    if sentence_two:
        parts.append(shorten(sentence_two, 180))
    parts.append("当前总结基于标题、摘要和元数据，未自动解析全文 PDF。")

    summary = " ".join(parts)
    tags = keywords[:8]
    basis = "abstract+metadata" if paper.abstract else "metadata-only"
    return summary, basis, tags
