from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from paper_monitor.models import PaperCandidate, PaperRecord, TopicConfig, TopicEvaluation
from paper_monitor.utils import keyword_in_text, normalize_title, unique_strings


def _combined_text(title: str, abstract: str, venue: str, categories: list[str], tags: list[str]) -> str:
    parts = [
        title,
        abstract,
        venue,
        " ".join(categories),
        " ".join(tags),
    ]
    return normalize_title(" ".join(part for part in parts if part))


def _category_hits(categories: list[str], categories_text: str) -> list[str]:
    hits: list[str] = []
    for category in categories:
        needle = normalize_title(category)
        if needle and needle in categories_text:
            hits.append(category)
    return hits


def _coerce_metric(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str) and value.strip().isdigit():
        return max(int(value.strip()), 0)
    return 0


def _scan_metadata_metric(payload: Any, keys: set[str], *, depth: int = 0) -> int:
    if depth > 4:
        return 0
    best = 0
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in keys:
                best = max(best, _coerce_metric(value))
                continue
            best = max(best, _scan_metadata_metric(value, keys, depth=depth + 1))
    elif isinstance(payload, list):
        for value in payload[:20]:
            best = max(best, _scan_metadata_metric(value, keys, depth=depth + 1))
    return best


def citation_metrics_from_metadata(metadata: dict[str, Any]) -> tuple[int, int]:
    ranking = metadata.get("ranking", {}) if isinstance(metadata, dict) else {}
    citation_count = _coerce_metric(ranking.get("citation_count")) if isinstance(ranking, dict) else 0
    influential_count = (
        _coerce_metric(ranking.get("influential_citation_count")) if isinstance(ranking, dict) else 0
    )
    if citation_count <= 0:
        citation_count = _scan_metadata_metric(
            metadata,
            {"citation_count", "citationCount", "cited_by_count", "citedByCount"},
        )
    if influential_count <= 0:
        influential_count = _scan_metadata_metric(
            metadata,
            {"influential_citation_count", "influentialCitationCount"},
        )
    return citation_count, influential_count


def _recency_boost(year: int | None, published_at: str | None, updated_at: str | None) -> float:
    year_value = year
    for value in (updated_at, published_at):
        if value and len(value) >= 4 and value[:4].isdigit():
            year_value = int(value[:4])
            break
    if year_value is None:
        return 0.0
    current_year = datetime.now().year
    age = max(current_year - year_value, 0)
    if age <= 1:
        return 4.0
    if age <= 3:
        return 3.0
    if age <= 5:
        return 2.0
    if age <= 8:
        return 1.0
    return 0.0


def _citation_boost(citation_count: int, influential_count: int) -> float:
    if citation_count <= 0 and influential_count <= 0:
        return 0.0
    score = 0.0
    if citation_count > 0:
        score += min(16.0, math.log10(citation_count + 1) * 7.0)
        if citation_count >= 50:
            score += 1.5
        if citation_count >= 200:
            score += 2.5
        if citation_count >= 1000:
            score += 2.0
    if influential_count > 0:
        score += min(6.0, math.log10(influential_count + 1) * 3.5)
    return min(score, 24.0)


def _evaluate_topic(
    *,
    title: str,
    abstract: str,
    venue: str,
    categories: list[str],
    tags: list[str],
    metadata: dict[str, Any],
    published_at: str | None,
    updated_at: str | None,
    year: int | None,
    topic: TopicConfig,
) -> TopicEvaluation:
    text = _combined_text(title, abstract, venue, categories, tags)
    venue_text = normalize_title(venue)
    categories_text = normalize_title(" ".join(categories))

    score = 0.0
    matched_keywords: list[str] = []
    reasons: list[str] = []
    missing_required_groups = 0
    missing_soft_groups = 0

    for group in topic.required_keyword_groups:
        matches = [keyword for keyword in group if keyword_in_text(keyword, text)]
        if matches:
            score += 10.0 + max(len(matches) - 1, 0)
            matched_keywords.extend(matches)
            reasons.append(f"命中硬性关键词组: {', '.join(matches[:3])}")
        else:
            missing_required_groups += 1

    for group in topic.must_match_groups:
        matches = [keyword for keyword in group if keyword_in_text(keyword, text)]
        if matches:
            score += 6.0 + max(len(matches) - 1, 0)
            matched_keywords.extend(matches)
            reasons.append(f"命中必要关键词组: {', '.join(matches[:3])}")
        else:
            missing_soft_groups += 1

    if topic.must_match_groups and missing_soft_groups == 0:
        score += 4.0
    elif topic.must_match_groups:
        score -= 3.0 * missing_soft_groups

    positive_hits = [keyword for keyword in topic.positive_keywords if keyword_in_text(keyword, text)]
    if positive_hits:
        score += 2.5 * len(positive_hits)
        matched_keywords.extend(positive_hits)
        reasons.append(f"命中增强关键词: {', '.join(positive_hits[:6])}")

    priority_category_hits = _category_hits(topic.priority_arxiv_categories, categories_text)
    if priority_category_hits:
        score += 4.0 * len(priority_category_hits)
        matched_keywords.extend(priority_category_hits)
        reasons.append(f"命中优先 arXiv 分类: {', '.join(priority_category_hits[:4])}")

    category_hits = [
        category
        for category in _category_hits(topic.arxiv_categories, categories_text)
        if category not in priority_category_hits
    ]
    if category_hits:
        score += 2.0 * len(category_hits)
        matched_keywords.extend(category_hits)
        reasons.append(f"命中 arXiv 分类: {', '.join(category_hits[:4])}")

    priority_venue_hits = [keyword for keyword in topic.priority_venue_keywords if keyword_in_text(keyword, venue_text)]
    if priority_venue_hits:
        score += 3.5 * len(priority_venue_hits)
        matched_keywords.extend(priority_venue_hits)
        reasons.append(f"命中优先 venue 线索: {', '.join(priority_venue_hits[:4])}")

    venue_hits = [
        keyword
        for keyword in topic.dblp_venue_keywords
        if keyword not in priority_venue_hits and keyword_in_text(keyword, venue_text)
    ]
    if venue_hits:
        score += 2.0 * len(venue_hits)
        matched_keywords.extend(venue_hits)
        reasons.append(f"命中 venue 线索: {', '.join(venue_hits[:4])}")

    citation_count, influential_count = citation_metrics_from_metadata(metadata)
    citation_bonus = _citation_boost(citation_count, influential_count)
    if citation_bonus > 0:
        score += citation_bonus
        reasons.append(f"引用加分: citations={citation_count}, influential={influential_count}")

    recency_bonus = _recency_boost(year, published_at, updated_at)
    if recency_bonus > 0:
        score += recency_bonus
        reasons.append(f"时间加分: {year or (published_at or updated_at or '未知')}")

    exclude_hits = [keyword for keyword in topic.exclude_keywords if keyword_in_text(keyword, text)]
    if exclude_hits:
        score -= 6.0 * len(exclude_hits)
        reasons.append(f"命中排除词: {', '.join(exclude_hits[:4])}")

    matched_keywords = unique_strings(matched_keywords)

    if missing_required_groups > 0:
        classification = "irrelevant"
        reasons.append("缺少硬性关键词组，判定为不相关")
    elif exclude_hits and score < topic.threshold * 0.85:
        classification = "irrelevant"
    elif score >= topic.threshold:
        classification = "relevant"
    elif score >= max(topic.threshold * 0.65, 10.0):
        classification = "maybe"
    else:
        classification = "irrelevant"

    return TopicEvaluation(
        topic_id=topic.id,
        topic_name=topic.display_name,
        score=round(score, 2),
        classification=classification,
        matched_keywords=matched_keywords,
        reasons=reasons or ["仅命中弱相关线索"],
    )


def evaluate_paper_against_topic(paper: PaperRecord, topic: TopicConfig) -> TopicEvaluation:
    return _evaluate_topic(
        title=paper.title,
        abstract=paper.abstract,
        venue=paper.venue,
        categories=paper.categories,
        tags=paper.tags,
        metadata=paper.metadata,
        published_at=paper.published_at,
        updated_at=paper.updated_at,
        year=paper.year,
        topic=topic,
    )


def evaluate_candidate_against_topic(candidate: PaperCandidate, topic: TopicConfig) -> TopicEvaluation:
    return _evaluate_topic(
        title=candidate.title,
        abstract=candidate.abstract,
        venue=candidate.venue,
        categories=candidate.categories,
        tags=[],
        metadata=candidate.raw,
        published_at=candidate.published_at,
        updated_at=candidate.updated_at,
        year=candidate.year,
        topic=topic,
    )


def evaluate_seed_paper_for_topic(paper: PaperRecord, topic: TopicConfig) -> TopicEvaluation:
    return TopicEvaluation(
        topic_id=topic.id,
        topic_name=topic.display_name,
        score=1000.0,
        classification="relevant",
        matched_keywords=["curated-seed"],
        reasons=["预置种子论文，直接纳入该领域"],
    )
