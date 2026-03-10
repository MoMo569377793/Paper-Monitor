from __future__ import annotations

from paper_monitor.models import PaperRecord, TopicConfig, TopicEvaluation
from paper_monitor.utils import keyword_in_text, normalize_whitespace, unique_strings


def _combined_text(paper: PaperRecord) -> str:
    parts = [
        paper.title,
        paper.abstract,
        paper.venue,
        " ".join(paper.categories),
        " ".join(paper.tags),
    ]
    return normalize_whitespace(" ".join(part for part in parts if part)).lower()


def evaluate_paper_against_topic(paper: PaperRecord, topic: TopicConfig) -> TopicEvaluation:
    text = _combined_text(paper)
    venue_text = normalize_whitespace(paper.venue).lower()
    categories_text = " ".join(category.lower() for category in paper.categories)

    score = 0.0
    matched_keywords: list[str] = []
    reasons: list[str] = []
    missing_required_groups = 0

    for group in topic.must_match_groups:
        matches = [keyword for keyword in group if keyword_in_text(keyword, text)]
        if matches:
            score += 8.0 + max(len(matches) - 1, 0)
            matched_keywords.extend(matches)
            reasons.append(f"命中必要关键词组: {', '.join(matches[:3])}")
        else:
            missing_required_groups += 1

    if topic.must_match_groups and missing_required_groups == 0:
        score += 6.0
    elif topic.must_match_groups:
        score -= 6.0 * missing_required_groups

    positive_hits = [keyword for keyword in topic.positive_keywords if keyword_in_text(keyword, text)]
    if positive_hits:
        score += 3.0 * len(positive_hits)
        matched_keywords.extend(positive_hits)
        reasons.append(f"命中增强关键词: {', '.join(positive_hits[:5])}")

    category_hits = [category for category in topic.arxiv_categories if category.lower() in categories_text]
    if category_hits:
        score += 2.5 * len(category_hits)
        matched_keywords.extend(category_hits)
        reasons.append(f"命中 arXiv 分类: {', '.join(category_hits[:4])}")

    venue_hits = [keyword for keyword in topic.dblp_venue_keywords if keyword_in_text(keyword, venue_text)]
    if venue_hits:
        score += 2.0 * len(venue_hits)
        matched_keywords.extend(venue_hits)
        reasons.append(f"命中 venue 线索: {', '.join(venue_hits[:4])}")

    exclude_hits = [keyword for keyword in topic.exclude_keywords if keyword_in_text(keyword, text)]
    if exclude_hits:
        score -= 5.0 * len(exclude_hits)
        reasons.append(f"命中排除词: {', '.join(exclude_hits[:4])}")

    matched_keywords = unique_strings(matched_keywords)

    if exclude_hits and score < topic.threshold * 0.6:
        classification = "irrelevant"
    elif score >= topic.threshold:
        classification = "relevant"
    elif score >= max(topic.threshold * 0.6, 8.0):
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
