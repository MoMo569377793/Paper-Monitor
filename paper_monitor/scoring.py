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
    ]
    return normalize_title(" ".join(part for part in parts if part))


def _category_hits(categories: list[str], categories_text: str) -> list[str]:
    hits: list[str] = []
    for category in categories:
        needle = normalize_title(category)
        if needle and needle in categories_text:
            hits.append(category)
    return hits


def _keyword_group_matches(group: list[str], text: str) -> list[str]:
    return [keyword for keyword in group if keyword_in_text(keyword, text)]


def _summary_text_for_scoring(summary_text: str, llm_summary: dict[str, Any] | None) -> str:
    parts: list[str] = [summary_text or ""]
    if isinstance(llm_summary, dict):
        for key in ("overview", "problem", "method", "results", "application"):
            value = llm_summary.get(key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                parts.extend(str(item) for item in value[:8] if isinstance(item, str))
    return normalize_title(" ".join(part for part in parts if part))


def _summary_keyword_hits(keywords: list[str], text: str) -> list[str]:
    return [keyword for keyword in keywords if keyword_in_text(keyword, text)]


def _summary_rerank_signal(
    topic: TopicConfig,
    *,
    summary_text: str,
    summary_basis: str,
    llm_summary: dict[str, Any] | None,
) -> tuple[float, list[str], bool]:
    text = _summary_text_for_scoring(summary_text, llm_summary)
    if not text:
        return 0.0, [], False

    reasons: list[str] = []
    score = 0.0
    force_irrelevant = False
    summary_basis_value = str(summary_basis or "").strip().lower()

    if topic.id == "matrix_free_fem":
        core_hits = _summary_keyword_hits(
            [
                "matrix-free",
                "matrix free",
                "partial assembly",
                "operator evaluation",
                "operator application",
                "sum factorization",
                "sum-factorization",
                "spectral element",
                "discontinuous galerkin",
            ],
            text,
        )
        impl_hits = _summary_keyword_hits(
            [
                "gpu",
                "cuda",
                "hip",
                "simd",
                "vectorization",
                "avx",
                "roofline",
                "throughput",
                "benchmark",
                "performance",
                "kokkos",
                "raja",
                "openmp",
                "multigrid",
                "precondition",
                "cache blocking",
            ],
            text,
        )
        cross_domain_hits = _summary_keyword_hits(
            [
                "large language model",
                "llm",
                "attention",
                "prompt",
                "recommendation",
                "dehazing",
                "photovoltaic",
                "vision token",
            ],
            text,
        )
        if core_hits:
            score += 6.0 + min(len(core_hits), 3)
            reasons.append(f"摘要确认 matrix-free 主题: {', '.join(core_hits[:4])}")
        if impl_hits:
            score += 5.0 + min(len(impl_hits), 3)
            reasons.append(f"摘要体现实现/性能讨论: {', '.join(impl_hits[:4])}")
        if core_hits and impl_hits:
            score += 3.0
        if summary_basis_value in {"llm+pdf+metadata", "llm+fulltext+metadata"} and (core_hits or impl_hits):
            score += 2.0
            reasons.append(f"摘要来源加权: {summary_basis_value}")
        if core_hits and not impl_hits and summary_basis_value in {"llm+pdf+metadata", "llm+fulltext+metadata"}:
            force_irrelevant = True
            reasons.append("摘要未体现实现/性能层面的讨论，判定为不符合 Matrix-Free 主题要求")
        if cross_domain_hits and not core_hits:
            force_irrelevant = True
            reasons.append(f"摘要显示更像跨域误匹配: {', '.join(cross_domain_hits[:4])}")
        return score, reasons, force_irrelevant

    if topic.id == "ai_operator_acceleration":
        impl_hits = _summary_keyword_hits(
            [
                "kernel",
                "compiler",
                "runtime",
                "tensor core",
                "cutlass",
                "triton",
                "tvm",
                "mlir",
                "halide",
                "fusion",
                "roofline",
                "benchmark",
                "throughput",
                "latency",
                "bandwidth",
                "cuda",
                "cublas",
                "cudnn",
                "vllm",
                "serving",
                "autotuning",
                "tiling",
                "scheduling",
                "fp8",
                "wgmma",
                "pim",
                "dpu",
                "accelerator",
                "dma",
                "pcie",
                "hbm",
                "jit",
                "sve",
                "sme",
                "simd",
            ],
            text,
        )
        workload_hits = _summary_keyword_hits(
            [
                "attention",
                "flashattention",
                "gemm",
                "matmul",
                "kv cache",
                "llm inference",
                "llm training",
                "transformer",
                "spmm",
                "tensor core",
            ],
            text,
        )
        system_metric_hits = _summary_keyword_hits(
            [
                "speedup",
                "throughput",
                "latency",
                "roofline",
                "bandwidth",
                "fps",
                "tflops",
                "gops",
                "memory",
                "end-to-end",
            ],
            text,
        )
        off_topic_hits = _summary_keyword_hits(
            [
                "prompt highlighting",
                "biasbios",
                "counterfact",
                "pronoun change",
                "scienceqa",
                "textvqa",
                "vqav2",
                "super-resolution",
                "image dehazing",
                "photovoltaic",
                "recommendation",
                "reservoir computing",
                "reservoir",
                "token pruning",
                "token reduction",
                "multimodal",
                "visual token",
                "distillation",
                "pruning",
            ],
            text,
        )
        if impl_hits:
            score += 5.0 + min(len(impl_hits), 4)
            reasons.append(f"摘要体现内核/编译器/系统实现: {', '.join(impl_hits[:5])}")
        if impl_hits and workload_hits:
            score += 4.0 + min(len(workload_hits), 3)
            reasons.append(f"摘要体现算子/工作负载对象: {', '.join(workload_hits[:4])}")
        if system_metric_hits:
            score += 2.0 + min(len(system_metric_hits), 3)
            reasons.append(f"摘要体现性能指标或系统结果: {', '.join(system_metric_hits[:4])}")
        if summary_basis_value in {"llm+pdf+metadata", "llm+fulltext+metadata"} and (impl_hits or workload_hits):
            score += 2.0
            reasons.append(f"摘要来源加权: {summary_basis_value}")
        if off_topic_hits and not impl_hits:
            force_irrelevant = True
            reasons.append(f"摘要显示更偏模型/任务论文而非实现优化: {', '.join(off_topic_hits[:4])}")
        elif off_topic_hits:
            score -= 3.0 + min(len(off_topic_hits), 3)
            reasons.append(f"摘要包含偏算法/任务线索，适度降权: {', '.join(off_topic_hits[:4])}")
        return score, reasons, force_irrelevant

    return 0.0, [], False


def _evaluate_required_keyword_lanes(
    lanes: list[list[list[str]]],
    text: str,
) -> tuple[bool, float, list[str], list[str]]:
    best_complete: tuple[float, list[str], list[str]] | None = None
    best_partial: tuple[int, int, list[str]] | None = None

    for lane_index, lane in enumerate(lanes, start=1):
        lane_keywords: list[str] = []
        lane_group_summaries: list[str] = []
        lane_score = 0.0
        matched_group_count = 0

        for group in lane:
            matches = _keyword_group_matches(group, text)
            if matches:
                matched_group_count += 1
                lane_score += 10.0 + max(len(matches) - 1, 0)
                lane_keywords.extend(matches)
                lane_group_summaries.append(" / ".join(matches[:3]))

        if matched_group_count == len(lane):
            lane_score += 2.0
            lane_reasons = [f"命中硬性准入路径 {lane_index}: {' ; '.join(lane_group_summaries[:3])}"]
            candidate = (lane_score, unique_strings(lane_keywords), lane_reasons)
            if best_complete is None or candidate[0] > best_complete[0] or (
                candidate[0] == best_complete[0] and len(candidate[1]) > len(best_complete[1])
            ):
                best_complete = candidate
        elif matched_group_count > 0:
            partial = (
                matched_group_count,
                len(lane),
                [f"最接近的硬性准入路径 {lane_index}: 已命中 {matched_group_count}/{len(lane)} 组"],
            )
            if best_partial is None or partial[:2] > best_partial[:2]:
                best_partial = partial

    if best_complete is not None:
        return True, best_complete[0], best_complete[1], best_complete[2]
    if best_partial is not None:
        return False, 0.0, [], best_partial[2]
    return False, 0.0, [], ["未命中任何硬性准入路径"]


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
    summary_text: str = "",
    summary_basis: str = "",
    llm_summary: dict[str, Any] | None = None,
) -> TopicEvaluation:
    text = _combined_text(title, abstract, venue, categories, tags)
    venue_text = normalize_title(venue)
    categories_text = normalize_title(" ".join(categories))

    score = 0.0
    matched_keywords: list[str] = []
    reasons: list[str] = []
    missing_required_groups = 0
    missing_soft_groups = 0

    if topic.required_keyword_lanes:
        lane_satisfied, lane_score, lane_keywords, lane_reasons = _evaluate_required_keyword_lanes(
            topic.required_keyword_lanes,
            text,
        )
        if lane_satisfied:
            score += lane_score
            matched_keywords.extend(lane_keywords)
            reasons.extend(lane_reasons)
        else:
            missing_required_groups = 1
            reasons.extend(lane_reasons)
    else:
        for group in topic.required_keyword_groups:
            matches = _keyword_group_matches(group, text)
            if matches:
                score += 10.0 + max(len(matches) - 1, 0)
                matched_keywords.extend(matches)
                reasons.append(f"命中硬性关键词组: {', '.join(matches[:3])}")
            else:
                missing_required_groups += 1

    for group in topic.must_match_groups:
        matches = _keyword_group_matches(group, text)
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

    summary_score, summary_reasons, summary_force_irrelevant = _summary_rerank_signal(
        topic,
        summary_text=summary_text,
        summary_basis=summary_basis,
        llm_summary=llm_summary,
    )
    if summary_score:
        score += summary_score
    reasons.extend(summary_reasons)

    matched_keywords = unique_strings(matched_keywords)

    if summary_force_irrelevant:
        classification = "irrelevant"
        reasons.append("摘要级二次重评分判定为不相关")
    elif topic.id == "ai_operator_acceleration" and exclude_hits:
        classification = "irrelevant"
        reasons.append("命中 AI 主题的算法/模型排除词，判定为不相关")
    elif missing_required_groups > 0:
        classification = "irrelevant"
        if topic.required_keyword_lanes:
            reasons.append("缺少任一硬性准入路径，判定为不相关")
        else:
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
        summary_text=paper.summary_text,
        summary_basis=paper.summary_basis,
        llm_summary=paper.llm_summary,
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
