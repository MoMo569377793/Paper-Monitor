from __future__ import annotations

import logging

from paper_monitor.fetchers.arxiv import ArxivFetcher
from paper_monitor.fetchers.dblp import DBLPFetcher
from paper_monitor.fetchers.scholar_alerts import ScholarAlertsFetcher
from paper_monitor.models import FetchPlan, PaperCandidate, RunStats, Settings
from paper_monitor.progress import ProgressBar
from paper_monitor.scoring import evaluate_paper_against_topic
from paper_monitor.storage import Database
from paper_monitor.summarize import build_paper_summary
from paper_monitor.utils import normalize_title


LOGGER = logging.getLogger(__name__)


class MonitorPipeline:
    def __init__(self, settings: Settings, db: Database) -> None:
        self.settings = settings
        self.db = db
        self.fetchers = {
            "arxiv": ArxivFetcher(settings.arxiv),
            "dblp": DBLPFetcher(settings.dblp),
            "google_scholar_alerts": ScholarAlertsFetcher(settings.scholar_alerts, settings.timezone),
        }

    def run_fetch(
        self,
        selected_sources: set[str] | None = None,
        *,
        start_year: int | None = None,
        recent_limit: int | None = None,
        page_size: int | None = None,
    ) -> RunStats:
        stats = RunStats()
        seen_source_items: set[tuple[str, str]] = set()
        fetch_plan = FetchPlan(
            start_year=start_year if start_year is not None else self.settings.bootstrap.start_year,
            recent_limit=recent_limit if recent_limit is not None else self.settings.bootstrap.recent_limit,
            page_size=page_size or self.settings.bootstrap.page_size,
        )
        fetch_bar = ProgressBar("抓取", self._count_fetch_steps(selected_sources))

        for topic in self.settings.topics:
            topic_candidates: list[PaperCandidate] = []
            for source_name, queries in topic.source_queries.items():
                if selected_sources and source_name not in selected_sources:
                    continue
                fetcher = self.fetchers.get(source_name)
                if fetcher is None or not fetcher.enabled:
                    continue

                try:
                    candidates = fetcher.fetch(topic, queries, fetch_plan, self._make_fetch_progress(fetch_bar))
                except Exception as exc:  # pragma: no cover - defensive logging
                    LOGGER.exception("fetch failed for %s / %s", topic.id, source_name)
                    fetch_bar.set_detail(f"{source_name} {topic.id} 失败: {exc}")
                    stats.errors.append(f"{source_name}:{topic.id}:{exc}")
                    continue

                stats.by_source[source_name] = stats.by_source.get(source_name, 0) + len(candidates)
                stats.fetched += len(candidates)
                topic_candidates.extend(candidates)

            for candidate in self._select_topic_candidates(topic_candidates, fetch_plan):
                key = (candidate.source_name, candidate.source_paper_id)
                if key in seen_source_items:
                    continue
                seen_source_items.add(key)
                self._process_candidate(candidate, stats)

        fetch_bar.close(f"抓取完成 {stats.fetched} 条候选")
        return stats

    def _select_topic_candidates(self, candidates: list[PaperCandidate], fetch_plan: FetchPlan) -> list[PaperCandidate]:
        if not candidates:
            return []
        if not fetch_plan.recent_limit:
            return candidates

        grouped: dict[str, list[PaperCandidate]] = {}
        for candidate in candidates:
            grouped.setdefault(self._logical_paper_key(candidate), []).append(candidate)

        ordered_groups = sorted(
            grouped.values(),
            key=lambda items: max(self._candidate_sort_key(item) for item in items),
            reverse=True,
        )[: fetch_plan.recent_limit]

        selected: list[PaperCandidate] = []
        for items in ordered_groups:
            selected.extend(
                sorted(
                    items,
                    key=lambda item: (
                        self._candidate_sort_key(item),
                        item.source_name,
                        item.source_paper_id,
                    ),
                    reverse=True,
                )
            )
        return selected

    def _logical_paper_key(self, candidate: PaperCandidate) -> str:
        if candidate.doi:
            return f"doi:{candidate.doi.lower()}"
        if candidate.arxiv_id:
            return f"arxiv:{candidate.arxiv_id.lower()}"
        return f"title:{normalize_title(candidate.title)}"

    def _candidate_sort_key(self, candidate: PaperCandidate) -> str:
        return candidate.updated_at or candidate.published_at or str(candidate.year or 0)

    def _count_fetch_steps(self, selected_sources: set[str] | None) -> int:
        total = 0
        for topic in self.settings.topics:
            for source_name, queries in topic.source_queries.items():
                if selected_sources and source_name not in selected_sources:
                    continue
                fetcher = self.fetchers.get(source_name)
                if fetcher is None or not fetcher.enabled:
                    continue
                total += max(len(queries), 1)
        return total or 1

    def _make_fetch_progress(self, progress_bar: ProgressBar):
        def callback(detail: str, advance: bool) -> None:
            if advance:
                progress_bar.advance(detail=detail)
                return
            progress_bar.set_detail(detail)

        return callback

    def _process_candidate(self, candidate: PaperCandidate, stats: RunStats) -> None:
        paper_id, created = self.db.upsert_paper(candidate)
        paper = self.db.get_paper(paper_id)

        evaluations = [evaluate_paper_against_topic(paper, topic) for topic in self.settings.topics]
        saved_matches = 0
        interesting_tags: list[str] = []
        for evaluation in evaluations:
            if evaluation.classification == "irrelevant":
                continue
            inserted = self.db.upsert_match(paper_id, evaluation)
            if inserted:
                saved_matches += 1
            interesting_tags.extend(evaluation.matched_keywords[:4])

        if paper.summary_basis.startswith("llm+") and paper.summary_text:
            self.db.update_paper_analysis(
                paper_id,
                paper.summary_text,
                paper.summary_basis,
                interesting_tags + paper.tags,
                llm_summary=paper.llm_summary,
            )
        else:
            summary_text, basis, tags = build_paper_summary(paper, evaluations)
            self.db.update_paper_analysis(paper_id, summary_text, basis, interesting_tags + tags)

        stats.processed += 1
        if created:
            stats.stored += 1
        stats.matched += saved_matches
