from __future__ import annotations

import logging

from paper_monitor.fetchers.arxiv import ArxivFetcher
from paper_monitor.fetchers.dblp import DBLPFetcher
from paper_monitor.fetchers.scholar_alerts import ScholarAlertsFetcher
from paper_monitor.models import PaperCandidate, RunStats, Settings
from paper_monitor.scoring import evaluate_paper_against_topic
from paper_monitor.storage import Database
from paper_monitor.summarize import build_paper_summary


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

    def run_fetch(self, selected_sources: set[str] | None = None) -> RunStats:
        stats = RunStats()
        seen_source_items: set[tuple[str, str]] = set()

        for topic in self.settings.topics:
            for source_name, queries in topic.source_queries.items():
                if selected_sources and source_name not in selected_sources:
                    continue
                fetcher = self.fetchers.get(source_name)
                if fetcher is None or not fetcher.enabled:
                    continue

                try:
                    candidates = fetcher.fetch(topic, queries)
                except Exception as exc:  # pragma: no cover - defensive logging
                    LOGGER.exception("fetch failed for %s / %s", topic.id, source_name)
                    stats.errors.append(f"{source_name}:{topic.id}:{exc}")
                    continue

                stats.by_source[source_name] = stats.by_source.get(source_name, 0) + len(candidates)
                stats.fetched += len(candidates)
                for candidate in candidates:
                    key = (candidate.source_name, candidate.source_paper_id)
                    if key in seen_source_items:
                        continue
                    seen_source_items.add(key)
                    self._process_candidate(candidate, stats)

        return stats

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

        summary_text, basis, tags = build_paper_summary(paper, evaluations)
        self.db.update_paper_analysis(paper_id, summary_text, basis, interesting_tags + tags)

        stats.processed += 1
        if created:
            stats.stored += 1
        stats.matched += saved_matches
