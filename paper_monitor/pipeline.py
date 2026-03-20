from __future__ import annotations

from dataclasses import replace
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from paper_monitor.fetchers.arxiv import ArxivFetcher
from paper_monitor.fetchers.dblp import DBLPFetcher
from paper_monitor.fetchers.scholar_alerts import ScholarAlertsFetcher
from paper_monitor.models import FetchPlan, PaperCandidate, RunStats, Settings
from paper_monitor.progress import ProgressBar
from paper_monitor.scoring import (
    citation_metrics_from_metadata,
    evaluate_candidate_against_topic,
    evaluate_paper_against_topic,
    evaluate_seed_paper_for_topic,
)
from paper_monitor.storage import Database
from paper_monitor.summarize import build_paper_summary
from paper_monitor.utils import normalize_title, now_iso, stable_hash


LOGGER = logging.getLogger(__name__)


class MonitorPipeline:
    def __init__(self, settings: Settings, db: Database) -> None:
        self.settings = settings
        self.db = db
        self.semantic_scholar_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        self._ranking_metadata_cache: dict[str, dict[str, int | str]] = {}
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
        since_last_run: bool = False,
    ) -> RunStats:
        stats = RunStats()
        seen_source_items: set[tuple[str, str]] = set()
        fetch_plan = FetchPlan(
            start_year=start_year if start_year is not None else self.settings.bootstrap.start_year,
            recent_limit=recent_limit if recent_limit is not None else self.settings.bootstrap.recent_limit,
            page_size=page_size or self.settings.bootstrap.page_size,
        )
        run_started_at = now_iso(self.settings.timezone)
        fetch_bar = ProgressBar("抓取", self._count_fetch_steps(selected_sources))

        for topic in self.settings.topics:
            topic_candidates: list[PaperCandidate] = []
            for source_name, queries in topic.source_queries.items():
                if selected_sources and source_name not in selected_sources:
                    continue
                fetcher = self.fetchers.get(source_name)
                if fetcher is None or not fetcher.enabled:
                    continue
                checkpoint_key = self._checkpoint_key(source_name, topic.id)
                source_plan = fetch_plan
                if since_last_run:
                    source_plan = replace(fetch_plan, since_at=self.db.get_checkpoint(checkpoint_key))

                try:
                    candidates = fetcher.fetch(topic, queries, source_plan, self._make_fetch_progress(fetch_bar))
                except Exception as exc:  # pragma: no cover - defensive logging
                    LOGGER.exception("fetch failed for %s / %s", topic.id, source_name)
                    fetch_bar.set_detail(f"{source_name} {topic.id} 失败: {exc}")
                    stats.errors.append(f"{source_name}:{topic.id}:{exc}")
                    continue
                if since_last_run:
                    self.db.set_checkpoint(checkpoint_key, run_started_at)

                stats.by_source[source_name] = stats.by_source.get(source_name, 0) + len(candidates)
                stats.fetched += len(candidates)
                topic_candidates.extend(candidates)

            for candidate in self._select_topic_candidates(topic, topic_candidates, fetch_plan):
                key = (candidate.source_name, candidate.source_paper_id)
                if key in seen_source_items:
                    continue
                seen_source_items.add(key)
                self._process_candidate(candidate, stats, only_new=since_last_run)

            for candidate in self._seed_candidates_for_topic(topic):
                key = (candidate.source_name, candidate.source_paper_id)
                if key in seen_source_items:
                    continue
                seen_source_items.add(key)
                self._process_candidate(candidate, stats, only_new=False)

        fetch_bar.close(f"抓取完成 {stats.fetched} 条候选")
        return stats

    def _checkpoint_key(self, source_name: str, topic_id: str) -> str:
        return f"fetch:{source_name}:{topic_id}:since"

    def _select_topic_candidates(
        self,
        topic,
        candidates: list[PaperCandidate],
        fetch_plan: FetchPlan,
    ) -> list[PaperCandidate]:
        if not candidates:
            return []
        self._annotate_candidates_with_ranking_metadata(candidates)

        grouped: dict[str, list[PaperCandidate]] = {}
        for candidate in candidates:
            grouped.setdefault(self._logical_paper_key(candidate), []).append(candidate)

        ranked_groups: list[dict[str, object]] = []
        for logical_key, items in grouped.items():
            best_candidate = max(items, key=lambda item: self._candidate_rank_tuple(item, topic))
            evaluation = evaluate_candidate_against_topic(best_candidate, topic)
            if evaluation.classification == "irrelevant":
                continue
            citation_count, influential_count = citation_metrics_from_metadata(best_candidate.raw)
            ranked_groups.append(
                {
                    "logical_key": logical_key,
                    "items": items,
                    "candidate": best_candidate,
                    "evaluation": evaluation,
                    "citation_count": citation_count,
                    "influential_count": influential_count,
                }
            )

        if not fetch_plan.recent_limit:
            ordered_groups = sorted(
                ranked_groups,
                key=self._fill_group_key,
                reverse=True,
            )
        else:
            ordered_groups = self._select_balanced_group_window(ranked_groups, fetch_plan.recent_limit)

        selected: list[PaperCandidate] = []
        for group_info in ordered_groups:
            items = group_info["items"]
            selected.extend(
                sorted(
                    items,
                    key=lambda item: self._candidate_rank_tuple(item, topic),
                    reverse=True,
                )
            )
        return selected

    def _select_balanced_group_window(self, ranked_groups: list[dict[str, object]], recent_limit: int) -> list[dict[str, object]]:
        if not ranked_groups or recent_limit <= 0:
            return []
        classic_quota = min(recent_limit // 2, 50)
        selected_keys: set[str] = set()
        selected: list[dict[str, object]] = []

        if classic_quota > 0:
            for group_info in sorted(ranked_groups, key=self._classic_group_key, reverse=True):
                if int(group_info["citation_count"]) <= 0:
                    continue
                logical_key = str(group_info["logical_key"])
                if logical_key in selected_keys:
                    continue
                selected.append(group_info)
                selected_keys.add(logical_key)
                if len(selected) >= classic_quota:
                    break

        for group_info in sorted(ranked_groups, key=self._recent_group_key, reverse=True):
            if len(selected) >= recent_limit:
                break
            logical_key = str(group_info["logical_key"])
            if logical_key in selected_keys:
                continue
            selected.append(group_info)
            selected_keys.add(logical_key)

        if len(selected) < recent_limit:
            for group_info in sorted(ranked_groups, key=self._fill_group_key, reverse=True):
                if len(selected) >= recent_limit:
                    break
                logical_key = str(group_info["logical_key"])
                if logical_key in selected_keys:
                    continue
                selected.append(group_info)
                selected_keys.add(logical_key)

        return selected

    def _logical_paper_key(self, candidate: PaperCandidate) -> str:
        if candidate.doi:
            return f"doi:{candidate.doi.lower()}"
        if candidate.arxiv_id:
            return f"arxiv:{candidate.arxiv_id.lower()}"
        return f"title:{normalize_title(candidate.title)}"

    def _candidate_sort_key(self, candidate: PaperCandidate) -> str:
        return candidate.updated_at or candidate.published_at or str(candidate.year or 0)

    def _candidate_rank_tuple(self, candidate: PaperCandidate, topic) -> tuple[float, int, int, str, str, str]:
        evaluation = evaluate_candidate_against_topic(candidate, topic)
        citation_count, influential_count = citation_metrics_from_metadata(candidate.raw)
        return (
            evaluation.score,
            citation_count,
            influential_count,
            self._candidate_sort_key(candidate),
            candidate.source_name,
            candidate.source_paper_id,
        )

    def _classic_group_key(self, group_info: dict[str, object]) -> tuple[int, int, float, str]:
        candidate = group_info["candidate"]
        evaluation = group_info["evaluation"]
        return (
            int(group_info["citation_count"]),
            int(group_info["influential_count"]),
            float(evaluation.score),
            self._candidate_sort_key(candidate),
        )

    def _recent_group_key(self, group_info: dict[str, object]) -> tuple[str, float, int]:
        candidate = group_info["candidate"]
        evaluation = group_info["evaluation"]
        return (
            self._candidate_sort_key(candidate),
            float(evaluation.score),
            int(group_info["citation_count"]),
        )

    def _fill_group_key(self, group_info: dict[str, object]) -> tuple[float, int, int, str]:
        candidate = group_info["candidate"]
        evaluation = group_info["evaluation"]
        return (
            float(evaluation.score),
            int(group_info["citation_count"]),
            int(group_info["influential_count"]),
            self._candidate_sort_key(candidate),
        )

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

    def _process_candidate(self, candidate: PaperCandidate, stats: RunStats, *, only_new: bool = False) -> None:
        paper_id, created = self.db.upsert_paper(candidate)
        if only_new and not created:
            return
        paper = self.db.get_paper(paper_id)
        forced_topic_ids = {
            str(item)
            for item in candidate.raw.get("forced_topic_ids", [])
            if isinstance(item, str)
        }
        evaluations = []
        for topic in self.settings.topics:
            if topic.id in forced_topic_ids:
                evaluations.append(
                    evaluate_seed_paper_for_topic(paper, topic)
                )
                continue
            evaluations.append(evaluate_paper_against_topic(paper, topic))
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
            stats.new_paper_ids.append(paper_id)
        stats.matched += saved_matches

    def _seed_candidates_for_topic(self, topic) -> list[PaperCandidate]:
        candidates: list[PaperCandidate] = []
        for seed in topic.seed_papers:
            source_key = seed.doi or seed.arxiv_id or seed.primary_url or seed.title
            candidate = PaperCandidate(
                source_name="seed",
                source_paper_id=f"seed-{stable_hash(source_key)}",
                query_text=f"seed:{topic.id}",
                title=seed.title,
                abstract=seed.abstract,
                authors=seed.authors,
                published_at=str(seed.year) if seed.year else None,
                updated_at=None,
                primary_url=seed.primary_url,
                pdf_url=seed.pdf_url,
                doi=seed.doi,
                arxiv_id=seed.arxiv_id,
                venue=seed.venue or "curated-seed",
                year=seed.year,
                categories=seed.categories,
                raw={
                    "seed": True,
                    "forced_topic_ids": [topic.id],
                    "seed_tags": seed.tags,
                    "ranking": {
                        "citation_count": 5000,
                        "influential_citation_count": 500,
                        "source": "curated_seed",
                    },
                },
            )
            candidates.append(candidate)
        return candidates

    def _annotate_candidates_with_ranking_metadata(self, candidates: list[PaperCandidate]) -> None:
        grouped: dict[str, list[PaperCandidate]] = {}
        for candidate in candidates:
            grouped.setdefault(self._logical_paper_key(candidate), []).append(candidate)

        for group_key, items in grouped.items():
            metrics = self._ranking_metadata_cache.get(group_key)
            if metrics is None:
                metrics = self._resolve_candidate_ranking_metadata(items[0])
                self._ranking_metadata_cache[group_key] = metrics
            if not metrics:
                continue
            for candidate in items:
                ranking = candidate.raw.get("ranking", {})
                if not isinstance(ranking, dict):
                    ranking = {}
                ranking.update(metrics)
                candidate.raw["ranking"] = ranking

    def _resolve_candidate_ranking_metadata(self, candidate: PaperCandidate) -> dict[str, int | str]:
        citation_count, influential_count = citation_metrics_from_metadata(candidate.raw)
        if citation_count > 0 or influential_count > 0:
            ranking = candidate.raw.get("ranking", {})
            source = ranking.get("source", "candidate") if isinstance(ranking, dict) else "candidate"
            return {
                "citation_count": citation_count,
                "influential_citation_count": influential_count,
                "source": str(source),
            }
        existing_paper_id = self.db.find_existing_paper_id(candidate)
        if existing_paper_id is not None:
            paper = self.db.get_paper(existing_paper_id)
            citation_count, influential_count = citation_metrics_from_metadata(paper.metadata)
            if citation_count > 0 or influential_count > 0:
                return {
                    "citation_count": citation_count,
                    "influential_citation_count": influential_count,
                    "source": "database",
                }

        fields = "title,year,citationCount,influentialCitationCount"
        for identifier in self._semantic_scholar_identifiers(candidate):
            payload = self._fetch_semantic_scholar_details(identifier, fields)
            metrics = self._semantic_scholar_metrics(payload)
            if metrics:
                return metrics
        payload = self._search_semantic_scholar_by_title(candidate.title, fields)
        return self._semantic_scholar_metrics(payload)

    def _semantic_scholar_identifiers(self, candidate: PaperCandidate) -> list[str]:
        identifiers: list[str] = []
        if candidate.doi:
            identifiers.append(f"DOI:{candidate.doi}")
        if candidate.arxiv_id:
            identifiers.append(f"ARXIV:{candidate.arxiv_id}")
            if candidate.arxiv_id.lower().startswith("abs/"):
                identifiers.append(f"ARXIV:{candidate.arxiv_id.rsplit('/', 1)[-1]}")
        return identifiers

    def _semantic_scholar_headers(self) -> dict[str, str]:
        headers = {"User-Agent": "paper-monitor/0.1 (+local)"}
        if self.semantic_scholar_api_key:
            headers["x-api-key"] = self.semantic_scholar_api_key
        return headers

    def _fetch_semantic_scholar_details(self, identifier: str, fields: str) -> dict:
        encoded_identifier = urllib.parse.quote(identifier, safe="")
        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/{encoded_identifier}"
            f"?fields={urllib.parse.quote(fields, safe=',')}"
        )
        last_error: Exception | None = None
        for attempt in range(3):
            request = urllib.request.Request(url, headers=self._semantic_scholar_headers())
            try:
                with urllib.request.urlopen(request, timeout=15) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 404:
                    return {}
                if exc.code == 429 and attempt < 2:
                    time.sleep(1 + attempt)
                    continue
                return {}
            except (urllib.error.URLError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1 + attempt)
                    continue
                break
        if last_error:
            LOGGER.debug("semantic scholar details lookup failed for %s: %s", identifier, last_error)
        return {}

    def _search_semantic_scholar_by_title(self, title: str, fields: str) -> dict:
        if not title:
            return {}
        params = urllib.parse.urlencode(
            {
                "query": title,
                "limit": 1,
                "fields": fields,
            }
        )
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
        last_error: Exception | None = None
        for attempt in range(3):
            request = urllib.request.Request(url, headers=self._semantic_scholar_headers())
            try:
                with urllib.request.urlopen(request, timeout=15) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 429 and attempt < 2:
                    time.sleep(1 + attempt)
                    continue
                return {}
            except (urllib.error.URLError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1 + attempt)
                    continue
                break
        if last_error:
            LOGGER.debug("semantic scholar title lookup failed for %s: %s", title, last_error)
        return {}

    def _semantic_scholar_metrics(self, payload: dict) -> dict[str, int | str]:
        if not payload:
            return {}
        candidate = payload
        if isinstance(payload.get("data"), list) and payload["data"]:
            candidate = payload["data"][0]
        citation_count = int(candidate.get("citationCount") or 0)
        influential_count = int(candidate.get("influentialCitationCount") or 0)
        if citation_count <= 0 and influential_count <= 0:
            return {}
        return {
            "citation_count": citation_count,
            "influential_citation_count": influential_count,
            "source": "semantic_scholar",
        }
