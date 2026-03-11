from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from paper_monitor.models import FetchPlan, GenericSourceConfig, PaperCandidate, TopicConfig
from paper_monitor.utils import normalize_whitespace


LOGGER = logging.getLogger(__name__)


class DBLPFetcher:
    def __init__(self, config: GenericSourceConfig) -> None:
        self.config = config
        self.enabled = config.enabled

    def fetch(
        self,
        topic: TopicConfig,
        queries: list[str],
        plan: FetchPlan | None = None,
        progress: Callable[[str, bool], None] | None = None,
    ) -> list[PaperCandidate]:
        if not self.enabled:
            return []
        plan = plan or FetchPlan(recent_limit=self.config.max_results, page_size=self.config.max_results)
        candidates: list[PaperCandidate] = []
        for index, query in enumerate(queries, start=1):
            if progress:
                progress(f"DBLP {topic.id} 查询 {index}/{len(queries)}", True)
            query_items = self._fetch_query(query, plan)
            candidates.extend(query_items)
            if progress:
                progress(f"DBLP {topic.id} 查询 {index}/{len(queries)} 命中 {len(query_items)}", False)
        candidates = self._apply_plan(candidates, plan)
        LOGGER.info("dblp topic=%s fetched=%s", topic.id, len(candidates))
        return candidates

    def _fetch_query(self, query: str, plan: FetchPlan) -> list[PaperCandidate]:
        page_size = max(1, min(plan.page_size or self.config.max_results, 1000))
        offset = 0
        scanned = 0
        scan_cap = max(plan.recent_limit or 0, self.config.max_results, page_size)
        if plan.start_year is not None:
            scan_cap = max(scan_cap, 300)

        items: list[PaperCandidate] = []
        while scanned < scan_cap:
            request_size = min(page_size, scan_cap - scanned)
            params = urllib.parse.urlencode({"q": query, "h": request_size, "f": offset, "format": "json"})
            url = f"https://dblp.org/search/publ/api?{params}"
            payload = self._fetch_with_retry(url, query)
            if not payload:
                break
            batch = self._parse_json(payload, query)
            if not batch:
                break
            items.extend(batch)
            scanned += len(batch)
            offset += len(batch)
            if len(batch) < request_size:
                break
        return items

    def _apply_plan(self, candidates: list[PaperCandidate], plan: FetchPlan) -> list[PaperCandidate]:
        filtered = candidates
        if plan.start_year is not None:
            filtered = [item for item in filtered if item.year is not None and item.year >= plan.start_year]
        filtered.sort(key=lambda item: (item.year or 0, item.title), reverse=True)
        if plan.recent_limit:
            filtered = filtered[: plan.recent_limit]
        return filtered

    def _fetch_with_retry(self, url: str, query: str) -> str | None:
        for attempt in range(3):
            request = urllib.request.Request(url, headers={"User-Agent": self.config.user_agent})
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                if exc.code >= 500 and attempt < 2:
                    time.sleep(1 + attempt)
                    continue
                LOGGER.warning("dblp query failed: %s (%s)", query, exc)
                return None
            except urllib.error.URLError as exc:
                LOGGER.warning("dblp query failed: %s (%s)", query, exc)
                return None
        return None

    def _parse_json(self, payload: str, query: str) -> list[PaperCandidate]:
        data = json.loads(payload)
        raw_hits = data.get("result", {}).get("hits", {}).get("hit", [])
        if isinstance(raw_hits, dict):
            raw_hits = [raw_hits]

        items: list[PaperCandidate] = []
        for hit in raw_hits:
            info = hit.get("info", {})
            title = normalize_whitespace(_ensure_string(info.get("title")))
            if not title:
                continue

            authors = _extract_authors(info)
            ee_value = info.get("ee", "")
            primary_url = _extract_primary_url(ee_value) or _ensure_string(info.get("url"))
            source_id = _ensure_string(info.get("key")) or primary_url or title
            venue = normalize_whitespace(_ensure_string(info.get("venue")))
            year_value = _extract_year(info.get("year"))
            doi = normalize_whitespace(_ensure_string(info.get("doi")))

            items.append(
                PaperCandidate(
                    source_name="dblp",
                    source_paper_id=source_id,
                    query_text=query,
                    title=title,
                    abstract="",
                    authors=authors,
                    published_at=str(year_value) if year_value else None,
                    updated_at=None,
                    primary_url=primary_url,
                    pdf_url="",
                    doi=doi,
                    arxiv_id="",
                    venue=venue or "DBLP",
                    year=year_value,
                    categories=[normalize_whitespace(_ensure_string(info.get("type")))],
                    raw={"hit": hit, "query": query},
                )
            )
        return items


def _ensure_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text
    return ""


def _extract_authors(info: dict[str, Any]) -> list[str]:
    raw_authors = info.get("authors", {})
    author_field = raw_authors.get("author", []) if isinstance(raw_authors, dict) else raw_authors
    if isinstance(author_field, dict):
        author_field = [author_field]
    authors: list[str] = []
    for item in author_field:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = _ensure_string(item)
        else:
            text = ""
        text = normalize_whitespace(text)
        if text:
            authors.append(text)
    return authors


def _extract_primary_url(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.startswith("http"):
                return item
            if isinstance(item, dict):
                text = _ensure_string(item)
                if text.startswith("http"):
                    return text
        return ""
    if isinstance(value, dict):
        text = _ensure_string(value)
        return text if text.startswith("http") else ""
    if isinstance(value, str):
        return value
    return ""


def _extract_year(value: Any) -> int | None:
    text = _ensure_string(value) if not isinstance(value, int) else str(value)
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return None
