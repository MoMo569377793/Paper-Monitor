from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from typing import Any

from paper_monitor.models import GenericSourceConfig, PaperCandidate, TopicConfig
from paper_monitor.utils import normalize_whitespace


LOGGER = logging.getLogger(__name__)


class DBLPFetcher:
    def __init__(self, config: GenericSourceConfig) -> None:
        self.config = config
        self.enabled = config.enabled

    def fetch(self, topic: TopicConfig, queries: list[str]) -> list[PaperCandidate]:
        if not self.enabled:
            return []
        candidates: list[PaperCandidate] = []
        for query in queries:
            params = urllib.parse.urlencode({"q": query, "h": self.config.max_results, "f": 0, "format": "json"})
            url = f"https://dblp.org/search/publ/api?{params}"
            request = urllib.request.Request(url, headers={"User-Agent": self.config.user_agent})
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                payload = response.read().decode("utf-8")
            candidates.extend(self._parse_json(payload, query))
        LOGGER.info("dblp topic=%s fetched=%s", topic.id, len(candidates))
        return candidates

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
