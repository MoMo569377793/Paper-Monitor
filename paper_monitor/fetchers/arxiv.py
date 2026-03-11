from __future__ import annotations

import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Callable

from paper_monitor.models import FetchPlan, GenericSourceConfig, PaperCandidate, TopicConfig
from paper_monitor.utils import normalize_whitespace


LOGGER = logging.getLogger(__name__)

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivFetcher:
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
                progress(f"arXiv {topic.id} 查询 {index}/{len(queries)}", True)
            query_items = self._fetch_query(query, plan)
            candidates.extend(query_items)
            if progress:
                progress(f"arXiv {topic.id} 查询 {index}/{len(queries)} 命中 {len(query_items)}", False)
        candidates = self._apply_plan(candidates, plan)
        LOGGER.info("arXiv topic=%s fetched=%s", topic.id, len(candidates))
        return candidates

    def _fetch_query(self, query: str, plan: FetchPlan) -> list[PaperCandidate]:
        page_size = max(1, min(plan.page_size or self.config.max_results, 100))
        offset = 0
        scanned = 0
        scan_cap = max(plan.recent_limit or 0, self.config.max_results, page_size)
        if plan.start_year is not None:
            scan_cap = max(scan_cap, 500)

        items: list[PaperCandidate] = []
        while scanned < scan_cap:
            request_size = min(page_size, scan_cap - scanned)
            params = urllib.parse.urlencode(
                {
                    "search_query": query,
                    "start": offset,
                    "max_results": request_size,
                    "sortBy": "lastUpdatedDate",
                    "sortOrder": "descending",
                }
            )
            url = f"http://export.arxiv.org/api/query?{params}"
            xml_text = self._request_feed(url)
            batch = self._parse_feed(xml_text, query)
            if not batch:
                break
            items.extend(batch)
            scanned += len(batch)
            offset += len(batch)
            if len(batch) < request_size:
                break
            if plan.start_year is not None and all((item.year or 0) < plan.start_year for item in batch):
                break
        return items

    def _apply_plan(self, candidates: list[PaperCandidate], plan: FetchPlan) -> list[PaperCandidate]:
        filtered = candidates
        if plan.start_year is not None:
            filtered = [item for item in filtered if item.year is not None and item.year >= plan.start_year]
        filtered.sort(key=lambda item: ((item.updated_at or item.published_at or ""), item.title), reverse=True)
        if plan.recent_limit:
            filtered = filtered[: plan.recent_limit]
        return filtered

    def _request_feed(self, url: str) -> str:
        last_error: Exception | None = None
        for attempt in range(3):
            request = urllib.request.Request(url, headers={"User-Agent": self.config.user_agent})
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code in {429, 500, 502, 503, 504} and attempt < 2:
                    time.sleep(2 + attempt)
                    continue
                raise
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2 + attempt)
                    continue
                raise
        raise RuntimeError(f"arXiv request failed after retries: {last_error}")

    def _parse_feed(self, xml_text: str, query: str) -> list[PaperCandidate]:
        root = ET.fromstring(xml_text)
        items: list[PaperCandidate] = []
        for entry in root.findall("atom:entry", ATOM_NS):
            id_text = entry.findtext("atom:id", default="", namespaces=ATOM_NS).strip()
            source_id = id_text.rsplit("/", 1)[-1]
            title = normalize_whitespace(entry.findtext("atom:title", default="", namespaces=ATOM_NS))
            summary = normalize_whitespace(entry.findtext("atom:summary", default="", namespaces=ATOM_NS))
            authors = [
                normalize_whitespace(node.findtext("atom:name", default="", namespaces=ATOM_NS))
                for node in entry.findall("atom:author", ATOM_NS)
            ]
            authors = [author for author in authors if author]
            categories = [node.attrib.get("term", "") for node in entry.findall("atom:category", ATOM_NS)]
            pdf_url = ""
            primary_url = id_text
            for link_node in entry.findall("atom:link", ATOM_NS):
                href = link_node.attrib.get("href", "")
                title_attr = link_node.attrib.get("title", "")
                rel = link_node.attrib.get("rel", "")
                if title_attr == "pdf" or "/pdf/" in href:
                    pdf_url = href
                if rel == "alternate" and href:
                    primary_url = href

            doi = entry.findtext("arxiv:doi", default="", namespaces=ATOM_NS).strip()
            items.append(
                PaperCandidate(
                    source_name="arxiv",
                    source_paper_id=source_id,
                    query_text=query,
                    title=title,
                    abstract=summary,
                    authors=authors,
                    published_at=entry.findtext("atom:published", default="", namespaces=ATOM_NS) or None,
                    updated_at=entry.findtext("atom:updated", default="", namespaces=ATOM_NS) or None,
                    primary_url=primary_url,
                    pdf_url=pdf_url,
                    doi=doi,
                    arxiv_id=source_id,
                    venue="arXiv",
                    year=_year_from_datetime(entry.findtext("atom:published", default="", namespaces=ATOM_NS)),
                    categories=[category for category in categories if category],
                    raw={"id": id_text, "query": query},
                )
            )
        return items


def _year_from_datetime(value: str) -> int | None:
    if len(value) >= 4 and value[:4].isdigit():
        return int(value[:4])
    return None
