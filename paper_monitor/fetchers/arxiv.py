from __future__ import annotations

import logging
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from paper_monitor.models import GenericSourceConfig, PaperCandidate, TopicConfig
from paper_monitor.utils import normalize_whitespace


LOGGER = logging.getLogger(__name__)

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivFetcher:
    def __init__(self, config: GenericSourceConfig) -> None:
        self.config = config
        self.enabled = config.enabled

    def fetch(self, topic: TopicConfig, queries: list[str]) -> list[PaperCandidate]:
        if not self.enabled:
            return []
        candidates: list[PaperCandidate] = []
        for query in queries:
            params = urllib.parse.urlencode(
                {
                    "search_query": query,
                    "start": 0,
                    "max_results": self.config.max_results,
                    "sortBy": "lastUpdatedDate",
                    "sortOrder": "descending",
                }
            )
            url = f"http://export.arxiv.org/api/query?{params}"
            request = urllib.request.Request(url, headers={"User-Agent": self.config.user_agent})
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                xml_text = response.read().decode("utf-8")
            candidates.extend(self._parse_feed(xml_text, query))
        LOGGER.info("arXiv topic=%s fetched=%s", topic.id, len(candidates))
        return candidates

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
                if title_attr == "pdf" or rel == "related":
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
