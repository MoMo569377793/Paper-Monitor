from __future__ import annotations

import hashlib
import imaplib
import logging
import os
import re
from datetime import datetime
from email import message_from_bytes
from email.header import decode_header
from email.message import Message
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Callable
from urllib.parse import parse_qs, urlparse

from paper_monitor.models import FetchPlan, PaperCandidate, ScholarAlertsConfig, TopicConfig
from paper_monitor.utils import normalize_whitespace


LOGGER = logging.getLogger(__name__)
GENERIC_LINK_TEXT = {
    "related articles",
    "cited by",
    "all versions",
    "view all",
    "more results",
    "create alert",
    "edit alert",
    "unsubscribe",
}


class ScholarAlertsFetcher:
    def __init__(self, config: ScholarAlertsConfig, timezone_name: str) -> None:
        self.config = config
        self.timezone_name = timezone_name
        self.enabled = bool(config.enabled and config.imap_host and config.username)

    def fetch(
        self,
        topic: TopicConfig,
        queries: list[str],
        plan: FetchPlan | None = None,
        progress: Callable[[str, bool], None] | None = None,
    ) -> list[PaperCandidate]:
        if not self.enabled:
            return []
        if progress:
            progress(f"Scholar Alerts {topic.id} 收取邮件", True)

        password = os.environ.get(self.config.password_env, "")
        if not password:
            LOGGER.warning("scholar alerts password env %s is missing", self.config.password_env)
            return []

        with imaplib.IMAP4_SSL(self.config.imap_host, self.config.imap_port, timeout=self.config.timeout_seconds) as mail:
            mail.login(self.config.username, password)
            mail.select(self.config.folder)
            criterion = self._search_criterion()
            status, search_data = mail.search(None, criterion)
            if status != "OK":
                return []

            message_ids = search_data[0].split()
            candidates: list[PaperCandidate] = []
            for message_id in message_ids:
                fetch_status, payload = mail.fetch(message_id, "(BODY.PEEK[])")
                if fetch_status != "OK" or not payload or not isinstance(payload[0], tuple):
                    continue
                message = message_from_bytes(payload[0][1])
                candidates.extend(self._parse_message(message))
                if self.config.search_criterion.upper() == "UNSEEN":
                    mail.store(message_id, "+FLAGS", "\\Seen")

            if progress:
                progress(f"Scholar Alerts {topic.id} 命中 {len(candidates)}", False)
            LOGGER.info("scholar alerts topic=%s fetched=%s", topic.id, len(candidates))
            return candidates

    def _search_criterion(self) -> str:
        if self.config.subject_keyword:
            return f'({self.config.search_criterion} SUBJECT "{self.config.subject_keyword}")'
        return f"({self.config.search_criterion})"

    def _parse_message(self, message: Message) -> list[PaperCandidate]:
        subject = _decode_header(message.get("Subject", ""))
        published_at = _parse_email_date(message.get("Date"))
        body_html, body_text = _extract_message_bodies(message)
        links = _extract_links(body_html) if body_html else []
        if not links and body_text:
            links = _extract_links_from_text(body_text)

        candidates: list[PaperCandidate] = []
        for title, url in links:
            cleaned_title = normalize_whitespace(title)
            if not cleaned_title:
                continue
            if cleaned_title.lower() in GENERIC_LINK_TEXT:
                continue
            if len(cleaned_title) < 12:
                continue
            resolved_url = _resolve_google_redirect(url)
            source_id = hashlib.sha1(f"{cleaned_title}|{resolved_url}".encode("utf-8")).hexdigest()
            candidates.append(
                PaperCandidate(
                    source_name="google_scholar_alerts",
                    source_paper_id=source_id,
                    query_text=subject,
                    title=cleaned_title,
                    abstract="",
                    authors=[],
                    published_at=published_at,
                    updated_at=published_at,
                    primary_url=resolved_url,
                    pdf_url="",
                    doi="",
                    arxiv_id="",
                    venue="Google Scholar Alerts",
                    year=_extract_year_from_date(published_at),
                    categories=[],
                    raw={"subject": subject, "date": published_at},
                )
            )
        return candidates


def _decode_header(value: str) -> str:
    parts = decode_header(value)
    chunks: list[str] = []
    for payload, encoding in parts:
        if isinstance(payload, bytes):
            chunks.append(payload.decode(encoding or "utf-8", errors="replace"))
        else:
            chunks.append(str(payload))
    return normalize_whitespace("".join(chunks))


def _parse_email_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return None


def _extract_message_bodies(message: Message) -> tuple[str, str]:
    html_parts: list[str] = []
    text_parts: list[str] = []
    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            if content_type not in {"text/plain", "text/html"}:
                continue
            payload = part.get_payload(decode=True) or b""
            charset = part.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
            if content_type == "text/html":
                html_parts.append(text)
            else:
                text_parts.append(text)
    else:
        payload = message.get_payload(decode=True) or b""
        charset = message.get_content_charset() or "utf-8"
        text = payload.decode(charset, errors="replace")
        if message.get_content_type() == "text/html":
            html_parts.append(text)
        else:
            text_parts.append(text)
    return "\n".join(html_parts), "\n".join(text_parts)


class _LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._current_href = ""
        self._current_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr_map = dict(attrs)
        self._current_href = attr_map.get("href", "") or ""
        self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._current_href:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._current_href:
            return
        text = normalize_whitespace("".join(self._current_text))
        self.links.append((text, self._current_href))
        self._current_href = ""
        self._current_text = []


def _extract_links(html_text: str) -> list[tuple[str, str]]:
    parser = _LinkCollector()
    parser.feed(html_text)
    items: list[tuple[str, str]] = []
    for text, href in parser.links:
        if not href.startswith("http"):
            continue
        lower_href = href.lower()
        if "scholar.google" not in lower_href and "http" in lower_href:
            items.append((text, href))
            continue
        if "scholar_url" in lower_href or "url=" in lower_href:
            items.append((text, href))
    return items


def _extract_links_from_text(text: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"(?P<title>.+?)\s+(?P<url>https?://\S+)")
    items: list[tuple[str, str]] = []
    for line in text.splitlines():
        match = pattern.search(line.strip())
        if match:
            items.append((match.group("title"), match.group("url")))
    return items


def _resolve_google_redirect(url: str) -> str:
    parsed = urlparse(url)
    if "scholar.google" not in parsed.netloc:
        return url
    query = parse_qs(parsed.query)
    targets = query.get("url")
    if targets:
        return targets[0]
    return url


def _extract_year_from_date(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).year
    except ValueError:
        return None
