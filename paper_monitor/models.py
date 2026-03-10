from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TopicConfig:
    id: str
    display_name: str
    description: str
    source_queries: dict[str, list[str]]
    must_match_groups: list[list[str]] = field(default_factory=list)
    positive_keywords: list[str] = field(default_factory=list)
    exclude_keywords: list[str] = field(default_factory=list)
    arxiv_categories: list[str] = field(default_factory=list)
    dblp_venue_keywords: list[str] = field(default_factory=list)
    threshold: float = 18.0


@dataclass(slots=True)
class GenericSourceConfig:
    enabled: bool = True
    max_results: int = 25
    timeout_seconds: int = 20
    user_agent: str = "paper-monitor/0.1 (+local)"


@dataclass(slots=True)
class ScholarAlertsConfig:
    enabled: bool = False
    imap_host: str = ""
    imap_port: int = 993
    username: str = ""
    password_env: str = "SCHOLAR_ALERTS_PASSWORD"
    folder: str = "INBOX"
    subject_keyword: str = "Google Scholar Alerts"
    search_criterion: str = "UNSEEN"
    timeout_seconds: int = 20


@dataclass(slots=True)
class ReportConfig:
    top_n_per_topic: int = 5
    lookback_days: int = 1
    include_maybe: bool = True


@dataclass(slots=True)
class Settings:
    base_dir: Path
    config_path: Path
    database_path: Path
    report_dir: Path
    export_dir: Path
    poll_minutes: int
    timezone: str
    arxiv: GenericSourceConfig
    dblp: GenericSourceConfig
    scholar_alerts: ScholarAlertsConfig
    report: ReportConfig
    topics: list[TopicConfig]


@dataclass(slots=True)
class PaperCandidate:
    source_name: str
    source_paper_id: str
    query_text: str
    title: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    published_at: str | None = None
    updated_at: str | None = None
    primary_url: str = ""
    pdf_url: str = ""
    doi: str = ""
    arxiv_id: str = ""
    venue: str = ""
    year: int | None = None
    categories: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PaperRecord:
    id: int
    title: str
    title_norm: str
    abstract: str
    authors: list[str]
    published_at: str | None
    updated_at: str | None
    primary_url: str
    pdf_url: str
    doi: str
    arxiv_id: str
    venue: str
    year: int | None
    categories: list[str]
    summary_text: str
    summary_basis: str
    tags: list[str]
    source_first: str
    created_at: str
    last_seen_at: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class TopicEvaluation:
    topic_id: str
    topic_name: str
    score: float
    classification: str
    matched_keywords: list[str]
    reasons: list[str]


@dataclass(slots=True)
class ReportEntry:
    topic_id: str
    topic_name: str
    score: float
    classification: str
    matched_keywords: list[str]
    reasons: list[str]
    paper: PaperRecord
    source_names: list[str]
    source_urls: list[str]


@dataclass(slots=True)
class RunStats:
    fetched: int = 0
    processed: int = 0
    stored: int = 0
    matched: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
