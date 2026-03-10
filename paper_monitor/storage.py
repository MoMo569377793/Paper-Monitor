from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from paper_monitor.models import PaperCandidate, PaperRecord, ReportEntry, TopicEvaluation
from paper_monitor.utils import (
    choose_earlier_date,
    choose_later_date,
    ensure_directory,
    json_dumps,
    normalize_title,
    now_iso,
    safe_json_loads,
    unique_strings,
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    title_norm TEXT NOT NULL,
    abstract TEXT NOT NULL DEFAULT '',
    authors_json TEXT NOT NULL DEFAULT '[]',
    published_at TEXT,
    updated_at TEXT,
    primary_url TEXT NOT NULL DEFAULT '',
    pdf_url TEXT NOT NULL DEFAULT '',
    doi TEXT NOT NULL DEFAULT '',
    arxiv_id TEXT NOT NULL DEFAULT '',
    venue TEXT NOT NULL DEFAULT '',
    year INTEGER,
    categories_json TEXT NOT NULL DEFAULT '[]',
    summary_text TEXT NOT NULL DEFAULT '',
    summary_basis TEXT NOT NULL DEFAULT 'metadata-only',
    tags_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    source_first TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_doi_unique
ON papers(doi)
WHERE doi != '';

CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_arxiv_unique
ON papers(arxiv_id)
WHERE arxiv_id != '';

CREATE INDEX IF NOT EXISTS idx_papers_title_norm
ON papers(title_norm);

CREATE TABLE IF NOT EXISTS paper_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL,
    source_name TEXT NOT NULL,
    source_paper_id TEXT NOT NULL,
    source_url TEXT NOT NULL DEFAULT '',
    query_text TEXT NOT NULL DEFAULT '',
    raw_json TEXT NOT NULL DEFAULT '{}',
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    FOREIGN KEY(paper_id) REFERENCES papers(id) ON DELETE CASCADE,
    UNIQUE(source_name, source_paper_id)
);

CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL,
    topic_id TEXT NOT NULL,
    topic_name TEXT NOT NULL,
    score REAL NOT NULL,
    classification TEXT NOT NULL,
    matched_keywords_json TEXT NOT NULL DEFAULT '[]',
    reasons_json TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL,
    FOREIGN KEY(paper_id) REFERENCES papers(id) ON DELETE CASCADE,
    UNIQUE(paper_id, topic_id)
);

CREATE TABLE IF NOT EXISTS checkpoints (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_type TEXT NOT NULL,
    report_date TEXT NOT NULL,
    path_md TEXT NOT NULL,
    path_html TEXT NOT NULL,
    path_json TEXT NOT NULL,
    meta_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE(report_type, report_date)
);
"""


class Database:
    def __init__(self, path: Path, timezone_name: str) -> None:
        self.path = path
        self.timezone_name = timezone_name
        ensure_directory(path.parent)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")

    def close(self) -> None:
        self.connection.close()

    def initialize(self) -> None:
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    def _find_existing_paper(self, candidate: PaperCandidate) -> sqlite3.Row | None:
        if candidate.doi:
            row = self.connection.execute("SELECT * FROM papers WHERE doi = ?", (candidate.doi,)).fetchone()
            if row:
                return row
        if candidate.arxiv_id:
            row = self.connection.execute("SELECT * FROM papers WHERE arxiv_id = ?", (candidate.arxiv_id,)).fetchone()
            if row:
                return row
        title_norm = normalize_title(candidate.title)
        return self.connection.execute(
            "SELECT * FROM papers WHERE title_norm = ? ORDER BY id LIMIT 1",
            (title_norm,),
        ).fetchone()

    def upsert_paper(self, candidate: PaperCandidate) -> tuple[int, bool]:
        existing = self._find_existing_paper(candidate)
        now = now_iso(self.timezone_name)
        title_norm = normalize_title(candidate.title)
        categories = unique_strings(candidate.categories)
        metadata_json = json_dumps(candidate.raw)

        if existing is None:
            cursor = self.connection.execute(
                """
                INSERT INTO papers (
                    title, title_norm, abstract, authors_json, published_at, updated_at,
                    primary_url, pdf_url, doi, arxiv_id, venue, year, categories_json,
                    summary_text, summary_basis, tags_json, metadata_json, source_first,
                    created_at, last_seen_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', 'metadata-only', '[]', ?, ?, ?, ?)
                """,
                (
                    candidate.title,
                    title_norm,
                    candidate.abstract or "",
                    json_dumps(unique_strings(candidate.authors)),
                    candidate.published_at,
                    candidate.updated_at,
                    candidate.primary_url,
                    candidate.pdf_url,
                    candidate.doi,
                    candidate.arxiv_id,
                    candidate.venue,
                    candidate.year,
                    json_dumps(categories),
                    metadata_json,
                    candidate.source_name,
                    now,
                    now,
                ),
            )
            paper_id = int(cursor.lastrowid)
            self._upsert_source_link(paper_id, candidate, now)
            self.connection.commit()
            return paper_id, True

        merged_authors = unique_strings(
            safe_json_loads(existing["authors_json"], []) + unique_strings(candidate.authors)
        )
        merged_categories = unique_strings(safe_json_loads(existing["categories_json"], []) + categories)
        merged_metadata = safe_json_loads(existing["metadata_json"], {})
        if isinstance(merged_metadata, dict):
            merged_metadata[candidate.source_name] = candidate.raw

        abstract = existing["abstract"] or ""
        if len(candidate.abstract or "") > len(abstract):
            abstract = candidate.abstract or ""

        primary_url = existing["primary_url"] or candidate.primary_url
        pdf_url = existing["pdf_url"] or candidate.pdf_url
        venue = existing["venue"] or candidate.venue
        year = existing["year"] or candidate.year
        doi = existing["doi"] or candidate.doi
        arxiv_id = existing["arxiv_id"] or candidate.arxiv_id

        self.connection.execute(
            """
            UPDATE papers
            SET title = ?, abstract = ?, authors_json = ?, published_at = ?, updated_at = ?,
                primary_url = ?, pdf_url = ?, doi = ?, arxiv_id = ?, venue = ?, year = ?,
                categories_json = ?, metadata_json = ?, last_seen_at = ?
            WHERE id = ?
            """,
            (
                existing["title"] or candidate.title,
                abstract,
                json_dumps(merged_authors),
                choose_earlier_date(existing["published_at"], candidate.published_at),
                choose_later_date(existing["updated_at"], candidate.updated_at),
                primary_url,
                pdf_url,
                doi,
                arxiv_id,
                venue,
                year,
                json_dumps(merged_categories),
                json_dumps(merged_metadata),
                now,
                existing["id"],
            ),
        )
        self._upsert_source_link(int(existing["id"]), candidate, now)
        self.connection.commit()
        return int(existing["id"]), False

    def _upsert_source_link(self, paper_id: int, candidate: PaperCandidate, now: str) -> None:
        existing = self.connection.execute(
            "SELECT id FROM paper_sources WHERE source_name = ? AND source_paper_id = ?",
            (candidate.source_name, candidate.source_paper_id),
        ).fetchone()
        if existing is None:
            self.connection.execute(
                """
                INSERT INTO paper_sources (
                    paper_id, source_name, source_paper_id, source_url, query_text, raw_json,
                    first_seen_at, last_seen_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper_id,
                    candidate.source_name,
                    candidate.source_paper_id,
                    candidate.primary_url,
                    candidate.query_text,
                    json_dumps(candidate.raw),
                    now,
                    now,
                ),
            )
            return

        self.connection.execute(
            """
            UPDATE paper_sources
            SET paper_id = ?, source_url = ?, query_text = ?, raw_json = ?, last_seen_at = ?
            WHERE id = ?
            """,
            (
                paper_id,
                candidate.primary_url,
                candidate.query_text,
                json_dumps(candidate.raw),
                now,
                existing["id"],
            ),
        )

    def get_paper(self, paper_id: int) -> PaperRecord:
        row = self.connection.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if row is None:
            raise KeyError(f"paper not found: {paper_id}")
        return self._row_to_paper(row)

    def _row_to_paper(self, row: sqlite3.Row) -> PaperRecord:
        return PaperRecord(
            id=int(row["id"]),
            title=row["title"],
            title_norm=row["title_norm"],
            abstract=row["abstract"] or "",
            authors=safe_json_loads(row["authors_json"], []),
            published_at=row["published_at"],
            updated_at=row["updated_at"],
            primary_url=row["primary_url"] or "",
            pdf_url=row["pdf_url"] or "",
            doi=row["doi"] or "",
            arxiv_id=row["arxiv_id"] or "",
            venue=row["venue"] or "",
            year=row["year"],
            categories=safe_json_loads(row["categories_json"], []),
            summary_text=row["summary_text"] or "",
            summary_basis=row["summary_basis"] or "metadata-only",
            tags=safe_json_loads(row["tags_json"], []),
            source_first=row["source_first"],
            created_at=row["created_at"],
            last_seen_at=row["last_seen_at"],
            metadata=safe_json_loads(row["metadata_json"], {}),
        )

    def update_paper_analysis(self, paper_id: int, summary_text: str, summary_basis: str, tags: list[str]) -> None:
        self.connection.execute(
            """
            UPDATE papers
            SET summary_text = ?, summary_basis = ?, tags_json = ?
            WHERE id = ?
            """,
            (summary_text, summary_basis, json_dumps(unique_strings(tags)), paper_id),
        )
        self.connection.commit()

    def upsert_match(self, paper_id: int, evaluation: TopicEvaluation) -> bool:
        now = now_iso(self.timezone_name)
        existing = self.connection.execute(
            "SELECT id FROM matches WHERE paper_id = ? AND topic_id = ?",
            (paper_id, evaluation.topic_id),
        ).fetchone()
        if existing is None:
            self.connection.execute(
                """
                INSERT INTO matches (
                    paper_id, topic_id, topic_name, score, classification,
                    matched_keywords_json, reasons_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper_id,
                    evaluation.topic_id,
                    evaluation.topic_name,
                    evaluation.score,
                    evaluation.classification,
                    json_dumps(evaluation.matched_keywords),
                    json_dumps(evaluation.reasons),
                    now,
                ),
            )
            self.connection.commit()
            return True

        self.connection.execute(
            """
            UPDATE matches
            SET topic_name = ?, score = ?, classification = ?, matched_keywords_json = ?,
                reasons_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                evaluation.topic_name,
                evaluation.score,
                evaluation.classification,
                json_dumps(evaluation.matched_keywords),
                json_dumps(evaluation.reasons),
                now,
                existing["id"],
            ),
        )
        self.connection.commit()
        return False

    def fetch_report_entries(self, start_at: str, end_at: str, include_maybe: bool) -> list[ReportEntry]:
        classifications = ("relevant", "maybe") if include_maybe else ("relevant",)
        placeholders = ", ".join(["?"] * len(classifications))
        rows = self.connection.execute(
            f"""
            SELECT
                m.topic_id,
                m.topic_name,
                m.score,
                m.classification,
                m.matched_keywords_json,
                m.reasons_json,
                p.*,
                GROUP_CONCAT(DISTINCT ps.source_name) AS source_names_csv,
                GROUP_CONCAT(DISTINCT ps.source_url) AS source_urls_csv
            FROM matches m
            JOIN papers p ON p.id = m.paper_id
            LEFT JOIN paper_sources ps ON ps.paper_id = p.id
            WHERE p.created_at BETWEEN ? AND ?
              AND m.classification IN ({placeholders})
            GROUP BY m.id
            ORDER BY m.topic_name ASC, m.score DESC, p.published_at DESC, p.id DESC
            """,
            (start_at, end_at, *classifications),
        ).fetchall()

        entries: list[ReportEntry] = []
        for row in rows:
            paper = self._row_to_paper(row)
            source_names = [name for name in (row["source_names_csv"] or "").split(",") if name]
            source_urls = [url for url in (row["source_urls_csv"] or "").split(",") if url]
            entries.append(
                ReportEntry(
                    topic_id=row["topic_id"],
                    topic_name=row["topic_name"],
                    score=float(row["score"]),
                    classification=row["classification"],
                    matched_keywords=safe_json_loads(row["matched_keywords_json"], []),
                    reasons=safe_json_loads(row["reasons_json"], []),
                    paper=paper,
                    source_names=source_names,
                    source_urls=source_urls,
                )
            )
        return entries

    def count_matches(self, start_at: str, end_at: str, include_maybe: bool) -> int:
        classifications = ("relevant", "maybe") if include_maybe else ("relevant",)
        placeholders = ", ".join(["?"] * len(classifications))
        row = self.connection.execute(
            f"""
            SELECT COUNT(*) AS count
            FROM matches m
            JOIN papers p ON p.id = m.paper_id
            WHERE p.created_at BETWEEN ? AND ?
              AND m.classification IN ({placeholders})
            """,
            (start_at, end_at, *classifications),
        ).fetchone()
        return int(row["count"])

    def record_report(
        self,
        report_type: str,
        report_date: str,
        path_md: str,
        path_html: str,
        path_json: str,
        meta: dict[str, Any],
    ) -> None:
        now = now_iso(self.timezone_name)
        self.connection.execute(
            """
            INSERT INTO reports (report_type, report_date, path_md, path_html, path_json, meta_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(report_type, report_date)
            DO UPDATE SET
                path_md = excluded.path_md,
                path_html = excluded.path_html,
                path_json = excluded.path_json,
                meta_json = excluded.meta_json,
                created_at = excluded.created_at
            """,
            (report_type, report_date, path_md, path_html, path_json, json_dumps(meta), now),
        )
        self.connection.commit()
