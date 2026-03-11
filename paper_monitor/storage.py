from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from paper_monitor.models import PaperCandidate, PaperLLMSummary, PaperRecord, ReportEntry, TopicEvaluation
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
    pdf_local_path TEXT NOT NULL DEFAULT '',
    pdf_status TEXT NOT NULL DEFAULT 'pending',
    pdf_downloaded_at TEXT,
    fulltext_txt_path TEXT NOT NULL DEFAULT '',
    fulltext_excerpt TEXT NOT NULL DEFAULT '',
    fulltext_status TEXT NOT NULL DEFAULT 'empty',
    page_count INTEGER,
    llm_summary_json TEXT NOT NULL DEFAULT '{}',
    analysis_updated_at TEXT,
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

CREATE TABLE IF NOT EXISTS paper_llm_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL,
    variant_id TEXT NOT NULL,
    variant_label TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT '',
    base_url TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    summary_text TEXT NOT NULL DEFAULT '',
    summary_basis TEXT NOT NULL DEFAULT '',
    tags_json TEXT NOT NULL DEFAULT '[]',
    structured_json TEXT NOT NULL DEFAULT '{}',
    usage_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(paper_id) REFERENCES papers(id) ON DELETE CASCADE,
    UNIQUE(paper_id, variant_id)
);

CREATE INDEX IF NOT EXISTS idx_paper_llm_summaries_paper
ON paper_llm_summaries(paper_id);
"""

PAPER_COLUMN_MIGRATIONS = {
    "pdf_local_path": "ALTER TABLE papers ADD COLUMN pdf_local_path TEXT NOT NULL DEFAULT ''",
    "pdf_status": "ALTER TABLE papers ADD COLUMN pdf_status TEXT NOT NULL DEFAULT 'pending'",
    "pdf_downloaded_at": "ALTER TABLE papers ADD COLUMN pdf_downloaded_at TEXT",
    "fulltext_txt_path": "ALTER TABLE papers ADD COLUMN fulltext_txt_path TEXT NOT NULL DEFAULT ''",
    "fulltext_excerpt": "ALTER TABLE papers ADD COLUMN fulltext_excerpt TEXT NOT NULL DEFAULT ''",
    "fulltext_status": "ALTER TABLE papers ADD COLUMN fulltext_status TEXT NOT NULL DEFAULT 'empty'",
    "page_count": "ALTER TABLE papers ADD COLUMN page_count INTEGER",
    "llm_summary_json": "ALTER TABLE papers ADD COLUMN llm_summary_json TEXT NOT NULL DEFAULT '{}'",
    "analysis_updated_at": "ALTER TABLE papers ADD COLUMN analysis_updated_at TEXT",
}


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
        self._run_migrations()
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_papers_fulltext_status ON papers(fulltext_status)"
        )
        self.connection.commit()

    def _run_migrations(self) -> None:
        existing_columns = {
            row["name"] for row in self.connection.execute("PRAGMA table_info(papers)").fetchall()
        }
        for column_name, ddl in PAPER_COLUMN_MIGRATIONS.items():
            if column_name in existing_columns:
                continue
            self.connection.execute(ddl)

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
            pdf_local_path=row["pdf_local_path"] or "",
            pdf_status=row["pdf_status"] or "pending",
            pdf_downloaded_at=row["pdf_downloaded_at"],
            fulltext_txt_path=row["fulltext_txt_path"] or "",
            fulltext_excerpt=row["fulltext_excerpt"] or "",
            fulltext_status=row["fulltext_status"] or "empty",
            page_count=row["page_count"],
            llm_summary=safe_json_loads(row["llm_summary_json"], {}),
            analysis_updated_at=row["analysis_updated_at"],
            source_first=row["source_first"],
            created_at=row["created_at"],
            last_seen_at=row["last_seen_at"],
            metadata=safe_json_loads(row["metadata_json"], {}),
        )

    def update_paper_analysis(
        self,
        paper_id: int,
        summary_text: str,
        summary_basis: str,
        tags: list[str],
        llm_summary: dict[str, Any] | None = None,
    ) -> None:
        now = now_iso(self.timezone_name)
        self.connection.execute(
            """
            UPDATE papers
            SET summary_text = ?, summary_basis = ?, tags_json = ?, llm_summary_json = ?, analysis_updated_at = ?
            WHERE id = ?
            """,
            (
                summary_text,
                summary_basis,
                json_dumps(unique_strings(tags)),
                json_dumps(llm_summary or {}),
                now,
                paper_id,
            ),
        )
        self.connection.commit()

    def update_paper_assets(
        self,
        paper_id: int,
        *,
        pdf_local_path: str,
        pdf_status: str,
        pdf_downloaded_at: str | None,
        fulltext_txt_path: str,
        fulltext_excerpt: str,
        fulltext_status: str,
        page_count: int | None,
    ) -> None:
        self.connection.execute(
            """
            UPDATE papers
            SET pdf_local_path = ?, pdf_status = ?, pdf_downloaded_at = ?, fulltext_txt_path = ?,
                fulltext_excerpt = ?, fulltext_status = ?, page_count = ?
            WHERE id = ?
            """,
            (
                pdf_local_path,
                pdf_status,
                pdf_downloaded_at,
                fulltext_txt_path,
                fulltext_excerpt,
                fulltext_status,
                page_count,
                paper_id,
            ),
        )
        self.connection.commit()

    def upsert_paper_llm_summary(
        self,
        paper_id: int,
        *,
        variant_id: str,
        variant_label: str,
        provider: str,
        base_url: str,
        model: str,
        summary_text: str,
        summary_basis: str,
        tags: list[str],
        structured: dict[str, Any],
        usage: dict[str, Any],
    ) -> None:
        now = now_iso(self.timezone_name)
        self.connection.execute(
            """
            INSERT INTO paper_llm_summaries (
                paper_id, variant_id, variant_label, provider, base_url, model,
                summary_text, summary_basis, tags_json, structured_json, usage_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id, variant_id)
            DO UPDATE SET
                variant_label = excluded.variant_label,
                provider = excluded.provider,
                base_url = excluded.base_url,
                model = excluded.model,
                summary_text = excluded.summary_text,
                summary_basis = excluded.summary_basis,
                tags_json = excluded.tags_json,
                structured_json = excluded.structured_json,
                usage_json = excluded.usage_json,
                updated_at = excluded.updated_at
            """,
            (
                paper_id,
                variant_id,
                variant_label,
                provider,
                base_url,
                model,
                summary_text,
                summary_basis,
                json_dumps(unique_strings(tags)),
                json_dumps(structured),
                json_dumps(usage),
                now,
                now,
            ),
        )
        self.connection.commit()

    def fetch_paper_llm_summaries(self, paper_ids: list[int] | None = None) -> dict[int, list[PaperLLMSummary]]:
        params: list[Any] = []
        sql = """
            SELECT paper_id, variant_id, variant_label, provider, base_url, model,
                   summary_text, summary_basis, tags_json, structured_json, usage_json,
                   created_at, updated_at
            FROM paper_llm_summaries
        """
        if paper_ids:
            placeholders = ", ".join(["?"] * len(paper_ids))
            sql += f" WHERE paper_id IN ({placeholders})"
            params.extend(paper_ids)
        sql += " ORDER BY variant_label ASC, model ASC, id ASC"
        rows = self.connection.execute(sql, params).fetchall()
        grouped: dict[int, list[PaperLLMSummary]] = {}
        for row in rows:
            grouped.setdefault(int(row["paper_id"]), []).append(
                PaperLLMSummary(
                    paper_id=int(row["paper_id"]),
                    variant_id=row["variant_id"],
                    variant_label=row["variant_label"],
                    provider=row["provider"] or "",
                    base_url=row["base_url"] or "",
                    model=row["model"] or "",
                    summary_text=row["summary_text"] or "",
                    summary_basis=row["summary_basis"] or "",
                    tags=safe_json_loads(row["tags_json"], []),
                    structured=safe_json_loads(row["structured_json"], {}),
                    usage=safe_json_loads(row["usage_json"], {}),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )
        return grouped

    def fetch_paper_llm_variant_ids(self, paper_id: int) -> set[str]:
        rows = self.connection.execute(
            "SELECT variant_id FROM paper_llm_summaries WHERE paper_id = ?",
            (paper_id,),
        ).fetchall()
        return {row["variant_id"] for row in rows}

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

    def fetch_paper_evaluations(self, paper_id: int) -> list[TopicEvaluation]:
        rows = self.connection.execute(
            """
            SELECT topic_id, topic_name, score, classification, matched_keywords_json, reasons_json
            FROM matches
            WHERE paper_id = ?
            ORDER BY score DESC, topic_name ASC
            """,
            (paper_id,),
        ).fetchall()
        return [
            TopicEvaluation(
                topic_id=row["topic_id"],
                topic_name=row["topic_name"],
                score=float(row["score"]),
                classification=row["classification"],
                matched_keywords=safe_json_loads(row["matched_keywords_json"], []),
                reasons=safe_json_loads(row["reasons_json"], []),
            )
            for row in rows
        ]

    def fetch_enrichment_candidates(
        self,
        limit: int,
        classifications: list[str],
        topic_ids: list[str] | None = None,
    ) -> list[PaperRecord]:
        if not classifications:
            return []
        placeholders = ", ".join(["?"] * len(classifications))
        params: list[Any] = list(classifications)
        topic_filter = ""
        if topic_ids:
            topic_placeholders = ", ".join(["?"] * len(topic_ids))
            topic_filter = f" AND m.topic_id IN ({topic_placeholders})"
            params.extend(topic_ids)
        params.append(limit)

        rows = self.connection.execute(
            f"""
            SELECT p.*, MAX(m.score) AS best_score, MIN(COALESCE(pls.variant_count, 0)) AS variant_count
            FROM papers p
            JOIN matches m ON m.paper_id = p.id
            LEFT JOIN (
                SELECT paper_id, COUNT(DISTINCT variant_id) AS variant_count
                FROM paper_llm_summaries
                GROUP BY paper_id
            ) pls ON pls.paper_id = p.id
            WHERE m.classification IN ({placeholders})
            {topic_filter}
            GROUP BY p.id
            ORDER BY variant_count ASC, best_score DESC, p.created_at DESC, p.id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._row_to_paper(row) for row in rows]

    def fetch_report_entries(self, start_at: str, end_at: str, include_maybe: bool) -> list[ReportEntry]:
        classifications = ("relevant", "maybe") if include_maybe else ("relevant",)
        placeholders = ", ".join(["?"] * len(classifications))
        start_day = start_at[:10]
        end_day = end_at[:10]
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
            WHERE (
                p.created_at BETWEEN ? AND ?
                OR substr(COALESCE(NULLIF(p.published_at, ''), p.created_at), 1, 10) BETWEEN ? AND ?
            )
              AND m.classification IN ({placeholders})
            GROUP BY m.id
            ORDER BY m.topic_name ASC, m.score DESC, p.published_at DESC, p.id DESC
            """,
            (start_at, end_at, start_day, end_day, *classifications),
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
        start_day = start_at[:10]
        end_day = end_at[:10]
        row = self.connection.execute(
            f"""
            SELECT COUNT(*) AS count
            FROM matches m
            JOIN papers p ON p.id = m.paper_id
            WHERE (
                p.created_at BETWEEN ? AND ?
                OR substr(COALESCE(NULLIF(p.published_at, ''), p.created_at), 1, 10) BETWEEN ? AND ?
            )
              AND m.classification IN ({placeholders})
            """,
            (start_at, end_at, start_day, end_day, *classifications),
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
