from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass, replace
from pathlib import Path

from paper_monitor.llm import LLMClient, looks_like_invalid_direct_pdf_summary
from paper_monitor.llm_registry import LLMRuntimeVariant
from paper_monitor.models import EnrichmentConfig, PaperRecord, RunStats, Settings
from paper_monitor.progress import ProgressBar
from paper_monitor.storage import Database
from paper_monitor.summarize import build_paper_summary
from paper_monitor.utils import (
    clean_extracted_text,
    command_exists,
    ensure_directory,
    now_iso,
    shorten,
    stable_hash,
)


LOGGER = logging.getLogger(__name__)

STREAM_RE = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.S)
TEXT_LITERAL_RE = re.compile(r"\(([^()]*)\)")
ARXIV_ID_RE = re.compile(r"(?:10\.48550/arxiv\.|arxiv\.org/(?:abs|pdf)/|abs/)(\d{4}\.\d{4,5}(?:v\d+)?)", re.I)


def _llm_route_label(llm_result) -> str:  # noqa: ANN001
    structured = llm_result.structured if isinstance(getattr(llm_result, "structured", None), dict) else {}
    source_mode = str(structured.get("source_mode", "")).strip().lower()
    pdf_strategy = str(
        structured.get("pdf_input_strategy") or structured.get("direct_pdf_strategy") or ""
    ).strip()
    direct_pdf_status = str(structured.get("direct_pdf_status", "")).strip().lower()
    if source_mode == "pdf_direct":
        return f"PDF/{pdf_strategy or 'direct'}"
    if source_mode == "fulltext_txt":
        if direct_pdf_status in {"unsupported", "request_failed", "disabled", "no_local_pdf", "too_large", "invalid_response"}:
            return f"全文回退/{direct_pdf_status}"
        return "全文"
    return "摘要"


def _summary_has_complete_pdf_output(summary) -> bool:  # noqa: ANN001
    if summary is None:
        return False
    structured = summary.structured if isinstance(getattr(summary, "structured", None), dict) else {}
    source_mode = str(structured.get("source_mode", "")).strip().lower()
    direct_pdf_status = str(structured.get("direct_pdf_status", "")).strip().lower()
    summary_text = str(getattr(summary, "summary_text", "") or "").strip()
    summary_basis = str(getattr(summary, "summary_basis", "") or "").strip().lower()
    structured_basis = str(structured.get("basis", "") or "").strip().lower()
    if not summary_text or source_mode != "pdf_direct" or direct_pdf_status not in {"", "used"}:
        return False
    if summary_basis != "llm+pdf+metadata":
        return False
    if structured_basis and structured_basis != "llm+pdf+metadata":
        return False
    return not looks_like_invalid_direct_pdf_summary(structured, summary_text)


def _normalize_variant_lookup_key(value: str) -> str:
    return str(value or "").strip().lower()


@dataclass(slots=True)
class DocumentArtifacts:
    pdf_local_path: str
    pdf_status: str
    pdf_downloaded_at: str | None
    fulltext_txt_path: str
    fulltext_excerpt: str
    fulltext_status: str
    page_count: int | None
    was_downloaded: bool
    was_extracted: bool


@dataclass(slots=True)
class PaperProcessingResult:
    paper: PaperRecord
    evaluations: list
    llm_results: list[tuple[LLMRuntimeVariant, object]]
    artifacts: DocumentArtifacts | None


class DocumentProcessor:
    def __init__(self, config: EnrichmentConfig, user_agent: str, timezone_name: str) -> None:
        self.config = config
        self.user_agent = user_agent
        self.timezone_name = timezone_name
        self.semantic_scholar_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        self.has_pdftotext = bool(config.use_pdftotext and command_exists("pdftotext"))
        self.has_pdfinfo = command_exists("pdfinfo")
        ensure_directory(config.pdf_dir)
        ensure_directory(config.text_dir)

    def can_try_pdf(self, paper: PaperRecord) -> bool:
        return bool(self.resolve_pdf_url(paper))

    def resolve_pdf_url(self, paper: PaperRecord) -> str:
        if self._looks_like_pdf_url(paper.pdf_url):
            return paper.pdf_url
        inferred_arxiv_id = self._infer_arxiv_id(paper)
        if inferred_arxiv_id:
            return f"https://arxiv.org/pdf/{inferred_arxiv_id}.pdf"
        if paper.arxiv_id:
            return f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"
        if "/abs/" in paper.primary_url and "arxiv.org" in paper.primary_url:
            return paper.primary_url.replace("/abs/", "/pdf/") + ".pdf"
        inferred_landing_pdf = self._infer_pdf_url_from_landing_page(paper.primary_url)
        if inferred_landing_pdf:
            return inferred_landing_pdf
        inferred_semantic_scholar_pdf = self._infer_pdf_url_from_semantic_scholar(paper)
        if inferred_semantic_scholar_pdf:
            return inferred_semantic_scholar_pdf
        if self._should_try_primary_url_as_pdf_candidate(paper):
            return paper.primary_url.strip()
        return ""

    def enrich(self, paper: PaperRecord, force: bool = False) -> DocumentArtifacts:
        pdf_url = self.resolve_pdf_url(paper)
        if not pdf_url:
            return DocumentArtifacts(
                pdf_local_path=paper.pdf_local_path,
                pdf_status="no-pdf",
                pdf_downloaded_at=paper.pdf_downloaded_at,
                fulltext_txt_path=paper.fulltext_txt_path,
                fulltext_excerpt=paper.fulltext_excerpt,
                fulltext_status=paper.fulltext_status or "empty",
                page_count=paper.page_count,
                was_downloaded=False,
                was_extracted=False,
            )

        pdf_path = self.config.pdf_dir / f"{self._paper_slug(paper)}.pdf"
        text_path = self.config.text_dir / f"{self._paper_slug(paper)}.txt"
        downloaded_at = paper.pdf_downloaded_at
        was_downloaded = False

        if force or self.config.redownload_existing or not pdf_path.exists():
            downloaded_at = now_iso(self.timezone_name)
            try:
                self._download_pdf(pdf_url, pdf_path)
                was_downloaded = True
            except (ValueError, urllib.error.URLError, OSError) as exc:
                LOGGER.warning("pdf candidate rejected for paper_id=%s url=%s: %s", paper.id, pdf_url, exc)
                return DocumentArtifacts(
                    pdf_local_path=paper.pdf_local_path,
                    pdf_status="no-pdf",
                    pdf_downloaded_at=paper.pdf_downloaded_at,
                    fulltext_txt_path=paper.fulltext_txt_path,
                    fulltext_excerpt=paper.fulltext_excerpt,
                    fulltext_status=paper.fulltext_status or "empty",
                    page_count=paper.page_count,
                    was_downloaded=False,
                    was_extracted=False,
                )

        page_count = self._extract_page_count(pdf_path) if pdf_path.exists() else paper.page_count

        fulltext = ""
        was_extracted = False
        if force or not text_path.exists() or self.config.redownload_existing:
            fulltext = self._extract_text(pdf_path)
            if fulltext:
                text_path.write_text(fulltext + "\n", encoding="utf-8")
                was_extracted = True
        elif text_path.exists():
            fulltext = text_path.read_text(encoding="utf-8", errors="replace")

        fulltext = clean_extracted_text(fulltext)
        return DocumentArtifacts(
            pdf_local_path=str(pdf_path) if pdf_path.exists() else "",
            pdf_status="downloaded" if pdf_path.exists() else "failed",
            pdf_downloaded_at=downloaded_at,
            fulltext_txt_path=str(text_path) if text_path.exists() else "",
            fulltext_excerpt=shorten(fulltext, self.config.excerpt_chars) if fulltext else "",
            fulltext_status="extracted" if fulltext else "failed",
            page_count=page_count,
            was_downloaded=was_downloaded,
            was_extracted=was_extracted,
        )

    def _paper_slug(self, paper: PaperRecord) -> str:
        if paper.arxiv_id:
            return paper.arxiv_id.replace("/", "_")
        if paper.doi:
            return stable_hash(paper.doi)
        return f"paper-{paper.id}-{stable_hash(paper.title)[:10]}"

    def _looks_like_pdf_url(self, url: str) -> bool:
        lowered = url.lower().strip()
        if not lowered:
            return False
        if lowered.endswith(".pdf"):
            return True
        return "/pdf/" in lowered

    def _infer_arxiv_id(self, paper: PaperRecord) -> str:
        candidates = [
            paper.arxiv_id,
            paper.pdf_url,
            paper.primary_url,
            paper.doi,
        ]
        if isinstance(paper.metadata, dict) and paper.metadata:
            candidates.append(str(paper.metadata))
        for candidate in candidates:
            text = str(candidate or "").strip()
            if not text:
                continue
            match = ARXIV_ID_RE.search(text)
            if match:
                return match.group(1)
        return ""

    def _should_try_primary_url_as_pdf_candidate(self, paper: PaperRecord) -> bool:
        url = paper.primary_url.strip()
        if not url:
            return False
        lowered = url.lower()
        if self._looks_like_pdf_url(lowered):
            return True
        if "doi.org/" in lowered and "10.48550/arxiv." not in lowered:
            return False
        metadata_text = str(paper.metadata or "").lower()
        if "'access': 'open'" in metadata_text or '"access": "open"' in metadata_text:
            return True
        return not lowered.startswith("https://doi.org/")

    def _download_pdf(self, url: str, destination: Path) -> None:
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(request, timeout=self.config.download_timeout_seconds) as response:
            payload = response.read()
            content_type = (response.headers.get("Content-Type") or "").lower()
            if not payload.startswith(b"%PDF") and "application/pdf" not in content_type:
                raise ValueError(f"candidate url did not return a pdf document: {url}")
            destination.write_bytes(payload)

    def _infer_pdf_url_from_landing_page(self, url: str) -> str:
        target = url.strip()
        if not target:
            return ""
        lowered = target.lower()
        if self._looks_like_pdf_url(lowered):
            return target
        request = urllib.request.Request(target, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=self.config.download_timeout_seconds) as response:
                content_type = (response.headers.get("Content-Type") or "").lower()
                if "html" not in content_type:
                    return ""
                final_url = response.geturl() or target
                body = response.read(512_000).decode("utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            return ""
        candidates = [
            r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+name=["\']dc\.identifier["\'][^>]+content=["\']([^"\']+\.pdf[^"\']*)["\']',
            r'<meta[^>]+property=["\']og:url["\'][^>]+content=["\']([^"\']+\.pdf[^"\']*)["\']',
            r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']',
        ]
        for pattern in candidates:
            for match in re.finditer(pattern, body, re.I):
                candidate = urllib.parse.urljoin(final_url, match.group(1).strip())
                if self._looks_like_pdf_url(candidate):
                    return candidate
        return ""

    def _infer_pdf_url_from_semantic_scholar(self, paper: PaperRecord) -> str:
        fields = "title,openAccessPdf,url,externalIds"
        for identifier in self._semantic_scholar_identifiers(paper):
            payload = self._fetch_semantic_scholar_details(identifier, fields)
            candidate = self._semantic_scholar_open_access_pdf(payload)
            if candidate:
                LOGGER.info(
                    "resolved pdf via Semantic Scholar details for paper_id=%s using %s",
                    paper.id,
                    identifier,
                )
                return candidate
        if not self.semantic_scholar_api_key:
            return ""
        payload = self._search_semantic_scholar_by_title(paper.title, fields)
        candidate = self._semantic_scholar_open_access_pdf(payload)
        if candidate:
            LOGGER.info(
                "resolved pdf via Semantic Scholar title search for paper_id=%s",
                paper.id,
            )
            return candidate
        return ""

    def _semantic_scholar_identifiers(self, paper: PaperRecord) -> list[str]:
        identifiers: list[str] = []
        if paper.doi:
            identifiers.append(f"DOI:{paper.doi.strip()}")
        inferred_arxiv_id = self._infer_arxiv_id(paper)
        if inferred_arxiv_id:
            identifiers.append(f"ARXIV:{re.sub(r'v\\d+$', '', inferred_arxiv_id, flags=re.I)}")
        elif paper.arxiv_id:
            identifiers.append(f"ARXIV:{re.sub(r'v\\d+$', '', paper.arxiv_id.strip(), flags=re.I)}")
        return identifiers

    def _semantic_scholar_headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }
        if self.semantic_scholar_api_key:
            headers["x-api-key"] = self.semantic_scholar_api_key
        return headers

    def _fetch_semantic_scholar_details(self, identifier: str, fields: str) -> dict:
        encoded_identifier = urllib.parse.quote(identifier, safe="")
        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/{encoded_identifier}"
            f"?fields={urllib.parse.quote(fields, safe=',')}"
        )
        for attempt in range(3):
            request = urllib.request.Request(url, headers=self._semantic_scholar_headers())
            try:
                with urllib.request.urlopen(request, timeout=self.config.download_timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8", errors="replace"))
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return {}
                if exc.code == 429 and attempt < 2:
                    time.sleep(1.0 + attempt)
                    continue
                LOGGER.info(
                    "Semantic Scholar details lookup failed for %s: http_%s",
                    identifier,
                    exc.code,
                )
                return {}
            except Exception as exc:  # noqa: BLE001
                LOGGER.info("Semantic Scholar details lookup failed for %s: %s", identifier, exc)
                return {}
        return {}

    def _search_semantic_scholar_by_title(self, title: str, fields: str) -> dict:
        query = urllib.parse.quote(title.strip())
        url = (
            "https://api.semanticscholar.org/graph/v1/paper/search"
            f"?query={query}&limit=1&fields={urllib.parse.quote(fields, safe=',')}"
        )
        payload = {}
        for attempt in range(3):
            request = urllib.request.Request(url, headers=self._semantic_scholar_headers())
            try:
                with urllib.request.urlopen(request, timeout=self.config.download_timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8", errors="replace"))
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < 2:
                    time.sleep(1.0 + attempt)
                    continue
                LOGGER.info("Semantic Scholar title search failed for %r: http_%s", title, exc.code)
                return {}
            except Exception as exc:  # noqa: BLE001
                LOGGER.info("Semantic Scholar title search failed for %r: %s", title, exc)
                return {}
        data = payload.get("data")
        if isinstance(data, list) and data:
            return data[0] if isinstance(data[0], dict) else {}
        return {}

    def _semantic_scholar_open_access_pdf(self, payload: dict) -> str:
        if not isinstance(payload, dict):
            return ""
        candidate = str((payload.get("openAccessPdf") or {}).get("url") or "").strip()
        if candidate and candidate.lower().startswith(("http://", "https://")):
            return candidate
        return ""

    def _extract_page_count(self, pdf_path: Path) -> int | None:
        if not self.has_pdfinfo:
            return None
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            if line.startswith("Pages:"):
                value = line.split(":", 1)[1].strip()
                if value.isdigit():
                    return int(value)
        return None

    def _extract_text(self, pdf_path: Path) -> str:
        if self.has_pdftotext:
            result = subprocess.run(
                ["pdftotext", "-layout", "-nopgbrk", str(pdf_path), "-"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                return clean_extracted_text(result.stdout)
        return self._fallback_extract_text(pdf_path)

    def _fallback_extract_text(self, pdf_path: Path) -> str:
        data = pdf_path.read_bytes()
        chunks: list[str] = []
        for stream in STREAM_RE.findall(data):
            decoded = stream
            for _ in range(2):
                try:
                    decoded = zlib.decompress(decoded)
                    break
                except zlib.error:
                    pass
            text = decoded.decode("latin-1", errors="ignore")
            literals = TEXT_LITERAL_RE.findall(text)
            if literals:
                chunks.extend(literals)
        return clean_extracted_text(" ".join(chunks))


class EnrichmentPipeline:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        document_processor: DocumentProcessor | None = None,
        llm_client: LLMClient | None = None,
        llm_variants: list[LLMRuntimeVariant] | None = None,
    ) -> None:
        self.settings = settings
        self.db = db
        user_agent = settings.arxiv.user_agent or settings.dblp.user_agent
        self.document_processor = document_processor or DocumentProcessor(
            settings.enrichment,
            user_agent=user_agent,
            timezone_name=settings.timezone,
        )
        if llm_variants is not None:
            self.llm_variants = llm_variants
        else:
            default_client = llm_client or LLMClient(settings.llm)
            self.llm_variants = [
                LLMRuntimeVariant(
                    variant_id=settings.llm.variant_id,
                    label=settings.llm.label or settings.llm.model or settings.llm.variant_id,
                    provider=settings.llm.provider,
                    base_url=settings.llm.base_url,
                    model=settings.llm.model,
                    config_path=settings.config_path,
                    client=default_client,
                )
            ]

    def run(
        self,
        *,
        limit: int | None = None,
        topic_ids: list[str] | None = None,
        classifications: list[str] | None = None,
        created_after: str | None = None,
        paper_ids: list[int] | None = None,
        force: bool = False,
        use_llm: bool | None = None,
        skip_document_processing: bool = False,
        workers: int = 1,
        secondary_priority_only: bool = False,
        secondary_top_per_topic: int = 3,
        secondary_min_score: float = 24.0,
        retry_from_variant: str | None = None,
        retry_from_status: str = "incomplete",
    ) -> RunStats:
        stats = RunStats()
        if not self.settings.enrichment.enabled and not force:
            return stats

        selected_classifications = classifications or self.settings.enrichment.process_classifications
        effective_limit = limit or self.settings.enrichment.max_papers_per_run
        if paper_ids:
            effective_limit = max(effective_limit, len(set(paper_ids)))
        fetch_limit = effective_limit
        if retry_from_variant and not paper_ids:
            fetch_limit = max(effective_limit * 10, 2000)
        candidates = self.db.fetch_enrichment_candidates(
            limit=fetch_limit,
            classifications=selected_classifications,
            topic_ids=topic_ids,
            created_after=created_after,
            paper_ids=paper_ids,
        )
        if retry_from_variant:
            candidates = self._filter_candidates_by_reference_variant(
                candidates,
                reference_variant=retry_from_variant,
                retry_status=retry_from_status,
            )
            candidates = candidates[:effective_limit]
        progress_bar = ProgressBar("增强", len(candidates) or 1)
        enabled_variants = [variant for variant in self.llm_variants if variant.client.enabled]
        use_llm_effective = bool(enabled_variants) if use_llm is None else (use_llm and bool(enabled_variants))
        worker_count = max(int(workers or 1), 1)
        target_variants_by_paper = self._build_target_variants_map(
            candidates,
            use_llm=use_llm_effective,
            secondary_priority_only=secondary_priority_only,
            secondary_top_per_topic=secondary_top_per_topic,
            secondary_min_score=secondary_min_score,
        )

        if worker_count > 1 and len(candidates) > 1:
            return self._run_concurrent(
                candidates,
                stats,
                force=force,
                use_llm=use_llm_effective,
                skip_document_processing=skip_document_processing,
                progress_bar=progress_bar,
                workers=worker_count,
                target_variants_by_paper=target_variants_by_paper,
            )

        for paper in candidates:
            title = shorten(paper.title, 56)
            if self._should_skip(
                paper,
                force=force,
                use_llm=use_llm_effective,
                skip_document_processing=skip_document_processing,
                target_variants=target_variants_by_paper.get(paper.id),
            ):
                stats.skipped += 1
                progress_bar.advance(detail=f"跳过 {title}")
                continue
            try:
                progress_bar.set_detail(f"处理中 {title}")
                self._enrich_paper(
                    paper,
                    stats,
                    force=force,
                    use_llm=use_llm_effective,
                    skip_document_processing=skip_document_processing,
                    progress_bar=progress_bar,
                    target_variants=target_variants_by_paper.get(paper.id),
                )
                progress_bar.advance(detail=f"完成 {title}")
            except (OSError, subprocess.SubprocessError, urllib.error.URLError) as exc:
                LOGGER.warning("enrichment failed for paper_id=%s: %s", paper.id, exc)
                stats.errors.append(f"paper:{paper.id}:{exc}")
                progress_bar.advance(detail=f"失败 {title}")
        progress_bar.close(
            f"增强完成 {stats.enriched} 篇, LLM 总结 {stats.llm_summaries} 条, 跳过 {stats.skipped} 篇"
        )
        return stats

    def _run_concurrent(
        self,
        candidates: list[PaperRecord],
        stats: RunStats,
        *,
        force: bool,
        use_llm: bool,
        skip_document_processing: bool,
        progress_bar: ProgressBar,
        workers: int,
        target_variants_by_paper: dict[int, list[LLMRuntimeVariant]],
    ) -> RunStats:
        jobs = []
        for paper in candidates:
            title = shorten(paper.title, 56)
            target_variants = target_variants_by_paper.get(paper.id, [])
            if self._should_skip(
                paper,
                force=force,
                use_llm=use_llm,
                skip_document_processing=skip_document_processing,
                target_variants=target_variants,
            ):
                stats.skipped += 1
                progress_bar.advance(detail=f"跳过 {title}")
                continue
            evaluations = self.db.fetch_paper_evaluations(paper.id)
            active_variants = (
                self._select_variants_needing_refresh(paper.id, target_variants, force=force) if use_llm else []
            )
            jobs.append((paper, evaluations, active_variants, title))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    self._process_paper_job,
                    paper,
                    evaluations,
                    active_variants,
                    force=force,
                    skip_document_processing=skip_document_processing,
                    use_llm=use_llm,
                ): (
                    paper,
                    title,
                )
                for paper, evaluations, active_variants, title in jobs
            }
            progress_bar.start_pulse(f"并发处理中 {len(future_map)} 篇论文")
            for future in as_completed(future_map):
                paper, title = future_map[future]
                try:
                    result = future.result()
                    if result.artifacts is not None:
                        self.db.update_paper_assets(
                            paper.id,
                            pdf_local_path=result.artifacts.pdf_local_path,
                            pdf_status=result.artifacts.pdf_status,
                            pdf_downloaded_at=result.artifacts.pdf_downloaded_at,
                            fulltext_txt_path=result.artifacts.fulltext_txt_path,
                            fulltext_excerpt=result.artifacts.fulltext_excerpt,
                            fulltext_status=result.artifacts.fulltext_status,
                            page_count=result.artifacts.page_count,
                        )
                        if result.artifacts.was_downloaded:
                            stats.downloaded_pdfs += 1
                        if result.artifacts.was_extracted:
                            stats.extracted_texts += 1
                    self._persist_paper_results(
                        result.paper,
                        result.evaluations,
                        result.llm_results,
                        stats,
                        use_llm=use_llm,
                    )
                    progress_bar.advance(detail=f"完成 {title}")
                except (OSError, subprocess.SubprocessError, urllib.error.URLError) as exc:
                    LOGGER.warning("enrichment failed for paper_id=%s: %s", paper.id, exc)
                    stats.errors.append(f"paper:{paper.id}:{exc}")
                    progress_bar.advance(detail=f"失败 {title}")
            progress_bar.stop_pulse()

        progress_bar.close(
            f"增强完成 {stats.enriched} 篇, LLM 总结 {stats.llm_summaries} 条, 跳过 {stats.skipped} 篇"
        )
        return stats

    def _generate_llm_results(
        self,
        paper: PaperRecord,
        evaluations,
        target_variants: list[LLMRuntimeVariant],
    ) -> list[tuple[LLMRuntimeVariant, object]]:
        llm_results: list[tuple[LLMRuntimeVariant, object]] = []
        for variant in target_variants:
            llm_result = variant.client.generate_summary(paper, evaluations)
            if llm_result is None:
                continue
            llm_results.append((variant, llm_result))
        return llm_results

    def _select_variants_needing_refresh(
        self,
        paper_id: int,
        target_variants: list[LLMRuntimeVariant],
        *,
        force: bool,
    ) -> list[LLMRuntimeVariant]:
        if force:
            return [variant for variant in target_variants if variant.client.enabled]
        summary_map = self.db.fetch_paper_llm_summary_map(paper_id)
        selected: list[LLMRuntimeVariant] = []
        for variant in target_variants:
            if not variant.client.enabled:
                continue
            summary = summary_map.get(variant.variant_id)
            if _summary_has_complete_pdf_output(summary):
                continue
            selected.append(variant)
        return selected

    def _process_paper_job(
        self,
        paper: PaperRecord,
        evaluations,
        target_variants: list[LLMRuntimeVariant],
        *,
        force: bool,
        skip_document_processing: bool,
        use_llm: bool,
    ) -> PaperProcessingResult:
        updated_paper = paper
        artifacts: DocumentArtifacts | None = None
        needs_document_processing = (
            not skip_document_processing
            and (force or paper.fulltext_status != "extracted" or paper.pdf_status == "pending")
        )
        if needs_document_processing:
            artifacts = self.document_processor.enrich(paper, force=force)
            updated_paper = replace(
                paper,
                pdf_local_path=artifacts.pdf_local_path,
                pdf_status=artifacts.pdf_status,
                pdf_downloaded_at=artifacts.pdf_downloaded_at,
                fulltext_txt_path=artifacts.fulltext_txt_path,
                fulltext_excerpt=artifacts.fulltext_excerpt,
                fulltext_status=artifacts.fulltext_status,
                page_count=artifacts.page_count,
            )

        llm_results: list[tuple[LLMRuntimeVariant, object]] = []
        if use_llm:
            llm_results = self._generate_llm_results(updated_paper, evaluations, target_variants)
        return PaperProcessingResult(
            paper=updated_paper,
            evaluations=evaluations,
            llm_results=llm_results,
            artifacts=artifacts,
        )

    def _filter_candidates_by_reference_variant(
        self,
        candidates: list[PaperRecord],
        *,
        reference_variant: str,
        retry_status: str,
    ) -> list[PaperRecord]:
        if not candidates:
            return candidates
        normalized_reference = _normalize_variant_lookup_key(reference_variant)
        if not normalized_reference:
            return candidates
        summary_map = self.db.fetch_paper_llm_summaries([paper.id for paper in candidates])
        filtered: list[PaperRecord] = []
        for paper in candidates:
            reference_summary = self._find_reference_variant_summary(
                summary_map.get(paper.id, []),
                normalized_reference,
            )
            if self._matches_retry_reference_status(reference_summary, retry_status):
                filtered.append(paper)
        LOGGER.info(
            "retry-from-variant filter applied: variant=%s status=%s kept=%s/%s",
            reference_variant,
            retry_status,
            len(filtered),
            len(candidates),
        )
        return filtered

    def _find_reference_variant_summary(
        self,
        summaries,
        normalized_reference: str,
    ):
        for summary in summaries:
            if _normalize_variant_lookup_key(summary.variant_id) == normalized_reference:
                return summary
            if _normalize_variant_lookup_key(summary.variant_label) == normalized_reference:
                return summary
        return None

    def _matches_retry_reference_status(self, summary, retry_status: str) -> bool:  # noqa: ANN001
        normalized_status = str(retry_status or "incomplete").strip().lower()
        if normalized_status == "missing":
            return summary is None
        if normalized_status == "fallback":
            return summary is not None and not _summary_has_complete_pdf_output(summary)
        return summary is None or not _summary_has_complete_pdf_output(summary)

    def _persist_paper_results(
        self,
        paper: PaperRecord,
        evaluations,
        llm_results: list[tuple[LLMRuntimeVariant, object]],
        stats: RunStats,
        *,
        use_llm: bool,
    ) -> None:
        primary_llm_result = None
        for variant, llm_result in llm_results:
            usage = {}
            if isinstance(llm_result.structured, dict):
                usage = llm_result.structured.get("usage", {})
            self.db.upsert_paper_llm_summary(
                paper.id,
                variant_id=variant.variant_id,
                variant_label=variant.label,
                provider=variant.provider,
                base_url=variant.base_url,
                model=variant.model,
                summary_text=llm_result.summary_text,
                summary_basis=llm_result.summary_basis,
                tags=llm_result.tags,
                structured=llm_result.structured,
                usage=usage if isinstance(usage, dict) else {},
            )
            if primary_llm_result is None:
                primary_llm_result = (variant, llm_result)
            stats.llm_summaries += 1

        if primary_llm_result is not None:
            primary_variant, llm_result = primary_llm_result
            legacy_structured = dict(llm_result.structured)
            legacy_structured.update(
                {
                    "variant_id": primary_variant.variant_id,
                    "variant_label": primary_variant.label,
                    "model": primary_variant.model,
                }
            )
            self.db.update_paper_analysis(
                paper.id,
                llm_result.summary_text,
                llm_result.summary_basis,
                llm_result.tags,
                llm_summary=legacy_structured,
            )
        elif use_llm and paper.summary_basis.startswith("llm+") and paper.summary_text:
            self.db.update_paper_analysis(
                paper.id,
                paper.summary_text,
                paper.summary_basis,
                paper.tags,
                llm_summary=paper.llm_summary,
            )
        else:
            summary_text, basis, tags = build_paper_summary(paper, evaluations)
            self.db.update_paper_analysis(paper.id, summary_text, basis, tags)

        stats.enriched += 1

    def _should_skip(
        self,
        paper: PaperRecord,
        *,
        force: bool,
        use_llm: bool,
        skip_document_processing: bool = False,
        target_variants: list[LLMRuntimeVariant] | None = None,
    ) -> bool:
        if force:
            return False
        needs_pdf = (
            not skip_document_processing
            and self.document_processor.can_try_pdf(paper)
            and paper.fulltext_status != "extracted"
        )
        needs_pdf_state_refresh = not skip_document_processing and paper.pdf_status == "pending"
        effective_variants = target_variants if target_variants is not None else self.llm_variants
        refresh_variants = (
            self._select_variants_needing_refresh(paper.id, effective_variants, force=False)
            if use_llm
            else []
        )
        needs_llm = use_llm and bool(refresh_variants)
        needs_offline_refresh = (
            not use_llm
            and paper.fulltext_status == "extracted"
            and paper.summary_basis not in {"fulltext+metadata", "llm+fulltext+metadata"}
        )
        needs_initial_summary = not paper.summary_text
        return not (needs_pdf or needs_pdf_state_refresh or needs_llm or needs_offline_refresh or needs_initial_summary)

    def _enrich_paper(
        self,
        paper: PaperRecord,
        stats: RunStats,
        *,
        force: bool,
        use_llm: bool,
        skip_document_processing: bool,
        progress_bar: ProgressBar | None = None,
        target_variants: list[LLMRuntimeVariant] | None = None,
    ) -> None:
        title = shorten(paper.title, 56)
        if not skip_document_processing and (force or paper.fulltext_status != "extracted" or paper.pdf_status == "pending"):
            if progress_bar:
                progress_bar.set_detail(f"下载/抽取 {title}")
                progress_bar.start_pulse(f"下载/抽取 {title}")
            try:
                artifacts = self.document_processor.enrich(paper, force=force)
            finally:
                if progress_bar:
                    progress_bar.stop_pulse(f"完成下载/抽取 {title}")
            self.db.update_paper_assets(
                paper.id,
                pdf_local_path=artifacts.pdf_local_path,
                pdf_status=artifacts.pdf_status,
                pdf_downloaded_at=artifacts.pdf_downloaded_at,
                fulltext_txt_path=artifacts.fulltext_txt_path,
                fulltext_excerpt=artifacts.fulltext_excerpt,
                fulltext_status=artifacts.fulltext_status,
                page_count=artifacts.page_count,
            )
            if artifacts.was_downloaded:
                stats.downloaded_pdfs += 1
            if artifacts.was_extracted:
                stats.extracted_texts += 1
            paper = self.db.get_paper(paper.id)
        elif skip_document_processing and progress_bar:
            progress_bar.set_detail(f"跳过 PDF，直接总结 {title}")

        evaluations = self.db.fetch_paper_evaluations(paper.id)
        llm_results = []
        if use_llm:
            candidate_variants = target_variants if target_variants is not None else self.llm_variants
            active_variants = self._select_variants_needing_refresh(paper.id, candidate_variants, force=force)
            for variant in active_variants:
                if progress_bar:
                    progress_bar.set_detail(f"{title} -> {variant.label}")
                    progress_bar.start_pulse(f"{title} -> {variant.label}")
                try:
                    llm_result = variant.client.generate_summary(paper, evaluations)
                finally:
                    if progress_bar:
                        progress_bar.stop_pulse(f"{title} -> {variant.label}")
                if llm_result is None:
                    if progress_bar:
                        progress_bar.set_detail(f"{title} -> {variant.label} [未生成]")
                    continue
                if progress_bar:
                    progress_bar.set_detail(f"{title} -> {variant.label} [{_llm_route_label(llm_result)}]")
                llm_results.append((variant, llm_result))

        if not llm_results and progress_bar:
            progress_bar.set_detail(f"离线总结 {title}")
        self._persist_paper_results(
            paper,
            evaluations,
            llm_results,
            stats,
            use_llm=use_llm,
        )

    def _build_target_variants_map(
        self,
        candidates: list[PaperRecord],
        *,
        use_llm: bool,
        secondary_priority_only: bool,
        secondary_top_per_topic: int,
        secondary_min_score: float,
    ) -> dict[int, list[LLMRuntimeVariant]]:
        enabled_variants = [variant for variant in self.llm_variants if variant.client.enabled]
        if not use_llm or not enabled_variants:
            return {paper.id: [] for paper in candidates}
        if not secondary_priority_only or len(enabled_variants) <= 1:
            return {paper.id: list(enabled_variants) for paper in candidates}

        primary_variant = enabled_variants[0]
        secondary_variants = enabled_variants[1:]
        priority_paper_ids = self._select_priority_paper_ids(
            candidates,
            top_per_topic=secondary_top_per_topic,
            min_score=secondary_min_score,
        )
        mapping: dict[int, list[LLMRuntimeVariant]] = {}
        for paper in candidates:
            target_variants = [primary_variant]
            if paper.id in priority_paper_ids:
                target_variants.extend(secondary_variants)
            mapping[paper.id] = target_variants
        return mapping

    def _select_priority_paper_ids(
        self,
        candidates: list[PaperRecord],
        *,
        top_per_topic: int,
        min_score: float,
    ) -> set[int]:
        top_limit = max(int(top_per_topic or 0), 0)
        selected: set[int] = set()
        by_topic: dict[str, list[tuple[float, str, int]]] = {}
        for paper in candidates:
            evaluations = self.db.fetch_paper_evaluations(paper.id)
            relevant_evaluations = [item for item in evaluations if item.classification == "relevant"]
            if not relevant_evaluations:
                continue
            best_score = max(item.score for item in relevant_evaluations)
            if best_score >= float(min_score):
                selected.add(paper.id)
            for evaluation in relevant_evaluations:
                timestamp = paper.published_at or paper.updated_at or paper.created_at
                by_topic.setdefault(evaluation.topic_id, []).append((evaluation.score, timestamp, paper.id))

        if top_limit > 0:
            for entries in by_topic.values():
                entries.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)
                for _, _, paper_id in entries[:top_limit]:
                    selected.add(paper_id)
        return selected
