from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import re
import subprocess
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path

from paper_monitor.llm import LLMClient
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
        if direct_pdf_status in {"unsupported", "request_failed", "disabled", "no_local_pdf"}:
            return f"全文回退/{direct_pdf_status}"
        return "全文"
    return "摘要"


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


class DocumentProcessor:
    def __init__(self, config: EnrichmentConfig, user_agent: str, timezone_name: str) -> None:
        self.config = config
        self.user_agent = user_agent
        self.timezone_name = timezone_name
        self.has_pdftotext = bool(config.use_pdftotext and command_exists("pdftotext"))
        self.has_pdfinfo = command_exists("pdfinfo")
        ensure_directory(config.pdf_dir)
        ensure_directory(config.text_dir)

    def can_try_pdf(self, paper: PaperRecord) -> bool:
        return bool(self.resolve_pdf_url(paper))

    def resolve_pdf_url(self, paper: PaperRecord) -> str:
        if self._looks_like_pdf_url(paper.pdf_url):
            return paper.pdf_url
        if paper.arxiv_id:
            return f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"
        if "/abs/" in paper.primary_url and "arxiv.org" in paper.primary_url:
            return paper.primary_url.replace("/abs/", "/pdf/") + ".pdf"
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
            self._download_pdf(pdf_url, pdf_path)
            was_downloaded = True

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

    def _download_pdf(self, url: str, destination: Path) -> None:
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(request, timeout=self.config.download_timeout_seconds) as response:
            destination.write_bytes(response.read())

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
    ) -> RunStats:
        stats = RunStats()
        if not self.settings.enrichment.enabled and not force:
            return stats

        selected_classifications = classifications or self.settings.enrichment.process_classifications
        effective_limit = limit or self.settings.enrichment.max_papers_per_run
        if paper_ids:
            effective_limit = max(effective_limit, len(set(paper_ids)))
        candidates = self.db.fetch_enrichment_candidates(
            limit=effective_limit,
            classifications=selected_classifications,
            topic_ids=topic_ids,
            created_after=created_after,
            paper_ids=paper_ids,
        )
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

        if skip_document_processing and use_llm_effective and worker_count > 1:
            return self._run_llm_only_concurrent(
                candidates,
                stats,
                force=force,
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

    def _run_llm_only_concurrent(
        self,
        candidates: list[PaperRecord],
        stats: RunStats,
        *,
        force: bool,
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
                use_llm=True,
                skip_document_processing=True,
                target_variants=target_variants,
            ):
                stats.skipped += 1
                progress_bar.advance(detail=f"跳过 {title}")
                continue
            evaluations = self.db.fetch_paper_evaluations(paper.id)
            existing_variant_ids = self.db.fetch_paper_llm_variant_ids(paper.id) if not force else set()
            active_variants = [
                variant
                for variant in target_variants
                if variant.client.enabled and (force or variant.variant_id not in existing_variant_ids)
            ]
            jobs.append((paper, evaluations, active_variants))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(self._generate_llm_results, paper, evaluations, target_variants): (
                    paper,
                    evaluations,
                )
                for paper, evaluations, target_variants in jobs
            }
            for future in as_completed(future_map):
                paper, evaluations = future_map[future]
                title = shorten(paper.title, 56)
                try:
                    llm_results = future.result()
                    self._persist_paper_results(
                        paper,
                        evaluations,
                        llm_results,
                        stats,
                        use_llm=True,
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
        existing_variant_ids = self.db.fetch_paper_llm_variant_ids(paper.id) if use_llm else set()
        effective_variants = target_variants if target_variants is not None else self.llm_variants
        target_variant_ids = {variant.variant_id for variant in effective_variants if variant.client.enabled}
        needs_llm = use_llm and bool(target_variant_ids - existing_variant_ids)
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
            artifacts = self.document_processor.enrich(paper, force=force)
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
        existing_variant_ids = self.db.fetch_paper_llm_variant_ids(paper.id) if use_llm and not force else set()
        llm_results = []
        if use_llm:
            candidate_variants = target_variants if target_variants is not None else self.llm_variants
            for variant in candidate_variants:
                if not variant.client.enabled:
                    continue
                if not force and variant.variant_id in existing_variant_ids:
                    continue
                if progress_bar:
                    progress_bar.set_detail(f"{title} -> {variant.label}")
                llm_result = variant.client.generate_summary(paper, evaluations)
                if llm_result is None:
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
