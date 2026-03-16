from __future__ import annotations

import logging
import time

from paper_monitor.enrichment import EnrichmentPipeline
from paper_monitor.llm_registry import LLMRuntimeVariant
from paper_monitor.models import Settings
from paper_monitor.pipeline import MonitorPipeline
from paper_monitor.reports import generate_report
from paper_monitor.storage import Database
from paper_monitor.utils import now_iso, today_string


LOGGER = logging.getLogger(__name__)
ENRICH_CHECKPOINT_KEY = "enrich:since"


def run_daemon(
    settings: Settings,
    db: Database,
    pipeline: MonitorPipeline,
    report_type: str,
    loop_limit: int | None = None,
    enrich: bool = False,
    use_llm: bool | None = None,
    llm_variants: list[LLMRuntimeVariant] | None = None,
    skip_document_processing: bool = False,
    workers: int = 1,
    since_last_run: bool = False,
    secondary_priority_only: bool = False,
    secondary_top_per_topic: int = 3,
    secondary_min_score: float = 24.0,
) -> None:
    loops = 0
    runtime_variants = llm_variants or []
    enrichment_pipeline = EnrichmentPipeline(settings, db, llm_variants=runtime_variants)
    if enrich and skip_document_processing and use_llm:
        LOGGER.warning("daemon 使用了 --skip-pdf + --with-llm，LLM 将无法读取完整 PDF 文本，只会基于标题/摘要生成总结。")
    while True:
        loops += 1
        LOGGER.info("starting fetch cycle %s", loops)
        stats = pipeline.run_fetch(since_last_run=since_last_run)
        enrichment_stats = None
        if enrich:
            enrich_started_at = now_iso(settings.timezone) if since_last_run else None
            enrichment_stats = enrichment_pipeline.run(
                paper_ids=stats.new_paper_ids if since_last_run else None,
                use_llm=use_llm,
                skip_document_processing=skip_document_processing,
                workers=workers,
                secondary_priority_only=secondary_priority_only,
                secondary_top_per_topic=secondary_top_per_topic,
                secondary_min_score=secondary_min_score,
            )
            if since_last_run and enrich_started_at is not None and not enrichment_stats.errors:
                db.set_checkpoint(ENRICH_CHECKPOINT_KEY, enrich_started_at)
        report_date = today_string(settings.timezone)
        paths = generate_report(
            db,
            settings,
            report_date=report_date,
            report_type=report_type,
            llm_variants=runtime_variants,
            use_llm_topic_digest=bool(use_llm),
        )
        LOGGER.info(
            "cycle %s finished, fetched=%s processed=%s enriched=%s report=%s",
            loops,
            stats.fetched,
            stats.processed,
            enrichment_stats.enriched if enrichment_stats else 0,
            paths,
        )

        if loop_limit and loops >= loop_limit:
            return

        sleep_seconds = max(settings.poll_minutes, 1) * 60
        LOGGER.info("sleeping for %s seconds", sleep_seconds)
        time.sleep(sleep_seconds)
