from __future__ import annotations

import logging
import time

from paper_monitor.enrichment import EnrichmentPipeline
from paper_monitor.llm_registry import LLMRuntimeVariant
from paper_monitor.models import Settings
from paper_monitor.pipeline import MonitorPipeline
from paper_monitor.reports import generate_report
from paper_monitor.storage import Database
from paper_monitor.utils import today_string


LOGGER = logging.getLogger(__name__)


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
) -> None:
    loops = 0
    runtime_variants = llm_variants or []
    enrichment_pipeline = EnrichmentPipeline(settings, db, llm_variants=runtime_variants)
    while True:
        loops += 1
        LOGGER.info("starting fetch cycle %s", loops)
        stats = pipeline.run_fetch()
        enrichment_stats = None
        if enrich:
            enrichment_stats = enrichment_pipeline.run(
                use_llm=use_llm,
                skip_document_processing=skip_document_processing,
                workers=workers,
            )
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
