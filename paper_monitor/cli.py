from __future__ import annotations

from pathlib import Path
import argparse
import logging

from paper_monitor.config import load_settings, write_default_config
from paper_monitor.enrichment import EnrichmentPipeline
from paper_monitor.llm import LLMClient
from paper_monitor.llm_registry import build_runtime_variants
from paper_monitor.pipeline import MonitorPipeline
from paper_monitor.progress import ProgressBar
from paper_monitor.reports import generate_comparison_report, generate_paper_reports, generate_report
from paper_monitor.scheduler import run_daemon
from paper_monitor.storage import Database
from paper_monitor.utils import ensure_directory, now_iso, to_day_bounds, today_string


LOGGER = logging.getLogger(__name__)


ENRICH_CHECKPOINT_KEY = "enrich:since"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地论文监控工具")
    parser.add_argument("--config", default="config/config.json", help="配置文件路径")
    parser.add_argument("--log-level", default="INFO", help="日志级别，默认 INFO")

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="创建默认配置并初始化目录")
    init_parser.add_argument("--force", action="store_true", help="覆盖现有配置文件")

    fetch_parser = subparsers.add_parser("fetch", help="仅抓取并写入数据库")
    fetch_parser.add_argument("--source", action="append", choices=["arxiv", "dblp", "google_scholar_alerts"])
    fetch_parser.add_argument("--start-year", type=int, help="首次回填时只保留不早于该年份的论文")
    fetch_parser.add_argument("--recent-limit", type=int, help="按时间顺序保留最近多少篇论文")
    fetch_parser.add_argument("--page-size", type=int, help="单次请求分页大小")
    fetch_parser.add_argument("--since-last-run", action="store_true", help="只处理自上次成功抓取后新增的论文")

    enrich_parser = subparsers.add_parser("enrich", help="下载 PDF、抽取全文并刷新摘要")
    enrich_parser.add_argument("--limit", type=int, help="本轮最多增强多少篇论文")
    enrich_parser.add_argument("--topic", action="append", help="仅增强指定 topic id")
    enrich_parser.add_argument("--classification", action="append", choices=["relevant", "maybe"])
    enrich_parser.add_argument("--force", action="store_true", help="忽略缓存，重新下载或重新抽取")
    enrich_parser.add_argument("--with-llm", action="store_true", help="若已配置 LLM，则强制尝试生成 LLM 摘要")
    enrich_parser.add_argument("--extra-config", action="append", help="额外加载一份配置文件中的 LLM 变体")
    enrich_parser.add_argument("--skip-pdf", action="store_true", help="跳过 PDF 下载与全文抽取，只基于标题/摘要生成总结")
    enrich_parser.add_argument("--workers", type=int, default=1, help="增强阶段并发 worker 数，默认 1")
    enrich_parser.add_argument("--since-last-run", action="store_true", help="只增强自上次成功增强后新增入库的论文")

    report_parser = subparsers.add_parser("report", help="仅生成报告")
    report_parser.add_argument("--date", help="报告日期，格式 YYYY-MM-DD，默认今天")
    report_parser.add_argument("--type", default="daily", choices=["daily", "weekly"])
    report_parser.add_argument("--days", type=int, help="统计窗口天数，默认使用配置中的 report.lookback_days")
    report_parser.add_argument("--with-llm", action="store_true", help="若已配置 LLM，则生成主题级聚合摘要")
    report_parser.add_argument("--extra-config", action="append", help="额外加载一份配置文件中的 LLM 变体")

    paper_report_parser = subparsers.add_parser("paper-report", help="生成单篇论文的 Markdown / HTML / JSON 报告")
    paper_report_parser.add_argument("--paper-id", type=int, action="append", help="指定一个或多个 paper_id")
    paper_report_parser.add_argument("--date", help="若不指定 --paper-id，则按时间窗口导出对应论文")
    paper_report_parser.add_argument("--type", default="daily", choices=["daily", "weekly"])
    paper_report_parser.add_argument("--days", type=int, help="统计窗口天数，默认使用配置中的 report.lookback_days")
    paper_report_parser.add_argument("--extra-config", action="append", help="额外加载一份配置文件中的 LLM 变体")

    compare_parser = subparsers.add_parser("compare-report", help="使用两套配置生成并排对比报告")
    compare_parser.add_argument("--date", help="报告日期，格式 YYYY-MM-DD，默认今天")
    compare_parser.add_argument("--type", default="weekly", choices=["daily", "weekly"])
    compare_parser.add_argument("--days", type=int, help="统计窗口天数，默认使用左侧配置中的 report.lookback_days")
    compare_parser.add_argument("--left-config", default="config/config.json", help="左侧配置文件路径")
    compare_parser.add_argument("--right-config", default="config/config.example.json", help="右侧配置文件路径")

    run_once_parser = subparsers.add_parser("run-once", help="抓取后立即生成报告")
    run_once_parser.add_argument("--date", help="报告日期，格式 YYYY-MM-DD，默认今天")
    run_once_parser.add_argument("--type", default="daily", choices=["daily", "weekly"])
    run_once_parser.add_argument("--days", type=int, help="统计窗口天数，默认使用配置中的 report.lookback_days")
    run_once_parser.add_argument("--source", action="append", choices=["arxiv", "dblp", "google_scholar_alerts"])
    run_once_parser.add_argument("--enrich", action="store_true", help="抓取后执行全文增强")
    run_once_parser.add_argument("--enrich-limit", type=int, help="本轮增强的论文数量上限")
    run_once_parser.add_argument("--with-llm", action="store_true", help="若已配置 LLM，则启用 LLM 摘要")
    run_once_parser.add_argument("--extra-config", action="append", help="额外加载一份配置文件中的 LLM 变体")
    run_once_parser.add_argument("--skip-pdf", action="store_true", help="跳过 PDF 下载与全文抽取，只基于标题/摘要生成总结")
    run_once_parser.add_argument("--workers", type=int, default=1, help="增强阶段并发 worker 数，默认 1")
    run_once_parser.add_argument("--start-year", type=int, help="首次回填时只保留不早于该年份的论文")
    run_once_parser.add_argument("--recent-limit", type=int, help="按时间顺序保留最近多少篇论文")
    run_once_parser.add_argument("--page-size", type=int, help="单次请求分页大小")
    run_once_parser.add_argument("--since-last-run", action="store_true", help="只处理自上次成功抓取后新增的论文")

    daemon_parser = subparsers.add_parser("daemon", help="持续轮询抓取，并覆盖更新当天报告")
    daemon_parser.add_argument("--type", default="daily", choices=["daily", "weekly"])
    daemon_parser.add_argument("--loops", type=int, help="仅运行指定轮次，便于测试")
    daemon_parser.add_argument("--enrich", action="store_true", help="每轮抓取后执行全文增强")
    daemon_parser.add_argument("--with-llm", action="store_true", help="若已配置 LLM，则启用 LLM 摘要")
    daemon_parser.add_argument("--extra-config", action="append", help="额外加载一份配置文件中的 LLM 变体")
    daemon_parser.add_argument("--skip-pdf", action="store_true", help="跳过 PDF 下载与全文抽取，只基于标题/摘要生成总结")
    daemon_parser.add_argument("--workers", type=int, default=1, help="增强阶段并发 worker 数，默认 1")
    daemon_parser.add_argument("--since-last-run", action="store_true", help="每轮仅处理自上次成功抓取后新增的论文")

    return parser


def _configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _open_database(config_path: str) -> tuple[Database, MonitorPipeline]:
    settings = load_settings(config_path)
    ensure_directory(settings.report_dir)
    ensure_directory(settings.export_dir)
    ensure_directory(settings.database_path.parent)
    ensure_directory(settings.enrichment.pdf_dir)
    ensure_directory(settings.enrichment.text_dir)
    db = Database(settings.database_path, settings.timezone)
    db.initialize()
    return db, MonitorPipeline(settings, db)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    if args.command == "init":
        config_path = write_default_config(args.config, force=args.force)
        settings = load_settings(config_path)
        ensure_directory(settings.report_dir)
        ensure_directory(settings.export_dir)
        ensure_directory(settings.database_path.parent)
        ensure_directory(settings.enrichment.pdf_dir)
        ensure_directory(settings.enrichment.text_dir)
        db = Database(settings.database_path, settings.timezone)
        db.initialize()
        db.close()
        print(f"已初始化配置: {config_path}")
        print(f"数据库路径: {settings.database_path}")
        return 0

    if args.command == "compare-report":
        left_settings = load_settings(args.left_config)
        right_settings = load_settings(args.right_config)
        if [topic.id for topic in left_settings.topics] != [topic.id for topic in right_settings.topics]:
            raise ValueError("compare-report 需要两份配置使用相同 topic 列表。")
        if left_settings.database_path != right_settings.database_path:
            raise ValueError("compare-report 需要两份配置指向同一个数据库，以便对比同一批论文。")

        compare_db = Database(left_settings.database_path, left_settings.timezone)
        compare_db.initialize()
        try:
            report_date = args.date or today_string(left_settings.timezone)
            variants = build_runtime_variants(left_settings, [args.right_config])
            paths = generate_comparison_report(
                compare_db,
                left_settings,
                report_date=report_date,
                report_type=args.type,
                variants=variants,
                lookback_days=args.days,
            )
        finally:
            compare_db.close()

        print(f"对比报告已生成: {paths['markdown']}")
        print(f"HTML: {paths['html']}")
        print(f"JSON: {paths['json']}")
        return 0

    db, pipeline = _open_database(args.config)
    settings = pipeline.settings
    extra_configs = getattr(args, "extra_config", None) or []
    runtime_variants = build_runtime_variants(settings, extra_configs)
    enabled_variants = [variant for variant in runtime_variants if variant.client.enabled]
    enrichment_pipeline = EnrichmentPipeline(settings, db, llm_variants=runtime_variants)
    llm_client = enabled_variants[0].client if enabled_variants else LLMClient(settings.llm)
    if getattr(args, "with_llm", False) and not enabled_variants:
        LOGGER.warning(
            "检测到 --with-llm，但当前没有可用的 LLM 变体。请检查 API key 环境变量和额外配置。"
        )
    try:
        if args.command == "fetch":
            selected_sources = set(args.source or [])
            stats = pipeline.run_fetch(
                selected_sources=selected_sources or None,
                start_year=args.start_year,
                recent_limit=args.recent_limit,
                page_size=args.page_size,
                since_last_run=args.since_last_run,
            )
            print(
                f"抓取完成: fetched={stats.fetched}, processed={stats.processed}, "
                f"stored={stats.stored}, matched={stats.matched}"
            )
            if stats.errors:
                print("错误:")
                for item in stats.errors:
                    print(f"- {item}")
            return 0

        if args.command == "enrich":
            if args.skip_pdf and args.with_llm:
                LOGGER.warning("当前使用了 --skip-pdf，LLM 将无法读取完整 PDF 文本，只会基于标题/摘要生成总结。")
            enrich_started_at = now_iso(settings.timezone) if args.since_last_run else None
            stats = enrichment_pipeline.run(
                limit=args.limit,
                topic_ids=args.topic,
                classifications=args.classification,
                created_after=db.get_checkpoint(ENRICH_CHECKPOINT_KEY) if args.since_last_run else None,
                force=args.force,
                use_llm=args.with_llm or None,
                skip_document_processing=args.skip_pdf,
                workers=args.workers,
            )
            if args.since_last_run and enrich_started_at is not None and not stats.errors:
                db.set_checkpoint(ENRICH_CHECKPOINT_KEY, enrich_started_at)
            print(
                f"增强完成: enriched={stats.enriched}, downloaded_pdfs={stats.downloaded_pdfs}, "
                f"extracted_texts={stats.extracted_texts}, llm_summaries={stats.llm_summaries}, skipped={stats.skipped}"
            )
            if stats.errors:
                print("错误:")
                for item in stats.errors:
                    print(f"- {item}")
            return 0

        if args.command == "report":
            report_date = args.date or today_string(settings.timezone)
            paths = generate_report(
                db,
                settings,
                report_date=report_date,
                report_type=args.type,
                lookback_days=args.days,
                llm_client=llm_client,
                llm_variants=enabled_variants,
                use_llm_topic_digest=args.with_llm,
            )
            print(f"报告已生成: {paths['markdown']}")
            print(f"HTML: {paths['html']}")
            print(f"JSON: {paths['json']}")
            print(f"单篇报告目录: {paths['papers_dir']}")
            return 0

        if args.command == "paper-report":
            if args.paper_id:
                paper_ids = args.paper_id
            else:
                report_date = args.date or today_string(settings.timezone)
                lookback_days = args.days or settings.report.lookback_days
                start_at, end_at = to_day_bounds(report_date, settings.timezone, lookback_days)
                entries = db.fetch_report_entries(start_at, end_at, settings.report.include_maybe)
                paper_ids = [entry.paper.id for entry in entries]
            unique_paper_ids = list(dict.fromkeys(paper_ids))
            progress_bar = ProgressBar("单篇报告", len(unique_paper_ids) or 1)
            outputs = generate_paper_reports(
                db,
                settings,
                unique_paper_ids,
                llm_variants=enabled_variants,
                progress_bar=progress_bar,
            )
            progress_bar.close(f"单篇报告已生成 {len(outputs)} 篇")
            if not outputs:
                print("没有找到可导出的论文。")
                return 0
            first = next(iter(outputs.values()))
            print(f"单篇报告数量: {len(outputs)}")
            print(f"示例 Markdown: {first['markdown']}")
            print(f"示例 HTML: {first['html']}")
            print(f"示例 JSON: {first['json']}")
            return 0

        if args.command == "run-once":
            selected_sources = set(args.source or [])
            stats = pipeline.run_fetch(
                selected_sources=selected_sources or None,
                start_year=args.start_year,
                recent_limit=args.recent_limit,
                page_size=args.page_size,
                since_last_run=args.since_last_run,
            )
            enrichment_stats = None
            if args.enrich:
                if args.skip_pdf and args.with_llm:
                    LOGGER.warning("当前使用了 --skip-pdf，LLM 将无法读取完整 PDF 文本，只会基于标题/摘要生成总结。")
                enrich_started_at = now_iso(settings.timezone) if args.since_last_run else None
                enrichment_stats = enrichment_pipeline.run(
                    limit=args.enrich_limit,
                    paper_ids=stats.new_paper_ids if args.since_last_run else None,
                    force=False,
                    use_llm=args.with_llm or None,
                    skip_document_processing=args.skip_pdf,
                    workers=args.workers,
                )
                if args.since_last_run and enrich_started_at is not None and not enrichment_stats.errors:
                    db.set_checkpoint(ENRICH_CHECKPOINT_KEY, enrich_started_at)
            report_date = args.date or today_string(settings.timezone)
            paths = generate_report(
                db,
                settings,
                report_date=report_date,
                report_type=args.type,
                lookback_days=args.days,
                llm_client=llm_client,
                llm_variants=enabled_variants,
                use_llm_topic_digest=args.with_llm,
            )
            print(
                f"运行完成: fetched={stats.fetched}, processed={stats.processed}, "
                f"stored={stats.stored}, matched={stats.matched}"
            )
            if enrichment_stats is not None:
                print(
                    f"增强结果: enriched={enrichment_stats.enriched}, "
                    f"downloaded_pdfs={enrichment_stats.downloaded_pdfs}, "
                    f"extracted_texts={enrichment_stats.extracted_texts}, "
                    f"llm_summaries={enrichment_stats.llm_summaries}, skipped={enrichment_stats.skipped}"
                )
            print(f"报告路径: {paths['markdown']}")
            print(f"单篇报告目录: {paths['papers_dir']}")
            if stats.errors:
                print("错误:")
                for item in stats.errors:
                    print(f"- {item}")
            if enrichment_stats and enrichment_stats.errors:
                print("增强错误:")
                for item in enrichment_stats.errors:
                    print(f"- {item}")
            return 0

        if args.command == "daemon":
            run_daemon(
                settings,
                db,
                pipeline,
                report_type=args.type,
                loop_limit=args.loops,
                enrich=args.enrich,
                use_llm=args.with_llm or None,
                llm_variants=enabled_variants,
                skip_document_processing=args.skip_pdf,
                workers=args.workers,
                since_last_run=args.since_last_run,
            )
            return 0

        parser.error(f"未知命令: {args.command}")
        return 2
    finally:
        db.close()
