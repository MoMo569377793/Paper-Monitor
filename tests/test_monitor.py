from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from unittest import mock
from pathlib import Path

from paper_monitor.cli import main as cli_main
from paper_monitor.config import load_settings
from paper_monitor.enrichment import (
    DocumentArtifacts,
    DocumentProcessor,
    EnrichmentPipeline,
    _summary_has_complete_pdf_output,
)
from paper_monitor.fetchers.arxiv import ArxivFetcher, extract_arxiv_id
from paper_monitor.fetchers.dblp import DBLPFetcher
from paper_monitor.llm import LLMClient, TASK_TOPIC_DIGEST, looks_like_invalid_direct_pdf_summary
from paper_monitor.llm_registry import LLMRuntimeVariant
from paper_monitor.models import FetchPlan, GenericSourceConfig, LLMResult
from paper_monitor.models import PaperCandidate, PaperLLMSummary, PaperRecord, ReportEntry, RunStats, TopicConfig, TopicEvaluation
from paper_monitor.pipeline import MonitorPipeline
from paper_monitor.reports import (
    _prepare_digest_entries_for_variant,
    generate_catalog_report,
    generate_comparison_report,
    generate_paper_reports,
    generate_preview_report,
    generate_report,
)
from paper_monitor.scoring import evaluate_candidate_against_topic, evaluate_paper_against_topic
from paper_monitor.storage import Database


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CONFIG_IKUN = REPO_ROOT / "config" / "config-ikun.json"
FIXTURE_CONFIG_POE = REPO_ROOT / "config" / "config-poe.json"


class MonitorPipelineTest(unittest.TestCase):
    def test_arxiv_request_retries_timeout_error(self) -> None:
        fetcher = ArxivFetcher(GenericSourceConfig(timeout_seconds=40))

        class DummyResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return b"<feed />"

        with mock.patch("urllib.request.urlopen", side_effect=[TimeoutError("timed out"), DummyResponse()]) as urlopen:
            payload = fetcher._request_feed("http://example.invalid/feed")

        self.assertEqual(payload, "<feed />")
        self.assertEqual(urlopen.call_count, 2)

    def test_dblp_request_retries_timeout_error(self) -> None:
        fetcher = DBLPFetcher(GenericSourceConfig(timeout_seconds=30))

        class DummyResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return b'{\"result\": {\"hits\": {\"hit\": []}}}'

        with mock.patch("urllib.request.urlopen", side_effect=[TimeoutError("timed out"), DummyResponse()]) as urlopen:
            payload = fetcher._fetch_with_retry("https://dblp.org/search/publ/api?q=test", "test")

        self.assertEqual(payload, '{"result": {"hits": {"hit": []}}}')
        self.assertEqual(urlopen.call_count, 2)

    def test_semantic_scholar_title_search_retries_timeout_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "config" / "config.json").write_text(FIXTURE_CONFIG_POE.read_text(encoding="utf-8"), encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            class DummyResponse:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def read(self) -> bytes:
                    return b'{"data":[{"title":"FlashAttention","citationCount":123,"influentialCitationCount":12}]}'

            with mock.patch("urllib.request.urlopen", side_effect=[TimeoutError("timed out"), DummyResponse()]) as urlopen:
                payload = pipeline._search_semantic_scholar_by_title(
                    "FlashAttention",
                    "title,year,citationCount,influentialCitationCount",
                )

            self.assertEqual(payload["data"][0]["citationCount"], 123)
            self.assertEqual(urlopen.call_count, 2)
            db.close()

    def test_extract_arxiv_id_supports_abs_and_pdf_urls(self) -> None:
        self.assertEqual(extract_arxiv_id("https://arxiv.org/abs/2501.12345v2"), "2501.12345v2")
        self.assertEqual(extract_arxiv_id("https://arxiv.org/pdf/2501.12345v2.pdf"), "2501.12345v2")
        self.assertEqual(extract_arxiv_id("10.48550/arXiv.2501.12345"), "2501.12345")

    def test_paper_preview_exports_files_without_touching_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "config" / "config.json").write_text(
                FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            exit_code = cli_main(
                [
                    "--config",
                    str(root / "config" / "config.json"),
                    "paper-preview",
                    "--title",
                    "Preview Only Matrix-Free GPU Operator Evaluation",
                    "--abstract",
                    "This paper studies matrix-free operator evaluation, GPU implementation, and performance portability.",
                    "--primary-url",
                    "https://example.com/preview-paper",
                    "--skip-pdf",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(list((root / "reports" / "preview").glob("*.md"))), 1)
            self.assertEqual(len(list((root / "exports" / "preview").glob("*.json"))), 1)
            self.assertFalse((root / "data" / "papers.db").exists())

    def test_paper_preview_can_autofill_metadata_from_arxiv_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "config" / "config.json").write_text(
                FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            feed = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2501.12345v2</id>
    <updated>2026-03-20T08:00:00Z</updated>
    <published>2026-03-19T08:00:00Z</published>
    <title>Autofilled Matrix-Free Preview Paper</title>
    <summary>We study matrix-free operator evaluation with GPU performance portability.</summary>
    <author><name>Alice</name></author>
    <author><name>Bob</name></author>
    <link rel="alternate" href="https://arxiv.org/abs/2501.12345v2" />
    <link title="pdf" href="https://arxiv.org/pdf/2501.12345v2.pdf" />
    <arxiv:doi>10.1000/autofill-preview</arxiv:doi>
    <category term="cs.NA" />
  </entry>
</feed>
"""

            class FakeResponse:
                def __init__(self, payload: str) -> None:
                    self.payload = payload.encode("utf-8")
                    self.headers = {"Content-Type": "application/atom+xml"}

                def read(self) -> bytes:
                    return self.payload

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb) -> None:
                    return None

            with mock.patch("paper_monitor.fetchers.arxiv.urllib.request.urlopen", return_value=FakeResponse(feed)):
                exit_code = cli_main(
                    [
                        "--config",
                        str(root / "config" / "config.json"),
                        "paper-preview",
                        "--primary-url",
                        "https://arxiv.org/abs/2501.12345v2",
                        "--skip-pdf",
                    ]
                )

            self.assertEqual(exit_code, 0)
            preview_json = next((root / "exports" / "preview").glob("*.json"))
            payload = json.loads(preview_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["paper"]["title"], "Autofilled Matrix-Free Preview Paper")
            self.assertEqual(payload["paper"]["doi"], "10.1000/autofill-preview")
            self.assertEqual(payload["paper"]["arxiv_id"], "2501.12345v2")

    def test_paper_set_pdf_updates_existing_paper_without_touching_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "config" / "config.json").write_text(
                FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            add_exit = cli_main(
                [
                    "--config",
                    str(root / "config" / "config.json"),
                    "paper-add",
                    "--topic",
                    "ai_operator_acceleration",
                    "--title",
                    "Manual PDF Patch Test",
                    "--primary-url",
                    "https://example.com/paper",
                    "--pdf-url",
                    "https://example.com/old.pdf",
                ]
            )
            self.assertEqual(add_exit, 0)

            settings = load_settings(root / "config" / "config.json")
            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            paper = db.find_papers(title_substring="Manual PDF Patch Test", limit=1)[0]
            db.update_paper_assets(
                paper.id,
                pdf_local_path="/tmp/old.pdf",
                pdf_status="downloaded",
                pdf_downloaded_at="2026-03-24T09:00:00+08:00",
                fulltext_txt_path="/tmp/old.txt",
                fulltext_excerpt="old excerpt",
                fulltext_status="extracted",
                page_count=12,
            )
            match_count_before = db.connection.execute(
                "SELECT COUNT(*) FROM matches WHERE paper_id = ?",
                (paper.id,),
            ).fetchone()[0]
            db.close()

            set_exit = cli_main(
                [
                    "--config",
                    str(root / "config" / "config.json"),
                    "paper-set-pdf",
                    "--paper-id",
                    str(paper.id),
                    "--pdf-url",
                    "file:///tmp/new.pdf",
                ]
            )
            self.assertEqual(set_exit, 0)

            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            updated = db.get_paper(paper.id)
            match_count_after = db.connection.execute(
                "SELECT COUNT(*) FROM matches WHERE paper_id = ?",
                (paper.id,),
            ).fetchone()[0]
            self.assertEqual(updated.pdf_url, "file:///tmp/new.pdf")
            self.assertEqual(updated.pdf_status, "pending")
            self.assertEqual(updated.pdf_local_path, "")
            self.assertEqual(updated.fulltext_txt_path, "")
            self.assertEqual(updated.fulltext_excerpt, "")
            self.assertEqual(updated.fulltext_status, "empty")
            self.assertIsNone(updated.page_count)
            self.assertEqual(match_count_before, match_count_after)
            db.close()

    def test_paper_find_no_pdf_lists_titles_and_summary_source_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "config" / "config.json").write_text(
                FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            settings = load_settings(root / "config" / "config.json")
            db = Database(settings.database_path, settings.timezone)
            db.initialize()

            no_pdf_id, _ = db.upsert_paper(
                PaperCandidate(
                    source_name="manual",
                    source_paper_id="manual-no-pdf",
                    query_text="manual",
                    title="No PDF Paper",
                    abstract="abstract",
                    primary_url="https://example.com/no-pdf",
                    venue="manual",
                    raw={"manual": True},
                )
            )
            with_pdf_id, _ = db.upsert_paper(
                PaperCandidate(
                    source_name="manual",
                    source_paper_id="manual-with-pdf",
                    query_text="manual",
                    title="With PDF Paper",
                    abstract="abstract",
                    primary_url="https://example.com/with-pdf",
                    venue="manual",
                    raw={"manual": True},
                )
            )
            db.update_paper_analysis(
                no_pdf_id,
                summary_text="summary",
                summary_basis="llm+abstract+metadata",
                tags=[],
                llm_summary={"source_mode": "abstract_metadata"},
            )
            db.update_paper_assets(
                with_pdf_id,
                pdf_local_path="/tmp/with.pdf",
                pdf_status="downloaded",
                pdf_downloaded_at="2026-03-24T10:00:00+08:00",
                fulltext_txt_path="/tmp/with.txt",
                fulltext_excerpt="excerpt",
                fulltext_status="extracted",
                page_count=3,
            )
            db.close()

            buffer = io.StringIO()
            with mock.patch("sys.stdout", new=buffer):
                exit_code = cli_main(
                    [
                        "--config",
                        str(root / "config" / "config.json"),
                        "paper-find",
                        "--no-pdf",
                        "--show-summary",
                        "--limit",
                        "10",
                    ]
                )

            self.assertEqual(exit_code, 0)
            output = buffer.getvalue()
            self.assertIn("当前库里没有本地 PDF 的论文有 1 篇", output)
            self.assertIn("- 1 No PDF Paper", output)
            self.assertIn("当前 primary_url: https://example.com/no-pdf", output)
            self.assertIn("当前默认摘要:", output)
            self.assertIn("summary", output)
            self.assertIn("这些论文当前的默认摘要来源：", output)
            self.assertIn("- llm+abstract+metadata / abstract_metadata: 1", output)
            self.assertNotIn("With PDF Paper", output)

    def test_complete_pdf_output_requires_pdf_basis(self) -> None:
        summary = type(
            "Summary",
            (),
            {
                "summary_text": "问题：A 方法：B 应用：C 结果：D",
                "summary_basis": "llm+abstract+metadata",
                "structured": {
                    "source_mode": "pdf_direct",
                    "direct_pdf_status": "used",
                    "basis": "llm+abstract+metadata",
                },
            },
        )()
        self.assertFalse(_summary_has_complete_pdf_output(summary))

    def test_since_last_run_only_processes_new_papers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            settings.topics = [settings.topics[0]]
            settings.topics[0].seed_papers = []
            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            old_candidate = PaperCandidate(
                source_name="arxiv",
                source_paper_id="2901.00001",
                query_text="matrix-free finite element",
                title="Matrix-Free FEM Baseline",
                abstract="matrix-free finite element, partial assembly, GPU implementation, and benchmark performance",
                authors=["Alice"],
                published_at="2026-03-10T09:00:00+08:00",
                updated_at="2026-03-10T09:00:00+08:00",
                primary_url="https://arxiv.org/abs/2901.00001",
                pdf_url="https://arxiv.org/pdf/2901.00001.pdf",
                doi="",
                arxiv_id="2901.00001",
                venue="arXiv",
                year=2026,
                categories=["cs.NA"],
                raw={"fixture": True, "ranking": {"citation_count": 5, "source": "fixture"}},
            )
            new_candidate = PaperCandidate(
                source_name="arxiv",
                source_paper_id="2901.00002",
                query_text="matrix-free finite element",
                title="Matrix-Free FEM Incremental",
                abstract="matrix-free finite element, multigrid, operator evaluation, and performance portability",
                authors=["Bob"],
                published_at="2026-03-10T11:00:00+08:00",
                updated_at="2026-03-10T11:00:00+08:00",
                primary_url="https://arxiv.org/abs/2901.00002",
                pdf_url="https://arxiv.org/pdf/2901.00002.pdf",
                doi="",
                arxiv_id="2901.00002",
                venue="arXiv",
                year=2026,
                categories=["cs.NA"],
                raw={"fixture": True, "ranking": {"citation_count": 6, "source": "fixture"}},
            )

            class FakeFetcher:
                enabled = True

                def __init__(self) -> None:
                    self.since_values: list[str | None] = []

                def fetch(self, topic, queries, plan, progress=None):  # noqa: ANN001, ARG002
                    self.since_values.append(plan.since_at)
                    if plan.since_at is None:
                        return [old_candidate]
                    return [old_candidate, new_candidate]

            class DisabledFetcher:
                enabled = False

            fake_fetcher = FakeFetcher()
            pipeline.fetchers = {
                "arxiv": fake_fetcher,
                "dblp": DisabledFetcher(),
                "google_scholar_alerts": DisabledFetcher(),
            }

            first = pipeline.run_fetch(selected_sources={"arxiv"}, since_last_run=True)
            self.assertEqual(first.stored, 1)
            self.assertEqual(first.processed, 1)
            self.assertEqual(len(first.new_paper_ids), 1)

            checkpoint_key = "fetch:arxiv:matrix_free_fem:since"
            checkpoint_value = db.get_checkpoint(checkpoint_key)
            self.assertIsNotNone(checkpoint_value)

            second = pipeline.run_fetch(selected_sources={"arxiv"}, since_last_run=True)
            self.assertEqual(second.stored, 1)
            self.assertEqual(second.processed, 1)
            self.assertEqual(len(second.new_paper_ids), 1)
            self.assertEqual(fake_fetcher.since_values[0], None)
            self.assertEqual(fake_fetcher.since_values[1], checkpoint_value)

            paper_count = db.connection.execute("SELECT COUNT(*) AS count FROM papers").fetchone()["count"]
            self.assertEqual(paper_count, 2)
            db.close()

    def test_topic_recent_limit_is_applied_after_cross_source_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            settings.llm.variant_id = "config-example-gpt-5-4"
            settings.llm.label = "gpt-5.4"
            settings.llm.model = "gpt-5.4"
            settings.llm.base_url = "https://example.invalid/v1"
            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            candidates: list[PaperCandidate] = []
            for index in range(70):
                candidates.append(
                    PaperCandidate(
                        source_name="arxiv",
                        source_paper_id=f"arxiv-{index}",
                        query_text="fixture",
                        title=f"Shared Paper {index}" if index < 20 else f"Arxiv Unique Paper {index}",
                        abstract="matrix-free finite element, partial assembly, GPU implementation, and benchmark performance",
                        authors=["Alice"],
                        published_at=f"2026-03-{28 - (index % 20):02d}T09:00:00+08:00",
                        updated_at=f"2026-03-{28 - (index % 20):02d}T09:00:00+08:00",
                        primary_url=f"https://arxiv.org/abs/{index}",
                        pdf_url=f"https://arxiv.org/pdf/{index}.pdf",
                        doi=f"10.1000/shared-{index}" if index < 20 else "",
                        arxiv_id=f"{index}",
                        venue="arXiv",
                        year=2026,
                        categories=["cs.NA"],
                        raw={"fixture": True, "ranking": {"citation_count": 2, "source": "fixture"}},
                    )
                )
            for index in range(70):
                candidates.append(
                    PaperCandidate(
                        source_name="dblp",
                        source_paper_id=f"dblp-{index}",
                        query_text="fixture",
                        title=f"Shared Paper {index}" if index < 20 else f"DBLP Unique Paper {index}",
                        abstract="matrix-free finite element, multigrid, operator evaluation, and performance portability",
                        authors=["Bob"],
                        published_at=f"2026-03-{28 - (index % 20):02d}",
                        updated_at=f"2026-03-{28 - (index % 20):02d}",
                        primary_url=f"https://dblp.org/rec/{index}",
                        pdf_url="",
                        doi=f"10.1000/shared-{index}" if index < 20 else "",
                        arxiv_id="",
                        venue="SC",
                        year=2026,
                        categories=["Conference"],
                        raw={"fixture": True, "ranking": {"citation_count": 2, "source": "fixture"}},
                    )
                )

            selected = pipeline._select_topic_candidates(
                settings.topics[0],
                candidates,
                FetchPlan(recent_limit=100, page_size=50),
            )
            logical_titles = {candidate.title for candidate in selected}

            self.assertEqual(len(logical_titles), 100)
            self.assertLessEqual(len(selected), 120)

            db.close()

    def test_matrix_free_topic_requires_direct_matrix_free_and_impl_signals(self) -> None:
        settings = load_settings(FIXTURE_CONFIG_IKUN)
        topic = next(item for item in settings.topics if item.id == "matrix_free_fem")

        broad_gpu_candidate = PaperCandidate(
            source_name="arxiv",
            source_paper_id="broad-1",
            query_text="fixture",
            title="GPU Acceleration for High-Order Finite Element Solvers",
            abstract="We accelerate high-order finite element kernels on GPUs with benchmarks.",
            authors=["Alice"],
            venue="SC",
            year=2025,
            categories=["cs.DC"],
            raw={},
        )
        self.assertEqual(evaluate_candidate_against_topic(broad_gpu_candidate, topic).classification, "irrelevant")

        math_only_candidate = PaperCandidate(
            source_name="arxiv",
            source_paper_id="math-1",
            query_text="fixture",
            title="Matrix-Free Finite Element Discretization with Error Analysis",
            abstract="We study a matrix-free finite element formulation and prove convergence and stability.",
            authors=["Bob"],
            venue="Journal of Numerical Analysis",
            year=2024,
            categories=["math.NA"],
            raw={},
        )
        self.assertEqual(evaluate_candidate_against_topic(math_only_candidate, topic).classification, "irrelevant")

        target_candidate = PaperCandidate(
            source_name="arxiv",
            source_paper_id="target-1",
            query_text="fixture",
            title="Matrix-Free Operator Evaluation for High-Order Finite Elements on GPUs",
            abstract="We present matrix-free operator evaluation with sum-factorization, CUDA backends, and roofline benchmarks.",
            authors=["Carol"],
            venue="SC",
            year=2025,
            categories=["cs.DC"],
            raw={"ranking": {"citation_count": 220}},
        )
        evaluation = evaluate_candidate_against_topic(target_candidate, topic)
        self.assertIn(evaluation.classification, {"relevant", "maybe"})
        self.assertGreaterEqual(evaluation.score, topic.threshold)

    def test_matrix_free_topic_rejects_cross_domain_matrix_free_contract_paper(self) -> None:
        settings = load_settings(FIXTURE_CONFIG_IKUN)
        topic = next(item for item in settings.topics if item.id == "matrix_free_fem")

        candidate = PaperCandidate(
            source_name="arxiv",
            source_paper_id="econ-1",
            query_text="fixture",
            title="Scalable Principal-Agent Contract Design via Gradient-Based Optimization",
            abstract=(
                "We introduce a matrix-free, variance-reduced bilevel optimization framework for contract design "
                "using implicit differentiation with conjugate gradients and benchmark CARA-Normal environments."
            ),
            authors=["Alice"],
            venue="arXiv",
            year=2025,
            categories=["econ.TH"],
            raw={},
        )

        self.assertEqual(evaluate_candidate_against_topic(candidate, topic).classification, "irrelevant")

    def test_matrix_free_topic_ignores_llm_tags_when_scoring_existing_papers(self) -> None:
        settings = load_settings(FIXTURE_CONFIG_IKUN)
        topic = next(item for item in settings.topics if item.id == "matrix_free_fem")

        paper = PaperRecord(
            id=98,
            title="OVGGT: O(1) Constant-Cost Streaming Visual Geometry Transformer",
            title_norm="ovgggt",
            abstract=(
                "We present a training-free framework for streaming visual geometry transformers that bounds GPU "
                "memory with a fixed budget while remaining compatible with FlashAttention."
            ),
            authors=["Alice"],
            published_at="2026-02-01T00:00:00+08:00",
            updated_at="2026-02-01T00:00:00+08:00",
            primary_url="https://example.com/ovggt",
            pdf_url="https://example.com/ovggt.pdf",
            doi="",
            arxiv_id="2603.05959",
            venue="arXiv",
            year=2026,
            categories=["cs.CV"],
            summary_text="",
            summary_basis="",
            tags=["与Matrix-Free/DG弱相关", "GPU推理优化", "训练免费方法"],
            pdf_local_path="",
            pdf_status="pending",
            pdf_downloaded_at=None,
            fulltext_txt_path="",
            fulltext_excerpt="",
            fulltext_status="empty",
            page_count=None,
            llm_summary={},
            analysis_updated_at=None,
            source_first="arxiv",
            created_at="2026-03-20T00:00:00+08:00",
            last_seen_at="2026-03-20T00:00:00+08:00",
            metadata={},
        )

        self.assertEqual(evaluate_paper_against_topic(paper, topic).classification, "irrelevant")

    def test_ai_topic_accepts_distributed_runtime_lane(self) -> None:
        settings = load_settings(FIXTURE_CONFIG_IKUN)
        topic = next(item for item in settings.topics if item.id == "ai_operator_acceleration")

        candidate = PaperCandidate(
            source_name="arxiv",
            source_paper_id="ai-runtime-1",
            query_text="fixture",
            title="AutoOverlap: Enabling Fine-Grained Overlap of Computation and Communication with Chunk-Based Scheduling",
            abstract=(
                "Communication has become a first-order bottleneck in large-scale GPU workloads. "
                "We present AutoOverlap, a compiler and runtime that enables automatic fine-grained "
                "overlap inside a single fused kernel. Implemented as a source-to-source compiler on "
                "Triton, AutoOverlap aligns computation with chunk availability and delivers up to "
                "4.7x speedup on multi-GPU workloads."
            ),
            authors=["Alice"],
            venue="MLSys",
            year=2026,
            categories=["cs.DC", "cs.LG"],
            raw={},
        )

        evaluation = evaluate_candidate_against_topic(candidate, topic)
        self.assertEqual(evaluation.classification, "relevant")
        self.assertTrue(any("硬性准入路径" in reason for reason in evaluation.reasons))

    def test_ai_topic_accepts_cpu_vector_lane(self) -> None:
        settings = load_settings(FIXTURE_CONFIG_IKUN)
        topic = next(item for item in settings.topics if item.id == "ai_operator_acceleration")

        candidate = PaperCandidate(
            source_name="dblp",
            source_paper_id="ai-cpu-1",
            query_text="fixture",
            title="ARM SME Microkernels for Transformer GEMM Inference",
            abstract=(
                "We implement SIMD microkernels using ARM SME and SVE for GEMM and attention in transformer "
                "inference, and report throughput and latency benchmarks."
            ),
            authors=["Alice"],
            venue="CGO",
            year=2025,
            categories=["cs.AR"],
            raw={},
        )

        self.assertEqual(evaluate_candidate_against_topic(candidate, topic).classification, "relevant")

    def test_ai_topic_rejects_token_pruning_paper(self) -> None:
        settings = load_settings(FIXTURE_CONFIG_IKUN)
        topic = next(item for item in settings.topics if item.id == "ai_operator_acceleration")

        candidate = PaperCandidate(
            source_name="arxiv",
            source_paper_id="ai-model-1",
            query_text="fixture",
            title="IDPruner: Harmonizing Importance and Diversity in Visual Token Pruning for MLLMs",
            abstract=(
                "We propose a visual token pruning method for multimodal large language models that reduces "
                "inference cost without introducing a new kernel implementation or runtime system."
            ),
            authors=["Alice"],
            venue="arXiv",
            year=2026,
            categories=["cs.CV", "cs.LG"],
            raw={},
        )

        self.assertEqual(evaluate_candidate_against_topic(candidate, topic).classification, "irrelevant")

    def test_ai_topic_summary_rerank_rejects_prompt_highlighting_paper(self) -> None:
        settings = load_settings(FIXTURE_CONFIG_IKUN)
        topic = next(item for item in settings.topics if item.id == "ai_operator_acceleration")

        paper = PaperRecord(
            id=228,
            title="Prism-Delta: Differential Subspace Steering for Prompt Highlighting in Large Language Models",
            title_norm="prism-delta",
            abstract=(
                "We improve prompt highlighting in large language models with differential subspace steering "
                "and report gains on BiasBios, CounterFact, and Pronoun Change."
            ),
            authors=["Alice"],
            published_at="2026-03-01T00:00:00+08:00",
            updated_at="2026-03-01T00:00:00+08:00",
            primary_url="https://example.com/prism",
            pdf_url="https://example.com/prism.pdf",
            doi="",
            arxiv_id="2603.00001",
            venue="arXiv",
            year=2026,
            categories=["cs.CL"],
            summary_text=(
                "The paper proposes prompt highlighting for large language models, evaluates on BiasBios, "
                "CounterFact, and Pronoun Change, and focuses on key/value steering rather than low-level "
                "system implementation."
            ),
            summary_basis="llm+pdf+metadata",
            tags=[],
            pdf_local_path="artifacts/pdfs/prism.pdf",
            pdf_status="downloaded",
            pdf_downloaded_at="2026-03-02T00:00:00+08:00",
            fulltext_txt_path="artifacts/text/prism.txt",
            fulltext_excerpt="",
            fulltext_status="extracted",
            page_count=12,
            llm_summary={"problem": "prompt highlighting", "application": "BiasBios and CounterFact"},
            analysis_updated_at="2026-03-02T00:00:00+08:00",
            source_first="arxiv",
            created_at="2026-03-02T00:00:00+08:00",
            last_seen_at="2026-03-02T00:00:00+08:00",
            metadata={},
        )

        evaluation = evaluate_paper_against_topic(paper, topic)
        self.assertEqual(evaluation.classification, "irrelevant")
        self.assertTrue(any("摘要级二次重评分" in reason for reason in evaluation.reasons))

    def test_ai_topic_summary_rerank_promotes_kernel_runtime_paper(self) -> None:
        settings = load_settings(FIXTURE_CONFIG_IKUN)
        topic = next(item for item in settings.topics if item.id == "ai_operator_acceleration")

        paper = PaperRecord(
            id=245,
            title="FlashDecoding++: Faster Large Language Model Inference with Asynchronization, Flat GEMM Optimization, and Heuristics.",
            title_norm="flashdecoding",
            abstract=(
                "We accelerate LLM inference with asynchronous softmax, flat GEMM, and heuristic dataflow."
            ),
            authors=["Alice"],
            published_at="2026-03-01T00:00:00+08:00",
            updated_at="2026-03-01T00:00:00+08:00",
            primary_url="https://example.com/flashdecoding",
            pdf_url="https://example.com/flashdecoding.pdf",
            doi="",
            arxiv_id="2603.00002",
            venue="arXiv",
            year=2026,
            categories=["cs.DC", "cs.LG"],
            summary_text=(
                "The paper implements asynchronous softmax and flat GEMM kernels for LLM inference, "
                "adds Tensor Core-aware tiling and runtime heuristics, and reports latency and throughput "
                "speedups on A100 and MI210."
            ),
            summary_basis="llm+pdf+metadata",
            tags=[],
            pdf_local_path="artifacts/pdfs/flashdecoding.pdf",
            pdf_status="downloaded",
            pdf_downloaded_at="2026-03-02T00:00:00+08:00",
            fulltext_txt_path="artifacts/text/flashdecoding.txt",
            fulltext_excerpt="",
            fulltext_status="extracted",
            page_count=14,
            llm_summary={"method": "kernel and runtime optimization", "results": "throughput speedups"},
            analysis_updated_at="2026-03-02T00:00:00+08:00",
            source_first="arxiv",
            created_at="2026-03-02T00:00:00+08:00",
            last_seen_at="2026-03-02T00:00:00+08:00",
            metadata={},
        )

        evaluation = evaluate_paper_against_topic(paper, topic)
        self.assertEqual(evaluation.classification, "relevant")
        self.assertGreaterEqual(evaluation.score, topic.threshold)
        self.assertTrue(any("摘要体现内核/编译器/系统实现" in reason for reason in evaluation.reasons))

    def test_ai_topic_rejects_tensorgalerkin_pde_system_paper(self) -> None:
        settings = load_settings(FIXTURE_CONFIG_IKUN)
        topic = next(item for item in settings.topics if item.id == "ai_operator_acceleration")

        candidate = PaperCandidate(
            source_name="arxiv",
            source_paper_id="ai-pde-1",
            query_text="fixture",
            title="Learning, Solving and Optimizing PDEs with TensorGalerkin",
            abstract=(
                "We present a high-performance GPU-compliant TensorGalerkin framework for linear system assembly "
                "in PDE solvers and physics-informed operator learning. The method tensorizes element-wise "
                "operations and uses sparse matrix multiplication for mesh-induced reductions."
            ),
            authors=["Alice"],
            venue="arXiv",
            year=2026,
            categories=["cs.LG"],
            raw={},
        )

        self.assertEqual(evaluate_candidate_against_topic(candidate, topic).classification, "irrelevant")

    def test_ai_topic_rejects_training_free_visual_geometry_flashattention_paper(self) -> None:
        settings = load_settings(FIXTURE_CONFIG_IKUN)
        topic = next(item for item in settings.topics if item.id == "ai_operator_acceleration")

        candidate = PaperCandidate(
            source_name="arxiv",
            source_paper_id="ai-vision-1",
            query_text="fixture",
            title="OVGGT: O(1) Constant-Cost Streaming Visual Geometry Transformer",
            abstract=(
                "Recent geometric foundation models achieve impressive reconstruction quality through "
                "all-to-all attention, yet their quadratic cost confines them to short offline sequences. "
                "We present a training-free framework that compresses the KV cache while remaining compatible "
                "with FlashAttention for long-horizon visual geometry inference."
            ),
            authors=["Alice"],
            venue="arXiv",
            year=2026,
            categories=["cs.CV"],
            raw={},
        )

        self.assertEqual(evaluate_candidate_against_topic(candidate, topic).classification, "irrelevant")

    def test_recent_limit_preserves_cited_classics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "config" / "config.json").write_text(
                FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            settings = load_settings(root / "config" / "config.json")
            topic = next(item for item in settings.topics if item.id == "ai_operator_acceleration")
            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            candidates: list[PaperCandidate] = []
            for index in range(80):
                candidates.append(
                    PaperCandidate(
                        source_name="dblp",
                        source_paper_id=f"recent-{index}",
                        query_text="fixture",
                        title=f"Recent Kernel Fusion Paper {index}",
                        abstract="Kernel fusion and GEMM optimization for transformer inference on GPUs.",
                        authors=["Alice"],
                        published_at=f"2026-03-{(index % 28) + 1:02d}",
                        updated_at=f"2026-03-{(index % 28) + 1:02d}",
                        primary_url=f"https://example.com/recent/{index}",
                        venue="MLSys",
                        year=2026,
                        categories=["cs.DC"],
                        raw={"ranking": {"citation_count": 3, "source": "fixture"}},
                    )
                )
            for index in range(80):
                candidates.append(
                    PaperCandidate(
                        source_name="dblp",
                        source_paper_id=f"classic-{index}",
                        query_text="fixture",
                        title=f"Classic GEMM Optimization Paper {index}",
                        abstract="GEMM optimization, Tensor Core scheduling, and system performance engineering for inference.",
                        authors=["Bob"],
                        published_at=f"2016-01-{(index % 28) + 1:02d}",
                        updated_at=f"2016-01-{(index % 28) + 1:02d}",
                        primary_url=f"https://example.com/classic/{index}",
                        venue="ISCA",
                        year=2016,
                        categories=["cs.AR"],
                        raw={"ranking": {"citation_count": 800, "influential_citation_count": 120, "source": "fixture"}},
                    )
                )

            selected = pipeline._select_topic_candidates(topic, candidates, FetchPlan(recent_limit=100, page_size=50))
            classic_count = len({item.title for item in selected if item.title.startswith("Classic GEMM Optimization")})
            self.assertGreaterEqual(classic_count, 45)
            db.close()

    def test_seed_candidates_are_forced_into_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "config" / "config.json").write_text(
                FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            settings = load_settings(root / "config" / "config.json")
            settings.topics = [next(item for item in settings.topics if item.id == "ai_operator_acceleration")]
            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            stats = RunStats()
            for candidate in pipeline._seed_candidates_for_topic(settings.topics[0]):
                pipeline._process_candidate(candidate, stats)
            row = db.connection.execute(
                """
                SELECT p.id
                FROM papers p
                JOIN matches m ON m.paper_id = p.id
                WHERE p.title = ? AND m.topic_id = ?
                """,
                ("FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", "ai_operator_acceleration"),
            ).fetchone()
            self.assertIsNotNone(row)
            db.close()

    def test_pipeline_scoring_and_report_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            settings.llm.variant_id = "config-example-gpt-5-4"
            settings.llm.label = "gpt-5.4"
            settings.llm.model = "gpt-5.4"
            settings.llm.base_url = "https://example.invalid/v1"
            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            matrix_candidate = PaperCandidate(
                source_name="arxiv",
                source_paper_id="2401.00001",
                query_text="all:\"matrix-free\" AND all:\"finite element\"",
                title="Matrix-Free Finite Element Operator Evaluation on GPUs",
                abstract=(
                    "We present a matrix-free finite element operator evaluation strategy "
                    "with partial assembly and sum factorization on GPUs."
                ),
                authors=["Alice", "Bob"],
                published_at="2026-03-10T09:00:00+08:00",
                updated_at="2026-03-10T09:00:00+08:00",
                primary_url="https://arxiv.org/abs/2401.00001",
                pdf_url="https://arxiv.org/pdf/2401.00001.pdf",
                doi="",
                arxiv_id="2401.00001",
                venue="arXiv",
                year=2026,
                categories=["cs.NA", "cs.MS"],
                raw={"fixture": True},
            )
            ai_candidate = PaperCandidate(
                source_name="dblp",
                source_paper_id="conf/example/flashattention",
                query_text="FlashAttention",
                title="Kernel Fusion for Transformer Inference with FlashAttention and Triton",
                abstract=(
                    "This paper studies kernel fusion for transformer inference and describes "
                    "a Triton-based code generation path for attention kernels."
                ),
                authors=["Carol"],
                published_at="2026-03-10",
                updated_at="2026-03-10",
                primary_url="https://dblp.org/rec/conf/example/flashattention",
                pdf_url="",
                doi="10.1000/example",
                arxiv_id="",
                venue="MLSys",
                year=2026,
                categories=["Conference"],
                raw={"fixture": True},
            )

            pipeline._process_candidate(matrix_candidate, RunStats())
            pipeline._process_candidate(ai_candidate, RunStats())

            class FakeRenderClient:
                enabled = False

            paths = generate_report(
                db,
                settings,
                report_date="2026-03-10",
                report_type="daily",
                lookback_days=1,
                llm_client=FakeRenderClient(),
            )
            topic_dir = Path(paths["topic_dir"])
            matrix_markdown = (topic_dir / "daily-2026-03-10--matrix_free_fem.md").read_text(encoding="utf-8")
            ai_markdown = (topic_dir / "daily-2026-03-10--ai_operator_acceleration.md").read_text(encoding="utf-8")
            matrix_html = (topic_dir / "daily-2026-03-10--matrix_free_fem.html").read_text(encoding="utf-8")
            ai_html = (topic_dir / "daily-2026-03-10--ai_operator_acceleration.html").read_text(encoding="utf-8")

            self.assertIn("Matrix-Free Finite Element Operator Evaluation on GPUs", matrix_markdown)
            self.assertIn("Kernel Fusion for Transformer Inference with FlashAttention and Triton", ai_markdown)
            self.assertIn("有限元分析 Matrix-Free 算法优化", matrix_html)
            self.assertIn("AI 算子加速", ai_html)

            db.close()

    def test_generate_report_handles_stored_variant_dicts_without_runtime_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            candidate = PaperCandidate(
                source_name="arxiv",
                source_paper_id="2509.00001",
                query_text="all:\"matrix-free\" AND all:\"finite element\"",
                title="Stored Variant Render Sample",
                abstract="matrix-free finite element operator application on GPUs",
                authors=["Alice"],
                published_at="2026-03-10T09:00:00+08:00",
                updated_at="2026-03-10T09:00:00+08:00",
                primary_url="https://arxiv.org/abs/2509.00001",
                pdf_url="https://arxiv.org/pdf/2509.00001.pdf",
                doi="",
                arxiv_id="2509.00001",
                venue="arXiv",
                year=2026,
                categories=["cs.NA"],
                raw={"fixture": True},
            )
            pipeline._process_candidate(candidate, RunStats())

            db.upsert_paper_llm_summary(
                1,
                variant_id="ikun",
                variant_label="ikun_gpt-5.4",
                provider="IkunCoding",
                base_url="https://api.ikuncode.cc/v1",
                model="gpt-5.4",
                summary_text="问题：矩阵自由有限元算子应用成本高。方法：使用 GPU 和 partial assembly。",
                summary_basis="llm+pdf+metadata",
                tags=["matrix-free"],
                structured={
                    "summary": "使用 GPU 和 partial assembly 的矩阵自由有限元总结。",
                    "problem": "矩阵自由有限元算子应用成本高。",
                    "method": "使用 GPU 和 partial assembly。",
                    "source_mode": "pdf_direct",
                    "direct_pdf_status": "used",
                },
                usage={},
            )

            class FakeRenderClient:
                enabled = False

            paths = generate_report(
                db,
                settings,
                report_date="2026-03-10",
                report_type="daily",
                lookback_days=1,
                llm_client=FakeRenderClient(),
                llm_variants=[],
                use_llm_topic_digest=False,
            )

            topic_dir = Path(paths["topic_dir"])
            markdown = (topic_dir / "daily-2026-03-10--matrix_free_fem.md").read_text(encoding="utf-8")
            self.assertIn("ikun_gpt-5.4 主题概览：未生成", markdown)
            self.assertIn("Stored Variant Render Sample", markdown)
            db.close()

    def test_generate_report_adds_fallback_review_when_window_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_POE.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            candidate = PaperCandidate(
                source_name="arxiv",
                source_paper_id="2502.00001",
                query_text="all:\"matrix-free\" AND all:\"finite element\"",
                title="Matrix-Free Fallback Review Sample",
                abstract="This paper studies matrix-free operator evaluation and performance portability for high-order finite elements on GPUs.",
                authors=["Alice"],
                published_at="2026-02-15T09:00:00+08:00",
                updated_at="2026-02-15T09:00:00+08:00",
                primary_url="https://arxiv.org/abs/2502.00001",
                pdf_url="https://arxiv.org/pdf/2502.00001.pdf",
                doi="",
                arxiv_id="2502.00001",
                venue="arXiv",
                year=2026,
                categories=["cs.NA"],
                raw={"fixture": True},
            )
            pipeline._process_candidate(candidate, RunStats())

            class FakeRenderClient:
                enabled = False

            paths = generate_report(
                db,
                settings,
                report_date="2026-03-10",
                report_type="daily",
                lookback_days=1,
                llm_client=FakeRenderClient(),
            )
            topic_dir = Path(paths["topic_dir"])
            markdown = (topic_dir / "daily-2026-03-10--matrix_free_fem.md").read_text(encoding="utf-8")
            report_json = json.loads((root / "exports" / "daily-2026-03-10--matrix_free_fem.json").read_text(encoding="utf-8"))

            self.assertIn("本窗口内没有新的严格命中论文。", markdown)
            self.assertIn("自动补充：近 `90` 天高分回顾", markdown)
            self.assertIn("Matrix-Free Fallback Review Sample", markdown)
            self.assertIn("fallback_reviews", report_json)
            self.assertIn("matrix_free_fem", report_json["fallback_reviews"])
            db.close()

    def test_ai_operator_priority_categories_and_venues_raise_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_POE.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            topic = next(topic for topic in settings.topics if topic.id == "ai_operator_acceleration")
            paper = PaperRecord(
                id=1,
                title="Tensor Core GEMM Optimization for Transformer Inference on GPUs",
                title_norm="tensor core gemm optimization for transformer inference on gpus",
                abstract="We optimize GEMM and matrix multiplication kernels for transformer inference with Tensor Core acceleration.",
                authors=["Alice"],
                published_at="2026-03-10",
                updated_at="2026-03-10",
                primary_url="https://example.com/paper",
                pdf_url="https://example.com/paper.pdf",
                doi="",
                arxiv_id="2905.00001",
                venue="MLSys",
                year=2026,
                categories=["cs.DC", "cs.AR"],
                summary_text="",
                summary_basis="metadata-only",
                tags=[],
                pdf_local_path="",
                pdf_status="pending",
                pdf_downloaded_at=None,
                fulltext_txt_path="",
                fulltext_excerpt="",
                fulltext_status="empty",
                page_count=None,
                llm_summary={},
                analysis_updated_at=None,
                source_first="arxiv",
                created_at="2026-03-10T09:00:00+08:00",
                last_seen_at="2026-03-10T09:00:00+08:00",
                metadata={},
            )

            evaluation = evaluate_paper_against_topic(paper, topic)
            self.assertEqual(evaluation.classification, "relevant")
            self.assertTrue(any("优先 arXiv 分类" in reason for reason in evaluation.reasons))
            self.assertTrue(any("优先 venue 线索" in reason for reason in evaluation.reasons))
            self.assertIn("cs.DC", evaluation.matched_keywords)
            self.assertIn("mlsys", evaluation.matched_keywords)

    def test_enrichment_pipeline_upgrades_summary_with_fulltext_and_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            settings.llm.enabled = True
            settings.llm.base_url = "http://example.invalid/v1"
            settings.llm.model = "dummy-model"

            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            candidate = PaperCandidate(
                source_name="arxiv",
                source_paper_id="2501.00001",
                query_text="all:\"matrix-free\" AND all:\"finite element\"",
                title="Matrix-Free Finite Element Preconditioners with Partial Assembly",
                abstract=(
                    "We present matrix-free finite element preconditioners with partial assembly "
                    "and show strong GPU performance."
                ),
                authors=["Alice"],
                published_at="2026-03-10T09:00:00+08:00",
                updated_at="2026-03-10T09:00:00+08:00",
                primary_url="https://arxiv.org/abs/2501.00001",
                pdf_url="https://arxiv.org/pdf/2501.00001.pdf",
                doi="",
                arxiv_id="2501.00001",
                venue="arXiv",
                year=2026,
                categories=["cs.NA"],
                raw={"fixture": True},
            )
            pipeline._process_candidate(candidate, RunStats())

            class FakeDocumentProcessor:
                def can_try_pdf(self, paper):  # noqa: ANN001
                    return True

                def enrich(self, paper, force=False):  # noqa: ANN001, ARG002
                    pdf_path = settings.enrichment.pdf_dir / "2501.00001.pdf"
                    text_path = settings.enrichment.text_dir / "2501.00001.txt"
                    pdf_path.parent.mkdir(parents=True, exist_ok=True)
                    text_path.parent.mkdir(parents=True, exist_ok=True)
                    pdf_path.write_bytes(b"%PDF-1.4\n")
                    text_content = (
                        "This work introduces a matrix-free finite element implementation with "
                        "partial assembly, multigrid preconditioners, and strong GPU scaling."
                    )
                    text_path.write_text(text_content, encoding="utf-8")
                    return DocumentArtifacts(
                        pdf_local_path=str(pdf_path),
                        pdf_status="downloaded",
                        pdf_downloaded_at="2026-03-10T10:00:00+08:00",
                        fulltext_txt_path=str(text_path),
                        fulltext_excerpt=text_content,
                        fulltext_status="extracted",
                        page_count=12,
                        was_downloaded=True,
                        was_extracted=True,
                    )

            class FakeLLMClient:
                enabled = True

                def generate_summary(self, paper, evaluations):  # noqa: ANN001
                    self.last_title = paper.title
                    self.last_scores = [item.score for item in evaluations]
                    return LLMResult(
                        summary_text="问题：高阶有限元矩阵自由算子成本高。方法：结合 partial assembly 与 multigrid。贡献：GPU 性能提升明显。",
                        summary_basis="llm+fulltext+metadata",
                        tags=["matrix-free", "partial assembly", "multigrid"],
                        structured={
                            "summary": "结合全文节选生成的中文总结。",
                            "tags": ["matrix-free", "partial assembly", "multigrid"],
                        },
                    )

            enrichment_pipeline = EnrichmentPipeline(
                settings,
                db,
                document_processor=FakeDocumentProcessor(),
                llm_client=FakeLLMClient(),
            )
            stats = enrichment_pipeline.run(limit=5, force=False, use_llm=True)
            self.assertEqual(stats.enriched, 1)
            self.assertEqual(stats.downloaded_pdfs, 1)
            self.assertEqual(stats.extracted_texts, 1)
            self.assertEqual(stats.llm_summaries, 1)

            paper = db.get_paper(1)
            self.assertEqual(paper.fulltext_status, "extracted")
            self.assertEqual(paper.summary_basis, "llm+fulltext+metadata")
            self.assertIn("partial assembly", paper.summary_text)
            self.assertEqual(paper.page_count, 12)

            class FakeRenderClient:
                enabled = False

            paths = generate_report(
                db,
                settings,
                report_date="2026-03-10",
                report_type="daily",
                lookback_days=1,
                llm_client=FakeRenderClient(),
            )
            topic_dir = Path(paths["topic_dir"])
            markdown = (topic_dir / "daily-2026-03-10--matrix_free_fem.md").read_text(encoding="utf-8")
            self.assertIn("全文状态", markdown)
            self.assertIn("llm+fulltext+metadata", markdown)

            db.close()

    def test_enrichment_workers_apply_to_full_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            settings.llm.enabled = True
            settings.llm.base_url = "http://example.invalid/v1"
            settings.llm.model = "dummy-model"

            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            for index in (1, 2):
                candidate = PaperCandidate(
                    source_name="arxiv",
                    source_paper_id=f"2501.1000{index}",
                    query_text="all:\"matrix-free\" AND all:\"finite element\"",
                    title=f"Matrix-Free Finite Element Paper {index}",
                    abstract="matrix-free finite element operator application on GPUs",
                    authors=["Alice"],
                    published_at=f"2026-03-10T0{index}:00:00+08:00",
                    updated_at=f"2026-03-10T0{index}:00:00+08:00",
                    primary_url=f"https://arxiv.org/abs/2501.1000{index}",
                    pdf_url=f"https://arxiv.org/pdf/2501.1000{index}.pdf",
                    doi="",
                    arxiv_id=f"2501.1000{index}",
                    venue="arXiv",
                    year=2026,
                    categories=["cs.NA"],
                    raw={"fixture": True},
                )
                pipeline._process_candidate(candidate, RunStats())

            class FakeDocumentProcessor:
                def can_try_pdf(self, paper):  # noqa: ANN001
                    return True

                def enrich(self, paper, force=False):  # noqa: ANN001, ARG002
                    pdf_path = settings.enrichment.pdf_dir / f"{paper.arxiv_id}.pdf"
                    text_path = settings.enrichment.text_dir / f"{paper.arxiv_id}.txt"
                    pdf_path.parent.mkdir(parents=True, exist_ok=True)
                    text_path.parent.mkdir(parents=True, exist_ok=True)
                    pdf_path.write_bytes(b"%PDF-1.4\n")
                    text_path.write_text(
                        f"{paper.title} full text with matrix-free operator application and GPU scaling.",
                        encoding="utf-8",
                    )
                    return DocumentArtifacts(
                        pdf_local_path=str(pdf_path),
                        pdf_status="downloaded",
                        pdf_downloaded_at="2026-03-10T10:00:00+08:00",
                        fulltext_txt_path=str(text_path),
                        fulltext_excerpt=f"{paper.title} full text excerpt",
                        fulltext_status="extracted",
                        page_count=10,
                        was_downloaded=True,
                        was_extracted=True,
                    )

            class FakeLLMClient:
                enabled = True

                def generate_summary(self, paper, evaluations):  # noqa: ANN001
                    return LLMResult(
                        summary_text=f"总结: {paper.title}",
                        summary_basis="llm+fulltext+metadata",
                        tags=["matrix-free", "gpu"],
                        structured={
                            "summary": paper.title,
                            "source_mode": "fulltext_txt",
                            "direct_pdf_status": "request_failed",
                        },
                    )

            enrichment_pipeline = EnrichmentPipeline(
                settings,
                db,
                document_processor=FakeDocumentProcessor(),
                llm_client=FakeLLMClient(),
            )
            stats = enrichment_pipeline.run(limit=10, force=False, use_llm=True, workers=2)

            self.assertEqual(stats.enriched, 2)
            self.assertEqual(stats.downloaded_pdfs, 2)
            self.assertEqual(stats.extracted_texts, 2)
            self.assertEqual(stats.llm_summaries, 2)
            self.assertEqual(db.connection.execute("SELECT COUNT(*) FROM paper_llm_summaries").fetchone()[0], 2)
            db.close()

    def test_enrichment_retries_fallback_variant_but_skips_completed_pdf_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            settings.llm.enabled = True
            settings.llm.base_url = "http://example.invalid/v1"
            settings.llm.model = "dummy-model"

            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)
            pipeline._process_candidate(
                PaperCandidate(
                    source_name="arxiv",
                    source_paper_id="2502.00001",
                    query_text="matrix-free finite element",
                    title="Matrix-Free Retry Sample",
                    abstract="matrix-free finite element operator application on GPUs",
                    authors=["Alice"],
                    published_at="2026-03-10T09:00:00+08:00",
                    updated_at="2026-03-10T09:00:00+08:00",
                    primary_url="https://arxiv.org/abs/2502.00001",
                    pdf_url="https://arxiv.org/pdf/2502.00001.pdf",
                    doi="",
                    arxiv_id="2502.00001",
                    venue="arXiv",
                    year=2026,
                    categories=["cs.DC"],
                    raw={"fixture": True},
                ),
                RunStats(),
            )

            db.upsert_paper_llm_summary(
                1,
                variant_id="ikun",
                variant_label="ikun_gpt-5.4",
                provider="IkunCoding",
                base_url="https://api.ikuncode.cc/v1",
                model="gpt-5.4",
                summary_text="旧的全文回退总结",
                summary_basis="llm+fulltext+metadata",
                tags=["matrix-free"],
                structured={"summary": "旧的全文回退总结", "source_mode": "fulltext_txt", "direct_pdf_status": "request_failed"},
                usage={},
            )
            db.upsert_paper_llm_summary(
                1,
                variant_id="poe",
                variant_label="poe_claude-opus-4.6",
                provider="openai_compatible",
                base_url="https://api.poe.com/v1",
                model="claude-opus-4.6",
                summary_text=(
                    "已有 PDF 直读总结：论文围绕 matrix-free 有限元算子在 GPU 上的加速实现，"
                    "详细说明了问题背景、partial assembly 方法和实验结果。"
                ),
                summary_basis="llm+pdf+metadata",
                tags=["matrix-free"],
                structured={
                    "summary": "论文围绕 matrix-free 有限元算子在 GPU 上的加速实现，详细说明了问题背景、partial assembly 方法和实验结果。",
                    "problem": "高阶有限元算子应用在 GPU 上面临带宽与访存瓶颈。",
                    "method": "结合 partial assembly 与矩阵自由实现。",
                    "application": "GPU 求解场景。",
                    "results": "性能与扩展性均有提升。",
                    "source_mode": "pdf_direct",
                    "direct_pdf_status": "used",
                },
                usage={},
            )

            dummy_client = LLMClient(settings.llm)
            variants = [
                LLMRuntimeVariant(
                    variant_id="ikun",
                    label="ikun_gpt-5.4",
                    provider="IkunCoding",
                    base_url="https://api.ikuncode.cc/v1",
                    model="gpt-5.4",
                    config_path=root / "config" / "config-ikun.json",
                    client=dummy_client,
                ),
                LLMRuntimeVariant(
                    variant_id="poe",
                    label="poe_claude-opus-4.6",
                    provider="openai_compatible",
                    base_url="https://api.poe.com/v1",
                    model="claude-opus-4.6",
                    config_path=root / "config" / "config-poe.json",
                    client=dummy_client,
                ),
            ]

            enrichment_pipeline = EnrichmentPipeline(settings, db, llm_variants=variants)
            refresh_variants = enrichment_pipeline._select_variants_needing_refresh(1, variants, force=False)  # noqa: SLF001

            self.assertEqual([variant.variant_id for variant in refresh_variants], ["ikun"])
            self.assertFalse(
                enrichment_pipeline._should_skip(  # noqa: SLF001
                    db.get_paper(1),
                    force=False,
                    use_llm=True,
                    target_variants=variants,
                )
            )
            db.close()

    def test_enrichment_can_filter_by_created_after(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            settings.llm.variant_id = "config-example-gpt-5-4"
            settings.llm.label = "gpt-5.4"
            settings.llm.model = "gpt-5.4"
            settings.llm.base_url = "https://example.invalid/v1"
            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            first = PaperCandidate(
                source_name="arxiv",
                source_paper_id="2910.00001",
                query_text="FlashAttention",
                title="Older FlashAttention Paper",
                abstract="kernel fusion for transformer inference",
                authors=["Alice"],
                published_at="2026-03-10T09:00:00+08:00",
                updated_at="2026-03-10T09:00:00+08:00",
                primary_url="https://arxiv.org/abs/2910.00001",
                pdf_url="https://arxiv.org/pdf/2910.00001.pdf",
                doi="",
                arxiv_id="2910.00001",
                venue="arXiv",
                year=2026,
                categories=["cs.LG"],
                raw={"fixture": True},
            )
            second = PaperCandidate(
                source_name="arxiv",
                source_paper_id="2910.00002",
                query_text="FlashAttention",
                title="Newer FlashAttention Paper",
                abstract="operator fusion and triton for transformer inference",
                authors=["Bob"],
                published_at="2026-03-10T11:00:00+08:00",
                updated_at="2026-03-10T11:00:00+08:00",
                primary_url="https://arxiv.org/abs/2910.00002",
                pdf_url="https://arxiv.org/pdf/2910.00002.pdf",
                doi="",
                arxiv_id="2910.00002",
                venue="arXiv",
                year=2026,
                categories=["cs.LG"],
                raw={"fixture": True},
            )
            pipeline._process_candidate(first, RunStats())
            pipeline._process_candidate(second, RunStats())

            db.connection.execute(
                "UPDATE papers SET created_at = ? WHERE arxiv_id = ?",
                ("2026-03-10T09:00:00+08:00", "2910.00001"),
            )
            db.connection.execute(
                "UPDATE papers SET created_at = ? WHERE arxiv_id = ?",
                ("2026-03-10T11:00:00+08:00", "2910.00002"),
            )
            db.connection.commit()

            class FakeLLMClient:
                enabled = True

                def __init__(self) -> None:
                    self.titles: list[str] = []

                def generate_summary(self, paper, evaluations):  # noqa: ANN001, ARG002
                    self.titles.append(paper.title)
                    return LLMResult(
                        summary_text=f"总结: {paper.title}",
                        summary_basis="llm+abstract+metadata",
                        tags=["flashattention"],
                        structured={"summary": paper.title},
                    )

            fake_llm = FakeLLMClient()
            enrichment_pipeline = EnrichmentPipeline(settings, db, llm_client=fake_llm)
            stats = enrichment_pipeline.run(
                limit=10,
                created_after="2026-03-10T10:00:00+08:00",
                use_llm=True,
                skip_document_processing=True,
            )

            self.assertEqual(stats.enriched, 1)
            self.assertEqual(stats.llm_summaries, 1)
            self.assertEqual(fake_llm.titles, ["Newer FlashAttention Paper"])
            db.close()

    def test_enrichment_can_retry_only_reference_variant_fallback_or_missing_papers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_POE.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            settings.llm.enabled = True
            settings.llm.base_url = "http://example.invalid/v1"
            settings.llm.model = "dummy-model"

            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            for index in range(1, 4):
                pipeline._process_candidate(
                    PaperCandidate(
                        source_name="arxiv",
                        source_paper_id=f"2503.0000{index}",
                        query_text="matrix-free finite element",
                        title=f"Retry Reference Sample {index}",
                        abstract="matrix-free finite element operator application on GPUs",
                        authors=["Alice"],
                        published_at=f"2026-03-10T0{index}:00:00+08:00",
                        updated_at=f"2026-03-10T0{index}:00:00+08:00",
                        primary_url=f"https://arxiv.org/abs/2503.0000{index}",
                        pdf_url=f"https://arxiv.org/pdf/2503.0000{index}.pdf",
                        doi="",
                        arxiv_id=f"2503.0000{index}",
                        venue="arXiv",
                        year=2026,
                        categories=["cs.DC"],
                        raw={"fixture": True},
                    ),
                    RunStats(),
                )

            db.upsert_paper_llm_summary(
                1,
                variant_id="ikun",
                variant_label="ikun_gpt-5.4",
                provider="IkunCoding",
                base_url="https://api.ikuncode.cc/v1",
                model="gpt-5.4",
                summary_text=(
                    "问题：高阶有限元 matrix-free 算子在 GPU 上面临访存瓶颈。"
                    "方法：论文基于完整 PDF 说明了 partial assembly、kernel fusion 与向量化实现。"
                    "应用：面向高性能科学计算。结果：给出了明确的性能收益与扩展性数据。"
                ),
                summary_basis="llm+pdf+metadata",
                tags=["matrix-free"],
                structured={
                    "summary": "论文基于完整 PDF 解释了 matrix-free 有限元算子在 GPU 上的优化方法和结果。",
                    "problem": "高阶有限元 matrix-free 算子在 GPU 上容易受访存和带宽瓶颈限制。",
                    "method": "结合 partial assembly、kernel fusion 和向量化布局优化实现高效算子应用。",
                    "application": "面向高性能科学计算中的高阶有限元求解与矩阵自由预条件过程。",
                    "results": "给出了可量化的性能收益、吞吐提升和可扩展性实验结果。",
                    "source_mode": "pdf_direct",
                    "direct_pdf_status": "used",
                    "basis": "llm+pdf+metadata",
                },
                usage={},
            )
            db.upsert_paper_llm_summary(
                2,
                variant_id="ikun",
                variant_label="ikun_gpt-5.4",
                provider="IkunCoding",
                base_url="https://api.ikuncode.cc/v1",
                model="gpt-5.4",
                summary_text="回退到全文文本的总结",
                summary_basis="llm+fulltext+metadata",
                tags=["matrix-free"],
                structured={
                    "summary": "回退到全文文本的总结",
                    "source_mode": "fulltext_txt",
                    "direct_pdf_status": "too_large",
                    "basis": "llm+fulltext+metadata",
                },
                usage={},
            )

            class FakePoeClient:
                enabled = True

                def __init__(self) -> None:
                    self.paper_ids: list[int] = []

                def generate_summary(self, paper, evaluations):  # noqa: ANN001, ARG002
                    self.paper_ids.append(paper.id)
                    return LLMResult(
                        summary_text=f"Poe summary for {paper.title}",
                        summary_basis="llm+abstract+metadata",
                        tags=["retry"],
                        structured={"summary": paper.title, "source_mode": "abstract_metadata"},
                    )

            fake_poe = FakePoeClient()
            variants = [
                LLMRuntimeVariant(
                    variant_id="poe",
                    label="poe_claude-opus-4.6",
                    provider="openai_compatible",
                    base_url="https://api.poe.com/v1",
                    model="claude-opus-4.6",
                    config_path=root / "config" / "config-poe.json",
                    client=fake_poe,
                )
            ]
            enrichment_pipeline = EnrichmentPipeline(settings, db, llm_variants=variants)
            candidates = db.fetch_enrichment_candidates(limit=10, classifications=["relevant", "maybe"])

            filtered_incomplete = enrichment_pipeline._filter_candidates_by_reference_variant(  # noqa: SLF001
                candidates,
                reference_variant="ikun",
                retry_status="incomplete",
            )
            filtered_fallback = enrichment_pipeline._filter_candidates_by_reference_variant(  # noqa: SLF001
                candidates,
                reference_variant="ikun_gpt-5.4",
                retry_status="fallback",
            )
            filtered_missing = enrichment_pipeline._filter_candidates_by_reference_variant(  # noqa: SLF001
                candidates,
                reference_variant="ikun",
                retry_status="missing",
            )

            self.assertEqual({paper.id for paper in filtered_incomplete}, {2, 3})
            self.assertEqual({paper.id for paper in filtered_fallback}, {2})
            self.assertEqual({paper.id for paper in filtered_missing}, {3})

            stats = enrichment_pipeline.run(
                limit=10,
                use_llm=True,
                skip_document_processing=True,
                retry_from_variant="ikun",
                retry_from_status="incomplete",
            )

            self.assertEqual(stats.enriched, 2)
            self.assertEqual(stats.llm_summaries, 2)
            self.assertEqual(set(fake_poe.paper_ids), {2, 3})
            db.close()

    def test_llm_summary_reads_fulltext_txt_and_uses_map_reduce(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            settings.llm.enabled = True
            settings.llm.provider = "openai_compatible"
            settings.llm.base_url = "https://example.invalid/v1"
            settings.llm.model = "dummy-model"
            settings.llm.api_key_env = ""
            settings.llm.max_input_chars = 180
            settings.llm.fulltext_chunk_chars = 140
            settings.llm.fulltext_chunk_overlap_chars = 20
            settings.llm.fulltext_max_chunks = 6

            fulltext_path = root / "paper.txt"
            fulltext_path.write_text(
                (
                    (
                        "Introduction. This paper studies matrix-free finite element operators and partial assembly. "
                        "It introduces a new fused operator application path and a cache-aware execution layout. "
                    )
                    * 12
                    + "\n\n"
                    + (
                        "Method. The solver combines matrix-free multigrid preconditioning with GPU-oriented tensor contractions. "
                        "Implementation details cover memory layout, element restriction, and kernel scheduling. "
                    )
                    * 12
                    + "\n\n"
                    + (
                        "Experiments. LATE_SECTION_MARKER. Experiments show strong scaling on GPUs, lower memory traffic, "
                        "and better throughput on high-order cardiac electrophysiology workloads. The paper also discusses ablation studies and solver robustness. "
                    )
                    * 12
                ),
                encoding="utf-8",
            )

            paper = PaperRecord(
                id=1,
                title="Matrix-Free FEM Full Paper",
                title_norm="matrix free fem full paper",
                abstract="A short abstract that should not be the only LLM input.",
                authors=["Alice", "Bob"],
                published_at="2026-03-10T09:00:00+08:00",
                updated_at="2026-03-10T09:00:00+08:00",
                primary_url="https://example.invalid/paper",
                pdf_url="https://example.invalid/paper.pdf",
                doi="",
                arxiv_id="",
                venue="arXiv",
                year=2026,
                categories=["cs.NA"],
                summary_text="",
                summary_basis="metadata-only",
                tags=[],
                pdf_local_path="",
                pdf_status="downloaded",
                pdf_downloaded_at="2026-03-10T10:00:00+08:00",
                fulltext_txt_path=str(fulltext_path),
                fulltext_excerpt="short excerpt without late section marker",
                fulltext_status="extracted",
                page_count=12,
                llm_summary={},
                analysis_updated_at=None,
                source_first="arxiv",
                created_at="2026-03-10T09:00:00+08:00",
                last_seen_at="2026-03-10T09:00:00+08:00",
                metadata={},
            )
            evaluations = [
                type(
                    "Eval",
                    (),
                    {
                        "classification": "relevant",
                        "topic_name": "有限元分析 Matrix-Free 算法优化",
                        "score": 30.0,
                        "matched_keywords": ["matrix-free", "finite element", "multigrid"],
                    },
                )()
            ]

            class InspectLLMClient(LLMClient):
                def __init__(self, config):  # noqa: ANN001
                    super().__init__(config)
                    self.enabled = True
                    self.calls: list[tuple[str, str]] = []

                def _request_text(self, **kwargs):  # noqa: ANN003
                    self.calls.append(("paper_chunk_note", kwargs["user_prompt"]))
                    return (
                        "分块概括：这一部分讨论 matrix-free 与 partial assembly。\n"
                        "关键方法：multigrid preconditioning。\n"
                        "结果/证据：GPU throughput improves。\n"
                        "应用场景：cardiac electrophysiology。\n"
                        "局限：requires high-order discretization。",
                        {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                    )

                def _request_structured_json(self, **kwargs):  # noqa: ANN003
                    self.calls.append((kwargs["schema_name"], kwargs["user_prompt"]))
                    return (
                        {
                            "summary": "论文针对矩阵自由高阶有限元算子求解的性能瓶颈，提出了融合算子应用与 multigrid 预条件的 GPU 实现，并在心脏电生理高阶工作负载上展示了更高吞吐与更低内存流量。",
                            "problem": "高阶 matrix-free FEM 的算子应用和求解开销较高。",
                            "method": "结合 fused operator application、partial assembly 和 multigrid 预条件。",
                            "application": "高阶有限元、GPU 求解、心脏电生理仿真。",
                            "results": "实验显示 GPU 吞吐提升、内存流量下降，并具有较好的强扩展性。",
                            "contributions": ["提出融合算子应用路径", "给出 multigrid GPU 实现"],
                            "limitations": ["依赖高阶离散场景"],
                            "tags": ["matrix-free", "multigrid", "gpu"],
                            "basis": "llm+fulltext+metadata",
                        },
                        {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
                    )

            client = InspectLLMClient(settings.llm)
            result = client.generate_summary(paper, evaluations)

            self.assertIsNotNone(result)
            self.assertEqual(result.summary_basis, "llm+fulltext+metadata")
            self.assertGreaterEqual(len([item for item in client.calls if item[0] == "paper_chunk_note"]), 2)
            self.assertTrue(any("LATE_SECTION_MARKER" in prompt for _, prompt in client.calls))
            self.assertIn("应用：高阶有限元", result.summary_text)
            self.assertGreaterEqual(int(result.structured.get("chunk_count", 0)), 2)

    def test_llm_summary_preserves_abstract_basis_even_when_fulltext_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "config" / "config.json").write_text(FIXTURE_CONFIG_POE.read_text(encoding="utf-8"), encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            settings.llm.enabled = True
            settings.llm.provider = "openai_compatible"
            settings.llm.base_url = "https://example.invalid/v1"
            settings.llm.model = "claude-opus-4.6"
            settings.llm.api_key_env = ""
            settings.llm.pdf_input_mode = "disable"

            fulltext_path = root / "artifacts" / "text" / "paper.txt"
            fulltext_path.parent.mkdir(parents=True, exist_ok=True)
            fulltext_path.write_text("fulltext exists", encoding="utf-8")

            paper = PaperRecord(
                id=1,
                title="Abstract only summary",
                title_norm="abstract only summary",
                abstract="This abstract is relevant.",
                authors=["Alice"],
                published_at="2026-03-01T00:00:00+08:00",
                updated_at="2026-03-01T00:00:00+08:00",
                primary_url="https://example.invalid/paper",
                pdf_url="https://example.invalid/paper.pdf",
                doi="",
                arxiv_id="",
                venue="",
                year=2026,
                categories=["cs.LG"],
                summary_text="",
                summary_basis="",
                tags=[],
                pdf_local_path="",
                pdf_status="missing",
                pdf_downloaded_at=None,
                fulltext_txt_path=str(fulltext_path),
                fulltext_excerpt="fulltext excerpt",
                fulltext_status="extracted",
                page_count=None,
                llm_summary={},
                analysis_updated_at="2026-03-01T00:00:00+08:00",
                source_first="manual",
                created_at="2026-03-01T00:00:00+08:00",
                last_seen_at="2026-03-01T00:00:00+08:00",
                metadata={},
            )
            evaluations = [
                TopicEvaluation(
                    topic_id="ai_operator_acceleration",
                    topic_name="AI 算子加速",
                    score=25,
                    classification="relevant",
                    matched_keywords=["compiler", "deep learning"],
                    reasons=[],
                )
            ]

            class AbstractOnlyClient(LLMClient):
                def _generate_summary_from_fulltext(self, paper, evaluations, fulltext):  # noqa: ANN001, ARG002
                    return None, {}

                def _request_structured_json(self, **kwargs):  # noqa: ANN003
                    return (
                        {
                            "summary": "只基于摘要和元数据生成总结。",
                            "problem": "问题描述。",
                            "method": "方法描述。",
                            "application": "应用描述。",
                            "results": "结果描述。",
                            "contributions": ["贡献"],
                            "limitations": ["局限"],
                            "tags": ["tag"],
                            "source_mode": "abstract_metadata",
                        },
                        {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                    )

            client = AbstractOnlyClient(settings.llm)
            result = client.generate_summary(paper, evaluations)

            self.assertIsNotNone(result)
            self.assertEqual(result.summary_basis, "llm+abstract+metadata")
            self.assertEqual(result.structured.get("source_mode"), "abstract_metadata")

    def test_llm_summary_prefers_direct_pdf_when_backend_supports_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            settings.llm.enabled = True
            settings.llm.provider = "openai_compatible"
            settings.llm.base_url = "https://example.invalid/v1"
            settings.llm.model = "dummy-model"
            settings.llm.api_key_env = ""
            settings.llm.pdf_input_mode = "auto"

            pdf_path = root / "paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n% fake pdf for test\n")
            fulltext_path = root / "paper.txt"
            fulltext_path.write_text("fallback fulltext should not be used", encoding="utf-8")

            paper = PaperRecord(
                id=1,
                title="Direct PDF Preferred Paper",
                title_norm="direct pdf preferred paper",
                abstract="An abstract.",
                authors=["Alice"],
                published_at="2026-03-10T09:00:00+08:00",
                updated_at="2026-03-10T09:00:00+08:00",
                primary_url="https://example.invalid/paper",
                pdf_url="https://example.invalid/paper.pdf",
                doi="",
                arxiv_id="",
                venue="arXiv",
                year=2026,
                categories=["cs.NA"],
                summary_text="",
                summary_basis="metadata-only",
                tags=[],
                pdf_local_path=str(pdf_path),
                pdf_status="downloaded",
                pdf_downloaded_at="2026-03-10T10:00:00+08:00",
                fulltext_txt_path=str(fulltext_path),
                fulltext_excerpt="fallback excerpt",
                fulltext_status="extracted",
                page_count=8,
                llm_summary={},
                analysis_updated_at=None,
                source_first="arxiv",
                created_at="2026-03-10T09:00:00+08:00",
                last_seen_at="2026-03-10T09:00:00+08:00",
                metadata={},
            )
            evaluations = [
                type(
                    "Eval",
                    (),
                    {
                        "classification": "relevant",
                        "topic_name": "有限元分析 Matrix-Free 算法优化",
                        "score": 30.0,
                        "matched_keywords": ["matrix-free", "finite element"],
                    },
                )()
            ]

            class PDFCapableClient(LLMClient):
                def __init__(self, config):  # noqa: ANN001
                    super().__init__(config)
                    self.enabled = True
                    self.pdf_request_count = 0

                def _post_json(self, url, payload, warn_on_error=True):  # noqa: ANN001, ARG002
                    messages = payload.get("messages", [])
                    user_content = messages[0].get("content", []) if messages else []
                    if isinstance(user_content, list):
                        self.pdf_request_count += 1
                        prompt_text = json.dumps(user_content, ensure_ascii=False)
                        if "supported" in prompt_text:
                            return {
                                "choices": [{"message": {"content": "{\"supported\": true}"}}],
                                "usage": {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
                            }
                        return {
                            "choices": [
                                {
                                    "message": {
                                        "content": (
                                            "摘要：模型直接阅读 PDF 后总结了整篇论文，并明确保留了全文上下文。\n"
                                            "问题：论文关注矩阵自由有限元算子效率。\n"
                                            "方法：直接从 PDF 中识别方法与实验。\n"
                                            "应用：高阶有限元与 GPU 求解。\n"
                                            "结果：实验显示吞吐提升。\n"
                                            "贡献：直接 PDF 阅读；保留全文上下文。\n"
                                            "局限：依赖后端支持 PDF。\n"
                                            "标签：matrix-free, gpu"
                                        )
                                    }
                                }
                            ],
                            "usage": {"input_tokens": 120, "output_tokens": 48, "total_tokens": 168},
                        }
                    raise AssertionError("this test should not call non-pdf chat requests")

                def _request_text(self, **kwargs):  # noqa: ANN003
                    raise AssertionError("fulltext fallback should not run when pdf direct is supported")

            client = PDFCapableClient(settings.llm)
            result = client.generate_summary(paper, evaluations)

            self.assertIsNotNone(result)
            self.assertEqual(result.summary_basis, "llm+pdf+metadata")
            self.assertEqual(result.structured.get("source_mode"), "pdf_direct")
            self.assertTrue(result.structured.get("pdf_input_used"))
            self.assertEqual(result.structured.get("pdf_input_strategy"), "chat_file")
            self.assertGreaterEqual(client.pdf_request_count, 2)

    def test_llm_pdf_request_retries_alternate_file_format_on_unknown_parameter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            settings.llm.enabled = True
            settings.llm.provider = "openai_compatible"
            settings.llm.base_url = "https://example.invalid/v1"
            settings.llm.model = "dummy-model"
            settings.llm.api_key_env = ""
            settings.llm.pdf_input_mode = "force"

            pdf_path = root / "paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n% fake pdf for test\n")

            paper = PaperRecord(
                id=1,
                title="Alternate PDF Format Paper",
                title_norm="alternate pdf format paper",
                abstract="An abstract.",
                authors=["Alice"],
                published_at="2026-03-10T09:00:00+08:00",
                updated_at="2026-03-10T09:00:00+08:00",
                primary_url="https://example.invalid/paper",
                pdf_url="https://example.invalid/paper.pdf",
                doi="",
                arxiv_id="",
                venue="arXiv",
                year=2026,
                categories=["cs.NA"],
                summary_text="",
                summary_basis="metadata-only",
                tags=[],
                pdf_local_path=str(pdf_path),
                pdf_status="downloaded",
                pdf_downloaded_at="2026-03-10T10:00:00+08:00",
                fulltext_txt_path="",
                fulltext_excerpt="",
                fulltext_status="empty",
                page_count=8,
                llm_summary={},
                analysis_updated_at=None,
                source_first="arxiv",
                created_at="2026-03-10T09:00:00+08:00",
                last_seen_at="2026-03-10T09:00:00+08:00",
                metadata={},
            )
            evaluations = [
                type(
                    "Eval",
                    (),
                    {
                        "classification": "relevant",
                        "topic_name": "有限元分析 Matrix-Free 算法优化",
                        "score": 30.0,
                        "matched_keywords": ["matrix-free", "finite element"],
                    },
                )()
            ]

            class RetryPDFClient(LLMClient):
                def __init__(self, config):  # noqa: ANN001
                    super().__init__(config)
                    self.enabled = True
                    self.attempts: list[str] = []

                def _post_json(self, url, payload, warn_on_error=True):  # noqa: ANN001, ARG002
                    messages = payload.get("messages", [])
                    user_content = messages[0].get("content", []) if messages else []
                    if not isinstance(user_content, list):
                        return {
                            "choices": [
                                {
                                    "message": {
                                        "content": (
                                            "{\"summary\":\"模型改用 input_file 后成功读取 PDF，并生成了结构化论文总结。\","
                                            "\"problem\":\"论文关注 matrix-free 有限元算子在 GPU 上的性能瓶颈。\","
                                            "\"method\":\"方法结合 partial assembly 与矩阵自由算子实现。\","
                                            "\"application\":\"适用于高阶有限元和 GPU 求解场景。\","
                                            "\"results\":\"实验结果显示吞吐与扩展性均得到提升。\","
                                            "\"contributions\":[\"给出结构化 PDF 直读总结\"],"
                                            "\"limitations\":[\"需要后端支持文件输入\"],\"tags\":[\"matrix-free\"],"
                                            "\"basis\":\"llm+pdf+metadata\"}"
                                        )
                                    }
                                }
                            ],
                            "usage": {"input_tokens": 30, "output_tokens": 20, "total_tokens": 50},
                        }
                    first_item = user_content[0]
                    if isinstance(first_item, dict) and first_item.get("type") == "file":
                        self.attempts.append("chat_file")
                        self._last_request_failure = (
                            "http_400:{\"error\":{\"message\":\"Unknown parameter: 'input[0].content[0].file'.\"}}"
                        )
                        return None
                    if isinstance(first_item, dict) and first_item.get("type") == "input_file":
                        self.attempts.append("chat_input_file")
                        return {
                            "choices": [
                                {
                                    "message": {
                                        "content": (
                                            "{\"summary\":\"模型改用 input_file 后成功读取 PDF。\","
                                            "\"problem\":\"问题。\",\"method\":\"方法。\",\"application\":\"应用。\","
                                            "\"results\":\"结果。\",\"contributions\":[\"贡献\"],"
                                            "\"limitations\":[\"局限\"],\"tags\":[\"matrix-free\"],"
                                            "\"basis\":\"llm+pdf+metadata\"}"
                                        )
                                    }
                                }
                            ],
                            "usage": {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
                        }
                    raise AssertionError("unexpected payload")

                def _request_text(self, **kwargs):  # noqa: ANN003
                    raise AssertionError("fulltext fallback should not run after alternate pdf format succeeds")

            client = RetryPDFClient(settings.llm)
            result = client.generate_summary(paper, evaluations)

            self.assertIsNotNone(result)
            self.assertEqual(client.attempts[:2], ["chat_file", "chat_input_file"])
            self.assertIn("chat_input_file", client.attempts)
            self.assertEqual(result.summary_basis, "llm+pdf+metadata")
            self.assertEqual(result.structured.get("source_mode"), "pdf_direct")
            self.assertEqual(result.structured.get("pdf_input_strategy"), "chat_input_file")

    def test_invalid_direct_pdf_response_is_not_treated_as_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            settings.llm.enabled = True
            settings.llm.provider = "openai_compatible"
            settings.llm.base_url = "https://example.invalid/v1"
            settings.llm.model = "dummy-model"
            settings.llm.api_key_env = ""
            settings.llm.pdf_input_mode = "force"

            pdf_path = root / "paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n% fake pdf for test\n")
            fulltext_path = root / "paper.txt"
            fulltext_path.write_text("fallback fulltext content", encoding="utf-8")

            paper = PaperRecord(
                id=1,
                title="Invalid Direct PDF Response Paper",
                title_norm="invalid direct pdf response paper",
                abstract="An abstract.",
                authors=["Alice"],
                published_at="2026-03-10T09:00:00+08:00",
                updated_at="2026-03-10T09:00:00+08:00",
                primary_url="https://example.invalid/paper",
                pdf_url="https://example.invalid/paper.pdf",
                doi="",
                arxiv_id="",
                venue="arXiv",
                year=2026,
                categories=["cs.NA"],
                summary_text="",
                summary_basis="metadata-only",
                tags=[],
                pdf_local_path=str(pdf_path),
                pdf_status="downloaded",
                pdf_downloaded_at="2026-03-10T10:00:00+08:00",
                fulltext_txt_path=str(fulltext_path),
                fulltext_excerpt="fallback excerpt",
                fulltext_status="extracted",
                page_count=8,
                llm_summary={},
                analysis_updated_at=None,
                source_first="arxiv",
                created_at="2026-03-10T09:00:00+08:00",
                last_seen_at="2026-03-10T09:00:00+08:00",
                metadata={},
            )
            evaluations = [
                type(
                    "Eval",
                    (),
                    {
                        "classification": "relevant",
                        "topic_name": "有限元分析 Matrix-Free 算法优化",
                        "score": 30.0,
                        "matched_keywords": ["matrix-free", "finite element"],
                    },
                )()
            ]

            class InvalidPDFClient(LLMClient):
                def __init__(self, config):  # noqa: ANN001
                    super().__init__(config)
                    self.enabled = True
                    self.fulltext_called = False

                def _post_json(self, url, payload, warn_on_error=True):  # noqa: ANN001, ARG002
                    messages = payload.get("messages", [])
                    user_content = messages[0].get("content", []) if messages else []
                    if isinstance(user_content, list):
                        return {
                            "choices": [
                                {
                                    "message": {
                                        "content": (
                                            "摘要：当前对话中未提供论文全文，因此我无法阅读全文后给出可靠总结。\n"
                                            "问题：缺少论文原文。\n"
                                            "方法：请发送 PDF。\n"
                                            "应用：无。\n"
                                            "结果：无。\n"
                                            "贡献：无。\n"
                                            "局限：缺少原文。\n"
                                            "标签：无"
                                        )
                                    }
                                }
                            ],
                            "usage": {"input_tokens": 80, "output_tokens": 40, "total_tokens": 120},
                        }
                    raise AssertionError("unexpected non-pdf request")

                def _request_text(self, **kwargs):  # noqa: ANN003
                    self.fulltext_called = True
                    return (
                        "分块概括：基于全文文本回退成功。\n"
                        "关键方法：fulltext fallback。\n"
                        "结果/证据：可继续生成结构化总结。\n"
                        "应用场景：测试。\n"
                        "局限：无。",
                        {"input_tokens": 12, "output_tokens": 6, "total_tokens": 18},
                    )

                def _request_structured_json(self, **kwargs):  # noqa: ANN003
                    return (
                        {
                            "summary": "通过全文文本回退生成总结。",
                            "problem": "问题。",
                            "method": "方法。",
                            "application": "应用。",
                            "results": "结果。",
                            "contributions": ["贡献"],
                            "limitations": ["局限"],
                            "tags": ["fallback"],
                            "basis": "llm+fulltext+metadata",
                        },
                        {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
                    )

            client = InvalidPDFClient(settings.llm)
            result = client.generate_summary(paper, evaluations)

            self.assertIsNotNone(result)
            self.assertTrue(client.fulltext_called)
            self.assertEqual(result.structured.get("source_mode"), "fulltext_txt")
            self.assertEqual(result.structured.get("direct_pdf_status"), "invalid_response")
            self.assertTrue(
                looks_like_invalid_direct_pdf_summary(
                    {"summary": "当前对话中未提供论文全文，因此无法总结。"},
                    "当前对话中未提供论文全文，因此无法总结。",
                )
            )

    def test_document_processor_can_infer_arxiv_pdf_from_dblp_metadata(self) -> None:
        settings = load_settings(FIXTURE_CONFIG_IKUN)
        processor = DocumentProcessor(settings.enrichment, "test-agent", settings.timezone)
        paper = PaperRecord(
            id=1,
            title="CoRR Paper",
            title_norm="corr paper",
            abstract="",
            authors=["Alice"],
            published_at="2026-03-10",
            updated_at="2026-03-10",
            primary_url="https://doi.org/10.48550/arXiv.2308.01792",
            pdf_url="",
            doi="10.48550/ARXIV.2308.01792",
            arxiv_id="",
            venue="CoRR",
            year=2023,
            categories=["cs.NA"],
            summary_text="",
            summary_basis="metadata-only",
            tags=[],
            pdf_local_path="",
            pdf_status="pending",
            pdf_downloaded_at=None,
            fulltext_txt_path="",
            fulltext_excerpt="",
            fulltext_status="empty",
            page_count=None,
            llm_summary={},
            analysis_updated_at=None,
            source_first="dblp",
            created_at="2026-03-10T09:00:00+08:00",
            last_seen_at="2026-03-10T09:00:00+08:00",
            metadata={
                "dblp": {
                    "hit": {
                        "info": {
                            "access": "open",
                            "doi": "10.48550/ARXIV.2308.01792",
                            "ee": "https://doi.org/10.48550/arXiv.2308.01792",
                        }
                    }
                }
            },
        )

        self.assertEqual(processor.resolve_pdf_url(paper), "https://arxiv.org/pdf/2308.01792.pdf")

    def test_document_processor_can_infer_pdf_from_landing_page_meta(self) -> None:
        settings = load_settings(FIXTURE_CONFIG_IKUN)
        processor = DocumentProcessor(settings.enrichment, "test-agent", settings.timezone)
        paper = PaperRecord(
            id=1,
            title="Landing page paper",
            title_norm="landing page paper",
            abstract="",
            authors=["Alice"],
            published_at="2026-03-10",
            updated_at="2026-03-10",
            primary_url="https://example.invalid/paper",
            pdf_url="",
            doi="",
            arxiv_id="",
            venue="USENIX",
            year=2026,
            categories=["cs.PF"],
            summary_text="",
            summary_basis="metadata-only",
            tags=[],
            pdf_local_path="",
            pdf_status="pending",
            pdf_downloaded_at=None,
            fulltext_txt_path="",
            fulltext_excerpt="",
            fulltext_status="empty",
            page_count=None,
            llm_summary={},
            analysis_updated_at=None,
            source_first="manual",
            created_at="2026-03-10T09:00:00+08:00",
            last_seen_at="2026-03-10T09:00:00+08:00",
            metadata={},
        )

        html = (
            '<html><head>'
            '<meta name="citation_pdf_url" content="https://example.invalid/files/paper.pdf" />'
            "</head><body></body></html>"
        ).encode("utf-8")

        class FakeResponse:
            def __init__(self, body: bytes) -> None:
                self._body = body
                self.headers = {"Content-Type": "text/html; charset=utf-8"}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, n: int = -1) -> bytes:
                return self._body if n < 0 else self._body[:n]

            def geturl(self) -> str:
                return "https://example.invalid/paper"

        with mock.patch("paper_monitor.enrichment.urllib.request.urlopen", return_value=FakeResponse(html)):
            self.assertEqual(processor.resolve_pdf_url(paper), "https://example.invalid/files/paper.pdf")

    def test_document_processor_can_infer_pdf_from_semantic_scholar_details(self) -> None:
        settings = load_settings(FIXTURE_CONFIG_IKUN)
        processor = DocumentProcessor(settings.enrichment, "test-agent", settings.timezone)
        paper = PaperRecord(
            id=1,
            title="Semantic Scholar DOI paper",
            title_norm="semantic scholar doi paper",
            abstract="",
            authors=["Alice"],
            published_at="2026-03-10",
            updated_at="2026-03-10",
            primary_url="https://doi.org/10.1000/example",
            pdf_url="",
            doi="10.1000/example",
            arxiv_id="",
            venue="Conference",
            year=2026,
            categories=["cs.PF"],
            summary_text="",
            summary_basis="metadata-only",
            tags=[],
            pdf_local_path="",
            pdf_status="pending",
            pdf_downloaded_at=None,
            fulltext_txt_path="",
            fulltext_excerpt="",
            fulltext_status="empty",
            page_count=None,
            llm_summary={},
            analysis_updated_at=None,
            source_first="manual",
            created_at="2026-03-10T09:00:00+08:00",
            last_seen_at="2026-03-10T09:00:00+08:00",
            metadata={},
        )

        payload = json.dumps(
            {
                "title": "Semantic Scholar DOI paper",
                "openAccessPdf": {"url": "https://downloads.example.invalid/paper.pdf"},
            }
        ).encode("utf-8")

        class FakeResponse:
            def __init__(self, body: bytes) -> None:
                self._body = body
                self.headers = {"Content-Type": "application/json"}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, n: int = -1) -> bytes:
                return self._body if n < 0 else self._body[:n]

            def geturl(self) -> str:
                return "https://api.semanticscholar.org/graph/v1/paper/DOI%3A10.1000%2Fexample"

        with mock.patch("paper_monitor.enrichment.urllib.request.urlopen", return_value=FakeResponse(payload)):
            self.assertEqual(processor.resolve_pdf_url(paper), "https://downloads.example.invalid/paper.pdf")

    def test_document_processor_retries_semantic_scholar_http_429(self) -> None:
        settings = load_settings(FIXTURE_CONFIG_IKUN)
        processor = DocumentProcessor(settings.enrichment, "test-agent", settings.timezone)

        payload = json.dumps(
            {
                "title": "Retry DOI paper",
                "openAccessPdf": {"url": "https://downloads.example.invalid/retry.pdf"},
            }
        ).encode("utf-8")

        class FakeResponse:
            def __init__(self, body: bytes) -> None:
                self._body = body
                self.headers = {"Content-Type": "application/json"}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, n: int = -1) -> bytes:
                return self._body if n < 0 else self._body[:n]

        calls = [
            urllib.error.HTTPError(
                "https://api.semanticscholar.org/graph/v1/paper/DOI%3A10.1000%2Fretry",
                429,
                "Too Many Requests",
                hdrs=None,
                fp=io.BytesIO(b"{}"),
            ),
            FakeResponse(payload),
        ]

        with mock.patch("paper_monitor.enrichment.urllib.request.urlopen", side_effect=calls), mock.patch(
            "paper_monitor.enrichment.time.sleep"
        ):
            result = processor._fetch_semantic_scholar_details("DOI:10.1000/retry", "title,openAccessPdf")

        self.assertEqual((result.get("openAccessPdf") or {}).get("url"), "https://downloads.example.invalid/retry.pdf")

    def test_document_processor_returns_no_pdf_when_download_url_is_forbidden(self) -> None:
        settings = load_settings(FIXTURE_CONFIG_IKUN)
        processor = DocumentProcessor(settings.enrichment, "test-agent", settings.timezone)
        paper = PaperRecord(
            id=1,
            title="Forbidden PDF paper",
            title_norm="forbidden pdf paper",
            abstract="",
            authors=["Alice"],
            published_at="2026-03-10",
            updated_at="2026-03-10",
            primary_url="https://example.invalid/paper",
            pdf_url="https://downloads.example.invalid/forbidden.pdf",
            doi="",
            arxiv_id="",
            venue="Conference",
            year=2026,
            categories=["cs.PF"],
            summary_text="",
            summary_basis="metadata-only",
            tags=[],
            pdf_local_path="",
            pdf_status="pending",
            pdf_downloaded_at=None,
            fulltext_txt_path="",
            fulltext_excerpt="",
            fulltext_status="empty",
            page_count=None,
            llm_summary={},
            analysis_updated_at=None,
            source_first="manual",
            created_at="2026-03-10T09:00:00+08:00",
            last_seen_at="2026-03-10T09:00:00+08:00",
            metadata={},
        )

        forbidden = urllib.error.HTTPError(
            paper.pdf_url,
            403,
            "Forbidden",
            hdrs=None,
            fp=io.BytesIO(b""),
        )

        with mock.patch("paper_monitor.enrichment.urllib.request.urlopen", side_effect=forbidden):
            artifacts = processor.enrich(paper, force=True)

        self.assertEqual(artifacts.pdf_status, "no-pdf")
        self.assertEqual(artifacts.fulltext_status, "empty")

    def test_parse_pdf_brief_summary_extracts_labeled_sections(self) -> None:
        settings = load_settings(FIXTURE_CONFIG_POE)
        client = LLMClient(settings.llm)
        parsed = client._parse_pdf_brief_summary(
            "摘要：这是一段摘要。\n"
            "问题：问题描述。\n"
            "方法：方法描述。\n"
            "应用：应用描述。\n"
            "结果：结果描述。\n"
            "贡献：贡献一；贡献二。\n"
            "局限：局限一；局限二。\n"
            "标签：tag-a, tag-b"
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["basis"], "llm+pdf+metadata")
        self.assertEqual(parsed["problem"], "问题描述。")
        self.assertEqual(parsed["contributions"], ["贡献一", "贡献二。"])
        self.assertEqual(parsed["tags"], ["tag-a", "tag-b"])

    def test_secondary_variants_can_be_restricted_to_priority_papers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "config" / "config.json").write_text(FIXTURE_CONFIG_POE.read_text(encoding="utf-8"), encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            db = Database(settings.database_path, settings.timezone)
            db.initialize()

            topic = next(item for item in settings.topics if item.id == "ai_operator_acceleration")
            candidates = []
            for index, score in enumerate([31.0, 22.0, 18.0], start=1):
                candidate = PaperCandidate(
                    source_name="manual",
                    source_paper_id=f"manual-{index}",
                    query_text="manual",
                    title=f"Priority Paper {index}",
                    abstract="kernel fusion and tensor compiler for deep learning inference",
                    authors=["Alice"],
                    published_at=f"2026-03-{10+index:02d}",
                    updated_at=f"2026-03-{10+index:02d}",
                    primary_url=f"https://example.com/{index}",
                    pdf_url="",
                    doi="",
                    arxiv_id="",
                    venue="MLSys",
                    year=2026,
                    categories=["cs.LG"],
                    raw={"fixture": True},
                )
                paper_id, _ = db.upsert_paper(candidate)
                db.upsert_match(
                    paper_id,
                    TopicEvaluation(
                        topic_id=topic.id,
                        topic_name=topic.display_name,
                        score=score,
                        classification="relevant",
                        matched_keywords=["kernel fusion"],
                        reasons=["fixture"],
                    ),
                )
                candidates.append(db.get_paper(paper_id))

            class DummyClient:
                def __init__(self, enabled=True):  # noqa: ANN001
                    self.enabled = enabled

            variants = [
                LLMRuntimeVariant(
                    variant_id="poe",
                    label="poe",
                    provider="openai_compatible",
                    base_url="https://example.invalid/v1",
                    model="gpt-5.4",
                    config_path=FIXTURE_CONFIG_POE,
                    client=DummyClient(),
                ),
                LLMRuntimeVariant(
                    variant_id="ikun",
                    label="ikun",
                    provider="IkunCoding",
                    base_url="https://example.invalid/v1",
                    model="gpt-5.4",
                    config_path=FIXTURE_CONFIG_IKUN,
                    client=DummyClient(),
                ),
            ]

            pipeline = EnrichmentPipeline(settings, db, llm_variants=variants)
            target_map = pipeline._build_target_variants_map(  # noqa: SLF001
                candidates,
                use_llm=True,
                secondary_priority_only=True,
                secondary_top_per_topic=1,
                secondary_min_score=30.0,
            )

            self.assertEqual([variant.variant_id for variant in target_map[candidates[0].id]], ["poe", "ikun"])
            self.assertEqual([variant.variant_id for variant in target_map[candidates[1].id]], ["poe"])
            self.assertEqual([variant.variant_id for variant in target_map[candidates[2].id]], ["poe"])
            db.close()

    def test_find_papers_supports_title_and_doi_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "config" / "config.json").write_text(FIXTURE_CONFIG_POE.read_text(encoding="utf-8"), encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            db = Database(settings.database_path, settings.timezone)
            db.initialize()

            candidate = PaperCandidate(
                source_name="manual",
                source_paper_id="manual-find-1",
                query_text="manual",
                title="Manual Tensor Core Study",
                abstract="tensor core optimization",
                authors=["Alice"],
                published_at="2026-03-16",
                updated_at="2026-03-16",
                primary_url="https://example.com/manual-study",
                pdf_url="https://example.com/manual-study.pdf",
                doi="10.1000/manual-study",
                arxiv_id="",
                venue="manual",
                year=2026,
                categories=["cs.LG"],
                raw={"fixture": True},
            )
            paper_id, _ = db.upsert_paper(candidate)

            by_title = db.find_papers(title_substring="tensor core", limit=10)
            by_doi = db.find_papers(doi="10.1000/manual-study", limit=10)

            self.assertEqual([paper.id for paper in by_title], [paper_id])
            self.assertEqual([paper.id for paper in by_doi], [paper_id])
            db.close()

    def test_llm_summary_falls_back_to_fulltext_when_pdf_direct_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            settings.llm.enabled = True
            settings.llm.provider = "openai_compatible"
            settings.llm.base_url = "https://example.invalid/v1"
            settings.llm.model = "dummy-model"
            settings.llm.api_key_env = ""
            settings.llm.fulltext_chunk_chars = 120
            settings.llm.fulltext_chunk_overlap_chars = 10

            pdf_path = root / "paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n% fake pdf for test\n")
            fulltext_path = root / "paper.txt"
            fulltext_path.write_text(
                "This full text should be used when the backend cannot read PDF directly. " * 10,
                encoding="utf-8",
            )

            paper = PaperRecord(
                id=1,
                title="Fallback Fulltext Paper",
                title_norm="fallback fulltext paper",
                abstract="An abstract.",
                authors=["Alice"],
                published_at="2026-03-10T09:00:00+08:00",
                updated_at="2026-03-10T09:00:00+08:00",
                primary_url="https://example.invalid/paper",
                pdf_url="https://example.invalid/paper.pdf",
                doi="",
                arxiv_id="",
                venue="arXiv",
                year=2026,
                categories=["cs.NA"],
                summary_text="",
                summary_basis="metadata-only",
                tags=[],
                pdf_local_path=str(pdf_path),
                pdf_status="downloaded",
                pdf_downloaded_at="2026-03-10T10:00:00+08:00",
                fulltext_txt_path=str(fulltext_path),
                fulltext_excerpt="fallback excerpt",
                fulltext_status="extracted",
                page_count=8,
                llm_summary={},
                analysis_updated_at=None,
                source_first="arxiv",
                created_at="2026-03-10T09:00:00+08:00",
                last_seen_at="2026-03-10T09:00:00+08:00",
                metadata={},
            )
            evaluations = [
                type(
                    "Eval",
                    (),
                    {
                        "classification": "relevant",
                        "topic_name": "有限元分析 Matrix-Free 算法优化",
                        "score": 30.0,
                        "matched_keywords": ["matrix-free", "finite element"],
                    },
                )()
            ]

            class PDFFallbackClient(LLMClient):
                def __init__(self, config):  # noqa: ANN001
                    super().__init__(config)
                    self.enabled = True
                    self.pdf_probe_attempts = 0
                    self.chunk_calls = 0

                def _post_json(self, url, payload, warn_on_error=True):  # noqa: ANN001, ARG002
                    messages = payload.get("messages", [])
                    user_content = messages[0].get("content", []) if messages else []
                    if isinstance(user_content, list):
                        self.pdf_probe_attempts += 1
                        return None
                    return None

                def _request_text(self, **kwargs):  # noqa: ANN003
                    self.chunk_calls += 1
                    return (
                        "分块概括：该分块来自完整抽取全文。\n"
                        "关键方法：matrix-free operator。\n"
                        "结果/证据：吞吐提升。\n"
                        "应用场景：高阶有限元。\n"
                        "局限：依赖特定离散。",
                        {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                    )

                def _request_structured_json(self, **kwargs):  # noqa: ANN003
                    return (
                        {
                            "summary": "在 PDF 不可直读时，回退到完整抽取全文并完成总结。",
                            "problem": "论文关注 matrix-free 效率问题。",
                            "method": "基于完整抽取全文做分块归纳。",
                            "application": "高阶有限元。",
                            "results": "性能提升。",
                            "contributions": ["回退逻辑生效"],
                            "limitations": ["依赖文本抽取质量"],
                            "tags": ["matrix-free"],
                            "basis": "llm+fulltext+metadata",
                        },
                        {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
                    )

            client = PDFFallbackClient(settings.llm)
            result = client.generate_summary(paper, evaluations)

            self.assertIsNotNone(result)
            self.assertEqual(result.summary_basis, "llm+fulltext+metadata")
            self.assertEqual(result.structured.get("source_mode"), "fulltext_txt")
            self.assertIn(result.structured.get("direct_pdf_status"), {"unsupported", "request_failed"})
            self.assertGreaterEqual(client.pdf_probe_attempts, 1)
            self.assertGreaterEqual(client.chunk_calls, 1)

    def test_ikun_chat_payload_includes_reasoning_effort(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "config" / "config.json").write_text(FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8"), encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            settings.llm.enabled = True
            settings.llm.provider = "openai_compatible"
            settings.llm.base_url = "https://example.invalid/v1"
            settings.llm.model = "gpt-5.4"
            settings.llm.api_key_env = ""
            settings.llm.model_reasoning_effort = "xhigh"

            class CaptureClient(LLMClient):
                def __init__(self, config):  # noqa: ANN001
                    super().__init__(config)
                    self.payloads: list[dict] = []

                def _post_json(self, url, payload, warn_on_error=True):  # noqa: ANN001, ARG002
                    self.payloads.append(payload)
                    return {
                        "choices": [{"message": {"content": "ok"}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    }

            client = CaptureClient(settings.llm)
            text, _ = client._post_chat_completions_text("sys", "user", task_name="paper_summary")

            self.assertEqual(text, "ok")
            self.assertEqual(client.payloads[0].get("reasoning_effort"), "xhigh")

    def test_poe_gemini_payload_maps_reasoning_to_thinking_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "config" / "config.json").write_text(FIXTURE_CONFIG_POE.read_text(encoding="utf-8"), encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            settings.llm.enabled = True
            settings.llm.api_key_env = ""
            settings.llm.model = "gemini-3.1-pro"
            settings.llm.extra_body = {"reasoning_effort": "xhigh"}

            class CaptureClient(LLMClient):
                def __init__(self, config):  # noqa: ANN001
                    super().__init__(config)
                    self.payloads: list[dict] = []

                def _post_json(self, url, payload, warn_on_error=True):  # noqa: ANN001, ARG002
                    self.payloads.append(payload)
                    return {
                        "choices": [{"message": {"content": "ok"}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    }

            client = CaptureClient(settings.llm)
            text, _ = client._post_chat_completions_text("sys", "user", task_name="paper_summary")

            self.assertEqual(text, "ok")
            self.assertEqual(client.payloads[0].get("extra_body", {}).get("reasoning_effort"), "xhigh")
            self.assertEqual(client.payloads[0].get("extra_body", {}).get("thinking_level"), "high")
            self.assertNotIn("reasoning_effort", client.payloads[0])

    def test_poe_claude_payload_uses_output_effort(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "config" / "config.json").write_text(FIXTURE_CONFIG_POE.read_text(encoding="utf-8"), encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            settings.llm.enabled = True
            settings.llm.api_key_env = ""
            settings.llm.model = "claude-opus-4.6"
            settings.llm.model_reasoning_effort = "xhigh"
            settings.llm.model_output_effort = "max"
            settings.llm.output_effort_by_task = {"paper_summary": "max"}
            settings.llm.extra_body = {}

            class CaptureClient(LLMClient):
                def __init__(self, config):  # noqa: ANN001
                    super().__init__(config)
                    self.payloads: list[dict] = []

                def _post_json(self, url, payload, warn_on_error=True):  # noqa: ANN001, ARG002
                    self.payloads.append(payload)
                    return {
                        "choices": [{"message": {"content": "ok"}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    }

            client = CaptureClient(settings.llm)
            text, _ = client._post_chat_completions_text("sys", "user", task_name="paper_summary")

            self.assertEqual(text, "ok")
            extra_body = client.payloads[0].get("extra_body", {})
            self.assertEqual(extra_body.get("output_effort"), "max")
            self.assertNotIn("reasoning_effort", extra_body)
            self.assertNotIn("reasoning_effort", client.payloads[0])

    def test_poe_claude_preserves_low_output_effort(self) -> None:
        settings = load_settings(FIXTURE_CONFIG_POE)
        settings.llm.enabled = True
        settings.llm.api_key_env = ""
        settings.llm.model = "claude-opus-4.6"
        settings.llm.output_effort_by_task = {"paper_chunk": "low"}
        settings.llm.model_output_effort = "max"

        class CaptureClient(LLMClient):
            def __init__(self, config):  # noqa: ANN001
                super().__init__(config)
                self.payloads: list[dict] = []

            def _post_json(self, url, payload, warn_on_error=True):  # noqa: ANN001, ARG002
                self.payloads.append(payload)
                return {
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }

        client = CaptureClient(settings.llm)
        text, _ = client._post_chat_completions_text("sys", "user", task_name="paper_chunk")

        self.assertEqual(text, "ok")
        self.assertEqual(client.payloads[0].get("extra_body", {}).get("output_effort"), "low")

    def test_poe_claude_enforces_min_output_budget_when_output_effort_is_enabled(self) -> None:
        settings = load_settings(FIXTURE_CONFIG_POE)
        settings.llm.enabled = True
        settings.llm.model = "claude-opus-4.6"
        settings.llm.model_output_effort = "max"
        settings.llm.output_effort_by_task = {"paper_chunk": "low"}
        settings.llm.max_output_tokens_by_task = {"paper_chunk": 800}

        client = LLMClient(settings.llm)

        self.assertEqual(client._max_output_tokens_for_task("paper_chunk"), 1200)  # noqa: SLF001
        self.assertEqual(client._max_output_tokens_for_task("paper_chunk", maximum=900), 1200)  # noqa: SLF001

    def test_poe_claude_payload_raises_explicit_low_token_budget_to_minimum(self) -> None:
        settings = load_settings(FIXTURE_CONFIG_POE)
        settings.llm.enabled = True
        settings.llm.api_key_env = ""
        settings.llm.model = "claude-opus-4.6"
        settings.llm.model_output_effort = "max"
        settings.llm.output_effort_by_task = {"paper_summary": "max"}

        class CaptureClient(LLMClient):
            def __init__(self, config):  # noqa: ANN001
                super().__init__(config)
                self.payloads: list[dict] = []

            def _post_json(self, url, payload, warn_on_error=True):  # noqa: ANN001, ARG002
                self.payloads.append(payload)
                return {
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }

        client = CaptureClient(settings.llm)
        text, _ = client._post_chat_completions_text(
            "sys",
            "user",
            max_output_tokens=800,
            task_name="paper_summary",
        )

        self.assertEqual(text, "ok")
        self.assertEqual(client.payloads[0].get("max_tokens"), 1200)

    def test_poe_claude_none_output_effort_does_not_raise_token_budget(self) -> None:
        settings = load_settings(FIXTURE_CONFIG_POE)
        settings.llm.enabled = True
        settings.llm.api_key_env = ""
        settings.llm.model = "claude-opus-4.6"
        settings.llm.model_output_effort = ""
        settings.llm.output_effort_by_task = {"paper_chunk": "none"}

        class CaptureClient(LLMClient):
            def __init__(self, config):  # noqa: ANN001
                super().__init__(config)
                self.payloads: list[dict] = []

            def _post_json(self, url, payload, warn_on_error=True):  # noqa: ANN001, ARG002
                self.payloads.append(payload)
                return {
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }

        client = CaptureClient(settings.llm)
        text, _ = client._post_chat_completions_text(
            "sys",
            "user",
            max_output_tokens=800,
            task_name="paper_chunk",
        )

        self.assertEqual(text, "ok")
        self.assertEqual(client.payloads[0].get("extra_body", {}).get("output_effort"), "none")
        self.assertEqual(client.payloads[0].get("max_tokens"), 800)

    def test_post_json_retries_http_524(self) -> None:
        settings = load_settings(FIXTURE_CONFIG_IKUN)
        settings.llm.enabled = True
        settings.llm.base_url = "https://example.invalid/v1"
        settings.llm.model = "gpt-5.4"
        settings.llm.api_key_env = ""
        client = LLMClient(settings.llm)

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"ok": true}'

        transient_524 = urllib.error.HTTPError(
            url="https://example.invalid/v1/chat/completions",
            code=524,
            msg="upstream timeout",
            hdrs=None,
            fp=io.BytesIO(b"error code: 524"),
        )

        with mock.patch(
            "paper_monitor.llm.urllib.request.urlopen",
            side_effect=[transient_524, FakeResponse()],
        ) as mocked_urlopen, mock.patch("paper_monitor.llm.time.sleep", return_value=None):
            result = client._post_json(
                "https://example.invalid/v1/chat/completions",
                {"model": "gpt-5.4", "messages": [{"role": "user", "content": "hello"}]},
                warn_on_error=False,
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(mocked_urlopen.call_count, 2)
        self.assertEqual(client._last_request_failure, "")

    def test_eof_failure_retries_alternate_pdf_strategy(self) -> None:
        settings = load_settings(FIXTURE_CONFIG_POE)
        settings.llm.enabled = True
        settings.llm.api_key_env = ""
        settings.llm.model = "claude-opus-4.6"
        client = LLMClient(settings.llm)

        self.assertTrue(
            client._should_retry_with_alternate_pdf_strategy(  # noqa: SLF001
                "<urlopen error [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1000)>"
            )
        )

    def test_topic_digest_can_override_reasoning_by_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "config" / "config.json").write_text(FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8"), encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            settings.llm.enabled = True
            settings.llm.provider = "openai_compatible"
            settings.llm.base_url = "https://example.invalid/v1"
            settings.llm.model = "gpt-5.4"
            settings.llm.api_key_env = ""
            settings.llm.enable_topic_digest = True
            settings.llm.model_reasoning_effort = "low"
            settings.llm.reasoning_by_task = {TASK_TOPIC_DIGEST: "xhigh"}

            paper = PaperRecord(
                id=1,
                title="Test Paper",
                title_norm="test paper",
                abstract="abstract",
                authors=["Alice"],
                published_at="2026-03-10T09:00:00+08:00",
                updated_at="2026-03-10T09:00:00+08:00",
                primary_url="https://example.invalid/paper",
                pdf_url="",
                doi="",
                arxiv_id="",
                venue="MLSys",
                year=2026,
                categories=["cs.LG"],
                summary_text="默认总结",
                summary_basis="llm+metadata",
                tags=["gemm"],
                pdf_local_path="",
                pdf_status="",
                pdf_downloaded_at=None,
                fulltext_txt_path="",
                fulltext_excerpt="",
                fulltext_status="",
                page_count=None,
                llm_summary={},
                analysis_updated_at=None,
                source_first="arxiv",
                created_at="2026-03-10T09:00:00+08:00",
                last_seen_at="2026-03-10T09:00:00+08:00",
                metadata={},
            )
            entry = ReportEntry(
                topic_id="ai_operator_acceleration",
                topic_name="AI 算子加速",
                score=30.0,
                classification="relevant",
                matched_keywords=["gemm", "tensor core"],
                reasons=["fixture"],
                paper=paper,
                source_names=["arxiv"],
                source_urls=["https://example.invalid/paper"],
            )

            class CaptureClient(LLMClient):
                def __init__(self, config):  # noqa: ANN001
                    super().__init__(config)
                    self.payloads: list[dict] = []

                def _post_json(self, url, payload, warn_on_error=True):  # noqa: ANN001, ARG002
                    self.payloads.append(payload)
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "topic_digest": {
                                                "overview": "概览",
                                                "highlights": ["亮点"],
                                                "watchlist": ["观察点"],
                                                "tags": ["gemm"],
                                            }
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    }

            client = CaptureClient(settings.llm)
            digest = client.generate_topic_digest("AI 算子加速", "desc", [entry])

            self.assertIsNotNone(digest)
            self.assertEqual(client.payloads[0].get("reasoning_effort"), "xhigh")

    def test_enrichment_candidates_prioritize_missing_llm_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            settings.llm.variant_id = "config-example-gpt-5-4"
            settings.llm.label = "gpt-5.4"
            settings.llm.model = "gpt-5.4"
            settings.llm.base_url = "https://example.invalid/v1"
            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            first = PaperCandidate(
                source_name="arxiv",
                source_paper_id="2801.00001",
                query_text="FlashAttention",
                title="FlashAttention Kernel Fusion for Incremental Scheduling",
                abstract="kernel fusion and compiler scheduling for transformer inference",
                authors=["Alice"],
                published_at="2026-03-10T09:00:00+08:00",
                updated_at="2026-03-10T09:00:00+08:00",
                primary_url="https://arxiv.org/abs/2801.00001",
                pdf_url="https://arxiv.org/pdf/2801.00001.pdf",
                doi="",
                arxiv_id="2801.00001",
                venue="arXiv",
                year=2026,
                categories=["cs.LG"],
                raw={"fixture": True},
            )
            second = PaperCandidate(
                source_name="arxiv",
                source_paper_id="2801.00002",
                query_text="FlashAttention",
                title="FlashAttention Kernel Fusion for Existing Papers",
                abstract="kernel fusion and compiler scheduling for transformer inference",
                authors=["Bob"],
                published_at="2026-03-10T09:00:00+08:00",
                updated_at="2026-03-10T09:00:00+08:00",
                primary_url="https://arxiv.org/abs/2801.00002",
                pdf_url="https://arxiv.org/pdf/2801.00002.pdf",
                doi="",
                arxiv_id="2801.00002",
                venue="arXiv",
                year=2026,
                categories=["cs.LG"],
                raw={"fixture": True},
            )
            pipeline._process_candidate(first, RunStats())
            pipeline._process_candidate(second, RunStats())
            db.upsert_paper_llm_summary(
                2,
                variant_id="config-example-gpt-5-4",
                variant_label="gpt-5.4",
                provider="openai_responses",
                base_url="https://example.invalid/v1",
                model="gpt-5.4",
                summary_text="已有摘要",
                summary_basis="llm+abstract+metadata",
                tags=["flashattention"],
                structured={"summary": "已有摘要"},
                usage={},
            )

            ordered = db.fetch_enrichment_candidates(limit=2, classifications=["relevant", "maybe"])
            self.assertEqual(ordered[0].id, 1)
            self.assertEqual(ordered[1].id, 2)

            db.close()

    def test_fetch_paper_llm_summaries_batches_large_id_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            db = Database(settings.database_path, settings.timezone)
            db.initialize()

            for paper_id in range(1, 251):
                db.connection.execute(
                    """
                    INSERT INTO papers (
                        id, title, title_norm, abstract, authors_json, published_at, updated_at,
                        primary_url, pdf_url, doi, arxiv_id, venue, year, categories_json,
                        summary_text, summary_basis, tags_json, metadata_json, source_first,
                        created_at, last_seen_at
                    ) VALUES (?, ?, ?, '', '[]', ?, ?, '', '', '', '', 'arXiv', 2026, '[]', '', 'metadata-only', '[]', '{}', 'arxiv', ?, ?)
                    """,
                    (
                        paper_id,
                        f"Paper {paper_id}",
                        f"paper {paper_id}",
                        "2026-03-10T09:00:00+08:00",
                        "2026-03-10T09:00:00+08:00",
                        "2026-03-10T09:00:00+08:00",
                        "2026-03-10T09:00:00+08:00",
                    ),
                )
                db.upsert_paper_llm_summary(
                    paper_id,
                    variant_id="config-example-gpt-5-4",
                    variant_label="gpt-5.4",
                    provider="openai_responses",
                    base_url="https://example.invalid/v1",
                    model="gpt-5.4",
                    summary_text=f"Summary {paper_id}",
                    summary_basis="llm+abstract+metadata",
                    tags=["tag"],
                    structured={"summary": f"Summary {paper_id}"},
                    usage={},
                )

            summaries = db.fetch_paper_llm_summaries(list(range(1, 251)))
            self.assertEqual(len(summaries), 250)
            self.assertEqual(summaries[250][0].summary_text, "Summary 250")
            db.close()

    def test_report_includes_topic_digest_when_llm_client_is_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            settings.llm.enable_topic_digest = True
            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            pipeline._process_candidate(
                PaperCandidate(
                    source_name="arxiv",
                    source_paper_id="2601.00001",
                    query_text="FlashAttention",
                    title="FlashAttention Kernel Fusion for Transformer Training",
                    abstract="We study kernel fusion and Triton code generation for transformer training.",
                    authors=["Alice"],
                    published_at="2026-03-10T09:00:00+08:00",
                    updated_at="2026-03-10T09:00:00+08:00",
                    primary_url="https://arxiv.org/abs/2601.00001",
                    pdf_url="https://arxiv.org/pdf/2601.00001.pdf",
                    doi="",
                    arxiv_id="2601.00001",
                    venue="arXiv",
                    year=2026,
                    categories=["cs.LG"],
                    raw={"fixture": True},
                ),
                RunStats(),
            )
            db.upsert_paper_llm_summary(
                1,
                variant_id=settings.llm.variant_id,
                variant_label=settings.llm.label,
                provider=settings.llm.provider,
                base_url=settings.llm.base_url,
                model=settings.llm.model,
                summary_text="这是 gpt-5.4 自己的逐篇总结。",
                summary_basis="llm+pdf+metadata",
                tags=["flashattention"],
                structured={
                    "summary": "这是 gpt-5.4 自己的逐篇总结。",
                    "basis": "llm+pdf+metadata",
                    "source_mode": "pdf_direct",
                    "pdf_input_strategy": "chat_file",
                    "direct_pdf_status": "used",
                },
                usage={"input_tokens": 100, "output_tokens": 30, "total_tokens": 130},
            )

            class FakeTopicDigestClient:
                enabled = True

                def __init__(self) -> None:
                    self.seen_summary_text = ""

                def generate_topic_digest(self, topic_name, description, entries):  # noqa: ANN001
                    if topic_name != "AI 算子加速":
                        return None
                    self.seen_summary_text = entries[0].paper.summary_text
                    return type(
                        "Digest",
                        (),
                        {
                            "overview": f"本窗口主题主要集中在 attention kernel 与 kernel fusion。输入摘要={self.seen_summary_text}",
                            "highlights": ["FlashAttention 仍然是主线", "Triton 和 MLIR 协同增多"],
                            "watchlist": ["关注训练侧 kernel autotuning"],
                            "tags": ["flashattention", "kernel fusion"],
                            "structured": {"usage": {"input_tokens": 1200, "output_tokens": 180, "total_tokens": 1380}},
                        },
                    )()

            fake_client = FakeTopicDigestClient()
            paths = generate_report(
                db,
                settings,
                report_date="2026-03-10",
                report_type="daily",
                lookback_days=1,
                llm_client=fake_client,
                use_llm_topic_digest=True,
            )
            topic_dir = Path(paths["topic_dir"])
            markdown = (topic_dir / "daily-2026-03-10--ai_operator_acceleration.md").read_text(encoding="utf-8")
            export = (root / "exports" / "daily-2026-03-10--ai_operator_acceleration.json").read_text(encoding="utf-8")
            self.assertEqual(fake_client.seen_summary_text, "这是 gpt-5.4 自己的逐篇总结。")
            self.assertIn("gpt-5.4 主题概览", markdown)
            self.assertIn("attention kernel 与 kernel fusion", markdown)
            self.assertIn("输入摘要=这是 gpt-5.4 自己的逐篇总结。", markdown)
            self.assertIn("gpt-5.4 聚合输入：使用 `1/1` 篇论文", markdown)
            self.assertIn("gpt-5.4 逐篇总结来源：本模型总结 `1` 篇，回退默认总结 `0` 篇", markdown)
            self.assertIn("gpt-5.4 Token", markdown)
            self.assertIn("\"variant_summary_count\": 1", export)
            self.assertIn("\"summary_source\": \"ikun_gpt-5.4 单篇总结\"", export)

            db.close()

    def test_report_can_limit_topic_digest_to_primary_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "config" / "config.json").write_text(FIXTURE_CONFIG_POE.read_text(encoding="utf-8"), encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            pipeline._process_candidate(
                PaperCandidate(
                    source_name="arxiv",
                    source_paper_id="2601.00002",
                    query_text="FlashAttention",
                    title="FlashAttention Compiler Fusion for Long Context Models",
                    abstract="We study attention kernels, fusion, and layout optimization for long context inference.",
                    authors=["Alice"],
                    published_at="2026-03-10T09:00:00+08:00",
                    updated_at="2026-03-10T09:00:00+08:00",
                    primary_url="https://arxiv.org/abs/2601.00002",
                    pdf_url="https://arxiv.org/pdf/2601.00002.pdf",
                    doi="",
                    arxiv_id="2601.00002",
                    venue="arXiv",
                    year=2026,
                    categories=["cs.LG"],
                    raw={"fixture": True},
                ),
                RunStats(),
            )

            class FakeDigestClient:
                enabled = True

                def __init__(self, overview: str) -> None:
                    self.overview = overview

                def generate_topic_digest(self, topic_name, description, entries):  # noqa: ANN001
                    if topic_name != "AI 算子加速":
                        return None
                    return type(
                        "Digest",
                        (),
                        {
                            "overview": self.overview,
                            "highlights": ["highlight"],
                            "watchlist": ["watch"],
                            "tags": ["tag"],
                            "structured": {"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
                        },
                    )()

            primary_variant = LLMRuntimeVariant(
                variant_id="poe",
                label="poe_claude-opus-4.6",
                provider="openai_compatible",
                base_url="https://api.poe.com/v1",
                model="claude-opus-4.6",
                config_path=root / "config" / "config.json",
                client=FakeDigestClient("primary digest"),
            )
            secondary_variant = LLMRuntimeVariant(
                variant_id="ikun",
                label="ikun_gpt-5.4",
                provider="IkunCoding",
                base_url="https://api.ikuncode.cc/v1",
                model="gpt-5.4",
                config_path=root / "config" / "config.json",
                client=FakeDigestClient("secondary digest"),
            )

            paths = generate_report(
                db,
                settings,
                report_date="2026-03-10",
                report_type="daily",
                llm_variants=[primary_variant, secondary_variant],
                topic_digest_variants=[primary_variant],
                use_llm_topic_digest=True,
            )
            topic_dir = Path(paths["topic_dir"])
            markdown = (topic_dir / "daily-2026-03-10--ai_operator_acceleration.md").read_text(encoding="utf-8")
            export = (root / "exports" / "daily-2026-03-10--ai_operator_acceleration.json").read_text(encoding="utf-8")

            self.assertIn("LLM 模型数：`2`", markdown)
            self.assertIn("primary digest", markdown)
            self.assertNotIn("secondary digest", markdown)
            self.assertIn("\"poe\"", export)
            self.assertNotIn("\"ikun\": {", export)

            db.close()

    def test_generate_report_writes_split_topic_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "config" / "config.json").write_text(FIXTURE_CONFIG_POE.read_text(encoding="utf-8"), encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            pipeline._process_candidate(
                PaperCandidate(
                    source_name="arxiv",
                    source_paper_id="2603.00011",
                    query_text="matrix-free finite element",
                    title="Matrix-Free DG Operator Evaluation on GPUs",
                    abstract="matrix-free operator evaluation with sum factorization on GPUs and performance portability",
                    authors=["Alice"],
                    published_at="2026-03-10T09:00:00+08:00",
                    updated_at="2026-03-10T09:00:00+08:00",
                    primary_url="https://arxiv.org/abs/2603.00011",
                    pdf_url="https://arxiv.org/pdf/2603.00011.pdf",
                    doi="",
                    arxiv_id="2603.00011",
                    venue="arXiv",
                    year=2026,
                    categories=["cs.NA"],
                    raw={"fixture": True},
                ),
                RunStats(),
            )
            pipeline._process_candidate(
                PaperCandidate(
                    source_name="arxiv",
                    source_paper_id="2603.00012",
                    query_text="flashattention triton compiler kernel fusion",
                    title="Kernel Fusion for Transformer Inference with Triton",
                    abstract="We optimize transformer inference with Triton kernel fusion and GPU runtime scheduling.",
                    authors=["Bob"],
                    published_at="2026-03-10T09:00:00+08:00",
                    updated_at="2026-03-10T09:00:00+08:00",
                    primary_url="https://arxiv.org/abs/2603.00012",
                    pdf_url="https://arxiv.org/pdf/2603.00012.pdf",
                    doi="",
                    arxiv_id="2603.00012",
                    venue="arXiv",
                    year=2026,
                    categories=["cs.LG"],
                    raw={"fixture": True},
                ),
                RunStats(),
            )

            paths = generate_report(
                db,
                settings,
                report_date="2026-03-10",
                report_type="daily",
                lookback_days=1,
            )

            topic_dir = Path(paths["topic_dir"])
            self.assertTrue(topic_dir.exists())
            matrix_md = topic_dir / "daily-2026-03-10--matrix_free_fem.md"
            ai_md = topic_dir / "daily-2026-03-10--ai_operator_acceleration.md"
            matrix_json = root / "exports" / "daily-2026-03-10--matrix_free_fem.json"
            ai_json = root / "exports" / "daily-2026-03-10--ai_operator_acceleration.json"
            self.assertTrue(matrix_md.exists())
            self.assertTrue(ai_md.exists())
            self.assertTrue(matrix_json.exists())
            self.assertTrue(ai_json.exists())
            matrix_text = matrix_md.read_text(encoding="utf-8")
            ai_text = ai_md.read_text(encoding="utf-8")
            self.assertIn("## 有限元分析 Matrix-Free 算法优化", matrix_text)
            self.assertNotIn("## AI 算子加速", matrix_text)
            self.assertIn("## AI 算子加速", ai_text)
            self.assertNotIn("## 有限元分析 Matrix-Free 算法优化", ai_text)

            db.close()

    def test_matrix_free_digest_selection_uses_bucketed_diversity(self) -> None:
        topic = TopicConfig(
            id="matrix_free_fem",
            display_name="有限元分析 Matrix-Free 算法优化",
            description="desc",
            source_queries={},
        )

        def make_entry(idx: int, title: str, abstract: str, score: float) -> ReportEntry:
            return ReportEntry(
                topic_id=topic.id,
                topic_name=topic.display_name,
                score=score,
                classification="relevant",
                matched_keywords=["matrix-free"],
                reasons=["fixture"],
                source_names=["seed"],
                source_urls=[],
                paper=PaperRecord(
                    id=idx,
                    title=title,
                    title_norm=title.lower(),
                    abstract=abstract,
                    authors=["Alice"],
                    published_at="2026-03-10T09:00:00+08:00",
                    updated_at="2026-03-10T09:00:00+08:00",
                    primary_url="https://example.com",
                    pdf_url="https://example.com/paper.pdf",
                    doi="",
                    arxiv_id="",
                    venue="arXiv",
                    year=2026,
                    categories=[],
                    summary_text=abstract,
                    summary_basis="llm+abstract+metadata",
                    tags=[],
                    pdf_local_path="",
                    pdf_status="pending",
                    pdf_downloaded_at=None,
                    fulltext_txt_path="",
                    fulltext_excerpt="",
                    fulltext_status="empty",
                    page_count=None,
                    llm_summary={},
                    analysis_updated_at=None,
                    source_first="seed",
                    created_at="2026-03-10T09:00:00+08:00",
                    last_seen_at="2026-03-10T09:00:00+08:00",
                    metadata={},
                ),
            )

        entries = [
            make_entry(i + 1, f"deal.II matrix-free framework paper {i}", "matrix-free finite element framework implementation on CPU", 1000 - i)
            for i in range(12)
        ]
        entries.extend(
            [
                make_entry(20, "Matrix-free DG operator evaluation with sum factorization", "matrix-free operator evaluation and sum factorization kernels", 880),
                make_entry(21, "GPU portability for matrix-free finite element operators", "matrix-free gpu portability on cuda hip sycl", 870),
                make_entry(22, "Multigrid for matrix-free high-order finite elements", "matrix-free multigrid preconditioner on GPU", 860),
                make_entry(23, "Matrix-free simplex and hybrid mesh extension", "matrix-free simplex hybrid mesh adaptive mesh refinement", 850),
                make_entry(24, "Additional operator kernel study", "matrix-free tensor product DG operator application", 840),
                make_entry(25, "Additional portability study", "matrix-free performance portability on amd and intel gpu", 830),
            ]
        )

        prepared, meta = _prepare_digest_entries_for_variant(
            topic,
            entries,
            {},
            variant_id="poe",
            entry_limit=12,
            variant_label="poe",
        )

        self.assertEqual(len(prepared), 14)
        self.assertTrue(str(meta["selection_mode"]).startswith("hybrid_top_"))
        bucket_counts = meta["bucket_counts"]
        self.assertIn("frameworks", bucket_counts)
        self.assertIn("operator_kernels", bucket_counts)
        self.assertIn("gpu_portability", bucket_counts)
        self.assertIn("multigrid", bucket_counts)
        self.assertIn("mesh_generalization", bucket_counts)
        scores = [entry.score for entry in prepared]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_ai_digest_selection_avoids_attention_only_top_slice(self) -> None:
        topic = TopicConfig(
            id="ai_operator_acceleration",
            display_name="AI 算子加速",
            description="desc",
            source_queries={},
        )

        def make_entry(idx: int, title: str, abstract: str, score: float) -> ReportEntry:
            return ReportEntry(
                topic_id=topic.id,
                topic_name=topic.display_name,
                score=score,
                classification="relevant",
                matched_keywords=["kernel"],
                reasons=["fixture"],
                source_names=["seed"],
                source_urls=[],
                paper=PaperRecord(
                    id=idx,
                    title=title,
                    title_norm=title.lower(),
                    abstract=abstract,
                    authors=["Alice"],
                    published_at="2026-03-10T09:00:00+08:00",
                    updated_at="2026-03-10T09:00:00+08:00",
                    primary_url="https://example.com",
                    pdf_url="https://example.com/paper.pdf",
                    doi="",
                    arxiv_id="",
                    venue="arXiv",
                    year=2026,
                    categories=[],
                    summary_text=abstract,
                    summary_basis="llm+abstract+metadata",
                    tags=[],
                    pdf_local_path="",
                    pdf_status="pending",
                    pdf_downloaded_at=None,
                    fulltext_txt_path="",
                    fulltext_excerpt="",
                    fulltext_status="empty",
                    page_count=None,
                    llm_summary={},
                    analysis_updated_at=None,
                    source_first="seed",
                    created_at="2026-03-10T09:00:00+08:00",
                    last_seen_at="2026-03-10T09:00:00+08:00",
                    metadata={},
                ),
            )

        entries = [
            make_entry(i + 100, f"FlashAttention variant {i}", "attention kernel optimization on Hopper", 1000 - i)
            for i in range(14)
        ]
        entries.extend(
            [
                make_entry(120, "CUTLASS GEMM Tensor Core kernels", "gemm matmul cutlass tensor core wgmma", 870),
                make_entry(121, "MLIR and Triton compiler stack for AI kernels", "mlir triton compiler tiling", 860),
                make_entry(122, "Kernel fusion runtime for LLM inference", "kernel fusion runtime vllm inference system", 850),
                make_entry(123, "ARM SME vectorization for transformer kernels", "arm sve sme simd kernel acceleration", 840),
                make_entry(124, "Additional TVM compiler autotuning", "tvm compiler autotuning tensor core", 830),
                make_entry(125, "Additional fusion training runtime", "training runtime fusion system", 820),
            ]
        )

        summaries = {
            120: [PaperLLMSummary(120, "poe", "poe", "openai_compatible", "https://api.example.com", "claude", "summary", "llm+pdf+metadata", [], {}, {}, "2026-03-10", "2026-03-10")],
            121: [PaperLLMSummary(121, "poe", "poe", "openai_compatible", "https://api.example.com", "claude", "summary", "llm+pdf+metadata", [], {}, {}, "2026-03-10", "2026-03-10")],
            122: [PaperLLMSummary(122, "poe", "poe", "openai_compatible", "https://api.example.com", "claude", "summary", "llm+pdf+metadata", [], {}, {}, "2026-03-10", "2026-03-10")],
            123: [PaperLLMSummary(123, "poe", "poe", "openai_compatible", "https://api.example.com", "claude", "summary", "llm+pdf+metadata", [], {}, {}, "2026-03-10", "2026-03-10")],
        }

        prepared, meta = _prepare_digest_entries_for_variant(
            topic,
            entries,
            summaries,
            variant_id="poe",
            entry_limit=12,
            variant_label="poe",
        )

        self.assertEqual(len(prepared), 16)
        self.assertTrue(str(meta["selection_mode"]).startswith("hybrid_top_"))
        bucket_counts = meta["bucket_counts"]
        self.assertIn("attention_kernels", bucket_counts)
        self.assertIn("gemm_tensorcore", bucket_counts)
        self.assertIn("compiler_stack", bucket_counts)
        self.assertIn("fusion_runtime_system", bucket_counts)
        self.assertIn("cpu_vector_arch", bucket_counts)
        scores = [entry.score for entry in prepared]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_report_renders_fulltext_scope_and_structured_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            settings.llm.variant_id = "config-example-gpt-5-4"
            settings.llm.label = "gpt-5.4"
            settings.llm.model = "gpt-5.4"
            settings.llm.base_url = "https://example.invalid/v1"
            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            pipeline._process_candidate(
                PaperCandidate(
                    source_name="arxiv",
                    source_paper_id="2751.00001",
                    query_text="matrix-free finite element",
                    title="Matrix-Free Fulltext Reporting Test on GPUs",
                    abstract="We study matrix-free operator evaluation, multigrid preconditioners, GPU kernels, and benchmark performance for high-order FEM.",
                    authors=["Alice"],
                    published_at="2026-03-10T09:00:00+08:00",
                    updated_at="2026-03-10T09:00:00+08:00",
                    primary_url="https://arxiv.org/abs/2751.00001",
                    pdf_url="https://arxiv.org/pdf/2751.00001.pdf",
                    doi="",
                    arxiv_id="2751.00001",
                    venue="arXiv",
                    year=2026,
                    categories=["cs.NA"],
                    raw={"fixture": True},
                ),
                RunStats(),
            )
            db.upsert_paper_llm_summary(
                1,
                variant_id="config-example-gpt-5-4",
                variant_label="gpt-5.4",
                provider="openai_responses",
                base_url="https://example.invalid/v1",
                model="gpt-5.4",
                summary_text="这是一个基于完整全文生成的结构化总结。",
                summary_basis="llm+fulltext+metadata",
                tags=["matrix-free", "multigrid"],
                structured={
                    "summary": "这是一个基于完整全文生成的结构化总结。",
                    "problem": "高阶 FEM 的矩阵自由算子与预条件效率问题。",
                    "method": "matrix-free + multigrid + partial assembly。",
                    "application": "高阶有限元与 GPU 求解。",
                    "results": "性能和扩展性优于基线。",
                    "contributions": ["提出新型预条件器", "给出 GPU 实现"],
                    "limitations": ["依赖高阶离散"],
                    "tags": ["matrix-free", "multigrid"],
                    "basis": "llm+fulltext+metadata",
                    "source_mode": "fulltext_txt",
                    "chunk_count": 4,
                    "direct_pdf_status": "unsupported",
                    "direct_pdf_strategy": "chat_file",
                },
                usage={"input_tokens": 1234, "output_tokens": 222, "total_tokens": 1456},
            )

            class FakeRenderClient:
                enabled = False

            paths = generate_report(
                db,
                settings,
                report_date="2026-03-10",
                report_type="daily",
                lookback_days=1,
                llm_client=FakeRenderClient(),
            )
            topic_dir = Path(paths["topic_dir"])
            markdown = (topic_dir / "daily-2026-03-10--matrix_free_fem.md").read_text(encoding="utf-8")
            html = (topic_dir / "daily-2026-03-10--matrix_free_fem.html").read_text(encoding="utf-8")
            export = (root / "exports" / "daily-2026-03-10--matrix_free_fem.json").read_text(encoding="utf-8")

            self.assertIn("输入依据：`已读取完整全文`", markdown)
            self.assertIn("本次总结读取了完整 PDF 提取全文，并按 4 个分块进行分析后聚合。", markdown)
            self.assertIn("直接 PDF 探测未通过（已尝试 `chat_file`）", markdown)
            self.assertIn("问题：高阶 FEM 的矩阵自由算子与预条件效率问题。", markdown)
            self.assertIn("方法：matrix-free + multigrid + partial assembly。", markdown)
            self.assertIn("输入依据 已读取完整全文", html)
            self.assertIn("\"direct_pdf_status\": \"unsupported\"", export)
            self.assertIn("\"summary_scope\": \"已读取完整全文\"", export)
            self.assertIn("\"chunk_count\": 4", export)

            db.close()

    def test_generate_paper_reports_exports_single_paper_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            pipeline._process_candidate(
                PaperCandidate(
                    source_name="arxiv",
                    source_paper_id="2761.00001",
                    query_text="matrix-free finite element",
                    title="Standalone Paper Report Test",
                    abstract="We study matrix-free operators and multigrid preconditioners.",
                    authors=["Alice"],
                    published_at="2026-03-10T09:00:00+08:00",
                    updated_at="2026-03-10T09:00:00+08:00",
                    primary_url="https://arxiv.org/abs/2761.00001",
                    pdf_url="https://arxiv.org/pdf/2761.00001.pdf",
                    doi="",
                    arxiv_id="2761.00001",
                    venue="arXiv",
                    year=2026,
                    categories=["cs.NA"],
                    raw={"fixture": True},
                ),
                RunStats(),
            )
            db.upsert_paper_llm_summary(
                1,
                variant_id="config-example-gpt-5-4",
                variant_label="gpt-5.4",
                provider="openai_responses",
                base_url="https://example.invalid/v1",
                model="gpt-5.4",
                summary_text="这是单篇导出的模型总结。",
                summary_basis="llm+pdf+metadata",
                tags=["matrix-free", "multigrid"],
                structured={
                    "summary": "这是单篇导出的模型总结。",
                    "problem": "矩阵自由算子效率。",
                    "method": "直接读取 PDF。",
                    "application": "高阶有限元。",
                    "results": "性能更好。",
                    "contributions": ["单篇导出"],
                    "limitations": ["依赖 PDF"],
                    "tags": ["matrix-free", "multigrid"],
                    "basis": "llm+pdf+metadata",
                    "source_mode": "pdf_direct",
                    "pdf_filename": "2761.00001.pdf",
                    "pdf_input_strategy": "chat_file",
                },
                usage={"input_tokens": 321, "output_tokens": 88, "total_tokens": 409},
            )

            outputs = generate_paper_reports(db, settings, [1])
            paper_paths = outputs[1]
            markdown = Path(paper_paths["markdown"]).read_text(encoding="utf-8")
            html = Path(paper_paths["html"]).read_text(encoding="utf-8")
            export = Path(paper_paths["json"]).read_text(encoding="utf-8")

            self.assertIn("单篇论文总结 - Standalone Paper Report Test", markdown)
            self.assertIn("输入依据：`已直接读取 PDF`", markdown)
            self.assertIn("策略 `chat_file`", markdown)
            self.assertIn("这是单篇导出的模型总结。", markdown)
            self.assertIn("已直接读取 PDF", html)
            self.assertIn("chat_file", export)
            self.assertIn("\"summary_scope\": \"已直接读取 PDF\"", export)

            db.close()

    def test_generate_preview_report_exports_single_paper_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            paper = PaperRecord(
                id=123456,
                title="Preview Export Sample",
                title_norm="preview export sample",
                abstract="matrix-free operator evaluation on GPUs",
                authors=["Alice"],
                published_at="2026-03-20",
                updated_at="2026-03-20",
                primary_url="https://example.com/preview",
                pdf_url="https://example.com/preview.pdf",
                doi="",
                arxiv_id="",
                venue="PreviewConf",
                year=2026,
                categories=["cs.DC"],
                summary_text="默认总结",
                summary_basis="fulltext+metadata",
                tags=["preview"],
                pdf_local_path="artifacts/pdfs/preview.pdf",
                pdf_status="downloaded",
                pdf_downloaded_at="2026-03-20T10:00:00+08:00",
                fulltext_txt_path="artifacts/text/preview.txt",
                fulltext_excerpt="fulltext excerpt",
                fulltext_status="extracted",
                page_count=12,
                llm_summary={},
                analysis_updated_at="2026-03-20T10:00:00+08:00",
                source_first="preview",
                created_at="2026-03-20T10:00:00+08:00",
                last_seen_at="2026-03-20T10:00:00+08:00",
                metadata={"preview": True},
            )
            evaluations = [
                TopicEvaluation(
                    topic_id="matrix_free_fem",
                    topic_name="有限元分析 Matrix-Free 算法优化",
                    score=42.0,
                    classification="relevant",
                    matched_keywords=["matrix-free", "gpu"],
                    reasons=["命中硬性关键词组"],
                )
            ]
            summaries = [
                PaperLLMSummary(
                    paper_id=paper.id,
                    variant_id="ikun",
                    variant_label="ikun_gpt-5.4",
                    provider="IkunCoding",
                    base_url="https://example.com/v1",
                    model="gpt-5.4",
                    summary_text="问题：A 方法：B 结果：C",
                    summary_basis="llm+pdf+metadata",
                    tags=["gpu"],
                    structured={"source_mode": "pdf_direct", "direct_pdf_status": "used"},
                    usage={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
                    created_at="2026-03-20T10:00:00+08:00",
                    updated_at="2026-03-20T10:00:00+08:00",
                )
            ]

            outputs = generate_preview_report(
                settings,
                paper,
                evaluations,
                source_names=["preview"],
                source_urls=["https://example.com/preview"],
                summaries=summaries,
                llm_variants=None,
                paper_add_suggestion={"recommended_topic_ids": ["matrix_free_fem"]},
            )
            markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
            export = Path(outputs["json"]).read_text(encoding="utf-8")

            self.assertEqual(set(outputs.keys()), {"markdown", "html", "json"})
            self.assertTrue(Path(outputs["markdown"]).exists())
            self.assertTrue(Path(outputs["html"]).exists())
            self.assertTrue(Path(outputs["json"]).exists())
            self.assertIn("单篇论文预分析 - Preview Export Sample", markdown)
            self.assertIn("未入库（预分析）", markdown)
            self.assertIn("\"persisted\": false", export)

    def test_generate_catalog_report_lists_stored_model_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            pipeline._process_candidate(
                PaperCandidate(
                    source_name="arxiv",
                    source_paper_id="2903.00001",
                    query_text="matrix-free finite element",
                    title="Matrix-Free Preconditioners for High-Order FEM on GPUs",
                    abstract="matrix-free finite element preconditioner, multigrid, GPU implementation, and benchmark performance portability",
                    authors=["Alice"],
                    published_at="2026-03-10T09:00:00+08:00",
                    updated_at="2026-03-10T09:00:00+08:00",
                    primary_url="https://arxiv.org/abs/2903.00001",
                    pdf_url="https://arxiv.org/pdf/2903.00001.pdf",
                    doi="",
                    arxiv_id="2903.00001",
                    venue="arXiv",
                    year=2026,
                    categories=["cs.NA"],
                    raw={"fixture": True},
                ),
                RunStats(),
            )
            db.upsert_paper_llm_summary(
                1,
                variant_id="poe",
                variant_label="poe_gemini-3.1-pro",
                provider="openai_compatible",
                base_url="https://api.poe.com/v1",
                model="gemini-3.1-pro",
                summary_text="Poe 总结",
                summary_basis="llm+fulltext+metadata",
                tags=["matrix-free"],
                structured={"summary": "Poe 总结", "source_mode": "fulltext_txt"},
                usage={},
            )
            db.upsert_paper_llm_summary(
                1,
                variant_id="ikun",
                variant_label="ikun_gpt-5.4",
                provider="IkunCoding",
                base_url="https://api.ikuncode.cc/v1",
                model="gpt-5.4",
                summary_text="Ikun 总结",
                summary_basis="llm+pdf+metadata",
                tags=["matrix-free"],
                structured={"summary": "Ikun 总结", "source_mode": "pdf_direct", "pdf_input_strategy": "chat_file"},
                usage={},
            )

            paths = generate_catalog_report(db, settings)
            topic_dir = Path(paths["topic_dir"])
            markdown = (topic_dir / "matrix_free_fem.md").read_text(encoding="utf-8")
            export = (root / "exports" / "catalog-matrix_free_fem.json").read_text(encoding="utf-8")

            self.assertIn("论文库总览", markdown)
            self.assertIn("Matrix-Free Preconditioners for High-Order FEM", markdown)
            self.assertIn("poe_gemini-3.1-pro", markdown)
            self.assertIn("ikun_gpt-5.4", markdown)
            self.assertIn("Poe 总结", markdown)
            self.assertIn("Ikun 总结", markdown)
            self.assertIn("\"variant_id\": \"poe\"", export)
            self.assertIn("\"variant_id\": \"ikun\"", export)

            db.close()

    def test_fetch_catalog_entries_orders_by_topic_then_score_desc(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_POE.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            topic = next(item for item in settings.topics if item.id == "ai_operator_acceleration")

            def add_candidate(index: int, title: str, score: float, published_at: str) -> None:
                candidate = PaperCandidate(
                    source_name="manual",
                    source_paper_id=f"fixture-{index}",
                    query_text="fixture",
                    title=title,
                    abstract="flashattention kernel implementation for transformer inference",
                    authors=["Alice"],
                    published_at=published_at,
                    updated_at=published_at,
                    primary_url=f"https://example.com/{index}",
                    pdf_url="",
                    doi="",
                    arxiv_id="",
                    venue="MLSys",
                    year=int(published_at[:4]),
                    categories=["cs.DC"],
                    raw={"fixture": True},
                )
                paper_id, _ = db.upsert_paper(candidate)
                db.upsert_match(
                    paper_id,
                    TopicEvaluation(
                        topic_id=topic.id,
                        topic_name=topic.display_name,
                        score=score,
                        classification="relevant",
                        matched_keywords=["flashattention"],
                        reasons=["fixture"],
                    ),
                )

            add_candidate(1, "Lower Score Recent", 30.0, "2026-03-10")
            add_candidate(2, "Higher Score Old", 70.0, "2024-03-10")
            add_candidate(3, "Mid Score Newer", 50.0, "2026-03-20")

            entries = [entry for entry in db.fetch_catalog_entries(include_maybe=True) if entry.topic_id == topic.id]
            self.assertEqual(
                [entry.paper.title for entry in entries[:3]],
                ["Higher Score Old", "Mid Score Newer", "Lower Score Recent"],
            )
            db.close()

    def test_generate_catalog_report_merges_requested_and_stored_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            pipeline._process_candidate(
                PaperCandidate(
                    source_name="arxiv",
                    source_paper_id="2903.00002",
                    query_text="matrix-free finite element",
                    title="Matrix-Free Kernel Optimization",
                    abstract="matrix-free finite element operator application on GPUs",
                    authors=["Alice"],
                    published_at="2026-03-10T09:00:00+08:00",
                    updated_at="2026-03-10T09:00:00+08:00",
                    primary_url="https://arxiv.org/abs/2903.00002",
                    pdf_url="https://arxiv.org/pdf/2903.00002.pdf",
                    doi="",
                    arxiv_id="2903.00002",
                    venue="arXiv",
                    year=2026,
                    categories=["cs.DC"],
                    raw={"fixture": True},
                ),
                RunStats(),
            )
            db.upsert_paper_llm_summary(
                1,
                variant_id="legacy",
                variant_label="legacy_model",
                provider="openai_compatible",
                base_url="https://legacy.example.com/v1",
                model="legacy-model",
                summary_text="Legacy 总结",
                summary_basis="llm+abstract+metadata",
                tags=["legacy"],
                structured={"summary": "Legacy 总结", "source_mode": "abstract"},
                usage={},
            )
            db.upsert_paper_llm_summary(
                1,
                variant_id="poe",
                variant_label="poe_gemini-3.1-pro",
                provider="openai_compatible",
                base_url="https://api.poe.com/v1",
                model="gemini-3.1-pro",
                summary_text="Poe 总结",
                summary_basis="llm+fulltext+metadata",
                tags=["matrix-free"],
                structured={"summary": "Poe 总结", "source_mode": "fulltext_txt"},
                usage={},
            )
            db.upsert_paper_llm_summary(
                1,
                variant_id="ikun",
                variant_label="ikun_gpt-5.4",
                provider="IkunCoding",
                base_url="https://api.ikuncode.cc/v1",
                model="gpt-5.4",
                summary_text="Ikun 总结",
                summary_basis="llm+pdf+metadata",
                tags=["matrix-free"],
                structured={"summary": "Ikun 总结", "source_mode": "pdf_direct", "pdf_input_strategy": "chat_file"},
                usage={},
            )

            client = LLMClient(settings.llm)
            variants = [
                LLMRuntimeVariant(
                    variant_id="poe",
                    label="config-poe.json / poe_gemini-3.1-pro",
                    provider="openai_compatible",
                    base_url="https://api.poe.com/v1",
                    model="gemini-3.1-pro",
                    config_path=root / "config" / "config-poe.json",
                    client=client,
                ),
                LLMRuntimeVariant(
                    variant_id="ikun",
                    label="config-ikun.json / ikun_gpt-5.4",
                    provider="IkunCoding",
                    base_url="https://api.ikuncode.cc/v1",
                    model="gpt-5.4",
                    config_path=root / "config" / "config-ikun.json",
                    client=client,
                ),
            ]

            paths = generate_catalog_report(db, settings, llm_variants=variants)
            topic_dir = Path(paths["topic_dir"])
            markdown = (topic_dir / "matrix_free_fem.md").read_text(encoding="utf-8")
            export = (root / "exports" / "catalog-matrix_free_fem.json").read_text(encoding="utf-8")

            self.assertIn("poe_gemini-3.1-pro", markdown)
            self.assertIn("ikun_gpt-5.4", markdown)
            self.assertIn("legacy_model", markdown)
            self.assertIn("\"variant_id\": \"poe\"", export)
            self.assertIn("\"variant_id\": \"ikun\"", export)
            self.assertIn("\"variant_id\": \"legacy\"", export)

            db.close()

    def test_generate_catalog_report_writes_split_topic_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "config" / "config.json").write_text(FIXTURE_CONFIG_POE.read_text(encoding="utf-8"), encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            pipeline._process_candidate(
                PaperCandidate(
                    source_name="arxiv",
                    source_paper_id="2603.00013",
                    query_text="matrix-free finite element",
                    title="GPU Matrix-Free FEM Benchmark",
                    abstract="matrix-free finite element operator application benchmark on GPUs with performance portability",
                    authors=["Alice"],
                    published_at="2026-03-10T09:00:00+08:00",
                    updated_at="2026-03-10T09:00:00+08:00",
                    primary_url="https://arxiv.org/abs/2603.00013",
                    pdf_url="https://arxiv.org/pdf/2603.00013.pdf",
                    doi="",
                    arxiv_id="2603.00013",
                    venue="arXiv",
                    year=2026,
                    categories=["cs.NA"],
                    raw={"fixture": True},
                ),
                RunStats(),
            )
            pipeline._process_candidate(
                PaperCandidate(
                    source_name="arxiv",
                    source_paper_id="2603.00014",
                    query_text="flashattention triton compiler kernel fusion",
                    title="CUTLASS and Triton for LLM Inference Kernels",
                    abstract="We study CUTLASS, Triton, GEMM tiling, and transformer inference kernel fusion.",
                    authors=["Bob"],
                    published_at="2026-03-10T09:00:00+08:00",
                    updated_at="2026-03-10T09:00:00+08:00",
                    primary_url="https://arxiv.org/abs/2603.00014",
                    pdf_url="https://arxiv.org/pdf/2603.00014.pdf",
                    doi="",
                    arxiv_id="2603.00014",
                    venue="arXiv",
                    year=2026,
                    categories=["cs.LG"],
                    raw={"fixture": True},
                ),
                RunStats(),
            )

            paths = generate_catalog_report(db, settings)

            topic_dir = Path(paths["topic_dir"])
            self.assertTrue(topic_dir.exists())
            matrix_md = topic_dir / "matrix_free_fem.md"
            ai_md = topic_dir / "ai_operator_acceleration.md"
            matrix_json = root / "exports" / "catalog-matrix_free_fem.json"
            ai_json = root / "exports" / "catalog-ai_operator_acceleration.json"
            self.assertTrue(matrix_md.exists())
            self.assertTrue(ai_md.exists())
            self.assertTrue(matrix_json.exists())
            self.assertTrue(ai_json.exists())
            matrix_text = matrix_md.read_text(encoding="utf-8")
            ai_text = ai_md.read_text(encoding="utf-8")
            self.assertIn("## 有限元分析 Matrix-Free 算法优化", matrix_text)
            self.assertNotIn("## AI 算子加速", matrix_text)
            self.assertIn("## AI 算子加速", ai_text)
            self.assertNotIn("## 有限元分析 Matrix-Free 算法优化", ai_text)

            db.close()

    def test_delete_paper_cascades_related_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            db = Database(settings.database_path, settings.timezone)
            db.initialize()

            candidate = PaperCandidate(
                source_name="manual",
                source_paper_id="manual-1",
                query_text="manual",
                title="Manual GPU Kernel Paper",
                abstract="kernel fusion compiler",
                authors=["Bob"],
                published_at="2026-03-10",
                updated_at="2026-03-10",
                primary_url="https://example.com/paper",
                pdf_url="https://example.com/paper.pdf",
                doi="10.1000/manual",
                arxiv_id="",
                venue="manual",
                year=2026,
                categories=["cs.LG"],
                raw={"fixture": True},
            )
            paper_id, created = db.upsert_paper(candidate)
            self.assertTrue(created)
            db.upsert_match(
                paper_id,
                TopicEvaluation(
                    topic_id="ai_operator_acceleration",
                    topic_name="AI 算子加速",
                    score=100.0,
                    classification="relevant",
                    matched_keywords=["manual"],
                    reasons=["用户手动加入论文"],
                ),
            )
            db.upsert_paper_llm_summary(
                paper_id,
                variant_id="poe",
                variant_label="poe_gemini-3.1-pro",
                provider="openai_compatible",
                base_url="https://api.poe.com/v1",
                model="gemini-3.1-pro",
                summary_text="manual summary",
                summary_basis="llm+abstract+metadata",
                tags=["manual"],
                structured={"summary": "manual summary"},
                usage={},
            )

            self.assertTrue(db.delete_paper(paper_id))
            self.assertFalse(db.delete_paper(paper_id))
            self.assertEqual(db.connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0], 0)
            self.assertEqual(db.connection.execute("SELECT COUNT(*) FROM paper_sources").fetchone()[0], 0)
            self.assertEqual(db.connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0], 0)
            self.assertEqual(db.connection.execute("SELECT COUNT(*) FROM paper_llm_summaries").fetchone()[0], 0)

            db.close()

    def test_comparison_report_contains_both_model_digests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            pipeline._process_candidate(
                PaperCandidate(
                    source_name="arxiv",
                    source_paper_id="2701.00001",
                    query_text="matrix-free finite element",
                    title="Matrix-Free Multigrid for High-Order FEM on GPUs",
                    abstract="We study matrix-free multigrid, partial assembly, GPU kernels, and strong-scaling benchmarks for high-order FEM.",
                    authors=["Alice"],
                    published_at="2026-03-10T09:00:00+08:00",
                    updated_at="2026-03-10T09:00:00+08:00",
                    primary_url="https://arxiv.org/abs/2701.00001",
                    pdf_url="https://arxiv.org/pdf/2701.00001.pdf",
                    doi="",
                    arxiv_id="2701.00001",
                    venue="arXiv",
                    year=2026,
                    categories=["cs.NA"],
                    raw={"fixture": True},
                ),
                RunStats(),
            )

            class FakeDigestClientA:
                enabled = True

                def generate_topic_digest(self, topic_name, description, entries):  # noqa: ANN001
                    if not entries:
                        return None
                    return type(
                        "Digest",
                        (),
                        {
                            "overview": f"A:{topic_name}",
                            "highlights": ["A1", "A2"],
                            "watchlist": ["A3"],
                            "tags": ["a"],
                            "structured": {"usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}},
                        },
                    )()

            class FakeDigestClientB:
                enabled = True

                def generate_topic_digest(self, topic_name, description, entries):  # noqa: ANN001
                    if not entries:
                        return None
                    return type(
                        "Digest",
                        (),
                        {
                            "overview": f"B:{topic_name}",
                            "highlights": ["B1", "B2"],
                            "watchlist": ["B3"],
                            "tags": ["b"],
                            "structured": {"usage": {"input_tokens": 110, "output_tokens": 22, "total_tokens": 132}},
                        },
                    )()

            paths = generate_comparison_report(
                db,
                settings,
                report_date="2026-03-10",
                report_type="weekly",
                variants=[
                    {
                        "slug": "config",
                        "label": "config-ikun.json",
                        "model": "gpt-5",
                        "base_url": "https://left.example/v1",
                        "llm_client": FakeDigestClientA(),
                    },
                    {
                        "slug": "config-poe",
                        "label": "config-poe.json",
                        "model": "gpt-5.4",
                        "base_url": "https://right.example/v1",
                        "llm_client": FakeDigestClientB(),
                    },
                ],
                lookback_days=7,
            )
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            self.assertIn("config-ikun.json", markdown)
            self.assertIn("config-poe.json", markdown)
            self.assertIn("A:有限元分析 Matrix-Free 算法优化", markdown)
            self.assertIn("B:有限元分析 Matrix-Free 算法优化", markdown)
            self.assertIn("gpt-5.4", markdown)

            db.close()


if __name__ == "__main__":
    unittest.main()
