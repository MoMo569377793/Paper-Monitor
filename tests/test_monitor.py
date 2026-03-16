from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from paper_monitor.config import load_settings
from paper_monitor.enrichment import DocumentArtifacts, EnrichmentPipeline
from paper_monitor.llm import LLMClient, TASK_TOPIC_DIGEST
from paper_monitor.llm_registry import LLMRuntimeVariant
from paper_monitor.models import FetchPlan, LLMResult
from paper_monitor.models import PaperCandidate, PaperRecord, ReportEntry, RunStats, TopicEvaluation
from paper_monitor.pipeline import MonitorPipeline
from paper_monitor.reports import generate_catalog_report, generate_comparison_report, generate_paper_reports, generate_report
from paper_monitor.scoring import evaluate_paper_against_topic
from paper_monitor.storage import Database


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CONFIG_IKUN = REPO_ROOT / "config" / "config-ikun.json"
FIXTURE_CONFIG_POE = REPO_ROOT / "config" / "config-poe.json"


class MonitorPipelineTest(unittest.TestCase):
    def test_since_last_run_only_processes_new_papers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = FIXTURE_CONFIG_IKUN.read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            settings.topics = [settings.topics[0]]
            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            old_candidate = PaperCandidate(
                source_name="arxiv",
                source_paper_id="2901.00001",
                query_text="matrix-free finite element",
                title="Matrix-Free FEM Baseline",
                abstract="matrix-free finite element and partial assembly",
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
                raw={"fixture": True},
            )
            new_candidate = PaperCandidate(
                source_name="arxiv",
                source_paper_id="2901.00002",
                query_text="matrix-free finite element",
                title="Matrix-Free FEM Incremental",
                abstract="matrix-free finite element and multigrid",
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
                raw={"fixture": True},
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
                        abstract="matrix-free finite element and partial assembly",
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
                        raw={"fixture": True},
                    )
                )
            for index in range(70):
                candidates.append(
                    PaperCandidate(
                        source_name="dblp",
                        source_paper_id=f"dblp-{index}",
                        query_text="fixture",
                        title=f"Shared Paper {index}" if index < 20 else f"DBLP Unique Paper {index}",
                        abstract="matrix-free finite element and multigrid",
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
                        raw={"fixture": True},
                    )
                )

            selected = pipeline._select_topic_candidates(candidates, FetchPlan(recent_limit=100, page_size=50))
            logical_titles = {candidate.title for candidate in selected}

            self.assertEqual(len(logical_titles), 100)
            self.assertLessEqual(len(selected), 120)

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
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            html = Path(paths["html"]).read_text(encoding="utf-8")

            self.assertIn("Matrix-Free Finite Element Operator Evaluation on GPUs", markdown)
            self.assertIn("Kernel Fusion for Transformer Inference with FlashAttention and Triton", markdown)
            self.assertIn("有限元分析 Matrix-Free 算法优化", html)
            self.assertIn("AI 算子加速", html)

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
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            self.assertIn("全文状态", markdown)
            self.assertIn("llm+fulltext+metadata", markdown)

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
            self.assertEqual(result.structured.get("direct_pdf_status"), "unsupported")
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
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            export = Path(paths["json"]).read_text(encoding="utf-8")
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
                    title="Matrix-Free Fulltext Reporting Test",
                    abstract="We study matrix-free operators and preconditioners for high-order FEM.",
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
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            html = Path(paths["html"]).read_text(encoding="utf-8")
            export = Path(paths["json"]).read_text(encoding="utf-8")

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
                    title="Matrix-Free Preconditioners for High-Order FEM",
                    abstract="matrix-free finite element preconditioner and multigrid",
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
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            export = Path(paths["json"]).read_text(encoding="utf-8")

            self.assertIn("论文库总览", markdown)
            self.assertIn("Matrix-Free Preconditioners for High-Order FEM", markdown)
            self.assertIn("poe_gemini-3.1-pro", markdown)
            self.assertIn("ikun_gpt-5.4", markdown)
            self.assertIn("Poe 总结", markdown)
            self.assertIn("Ikun 总结", markdown)
            self.assertIn("\"variant_id\": \"poe\"", export)
            self.assertIn("\"variant_id\": \"ikun\"", export)

            db.close()

    def test_generate_catalog_report_hides_unrequested_stored_variants(self) -> None:
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
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            export = Path(paths["json"]).read_text(encoding="utf-8")

            self.assertIn("poe_gemini-3.1-pro", markdown)
            self.assertIn("ikun_gpt-5.4", markdown)
            self.assertNotIn("legacy_model", markdown)
            self.assertIn("\"variant_id\": \"poe\"", export)
            self.assertIn("\"variant_id\": \"ikun\"", export)
            self.assertNotIn("\"variant_id\": \"legacy\"", export)

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
                    title="Matrix-Free Multigrid for High-Order FEM",
                    abstract="We study matrix-free multigrid and partial assembly for high-order FEM.",
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
