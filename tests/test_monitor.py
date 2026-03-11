from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from paper_monitor.config import load_settings
from paper_monitor.enrichment import DocumentArtifacts, EnrichmentPipeline
from paper_monitor.llm import LLMClient
from paper_monitor.models import FetchPlan, LLMResult
from paper_monitor.models import PaperCandidate, PaperRecord, RunStats
from paper_monitor.pipeline import MonitorPipeline
from paper_monitor.reports import generate_comparison_report, generate_report
from paper_monitor.storage import Database


class MonitorPipelineTest(unittest.TestCase):
    def test_since_last_run_only_processes_new_papers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = Path("/home/momo/git_ws/search/config/config.example.json").read_text(encoding="utf-8")
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
            source_config = Path("/home/momo/git_ws/search/config/config.example.json").read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
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
            source_config = Path("/home/momo/git_ws/search/config/config.example.json").read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
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

            paths = generate_report(db, settings, report_date="2026-03-10", report_type="daily", lookback_days=1)
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            html = Path(paths["html"]).read_text(encoding="utf-8")

            self.assertIn("Matrix-Free Finite Element Operator Evaluation on GPUs", markdown)
            self.assertIn("Kernel Fusion for Transformer Inference with FlashAttention and Triton", markdown)
            self.assertIn("有限元分析 Matrix-Free 算法优化", html)
            self.assertIn("AI 算子加速", html)

            db.close()

    def test_enrichment_pipeline_upgrades_summary_with_fulltext_and_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = Path("/home/momo/git_ws/search/config/config.example.json").read_text(encoding="utf-8")
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

            paths = generate_report(db, settings, report_date="2026-03-10", report_type="daily", lookback_days=1)
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            self.assertIn("全文状态", markdown)
            self.assertIn("llm+fulltext+metadata", markdown)

            db.close()

    def test_enrichment_can_filter_by_created_after(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = Path("/home/momo/git_ws/search/config/config.example.json").read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
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
            source_config = Path("/home/momo/git_ws/search/config/config.example.json").read_text(encoding="utf-8")
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

    def test_enrichment_candidates_prioritize_missing_llm_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = Path("/home/momo/git_ws/search/config/config.example.json").read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
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

    def test_report_includes_topic_digest_when_llm_client_is_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = Path("/home/momo/git_ws/search/config/config.example.json").read_text(encoding="utf-8")
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

            class FakeTopicDigestClient:
                enabled = True

                def generate_topic_digest(self, topic_name, description, entries):  # noqa: ANN001
                    if topic_name != "AI 算子加速":
                        return None
                    return type(
                        "Digest",
                        (),
                        {
                            "overview": "本窗口主题主要集中在 attention kernel 与 kernel fusion。",
                            "highlights": ["FlashAttention 仍然是主线", "Triton 和 MLIR 协同增多"],
                            "watchlist": ["关注训练侧 kernel autotuning"],
                            "tags": ["flashattention", "kernel fusion"],
                            "structured": {"usage": {"input_tokens": 1200, "output_tokens": 180, "total_tokens": 1380}},
                        },
                    )()

            paths = generate_report(
                db,
                settings,
                report_date="2026-03-10",
                report_type="daily",
                lookback_days=1,
                llm_client=FakeTopicDigestClient(),
                use_llm_topic_digest=True,
            )
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            self.assertIn("gpt-5.4 主题概览", markdown)
            self.assertIn("attention kernel 与 kernel fusion", markdown)
            self.assertIn("gpt-5.4 Token", markdown)

            db.close()

    def test_comparison_report_contains_both_model_digests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = Path("/home/momo/git_ws/search/config/config.example.json").read_text(encoding="utf-8")
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
                        "label": "config.json",
                        "model": "gpt-5",
                        "base_url": "https://left.example/v1",
                        "llm_client": FakeDigestClientA(),
                    },
                    {
                        "slug": "config-example",
                        "label": "config.example.json",
                        "model": "gpt-5.4",
                        "base_url": "https://right.example/v1",
                        "llm_client": FakeDigestClientB(),
                    },
                ],
                lookback_days=7,
            )
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            self.assertIn("config.json", markdown)
            self.assertIn("config.example.json", markdown)
            self.assertIn("A:有限元分析 Matrix-Free 算法优化", markdown)
            self.assertIn("B:有限元分析 Matrix-Free 算法优化", markdown)
            self.assertIn("gpt-5.4", markdown)

            db.close()


if __name__ == "__main__":
    unittest.main()
