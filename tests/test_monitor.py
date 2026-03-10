from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from paper_monitor.config import load_settings
from paper_monitor.models import PaperCandidate, RunStats
from paper_monitor.pipeline import MonitorPipeline
from paper_monitor.reports import generate_report
from paper_monitor.storage import Database


class MonitorPipelineTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
