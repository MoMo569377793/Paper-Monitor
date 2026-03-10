from __future__ import annotations

import json
from pathlib import Path

from paper_monitor.models import (
    GenericSourceConfig,
    ReportConfig,
    ScholarAlertsConfig,
    Settings,
    TopicConfig,
)
from paper_monitor.utils import ensure_directory


DEFAULT_CONFIG = {
    "database_path": "data/papers.db",
    "report_dir": "reports",
    "export_dir": "exports",
    "poll_minutes": 60,
    "timezone": "Asia/Shanghai",
    "sources": {
        "arxiv": {
            "enabled": True,
            "max_results": 30,
            "timeout_seconds": 20,
            "user_agent": "paper-monitor/0.1 (+local)",
        },
        "dblp": {
            "enabled": True,
            "max_results": 20,
            "timeout_seconds": 20,
            "user_agent": "paper-monitor/0.1 (+local)",
        },
        "google_scholar_alerts": {
            "enabled": False,
            "imap_host": "",
            "imap_port": 993,
            "username": "",
            "password_env": "SCHOLAR_ALERTS_PASSWORD",
            "folder": "INBOX",
            "subject_keyword": "Google Scholar Alerts",
            "search_criterion": "UNSEEN",
            "timeout_seconds": 20,
        },
    },
    "report": {
        "top_n_per_topic": 5,
        "lookback_days": 1,
        "include_maybe": True,
    },
    "topics": [
        {
            "id": "matrix_free_fem",
            "display_name": "有限元分析 Matrix-Free 算法优化",
            "description": "跟踪矩阵自由、部分装配、sum factorization、高阶有限元与相关 HPC 优化论文。",
            "source_queries": {
                "arxiv": [
                    "all:\"matrix-free\" AND all:\"finite element\"",
                    "all:\"partial assembly\" AND all:\"finite element\"",
                    "all:\"sum factorization\" AND all:\"finite element\"",
                ],
                "dblp": [
                    "\"matrix-free\" \"finite element\"",
                    "\"partial assembly\" \"finite element\"",
                    "\"sum factorization\" \"finite element\"",
                ],
                "google_scholar_alerts": [
                    "\"matrix-free\" \"finite element\"",
                    "\"partial assembly\" FEM",
                    "\"sum factorization\" finite element",
                ],
            },
            "must_match_groups": [
                ["matrix-free", "matrix free", "partial assembly", "sum factorization"],
                ["finite element", "finite elements", "fem", "high-order fem", "high order finite element"],
            ],
            "positive_keywords": [
                "libceed",
                "mfem",
                "deal.ii",
                "deal ii",
                "operator application",
                "multigrid",
                "preconditioner",
                "spectral element",
                "tensor contraction",
                "element restriction",
                "gpu acceleration",
            ],
            "exclude_keywords": [
                "matrix factorization",
                "graph neural network",
                "diffusion model",
                "recommendation system",
            ],
            "arxiv_categories": ["cs.NA", "cs.MS", "math.NA"],
            "dblp_venue_keywords": ["finite elements", "supercomputing", "high performance"],
            "threshold": 18,
        },
        {
            "id": "ai_operator_acceleration",
            "display_name": "AI 算子加速",
            "description": "跟踪算子优化、内核融合、编译器、推理与训练性能优化相关论文。",
            "source_queries": {
                "arxiv": [
                    "all:\"kernel fusion\"",
                    "all:\"operator fusion\"",
                    "all:\"tensor compiler\"",
                    "all:\"FlashAttention\"",
                    "all:\"MLIR\" AND all:\"deep learning\"",
                ],
                "dblp": [
                    "\"kernel fusion\" deep learning",
                    "\"tensor compiler\" deep learning",
                    "\"operator acceleration\" AI",
                    "FlashAttention",
                    "MLIR deep learning compiler",
                ],
                "google_scholar_alerts": [
                    "\"kernel fusion\" transformer",
                    "\"tensor compiler\" deep learning",
                    "\"operator acceleration\" AI",
                    "FlashAttention kernel optimization",
                ],
            },
            "must_match_groups": [
                ["kernel", "operator", "compiler", "fusion", "acceleration"],
                ["deep learning", "llm", "transformer", "tensor", "inference", "training"],
            ],
            "positive_keywords": [
                "triton",
                "tvm",
                "mlir",
                "xla",
                "tensorrt",
                "cuda",
                "gemm",
                "attention kernel",
                "flashattention",
                "cutlass",
                "graph compiler",
                "operator scheduling",
                "fusion pass",
                "tensorir",
                "code generation",
            ],
            "exclude_keywords": [
                "medical image",
                "operator theory",
                "linear operator without acceleration",
                "finite element operator",
            ],
            "arxiv_categories": ["cs.LG", "cs.DC", "cs.AR", "cs.MS", "cs.PF"],
            "dblp_venue_keywords": ["ppopp", "mlsys", "isca", "micro", "hpca", "asplos", "sc"],
            "threshold": 18,
        },
    ],
}


def _resolve_base_dir(config_path: Path) -> Path:
    if config_path.parent.name == "config":
        return config_path.parent.parent.resolve()
    return config_path.parent.resolve()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_settings(config_path: str | Path) -> Settings:
    path = Path(config_path).resolve()
    raw = _load_json(path)
    base_dir = _resolve_base_dir(path)

    sources = raw.get("sources", {})
    arxiv = GenericSourceConfig(**sources.get("arxiv", {}))
    dblp = GenericSourceConfig(**sources.get("dblp", {}))
    scholar_alerts = ScholarAlertsConfig(**sources.get("google_scholar_alerts", {}))
    report = ReportConfig(**raw.get("report", {}))

    topics = [TopicConfig(**topic_raw) for topic_raw in raw.get("topics", [])]
    if not topics:
        raise ValueError("配置文件中至少需要一个 topic。")

    return Settings(
        base_dir=base_dir,
        config_path=path,
        database_path=(base_dir / raw.get("database_path", "data/papers.db")).resolve(),
        report_dir=(base_dir / raw.get("report_dir", "reports")).resolve(),
        export_dir=(base_dir / raw.get("export_dir", "exports")).resolve(),
        poll_minutes=int(raw.get("poll_minutes", 60)),
        timezone=raw.get("timezone", "Asia/Shanghai"),
        arxiv=arxiv,
        dblp=dblp,
        scholar_alerts=scholar_alerts,
        report=report,
        topics=topics,
    )


def write_default_config(config_path: str | Path, force: bool = False) -> Path:
    path = Path(config_path).resolve()
    ensure_directory(path.parent)
    if path.exists() and not force:
        raise FileExistsError(f"配置文件已存在: {path}")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(DEFAULT_CONFIG, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path
