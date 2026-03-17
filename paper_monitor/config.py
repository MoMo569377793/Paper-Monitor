from __future__ import annotations

import json
from pathlib import Path

from paper_monitor.models import (
    BootstrapConfig,
    EnrichmentConfig,
    GenericSourceConfig,
    LLMConfig,
    PromptPaths,
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
    "bootstrap": {
        "start_year": None,
        "recent_limit": None,
        "page_size": 25,
    },
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
    "enrichment": {
        "enabled": True,
        "pdf_dir": "artifacts/pdfs",
        "text_dir": "artifacts/text",
        "download_timeout_seconds": 60,
        "max_papers_per_run": 20,
        "redownload_existing": False,
        "use_pdftotext": True,
        "excerpt_chars": 12000,
        "process_classifications": ["relevant", "maybe"],
    },
    "prompts": {
        "paper_summary_system": "prompts/paper_summary_system.txt",
        "paper_summary_user": "prompts/paper_summary_user.txt",
        "paper_chunk_system": "prompts/paper_chunk_system.txt",
        "paper_chunk_user": "prompts/paper_chunk_user.txt",
        "paper_reduce_system": "prompts/paper_reduce_system.txt",
        "paper_reduce_user": "prompts/paper_reduce_user.txt",
        "topic_digest_system": "prompts/topic_digest_system.txt",
        "topic_digest_user": "prompts/topic_digest_user.txt",
    },
    "llm": {
        "variant_id": "primary",
        "label": "主模型",
        "enabled": False,
        "provider": "openai_compatible",
        "base_url": "",
        "api_key_env": "LLM_API_KEY",
        "model": "",
        "timeout_seconds": 60,
        "temperature": 0.2,
        "model_reasoning_effort": "",
        "model_output_effort": "",
        "model_thinking_level": "",
        "max_input_chars": 16000,
        "max_output_tokens": 700,
        "store": False,
        "extra_body": {},
        "reasoning_by_task": {},
        "output_effort_by_task": {},
        "thinking_level_by_task": {},
        "enable_topic_digest": False,
        "topic_digest_entry_limit": 8,
        "fulltext_chunk_chars": 10000,
        "fulltext_chunk_overlap_chars": 1200,
        "fulltext_max_chunks": 12,
        "pdf_input_mode": "auto",
        "pdf_inline_max_bytes": 12582912,
    },
    "llm_variants": [],
    "topics": [
        {
            "id": "matrix_free_fem",
            "display_name": "有限元分析 Matrix-Free 算法优化",
            "description": "跟踪 matrix-free、partial assembly、sum factorization、高阶有限元在 GPU、多核、SIMD、性能可移植和算子应用优化方面的论文，更偏实现与性能优化而非纯数学分析。",
            "source_queries": {
                "arxiv": [
                    "all:\"matrix-free\" AND all:\"finite element\"",
                    "all:\"partial assembly\" AND all:\"finite element\"",
                    "all:\"sum factorization\" AND all:\"finite element\"",
                    "all:\"matrix-free\" AND all:\"operator application\"",
                    "all:\"matrix-free\" AND all:\"GPU\"",
                    "all:\"matrix-free\" AND all:\"performance portability\"",
                    "all:\"matrix-free\" AND all:\"vectorization\"",
                    "all:\"libCEED\"",
                ],
                "dblp": [
                    "\"matrix-free\" \"finite element\"",
                    "\"partial assembly\" \"finite element\"",
                    "\"sum factorization\" \"finite element\"",
                    "\"matrix-free\" \"operator application\"",
                    "\"matrix-free\" GPU",
                    "\"matrix-free\" \"performance portability\"",
                    "\"matrix-free\" vectorization",
                    "libCEED matrix-free",
                ],
                "google_scholar_alerts": [
                    "\"matrix-free\" \"finite element\"",
                    "\"partial assembly\" FEM",
                    "\"sum factorization\" finite element",
                    "\"matrix-free\" \"operator application\"",
                    "\"matrix-free\" GPU finite element",
                    "\"matrix-free\" \"performance portability\"",
                ],
            },
            "must_match_groups": [
                ["matrix-free", "matrix free", "partial assembly", "sum factorization"],
                ["finite element", "finite elements", "fem", "high-order fem", "high order finite element", "galerkin", "discontinuous galerkin", "dg", "spectral element"],
                ["operator", "application", "gpu", "performance", "vectorization", "parallel", "multigrid", "preconditioner", "simd", "scaling"],
            ],
            "positive_keywords": [
                "libceed",
                "mfem",
                "deal.ii",
                "deal ii",
                "operator application",
                "fused operator",
                "multigrid",
                "preconditioner",
                "spectral element",
                "tensor contraction",
                "element restriction",
                "performance portability",
                "roofline",
                "vectorization",
                "simd",
                "avx512",
                "sve",
                "gpu",
                "gpu acceleration",
                "cuda",
                "hip",
                "kokkos",
                "raja",
                "strong scaling",
                "weak scaling",
                "memory bandwidth",
                "cache blocking",
            ],
            "exclude_keywords": [
                "matrix factorization",
                "graph neural network",
                "diffusion model",
                "recommendation system",
                "a posteriori error",
                "a priori error",
                "well posedness",
                "existence and uniqueness",
                "convergence proof",
                "inverse problem",
            ],
            "arxiv_categories": ["cs.DC", "cs.MS", "cs.NA", "math.NA"],
            "priority_arxiv_categories": ["cs.DC", "cs.MS", "cs.NA"],
            "dblp_venue_keywords": ["finite elements", "supercomputing", "high performance", "sc", "ppopp", "ipdps", "ics", "cluster", "cgo", "asplos"],
            "priority_venue_keywords": ["sc", "ppopp", "ipdps", "ics", "cluster", "cgo", "asplos"],
            "threshold": 18,
        },
        {
            "id": "ai_operator_acceleration",
            "display_name": "AI 算子加速",
            "description": "跟踪 AI 算子优化、内核融合、编译器、GEMM/matmul、Tensor Core、ARM SVE/SME、推理与训练性能优化相关论文，并额外关注 CUTLASS、Triton、FlashAttention 系列扩展阅读线索。",
            "source_queries": {
                "arxiv": [
                    "all:\"kernel fusion\"",
                    "all:\"operator fusion\"",
                    "all:\"tensor compiler\"",
                    "all:\"FlashAttention\"",
                    "all:\"MLIR\" AND all:\"deep learning\"",
                    "all:\"CUTLASS\"",
                    "all:\"Triton\" AND all:\"compiler\"",
                    "all:\"GEMM optimization\"",
                    "all:\"matrix multiplication\" AND all:\"optimization\"",
                    "all:\"Tensor Core\" AND all:\"deep learning\"",
                    "all:\"SVE\" AND all:\"GEMM\"",
                    "all:\"SME\" AND all:\"GEMM\"",
                ],
                "dblp": [
                    "\"kernel fusion\" deep learning",
                    "\"tensor compiler\" deep learning",
                    "\"operator acceleration\" AI",
                    "FlashAttention",
                    "MLIR deep learning compiler",
                    "CUTLASS GEMM",
                    "Triton compiler deep learning",
                    "\"GEMM optimization\" accelerator",
                    "\"matrix multiplication\" optimization",
                    "\"Tensor Core\" deep learning",
                    "ARM SVE GEMM",
                    "ARM SME GEMM",
                ],
                "google_scholar_alerts": [
                    "\"kernel fusion\" transformer",
                    "\"tensor compiler\" deep learning",
                    "\"operator acceleration\" AI",
                    "FlashAttention kernel optimization",
                    "CUTLASS GEMM kernel",
                    "Triton kernel compiler",
                    "\"GEMM optimization\" AI",
                    "\"Tensor Core\" transformer",
                    "\"ARM SVE\" GEMM",
                    "\"ARM SME\" GEMM",
                ],
            },
            "must_match_groups": [
                ["kernel", "operator", "compiler", "fusion", "acceleration", "optimization", "optimized", "scheduling"],
                [
                    "deep learning",
                    "llm",
                    "transformer",
                    "tensor",
                    "inference",
                    "training",
                    "gemm",
                    "matmul",
                    "matrix multiplication",
                    "tensor core",
                    "arm",
                    "sve",
                    "sme",
                ],
            ],
            "positive_keywords": [
                "triton",
                "tvm",
                "mlir",
                "xla",
                "tensorrt",
                "cuda",
                "gemm",
                "matmul",
                "matrix multiplication",
                "attention kernel",
                "flashattention",
                "cutlass",
                "wmma",
                "mma",
                "tensor core",
                "tensor cores",
                "cublas",
                "rocblas",
                "onednn",
                "arm",
                "armv9",
                "sve",
                "sme",
                "neon",
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
            "priority_arxiv_categories": ["cs.DC", "cs.AR", "cs.PF", "cs.LG"],
            "dblp_venue_keywords": ["ppopp", "mlsys", "isca", "micro", "hpca", "asplos", "sc", "osdi", "sosp"],
            "priority_venue_keywords": ["mlsys", "ppopp", "sc", "isca", "micro", "asplos", "osdi", "sosp"],
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


def _slugify(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "variant"


def _build_llm_config(raw: dict, *, config_stem: str, index: int, primary: bool) -> LLMConfig:
    model = str(raw.get("model", "")).strip()
    base_variant = str(raw.get("variant_id", "")).strip()
    if not base_variant:
        suffix = model or ("primary" if primary else f"variant-{index}")
        base_variant = f"{config_stem}-{_slugify(suffix)}"
    label = str(raw.get("label", "")).strip() or model or base_variant
    return LLMConfig(
        variant_id=base_variant,
        label=label,
        enabled=bool(raw.get("enabled", False)),
        provider=str(raw.get("provider", "openai_compatible")),
        base_url=str(raw.get("base_url", "")),
        api_key_env=str(raw.get("api_key_env", "LLM_API_KEY")),
        model=model,
        timeout_seconds=int(raw.get("timeout_seconds", 60)),
        temperature=float(raw.get("temperature", 0.2)),
        model_reasoning_effort=str(raw.get("model_reasoning_effort", raw.get("reasoning_effort", ""))),
        model_output_effort=str(raw.get("model_output_effort", raw.get("output_effort", ""))),
        model_thinking_level=str(raw.get("model_thinking_level", raw.get("thinking_level", ""))),
        max_input_chars=int(raw.get("max_input_chars", 16000)),
        max_output_tokens=int(raw.get("max_output_tokens", 700)),
        store=bool(raw.get("store", False)),
        extra_body=dict(raw.get("extra_body", {})) if isinstance(raw.get("extra_body", {}), dict) else {},
        reasoning_by_task=(
            dict(raw.get("reasoning_by_task", {})) if isinstance(raw.get("reasoning_by_task", {}), dict) else {}
        ),
        output_effort_by_task=(
            dict(raw.get("output_effort_by_task", {}))
            if isinstance(raw.get("output_effort_by_task", {}), dict)
            else {}
        ),
        thinking_level_by_task=(
            dict(raw.get("thinking_level_by_task", {})) if isinstance(raw.get("thinking_level_by_task", {}), dict) else {}
        ),
        max_output_tokens_by_task=(
            {
                str(key): int(value)
                for key, value in dict(raw.get("max_output_tokens_by_task", {})).items()
            }
            if isinstance(raw.get("max_output_tokens_by_task", {}), dict)
            else {}
        ),
        enable_topic_digest=bool(raw.get("enable_topic_digest", False)),
        topic_digest_entry_limit=int(raw.get("topic_digest_entry_limit", 8)),
        fulltext_chunk_chars=int(raw.get("fulltext_chunk_chars", 10000)),
        fulltext_chunk_overlap_chars=int(raw.get("fulltext_chunk_overlap_chars", 1200)),
        fulltext_max_chunks=int(raw.get("fulltext_max_chunks", 12)),
        pdf_input_mode=str(raw.get("pdf_input_mode", "auto")),
        pdf_inline_max_bytes=int(raw.get("pdf_inline_max_bytes", 12582912)),
    )


def load_settings(config_path: str | Path) -> Settings:
    path = Path(config_path).resolve()
    raw = _load_json(path)
    base_dir = _resolve_base_dir(path)
    config_stem = path.stem

    bootstrap_raw = raw.get("bootstrap", {})
    bootstrap = BootstrapConfig(
        start_year=bootstrap_raw.get("start_year"),
        recent_limit=bootstrap_raw.get("recent_limit"),
        page_size=int(bootstrap_raw.get("page_size", 25)),
    )
    sources = raw.get("sources", {})
    arxiv = GenericSourceConfig(**sources.get("arxiv", {}))
    dblp = GenericSourceConfig(**sources.get("dblp", {}))
    scholar_alerts = ScholarAlertsConfig(**sources.get("google_scholar_alerts", {}))
    report = ReportConfig(**raw.get("report", {}))
    enrichment_raw = raw.get("enrichment", {})
    enrichment = EnrichmentConfig(
        enabled=bool(enrichment_raw.get("enabled", True)),
        pdf_dir=(base_dir / enrichment_raw.get("pdf_dir", "artifacts/pdfs")).resolve(),
        text_dir=(base_dir / enrichment_raw.get("text_dir", "artifacts/text")).resolve(),
        download_timeout_seconds=int(enrichment_raw.get("download_timeout_seconds", 60)),
        max_papers_per_run=int(enrichment_raw.get("max_papers_per_run", 20)),
        redownload_existing=bool(enrichment_raw.get("redownload_existing", False)),
        use_pdftotext=bool(enrichment_raw.get("use_pdftotext", True)),
        excerpt_chars=int(enrichment_raw.get("excerpt_chars", 12000)),
        process_classifications=list(enrichment_raw.get("process_classifications", ["relevant", "maybe"])),
    )
    prompts_raw = raw.get("prompts", {})
    prompt_paths = PromptPaths(
        paper_summary_system=(base_dir / prompts_raw.get("paper_summary_system", "prompts/paper_summary_system.txt")).resolve(),
        paper_summary_user=(base_dir / prompts_raw.get("paper_summary_user", "prompts/paper_summary_user.txt")).resolve(),
        paper_chunk_system=(base_dir / prompts_raw.get("paper_chunk_system", "prompts/paper_chunk_system.txt")).resolve(),
        paper_chunk_user=(base_dir / prompts_raw.get("paper_chunk_user", "prompts/paper_chunk_user.txt")).resolve(),
        paper_reduce_system=(base_dir / prompts_raw.get("paper_reduce_system", "prompts/paper_reduce_system.txt")).resolve(),
        paper_reduce_user=(base_dir / prompts_raw.get("paper_reduce_user", "prompts/paper_reduce_user.txt")).resolve(),
        topic_digest_system=(base_dir / prompts_raw.get("topic_digest_system", "prompts/topic_digest_system.txt")).resolve(),
        topic_digest_user=(base_dir / prompts_raw.get("topic_digest_user", "prompts/topic_digest_user.txt")).resolve(),
    )
    llm = _build_llm_config(raw.get("llm", {}), config_stem=config_stem, index=0, primary=True)
    llm_variants = [
        _build_llm_config(item, config_stem=config_stem, index=index, primary=False)
        for index, item in enumerate(raw.get("llm_variants", []), start=1)
    ]

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
        bootstrap=bootstrap,
        arxiv=arxiv,
        dblp=dblp,
        scholar_alerts=scholar_alerts,
        report=report,
        enrichment=enrichment,
        prompt_paths=prompt_paths,
        llm=llm,
        llm_variants=llm_variants,
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
