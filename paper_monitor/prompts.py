from __future__ import annotations

from pathlib import Path

from paper_monitor.models import PromptPaths
from paper_monitor.utils import ensure_directory


DEFAULT_PAPER_SUMMARY_SYSTEM = (
    "你是一个严谨的中文论文分析助手。"
    "请根据提供的标题、摘要、全文信息和相关性标签，输出简洁、具体、避免空话的 JSON。"
    "如果请求里直接附带 PDF 文件，应优先阅读全文后再总结。"
)

DEFAULT_PAPER_SUMMARY_USER = """请仅依据提供的信息进行总结，不要虚构实验结果。
summary 需要 120-220 字中文；problem/method/application/results 用 1-2 句；contributions 和 limitations 各给 2-4 条短句数组；tags 给 3-8 个关键词。
特别关注：
1. 这篇论文解决了什么问题
2. 提出了什么核心方法
3. 主要应用在什么领域或场景
4. 论文给出了什么关键结果或证据
5. 如果本次请求直接附带 PDF 文件，请以 PDF 全文为准，basis 输出 llm+pdf+metadata
6. 如果本次请求提供的是完整抽取全文文本，basis 输出 llm+fulltext+metadata
7. 如果本次只能看到摘要和元数据，basis 输出 llm+abstract+metadata

标题: {title}
作者: {authors}
发表信息: venue={venue}, published_at={published_at}
相关主题:
{topics_text}

摘要:
{abstract}

完整全文分析说明:
{fulltext}
"""

DEFAULT_PAPER_CHUNK_SYSTEM = (
    "你是一个严谨的中文论文分析助手。"
    "你正在阅读一篇论文的一个正文分块。"
    "请只提取这一分块中明确出现的事实、方法、实验线索和应用场景，不要脑补整篇论文没有写出的内容。"
)

DEFAULT_PAPER_CHUNK_USER = """你正在阅读论文《{title}》的正文分块 {chunk_index}/{chunk_total}。
请只基于当前分块内容，写一份供后续聚合使用的中文备忘录。

输出格式要求：
1. 只输出正文，不要输出 JSON，不要输出 Markdown 代码块
2. 使用以下固定小标题：
分块概括：
关键方法：
结果/证据：
应用场景：
局限：
3. 每个小标题下写 1-4 条短句，保留具体方法、实验、结论和适用场景
4. 不要脑补整篇论文没有写出的信息

标题: {title}
作者: {authors}
发表信息: venue={venue}, published_at={published_at}
相关主题:
{topics_text}

摘要:
{abstract}

当前正文分块:
{chunk_text}
"""

DEFAULT_PAPER_REDUCE_SYSTEM = (
    "你是一个严谨的中文论文分析助手。"
    "你已经看过一篇论文全文的多段分块笔记。"
    "现在请基于全部分块笔记，写出高质量、结构化、面向研究监控场景的最终总结。"
    "不要输出思考过程，不要输出无法从材料中支持的结论。"
)

DEFAULT_PAPER_REDUCE_USER = """请基于整篇论文的分块分析结果，生成最终中文总结。

目标：
1. 明确这篇论文解决了什么问题
2. 明确提出了什么核心方法
3. 明确适用于什么任务、场景或领域
4. 明确论文给出的关键结果、证据或实验结论
5. 明确主要贡献和局限

输出要求：
- summary：150-260 字，必须体现“问题、方法、应用、结果”
- problem/method/application/results：各 1-2 句
- contributions：2-4 条
- limitations：2-4 条
- tags：3-8 个中文或英文关键词
- basis：如果分块来自完整 PDF，请输出 llm+fulltext+metadata

标题: {title}
作者: {authors}
发表信息: venue={venue}, published_at={published_at}
相关主题:
{topics_text}

摘要:
{abstract}

整篇论文的分块分析结果:
{chunk_notes}
"""

DEFAULT_TOPIC_DIGEST_SYSTEM = (
    "你是一个严谨的中文研究情报分析助手。"
    "请根据一个主题下的多篇论文信息，生成简洁的中文主题摘要，突出趋势、代表工作和建议关注点。"
)

DEFAULT_TOPIC_DIGEST_USER = """主题: {topic_name}
说明: {description}
论文数量: {paper_count}

请总结这个主题在当前时间窗口的趋势。
highlights 应是 2-4 条关键观察，watchlist 应是 2-4 条建议关注的论文或技术线索。

{paper_blocks}
"""


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class PromptLibrary:
    def __init__(self, prompt_paths: PromptPaths) -> None:
        self.prompt_paths = prompt_paths
        self._ensure_defaults()

    def _ensure_defaults(self) -> None:
        self._ensure_file(self.prompt_paths.paper_summary_system, DEFAULT_PAPER_SUMMARY_SYSTEM)
        self._ensure_file(self.prompt_paths.paper_summary_user, DEFAULT_PAPER_SUMMARY_USER)
        self._ensure_file(self.prompt_paths.paper_chunk_system, DEFAULT_PAPER_CHUNK_SYSTEM)
        self._ensure_file(self.prompt_paths.paper_chunk_user, DEFAULT_PAPER_CHUNK_USER)
        self._ensure_file(self.prompt_paths.paper_reduce_system, DEFAULT_PAPER_REDUCE_SYSTEM)
        self._ensure_file(self.prompt_paths.paper_reduce_user, DEFAULT_PAPER_REDUCE_USER)
        self._ensure_file(self.prompt_paths.topic_digest_system, DEFAULT_TOPIC_DIGEST_SYSTEM)
        self._ensure_file(self.prompt_paths.topic_digest_user, DEFAULT_TOPIC_DIGEST_USER)

    def _ensure_file(self, path: Path, content: str) -> None:
        ensure_directory(path.parent)
        if not path.exists():
            path.write_text(content.strip() + "\n", encoding="utf-8")

    def paper_summary_system(self) -> str:
        return self.prompt_paths.paper_summary_system.read_text(encoding="utf-8").strip()

    def paper_summary_user(self, context: dict[str, str]) -> str:
        template = self.prompt_paths.paper_summary_user.read_text(encoding="utf-8")
        return template.format_map(_SafeDict(context)).strip()

    def paper_chunk_system(self) -> str:
        return self.prompt_paths.paper_chunk_system.read_text(encoding="utf-8").strip()

    def paper_chunk_user(self, context: dict[str, str]) -> str:
        template = self.prompt_paths.paper_chunk_user.read_text(encoding="utf-8")
        return template.format_map(_SafeDict(context)).strip()

    def paper_reduce_system(self) -> str:
        return self.prompt_paths.paper_reduce_system.read_text(encoding="utf-8").strip()

    def paper_reduce_user(self, context: dict[str, str]) -> str:
        template = self.prompt_paths.paper_reduce_user.read_text(encoding="utf-8")
        return template.format_map(_SafeDict(context)).strip()

    def topic_digest_system(self) -> str:
        return self.prompt_paths.topic_digest_system.read_text(encoding="utf-8").strip()

    def topic_digest_user(self, context: dict[str, str]) -> str:
        template = self.prompt_paths.topic_digest_user.read_text(encoding="utf-8")
        return template.format_map(_SafeDict(context)).strip()
