from __future__ import annotations

from pathlib import Path

from paper_monitor.models import PromptPaths
from paper_monitor.utils import ensure_directory


DEFAULT_PAPER_SUMMARY_SYSTEM = (
    "你是一个严谨的中文论文分析助手。"
    "请根据提供的标题、摘要、全文节选和相关性标签，输出简洁、具体、避免空话的 JSON。"
)

DEFAULT_PAPER_SUMMARY_USER = """请仅依据提供的信息进行总结，不要虚构实验结果。
summary 需要 120-220 字中文；problem/method 用 1-2 句；contributions 和 limitations 各给 2-4 条短句数组；tags 给 3-8 个关键词。
特别关注：
1. 这篇论文解决了什么问题
2. 提出了什么核心方法
3. 主要应用在什么领域或场景

标题: {title}
作者: {authors}
发表信息: venue={venue}, published_at={published_at}
相关主题:
{topics_text}

摘要:
{abstract}

全文节选:
{fulltext}
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

    def topic_digest_system(self) -> str:
        return self.prompt_paths.topic_digest_system.read_text(encoding="utf-8").strip()

    def topic_digest_user(self, context: dict[str, str]) -> str:
        template = self.prompt_paths.topic_digest_user.read_text(encoding="utf-8")
        return template.format_map(_SafeDict(context)).strip()
