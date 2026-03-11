from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from paper_monitor.config import load_settings
from paper_monitor.llm import LLMClient
from paper_monitor.models import LLMConfig, Settings
from paper_monitor.prompts import PromptLibrary


@dataclass(slots=True)
class LLMRuntimeVariant:
    variant_id: str
    label: str
    provider: str
    base_url: str
    model: str
    config_path: Path
    client: LLMClient


def _iter_llm_configs(settings: Settings) -> list[LLMConfig]:
    return [settings.llm, *settings.llm_variants]


def _validate_compatible_settings(base: Settings, other: Settings) -> None:
    if base.database_path != other.database_path:
        raise ValueError("多模型模式要求所有配置指向同一个数据库。")
    if [topic.id for topic in base.topics] != [topic.id for topic in other.topics]:
        raise ValueError("多模型模式要求所有配置使用相同的 topic 列表。")


def build_runtime_variants(settings: Settings, extra_config_paths: list[str] | None = None) -> list[LLMRuntimeVariant]:
    prompt_library = PromptLibrary(settings.prompt_paths)
    all_settings = [settings]
    for config_path in extra_config_paths or []:
        loaded = load_settings(config_path)
        _validate_compatible_settings(settings, loaded)
        all_settings.append(loaded)

    variants: list[LLMRuntimeVariant] = []
    seen_ids: set[str] = set()
    for current in all_settings:
        config_label = current.config_path.name
        for llm_config in _iter_llm_configs(current):
            variant_id = llm_config.variant_id
            if variant_id in seen_ids:
                raise ValueError(f"检测到重复的 LLM variant_id: {variant_id}")
            seen_ids.add(variant_id)
            client = LLMClient(llm_config, prompt_library=prompt_library)
            variants.append(
                LLMRuntimeVariant(
                    variant_id=variant_id,
                    label=f"{config_label} / {llm_config.label or llm_config.model or variant_id}",
                    provider=llm_config.provider,
                    base_url=llm_config.base_url,
                    model=llm_config.model,
                    config_path=current.config_path,
                    client=client,
                )
            )
    return variants
