from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


SPACE_RE = re.compile(r"\s+")
TITLE_RE = re.compile(r"[^a-z0-9]+")
SENTENCE_RE = re.compile(r"(?<=[.!?;。！？；])\s+")


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_whitespace(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def normalize_title(text: str) -> str:
    lowered = normalize_whitespace(text).lower()
    return TITLE_RE.sub(" ", lowered).strip()


def keyword_in_text(keyword: str, text: str) -> bool:
    normalized_keyword = normalize_title(keyword)
    return normalized_keyword in text


def unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = normalize_whitespace(value)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def safe_json_loads(text: str, default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def now_iso(timezone_name: str) -> str:
    return datetime.now(ZoneInfo(timezone_name)).isoformat(timespec="seconds")


def to_day_bounds(day_text: str, timezone_name: str, lookback_days: int = 1) -> tuple[str, str]:
    tz = ZoneInfo(timezone_name)
    target_date = date.fromisoformat(day_text)
    start_day = target_date - timedelta(days=max(lookback_days - 1, 0))
    start_at = datetime.combine(start_day, datetime.min.time(), tz).isoformat(timespec="seconds")
    end_at = datetime.combine(target_date, datetime.max.time().replace(microsecond=0), tz).isoformat(
        timespec="seconds"
    )
    return start_at, end_at


def today_string(timezone_name: str) -> str:
    return datetime.now(ZoneInfo(timezone_name)).date().isoformat()


def parse_source_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    candidate = value.strip()
    if len(candidate) == 4 and candidate.isdigit():
        return datetime.fromisoformat(f"{candidate}-01-01T00:00:00+00:00")
    if len(candidate) == 10 and candidate[4] == "-" and candidate[7] == "-":
        return datetime.fromisoformat(f"{candidate}T00:00:00+00:00")
    try:
        normalized = candidate.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


def choose_earlier_date(left: str | None, right: str | None) -> str | None:
    left_dt = parse_source_datetime(left)
    right_dt = parse_source_datetime(right)
    if left_dt and right_dt:
        return left if left_dt <= right_dt else right
    return left or right


def choose_later_date(left: str | None, right: str | None) -> str | None:
    left_dt = parse_source_datetime(left)
    right_dt = parse_source_datetime(right)
    if left_dt and right_dt:
        return left if left_dt >= right_dt else right
    return right or left


def split_sentences(text: str) -> list[str]:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return []
    parts = SENTENCE_RE.split(cleaned)
    return [part.strip() for part in parts if part.strip()]


def shorten(text: str, limit: int) -> str:
    cleaned = normalize_whitespace(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(limit - 3, 0)].rstrip() + "..."


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def clean_extracted_text(text: str) -> str:
    cleaned = text.replace("\x00", " ")
    cleaned = cleaned.replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
