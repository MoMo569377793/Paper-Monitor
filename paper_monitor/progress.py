from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TextIO


def _clean_detail(detail: str, limit: int = 72) -> str:
    text = " ".join(str(detail).split()).strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


@dataclass(slots=True)
class ProgressBar:
    label: str
    total: int
    enabled: bool = True
    width: int = 28
    stream: TextIO | None = None
    current: int = 0
    detail: str = ""
    _last_render_width: int = 0

    def __post_init__(self) -> None:
        self.stream = self.stream or sys.stderr
        self.total = max(int(self.total), 1)
        self.current = 0
        self.detail = ""
        self._last_render_width = 0
        self.enabled = bool(self.enabled and getattr(self.stream, "isatty", lambda: False)())
        if self.enabled:
            self._render()

    def set_detail(self, detail: str) -> None:
        if not self.enabled:
            return
        self.detail = _clean_detail(detail)
        self._render()

    def advance(self, step: int = 1, detail: str | None = None) -> None:
        if not self.enabled:
            return
        self.current = min(self.total, self.current + max(step, 0))
        if detail is not None:
            self.detail = _clean_detail(detail)
        self._render()

    def close(self, detail: str | None = None) -> None:
        if not self.enabled:
            return
        self.current = self.total
        if detail is not None:
            self.detail = _clean_detail(detail)
        self._render()
        self.stream.write("\n")
        self.stream.flush()

    def _render(self) -> None:
        ratio = self.current / self.total if self.total else 1.0
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        prefix = f"{self.label} [{bar}] {self.current}/{self.total}"
        text = f"{prefix} {self.detail}".rstrip()
        padded = text.ljust(self._last_render_width)
        self.stream.write("\r" + padded)
        self.stream.flush()
        self._last_render_width = max(self._last_render_width, len(text))
