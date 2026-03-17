"""Helpers for visible progress during long synchronous phases."""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable, TextIO


@dataclass
class HeartbeatFrames:
    """Generate dot suffixes from one to six, then reset."""

    max_dots: int = 6
    _count: int = 0

    def next_suffix(self) -> str:
        self._count = 1 if self._count >= self.max_dots else self._count + 1
        return "." * self._count


def run_with_heartbeat(
    message: str,
    fn: Callable[[], Any],
    *,
    interval: float = 0.5,
    stream: TextIO | None = None,
) -> Any:
    """Run a blocking function while emitting a lightweight heartbeat."""

    output = stream or sys.stdout
    stop_event = threading.Event()
    frames = HeartbeatFrames()

    def _render() -> None:
        output.write(f"\r{message}{frames.next_suffix()}")
        output.flush()

    def _worker() -> None:
        while not stop_event.wait(interval):
            _render()

    _render()
    thread = threading.Thread(target=_worker, name="sft-label-heartbeat", daemon=True)
    thread.start()
    try:
        return fn()
    finally:
        stop_event.set()
        thread.join(timeout=max(interval, 0.05) * 2)
        output.write("\n")
        output.flush()
