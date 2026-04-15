"""
Cross-layer memory protection and entropy monitoring.

ProtectedMemory: append-only key-value store with per-agent provenance.
EntropyMonitor: sliding-window violation rate tracker.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class MemoryEntry:
    key: str
    value: str
    source_type: str        # "user" | "tool" | "agent" | "system"
    source_agent: str = ""
    timestamp: str = ""
    hash: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.hash:
            self.hash = hashlib.sha256(
                self.value.encode()
            ).hexdigest()[:16]


MemorySnapshot = dict[str, list[MemoryEntry]]


class ProtectedMemory:
    """Append-only memory store with per-agent provenance tracking."""

    def __init__(self):
        self._store: dict[str, list[MemoryEntry]] = {}

    def write(self, key: str, value: str,
              source_type: str = "agent",
              source_agent: str = ""):
        entry = MemoryEntry(
            key=key, value=value,
            source_type=source_type,
            source_agent=source_agent,
        )
        self._store.setdefault(key, []).append(entry)

    def read(self, key: str) -> MemoryEntry | None:
        entries = self._store.get(key, [])
        return entries[-1] if entries else None

    def snapshot(self) -> MemorySnapshot:
        return {k: list(v) for k, v in self._store.items()}

    def restore(self, snap: MemorySnapshot):
        self._store = {k: list(v) for k, v in snap.items()}

    def detect_anomalies(self) -> list[str]:
        """Check for anomalous memory patterns."""
        anomalies = []
        total = sum(len(v) for v in self._store.values())
        if total == 0:
            return anomalies

        # Check for key mutations (same key written by different agents)
        for key, entries in self._store.items():
            agents = set(e.source_agent for e in entries if e.source_agent)
            if len(agents) > 1:
                anomalies.append(
                    f"Key '{key}' written by multiple agents: {agents}"
                )

        # Check for high ratio of non-system writes
        non_system = sum(
            1 for entries in self._store.values()
            for e in entries if e.source_type != "system"
        )
        if total > 5 and non_system / total > 0.8:
            anomalies.append(
                f"High ratio of non-system writes: {non_system}/{total}"
            )

        # Check single-agent write dominance
        agent_counts: dict[str, int] = {}
        for entries in self._store.values():
            for e in entries:
                if e.source_agent:
                    agent_counts[e.source_agent] = agent_counts.get(
                        e.source_agent, 0
                    ) + 1
        if agent_counts and total > 5:
            max_agent = max(agent_counts.values())
            if max_agent / total > 0.7:
                dominant = [a for a, c in agent_counts.items() if c == max_agent]
                anomalies.append(
                    f"Single-agent write dominance: {dominant[0]} "
                    f"({max_agent}/{total})"
                )

        return anomalies


class EntropyMonitor:
    """Sliding-window entropy monitor tracking violation rates."""

    def __init__(self, window_size: int = 20, threshold: float = 0.3):
        self.window_size = window_size
        self.threshold = threshold
        self._window: deque[bool] = deque(maxlen=window_size)

    def record_action(self, violated: bool = False):
        self._window.append(violated)

    def entropy_score(self) -> float:
        """Violation rate in the current window."""
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    def is_drifting(self, threshold: float | None = None) -> bool:
        t = threshold if threshold is not None else self.threshold
        return self.entropy_score() > t
