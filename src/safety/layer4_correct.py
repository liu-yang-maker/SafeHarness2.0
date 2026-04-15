"""
Layer 4 — CORRECT: Safe Rollback & Adaptive Degradation/Recovery.

Per-agent checkpointing, cascading failure detection, and quarantine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.tools.registry import TIER_ORDER
from src.tools.simulated_tools import SimulatedEnvironment
from src.safety.memory_protection import MemorySnapshot


@dataclass
class Checkpoint:
    step: int
    agent_id: str
    filesystem_snapshot: dict[str, str]
    history_length: int
    memory_snapshot: MemorySnapshot | None = None


@dataclass
class CascadingEvent:
    """Record of a detected cascading failure."""
    trigger_agent: str
    affected_agents: list[str]
    step: int
    severity: str = "medium"     # low | medium | high


# Degradation level -> max allowed tier
_DEGRADATION_MAP = {
    0: "destructive",   # normal
    1: "network",       # destructive disabled
    2: "execute",       # network disabled
    3: "write",         # execute disabled
    4: "read_only",     # write disabled
}


class RollbackLayer:
    """L4 — Correct: per-agent checkpointing, rollback, degradation."""

    def __init__(self, config: dict | None = None, enabled: bool = True):
        self.config = config or {}
        self.enabled = enabled
        self._checkpoint_interval = self.config.get("checkpoint_interval", 3)
        self._max_degradation = self.config.get("max_degradation_level", 4)
        self._recovery_window = self.config.get("recovery_window", 5)

        # Per-agent state
        self._per_agent_checkpoints: dict[str, list[Checkpoint]] = {}
        self._per_agent_degradation: dict[str, int] = {}
        self._per_agent_consecutive_safe: dict[str, int] = {}
        self._per_agent_step: dict[str, int] = {}
        self._quarantined: set[str] = set()
        self._cascading_events: list[CascadingEvent] = []

    def maybe_checkpoint(self, step: int, env: SimulatedEnvironment,
                         agent_id: str = "primary",
                         memory_snapshot: MemorySnapshot | None = None):
        """Create checkpoint if at the right interval."""
        if not self.enabled:
            return

        self._per_agent_step[agent_id] = step
        if step % self._checkpoint_interval == 0:
            cp = Checkpoint(
                step=step,
                agent_id=agent_id,
                filesystem_snapshot=dict(env.filesystem),
                history_length=len(env.history),
                memory_snapshot=memory_snapshot,
            )
            self._per_agent_checkpoints.setdefault(agent_id, []).append(cp)

    def rollback(self, env: SimulatedEnvironment,
                 agent_id: str = "primary") -> Checkpoint | None:
        """Restore to last checkpoint for the given agent."""
        if not self.enabled:
            return None
        checkpoints = self._per_agent_checkpoints.get(agent_id, [])
        if not checkpoints:
            return None

        cp = checkpoints[-1]
        env.filesystem = dict(cp.filesystem_snapshot)
        env.history = env.history[:cp.history_length]
        return cp

    def escalate_degradation(self, agent_id: str = "primary") -> int:
        """Increase restriction level for agent."""
        current = self._per_agent_degradation.get(agent_id, 0)
        new_level = min(current + 1, self._max_degradation)
        self._per_agent_degradation[agent_id] = new_level
        self._per_agent_consecutive_safe[agent_id] = 0
        return new_level

    def maybe_recover(self, agent_id: str = "primary") -> bool:
        """After recovery_window consecutive safe actions, reduce degradation."""
        if not self.enabled:
            return False

        safe_count = self._per_agent_consecutive_safe.get(agent_id, 0) + 1
        self._per_agent_consecutive_safe[agent_id] = safe_count

        current = self._per_agent_degradation.get(agent_id, 0)
        if current > 0 and safe_count >= self._recovery_window:
            self._per_agent_degradation[agent_id] = current - 1
            self._per_agent_consecutive_safe[agent_id] = 0
            return True
        return False

    def max_allowed_tier(self, agent_id: str = "primary") -> str:
        """Get max tier name based on current degradation level."""
        level = self._per_agent_degradation.get(agent_id, 0)
        return _DEGRADATION_MAP.get(level, "read_only")

    def detect_cascade(self, message_log: list,
                       degraded_agents: set[str] | None = None) -> list[CascadingEvent]:
        """Analyze message log for cascading failure patterns.

        Checks if degraded agents sent messages to other agents that
        subsequently also got degraded.
        """
        if not self.enabled or degraded_agents is None:
            return []

        events = []
        for msg in message_log:
            if hasattr(msg, "blocked") and msg.blocked:
                continue
            sender = getattr(msg, "sender_id", "")
            receiver = getattr(msg, "receiver_id", "")
            if sender in degraded_agents and receiver not in degraded_agents:
                # Degraded agent communicating with non-degraded agent
                if receiver in self._per_agent_degradation:
                    event = CascadingEvent(
                        trigger_agent=sender,
                        affected_agents=[receiver],
                        step=self._per_agent_step.get(sender, 0),
                        severity="medium",
                    )
                    events.append(event)
                    self._cascading_events.append(event)

        return events

    def quarantine_agent(self, agent_id: str):
        """Set agent to maximum degradation and flag for isolation."""
        self._per_agent_degradation[agent_id] = self._max_degradation
        self._quarantined.add(agent_id)

    @property
    def quarantined_agents(self) -> set[str]:
        return set(self._quarantined)

    @property
    def cascading_events(self) -> list[CascadingEvent]:
        return list(self._cascading_events)

    def record_safe_action(self, agent_id: str = "primary"):
        """Record that agent took a safe action (for recovery tracking)."""
        self._per_agent_consecutive_safe[agent_id] = \
            self._per_agent_consecutive_safe.get(agent_id, 0) + 1

    def reset_safe_counter(self, agent_id: str = "primary"):
        """Reset safe counter on violation."""
        self._per_agent_consecutive_safe[agent_id] = 0
