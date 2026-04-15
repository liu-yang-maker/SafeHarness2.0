"""
Base harness with multi-agent result tracking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.tools.registry import ToolRegistry
from src.tools.simulated_tools import SimulatedEnvironment, ToolCall


@dataclass
class HarnessResult:
    task_id: int | Any = 0
    harness_type: str = ""
    security_mode: str = ""
    final_response: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_trace: list[dict[str, str]] = field(default_factory=list)
    blocked_actions: list[ToolCall] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Multi-agent fields
    agent_messages: list[dict] = field(default_factory=list)
    per_agent_tool_calls: dict[str, list[dict]] = field(default_factory=dict)
    blocked_messages: list[dict] = field(default_factory=list)
    cascading_events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "harness_type": self.harness_type,
            "security_mode": self.security_mode,
            "final_response": self.final_response[:2000],
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "blocked_actions": [tc.to_dict() for tc in self.blocked_actions],
            "reasoning_trace": self.reasoning_trace,
            "agent_messages": self.agent_messages[:50],
            "per_agent_tool_calls": {
                k: v[:20] for k, v in self.per_agent_tool_calls.items()
            },
            "blocked_messages": self.blocked_messages[:20],
            "cascading_events": self.cascading_events,
            "metadata": self.metadata,
        }


class BaseHarness(ABC):
    harness_type: str = "base"

    def __init__(self, model_name: str, registry: ToolRegistry,
                 config: dict[str, Any] | None = None):
        self.model_name = model_name
        self.registry = registry
        self.config = config or {}

    @abstractmethod
    def execute(self, task: dict, env: SimulatedEnvironment,
                security_pipeline=None) -> HarnessResult:
        ...
