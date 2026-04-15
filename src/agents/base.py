"""
Core multi-agent infrastructure: AgentIdentity, Message, MessageBus, BaseAgent.

Every inter-agent message flows through the MessageBus, which intercepts for
security checks (L5 authentication + L1 content filtering).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, TYPE_CHECKING

from src.llm_client import call_model
from src.tools.registry import ToolRegistry
from src.tools.simulated_tools import SimulatedEnvironment, ToolCall

if TYPE_CHECKING:
    pass  # avoid circular imports with safety module


class AgentRole(Enum):
    MANAGER = "manager"
    WORKER = "worker"
    PROPOSER = "proposer"
    CRITIC = "critic"
    ANALYST = "analyst"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    PEER = "peer"


# Default max tool tiers per role
_ROLE_DEFAULT_TIER: dict[AgentRole, str] = {
    AgentRole.MANAGER:  "network",
    AgentRole.WORKER:   "write",
    AgentRole.PROPOSER: "network",
    AgentRole.CRITIC:   "read_only",
    AgentRole.ANALYST:  "read_only",
    AgentRole.EXECUTOR: "execute",
    AgentRole.REVIEWER: "read_only",
    AgentRole.PEER:     "write",
}

_ROLE_DEFAULT_TRUST: dict[AgentRole, float] = {
    AgentRole.MANAGER:  0.8,
    AgentRole.WORKER:   0.6,
    AgentRole.PROPOSER: 0.7,
    AgentRole.CRITIC:   0.8,
    AgentRole.ANALYST:  0.7,
    AgentRole.EXECUTOR: 0.6,
    AgentRole.REVIEWER: 0.8,
    AgentRole.PEER:     0.7,
}


@dataclass
class AgentIdentity:
    """Cryptographic identity for an agent in the system."""
    agent_id: str
    role: AgentRole
    trust_level: float = 0.7
    secret_key: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    max_tool_tier: str = "write"

    def __post_init__(self):
        if not self.secret_key:
            self.secret_key = uuid.uuid4().hex
        if not self.trust_level:
            self.trust_level = _ROLE_DEFAULT_TRUST.get(self.role, 0.5)
        if not self.max_tool_tier:
            self.max_tool_tier = _ROLE_DEFAULT_TIER.get(self.role, "write")

    def sign_message(self, content: str) -> str:
        """HMAC-SHA256 signature over message content."""
        return hmac.new(
            self.secret_key.encode(), content.encode(), hashlib.sha256
        ).hexdigest()


@dataclass
class Message:
    """Inter-agent message with provenance and authentication."""
    msg_id: str = ""
    sender_id: str = ""
    receiver_id: str = ""         # "*" for broadcast
    content: str = ""
    msg_type: str = "discussion"  # task_delegation | result | feedback |
                                   # vote | discussion | critique
    signature: str = ""
    provenance_chain: list[str] = field(default_factory=list)
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    blocked: bool = False
    block_reason: str = ""

    def __post_init__(self):
        if not self.msg_id:
            self.msg_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "msg_id": self.msg_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "content": self.content[:500],
            "msg_type": self.msg_type,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
        }


class MessageBus:
    """Central message router with security interception.

    Every message passes through:
      1. L5 (Coordinate): sender authentication + trust check
      2. L1 (Inform): content filtering with source="inter_agent"
    """

    def __init__(self, security_pipeline=None):
        self.security_pipeline = security_pipeline
        self.message_log: list[Message] = []
        self._agent_registry: dict[str, AgentIdentity] = {}
        self._lock = threading.Lock()

    def register_agent(self, identity: AgentIdentity):
        self._agent_registry[identity.agent_id] = identity

    def send(self, message: Message) -> Message | None:
        """Route a message through security checks, then deliver.
        Returns the (possibly filtered) message, or None if blocked.
        """
        with self._lock:
            # L5: authenticate sender
            if self.security_pipeline is not None:
                auth = self.security_pipeline.authenticate_message(message)
                if not auth.authenticated:
                    message.blocked = True
                    message.block_reason = auth.reason
                    self.message_log.append(message)
                    return None

            # L1: filter content
            if self.security_pipeline is not None:
                filtered = self.security_pipeline.filter_context(
                    message.content,
                    source="inter_agent",
                    source_agent=message.sender_id,
                    provenance_chain=message.provenance_chain,
                )
                message.content = filtered

            self.message_log.append(message)
            return message

    def get_messages_for(self, agent_id: str,
                         msg_type: str | None = None) -> list[Message]:
        """Retrieve messages addressed to agent_id or broadcast."""
        results = []
        for msg in self.message_log:
            if msg.blocked:
                continue
            if msg.receiver_id in (agent_id, "*"):
                if msg_type is None or msg.msg_type == msg_type:
                    results.append(msg)
        return results

    def get_communication_graph(self) -> dict[str, list[str]]:
        """Return adjacency list of who communicated with whom."""
        graph: dict[str, list[str]] = {}
        for msg in self.message_log:
            if not msg.blocked:
                graph.setdefault(msg.sender_id, []).append(msg.receiver_id)
        return graph


class BaseAgent:
    """Abstract base for all agent types in the multi-agent system."""

    def __init__(self, identity: AgentIdentity, model_name: str,
                 registry: ToolRegistry, bus: MessageBus,
                 config: dict[str, Any] | None = None):
        self.identity = identity
        self.model_name = model_name
        self.registry = registry
        self.bus = bus
        self.config = config or {}
        self.conversation_history: list[dict] = []

    @property
    def agent_id(self) -> str:
        return self.identity.agent_id

    def build_system_prompt(self, task: dict, role_instruction: str = "") -> str:
        """Build a system prompt including available tools and role."""
        # Only show tools this agent is allowed to use
        tool_names = self.identity.allowed_tools or None
        tool_desc = self.registry.descriptions_for_prompt(tool_names)

        prompt = (
            f"You are agent '{self.identity.agent_id}' with role "
            f"'{self.identity.role.value}'.\n"
        )
        if role_instruction:
            prompt += role_instruction + "\n"

        if tool_desc:
            prompt += (
                "\nAvailable tools:\n" + tool_desc + "\n\n"
                "To use a tool, respond with a JSON block:\n"
                '```json\n{"tool": "<tool_name>", "arguments": {...}}\n```\n'
            )
        return prompt

    def call_llm(self, messages: list[dict]) -> str | None:
        """Call LLM with this agent's conversation."""
        return call_model(self.model_name, messages)

    def send_message(self, receiver_id: str, content: str,
                     msg_type: str = "discussion", **kwargs) -> Message | None:
        """Construct, sign, and send a message via the bus."""
        msg = Message(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            content=content,
            msg_type=msg_type,
            signature=self.identity.sign_message(content),
            provenance_chain=[self.agent_id],
            metadata=kwargs,
        )
        return self.bus.send(msg)

    def receive_messages(self, msg_type: str | None = None) -> list[Message]:
        """Retrieve pending messages from the bus."""
        return self.bus.get_messages_for(self.agent_id, msg_type)

    def parse_tool_call(self, text: str) -> dict | None:
        """Extract tool-call JSON from model output."""
        # Try ```json ... ``` blocks first
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
                if "tool" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass

        # Try raw JSON braces
        start = text.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(text[start:i + 1])
                            if "tool" in parsed:
                                return parsed
                        except json.JSONDecodeError:
                            pass
                        break
        return None

    def try_tool_call(self, parsed: dict, env: SimulatedEnvironment,
                      security_pipeline=None, task: dict | None = None) -> ToolCall:
        """Execute a tool call with security checking."""
        tool_name = parsed.get("tool", "")
        arguments = parsed.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        # Security check
        if security_pipeline is not None:
            decision = security_pipeline.check_action(
                tool_name, arguments, env, agent_id=self.agent_id
            )
            if decision.blocked:
                return env.record_blocked(
                    tool_name, arguments, decision.reason,
                    agent_id=self.agent_id
                )

        tc = env.execute(tool_name, arguments, agent_id=self.agent_id)

        # Inject indirect payload if present (for attack simulation)
        if task and task.get("_indirect_payload"):
            tc.result += "\n" + task["_indirect_payload"]

        # Filter tool output through L1
        if security_pipeline is not None:
            tc.result = security_pipeline.filter_context(
                tc.result, source="tool_output",
                source_agent=self.agent_id,
            )

        return tc
