"""
Layer 3 — CONSTRAIN: Privilege-Separated Tool Control.

Per-agent capability tokens, cross-agent access control, HMAC signature
verification, and tier ceiling enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.tools.registry import (
    ToolRegistry, CapabilityToken, TIER_ORDER, tier_index,
    verify_signature, verify_token_signature,
)


@dataclass
class L3Decision:
    allowed: bool
    reason: str = ""


_DENIED_PATHS = {"/etc", "/root", "/var", "/sys", "/proc", "/etc/passwd",
                 "/etc/shadow"}


class PrivilegeSeparationLayer:
    """L3 — Constrain: per-agent privilege separation."""

    def __init__(self, registry: ToolRegistry,
                 max_allowed_tier: str = "network",
                 config: dict | None = None,
                 enabled: bool = True):
        self.registry = registry
        self.config = config or {}
        self.enabled = enabled
        self._global_max_tier = max_allowed_tier

        # Per-agent state
        self._tokens: dict[str, dict[str, CapabilityToken]] = {}
        self._agent_tiers: dict[str, str] = {}

        denied = self.config.get("denied_paths", list(_DENIED_PATHS))
        self._denied_paths = set(denied)

    def issue_session_tokens(self, agent_id: str = "primary",
                             max_tier: str | None = None,
                             ttl_seconds: int | None = None,
                             max_invocations: int | None = None):
        """Issue tokens for all tools at or below agent's tier ceiling."""
        tier = max_tier or self._global_max_tier
        self._agent_tiers[agent_id] = tier
        ttl = ttl_seconds or self.config.get("token_ttl", 600)
        max_inv = max_invocations or self.config.get("token_max_invocations", 50)

        cutoff = tier_index(tier)
        self._tokens.setdefault(agent_id, {})

        for tool in self.registry.list_tools():
            if tier_index(tool.risk_tier) <= cutoff:
                token = self.registry.issue_token(
                    tool.name, agent_id, ttl, max_inv,
                )
                self._tokens[agent_id][tool.name] = token

    def set_agent_tier_ceiling(self, agent_id: str, max_tier: str):
        """Dynamically adjust an agent's tier ceiling."""
        self._agent_tiers[agent_id] = max_tier

    def check(self, tool_name: str, arguments: dict,
              agent_id: str = "primary") -> L3Decision:
        """Full privilege check for a tool call."""
        if not self.enabled:
            return L3Decision(allowed=True)

        # 1. Tool exists
        tool = self.registry.get(tool_name)
        if tool is None:
            return L3Decision(allowed=False,
                              reason=f"Unknown tool: {tool_name}")

        # 2. Signature integrity
        if not verify_signature(tool):
            return L3Decision(allowed=False,
                              reason=f"Tool signature mismatch: {tool_name} "
                                     f"(possible tampering)")

        # 3. Tier ceiling check
        agent_tier = self._agent_tiers.get(agent_id, self._global_max_tier)
        if tier_index(tool.risk_tier) > tier_index(agent_tier):
            return L3Decision(
                allowed=False,
                reason=f"Tool '{tool_name}' (tier={tool.risk_tier}) "
                       f"exceeds agent '{agent_id}' ceiling ({agent_tier})",
            )

        # 4. Capability token check
        agent_tokens = self._tokens.get(agent_id, {})
        token = agent_tokens.get(tool_name)
        if token is None:
            return L3Decision(allowed=False,
                              reason=f"No token for {agent_id}/{tool_name}")

        if not verify_token_signature(token):
            return L3Decision(allowed=False,
                              reason=f"Token signature invalid: {tool_name}")

        if not token.is_valid():
            return L3Decision(allowed=False,
                              reason=f"Token expired or exhausted: {tool_name}")

        # 5. Path-based constraints
        for key in ("path", "dest", "device"):
            val = arguments.get(key, "")
            if val:
                # Path traversal
                if ".." in val:
                    return L3Decision(allowed=False,
                                      reason=f"Path traversal detected: {val}")
                # Denied paths
                import os
                norm = os.path.normpath(val)
                if norm in self._denied_paths:
                    return L3Decision(allowed=False,
                                      reason=f"Denied path: {val}")

        # Consume token
        token.consume()
        return L3Decision(allowed=True)
