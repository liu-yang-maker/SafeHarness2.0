"""
Layer 5 — COORDINATE: Agent Authentication, Trust Propagation,
Communication Graph Monitoring.

New layer specific to multi-agent systems.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AuthResult:
    authenticated: bool
    reason: str = ""


@dataclass
class TrustState:
    agent_id: str
    trust_score: float = 0.7
    violations: int = 0
    successful_interactions: int = 0


class CoordinationLayer:
    """L5 — Coordinate: multi-agent security coordination.

    1. Agent authentication: verify message HMAC signatures
    2. Trust propagation: per-agent trust scores
    3. Communication graph monitoring: detect anomalous patterns
    4. Message rate limiting: prevent flooding
    """

    def __init__(self, config: dict | None = None, enabled: bool = True):
        self.config = config or {}
        self.enabled = enabled

        self._trust_states: dict[str, TrustState] = {}
        self._agent_secrets: dict[str, str] = {}
        self._comm_graph: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._message_rates: dict[str, list[float]] = defaultdict(list)
        self._expected_topology: dict[str, list[str]] | None = None

        # Config
        self._trust_decay = self.config.get("trust_decay_rate", 0.1)
        self._trust_recovery = self.config.get("trust_recovery_rate", 0.02)
        self._quarantine_threshold = self.config.get("trust_quarantine_threshold", 0.3)
        self._rate_limit = self.config.get("rate_limit_per_minute", 30)

    def register_agent(self, agent_id: str, secret_key: str,
                       initial_trust: float = 0.7):
        """Register an agent's identity and initialize trust state."""
        if not self.enabled:
            return
        self._agent_secrets[agent_id] = secret_key
        self._trust_states[agent_id] = TrustState(
            agent_id=agent_id, trust_score=initial_trust,
        )

    def set_expected_topology(self, topology: dict[str, list[str]]):
        """Set the expected communication topology for anomaly detection."""
        self._expected_topology = topology

    def authenticate_message(self, message) -> AuthResult:
        """Verify message signature against registered sender."""
        if not self.enabled:
            return AuthResult(authenticated=True)

        sender_id = message.sender_id
        content = message.content
        signature = message.signature

        # Check sender is registered
        if sender_id not in self._agent_secrets:
            return AuthResult(
                authenticated=False,
                reason=f"Unknown sender: {sender_id}",
            )

        # Check quarantine
        trust = self._trust_states.get(sender_id)
        if trust and trust.trust_score < self._quarantine_threshold:
            return AuthResult(
                authenticated=False,
                reason=f"Sender {sender_id} quarantined (trust={trust.trust_score:.2f})",
            )

        # Verify HMAC signature
        secret = self._agent_secrets[sender_id]
        expected = hmac.new(
            secret.encode(), content.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(signature, expected):
            self.update_trust(sender_id, "impersonation_attempt")
            return AuthResult(
                authenticated=False,
                reason=f"Signature mismatch for sender {sender_id} "
                       f"(possible impersonation)",
            )

        # Rate limit check
        if not self._check_rate_limit(sender_id):
            self.update_trust(sender_id, "rate_limit_hit")
            return AuthResult(
                authenticated=False,
                reason=f"Rate limit exceeded for {sender_id}",
            )

        # Record communication
        self.record_communication(sender_id, message.receiver_id)
        return AuthResult(authenticated=True)

    def update_trust(self, agent_id: str, event: str):
        """Update trust score based on event."""
        if not self.enabled or agent_id not in self._trust_states:
            return

        state = self._trust_states[agent_id]

        if event == "safe_action":
            state.trust_score = min(1.0, state.trust_score + self._trust_recovery)
            state.successful_interactions += 1
        elif event == "unsafe_action":
            state.trust_score = max(0.0, state.trust_score - self._trust_decay)
            state.violations += 1
        elif event == "blocked_action":
            state.trust_score = max(0.0, state.trust_score - self._trust_decay * 0.5)
            state.violations += 1
        elif event == "impersonation_attempt":
            state.trust_score = max(0.0, state.trust_score - self._trust_decay * 2)
            state.violations += 1
        elif event == "rate_limit_hit":
            state.trust_score = max(0.0, state.trust_score - self._trust_decay * 0.3)

    def get_trust(self, agent_id: str) -> float:
        """Get current trust score for an agent."""
        state = self._trust_states.get(agent_id)
        return state.trust_score if state else 0.0

    def is_quarantined(self, agent_id: str) -> bool:
        trust = self.get_trust(agent_id)
        return trust < self._quarantine_threshold

    def check_communication_graph(self) -> list[dict]:
        """Analyze communication patterns for anomalies."""
        if not self.enabled:
            return []

        anomalies = []

        # Check for unexpected edges
        if self._expected_topology:
            for sender, receivers in self._comm_graph.items():
                expected = set(self._expected_topology.get(sender, []))
                for receiver in receivers:
                    if receiver != "*" and receiver not in expected:
                        anomalies.append({
                            "type": "unexpected_edge",
                            "sender": sender,
                            "receiver": receiver,
                            "detail": f"Communication outside expected topology",
                        })

        # Check for unusual volume
        if self._comm_graph:
            all_counts = [
                count
                for receivers in self._comm_graph.values()
                for count in receivers.values()
            ]
            if all_counts:
                avg = sum(all_counts) / len(all_counts)
                threshold = max(avg * 3, 10)
                for sender, receivers in self._comm_graph.items():
                    for receiver, count in receivers.items():
                        if count > threshold:
                            anomalies.append({
                                "type": "high_volume",
                                "sender": sender,
                                "receiver": receiver,
                                "count": count,
                                "detail": f"Unusually high message volume",
                            })

        return anomalies

    def record_communication(self, sender_id: str, receiver_id: str):
        """Update the communication graph."""
        self._comm_graph[sender_id][receiver_id] += 1

    def _check_rate_limit(self, agent_id: str) -> bool:
        """Returns True if agent is within rate limits."""
        now = time.time()
        window = 60.0  # 1 minute
        timestamps = self._message_rates[agent_id]
        # Remove old timestamps
        self._message_rates[agent_id] = [
            t for t in timestamps if now - t < window
        ]
        if len(self._message_rates[agent_id]) >= self._rate_limit:
            return False
        self._message_rates[agent_id].append(now)
        return True
