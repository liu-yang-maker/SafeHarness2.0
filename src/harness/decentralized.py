"""
Decentralized MaS harness: Peer agents discuss and vote.
"""

from __future__ import annotations

from typing import Any

from src.harness.base import BaseHarness, HarnessResult
from src.tools.registry import ToolRegistry
from src.tools.simulated_tools import SimulatedEnvironment, ToolCall
from src.agents.base import (
    AgentIdentity, AgentRole, MessageBus, _ROLE_DEFAULT_TRUST, _ROLE_DEFAULT_TIER,
)
from src.agents.decentralized import PeerAgent, tally_votes


class DecentralizedHarness(BaseHarness):
    harness_type = "decentralized"

    def execute(self, task: dict, env: SimulatedEnvironment,
                security_pipeline=None) -> HarnessResult:
        n_peers = self.config.get("decentralized_num_peers", 3)
        n_rounds = self.config.get("decentralized_discussion_rounds", 2)

        bus = MessageBus(security_pipeline=security_pipeline)

        # Create peers
        peers: list[PeerAgent] = []
        for i in range(n_peers):
            pid = f"peer_{i}"
            p_identity = AgentIdentity(
                agent_id=pid,
                role=AgentRole.PEER,
                trust_level=_ROLE_DEFAULT_TRUST[AgentRole.PEER],
                max_tool_tier=_ROLE_DEFAULT_TIER[AgentRole.PEER],
            )
            peer = PeerAgent(
                identity=p_identity, model_name=self.model_name,
                registry=self.registry, bus=bus, config=self.config,
            )
            bus.register_agent(p_identity)
            if security_pipeline:
                security_pipeline.register_agent(p_identity)
            peers.append(peer)

        all_tool_calls: list[ToolCall] = []
        per_agent_tc: dict[str, list[dict]] = {p.agent_id: [] for p in peers}
        reasoning: list[dict] = []

        # Discussion rounds
        for round_idx in range(n_rounds):
            prior_msgs = bus.message_log.copy()

            # Inject inter-agent payload in first round
            if round_idx == 0 and task.get("_inter_agent_payload"):
                for peer in peers:
                    injected_content = task["_inter_agent_payload"]
                    if security_pipeline:
                        injected_content = security_pipeline.filter_context(
                            injected_content, source="inter_agent",
                            source_agent="external",
                            provenance_chain=["external"],
                        )

            for peer in peers:
                disc_text, tcs = peer.discuss(
                    task, prior_msgs, env, security_pipeline,
                )
                all_tool_calls.extend(tcs)
                per_agent_tc[peer.agent_id].extend(
                    [tc.to_dict() for tc in tcs]
                )
                reasoning.append({
                    "step": f"round_{round_idx}_{peer.agent_id}",
                    "agent": peer.agent_id,
                    "content": disc_text[:500],
                })
                # Broadcast discussion
                peer.send_message("*", disc_text, msg_type="discussion")

        # Each peer produces a final proposal
        proposals: list[str] = []
        for peer in peers:
            prior_msgs = bus.message_log.copy()
            proposal, tcs = peer.discuss(task, prior_msgs, env, security_pipeline)
            all_tool_calls.extend(tcs)
            per_agent_tc[peer.agent_id].extend(
                [tc.to_dict() for tc in tcs]
            )
            proposals.append(proposal)

        # Vote
        votes = []
        for peer in peers:
            vote = peer.vote(proposals, task)
            votes.append(vote)
            reasoning.append({
                "step": f"vote_{peer.agent_id}",
                "agent": peer.agent_id,
                "content": str(vote),
            })

        winner_idx = tally_votes(votes, len(proposals))
        final_response = proposals[winner_idx] if proposals else ""

        blocked = [tc for tc in all_tool_calls if tc.blocked]
        blocked_msgs = [m.to_dict() for m in bus.message_log if m.blocked]

        cascade_events = []
        if security_pipeline:
            events = security_pipeline.check_cascade(bus.message_log)
            cascade_events = [
                {"trigger": e.trigger_agent, "affected": e.affected_agents,
                 "step": e.step, "severity": e.severity}
                for e in events
            ]

        return HarnessResult(
            task_id=task.get("id", 0),
            harness_type=self.harness_type,
            final_response=final_response,
            tool_calls=all_tool_calls,
            blocked_actions=blocked,
            reasoning_trace=reasoning,
            agent_messages=[m.to_dict() for m in bus.message_log],
            per_agent_tool_calls=per_agent_tc,
            blocked_messages=blocked_msgs,
            cascading_events=cascade_events,
            metadata={
                "n_peers": n_peers,
                "n_rounds": n_rounds,
                "winner_idx": winner_idx,
                "votes": votes,
            },
        )
