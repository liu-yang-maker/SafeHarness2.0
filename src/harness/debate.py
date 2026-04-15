"""
Debate MaS harness: Proposer generates, Critic evaluates, iterate.
"""

from __future__ import annotations

from typing import Any

from src.harness.base import BaseHarness, HarnessResult
from src.tools.registry import ToolRegistry
from src.tools.simulated_tools import SimulatedEnvironment, ToolCall
from src.agents.base import (
    AgentIdentity, AgentRole, MessageBus, _ROLE_DEFAULT_TRUST, _ROLE_DEFAULT_TIER,
)
from src.agents.debate import ProposerAgent, CriticAgent


class DebateHarness(BaseHarness):
    harness_type = "debate"

    def execute(self, task: dict, env: SimulatedEnvironment,
                security_pipeline=None) -> HarnessResult:
        max_rounds = self.config.get("debate_max_rounds", 3)

        bus = MessageBus(security_pipeline=security_pipeline)

        # Create proposer
        proposer_id = AgentIdentity(
            agent_id="proposer",
            role=AgentRole.PROPOSER,
            trust_level=_ROLE_DEFAULT_TRUST[AgentRole.PROPOSER],
            max_tool_tier=_ROLE_DEFAULT_TIER[AgentRole.PROPOSER],
        )
        proposer = ProposerAgent(
            identity=proposer_id, model_name=self.model_name,
            registry=self.registry, bus=bus, config=self.config,
        )
        bus.register_agent(proposer_id)
        if security_pipeline:
            security_pipeline.register_agent(proposer_id)

        # Create critic
        critic_id = AgentIdentity(
            agent_id="critic",
            role=AgentRole.CRITIC,
            trust_level=_ROLE_DEFAULT_TRUST[AgentRole.CRITIC],
            max_tool_tier=_ROLE_DEFAULT_TIER[AgentRole.CRITIC],
        )
        critic = CriticAgent(
            identity=critic_id, model_name=self.model_name,
            registry=self.registry, bus=bus, config=self.config,
        )
        bus.register_agent(critic_id)
        if security_pipeline:
            security_pipeline.register_agent(critic_id)

        all_tool_calls: list[ToolCall] = []
        per_agent_tc: dict[str, list[dict]] = {"proposer": [], "critic": []}
        reasoning: list[dict] = []
        final_response = ""
        feedback = None

        for round_idx in range(max_rounds):
            # Inject inter-agent payload into feedback if attack present
            if task.get("_inter_agent_payload") and feedback:
                feedback += "\n" + task["_inter_agent_payload"]
                if security_pipeline:
                    feedback = security_pipeline.filter_context(
                        feedback, source="inter_agent",
                        source_agent="critic",
                        provenance_chain=["critic"],
                    )

            # Proposer generates/revises
            proposal, tcs = proposer.propose(
                task, env, security_pipeline, feedback=feedback,
            )
            all_tool_calls.extend(tcs)
            per_agent_tc["proposer"].extend([tc.to_dict() for tc in tcs])
            final_response = proposal

            reasoning.append({
                "step": f"round_{round_idx}_proposal",
                "agent": "proposer",
                "content": proposal[:500],
            })

            # Proposer sends proposal to critic
            proposer.send_message("critic", proposal, msg_type="discussion")

            # Critic evaluates
            critique_text, approved = critic.critique(proposal, task)
            reasoning.append({
                "step": f"round_{round_idx}_critique",
                "agent": "critic",
                "content": critique_text[:500],
                "approved": approved,
            })

            # Critic sends feedback
            critic.send_message("proposer", critique_text, msg_type="critique")

            if approved:
                break

            feedback = critique_text

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
            metadata={"rounds": len(reasoning) // 2},
        )
