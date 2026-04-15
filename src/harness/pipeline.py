"""
Sequential Pipeline MaS harness: Analyst -> Executor -> Reviewer.
"""

from __future__ import annotations

from typing import Any

from src.harness.base import BaseHarness, HarnessResult
from src.tools.registry import ToolRegistry
from src.tools.simulated_tools import SimulatedEnvironment, ToolCall
from src.agents.base import (
    AgentIdentity, AgentRole, MessageBus, _ROLE_DEFAULT_TRUST, _ROLE_DEFAULT_TIER,
)
from src.agents.pipeline import AnalystAgent, ExecutorAgent, ReviewerAgent


class PipelineHarness(BaseHarness):
    harness_type = "pipeline"

    def execute(self, task: dict, env: SimulatedEnvironment,
                security_pipeline=None) -> HarnessResult:
        max_revisions = self.config.get("pipeline_max_revisions", 2)

        bus = MessageBus(security_pipeline=security_pipeline)

        # Create agents
        analyst_id = AgentIdentity(
            agent_id="analyst",
            role=AgentRole.ANALYST,
            trust_level=_ROLE_DEFAULT_TRUST[AgentRole.ANALYST],
            max_tool_tier=_ROLE_DEFAULT_TIER[AgentRole.ANALYST],
        )
        analyst = AnalystAgent(
            identity=analyst_id, model_name=self.model_name,
            registry=self.registry, bus=bus, config=self.config,
        )
        bus.register_agent(analyst_id)
        if security_pipeline:
            security_pipeline.register_agent(analyst_id)

        executor_id = AgentIdentity(
            agent_id="executor",
            role=AgentRole.EXECUTOR,
            trust_level=_ROLE_DEFAULT_TRUST[AgentRole.EXECUTOR],
            max_tool_tier=_ROLE_DEFAULT_TIER[AgentRole.EXECUTOR],
        )
        executor = ExecutorAgent(
            identity=executor_id, model_name=self.model_name,
            registry=self.registry, bus=bus, config=self.config,
        )
        bus.register_agent(executor_id)
        if security_pipeline:
            security_pipeline.register_agent(executor_id)

        reviewer_id = AgentIdentity(
            agent_id="reviewer",
            role=AgentRole.REVIEWER,
            trust_level=_ROLE_DEFAULT_TRUST[AgentRole.REVIEWER],
            max_tool_tier=_ROLE_DEFAULT_TIER[AgentRole.REVIEWER],
        )
        reviewer = ReviewerAgent(
            identity=reviewer_id, model_name=self.model_name,
            registry=self.registry, bus=bus, config=self.config,
        )
        bus.register_agent(reviewer_id)
        if security_pipeline:
            security_pipeline.register_agent(reviewer_id)

        all_tool_calls: list[ToolCall] = []
        per_agent_tc: dict[str, list[dict]] = {
            "analyst": [], "executor": [], "reviewer": [],
        }
        reasoning: list[dict] = []

        # Step 1: Analyst
        plan = analyst.analyze(task)

        # Inject inter-agent payload if attack present
        if task.get("_inter_agent_payload"):
            plan += "\n" + task["_inter_agent_payload"]
            if security_pipeline:
                plan = security_pipeline.filter_context(
                    plan, source="inter_agent",
                    source_agent="analyst",
                    provenance_chain=["analyst"],
                )

        reasoning.append({
            "step": "analysis",
            "agent": "analyst",
            "content": plan[:500],
        })
        analyst.send_message("executor", plan, msg_type="result")

        # Step 2-3: Executor + Reviewer loop
        final_response = ""
        for rev in range(max_revisions + 1):
            exec_result, tcs = executor.execute_plan(
                plan, task, env, security_pipeline,
            )
            all_tool_calls.extend(tcs)
            per_agent_tc["executor"].extend([tc.to_dict() for tc in tcs])
            final_response = exec_result

            reasoning.append({
                "step": f"execution_{rev}",
                "agent": "executor",
                "content": exec_result[:500],
            })
            executor.send_message("reviewer", exec_result, msg_type="result")

            # Reviewer
            review_text, approved = reviewer.review(exec_result, task)
            reasoning.append({
                "step": f"review_{rev}",
                "agent": "reviewer",
                "content": review_text[:500],
                "approved": approved,
            })
            reviewer.send_message("executor", review_text, msg_type="feedback")

            if approved:
                break

            # Update plan with feedback for next iteration
            plan = f"Previous feedback: {review_text}\nRevise execution."

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
            metadata={"revisions": len(reasoning) // 2},
        )
