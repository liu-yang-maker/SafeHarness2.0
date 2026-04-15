"""
Hierarchical MaS harness: Manager decomposes, Workers execute, Manager synthesizes.
"""

from __future__ import annotations

from typing import Any

from src.harness.base import BaseHarness, HarnessResult
from src.tools.registry import ToolRegistry
from src.tools.simulated_tools import SimulatedEnvironment, ToolCall
from src.agents.base import (
    AgentIdentity, AgentRole, MessageBus, _ROLE_DEFAULT_TRUST, _ROLE_DEFAULT_TIER,
)
from src.agents.hierarchical import ManagerAgent, WorkerAgent


class HierarchicalHarness(BaseHarness):
    harness_type = "hierarchical"

    def execute(self, task: dict, env: SimulatedEnvironment,
                security_pipeline=None) -> HarnessResult:
        n_workers = self.config.get("hierarchical_num_workers", 2)

        # Create message bus
        bus = MessageBus(security_pipeline=security_pipeline)

        # Create manager
        manager_id = AgentIdentity(
            agent_id="manager",
            role=AgentRole.MANAGER,
            trust_level=_ROLE_DEFAULT_TRUST[AgentRole.MANAGER],
            max_tool_tier=_ROLE_DEFAULT_TIER[AgentRole.MANAGER],
        )
        manager = ManagerAgent(
            identity=manager_id, model_name=self.model_name,
            registry=self.registry, bus=bus, config=self.config,
        )
        bus.register_agent(manager_id)
        if security_pipeline:
            security_pipeline.register_agent(manager_id)

        # Create workers
        workers: list[WorkerAgent] = []
        worker_ids: list[str] = []
        for i in range(n_workers):
            wid = f"worker_{i}"
            worker_ids.append(wid)
            w_identity = AgentIdentity(
                agent_id=wid,
                role=AgentRole.WORKER,
                trust_level=_ROLE_DEFAULT_TRUST[AgentRole.WORKER],
                max_tool_tier=_ROLE_DEFAULT_TIER[AgentRole.WORKER],
            )
            w = WorkerAgent(
                identity=w_identity, model_name=self.model_name,
                registry=self.registry, bus=bus, config=self.config,
            )
            bus.register_agent(w_identity)
            if security_pipeline:
                security_pipeline.register_agent(w_identity)
            workers.append(w)

        # Step 1: Manager decomposes task
        subtasks = manager.decompose_task(task, worker_ids)

        # Step 2: Delegate
        manager.delegate(subtasks)

        # Step 3: Workers execute
        all_tool_calls: list[ToolCall] = []
        per_agent_tc: dict[str, list[dict]] = {}
        worker_results: list[str] = []

        for i, worker in enumerate(workers):
            # Get delegation message
            delegation_msgs = worker.receive_messages(msg_type="task_delegation")
            instruction = task.get("instruction", "")
            if delegation_msgs:
                instruction = delegation_msgs[-1].content

            # Inject inter-agent payload if attack present
            if task.get("_inter_agent_payload") and delegation_msgs:
                instruction += "\n" + task["_inter_agent_payload"]
                # Re-filter through L1
                if security_pipeline:
                    instruction = security_pipeline.filter_context(
                        instruction, source="inter_agent",
                        source_agent="manager",
                        provenance_chain=["manager"],
                    )

            result_text, tcs = worker.execute_subtask(
                instruction, task, env, security_pipeline,
            )
            worker_results.append(result_text)
            all_tool_calls.extend(tcs)
            per_agent_tc[worker.agent_id] = [tc.to_dict() for tc in tcs]

            # Worker sends result back
            worker.send_message("manager", result_text, msg_type="result")

        # Step 4: Manager synthesizes
        final_response = manager.synthesize(worker_results, task)

        # Collect blocked actions and messages
        blocked = [tc for tc in all_tool_calls if tc.blocked]
        blocked_msgs = [m.to_dict() for m in bus.message_log if m.blocked]

        # Check for cascading events
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
            agent_messages=[m.to_dict() for m in bus.message_log],
            per_agent_tool_calls=per_agent_tc,
            blocked_messages=blocked_msgs,
            cascading_events=cascade_events,
            metadata={
                "n_workers": n_workers,
                "n_subtasks": len(subtasks),
            },
        )
