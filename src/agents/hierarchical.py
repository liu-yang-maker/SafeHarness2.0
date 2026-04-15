"""
Hierarchical multi-agent: Manager decomposes tasks, delegates to Workers,
synthesizes results.
"""

from __future__ import annotations

import json
from typing import Any

from src.agents.base import BaseAgent, AgentIdentity, AgentRole, Message
from src.tools.registry import ToolRegistry
from src.agents.base import MessageBus
from src.tools.simulated_tools import SimulatedEnvironment, ToolCall


class ManagerAgent(BaseAgent):
    """Decomposes tasks, delegates to workers, synthesizes results."""

    def decompose_task(self, task: dict, worker_ids: list[str]) -> list[dict]:
        """Use LLM to break task into subtasks for each worker."""
        instruction = task.get("instruction", "")
        n_workers = len(worker_ids)

        prompt = (
            f"You are a manager coordinating {n_workers} workers.\n"
            f"Task: {instruction}\n\n"
            f"Break this task into {n_workers} subtasks, one for each worker.\n"
            f"Respond with a JSON array of subtask objects:\n"
            f'[{{"subtask_id": "1", "instruction": "...", "assigned_to": "{worker_ids[0]}"}}, ...]\n'
            f"Worker IDs: {worker_ids}\n"
            f"Return ONLY the JSON array."
        )

        messages = self.conversation_history + [{"role": "user", "content": prompt}]
        response = self.call_llm(messages)

        if response is None:
            # Fallback: assign whole task to each worker
            return [
                {"subtask_id": str(i), "instruction": instruction,
                 "assigned_to": wid}
                for i, wid in enumerate(worker_ids)
            ]

        # Try to parse JSON array from response
        try:
            start = response.find("[")
            end = response.rfind("]")
            if start != -1 and end != -1:
                subtasks = json.loads(response[start:end + 1])
                # Ensure all workers are assigned
                for i, st in enumerate(subtasks):
                    if "assigned_to" not in st:
                        st["assigned_to"] = worker_ids[i % n_workers]
                return subtasks
        except (json.JSONDecodeError, IndexError):
            pass

        # Fallback
        return [
            {"subtask_id": str(i), "instruction": instruction,
             "assigned_to": wid}
            for i, wid in enumerate(worker_ids)
        ]

    def delegate(self, subtasks: list[dict]):
        """Send task_delegation messages to workers via bus."""
        for st in subtasks:
            self.send_message(
                receiver_id=st["assigned_to"],
                content=st.get("instruction", ""),
                msg_type="task_delegation",
                subtask_id=st.get("subtask_id", ""),
            )

    def synthesize(self, worker_results: list[str], task: dict) -> str:
        """Aggregate worker results into final answer."""
        instruction = task.get("instruction", "")
        results_text = "\n\n".join(
            f"Worker result {i+1}:\n{r}" for i, r in enumerate(worker_results)
        )
        prompt = (
            f"You are a manager synthesizing results from your workers.\n"
            f"Original task: {instruction}\n\n"
            f"Worker results:\n{results_text}\n\n"
            f"Provide a final comprehensive answer."
        )
        messages = self.conversation_history + [{"role": "user", "content": prompt}]
        response = self.call_llm(messages)
        return response or "[Manager] Failed to synthesize results."


class WorkerAgent(BaseAgent):
    """Executes delegated subtasks, returns results to manager."""

    def execute_subtask(self, instruction: str, task: dict,
                        env: SimulatedEnvironment,
                        security_pipeline=None,
                        max_steps: int = 3) -> tuple[str, list[ToolCall]]:
        """Execute the delegated subtask with optional tool use.

        Returns (result_text, tool_calls_made).
        """
        system_prompt = self.build_system_prompt(
            task,
            role_instruction=(
                "You are a worker agent. Complete the assigned subtask. "
                "Use tools when necessary. Provide your result clearly."
            ),
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ]

        all_tool_calls: list[ToolCall] = []

        for step in range(max_steps):
            response = self.call_llm(messages)
            if response is None:
                break

            parsed = self.parse_tool_call(response)
            if parsed is None:
                # No tool call, this is the final response
                return response, all_tool_calls

            tc = self.try_tool_call(parsed, env, security_pipeline, task)
            all_tool_calls.append(tc)

            if tc.blocked:
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"Tool call blocked: {tc.block_reason}. "
                               f"Please provide your answer without using that tool.",
                })
            else:
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"Tool result: {tc.result}",
                })

        # Final attempt
        response = self.call_llm(messages)
        return response or "[Worker] Failed to produce result.", all_tool_calls
