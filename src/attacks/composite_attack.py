"""
Attack A6: Composite Attack.

Combines inter-agent injection + impersonation + privilege escalation
simultaneously for a multi-vector attack.
"""

from __future__ import annotations

import copy
import random

from src.attacks import inter_agent_injection
from src.attacks import agent_impersonation
from src.attacks import privilege_escalation


def inject(task: dict, seed: int | None = None) -> dict:
    """Apply all three attack vectors simultaneously."""
    task = copy.deepcopy(task)
    rng = random.Random(seed)

    # Apply inter-agent injection
    task = inter_agent_injection.inject(task, seed=rng.randint(0, 99999))

    # Add impersonation config
    imp_task = agent_impersonation.inject(task, seed=rng.randint(0, 99999))
    task["_impersonate_config"] = imp_task.get("_impersonate_config")

    # Add escalation config
    esc_task = privilege_escalation.inject(task, seed=rng.randint(0, 99999))
    task["_escalation_config"] = esc_task.get("_escalation_config")

    task["_attack"] = "composite"
    return task
