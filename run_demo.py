"""
SafeHarness 2.0 — Quick Demo.

Runs a small number of tasks through all 4 MaS types with SafeHarness2
pipeline, demonstrating the full end-to-end flow.

Usage:
  python run_demo.py
  python run_demo.py --n 1 --harness debate
  python run_demo.py --n 2 --attack inter_agent_injection
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.llm_client import DEFAULT_MODEL
from src.data_loader import load_subset
from src.tools.registry import build_default_registry
from src.tools.simulated_tools import SimulatedEnvironment
from src.harness.hierarchical import HierarchicalHarness
from src.harness.debate import DebateHarness
from src.harness.pipeline import PipelineHarness
from src.harness.decentralized import DecentralizedHarness
from src.safety import SecurityPipeline2, NoDefensePipeline
from src.evaluation import rule_checker, llm_judge, metrics
from src.attacks import (
    inter_agent_injection, agent_impersonation, privilege_escalation,
    cascading_failure, information_leakage, composite_attack,
)


HARNESS_MAP = {
    "hierarchical": HierarchicalHarness,
    "debate": DebateHarness,
    "pipeline": PipelineHarness,
    "decentralized": DecentralizedHarness,
}

ATTACK_MAP = {
    "inter_agent_injection": inter_agent_injection.inject,
    "agent_impersonation": agent_impersonation.inject,
    "privilege_escalation": privilege_escalation.inject,
    "cascading_failure": cascading_failure.inject,
    "information_leakage": information_leakage.inject,
    "composite": composite_attack.inject,
}

HARNESS_CONFIG = {
    "hierarchical_num_workers": 2,
    "debate_max_rounds": 2,
    "pipeline_max_revisions": 1,
    "decentralized_num_peers": 3,
    "decentralized_discussion_rounds": 1,
}

SAFETY_CONFIG = {
    "layer1": {"always_semantic": False},
    "layer2": {"tau_low": 0.3, "tau_high": 0.7},
    "layer3": {"token_ttl": 600, "token_max_invocations": 50},
    "layer4": {"checkpoint_interval": 3, "max_degradation_level": 4, "recovery_window": 5},
    "layer5": {"trust_decay_rate": 0.1, "trust_recovery_rate": 0.02,
               "trust_quarantine_threshold": 0.3, "rate_limit_per_minute": 30},
    "entropy": {"window_size": 20, "threshold": 0.3},
}


def run_single(task: dict, harness_type: str, model_name: str,
               with_defense: bool = True,
               attack_name: str | None = None,
               judge_model: str | None = None):
    """Run a single task and return summary dict."""
    registry = build_default_registry()
    jm = judge_model or model_name

    # Apply attack
    original_task = task
    if attack_name and attack_name in ATTACK_MAP:
        task = ATTACK_MAP[attack_name](task, seed=task.get("id", 0))

    # Build harness & pipeline
    harness_cls = HARNESS_MAP[harness_type]
    harness = harness_cls(model_name, registry, HARNESS_CONFIG)

    if with_defense:
        pipeline = SecurityPipeline2(registry, SAFETY_CONFIG, jm)
    else:
        pipeline = NoDefensePipeline()

    env = SimulatedEnvironment()
    t0 = time.time()
    result = harness.execute(task, env, pipeline)
    elapsed = time.time() - t0

    # Evaluate
    rr = rule_checker.check(result)
    jr = llm_judge.evaluate(result, original_task, judge_model=jm)

    label_str = {-1: "FAILED", 0: "SAFE", 1: "PARTIAL_UNSAFE", 2: "FULLY_UNSAFE"}.get(
        jr.label, "UNKNOWN"
    )

    return {
        "task_id": task.get("id", "?"),
        "instruction": task.get("instruction", "")[:100],
        "harness": harness_type,
        "defense": "SafeHarness2" if with_defense else "NoDefense",
        "attack": attack_name or "clean",
        "judge_label": label_str,
        "rule_unsafe": rr.is_unsafe,
        "triggered_rules": rr.triggered_rules,
        "blocked_actions": len(result.blocked_actions),
        "blocked_messages": len(result.blocked_messages),
        "agent_messages": len(result.agent_messages),
        "tool_calls": len(result.tool_calls),
        "elapsed_s": round(elapsed, 1),
        "response_preview": (result.final_response or "")[:200],
    }


def main():
    parser = argparse.ArgumentParser(description="SafeHarness 2.0 Demo")
    parser.add_argument("--n", type=int, default=2, help="Number of tasks")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--harness", type=str, default=None,
                        help="Specific harness (default: all)")
    parser.add_argument("--attack", type=str, default=None,
                        help="Specific attack to test")
    parser.add_argument("--no-defense", action="store_true",
                        help="Run without SafeHarness2 defense")
    args = parser.parse_args()

    print("=" * 70)
    print("SafeHarness 2.0 — Demo")
    print("=" * 70)

    tasks = load_subset(args.n, sequential=True)
    print(f"Loaded {len(tasks)} tasks")

    harness_types = [args.harness] if args.harness else list(HARNESS_MAP.keys())
    all_results = []

    for task in tasks:
        tid = task.get("id", "?")
        instruction = task.get("instruction", "")[:80]
        print(f"\n{'─' * 70}")
        print(f"Task {tid}: {instruction}...")
        print(f"Risks: {task.get('risks', [])}")

        for ht in harness_types:
            # Clean run with defense
            print(f"\n  [{ht.upper()}] SafeHarness2 | clean")
            try:
                r = run_single(
                    task, ht, args.model,
                    with_defense=not args.no_defense,
                )
                all_results.append(r)
                print(f"    Label: {r['judge_label']} | "
                      f"Blocked: {r['blocked_actions']} actions, "
                      f"{r['blocked_messages']} msgs | "
                      f"{r['elapsed_s']}s")
            except Exception as e:
                print(f"    ERROR: {e}")

            # Attack run (if specified)
            if args.attack:
                print(f"\n  [{ht.upper()}] SafeHarness2 | {args.attack}")
                try:
                    r = run_single(
                        task, ht, args.model,
                        with_defense=not args.no_defense,
                        attack_name=args.attack,
                    )
                    all_results.append(r)
                    print(f"    Label: {r['judge_label']} | "
                          f"Blocked: {r['blocked_actions']} actions, "
                          f"{r['blocked_messages']} msgs | "
                          f"{r['elapsed_s']}s")
                except Exception as e:
                    print(f"    ERROR: {e}")

    # Summary
    print(f"\n{'=' * 70}")
    print("DEMO SUMMARY")
    print(f"{'=' * 70}")
    safe = sum(1 for r in all_results if r["judge_label"] == "SAFE")
    unsafe = sum(1 for r in all_results if "UNSAFE" in r["judge_label"])
    failed = sum(1 for r in all_results if r["judge_label"] == "FAILED")
    print(f"  Total runs:  {len(all_results)}")
    print(f"  SAFE:        {safe}")
    print(f"  UNSAFE:      {unsafe}")
    print(f"  FAILED:      {failed}")

    # Save
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "demo_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
