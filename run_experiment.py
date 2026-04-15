"""
SafeHarness 2.0 — Full Experiment Runner.

Runs the complete experiment matrix:
  models x harness_types x security_modes x tasks x attacks

Usage:
  python run_experiment.py --subset 5
  python run_experiment.py --subset 10 --harness debate --security safe_harness_2
  python run_experiment.py --ablation --subset 5
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

import yaml

from src.llm_client import call_model, DEFAULT_MODEL
from src.data_loader import load_subset, download_dataset
from src.tools.registry import build_default_registry, ToolRegistry
from src.tools.simulated_tools import SimulatedEnvironment
from src.harness.base import HarnessResult
from src.harness.hierarchical import HierarchicalHarness
from src.harness.debate import DebateHarness
from src.harness.pipeline import PipelineHarness
from src.harness.decentralized import DecentralizedHarness
from src.safety import (
    SecurityPipeline2, NoDefensePipeline, SystemPromptDefense,
    SingleGuardPipeline, IndependentGuardPipeline,
)
from src.attacks import (
    inter_agent_injection, agent_impersonation, privilege_escalation,
    cascading_failure, information_leakage, composite_attack,
)
from src.evaluation import rule_checker, llm_judge, metrics

_PROJECT_ROOT = Path(__file__).resolve().parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "experiment.yaml"


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


def _build_security(mode: str, registry: ToolRegistry,
                    cfg: dict, judge_model: str,
                    disable_layers: list[str] | None = None):
    safety_cfg = cfg.get("safety", {})
    if disable_layers:
        safety_cfg = dict(safety_cfg)
        safety_cfg["disable_layers"] = disable_layers

    if mode == "no_defense":
        return NoDefensePipeline()
    elif mode == "system_prompt":
        return SystemPromptDefense()
    elif mode == "single_guard":
        return SingleGuardPipeline(registry, safety_cfg, judge_model)
    elif mode == "independent_guard":
        return IndependentGuardPipeline(registry, safety_cfg, judge_model)
    elif mode == "safe_harness_2":
        return SecurityPipeline2(registry, safety_cfg, judge_model)
    else:
        raise ValueError(f"Unknown security mode: {mode}")


@dataclass
class Job:
    task: dict
    model_name: str
    harness_type: str
    security_mode: str
    attack: str | None
    attack_fn: Callable | None
    ablation_name: str | None = None
    disable_layers: list[str] | None = None


@dataclass
class Record:
    model: str
    harness: str
    security_mode: str
    attack: str | None
    task_id: Any
    risks: list[str]
    judge_label: int
    judge_reasoning: str
    rule_unsafe: bool
    rule_details: str
    blocked_count: int
    blocked_messages: int
    cascade_events: int
    elapsed_s: float
    ablation: str | None = None


def _run_job(job: Job, cfg: dict) -> Record:
    """Execute one experiment job."""
    t0 = time.time()
    registry = build_default_registry()
    judge_model = cfg.get("models", {}).get("judge_model", DEFAULT_MODEL)

    # Apply attack
    task = job.task
    if job.attack_fn:
        task = job.attack_fn(task, seed=task.get("id", 0))

    # Build harness
    harness_cls = HARNESS_MAP[job.harness_type]
    harness_cfg = cfg.get("harness", {})
    harness = harness_cls(job.model_name, registry, harness_cfg)

    # Build security pipeline
    pipeline = _build_security(
        job.security_mode, registry, cfg, judge_model, job.disable_layers,
    )

    # Execute
    env = SimulatedEnvironment()
    try:
        result = harness.execute(task, env, pipeline)
        result.security_mode = job.security_mode
    except Exception as e:
        result = HarnessResult(
            task_id=task.get("id", 0),
            harness_type=job.harness_type,
            security_mode=job.security_mode,
            final_response=f"[ERROR] {type(e).__name__}: {e}",
        )

    # Evaluate
    try:
        rr = rule_checker.check(result)
    except Exception:
        rr = rule_checker.RuleCheckResult(is_unsafe=False)

    try:
        jr = llm_judge.evaluate(result, task, judge_model=judge_model)
    except Exception:
        jr = llm_judge.JudgeResult(label=-1, reasoning="Judge error")

    elapsed = time.time() - t0

    n_cascade = (len(result.cascading_events)
                 if isinstance(result.cascading_events, list) else 0)

    return Record(
        model=job.model_name,
        harness=job.harness_type,
        security_mode=job.security_mode,
        attack=job.attack,
        task_id=task.get("id", 0),
        risks=task.get("risks", []),
        judge_label=jr.label,
        judge_reasoning=jr.reasoning[:300],
        rule_unsafe=rr.is_unsafe,
        rule_details="; ".join(rr.triggered_rules),
        blocked_count=len(result.blocked_actions),
        blocked_messages=len(result.blocked_messages),
        cascade_events=n_cascade,
        elapsed_s=round(elapsed, 2),
        ablation=job.ablation_name,
    )


def main():
    parser = argparse.ArgumentParser(description="SafeHarness 2.0 Experiment")
    parser.add_argument("--config", type=str, default=str(_CONFIG_PATH))
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--harness", type=str, default=None,
                        help="Run only this harness type")
    parser.add_argument("--security", type=str, default=None,
                        help="Run only this security mode")
    parser.add_argument("--attack", type=str, default=None,
                        help="Run only this attack type (or 'none' for clean)")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--skip-jobs", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load data
    n = args.subset or cfg.get("data", {}).get("experiment_subset_size", 50)
    dataset_file = cfg.get("data", {}).get("dataset_file")
    sequential = cfg.get("data", {}).get("sequential_subset", True)
    tasks = load_subset(n, dataset_file=dataset_file, sequential=sequential)
    print(f"[experiment] Loaded {len(tasks)} tasks")

    # Build job list
    models = [args.model] if args.model else cfg.get("models", {}).get("agent_models", [DEFAULT_MODEL])
    harness_types = [args.harness] if args.harness else cfg.get("harness", {}).get("types", ["debate"])
    security_modes = [args.security] if args.security else cfg.get("security_modes", ["safe_harness_2"])
    attack_types = cfg.get("attacks", {}).get("types", [])

    if args.attack:
        if args.attack == "none":
            attack_types = []
        else:
            attack_types = [args.attack]

    jobs: list[Job] = []
    for model in models:
        for ht in harness_types:
            for sm in security_modes:
                for task in tasks:
                    # Clean run
                    jobs.append(Job(
                        task=task, model_name=model,
                        harness_type=ht, security_mode=sm,
                        attack=None, attack_fn=None,
                    ))
                    # Attack runs
                    for at in attack_types:
                        if at in ATTACK_MAP:
                            jobs.append(Job(
                                task=task, model_name=model,
                                harness_type=ht, security_mode=sm,
                                attack=at, attack_fn=ATTACK_MAP[at],
                            ))

    # Ablation mode
    if args.ablation:
        ablation_cfgs = cfg.get("ablation", {}).get("configs", [])
        ablation_jobs = []
        for abl in ablation_cfgs:
            name = abl["name"]
            dl = abl.get("disable_layers", [])
            if name == "full":
                continue  # already covered
            for task in tasks:
                for ht in harness_types:
                    ablation_jobs.append(Job(
                        task=task, model_name=models[0],
                        harness_type=ht, security_mode="safe_harness_2",
                        attack=None, attack_fn=None,
                        ablation_name=name, disable_layers=dl,
                    ))
        jobs.extend(ablation_jobs)

    # Skip
    jobs = jobs[args.skip_jobs:]
    print(f"[experiment] {len(jobs)} jobs to run")

    # Output
    output_dir = Path(cfg.get("project", {}).get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "experiment_results.jsonl"
    json_path = output_dir / "experiment_results.json"

    # Execute
    max_workers = args.workers or cfg.get("parallel", {}).get("max_workers", 4)
    records: list[Record] = []

    with open(jsonl_path, "a", encoding="utf-8") as log_f:
        # Sequential execution for stability (LLM calls are the bottleneck)
        for i, job in enumerate(jobs):
            tag = (f"[{i+1}/{len(jobs)}] {job.model_name}/{job.harness_type}/"
                   f"{job.security_mode}/{job.attack or 'clean'}")
            print(f"{tag} ...", flush=True)

            try:
                record = _run_job(job, cfg)
                records.append(record)
                log_f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                log_f.flush()

                label_str = {-1: "FAILED", 0: "SAFE", 1: "PARTIAL", 2: "UNSAFE"}.get(
                    record.judge_label, "?"
                )
                print(f"  -> {label_str} | blocked={record.blocked_count} "
                      f"| {record.elapsed_s:.1f}s")
            except Exception as e:
                print(f"  -> ERROR: {e}")

    # Compute metrics
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Group by security mode
    by_mode: dict[str, list[Record]] = {}
    for r in records:
        key = r.security_mode
        if r.ablation:
            key = f"safe_harness_2 ({r.ablation})"
        by_mode.setdefault(key, []).append(r)

    # For each group, compute metrics by reconstructing tuples
    for mode, mode_records in sorted(by_mode.items()):
        # Build pseudo-tuples for compute_metrics
        tuples = []
        for r in mode_records:
            pseudo_task = {
                "id": r.task_id,
                "risks": r.risks,
                "fulfillable": 1,
                "_attack": r.attack,
            }
            pseudo_hr = type("HR", (), {
                "final_response": "",
                "tool_calls": [],
                "blocked_actions": [None] * r.blocked_count,
                "agent_messages": [None] * 5,
                "blocked_messages": [None] * r.blocked_messages,
                "cascading_events": [None] * r.cascade_events,
            })()
            pseudo_jr = type("JR", (), {"label": r.judge_label})()
            pseudo_rr = type("RR", (), {"is_unsafe": r.rule_unsafe})()
            tuples.append((pseudo_task, pseudo_hr, pseudo_jr, pseudo_rr))

        m = metrics.compute_metrics(tuples)
        summary = m.summary()
        print(f"\n  [{mode}] ({len(mode_records)} jobs)")
        for k, v in summary.items():
            print(f"    {k:>6}: {v}")

    # Save final JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, ensure_ascii=False, indent=2)
    print(f"\n[experiment] Results saved to {jsonl_path} and {json_path}")


if __name__ == "__main__":
    main()
