# SafeHarness 2.0 — Multi-Agent Safety Harness

A 5-layer security pipeline for multi-agent LLM systems, extending SafeHarness v1 from single-agent to multi-agent architectures.

## Architecture

### 5-Layer Security Pipeline

| Layer | Name | Function |
|-------|------|----------|
| L1 | INFORM | Adversarial context filtering with inter-agent provenance tracking |
| L2 | VERIFY | Three-tier causal safety verification (rule → LLM → causal diagnostics) |
| L3 | CONSTRAIN | Per-agent capability tokens, cross-agent access control, HMAC signatures |
| L4 | CORRECT | Per-agent checkpointing, rollback, cascading failure detection, quarantine |
| L5 | COORDINATE | Agent authentication, trust propagation, communication graph monitoring |

### 4 Multi-Agent System Architectures

- **Hierarchical**: Manager decomposes tasks, delegates to Workers, synthesizes results
- **Debate**: Proposer generates, Critic evaluates, iterates until approved
- **Pipeline**: Analyst → Executor → Reviewer, strict sequential chain
- **Decentralized**: Peer agents discuss and vote on decisions

### 6 Multi-Agent Attack Types

| Attack | Target | Description |
|--------|--------|-------------|
| A1: Inter-Agent Injection | MessageBus content | Malicious payload in inter-agent messages |
| A2: Agent Impersonation | L5 authentication | Forged sender identity in messages |
| A3: Privilege Escalation | L3 tier ceilings | Worker requests above-tier tools via delegation |
| A4: Cascading Failure | L4 containment | Compromise propagation across agents |
| A5: Information Leakage | Cross-agent data | Data exfiltration via inter-agent messages |
| A6: Composite | Multiple layers | Combined multi-vector attack |

### 4 Baselines (+ SafeHarness2)

| Baseline | Layers | Per-Agent | Cross-Agent |
|----------|--------|-----------|-------------|
| NoDefense | None | No | No |
| SystemPromptDefense | Prompt only | No | No |
| SingleGuard | L1+L2 (shared) | No | No |
| IndependentGuard | L1+L2 (per-agent) | Yes | No |
| **SafeHarness2** | **L1+L2+L3+L4+L5** | **Yes** | **Yes** |

### Metrics

Standard: UBR (Unsafe Behaviour Rate), ASR (Attack Success Rate), TCR (Task Completion Rate), UA (Utility Under Attack)

Multi-agent: CDR (Cascade Detection Rate), IASS (Inter-Agent Safety Score), CS (Containment Score)

## Project Structure

```
SafeHarness2.0/
├── llm_api_openai.py              # LLM API access (pre-configured)
├── config/experiment.yaml         # Experiment configuration
├── src/
│   ├── llm_client.py              # Unified LLM call wrapper
│   ├── data_loader.py             # Agent-SafetyBench download/load
│   ├── tools/                     # Tool registry + simulated execution
│   ├── agents/                    # Agent types (hierarchical, debate, pipeline, decentralized)
│   ├── harness/                   # MaS harness orchestrators
│   ├── safety/                    # 5-layer security pipeline + baselines
│   ├── attacks/                   # 6 attack injection modules
│   └── evaluation/                # Rule checker + LLM judge + metrics
├── run_experiment.py              # Full experiment runner
├── run_demo.py                    # Quick demo
└── requirements.txt
```

## Benchmark

Uses [Agent-SafetyBench](https://github.com/thu-coai/Agent-SafetyBench) (2,000 tasks, 349 environments, 8 risk categories) from Tsinghua University. Auto-downloaded via HuggingFace or GitHub.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quick demo (2 tasks, all 4 MaS types, SafeHarness2 defense)
python run_demo.py --n 2

# Demo with specific harness and attack
python run_demo.py --n 1 --harness debate --attack inter_agent_injection

# Full experiment (5 tasks, all configurations)
python run_experiment.py --subset 5

# Experiment with specific settings
python run_experiment.py --subset 10 --harness hierarchical --security safe_harness_2

# Ablation study
python run_experiment.py --subset 5 --ablation
```

## Data Flow

```
Task (Agent-SafetyBench)
  → [Attack injection] (optional: modify task with attack payload)
  → MaS Harness (create agents + MessageBus)
      → Agent sends message → MessageBus.send()
          → L5: authenticate sender (HMAC verify)
          → L1: filter content (detect injection)
      → Agent proposes tool call → SecurityPipeline.check_action()
          → L4: checkpoint agent state
          → L3: per-agent token + tier ceiling check
          → L2: tiered safety verification
          → L4: degradation/recovery
      → Agent executes tool (simulated)
      → Repeat until task complete
  → Evaluation
      → Rule checker (deterministic patterns)
      → LLM judge (multi-agent-aware safety evaluation)
  → Metrics (UBR, ASR, TCR, CDR, IASS, CS)
```

## Configuration

Edit `config/experiment.yaml` to customize:

- **models**: Which LLM to use (default: `doubao_seed2`)
- **harness.types**: Which MaS architectures to test
- **security_modes**: Which defense baselines to compare
- **attacks.types**: Which attacks to inject
- **safety.layer1-5**: Per-layer parameters (thresholds, token TTL, trust decay, etc.)
- **ablation**: Layer-disable configurations for ablation study

## Key Design Decisions

1. **MessageBus as Security Interception Point**: All inter-agent messages route through L5 authentication + L1 filtering.
2. **Per-Agent State**: L3 tokens, L4 checkpoints/degradation, and L5 trust scores are all per-agent.
3. **Simulated Execution**: All tool calls return mock results for safe, reproducible experiments.
4. **Attack via Task Dict**: Following SafeHarness v1 pattern — attacks set fields on task dict, harnesses inject at appropriate points.
