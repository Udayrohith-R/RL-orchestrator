# Orchestrator-in-the-Loop for Distributed RLHF

> Manages the Precompute-Train-Evaluate cycle of RLHF across heterogeneous compute, ensuring GPUs never sit idle waiting for reward model scores.

## The Problem

In distributed RLHF training, the Policy (Actor), Reward Model, and Reference Model run on different GPU clusters with different throughputs. The Reward Model is typically 2-4x slower than the Policy, causing the training step to starve while waiting for scores. This wastes thousands of dollars in idle GPU time per training run.

## The Solution

```
  ┌─────────────────────────────────────────────────────────────┐
  │                    RL ORCHESTRATOR                           │
  │                                                             │
  │  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐  │
  │  │  Policy   │───►│  Generation  │───►│  Scoring Queue   │  │
  │  │  Actor    │    │  Buffer      │    │  (Reward + Ref)  │  │
  │  │  (Fast)   │    │              │    │  (Slower)        │  │
  │  └──────────┘    └──────────────┘    └────────┬─────────┘  │
  │                                                │            │
  │                   ┌────────────────────────────┘            │
  │                   ▼                                         │
  │  ┌──────────────────────────────┐    ┌──────────────────┐  │
  │  │    Dynamic Batch Manager     │───►│  Training Step   │  │
  │  │    (Adaptive sizing)         │    │  (PPO Update)    │  │
  │  └──────────────────────────────┘    └──────────────────┘  │
  │                                                             │
  │  ┌──────────────────────────────────────────────────────┐  │
  │  │              Component Auto-Scaler                    │  │
  │  │  Monitors throughput → Scales bottleneck workers      │  │
  │  └──────────────────────────────────────────────────────┘  │
  └─────────────────────────────────────────────────────────────┘
```

## Key Features

| Feature | Description |
|---|---|
| **Dynamic Batching** | Adjusts batch size based on buffer pressure to prevent GPU idle |
| **Async Prefetching** | Generates samples ahead of training loop |
| **Parallel Scoring** | Reward + Reference computed concurrently |
| **Auto-Scaling** | Monitors utilization, scales bottleneck components |
| **Starvation Prevention** | Partial batches when buffer draining |

## Usage

```python
from rl_orchestrator import RLOrchestrator

orchestrator = RLOrchestrator(
    batch_size=32,
    num_generation_workers=4,
    num_reward_workers=2,
    num_reference_workers=2,
)

orchestrator.run(
    num_prompts=10000,
    num_training_steps=1000,
)
```

## Running the Demo

```bash
python rl_orchestrator.py
```

## Production Roadmap

- [ ] Ray/Kubernetes deployment manifests
- [ ] Real PyTorch model integration
- [ ] GPU memory-aware scheduling
- [ ] Prometheus metrics + Grafana dashboard
- [ ] Multi-node fault tolerance
- [ ] Integration with Shadow Evaluator for stability monitoring

## Author

Uday — ML Infrastructure Engineer | Ex-Google DeepMind (Gemini)
