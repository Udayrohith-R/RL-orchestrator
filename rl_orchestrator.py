# Orchestrator-in-the-Loop for Distributed RLHF
# ================================================
# A framework that manages the Precompute-Train-Evaluate cycle of RLHF
# across heterogeneous compute, ensuring GPUs never sit idle waiting
# for reward model scores.
#
# Architecture:
# 1. Distributed actor system (Policy, Reward, Reference on different nodes)
# 2. Dynamic batching to handle speed mismatches between components
# 3. Auto-scaling of slow components to prevent starvation
# 4. Async sampling buffer for decoupled generation and training
#
# Author: Uday
# Target: Anthropic RL Engineering Team

import os
import time
import json
import threading
import logging
import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple, Any
from collections import deque
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import queue

# NOTE ON GIL CONTENTION:
# This prototype uses ThreadPoolExecutor for simulation simplicity (time.sleep()
# releases the GIL). In production, CPU-bound tokenization, batch collation, and
# data serialization MUST use ProcessPoolExecutor or a distributed actor framework
# (Ray, Kubernetes Jobs) to avoid GIL contention starving GPU training loops.
# See gil-free-data-pipeline repo for the GIL-bypass tokenization approach.

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("rl_orchestrator")


# ============================================================
# PART 1: Core Data Structures
# ============================================================

@dataclass
class RLSample:
    """Single RL training sample with all components."""
    sample_id: str
    prompt_tokens: List[int]
    generated_tokens: List[int]
    policy_logprobs: List[float]
    reference_logprobs: Optional[List[float]] = None
    reward_score: Optional[float] = None
    advantages: Optional[List[float]] = None
    timestamp_generated: float = 0.0
    timestamp_scored: float = 0.0
    timestamp_ready: float = 0.0


@dataclass
class RLBatch:
    """Batch of samples ready for training."""
    batch_id: str
    samples: List[RLSample]
    batch_size: int
    avg_reward: float = 0.0
    avg_kl: float = 0.0
    is_complete: bool = False


class ComponentStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    FAILED = "failed"


@dataclass
class ComponentHealth:
    """Health metrics for a distributed component."""
    name: str
    status: ComponentStatus
    throughput_per_sec: float
    queue_depth: int
    avg_latency_ms: float
    num_workers: int
    utilization: float  # 0.0 to 1.0


# ============================================================
# PART 2: Simulated Distributed Components
# ============================================================

class PolicyActor:
    """
    Simulates the Policy model (Actor) that generates responses.
    In production: runs on GPU nodes, generates tokens autoregressively.
    """
    
    def __init__(self, generation_latency_ms: float = 50.0, num_workers: int = 4):
        self.latency = generation_latency_ms / 1000.0
        self.num_workers = num_workers
        self._total_generated = 0
        self._start_time = time.time()
        self._lock = threading.Lock()
    
    def generate(self, prompt_tokens: List[int], max_new_tokens: int = 256) -> Dict:
        """Generate a response for a prompt."""
        # Simulate generation latency
        time.sleep(self.latency * (0.8 + random.random() * 0.4))
        
        # Simulate generated tokens and logprobs
        gen_tokens = [random.randint(0, 50256) for _ in range(max_new_tokens)]
        logprobs = [random.gauss(-2.0, 0.5) for _ in range(max_new_tokens)]
        
        with self._lock:
            self._total_generated += 1
        
        return {
            "generated_tokens": gen_tokens,
            "policy_logprobs": logprobs,
            "latency_ms": self.latency * 1000,
        }
    
    def get_health(self) -> ComponentHealth:
        elapsed = time.time() - self._start_time
        throughput = self._total_generated / elapsed if elapsed > 0 else 0
        return ComponentHealth(
            name="PolicyActor",
            status=ComponentStatus.BUSY,
            throughput_per_sec=throughput,
            queue_depth=0,
            avg_latency_ms=self.latency * 1000,
            num_workers=self.num_workers,
            utilization=min(1.0, throughput / (self.num_workers / self.latency)),
        )


class RewardModel:
    """
    Simulates the Reward Model that scores generated responses.
    In production: separate GPU cluster, often the bottleneck.
    """
    
    def __init__(self, scoring_latency_ms: float = 30.0, num_workers: int = 2):
        self.latency = scoring_latency_ms / 1000.0
        self.num_workers = num_workers
        self._total_scored = 0
        self._start_time = time.time()
        self._lock = threading.Lock()
    
    def score(self, prompt_tokens: List[int], generated_tokens: List[int]) -> Dict:
        """Score a generated response."""
        time.sleep(self.latency * (0.8 + random.random() * 0.4))
        
        # Simulate reward score
        reward = random.gauss(0.5, 0.3)
        
        with self._lock:
            self._total_scored += 1
        
        return {
            "reward_score": reward,
            "latency_ms": self.latency * 1000,
        }
    
    def get_health(self) -> ComponentHealth:
        elapsed = time.time() - self._start_time
        throughput = self._total_scored / elapsed if elapsed > 0 else 0
        return ComponentHealth(
            name="RewardModel",
            status=ComponentStatus.BUSY,
            throughput_per_sec=throughput,
            queue_depth=0,  # In production: track via distributed task queue (Ray/Celery)
            avg_latency_ms=self.latency * 1000,
            num_workers=self.num_workers,
            utilization=min(1.0, throughput / (self.num_workers / self.latency)),
        )


class ReferenceModel:
    """
    Simulates the frozen Reference Model for KL penalty computation.
    In production: same architecture as policy, frozen weights.
    """
    
    def __init__(self, inference_latency_ms: float = 20.0, num_workers: int = 2):
        self.latency = inference_latency_ms / 1000.0
        self.num_workers = num_workers
        self._total_computed = 0
        self._start_time = time.time()
        self._lock = threading.Lock()
    
    def compute_logprobs(self, prompt_tokens: List[int], 
                         generated_tokens: List[int]) -> Dict:
        """Compute log probabilities under the reference policy."""
        time.sleep(self.latency * (0.8 + random.random() * 0.4))
        
        ref_logprobs = [random.gauss(-2.0, 0.3) for _ in range(len(generated_tokens))]
        
        with self._lock:
            self._total_computed += 1
        
        return {
            "reference_logprobs": ref_logprobs,
            "latency_ms": self.latency * 1000,
        }
    
    def get_health(self) -> ComponentHealth:
        elapsed = time.time() - self._start_time
        throughput = self._total_computed / elapsed if elapsed > 0 else 0
        return ComponentHealth(
            name="ReferenceModel",
            status=ComponentStatus.BUSY,
            throughput_per_sec=throughput,
            queue_depth=0,
            avg_latency_ms=self.latency * 1000,
            num_workers=self.num_workers,
            utilization=min(1.0, throughput / (self.num_workers / self.latency)),
        )


# ============================================================
# PART 3: Dynamic Batch Manager
# ============================================================

class DynamicBatchManager:
    """
    Manages dynamic batching to handle speed mismatches between
    the Policy, Reward, and Reference models.
    
    Key insight: The Reward Model is typically the bottleneck.
    This manager adjusts the sampling buffer to prevent the
    training step from starving while waiting for reward scores.
    """
    
    def __init__(
        self,
        target_batch_size: int = 32,
        min_batch_size: int = 8,
        max_buffer_size: int = 256,
        starvation_timeout_ms: float = 5000.0,
    ):
        self.target_batch_size = target_batch_size
        self.min_batch_size = min_batch_size
        self.max_buffer_size = max_buffer_size
        self.starvation_timeout = starvation_timeout_ms / 1000.0
        
        # Buffers
        self._generation_buffer: deque = deque(maxlen=max_buffer_size)
        self._scoring_buffer: deque = deque(maxlen=max_buffer_size)
        self._ready_buffer: deque = deque(maxlen=max_buffer_size)
        
        self._lock = threading.Lock()
        
        # Metrics
        self._starvation_count = 0
        self._batches_formed = 0
        self._dynamic_adjustments = 0
    
    def add_generated(self, sample: RLSample):
        """Add a newly generated sample to the scoring queue."""
        with self._lock:
            self._generation_buffer.append(sample)
    
    def add_scored(self, sample: RLSample):
        """Add a scored sample to the ready buffer."""
        with self._lock:
            self._ready_buffer.append(sample)
    
    def try_form_batch(self) -> Optional[RLBatch]:
        """
        Try to form a training batch from ready samples.
        Uses dynamic sizing based on buffer pressure.
        """
        with self._lock:
            available = len(self._ready_buffer)
            
            if available == 0:
                return None
            
            # Dynamic batch sizing
            # If buffer is accumulating (reward model faster), use target size
            # If buffer is draining (reward model slower), use smaller batches
            if available >= self.target_batch_size:
                batch_size = self.target_batch_size
            elif available >= self.min_batch_size:
                # Partial batch to prevent GPU idle time
                batch_size = available
                self._dynamic_adjustments += 1
            else:
                return None  # Wait for more samples
            
            samples = [self._ready_buffer.popleft() for _ in range(batch_size)]
        
        self._batches_formed += 1
        
        avg_reward = sum(s.reward_score for s in samples if s.reward_score) / len(samples)
        
        return RLBatch(
            batch_id=f"batch_{self._batches_formed}",
            samples=samples,
            batch_size=len(samples),
            avg_reward=avg_reward,
            is_complete=True,
        )
    
    def get_pending_generation(self, n: int = 1) -> List[RLSample]:
        """Get samples that need reward scoring."""
        with self._lock:
            result = []
            for _ in range(min(n, len(self._generation_buffer))):
                result.append(self._generation_buffer.popleft())
            return result
    
    def get_metrics(self) -> Dict:
        return {
            "generation_buffer_depth": len(self._generation_buffer),
            "scoring_buffer_depth": len(self._scoring_buffer),
            "ready_buffer_depth": len(self._ready_buffer),
            "batches_formed": self._batches_formed,
            "starvation_events": self._starvation_count,
            "dynamic_adjustments": self._dynamic_adjustments,
        }


# ============================================================
# PART 4: Auto-Scaler
# ============================================================

class ComponentAutoScaler:
    """
    Monitors component health and automatically scales workers
    to prevent bottlenecks.
    
    Key logic: If the Reward Model throughput < Policy throughput,
    scale up Reward Model workers to prevent training starvation.
    """
    
    def __init__(
        self,
        scale_up_threshold: float = 0.85,   # utilization above this = scale up
        scale_down_threshold: float = 0.3,   # utilization below this = scale down
        cooldown_seconds: float = 30.0,
        max_workers_per_component: int = 16,
    ):
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown = cooldown_seconds
        self.max_workers = max_workers_per_component
        
        self._last_scale_time: Dict[str, float] = {}
        self._scale_events: List[Dict] = []
    
    def evaluate(self, components: Dict[str, ComponentHealth]) -> List[Dict]:
        """
        Evaluate component health and recommend scaling actions.
        Returns list of scaling recommendations.
        """
        actions = []
        current_time = time.time()
        
        # Find the bottleneck
        throughputs = {
            name: health.throughput_per_sec 
            for name, health in components.items()
        }
        
        if not throughputs:
            return actions
        
        min_throughput_name = min(throughputs, key=throughputs.get)
        max_throughput = max(throughputs.values())
        
        for name, health in components.items():
            # Check cooldown
            last_scale = self._last_scale_time.get(name, 0)
            if current_time - last_scale < self.cooldown:
                continue
            
            # Scale up if overloaded
            if health.utilization > self.scale_up_threshold:
                if health.num_workers < self.max_workers:
                    new_workers = min(
                        health.num_workers + 2,
                        self.max_workers
                    )
                    action = {
                        "component": name,
                        "action": "scale_up",
                        "current_workers": health.num_workers,
                        "recommended_workers": new_workers,
                        "reason": f"Utilization {health.utilization:.1%} > threshold {self.scale_up_threshold:.1%}",
                    }
                    actions.append(action)
                    self._last_scale_time[name] = current_time
                    self._scale_events.append(action)
            
            # Scale down if underutilized
            elif health.utilization < self.scale_down_threshold:
                if health.num_workers > 1:
                    new_workers = max(1, health.num_workers - 1)
                    action = {
                        "component": name,
                        "action": "scale_down",
                        "current_workers": health.num_workers,
                        "recommended_workers": new_workers,
                        "reason": f"Utilization {health.utilization:.1%} < threshold {self.scale_down_threshold:.1%}",
                    }
                    actions.append(action)
                    self._last_scale_time[name] = current_time
                    self._scale_events.append(action)
        
        return actions


# ============================================================
# PART 5: RL Orchestrator (Main System)
# ============================================================

class RLOrchestrator:
    """
    Main orchestration system for distributed RLHF training.
    
    Manages the full Precompute → Train → Evaluate cycle:
    1. Policy generates responses (GPU cluster A)
    2. Reference model computes KL baseline (GPU cluster B)
    3. Reward model scores responses (GPU cluster C)
    4. Dynamic batcher forms training batches
    5. Training step updates policy weights
    
    The orchestrator ensures no GPU cluster sits idle by:
    - Prefetching generations ahead of the training loop
    - Dynamically sizing batches based on component speeds
    - Auto-scaling slow components
    
    Usage:
        orchestrator = RLOrchestrator(
            batch_size=32,
            num_generation_workers=4,
            num_reward_workers=2,
        )
        
        orchestrator.run(
            prompts=training_prompts,
            num_training_steps=1000,
        )
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        num_generation_workers: int = 4,
        num_reward_workers: int = 2,
        num_reference_workers: int = 2,
        prefetch_multiplier: int = 3,
        generation_latency_ms: float = 50.0,
        reward_latency_ms: float = 80.0,   # Reward model is slower
        reference_latency_ms: float = 20.0,
    ):
        # Components
        self.policy = PolicyActor(
            generation_latency_ms=generation_latency_ms,
            num_workers=num_generation_workers,
        )
        self.reward_model = RewardModel(
            scoring_latency_ms=reward_latency_ms,
            num_workers=num_reward_workers,
        )
        self.reference_model = ReferenceModel(
            inference_latency_ms=reference_latency_ms,
            num_workers=num_reference_workers,
        )
        
        # Batch management
        self.batch_manager = DynamicBatchManager(
            target_batch_size=batch_size,
            min_batch_size=batch_size // 4,
            max_buffer_size=batch_size * prefetch_multiplier,
        )
        
        # Auto-scaler
        self.auto_scaler = ComponentAutoScaler()
        
        # Thread pools for async operations
        # NOTE: Using ThreadPoolExecutor here for simulation (time.sleep releases GIL).
        # Production MUST use ProcessPoolExecutor or Ray actors for CPU-bound work
        # (tokenization, collation) to avoid GIL starvation of GPU training loops.
        self._gen_pool = ThreadPoolExecutor(max_workers=num_generation_workers)
        self._reward_pool = ThreadPoolExecutor(max_workers=num_reward_workers)
        self._ref_pool = ThreadPoolExecutor(max_workers=num_reference_workers)
        
        # Clean shutdown flag — prevents zombie threads in distributed PyTorch envs
        self._stop_event = threading.Event()
        
        # Metrics
        self._training_steps_completed = 0
        self._total_samples_processed = 0
        self._gpu_idle_time_ms = 0
        self._start_time = None
    
    def _generate_and_enqueue(self, prompt_tokens: List[int], sample_id: str):
        """Generate a response and add to scoring queue."""
        result = self.policy.generate(prompt_tokens)
        
        sample = RLSample(
            sample_id=sample_id,
            prompt_tokens=prompt_tokens,
            generated_tokens=result["generated_tokens"],
            policy_logprobs=result["policy_logprobs"],
            timestamp_generated=time.time(),
        )
        
        self.batch_manager.add_generated(sample)
    
    def _score_sample(self, sample: RLSample):
        """Score a sample with reward and reference models in parallel."""
        # Run reward and reference in parallel
        reward_future = self._reward_pool.submit(
            self.reward_model.score,
            sample.prompt_tokens,
            sample.generated_tokens,
        )
        ref_future = self._ref_pool.submit(
            self.reference_model.compute_logprobs,
            sample.prompt_tokens,
            sample.generated_tokens,
        )
        
        # Collect results with timeouts to prevent indefinite blocking
        # if a node fails or hangs. Gracefully drop sample on failure.
        try:
            reward_result = reward_future.result(timeout=5.0)
            ref_result = ref_future.result(timeout=5.0)
        except TimeoutError:
            logger.warning(f"Scoring timeout for {sample.sample_id} — dropping sample")
            reward_future.cancel()
            ref_future.cancel()
            return
        except Exception as e:
            logger.error(f"Scoring failed for {sample.sample_id}: {e} — dropping sample")
            return
        
        sample.reward_score = reward_result["reward_score"]
        sample.reference_logprobs = ref_result["reference_logprobs"]
        sample.timestamp_scored = time.time()
        
        # Compute KL penalty between policy and reference
        kl_penalty = 0.0
        if sample.reference_logprobs:
            for p_lp, r_lp in zip(sample.policy_logprobs, sample.reference_logprobs):
                kl_penalty += (p_lp - r_lp)
            kl_penalty /= len(sample.reference_logprobs)
        
        # TODO (Production): Replace scalar advantage broadcast with Generalized
        # Advantage Estimation (GAE, λ=0.95) computed over the token-level trajectory.
        # GAE properly credits specific tokens for the reward using temporal-difference
        # residuals: A_t = Σ(γλ)^l * δ_{t+l}, where δ_t = r_t + γV(s_{t+1}) - V(s_t).
        # The current broadcast assigns the same advantage to every token, which
        # works for prototyping but reduces training signal quality at scale.
        reward_with_kl = sample.reward_score - 0.1 * kl_penalty
        sample.advantages = [reward_with_kl] * len(sample.generated_tokens)
        sample.timestamp_ready = time.time()
        
        self.batch_manager.add_scored(sample)
    
    def _simulate_training_step(self, batch: RLBatch) -> Dict:
        """Simulate a PPO training step."""
        # In production: actual gradient computation and optimizer step
        time.sleep(0.02)  # Simulate ~20ms training step
        
        self._training_steps_completed += 1
        self._total_samples_processed += batch.batch_size
        
        return {
            "step": self._training_steps_completed,
            "batch_size": batch.batch_size,
            "avg_reward": batch.avg_reward,
            "loss": random.gauss(2.0, 0.3),
        }
    
    def run(self, num_prompts: int = 200, num_training_steps: int = 20):
        """
        Run the distributed RLHF training loop.
        """
        print("=" * 70)
        print("RL ORCHESTRATOR — DISTRIBUTED RLHF TRAINING")
        print("=" * 70)
        print(f"\nConfig:")
        print(f"  Prompts: {num_prompts}")
        print(f"  Target training steps: {num_training_steps}")
        print(f"  Policy workers: {self.policy.num_workers}")
        print(f"  Reward workers: {self.reward_model.num_workers}")
        print(f"  Reference workers: {self.reference_model.num_workers}")
        print(f"  Reward model latency: {self.reward_model.latency*1000:.0f}ms "
              f"(intentionally slower to demonstrate dynamic batching)\n")
        
        self._start_time = time.time()
        
        # Generate synthetic prompts
        prompts = [
            [random.randint(0, 50256) for _ in range(random.randint(50, 200))]
            for _ in range(num_prompts)
        ]
        
        # Phase 1: Prefetch generations
        print("--- Phase 1: Prefetching generations ---")
        gen_futures = []
        for i, prompt in enumerate(prompts):
            future = self._gen_pool.submit(
                self._generate_and_enqueue, prompt, f"sample_{i}"
            )
            gen_futures.append(future)
        
        # Phase 2: Score samples as they become available
        print("--- Phase 2: Scoring samples (async) ---")
        score_thread = threading.Thread(
            target=self._scoring_loop, daemon=True
        )
        score_thread.start()
        
        # Phase 3: Training loop
        print("--- Phase 3: Training loop ---")
        steps_completed = 0
        
        while steps_completed < num_training_steps:
            # Try to form a batch
            batch = self.batch_manager.try_form_batch()
            
            if batch is None:
                # GPU would be idle here — this is what we want to minimize
                self._gpu_idle_time_ms += 10
                time.sleep(0.01)
                continue
            
            # Execute training step
            result = self._simulate_training_step(batch)
            steps_completed += 1
            
            if steps_completed % 5 == 0:
                metrics = self.batch_manager.get_metrics()
                print(f"  Step {steps_completed}: batch_size={batch.batch_size}, "
                      f"reward={batch.avg_reward:.3f}, "
                      f"ready_buffer={metrics['ready_buffer_depth']}, "
                      f"gen_buffer={metrics['generation_buffer_depth']}")
                
                # Auto-scaling check
                component_health = {
                    "policy": self.policy.get_health(),
                    "reward": self.reward_model.get_health(),
                    "reference": self.reference_model.get_health(),
                }
                
                scale_actions = self.auto_scaler.evaluate(component_health)
                for action in scale_actions:
                    print(f"  [AutoScaler] {action['component']}: {action['action']} "
                          f"({action['current_workers']} → {action['recommended_workers']} workers) "
                          f"— {action['reason']}")
        
        # Summary
        elapsed = time.time() - self._start_time
        print(f"\n{'=' * 70}")
        print("ORCHESTRATION SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Total time: {elapsed:.1f}s")
        print(f"  Training steps: {steps_completed}")
        print(f"  Samples processed: {self._total_samples_processed}")
        print(f"  GPU idle time: {self._gpu_idle_time_ms:.0f}ms "
              f"({self._gpu_idle_time_ms/1000/elapsed*100:.1f}%)")
        print(f"  Throughput: {self._total_samples_processed/elapsed:.1f} samples/sec")
        
        batch_metrics = self.batch_manager.get_metrics()
        print(f"  Dynamic batch adjustments: {batch_metrics['dynamic_adjustments']}")
        print(f"  Starvation events: {batch_metrics['starvation_events']}")
        
        # Cleanup — signal stop event first, then shutdown pools
        # This prevents zombie threads and ensures clean resource release
        self._stop_event.set()
        self._gen_pool.shutdown(wait=True, cancel_futures=True)
        self._reward_pool.shutdown(wait=True, cancel_futures=True)
        self._ref_pool.shutdown(wait=True, cancel_futures=True)
    
    def _scoring_loop(self):
        """Background loop that scores generated samples.
        Uses stop_event for clean shutdown — prevents zombie threads/unclosed
        sockets in distributed PyTorch environments if main thread crashes.
        """
        while not self._stop_event.is_set():
            samples = self.batch_manager.get_pending_generation(n=4)
            if not samples:
                # Use wait() instead of sleep() so stop_event interrupts immediately
                self._stop_event.wait(timeout=0.01)
                continue
            
            for sample in samples:
                if self._stop_event.is_set():
                    break
                try:
                    self._score_sample(sample)
                except Exception as e:
                    logger.error(f"Scoring failed for {sample.sample_id}: {e}")


if __name__ == "__main__":
    orchestrator = RLOrchestrator(
        batch_size=16,
        num_generation_workers=4,
        num_reward_workers=2,
        num_reference_workers=2,
        generation_latency_ms=30.0,
        reward_latency_ms=80.0,     # Intentionally slow to show dynamic batching
        reference_latency_ms=20.0,
    )
    
    orchestrator.run(num_prompts=100, num_training_steps=15)
