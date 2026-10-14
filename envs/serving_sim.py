"""Fluid model of LLM serving batching.

Agent sees mean tokens/s. Invariants:
  - TTFT SLO for interactive sessions
  - KV-cache never exceeds GPU memory
  - no request starved beyond a deadline
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from src.types import Action, InvariantViolation, StepResult


class ServingBatchEnv:
    name = "serving"

    def __init__(self, n_req: int = 120, seed: int = 0):
        self.n_req = n_req
        self.gpu_mem = 80.0  # GB
        self.kv_per_token = 0.0005
        self.slo_ttft = 0.35
        self.max_starve = 4.0
        self.knobs = {
            "max_num_seqs": 16.0,
            "prefill_priority": 1.0,
            "gpu_mem_util": 0.85,
        }
        self._snapshot: Dict[str, Any] = {}
        self._last_metrics: dict = {}

    def higher_is_better(self) -> bool:
        return True  # tokens / s

    def reset(self, seed: int = 0) -> dict:
        rng = random.Random(seed)
        self.reqs = []
        t = 0.0
        for i in range(self.n_req):
            t += rng.expovariate(8.0)
            interactive = rng.random() < 0.4
            self.reqs.append(
                {
                    "arrival": t,
                    "prefill": rng.randint(32, 512),
                    "decode": rng.randint(16, 256),
                    "interactive": interactive,
                }
            )
        self.knobs = {"max_num_seqs": 16.0, "prefill_priority": 1.0, "gpu_mem_util": 0.85}
        self._last_metrics = self._simulate()
        self._snapshot = {"knobs": dict(self.knobs)}
        return self.observe()

    def observe(self) -> dict:
        m = self._last_metrics
        return {
            "task": "Maximize tokens/s. Set max_num_seqs, prefill_priority, gpu_mem_util.",
            "knobs": dict(self.knobs),
            "tokens_per_s": round(m["tps"], 2),
            "ranges": {
                "max_num_seqs": [1, 64],
                "prefill_priority": [0.1, 4.0],
                "gpu_mem_util": [0.4, 0.99],
            },
            "hard_constraints": [
                f"p99 TTFT must be <= {self.slo_ttft}s",
                "KV-cache peak must stay under gpu_mem * gpu_mem_util",
                f"no request may starve longer than {self.max_starve}s",
            ],
        }

    def preview(self, action: Action) -> List[InvariantViolation]:
        trial = dict(self.knobs)
        trial.update({k: float(v) for k, v in action.knobs.items() if k in trial})
        return self._check(self._simulate(trial), trial)

    def step(self, action: Action) -> StepResult:
        self._snapshot = {"knobs": dict(self.knobs)}
        self.knobs.update({k: float(v) for k, v in action.knobs.items() if k in self.knobs})
        self._last_metrics = self._simulate()
        return StepResult(
            applied=True,
            rejected_by_gate=False,
            rolled_back=False,
            objective=self.objective(),
            violations=self._check(self._last_metrics, self.knobs),
            info=dict(self._last_metrics),
        )

    def rollback(self) -> None:
        self.knobs = dict(self._snapshot["knobs"])
        self._last_metrics = self._simulate()

    def objective(self) -> float:
        return float(self._last_metrics["tps"])

    def _check(self, m: dict, knobs: dict) -> List[InvariantViolation]:
        out = []
        if m["peak_mem"] > self.gpu_mem * float(knobs["gpu_mem_util"]) + 1e-6:
            out.append(InvariantViolation("kv_oom", f"peak_mem={m['peak_mem']:.1f}GB"))
        if m["p99_ttft"] > self.slo_ttft:
            out.append(InvariantViolation("ttft_slo", f"p99_ttft={m['p99_ttft']:.3f}"))
        if m["max_starve"] > self.max_starve:
            out.append(InvariantViolation("no_starvation", f"starve={m['max_starve']:.2f}"))
        return out

    def _simulate(self, knobs: dict | None = None) -> dict:
        k = knobs or self.knobs
        max_seqs = int(max(1, min(64, k["max_num_seqs"])))
        pref_p = max(0.1, k["prefill_priority"])
        mem_cap = self.gpu_mem * float(k["gpu_mem_util"])

        # Cheap fluid model: higher batch → tps up, TTFT up, mem up.
        tps = 120.0 * (max_seqs**0.55) * (0.7 + 0.15 * pref_p)
        p99_ttft = 0.08 + 0.012 * max_seqs + 0.04 * (2.0 - min(pref_p, 2.0))
        peak_mem = 18.0 + 0.9 * max_seqs + 12.0 * (k["gpu_mem_util"] - 0.5)
        max_starve = 0.3 + 0.08 * max_seqs / max(pref_p, 0.2)
        # Interactive mix penalty if prefill_priority is low
        if pref_p < 0.6:
            p99_ttft *= 1.8
            max_starve *= 2.2
        return {
            "tps": tps,
            "p99_ttft": p99_ttft,
            "peak_mem": peak_mem,
            "max_starve": max_starve,
            "mem_cap": mem_cap,
        }
