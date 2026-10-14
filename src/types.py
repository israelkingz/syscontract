from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Action:
    """A proposed configuration delta. Keys are env-specific knobs."""

    knobs: Dict[str, Any]
    rationale: str = ""


@dataclass
class InvariantViolation:
    name: str
    detail: str


@dataclass
class StepResult:
    applied: bool
    rejected_by_gate: bool
    rolled_back: bool
    objective: float
    violations: List[InvariantViolation] = field(default_factory=list)
    tokens: int = 0
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeLog:
    env_name: str
    controller: str
    steps: List[StepResult] = field(default_factory=list)
    proposals: List[Action] = field(default_factory=list)
    proposal_tokens: List[int] = field(default_factory=list)
    stopped_reason: str = ""
    tokens_total: int = 0
    best_safe_objective: Optional[float] = None
    final_objective: Optional[float] = None
    violation_steps: int = 0
    gated_rejects: int = 0
    tokens_after_last_improve: int = 0
    final_hold: int = 1
