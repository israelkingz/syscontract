"""Stop-Check-Act runtime.

Every agent proposal is admitted only if a machine-checked envelope holds.
A sequential stopping rule ends the episode when expected gain no longer
justifies the token cost. Failed or violating actions roll back.
"""

from __future__ import annotations

from typing import Protocol

from src.types import Action, EpisodeLog, StepResult


class SystemsEnv(Protocol):
    name: str

    def reset(self, seed: int = 0) -> dict: ...
    def observe(self) -> dict: ...
    def preview(self, action: Action) -> list[InvariantViolation]: ...
    def step(self, action: Action) -> StepResult: ...
    def rollback(self) -> None: ...
    def objective(self) -> float: ...
    def higher_is_better(self) -> bool: ...


class Agent(Protocol):
    def propose(self, observation: dict, history: list[StepResult]) -> tuple[Action, int]:
        """Return (action, tokens_used)."""


def _hold(env: SystemsEnv) -> int:
    try:
        return 0 if env.preview(Action(knobs={})) else 1
    except Exception:
        return 1


def _attach_action(result: StepResult, action: Action) -> StepResult:
    result.info = dict(result.info or {})
    result.info.setdefault("action", action.knobs)
    return result


class SysContract:
    def __init__(
        self,
        env: SystemsEnv,
        agent: Agent,
        max_steps: int = 8,
        min_steps: int = 2,
        epsilon: float = 0.01,
        patience: int = 2,
        token_budget: int = 12_000,
        gate: bool = True,
        stop: bool = True,
        rollback_on_violation: bool = True,
    ):
        self.env = env
        self.agent = agent
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.epsilon = epsilon
        self.patience = patience
        self.token_budget = token_budget
        self.gate = gate
        self.stop = stop
        self.rollback_on_violation = rollback_on_violation

    def run(self, seed: int = 0) -> EpisodeLog:
        self.env.reset(seed)
        log = EpisodeLog(env_name=self.env.name, controller="syscontract")
        best = self.env.objective()
        last_improve_tokens = 0
        stale = 0
        better = (lambda a, b: a > b) if self.env.higher_is_better() else (lambda a, b: a < b)

        def rel(new, old):
            return abs(new - old) / max(abs(old), 1e-9)

        for t in range(self.max_steps):
            if log.tokens_total >= self.token_budget:
                log.stopped_reason = "token_budget"
                break

            obs = self.env.observe()
            action, tokens = self.agent.propose(obs, log.steps)
            log.proposals.append(action)
            log.proposal_tokens.append(tokens)
            log.tokens_total += tokens

            if self.gate:
                violations = self.env.preview(action)
                if violations:
                    log.gated_rejects += 1
                    log.steps.append(
                        StepResult(
                            applied=False,
                            rejected_by_gate=True,
                            rolled_back=False,
                            objective=self.env.objective(),
                            violations=violations,
                            tokens=tokens,
                            info={"action": action.knobs},
                        )
                    )
                    stale += 1
                    if self.stop and t + 1 >= self.min_steps and stale >= self.patience:
                        log.stopped_reason = "stop_after_rejects"
                        break
                    continue

            result = _attach_action(self.env.step(action), action)
            result.tokens = tokens
            if result.violations and self.rollback_on_violation:
                self.env.rollback()
                result.rolled_back = True
                result.objective = self.env.objective()
                log.violation_steps += 1
            log.steps.append(result)

            obj = result.objective
            if better(obj, best) and rel(obj, best) >= self.epsilon:
                best = obj
                last_improve_tokens = log.tokens_total
                stale = 0
            else:
                stale += 1

            if self.stop and t + 1 >= self.min_steps and stale >= self.patience:
                log.stopped_reason = "sequential_stop"
                break
        else:
            log.stopped_reason = "max_steps"

        log.best_safe_objective = best
        log.final_objective = self.env.objective()
        log.tokens_after_last_improve = max(0, log.tokens_total - last_improve_tokens)
        log.final_hold = _hold(self.env)
        return log


class UnconstrainedLoop:
    """ReAct-style loop: apply whatever the agent emits. No gate, no stop, no rollback."""

    def __init__(self, env: SystemsEnv, agent: Agent, max_steps: int = 8, token_budget: int = 12_000):
        self.env = env
        self.agent = agent
        self.max_steps = max_steps
        self.token_budget = token_budget

    def run(self, seed: int = 0) -> EpisodeLog:
        self.env.reset(seed)
        log = EpisodeLog(env_name=self.env.name, controller="unconstrained")
        start_tokens_at_best = 0
        best = self.env.objective()
        better = (lambda a, b: a > b) if self.env.higher_is_better() else (lambda a, b: a < b)

        for _ in range(self.max_steps):
            if log.tokens_total >= self.token_budget:
                log.stopped_reason = "token_budget"
                break
            action, tokens = self.agent.propose(self.env.observe(), log.steps)
            log.proposals.append(action)
            log.proposal_tokens.append(tokens)
            log.tokens_total += tokens
            result = _attach_action(self.env.step(action), action)
            result.tokens = tokens
            if result.violations:
                log.violation_steps += 1
            log.steps.append(result)
            if better(result.objective, best):
                best = result.objective
                start_tokens_at_best = log.tokens_total

        log.stopped_reason = log.stopped_reason or "max_steps"
        log.best_safe_objective = best
        log.final_objective = self.env.objective()
        log.tokens_after_last_improve = max(0, log.tokens_total - start_tokens_at_best)
        log.final_hold = _hold(self.env)
        return log
