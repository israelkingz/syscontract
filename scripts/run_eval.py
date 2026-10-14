#!/usr/bin/env python3
"""Run SysContract vs unconstrained vs default vs random, including GPT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT.parent / ".env")
load_dotenv(ROOT / ".env")

from envs.cluster import ClusterPackEnv
from envs.serving_sim import ServingBatchEnv
from envs.sqlite_tune import SQLiteTuneEnv
from src.agents import DefaultAgent, LLMAgent, MetricHacker, RandomAgent, ReplayAgent
from src.runtime import SysContract, UnconstrainedLoop, _hold
from src.types import EpisodeLog


def summarize(log: EpisodeLog) -> dict:
    return {
        "env": log.env_name,
        "controller": log.controller,
        "stopped": log.stopped_reason,
        "tokens": log.tokens_total,
        "tokens_after_last_improve": log.tokens_after_last_improve,
        "violation_steps": log.violation_steps,
        "gated_rejects": log.gated_rejects,
        "final_objective": log.final_objective,
        "best_safe_objective": log.best_safe_objective,
        "n_steps": len(log.steps),
        "any_violation": int(log.violation_steps > 0),
        "final_hold": log.final_hold,
        "n_proposals": len(log.proposals),
    }


def make_env(name: str):
    if name == "cluster":
        return ClusterPackEnv()
    if name == "serving":
        return ServingBatchEnv()
    if name == "sqlite":
        return SQLiteTuneEnv()
    raise SystemExit(f"unknown env {name}")


def make_agent(name: str, seed: int, model: str, aware: bool = False):
    if name == "default":
        return DefaultAgent()
    if name == "random":
        return RandomAgent(seed)
    if name == "hacker":
        return MetricHacker(seed)
    if name == "llm":
        return LLMAgent(model=model, aware=aware)
    raise SystemExit(f"unknown agent {name}")


def run_default(env_name: str, seed: int) -> dict:
    env = make_env(env_name)
    env.reset(seed)
    log = EpisodeLog(env_name=env.name, controller="default")
    log.final_objective = env.objective()
    log.best_safe_objective = env.objective()
    log.final_hold = _hold(env)
    log.stopped_reason = "default"
    row = summarize(log)
    row["seed"] = seed
    row["agent"] = "default"
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="cluster", choices=["cluster", "serving", "sqlite"])
    p.add_argument("--agent", default="llm", choices=["default", "random", "hacker", "llm"])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--aware", action="store_true", help="Show hard constraints in the GPT prompt")
    p.add_argument("--unconstrained-only", action="store_true")
    p.add_argument("--replay-gate", action="store_true", help="Skip live SysContract GPT (replay only)")
    p.add_argument("--out", default="")
    args = p.parse_args()

    rows = []
    if args.agent == "default":
        for seed in range(args.seeds):
            row = run_default(args.env, seed)
            rows.append(row)
            print(json.dumps(row), flush=True)
    else:
        for seed in range(args.seeds):
            env_u = make_env(args.env)
            agent_u = make_agent(args.agent, seed, args.model, aware=args.aware)
            log_u = UnconstrainedLoop(env_u, agent_u, max_steps=args.steps).run(seed=seed)
            if args.aware:
                log_u.controller = "unconstrained_aware"
            row_u = summarize(log_u)
            row_u["seed"] = seed
            row_u["agent"] = "llm_aware" if args.aware else args.agent
            rows.append(row_u)
            print(json.dumps(row_u), flush=True)

            if args.unconstrained_only:
                continue

            env_r = make_env(args.env)
            agent_r = ReplayAgent(log_u.proposals, log_u.proposal_tokens)
            log_r = SysContract(env_r, agent_r, max_steps=args.steps).run(seed=seed)
            log_r.controller = "syscontract_replay"
            row_r = summarize(log_r)
            row_r["seed"] = seed
            row_r["agent"] = row_u["agent"]
            rows.append(row_r)
            print(json.dumps(row_r), flush=True)

            if not args.replay_gate:
                env_s = make_env(args.env)
                agent_s = make_agent(args.agent, seed + 10_000, args.model, aware=args.aware)
                log_s = SysContract(env_s, agent_s, max_steps=args.steps).run(seed=seed)
                row_s = summarize(log_s)
                row_s["seed"] = seed
                row_s["agent"] = row_u["agent"]
                rows.append(row_s)
                print(json.dumps(row_s), flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
