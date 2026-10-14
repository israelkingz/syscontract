#!/usr/bin/env python3
"""Aggregate eval JSON into means for the paper table."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def j_of(row):
    ctrl = row["controller"]
    if ctrl.startswith("syscontract"):
        return row.get("best_safe_objective", row["final_objective"])
    return row["final_objective"]


def main():
    files = [Path(f) for f in sys.argv[1:]] if len(sys.argv) > 1 else [
        p for p in (ROOT / "results").glob("*.json")
        if p.name not in {"agg.json", "smoke_llm.json"}
    ]
    rows = []
    for p in files:
        data = json.loads(p.read_text())
        if isinstance(data, list):
            rows.extend(data)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["env"], r.get("agent"), r["controller"])].append(r)

    out = []
    print(f"{'env':8} {'agent':8} {'controller':20} {'J':>10} {'hold':>6} {'tok':>8} {'after':>8} {'n':>3}")
    for key in sorted(groups):
        env, agent, ctrl = key
        g = groups[key]
        rec = {
            "env": env,
            "agent": agent,
            "controller": ctrl,
            "J": round(mean([j_of(x) for x in g]), 3),
            "hold": round(mean([x.get("final_hold", 1 - x.get("any_violation", 0)) for x in g]), 2),
            "tokens": round(mean([x["tokens"] for x in g]), 1),
            "tokens_after_plateau": round(mean([x["tokens_after_last_improve"] for x in g]), 1),
            "n": len(g),
        }
        out.append(rec)
        print(
            f"{env:8} {str(agent):8} {ctrl:20} {rec['J']:10.3f} {rec['hold']:6.2f} "
            f"{rec['tokens']:8.1f} {rec['tokens_after_plateau']:8.1f} {rec['n']:3d}"
        )

    dest = ROOT / "results" / "agg.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
