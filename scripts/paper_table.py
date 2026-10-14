#!/usr/bin/env python3
"""Build mean±std table rows from eval JSON (seeds 0..n-1)."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def mean_std(xs):
    xs = [float(x) for x in xs]
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v)


def j_of(row):
    if str(row.get("controller", "")).startswith("syscontract"):
        return row.get("best_safe_objective", row["final_objective"])
    return row["final_objective"]


def main():
    files = [Path(p) for p in sys.argv[1:]]
    rows = []
    for p in files:
        data = json.loads(p.read_text())
        rows.extend(data if isinstance(data, list) else [data])

    groups = defaultdict(list)
    for r in rows:
        if r.get("seed", 0) > 4:
            continue
        groups[(r["env"], r.get("agent"), r["controller"])].append(r)

    print(f"{'env':8} {'agent':10} {'ctrl':22} {'J':>16} {'hold':>6} {'tok':>8} {'n':>3}")
    out = []
    for key in sorted(groups):
        env, agent, ctrl = key
        g = groups[key]
        jm, js = mean_std([j_of(x) for x in g])
        hm, _ = mean_std([x.get("final_hold", 1) for x in g])
        tm, ts = mean_std([x["tokens"] for x in g])
        am, _ = mean_std([x["tokens_after_last_improve"] for x in g])
        rec = {
            "env": env,
            "agent": agent,
            "controller": ctrl,
            "J": round(jm, 3),
            "J_std": round(js, 3),
            "hold": round(hm, 2),
            "tokens": round(tm, 1),
            "tokens_std": round(ts, 1),
            "after": round(am, 1),
            "n": len(g),
        }
        out.append(rec)
        print(
            f"{env:8} {str(agent):10} {ctrl:22} {jm:7.3f}±{js:<6.3f} {hm:6.2f} {tm:8.1f} {len(g):3d}"
        )
    dest = ROOT / "results" / "paper_table.json"
    dest.write_text(json.dumps(out, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
