#!/usr/bin/env python3
"""
ake.py — ratios & factorials for a flow JSON consumed by the DQN trainer.

Loads a flow_*.json (list of {route, interval, vehicle, ...} dicts), groups
flows by direction suffix (0=WE, 1=NS, 2=EW, 3=SN), and prints:

  RATIOS
    · Per-direction demand (veh/h), share %, mean interval
    · Pairwise demand ratios     (WE:EW, NS:SN, EW-axis:NS-axis, max:min)
    · Pairwise flow-count ratios (n_we:n_ew, n_ns:n_sn, axis ratios)

  FACTORIALS
    · Exact factorials of small counts  (n_flows!, per-direction n!)
    · Factorials of integer intervals   (2.0 s -> 2! = 2)
    · log10(n!) for per-direction demand — avoids 4-digit-exponent overflow

  DETAIL
    · Top-N flows by demand, with route length, direction, interval

Usage
-----
    python ake.py flow_3_4.json
    python ake.py flow_3_4.json --json        # machine-readable
    python ake.py flow_3_4.json --top 10      # top-10 by demand
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

DIR_NAMES = {0: "WE (east)", 1: "NS (south)", 2: "EW (west)", 3: "SN (north)"}
DIR_SHORT = {0: "WE", 1: "NS", 2: "EW", 3: "SN"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def parse_road(road_id: str) -> tuple[int, int, int]:
    """road_{x}_{y}_{d} -> (x, y, d)."""
    p = road_id.split("_")
    return int(p[1]), int(p[2]), int(p[3])


def direction_of(flow: dict) -> int:
    return parse_road(flow["route"][0])[2]


def interval_of(flow: dict) -> float:
    """Interval may be at top level or inside a SUMO-style .vehicle block."""
    if "interval" in flow:
        return float(flow["interval"])
    veh = flow.get("vehicle", {})
    if "interval" in veh:
        return float(veh["interval"])
    return 1.0


def demand_per_hour(interval: float) -> float:
    return 3600.0 / interval if interval > 0 else 0.0


def log10_factorial(n: int) -> float:
    """log10(n!) via lgamma — safe for any n that fits in a Python int."""
    if n < 0:
        return float("nan")
    return math.lgamma(n + 1) / math.log(10)


def fmt_factorial(n: int) -> str:
    """Exact for n <= 20, log10 form otherwise."""
    if n < 0:
        return "n/a"
    if n <= 20:
        return f"{n}! = {math.factorial(n):,}"
    return f"{n}! ≈ 10^{log10_factorial(n):.3f}"


def safe_ratio(a: float, b: float) -> float | None:
    return a / b if b else None


# --------------------------------------------------------------------------- #
# core report
# --------------------------------------------------------------------------- #
def build_report(flows: list[dict]) -> dict:
    # Per-flow rows
    detail: list[dict] = []
    for i, f in enumerate(flows, start=1):
        d = direction_of(f)
        iv = interval_of(f)
        detail.append({
            "flow_index":         i,
            "direction":          d,
            "direction_short":    DIR_SHORT[d],
            "n_segments":         len(f["route"]),
            "interval_s":         iv,
            "demand_veh_per_h":   demand_per_hour(iv),
        })

    # Per-direction aggregate
    by_dir: dict[int, list[dict]] = defaultdict(list)
    for row in detail:
        by_dir[row["direction"]].append(row)

    per_dir: dict[int, dict] = {}
    total_count = 0
    total_demand = 0.0
    for d in sorted(by_dir):
        rows = by_dir[d]
        cnt = len(rows)
        total_veh = sum(r["demand_veh_per_h"] for r in rows)
        avg_iv = sum(r["interval_s"] for r in rows) / cnt
        per_dir[d] = {
            "name":                    DIR_NAMES[d],
            "short":                   DIR_SHORT[d],
            "count":                   cnt,
            "total_demand_veh_per_h":  total_veh,
            "mean_interval_s":         avg_iv,
            "min_interval_s":          min(r["interval_s"] for r in rows),
            "max_interval_s":          max(r["interval_s"] for r in rows),
        }
        total_count += cnt
        total_demand += total_veh

    for d in per_dir:
        per_dir[d]["share_pct"] = (
            100.0 * per_dir[d]["total_demand_veh_per_h"] / total_demand
            if total_demand else 0.0
        )

    # Ratios
    we = per_dir.get(0, {}).get("total_demand_veh_per_h", 0.0)
    ew = per_dir.get(2, {}).get("total_demand_veh_per_h", 0.0)
    ns = per_dir.get(1, {}).get("total_demand_veh_per_h", 0.0)
    sn = per_dir.get(3, {}).get("total_demand_veh_per_h", 0.0)
    we_c = per_dir.get(0, {}).get("count", 0)
    ew_c = per_dir.get(2, {}).get("count", 0)
    ns_c = per_dir.get(1, {}).get("count", 0)
    sn_c = per_dir.get(3, {}).get("count", 0)

    positive = [v for v in (we, ew, ns, sn) if v > 0]
    ratios = {
        "demand_WE_over_EW":            safe_ratio(we, ew),
        "demand_NS_over_SN":            safe_ratio(ns, sn),
        "demand_EWaxis_over_NSaxis":    safe_ratio(we + ew, ns + sn),
        "demand_max_over_min":          (safe_ratio(max(positive), min(positive))
                                         if len(positive) >= 2 else None),
        "count_WE_over_EW":             safe_ratio(we_c, ew_c),
        "count_NS_over_SN":             safe_ratio(ns_c, sn_c),
        "count_EWaxis_over_NSaxis":     safe_ratio(we_c + ew_c, ns_c + sn_c),
    }

    # Factorials
    intervals = sorted(set(round(r["interval_s"], 6) for r in detail))
    interval_facts: dict[str, str] = {}
    for iv in intervals:
        if iv > 0 and abs(iv - round(iv)) < 1e-9:
            interval_facts[f"{iv:g}s"] = fmt_factorial(int(round(iv)))
        else:
            interval_facts[f"{iv:g}s"] = "non-integer — skipped"

    demand_facts: dict[str, dict] = {}
    for d in per_dir:
        total = per_dir[d]["total_demand_veh_per_h"]
        n = int(round(total))
        demand_facts[DIR_SHORT[d]] = {
            "veh_per_h":       total,
            "log10_factorial": log10_factorial(n),
        }

    return {
        "total_flows":               total_count,
        "total_demand_veh_per_h":    total_demand,
        "per_direction":             per_dir,
        "ratios":                    ratios,
        "factorials": {
            "n_flows":                 total_count,
            "n_flows_factorial":       fmt_factorial(total_count),
            "per_direction_counts": {
                DIR_SHORT[d]: {
                    "count":     per_dir[d]["count"],
                    "factorial": fmt_factorial(per_dir[d]["count"]),
                } for d in per_dir
            },
            "interval_factorials":     interval_facts,
            "demand_log10_factorials": demand_facts,
        },
        "flows":                     detail,
    }


# --------------------------------------------------------------------------- #
# pretty printer
# --------------------------------------------------------------------------- #
def print_report(rep: dict, top: int) -> None:
    bar = "=" * 78
    print(bar)
    print(f" FLOW ANALYSIS — {rep['total_flows']} flows · "
          f"total demand ≈ {rep['total_demand_veh_per_h']:.1f} veh/h")
    print(bar)

    # per-direction
    print("\n PER-DIRECTION")
    print("-" * 78)
    print(f"  {'dir':<6}{'name':<16}{'flows':>7}{'demand':>14}"
          f"{'share':>9}{'mean int':>11}")
    for d in sorted(rep["per_direction"]):
        info = rep["per_direction"][d]
        print(f"  {info['short']:<6}{info['name']:<16}"
              f"{info['count']:>7}"
              f"{info['total_demand_veh_per_h']:>11.1f}/h"
              f"{info['share_pct']:>8.1f}%"
              f"{info['mean_interval_s']:>10.3f}s")

    # ratios
    print("\n RATIOS")
    print("-" * 78)
    for key, val in rep["ratios"].items():
        if val is None:
            print(f"  {key:<32} n/a (division by zero)")
        else:
            print(f"  {key:<32} {val:>10.4f}")

    # factorials
    print("\n FACTORIALS")
    print("-" * 78)
    fac = rep["factorials"]
    print(f"  total flows          {fac['n_flows_factorial']}")
    for short, info in fac["per_direction_counts"].items():
        print(f"  per-direction {short:<5}    {info['factorial']}")
    for iv, txt in fac["interval_factorials"].items():
        print(f"  interval {iv:<8}     {txt}")
    print()
    for short, info in fac["demand_log10_factorials"].items():
        print(f"  {short} demand {info['veh_per_h']:>8.1f} veh/h → "
              f"log10(n!) ≈ {info['log10_factorial']:.3f}")
    print("\n  note: factorials of veh/h are combinatorially meaningful "
          "(orderings of arrivals),\n"
          "        not a quantity the trainer uses — presented for completeness.")

    # top flows
    top = min(max(0, top), len(rep["flows"]))
    if top:
        rows = sorted(rep["flows"], key=lambda r: -r["demand_veh_per_h"])[:top]
        print(f"\n TOP-{top} FLOWS (highest demand)")
        print("-" * 78)
        print(f"  {'#':>3}  {'dir':<6}{'segments':>9}{'interval':>10}{'demand':>12}")
        for r in rows:
            print(f"  {r['flow_index']:>3}  {r['direction_short']:<6}"
                  f"{r['n_segments']:>9}{r['interval_s']:>9.3f}s"
                  f"{r['demand_veh_per_h']:>10.1f}/h")
    print()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("flow_file",
                   help="path to a flow_*.json (list of {route, interval} dicts)")
    p.add_argument("--json", action="store_true",
                   help="emit machine-readable JSON instead of a table")
    p.add_argument("--top", type=int, default=5,
                   help="print the top-N highest-demand flows (default 5)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.isfile(args.flow_file):
        print(f"[ake] file not found: {args.flow_file}", file=sys.stderr)
        return 1

    with open(args.flow_file) as f:
        flows = json.load(f)

    if not isinstance(flows, list) or not flows:
        print(f"[ake] expected a non-empty JSON list in {args.flow_file}",
              file=sys.stderr)
        return 1

    try:
        rep = build_report(flows)
    except (KeyError, ValueError) as e:
        print(f"[ake] malformed flow entry: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        print_report(rep, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())