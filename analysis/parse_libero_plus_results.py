"""
Parse LIBERO-Plus evaluation log and break down results by perturbation type
using the official task_classification.json from the LIBERO-Plus repo.

Usage:
    python parse_libero_plus_results.py EVAL-log.txt task_classification.json [--suite libero_spatial]
"""

import argparse
import json
import os
import re
from collections import defaultdict


def build_classification_lookup(classification, suite):
    """Build lookup: task_id (0-based) -> {category, difficulty_level}."""
    tasks = classification[suite]
    # The tasks are ordered by id (1-based), matching the eval order
    lookup = {}
    for entry in tasks:
        task_id = entry["id"] - 1  # Convert to 0-based
        lookup[task_id] = {
            "category": entry["category"],
            "difficulty": entry["difficulty_level"],
            "name": entry["name"],
        }
    return lookup


def parse_eval_log(log_path):
    """Parse the eval log to extract per-task results in order."""
    results = []
    current_task = None

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            m = re.match(r"^Task:\s+(.+)$", line)
            if m:
                current_task = m.group(1).strip()
                continue
            m = re.match(r"^Success:\s+(True|False)", line)
            if m and current_task is not None:
                results.append((current_task, m.group(1) == "True"))
                current_task = None

    return results


def main():
    parser = argparse.ArgumentParser(description="Parse LIBERO-Plus eval results by perturbation type")
    parser.add_argument("log_file", help="Path to eval log .txt or .out file")
    parser.add_argument("classification", help="Path to task_classification.json")
    parser.add_argument("--suite", default="libero_spatial",
                        help="Suite name (default: libero_spatial)")
    args = parser.parse_args()

    # Load classification
    with open(args.classification) as f:
        classification = json.load(f)

    if args.suite not in classification:
        print(f"Suite '{args.suite}' not found. Available: {list(classification.keys())}")
        return

    lookup = build_classification_lookup(classification, args.suite)

    # Parse eval log
    results = parse_eval_log(args.log_file)
    if not results:
        print(f"No results found in {args.log_file}")
        return

    # Match results to classification by task order
    cats = defaultdict(lambda: {"total": 0, "success": 0})
    cats_diff = defaultdict(lambda: defaultdict(lambda: {"total": 0, "success": 0}))

    for task_id, (task_name, success) in enumerate(results):
        if task_id in lookup:
            cat = lookup[task_id]["category"]
            diff = lookup[task_id]["difficulty"]
        else:
            cat = "Unknown"
            diff = None

        cats[cat]["total"] += 1
        cats[cat]["success"] += int(success)
        if diff is not None:
            cats_diff[cat][diff]["total"] += 1
            cats_diff[cat][diff]["success"] += int(success)

    # Print per-perturbation results
    print(f"Parsed {len(results)}/{len(lookup)} tasks from {os.path.basename(args.log_file)}")
    print(f"Suite: {args.suite}\n")
    print(f"{'Perturbation Type':<25} {'Success':>8} {'Total':>8} {'Rate':>8}")
    print("=" * 55)

    total_s, total_t = 0, 0
    for cat in sorted(cats.keys()):
        s = cats[cat]["success"]
        t = cats[cat]["total"]
        rate = s / t * 100 if t > 0 else 0
        print(f"{cat:<25} {s:>8} {t:>8} {rate:>7.1f}%")
        total_s += s
        total_t += t

    print("-" * 55)
    rate = total_s / total_t * 100 if total_t > 0 else 0
    print(f"{'TOTAL':<25} {total_s:>8} {total_t:>8} {rate:>7.1f}%")
    print("=" * 55)

    # Print per-difficulty breakdown
    print(f"\n{'Perturbation Type':<25} {'L1':>7} {'L2':>7} {'L3':>7} {'L4':>7} {'L5':>7}")
    print("=" * 65)
    for cat in sorted(cats_diff.keys()):
        row = f"{cat:<25}"
        for level in range(1, 6):
            d = cats_diff[cat].get(level, {"total": 0, "success": 0})
            if d["total"] > 0:
                rate = d["success"] / d["total"] * 100
                row += f" {rate:>6.1f}%"
            else:
                row += f" {'--':>6} "
        print(row)
    print("=" * 65)


if __name__ == "__main__":
    main()
