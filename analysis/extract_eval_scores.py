"""
Extract per-task success rates from ReasonVLA evaluation log files.

Usage:
    python extract_eval_scores.py <log_file.out> [log_file2.out ...]
    python extract_eval_scores.py /path/to/logs/reasonvla-eval_*.out
"""

import sys
import re
import os


def extract_scores(filepath):
    """Extract experiment info and per-task success rates from a log file."""
    with open(filepath) as f:
        content = f.read()

    # Extract experiment info from header
    info = {}
    for key in ["Checkpoint", "Stage", "LoRA dir", "Hidden layer", "Feedback", "Trials/task"]:
        match = re.search(rf"{key}:\s+(.+)", content)
        if match:
            info[key] = match.group(1).strip()

    # Extract job ID from filename
    job_match = re.search(r"(\d{7})", os.path.basename(filepath))
    job_id = job_match.group(1) if job_match else "unknown"

    # Extract run ID from checkpoint path (supports multiple run-folder prefixes)
    ckpt = info.get("Checkpoint", "")
    run_match = (re.search(r"reason-vla-(\d+)", ckpt)
                 or re.search(r"rvla-projcrossattn-(\d+)", ckpt)
                 or re.search(r"rvla-textcrossattn-(\d+)", ckpt))
    run_id = run_match.group(1) if run_match else "unknown"

    # Extract checkpoint step
    ckpt_match = re.search(r"checkpoint-(\d+)", info.get("Checkpoint", ""))
    ckpt_step = ckpt_match.group(1) if ckpt_match else "final"

    # Extract per-task success rates
    task_rates = re.findall(r"Current task success rate: ([\d.]+)", content)
    task_pcts = [round(float(r) * 100) for r in task_rates]

    # Calculate average
    avg = sum(task_pcts) / len(task_pcts) if task_pcts else 0

    return {
        "job_id": job_id,
        "run_id": run_id,
        "ckpt_step": ckpt_step,
        "stage": info.get("Stage", "?"),
        "hidden_layer": info.get("Hidden layer", "?"),
        "feedback": info.get("Feedback", "?"),
        "task_rates": task_pcts,
        "avg": avg,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_eval_scores.py <log_file.out> [log_file2.out ...]")
        sys.exit(1)

    results = []
    for filepath in sys.argv[1:]:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}", file=sys.stderr)
            continue
        try:
            result = extract_scores(filepath)
            if result["task_rates"]:
                results.append(result)
            else:
                print(f"No task results found in: {filepath}", file=sys.stderr)
        except Exception as e:
            print(f"Error processing {filepath}: {e}", file=sys.stderr)

    if not results:
        print("No results found.")
        return

    # Print header
    print(f"{'Experiment':<45} {'hl':<5} {'stage':<6} {'T0':<5} {'T1':<5} {'T2':<5} {'T3':<5} {'T4':<5} {'T5':<5} {'T6':<5} {'T7':<5} {'T8':<5} {'T9':<5} {'Avg':<6}")
    print("-" * 120)

    for r in results:
        name = f"rvla-{r['run_id']}-ckpt{r['ckpt_step']}-s{r['stage']}"
        tasks = r["task_rates"]
        task_strs = [f"{t}%" for t in tasks[:10]]
        # Pad if fewer than 10 tasks
        while len(task_strs) < 10:
            task_strs.append("-")
        print(f"{name:<45} {r['hidden_layer']:<5} {r['stage']:<6} {task_strs[0]:<5} {task_strs[1]:<5} {task_strs[2]:<5} {task_strs[3]:<5} {task_strs[4]:<5} {task_strs[5]:<5} {task_strs[6]:<5} {task_strs[7]:<5} {task_strs[8]:<5} {task_strs[9]:<5} {r['avg']:.1f}%")


if __name__ == "__main__":
    main()
