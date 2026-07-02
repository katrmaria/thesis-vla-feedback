"""
Extract per-task success rates from attention modulation evaluation logs.
Supports: attention mask (vision-side), test-time attn mod v1/v2 (language-side),
ViT attention bias, Song faithful, logit bias, attn gate, text cross-attention.

Usage:
    python extract_attn_eval_scores.py <log_file.out> [log_file2.out ...]
    python extract_attn_eval_scores.py /path/to/logs/attnmask-repro_*.out /path/to/logs/ttamv2-repro_*.out
"""

import sys
import re
import os


def extract_key(content, key):
    """Extract a value from 'Key:  value' pattern."""
    match = re.search(rf"{key}:\s+(.+)", content)
    return match.group(1).strip() if match else None


def detect_type(content, filename):
    """Detect experiment type from log content."""
    if "ViTAttentionBiasVLA" in content or "vit_bias" in content:
        return "vit_bias"
    if "method: song" in content or "Method: song" in content:
        return "song"
    if "method: logit_bias" in content or "Method: logit_bias" in content:
        return "logit_bias"
    if "method: attn_gate" in content or "Method: attn_gate" in content:
        return "attn_gate"
    if "TestTimeAttentionModulation" in content or "Teacher:" in content:
        return "ttam_v1"
    if "AttentionMaskVLA" in content or "Attn layers:" in content or "attnmask" in filename:
        return "attn_mask"
    if "TextCrossAttn" in content or "textcrossattn" in filename.lower():
        return "text_crossattn"
    if "ReasonVLA" in content or "reason" in filename.lower():
        return "reasonvla"
    return "unknown"


def extract_scores(filepath):
    """Extract experiment info and per-task success rates from a log file."""
    with open(filepath) as f:
        content = f.read()

    filename = os.path.basename(filepath)

    # Job ID from filename
    job_match = re.search(r"(\d{7})", filename)
    job_id = job_match.group(1) if job_match else "?"

    # Per-task success rates
    task_rates = re.findall(r"Current task success rate: ([\d.]+)", content)
    task_pcts = [round(float(r) * 100) for r in task_rates]
    avg = sum(task_pcts) / len(task_pcts) if task_pcts else 0

    exp_type = detect_type(content, filename)

    if exp_type == "attn_mask":
        mode = extract_key(content, "Mode display") or extract_key(content, "Mode") or "?"
        floor_val = extract_key(content, "Attn floor") or "?"
        alpha = extract_key(content, "Alpha") or ""
        name = f"{mode}, floor={floor_val}"
        if alpha and "additive" in mode.lower():
            name += f", α={alpha}"

    elif exp_type == "ttam_v1":
        alpha = extract_key(content, "Alpha") or "?"
        inject = extract_key(content, "Inject layer") or "?"
        teacher = extract_key(content, "Teacher") or "?"
        name = f"residual α={alpha}, inj=L{inject}, T={teacher}"

    elif exp_type == "song":
        rho = extract_key(content, "Rho") or "?"
        lam = extract_key(content, "Lambda") or "?"
        pre = extract_key(content, "Pre layer") or "?"
        post = extract_key(content, "Post layer") or "?"
        review = extract_key(content, "Review layer") or "?"
        name = f"Song ρ={rho}, λ={lam}, |L{post}-L{pre}|, rev=L{review}"

    elif exp_type == "logit_bias":
        beta = extract_key(content, "Beta") or "?"
        layers = extract_key(content, "Bias layers") or "L10-15"
        teacher = extract_key(content, "Teacher") or "?"
        name = f"logit bias β={beta}, {layers}, T={teacher}"

    elif exp_type == "attn_gate":
        layers = extract_key(content, "Gate layers") or "L10-15"
        teacher = extract_key(content, "Teacher") or "?"
        name = f"attn gate {layers}, T={teacher}"

    elif exp_type == "vit_bias":
        beta = extract_key(content, "Beta") or "?"
        teacher = extract_key(content, "Teacher") or "?"
        name = f"ViT attn bias β={beta}, T={teacher}"

    elif exp_type == "text_crossattn":
        ckpt = extract_key(content, "Checkpoint") or "?"
        stage = extract_key(content, "Stage") or "?"
        hl = extract_key(content, "Hidden layer") or "?"
        ckpt_match = re.search(r"checkpoint-(\d+)", ckpt)
        ckpt_step = ckpt_match.group(1) if ckpt_match else "final"
        name = f"text-crossattn ckpt{ckpt_step} s{stage} hl={hl}"

    elif exp_type == "reasonvla":
        ckpt = extract_key(content, "Checkpoint") or "?"
        stage = extract_key(content, "Stage") or "?"
        hl = extract_key(content, "Hidden layer") or "?"
        ckpt_match = re.search(r"checkpoint-(\d+)", ckpt)
        ckpt_step = ckpt_match.group(1) if ckpt_match else "final"
        name = f"ReasonVLA ckpt{ckpt_step} s{stage} hl={hl}"

    else:
        name = f"unknown ({job_id})"

    return {
        "name": name,
        "job_id": job_id,
        "type": exp_type,
        "task_rates": task_pcts,
        "avg": avg,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_attn_eval_scores.py <log_file.out> [...]")
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

    # Group by type
    vision = [r for r in results if r["type"] in ("attn_mask", "vit_bias")]
    language = [r for r in results if r["type"] in ("ttam_v1", "song", "logit_bias", "attn_gate")]
    other = [r for r in results if r["type"] in ("text_crossattn", "reasonvla", "unknown")]

    def print_group(title, group):
        if not group:
            return
        print(f"\n{'=' * 130}")
        print(f"  {title}")
        print(f"{'=' * 130}")
        print(f"{'Experiment':<50} {'Job':<9} {'T0':<5} {'T1':<5} {'T2':<5} {'T3':<5} {'T4':<5} {'T5':<5} {'T6':<5} {'T7':<5} {'T8':<5} {'T9':<5} {'Avg':<6}")
        print("-" * 130)
        for r in group:
            tasks = r["task_rates"]
            task_strs = [f"{t}%" for t in tasks[:10]]
            while len(task_strs) < 10:
                task_strs.append("-")
            print(f"{r['name']:<50} {r['job_id']:<9} {task_strs[0]:<5} {task_strs[1]:<5} {task_strs[2]:<5} {task_strs[3]:<5} {task_strs[4]:<5} {task_strs[5]:<5} {task_strs[6]:<5} {task_strs[7]:<5} {task_strs[8]:<5} {task_strs[9]:<5} {r['avg']:.1f}%")

    print_group("VISION-SIDE", vision)
    print_group("LANGUAGE-SIDE", language)
    print_group("OTHER", other)


if __name__ == "__main__":
    main()
