#!/usr/bin/env python3
"""Collect lm-evaluation-harness results into a comparison table.

Scans results/benchmarks/<tag>/ directories, extracts key metrics from
lm-eval JSON output, and produces a markdown or CSV comparison table.

Usage:
    python collect_results.py results/benchmarks/
    python collect_results.py results/benchmarks/ --format csv --output results/comparison.csv
    python collect_results.py results/benchmarks/ --tags "base compressed finetuned"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Benchmark → (lm-eval task key, metric key, display name)
BENCHMARKS = {
    "mmlu": ("mmlu", "acc,none", "MMLU (5-shot)"),
    "ceval": ("ceval-valid", "acc,none", "C-Eval (5-shot)"),
    "truthfulqa": ("truthfulqa_mc2", "acc,none", "TruthfulQA (mc2)"),
    "arc_challenge": ("arc_challenge", "acc_norm,none", "ARC-C (25-shot)"),
    "hellaswag": ("hellaswag", "acc_norm,none", "HellaSwag (10-shot)"),
    "winogrande": ("winogrande", "acc,none", "Winogrande (5-shot)"),
    "bbh": ("bbh_zeroshot", "exact_match,none", "BBH (0-shot)"),
    "gsm8k": ("gsm8k", "exact_match,flexible-extract", "GSM8K (5-shot)"),
    "ifeval": ("ifeval", "prompt_level_strict_acc,none", "IFEval"),
    "mbpp": ("mbpp", "pass@1,none", "MBPP (3-shot)"),
}


def find_result_json(bench_dir: Path) -> Path | None:
    """Find the lm-eval results JSON file in a benchmark output directory."""
    # lm-eval outputs to <output_path>/<model_name>/results_*.json
    for f in bench_dir.rglob("results_*.json"):
        return f
    return None


def extract_metric(result_json: Path, task_key: str, metric_key: str) -> float | None:
    """Extract a metric value from an lm-eval results JSON file."""
    with open(result_json) as f:
        data = json.load(f)

    results = data.get("results", {})

    # Direct task match
    if task_key in results:
        val = results[task_key].get(metric_key)
        if val is not None:
            return round(val * 100, 1)

    # For aggregate tasks (bbh, mmlu), look for the group result
    for key, metrics in results.items():
        if task_key in key:
            val = metrics.get(metric_key)
            if val is not None:
                return round(val * 100, 1)

    # Try without the ,none suffix
    base_metric = metric_key.split(",")[0]
    for key, metrics in results.items():
        if task_key in key:
            for mk, mv in metrics.items():
                if mk.startswith(base_metric) and isinstance(mv, (int, float)):
                    return round(mv * 100, 1)

    return None


def collect_tag(tag_dir: Path) -> dict[str, float | None]:
    """Collect all benchmark results for a single tag."""
    row: dict[str, float | None] = {}
    for bench_name, (task_key, metric_key, _) in BENCHMARKS.items():
        bench_dir = tag_dir / bench_name
        if not bench_dir.exists():
            row[bench_name] = None
            continue
        result_file = find_result_json(bench_dir)
        if result_file is None:
            row[bench_name] = None
            continue
        row[bench_name] = extract_metric(result_file, task_key, metric_key)
    return row


def format_markdown(tags: list[str], table: dict[str, dict]) -> str:
    """Format results as a markdown table."""
    lines = []
    header = "| Benchmark |" + "|".join(f" {t} " for t in tags) + "|"
    sep = "|-----------|" + "|".join("------:" for _ in tags) + "|"
    lines.append(header)
    lines.append(sep)

    for bench_name, (_, _, display) in BENCHMARKS.items():
        row = f"| {display} |"
        for tag in tags:
            val = table[tag].get(bench_name)
            row += f" {val if val is not None else '-'} |"
        lines.append(row)

    return "\n".join(lines)


def format_csv(tags: list[str], table: dict[str, dict]) -> str:
    """Format results as CSV."""
    lines = []
    lines.append("Benchmark," + ",".join(tags))

    for bench_name, (_, _, display) in BENCHMARKS.items():
        row = display
        for tag in tags:
            val = table[tag].get(bench_name)
            row += f",{val if val is not None else ''}"
        lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Collect lm-eval results into a comparison table"
    )
    parser.add_argument(
        "results_dir", type=str,
        help="Path to results/benchmarks/ directory"
    )
    parser.add_argument(
        "--tags", type=str, default=None,
        help="Space-separated list of tags to include (default: all subdirs)"
    )
    parser.add_argument(
        "--format", choices=["markdown", "csv"], default="markdown",
        help="Output format (default: markdown)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (default: stdout)"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Discover tags
    if args.tags:
        tags = args.tags.split()
    else:
        tags = sorted(
            d.name for d in results_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    if not tags:
        print("No result directories found.", file=sys.stderr)
        sys.exit(1)

    print(f"Collecting results for {len(tags)} tags: {', '.join(tags)}", file=sys.stderr)

    # Collect
    table: dict[str, dict] = {}
    for tag in tags:
        tag_dir = results_dir / tag
        if not tag_dir.exists():
            print(f"  Warning: {tag_dir} not found, skipping", file=sys.stderr)
            continue
        table[tag] = collect_tag(tag_dir)

    # Format
    if args.format == "csv":
        output = format_csv(list(table.keys()), table)
    else:
        output = format_markdown(list(table.keys()), table)

    # Output
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output + "\n")
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
