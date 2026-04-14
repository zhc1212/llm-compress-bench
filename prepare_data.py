#!/usr/bin/env python3
"""Prepare mixed training/calibration data for LLM compression benchmarking.

Two modes:
  finetune    — JSONL with chat-formatted data from 6 sources
  calibration — tokenized tensor (.pt) from 11 sources for SVD calibration

Usage:
    # Finetune data
    python prepare_data.py --mode finetune --output data/finetune.jsonl \
        [--config configs/default_mix.yaml] [--seed 42] [--total N]

    # Calibration data
    python prepare_data.py --mode calibration --tokenizer Qwen/Qwen3-14B \
        --output data/calibration.pt [--config configs/calibration_mix.yaml] \
        [--nsamples 4096] [--seqlen 2048] [--seed 42]
"""

from __future__ import annotations

import os

# Default to HF mirror for faster downloads in China; override with HF_ENDPOINT env var
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Cache datasets/models under ~/datasets/ by default; override with HF_HOME env var
os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), "datasets"))
# Default local dataset directory; override with RAW_DATASET_DIR env var
os.environ.setdefault("RAW_DATASET_DIR", os.path.join(os.path.expanduser("~"), "datasets"))

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from datasets import get_dataset_config_names, load_dataset

# ---------------------------------------------------------------------------
# YAML config loading with built-in defaults
# ---------------------------------------------------------------------------

_DEFAULT_FINETUNE_SOURCES: dict[str, dict[str, Any]] = {
    "sharegpt": {
        "hf_id": "shareAI/ShareGPT-Chinese-English-90k",
        "ratio": 0.40,
        "format": "chat",
        "data_files": [
            "sharegpt_jsonl/common_zh_70k.jsonl",
            "sharegpt_jsonl/common_en_70k.jsonl",
        ],
    },
    "mmlu": {
        "hf_id": "cais/mmlu",
        "config": "auxiliary_train",
        "split": "train",
        "ratio": 0.25,
        "format": "mcq_en",
    },
    "gsm8k": {
        "hf_id": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "ratio": 0.20,
        "format": "qa",
    },
    "mbpp": {
        "hf_id": "google-research-datasets/mbpp",
        "config": "full",
        "split": "train",
        "ratio": 0.05,
        "format": "code",
    },
    "cmmlu": {
        "hf_id": "haonan-li/cmmlu",
        "split": "dev",
        "ratio": 0.05,
        "format": "mcq_zh",
    },
    "ceval": {
        "hf_id": "ceval/ceval-exam",
        "split": "val",
        "ratio": 0.05,
        "format": "mcq_zh_ceval",
    },
}

_DEFAULT_CALIBRATION_SOURCES: dict[str, dict[str, Any]] = {
    "sharegpt_chat": {
        "hf_id": "shareAI/ShareGPT-Chinese-English-90k",
        "ratio": 0.20,
        "format": "chat",
        "data_files": [
            "sharegpt_jsonl/common_zh_70k.jsonl",
            "sharegpt_jsonl/common_en_70k.jsonl",
        ],
    },
    "zh_wikipedia": {
        "hf_id": "wikimedia/wikipedia",
        "config": "20231101.zh",
        "split": "train",
        "ratio": 0.15,
        "format": "text",
        "min_chars": 500,
    },
    "en_wikipedia": {
        "hf_id": "wikimedia/wikipedia",
        "config": "20231101.en",
        "split": "train",
        "ratio": 0.15,
        "format": "text",
        "min_chars": 500,
    },
    "en_fineweb_edu": {
        "hf_id": "HuggingFaceFW/fineweb-edu",
        "split": "train",
        "ratio": 0.10,
        "format": "text",
        "streaming": True,
    },
    "zh_fineweb_edu": {
        "hf_id": "opencsg/Fineweb-Edu-Chinese-V2.1",
        "split": "train",
        "ratio": 0.08,
        "format": "text",
        "streaming": True,
    },
    "skypile": {
        "hf_id": "Skywork/SkyPile-150B",
        "split": "train",
        "ratio": 0.07,
        "format": "text",
        "streaming": True,
        "min_chars": 500,
    },
    "mmlu_train": {
        "hf_id": "cais/mmlu",
        "config": "auxiliary_train",
        "split": "train",
        "ratio": 0.07,
        "format": "mcq_en",
    },
    "ceval_train": {
        "hf_id": "ceval/ceval-exam",
        "split": "val",
        "ratio": 0.05,
        "format": "mcq_zh_ceval",
    },
    "gsm8k_train": {
        "hf_id": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "ratio": 0.05,
        "format": "qa",
    },
    "arc_train": {
        "hf_id": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "split": "train",
        "ratio": 0.04,
        "format": "arc",
    },
    "mbpp_train": {
        "hf_id": "google-research-datasets/mbpp",
        "config": "full",
        "split": "train",
        "ratio": 0.04,
        "format": "code",
    },
}

_ABCD = ["A", "B", "C", "D"]


def load_config(config_path: str | None, mode: str) -> dict[str, dict[str, Any]]:
    """Load source definitions from YAML config, or return built-in defaults."""
    defaults = (
        _DEFAULT_FINETUNE_SOURCES if mode == "finetune"
        else _DEFAULT_CALIBRATION_SOURCES
    )
    if config_path is None:
        return {k: dict(v) for k, v in defaults.items()}

    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    sources = cfg.get("sources", {})
    if not sources:
        raise ValueError(f"Config {config_path} has no 'sources' key")
    return sources


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_ROLE_MAP = {"human": "user", "gpt": "assistant", "assistant": "assistant", "system": "system"}


def _coalesce_turns(turns: list[dict]) -> list[dict]:
    """Merge adjacent same-role turns."""
    if not turns:
        return turns
    merged: list[dict] = [turns[0].copy()]
    for t in turns[1:]:
        if t["role"] == merged[-1]["role"]:
            merged[-1]["content"] += "\n" + t["content"]
        else:
            merged.append(t.copy())
    return merged


def _parse_sharegpt_conversation(raw_convs: list) -> list[dict] | None:
    """Parse ShareGPT conversation into normalised turns.

    Handles two formats:
      A) [{"role": "human", "content": "..."}, ...]
      B) [{"human": "...", "assistant": "..."}, ...]  (Firefly)
    """
    turns: list[dict] = []
    for item in raw_convs:
        if "role" in item and "content" in item:
            role = _ROLE_MAP.get(item["role"])
            if role is None:
                continue
            content = item["content"]
            if isinstance(content, str) and content.strip():
                turns.append({"role": role, "content": content.strip()})
        elif "human" in item and "assistant" in item:
            h, a = item["human"], item["assistant"]
            if isinstance(h, str) and h.strip():
                turns.append({"role": "user", "content": h.strip()})
            if isinstance(a, str) and a.strip():
                turns.append({"role": "assistant", "content": a.strip()})
    turns = _coalesce_turns(turns)
    return turns if len(turns) >= 2 else None


def _tokenize_and_window(
    text: str, tokenizer, seqlen: int, rng: random.Random,
    add_special_tokens: bool = False,
) -> torch.LongTensor | None:
    """Tokenize text, take one random contiguous window of seqlen tokens."""
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids[0]
    if ids.numel() < seqlen:
        return None
    start = rng.randint(0, ids.numel() - seqlen)
    return ids[start : start + seqlen].unsqueeze(0)


# ---------------------------------------------------------------------------
# Finetune mode: dataset loaders
# ---------------------------------------------------------------------------

def _load_sharegpt_from_local(local_path: str | Path) -> list[dict]:
    """Load ShareGPT chat data from local JSONL files."""
    local_path = Path(local_path)
    results: list[dict] = []
    skipped = 0

    # Handle directory with config subdirectories
    if local_path.is_dir():
        jsonl_files = list(local_path.glob("**/*.jsonl"))
        for jsonl_file in jsonl_files:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        convs = row.get("conversations") or row.get("conversation")
                        if not convs:
                            skipped += 1
                            continue
                        turns = _parse_sharegpt_conversation(convs)
                        if turns is None:
                            skipped += 1
                            continue
                        results.append({"source": "sharegpt", "conversations": turns})
                    except Exception:
                        skipped += 1
                        continue
    # Handle single JSONL file
    elif local_path.is_file():
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    convs = row.get("conversations") or row.get("conversation")
                    if not convs:
                        skipped += 1
                        continue
                    turns = _parse_sharegpt_conversation(convs)
                    if turns is None:
                        skipped += 1
                        continue
                    results.append({"source": "sharegpt", "conversations": turns})
                except Exception:
                    skipped += 1
                    continue

    print(f"  [sharegpt] {len(results)} conversations ({skipped} skipped)")
    return results


def _load_finetune_sharegpt(spec: dict) -> list[dict]:
    """Load ShareGPT chat data as conversation dicts."""
    print("  [sharegpt] Loading ShareGPT Chinese+English chat data...")

    # Check for local path
    local_path = spec.get("local_path")
    if not local_path:
        raw_dataset_dir = os.environ.get("RAW_DATASET_DIR")
        if raw_dataset_dir:
            dataset_name = spec["hf_id"].split("/")[-1]
            local_path = Path(raw_dataset_dir) / dataset_name

    if local_path and Path(local_path).exists():
        print(f"  [sharegpt] Loading from local: {local_path}")
        return _load_sharegpt_from_local(local_path)

    # Download from Hugging Face
    kwargs: dict[str, Any] = {"split": "train"}
    if "data_files" in spec:
        kwargs["data_files"] = spec["data_files"]
    ds = load_dataset(spec["hf_id"], **kwargs)

    results: list[dict] = []
    skipped = 0
    for row in ds:
        convs = row.get("conversations") or row.get("conversation")
        if not convs:
            skipped += 1
            continue
        turns = _parse_sharegpt_conversation(convs)
        if turns is None:
            skipped += 1
            continue
        results.append({"source": "sharegpt", "conversations": turns})
    print(f"  [sharegpt] {len(results)} conversations ({skipped} skipped)")
    return results


def _load_jsonl_from_local(local_path: str | Path) -> list[dict]:
    """Load data from local JSONL file."""
    local_path = Path(local_path)
    results = []

    if local_path.is_dir():
        jsonl_files = list(local_path.glob("**/*.jsonl"))
    else:
        jsonl_files = [local_path]

    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except Exception:
                        pass
    return results


def _load_finetune_mcq_en(name: str, spec: dict) -> list[dict]:
    """Load English MCQ dataset (MMLU format)."""
    print(f"  [{name}] Loading {spec['hf_id']}...")

    # Check for local path
    local_path = spec.get("local_path")
    if not local_path:
        raw_dataset_dir = os.environ.get("RAW_DATASET_DIR")
        if raw_dataset_dir:
            dataset_name = spec["hf_id"].split("/")[-1]
            local_path = Path(raw_dataset_dir) / dataset_name

    if local_path and Path(local_path).exists():
        print(f"  [{name}] Loading from local: {local_path}")
        all_rows = _load_jsonl_from_local(local_path)
        results = []
        for row in all_rows:
            choices = row["choices"]
            q = (f"Question: {row['question']}\n"
                 f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}")
            a = f"The answer is {_ABCD[row['answer']]}."
            results.append({
                "source": name,
                "conversations": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
            })
        print(f"  [{name}] {len(results)} samples")
        return results

    # Download from Hugging Face
    kwargs: dict[str, Any] = {"split": spec.get("split", "train"), "trust_remote_code": True}
    if "config" in spec:
        ds = load_dataset(spec["hf_id"], spec["config"], **kwargs)
    else:
        ds = load_dataset(spec["hf_id"], **kwargs)

    results: list[dict] = []
    for row in ds:
        choices = row["choices"]
        q = (f"Question: {row['question']}\n"
             f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}")
        a = f"The answer is {_ABCD[row['answer']]}."
        results.append({
            "source": name,
            "conversations": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
        })
    print(f"  [{name}] {len(results)} samples")
    return results


def _load_finetune_qa(name: str, spec: dict) -> list[dict]:
    """Load QA dataset (GSM8K format)."""
    print(f"  [{name}] Loading {spec['hf_id']}...")

    # Check for local path
    local_path = spec.get("local_path")
    if not local_path:
        raw_dataset_dir = os.environ.get("RAW_DATASET_DIR")
        if raw_dataset_dir:
            dataset_name = spec["hf_id"].split("/")[-1]
            local_path = Path(raw_dataset_dir) / dataset_name

    if local_path and Path(local_path).exists():
        print(f"  [{name}] Loading from local: {local_path}")
        all_rows = _load_jsonl_from_local(local_path)
        results = []
        for row in all_rows:
            results.append({
                "source": name,
                "conversations": [
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row["answer"]},
                ],
            })
        print(f"  [{name}] {len(results)} samples")
        return results

    # Download from Hugging Face
    kwargs: dict[str, Any] = {"split": spec.get("split", "train"), "trust_remote_code": True}
    if "config" in spec:
        ds = load_dataset(spec["hf_id"], spec["config"], **kwargs)
    else:
        ds = load_dataset(spec["hf_id"], **kwargs)

    results: list[dict] = []
    for row in ds:
        results.append({
            "source": name,
            "conversations": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ],
        })
    print(f"  [{name}] {len(results)} samples")
    return results


def _load_finetune_code(name: str, spec: dict) -> list[dict]:
    """Load code dataset (MBPP format)."""
    print(f"  [{name}] Loading {spec['hf_id']}...")

    # Check for local path
    local_path = spec.get("local_path")
    if not local_path:
        raw_dataset_dir = os.environ.get("RAW_DATASET_DIR")
        if raw_dataset_dir:
            dataset_name = spec["hf_id"].split("/")[-1]
            local_path = Path(raw_dataset_dir) / dataset_name

    if local_path and Path(local_path).exists():
        print(f"  [{name}] Loading from local: {local_path}")
        all_rows = _load_jsonl_from_local(local_path)
        results = []
        for row in all_rows:
            test_str = "\n".join(row["test_list"])
            q = f"Write a Python function for the following task:\n{row['text']}\n\nTest cases:\n{test_str}"
            a = f"```python\n{row['code']}\n```"
            results.append({
                "source": name,
                "conversations": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
            })
        print(f"  [{name}] {len(results)} samples")
        return results

    # Download from Hugging Face
    kwargs: dict[str, Any] = {"split": spec.get("split", "train"), "trust_remote_code": True}
    if "config" in spec:
        ds = load_dataset(spec["hf_id"], spec["config"], **kwargs)
    else:
        ds = load_dataset(spec["hf_id"], **kwargs)

    results: list[dict] = []
    for row in ds:
        test_str = "\n".join(row["test_list"])
        q = f"Write a Python function for the following task:\n{row['text']}\n\nTest cases:\n{test_str}"
        a = f"```python\n{row['code']}\n```"
        results.append({
            "source": name,
            "conversations": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
        })
    print(f"  [{name}] {len(results)} samples")
    return results


def _load_finetune_mcq_zh(name: str, spec: dict) -> list[dict]:
    """Load Chinese MCQ dataset (CMMLU format)."""
    print(f"  [{name}] Loading {spec['hf_id']} (all subjects)...")

    # Check for local path
    local_path = spec.get("local_path")
    if not local_path:
        raw_dataset_dir = os.environ.get("RAW_DATASET_DIR")
        if raw_dataset_dir:
            dataset_name = spec["hf_id"].split("/")[-1]
            local_path = Path(raw_dataset_dir) / dataset_name

    if local_path and Path(local_path).exists():
        print(f"  [{name}] Loading from local: {local_path}")
        all_rows = _load_jsonl_from_local(local_path)
        results = []
        for row in all_rows:
            q_text = row.get("Question") or row.get("question", "")
            q = f"问题：{q_text}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"
            ans = row.get("Answer") or row.get("answer", "")
            a = f"答案是 {ans}。"
            results.append({
                "source": name,
                "conversations": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
            })
        print(f"  [{name}] {len(results)} samples")
        return results

    # Download from Hugging Face
    try:
        configs = get_dataset_config_names(spec["hf_id"])
    except Exception:
        configs = []

    results: list[dict] = []
    split = spec.get("split", "dev")
    for cfg in configs:
        try:
            ds = load_dataset(spec["hf_id"], cfg, split=split, trust_remote_code=True)
        except Exception:
            continue
        for row in ds:
            # CMMLU uses 'Question', ceval uses 'question'
            q_text = row.get("Question") or row.get("question", "")
            q = f"问题：{q_text}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"
            ans = row.get("Answer") or row.get("answer", "")
            a = f"答案是 {ans}。"
            results.append({
                "source": name,
                "conversations": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
            })

    # CMMLU fallback: try loading from zip
    if not results and "cmmlu" in name.lower():
        try:
            from huggingface_hub import hf_hub_download
            import csv
            import io
            import zipfile

            print(f"  [{name}] Trying zip fallback...")
            local_zip = hf_hub_download(spec["hf_id"], "cmmlu_v1_0_1.zip", repo_type="dataset")
            with zipfile.ZipFile(local_zip) as z:
                dev_csvs = [n for n in z.namelist() if n.startswith("dev/") and n.endswith(".csv")]
                for csv_name in dev_csvs:
                    with z.open(csv_name) as f:
                        reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
                        for row in reader:
                            if len(row) < 6 or row[5] not in "ABCD":
                                continue
                            q = f"问题：{row[0]}\nA. {row[1]}\nB. {row[2]}\nC. {row[3]}\nD. {row[4]}"
                            results.append({
                                "source": name,
                                "conversations": [
                                    {"role": "user", "content": q},
                                    {"role": "assistant", "content": f"答案是 {row[5]}。"},
                                ],
                            })
        except Exception as e:
            print(f"  [{name}] Zip fallback failed: {e}")

    print(f"  [{name}] {len(results)} samples")
    return results


def _load_finetune_mcq_zh_ceval(name: str, spec: dict) -> list[dict]:
    """Load C-Eval dataset (separate handler due to different field names)."""
    print(f"  [{name}] Loading {spec['hf_id']} (all subjects)...")

    # Check for local path
    local_path = spec.get("local_path")
    if not local_path:
        raw_dataset_dir = os.environ.get("RAW_DATASET_DIR")
        if raw_dataset_dir:
            dataset_name = spec["hf_id"].split("/")[-1]
            local_path = Path(raw_dataset_dir) / dataset_name

    if local_path and Path(local_path).exists():
        print(f"  [{name}] Loading from local: {local_path}")
        all_rows = _load_jsonl_from_local(local_path)
        results = []
        for row in all_rows:
            q = (f"问题：{row['question']}\n"
                 f"A. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}")
            a = f"答案是 {row['answer']}。"
            results.append({
                "source": name,
                "conversations": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
            })
        print(f"  [{name}] {len(results)} samples")
        return results

    # Download from Hugging Face
    try:
        configs = get_dataset_config_names(spec["hf_id"])
    except Exception:
        configs = []

    results: list[dict] = []
    split = spec.get("split", "val")
    for cfg in configs:
        try:
            ds = load_dataset(spec["hf_id"], cfg, split=split, trust_remote_code=True)
        except Exception:
            continue
        for row in ds:
            q = (f"问题：{row['question']}\n"
                 f"A. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}")
            a = f"答案是 {row['answer']}。"
            results.append({
                "source": name,
                "conversations": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
            })
    print(f"  [{name}] {len(results)} samples ({len(configs)} subjects)")
    return results


_FINETUNE_FORMAT_LOADERS = {
    "chat": lambda name, spec: _load_finetune_sharegpt(spec),
    "mcq_en": _load_finetune_mcq_en,
    "qa": _load_finetune_qa,
    "code": _load_finetune_code,
    "mcq_zh": _load_finetune_mcq_zh,
    "mcq_zh_ceval": _load_finetune_mcq_zh_ceval,
}


def run_finetune(sources: dict[str, dict], output: str, seed: int, total: int | None) -> None:
    """Load, mix, and write finetune JSONL."""
    rng = random.Random(seed)

    # Load all source pools
    pools: dict[str, list[dict]] = {}
    for name, spec in sources.items():
        fmt = spec.get("format", "qa")
        loader = _FINETUNE_FORMAT_LOADERS.get(fmt)
        if loader is None:
            print(f"  WARNING: unknown format '{fmt}' for source '{name}', skipping")
            pools[name] = []
            continue
        try:
            pools[name] = loader(name, spec)
        except Exception as e:
            print(f"  ERROR loading {name}: {e}")
            pools[name] = []

    # Determine total sample count
    ratios = {name: spec.get("ratio", 0) for name, spec in sources.items()}
    if total is None:
        # Auto-calculate: find the source with the most data relative to its ratio
        # so no source needs excessive oversampling
        max_ratio_source = max(ratios, key=lambda n: ratios[n]) if ratios else None
        if max_ratio_source and ratios[max_ratio_source] > 0:
            total = int(len(pools.get(max_ratio_source, [])) / ratios[max_ratio_source])
        else:
            total = sum(len(v) for v in pools.values())
        print(f"\nAuto total = {total}")
    else:
        print(f"\nOverridden total = {total}")

    # Mix by ratio
    mixed: list[dict] = []
    print("\nMixing:")
    for name, spec in sources.items():
        ratio = spec.get("ratio", 0)
        pool = pools.get(name, [])
        target = int(total * ratio)
        if not pool:
            print(f"  {name}: 0 raw -> skip (target {target})")
            continue
        if len(pool) >= target:
            sampled = rng.sample(pool, k=target)
            mode = "subsampled"
        else:
            sampled = rng.choices(pool, k=target)
            mode = "oversampled"
        mixed.extend(sampled)
        print(f"  {name}: {len(pool)} raw -> {len(sampled)} ({mode})")

    rng.shuffle(mixed)
    print(f"\nTotal mixed: {len(mixed)}")

    # Write JSONL
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in mixed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Written {len(mixed)} samples to {out_path}")


# ---------------------------------------------------------------------------
# Calibration mode: tokenize + random window
# ---------------------------------------------------------------------------

def _collect_from_streaming(
    dataset_iter, tokenizer, seqlen: int, rng: random.Random,
    target: int, text_field: str = "text", min_chars: int = 0,
    source_name: str = "",
) -> list[torch.LongTensor]:
    """Collect tokenized windows from a streaming dataset iterator."""
    samples: list[torch.LongTensor] = []
    examined = 0
    for row in dataset_iter:
        text = row.get(text_field, "")
        if not isinstance(text, str) or len(text) < min_chars:
            continue
        examined += 1
        window = _tokenize_and_window(text, tokenizer, seqlen, rng)
        if window is not None:
            samples.append(window)
            if len(samples) % 100 == 0:
                print(f"    [{source_name}] {len(samples)}/{target} (examined {examined})")
            if len(samples) >= target:
                break
    return samples


def _load_calib_chat(
    name: str, spec: dict, tokenizer, seqlen: int, rng: random.Random, target: int,
) -> list[torch.LongTensor]:
    """Load ShareGPT chat data as tokenized windows."""
    print(f"  [{name}] Loading chat data...")

    # Check for local path
    local_path = spec.get("local_path")
    if not local_path:
        raw_dataset_dir = os.environ.get("RAW_DATASET_DIR")
        if raw_dataset_dir:
            dataset_name = spec["hf_id"].split("/")[-1]
            local_path = Path(raw_dataset_dir) / dataset_name

    if local_path and Path(local_path).exists():
        print(f"  [{name}] Loading from local: {local_path}")
        samples: list[torch.LongTensor] = []
        skipped = 0

        local_path = Path(local_path)
        if local_path.is_dir():
            jsonl_files = list(local_path.glob("**/*.jsonl"))
        else:
            jsonl_files = [local_path]

        for jsonl_file in jsonl_files:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        skipped += 1
                        continue

                    convs = row.get("conversations") or row.get("conversation")
                    if not convs:
                        skipped += 1
                        continue
                    turns = _parse_sharegpt_conversation(convs)
                    if turns is None:
                        skipped += 1
                        continue
                    try:
                        templated = tokenizer.apply_chat_template(
                            turns, tokenize=False, add_generation_prompt=False,
                        )
                    except Exception:
                        skipped += 1
                        continue
                    window = _tokenize_and_window(templated, tokenizer, seqlen, rng)
                    if window is not None:
                        samples.append(window)
                        if len(samples) % 100 == 0:
                            print(f"    [{name}] {len(samples)}/{target}")
                        if len(samples) >= target:
                            break

        print(f"  [{name}] {len(samples)}/{target} (skipped {skipped})")
        return samples

    # Download from Hugging Face
    kwargs: dict[str, Any] = {"split": "train"}
    if "data_files" in spec:
        kwargs["data_files"] = spec["data_files"]
    ds = load_dataset(spec["hf_id"], **kwargs)

    samples: list[torch.LongTensor] = []
    skipped = 0
    for row in ds:
        convs = row.get("conversations") or row.get("conversation")
        if not convs:
            skipped += 1
            continue
        turns = _parse_sharegpt_conversation(convs)
        if turns is None:
            skipped += 1
            continue
        try:
            templated = tokenizer.apply_chat_template(
                turns, tokenize=False, add_generation_prompt=False,
            )
        except Exception:
            skipped += 1
            continue
        window = _tokenize_and_window(templated, tokenizer, seqlen, rng)
        if window is not None:
            samples.append(window)
            if len(samples) % 100 == 0:
                print(f"    [{name}] {len(samples)}/{target}")
            if len(samples) >= target:
                break
    print(f"  [{name}] {len(samples)}/{target} (skipped {skipped})")
    return samples


def _load_calib_text(
    name: str, spec: dict, tokenizer, seqlen: int, rng: random.Random, target: int,
) -> list[torch.LongTensor]:
    """Load plain-text dataset as tokenized windows."""
    print(f"  [{name}] Loading text data from {spec['hf_id']}...")

    # Check for local path
    local_path = spec.get("local_path")
    if not local_path:
        raw_dataset_dir = os.environ.get("RAW_DATASET_DIR")
        if raw_dataset_dir:
            dataset_name = spec["hf_id"].split("/")[-1]
            local_path = Path(raw_dataset_dir) / dataset_name

    if local_path and Path(local_path).exists():
        print(f"  [{name}] Loading from local: {local_path}")
        min_chars = spec.get("min_chars", 0)
        text_field = spec.get("text_field", "text")

        local_path = Path(local_path)
        if local_path.is_dir():
            jsonl_files = list(local_path.glob("**/*.jsonl"))
        else:
            jsonl_files = [local_path]

        all_rows = []
        for jsonl_file in jsonl_files:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            all_rows.append(json.loads(line))
                        except Exception:
                            pass

        indices = list(range(len(all_rows)))
        rng.shuffle(indices)
        samples: list[torch.LongTensor] = []
        for idx in indices:
            text = all_rows[idx].get(text_field, "")
            if not isinstance(text, str) or len(text) < min_chars:
                continue
            window = _tokenize_and_window(text, tokenizer, seqlen, rng)
            if window is not None:
                samples.append(window)
                if len(samples) % 100 == 0:
                    print(f"    [{name}] {len(samples)}/{target}")
                if len(samples) >= target:
                    break

        print(f"  [{name}] {len(samples)}/{target}")
        return samples

    # Download from Hugging Face
    streaming = spec.get("streaming", False)
    min_chars = spec.get("min_chars", 0)
    text_field = spec.get("text_field", "text")

    load_kwargs: dict[str, Any] = {"split": spec.get("split", "train")}
    if "config" in spec:
        load_kwargs["name"] = spec["config"]
    if streaming:
        load_kwargs["streaming"] = True

    ds = load_dataset(spec["hf_id"], **load_kwargs)

    if streaming:
        samples = _collect_from_streaming(
            iter(ds), tokenizer, seqlen, rng, target,
            text_field=text_field, min_chars=min_chars, source_name=name,
        )
    else:
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        samples: list[torch.LongTensor] = []
        for idx in indices:
            text = ds[idx].get(text_field, "")
            if not isinstance(text, str) or len(text) < min_chars:
                continue
            window = _tokenize_and_window(text, tokenizer, seqlen, rng)
            if window is not None:
                samples.append(window)
                if len(samples) % 100 == 0:
                    print(f"    [{name}] {len(samples)}/{target}")
                if len(samples) >= target:
                    break

    print(f"  [{name}] {len(samples)}/{target}")
    return samples


def _load_calib_benchmark(
    name: str, spec: dict, tokenizer, seqlen: int, rng: random.Random, target: int,
) -> list[torch.LongTensor]:
    """Load benchmark QA/code data as tokenized windows via chat template."""
    fmt = spec.get("format", "qa")
    print(f"  [{name}] Loading benchmark data ({fmt})...")

    # Check for local path
    local_path = spec.get("local_path")
    if not local_path:
        raw_dataset_dir = os.environ.get("RAW_DATASET_DIR")
        if raw_dataset_dir:
            dataset_name = spec["hf_id"].split("/")[-1]
            local_path = Path(raw_dataset_dir) / dataset_name

    if local_path and Path(local_path).exists():
        print(f"  [{name}] Loading from local: {local_path}")

        local_path = Path(local_path)
        if local_path.is_dir():
            jsonl_files = list(local_path.glob("**/*.jsonl"))
        else:
            jsonl_files = [local_path]

        all_rows = []
        for jsonl_file in jsonl_files:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            all_rows.append(json.loads(line))
                        except Exception:
                            pass

        rng.shuffle(all_rows)
    else:
        # Load dataset from Hugging Face
        load_kwargs: dict[str, Any] = {"split": spec.get("split", "train"), "trust_remote_code": True}

        # Handle multi-config datasets (C-Eval)
        if fmt == "mcq_zh_ceval":
            try:
                configs = get_dataset_config_names(spec["hf_id"])
            except Exception:
                configs = []
            all_rows: list[dict] = []
            for cfg in configs:
                try:
                    ds = load_dataset(spec["hf_id"], cfg, **load_kwargs)
                    all_rows.extend(list(ds))
                except Exception:
                    continue
            rng.shuffle(all_rows)
        else:
            if "config" in spec:
                ds = load_dataset(spec["hf_id"], spec["config"], **load_kwargs)
            else:
                ds = load_dataset(spec["hf_id"], **load_kwargs)
            all_rows = list(ds)
            rng.shuffle(all_rows)

    # Format function based on type
    def _format_row(row: dict) -> list[dict] | None:
        if fmt == "mcq_en":
            choices = row["choices"]
            q = (f"Question: {row['question']}\n"
                 f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}")
            a = f"The answer is {_ABCD[row['answer']]}."
            return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        elif fmt == "mcq_zh_ceval":
            q = (f"问题：{row['question']}\n"
                 f"A. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}")
            a = f"答案是 {row['answer']}。"
            return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        elif fmt == "qa":
            return [{"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row["answer"]}]
        elif fmt == "arc":
            choices = row["choices"]
            labels, texts = choices["label"], choices["text"]
            opts = "\n".join(f"{lbl}. {t}" for lbl, t in zip(labels, texts))
            q = f"Question: {row['question']}\n{opts}"
            a = f"The answer is {row['answerKey']}."
            return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        elif fmt == "code":
            test_str = "\n".join(row["test_list"])
            q = f"Write a Python function:\n{row['text']}\n\nTest cases:\n{test_str}"
            a = f"```python\n{row['code']}\n```"
            return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        return None

    samples: list[torch.LongTensor] = []
    for row in all_rows:
        if len(samples) >= target:
            break
        turns = _format_row(row)
        if turns is None:
            continue
        try:
            text = tokenizer.apply_chat_template(
                turns, tokenize=False, add_generation_prompt=False,
            )
        except Exception:
            continue
        window = _tokenize_and_window(text, tokenizer, seqlen, rng)
        if window is not None:
            samples.append(window)

    print(f"  [{name}] {len(samples)}/{target}")
    return samples


def run_calibration(
    sources: dict[str, dict], output: str, tokenizer_name: str,
    nsamples: int, seqlen: int, seed: int,
) -> None:
    """Load, tokenize, and save calibration tensor."""
    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    rng = random.Random(seed)

    # Compute per-source targets
    source_list = list(sources.items())
    targets: dict[str, int] = {}
    for name, spec in source_list:
        targets[name] = int(nsamples * spec.get("ratio", 0))
    # Distribute rounding remainder to first source
    remainder = nsamples - sum(targets.values())
    if remainder > 0:
        targets[source_list[0][0]] += remainder

    print(f"\nCollecting {nsamples} calibration samples (seqlen={seqlen}):")
    for name, tgt in targets.items():
        print(f"  {name}: {tgt}")

    # Load each source with graceful fallback
    all_samples: list[torch.LongTensor] = []
    failed: list[str] = []
    succeeded: list[str] = []

    for name, spec in source_list:
        target = targets[name]
        fmt = spec.get("format", "text")
        try:
            if fmt == "chat":
                samples = _load_calib_chat(name, spec, tokenizer, seqlen, rng, target)
            elif fmt == "text":
                samples = _load_calib_text(name, spec, tokenizer, seqlen, rng, target)
            else:
                samples = _load_calib_benchmark(name, spec, tokenizer, seqlen, rng, target)

            all_samples.extend(samples)
            if samples:
                succeeded.append(name)
            else:
                print(f"  WARNING: [{name}] returned 0 samples")
                failed.append(name)
        except Exception as e:
            print(f"  WARNING: [{name}] failed: {e}")
            failed.append(name)

    # Redistribute failed quotas proportionally
    if failed and succeeded:
        failed_total = sum(targets[n] for n in failed)
        succ_ratios = {n: sources[n].get("ratio", 0) for n in succeeded}
        total_ratio = sum(succ_ratios.values())
        if total_ratio > 0:
            for name in succeeded:
                extra = int(failed_total * succ_ratios[name] / total_ratio)
                if extra <= 0:
                    continue
                print(f"  Redistributing: {extra} extra from [{name}]...")
                spec = sources[name]
                fmt = spec.get("format", "text")
                try:
                    if fmt == "chat":
                        extra_s = _load_calib_chat(name, spec, tokenizer, seqlen, rng, extra)
                    elif fmt == "text":
                        extra_s = _load_calib_text(name, spec, tokenizer, seqlen, rng, extra)
                    else:
                        extra_s = _load_calib_benchmark(name, spec, tokenizer, seqlen, rng, extra)
                    all_samples.extend(extra_s)
                    print(f"  [{name}] extra: {len(extra_s)}/{extra}")
                except Exception as e:
                    print(f"  WARNING: [{name}] extra load failed: {e}")

    # Shuffle and save
    rng.shuffle(all_samples)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_samples, str(out_path))
    print(f"\nCalibration data: {len(all_samples)} samples "
          f"(target {nsamples}, {len(failed)} sources failed)")
    print(f"Saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare mixed data for LLM compression benchmarking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", required=True, choices=["finetune", "calibration"],
                        help="finetune (JSONL) or calibration (tokenized .pt)")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--config", default=None,
                        help="YAML config with source definitions (optional)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Finetune-specific
    parser.add_argument("--total", type=int, default=None,
                        help="[finetune] Total samples (auto if omitted)")

    # Calibration-specific
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="[calibration] HF tokenizer name or path")
    parser.add_argument("--nsamples", type=int, default=4096,
                        help="[calibration] Number of calibration samples")
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="[calibration] Sequence length per sample")

    args = parser.parse_args()

    # Respect HF_ENDPOINT if set
    hf_endpoint = os.environ.get("HF_ENDPOINT")
    if hf_endpoint:
        print(f"Using HF_ENDPOINT={hf_endpoint}")

    sources = load_config(args.config, args.mode)
    print(f"Mode: {args.mode}")
    print(f"Sources: {list(sources.keys())}")
    print(f"Seed: {args.seed}")

    if args.mode == "finetune":
        run_finetune(sources, args.output, args.seed, args.total)
    else:
        if args.tokenizer is None:
            parser.error("--tokenizer is required for calibration mode")
        run_calibration(sources, args.output, args.tokenizer, args.nsamples, args.seqlen, args.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
