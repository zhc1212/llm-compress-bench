#!/usr/bin/env python3
"""Download datasets from Hugging Face Hub to specified locations.

Usage:
    # Download a simple dataset
    python download_dataset --dataset "allenai/ai2_arc" --output "data/arc"

    # Download with specific config and split
    python download_dataset --dataset "cais/mmlu" --config "auxiliary_train" --split "train" --output "data/mmlu"

    # Download with streaming support
    python download_dataset --dataset "HuggingFaceFW/fineweb-edu" --split "train" --output "data/fineweb" --streaming

    # Download multiple configs
    python download_dataset --dataset "ceval/ceval-exam" --split "val" --output "data/ceval" --multiple-configs
"""

#!/usr/bin/env python3
"""Download all datasets used in prepare_data.py to specified locations.

This script downloads all the datasets needed for both finetune and calibration modes.
The datasets are saved to organized directories for easy access.
"""

from __future__ import annotations

import os
import json
from pathlib import Path

# Default to HF mirror for faster downloads in China; override with HF_ENDPOINT env var
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Cache datasets/models under ~/datasets/ by default; override with HF_HOME env var
os.environ.setdefault("RAW_DATASET_DIR", os.path.join(os.path.expanduser("~"), "datasets"))

from datasets import get_dataset_config_names, load_dataset


def download_single_dataset(
    dataset_id: str,
    output_path: str,
    config: str | None = None,
    split: str = "train",
    streaming: bool = False,
    trust_remote_code: bool = False,
    multiple_configs: bool = False,
    data_files: list[str] | None = None,
) -> None:
    """Download a single dataset from Hugging Face Hub."""
    print(f"\nDownloading dataset: {dataset_id}")
    print(f"Output: {output_path}")
    print(f"Config: {config or 'default'}")
    print(f"Split: {split}")
    print(f"Streaming: {streaming}")
    print("-" * 50)

    # Create output directory
    output_dir = Path(output_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Handle multiple configs
    if multiple_configs:
        print(f"Available configs: {get_dataset_config_names(dataset_id)}")
        configs = get_dataset_config_names(dataset_id)
    else:
        configs = [config] if config else [None]

    for cfg in configs:
        if cfg:
            print(f"\nLoading config: {cfg}")
            dataset = load_dataset(
                dataset_id,
                name=cfg,
                split=split,
                streaming=streaming,
                trust_remote_code=trust_remote_code,
                data_files=data_files,
            )
        else:
            print(f"\nLoading default config")
            dataset = load_dataset(
                dataset_id,
                split=split,
                streaming=streaming,
                trust_remote_code=trust_remote_code,
                data_files=data_files,
            )

        # Convert streaming dataset to list if needed
        if streaming:
            # For streaming, save as shards
            config_name = cfg or "default"
            shard_dir = output_dir / config_name / split
            shard_dir.mkdir(parents=True, exist_ok=True)

            # Convert to list and save in chunks
            data_list = []
            shard_count = 0
            for i, item in enumerate(dataset):
                data_list.append(item)
                if len(data_list) >= 1000:  # Save every 1000 samples
                    shard_file = shard_dir / f"shard-{shard_count}.jsonl"
                    with open(shard_file, "w", encoding="utf-8") as f:
                        for data in data_list:
                            f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    shard_count += 1
                    data_list = []

            # Save remaining data
            if data_list:
                shard_file = shard_dir / f"shard-{shard_count}.jsonl"
                with open(shard_file, "w", encoding="utf-8") as f:
                    for data in data_list:
                        f.write(json.dumps(data, ensure_ascii=False) + "\n")

            print(f"Saved {shard_count + 1} shards to {shard_dir}")
        else:
            # Convert to list and save as JSONL
            data_list = list(dataset)
            config_name = cfg or "default"

            # Save as single JSONL file
            output_file = output_dir / config_name / f"{split}.jsonl"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                for item in data_list:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            print(f"Saved {len(data_list)} samples to {output_file}")

            # Save metadata
            metadata = {
                "dataset_id": dataset_id,
                "config": config_name,
                "split": split,
                "num_samples": len(data_list),
                "streaming": streaming,
                "trust_remote_code": trust_remote_code,
            }

            metadata_file = output_dir / config_name / "metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("✓ Download completed!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download all datasets for LLM compression benchmarking")
    parser.add_argument("--output-dir", type=str, default="datasets",
                        help="Output directory for downloaded datasets (default: datasets)")
    args = parser.parse_args()

    # Show HF endpoint info
    hf_endpoint = os.environ.get("HF_ENDPOINT")
    if hf_endpoint:
        print(f"Using HF_ENDPOINT={hf_endpoint}")

    # Show HF cache location
    hf_home = os.environ.get("HF_HOME")
    print(f"HF cache location: {hf_home}")

    print("Downloading all datasets for LLM compression benchmarking")
    print("=" * 70)

    # Base directory for all downloads
    base_dir = Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # List of all datasets to download
    datasets = [
        # Finetune datasets
        {
            "id": "shareAI/ShareGPT-Chinese-English-90k",
            "output": str(base_dir / "sharegpt"),
            "data_files": ["sharegpt_jsonl/common_zh_70k.jsonl", "sharegpt_jsonl/common_en_70k.jsonl"],
        },
        {
            "id": "cais/mmlu",
            "output": str(base_dir / "mmlu"),
            "config": "auxiliary_train",
        },
        {
            "id": "openai/gsm8k",
            "output": str(base_dir / "gsm8k"),
            "config": "main",
        },
        {
            "id": "google-research-datasets/mbpp",
            "output": str(base_dir / "mbpp"),
            "config": "full",
        },
        {
            "id": "haonan-li/cmmlu",
            "output": str(base_dir / "cmmlu"),
            "split": "dev",
        },
        {
            "id": "ceval/ceval-exam",
            "output": str(base_dir / "ceval"),
            "split": "val",
        },
        # Calibration datasets
        {
            "id": "shareAI/ShareGPT-Chinese-English-90k",
            "output": str(base_dir / "sharegpt_calib"),
            "data_files": ["sharegpt_jsonl/common_zh_70k.jsonl", "sharegpt_jsonl/common_en_70k.jsonl"],
        },
        {
            "id": "wikimedia/wikipedia",
            "output": str(base_dir / "wikipedia"),
            "multiple_configs": True,
        },
        {
            "id": "HuggingFaceFW/fineweb-edu",
            "output": str(base_dir / "fineweb_edu"),
            "streaming": True,
        },
        {
            "id": "opencsg/Fineweb-Edu-Chinese-V2.1",
            "output": str(base_dir / "fineweb_edu_chinese"),
            "streaming": True,
        },
        {
            "id": "Skywork/SkyPile-150B",
            "output": str(base_dir / "skypile"),
            "streaming": True,
        },
        {
            "id": "cais/mmlu",
            "output": str(base_dir / "mmlu_train"),
            "config": "auxiliary_train",
        },
        {
            "id": "ceval/ceval-exam",
            "output": str(base_dir / "ceval_train"),
            "split": "val",
        },
        {
            "id": "openai/gsm8k",
            "output": str(base_dir / "gsm8k_train"),
            "config": "main",
        },
        {
            "id": "allenai/ai2_arc",
            "output": str(base_dir / "arc"),
            "config": "ARC-Challenge",
        },
        {
            "id": "google-research-datasets/mbpp",
            "output": str(base_dir / "mbpp_train"),
            "config": "full",
        },
    ]

    # Download each dataset
    for ds in datasets:
        try:
            download_single_dataset(
                dataset_id=ds["id"],
                output_path=ds["output"],
                split=ds.get("split", "train"),
                config=ds.get("config"),
                streaming=ds.get("streaming", False),
                trust_remote_code=ds.get("trust_remote_code", False),
                multiple_configs=ds.get("multiple_configs", False),
                data_files=ds.get("data_files"),
            )
        except Exception as e:
            print(f"✗ Failed to download {ds['id']}: {e}")

    print("\n" + "=" * 70)
    print("All download tasks completed!")
    print(f"\nAll datasets are downloaded to: {base_dir.absolute()}")
    print("\nYou can now run:")
    print("python prepare_data.py --mode finetune --output data/finetune.jsonl --config configs/default_mix.yaml")
    print("python prepare_data.py --mode calibration --tokenizer Qwen/Qwen3-14B --output data/calibration.pt --config configs/calibration_mix.yaml")


if __name__ == "__main__":
    main()