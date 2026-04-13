# llm-compress-bench

Standardized benchmark suite for evaluating LLM compression methods. Two scripts cover the full pipeline:

1. **`prepare_data.py`** — Prepare mixed training/calibration data from HuggingFace
2. **`run_benchmarks.sh`** — Run standardized benchmarks via lm-evaluation-harness

## Quick Start

```bash
git clone https://github.com/zhc1212/llm-compress-bench.git
cd llm-compress-bench
bash setup.sh

# Run all 10 benchmarks on a model
CUDA_VISIBLE_DEVICES=0 bash run_benchmarks.sh Qwen/Qwen3-14B base all

# Collect results into a table
python collect_results.py results/benchmarks/
```

## Data Preparation

### Fine-tuning Data (JSONL)

Generates chat-formatted JSONL from 6 sources for LoRA/SFT recovery after compression.

```bash
python prepare_data.py --mode finetune --output data/finetune.jsonl
```

Default mix (configurable via `--config configs/default_mix.yaml`):

| Source | HF Dataset | Ratio | Format |
|--------|-----------|-------|--------|
| ShareGPT | shareAI/ShareGPT-Chinese-English-90k | 40% | Multi-turn chat |
| MMLU | cais/mmlu (auxiliary_train) | 25% | MCQ → "The answer is X." |
| GSM8K | openai/gsm8k | 20% | Step-by-step math |
| MBPP | google-research-datasets/mbpp | 5% | Code generation |
| CMMLU | haonan-li/cmmlu | 5% | MCQ → "答案是 X。" |
| C-Eval | ceval/ceval-exam | 5% | MCQ → "答案是 X。" |

### Calibration Data (Tensors)

Generates tokenized tensors from 11 diverse sources for SVD/compression calibration.

```bash
python prepare_data.py --mode calibration \
    --tokenizer Qwen/Qwen3-14B \
    --nsamples 4096 --seqlen 2048 \
    --output data/calibration.pt
```

Default: 4096 samples × 2048 tokens = **~8.4M tokens**. Sources include Wikipedia (zh/en), Fineweb-Edu, SkyPile, ShareGPT, and benchmark training sets. See `configs/calibration_mix.yaml` for full breakdown.

## Benchmarks

### Standard Benchmarks (10)

```bash
# All benchmarks
bash run_benchmarks.sh <model_path> <tag> all

# Selected benchmarks
bash run_benchmarks.sh <model_path> <tag> "mmlu gsm8k ifeval"
```

| Benchmark | Task Type | Few-shot | Metric | Chat Template |
|-----------|-----------|----------|--------|---------------|
| MMLU | loglikelihood | 5 | acc | No |
| C-Eval | loglikelihood | 5 | acc | No |
| TruthfulQA | loglikelihood | 0 (mc2) | acc | No |
| ARC-C | loglikelihood | 25 | acc_norm | No |
| HellaSwag | loglikelihood | 10 | acc_norm | No |
| Winogrande | loglikelihood | 5 | acc | No |
| BBH | generate_until | 0 | exact_match | Yes |
| GSM8K | generate_until | 5 | flexible-extract | Yes |
| IFEval | generate_until | 0 | prompt_strict_acc | Yes |
| MBPP | generate_until | 3 | pass@1 | Yes |

**Protocol**: Generate tasks use `--apply_chat_template --system_instruction /no_think --log_samples --batch_size auto:4`.

### Thinking Mode (Optional)

For models with reasoning capability (e.g., Qwen3):

```bash
bash run_benchmarks.sh <model_path> <tag> "gsm8k_thinking math500_thinking" --thinking
```

Uses custom task YAMLs in `configs/tasks/` with `enable_thinking=true`, sampling `temperature=0.6, top_p=0.95, max_gen_toks=4096`.

### Chat Quality Benchmark

32 heuristic-scored prompts across 8 categories (no API keys needed):

```bash
python chat_bench.py --model_path <path> --tag "my model" --dtype bfloat16
```

Categories: instruction_following, format_compliance, chinese_fluency, math_reasoning, coding, safety, creativity, multi_turn.

## Collecting Results

```bash
# Markdown table (stdout)
python collect_results.py results/benchmarks/

# CSV output
python collect_results.py results/benchmarks/ --format csv --output comparison.csv

# Specific tags only
python collect_results.py results/benchmarks/ --tags "base compressed finetuned"
```

## Options

```bash
# Custom dtype and batch size
bash run_benchmarks.sh <model> <tag> all --dtype float16 --batch-size 8

# Custom lm_eval path
LM_EVAL=/path/to/lm_eval bash run_benchmarks.sh <model> <tag> all

# Custom system instruction
bash run_benchmarks.sh <model> <tag> all --system-instruction ""
```

## File Structure

```
llm-compress-bench/
├── prepare_data.py          # Data preparation (finetune JSONL / calibration tensors)
├── run_benchmarks.sh        # Standardized benchmark runner
├── chat_bench.py            # Chat quality evaluation (32 prompts, heuristic scoring)
├── collect_results.py       # Result aggregation → markdown/CSV tables
├── setup.sh                 # Dependency installation
├── configs/
│   ├── default_mix.yaml     # Fine-tuning data mix config
│   ├── calibration_mix.yaml # Calibration data mix config
│   └── tasks/               # Custom lm-eval task YAMLs
│       ├── gsm8k_thinking.yaml
│       └── math500_thinking.yaml
└── LICENSE
```

## Requirements

- Python 3.10+
- PyTorch with CUDA
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) v0.4.11
- HuggingFace `transformers` and `datasets`
