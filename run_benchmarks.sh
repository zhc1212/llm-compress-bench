#!/bin/bash
# run_benchmarks.sh — Standardized LLM benchmarks via lm-evaluation-harness
# Usage: bash run_benchmarks.sh <model_path> <tag> <benchmarks> [options]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
DTYPE="bfloat16"
BATCH_SIZE="auto:4"
SYSTEM_INSTRUCTION="/no_think"
THINKING=false

# ── Help ──────────────────────────────────────────────────────────────────────
usage() {
    cat <<'HELP'
Usage: bash run_benchmarks.sh <model_path> <tag> <benchmarks> [options]

Arguments:
  model_path    Path or HF id of the model to evaluate
  tag           Output subdirectory name (results/benchmarks/<tag>/)
  benchmarks    Space-separated list, or "all" for all 10 standard benchmarks

Options:
  --dtype TYPE              Model dtype (default: bfloat16)
  --batch-size SIZE         Batch size (default: auto:4)
  --system-instruction STR  System instruction for chat benchmarks (default: /no_think)
  --thinking                Enable thinking-mode benchmarks (uses custom task yamls)
  -h, --help                Show this help

Standard benchmarks (10):
  mmlu  ceval  truthfulqa  arc_challenge  hellaswag
  winogrande  bbh  gsm8k  ifeval  mbpp

Thinking-mode benchmarks (use --thinking):
  gsm8k_thinking  math500_thinking

Examples:
  bash run_benchmarks.sh meta-llama/Llama-3-8B llama3-8b all
  bash run_benchmarks.sh ./compressed llama3-compressed "mmlu gsm8k ifeval"
  bash run_benchmarks.sh ./model tag all --dtype float16 --batch-size 8
  bash run_benchmarks.sh ./model tag "gsm8k_thinking math500_thinking" --thinking
HELP
    exit 0
}

# ── Parse positional args ─────────────────────────────────────────────────────
[[ $# -lt 1 ]] && usage
[[ "$1" == "-h" || "$1" == "--help" ]] && usage
[[ $# -lt 3 ]] && { echo "Error: requires 3 positional args: model_path tag benchmarks"; usage; }

MODEL_PATH="$1"
TAG="$2"
BENCHMARKS="$3"
shift 3

# ── Parse optional args ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dtype)          DTYPE="$2";              shift 2 ;;
        --batch-size)     BATCH_SIZE="$2";         shift 2 ;;
        --system-instruction) SYSTEM_INSTRUCTION="$2"; shift 2 ;;
        --thinking)       THINKING=true;           shift   ;;
        -h|--help)        usage ;;
        *) echo "Error: unknown option '$1'"; exit 1 ;;
    esac
done

# ── Find lm_eval ─────────────────────────────────────────────────────────────
if [[ -z "${LM_EVAL:-}" ]]; then
    LM_EVAL="$(command -v lm_eval 2>/dev/null || true)"
fi
if [[ -z "$LM_EVAL" || ! -x "$LM_EVAL" ]]; then
    echo "Error: lm_eval not found. Install lm-evaluation-harness or set LM_EVAL env var."
    exit 1
fi
echo "Using lm_eval: $LM_EVAL"

# ── Expand "all" ─────────────────────────────────────────────────────────────
ALL_BENCHMARKS="mmlu ceval truthfulqa arc_challenge hellaswag winogrande bbh gsm8k ifeval mbpp"
if [[ "$BENCHMARKS" == "all" ]]; then
    BENCHMARKS="$ALL_BENCHMARKS"
fi

# ── Output directory ─────────────────────────────────────────────────────────
OUTDIR="$SCRIPT_DIR/results/benchmarks/$TAG"
mkdir -p "$OUTDIR"
echo "Results will be saved to: $OUTDIR"

# ── Model args ───────────────────────────────────────────────────────────────
MODEL_ARGS="pretrained=$MODEL_PATH,dtype=$DTYPE,trust_remote_code=True"
# ── Benchmark definitions ────────────────────────────────────────────────────
# run_bench <name> <task> <num_fewshot> <type: ll|gen> [extra_flags...]
run_bench() {
    local name="$1" task="$2" fewshot="$3" btype="$4"
    shift 4
    local extra_flags=("$@")

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Running: $name  (task=$task, fewshot=$fewshot, type=$btype)"
    echo "════════════════════════════════════════════════════════════════"

    local cmd=("$LM_EVAL"
        --model hf
        --model_args "$MODEL_ARGS"
        --tasks "$task"
        --num_fewshot "$fewshot"
        --output_path "$OUTDIR/$name"
    )

    if [[ "$btype" == "gen" ]]; then
        cmd+=(--apply_chat_template
              --system_instruction "$SYSTEM_INSTRUCTION"
              --log_samples
              --batch_size "$BATCH_SIZE")
    else
        cmd+=(--batch_size "$BATCH_SIZE")
    fi

    # Append any extra flags
    if [[ ${#extra_flags[@]} -gt 0 ]]; then
        cmd+=("${extra_flags[@]}")
    fi

    echo "CMD: ${cmd[*]}"
    "${cmd[@]}"
    echo "Done: $name"
}

# ── Run each benchmark ───────────────────────────────────────────────────────
FAILED=()
PASSED=()

for bench in $BENCHMARKS; do
    (
    case "$bench" in
        # ── Loglikelihood benchmarks (no chat template) ──
        mmlu)
            run_bench mmlu mmlu 5 ll
            ;;
        ceval)
            run_bench ceval ceval-valid 5 ll
            ;;
        truthfulqa)
            run_bench truthfulqa truthfulqa_mc2 0 ll
            ;;
        arc_challenge)
            run_bench arc_challenge arc_challenge 25 ll
            ;;
        hellaswag)
            run_bench hellaswag hellaswag 10 ll
            ;;
        winogrande)
            run_bench winogrande winogrande 5 ll
            ;;

        # ── Generate benchmarks (with chat template) ──
        bbh)
            run_bench bbh bbh_zeroshot 0 gen
            ;;
        gsm8k)
            run_bench gsm8k gsm8k 5 gen
            ;;
        ifeval)
            run_bench ifeval ifeval 0 gen
            ;;
        mbpp)
            export HF_ALLOW_CODE_EVAL=1
            run_bench mbpp mbpp 3 gen --confirm_run_unsafe_code
            ;;

        # ── Thinking-mode benchmarks ──
        gsm8k_thinking)
            if [[ "$THINKING" != true ]]; then
                echo "Warning: $bench requires --thinking flag, skipping."
                exit 0
            fi
            THINK_MODEL_ARGS="$MODEL_ARGS,enable_thinking=true"
            echo ""
            echo "════════════════════════════════════════════════════════════════"
            echo "  Running: gsm8k_thinking  (thinking mode)"
            echo "════════════════════════════════════════════════════════════════"
            THINK_CMD=("$LM_EVAL"
                --model hf
                --model_args "$THINK_MODEL_ARGS"
                --tasks gsm8k_thinking
                --include_path "$SCRIPT_DIR/configs/tasks"
                --num_fewshot 5
                --output_path "$OUTDIR/gsm8k_thinking"
                --apply_chat_template
                --log_samples
                --batch_size "$BATCH_SIZE"
            )
            echo "CMD: ${THINK_CMD[*]}"
            "${THINK_CMD[@]}"
            echo "Done: gsm8k_thinking"
            ;;
        math500_thinking)
            if [[ "$THINKING" != true ]]; then
                echo "Warning: $bench requires --thinking flag, skipping."
                exit 0
            fi
            THINK_MODEL_ARGS="$MODEL_ARGS,enable_thinking=true"
            echo ""
            echo "════════════════════════════════════════════════════════════════"
            echo "  Running: math500_thinking  (thinking mode)"
            echo "════════════════════════════════════════════════════════════════"
            THINK_CMD=("$LM_EVAL"
                --model hf
                --model_args "$THINK_MODEL_ARGS"
                --tasks math500_thinking
                --include_path "$SCRIPT_DIR/configs/tasks"
                --num_fewshot 0
                --output_path "$OUTDIR/math500_thinking"
                --apply_chat_template
                --log_samples
                --batch_size "$BATCH_SIZE"
            )
            echo "CMD: ${THINK_CMD[*]}"
            "${THINK_CMD[@]}"
            echo "Done: math500_thinking"
            ;;

        *)
            echo "Error: unknown benchmark '$bench'"
            echo "Valid: $ALL_BENCHMARKS gsm8k_thinking math500_thinking"
            exit 1
            ;;
    esac
    ) && PASSED+=("$bench") || FAILED+=("$bench")
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Benchmark Summary"
echo "════════════════════════════════════════════════════════════════"
echo "  Model:   $MODEL_PATH"
echo "  Tag:     $TAG"
echo "  Output:  $OUTDIR"
echo "  Dtype:   $DTYPE"
echo ""
if [[ ${#PASSED[@]} -gt 0 ]]; then
    echo "  Passed (${#PASSED[@]}): ${PASSED[*]}"
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  FAILED (${#FAILED[@]}): ${FAILED[*]}"
    echo ""
    echo "Some benchmarks failed. Check logs above for details."
    exit 1
fi
echo ""
echo "All benchmarks completed successfully."
