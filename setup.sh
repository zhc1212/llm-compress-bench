#!/bin/bash
# setup.sh — Install dependencies for llm-compress-bench
set -e

echo "Installing llm-compress-bench dependencies..."

# Core
pip install torch transformers datasets pyyaml

# lm-evaluation-harness with extras
pip install "lm-eval[math,ifeval]==0.4.11"

# Code evaluation (for MBPP/HumanEval)
pip install "lm-eval[code_eval]==0.4.11"

echo ""
echo "Done. Verify with: lm_eval --help"
