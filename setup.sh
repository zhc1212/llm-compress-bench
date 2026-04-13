#!/bin/bash
# setup.sh — Install dependencies for llm-compress-bench
set -e

echo "Installing llm-compress-bench dependencies..."

# Core
pip install torch transformers datasets pyyaml

# lm-evaluation-harness with all extras
pip install "lm-eval[math,ifeval,code_eval]==0.4.11"

echo ""
echo "Done. Verify with: lm_eval --help"
