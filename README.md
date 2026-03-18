

# Lambda-RLM
Code for $\lambda$-Recursive Language Models: typed functional recursion for reliable long-context reasoning. Lambda-RLM provides a deterministic alternative to open-ended REPL-based recursive reasoning by replacing the unbounded control loop with a bounded split / map / reduce-style executor ($\Phi$). The goal is to make long-context reasoning more predictable, easier to benchmark, and easier to compare against standard Recursive Language Models.

## Installation

From the project root (the directory containing pyproject.toml):

```bash
conda create -n lambda-rlm python=3.11 -y
conda activate lambda-rlm

pip install -e .
```

## NVIDIA NIM API setup

Set your API key:

```bash
export NVIDIA_API_KEY="your_nvidia_key"
```

## Benchmark Usage

The main usage of this repository is the benchmark entry point for running experiments with same dataset and comparing the behavior, latency, and output quality under the same setup between:
- Normal RLM (`rlm`)
- Lambda-RLM (`lambda_rlm`)

### Quick start: compare both methods on the same dataset

python benchmarks/benchmark.py \
  --datasets oolong \
  --methods rlm lambda_rlm \
  --n-samples-per-bucket 2 \
  --max-iter 8 \
  --max-depth 2 \
  --context-window 100000 \
  --output-dir ./results_compare

Outputs are written to the specified output directory, typically including:
- `results.json`
- `stats.json`
- `averages.json`

### Run only Normal RLM

python benchmarks/benchmark.py \
  --datasets oolong \
  --methods rlm \
  --max-iter 10 \
  --max-depth 2 \
  --n-samples-per-bucket 5 \
  --output-dir ./results_rlm

### Run only Lambda-RLM

python benchmarks/benchmark.py \
  --datasets oolong \
  --methods lambda_rlm \
  --context-window 100000 \
  --n-samples-per-bucket 5 \
  --output-dir ./results_lambda

## Repository structure

This repository has two main parts: Normal RLM and Lambda-RLM.

### 1. Normal RLM

The Normal RLM components are used as the comparison baseline in this repository.

These components are derived from the upstream RLM implementation:
`https://github.com/alexzhang13/rlm`

Key files:
- `rlm/core/rlm.py` — main REPL-based RLM loop
- `rlm/environments/local_repl.py` — sandboxed Python REPL execution, context storage, and helper functions
- `rlm/utils/parsing.py` — parsing of ```repl``` code blocks and FINAL markers; formatting of execution output back into the model history
- `rlm/clients/openai.py` — OpenAI-compatible client used with NVIDIA NIM

### 2. Lambda-RLM

The Lambda-RLM implementation is the new deterministic method added in this repository.

Key file:
- `rlm/lambda_rlm.py` — LambdaRLM implementation, including task detection, planning, and deterministic execution through Φ

## Upstream attribution

This repository uses upstream Normal RLM components for comparison.

Upstream repository:
`https://github.com/alexzhang13/rlm`

The upstream code is licensed under the MIT License. See `THIRD_PARTY_NOTICES.md` for attribution and licensing details.