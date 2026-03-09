

# Lambda-RLM

This repository provides two long-context reasoning methods:

- Normal RLM (rlm): An iterative REPL-based Recursive Language Model based on the RLM implementation: https://github.com/alexzhang13/rlm.
- Lambda-RLM (lambda_rlm): A deterministic alternative that replaces the open-ended REPL loop with a bounded, planned split/map/reduce-style executor (Φ). It performs bounded leaf calls and deterministic composition.

## Repository structure

Core (Normal RLM / REPL loop):
- rlm/core/rlm.py — main RLM loop (RLM.completion)
- rlm/environments/local_repl.py — sandboxed Python REPL execution + context storage + injected helper functions
- rlm/utils/parsing.py — parses ```repl``` code blocks and FINAL markers; formats execution output back into the LLM history

Lambda-RLM:
- rlm/lambda_rlm.py — LambdaRLM implementation (task detection, deterministic plan, deterministic executor Φ)

Benchmark / evaluation:
- benchmarks/benchmark.py — runs experiments and compares methods; writes JSON outputs

## Installation

From the project root (the directory containing pyproject.toml):

```bash
conda create -n lambda-rlm python=3.11 -y
conda activate lambda-rlm

pip install -e .

pip install datasets
pip install matplotlib numpy
```
## NVIDIA NIM API setup

Set your API key:

```bash
export NVIDIA_API_KEY="your_nvidia_key"
```


## Quick start: run Normal RLM and Lambda-RLM on the same dataset
### Normal RLM

--max-iter: Max REPL iterations. More iterations can help on harder tasks but increases cost/latency.
--max-depth: Recursion depth for sub-calls (if enabled). Larger depth can allow more nested calls.

Example:

```bash
python benchmarks/benchmark.py \
  --datasets oolong \
  --methods rlm \
  --max-iter 10 \
  --max-depth 2 \
  --n-samples-per-bucket 5 \
  --output-dir ./results_rlm
  ```

## Lambda-RLM

--context-window: Lambda-RLM budgeting parameter (in characters). This affects how aggressively Lambda-RLM splits the context.

Example:

``` bash
python benchmarks/benchmark.py \
  --datasets oolong \
  --methods lambda_rlm \
  --context-window 100000 \
  --n-samples-per-bucket 5 \
  --output-dir ./results_lambda
  ```
