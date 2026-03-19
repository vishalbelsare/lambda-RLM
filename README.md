# $\lambda$-RLM
Code for **The $\mathbf{Y}$-Combinator for LLMs: \\ Solving Long-Context Rot with $\lambda$-Calculus**: a framework for long-context reasoning that replaces free-form recursive code generation with a typed functional runtime grounded in $\lambda$-calculus.

<p align="center">
  <img src="intro.png" alt="Lambda-RLM results figure" width="900" />
</p>

Direct LLM inference and standard RLM inference are constrained by context windows and can rely on hard-to-predict decomposition strategies. $\lambda$-RLM addresses this by:

- **Planning decomposition ahead of execution** with a deterministic recursive strategy
- **Expressing inference through functional structure**, with model calls at local steps and symbolic operators for composition
- **Breaking long inputs into manageable chunks** that fit within the model context window
- **Applying the model only to bounded leaf subproblems and combining intermediate results** through structured operators such as `SPLIT`, `MAP`, `FILTER`, `REDUCE`, `CONCAT` and `CROSS`.

---

## Installation and Setup

From the project root (the directory containing pyproject.toml):

```bash
conda create -n lambda-rlm python=3.11 -y
conda activate lambda-rlm

pip install -e .
```
The project supports multiple API-compatible model providers. For example, you can request a [NVIDIA NIM API key](https://build.nvidia.com) or a [TOGETHER AI API key](https://api.together.ai/) to access the available model backends. Set your API key as an environment variable:

```bash
export NVIDIA_API_KEY="nvapi-..."
```

```bash
export TOGETHER_API_KEY="tgp_..."
```

### Supported datasets:
- `sniah` — Sequential-NIAH examples loaded from the public GitHub JSONL source
- `oolong` — single-document QA examples loaded from `THUDM/LongBench-v2`
- `browsecomp` — multi-document QA examples loaded from `THUDM/LongBench-v2`
- `codeqa` — code repository understanding examples loaded from a local JSONL file or from `THUDM/LongBench-v2`

## Usage

### Quick Start

```python
import os
from rlm import LambdaRLM

document = """
This report discusses the development of a new battery technology.
It covers technical design choices, manufacturing trade-offs, safety concerns,
cost reduction strategies, and future commercialization plans.

One section focuses on performance improvements in energy density.
Another section discusses supply-chain risks and regulatory constraints.
The final section outlines expected market impact and open research questions.
"""

prompt = f"""Context:
{document}

Question: Summarize the main ideas discussed in this document.

Answer:"""

rlm = LambdaRLM(
    backend_kwargs={
        "model_name": "meta/llama-3.3-70b-instruct",
        "api_key": os.environ["NVIDIA_API_KEY"],
        "base_url": "https://integrate.api.nvidia.com/v1",
    }
)

result = rlm.completion(prompt)
print(result.response)
```

## Repository structure
### Normal RLM

This repository uses upstream Normal RLM components for comparison: `https://github.com/alexzhang13/rlm`

The upstream code is licensed under the MIT License. See `THIRD_PARTY_NOTICES.md` for attribution and licensing details.

Key files:
- `rlm/core/rlm.py` — main REPL-based RLM loop
- `rlm/environments/local_repl.py` — sandboxed Python REPL execution, context storage, and helper functions
- `rlm/utils/parsing.py` — parsing of ```repl``` code blocks and FINAL markers; formatting of execution output back into the model history
- `rlm/clients/openai.py` — OpenAI-compatible client used with NVIDIA NIM

### $\lambda$-RLM

- `rlm/lambda_rlm.py` — LambdaRLM implementation, including task detection, planning, and deterministic execution through $\Phi$

## Benchmarking

The benchmark entry point is used for running experiments with same dataset and comparing the behavior, latency, and output quality under the same setup between Normal RLM (rlm) and Lambda-RLM (lambda_rlm)


### Compare both methods on the same dataset

```bash
python benchmarks/benchmark.py --datasets sniah --model meta/llama-3.3-70b-instruct --methods rlm lambda_rlm --n-samples-per-bucket 2 --max-iter 8 --max-depth 2 --context-window 100000 --output-dir ./results/llama-3.3-70b-instruct
```

Outputs are written to the specified output directory, typically including:
- `results.json`
- `stats.json`
- `averages.json`

### Run only Normal RLM

```bash
python benchmarks/benchmark.py --datasets sniah --model meta/llama-3.3-70b-instruct --methods rlm --n-samples-per-bucket 2 --max-iter 8 --max-depth 2 --context-window 100000 --output-dir ./results/llama-3.3-70b-instruct_rlm
```

### Run only $\lambda$-RLM

```bash
python benchmarks/benchmark.py --datasets sniah --model meta/llama-3.3-70b-instruct --methods lambda_rlm --n-samples-per-bucket 2 --max-iter 8 --max-depth 2 --context-window 100000 --output-dir ./results/llama-3.3-70b-instruct_lambda_rlm
```