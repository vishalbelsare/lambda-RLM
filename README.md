

# $\lambda$-RLM
Code for $\lambda$-Recursive Language Models: typed functional recursion for reliable long-context reasoning. 

Standard LLM inference is limited by context windows and is opaque — models improvise decomposition unpredictably. $\lambda$-RLM solves this by:

- **Pre-computing** the optimal decomposition plan before any LLM call (deterministic)
- **Mapping inference** to Lambda Calculus primitives: β-reduction at leaves, symbolic combinators for composition
- **Splitting** long inputs into parallel chunks where each fits in the model's context window
- **Composing** results via pre-verified operators (MERGE_COUNTS, SEARCH_UNION, SUMMARIZE_REDUCE, etc.)

---

## Installation and Setup

From the project root (the directory containing pyproject.toml):

```bash
conda create -n lambda-rlm python=3.11 -y
conda activate lambda-rlm

pip install -e .
```
We support access models through APIs, for example, you can request a [NVIDIA NIM API key](https://build.nvidia.com) to access available models.

```bash
export NVIDIA_API_KEY="nvapi-..."
```


### Supported Task Types

| Task | Composition Operator | Strategy |
|---|---|---|
| `aggregation` | MERGE_COUNTS | SPLIT → MAP → sum counts |
| `search` | SEARCH_UNION | SPLIT → MAP → union doc IDs |
| `classification` | CONCAT | SPLIT → MAP → concat labels |
| `pairwise` | CONCAT + symbolic CROSS | MAP(classify) → CROSS(O(N²), free) |
| `summarization` | SUMMARIZE_REDUCE + M | SPLIT → MAP → M(final synthesis) |
| `extraction` | CONCAT | SPLIT → MAP → concat fields |
| `code_understanding` | CONCAT | SPLIT → MAP → concat analysis |
| `multi_hop` | SUMMARIZE_REDUCE + M | SPLIT → MAP(extract) → M(synthesize) |


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
        "model_name": "qwen/qwen3-next-80b-a3b-thinking",
        "api_key": os.environ["NVIDIA_API_KEY"],
        "base_url": "https://integrate.api.nvidia.com/v1",
    }
)

result = rlm.completion(prompt)
print(result.response)
```

## Benchmarking

The benchmark entry point is used for running experiments with same dataset and comparing the behavior, latency, and output quality under the same setup between Normal RLM (rlm) and Lambda-RLM (lambda_rlm)

### Supported datasets:
- `sniah` — Sequential-NIAH examples loaded from a local JSONL file or from the public GitHub JSONL source
- `oolong` — single-document QA examples loaded from `THUDM/LongBench-v2`
- `browsecomp` — multi-document QA examples loaded from `THUDM/LongBench-v2`
- `codeqa` — code repository understanding examples loaded from a local JSONL file or from `THUDM/LongBench-v2`

### Supported models:
- todo

### Compare both methods on the same dataset

```bash
python benchmarks/benchmark.py --datasets sniah --methods rlm lambda_rlm --n-samples-per-bucket 1 --max-iter 8 --max-depth 2 --context-window 100000 --output-dir ./results_compare
```

Outputs are written to the specified output directory, typically including:
- `results.json`
- `stats.json`
- `averages.json`

### Run only Normal RLM

```bash
python benchmarks/benchmark.py --datasets sniah --methods rlm --n-samples-per-bucket 1 --max-iter 8 --max-depth 2 --context-window 100000 --output-dir ./results_compare
```

### Run only Lambda-RLM

```bash
python benchmarks/benchmark.py --datasets sniah --methods lambda_rlm --n-samples-per-bucket 1 --max-iter 8 --max-depth 2 --context-window 100000 --output-dir ./results_compare
```

## Repository structure
### Normal RLM

This repository uses upstream Normal RLM components for comparison.

Upstream repository:
`https://github.com/alexzhang13/rlm`

The upstream code is licensed under the MIT License. See `THIRD_PARTY_NOTICES.md` for attribution and licensing details.

Key files:
- `rlm/core/rlm.py` — main REPL-based RLM loop
- `rlm/environments/local_repl.py` — sandboxed Python REPL execution, context storage, and helper functions
- `rlm/utils/parsing.py` — parsing of ```repl``` code blocks and FINAL markers; formatting of execution output back into the model history
- `rlm/clients/openai.py` — OpenAI-compatible client used with NVIDIA NIM

### Lambda-RLM

- `rlm/lambda_rlm.py` — LambdaRLM implementation, including task detection, planning, and deterministic execution through Φ