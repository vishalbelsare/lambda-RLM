"""
benchmark.py — Normal RLM vs λ-RLM, stratified by context-length buckets

Context buckets (characters):
  8k | 16k | 33k | 66k | 131k | 262k | 524k | 1M

For each dataset × bucket × method we report:
  • Token F1 accuracy
  • Wall-clock latency (seconds)

Final row is the macro-average across all non-empty buckets.

Datasets:
  oolong     – oolongbench/oolong-real
  sniah      – Sequential-NIAH (GitHub JSONL, explicitly length-stratified)
  browsecomp – Tevatron/browsecomp-plus
  codeqa     – configurable local JSONL or code_search_net fallback

Usage:
  python benchmarks/benchmark.py \\
      --n-samples-per-bucket 5 \\
      --datasets oolong sniah browsecomp codeqa \\
      --output-dir ./benchmark_results
"""

from __future__ import annotations

import argparse
import json
import math
import sys
# Ensure the project root (parent of benchmarks/) is on sys.path so `import rlm` works
# whether the script is run as `python benchmarks/benchmark.py` or `python -m benchmarks.benchmark`.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import os
import re
import string
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    from datasets import load_dataset as hf_load_dataset
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    print("WARNING: `datasets` not installed — HuggingFace datasets skipped.")
    print("         pip install datasets")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False
    print("WARNING: `matplotlib`/`numpy` not installed — plots skipped.")
    print("         pip install matplotlib numpy")

try:
    import requests as _requests_mod
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


# ── Context-length buckets: 2^13 → 2^18 tokens ────────────────────────────────
# Six bins, one per power of 2 from 2^13 (8 192) to 2^18 (262 144).
# Bin boundaries are geometric midpoints between consecutive powers:
#   gm(2^n, 2^(n+1)) = 2^(n+0.5)
# so each sample is assigned to the nearest power-of-2 bucket.
#
#   Label  Centre     Lower boundary   Upper boundary
#   8k     2^13=8192       0           2^13.5≈11585
#   16k    2^14=16384  11585           2^14.5≈23170
#   32k    2^15=32768  23170           2^15.5≈46341
#   64k    2^16=65536  46341           2^16.5≈92682
#   128k   2^17=131072 92682           2^17.5≈185364
#   256k   2^18=262144 185364          ∞
#
# Tokens for non-S-Niah datasets are estimated as chars // 4.

BIN_LABELS: list[str] = ["8k", "16k", "32k", "64k", "128k", "256k"]
_BIN_EXPS:  list[int]  = list(range(13, 19))          # 13..18
_BIN_BOUNDARIES: list[float] = [
    2 ** (e + 0.5) for e in _BIN_EXPS[:-1]            # 5 midpoints
]


def assign_bin(token_len: int) -> str:
    """Assign token_len to the nearest power-of-2 bucket in [2^13, 2^18]."""
    for i, boundary in enumerate(_BIN_BOUNDARIES):
        if token_len < boundary:
            return BIN_LABELS[i]
    return BIN_LABELS[-1]    # 256k bucket (anything ≥ 2^17.5)


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class Sample:
    dataset:   str
    idx:       int
    context:   str   # full context text
    question:  str
    gold:      str
    ctx_len:   int   # len(context) in characters
    token_len: int   # context length in tokens (exact if known, else chars//4)
    bin_label: str   # one of BIN_LABELS (assigned from token_len)


@dataclass
class Result:
    dataset:    str
    idx:        int
    bin_label:  str
    method:     str   # "rlm" | "lambda_rlm"
    prediction: str
    gold:       str
    f1:         float
    contains:   float
    exact:      float
    latency:    float
    error:      str | None = None


@dataclass
class BinStats:
    dataset:    str
    method:     str
    bin_label:  str
    n:          int
    f1:         float   # mean
    contains:   float   # mean
    exact:      float   # mean
    latency:    float   # mean seconds
    errors:     int


# ── Evaluation ────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def _f1(pred: str, gold: str) -> float:
    p_toks = _norm(pred).split()
    g_toks = _norm(gold).split()
    if not p_toks or not g_toks:
        return 0.0
    common = set(p_toks) & set(g_toks)
    if not common:
        return 0.0
    prec = len(common) / len(p_toks)
    rec  = len(common) / len(g_toks)
    return 2 * prec * rec / (prec + rec)


def _contains(pred: str, gold: str) -> float:
    return float(_norm(gold) in _norm(pred))


def _exact(pred: str, gold: str) -> float:
    return float(_norm(pred) == _norm(gold))


# ── Dataset loaders ───────────────────────────────────────────────────────────

_MAX_CTX = 1_200_000  # hard cap to avoid OOM


def _make_sample(dataset: str, idx: int, ctx: str, q: str, gold: str,
                  token_len: int | None = None) -> Sample:
    ctx = ctx[:_MAX_CTX]
    tlen = token_len if token_len is not None else len(ctx) // 4
    return Sample(dataset=dataset, idx=idx, context=ctx, question=q, gold=gold,
                  ctx_len=len(ctx), token_len=tlen, bin_label=assign_bin(tlen))


# LongBench-v2 length label → approximate token count for bin assignment.
# Schema: "short" ~8k, "medium" ~32k, "long" ~128k, "extra-long" ~256k.
_LBV2_LEN_TOKENS: dict[str, int] = {
    "short":      8_192,
    "medium":    32_768,
    "long":     131_072,
    "extra-long": 262_144,
}


def _load_longbench_v2(
    max_per_bin: int,
    domain_filter: str | None,
    dataset_name: str,
    display_name: str,
) -> list[Sample]:
    """
    Load THUDM/LongBench-v2, optionally filtered by domain.

    Schema: _id, domain, sub_domain, difficulty, length (short/medium/long/extra-long),
            question, choice_A-D, answer (A/B/C/D), context.
    Gold = full text of the correct choice, so F1 can be computed on free-text output.
    Context starts with a UUID line; we strip it to get the actual document.
    """
    if not _HF_AVAILABLE:
        print(f"  [SKIP] {display_name}: `datasets` not available")
        return []
    try:
        ds = hf_load_dataset("THUDM/LongBench-v2", split="train")
    except Exception as e:
        print(f"  [SKIP] {display_name}: could not load LongBench-v2 — {e}")
        return []

    bin_counts: dict[str, int] = defaultdict(int)
    samples: list[Sample] = []
    for i, row in enumerate(ds):
        if domain_filter and domain_filter not in row.get("domain", ""):
            continue
        q = str(row.get("question", "")).strip()
        correct_key = f"choice_{row.get('answer', 'A')}"
        gold = str(row.get(correct_key, "")).strip()
        if not q or not gold:
            continue
        # Strip leading UUID line from context (format: "<uuid>\n<document>")
        ctx_raw = str(row.get("context", ""))
        first_nl = ctx_raw.find("\n")
        ctx = ctx_raw[first_nl + 1:].strip() if first_nl != -1 and first_nl < 50 else ctx_raw
        token_len = _LBV2_LEN_TOKENS.get(str(row.get("length", "short")), 8_192)
        s = _make_sample(dataset_name, i, ctx, q, gold, token_len=token_len)
        if bin_counts[s.bin_label] < max_per_bin:
            samples.append(s)
            bin_counts[s.bin_label] += 1
        if all(v >= max_per_bin for v in bin_counts.values()) and len(bin_counts) == len(BIN_LABELS):
            break

    _print_load_summary(display_name, samples)
    return samples


def load_oolong(max_per_bin: int) -> list[Sample]:
    """Single-Document QA from LongBench-v2 (replaces unavailable oolongbench)."""
    return _load_longbench_v2(
        max_per_bin,
        domain_filter="Single-Document QA",
        dataset_name="oolong",
        display_name="OOLONG",
    )


def load_sniah(max_per_bin: int, local_path: str | None = None) -> list[Sample]:
    _RAW_URL = (
        "https://raw.githubusercontent.com/miraclefish/Sequential-NIAH-Benchmark"
        "/main/data/test_data/test_data_for_infer.jsonl"
    )
    rows: list[dict] = []

    if local_path and os.path.exists(local_path):
        with open(local_path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
    elif _REQUESTS_AVAILABLE:
        try:
            r = _requests_mod.get(_RAW_URL, timeout=60)
            r.raise_for_status()
            rows = [json.loads(l) for l in r.text.splitlines() if l.strip()]
        except Exception as e:
            print(f"  [SKIP] S-Niah: fetch failed – {e}")
            return []
    else:
        print("  [SKIP] S-Niah: no local path and `requests` not installed")
        return []

    if not rows:
        print("  [SKIP] S-Niah: empty")
        return []

    # Actual schema: md5, length, lang, num_needles, source_QA, ppl_QA,
    #                question, gt_answer, raw_question
    # `length`       — target context length in tokens (use directly for binning)
    # `question`     — full "Document:\n...\nQuestion: ..." string (the haystack)
    # `raw_question` — the actual question without context
    # `gt_answer`    — gold answer
    # `source_QA`/`ppl_QA` — string tags ("syn", etc.), not usable as context
    row0 = rows[0]
    if "question" not in row0 or "gt_answer" not in row0:
        print(f"  [SKIP] S-Niah: unexpected schema {list(row0.keys())}")
        return []

    def _sniah_context(row: dict) -> str:
        """Extract the document portion from the 'question' haystack field.

        The field has the form:
            Document:
            <haystack text>
            ...
            Question: <question text>

        We strip the trailing 'Question: ...' line to isolate the document.
        """
        full = str(row.get("question", ""))
        # Split off the trailing "Question: ..." if present
        sep = "\nQuestion: "
        idx = full.rfind(sep)
        if idx != -1:
            return full[:idx].strip()
        return full.strip()

    bin_counts: dict[str, int] = defaultdict(int)
    samples: list[Sample] = []
    for i, row in enumerate(rows):
        gold_raw = row["gt_answer"]
        gold = (gold_raw[0] if isinstance(gold_raw, list) else str(gold_raw)).strip()
        # Prefer raw_question (clean question without context), fall back to question
        q   = str(row.get("raw_question") or row["question"]).strip()
        ctx = _sniah_context(row)
        # Use the explicit `length` token count for bin assignment
        token_len = int(row.get("length", len(ctx) // 4))
        s = _make_sample("sniah", i, ctx, q, gold, token_len=token_len)
        if bin_counts[s.bin_label] < max_per_bin:
            samples.append(s)
            bin_counts[s.bin_label] += 1

    _print_load_summary("S-Niah", samples)
    return samples


def load_browsecomp(max_per_bin: int) -> list[Sample]:
    """Multi-Document QA from LongBench-v2 (replaces encrypted browsecomp-plus)."""
    return _load_longbench_v2(
        max_per_bin,
        domain_filter="Multi-Document QA",
        dataset_name="browsecomp",
        display_name="BrowseComp+",
    )


def load_codeqa(max_per_bin: int, local_path: str | None = None) -> list[Sample]:
    if local_path and os.path.exists(local_path):
        rows = [json.loads(l) for l in open(local_path) if l.strip()]
        bin_counts: dict[str, int] = defaultdict(int)
        samples: list[Sample] = []
        for i, row in enumerate(rows):
            q_k   = next((k for k in ["question","query","docstring"]  if k in row), None)
            ans_k = next((k for k in ["answer","code","label","output"] if k in row), None)
            ctx_k = next((k for k in ["context","code","body","text"]   if k in row), None)
            if not q_k or not ans_k:
                continue
            ctx  = str(row.get(ctx_k, "")) if ctx_k else ""
            q    = str(row[q_k])
            gold = str(row[ans_k])
            s = _make_sample("codeqa", i, ctx, q, gold)
            if bin_counts[s.bin_label] < max_per_bin:
                samples.append(s)
                bin_counts[s.bin_label] += 1
        _print_load_summary("CodeQA", samples)
        return samples

    # HF fallback: Code Repository Understanding from LongBench-v2
    return _load_longbench_v2(
        max_per_bin,
        domain_filter="Code Repository Understanding",
        dataset_name="codeqa",
        display_name="CodeQA",
    )


def _print_load_summary(name: str, samples: list[Sample]) -> None:
    counts = defaultdict(int)
    for s in samples:
        counts[s.bin_label] += 1
    line = "  ".join(f"{b}:{counts.get(b,0)}" for b in BIN_LABELS)
    print(f"  {name}: {len(samples)} samples   [{line}]")


# ── Runner ────────────────────────────────────────────────────────────────────

def _build_prompt(s: Sample) -> str:
    if s.context.strip():
        return f"Context:\n{s.context}\n\nQuestion: {s.question}\n\nAnswer:"
    return f"Question: {s.question}\n\nAnswer:"


def run_sample(method: str, runner, sample: Sample) -> Result:
    prompt = _build_prompt(sample)
    t0 = time.perf_counter()
    error = None
    prediction = ""
    try:
        out = runner(prompt)
        prediction = out.response if hasattr(out, "response") else str(out)
    except Exception as e:
        error = str(e)
    latency = time.perf_counter() - t0
    return Result(
        dataset=sample.dataset, idx=sample.idx,
        bin_label=sample.bin_label, method=method,
        prediction=prediction, gold=sample.gold,
        f1=_f1(prediction, sample.gold),
        contains=_contains(prediction, sample.gold),
        exact=_exact(prediction, sample.gold),
        latency=latency, error=error,
    )


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(results: list[Result]) -> list[BinStats]:
    groups: dict[tuple[str, str, str], list[Result]] = defaultdict(list)
    for r in results:
        groups[(r.dataset, r.method, r.bin_label)].append(r)

    stats: list[BinStats] = []
    for (ds, method, bl), items in sorted(groups.items()):
        n = len(items)
        stats.append(BinStats(
            dataset=ds, method=method, bin_label=bl, n=n,
            f1      =sum(r.f1       for r in items) / n,
            contains=sum(r.contains for r in items) / n,
            exact   =sum(r.exact    for r in items) / n,
            latency =sum(r.latency  for r in items) / n,
            errors  =sum(1 for r in items if r.error),
        ))
    return stats


def _avg_row(stats: list[BinStats], dataset: str, method: str) -> BinStats | None:
    """Macro-average across all bins for a (dataset, method) pair."""
    items = [s for s in stats if s.dataset == dataset and s.method == method and s.n > 0]
    if not items:
        return None
    n_total = sum(s.n for s in items)
    return BinStats(
        dataset=dataset, method=method, bin_label="AVG",
        n=n_total,
        f1      =sum(s.f1       for s in items) / len(items),
        contains=sum(s.contains for s in items) / len(items),
        exact   =sum(s.exact    for s in items) / len(items),
        latency =sum(s.latency  for s in items) / len(items),
        errors  =sum(s.errors   for s in items),
    )


# ── Console table ─────────────────────────────────────────────────────────────

_METHOD_LABEL = {"rlm": "Normal RLM", "lambda_rlm": "λ-RLM"}
_NAN = "  --  "


def _pct(v: float | None) -> str:
    return f"{v*100:5.1f}%" if v is not None else _NAN


def _sec(v: float | None) -> str:
    return f"{v:6.2f}s" if v is not None else _NAN


def print_table(stats: list[BinStats], datasets: list[str], methods: list[str]) -> None:
    idx: dict[tuple[str, str, str], BinStats] = {
        (s.dataset, s.method, s.bin_label): s for s in stats
    }

    col_w = 9
    header_bins = BIN_LABELS + ["AVG"]

    for ds in datasets:
        print(f"\n{'═'*100}")
        print(f"  Dataset: {ds.upper()}")
        print(f"{'═'*100}")

        # Header row
        hdr = f"  {'Method':<14}  {'Metric':<10}"
        for b in header_bins:
            hdr += f"  {b:>{col_w}}"
        print(hdr)
        print(f"  {'-'*14}  {'-'*10}" + f"  {'-'*col_w}" * len(header_bins))

        for method in methods:
            avg = _avg_row(stats, ds, method)
            label = _METHOD_LABEL.get(method, method)

            # F1 row
            row_f1 = f"  {label:<14}  {'F1 (%)':10}"
            for b in BIN_LABELS:
                s = idx.get((ds, method, b))
                row_f1 += f"  {_pct(s.f1 if s else None):>{col_w}}"
            row_f1 += f"  {_pct(avg.f1 if avg else None):>{col_w}}"
            print(row_f1)

            # Latency row (indented under same method)
            row_lat = f"  {'':14}  {'Time (s)':10}"
            for b in BIN_LABELS:
                s = idx.get((ds, method, b))
                row_lat += f"  {_sec(s.latency if s else None):>{col_w}}"
            row_lat += f"  {_sec(avg.latency if avg else None):>{col_w}}"
            print(row_lat)

            print()


# ── Plots ─────────────────────────────────────────────────────────────────────

# Actual token counts for each bin label — used as log-scale x positions.
_BIN_X: list[int] = [2 ** e for e in _BIN_EXPS]   # [8192, 16384, ..., 262144]

# One distinctive color + marker per dataset (up to 8 datasets).
_DS_COLORS  = ["#E07B39", "#3A86C8", "#33A474", "#C94040",
               "#8B5CF6", "#EC4899", "#F59E0B", "#6B7280"]
_DS_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X"]

_METHOD_TITLE = {"rlm": "RLM", "lambda_rlm": "λ-RLM"}


def _make_score_plot(ax, stats_idx, datasets: list[str], method: str,
                     model_label: str) -> None:
    """Draw one score-vs-context-length plot (log scale) on *ax*."""

    ds_colors  = {ds: _DS_COLORS[i % len(_DS_COLORS)]  for i, ds in enumerate(datasets)}
    ds_markers = {ds: _DS_MARKERS[i % len(_DS_MARKERS)] for i, ds in enumerate(datasets)}

    # Collect (ds, xs, ys) first so we can stagger right-side labels
    series: list[tuple[str, list, list]] = []
    for ds in datasets:
        xs, ys = [], []
        for xi, b in zip(_BIN_X, BIN_LABELS):
            s = stats_idx.get((ds, method, b))
            if s and s.n > 0:
                xs.append(xi)
                ys.append(s.f1 * 100)
        if ys:
            series.append((ds, xs, ys))

    for ds, xs, ys in series:
        ax.plot(xs, ys,
                color=ds_colors[ds], marker=ds_markers[ds],
                linewidth=2, markersize=8, label=ds.upper())

    # Stagger right-side labels by last-y-value to prevent overlap
    label_items = [(ys[-1], ds, xs[-1]) for ds, xs, ys in series if ys]
    label_items.sort(key=lambda t: t[0], reverse=True)
    prev_y = None
    MIN_GAP = 4.5   # percentage points minimum gap
    for i, (last_y, ds, last_x) in enumerate(label_items):
        if prev_y is not None and (prev_y - last_y) < MIN_GAP:
            last_y = prev_y - MIN_GAP
        ax.annotate(
            ds.upper(),
            xy=(last_x, last_y),
            xytext=(8, 0), textcoords="offset points",
            va="center", fontsize=9,
            color=ds_colors[ds], fontweight="bold",
        )
        prev_y = last_y

    # Shade last bin as "extrapolation" region (lightest in dataset)
    shade_x = _BIN_X[-1]
    ax.axvspan(shade_x / 2 ** 0.5, shade_x * 2 ** 0.5,
               color="#B2EBE0", alpha=0.45, zorder=0)

    ax.set_xscale("log", base=2)
    ax.set_xticks(_BIN_X)
    ax.set_xticklabels(BIN_LABELS, fontsize=10)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("Input Context Length (log scale)", fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_xlim(_BIN_X[0] / 1.4, _BIN_X[-1] * 1.9)
    ax.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.grid(axis="x", linestyle=":", alpha=0.25)
    ax.set_title(f"{_METHOD_TITLE.get(method, method)}({model_label})",
                 fontsize=13, fontweight="bold")


def make_plots(stats: list[BinStats], datasets: list[str], methods: list[str],
               output_dir: Path, model_label: str = "model") -> None:
    if not _PLOT_AVAILABLE:
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    idx: dict[tuple[str, str, str], BinStats] = {
        (s.dataset, s.method, s.bin_label): s for s in stats
    }

    # ── One score plot per method (all datasets as lines) ────────────────────
    for method in methods:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        _make_score_plot(ax, idx, datasets, method, model_label)
        fig.tight_layout()
        fname = plots_dir / f"{method}_score_by_ctx.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")

    # ── Side-by-side: RLM vs λ-RLM on one figure ────────────────────────────
    if len(methods) >= 2:
        fig, axes = plt.subplots(1, len(methods), figsize=(9 * len(methods), 5.5),
                                  sharey=True)
        for ax, method in zip(axes, methods):
            _make_score_plot(ax, idx, datasets, method, model_label)
        fig.suptitle("Normal RLM vs λ-RLM — Score by Context Length",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        fname = plots_dir / "rlm_vs_lambda_score.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")

    # ── Delta F1 heatmap: rows=datasets, cols=bins ────────────────────────────
    if "rlm" in methods and "lambda_rlm" in methods and _PLOT_AVAILABLE:
        data_matrix = []
        row_labels  = []
        for ds in datasets:
            row = []
            for b in BIN_LABELS:
                rlm_s  = idx.get((ds, "rlm",        b))
                lrlm_s = idx.get((ds, "lambda_rlm", b))
                if rlm_s and lrlm_s:
                    row.append((lrlm_s.f1 - rlm_s.f1) * 100)
                else:
                    row.append(float("nan"))
            data_matrix.append(row)
            row_labels.append(ds)

        matrix_np = np.array(data_matrix, dtype=float)
        fig, ax = plt.subplots(figsize=(12, max(3, len(datasets) * 1.2)))
        vmax = max(abs(np.nanmax(matrix_np)), abs(np.nanmin(matrix_np)), 1.0)
        im = ax.imshow(matrix_np, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(BIN_LABELS)))
        ax.set_xticklabels(BIN_LABELS, fontsize=10)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=10)
        ax.set_title("Δ F1: λ-RLM minus Normal RLM (pp)", fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Δ F1 (percentage points)")
        for r in range(len(row_labels)):
            for c in range(len(BIN_LABELS)):
                v = matrix_np[r, c]
                if not math.isnan(v):
                    ax.text(c, r, f"{v:+.1f}", ha="center", va="center",
                            fontsize=8, color="black")
        fig.tight_layout()
        fname = plots_dir / "delta_f1_heatmap.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved: {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Benchmark Normal RLM vs λ-RLM stratified by context length."
    )
    p.add_argument("--backend",            default="openai")
    p.add_argument("--model",              default="qwen/qwen3-next-80b-a3b-thinking")
    p.add_argument("--base-url",           default="https://integrate.api.nvidia.com/v1")
    p.add_argument("--api-key",            default=None)
    p.add_argument("--n-samples-per-bucket", type=int, default=5,
                   help="Max samples per context-length bucket per dataset")
    p.add_argument("--datasets",  nargs="+",
                   default=["oolong", "sniah", "browsecomp", "codeqa"],
                   choices=["oolong", "sniah", "browsecomp", "codeqa"])
    p.add_argument("--methods",   nargs="+", default=["rlm", "lambda_rlm"],
                   choices=["rlm", "lambda_rlm"])
    p.add_argument("--output-dir",         default="./benchmark_results")
    p.add_argument("--sniah-path",         default=None)
    p.add_argument("--codeqa-path",        default=None)
    p.add_argument("--context-window",     type=int, default=100_000,
                   help="λ-RLM context window in chars (default 100k ≈ 25k tokens)")
    p.add_argument("--max-depth",          type=int, default=2,
                   help="Normal RLM max_depth")
    p.add_argument("--max-iter",           type=int, default=10,
                   help="Normal RLM max_iterations")
    p.add_argument("--dry-run",            action="store_true",
                   help="Load datasets only, skip LLM calls")
    args = p.parse_args()
    args.methods = list(dict.fromkeys(args.methods))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    N = args.n_samples_per_bucket

    # ── Load datasets ─────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading datasets (up to {N} samples per bucket per dataset)...")
    all_samples: list[Sample] = []
    if "oolong"     in args.datasets: all_samples += load_oolong(N)
    if "sniah"      in args.datasets: all_samples += load_sniah(N, args.sniah_path)
    if "browsecomp" in args.datasets: all_samples += load_browsecomp(N)
    if "codeqa"     in args.datasets: all_samples += load_codeqa(N, args.codeqa_path)

    if not all_samples:
        print("No samples loaded. Exiting.")
        return

    datasets_present = sorted({s.dataset for s in all_samples})
    print(f"\n  Total samples: {len(all_samples)}  across datasets: {datasets_present}")
    print(f"  Bucket distribution:")
    for ds in datasets_present:
        counts = defaultdict(int)
        for s in all_samples:
            if s.dataset == ds:
                counts[s.bin_label] += 1
        row = "  ".join(f"{b}:{counts.get(b,0):2d}" for b in BIN_LABELS)
        print(f"    {ds:<15} {row}")

    if args.dry_run:
        print("\n[DRY RUN] Skipping LLM calls.")
        for s in all_samples[:4]:
            print(f"  [{s.dataset}/{s.bin_label}] ctx={s.ctx_len:,}c  q={s.question[:70]!r}")
        return

    # ── Build runners ─────────────────────────────────────────────────────────
    print("\n[2/4] Building runners...")

    base_url = getattr(args, "base_url", None)
    api_key  = args.api_key
    if api_key is None:
        if base_url and "nvidia" in base_url:
            api_key = os.environ.get("NVIDIA_API_KEY")
        else:
            api_key = os.environ.get("OPENAI_API_KEY")

    backend_kwargs: dict[str, Any] = {
        "model_name":  args.model,
        "temperature": 0.6,
        "top_p":       0.7,
        "max_tokens":  4096,
        "stream":      True,
    }
    if api_key:
        backend_kwargs["api_key"] = api_key
    if base_url:
        backend_kwargs["base_url"] = base_url

    runners: dict[str, Any] = {}
    for method in args.methods:
        if method == "rlm":
            from rlm import RLM
            runners["rlm"] = RLM(
                backend=args.backend,
                backend_kwargs=backend_kwargs.copy(),
                environment="local",
                max_depth=args.max_depth,
                max_iterations=args.max_iter,
            ).completion
            print(f"  Normal RLM : {args.model}  depth={args.max_depth}  iter={args.max_iter}")
        elif method == "lambda_rlm":
            from rlm import LambdaRLM
            runners["lambda_rlm"] = LambdaRLM(
                backend=args.backend,
                backend_kwargs=backend_kwargs.copy(),
                context_window_chars=args.context_window,
                verbose=False,
            ).completion
            print(f"  λ-RLM      : {args.model}  ctx_window={args.context_window:,}c")

    # ── Run ───────────────────────────────────────────────────────────────────
    total_runs = len(all_samples) * len(runners)
    print(f"\n[3/4] Running benchmark ({len(all_samples)} samples × {len(runners)} methods"
          f" = {total_runs} total calls)...")

    all_results: list[Result] = []
    run_i = 0
    for method, runner in runners.items():
        print(f"\n  ── {_METHOD_LABEL.get(method, method)} ──")
        for s in all_samples:
            run_i += 1
            print(f"  [{run_i:>4}/{total_runs}] {s.dataset}/{s.bin_label} "
                  f"({s.ctx_len:>8,}c)  {s.question[:55]!r}", end="", flush=True)
            result = run_sample(method, runner, s)
            all_results.append(result)
            status = f"  f1={result.f1:.2f}  {result.latency:.1f}s"
            if result.error:
                status += f"  ERR:{result.error[:50]}"
            print(status)

    # ── Aggregate & report ────────────────────────────────────────────────────
    print("\n[4/4] Aggregating and saving results...")
    stats = aggregate(all_results)

    print_table(stats, datasets_present, args.methods)

    # Save results
    (output_dir / "results.json").write_text(
        json.dumps([asdict(r) for r in all_results], indent=2))
    (output_dir / "stats.json").write_text(
        json.dumps([asdict(s) for s in stats], indent=2))

    # Compute and save overall averages
    overall: list[dict] = []
    for ds in datasets_present:
        for method in args.methods:
            avg = _avg_row(stats, ds, method)
            if avg:
                overall.append(asdict(avg))
    (output_dir / "averages.json").write_text(json.dumps(overall, indent=2))

    print(f"\n  Results  → {output_dir / 'results.json'}")
    print(f"  Stats    → {output_dir / 'stats.json'}")
    print(f"  Averages → {output_dir / 'averages.json'}")

    make_plots(stats, datasets_present, args.methods, output_dir,
               model_label=args.model.split("/")[-1])

    # ── Print final average summary ───────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  FINAL AVERAGES (macro-avg across non-empty context-length buckets)")
    print(f"{'═'*70}")
    print(f"  {'Dataset':<15}  {'Method':<14}  {'F1':>8}  {'Contains':>10}  {'Latency':>9}")
    print(f"  {'-'*15}  {'-'*14}  {'-'*8}  {'-'*10}  {'-'*9}")
    for ds in datasets_present:
        for method in args.methods:
            avg = _avg_row(stats, ds, method)
            if avg:
                print(f"  {ds:<15}  {_METHOD_LABEL.get(method,method):<14}  "
                      f"{avg.f1*100:>7.2f}%  {avg.contains*100:>9.2f}%  "
                      f"{avg.latency:>8.2f}s")
    print()


if __name__ == "__main__":
    main()
