"""Corpus metrics: domain coverage, token-level diversity and source distribution.

* Coverage: number of distinct domains/TLDs, documents per domain, top-k domain
  share, Shannon entropy of the domain distribution (and its normalized form),
  effective number of domains (exp(entropy)), Gini coefficient, and the share of
  input domains still represented after filtering.
* Diversity: distinct-n ratios (unique n-grams / total n-grams) over a reservoir
  sample of documents, both corpus-level and averaged per document, plus
  vocabulary size and type-token ratio.
* Source distribution: documents and tokens by TLD, language and WARC source,
  quality score quantiles, and document retention through every pipeline stage
  (read from each stage's `_report.json`).

Usage:
    python -m src.metrics [--config config.yaml] [--input work/quality_filtered] [--reference work/extracted]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .utils import get_logger, load_config, read_jsonl, resolve_path, stage_dir, tokenize, write_json

log = get_logger("metrics")

STAGES = ["extracted", "deduped", "lang_filtered", "quality_filtered"]


# --------------------------------------------------------------------------- distribution stats

def shannon_entropy(counts: Iterable[int]) -> float:
    counts = [c for c in counts if c > 0]
    total = sum(counts)
    return -sum(c / total * math.log(c / total) for c in counts) if total else 0.0


def gini(counts: Iterable[int]) -> float:
    """0 = documents spread evenly across domains, ->1 = concentrated in a few."""
    xs = sorted(counts)
    n, total = len(xs), sum(xs)
    if n == 0 or total == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cum) / (n * total) - (n + 1) / n


def quantiles(values: list[float], qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> dict[str, float]:
    if not values:
        return {}
    xs = sorted(values)
    return {f"p{int(q * 100)}": round(xs[min(int(q * len(xs)), len(xs) - 1)], 4) for q in qs}


# --------------------------------------------------------------------------- metric groups

def coverage_metrics(domain_counts: Counter, tld_counts: Counter, top_k: int,
                     reference_domains: set[str] | None = None) -> dict[str, Any]:
    n_docs = sum(domain_counts.values())
    per_domain = sorted(domain_counts.values())
    entropy = shannon_entropy(per_domain)
    max_entropy = math.log(len(per_domain)) if len(per_domain) > 1 else 1.0
    top = domain_counts.most_common(top_k)
    out = {
        "documents": n_docs,
        "unique_domains": len(domain_counts),
        "unique_tlds": len(tld_counts),
        "docs_per_domain": {
            "mean": round(n_docs / max(len(domain_counts), 1), 3),
            "max": per_domain[-1] if per_domain else 0,
            **quantiles(per_domain, (0.5, 0.9, 0.99)),
        },
        "singleton_domain_share": round(sum(1 for c in per_domain if c == 1) / max(len(per_domain), 1), 4),
        f"top_{top_k}_domain_share": round(sum(c for _, c in top) / max(n_docs, 1), 4),
        "domain_entropy": round(entropy, 4),
        "domain_entropy_normalized": round(entropy / max_entropy, 4),
        "effective_num_domains": round(math.exp(entropy), 1),
        "domain_gini": round(gini(per_domain), 4),
        "top_domains": [{"domain": d, "documents": c, "share": round(c / n_docs, 4)} for d, c in top],
    }
    if reference_domains:
        kept = set(domain_counts) & reference_domains
        out["reference_domains"] = len(reference_domains)
        out["domain_coverage_vs_reference"] = round(len(kept) / len(reference_domains), 4)
    return out


def diversity_metrics(sample_texts: list[str], orders: list[int]) -> dict[str, Any]:
    corpus_ngrams = {n: Counter() for n in orders}
    per_doc = {n: [] for n in orders}
    vocab: Counter = Counter()
    total_tokens = 0
    for text in sample_texts:
        tokens = tokenize(text)
        total_tokens += len(tokens)
        vocab.update(tokens)
        for n in orders:
            grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
            if grams:
                corpus_ngrams[n].update(grams)
                per_doc[n].append(len(set(grams)) / len(grams))

    distinct = {}
    for n in orders:
        total = sum(corpus_ngrams[n].values())
        distinct[f"distinct_{n}"] = {
            "corpus": round(len(corpus_ngrams[n]) / total, 4) if total else 0.0,
            "mean_per_doc": round(sum(per_doc[n]) / len(per_doc[n]), 4) if per_doc[n] else 0.0,
            "unique": len(corpus_ngrams[n]),
            "total": total,
        }
    return {
        "sample_documents": len(sample_texts),
        "sample_tokens": total_tokens,
        "vocabulary_size": len(vocab),
        "type_token_ratio": round(len(vocab) / max(total_tokens, 1), 4),
        **distinct,
    }


def stage_retention(work_dir: Path) -> list[dict[str, Any]]:
    rows = []
    first = None
    for stage in STAGES:
        report = work_dir / stage / "_report.json"
        if not report.exists():
            continue
        r = json.loads(report.read_text())
        out = r.get("documents", r.get("kept"))
        first = first or out
        rows.append({"stage": r.get("stage", stage), "input": r.get("input", r.get("responses")),
                     "output": out, "retained_vs_extracted": round(out / first, 4) if first else None})
    return rows


# --------------------------------------------------------------------------- driver

def compute_metrics(cfg: dict[str, Any], input_path: str | Path | None = None,
                    reference_path: str | Path | None = None) -> dict[str, Any]:
    mcfg = cfg["metrics"]
    rng = random.Random(0)
    source = input_path or stage_dir(cfg, "quality_filtered")

    domains: Counter = Counter()
    tlds: Counter = Counter()
    langs: Counter = Counter()
    sources: Counter = Counter()
    tokens_by_tld: Counter = Counter()
    quality_scores: list[float] = []
    doc_lengths: list[int] = []
    sample: list[str] = []
    k = mcfg["diversity_sample_docs"]

    for i, doc in enumerate(read_jsonl(source)):
        n_tokens = len(tokenize(doc["text"]))
        domains[doc["domain"]] += 1
        tlds[doc["tld"]] += 1
        langs[doc.get("lang", "unknown")] += 1
        sources[doc.get("source", "unknown")] += 1
        tokens_by_tld[doc["tld"]] += n_tokens
        doc_lengths.append(n_tokens)
        if "quality_score" in doc:
            quality_scores.append(doc["quality_score"])
        # Reservoir sampling keeps a uniform sample for the (memory-heavy) n-gram counts.
        if i < k:
            sample.append(doc["text"])
        elif (j := rng.randint(0, i)) < k:
            sample[j] = doc["text"]

    if not domains:
        raise ValueError(f"No documents found in {source}")

    reference_domains = None
    reference = reference_path or stage_dir(cfg, "extracted")
    if Path(reference).exists() and Path(reference).resolve() != Path(source).resolve():
        reference_domains = {d["domain"] for d in read_jsonl(reference)}

    total_docs = sum(domains.values())
    total_tokens = sum(doc_lengths)
    metrics = {
        "input": str(source),
        "coverage": coverage_metrics(domains, tlds, mcfg["top_k_domains"], reference_domains),
        "diversity": diversity_metrics(sample, mcfg["ngram_orders"]),
        "source_distribution": {
            "total_tokens": total_tokens,
            "tokens_per_doc": {"mean": round(total_tokens / total_docs, 1), **quantiles(doc_lengths)},
            "by_tld": [{"tld": t, "documents": c, "doc_share": round(c / total_docs, 4),
                        "token_share": round(tokens_by_tld[t] / max(total_tokens, 1), 4)}
                       for t, c in tlds.most_common(mcfg["top_k_domains"])],
            "by_language": dict(langs.most_common()),
            "by_warc_source": dict(sources.most_common()),
            "quality_score": quantiles(quality_scores),
        },
        "stage_retention": stage_retention(resolve_path(cfg, cfg["pipeline"]["work_dir"])),
    }

    out_dir = stage_dir(cfg, "metrics")
    write_json(out_dir / "metrics.json", metrics)
    cov, div = metrics["coverage"], metrics["diversity"]
    log.info("Coverage: %d docs, %d domains, %d TLDs, normalized entropy %.3f, Gini %.3f",
             cov["documents"], cov["unique_domains"], cov["unique_tlds"],
             cov["domain_entropy_normalized"], cov["domain_gini"])
    log.info("Diversity: %s", ", ".join(f"distinct-{n}={div[f'distinct_{n}']['corpus']:.3f}"
                                         for n in mcfg["ngram_orders"]))
    log.info("Wrote %s", out_dir / "metrics.json")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", default=None, help="documents to measure (default: final stage output)")
    parser.add_argument("--reference", default=None, help="corpus to compare domain coverage against")
    args = parser.parse_args()
    compute_metrics(load_config(args.config), args.input, args.reference)


if __name__ == "__main__":
    main()
