"""Distributed exact + MinHash-LSH near-duplicate removal with PySpark.

Stages:
  1. Exact dedup: key by SHA-1 of normalized text, keep the smallest doc id.
  2. Domain partitioning: a custom partitioner groups each domain's documents in
     one partition (hot domains are salted across several).
  3. Local pass (mapPartitions): compute MinHash signatures and drop near-duplicates
     inside each partition with an in-memory LSH index. Most near-duplicates are
     intra-site (templated pages, pagination, mirrored articles), so this removes
     them before any shuffle.
  4. Global pass: split signatures into `b` bands of `r` rows (chosen for the
     Jaccard threshold), group by (band, band-hash) to find candidate pairs across
     partitions, verify each candidate with the estimated Jaccard similarity, and
     drop the document with the larger id of every verified pair.

Usage:
    spark-submit spark/spark_dedup.py --config config.yaml --input work/extracted --output work/deduped
    python spark/spark_dedup.py ...      # uses spark.master from config.yaml (local[*] by default)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.spark_utils import (DomainPartitioner, create_spark, input_glob, is_local_path,  # noqa: E402
                             save_jsonl)
from src.utils import get_logger, load_config, resolve_path, stage_dir, write_json  # noqa: E402

log = get_logger("spark_dedup")


def lsh_params(threshold: float, num_perm: int) -> tuple[int, int]:
    """(bands, rows) minimizing weighted false positive/negative probability, as datasketch does."""
    from datasketch.lsh import _optimal_param

    return _optimal_param(threshold, num_perm, 0.5, 0.5)


def _local_dedup(params: dict[str, Any]):
    """mapPartitions function: signatures + in-partition near-duplicate removal."""

    def run(rows: Iterator[tuple[Any, dict[str, Any]]]):
        from src.dedup import NearDuplicateIndex, minhash_signature

        index = NearDuplicateIndex(params["threshold"], params["num_perm"], params["seed"])
        # Sort for a deterministic "first occurrence" within the partition.
        for _, doc in sorted(rows, key=lambda kv: kv[1]["id"]):
            sig = minhash_signature(doc["text"], params["num_perm"], params["ngram_size"], params["seed"])
            if index.is_duplicate(doc["id"], sig) is None:
                yield doc["id"], (doc, sig)
            else:
                yield "__dup__", None

    return run


def _band_keys(bands: int, rows: int):
    def run(kv):
        doc_id, (_, sig) = kv
        for b in range(bands):
            digest = hashlib.blake2b(sig[b * rows:(b + 1) * rows].tobytes(), digest_size=8).hexdigest()
            yield (b, digest), doc_id

    return run


def _bucket_edges(kv):
    """Star edges from every bucket member to the bucket's smallest id."""
    ids = sorted(set(kv[1]))
    for other in ids[1:]:
        yield other, ids[0]


def run(spark, cfg: dict[str, Any], input_path: str, output_path: str) -> dict[str, Any]:
    import numpy as np

    sc = spark.sparkContext
    dcfg, scfg = cfg["dedup"], cfg["spark"]
    params = {k: dcfg[k] for k in ("threshold", "num_perm", "ngram_size", "seed")}
    from src.dedup import exact_hash
    import json

    docs = sc.textFile(input_glob(input_path)).map(json.loads)
    n_input = docs.count()

    if dcfg["exact"]:
        docs = (docs.map(lambda d: (exact_hash(d["text"]), d))
                .reduceByKey(lambda a, b: a if a["id"] <= b["id"] else b)
                .values())
    docs = docs.cache()
    n_after_exact = docs.count()

    partitioner = DomainPartitioner.fit(docs, scfg["num_partitions"], scfg["hot_domain_share"], scfg["max_salt"])
    local = partitioner.partition(docs).mapPartitions(_local_dedup(params), preservesPartitioning=True).cache()
    n_local_dups = local.filter(lambda kv: kv[0] == "__dup__").count()
    sigs = local.filter(lambda kv: kv[0] != "__dup__")

    bands, rows = lsh_params(params["threshold"], params["num_perm"])
    threshold = params["threshold"]
    candidates = (sigs.flatMap(_band_keys(bands, rows))
                  .groupByKey()
                  .filter(lambda kv: len(kv[1]) > 1)
                  .flatMap(_bucket_edges)
                  .distinct())
    sig_only = sigs.mapValues(lambda v: v[1])
    duplicates = (candidates.join(sig_only)                                  # (a, (m, sig_a))
                  .map(lambda kv: (kv[1][0], (kv[0], kv[1][1])))           # (m, (a, sig_a))
                  .join(sig_only)                                          # (m, ((a, sig_a), sig_m))
                  .filter(lambda kv: float(np.mean(kv[1][0][1] == kv[1][1])) >= threshold)
                  .map(lambda kv: (kv[1][0][0], None))
                  .distinct())
    kept = sigs.subtractByKey(duplicates).map(lambda kv: kv[1][0])

    save_jsonl(kept, output_path)
    n_kept = sc.textFile(input_glob(output_path)).count()

    report = {
        "stage": "dedup", "engine": "spark", "input": n_input,
        "exact_duplicates": n_input - n_after_exact,
        "near_duplicates_local": n_local_dups,
        "near_duplicates_global": n_after_exact - n_local_dups - n_kept,
        "kept": n_kept,
        "removed_pct": round(100 * (1 - n_kept / max(n_input, 1)), 2),
        "threshold": threshold, "num_perm": params["num_perm"], "lsh_bands": bands, "lsh_rows": rows,
        "num_partitions": scfg["num_partitions"], "salted_domains": partitioner.salts,
    }
    if is_local_path(output_path):
        write_json(Path(output_path.removeprefix("file://")) / "_report.json", report)
    log.info("Spark dedup: %d in, %d exact, %d local + %d global near duplicates, %d kept",
             n_input, report["exact_duplicates"], n_local_dups, report["near_duplicates_global"], n_kept)
    docs.unpersist()
    local.unpersist()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_path = str(resolve_path(cfg, args.input)) if args.input and is_local_path(args.input) else (
        args.input or str(stage_dir(cfg, "extracted")))
    output_path = str(resolve_path(cfg, args.output)) if args.output and is_local_path(args.output) else (
        args.output or str(stage_dir(cfg, "deduped")))
    spark = create_spark(cfg)
    try:
        run(spark, cfg, input_path, output_path)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
