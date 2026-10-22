"""Distributed language + quality filtering with PySpark.

Documents are partitioned by domain with the custom `DomainPartitioner`, then
each partition is filtered in a single pass (`mapPartitions`):

  * `lang`:    fastText lid.176 language ID; keep target languages above the
               confidence threshold;
  * `quality`: Gopher-style heuristic rules, then the fastText quality
               classifier; keep documents with P(hq) >= cutoff.

Model files are shipped to executors with `SparkContext.addFile` and loaded once
per Python worker (cached at module level and reused across partitions). Output
shards stay grouped by domain, and the job writes drop-reason counts and
per-domain retention to `_report.json`.

Usage:
    spark-submit spark/spark_filter.py --config config.yaml --steps lang,quality \
        --input work/deduped --output work/quality_filtered
"""

from __future__ import annotations

import argparse
import sys
from operator import add
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.spark_utils import DomainPartitioner, create_spark, input_glob, is_local_path, save_jsonl  # noqa: E402
from src.utils import get_logger, load_config, resolve_path, stage_dir, write_json  # noqa: E402

log = get_logger("spark_filter")

KEPT = "kept"


def _filter_partition(steps: tuple[str, ...], params: dict[str, Any]):
    def run(rows: Iterator[tuple[Any, dict[str, Any]]]):
        from src.quality_scorer import heuristic_failures
        from src.spark_utils import worker_model

        lid = worker_model("lang", params["lang_model"]) if "lang" in steps else None
        clf = worker_model("quality", params["quality_model"]) if "quality" in steps else None
        for _, doc in rows:
            reason = None
            if lid is not None:
                lang, score = lid.predict(doc["text"])
                if lang not in params["target_languages"]:
                    reason = "wrong_language"
                elif score < params["min_confidence"]:
                    reason = "low_confidence"
                else:
                    doc["lang"], doc["lang_score"] = lang, round(score, 4)
            if reason is None and clf is not None:
                failures = heuristic_failures(doc["text"], params["heuristics"])
                if failures:
                    reason = f"heuristic:{failures[0]}"
                else:
                    q = clf.score(doc["text"])
                    if q < params["cutoff"]:
                        reason = "low_quality_score"
                    else:
                        doc["quality_score"] = round(q, 4)
            yield reason or KEPT, doc

    return run


def run(spark, cfg: dict[str, Any], input_path: str, output_path: str,
        steps: tuple[str, ...] = ("lang", "quality")) -> dict[str, Any]:
    import json

    sc = spark.sparkContext
    scfg, lcfg, qcfg = cfg["spark"], cfg["lang_filter"], cfg["quality"]
    params: dict[str, Any] = {
        "target_languages": set(lcfg["target_languages"]), "min_confidence": lcfg["min_confidence"],
        "heuristics": qcfg["heuristics"], "cutoff": qcfg["cutoff"],
    }
    if "lang" in steps:
        from src.lang_filter import ensure_model

        lang_model = ensure_model(resolve_path(cfg, lcfg["model_path"]), lcfg["model_url"])
        sc.addFile(str(lang_model))
        params["lang_model"] = lang_model.name
    if "quality" in steps:
        quality_model = resolve_path(cfg, qcfg["model_path"])
        if not quality_model.exists():
            raise FileNotFoundError(f"{quality_model} not found: run scripts/train_quality_model.py first")
        sc.addFile(str(quality_model))
        params["quality_model"] = quality_model.name

    docs = sc.textFile(input_glob(input_path)).map(json.loads)
    partitioner = DomainPartitioner.fit(docs, scfg["num_partitions"], scfg["hot_domain_share"], scfg["max_salt"])
    results = (partitioner.partition(docs)
               .mapPartitions(_filter_partition(steps, params), preservesPartitioning=True)
               .cache())

    reasons = results.map(lambda r: (r[0], 1)).reduceByKey(add).collectAsMap()
    per_domain = (results.map(lambda r: (r[1]["domain"], (int(r[0] == KEPT), 1)))
                  .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])))
    top_domains = per_domain.takeOrdered(20, key=lambda kv: -kv[1][1])
    domains_in = per_domain.count()
    domains_kept = per_domain.filter(lambda kv: kv[1][0] > 0).count()

    save_jsonl(results.filter(lambda r: r[0] == KEPT).map(lambda r: r[1]), output_path)

    n_input = sum(reasons.values())
    report = {
        "stage": "quality" if "quality" in steps else "lang_filter", "engine": "spark", "steps": list(steps),
        "input": n_input, "kept": reasons.get(KEPT, 0),
        "drop_reasons": {k: v for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]) if k != KEPT},
        "domains_in": domains_in, "domains_kept": domains_kept,
        "top_domains": [{"domain": d, "input": t, "kept": k} for d, (k, t) in top_domains],
        "num_partitions": scfg["num_partitions"], "salted_domains": partitioner.salts,
    }
    if is_local_path(output_path):
        write_json(Path(output_path.removeprefix("file://")) / "_report.json", report)
    log.info("Spark filter %s: %d in, %d kept, %d/%d domains kept; drops: %s", "+".join(steps), n_input,
             report["kept"], domains_kept, domains_in,
             ", ".join(f"{k}={v}" for k, v in report["drop_reasons"].items()))
    results.unpersist()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--steps", default="lang,quality", help="comma-separated subset of: lang, quality")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    steps = tuple(s.strip() for s in args.steps.split(",") if s.strip())
    if not steps or set(steps) - {"lang", "quality"}:
        raise SystemExit("--steps must be a subset of: lang, quality")
    default_out = "quality_filtered" if "quality" in steps else "lang_filtered"
    input_path = str(resolve_path(cfg, args.input)) if args.input and is_local_path(args.input) else (
        args.input or str(stage_dir(cfg, "deduped")))
    output_path = str(resolve_path(cfg, args.output)) if args.output and is_local_path(args.output) else (
        args.output or str(stage_dir(cfg, default_out)))
    spark = create_spark(cfg)
    try:
        run(spark, cfg, input_path, output_path, steps)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
