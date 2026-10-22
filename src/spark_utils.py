"""Spark helpers: session creation and domain-aware custom partitioning.

`DomainPartitioner` routes every document of a domain to the same partition, so
per-partition work (local near-duplicate detection, per-domain statistics, model
loading) sees whole sites at once and output shards are grouped by domain.
Domains that hold more than `hot_domain_share` of the corpus are salted across
several partitions so a single huge site cannot create a straggler task.

Partition ids use CRC32 rather than Python's `hash()`, which is randomized per
process and would route the same key differently on different executors.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import zipfile
import zlib
from operator import add
from pathlib import Path
from typing import Any

from .utils import ROOT, get_logger

log = get_logger("spark")


def stable_hash(value: str) -> int:
    return zlib.crc32(value.encode("utf-8"))


class DomainPartitioner:
    """Partition function for `rdd.partitionBy(num_partitions, partitioner)` on (domain, salt) keys."""

    def __init__(self, num_partitions: int, salts: dict[str, int] | None = None):
        self.num_partitions = num_partitions
        self.salts = salts or {}

    def key(self, doc: dict[str, Any]) -> tuple[str, int]:
        domain = doc.get("domain", "")
        n = self.salts.get(domain, 1)
        salt = stable_hash(doc["id"]) % n if n > 1 else 0
        return domain, salt

    def __call__(self, key: tuple[str, int]) -> int:
        domain, salt = key
        return stable_hash(f"{domain}#{salt}") % self.num_partitions

    @classmethod
    def fit(cls, docs_rdd, num_partitions: int, hot_domain_share: float, max_salt: int) -> "DomainPartitioner":
        """Counts documents per domain and salts the hot ones in proportion to their size."""
        total = docs_rdd.count()
        threshold = max(int(total * hot_domain_share), 1)
        hot = (docs_rdd.map(lambda d: (d.get("domain", ""), 1))
               .reduceByKey(add)
               .filter(lambda kv: kv[1] > threshold)
               .collectAsMap())
        fair_share = max(total / num_partitions, 1)
        salts = {d: min(max_salt, max(2, math.ceil(c / fair_share))) for d, c in hot.items()}
        if salts:
            log.info("Salting %d hot domains: %s", len(salts),
                     ", ".join(f"{d}x{s}" for d, s in sorted(salts.items(), key=lambda kv: -kv[1])[:5]))
        return cls(num_partitions, salts)

    def partition(self, docs_rdd):
        """Returns a (key, doc) RDD partitioned by domain."""
        return docs_rdd.map(lambda d: (self.key(d), d)).partitionBy(self.num_partitions, self)


_WORKER_MODELS: dict[str, Any] = {}


def worker_model(kind: str, filename: str):
    """Loads a model shipped with `sc.addFile` once per Python worker.

    The cache lives in this importable module, so it survives across tasks when
    Spark reuses Python workers (spark.python.worker.reuse, on by default).
    """
    if kind not in _WORKER_MODELS:
        from pyspark import SparkFiles

        if kind == "lang":
            from .lang_filter import LanguageIdentifier

            _WORKER_MODELS[kind] = LanguageIdentifier(SparkFiles.get(filename))
        elif kind == "quality":
            from .quality_scorer import QualityClassifier

            _WORKER_MODELS[kind] = QualityClassifier(SparkFiles.get(filename))
        else:
            raise ValueError(f"unknown model kind {kind!r}")
    return _WORKER_MODELS[kind]


def _zip_packages() -> str:
    """Zips the `src` and `spark` packages so executors can import them (`sc.addPyFile`)."""
    path = Path(tempfile.gettempdir()) / f"pretrain_pkgs_{os.getpid()}.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for package in ("src", "spark"):
            for f in (ROOT / package).rglob("*.py"):
                zf.write(f, f.relative_to(ROOT))
    return str(path)


def create_spark(cfg: dict[str, Any]):
    from pyspark import SparkConf
    from pyspark.sql import SparkSession

    scfg = cfg["spark"]
    # Local mode: run Python workers with the driver's interpreter (same venv and version).
    # On a cluster, set PYSPARK_PYTHON to the executors' environment instead.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    conf = SparkConf()
    builder = SparkSession.builder.appName(scfg["app_name"])
    if not conf.contains("spark.master"):  # spark-submit --master takes precedence
        builder = builder.master(scfg["master"])
    for k, v in (scfg.get("conf") or {}).items():
        if not conf.contains(k):
            builder = builder.config(k, str(v))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    spark.sparkContext.addPyFile(_zip_packages())
    return spark


def is_local_path(path: str) -> bool:
    return "://" not in path or path.startswith("file://")


def prepare_output(path: str) -> None:
    """Spark refuses to overwrite with saveAsTextFile: clear a previous local output directory."""
    if is_local_path(path):
        import shutil

        p = Path(path.removeprefix("file://"))
        if p.exists():
            shutil.rmtree(p)


def save_jsonl(rdd, path: str) -> None:
    import json

    prepare_output(path)
    rdd.map(lambda d: json.dumps(d, ensure_ascii=False)).saveAsTextFile(
        path, compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec"
    )


def input_glob(path: str) -> str:
    """Accepts a directory of JSONL shards (ours or Spark part files) or an explicit glob."""
    if is_local_path(path) and Path(path).is_dir():
        p = Path(path)
        if list(p.glob("part-*")):
            return str(p / "part-*")
        return str(p / "*.jsonl*")
    return path
