"""Orchestrates the full curation pipeline.

    download -> extract -> dedup -> lang_filter -> quality -> metrics

Each stage reads the previous stage's output directory under `pipeline.work_dir`
and writes its own shards plus a `_report.json`, so any stage can be re-run on
its own. With `--engine spark` (or `pipeline.engine: spark`), dedup, language
filtering and quality filtering run as distributed PySpark jobs
(`spark/spark_dedup.py`, `spark/spark_filter.py`) in one shared SparkSession;
download, extraction and metrics stay local.

If the quality model does not exist yet, the quality stage trains it first
(`scripts/train_quality_model.py`: Wikipedia positives, held-out crawl negatives).

Usage:
    python -m src.pipeline                                  # all stages, local engine
    python -m src.pipeline --engine spark                   # distributed dedup + filtering
    python -m src.pipeline --stages dedup,lang_filter       # a subset
    python -m src.pipeline --from-stage quality             # resume from a stage
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .utils import get_logger, load_config, resolve_path, stage_dir, write_json

log = get_logger("pipeline")

STAGES = ["download", "extract", "dedup", "lang_filter", "quality", "metrics"]


class Pipeline:
    def __init__(self, cfg: dict[str, Any], engine: str | None = None):
        self.cfg = cfg
        self.engine = engine or cfg["pipeline"].get("engine", "local")
        if self.engine not in ("local", "spark"):
            raise ValueError(f"engine must be 'local' or 'spark', got {self.engine!r}")
        self._spark = None

    # ---------------------------------------------------------------- spark session
    @property
    def spark(self):
        if self._spark is None:
            from .spark_utils import create_spark

            self._spark = create_spark(self.cfg)
            log.info("Started Spark (%s, %d partitions)", self._spark.sparkContext.master,
                     self.cfg["spark"]["num_partitions"])
        return self._spark

    def close(self) -> None:
        if self._spark is not None:
            self._spark.stop()
            self._spark = None

    # ---------------------------------------------------------------- stages
    def download(self) -> dict[str, Any]:
        from .downloader import download

        files = download(self.cfg)
        return {"stage": "download", "files": [f.name for f in files]}

    def extract(self) -> dict[str, Any]:
        from .extractor import extract

        return extract(self.cfg)

    def dedup(self) -> dict[str, Any]:
        if self.engine == "spark":
            from spark.spark_dedup import run

            return run(self.spark, self.cfg, str(stage_dir(self.cfg, "extracted")),
                       str(stage_dir(self.cfg, "deduped")))
        from .dedup import dedup

        return dedup(self.cfg)

    def lang_filter(self) -> dict[str, Any]:
        if self.engine == "spark":
            from spark.spark_filter import run

            return run(self.spark, self.cfg, str(stage_dir(self.cfg, "deduped")),
                       str(stage_dir(self.cfg, "lang_filtered")), steps=("lang",))
        from .lang_filter import lang_filter

        return lang_filter(self.cfg)

    def quality(self) -> dict[str, Any]:
        self._ensure_quality_model()
        if self.engine == "spark":
            from spark.spark_filter import run

            return run(self.spark, self.cfg, str(stage_dir(self.cfg, "lang_filtered")),
                       str(stage_dir(self.cfg, "quality_filtered")), steps=("quality",))
        from .quality_scorer import quality_filter

        return quality_filter(self.cfg)

    def metrics(self) -> dict[str, Any]:
        from .metrics import compute_metrics

        m = compute_metrics(self.cfg)
        return {"stage": "metrics", "documents": m["coverage"]["documents"],
                "unique_domains": m["coverage"]["unique_domains"]}

    def _ensure_quality_model(self) -> None:
        model = resolve_path(self.cfg, self.cfg["quality"]["model_path"])
        if model.exists():
            return
        log.info("Quality model %s not found: training it now", model)
        script = Path(__file__).resolve().parents[1] / "scripts" / "train_quality_model.py"
        subprocess.run([sys.executable, str(script), "--config", self.cfg["_config_path"]], check=True)

    # ---------------------------------------------------------------- driver
    def run(self, stages: list[str]) -> dict[str, Any]:
        summary: dict[str, Any] = {"engine": self.engine, "stages": []}
        t_start = time.time()
        try:
            for stage in stages:
                log.info("=== %s (%s) ===", stage, self.engine if stage in ("dedup", "lang_filter", "quality")
                         else "local")
                t0 = time.time()
                report = getattr(self, stage)()
                summary["stages"].append({
                    "stage": stage, "seconds": round(time.time() - t0, 1),
                    **{k: v for k, v in report.items() if isinstance(v, (int, float, str)) and k != "stage"},
                })
        finally:
            self.close()
        summary["total_seconds"] = round(time.time() - t_start, 1)
        write_json(stage_dir(self.cfg, "metrics") / "pipeline_summary.json", summary)
        for row in summary["stages"]:
            counts = ", ".join(f"{k}={row[k]}" for k in ("input", "documents", "kept") if k in row)
            log.info("%-12s %6.1fs  %s", row["stage"], row["seconds"], counts)
        return summary


def parse_stages(args: argparse.Namespace, cfg: dict[str, Any]) -> list[str]:
    stages = [s.strip() for s in args.stages.split(",")] if args.stages else list(cfg["pipeline"]["stages"])
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise SystemExit(f"Unknown stages: {', '.join(sorted(unknown))}. Valid: {', '.join(STAGES)}")
    if args.from_stage:
        stages = [s for s in STAGES[STAGES.index(args.from_stage):] if s in stages]
    return [s for s in STAGES if s in stages]  # always run in pipeline order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--engine", choices=["local", "spark"], default=None)
    parser.add_argument("--stages", default=None, help=f"comma-separated subset of: {','.join(STAGES)}")
    parser.add_argument("--from-stage", choices=STAGES, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    Pipeline(cfg, args.engine).run(parse_stages(args, cfg))


if __name__ == "__main__":
    main()
