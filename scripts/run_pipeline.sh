#!/usr/bin/env bash
# Run the pre-training data pipeline.
#
#   scripts/run_pipeline.sh local  [pipeline args]   # every stage on this machine
#   scripts/run_pipeline.sh spark  [pipeline args]   # dedup + filtering on Spark (spark.master from config.yaml)
#   scripts/run_pipeline.sh submit                    # each Spark job via spark-submit (for a cluster)
#
# Environment:
#   CONFIG             config file (default: config.yaml)
#   PYTHON             python interpreter (default: .venv/bin/python if present, else python3)
#   SPARK_MASTER       --master for spark-submit (default: local[*])
#   SPARK_SUBMIT_ARGS  extra spark-submit arguments, e.g. "--num-executors 20 --executor-memory 8g"
set -euo pipefail

cd "$(dirname "$0")/.."
MODE="${1:-local}"
shift || true
CONFIG="${CONFIG:-config.yaml}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi
WORK_DIR="$("$PYTHON" -c "from src.utils import load_config, resolve_path; c = load_config('$CONFIG'); print(resolve_path(c, c['pipeline']['work_dir']))")"
QUALITY_MODEL="$("$PYTHON" -c "from src.utils import load_config, resolve_path; c = load_config('$CONFIG'); print(resolve_path(c, c['quality']['model_path']))")"

case "$MODE" in
  local)
    exec "$PYTHON" -m src.pipeline --config "$CONFIG" --engine local "$@"
    ;;
  spark)
    exec "$PYTHON" -m src.pipeline --config "$CONFIG" --engine spark "$@"
    ;;
  submit)
    command -v spark-submit >/dev/null || { echo "spark-submit not found on PATH" >&2; exit 1; }
    mkdir -p dist
    rm -f dist/pipeline.zip
    zip -qr dist/pipeline.zip src spark -i '*.py'
    SUBMIT=(spark-submit --master "${SPARK_MASTER:-local[*]}" --py-files dist/pipeline.zip ${SPARK_SUBMIT_ARGS:-})

    "$PYTHON" -m src.pipeline --config "$CONFIG" --stages download,extract
    "${SUBMIT[@]}" spark/spark_dedup.py --config "$CONFIG" \
      --input "$WORK_DIR/extracted" --output "$WORK_DIR/deduped"
    "${SUBMIT[@]}" spark/spark_filter.py --config "$CONFIG" --steps lang \
      --input "$WORK_DIR/deduped" --output "$WORK_DIR/lang_filtered"
    if [[ ! -f "$QUALITY_MODEL" ]]; then
      "$PYTHON" scripts/train_quality_model.py --config "$CONFIG"
    fi
    "${SUBMIT[@]}" spark/spark_filter.py --config "$CONFIG" --steps quality \
      --input "$WORK_DIR/lang_filtered" --output "$WORK_DIR/quality_filtered"
    "$PYTHON" -m src.pipeline --config "$CONFIG" --stages metrics
    ;;
  *)
    echo "usage: $0 {local|spark|submit} [pipeline args]" >&2
    exit 2
    ;;
esac
