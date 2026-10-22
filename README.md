# pretrain-data-pipeline

A pipeline for curating language-model pre-training text from [Common Crawl](https://commoncrawl.org). It downloads raw WARC segments, extracts the main text of each page, removes exact and near-duplicate documents, keeps the target languages, scores document quality with a fastText classifier, and reports coverage, diversity and source-distribution metrics for the resulting corpus.

Every stage runs on a single machine. The heavy stages (deduplication and filtering) also run as PySpark jobs that partition documents by domain.

```
 Common Crawl            ┌──────────┐   ┌─────────┐   ┌─────────┐   ┌─────────────┐   ┌──────────────┐   ┌─────────┐
 warc.paths.gz  ───────► │ download │──►│ extract │──►│  dedup  │──►│ lang_filter │──►│ quality      │──►│ metrics │
 (data.commoncrawl.org)  └──────────┘   └─────────┘   └─────────┘   └─────────────┘   └──────────────┘   └─────────┘
                          WARC .gz       trafilatura   exact SHA-1   fastText          Gopher rules +      coverage,
                          (full or       boilerplate   + MinHash-    lid.176.bin       fastText hq/lq      distinct-n,
                          streamed       removal       LSH           (176 langs)       classifier          sources
                          sample)                      ─────────── Spark: domain-partitioned jobs ──────
```

## Pipeline stages

| Stage | Module | What it does | Output (`work/…`) |
| --- | --- | --- | --- |
| **download** | `src/downloader.py` | Resolves the crawl (or `latest`) and reads its `warc.paths.gz` manifest. Fetches WARC segments with retries and backoff. It either downloads full, resumable files or streams just the first *N* records of each segment, rewritten as a valid WARC. | `warc/*.warc.gz` |
| **extract** | `src/extractor.py` | Iterates HTTP 200 HTML responses with `warcio`, decodes the charset, and strips boilerplate with `trafilatura` (navigation, footers, cookie banners, ads). Drops pages with too little text. Runs one process per WARC file. | `extracted/*.jsonl.gz` |
| **dedup** | `src/dedup.py` | Removes exact duplicates (SHA-1 of normalized text). Then finds near-duplicates with MinHash over word 5-gram shingles and a `datasketch` MinHashLSH index. Each candidate is verified with estimated Jaccard ≥ `threshold`, and the first occurrence is kept. | `deduped/` |
| **lang_filter** | `src/lang_filter.py` | Runs fastText `lid.176.bin` language ID (downloaded on first use). Keeps `target_languages` above `min_confidence`. | `lang_filtered/` |
| **quality** | `src/quality_scorer.py` | Applies Gopher-style heuristic rules (word count, mean word length, symbol ratio, bullet and ellipsis lines, alphabetic ratio, repeated lines). Then a fastText classifier gives `P(high quality)`, and documents below `cutoff` are dropped. | `quality_filtered/` |
| **metrics** | `src/metrics.py` | Computes coverage across domains, token-level diversity and source distribution, plus document retention through every stage. | `metrics/metrics.json` |

Each stage writes gzipped JSONL shards and a `_report.json` with its counts. Each stage can be rerun on its own.

Document schema: `{"id", "url", "domain", "tld", "warc_date", "source", "text"}`. Later stages add `lang`, `lang_score` and `quality_score`.

### Quality classifier

`scripts/train_quality_model.py` trains a binary fastText model, following the approach used for GPT-3 and CCNet-style filtering:

- **Positives** (`__label__hq`): curated reference text. By default this is a random sample of English Wikipedia articles from the Hugging Face `wikimedia/wikipedia` dataset, cached under `models/`. Any JSONL file with a `text` field also works.
- **Negatives** (`__label__lq`): target-language pages from a *held-out* sample of the last segments of the crawl. The pipeline processes segments from the start of the manifest, so the model never scores documents it was trained on. In testing, scoring a model's own training negatives made it about 7× stricter than it was on unseen pages.

The classes are balanced, and both are normalized identically at training and inference time (`classifier_text`). The script reports validation precision, recall and AUC to `models/quality_model.json`. If the model is missing, the quality stage trains it automatically.

### Metrics

| Group | Metrics |
| --- | --- |
| Coverage | unique domains and TLDs; docs per domain (mean, p50, p90, p99, max); singleton-domain share; top-k domain share; Shannon entropy of the domain distribution (raw and normalized); effective number of domains (`exp(H)`); Gini coefficient; share of extracted domains still present after filtering |
| Diversity | distinct-n ratios for n = 1–4 (unique n-grams / total n-grams), corpus-level and mean per document, over a reservoir sample; vocabulary size; type-token ratio |
| Source distribution | documents and token share by TLD, language and WARC source; tokens-per-document quantiles; quality score quantiles; per-stage retention |

## Spark

`spark/spark_dedup.py` and `spark/spark_filter.py` are the distributed counterparts of the dedup, language-filter and quality stages.

**Custom partitioning by domain** (`src/spark_utils.py: DomainPartitioner`). Documents are keyed by `(domain, salt)` and routed with a CRC32-based partition function. CRC32 is used because Python's `hash()` is randomized per process. This keeps every page of a site in one partition, so per-partition work sees whole domains, and output shards are grouped by domain. Domains holding more than `hot_domain_share` of the corpus are salted across up to `max_salt` partitions in proportion to their size, so one large site doesn't create a straggler task.

**`spark_dedup.py`** runs four steps:

1. **Exact dedup.** `reduceByKey` on the normalized-text SHA-1, keeping the smallest id.
2. **Domain partitioning.** Apply the partitioner described above.
3. **Local near-dedup.** `mapPartitions` computes MinHash signatures and drops near-duplicates inside each partition. Most near-duplicates are within a site (templated pages, pagination, mirrored articles), so they are removed before any shuffle.
4. **Global near-dedup.** Signatures are split into `b` bands of `r` rows, chosen for the threshold the same way `datasketch` does. Grouping by `(band, band-hash)` produces cross-partition candidate pairs. Each pair is verified by estimated Jaccard similarity, and the document with the larger id is dropped.

**`spark_filter.py`** runs language ID, heuristics and the quality classifier in one `mapPartitions` pass over domain-partitioned data:

- `--steps lang`, `--steps quality`, or both.
- Model files are shipped with `SparkContext.addFile` and loaded once per Python worker.
- The report includes drop-reason counts and per-domain retention.

## Running locally

Requires Python 3.10+. Java 17+ is needed only for Spark.

```bash
git clone https://github.com/swan28pande/pretrain-data-pipeline.git
cd pretrain-data-pipeline
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Full pipeline with the defaults in config.yaml (2 sampled segments of the latest crawl)
python -m src.pipeline
# or
scripts/run_pipeline.sh local
```

Run or resume individual stages:

```bash
python -m src.pipeline --stages download,extract
python -m src.pipeline --from-stage dedup
python -m src.dedup --input work/extracted          # every stage module is also a CLI
python scripts/train_quality_model.py               # (re)train the quality classifier
python -m src.metrics --input work/quality_filtered --reference work/extracted
```

Scale up by editing `config.yaml`. For example, set `download.num_segments: 100` and `download.max_records_per_segment: null` to download complete segments.

## Running on Spark

```bash
# Dedup + filtering as Spark jobs inside the pipeline (spark.master from config.yaml, default local[*])
python -m src.pipeline --engine spark
scripts/run_pipeline.sh spark

# On a cluster: each stage is submitted with spark-submit
SPARK_MASTER=yarn SPARK_SUBMIT_ARGS="--num-executors 50 --executor-memory 8g" scripts/run_pipeline.sh submit

# Or submit the jobs directly (input/output can be HDFS/S3 URIs)
zip -qr dist/pipeline.zip src spark -i '*.py'
spark-submit --master yarn --py-files dist/pipeline.zip spark/spark_dedup.py \
    --config config.yaml --input s3a://bucket/extracted --output s3a://bucket/deduped
spark-submit --master yarn --py-files dist/pipeline.zip spark/spark_filter.py \
    --config config.yaml --steps lang,quality --input s3a://bucket/deduped --output s3a://bucket/filtered
```

Executors need the Python dependencies from `requirements.txt`. Use a cluster image or `--archives` with a packed environment. `create_spark` also ships the `src` and `spark` packages to executors automatically.

## Sample run

These numbers come from a development-sized run on `CC-MAIN-2026-43`: 10 segments, streaming the first 6,000 WARC records of each (20,000 HTTP responses), on an 8-core laptop. The local and Spark engines produced the same final corpus: 321 documents from 272 domains.

| Stage | In | Out | Kept | Notes |
| --- | ---: | ---: | ---: | --- |
| extract | 20,000 responses | 11,309 | 56.5% | the rest were non-HTML, non-200, or under 200 characters of main text |
| dedup | 11,309 | 10,828 | 95.7% | 363 exact and 118 near-duplicates (mostly the same spam page across many domains) |
| lang_filter (`en`) | 10,828 | 3,162 | 29.2% | top languages: en 3,452 · zh 1,904 · ru 1,102 · ja 480 · fr 465 |
| quality (cutoff 0.3) | 3,162 | 321 | 10.2% | 496 failed heuristics; 2,345 scored below the cutoff |

Quality classifier: 3,000 Wikipedia positives and 2,843 held-out crawl negatives. Validation AUC is 0.988, with precision 0.91 and recall 0.99 at the 0.3 cutoff.

Final-corpus metrics (`work/metrics/metrics.json`):

| Metric | Value |
| --- | --- |
| Documents / tokens | 321 / 218,443 (median 276 tokens per doc) |
| Unique domains / TLDs | 272 / 46 |
| Domain entropy (normalized) / effective domains / Gini | 0.990 / 257 / 0.131 |
| Top-20 domain share | 14.3% |
| Distinct-1 / 2 / 3 / 4 (corpus) | 0.109 / 0.575 / 0.827 / 0.896 |
| TLD share (docs) | .com 48% · .org 16% · .uk 4% · .net 3% · .edu 2% |
| Quality score p10 / p50 / p90 | 0.33 / 0.52 / 0.87 |

At this scale the local engine is faster: about 40 seconds from dedup to metrics, against about 3.5 minutes on Spark `local[*]`. Spark pays per-task scheduling costs, and each Python worker loads the 126 MB `lid.176.bin`. Spark is for corpora that don't fit on one machine, where the same jobs run unchanged on a cluster.

## Docker

```bash
docker build -t pretrain-data-pipeline .
docker run --rm -v "$PWD/work:/app/work" -v "$PWD/models:/app/models" pretrain-data-pipeline local
docker run --rm -v "$PWD/work:/app/work" -v "$PWD/models:/app/models" pretrain-data-pipeline spark
```

The image includes Java 17 for PySpark. Mount `work/` and `models/` to keep outputs and downloaded models between runs.

## Configuration

All parameters live in [`config.yaml`](config.yaml):

| Section | Key parameters |
| --- | --- |
| `pipeline` | `work_dir`, `engine` (`local`/`spark`), `num_workers`, `stages` |
| `download` | `crawl` (`latest` or `CC-MAIN-YYYY-WW`), `num_segments`, `segment_offset`, `max_records_per_segment`, retries |
| `extract` | `min_text_chars`, `max_html_bytes`, trafilatura `favor_precision` / tables / comments |
| `dedup` | `exact`, `threshold` (Jaccard similarity, default 0.8), `num_perm`, `ngram_size`, `seed` |
| `lang_filter` | `target_languages` (default `[en]`), `min_confidence`, model path / URL |
| `quality` | `cutoff` (P(hq) threshold), `model_path`, `heuristics.*` rule thresholds |
| `quality_training` | positive / negative sources and counts, fastText hyperparameters |
| `metrics` | `ngram_orders`, `diversity_sample_docs`, `top_k_domains` |
| `spark` | `master`, `num_partitions`, `hot_domain_share`, `max_salt`, extra `conf` |

## Project structure

```
pretrain-data-pipeline/
├── config.yaml
├── src/
│   ├── downloader.py       # Common Crawl WARC segments
│   ├── extractor.py        # WARC -> main text (trafilatura)
│   ├── dedup.py            # exact + MinHash-LSH dedup (datasketch)
│   ├── lang_filter.py      # fastText lid.176 language ID
│   ├── quality_scorer.py   # heuristic rules + fastText quality classifier
│   ├── metrics.py          # coverage, diversity, source distribution
│   ├── pipeline.py         # orchestration, local or Spark engine
│   ├── spark_utils.py      # SparkSession, DomainPartitioner, I/O helpers
│   └── utils.py            # config, JSONL I/O, URL and text helpers
├── spark/
│   ├── spark_dedup.py      # distributed MinHash dedup
│   └── spark_filter.py     # distributed language + quality filtering
├── scripts/
│   ├── train_quality_model.py
│   └── run_pipeline.sh
├── requirements.txt
└── Dockerfile
```
