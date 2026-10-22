"""Train the fastText quality classifier.

Positives (`__label__hq`) are curated reference text: a random sample of English
Wikipedia articles (Hugging Face `wikimedia/wikipedia`, cached locally), or any JSONL file with a
`text` field (books, papers, a hand-curated set...). Negatives (`__label__lq`)
are crawled documents in the target language: by default a held-out sample from
the last segments of the crawl (never the segments the pipeline processes), or
any JSONL file/directory. The classifier therefore learns "how much does this page
look like curated text", the approach used for GPT-3 and CCNet-style filtering.

Writes the model to `quality.model_path` and a JSON report with validation
precision / recall / AUC next to it.

Usage:
    python scripts/train_quality_model.py [--config config.yaml] [--positives wikipedia|path.jsonl]
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.quality_scorer import HQ_LABEL, LQ_LABEL, classifier_text  # noqa: E402
from src.utils import JsonlWriter, get_logger, load_config, read_jsonl, resolve_path, write_json  # noqa: E402

log = get_logger("train_quality_model")

# Cleaned English Wikipedia (wikimedia/wikipedia on the Hugging Face Hub), read
# through the dataset viewer API in windows of up to 100 rows at random offsets.
WIKI_ROWS_API = "https://datasets-server.huggingface.co/rows"
WIKI_DATASET = {"dataset": "wikimedia/wikipedia", "config": "20231101.en", "split": "train"}
WIKI_NUM_ROWS = 6_407_814
WINDOW = 100
USER_AGENT = "pretrain-data-pipeline/0.1 (+https://github.com/swan28pande/pretrain-data-pipeline)"
MIN_WORDS = 100


def _get_with_backoff(session: requests.Session, params: dict, retries: int = 6) -> dict:
    delay = 2.0
    for attempt in range(retries):
        resp = session.get(WIKI_ROWS_API, params=params, timeout=60)
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            wait = float(resp.headers.get("Retry-After", delay))
            log.warning("HTTP %d from dataset viewer, retrying in %.0fs", resp.status_code, wait)
            time.sleep(wait)
            delay = min(delay * 2, 60)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("unreachable")


def fetch_wikipedia(n: int, cache: Path, seed: int) -> list[str]:
    """Samples `n` Wikipedia articles with at least MIN_WORDS words (cached as JSONL)."""
    texts = [d["text"] for d in read_jsonl(cache)] if cache.exists() else []
    if len(texts) >= n:
        return texts[:n]
    log.info("Sampling %d Wikipedia articles from %s", n, WIKI_DATASET["dataset"])
    rng = random.Random(seed)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    texts = []
    while len(texts) < n:
        offset = rng.randrange(0, WIKI_NUM_ROWS - WINDOW)
        rows = _get_with_backoff(session, {**WIKI_DATASET, "offset": offset, "length": WINDOW})["rows"]
        texts.extend(r["row"]["text"] for r in rows if len(r["row"]["text"].split()) >= MIN_WORDS)
        log.info("  %d / %d articles", min(len(texts), n), n)
    texts = texts[:n]
    with JsonlWriter(cache) as w:
        for t in texts:
            w.write({"text": t})
    return texts


def fetch_crawl_negatives(cfg: dict) -> list[str]:
    """Builds a held-out negative set from the *last* segments of the crawl.

    The pipeline processes segments from the start of the manifest
    (`download.segment_offset`), so these documents are never scored by the model
    they trained. Scoring a model's own training negatives makes it far stricter
    than it is on unseen text. Results are cached under models/.
    """
    from src.downloader import _session, _with_retries, download_sample, list_warc_paths, resolve_crawl
    from src.extractor import iter_documents
    from src.lang_filter import LanguageIdentifier, ensure_model

    tcfg, dcfg, lcfg = cfg["quality_training"], cfg["download"], cfg["lang_filter"]
    model_dir = resolve_path(cfg, cfg["quality"]["model_path"]).parent
    session = _session()
    crawl = resolve_crawl(dcfg["crawl"], session, dcfg["timeout_s"])
    cache = model_dir / f"crawl_negatives_{crawl}.jsonl.gz"
    if cache.exists():  # delete the cache file to rebuild with different settings
        return [d["text"] for d in read_jsonl(cache)]

    paths = list_warc_paths(crawl, dcfg["base_url"], session, dcfg["timeout_s"])
    lid = LanguageIdentifier(ensure_model(resolve_path(cfg, lcfg["model_path"]), lcfg["model_url"]))
    targets = set(lcfg["target_languages"])
    warc_dir = model_dir / "negative_warcs"
    warc_dir.mkdir(parents=True, exist_ok=True)
    texts = []
    for rel in reversed(paths[-tcfg["negative_segments"]:]):
        dest = warc_dir / Path(rel).name
        if not dest.exists():
            log.info("Sampling held-out negative segment %s", dest.name)
            _with_retries(lambda u=f"{dcfg['base_url']}/{rel}", d=dest: download_sample(
                u, d, tcfg["negative_records_per_segment"], session, dcfg["timeout_s"]),
                retries=dcfg["max_retries"], what=dest.name)
        for doc in iter_documents(dest, cfg, Counter()):
            lang, score = lid.predict(doc["text"])
            if lang in targets and score >= lcfg["min_confidence"]:
                texts.append(doc["text"])
        log.info("  %d held-out negative documents", len(texts))
    with JsonlWriter(cache) as w:
        for t in texts:
            w.write({"text": t})
    return texts


def reservoir_sample(docs, k: int, rng: random.Random) -> list[str]:
    sample: list[str] = []
    for i, doc in enumerate(docs):
        if i < k:
            sample.append(doc["text"])
        else:
            j = rng.randint(0, i)
            if j < k:
                sample[j] = doc["text"]
    return sample


def roc_auc(labels: list[int], scores: list[float]) -> float:
    """AUC via the rank-sum (Mann-Whitney U) formulation."""
    order = sorted(range(len(scores)), key=scores.__getitem__)
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):  # average ranks for ties
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    rank_sum = sum(r for r, y in zip(ranks, labels) if y)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--positives", default=None, help='"wikipedia" or a JSONL file with a text field')
    parser.add_argument("--negatives", default=None, help="JSONL file/dir of crawled documents")
    args = parser.parse_args()

    import fasttext

    cfg = load_config(args.config)
    tcfg = cfg["quality_training"]
    rng = random.Random(tcfg["seed"])
    model_path = resolve_path(cfg, cfg["quality"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    neg_source = args.negatives or tcfg["negatives"]
    if neg_source == "commoncrawl":
        pool = fetch_crawl_negatives(cfg)
        negatives = rng.sample(pool, min(len(pool), tcfg["num_negatives"]))
    else:
        neg_source = str(resolve_path(cfg, neg_source))
        negatives = reservoir_sample(read_jsonl(neg_source), tcfg["num_negatives"], rng)
    if not negatives:
        raise SystemExit(f"No negative documents from {neg_source}")

    pos_source = args.positives or tcfg["positives"]
    if pos_source == "wikipedia":
        positives = fetch_wikipedia(tcfg["num_positives"], model_path.parent / "wikipedia_positives.jsonl.gz",
                                    tcfg["seed"])
    else:
        positives = reservoir_sample(read_jsonl(resolve_path(cfg, pos_source)), tcfg["num_positives"], rng)

    # Balance the classes so the prior does not dominate P(hq).
    n = min(len(positives), len(negatives))
    examples = [(HQ_LABEL, t) for t in rng.sample(positives, n)] + [(LQ_LABEL, t) for t in rng.sample(negatives, n)]
    rng.shuffle(examples)
    n_valid = max(int(len(examples) * tcfg["validation_split"]), 1)
    valid, train = examples[:n_valid], examples[n_valid:]
    log.info("Training on %d examples (%d per class), validating on %d", len(train), n, len(valid))

    with tempfile.TemporaryDirectory() as tmp:
        train_file = Path(tmp) / "train.txt"
        train_file.write_text("".join(f"{label} {classifier_text(t)}\n" for label, t in train), encoding="utf-8")
        model = fasttext.train_supervised(
            input=str(train_file), epoch=tcfg["epochs"], lr=tcfg["learning_rate"],
            wordNgrams=tcfg["word_ngrams"], dim=tcfg["dim"], bucket=tcfg["bucket"], minCount=2,
            loss="softmax", verbose=0,
        )

    cutoff = cfg["quality"]["cutoff"]
    labels, scores = [], []
    for label, text in valid:
        pred, prob = model.predict(classifier_text(text), k=2)
        p = dict(zip(pred, prob))
        labels.append(int(label == HQ_LABEL))
        scores.append(float(p.get(HQ_LABEL, 1 - p.get(LQ_LABEL, 0.0))))
    tp = sum(1 for y, s in zip(labels, scores) if y and s >= cutoff)
    fp = sum(1 for y, s in zip(labels, scores) if not y and s >= cutoff)
    fn = sum(1 for y, s in zip(labels, scores) if y and s < cutoff)
    metrics = {
        "train_examples": len(train), "valid_examples": len(valid), "cutoff": cutoff,
        "precision": round(tp / max(tp + fp, 1), 4), "recall": round(tp / max(tp + fn, 1), 4),
        "auc": round(roc_auc(labels, scores), 4),
        "positives": pos_source, "negatives": neg_source,
    }

    model.save_model(str(model_path))
    write_json(model_path.with_suffix(".json"), metrics)
    log.info("Validation: precision=%.3f recall=%.3f AUC=%.3f at cutoff %.2f",
             metrics["precision"], metrics["recall"], metrics["auc"], cutoff)
    log.info("Saved quality model to %s", model_path)


if __name__ == "__main__":
    main()
