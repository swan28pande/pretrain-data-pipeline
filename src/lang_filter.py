"""Language identification with fastText `lid.176.bin`.

Predicts the language of each document (176 languages) and keeps documents
whose top language is in `target_languages` with confidence >= `min_confidence`.
The model is downloaded on first use. Kept documents gain `lang` and
`lang_score` fields.

Usage:
    python -m src.lang_filter [--config config.yaml] [--input work/deduped]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import requests

from .utils import (JsonlWriter, clear_outputs, get_logger, load_config, read_jsonl, resolve_path,
                    stage_dir, write_json)

log = get_logger("lang_filter")

# fastText predicts on a single line; long documents are truncated for speed
# (the first few thousand characters are plenty for language ID).
MAX_CHARS = 5_000


def ensure_model(path: Path, url: str) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading language ID model %s -> %s", url, path)
    tmp = path.with_name(path.name + ".part")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(path)
    return path


def load_model(path: str | Path):
    import fasttext

    fasttext.FastText.eprint = lambda *args, **kwargs: None  # silence load-time warning
    return fasttext.load_model(str(path))


class LanguageIdentifier:
    def __init__(self, model_path: str | Path, k: int = 1):
        self.model = load_model(model_path)
        self.k = k

    def predict(self, text: str) -> tuple[str, float]:
        labels, scores = self.model.predict(text[:MAX_CHARS].replace("\n", " "), k=self.k)
        return labels[0].removeprefix("__label__"), float(scores[0])


def lang_filter(cfg: dict[str, Any], input_path: str | Path | None = None) -> dict[str, Any]:
    lcfg = cfg["lang_filter"]
    model_path = ensure_model(resolve_path(cfg, lcfg["model_path"]), lcfg["model_url"])
    lid = LanguageIdentifier(model_path)
    targets = set(lcfg["target_languages"])
    min_conf = lcfg["min_confidence"]

    source = input_path or stage_dir(cfg, "deduped")
    out_dir = stage_dir(cfg, "lang_filtered")
    clear_outputs(out_dir)

    detected: Counter = Counter()
    counts = {"input": 0, "kept": 0, "wrong_language": 0, "low_confidence": 0}
    with JsonlWriter(out_dir / "lang-00000.jsonl.gz") as writer:
        for doc in read_jsonl(source):
            counts["input"] += 1
            lang, score = lid.predict(doc["text"])
            detected[lang] += 1
            if lang not in targets:
                counts["wrong_language"] += 1
            elif score < min_conf:
                counts["low_confidence"] += 1
            else:
                doc["lang"], doc["lang_score"] = lang, round(score, 4)
                writer.write(doc)
                counts["kept"] += 1

    report = {"stage": "lang_filter", **counts, "target_languages": sorted(targets),
              "min_confidence": min_conf, "detected_languages": dict(detected.most_common(30))}
    write_json(out_dir / "_report.json", report)
    log.info("Language filter: %(input)d in, %(kept)d kept, %(wrong_language)d other languages, "
             "%(low_confidence)d low confidence", counts)
    log.info("Top detected languages: %s", ", ".join(f"{k}={v}" for k, v in detected.most_common(8)))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", default=None)
    args = parser.parse_args()
    lang_filter(load_config(args.config), args.input)


if __name__ == "__main__":
    main()
