"""Document quality scoring.

Two layers, applied in order:

1. Heuristic filters (Gopher-style rules from Rae et al., 2021): word count,
   mean word length, symbol-to-word ratio, bullet / ellipsis line ratios,
   alphabetic word ratio and duplicate-line ratio. Cheap and catch most spam,
   menus and boilerplate that survived extraction.
2. A fastText classifier trained to separate curated reference text
   (`__label__hq`, e.g. Wikipedia) from random crawl text (`__label__lq`), as in
   GPT-3 / CCNet style filtering. `quality_score` = P(hq); documents below
   `quality.cutoff` are dropped. Train it with `scripts/train_quality_model.py`.

Usage:
    python -m src.quality_scorer [--config config.yaml] [--input work/lang_filtered]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .utils import (JsonlWriter, clear_outputs, get_logger, load_config, normalize_text, read_jsonl,
                    resolve_path, stage_dir, write_json)

log = get_logger("quality_scorer")

HQ_LABEL, LQ_LABEL = "__label__hq", "__label__lq"
MAX_WORDS = 1_000  # the classifier sees the first N words of a document
BULLETS = ("•", "-", "*", "·", "–", "‣", "▪", "►")


def classifier_text(text: str) -> str:
    """Single-line normalized text fed to fastText, shared by training and inference."""
    return " ".join(normalize_text(text).split()[:MAX_WORDS])


def heuristic_failures(text: str, rules: dict[str, float]) -> list[str]:
    """Names of the heuristic rules a document violates (empty list = passes)."""
    words = text.split()
    n_words = len(words)
    if n_words < rules["min_words"]:
        return ["too_few_words"]
    failures = []
    if n_words > rules["max_words"]:
        failures.append("too_many_words")

    mean_len = sum(len(w) for w in words) / n_words
    if not rules["min_mean_word_length"] <= mean_len <= rules["max_mean_word_length"]:
        failures.append("mean_word_length")

    symbols = text.count("#") + text.count("...") + text.count("…")
    if symbols / n_words > rules["max_symbol_to_word_ratio"]:
        failures.append("symbol_ratio")

    alpha_words = sum(any(c.isalpha() for c in w) for w in words)
    if alpha_words / n_words < rules["min_alpha_word_ratio"]:
        failures.append("alpha_word_ratio")

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        if sum(l.startswith(BULLETS) for l in lines) / len(lines) > rules["max_bullet_line_ratio"]:
            failures.append("bullet_lines")
        if sum(l.endswith(("...", "…")) for l in lines) / len(lines) > rules["max_ellipsis_line_ratio"]:
            failures.append("ellipsis_lines")
        if 1 - len(set(lines)) / len(lines) > rules["max_duplicate_line_ratio"]:
            failures.append("duplicate_lines")
    return failures


class QualityClassifier:
    def __init__(self, model_path: str | Path):
        import fasttext

        fasttext.FastText.eprint = lambda *args, **kwargs: None
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Quality model {path} not found. Train it with: python scripts/train_quality_model.py"
            )
        self.model = fasttext.load_model(str(path))

    def score(self, text: str) -> float:
        """Probability that `text` resembles the curated (high-quality) reference corpus."""
        labels, probs = self.model.predict(classifier_text(text), k=2)
        scores = dict(zip(labels, probs))
        return float(scores.get(HQ_LABEL, 1.0 - scores.get(LQ_LABEL, 0.0)))


class QualityScorer:
    """Heuristics + classifier; `evaluate` returns (quality_score or None, drop_reason or None)."""

    def __init__(self, cfg: dict[str, Any]):
        qcfg = cfg["quality"]
        self.rules = qcfg["heuristics"]
        self.cutoff = qcfg["cutoff"]
        self.classifier = QualityClassifier(resolve_path(cfg, qcfg["model_path"]))

    def evaluate(self, text: str) -> tuple[float | None, str | None]:
        failures = heuristic_failures(text, self.rules)
        if failures:
            return None, f"heuristic:{failures[0]}"
        score = self.classifier.score(text)
        return score, (None if score >= self.cutoff else "low_quality_score")


def quality_filter(cfg: dict[str, Any], input_path: str | Path | None = None) -> dict[str, Any]:
    scorer = QualityScorer(cfg)
    source = input_path or stage_dir(cfg, "lang_filtered")
    out_dir = stage_dir(cfg, "quality_filtered")
    clear_outputs(out_dir)

    reasons: Counter = Counter()
    histogram = [0] * 10
    counts = {"input": 0, "kept": 0}
    with JsonlWriter(out_dir / "quality-00000.jsonl.gz") as writer:
        for doc in read_jsonl(source):
            counts["input"] += 1
            score, reason = scorer.evaluate(doc["text"])
            if score is not None:
                histogram[min(int(score * 10), 9)] += 1
            if reason:
                reasons[reason] += 1
                continue
            doc["quality_score"] = round(score, 4)
            writer.write(doc)
            counts["kept"] += 1

    report = {
        "stage": "quality", **counts, "cutoff": scorer.cutoff,
        "drop_reasons": dict(reasons.most_common()),
        "score_histogram": {f"{i / 10:.1f}-{(i + 1) / 10:.1f}": c for i, c in enumerate(histogram)},
    }
    write_json(out_dir / "_report.json", report)
    log.info("Quality filter: %d in, %d kept (cutoff %.2f); drops: %s", counts["input"], counts["kept"],
             scorer.cutoff, ", ".join(f"{k}={v}" for k, v in reasons.most_common()))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", default=None)
    args = parser.parse_args()
    quality_filter(load_config(args.config), args.input)


if __name__ == "__main__":
    main()
