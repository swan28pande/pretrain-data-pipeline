"""Shared helpers: config loading, JSONL I/O, URL/domain parsing, text normalization."""

from __future__ import annotations

import glob
import gzip
import io
import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import urlsplit

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def get_logger(name: str) -> logging.Logger:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=os.environ.get("LOG_LEVEL", "INFO"),
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    return logging.getLogger(name)


# --------------------------------------------------------------------------- config

def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Loads the YAML config. Relative paths in it are resolved against the config file's directory."""
    path = Path(path or DEFAULT_CONFIG).resolve()
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_dir"] = str(path.parent)
    cfg["_config_path"] = str(path)
    return cfg


def resolve_path(cfg: dict[str, Any], value: str | os.PathLike) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return p if p.is_absolute() else Path(cfg.get("_config_dir", ".")) / p


def stage_dir(cfg: dict[str, Any], stage: str) -> Path:
    """Output directory of a pipeline stage, e.g. <work_dir>/extracted."""
    return resolve_path(cfg, cfg["pipeline"]["work_dir"]) / stage


# --------------------------------------------------------------------------- JSONL I/O

def _open_text(path: str | os.PathLike, mode: str = "rt"):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8") if "t" in mode else gzip.open(path, mode)
    return open(path, mode, encoding="utf-8")


def list_jsonl(source: str | os.PathLike | Iterable[str | os.PathLike]) -> list[str]:
    """Expands a file, directory (including Spark `part-*` output) or glob into sorted JSONL files."""
    if not isinstance(source, (str, os.PathLike)):
        files: list[str] = []
        for s in source:
            files.extend(list_jsonl(s))
        return files
    source = str(source)
    if os.path.isdir(source):
        candidates = glob.glob(os.path.join(source, "**", "*"), recursive=True)
        return sorted(
            f for f in candidates
            if os.path.isfile(f)
            and (f.endswith((".jsonl", ".jsonl.gz", ".json.gz")) or os.path.basename(f).startswith("part-"))
            and not os.path.basename(f).startswith((".", "_"))
        )
    return sorted(glob.glob(source)) if any(c in source for c in "*?[") else [source]


def read_jsonl(source: str | os.PathLike | Iterable[str | os.PathLike]) -> Iterator[dict[str, Any]]:
    for path in list_jsonl(source):
        with _open_text(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


class JsonlWriter:
    """Writes documents to a gzip-compressed JSONL file atomically (temp file + rename)."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.path.with_name(self.path.name + ".tmp")
        self._raw = open(self._tmp, "wb")
        self._gz = gzip.GzipFile(fileobj=self._raw, mode="wb", compresslevel=6)
        self._f = io.TextIOWrapper(self._gz, encoding="utf-8")
        self.count = 0

    def write(self, doc: dict[str, Any]) -> None:
        self._f.write(json.dumps(doc, ensure_ascii=False))
        self._f.write("\n")
        self.count += 1

    def close(self) -> None:
        self._f.close()
        self._raw.close()
        os.replace(self._tmp, self.path)

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.close()
        else:
            self._f.close()
            self._raw.close()
            self._tmp.unlink(missing_ok=True)


def write_json(path: str | os.PathLike, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def clear_outputs(directory: Path) -> None:
    """Removes previous stage outputs (ours or Spark's part files) so reruns never mix shards."""
    if directory.exists():
        for pattern in ("*.jsonl.gz", "part-*", ".part-*.crc", "_SUCCESS", "._SUCCESS.crc"):
            for f in directory.glob(pattern):
                f.unlink()


# --------------------------------------------------------------------------- URLs and text

def url_domain(url: str) -> str:
    """Host name without a leading `www.` (e.g. https://www.bbc.co.uk/x -> bbc.co.uk)."""
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def url_tld(url_or_domain: str) -> str:
    domain = url_domain(url_or_domain) if "/" in url_or_domain else url_or_domain
    return domain.rsplit(".", 1)[-1] if "." in domain else domain


def normalize_text(text: str) -> str:
    """Normalization used before hashing: NFKC, lowercase, no punctuation, collapsed whitespace."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())
