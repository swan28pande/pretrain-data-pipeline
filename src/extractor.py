"""Extract main text from WARC response records.

For every HTTP 200 `response` record with an HTML payload, trafilatura strips
boilerplate (navigation, footers, ads, cookie banners) and returns the main
content. Each WARC file is processed by its own worker process and written to
`<work_dir>/extracted/<warc-name>.jsonl.gz` with one document per line:

    {"id", "url", "domain", "tld", "warc_date", "source", "text"}

Usage:
    python -m src.extractor [--config config.yaml] [--input work/warc]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import trafilatura
from warcio.archiveiterator import ArchiveIterator

from .utils import JsonlWriter, get_logger, load_config, stage_dir, url_domain, url_tld, write_json

log = get_logger("extractor")
# trafilatura logs a warning for every page it discards.
logging.getLogger("trafilatura").setLevel(logging.ERROR)

_CHARSET_RE = re.compile(rb"charset=[\"']?([\w-]+)", re.IGNORECASE)


def _decode(payload: bytes, content_type: str) -> str:
    match = _CHARSET_RE.search(content_type.encode()) or _CHARSET_RE.search(payload[:2048])
    encoding = match.group(1).decode() if match else "utf-8"
    try:
        return payload.decode(encoding, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def iter_documents(warc_path: str | Path, cfg: dict[str, Any], stats: Counter) -> Iterator[dict[str, Any]]:
    ecfg = cfg["extract"]
    source = Path(warc_path).name
    with open(warc_path, "rb") as f:
        for record in ArchiveIterator(f):
            if record.rec_type != "response":
                continue
            stats["responses"] += 1
            http = record.http_headers
            if http is None or http.get_statuscode() != "200":
                stats["skipped_status"] += 1
                continue
            content_type = http.get_header("Content-Type", "") or ""
            if "html" not in content_type.lower():
                stats["skipped_non_html"] += 1
                continue
            payload = record.content_stream().read()
            if len(payload) > ecfg["max_html_bytes"]:
                stats["skipped_too_large"] += 1
                continue

            url = record.rec_headers.get_header("WARC-Target-URI")
            text = trafilatura.extract(
                _decode(payload, content_type),
                url=url,
                favor_precision=ecfg["favor_precision"],
                include_tables=ecfg["include_tables"],
                include_comments=ecfg["include_comments"],
                deduplicate=True,
            )
            if not text or len(text) < ecfg["min_text_chars"]:
                stats["skipped_short"] += 1
                continue

            stats["documents"] += 1
            domain = url_domain(url)
            yield {
                "id": hashlib.sha1(url.encode()).hexdigest()[:16],
                "url": url,
                "domain": domain,
                "tld": url_tld(domain),
                "warc_date": record.rec_headers.get_header("WARC-Date"),
                "source": source,
                "text": text,
            }


def extract_file(warc_path: str, cfg: dict[str, Any], out_dir: str) -> dict[str, Any]:
    stats: Counter = Counter()
    out = Path(out_dir) / (Path(warc_path).name.removesuffix(".warc.gz") + ".jsonl.gz")
    with JsonlWriter(out) as writer:
        for doc in iter_documents(warc_path, cfg, stats):
            writer.write(doc)
    return {"file": Path(warc_path).name, **stats}


def extract(cfg: dict[str, Any], input_dir: str | Path | None = None) -> dict[str, Any]:
    in_dir = Path(input_dir) if input_dir else stage_dir(cfg, "warc")
    out_dir = stage_dir(cfg, "extracted")
    out_dir.mkdir(parents=True, exist_ok=True)
    warcs = sorted(str(p) for p in in_dir.glob("*.warc.gz"))
    if not warcs:
        raise FileNotFoundError(f"No .warc.gz files in {in_dir}: run the download stage first")

    log.info("Extracting %d WARC files with %d workers", len(warcs), cfg["pipeline"]["num_workers"])
    totals: Counter = Counter()
    per_file = []
    with ProcessPoolExecutor(max_workers=cfg["pipeline"]["num_workers"]) as pool:
        futures = [pool.submit(extract_file, w, cfg, str(out_dir)) for w in warcs]
        for fut in futures:
            result = fut.result()
            per_file.append(result)
            totals.update({k: v for k, v in result.items() if k != "file"})
            log.info("%s: %d documents from %d responses", result["file"],
                     result.get("documents", 0), result.get("responses", 0))

    report = {"stage": "extract", "input_files": len(warcs), **totals, "per_file": per_file}
    write_json(out_dir / "_report.json", report)
    log.info("Extracted %d documents from %d responses", totals["documents"], totals["responses"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", default=None, help="directory of .warc.gz files")
    args = parser.parse_args()
    extract(load_config(args.config), args.input)


if __name__ == "__main__":
    main()
