"""Download Common Crawl WARC segments.

Resolves the crawl's `warc.paths.gz` manifest and fetches WARC files from
data.commoncrawl.org with retries and exponential backoff. Two modes:

* full download: the complete ~1 GB segment, resumable via HTTP Range requests;
* sampled download (`max_records_per_segment`): streams the segment and keeps the
  first N records, re-written as a valid gzipped WARC. Useful for development
  and for sizing a run before committing to full segments.

Usage:
    python -m src.downloader [--config config.yaml] [--crawl CC-MAIN-2024-33] [--num-segments 2]
"""

from __future__ import annotations

import argparse
import gzip
import io
import time
from pathlib import Path
from typing import Any

import requests
from warcio.archiveiterator import ArchiveIterator
from warcio.warcwriter import WARCWriter

from .utils import get_logger, load_config, stage_dir

log = get_logger("downloader")

COLLINFO_URL = "https://index.commoncrawl.org/collinfo.json"
USER_AGENT = "pretrain-data-pipeline/0.1 (+https://github.com/swan28pande/pretrain-data-pipeline)"
CHUNK = 1 << 20


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def _with_retries(fn, *, retries: int, what: str):
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except (requests.RequestException, OSError, EOFError) as e:
            # Common Crawl answers 503 "Slow Down" when rate limited: back off and retry.
            if attempt == retries:
                raise
            log.warning("%s failed (%s), retry %d/%d in %.0fs", what, e, attempt, retries - 1, delay)
            time.sleep(delay)
            delay = min(delay * 2, 60)


def resolve_crawl(crawl: str, session: requests.Session, timeout: float) -> str:
    if crawl != "latest":
        return crawl
    resp = session.get(COLLINFO_URL, timeout=timeout)
    resp.raise_for_status()
    latest = resp.json()[0]["id"]
    log.info("Resolved latest crawl: %s", latest)
    return latest


def list_warc_paths(crawl: str, base_url: str, session: requests.Session, timeout: float) -> list[str]:
    url = f"{base_url}/crawl-data/{crawl}/warc.paths.gz"
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return gzip.decompress(resp.content).decode().split()


def download_full(url: str, dest: Path, session: requests.Session, timeout: float) -> None:
    """Downloads a whole file, resuming a partial `.part` file with a Range request."""
    part = dest.with_name(dest.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with session.get(url, headers=headers, stream=True, timeout=timeout) as resp:
        if resp.status_code == 416:  # already complete
            part.replace(dest)
            return
        resp.raise_for_status()
        mode = "ab" if offset and resp.status_code == 206 else "wb"
        with open(part, mode) as f:
            for chunk in resp.iter_content(CHUNK):
                f.write(chunk)
    part.replace(dest)


def download_sample(url: str, dest: Path, max_records: int, session: requests.Session, timeout: float) -> int:
    """Streams a WARC and writes its first `max_records` records to `dest`."""
    tmp = dest.with_name(dest.name + ".part")
    written = skipped = 0
    with session.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        resp.raw.decode_content = False
        with open(tmp, "wb") as out:
            for record in ArchiveIterator(resp.raw):
                # Serialize into a buffer first: a record warcio cannot re-encode (e.g. non-ASCII
                # HTTP headers) is skipped instead of leaving a half-written gzip member.
                buf = io.BytesIO()
                try:
                    WARCWriter(buf, gzip=True).write_record(record)
                except (UnicodeEncodeError, ValueError):
                    skipped += 1
                    continue
                out.write(buf.getvalue())
                written += 1
                if written >= max_records:
                    break
    tmp.replace(dest)
    if skipped:
        log.warning("Skipped %d records of %s that could not be re-encoded", skipped, dest.name)
    return written


def download(cfg: dict[str, Any], crawl: str | None = None, num_segments: int | None = None) -> list[Path]:
    dcfg = cfg["download"]
    out_dir = stage_dir(cfg, "warc")
    out_dir.mkdir(parents=True, exist_ok=True)
    session = _session()
    timeout, retries = dcfg["timeout_s"], dcfg["max_retries"]

    crawl = resolve_crawl(crawl or dcfg["crawl"], session, timeout)
    paths = _with_retries(lambda: list_warc_paths(crawl, dcfg["base_url"], session, timeout),
                          retries=retries, what="warc.paths.gz")
    start = dcfg.get("segment_offset", 0)
    n = num_segments or dcfg["num_segments"]
    selected = paths[start:start + n]
    log.info("%s has %d WARC segments; fetching %d", crawl, len(paths), len(selected))

    max_records = dcfg.get("max_records_per_segment")
    files = []
    for rel in selected:
        url = f"{dcfg['base_url']}/{rel}"
        dest = out_dir / Path(rel).name
        if dest.exists():
            log.info("Already downloaded: %s", dest.name)
            files.append(dest)
            continue
        t0 = time.time()
        if max_records:
            count = _with_retries(lambda u=url, d=dest: download_sample(u, d, max_records, session, timeout),
                                  retries=retries, what=dest.name)
            log.info("Sampled %d records from %s in %.1fs", count, dest.name, time.time() - t0)
        else:
            _with_retries(lambda u=url, d=dest: download_full(u, d, session, timeout),
                          retries=retries, what=dest.name)
            log.info("Downloaded %s (%.1f MB) in %.1fs", dest.name, dest.stat().st_size / 1e6, time.time() - t0)
        files.append(dest)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--crawl", default=None)
    parser.add_argument("--num-segments", type=int, default=None)
    args = parser.parse_args()
    download(load_config(args.config), args.crawl, args.num_segments)


if __name__ == "__main__":
    main()
