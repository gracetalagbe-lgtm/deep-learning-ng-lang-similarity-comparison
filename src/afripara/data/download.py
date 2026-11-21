"""Download registered data sources into data/raw/ (immutable originals).

Features:
  - downloads every registered file of a source (or --all sources);
  - skips files that already exist with the same sha256 (idempotent);
  - writes/updates data/raw/provenance.json: url, sha256, byte size,
    download timestamp (UTC), licence, citation per source;
  - never mutates an existing raw file: if content changed, the download is
    REFUSED unless --force is given (raw originals are immutable).

Usage:
    python -m afripara.data.download --source afrixnli
    python -m afripara.data.download --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from afripara.config import PROJECT_ROOT, resolve_path
from afripara.data.sources import SOURCES, get_source
from afripara.utils.logging_utils import setup_logging

logger = logging.getLogger("afripara.download")

RAW_DIR_NAME = "data_raw"
PROVENANCE_NAME = "provenance.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_provenance(raw_dir: Path) -> dict:
    prov_path = raw_dir / PROVENANCE_NAME
    if prov_path.exists():
        with prov_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"downloaded_at": None, "sources": {}}


def save_provenance(raw_dir: Path, prov: dict) -> None:
    prov["downloaded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prov_path = raw_dir / PROVENANCE_NAME
    with prov_path.open("w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2, ensure_ascii=False)
    logger.info("Provenance written to %s", prov_path)


def download_file(url: str, dest: Path, force: bool = False) -> str:
    """Download ``url`` to ``dest``; returns the file's sha256.

    Refuses to overwrite an existing file whose content differs (immutable
    raw originals) unless ``force`` is set.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "afripara-msc-research/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as fh:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    digest = sha256_of(tmp)
    if dest.exists():
        existing = sha256_of(dest)
        if existing == digest:
            tmp.unlink()
            logger.info("unchanged: %s", dest)
            return digest
        if not force:
            tmp.unlink()
            raise RuntimeError(
                f"Refusing to overwrite changed raw file {dest} "
                f"(existing sha256={existing[:12]}..., new={digest[:12]}...). Use --force."
            )
    tmp.replace(dest)
    logger.info("downloaded: %s (%d bytes) from %s", dest, dest.stat().st_size, url)
    return digest


def download_source(name: str, raw_dir: Path, force: bool = False) -> dict:
    """Download all files of one source; returns its provenance record."""
    src = get_source(name)
    record = {
        "title": src["title"],
        "description": src["description"],
        "languages": src["languages"],
        "license": src["license"],
        "license_note": src["license_note"],
        "citation": src["citation"],
        "files": {},
    }
    failures = []
    for url, rel in src["files"]:
        dest = raw_dir / rel
        try:
            digest = download_file(url, dest, force=force)
            record["files"][rel] = {
                "url": url,
                "sha256": digest,
                "bytes": dest.stat().st_size,
            }
        except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, OSError) as exc:
            failures.append((rel, str(exc)))
            logger.error("FAILED %s: %s", rel, exc)
    record["failures"] = failures
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", default=[], help="source name (repeatable)")
    parser.add_argument("--all", action="store_true", help="download every registered source")
    parser.add_argument("--force", action="store_true", help="allow overwriting changed raw files")
    args = parser.parse_args(argv)

    setup_logging()
    names = sorted(SOURCES) if args.all else args.source
    if not names:
        parser.error("give --source <name> (repeatable) or --all")
        return 2

    raw_dir = PROJECT_ROOT / "data" / "raw"
    try:
        from afripara.config import load_config

        raw_dir = resolve_path(load_config("default"), RAW_DIR_NAME)
    except Exception:  # pragma: no cover - config should exist in practice
        logger.warning("Could not load default config; using %s", raw_dir)

    prov = load_provenance(raw_dir)
    total_failures = 0
    for name in names:
        logger.info("=== source: %s (%s) ===", name, get_source(name)["license"])
        prov["sources"].setdefault(name, {})
        prov["sources"][name] = download_source(name, raw_dir, force=args.force)
        failures = prov["sources"][name].get("failures", [])
        total_failures += len(failures)

    save_provenance(raw_dir, prov)
    logger.info("Done. %d failure(s).", total_failures)
    return 1 if total_failures else 0


if __name__ == "__main__":
    sys.exit(main())
