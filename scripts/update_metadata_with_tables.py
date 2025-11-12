#!/usr/bin/env python3
"""Attach restructured table text to existing metadata JSON files.

Reads JSON files from a metadata directory (default: Processed/metadata),
looks for matching table files in a table directory (default: Processed/out_all_structured),
and writes updated JSONs either in-place (with optional .bak backups) or to an output directory.

Usage examples:
  python3 scripts/update_metadata_with_tables.py --metadata-dir Processed/metadata \
      --table-dir Processed/out_all_structured --backup

"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional
import shutil


def read_text_if_nonblank(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        # fallback with replace errors
        text = path.read_text(encoding="utf-8", errors="replace")
    if not text or text.strip() == "":
        return None
    return text


def find_table_file(table_dir: Path, record_id: str) -> Optional[Path]:
    # Common exact names
    for ext in (".md", ".txt", ".markdown"):
        p = table_dir / f"{record_id}{ext}"
        if p.exists():
            return p

    # Fallback: glob for files that start with the record id
    candidates = sorted(table_dir.glob(f"{record_id}*"))
    if candidates:
        # prefer .md if present among candidates
        for c in candidates:
            if c.suffix.lower() == ".md":
                return c
        return candidates[0]

    return None


def backup_file(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    logging.info("Wrote backup %s", bak)


def process_metadata_file(
    meta_path: Path,
    table_dir: Path,
    out_path: Path,
    overwrite: bool = True,
    backup: bool = True,
    dry_run: bool = False,
    force: bool = False,
):
    record_id = meta_path.stem
    logging.debug("Reading metadata %s", meta_path)
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.error("Failed to read JSON %s: %s", meta_path, exc)
        return False, "json-read-error"

    # Prefer an explicit record_id field if present
    rid = meta.get("record_id") or record_id

    table_file = find_table_file(table_dir, rid)
    table_text = None
    table_source = None
    if table_file:
        table_text = read_text_if_nonblank(table_file)
        if table_text is not None:
            table_source = str(table_file)

    # If table_text is None, we will set table_text to None in JSON
    # Decide whether to overwrite existing field
    if ("table_text" in meta) and (not force):
        logging.info("Skipping %s: metadata already contains table_text (use --force to overwrite)", meta_path.name)
        return False, "skipped-exists"

    meta["table_text"] = table_text
    meta["table_source"] = table_source

    # Determine where to write
    if out_path is None:
        write_path = meta_path
    else:
        write_path = out_path / meta_path.name

    if dry_run:
        logging.info("Dry-run: would write %s (table: %s)", write_path, bool(table_text))
        return True, "dry-run"

    # Ensure parent exists
    write_path.parent.mkdir(parents=True, exist_ok=True)

    # If overwriting original and backup requested
    if write_path.exists() and (write_path == meta_path) and backup:
        try:
            backup_file(meta_path)
        except Exception as exc:
            logging.warning("Backup failed for %s: %s", meta_path, exc)

    try:
        write_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("Wrote updated metadata to %s", write_path)
        return True, "written"
    except Exception as exc:
        logging.error("Failed to write %s: %s", write_path, exc)
        return False, "write-error"


def main(argv=None):
    p = argparse.ArgumentParser(description="Attach table_text to existing metadata JSON files")
    p.add_argument("--metadata-dir", default="Processed/metadata", help="Directory with metadata JSON files")
    p.add_argument("--table-dir", default="Processed/out_all_structured", help="Directory with restructured table files")
    p.add_argument("--out-dir", default=None, help="Directory to write updated metadata files. If omitted, overwrite metadata-dir")
    p.add_argument("--backup", action="store_true", help="When overwriting originals, create .bak copies")
    p.add_argument("--dry-run", action="store_true", help="Show what would be done without writing files")
    p.add_argument("--force", action="store_true", help="Overwrite existing table_text fields if present")
    p.add_argument("--log-level", default="INFO")

    args = p.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    metadata_dir = Path(args.metadata_dir).expanduser().resolve()
    table_dir = Path(args.table_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None

    if not metadata_dir.exists():
        logging.error("Metadata dir not found: %s", metadata_dir)
        return 2

    if not table_dir.exists():
        logging.warning("Table dir not found: %s (will still proceed but no tables will be attached)", table_dir)

    meta_files = sorted(metadata_dir.glob("*.json"))
    if not meta_files:
        logging.info("No metadata JSON files found in %s", metadata_dir)
        return 0

    stats = {"processed": 0, "written": 0, "skipped": 0, "errors": 0}

    for meta_path in meta_files:
        stats["processed"] += 1
        ok, reason = process_metadata_file(
            meta_path,
            table_dir,
            out_dir,
            overwrite=True,
            backup=args.backup,
            dry_run=args.dry_run,
            force=args.force,
        )
        if ok and reason == "written":
            stats["written"] += 1
        elif ok:
            stats["skipped"] += 1
        else:
            stats["errors"] += 1

    logging.info("Summary: processed=%d written=%d skipped=%d errors=%d", stats["processed"], stats["written"], stats["skipped"], stats["errors"]) 
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
