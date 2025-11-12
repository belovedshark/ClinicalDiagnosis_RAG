#!/usr/bin/env python3
"""Chunk records (text, table, images) into per-record JSON files containing an array of chunks.

This implements the "one JSON per record" storage option described in the plan.

Usage:
  python3 scripts/chunk_and_embed.py --metadata-dir Processed/metadata --out-dir Processed_embeddings/embeddings

By default the script writes per-record JSON files named {record_id}.json under the out-dir.
No embeddings are computed by this script; it only produces chunk metadata and text/table/image entries.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import List, Optional


def normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def split_paragraphs(markdown_text: str) -> List[str]:
    """Split markdown text into paragraph-like blocks.

    Heuristics:
    - Split on two or more newlines.
    - Keep headings (#) attached to the following paragraph.
    - Merge very short paragraphs (<40 chars) with previous when possible.
    """
    text = normalize_newlines(markdown_text or "")
    # Split on two or more newlines
    raw_blocks = re.split(r"\n\s*\n", text)
    blocks: List[str] = []
    i = 0
    while i < len(raw_blocks):
        b = raw_blocks[i].strip()
        if not b:
            i += 1
            continue

        # If block is a heading (starts with #), attach it to next block if exists
        if re.match(r"^#{1,6}\s+", b) and i + 1 < len(raw_blocks):
            next_b = raw_blocks[i + 1].strip()
            combined = b + "\n\n" + next_b if next_b else b
            blocks.append(combined.strip())
            i += 2
            continue

        blocks.append(b)
        i += 1

    # Merge very short blocks into previous if possible
    merged: List[str] = []
    for b in blocks:
        if merged and len(b) < 40:
            merged[-1] = (merged[-1] + "\n\n" + b).strip()
        else:
            merged.append(b)

    return merged


def extract_tables_from_text(text: str) -> List[str]:
    """Heuristic extraction of Markdown tables from text.

    Scans the text line-by-line and groups contiguous lines that contain '|' as table blocks.
    Returns a list of table block strings (preserving newlines).
    """
    if not text:
        return []
    lines = normalize_newlines(text).split("\n")
    tables: List[List[str]] = []
    current: Optional[List[str]] = None

    def is_table_line(line: str) -> bool:
        # A line is considered part of a table if it contains a pipe '|' with at least one pipe-separated column
        if "|" in line:
            # crude check: at least one pipe and some non-pipe content
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2 and any(p for p in parts):
                return True
        # also treat separator lines like '|---|---|' or '---|---'
        if re.match(r"^\s*\|?\s*[:-]+\s*(\|\s*[:-]+\s*)+\|?\s*$", line):
            return True
        return False

    for ln in lines:
        if is_table_line(ln):
            if current is None:
                current = []
            current.append(ln)
        else:
            if current is not None:
                # end of table
                tables.append(current)
                current = None
    if current is not None:
        tables.append(current)

    # join lines into blocks and strip outer blank lines
    table_blocks = ["\n".join(block).strip() for block in tables if any(line.strip() for line in block)]
    return table_blocks


def gather_image_paths(meta: dict) -> List[str]:
    # look for common locations in the metadata JSON
    paths = []
    if "image_paths" in meta and isinstance(meta["image_paths"], list):
        paths.extend(meta["image_paths"])
    # nested under metadata.image_paths
    m = meta.get("metadata")
    if isinstance(m, dict):
        ip = m.get("image_paths") or m.get("images")
        if isinstance(ip, list):
            paths.extend(ip)

    # fallback keys
    for k in ("images", "image_files", "image_paths"):
        v = meta.get(k)
        if isinstance(v, list):
            paths.extend(v)

    # deduplicate while preserving order
    seen = set()
    out = []
    for p in paths:
        if not isinstance(p, str):
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def build_chunks_for_record(meta: dict, record_id: str) -> List[dict]:
    chunks: List[dict] = []

    # Text chunks
    md_text = meta.get("markdown_text") or meta.get("text") or ""
    if md_text and md_text.strip():
        paragraphs = split_paragraphs(md_text)
        for i, p in enumerate(paragraphs, start=1):
            chunk = {
                "record_id": record_id,
                "chunk_id": f"{record_id}__t{i}",
                "type": "text",
                "text": p,
                "image_path": None,
                "metadata": {
                    "position": {"index": i, "total": len(paragraphs)},
                    "original_length_chars": len(p),
                },
                "embedding": None,
            }
            chunks.append(chunk)

    # Table chunks
    table_text = meta.get("table_text") or meta.get("tables") or ""
    table_blocks: List[str] = []
    table_sources: List[Optional[str]] = []
    if isinstance(table_text, str) and table_text.strip():
        extracted = extract_tables_from_text(table_text)
        if extracted:
            table_blocks = extracted
        else:
            table_blocks = [table_text.strip()]
        # use provided table_source if present
        source = meta.get("table_source")
        table_sources = [source] * len(table_blocks)

    # If no table_text found, attempt to detect tables inside markdown_text
    if not table_blocks:
        md_text = meta.get("markdown_text") or ""
        if md_text and md_text.strip():
            extracted_from_md = extract_tables_from_text(md_text)
            if extracted_from_md:
                table_blocks = extracted_from_md
                # mark source as markdown_path if available
                table_sources = [meta.get("markdown_path") or meta.get("markdown") or None] * len(table_blocks)

    for j, tb in enumerate(table_blocks, start=1):
        chunk = {
            "record_id": record_id,
            "chunk_id": f"{record_id}__tab{j}",
            "type": "table",
            "text": tb,
            "image_path": None,
            "metadata": {"position": {"table_index": j, "total_tables": len(table_blocks)}, "original_length_chars": len(tb)},
            "table_source": table_sources[j - 1] if table_sources else None,
            "embedding": None,
        }
        chunks.append(chunk)

    # Image chunks
    image_paths = gather_image_paths(meta)
    for k, ip in enumerate(image_paths, start=1):
        chunk = {
            "record_id": record_id,
            "chunk_id": f"{record_id}__img{k}",
            "type": "image",
            "text": None,
            "image_path": ip,
            "metadata": {"position": {"image_index": k, "total_images": len(image_paths)}},
            "embedding": None,
        }
        chunks.append(chunk)

    return chunks


def process_all(metadata_dir: Path, out_dir: Path, dry_run: bool = False, overwrite: bool = True):
    metadata_dir = metadata_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_files = sorted(metadata_dir.glob("*.json"))
    logging.info("Found %d metadata files in %s", len(meta_files), metadata_dir)
    # default table_dir near Processed/out_all_structured — used as a fallback when metadata lacks table_text
    table_dir = Path("Processed/out_all_structured").expanduser().resolve()

    for meta_path in meta_files:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logging.error("Failed to read metadata %s: %s", meta_path, exc)
            continue

        record_id = meta.get("record_id") or meta_path.stem
        logging.info("Processing record %s", record_id)

        # If metadata doesn't include markdown_text, try to read from markdown_path referenced in the metadata
        if not meta.get("markdown_text"):
            md_path_value = meta.get("markdown_path")
            if md_path_value:
                md_path = Path(md_path_value)
                # if path is relative, resolve relative to repo root or metadata dir
                if not md_path.is_absolute():
                    candidate = metadata_dir.parent / md_path
                    if candidate.exists():
                        md_path = candidate
                    else:
                        md_path = metadata_dir / md_path.name

                if md_path.exists():
                    try:
                        meta["markdown_text"] = md_path.read_text(encoding="utf-8")
                        logging.debug("Loaded markdown_text from %s for %s", md_path, record_id)
                    except Exception:
                        meta["markdown_text"] = md_path.read_text(encoding="utf-8", errors="replace")

        # If metadata doesn't include table_text, try to find a structured table file in the table_dir
        if not meta.get("table_text"):
            # look for common table file names
            found_table = None
            for ext in (".md", ".txt", ".markdown"):
                p = table_dir / f"{record_id}{ext}"
                if p.exists():
                    found_table = p
                    break
            if found_table is None:
                # fallback glob
                candidates = sorted(table_dir.glob(f"{record_id}*"))
                if candidates:
                    # prefer .md among candidates
                    for c in candidates:
                        if c.suffix.lower() == ".md":
                            found_table = c
                            break
                    if found_table is None:
                        found_table = candidates[0]

            if found_table is not None and found_table.exists():
                try:
                    meta["table_text"] = found_table.read_text(encoding="utf-8")
                    meta["table_source"] = str(found_table)
                    logging.debug("Loaded table_text from %s for %s", found_table, record_id)
                except Exception:
                    meta["table_text"] = found_table.read_text(encoding="utf-8", errors="replace")

        chunks = build_chunks_for_record(meta, record_id)

        out_path = out_dir / f"{record_id}.json"
        if out_path.exists() and not overwrite:
            logging.info("Skipping existing %s (overwrite disabled)", out_path)
            continue

        out_obj = {"record_id": record_id, "chunks": chunks}

        if dry_run:
            logging.info("Dry-run: would write %s with %d chunks", out_path, len(chunks))
            continue

        try:
            out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
            logging.info("Wrote %s (%d chunks)", out_path, len(chunks))
        except Exception as exc:
            logging.error("Failed to write %s: %s", out_path, exc)


def parse_args():
    p = argparse.ArgumentParser(description="Chunk records into per-record JSON files containing arrays of chunks")
    p.add_argument("--metadata-dir", default="Processed/metadata", help="Directory with metadata JSON files")
    p.add_argument("--out-dir", default="Processed_embeddings/embeddings", help="Output directory for per-record chunk JSON files")
    p.add_argument("--dry-run", action="store_true", help="Don't write files; just log what would be done")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    process_all(Path(args.metadata_dir), Path(args.out_dir), dry_run=args.dry_run, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
