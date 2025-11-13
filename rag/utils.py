"""Small utilities for handling table-like queries.

This file implements a conservative detector for pipe-style markdown tables
and a tiny parser that extracts headers and rows and a compact flat_text
representation for embedding.
"""
import re
from typing import Dict, List, Any


SEPARATOR_RE = re.compile(r"^\s*\|?\s*[-:]{3,}\s*(\|\s*[-:]{3,}\s*)*$")


def is_table_query(text: str) -> bool:
    """Return True if the input text looks like it contains a markdown table

    Heuristics used:
    - A line with at least one '|' character and a following line that looks like
      a markdown table separator (e.g. |---|-----|) OR
    - A leading label starting with 'TABLE' (case-insensitive)
    """
    if not text:
        return False
    lines = text.splitlines()
    # Leading TABLE  or Table 1. style label
    first = lines[0].strip() if lines else ""
    if first.upper().startswith("TABLE"):
        return True

    for i in range(len(lines) - 1):
        if '|' in lines[i] and SEPARATOR_RE.match(lines[i + 1].strip()):
            return True

    return False


def parse_query_table(text: str) -> Dict[str, Any]:
    """Parse a lightweight markdown table from the query text.

    Returns a dict with keys: headers (list), rows (list of lists), label (optional),
    caption (optional), flat_text (compact string to embed).

    The parser is intentionally forgiving: it finds the first table-looking block
    and extracts header + rows. If parsing fails, returns {}.
    """
    if not is_table_query(text):
        return {}
    lines = text.splitlines()

    label = None
    caption = None
    # If first line starts with TABLE, capture as label and drop it
    if lines and lines[0].strip().upper().startswith('TABLE'):
        label = lines[0].strip()
        lines = lines[1:]

    # Find the first header + separator pair
    header_idx = None
    for i in range(len(lines) - 1):
        if '|' in lines[i] and SEPARATOR_RE.match(lines[i + 1].strip()):
            header_idx = i
            break

    if header_idx is None:
        return {}

    raw_header = lines[header_idx]
    headers = [h.strip() for h in raw_header.strip().strip('|').split('|')]

    rows = []
    for r in lines[header_idx + 2:]:
        if not r.strip():
            break
        # Stop if we hit a line that looks like non-table text (no pipe and short)
        if '|' not in r:
            break
        cells = [c.strip() for c in r.strip().strip('|').split('|')]
        # Pad or trim cells to header length
        if len(cells) < len(headers):
            cells += [""] * (len(headers) - len(cells))
        elif len(cells) > len(headers):
            cells = cells[:len(headers)]
        rows.append(cells)

    flat = ''
    if headers:
        first_row = rows[0] if rows else []
        csv_like = ", ".join([f"{h}: {v}" for h, v in zip(headers, first_row)])
        flat = " | ".join(headers) + " -- " + csv_like

    return {
        'headers': headers,
        'rows': rows,
        'label': label,
        'caption': caption,
        'flat_text': flat,
    }
