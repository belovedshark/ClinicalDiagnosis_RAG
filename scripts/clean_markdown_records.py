#!/usr/bin/env python3
"""Clean markdown records in `Processed/markdown` by removing:

- image markdowns and their paths
- figure labels/captions (uses the existing extraction logic)
- "Further reading" sections
- Markdown pipe-style tables
- leading author names written in ALL CAPS in the first paragraph

Backups of modified files are created as `<filename>.bak` by default.

Usage:
    python3 scripts/clean_markdown_records.py --dir Processed/markdown --recursive

"""
from pathlib import Path
import argparse
import re
import shutil
import importlib.util


def load_extract_module():
    # load scripts/extract_figures_to_md.py dynamically so this script can reuse
    # its extraction regexes and logic without requiring package imports.
    here = Path(__file__).parent
    path = here / 'extract_figures_to_md.py'
    spec = importlib.util.spec_from_file_location('extract_figures_to_md', str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def remove_captions_and_images(text, mod):
    # Get captions and images from original text
    captions, images = mod.extract_from_text(text)

    # Remove image markdowns first (all occurrences)
    text_no_images = mod.IMAGE_MD_RE.sub('', text)

    # Remove captions: try to remove the exact captured caption text near its
    # original position; fallback to a single global replace.
    # Iterate from end to start to preserve positions while cutting.
    for pos, cap in sorted(captions, key=lambda x: x[0], reverse=True):
        if not cap:
            continue
        # try to find caption near original position (allow some leeway)
        start_search = max(0, pos - 200)
        end_search = min(len(text_no_images), pos + len(cap) + 200)
        idx = text_no_images.find(cap, start_search, end_search)
        if idx != -1:
            text_no_images = text_no_images[:idx] + text_no_images[idx + len(cap):]
        else:
            # fallback: replace first occurrence
            text_no_images = text_no_images.replace(cap, '', 1)

    # collapse excessive blank lines
    text_no_images = re.sub(r"\n{3,}", "\n\n", text_no_images)
    return text_no_images


def remove_further_reading(text):
    # Find 'Further reading' anywhere in the document (case-insensitive).
    # It may appear mid-paragraph (e.g. after a sentence) so we don't anchor to
    # line-start. Remove from that point up to the next markdown heading (any
    # level) or EOF. This will remove lists of citations following the marker.
    m = re.search(r'(?i)Further\s+Reading\b', text)
    if not m:
        return text
    start = m.start()
    # find next markdown heading after the marker; if none, remove to EOF
    next_h = re.search(r'(?im)^\s*#{1,6}\s+.*$', text[m.end():])
    if next_h:
        end = m.end() + next_h.start()
    else:
        end = len(text)
    new_text = text[:start] + text[end:]
    new_text = re.sub(r"\n{3,}", "\n\n", new_text)
    return new_text


def remove_markdown_tables(text):
    """Remove Markdown pipe-style tables.

    Detects a header line containing "|" followed by a separator line
    with dashes/colons (e.g. "| --- | --- |") and removes the header,
    the separator and following rows until the first blank line or a
    non-table line.
    """
    lines = text.splitlines(keepends=True)
    out_lines = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        # candidate header: contains '|' and at least one alphanumeric
        if '|' in line and re.search(r'\w', line):
            # lookahead for separator line on next line
            if i + 1 < n and re.match(r"^\s*\|?\s*[:\- ]{3,}.*\|?\s*$", lines[i+1]):
                # start of table, skip header and separator
                i += 2
                # skip following table rows (lines containing '|' or cells)
                while i < n and ("|" in lines[i] or lines[i].strip() and re.search(r"\S\s+\S", lines[i])):
                    # stop when we hit a blank line that likely ends the table
                    if lines[i].strip() == '':
                        i += 1
                        break
                    i += 1
                # after skipping, continue without appending table lines
                continue
        out_lines.append(line)
        i += 1

    new_text = ''.join(out_lines)
    # collapse excessive blank lines
    new_text = re.sub(r"\n{3,}", "\n\n", new_text)
    return new_text


def remove_table_like_segments(text):
    """Remove table-like segments detected by scripts/extract_tables_md.py

    This reuses the extractor's start/end logic (TABLE n start signal and
    configured end signals) to identify table segments and remove them.
    """
    try:
        spec_path = Path(__file__).parent / 'extract_tables_md.py'
        spec = importlib.util.spec_from_file_location('extract_tables_md', str(spec_path))
        extractmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extractmod)
    except Exception:
        # If module can't be loaded, return original text
        return text

    cfg = extractmod.default_config()
    results = extractmod.extract_tables_from_text(text, cfg, debug=False)
    if not results:
        return text

    # Remove ranges from end->start to preserve indices
    new_text = text
    for r in sorted(results, key=lambda x: x['start_pos'], reverse=True):
        s = r['start_pos']
        e = r['end_pos']
        # safety: ensure indices are within bounds
        s = max(0, min(len(new_text), s))
        e = max(0, min(len(new_text), e))
        new_text = new_text[:s] + new_text[e:]

    new_text = re.sub(r"\n{3,}", "\n\n", new_text)
    return new_text


def remove_images_section(text):
    """Remove a trailing '## Images' section (heading and following content).

    This removes a markdown heading like '## Images' and everything from that
    heading to the next markdown heading (if any) or to EOF. Case-insensitive.
    """
    m = re.search(r'(?im)^\s*##\s*Images\b.*$', text)
    if not m:
        return text
    start = m.start()
    # find next heading after the marker; if none, remove to EOF
    next_h = re.search(r'(?im)^\s*#{1,6}\s+.*$', text[m.end():])
    if next_h:
        end = m.end() + next_h.start()
    else:
        end = len(text)
    new_text = text[:start] + text[end:]
    new_text = re.sub(r"\n{3,}", "\n\n", new_text)
    return new_text


def remove_images_trailing(text):
    """Aggressively remove a trailing '## Images' section by deleting from the
    last occurrence of a '## Images' heading to EOF.

    This is more aggressive than `remove_images_section` and will remove the
    remainder of the file after the final '## Images' heading (case-insensitive).
    """
    # find all matches and pick the last
    matches = list(re.finditer(r'(?im)^\s*##\s*images\b', text))
    if matches:
        last = matches[-1]
        start = last.start()
        new_text = text[:start]
    else:
        # fallback: find the last literal occurrence of '## images' case-insensitive
        idx = text.lower().rfind('## images')
        if idx == -1:
            return text
        new_text = text[:idx]
    new_text = re.sub(r"\n{3,}", "\n\n", new_text)
    return new_text


def remove_leading_author(text):
    # Identify the first paragraph (up to first blank line)
    parts = re.split(r"\n\s*\n", text, maxsplit=1)
    first = parts[0]
    rest = parts[1] if len(parts) > 1 else ''

    # Find the first run of consecutive ALL-CAPS words (allow initials with dots)
    words = re.findall(r"\S+", first)
    run_start = None
    run_end = None
    for i in range(len(words)):
        j = i
        while j < len(words) and re.match(r"^[A-Z][A-Z\.\-]*$", words[j]):
            j += 1
        if j > i:
            run_start, run_end = i, j
            break

    if run_start is None:
        return text

    # Reconstruct positions to remove the substring from the first paragraph
    # Find index of the run's first and last token in 'first'
    # Use incremental search to be robust to spacing
    idx = 0
    first_lower = first
    start_idx = None
    for k, w in enumerate(words):
        pos = first.find(w, idx)
        if pos == -1:
            break
        if k == run_start:
            start_idx = pos
        if k == run_end - 1:
            end_idx = pos + len(w)
            break
        idx = pos + len(w)

    if start_idx is None:
        return text

    # Remove the substring and tidy spacing
    new_first = first[:start_idx] + first[end_idx:]
    new_first = re.sub(r"\s{2,}", " ", new_first).strip()

    if rest:
        new_text = new_first + "\n\n" + rest
    else:
        new_text = new_first

    # If removing author left an empty heading line (like an initial '#'), tidy it
    new_text = re.sub(r"^#\s*\n", "", new_text, flags=re.M)
    return new_text


def process_file(p: Path, mod, backup=True):
    text = p.read_text(encoding='utf-8')
    orig = text

    # Step 1: remove captions and images
    text = remove_captions_and_images(text, mod)

    # Step 2: remove 'Further reading' section
    text = remove_further_reading(text)

    # Step 2.5: remove markdown tables
    text = remove_markdown_tables(text)

    # Step 2.75: remove table-like segments detected by extract_tables_md
    text = remove_table_like_segments(text)

    # Step 2.9: remove trailing '## Images' section
    text = remove_images_section(text)

    # Step 2.95: aggressively remove any trailing '## Images' remainder
    text = remove_images_trailing(text)

    # Step 3: remove leading all-caps author in first paragraph
    text = remove_leading_author(text)

    # Final tidying: strip trailing spaces and ensure newline at EOF
    text = text.rstrip() + '\n'

    if text != orig:
        if backup:
            bak = p.with_suffix(p.suffix + '.bak')
            shutil.copy2(p, bak)
        p.write_text(text, encoding='utf-8')
        return True
    return False


def process_path(path: Path, recursive: bool, backup: bool):
    mod = load_extract_module()
    results = []
    if path.is_file():
        if 'markdown_img' in str(path.parent) or path.name.endswith('_markdown_img.md'):
            print(f"SKIP (generated file): {path}")
            return results
        changed = process_file(path, mod, backup=backup)
        results.append((path, changed))
        return results

    it = path.rglob('*.md') if recursive else path.glob('*.md')
    for md in sorted(it):
        if 'markdown_img' in str(md.parent) or md.name.endswith('_markdown_img.md'):
            print(f"SKIP (generated file): {md}")
            continue
        changed = process_file(md, mod, backup=backup)
        results.append((md, changed))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', type=str, help='Directory of markdown files', required=True)
    ap.add_argument('--recursive', action='store_true')
    ap.add_argument('--no-backup', dest='backup', action='store_false', help='Do not create .bak backups')
    args = ap.parse_args()

    p = Path(args.dir)
    if not p.exists():
        ap.error(f"Path does not exist: {p}")

    results = process_path(p, recursive=args.recursive, backup=args.backup)
    for md, changed in results:
        print(f"{md} -> {'UPDATED' if changed else 'UNCHANGED'}")


if __name__ == '__main__':
    main()
