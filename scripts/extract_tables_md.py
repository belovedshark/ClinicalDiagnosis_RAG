#!/usr/bin/env python3
"""Extract table-like segments from markdown files.

Writes a single output markdown per input named <input-basename>-tables.md with excerpts
separated by a line containing exactly four equals signs (====). Also writes a JSON
summary if requested.

Usage examples:
  python scripts/extract_tables_md.py --input Processed/markdown/1---A-...md --output-dir ./out

The script keeps external dependencies to the Python stdlib.
"""
import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def compile_regex(pattern: str, case_sensitive: bool) -> re.Pattern:
    flags = re.MULTILINE
    if not case_sensitive:
        flags |= re.IGNORECASE
    return re.compile(pattern, flags)


def find_start_matches(text: str, start_re: re.Pattern) -> List[re.Match]:
    return list(start_re.finditer(text))


def find_earliest_end(text: str, start_pos: int, end_res: List[Tuple[re.Pattern, str]], debug: bool = False) -> Optional[Tuple[re.Match, str]]:
    """Return the earliest (closest) end-match and the pattern string that matched.

    end_res is a list of tuples (compiled_regex, pattern_string).
    """
    earliest = None
    earliest_pat = None
    for r, pat in end_res:
        m = r.search(text, pos=start_pos)
        if m:
            if debug:
                snippet = text[m.start(): m.end()]
                print(f"  [debug] end-signal pattern '{pat}' matched at {m.start()}..{m.end()} -> '{snippet}'")
            if earliest is None or m.start() < earliest.start():
                earliest = m
                earliest_pat = pat
    if earliest:
        return earliest, earliest_pat
    return None


def nth_word_end_pos(text: str, start_pos: int, n: int) -> int:
    # Find character index after the Nth word starting from start_pos
    it = re.finditer(r"\S+", text[start_pos:])
    last_end = start_pos
    count = 0
    for m in it:
        count += 1
        last_end = start_pos + m.end()
        if count >= n:
            break
    return last_end


def line_number_of_pos(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def extract_tables_from_text(text: str, config: dict, debug: bool = False) -> List[dict]:
    # Start signal should be case-sensitive (look for capitalized 'TABLE')
    start_re = compile_regex(config["start_signal_regex"], True)
    # End signals should be case-insensitive
    end_res = [(compile_regex(p, False), p) for p in config.get("end_signals", [])]

    matches = find_start_matches(text, start_re)
    results = []
    if debug:
        print(f"[debug] Found {len(matches)} start match(es)")
        for mi, mm in enumerate(matches, start=1):
            snippet = text[mm.start(): mm.end()+50].splitlines()[0]
            print(f"  [debug] start#{mi}: pos={mm.start()}..{mm.end()} -> '{snippet}'")
    for idx, m in enumerate(matches, start=1):
        # start at the match start (useful when the markdown has few newlines)
        start_idx = m.start()

        # search for earliest end AFTER the match end to avoid earlier anchors
        search_start = m.end()
        end_match_info = find_earliest_end(text, search_start, end_res, debug=debug)
        end_match = end_match_info[0] if end_match_info else None
        end_pattern = end_match_info[1] if end_match_info else None
        include_end = config.get("include_end_signal_line", False)
        if end_match:
            end_idx = end_match.start()
            if debug:
                print(f"  [debug] selected end at {end_idx} using pattern {end_pattern}")
            if include_end:
                # include the end line as well
                nl = text.find("\n", end_match.start())
                end_idx = nl + 1 if nl != -1 else len(text)
        else:
            # fallback to max_words
            max_words = int(config.get("max_words", 500))
            end_idx = nth_word_end_pos(text, start_idx, max_words)

        excerpt = text[start_idx:end_idx].strip()
        results.append(
            {
                "id": idx,
                "start_pos": start_idx,
                "end_pos": end_idx,
                "start_line": line_number_of_pos(text, start_idx),
                "end_line": line_number_of_pos(text, end_idx),
                "word_count": len(re.findall(r"\S+", excerpt)),
                "excerpt": excerpt,
            }
        )
        if debug:
            print(f"  [debug] excerpt for start#{idx}: chars {start_idx}..{end_idx}, approx words {len(re.findall(r'\S+', text[start_idx:end_idx].strip()))}")

    return results


def process_file(path: Path, config: dict, output_dir: Optional[Path], output_json: Optional[Path], individual_files: bool, debug: bool = False) -> List[dict]:
    text = path.read_text(encoding="utf-8")
    results = extract_tables_from_text(text, config, debug=debug)

    basename = path.stem
    out_md_name = f"{basename}-tables.md"
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_md_path = output_dir / out_md_name
    else:
        out_md_path = path.parent / out_md_name

    # Write combined markdown with ==== separators
    with out_md_path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(results):
            f.write(r["excerpt"].rstrip())
            if i != len(results) - 1:
                f.write("\n====\n")
    if debug:
        print(f"[debug] Wrote combined markdown to {out_md_path} ({len(results)} excerpt(s))")

    # Optionally write individual files
    if individual_files and output_dir:
        for r in results:
            fname = output_dir / f"{basename}_table_{r['id']}.md"
            fname.write_text(r["excerpt"], encoding="utf-8")
            if debug:
                print(f"[debug] Wrote individual file {fname}")

    # Write JSON summary if requested
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        summary = {"source": str(path), "tables": results}
        output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return results


def default_config() -> dict:
    return {
    "start_signal_regex": r"\bTABLE\s+\d+(?:\.\d+)?\b",
        # Use simple, unanchored end signals (match anywhere).
        # '#' is used as a plain symbol (no start-of-line requirement) to catch page markers like '10 # Page 2'.
        "end_signals": [
            r"(?i)\bTABLE\b",
            r"(?i)\bFIGURE\b",
            r"(?i)\bCHAPTER\b",
            r"#",
            r"(?i)\bREFERENCES\b",
        ],
        "max_words": 500,
        "include_end_signal_line": False,
        "context_lines": 0,
        "case_sensitive": False,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Extract table segments from markdown files")
    p.add_argument("--input", required=True, help="Path to markdown file or directory containing markdown files")
    p.add_argument("--output-dir", help="Directory to place outputs (default: same directory as source)")
    p.add_argument("--output-json", help="Path to write JSON summary (optional)")
    p.add_argument("--max-words", type=int, help="Max words fallback when no end signal found")
    p.add_argument("--start-regex", help="Start signal regex (overrides default)")
    p.add_argument("--end-signals", nargs="*", help="List of end-signal regexes (space separated) to search for")
    p.add_argument("--include-end", action="store_true", help="Include the matched end-signal line in the excerpt")
    p.add_argument("--individual-files", action="store_true", help="Also write individual table_<n>.md files in output-dir")
    p.add_argument("--case-sensitive", action="store_true", help="Make regex matching case-sensitive")
    p.add_argument("--debug", action="store_true", help="Print debug information during extraction")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = default_config()
    if args.max_words:
        cfg["max_words"] = args.max_words
    if args.start_regex:
        cfg["start_signal_regex"] = args.start_regex
    if args.end_signals:
        cfg["end_signals"] = args.end_signals
    if args.include_end:
        cfg["include_end_signal_line"] = True
    if args.case_sensitive:
        cfg["case_sensitive"] = True

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else None
    output_json = Path(args.output_json) if args.output_json else None

    targets = []
    debug = getattr(args, "debug", False)
    if input_path.is_dir():
        targets = list(input_path.glob("*.md"))
    elif input_path.is_file():
        targets = [input_path]
    else:
        print(f"Input path {input_path} not found")
        return
    overall_summary = []
    for t in targets:
        print(f"Processing {t}")
        # per-file json path default
        per_json = output_json
        if per_json is None and output_dir:
            per_json = output_dir / f"{t.stem}-tables.json"
        elif per_json is None:
            per_json = t.parent / f"{t.stem}-tables.json"
        res = process_file(t, cfg, output_dir, per_json, args.individual_files, debug=debug)
        overall_summary.append({"source": str(t), "tables": res})
    # write combined json in output_dir if requested via --output-json
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(overall_summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
