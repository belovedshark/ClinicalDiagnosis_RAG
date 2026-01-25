import os
import re
import json
from typing import List, Dict, Optional


def list_markdown_records(processed_md_dir: str) -> List[str]:
    files = [f for f in os.listdir(processed_md_dir) if f.lower().endswith('.md')]
    files.sort()
    return files


def read_file_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def extract_image_paths_from_markdown(md_text: str) -> List[str]:
    # match markdown image syntax ![alt](path)
    img_pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    matches = img_pattern.findall(md_text)
    # also match HTML <img src="...">
    html_pattern = re.compile(r"<img[^>]+src=[\"']([^\"']+)[\"'][^>]*>")
    matches += html_pattern.findall(md_text)
    # normalize
    cleaned = [m.strip() for m in matches]
    return cleaned


def extract_plain_text_from_markdown(md_text: str) -> str:
    # Very small markdown -> plain text stripper. This is not a full md parser
    text = md_text
    # remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # remove inline code
    text = re.sub(r"`[^`]*`", "", text)
    # remove images and links but keep alt text for images
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # remove headings markup
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    # remove remaining markdown characters (*, >, -, etc.) at line starts
    text = re.sub(r"^[\-\*>\s]+", "", text, flags=re.M)
    # collapse multiple blank lines
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def find_matching_table_file(basename: str, table_dir: str) -> Optional[str]:
    if not table_dir or not os.path.isdir(table_dir):
        return None
    # direct startswith match
    for f in os.listdir(table_dir):
        if f.startswith(basename):
            return os.path.join(table_dir, f)
    # try matching by numeric prefix (e.g., '12' for '12---...')
    prefix = basename.split('---')[0]
    if prefix:
        for f in os.listdir(table_dir):
            if f.startswith(prefix + '---') or f.startswith(prefix + '_') or f.startswith(prefix + '-') or f.startswith(prefix + '.') or f.startswith(prefix):
                return os.path.join(table_dir, f)
    return None


def get_table_dir(processed_dir: str) -> Optional[str]:
    """Return the existing table directory under processed_dir, checking common names."""
    candidates = [
        os.path.join(processed_dir, 'out_all_structured'),
        os.path.join(processed_dir, 'out_all_strcutured'),
        os.path.join(processed_dir, 'out_all'),
        os.path.join(processed_dir, 'out_all_structured/'),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def load_table_structured(path: str):
    # Try JSON then CSV then return raw text
    try:
        if path.lower().endswith('.json'):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif path.lower().endswith('.csv'):
            import csv
            rows = []
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
            return rows
        else:
            # fallback: return file text
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception:
        return None
