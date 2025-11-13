#!/usr/bin/env python3
"""
embedding_clip.py

Chunk markdowns under <input_root>/markdown and compute CLIP text & image embeddings.
Saves embeddings to <output_root>/embeddings.

python scripts/embedding_clip.py --input-root Processed --output-root Processed_embeddings --model openai/clip-vit-base-patch32 --device cpu

Usage:
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import re
from typing import List, Dict, Optional

def tokenize_chunks(tokenizer, text: str, max_tokens:int, overlap:int):
    # token_ids = tokenizer.encode(text, add_special_tokens=False)
    encoded = tokenizer(text, add_special_tokens=False)['input_ids']
    chunks = []
    step = max_tokens - overlap if max_tokens > overlap else max_tokens
    i = 0
    cid = 0
    while i < len(encoded):
        chunk_ids = encoded[i:i+max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        chunks.append({'id': cid, 'text': chunk_text, 'start_token': i, 'end_token': i+len(chunk_ids)})
        cid += 1
        if i + max_tokens >= len(encoded):
            break
        i += step
    chunks = merge_short_chunks(chunks)
    return chunks

def chunk_text_charwise(text: str, max_chars:int, overlap:int):
    if not text:
        return []
    chunks = []
    start = 0
    step = max_chars - overlap if max_chars > overlap else max_chars
    cid = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        snippet = text[start:end].strip()
        if snippet:
            chunks.append({'id': cid, 'text': snippet, 'start_char': start, 'end_char': end})
            cid += 1
        if end == len(text):
            break
        start += step
    return chunks

def merge_short_chunks(chunks, min_len=50):
    merged = []
    buffer = ""
    for c in chunks:
        if len(c["text"]) < min_len:
            buffer += " " + c["text"]
        else:
            if buffer:
                merged.append({"id": len(merged), "text": buffer.strip()})
                buffer = ""
            merged.append(c)
    if buffer:
        merged.append({"id": len(merged), "text": buffer.strip()})
    return merged


def clean_text(text: str) -> str:
    """Lightweight cleaning for OCR/extraction artifacts:
    - remove page header lines like 'page 12' or '# page 12'
    - collapse spaces between digits (e.g. '1 4' -> '14')
    - normalize spaces around hyphens (e.g. ' - ' -> '-')
    - collapse extra whitespace and trim
    """
    if not text:
        return text

    # Normalize unicode whitespace (non-breaking, zero-width) to regular spaces
    text = re.sub(r'[\u00A0\u200B\u202F]+', ' ', text)

    # Remove page header lines like '# Page 12' but avoid removing long lines
    # that accidentally begin with the word 'Page' followed by the article title
    # (some files have long single-line paragraphs). Filter line-by-line
    # and only drop short header lines.
    original_text = text
    lines = text.splitlines()
    kept_lines = []
    for ln in lines:
        if re.match(r'(?i)^[#\s]*page\b', ln):
            # drop only if the line looks like a short page header
            if len(ln.strip()) < 120:
                continue
        kept_lines.append(ln)
    text = "\n".join(kept_lines)

    # Remove obvious image/file markers like '_2022_' or 'img 1'
    text = re.sub(r'[_\-]*\s*\d{4}\s*[_\-]*', ' ', text)
    text = re.sub(r'img\s*\d+', '', text, flags=re.IGNORECASE)

    # Remove underscores and multiple hyphens
    text = re.sub(r'[_\-]{2,}', '-', text)

    # Collapse spaced-out letters/numbers: '2 0 2 2' -> '2022', 'c a s e' -> 'case'
    text = re.sub(r'(?<=\b\w)\s+(?=\w\b)', '', text)

    # Remove isolated underscores and stray punctuation
    text = re.sub(r'[_/\\]+', ' ', text)

    # Normalize spaces around hyphens or commas
    text = re.sub(r'\s*-\s*', '-', text)
    text = re.sub(r'\s*,\s*', ', ', text)

    # Collapse multiple spaces or tabs
    text = re.sub(r'[ \t]+', ' ', text)

    # Collapse multiple blank lines
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    # Remove leftover punctuation artifacts
    text = re.sub(r'[-–]{2,}', '-', text)
    text = re.sub(r'[()]', '', text)
    
    # Join split words from OCR (e.g. "infec- tion" → "infection")
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    
    # Normalize spacing around punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    # --- Additional safe fixes for numeric and OCR artifacts ---
    # Join digit groups that were split by spaces (e.g. '1 0 3' -> '103')
    # Join any whitespace between digits (covers NBSP/zero-width added above)
    text = re.sub(r'(?<=\d)\s+(?=\d)', '', text)
    # Ensure a space before common medical units when they were glued to numbers
    text = re.sub(r'(?i)(\d+)(?=(mmhg|bpm|kg|cm|%))', r"\1 ", text)
    # Normalize common unit casing
    text = re.sub(r'(?i)\bmmhg\b', 'mmHg', text)
    # --- Blood-pressure reconstruction heuristic ---
    # Convert four-digit numbers followed by mmHg (e.g. '9060 mmHg') into '90/60 mmHg'
    # only when the split into two 2-digit numbers yields plausible BP values.
    def _format_bp(match):
        num = match.group(1)
        unit = match.group(2)
        # split into two 2-digit parts
        a = int(num[:2])
        b = int(num[2:])
        # sanity ranges: systolic 30-250, diastolic 20-200
        if 30 <= a <= 250 and 20 <= b <= 200:
            return f"{a}/{b} {unit}"
        return match.group(0)

    text = re.sub(r'\b(\d{4})\s*(mmHg)\b', _format_bp, text)
    # Fix decimals where spaces surround the dot: '39 . 6' -> '39.6'
    text = re.sub(r'(?<=\d)\s*\.\s*(?=\d)', '.', text)
    # Remove spaces between digits and percent sign: '1 5 %' -> '15%'
    text = re.sub(r'(?<=\d)\s+%(?=\s|$)', '%', text)
    # Collapse spaced-out single letters/numbers left from OCR (e.g. 'c a s e' -> 'case')
    # But be conservative: only collapse when it's sequences of single letters separated by spaces
    text = re.sub(r'(?:(?<=\s)|^)(?:[A-Za-z0-9]\s+){2,}[A-Za-z0-9](?=(?:\s|$))',
                  lambda m: m.group(0).replace(' ', ''), text)

    # Remove stray multi-space clusters again
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def extract_tables_from_markdown(text: str) -> List[Dict[str, Optional[str]]]:
    """Find simple markdown-style tables (pipe-separated) and TABLE label/caption blocks.
    Returns a list of dicts: {'text': <table_text>, 'label': <label or None>, 'caption': <caption or None>}.

    Heuristic approach:
    - Detect contiguous blocks where a header line contains '|' and the next line looks like a separator (---|:---)
      (common markdown table). Capture the whole contiguous block of pipe-containing lines.
    - For each captured block, look up to 3 non-empty previous lines for a TABLE label/caption like "TABLE 4.1 ..." or "Table 4.1 ...".
    - Remove found table blocks from the original text by replacing with a single blank line to avoid double-embedding.
    """
    lines = text.splitlines()
    i = 0
    n = len(lines)
    tables = []
    to_remove_ranges = []
    inline_substrings: List[str] = []

    def find_label_caption(before_lines: List[str]) -> tuple[Optional[str], Optional[str]]:
        label = None
        caption = None
        for ln in reversed(before_lines[-3:]):
            m = re.match(r'(?i)^(TABLE|Table)\s+([^\-\n\r]+)\s*-?\s*(.*)$', ln.strip())
            if m:
                label = m.group(1) + ' ' + m.group(2).strip()
                # remaining part may be caption
                rem = m.group(3).strip()
                if rem:
                    caption = rem
                break
        return label, caption

    while i < n:
        # pipe-style table detection: a line with | and following line that contains >=3 hyphens or pipes
        if '|' in lines[i] and i + 1 < n and re.search(r'^[\s\|:>-]*-+[\s\|:-]*$', lines[i+1]):
            start = i
            # capture preceding header line if it's part of the table header
            # grab contiguous pipe-containing lines
            j = i
            while j < n and ('|' in lines[j] or re.match(r'^[\s\|:>-]*-+[\s\|:-]*$', lines[j])):
                j += 1
            table_block = '\n'.join(lines[start:j]).strip()
            # look back for label/caption within previous 3 non-empty lines
            before = [ln for ln in lines[:start] if ln.strip()]
            label, caption = find_label_caption(before)
            tables.append({'text': table_block, 'label': label, 'caption': caption})
            to_remove_ranges.append((start, j))
            i = j
            continue
        # fallback: a standalone TABLE label line followed by an indented/code/table block
        m = re.match(r'(?i)^(TABLE|Table)\b', lines[i].strip())
        if m:
            start = i
            j = i + 1
            # include subsequent non-empty lines until a blank line or a heading/page marker
            # but limit capture to avoid swallowing the whole document when tables are inline in long paragraphs
            max_lines = 20
            while j < n and j < start + max_lines and lines[j].strip() and not re.match(r'(?i)^#\s*page\b', lines[j]) and not re.match(r'^#{1,6}\s', lines[j]):
                j += 1
            block = '\n'.join(lines[start:j]).strip()
            label = lines[start].strip()
            caption = None
            # try to extract a short caption from the first line after label if present
            if start + 1 < n and lines[start + 1].strip():
                caption = lines[start + 1].strip()
            tables.append({'text': block, 'label': label, 'caption': caption})
            to_remove_ranges.append((start, j))
            i = j
            continue

        # additional heuristic: detect inline 'TABLE' occurrences embedded in paragraphs
        if re.search(r'(?i)\bTABLE\b', lines[i]) and '|' not in lines[i]:
            # capture from this line until the next blank line or page/heading marker
            # but limit the number of lines captured to avoid greedy matches in single-paragraph files
            start = i
            j = i + 1
            max_lines_inline = 10
            while j < n and j < start + max_lines_inline and lines[j].strip() and not re.match(r'(?i)^#\s*page\b', lines[j]) and not re.match(r'^#{1,6}\s', lines[j]):
                j += 1
            # extract the label token from the line if present
            lm = re.search(r'(?i)(TABLE\s*\d+[\.\d]*)', lines[i])
            label = lm.group(0) if lm else None
            caption = None
            if lm:
                rest = lines[i][lm.end():].strip()
                if rest:
                    caption = rest[:120].strip()

            # If the file is a single very long line or the line itself is huge, extract a window
            # around the TABLE match instead of swallowing the whole long line.
            LONG_LINE_THRESHOLD = 800
            WINDOW = 400
            line_text = lines[i]
            if lm and (len(line_text) > LONG_LINE_THRESHOLD or (n == 1 and len(line_text) > 400)):
                sidx, eidx = lm.span()
                start_char = max(0, sidx - WINDOW)
                end_char = min(len(line_text), eidx + WINDOW)
                block = line_text[start_char:end_char].strip()
                # record substring to remove later from the large single line
                inline_substrings.append(block)
                tables.append({'text': block, 'label': label, 'caption': caption})
                i = j
                continue

            block = '\n'.join(lines[start:j]).strip()
            tables.append({'text': block, 'label': label, 'caption': caption})
            to_remove_ranges.append((start, j))
            i = j
            continue

        i += 1

    # Remove ranges from the original text by replacing with a single blank line (preserve positions roughly)
    if to_remove_ranges:
        new_lines = []
        ri = 0
        ranges = sorted(to_remove_ranges)
        for (s, e) in ranges:
            # append lines before s that haven't been appended
            while ri < s and ri < n:
                new_lines.append(lines[ri])
                ri += 1
            # insert a blank line placeholder
            new_lines.append('')
            ri = e
        while ri < n:
            new_lines.append(lines[ri])
            ri += 1
        new_text = '\n'.join(new_lines)
    else:
        new_text = text

    return tables, new_text


def parse_structured_tables_file(text: str) -> List[Dict[str, Optional[str]]]:
    """Parse an out_all_structured file content into table blocks.
    Strategy: iterate lines, start a block when a line begins with TABLE/Table, collect until a blank line.
    Returns list of {'text', 'label', 'caption'}.
    """
    lines = text.splitlines()
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i].strip()
        if re.match(r'(?i)^(TABLE|Table)\b', ln):
            start = i
            # include subsequent lines until a blank line
            j = i + 1
            while j < n and lines[j].strip() != '':
                j += 1
            block_lines = lines[start:j]
            block = '\n'.join(block_lines).strip()
            # extract label and caption from first line
            m = re.match(r'(?i)^(TABLE\s*[^\s]*)(?:\s*(.*))?$', block_lines[0].strip())
            if m:
                label = m.group(1).strip()
                caption = m.group(2).strip() if m.group(2) else None
            else:
                label = None
                caption = None
            blocks.append({'text': block, 'label': label, 'caption': caption})
            i = j
            continue
        i += 1
    return blocks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', default='Processed', help='Root with markdown/ and images/')
    parser.add_argument('--output-root', default='Processed', help='Root to write embeddings into <output-root>/embeddings/')
    parser.add_argument('--model', default='openai/clip-vit-base-patch32', help='HF CLIP model name')
    parser.add_argument('--device', default='cpu', help='torch device (cpu or cuda)')
    parser.add_argument('--max-tokens', type=int, default=77, help='Max tokens per text chunk (CLIP default 77)')
    parser.add_argument('--token-overlap', type=int, default=10, help='Token overlap between chunks')
    parser.add_argument('--max-chars', type=int, default=1000, help='(fallback) char chunk size when tokenizer unavailable')
    args = parser.parse_args()

    input_root = Path(args.input_root)
    markdown_dir = input_root / 'markdown'
    images_root = input_root / 'images'
    out_emb = Path(args.output_root) / 'embeddings'
    # out_emb.mkdir(parents=True, exist_ok=True)
    out_text = out_emb / 'text'; out_img = out_emb / 'image'
    out_text.mkdir(parents=True, exist_ok=True)
    out_img.mkdir(parents=True, exist_ok=True)

    if not markdown_dir.exists():
        print('Markdown folder not found:', markdown_dir)
        return

    # Load CLIP via transformers
    try:
        from transformers import CLIPModel, CLIPProcessor
        import torch
    except Exception as e:
        raise RuntimeError('Install transformers and torch: pip install transformers torch') from e

    model = CLIPModel.from_pretrained(args.model).to(args.device)
    processor = CLIPProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer  # use tokenizer for token-based chunking

    summary = []
    for md in sorted(markdown_dir.glob('*.md')):
        stem = md.stem
        text = md.read_text(encoding='utf-8')

        # --- Remove image blocks / references BEFORE cleaning ---
        # Do this prior to clean_text so the cleaning rules don't accidentally
        # collapse or remove the main prose and leave only image markers.
        # Split at a heading like '## Images' (case-insensitive) if present and keep the part before it.
        parts = re.split(r"\n#{1,6}\s*images\b", text, flags=re.IGNORECASE)
        if parts:
            text = parts[0]

        # Remove inline markdown image references: ![alt](path)
        text = re.sub(r'!\[[^\]]*\]\([^\)]*\)', ' ', text)
        # Remove stray image path fragments like '../images/...' or filenames ending in .jpeg/.jpg/.png
        text = re.sub(r'\.{2}/images/\S+', ' ', text)
        text = re.sub(r'\S+\.(?:jpe?g|png|webp)\b', ' ', text)

        # lightweight cleaning to fix OCR/extraction spacing artifacts
        text = clean_text(text)

        # Prefer structured table files under <input_root>/out_all_structured if present
        tables = []
        structured_dir = input_root / 'out_all_structured'
        if structured_dir.exists():
            candidates = sorted(structured_dir.glob(f"{stem}*tables.md"))
            if candidates:
                for cf in candidates:
                    s = cf.read_text(encoding='utf-8')
                    parsed = parse_structured_tables_file(s)
                    if parsed:
                        tables.extend(parsed)
        # If no structured tables found, fall back to inline detection and removal
        if not tables:
            tables, text = extract_tables_from_markdown(text)

        # Token-based chunking (preferred) for the remaining prose
        chunks = tokenize_chunks(tokenizer, text, max_tokens=args.max_tokens, overlap=args.token_overlap)
        if not chunks:
            # fallback to char-chunking
            chunks = chunk_text_charwise(text, max_chars=args.max_chars, overlap=int(args.max_chars*0.1))

        texts = [c['text'] for c in chunks]
        # text embeddings saved under <output-root>/embeddings/text/
        text_emb_path = out_text / (stem + '_text.npy')

        if texts:
            # process in batches to avoid OOM
            batch_size = 16
            all_embs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = processor(text=batch, return_tensors='pt', padding=True, truncation=True).to(args.device)
                with torch.no_grad():
                    emb = model.get_text_features(**inputs)
                all_embs.append(emb.cpu().numpy())
            all_emb = np.vstack(all_embs)
        else:
            # model.text_projection may be a torch.nn.Linear (has out_features) or a Tensor.
            try:
                out_dim = int(model.text_projection.out_features)
            except Exception:
                try:
                    out_dim = int(model.text_projection.weight.shape[0])
                except Exception:
                    # fallback: try numpy shape attribute (if it's a Tensor)
                    out_dim = int(getattr(model.text_projection, 'shape', (0, 0))[1])
            all_emb = np.zeros((0, out_dim), dtype=np.float32)

        # --- Table embeddings (embed each table block as a single chunk) ---
        table_emb_path = out_text / (stem + '_table.npy')
        table_texts = [t['text'] for t in tables]
        if table_texts:
            batch_size = 16
            all_table_embs = []
            for i in range(0, len(table_texts), batch_size):
                batch = table_texts[i:i+batch_size]
                inputs = processor(text=batch, return_tensors='pt', padding=True, truncation=True).to(args.device)
                with torch.no_grad():
                    emb = model.get_text_features(**inputs)
                all_table_embs.append(emb.cpu().numpy())
            table_all_emb = np.vstack(all_table_embs)
        else:
            # if no table texts, create empty array with same dim as text projection
            try:
                tdim = int(model.text_projection.out_features)
            except Exception:
                try:
                    tdim = int(model.text_projection.weight.shape[0])
                except Exception:
                    tdim = int(getattr(model.text_projection, 'shape', (0, 0))[1])
            table_all_emb = np.zeros((0, tdim), dtype=np.float32)

        np.save(text_emb_path, all_emb)
        np.save(table_emb_path, table_all_emb)

        np.save(text_emb_path, all_emb)

        # images
        img_dir = images_root / stem
        # image embeddings saved under <output-root>/embeddings/image/
        image_emb_path = out_img / (stem + '_images.npy')
        image_list = []
        if img_dir.exists():
            image_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg','.webp')])
            if image_paths:
                batch_size = 8
                all_i_emb = []
                from PIL import Image
                for i in range(0, len(image_paths), batch_size):
                    batch_paths = image_paths[i:i+batch_size]
                    images = [Image.open(p).convert('RGB') for p in batch_paths]
                    inputs = processor(images=images, return_tensors='pt').to(args.device)
                    with torch.no_grad():
                        i_emb = model.get_image_features(**inputs)
                    all_i_emb.append(i_emb.cpu().numpy())
                all_i_emb = np.vstack(all_i_emb)
                np.save(image_emb_path, all_i_emb)
                image_list = [str(p) for p in image_paths]
            else:
                try:
                    out_dim = int(model.visual_projection.out_features)
                except Exception:
                    try:
                        out_dim = int(model.visual_projection.weight.shape[0])
                    except Exception:
                        out_dim = int(getattr(model.visual_projection, 'shape', (0, 0))[1])
                np.save(image_emb_path, np.zeros((0, out_dim), dtype=np.float32))
        else:
            try:
                out_dim = int(model.visual_projection.out_features)
            except Exception:
                try:
                    out_dim = int(model.visual_projection.weight.shape[0])
                except Exception:
                    out_dim = int(getattr(model.visual_projection, 'shape', (0, 0))[1])
            np.save(image_emb_path, np.zeros((0, out_dim), dtype=np.float32))

        # Write index for this file
        # index = {
        #     'file': md.name,
        #     'n_chunks': len(chunks),
        #     'text_embedding': str(text_emb_path),
        #     'image_embedding': str(image_emb_path),
        #     'chunks': chunks,
        #     'images': image_list,
        # }
        index = {
            'file': md.name,
            'n_chunks': len(chunks) + len(tables),
            'text_embedding': str(text_emb_path),
            'table_embedding': str(table_emb_path),
            'image_embedding': str(image_emb_path),
            'chunks': [
                {'id': c['id'], 'text': c['text'], 'type': 'text'} for c in chunks
            ],
            'tables': [
                {'id': idx, 'text': t.get('text'), 'label': t.get('label'), 'caption': t.get('caption'), 'type': 'table'}
                for idx, t in enumerate(tables)
            ],
            'images': [{'path': p, 'type': 'image'} for p in image_list],
        }
        (out_emb / (stem + '.json')).write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')
        summary.append(index)
    metadata = {
        'model': args.model,
        'device': args.device,
        'num_files': len(summary),
        'embedding_dim_text': all_emb.shape[1] if len(summary) and 'text_embedding' in summary[-1] else 0,
        'embedding_dim_table': table_all_emb.shape[1] if len(summary) and 'table_embedding' in summary[-1] else 0,
        'embedding_dim_image': all_i_emb.shape[1] if len(summary) and 'image_embedding' in summary[-1] else 0
    }
    (out_emb / 'summary.json').write_text(json.dumps(metadata, indent=2))
    print('Saved embeddings to', out_emb)

if __name__ == '__main__':
    main()