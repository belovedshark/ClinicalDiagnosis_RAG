#!/usr/bin/env python3
"""
embedding_clip.py

Upgraded embedding pipeline:
- Reading-order assembly from: markdown (text), markdown_img (captions + image paths), out_all_structured (tables), images dir
- Semantic chunking (sentence split -> group to target token count with overlap)
- Text & table embeddings with a GTE-style model (default: thelper/gte-small)
- Image embeddings with CLIP image encoder (openai/clip-vit-base-patch32)
- Avoid passing long text to CLIP (CLIP used only for images)
- Outputs per-case .npy embeddings and a JSON index preserving reading order

Usage example:
python scripts/embedding_clip.py --input-root Processed --output-root Processed_embeddings \
  --text-model thelper/gte-small --image-model openai/clip-vit-base-patch32 --device cpu

"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import re
from typing import List, Dict, Optional, Tuple
import sys
import os

# try to import nltk punkt
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except Exception:
    try:
        import nltk
        nltk.download('punkt')
    except Exception:
        pass

# ------------------ Utilities ------------------

def safe_read_text(p: Path) -> str:
    try:
        return p.read_text(encoding='utf-8')
    except Exception:
        try:
            return p.read_text(encoding='latin-1')
        except Exception:
            return ''


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


# simple caption parser for markdown_img files
CAPTION_LINE_RE = re.compile(r"^\s*(?:[-*\u2022]?\s*)?(Fig(?:ure)?\.?)\s*\d+(?:\.\d+)*(?:[A-Za-z])?\b.*", re.I)
IMAGE_MD_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
PATH_SUFFIX_RE = re.compile(r"\.(?:jpe?g|png|webp|bmp)$", re.I)


def parse_markdown_img(md_text: str) -> List[Dict[str, str]]:
    """Parse a markdown_img file into a list of {'label','caption','path'} preserving order."""
    lines = md_text.splitlines()
    items: List[Dict[str, str]] = []
    for ln in lines:
        ln_strip = ln.strip()
        # try line-level caption
        if CAPTION_LINE_RE.match(ln_strip):
            # capture the caption line and find any path within same or next lines
            caption = ln_strip
            items.append({'label': None, 'caption': caption, 'path': None})
            continue
        # try inline image markdown
        m = IMAGE_MD_RE.search(ln)
        if m:
            path = m.group(1).strip()
            # attach to last caption if present and missing path
            if items and items[-1]['path'] is None:
                items[-1]['path'] = path
            else:
                items.append({'label': None, 'caption': None, 'path': path})
    # normalize paths (strip surrounding quotes, spaces)
    for it in items:
        if it['path']:
            it['path'] = it['path'].strip().strip('"').strip("'")
    return items


# ------------------ Chunking ------------------

def sentence_split(text: str) -> List[str]:
    try:
        from nltk.tokenize import sent_tokenize
        sents = sent_tokenize(text)
        return [s.strip() for s in sents if s.strip()]
    except Exception:
        # fallback simple split
        sents = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', text) if s.strip()]
        return sents

def merge_short_chunks(chunks, min_len=80):
    """
    Merge adjacent chunks that are too short so we avoid tiny meaningless fragments.
    chunks = list of {'id','text',...}
    """
    if not chunks:
        return chunks

    merged = []
    buf = chunks[0]['text']

    for c in chunks[1:]:
        if len(buf) < min_len:
            buf += " " + c['text']
        else:
            merged.append({'id': 0, 'text': buf})
            buf = c['text']

    merged.append({'id': 0, 'text': buf})
    # reassign proper IDs
    for i, m in enumerate(merged):
        m['id'] = i

    return merged

def semantic_chunker_using_tokenizer(tokenizer, text: str, chunk_tokens: int, overlap_tokens: int) -> List[Dict]:
    """Split by sentences then group sentences until chunk_tokens (approx via tokenizer)"""
    if not text or text.strip() == '':
        return []
    sents = sentence_split(text)
    sent_tokens: List[int] = []
    for s in sents:
        try:
            ids = tokenizer(s, add_special_tokens=False)['input_ids']
            sent_tokens.append(len(ids))
        except Exception:
            sent_tokens.append(max(1, len(s.split())))

    chunks = []
    i = 0
    cid = 0
    total_sents = len(sents)
    while i < total_sents:
        cur = 0
        start = i
        parts = []
        while i < total_sents and (cur < chunk_tokens or len(parts) == 0):
            parts.append(sents[i])
            cur += sent_tokens[i]
            i += 1
        chunk_text = ' '.join(parts).strip()
        start_token = sum(sent_tokens[:start])
        end_token = start_token + cur
        chunks.append({'id': cid, 'text': chunk_text, 'start_token': start_token, 'end_token': end_token})
        cid += 1
        if i >= total_sents:
            break
        # create overlap: move i back to sentence index corresponding to end_token - overlap_tokens
        target = max(0, end_token - overlap_tokens)
        acc = 0
        new_i = 0
        for si, tlen in enumerate(sent_tokens):
            if acc + tlen > target:
                new_i = si
                break
            acc += tlen
        i = new_i
    # merge very short chunks
    chunks = merge_short_chunks(chunks, min_len=80)
    return chunks


# ------------------ Embedding helpers ------------------

def chunk_text_charwise(text: str, max_chars: int = 1000, overlap: int = 100):
    chunks = []
    start = 0
    n = len(text)
    cid = 0
    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end].strip()
        chunks.append({"id": cid, "text": chunk})
        cid += 1
        start = max(0, end - overlap)
    return chunks

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    if attention_mask is None:
        return token_embeddings.mean(1)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# ------------------ Main ------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', default='Processed')
    parser.add_argument('--output-root', default='Processed_embeddings')
    parser.add_argument('--text-model', default='thelper/gte-small')
    parser.add_argument('--image-model', default='openai/clip-vit-base-patch32')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--chunk-tokens', type=int, default=320)
    parser.add_argument('--chunk-overlap', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--markdown-img-dir', default='markdown_img')
    parser.add_argument('--tables-dir', default='out_all_structured')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    input_root = Path(args.input_root)
    markdown_dir = input_root / 'markdown'
    markdown_img_dir = input_root / args.markdown_img_dir
    images_root = input_root / 'images'
    tables_dir = input_root / args.tables_dir

    out_root = Path(args.output_root)
    out_emb = out_root / 'embeddings'
    out_text = out_emb / 'text'; out_table = out_emb / 'table'; out_img = out_emb / 'image'
    out_text.mkdir(parents=True, exist_ok=True)
    out_table.mkdir(parents=True, exist_ok=True)
    out_img.mkdir(parents=True, exist_ok=True)

    if not markdown_dir.exists():
        print('markdown folder missing:', markdown_dir)
        return

    # load models
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel, CLIPModel, CLIPProcessor
    except Exception as e:
        print('Please install transformers and torch:', e)
        return

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    print('Loading text tokenizer/model', args.text_model)
    try:
        text_tokenizer = AutoTokenizer.from_pretrained(args.text_model)
        text_model = AutoModel.from_pretrained(args.text_model).to(device)
    except Exception as e:
        print('Failed to load text model:', e)
        text_tokenizer = None
        text_model = None

    print('Loading CLIP image model', args.image_model)
    try:
        clip_model = CLIPModel.from_pretrained(args.image_model).to(device)
        clip_processor = CLIPProcessor.from_pretrained(args.image_model)
    except Exception as e:
        print('Failed to load CLIP model:', e)
        clip_model = None
        clip_processor = None

    # determine safe batch size on CPU
    def safe_batch(default):
        if device.type == 'cpu':
            return min(default, 4)
        return default
    bs = safe_batch(args.batch_size)

    summary = []

    # pre-read markdown_img files
    md_img_map: Dict[str, List[Dict[str, str]]] = {}
    if markdown_img_dir.exists():
        for mdf in sorted(markdown_img_dir.glob('*.md')):
            txt = safe_read_text(mdf)
            md_img_map[mdf.stem] = parse_markdown_img(txt)

    # pre-read structured tables
    tables_map: Dict[str, List[Dict[str, Optional[str]]]] = {}
    if tables_dir.exists():
        for tf in sorted(tables_dir.glob('*.md')):
            txt = safe_read_text(tf)
            parsed = []
            try:
                parsed = parse_structured_tables_file(txt)
            except Exception:
                parsed = []
            if parsed:
                tables_map[tf.stem] = parsed

    # iterate markdown files
    for md in sorted(markdown_dir.glob('*.md')):
        stem = md.stem
        raw = safe_read_text(md)
        if not raw.strip():
            print('Skipping empty', md.name)
            continue

        # split out an images section if present (we will use markdown_img mapping instead)
        # Remove inline image references from main text to avoid duplication
        text_only = re.sub(r'!\[[^\]]*\]\([^\)]+\)', ' ', raw)

        # clean common 'Further reading' and reference sections by splitting
        text_only = re.split(r'(?i)^further reading', text_only, flags=re.MULTILINE)[0]
        text_only = re.split(r'(?i)^references?:', text_only, flags=re.MULTILINE)[0]

        # We'll assemble a reading-order sequence of items: each item = (type, payload)
        # types: 'text' (string), 'image' (dict label/caption/path), 'table' (dict)
        reading_sequence: List[Tuple[str, object]] = []

        # Strategy: naive approach — go through original raw text lines and whenever we
        # encounter an image markdown or a caption token, we insert image-chunk; otherwise
        # collect paragraphs into text blocks to be chunked.

        # Create quick lookup of images/captions for this stem
        images_for_case = md_img_map.get(stem, [])
        tables_for_case = tables_map.get(stem, []) if stem in tables_map else []

        # Build a list of image placeholders positions in raw text (line index)
        raw_lines = raw.splitlines()
        image_positions = []  # list of (line_idx, path_or_none)
        for idx, ln in enumerate(raw_lines):
            m = IMAGE_MD_RE.search(ln)
            if m:
                pth = m.group(1).strip().strip('"').strip("'")
                image_positions.append((idx, pth))
            else:
                # also detect lines that look like 'Fig. X' captions
                if CAPTION_LINE_RE.match(ln.strip()):
                    image_positions.append((idx, None))

        # We'll scan raw_lines and create text blocks between image positions
        last_idx = 0
        img_idx = 0
        for pos_idx, pos in enumerate(image_positions):
            line_idx, pth = pos
            # take lines from last_idx up to line_idx as a text block
            block = '\n'.join(raw_lines[last_idx:line_idx]).strip()
            if block:
                reading_sequence.append(('text', block))
            # Insert image chunk: try to find match from markdown_img map by order
            # prefer matching by path; else assign next available caption
            assigned = None
            if pth:
                # normalize path tail
                tail = Path(pth).name
                for it in images_for_case:
                    if it.get('path') and Path(it['path']).name == tail:
                        assigned = it
                        break
            if assigned is None and images_for_case and img_idx < len(images_for_case):
                assigned = images_for_case[img_idx]
                img_idx += 1
            if assigned:
                reading_sequence.append(('image', assigned))
            else:
                # no assigned caption; create placeholder using path if present
                reading_sequence.append(('image', {'caption': None, 'path': pth}))
            last_idx = line_idx + 1
        # tail text
        tail_block = '\n'.join(raw_lines[last_idx:]).strip()
        if tail_block:
            reading_sequence.append(('text', tail_block))

        # If there are images not yet used, append them at end
        if images_for_case and any(it.get('path') for it in images_for_case):
            used_paths = {it.get('path') for ttype, it in reading_sequence if ttype == 'image' and isinstance(it, dict) and it.get('path')}
            for it in images_for_case:
                if it.get('path') and it.get('path') not in used_paths:
                    reading_sequence.append(('image', it))

        # Insert tables: attempt to find insertion point by searching for 'Table' label in text blocks
        for table in tables_for_case:
            inserted = False
            for idx, (ttype, payload) in enumerate(reading_sequence):
                if ttype == 'text' and isinstance(payload, str) and table.get('label') and table['label'][:10] in payload:
                    reading_sequence.insert(idx+1, ('table', table))
                    inserted = True
                    break
            if not inserted:
                reading_sequence.append(('table', table))

        # DEBUG: write assembled reading-order markdown if requested
        if args.debug:
            dbg_path = out_root / f'{stem}_reading_order.md'
            with dbg_path.open('w', encoding='utf-8') as fh:
                for ttype, payload in reading_sequence:
                    if ttype == 'text':
                        fh.write('\n\n## TEXT BLOCK\n\n')
                        fh.write(payload + '\n')
                    elif ttype == 'image':
                        fh.write('\n\n## IMAGE BLOCK\n\n')
                        fh.write(json.dumps(payload, ensure_ascii=False) + '\n')
                    else:
                        fh.write('\n\n## TABLE BLOCK\n\n')
                        fh.write(json.dumps(payload, ensure_ascii=False) + '\n')

        # Now chunk each text block semantically and build final chunks list in reading order
        final_chunks: List[Dict] = []
        chunk_counter = 0
        for ttype, payload in reading_sequence:
            if ttype == 'text':
                # semantic chunk into multiple chunks
                if text_tokenizer is not None:
                    parts = semantic_chunker_using_tokenizer(text_tokenizer, payload, chunk_tokens=args.chunk_tokens, overlap_tokens=args.chunk_overlap)
                else:
                    # fallback charwise
                    parts = chunk_text_charwise(payload, max_chars=1000, overlap=100)
                for p in parts:
                    final_chunks.append({'id': f'{stem}_text_{chunk_counter}', 'type': 'text', 'text': p['text']})
                    chunk_counter += 1
            elif ttype == 'table':
                final_chunks.append({'id': f'{stem}_table_{chunk_counter}', 'type': 'table', 'text': table.get('text'), 'label': table.get('label'), 'caption': table.get('caption')})
                chunk_counter += 1
            elif ttype == 'image':
                # image payload: expect dict with path and caption
                img_payload = payload if isinstance(payload, dict) else {'caption': None, 'path': str(payload)}
                final_chunks.append({'id': f'{stem}_image_{chunk_counter}', 'type': 'image', 'caption': img_payload.get('caption'), 'path': img_payload.get('path')})
                chunk_counter += 1

        # Prepare lists for embedding
        text_chunks = [c for c in final_chunks if c['type'] == 'text']
        table_chunks = [c for c in final_chunks if c['type'] == 'table']
        image_chunks = [c for c in final_chunks if c['type'] == 'image']

        # Embed text_chunks using text_model
        text_embs = np.zeros((0, 1), dtype=np.float32)
        if text_chunks and text_model is not None:
            # process in streaming batches
            all_t_embs = []
            for i in range(0, len(text_chunks), bs):
                batch = [c['text'] for c in text_chunks[i:i+bs]]
                try:
                    inputs = text_tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)
                    with torch.no_grad():
                        outputs = text_model(**inputs)
                        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                            emb = outputs.pooler_output
                        else:
                            emb = mean_pooling(outputs, inputs.get('attention_mask'))
                    all_t_embs.append(emb.cpu().numpy())
                except RuntimeError as e:
                    print('RuntimeError during text embedding batch, retrying smaller batch:', e)
                    # try one-by-one fallback
                    for s in batch:
                        try:
                            inp = text_tokenizer(s, return_tensors='pt', padding=True, truncation=True).to(device)
                            with torch.no_grad():
                                out = text_model(**inp)
                                if hasattr(out, 'pooler_output') and out.pooler_output is not None:
                                    embv = out.pooler_output
                                else:
                                    embv = mean_pooling(out, inp.get('attention_mask'))
                            all_t_embs.append(embv.cpu().numpy())
                        except Exception as e2:
                            print('Failed to embed single text chunk:', e2)
            if all_t_embs:
                text_embs = np.vstack([a if a.ndim==2 else a.reshape(1,-1) for a in all_t_embs])
        else:
            # empty array with inferred dim if possible
            text_embs = np.zeros((0, text_model.config.hidden_size if text_model is not None else 0), dtype=np.float32)

        # Embed tables using text_model (each table as single chunk)
        table_embs = np.zeros((0,1), dtype=np.float32)
        if table_chunks and text_model is not None:
            all_tab_embs = []
            for i in range(0, len(table_chunks), bs):
                batch = [c['text'] for c in table_chunks[i:i+bs]]
                inputs = text_tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)
                with torch.no_grad():
                    outputs = text_model(**inputs)
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        emb = outputs.pooler_output
                    else:
                        emb = mean_pooling(outputs, inputs.get('attention_mask'))
                all_tab_embs.append(emb.cpu().numpy())
            if all_tab_embs:
                table_embs = np.vstack(all_tab_embs)
        else:
            table_embs = np.zeros((0, text_model.config.hidden_size if text_model is not None else 0), dtype=np.float32)

        # Embed images using CLIP image encoder only
        image_embs = np.zeros((0,1), dtype=np.float32)
        image_paths_list: List[str] = []
        if image_chunks and clip_model is not None and clip_processor is not None:
            from PIL import Image
            all_img_embs = []
            batch_size_img = min(8, bs*2)
            for i in range(0, len(image_chunks), batch_size_img):
                batch_items = image_chunks[i:i+batch_size_img]
                images = []
                valid_paths = []
                for it in batch_items:
                    p = it.get('path')
                    if not p:
                        images.append(Image.new('RGB', (224,224), color=(255,255,255)))
                        valid_paths.append(None)
                        continue
                    # try various base paths
                    candidate = Path(p)
                    if not candidate.exists():
                        candidate = images_root / candidate
                    if not candidate.exists():
                        # try relative to markdown dir
                        candidate = markdown_dir / candidate
                    if not candidate.exists():
                        print('Image not found:', p)
                        images.append(Image.new('RGB', (224,224), color=(255,255,255)))
                        valid_paths.append(str(p))
                        continue
                    try:
                        img = Image.open(candidate).convert('RGB')
                        images.append(img)
                        valid_paths.append(str(candidate))
                    except Exception as e:
                        print('Failed open image', candidate, e)
                        images.append(Image.new('RGB', (224,224), color=(255,255,255)))
                        valid_paths.append(str(candidate))
                inputs = clip_processor(images=images, return_tensors='pt', padding=True).to(device)
                with torch.no_grad():
                    i_out = clip_model.get_image_features(**inputs)
                all_img_embs.append(i_out.cpu().numpy())
                image_paths_list.extend(valid_paths)
            if all_img_embs:
                image_embs = np.vstack(all_img_embs)
        else:
            image_embs = np.zeros((0, clip_model.visual_projection.out_features if clip_model is not None else 0), dtype=np.float32)

        # Persist embeddings
        text_emb_path = out_text / (stem + '_text.npy')
        table_emb_path = out_table / (stem + '_table.npy')
        image_emb_path = out_img / (stem + '_images.npy')
        np.save(text_emb_path, text_embs)
        np.save(table_emb_path, table_embs)
        np.save(image_emb_path, image_embs)

        # write per-case index preserving reading order
        index = {
            'file': md.name,
            'n_chunks': len(final_chunks),
            'text_embedding': str(text_emb_path),
            'table_embedding': str(table_emb_path),
            'image_embedding': str(image_emb_path),
            'chunks': []
        }
        # for each final chunk include pointer to embedding index (text/table/image) and order
        t_idx = 0; tab_idx = 0; img_idx = 0
        for c in final_chunks:
            rec = {'id': c['id'], 'type': c['type']}
            if c['type'] == 'text':
                rec['text'] = c['text']
                rec['embedding_index'] = t_idx
                t_idx += 1
            elif c['type'] == 'table':
                rec['text'] = c.get('text')
                rec['label'] = c.get('label')
                rec['caption'] = c.get('caption')
                rec['embedding_index'] = tab_idx
                tab_idx += 1
            elif c['type'] == 'image':
                rec['caption'] = c.get('caption')
                rec['path'] = c.get('path')
                rec['embedding_index'] = img_idx
                img_idx += 1
            index['chunks'].append(rec)

        write_json(out_emb / (stem + '.json'), index)
        summary.append(index)

    # write global summary
    meta = {
        'text_model': args.text_model,
        'image_model': args.image_model,
        'device': str(device),
        'num_files': len(summary)
    }
    write_json(out_emb / 'summary.json', meta)
    print('Done. Saved embeddings to', out_emb)


if __name__ == '__main__':
    main()