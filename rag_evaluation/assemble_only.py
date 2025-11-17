#!/usr/bin/env python3
"""Assemble per-record context packs (text, table, images) and save to evaluation_kits/context_packs/.

Run:
  python -m rag_evaluation.assemble_only --limit 10
"""

import os
import json
import argparse
from . import utils


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def assemble(processed_dir: str = 'Processed', limit: int = 10):
    md_dir = os.path.join(processed_dir, 'markdown')
    table_dir = utils.get_table_dir(processed_dir)
    images_dir = os.path.join(processed_dir, 'images')

    out_context_dir = os.path.join('evaluation_kits', 'context_packs')
    ensure_dir(out_context_dir)
    out_context_md_dir = os.path.join('evaluation_kits', 'context_packs_md')
    ensure_dir(out_context_md_dir)

    files = utils.list_markdown_records(md_dir)
    files = files[:limit]

    for fn in files:
        path = os.path.join(md_dir, fn)
        basename = os.path.splitext(fn)[0]
        print(f"Assembling {fn}...")
        raw_md = utils.read_file_text(path)
        text_section = utils.extract_plain_text_from_markdown(raw_md)
        image_refs = utils.extract_image_paths_from_markdown(raw_md)
        images = []
        for im in image_refs:
            im_path = os.path.join(images_dir, os.path.basename(im))
            if os.path.exists(im_path):
                images.append({"path": im_path, "caption": ""})
            else:
                images.append({"path": im, "caption": ""})

        table_file = utils.find_matching_table_file(basename, table_dir)
        table_section = None
        table_structured = None
        if table_file:
            table_structured = utils.load_table_structured(table_file)
            if isinstance(table_structured, list):
                if table_structured:
                    cols = list(table_structured[0].keys())
                    header = ' | '.join(cols)
                    sep = ' | '.join(['---'] * len(cols))
                    rows = [' | '.join(str(r.get(c, '')) for c in cols) for r in table_structured]
                    table_section = '\n'.join([header, sep] + rows)
                else:
                    table_section = ''
            elif isinstance(table_structured, dict):
                table_section = json.dumps(table_structured)
            else:
                table_section = str(table_structured)

        # build markdown-formatted context
        md_parts = []
        md_parts.append('=== TEXT SECTION ===\n')
        md_parts.append(text_section or '')
        md_parts.append('\n\n=== TABLE SECTION ===\n')
        md_parts.append(table_section or '')
        md_parts.append('\n\n=== IMAGE SECTION ===\n')
        # include image links if available
        if images:
            for im in images:
                # use path in angle brackets to avoid markdown issues
                md_parts.append(f"- {im['path']}")
        else:
            md_parts.append('')

        context_markdown = '\n'.join(md_parts).strip()

        context_pack = {
            "record_id": fn,
            "source_files": {
                "markdown": path,
                "tables": table_file if table_file else None,
                "images": [i['path'] for i in images]
            },
            "context_pack": {
                "text_section": text_section,
                "table_section": table_section,
                "table_structured": table_structured,
                "images": images,
            }
        }

        # attach markdown string to JSON and save a .md file
        context_pack['context_markdown'] = context_markdown

        out_path = os.path.join(out_context_dir, f"{basename}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(context_pack, f, ensure_ascii=False, indent=2)

        md_out_path = os.path.join(out_context_md_dir, f"{basename}.md")
        with open(md_out_path, 'w', encoding='utf-8') as f:
            f.write(context_markdown)

        print(f"Wrote {out_path}")
        print(f"Wrote {md_out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--processed', default='Processed')
    p.add_argument('--limit', type=int, default=10)
    args = p.parse_args()
    assemble(processed_dir=args.processed, limit=args.limit)


if __name__ == '__main__':
    main()
