#!/usr/bin/env python3
"""Extract figure captions/labels and their image paths from markdown files.

For each input markdown file the script writes a new markdown file under
`Processed/markdown_img/` named `<original_basename>_markdown_img.md` that
contains caption lines followed by the matching image markdown, for example:

• Fig. 1.1 Oral bleeding in Ebola virus disease.

![figure](../images/.../img1.jpeg)

The pairing is performed sequentially: captions are matched to image
markdowns in document order. Any leftover images are appended at the end.

Usage:
  python3 scripts/extract_figures_to_md.py --path Processed/markdown/file.md
  python3 scripts/extract_figures_to_md.py --dir Processed/markdown --recursive

"""
import argparse
import re
from pathlib import Path


CAPTION_LINE_RE = re.compile(r"^\s*(?:[-*•]\s*)?Fig\.?\s*\d+(?:\.\d+)*\b.*$", re.I | re.M)
# Inline caption: optional bullet, 'Fig' (with optional dot), number (e.g. 1, 1.1), then text until a period
# Inline caption: optional bullet, 'Fig' (with optional dot), number (e.g. 1, 1.1), then capture
# the rest of the line. We previously tried to stop at the first period which caused
# truncation when sublabels like '(A)' appeared immediately after the figure number.
CAPTION_INLINE_RE = re.compile(r"(?:[-*•]?\s*)(Fig(?:\.)?\s*\d+(?:\.\d+)*)", re.I)
IMAGE_MD_RE = re.compile(r"(!\[[^\]]*\]\(([^)]+)\))")

# Common abbreviations that should not mark the end of a caption sentence
ABBREVIATIONS = {"e.g", "i.e", "et al", "dr", "mr", "mrs", "ms", "prof", "inc", "ltd", "jr", "sr", "vs", "etc", "fig"}

def extract_from_text(text):
    # Collect candidate captions with positions (pos, text)
    caption_candidates = []
    # 1) Capture line-starting captions using multiline regex to get positions
    for m in CAPTION_LINE_RE.finditer(text):
        s = m.group(0).strip()
        # skip if the whole line is enclosed in parentheses (listing, not a real caption)
        if s.startswith('(') and s.endswith(')'):
            continue
        # skip if this line contains a figure number immediately followed by a closing
        # parenthesis (e.g. 'Fig. 2.1)') which indicates a listing/inline reference
        if re.search(r"Fig(?:\.)?\s*\d+(?:\.\d+)*\s*\)", s, re.I):
            continue
        caption_candidates.append((m.start(), s, 'line'))

    # 2) Also capture inline captions (e.g., '... Further Reading ... • Fig. 1.1 Oral bleeding...')
    # Use finditer to preserve document order; record (pos, caption_text)
    inline_caps = []
    for m in CAPTION_INLINE_RE.finditer(text):
        # prefer to extract the full paragraph (up to the next blank line) containing
        # the figure label — captions sometimes wrap across soft newlines.
        para_start = text.rfind('\n\n', 0, m.start())
        if para_start == -1:
            para_start = text.rfind('\n', 0, m.start()) + 1
        else:
            para_start = para_start + 2
        para_end = text.find('\n\n', m.end())
        if para_end == -1:
            para_end = len(text)
        # Try to extract a sensible caption sentence starting at the 'Fig' match.
        # If there are no sublabels like '(A)' in the paragraph, treat the first
        # period after the figure label as the sentence end. If sublabels exist,
        # allow collecting multi-part sublabel sentences.
        caption_start = m.start()
        cap_text = None
        para_sub = text[m.start():para_end]
        has_sublabel = re.search(r"\([A-Z]\)", para_sub)

        # If the match includes a visible bullet (e.g. '• Fig...') that occurs
        # in the middle of a paragraph, prefer to treat the bullet as the start
        # of the caption so we don't include the preceding sentence.
        # robustly detect a nearby bullet like '• Fig' that may occur inside
        # a paragraph (e.g. "... unable to • Fig. 71.1 ..."). If found, treat the
        # bullet position as the caption start so we don't include the previous sentence.
        win_start = max(0, m.start() - 6)
        win_end = min(len(text), m.start() + 6)
        window = text[win_start:win_end]
        if re.search(r"[-*•]\s*Fig", window):
            # find the bullet position relative to the window
            bm = re.search(r"([-*•])\s*Fig", window)
            if bm:
                bullet_rel = bm.start(1)
                bullet_abs = win_start + bullet_rel
                if bullet_abs > para_start:
                    para_start = bullet_abs
                    caption_start = bullet_abs
                    para_sub = text[caption_start:para_end]
                    has_sublabel = re.search(r"\([A-Z]\)", para_sub)

        if not has_sublabel:
            # take first period after the match as caption end (if present within paragraph)
            # but skip periods that are likely decimals (e.g. '0.7'), ellipses, initials (J.),
            # or common abbreviations (e.g. 'et al.').
            search_pos = m.end()
            cap_text = None
            while True:
                pdot = text.find('.', search_pos)
                if pdot == -1 or pdot >= para_end:
                    break
                # find previous non-space character
                i = pdot - 1
                while i >= 0 and text[i] in ' \t\n\r':
                    i -= 1
                prevc = text[i] if i >= 0 else ''
                # find next non-space character
                j = pdot + 1
                while j < len(text) and text[j] in ' \t\n\r':
                    j += 1
                nextc = text[j] if j < len(text) else ''

                # skip if both surrounding non-space chars are digits (decimal number)
                if prevc.isdigit() and nextc.isdigit():
                    search_pos = pdot + 1
                    continue
                # skip ellipses '...'
                if prevc == '.' or nextc == '.':
                    search_pos = pdot + 1
                    continue

                # Extract the token immediately before the period to detect initials/abbrev
                k = pdot - 1
                while k >= 0 and text[k] not in ' \t\n\r':
                    k -= 1
                token = text[k+1:pdot].strip()
                token_norm = re.sub(r"[^A-Za-z]", "", token).lower()
                if (len(token_norm) == 1 and token_norm.isalpha()) or (token_norm in ABBREVIATIONS):
                    search_pos = pdot + 1
                    continue

                # otherwise accept this period as sentence end
                cap_text = text[caption_start:pdot+1].strip()
                break

            if cap_text is None:
                para = text[para_start:para_end].strip()
                if para.startswith('(') and para.endswith(')'):
                    continue
                cap_text = para
        else:
            # search for a period that plausibly ends the caption (period followed by
            # whitespace and an uppercase letter or '(' + uppercase). If none found,
            # fall back to the paragraph.
            search_pos = m.end()
            while True:
                pdot = text.find('.', search_pos)
                if pdot == -1 or pdot >= para_end:
                    break
                # find next non-space character after pdot
                j = pdot + 1
                while j < len(text) and text[j] in ' \t\n\r':
                    j += 1
                if j >= len(text):
                    cap_text = text[caption_start:pdot+1].strip()
                    break
                nxt = text[j]

                # Extract token immediately before the period to check for initials/abbrev
                k = pdot - 1
                while k >= 0 and text[k] not in ' \t\n\r':
                    k -= 1
                token = text[k+1:pdot].strip()
                token_norm = re.sub(r"[^A-Za-z]", "", token).lower()
                if (len(token_norm) == 1 and token_norm.isalpha()) or (token_norm in ABBREVIATIONS):
                    search_pos = pdot + 1
                    continue

                # accept period as sentence end if next is uppercase or '(' followed by uppercase
                if nxt.isupper() or (nxt == '(' and j+1 < len(text) and text[j+1].isupper()):
                    cap_text = text[caption_start:pdot+1].strip()
                    break
                # otherwise continue searching after this pdot
                search_pos = pdot + 1

            if cap_text is None:
                para = text[para_start:para_end].strip()
                # if the paragraph is wholly parenthesized, skip it
                if para.startswith('(') and para.endswith(')'):
                    continue
                cap_text = para
        # If caption contains a sublabel like '(A)', try to include subsequent
        # sublabel sentences '(B)', '(C)' etc. from the same paragraph so multi-part
        # captions aren't split.
        try:
            para = text[para_start:para_end]
        except Exception:
            para = ''

        if re.search(r"\([A-Z]\)", cap_text):
            extras = []
            for m2 in re.finditer(r"\([A-Z]\)", para):
                abspos = para_start + m2.start()
                # only consider sublabels that occur at or after the caption start
                if abspos < caption_start:
                    continue
                # find sentence end (period) after this sublabel
                pdot = text.find('.', abspos)
                if pdot != -1 and pdot < para_end:
                    seg = text[abspos:pdot+1].strip()
                    # avoid duplicating the (A) sentence if already present
                    if seg not in cap_text and seg not in extras:
                        extras.append(seg)
            if extras:
                cap_text = cap_text.rstrip() + ' ' + ' '.join(extras)

        # skip if the captured caption contains a figure number followed immediately by ')'
        if re.search(r"Fig(?:\.)?\s*\d+(?:\.\d+)*\s*\)", cap_text, re.I):
            continue
        # if there's a nearby bullet like '• Fig' treat this as a line-like caption
        win_start2 = max(0, m.start() - 6)
        win_end2 = min(len(text), m.start() + 6)
        window2 = text[win_start2:win_end2]
        kind = 'line' if re.search(r"[-*•]\s*Fig", window2) else 'inline'
        inline_caps.append((m.start(), cap_text, kind))

    # Add inline captions candidates
    for pos, cap, kind in inline_caps:
        caption_candidates.append((pos, cap, kind))

    # Deduplicate caption candidates by figure number: keep the longest caption for each figure
    captions_by_fig = {}
    other_captions = []
    FIG_NUM_RE = re.compile(r"Fig(?:\.)?\s*(\d+(?:\.\d+)*)", re.I)
    for pos, cap, kind in sorted(caption_candidates, key=lambda x: x[0]):
        mfig = FIG_NUM_RE.search(cap)
        if mfig:
            fnum = mfig.group(1)
            if fnum in captions_by_fig:
                # prefer line-start captions over inline ones; otherwise prefer longer text
                existing = captions_by_fig[fnum]
                if existing.get('kind') == 'inline' and kind == 'line':
                    captions_by_fig[fnum] = {'pos': pos, 'text': cap, 'kind': kind}
                elif existing.get('kind') == kind:
                    if len(cap) > len(existing['text']):
                        captions_by_fig[fnum] = {'pos': pos, 'text': cap, 'kind': kind}
            else:
                captions_by_fig[fnum] = {'pos': pos, 'text': cap, 'kind': kind}
        else:
            other_captions.append({'pos': pos, 'text': cap, 'kind': kind})

    # Build final captions list ordered by position: include deduped fig captions and others
    captions = []
    for fnum, info in captions_by_fig.items():
        captions.append((info['pos'], info['text']))
    for info in other_captions:
        captions.append((info['pos'], info['text']))
    captions.sort(key=lambda x: x[0])

    images = []
    for m in IMAGE_MD_RE.finditer(text):
        full = m.group(1)
        path = m.group(2)
        images.append({'pos': m.start(), 'full': full, 'path': path, 'paired': False})

    return captions, images


def write_output_for_file(src_path: Path, out_dir: Path):
    text = src_path.read_text(encoding='utf-8')
    captions, images = extract_from_text(text)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{src_path.stem}_markdown_img.md"

    with out_file.open('w', encoding='utf-8') as f:
        # Pair captions with images. If a caption contains sublabels like (A),(B),
        # consume that many images (if available). Otherwise consume one image.
        unpaired_images = [img for img in images]
        # helper to pop next unpaired image
        def pop_next_image():
            for img in unpaired_images:
                if not img['paired']:
                    img['paired'] = True
                    return img
            return None

        for pos, cap in captions:
            f.write(cap + '\n\n')
            # count sublabels (A),(B), etc.
            sublabels = re.findall(r"\(([A-Z])\)", cap)
            need = max(1, len(sublabels))
            assigned = 0
            while assigned < need:
                img = pop_next_image()
                if img is None:
                    break
                f.write(f"![figure]({img['path']})\n\n")
                assigned += 1

        # Append any remaining unpaired images
        for img in unpaired_images:
            if not img['paired']:
                f.write(f"![figure]({img['path']})\n\n")

    return out_file


def process_path(p: Path, out_dir: Path, recursive=False):
    results = []
    if p.is_file():
        # avoid processing already-generated outputs
        if 'markdown_img' in str(p.parent) or p.name.endswith('_markdown_img.md'):
            print(f'SKIP (generated file): {p}')
        else:
            results.append((p, write_output_for_file(p, out_dir)))
    else:
        it = p.rglob('*.md') if recursive else p.glob('*.md')
        for md in sorted(it):
            # skip files already in the output directory to avoid double-processing
            if 'markdown_img' in str(md.parent) or md.name.endswith('_markdown_img.md'):
                print(f'SKIP (generated file): {md}')
                continue
            results.append((md, write_output_for_file(md, out_dir)))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', type=str, help='Path to markdown file')
    ap.add_argument('--dir', type=str, help='Directory of markdown files')
    ap.add_argument('--recursive', action='store_true')
    ap.add_argument('--out', type=str, default='Processed/markdown_img', help='Output directory')
    args = ap.parse_args()

    targets = []
    if args.path:
        targets.append(Path(args.path))
    if args.dir:
        targets.append(Path(args.dir))
    if not targets:
        ap.error('Provide --path or --dir')

    out_dir = Path(args.out)
    all_results = []
    for t in targets:
        if not t.exists():
            print(f'SKIP: {t} does not exist')
            continue
        all_results.extend(process_path(t, out_dir, recursive=args.recursive))

    for src, out in all_results:
        print(f'{src} -> {out}')


if __name__ == '__main__':
    main()
