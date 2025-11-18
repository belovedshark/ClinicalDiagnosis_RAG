#!/usr/bin/env python3
"""
ingest_qdrant.py

Ingest embeddings from Processed_embeddings/embeddings into a local Qdrant server.

Usage:
  # start Qdrant (one-time)
  docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage -d qdrant/qdrant

  # install client in venv
  pip install qdrant-client

  # run this script
  python scripts/ingest_qdrant.py --emb-dir Processed_embeddings/embeddings --collection documents

"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from tqdm import tqdm

# Import helper from sibling module. When running this file as a script
# (python scripts/ingest_qdrant.py) the `scripts/` directory is on sys.path
# so import the module directly rather than as package `scripts.embedding_clip`.
from embedding_clip import clean_text


def ingest(emb_dir: Path, collection_name: str, host: str = 'localhost', port: int = 6333, dry_run: bool = False):
    """Ingest embeddings into Qdrant. If dry_run is True, do not connect to Qdrant and only report counts."""
    client = None
    if not dry_run:
        client = QdrantClient(url=f'http://{host}:{port}')

    # find sample embeddings to derive vector dimensions
    text_dir = emb_dir / 'text'
    image_dir = emb_dir / 'image'
    
    text_sample = next(text_dir.glob('*_text.npy'), None) if text_dir.exists() else None
    image_sample = next(image_dir.glob('*_images.npy'), None) if image_dir.exists() else None
    
    if text_sample is None:
        raise SystemExit('No text embeddings found in ' + str(emb_dir))
    
    text_dim = np.load(text_sample).shape[1]
    print(f'Text/Table vector dimension: {text_dim}')
    
    image_dim = None
    if image_sample:
        image_dim = np.load(image_sample).shape[1]
        print(f'Image vector dimension: {image_dim}')
    
    # create or recreate collection with named vectors for different dimensions
    if not dry_run:
        vectors_config = {
            'text': rest.VectorParams(size=text_dim, distance=rest.Distance.COSINE),
        }
        if image_dim:
            vectors_config['image'] = rest.VectorParams(size=image_dim, distance=rest.Distance.COSINE)
        
        # Use delete + create instead of deprecated recreate_collection
        try:
            client.delete_collection(collection_name=collection_name)
        except Exception:
            pass  # Collection doesn't exist yet
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
        )

    batch = []
    next_id = 1
    BATCH_SIZE = 256

    total_candidates = 0
    total_text = 0
    total_tables = 0
    total_images = 0

    json_files = sorted(emb_dir.glob('*.json'))
    print(f'\n📥 Ingesting {len(json_files)} documents into Qdrant...\n')
    
    for jsonf in tqdm(json_files, desc="Ingesting", unit="doc"):
        meta = json.loads(jsonf.read_text(encoding='utf-8'))
        stem = jsonf.stem
        # text, table and image npy files are stored under respective subfolders
        text_npy = emb_dir / 'text' / (stem + '_text.npy')
        table_npy = emb_dir / 'table' / (stem + '_table.npy')
        img_npy = emb_dir / 'image' / (stem + '_images.npy')

        # get all chunks from metadata
        chunks = meta.get('chunks', [])
        
        # separate chunks by type
        text_chunks = [c for c in chunks if c.get('type') == 'text']
        table_chunks = [c for c in chunks if c.get('type') == 'table']
        image_chunks = [c for c in chunks if c.get('type') == 'image']

        # ingest text embeddings
        if text_npy.exists() and text_chunks:
            arr = np.load(text_npy)
            total_text += int(arr.shape[0])
            for i, vec in enumerate(arr):
                if dry_run:
                    total_candidates += 1
                    continue
                # get the corresponding chunk
                chunk_text = ''
                if i < len(text_chunks):
                    chunk_text = text_chunks[i].get('text', '')
                payload = {
                    'file': meta.get('file'),
                    'type': 'text',
                    'chunk_id': i,
                    'text': clean_text(chunk_text) if chunk_text else ''
                }
                batch.append(rest.PointStruct(id=next_id, vector={'text': vec.tolist()}, payload=payload))
                next_id += 1
                if len(batch) >= BATCH_SIZE:
                    client.upsert(collection_name=collection_name, points=batch)
                    batch = []

        # ingest table embeddings
        if table_npy.exists() and table_chunks:
            arr = np.load(table_npy)
            total_tables += int(arr.shape[0])
            total_candidates += int(arr.shape[0])
            for i, vec in enumerate(arr):
                if dry_run:
                    continue
                # get the corresponding table chunk
                table_text = ''
                table_label = None
                table_caption = None
                if i < len(table_chunks):
                    table_text = table_chunks[i].get('text', '')
                    table_label = table_chunks[i].get('label')
                    table_caption = table_chunks[i].get('caption')
                payload = {
                    'file': meta.get('file'),
                    'type': 'table',
                    'table_id': i,
                    'text': clean_text(table_text) if table_text else '',
                    'label': table_label,
                    'caption': table_caption,
                }
                batch.append(rest.PointStruct(id=next_id, vector={'text': vec.tolist()}, payload=payload))
                next_id += 1
                if len(batch) >= BATCH_SIZE:
                    client.upsert(collection_name=collection_name, points=batch)
                    batch = []

        # ingest image embeddings
        if img_npy.exists() and image_chunks:
            arr = np.load(img_npy)
            total_images += int(arr.shape[0])
            for i, vec in enumerate(arr):
                if dry_run:
                    total_candidates += 1
                    continue
                # get the corresponding image chunk
                caption = None
                image_path = None
                if i < len(image_chunks):
                    caption = image_chunks[i].get('caption')
                    image_path = image_chunks[i].get('path')
                payload = {
                    'file': meta.get('file'),
                    'type': 'image',
                    'image_idx': i,
                    'caption': caption,
                    'image_path': image_path
                }
                batch.append(rest.PointStruct(id=next_id, vector={'image': vec.tolist()}, payload=payload))
                next_id += 1
                if len(batch) >= BATCH_SIZE:
                    client.upsert(collection_name=collection_name, points=batch)
                    batch = []

    if not dry_run and batch:
        client.upsert(collection_name=collection_name, points=batch)

    if dry_run:
        print('Dry-run completed. No data was sent to Qdrant.')
        print(f'  Files inspected: {len(list(emb_dir.glob("*.json")))}')
        print(f'  Text vectors: {total_text}')
        print(f'  Table vectors: {total_tables}')
        print(f'  Image vectors: {total_images}')
        print(f'  Total candidate points: {total_candidates}')
    else:
        print(f'\n✅ Ingest completed!')
        print(f'📊 Statistics:')
        print(f'   - Text chunks: {total_text}')
        print(f'   - Tables: {total_tables}')
        print(f'   - Images: {total_images}')
        print(f'   - Total points inserted: {next_id - 1}')
        print(f'   - Collection: "{collection_name}"')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb-dir', default='Processed_embeddings/embeddings', help='Embeddings directory')
    parser.add_argument('--collection', default='documents', help='Qdrant collection name')
    parser.add_argument('--host', default='localhost', help='Qdrant host')
    parser.add_argument('--port', type=int, default=6333, help='Qdrant port')
    parser.add_argument('--dry-run', action='store_true', help='Do not connect to Qdrant; just validate files and print counts')
    args = parser.parse_args()

    emb_dir = Path(args.emb_dir)
    if not emb_dir.exists():
        raise SystemExit('Embeddings folder not found: ' + str(emb_dir))

    ingest(emb_dir, args.collection, args.host, args.port, dry_run=args.dry_run)
