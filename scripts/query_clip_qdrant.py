"""
query_clip_qdrant.py

Query Qdrant with text or image using CLIP embeddings.
Example:
  python scripts/query_clip_qdrant.py --collection documents --query "A woman with fever and rash" --top-k 3
  
  python scripts/query_clip_qdrant.py --collection documents --image "test_images/test.jpeg" --top-k 3
  
  python scripts/query_clip_qdrant.py --collection documents --query "A woman with fever and rash" --top-k 3 --dump-top-chunks top_chunks.txt
  
  sample: (replace "A woman with fever and rash" with the below query)
  A 53-year-old man from Malawi presents with a 3-month history of productive cough, night sweats, and weight loss. He is HIV-positive with advanced immunosuppression (CD4 count 54 cells/μL). Sputum smears for tuberculosis are negative, and chest X-ray shows only bilateral hilar prominence. He has a low-grade fever and mild anaemia
"""

import argparse
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', default='documents', help='Qdrant collection name')
    parser.add_argument('--host', default='http://localhost:6333', help='Qdrant URL')
    parser.add_argument('--query', type=str, help='Text query')
    parser.add_argument('--image', type=str, help='Path to image query')
    parser.add_argument('--model', default='openai/clip-vit-base-patch32', help='CLIP model name')
    parser.add_argument('--device', default='cpu', help='torch device (cpu or cuda)')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results to retrieve')
    parser.add_argument('--dump-top-chunks', type=str, default=None,
                        help='Path to write the top-k chunk contents for debugging')
    args = parser.parse_args()

    # Load CLIP
    model = CLIPModel.from_pretrained(args.model).to(args.device)
    processor = CLIPProcessor.from_pretrained(args.model)

    # Connect to Qdrant
    client = QdrantClient(url=args.host)
    print(f"Connected to Qdrant at {args.host}")

    # Create query vector (text or image)
    if args.query:
        inputs = processor(text=args.query, return_tensors='pt', truncation=True).to(args.device)
        with torch.no_grad():
            query_emb = model.get_text_features(**inputs).cpu().numpy()[0]
        print(f"Generated text embedding for query: {args.query}")

    elif args.image:
        img = Image.open(args.image).convert('RGB')
        inputs = processor(images=img, return_tensors='pt').to(args.device)
        with torch.no_grad():
            query_emb = model.get_image_features(**inputs).cpu().numpy()[0]
        print(f"Generated image embedding for {args.image}")

    else:
        raise ValueError("Provide either --query (text) or --image (path).")

    # Search Qdrant. Different qdrant-client versions expose different methods.
    # Prefer `search` (returns a list of scored points). Fall back to `query_points`
    # for older/newer variants if needed.
    if hasattr(client, 'search'):
        results = client.search(
            collection_name=args.collection,
            query_vector=query_emb.tolist(),
            limit=args.top_k,
            with_payload=True,
        )
    else:
        # Older API variant used in some client versions
        results = client.query_points(
            collection_name=args.collection,
            query_vector=query_emb.tolist(),
            limit=args.top_k
        )

    # Display results (handle both return shapes)
    print(f"\nTop {args.top_k} results:")

    # `results` may be an object with `.points` (query_points) or a list (search)
    if hasattr(results, 'points'):
        points = results.points
    else:
        points = results

    # If requested, dump the full content of each top chunk to a file for debugging
    if args.dump_top_chunks:
        out_path = args.dump_top_chunks
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(f"Query: {args.query or args.image}\n\n")
                for idx, r in enumerate(points, start=1):
                    # Support both dict-like and object-like points
                    if isinstance(r, dict):
                        payload = r.get('payload', {}) or {}
                        rid = r.get('id')
                        score = r.get('score')
                    else:
                        payload = getattr(r, 'payload', {}) or {}
                        rid = getattr(r, 'id', None)
                        score = getattr(r, 'score', None)

                    f.write(f"--- Result {idx} ---\n")
                    f.write(f"ID: {rid}\n")
                    f.write(f"Score: {score}\n")
                    if 'file' in payload:
                        f.write(f"File: {payload.get('file')}\n")
                    if 'text' in payload:
                        f.write("Text:\n")
                        # write full text (no truncation) followed by a separator
                        f.write(payload.get('text', '') + "\n")
                    if 'image_path' in payload:
                        f.write(f"Image: {payload.get('image_path')}\n")

                    # Dump any remaining payload keys for completeness
                    other_keys = [k for k in payload.keys() if k not in ('text', 'file', 'image_path')]
                    if other_keys:
                        f.write("Other payload fields:\n")
                        for k in other_keys:
                            f.write(f"  {k}: {payload.get(k)}\n")
                    f.write("\n")
            print(f"Wrote top {args.top_k} chunks to {out_path}")
        except Exception as e:
            print(f"Failed to write top chunks to {out_path}: {e}")

    for r in points:
        # Support both dict-like and object-like points
        if isinstance(r, dict):
            payload = r.get('payload', {}) or {}
            rid = r.get('id')
            score = r.get('score')
        else:
            payload = getattr(r, 'payload', {}) or {}
            rid = getattr(r, 'id', None)
            score = getattr(r, 'score', None)

        score_str = f"{score:.4f}" if (score is not None) else "N/A"
        print(f"ID: {rid}  |  Score: {score_str}")
        if 'file' in payload:
            print(f"  File: {payload['file']}")
        if 'text' in payload:
            txt = payload.get('text', '')
            print(f"  Text: {txt[:150]}...")
        if 'image_path' in payload:
            print(f"  Image: {payload['image_path']}")
        print("-" * 80)

if __name__ == '__main__':
    main()