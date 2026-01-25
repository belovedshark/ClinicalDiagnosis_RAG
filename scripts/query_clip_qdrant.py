"""
query_clip_qdrant.py

Query Qdrant with text or image using CLIP embeddings.
Example:
  python scripts/query_clip_qdrant.py --collection documents --query "A woman with fever and rash" --top-k 3
  
  python scripts/query_clip_qdrant.py --collection documents --image "test_images/test.jpeg" --top-k 3
  
  python scripts/query_clip_qdrant.py --collection documents --query "A woman with fever and rash" --top-k 3 --dump-top-chunks top_chunks.txt
  
  sample: (replace "A woman with fever and rash" with the below query)
  A 53-year-old man from Malawi presents with a 3-month history of productive cough, night sweats, and weight loss. He is HIV-positive with advanced immunosuppression (CD4 count 54 cells/μL). Sputum smears for tuberculosis are negative, and chest X-ray shows only bilateral hilar prominence. He has a low-grade fever and mild anaemia
  
  A 67-year-old German woman who has lived as an expatriate in Cameroon for the past 4 years presents with palpitations at a tropical medicine clinic in Germany. She also reports tran-sient subcutaneous swellings for the past year.
"""

import argparse
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModel
from PIL import Image
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', default='documents', help='Qdrant collection name')
    parser.add_argument('--host', default='http://localhost:6333', help='Qdrant URL')
    parser.add_argument('--query', type=str, help='Text query')
    parser.add_argument('--image', type=str, help='Path to image query')
    parser.add_argument('--text-model', default='thenlper/gte-small', help='Text embedding model (must match ingestion)')
    parser.add_argument('--image-model', default='openai/clip-vit-base-patch32', help='Image embedding model (CLIP)')
    parser.add_argument('--device', default='cpu', help='torch device (cpu or cuda)')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results to retrieve')
    parser.add_argument('--dump-top-chunks', type=str, default=None,
                        help='Path to write the top-k chunk contents for debugging')
    args = parser.parse_args()

    # Load models based on query type
    text_model = None
    text_tokenizer = None
    clip_model = None
    clip_processor = None
    
    if args.query:
        # Load text embedding model (same as used in ingestion)
        print(f"Loading text model: {args.text_model}")
        text_tokenizer = AutoTokenizer.from_pretrained(args.text_model)
        text_model = AutoModel.from_pretrained(args.text_model).to(args.device)
    
    if args.image:
        # Load CLIP for image queries
        print(f"Loading image model: {args.image_model}")
        clip_model = CLIPModel.from_pretrained(args.image_model).to(args.device)
        clip_processor = CLIPProcessor.from_pretrained(args.image_model)

    # Connect to Qdrant
    client = QdrantClient(url=args.host)
    print(f"Connected to Qdrant at {args.host}")

    # Create query vector (text or image)
    is_image_query = False
    if args.query:
        # Use text embedding model (GTE) for text queries
        inputs = text_tokenizer(args.query, return_tensors='pt', padding=True, truncation=True, max_length=512).to(args.device)
        with torch.no_grad():
            outputs = text_model(**inputs)
            # Use mean pooling for GTE model
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                query_emb = outputs.pooler_output.cpu().numpy()[0]
            else:
                # Mean pooling
                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                query_emb = (sum_embeddings / sum_mask).cpu().numpy()[0]
        print(f"Generated text embedding (384-dim) for query: {args.query[:60]}...")
        vector_name = 'text'

    elif args.image:
        # Use CLIP for image queries
        img = Image.open(args.image).convert('RGB')
        inputs = clip_processor(images=img, return_tensors='pt').to(args.device)
        with torch.no_grad():
            query_emb = clip_model.get_image_features(**inputs).cpu().numpy()[0]
        print(f"Generated image embedding (512-dim) for {args.image}")
        vector_name = 'image'
        is_image_query = True

    else:
        raise ValueError("Provide either --query (text) or --image (path).")

    # Search Qdrant using named vectors
    # For qdrant-client v1.16.0+, use query with NamedVector
    from qdrant_client.models import NamedVector, QueryRequest
    
    results = client.query_points(
        collection_name=args.collection,
        query=query_emb.tolist(),
        using=vector_name,
        limit=args.top_k,
        with_payload=True,
    ).points

    # Display results
    print(f"\nTop {args.top_k} results:")
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