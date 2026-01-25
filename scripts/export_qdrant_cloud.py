from qdrant_client import QdrantClient
from qdrant_client.http import models
import time
from pathlib import Path
import os

# Try to auto-load a local .env file for development convenience. If
# python-dotenv is not installed, we'll continue but require the
# environment variables to be set externally.
try:
    from dotenv import load_dotenv
    repo_root = Path(__file__).resolve().parent.parent
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment from {env_path}")
    else:
        print("No .env file found at project root; using environment variables.")
except Exception:
    print("python-dotenv not installed; ensure QDRANT_URL and QDRANT_API_KEY are set in the environment.")

# Connect to local Qdrant
local = QdrantClient(url="http://localhost:6333")

# Helper: stream points from local Qdrant in pages to avoid loading everything into memory
def stream_local_points(collection_name: str, page_size: int = 200):
    cursor = None
    while True:
        points, cursor = local.scroll(
            collection_name=collection_name,
            limit=page_size,
            offset=cursor,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break
        yield points
        if cursor is None:
            break

# Connect to Qdrant Cloud
import os

# Read Qdrant Cloud connection info from environment (safer than hardcoding)
QDRANT_CLOUD_URL = os.environ.get("QDRANT_URL")
QDRANT_CLOUD_API_KEY = os.environ.get("QDRANT_API_KEY")

if not QDRANT_CLOUD_URL or not QDRANT_CLOUD_API_KEY:
    raise RuntimeError(
        "QDRANT_URL and QDRANT_API_KEY must be set in the environment to export to Qdrant Cloud."
    )

cloud = QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_CLOUD_API_KEY)

# Configurable batch/page size and retry parameters via env
PAGE_SIZE = int(os.environ.get('EXPORT_BATCH_SIZE', '200'))
RETRIES = int(os.environ.get('EXPORT_RETRIES', '5'))
RETRY_BASE_DELAY = float(os.environ.get('EXPORT_RETRY_DELAY', '0.5'))
collection_name = 'documents'

# Scan the local collection to determine if vectors are single or named, and
# collect the full set of named vector names (so the cloud collection is
# created with every vector name that will appear during upsert).
vec_dim = None
named_vectors = {}
found_any = False
for page in stream_local_points(collection_name, page_size=PAGE_SIZE):
    found_any = True
    for p in page:
        v = getattr(p, 'vector', None)
        if v is None:
            continue
        # qdrant can store either a single vector (list) or named vectors (dict)
        if isinstance(v, dict):
            for k, vv in v.items():
                size = len(vv)
                if k in named_vectors:
                    if named_vectors[k].size != size:
                        raise RuntimeError(f'Inconsistent vector size for named vector "{k}"')
                else:
                    named_vectors[k] = models.VectorParams(size=size, distance=models.Distance.COSINE)
        else:
            if vec_dim is None:
                vec_dim = len(v)
            elif vec_dim != len(v):
                raise RuntimeError('Inconsistent vector sizes detected in collection')

if not found_any:
    raise RuntimeError(f'No points found in local collection "{collection_name}"')

if named_vectors:
    print(f'Detected named vectors {list(named_vectors.keys())}; ensuring cloud collection "{collection_name}" exists (this will recreate it)')
    cloud.recreate_collection(collection_name=collection_name, vectors_config=named_vectors)
else:
    if vec_dim is None:
        raise RuntimeError('Could not determine vector dimension from local points')
    print(f'Detected vector_dim={vec_dim}; ensuring cloud collection "{collection_name}" exists (this will recreate it)')
    cloud.recreate_collection(collection_name=collection_name, vectors_config=models.VectorParams(size=vec_dim, distance=models.Distance.COSINE))

def upsert_with_retries(points_batch, retries=RETRIES, base_delay=RETRY_BASE_DELAY):
    attempt = 0
    while True:
        try:
            cloud.upsert(collection_name=collection_name, points=points_batch)
            return
        except Exception as e:
            attempt += 1
            if attempt > retries:
                print(f'Upsert failed after {retries} retries: {e}')
                raise
            delay = base_delay * (2 ** (attempt - 1))
            print(f'Upsert attempt {attempt} failed, retrying in {delay:.1f}s: {e}')
            time.sleep(delay)

total_uploaded = 0
page_idx = 0
for page in stream_local_points(collection_name, page_size=PAGE_SIZE):
    page_idx += 1
    formatted = [models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in page]
    # upload the page as a single batch (Qdrant accepts lists up to reasonable size),
    # but we'll sub-slice if the page is very large
    for i in range(0, len(formatted), PAGE_SIZE):
        batch = formatted[i:i+PAGE_SIZE]
        print(f'⬆️ Uploading page {page_idx} batch {i//PAGE_SIZE + 1} ({len(batch)} points)')
        upsert_with_retries(batch)
        total_uploaded += len(batch)
        time.sleep(0.2)

print(f'✅ Successfully uploaded {total_uploaded} points to Qdrant Cloud in batches!')