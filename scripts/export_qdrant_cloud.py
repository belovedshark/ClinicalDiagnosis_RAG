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

# Retrieve all data
points, _ = local.scroll(
    collection_name="documents",
    limit=10000,
    with_payload=True,
    with_vectors=True
)

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

# Ensure the collection exists
if "documents" not in [c.name for c in cloud.get_collections().collections]:
    cloud.recreate_collection(
        collection_name="documents",
        vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE)
    )

# Convert local records into PointStruct format
formatted_points = [
    models.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
    for p in points
]

# ✅ Upload in small batches to avoid timeout
BATCH_SIZE = 200
for i in range(0, len(formatted_points), BATCH_SIZE):
    batch = formatted_points[i : i + BATCH_SIZE]
    print(f"⬆️ Uploading batch {i//BATCH_SIZE + 1} ({len(batch)} points)...")
    cloud.upsert(collection_name="documents", points=batch)
    time.sleep(1)  # optional pause to be gentle with the API

print(f"✅ Successfully uploaded {len(formatted_points)} points to Qdrant Cloud in batches!")