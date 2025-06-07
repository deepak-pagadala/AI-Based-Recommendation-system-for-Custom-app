# embed_resources.py

import os
import hashlib
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
import openai

# 1) Load environment variables
load_dotenv()  # expects .env next to this script
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

openai.api_key = OPENAI_API_KEY

# 2) Initialize Qdrant client
qdrant = QdrantClient(url=QDRANT_URL)

# 3) Qdrant collection settings
COLLECTION_NAME = "tancho_resources"
VECTOR_SIZE = 3072  # OpenAI text-embedding-3-large

# Delete and create collection to ensure correct size
if qdrant.collection_exists(collection_name=COLLECTION_NAME):
    print(f"Deleting existing collection '{COLLECTION_NAME}' to fix vector size...")
    qdrant.delete_collection(collection_name=COLLECTION_NAME)

print(f"Creating collection '{COLLECTION_NAME}' with vector size {VECTOR_SIZE}...")
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=rest.VectorParams(size=VECTOR_SIZE, distance=rest.Distance.COSINE),
)

# 5) Read your CSV
df = pd.read_csv("resources/resources.csv")

def md5_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

for _, row in df.iterrows():
    # Extract exact column names
    resource_id = str(row["No. "]).strip()
    name        = str(row["Name "]).strip()
    language    = str(row["Langugage"]).strip()
    category    = str(row["Type"]).strip()
    description = str(row["Description"]).strip()
    topics      = [t.strip() for t in str(row["Key topics"]).split(",")]
    difficulty  = str(row["Difficulty"]).strip()
    study_time_raw = str(row["Study Time"]).strip()
    
    # Extract first number from study_time_raw (handles "45-60 mins" etc.)
    import re
    match = re.search(r"\d+", study_time_raw)
    study_time = int(match.group()) if match else 0

    # Prepare text and compute hash
    text_to_embed = description + " " + " ".join(topics)
    current_hash  = md5_hash(text_to_embed)

    # Skip if unchanged
    try:
        existing = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=[int(resource_id)])
        if existing and existing[0].payload.get("vectorHash") == current_hash:
            print(f"Skipping {resource_id} ({name}): no change.")
            continue
    except Exception:
        pass

    # Get embedding via new OpenAI API
    resp = openai.embeddings.create(
        model="text-embedding-3-large",
        input=[text_to_embed]
    )
    vector = resp.data[0].embedding

    # Prepare payload
    payload = {
        "name":               name,
        "description":        description,
        "topics":             topics,
        "language":           language,
        "category":           category,
        "difficulty":         difficulty,
        "estimatedStudyTime": study_time,
        "vectorHash":         current_hash
    }

    # Upsert into Qdrant
    point = rest.PointStruct(
        id=int(resource_id),
        vector=vector,
        payload=payload
    )
    qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])
    print(f"Upserted {resource_id}: {name}")

print("All resources embedded and upserted into Qdrant.")
