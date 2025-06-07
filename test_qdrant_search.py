import os
from dotenv import load_dotenv
import openai
from qdrant_client import QdrantClient

# Load .env (API keys, etc.)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "tancho_resources"

# Set up clients
qdrant = QdrantClient(url=QDRANT_URL)

# Define your test query
query_text = "How can I learn flower names in Japanese?"

# 1. Get query embedding
resp = openai.embeddings.create(
    model="text-embedding-3-large",
    input=[query_text]
)
query_vector = resp.data[0].embedding

# 2. Query Qdrant for top 3 resources
hits = qdrant.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    limit=3
)

# 3. Print results
print("\nTop matches for:", query_text)
for hit in hits:
    payload = hit.payload
    print(f"\n• {payload['name']} (ID={hit.id})")
    print(f"  Description: {payload['description']}")
    print(f"  Topics: {payload['topics']}")
    print(f"  Difficulty: {payload['difficulty']}")
print("\n✅ Qdrant semantic search is working.")
