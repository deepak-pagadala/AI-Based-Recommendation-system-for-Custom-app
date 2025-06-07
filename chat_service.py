# chat_service.py

import os
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "tancho_resources"

app = FastAPI()
qdrant = QdrantClient(url=QDRANT_URL)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]
    top_k: int = 3

def retrieve_resources(query, top_k=3):
    resp = openai.embeddings.create(
        model="text-embedding-3-large",
        input=[query]
    )
    query_vector = resp.data[0].embedding

    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    resources = [
        {
            "name": hit.payload.get("name"),
            "description": hit.payload.get("description"),
            "topics": hit.payload.get("topics"),
            "difficulty": hit.payload.get("difficulty"),
            "estimatedStudyTime": hit.payload.get("estimatedStudyTime")
        }
        for hit in hits
    ]
    return resources

@app.post("/chat")
def chat_endpoint(body: ChatRequest):
    messages = [m.dict() for m in body.messages]
    top_k = body.top_k
    user_message = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

    # Step 1: Retrieve relevant resources
    resources = retrieve_resources(user_message, top_k=top_k)

    # Step 2: Build system prompt
    system_prompt = (
        "You are Tancho's AI language tutor. "
        "You have access to resources like books and podcasts from both Japanese and Korean languages. "
        "If a user asks about a resource, just try to convince them about how the resource helps them"
        "Get creative, do not repeat the description for the resource."
        "Dont always recommend something unless the user asks or feels like is struggling to understand"
        "Recommend only from the following resources and do not mention any outside apps or websites. "
        "If there is no relevant resource, say so. "
        f"Resources: {json.dumps(resources, ensure_ascii=False)}"
    )

    full_messages = [
        {"role": "system", "content": system_prompt},
        *messages
    ]

    # Step 3: Stream OpenAI response
    def stream_openai():
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=full_messages,
            stream=True,
            temperature=0.7,
        )
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                data = {"choices": [{"delta": {"content": content}}]}
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_openai(), media_type="text/event-stream")
