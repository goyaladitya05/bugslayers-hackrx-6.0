from fastapi import FastAPI, File, UploadFile, Form, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import requests
import os
from tempfile import NamedTemporaryFile

from loader import load_and_chunk_document
from embedder import embed_and_store_chunks
from chain import query_rag_chain
from llm_client import embedding_model

# === CONFIGURATION ===
API_TOKEN = "17d6ae50ac1c5917f307b509de33465b4081ed3b27e6f6d6700119da25fb35a7"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# === AUTHENTICATION ===
def authenticate(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing token")

# === REQUEST BODY MODEL ===
class RunRequest(BaseModel):
    documents: str  # URL to a PDF
    questions: list[str]

# === MAIN ENDPOINT ===
@app.post("/api/v1/hackrx/run", dependencies=[Depends(authenticate)])
async def run_submission(req: RunRequest):
    try:
        # Step 1: Download the PDF file
        response = requests.get(req.documents)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download the document.")

        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        # Step 2: Load and chunk the document
        docs = load_and_chunk_document(tmp_path)

        # Step 3: Embed and store chunks
        embed_and_store_chunks(docs, embedding_model, persist_directory="faiss_index")

        # Step 4: Answer all questions
        results = [query_rag_chain(q, k=5) for q in req.questions]

        return {"answers": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
