from fastapi import FastAPI, Request, HTTPException, Depends
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

# Constants
API_TOKEN = "17d6ae50ac1c5917f307b509de33465b4081ed3b27e6f6d6700119da25fb35a7"

# FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Request model
class RunRequest(BaseModel):
    documents: str  # URL to PDF
    questions: list[str]

# Auth dependency
def authenticate(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

# Main route
@app.post("/api/v1/hackrx/run", dependencies=[Depends(authenticate)])
async def run_submission(req: RunRequest):
    try:
        # 1. Download the PDF
        response = requests.get(req.documents)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download document.")

        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        # 2. Process document
        docs = load_and_chunk_document(tmp_path)
        embed_and_store_chunks(docs, embedding_model, persist_directory="faiss_index")

        # 3. Answer each question
        results = []
        for question in req.questions:
            answer = query_rag_chain(question, k=5)
            results.append(answer)

        return {"answers": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
