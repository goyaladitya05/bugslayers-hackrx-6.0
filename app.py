from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from loader import load_and_chunk_document
from embedder import embed_and_store_chunks, load_vectorstore
from chain import query_rag_chain
from llm_client import embedding_model  # make sure embedding_model is imported
import os
import shutil

app = FastAPI()

# Enable CORS for all origins (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------- Upload route -----------
import time
@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG API. Use /upload/ to POST a file and /query/ to ask questions."}


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    start_time = time.time()

    # Save file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print("[DEBUG] File saved")

    # Load and chunk
    docs = load_and_chunk_document(file_path)
    print(f"[DEBUG] Loaded and chunked into {len(docs)} chunks")

    # Embed and store
    embed_and_store_chunks(docs, embedding_model, persist_directory="faiss_index")
    print("[DEBUG] Embedding + storing done")

    end_time = time.time()
    print(f"[DEBUG] Total upload processing time: {end_time - start_time:.2f} seconds")

    return {"status": "File uploaded and processed successfully"}


# ----------- Query route -----------
@app.post("/query/")
async def ask_question(question: str = Form(...)):
    try:
        answer = query_rag_chain(question, k=5)
        return {"response": answer}
    except Exception as e:
        return {"error": str(e)}
