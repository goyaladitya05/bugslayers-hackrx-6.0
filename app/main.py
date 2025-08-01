from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import tempfile, requests, os
import fitz
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

# Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = ""
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

class HackRxInputSimple(BaseModel):
    documents: str
    questions: list[str]

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed_texts(texts, model):
    return model.encode(texts, convert_to_numpy=True)

@app.post("/hackrx/run")
def hackrx_run(
    payload: HackRxInputSimple,
    authorization: str = Header(None)
):
    # Download PDF to temp file
    response = requests.get(payload.documents)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download document")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        pdf_path = tmp.name

    # Extract and chunk text
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    # Embed chunks using a local model (for semantic search)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = embed_texts(chunks, embedder)

    # Gemini model for answering
    gemini = genai.GenerativeModel("models/gemini-2.5-pro")

    results = []
    for question in payload.questions:
        # Embed the question
        q_emb = embed_texts([question], embedder)
        # Find the most relevant chunk
        sims = cosine_similarity(q_emb, chunk_embeddings)[0]
        best_idx = int(np.argmax(sims))
        best_chunk = chunks[best_idx]

        # Ask Gemini using only the best chunk
        prompt = f"Given this policy excerpt:\n\n{best_chunk}\n\nAnswer this question: {question}"
        response = gemini.generate_content(prompt)
        answer = response.text if hasattr(response, "text") else str(response)
        results.append({
            "question": question,
            "answer": answer,
            "context_excerpt": best_chunk
        })

    return {"answers": results}
