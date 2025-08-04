from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import tempfile, requests, os
import fitz
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import docx
from email import policy
from email.parser import BytesParser
import mimetypes
import faiss
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

os.environ["GOOGLE_API_KEY"] = "AIzaSyB4mLfobM_4ExIZPoqgU1c5O-MmHsbuPw0"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY environment variable not set.")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class HackRxInputSimple(BaseModel):
    documents: str
    questions: list[str]

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return "".join(page.get_text() for page in doc)

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_eml(file_path):
    with open(file_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    return msg.get_body(preferencelist=('plain')).get_content()

def extract_text(file_path):
    mime, _ = mimetypes.guess_type(file_path)
    if mime == "application/pdf":
        return extract_text_from_pdf(file_path)
    elif mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(file_path)
    elif mime == "message/rfc822" or file_path.endswith(".eml"):
        return extract_text_from_eml(file_path)
    else:
        raise ValueError("Unsupported file type")

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_faiss(index, query_emb, top_k=1):
    D, I = index.search(query_emb, top_k)
    return I[0]

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
    # Download the document
    response = requests.get(payload.documents)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download document")

    # Detect content type from HTTP header
    content_type = response.headers.get("Content-Type", "")
    if "pdf" in content_type:
        suffix = ".pdf"
    elif "word" in content_type:
        suffix = ".docx"
    elif "rfc822" in content_type or "message" in content_type:
        suffix = ".eml"
    else:
        suffix = ".bin"  # fallback

    # Save file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(response.content)
        file_path = tmp.name

    # Extract text and chunk
    text = extract_text(file_path)
    chunks = chunk_text(text)

    # Embed and build index
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = embed_texts(chunks, embedder)
    faiss_index = build_faiss_index(chunk_embeddings)

    # Gemini model
    gemini = genai.GenerativeModel("models/gemini-2.5-pro")

    results = []
    for question in payload.questions:
        q_emb = embed_texts([question], embedder)
        indices = search_faiss(faiss_index, q_emb, top_k=1)
        best_idx = int(indices[0])
        best_chunk = chunks[best_idx]
        similarity_score = float(cosine_similarity(q_emb, chunk_embeddings)[0][best_idx])

        prompt = (
            f"You are an insurance policy expert. Refer only to the provided excerpt below. "
            f"--- START OF EXCERPT --- {best_chunk} --- END OF EXCERPT --- "
            f"Answer the following question strictly based on the above excerpt, using professional and precise language: "
            f"{question} "
            f"Do not guess. If the answer is not present, respond with: 'Information not available in the provided excerpt.'"
        )

        try:
            response = gemini.generate_content(prompt)
            answer = response.text if hasattr(response, "text") else str(response)
        except Exception as e:
            answer = f"Gemini error: {e}"

        results.append({
            "question": question,
            "answer": answer,
            #"context_excerpt": best_chunk,
            #"similarity_score": similarity_score
        })

    return {"answers": results}

@app.get("/health")
def health_check():
    return {"status": "ok"}