from typing import List
from langchain_core.documents import Document
from embedder import load_vectorstore
from retriever import search_similar_chunks
from llm_client import get_llm_response
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

# Load the embedding model (only once)
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Load vectorstore from disk (e.g., FAISS)
vectorstore = load_vectorstore("faiss_index", embedding_model)

def query_rag_chain(user_query: str, k: int = 5) -> str:
    """
    Full RAG pipeline: retrieves relevant chunks and gets LLM response.

    Args:
        user_query (str): Natural language question from the user.
        k (int): Number of top chunks to retrieve (default = 5).

    Returns:
        str: Generated response from LLM.
    """
    # Step 1: Retrieve top-k relevant documents
    retrieved_docs: List[Document] = search_similar_chunks(user_query, vectorstore, k=k)

    # Step 2: Generate response using LLM
    response: str = get_llm_response(user_query, retrieved_docs)

    return response
