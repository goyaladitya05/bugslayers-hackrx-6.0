from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from typing import List
import os


def load_vectorstore(faiss_index_path: str, embedding_model: Embeddings) -> FAISS:
    """
    Loads a FAISS index from disk.

    Args:
        faiss_index_path (str): Path where the FAISS index is stored.
        embedding_model (Embeddings): The same embedding model used during storage.

    Returns:
        FAISS: The loaded vector store object.
    """
    return FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)


def search_similar_chunks(query: str, vectorstore: FAISS, k: int = 5) -> List[Document]:
    """
    Searches the FAISS index for the top-k most similar chunks to a given query.

    Args:
        query (str): The user's search query.
        vectorstore (FAISS): Loaded FAISS vector store.
        k (int): Number of top results to return.

    Returns:
        List[Document]: List of matching document chunks.
    """
    return vectorstore.similarity_search(query, k=k)
