from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_and_chunk_document(file_path):
    loader = TextLoader(file_path)  # Or PDFLoader, etc.
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    return docs  # ✅ This should be list[Document]


# embedder.py or similar

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def load_vectorstore(index_path: str, embedding_model: Embeddings) -> FAISS:
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No FAISS index found at {index_path}")

    return FAISS.load_local(index_path, embeddings=embedding_model, allow_dangerous_deserialization=True)


def embed_and_store_chunks(
    docs: list[Document],
    embedding_model: Embeddings,
    persist_directory: str = "faiss_index"
):
    if not docs or not isinstance(docs[0], Document):
        raise ValueError("Docs must be a list of langchain Document objects.")

    vector_store = FAISS.from_documents(docs, embedding_model)
    vector_store.save_local(persist_directory)

