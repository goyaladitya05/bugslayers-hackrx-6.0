from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

def load_and_chunk_document(file_path: str) -> list[Document]:
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext == '.docx':
        loader = Docx2txtLoader(file_path)
    elif ext == '.eml':
        loader = UnstructuredEmailLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # This returns a list of Document objects
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_documents = splitter.split_documents(documents)

    print("[DEBUG] Loaded", len(chunked_documents), "document chunks")
    print("[DEBUG] Type of first chunk:", type(chunked_documents[0]))
    print("[DEBUG] Preview:", chunked_documents[0].page_content[:200])

    return chunked_documents
