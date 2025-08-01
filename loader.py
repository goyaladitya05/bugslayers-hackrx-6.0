#from typing import list
from langchain.document_loaders import PyPDFLoader #splits pdf into page level docs
from langchain.text_splitter import RecursiveCharacterTextSplitter #breaks documents into small chunks for sending to llms, as they have a token window

def load_and_chunk_pdf(path: str) -> list:
    loader = PyPDFLoader(path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        seperators=["\n\n","\n","."," "]
    )

    docs=splitter.split_documents(pages)
    return [doc.page_content for doc in docs]

x=load_and_chunk_pdf("wang24n.pdf")
print(x)