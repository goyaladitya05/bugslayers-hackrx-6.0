# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.docstore.document import Document
# from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI

# def embed_chunks(chunks):
#     embeddings = OpenAIEmbeddings()
#     docs = [Document(page_content=chunk) for chunk in chunks]
#     vector_db = FAISS.from_documents(docs, embeddings)
#     return vector_db

# def build_qa_chain(vector_db):
#     llm = ChatOpenAI(model="gpt-3.5-turbo")
#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=vector_db.as_retriever(search_kwargs={"k": 3})
#     )
#     return qa
