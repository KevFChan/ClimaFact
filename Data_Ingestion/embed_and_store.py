#Embeds the chunks and saves them in FAISS or Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle

def embed_and_save(chunks, save_path = "vectorstore/faiss_index"):
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(save_path)

