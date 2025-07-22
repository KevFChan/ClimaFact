#Embeds the chunks and saves them in FAISS or Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import pickle

def embed_and_save(chunks, api_key, save_path = "vectorstore/faiss_index"):
    embeddings = OpenAIEmbeddings(openai_api_key = api_key)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(save_path)

