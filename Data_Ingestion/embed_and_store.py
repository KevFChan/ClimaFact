#Embeds the chunks and saves them in FAISS or Chroma
#from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle

def embed_and_save(chunks, save_path = "vectorstore/faiss_index"):
    """"
    Embeds and stores the document once chunked

    Args:
        chunks (List[Document]): List of text chunks to embed
        api_key (str): The OpenAI API key
        save_path (str): Directory where the FAISS index will be saved

    Returns:
        None, it just saves the chunks
    """

    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(save_path)

