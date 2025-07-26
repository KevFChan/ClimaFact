#We will retrieve relevant documents from our vector store
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

#Load the environment variables
load_dotenv()

def load_retriever(
        vectorstore_path: str = "vectorstore/faiss_index",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        k: int = 3
):
    """
    Loads the vector store and returns a retriever that returns the top-k relevant documents.

    Args:
        vectorstore_path: Path to the FAISS index directory
        model_name: Name of the HuggingFace embedding model to use
        k: Number of documents to retrieve

    Returns:
        A configured retriever object
    """

    embedding_model = HuggingFaceEmbeddings(model_name = model_name)
    db = FAISS.load_local(vectorstore_path, embeddings = embedding_model, allow_dangerous_deserialization = True)
    retriever = db.as_retriever(search_type = "similarity", search_kwargs = {"k": k})

    return retriever

