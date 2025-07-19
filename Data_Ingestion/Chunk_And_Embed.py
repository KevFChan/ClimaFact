import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

from load_documents import load_pdfs, load_html
from chunk_documents import chunk_docs
from embed_and_store import embed_and_save

load_dotenv()

#Load the document
loader = PyPDFLoader("epa_climate_plan.pdf")
docs = loader.load()

#Chunk it 
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
chunks = splitter.split_documents(docs)

#Embed
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)
db.save_local("faiss_index")
