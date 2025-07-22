import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

from load_documents import load_pdfs, load_html
from chunk_documents import chunk_docs
from embed_and_store import embed_and_save

#Get the API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

#Load the document
pdf_docs = load_pdfs("data/pdf")
html_docs = load_html("data/html")
all_docs = pdf_docs + html_docs

#Chunk them all
chunks = chunk_docs(all_docs)

#Embed and store them
embed_and_save(chunks, api_key = api_key)
