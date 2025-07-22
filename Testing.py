from data_ingestion.load_documents import load_pdfs
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
import os
