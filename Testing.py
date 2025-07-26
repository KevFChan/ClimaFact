from data_ingestion.load_documents import load_pdfs
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
import os

#Load environment variables 
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

#Load the FAISS vector store
vectorstore_path = "vectorstore/faiss_index"
embedding = OpenAIEmbeddings(openai_api_key = api_key)
db = FAISS.load_local(vectorstore_path, embedding, allow_dangerous_deserialization = True)

#Define the test query
query = "What actions is the EPA taking to address climate adaptation?"

#Run similarity search
docs = db.similarity_search(query, k = 3)

for i, doc in enumerate(docs, 1):
    print(f"\n--- Result #{i} ---")
    print(doc.page_content[:1000])


