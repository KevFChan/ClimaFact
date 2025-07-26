#This RAG pipeline script with:
#1) Load the FAISS vector store
#2) Accept a user query
#3) Retrieves the top-k matching documents
#4) Use the LLM to generate an answer

from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from retriever import load_retriever
from prompt_formatter import get_prompt_template
from qa_generator import generate_answer

from dotenv import load_dotenv
import os

#Load the environment variables
load_dotenv()

# #1) Load the vector store
# retriever = load_retriever()

# #2) Define the query
# query = input("Ask a climate policy question: ")

# #3) Retrieve the top-k matching documents
# retriever = db.as_retriever(search_type = "similarity", search_kwargs = {"k": 3})
# docs = retriever.get_relevant_documents(query)

# #Print the context
# print("\n📄 Top Retrieved Chunks:")
# for i, doc in enumerate(docs):
#     print(f"\n--- Document {i+1} ---")
#     #Preview the first 1000 characters
#     print(doc.page_content[:1000])

# #4) Use the Hugging Face LLM to generate the answer
# USE_LLM = True

# if USE_LLM:
#     prompt_template = get_prompt_template()
    
#     answer = generate_answer(query, retriever, prompt_template)

#     print("\n🤖 Answer:")
#     print(answer)
def main():
    #1) Prompt the user for a climate policy question
    query = input("🌍 Ask a climate policy question: ")

    #2) Get retriever and prompt templates
    retriever = load_retriever()
    prompt_template = get_prompt_template()

    #3) Run the RAG pipeline
    answer = generate_answer(query, retriever, prompt_template)

    #4) Display the answer
    print("\n🤖 Answer:")
    print(answer)

if __name__ == "__main__":
    main()