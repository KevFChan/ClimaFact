#The purpose of this script is to contain the logic for generating an answer using an LLM, given a retriever and a prompt template

from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
import os

#Load the API key
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def build_llm():
    """Load and configure the Hugging Face model"""

    if not hf_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables.")
    
    return HuggingFaceEndpoint(
        repo_id = "google/flan-t5-xl",
        #endpoint_url = "https://api-inference.huggingface.co/models/google/flan-t5-xl",
        task = "text2text-generation",
        huggingfacehub_api_token = hf_token,
        temperature = 0.5,
        max_new_tokens = 512
    )

def generate_answer(query: str, retriever, prompt_template: PromptTemplate):
    """
    Run a retrieval-augmented generation chain to answer a query

    Parameters:
        query (str): The input question
        retriever: A retriever object that fetches relevant docs
        prompt_template (PromptTemplate): Formatting for the LLM input

    Returns:
        str: The generated answer
    """

    llm = build_llm()

    chain = RetrievalQA.from_chain_type(
        llm = llm,
        retriever = retriever,
        chain_type_kwargs = {"prompt": prompt_template}
    )

    return chain.invoke(query)

