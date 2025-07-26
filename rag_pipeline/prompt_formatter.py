#This script will modularize how the prompts are constructured

from langchain_core.prompts import PromptTemplate

def get_prompt_template():
    """
    Returns a PromptTemplate that injects context and question into a system prompt designed for climate policy Q&A
    """

    template = """
You are a climate policy expert. Use the context below to answer the question as clearly and concisely as possible. If the context does not contain relevant information,
say you don't know.

Context:
{context}

Question:
{question}

Answer:
"""
    return PromptTemplate(
        input_variables = ["context", "question"],
        template = template.strip()
    )

