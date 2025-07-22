#Handles reading PDF and HTML into raw text 
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader
from pathlib import Path

def load_pdfs(pdf_dir):
    pdfs = list(Path(pdf_dir).glob("*.pdf"))
    all_docs = []

    for pdf in pdfs:
        loader = PyPDFLoader(str(pdf))
        docs = loader.load()
        all_docs.extend(docs)

    return all_docs


def load_html(html_dir):
    htmls = list(Path(html_dir).glob("*.pdf"))
    all_docs = []
    
    for html in htmls:
        loader = UnstructuredHTMLLoader(str(html))
        docs = loader.load()
        all_docs.extend(docs)

    return all_docs