#Chunk the documents into token pieces
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_docs(docs, chunk_size = 1000, chunk_overlap = 200):
    """"
    Split documents into overlapping text chunks for embedding

    Args:
        docs (List[Document]): LangChain document to split
        chunk_size (int): Size of each text chunk
        chunk_overlap (int): Overlap between chunks

    Returns:
        List[Document]: List of chunked documents
    """
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    return splitter.split_documents(docs)

