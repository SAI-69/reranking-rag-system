"""
1. load the pdf from data folder
2 extract the text from the pdf
3. split the text into chunks
    3.1 we can use a simple split method like splitting
    3.2 follow proper chunking strategy
    3.3 chunk size = x tokens
    3.4 chunk overlap y tokens
4. create embeddings for the chunks
    4.1 choose the embedding model (gemini-embedding-2-preview or gemini-embedding-001)
    4.2 choose the dimension of the embeddings
    4.3 create the embeddings for each chunk
5. store those embeddings in a vector database
    5.1 our prefered vector db is pgvector
    5.2 we have to activate pgvector extension in our postgres db
    5.3 we have to create table to store the embeddings
    5.4 we have to insert embeddings to the table"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from src.core.db import get_vector_store

load_dotenv()
PG_CONNECTION = os.getenv("PG_CONNECTION_STRING")

def ingest_pdf(file_path,file_category):
    """Ingest a PDF file and save it in vector database"""

    # 1. load the pdf from data folder
    loader = PyPDFLoader(file_path)

    # 2 extract the text from the pdf
    docs=loader.load()    
    for doc in docs:
        doc.metadata.update(
            {
            "page": doc.metadata.get("page",None),
            "title": doc.metadata.get("title",None),
            "source": file_path,
            }             
        )
    
    # 3. split the text into chunks
    
    splitter = RecursiveCharacterTextSplitter(
        #Character based chunking. For token based chunking use "RecursiveCharacterTextSplitter.from_tiktoken_encoder"
        chunk_size = 500, 
        chunk_overlap = 100
    )
    chunks = splitter.split_documents(docs)
    print("Chunks ",len(chunks))

    # 4. create embeddings for the chunks
    # 5. store those embeddings in a vector database
    vector_store = get_vector_store(file_category)

    vector_store.add_documents(chunks)
    print("RAG ingestion sucessfull")
    
    
        
# ingest_pdf("data/HR_Knowledge_Base_2025.pdf","reranking_db")
# ingest_pdf("data/HR_Knowledge_Base_2026.pdf","reranking_db") 

# print("============ Ingestion Completed ============")