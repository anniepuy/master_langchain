"""
Title: Embedding FAISS Setup Standalone
Purpose: Supports RAG Q&A with FAISS, LangChain & Ollama. Uses Q&A to answer questions based on the cosine simularity.
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
"""

import os
import warnings
from dotenv import load_dotenv

os.environ
warnings.filterwarnings("ignore")
load_dotenv()

#1 Document Loader 
from langchain_community.document_loaders import PyMuPDFLoader

pdfs = []
for root, dirs, files in os.walk("rag_with_vector/rag_dataset"):
    for file in files:
        if file.endswith(".pdf"):
            pdfs.append(os.path.join(root, file))

#after getting allow the pdfs, now we will read the documents page wise
docs = []
for pdfs in pdfs:
    loader = PyMuPDFLoader(pdfs)
    temp = loader.load()
    docs.extend(temp)


#check the docs uploaded
# print(len(docs))

#2 Apply text splitter (aka document chunks)
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)
#print(len(chunks))

#verify tockens per chunk
#import tiktoken
#encoding = tiktoken.encoding_for_model("gpt-4o-mini")
#encoding.encode(chunks[1].page_content)
#print(len(encoding.encode(chunks[1].page_content)))

#3 Create Vector Embedding with FAISS
#ollama pull nomic-embed-text
from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
#stores the vector in RAM memory - depricated
from langchain_community.docstore.in_memory import InMemoryDocstore

embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url='http://localhost:11434')

#single vector check
vector = embeddings.embed_query("Hello worlds")
#print(len(vector))

#4. Implement FAISS
index = faiss.IndexFlatL2(len(vector))
print(index.ntotal, index.d)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

vector_store.index.ntotal, vector_store.index.d

ids = vector_store.add_documents(documents = chunks)

#verify
#print(len(ids), vector_store.index.ntotal)
#print(ids)

#5. Retrieve & Search the vector stores from the InMemoryDocstore
#question = "How to gain muscle mass?"
#docs2 = vector_store.search(query=question, k=5, search_type="similarity")
#print(docs2)

#store the vector store
db_name = "health_supplements"
vector_store.save_local(db_name)
