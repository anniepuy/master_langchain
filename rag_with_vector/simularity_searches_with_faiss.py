"""
Title: Simularity searches with FAISS - uses multiple simularity searches 
Purpose: takes teh RAG/d documents set up in embedding_faiss_setup_standalone.py and uses the LLM to answer questions
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
"""

import os
import warnings
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

os.environ
warnings.filterwarnings("ignore")
load_dotenv()

embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url='http://localhost:11434')
db_name_path = "/Users/annhagan/Source/master_langchain/health_supplements"
vector_store = FAISS.load_local(db_name_path, embeddings, allow_dangerous_deserialization=True)

### Simularity Retreival
question = "how do I gain muscle mass?"
docs = vector_store.search(question, k=5, search_type="similarity")

### convert vector store to retrieval for LangChain Runnables
retriever = vector_store.as_retriever(search_type="similarity", 
                                      search_kwargs = {'k': 3})
result = retriever.invoke(question)
print(result)

##MMR Retreival
### Simularity Retreival
retriever3 = vector_store.as_retriever(search_type="mmr", 
                                      search_kwargs = {'k': 3, 'fetch_k': 3,
                                       "lambda_mult": 1 })
result3 = retriever3.invoke(question)
print("\n\n" + "MMR:")
print(result3)

# Simularity score threshold search 
retriever2 = vector_store.as_retriever(search_type='similarity_score_threshold', 
                                      search_kwargs = {'k': 3, 'score_threshold': 0.1})
result2 = retriever2.invoke(question)
print("\n\n" + "Similarity Score Threshold:")
print(result2)