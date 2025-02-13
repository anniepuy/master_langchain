"""
Title: Personal Health Agent
Purpose: Agent that serves as a personal health assistant. LLM will decide does it use the Tavily Search or use the Health Supplement list
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
"""

from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults
from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

load_dotenv()

llm = ChatOllama(model='llama3.2', base_url="http://localhost:11434")

#Step 1: Define Tools
#Tavily Web Search
@tool
def search(query: str) -> str:
    """
    Search the web for realtime and latest information. For example, stock market news, news, weather, etc.
    """
    search = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True,
    )
    response = search.invoke(query)
    return response

# Health Supplement Retriever
db_name_path = "/Users/annhagan/Source/master_langchain/health_supplements"
embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url='http://localhost:11434')
vector_store = FAISS.load_local(db_name_path, embeddings, allow_dangerous_deserialization=True)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 3})

retriever

# Final Tool to help with retriever
@tool
def health_supplements(query: str) -> str:
    """
    Search the information about Health Supplements.
    For any questions about Health and Gym Supplements, you must use this tool.

    Args:
    query: The search query
    """
    response = retriever.invoke(query)
    return response

#test the tool
#print(health_supplements.invoke("What supplements help gain muscle mass?"))

#Step 2: Define the Agent