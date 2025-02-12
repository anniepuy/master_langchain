"""
Title: Using LangChain Search Tools
Purpose: Tool calling to support agents
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
Requires pip install duckduckgo-search wikipedia yfinance xmltodict tavily-python

"""
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

#base set up
base_url = "http://localhost:11434"

model = 'llama3.2'

llm = ChatOllama(
    base_url=base_url,
    model = model,
    temperature = 1,
)

## Using DuckDuckGo Search 



